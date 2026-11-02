from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from scripts.smoke_test_xlsr import build_tiny_classifier, run_smoke
from src.models.xlsr import MeanStdPooling, XLSRBinaryClassifier, load_xlsr_backbone


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/xlsr_smoke.json"
HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None


class MeanStdPoolingTests(unittest.TestCase):
    def test_known_population_statistics(self) -> None:
        features = torch.tensor([[[1.0, 2.0], [3.0, 6.0]]])
        actual = MeanStdPooling()(features)
        self.assertTrue(torch.allclose(actual, torch.tensor([[2.0, 4.0, 1.0, 2.0]])))

    def test_single_frame_and_constant_features_have_finite_backward(self) -> None:
        for frames in (1, 4):
            features = torch.ones(2, frames, 3, requires_grad=True)
            output = MeanStdPooling()(features)
            output.sum().backward()
            self.assertEqual(output.shape, (2, 6))
            self.assertTrue(torch.isfinite(features.grad).all())

    def test_rejects_empty_wrong_rank_and_nonfinite_features(self) -> None:
        for features in (torch.empty(0, 2, 3), torch.empty(2, 0, 3), torch.empty(2, 3, 0),
                         torch.ones(2, 3), torch.ones(2, 3, 4, dtype=torch.int64),
                         torch.full((2, 3, 4), float("nan"))):
            with self.subTest(shape=features.shape), self.assertRaises(ValueError):
                MeanStdPooling()(features)

    def test_hub_revision_must_be_immutable_before_loading(self) -> None:
        for revision in (None, "main", "abc123"):
            with self.assertRaisesRegex(ValueError, "commit SHA"):
                load_xlsr_backbone("facebook/wav2vec2-xls-r-300m", revision=revision)


@unittest.skipUnless(HAS_TRANSFORMERS, "Cần requirements-xlsr.txt để kiểm thử encoder thật thu nhỏ.")
class XLSRClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def setUp(self) -> None:
        torch.manual_seed(2026)

    def test_x0_only_head_updates_and_embeddings_are_deterministic_in_train_mode(self) -> None:
        model = build_tiny_classifier(self.config, 0)
        model.eval().train()
        waveform = torch.randn(2, 64_000) * 0.05
        first = model.extract_embeddings(waveform)
        second = model.extract_embeddings(waveform)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(first.requires_grad)
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        optimizer = torch.optim.Adam(model.optimizer_groups(encoder_lr=1e-5, classifier_lr=1e-3))
        logits = model(waveform)
        self.assertEqual(logits.shape, (2,))
        nn.BCEWithLogitsLoss()(logits, torch.tensor([0.0, 1.0])).backward()
        optimizer.step()
        self.assertTrue(any(not torch.equal(before[name], value) for name, value in model.named_parameters()
                            if name.startswith("classifier.")))
        for name, parameter in model.backbone.named_parameters():
            self.assertFalse(parameter.requires_grad, name)
            self.assertIsNone(parameter.grad, name)
            self.assertTrue(torch.equal(before[f"backbone.{name}"], parameter), name)

    def test_x1_only_last_block_and_head_receive_gradients_and_update(self) -> None:
        model = build_tiny_classifier(self.config, 1)
        model.eval().train()
        self.assertTrue(model.classifier.training)
        self.assertFalse(model.backbone.training)
        self.assertFalse(model.backbone.feature_extractor.training)
        self.assertFalse(model.backbone.encoder.layers[0].training)
        self.assertTrue(model.backbone.encoder.layers[1].training)
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        optimizer = torch.optim.Adam(model.optimizer_groups(encoder_lr=1e-3, classifier_lr=1e-3))
        nn.BCEWithLogitsLoss()(model(torch.randn(2, 64_000)), torch.tensor([0.0, 1.0])).backward()
        for name, parameter in model.named_parameters():
            expected_trainable = name.startswith(("classifier.", "backbone.encoder.layers.1."))
            self.assertEqual(parameter.requires_grad, expected_trainable, name)
            self.assertEqual(parameter.grad is not None, expected_trainable, name)
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        optimizer.step()
        changed = [name for name, value in model.named_parameters() if not torch.equal(value, before[name])]
        self.assertTrue(any(name.startswith("backbone.encoder.layers.1.") for name in changed))
        self.assertTrue(any(name.startswith("classifier.") for name in changed))
        self.assertTrue(all(name.startswith(("classifier.", "backbone.encoder.layers.1.")) for name in changed))
        model.eval()
        self.assertTrue(all(not module.training for module in model.modules()))

    def test_passes_waveform_unchanged_to_backbone(self) -> None:
        model = build_tiny_classifier(self.config, 0)
        seen: list[torch.Tensor] = []
        handle = model.backbone.register_forward_pre_hook(lambda module, inputs: seen.append(inputs[0].clone()))
        waveform = torch.randn(1, 64_000) * 0.07 + 0.02
        original = waveform.clone()
        with torch.no_grad():
            self.assertEqual(model(waveform).shape, (1,))
        handle.remove()
        self.assertTrue(torch.equal(original, waveform))
        self.assertTrue(torch.equal(original, seen[0]))

    def test_rejects_invalid_audio_and_sample_rate(self) -> None:
        model = build_tiny_classifier(self.config, 0)
        for waveform in (torch.zeros(64_000), torch.zeros(1, 1, 64_000), torch.zeros(1, 16_000),
                         torch.zeros(0, 64_000), torch.zeros(1, 64_000, dtype=torch.float64),
                         torch.full((1, 64_000), float("inf"))):
            with self.subTest(shape=waveform.shape), self.assertRaises(ValueError):
                model(waveform)
        with self.assertRaisesRegex(ValueError, "16.000 Hz"):
            model(torch.zeros(1, 64_000), sample_rate=40_000)

    def test_rejects_full_finetuning_and_invalid_layer_counts(self) -> None:
        for layers in (-1, 2, 3, True, 0.5):
            with self.subTest(layers=layers), self.assertRaisesRegex(ValueError, "Số tầng mở"):
                build_tiny_classifier(self.config, layers)
        with self.assertRaisesRegex(ValueError, "giao diện Wav2Vec2Model"):
            XLSRBinaryClassifier(nn.Linear(2, 2))

    def test_optimizer_groups_cover_trainable_parameters_once(self) -> None:
        for layers in (0, 1):
            model = build_tiny_classifier(self.config, layers)
            groups = model.optimizer_groups(encoder_lr=1e-5, classifier_lr=1e-3)
            ids = [id(parameter) for group in groups for parameter in group["params"]]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(set(ids), {id(parameter) for parameter in model.parameters() if parameter.requires_grad})
            self.assertEqual(groups[0]["lr"], 1e-3)
            self.assertEqual(groups[0]["name"], "classifier")
            self.assertEqual(len(groups), layers + 1)
            if layers:
                self.assertEqual(groups[1]["lr"], 1e-5)
                self.assertEqual(groups[1]["name"], "encoder")
            for invalid_lr in (0.0, -1.0, float("nan"), float("inf")):
                with self.assertRaisesRegex(ValueError, "Learning rate"):
                    model.optimizer_groups(encoder_lr=invalid_lr, classifier_lr=1e-3)

    def test_local_pretrained_loading_and_state_dict_roundtrip(self) -> None:
        model = build_tiny_classifier(self.config, 1).eval()
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            model.backbone.save_pretrained(directory)
            restored = XLSRBinaryClassifier(load_xlsr_backbone(directory), num_unfrozen_layers=1,
                                           **self.config["classifier"])
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        restored.eval()
        waveform = torch.randn(2, 64_000)
        with torch.no_grad():
            self.assertTrue(torch.equal(model(waveform), restored(waveform)))
        self.assertTrue(all(not parameter.requires_grad for parameter in restored.backbone.feature_extractor.parameters()))

    def test_smoke_report_proves_both_modes_without_dataset(self) -> None:
        report = run_smoke(CONFIG_PATH)
        self.assertEqual(report["status"], "passed")
        self.assertEqual((report["dataset_samples"], report["dataset_shards"]), (0, 0))
        for result in report["results"].values():
            self.assertTrue(result["passed"])
            self.assertEqual(result["state_dict_restore_max_difference"], 0.0)


if __name__ == "__main__":
    unittest.main()
