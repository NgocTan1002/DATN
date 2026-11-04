from __future__ import annotations

import json
import itertools
import random
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from src.training import ResumableBatchSampler, TrainingProgress
from src.training.xlsr_checkpoint import (
    XLSRTrainingRunIdentity,
    load_xlsr_training_checkpoint,
    save_xlsr_training_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]


class TinyResumeXLSR(nn.Module):
    def __init__(self, num_unfrozen_layers: int) -> None:
        super().__init__()
        self.num_unfrozen_layers = num_unfrozen_layers
        self.backbone = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )
        self.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(2, 1))
        self.backbone.requires_grad_(False)
        if num_unfrozen_layers:
            self.backbone[-1].requires_grad_(True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(value)).squeeze(-1)

    def optimizer_groups(
        self, *, encoder_lr: float, classifier_lr: float
    ) -> list[dict[str, object]]:
        groups: list[dict[str, object]] = [
            {
                "name": "classifier",
                "params": list(self.classifier.parameters()),
                "lr": classifier_lr,
            }
        ]
        encoder = [
            parameter for parameter in self.backbone.parameters() if parameter.requires_grad
        ]
        if encoder:
            groups.append({"name": "encoder", "params": encoder, "lr": encoder_lr})
        return groups


class XLSRCheckpointTests(unittest.TestCase):
    @staticmethod
    def identity() -> XLSRTrainingRunIdentity:
        return XLSRTrainingRunIdentity(
            manifest_sha256="1" * 64,
            audio_config_sha256="2" * 64,
            sample_selection_sha256="3" * 64,
            pretrained_model_id="facebook/wav2vec2-xls-r-300m",
            pretrained_revision="4" * 40,
            pretrained_weights_sha256="5" * 64,
            model_config_sha256="6" * 64,
            preprocessor_config_sha256="7" * 64,
            amplitude_policy="peak",
            waveform_normalization="pipeline_only",
            seed=2026,
            mode="X1",
            num_unfrozen_layers=1,
            classifier_hidden_features=8,
            classifier_dropout=0.4,
            batch_size=2,
            optimizer_name="Adam",
            encoder_learning_rate=1e-4,
            classifier_learning_rate=1e-3,
            step_budget=4,
        )

    @staticmethod
    def build() -> tuple[TinyResumeXLSR, torch.optim.Optimizer]:
        torch.manual_seed(99)
        model = TinyResumeXLSR(1)
        optimizer = torch.optim.Adam(
            model.optimizer_groups(encoder_lr=1e-4, classifier_lr=1e-3)
        )
        return model, optimizer

    @staticmethod
    def seed_training() -> None:
        random.seed(2026)
        np.random.seed(2026)
        torch.manual_seed(2026)

    @staticmethod
    def train_remaining(
        model: TinyResumeXLSR,
        optimizer: torch.optim.Optimizer,
        sampler: ResumableBatchSampler,
        *,
        maximum_batches: int | None = None,
    ) -> list[float]:
        features = torch.tensor(
            [[-1.0, 0.5], [0.1, 0.3], [0.2, -0.7], [0.9, 0.4],
             [-0.5, -0.2], [0.7, -0.8], [0.6, 0.2], [-0.3, 0.9]]
        )
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
        losses: list[float] = []
        model.train()
        batches = iter(sampler)
        if maximum_batches is not None:
            batches = itertools.islice(batches, maximum_batches)
        for indices in batches:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.BCEWithLogitsLoss()(model(features[indices]), labels[indices])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        return losses

    def test_interrupted_resume_matches_losses_and_parameters(self) -> None:
        self.seed_training()
        continuous_model, continuous_optimizer = self.build()
        self.seed_training()
        continuous_sampler = ResumableBatchSampler(
            dataset_size=8, batch_size=2, seed=2026
        )
        continuous_losses = self.train_remaining(
            continuous_model, continuous_optimizer, continuous_sampler
        )

        partial_model, partial_optimizer = self.build()
        self.seed_training()
        partial_sampler = ResumableBatchSampler(
            dataset_size=8, batch_size=2, seed=2026
        )
        partial_losses = self.train_remaining(
            partial_model, partial_optimizer, partial_sampler, maximum_batches=2
        )
        self.assertEqual(partial_sampler.batch_cursor, 2)

        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            path = Path(temporary) / "xlsr.pt"
            save_xlsr_training_checkpoint(
                path,
                model=partial_model,
                optimizer=partial_optimizer,
                progress=TrainingProgress(
                    global_step=2, epoch=0, batch_cursor=partial_sampler.batch_cursor
                ),
                identity=self.identity(),
                sampler=partial_sampler,
            )
            stored = torch.load(path, weights_only=True)
            self.assertTrue(stored["trainable_state_dict"])
            self.assertTrue(
                all(
                    name.startswith(("backbone.2.", "classifier."))
                    for name in stored["trainable_state_dict"]
                )
            )
            self.assertFalse(path.with_name("xlsr.pt.tmp").exists())

            resumed_model, resumed_optimizer = self.build()
            resumed_sampler = ResumableBatchSampler(
                dataset_size=8, batch_size=2, seed=2026
            )
            state = load_xlsr_training_checkpoint(
                path,
                model=resumed_model,
                optimizer=resumed_optimizer,
                expected_identity=self.identity(),
                sampler=resumed_sampler,
                device=torch.device("cpu"),
            )
            self.assertEqual(state.progress.global_step, 2)
            suffix_losses = self.train_remaining(
                resumed_model, resumed_optimizer, resumed_sampler
            )

        self.assertEqual(partial_losses + suffix_losses, continuous_losses)
        for expected, actual in zip(
            continuous_model.parameters(), resumed_model.parameters()
        ):
            self.assertTrue(torch.equal(expected, actual))

    def test_identity_and_optimizer_group_mismatch_are_rejected(self) -> None:
        model, optimizer = self.build()
        sampler = ResumableBatchSampler(dataset_size=8, batch_size=2, seed=2026)
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            path = Path(temporary) / "xlsr.pt"
            save_xlsr_training_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                progress=TrainingProgress(global_step=0, epoch=0, batch_cursor=0),
                identity=self.identity(),
                sampler=sampler,
            )
            other_model, other_optimizer = self.build()
            with self.assertRaisesRegex(ValueError, "Identity checkpoint XLS-R"):
                load_xlsr_training_checkpoint(
                    path,
                    model=other_model,
                    optimizer=other_optimizer,
                    expected_identity=replace(
                        self.identity(), pretrained_revision="8" * 40
                    ),
                    sampler=ResumableBatchSampler(
                        dataset_size=8, batch_size=2, seed=2026
                    ),
                    device=torch.device("cpu"),
                )

        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            wrong_optimizer = torch.optim.Adam(
                model.optimizer_groups(encoder_lr=2e-4, classifier_lr=1e-3)
            )
            with self.assertRaisesRegex(ValueError, "Learning rate/group"):
                save_xlsr_training_checkpoint(
                    Path(temporary) / "wrong.pt",
                    model=model,
                    optimizer=wrong_optimizer,
                    progress=TrainingProgress(
                        global_step=0, epoch=0, batch_cursor=0
                    ),
                    identity=self.identity(),
                    sampler=sampler,
                )

    def test_identity_requires_explicit_mode_normalization_and_revision(self) -> None:
        for changed in (
            {"mode": "X0"},
            {"waveform_normalization": "unresolved"},
            {"pretrained_revision": "main"},
            {"seed": 1},
        ):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                replace(self.identity(), **changed)

    def test_candidate_config_is_pinned_but_not_marked_ready(self) -> None:
        candidate = json.loads(
            (ROOT / "configs/xlsr_pretrained_candidate.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertIn("not_ready", candidate["status"])
        self.assertEqual(
            candidate["source"]["revision"],
            "1a640f32ac3e39899438a2931f9924c02f080a54",
        )
        self.assertTrue(candidate["source"]["weights_downloaded"])
        self.assertEqual(
            candidate["source"]["weights_sha256"],
            "d5e490574712ad0a6736923b9ed11d4cd51c78609c36205f704fc4e87b11d2e0",
        )
        self.assertFalse(candidate["project_protocol"]["use_test"])
        self.assertTrue(
            all(
                value is None
                for value in candidate["unresolved_scientific_choices"].values()
            )
        )


if __name__ == "__main__":
    unittest.main()
