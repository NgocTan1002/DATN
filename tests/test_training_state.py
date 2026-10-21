from __future__ import annotations

import random
import tempfile
import unittest
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from src.training import (
    ResumableBatchSampler,
    TrainingProgress,
    TrainingRunIdentity,
    load_training_checkpoint,
    save_training_checkpoint,
)


class ResumableBatchSamplerTests(unittest.TestCase):
    def test_resume_returns_exact_remaining_batches(self) -> None:
        continuous = ResumableBatchSampler(
            dataset_size=10,
            batch_size=3,
            seed=2026,
            epoch=0,
            shuffle=True,
        )
        expected_batches = list(continuous)

        interrupted = ResumableBatchSampler(
            dataset_size=10,
            batch_size=3,
            seed=2026,
            epoch=0,
            shuffle=True,
        )
        iterator = iter(interrupted)
        completed_batches = [next(iterator), next(iterator)]

        resumed = ResumableBatchSampler(
            dataset_size=10,
            batch_size=3,
            seed=2026,
            epoch=0,
            shuffle=True,
        )
        resumed.load_state_dict(interrupted.state_dict())

        self.assertEqual(completed_batches + list(resumed), expected_batches)


class TrainingCheckpointTests(unittest.TestCase):
    @staticmethod
    def _identity() -> TrainingRunIdentity:
        return TrainingRunIdentity(
            manifest_sha256="a" * 64,
            config_sha256="b" * 64,
            sample_selection_sha256="c" * 64,
            seed=2026,
            run_mode="pilot",
            amplitude_policy="none",
            batch_size=2,
            optimizer_name="Adam",
            learning_rate=0.001,
            step_budget=4,
        )

    @staticmethod
    def _train_batches(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        batches: Iterable[list[int]],
        features: torch.Tensor,
        targets: torch.Tensor,
    ) -> list[float]:
        losses: list[float] = []
        loss_function = torch.nn.MSELoss()
        for indices in batches:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features[indices]).squeeze(1)
            loss = loss_function(prediction, targets[indices])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        return losses

    def test_checkpoint_restores_complete_training_state(self) -> None:
        random.seed(2026)
        np.random.seed(2026)
        torch.manual_seed(2026)
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss = model(torch.ones(2, 2)).sum()
        loss.backward()
        optimizer.step()
        sampler = ResumableBatchSampler(
            dataset_size=8,
            batch_size=2,
            seed=2026,
            epoch=0,
            shuffle=True,
        )
        next(iter(sampler))
        progress = TrainingProgress(global_step=1, epoch=0, batch_cursor=1)
        identity = self._identity()

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "training.pt"
            save_training_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                progress=progress,
                identity=identity,
                sampler=sampler,
            )
            expected_python = random.random()
            expected_numpy = float(np.random.random())
            expected_torch = torch.rand(3)
            expected_remaining = list(sampler)

            random.seed(999)
            np.random.seed(999)
            torch.manual_seed(999)
            restored_model = torch.nn.Linear(2, 1)
            restored_optimizer = torch.optim.Adam(
                restored_model.parameters(), lr=0.001
            )
            restored_sampler = ResumableBatchSampler(
                dataset_size=8,
                batch_size=2,
                seed=2026,
                epoch=0,
                shuffle=True,
            )

            restored = load_training_checkpoint(
                checkpoint_path,
                model=restored_model,
                optimizer=restored_optimizer,
                expected_identity=identity,
                sampler=restored_sampler,
                device=torch.device("cpu"),
            )

            self.assertEqual(restored.progress, progress)
            self.assertEqual(restored.source_path, checkpoint_path.resolve())
            self.assertEqual(list(restored_sampler), expected_remaining)
            self.assertTrue(restored_optimizer.state_dict()["state"])
            for expected, actual in zip(
                model.parameters(), restored_model.parameters()
            ):
                self.assertTrue(torch.equal(expected, actual))
            self.assertEqual(random.random(), expected_python)
            self.assertEqual(float(np.random.random()), expected_numpy)
            self.assertTrue(torch.equal(torch.rand(3), expected_torch))

    def test_interrupted_run_matches_continuous_run(self) -> None:
        torch.manual_seed(7)
        initial_model = torch.nn.Linear(2, 1)
        initial_state = deepcopy(initial_model.state_dict())
        features = torch.arange(16, dtype=torch.float32).reshape(8, 2) / 10.0
        targets = torch.linspace(0.0, 1.0, 8)
        identity = TrainingRunIdentity(
            manifest_sha256="a" * 64,
            config_sha256="b" * 64,
            sample_selection_sha256="c" * 64,
            seed=2026,
            run_mode="pilot",
            amplitude_policy="none",
            batch_size=2,
            optimizer_name="SGD",
            learning_rate=0.05,
            step_budget=4,
        )

        continuous_model = torch.nn.Linear(2, 1)
        continuous_model.load_state_dict(initial_state)
        continuous_optimizer = torch.optim.SGD(
            continuous_model.parameters(), lr=0.05, momentum=0.9
        )
        continuous_sampler = ResumableBatchSampler(
            dataset_size=8, batch_size=2, seed=2026, shuffle=True
        )
        continuous_losses = self._train_batches(
            continuous_model,
            continuous_optimizer,
            continuous_sampler,
            features,
            targets,
        )

        interrupted_model = torch.nn.Linear(2, 1)
        interrupted_model.load_state_dict(initial_state)
        interrupted_optimizer = torch.optim.SGD(
            interrupted_model.parameters(), lr=0.05, momentum=0.9
        )
        interrupted_sampler = ResumableBatchSampler(
            dataset_size=8, batch_size=2, seed=2026, shuffle=True
        )
        sampler_iterator = iter(interrupted_sampler)
        first_indices = next(sampler_iterator)
        second_indices = next(sampler_iterator)
        prefix_losses = self._train_batches(
            interrupted_model,
            interrupted_optimizer,
            [first_indices, second_indices],
            features,
            targets,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "interrupted.pt"
            save_training_checkpoint(
                checkpoint_path,
                model=interrupted_model,
                optimizer=interrupted_optimizer,
                progress=TrainingProgress(
                    global_step=2,
                    epoch=0,
                    batch_cursor=interrupted_sampler.batch_cursor,
                ),
                identity=identity,
                sampler=interrupted_sampler,
            )
            resumed_model = torch.nn.Linear(2, 1)
            resumed_optimizer = torch.optim.SGD(
                resumed_model.parameters(), lr=0.05, momentum=0.9
            )
            resumed_sampler = ResumableBatchSampler(
                dataset_size=8, batch_size=2, seed=2026, shuffle=True
            )
            load_training_checkpoint(
                checkpoint_path,
                model=resumed_model,
                optimizer=resumed_optimizer,
                expected_identity=identity,
                sampler=resumed_sampler,
                device=torch.device("cpu"),
            )
            suffix_losses = self._train_batches(
                resumed_model,
                resumed_optimizer,
                resumed_sampler,
                features,
                targets,
            )

        self.assertEqual(prefix_losses + suffix_losses, continuous_losses)
        for expected, actual in zip(
            continuous_model.parameters(), resumed_model.parameters()
        ):
            self.assertTrue(torch.equal(expected, actual))

    def test_checkpoint_rejects_run_identity_mismatch(self) -> None:
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        sampler = ResumableBatchSampler(
            dataset_size=8, batch_size=2, seed=2026, shuffle=True
        )
        identity = self._identity()
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "identity.pt"
            save_training_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                progress=TrainingProgress(global_step=0, epoch=0, batch_cursor=0),
                identity=identity,
                sampler=sampler,
            )
            restored_model = torch.nn.Linear(2, 1)

            with self.assertRaisesRegex(ValueError, "Identity checkpoint"):
                load_training_checkpoint(
                    checkpoint_path,
                    model=restored_model,
                    optimizer=torch.optim.Adam(
                        restored_model.parameters(), lr=0.001
                    ),
                    expected_identity=replace(identity, amplitude_policy="peak"),
                    sampler=ResumableBatchSampler(
                        dataset_size=8,
                        batch_size=2,
                        seed=2026,
                        shuffle=True,
                    ),
                    device=torch.device("cpu"),
                )

            for incompatible in (
                replace(identity, run_mode="full_training"),
                replace(identity, step_budget=5),
            ):
                with self.assertRaisesRegex(ValueError, "Identity checkpoint"):
                    load_training_checkpoint(
                        checkpoint_path,
                        model=restored_model,
                        optimizer=torch.optim.Adam(
                            restored_model.parameters(), lr=0.001
                        ),
                        expected_identity=incompatible,
                        sampler=ResumableBatchSampler(
                            dataset_size=8,
                            batch_size=2,
                            seed=2026,
                            shuffle=True,
                        ),
                        device=torch.device("cpu"),
                    )

    def test_checkpoint_missing_fields_is_rejected(self) -> None:
        identity = self._identity()
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "missing.pt"
            torch.save({"checkpoint_version": 1}, checkpoint_path)
            model = torch.nn.Linear(2, 1)

            with self.assertRaisesRegex(ValueError, "thiếu trường bắt buộc"):
                load_training_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
                    expected_identity=identity,
                    sampler=ResumableBatchSampler(
                        dataset_size=8,
                        batch_size=2,
                        seed=2026,
                        shuffle=True,
                    ),
                    device=torch.device("cpu"),
                )


if __name__ == "__main__":
    unittest.main()
