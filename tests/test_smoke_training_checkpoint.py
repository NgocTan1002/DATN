from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import psutil
from torch.utils.data import DataLoader

from scripts.smoke_train_lfcc_lcnn import (
    evaluate_development,
    load_resume_checkpoint,
)


class MeanLogit(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.mean(dim=(1, 2))


class SmokeTrainingCheckpointTests(unittest.TestCase):
    def _checkpoint(self, path: Path) -> tuple[torch.nn.Module, dict[str, object]]:
        torch.manual_seed(2026)
        model = torch.nn.Linear(3, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss = model(torch.ones(2, 3)).sum()
        loss.backward()
        optimizer.step()
        config: dict[str, object] = {"seed": 2026, "name": "test"}
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "seed": 2026,
                "steps": 7,
                "epochs_completed": 2,
                "amplitude_policy": "none",
                "batch_size": 8,
                "learning_rate": 0.001,
            },
            path,
        )
        return model, config

    def test_load_resume_checkpoint_restores_model_optimizer_and_counters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "resume.pt"
            expected_model, config = self._checkpoint(checkpoint_path)
            restored_model = torch.nn.Linear(3, 1)
            restored_optimizer = torch.optim.Adam(
                restored_model.parameters(), lr=0.001
            )

            state = load_resume_checkpoint(
                checkpoint_path,
                model=restored_model,
                optimizer=restored_optimizer,
                config=config,
                seed=2026,
                amplitude_policy="none",
                batch_size=8,
                learning_rate=0.001,
                device=torch.device("cpu"),
            )

            self.assertEqual(state.steps, 7)
            self.assertEqual(state.epochs_completed, 2)
            self.assertEqual(state.source_path, checkpoint_path.resolve())
            self.assertTrue(restored_optimizer.state_dict()["state"])
            for expected, actual in zip(
                expected_model.parameters(), restored_model.parameters()
            ):
                self.assertTrue(torch.equal(expected, actual))

    def test_load_resume_checkpoint_rejects_protocol_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "resume.pt"
            _, config = self._checkpoint(checkpoint_path)
            model = torch.nn.Linear(3, 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            with self.assertRaisesRegex(ValueError, "amplitude policy"):
                load_resume_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    seed=2026,
                    amplitude_policy="peak",
                    batch_size=8,
                    learning_rate=0.001,
                    device=torch.device("cpu"),
                )

    def test_development_reports_loader_and_model_time(self) -> None:
        loader = DataLoader(
            [
                {
                    "waveform": torch.tensor([0.1, 0.2, 0.3]),
                    "label": torch.tensor(0.0),
                },
                {
                    "waveform": torch.tensor([-0.1, -0.2, -0.3]),
                    "label": torch.tensor(1.0),
                },
            ],
            batch_size=2,
            shuffle=False,
        )

        loss, examples, _, loader_seconds, model_seconds = evaluate_development(
            model=MeanLogit(),
            lfcc=torch.nn.Identity(),
            loader=loader,
            loss_function=torch.nn.BCEWithLogitsLoss(),
            device=torch.device("cpu"),
            process=psutil.Process(),
        )

        self.assertEqual(examples, 2)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertGreaterEqual(loader_seconds, 0.0)
        self.assertGreaterEqual(model_seconds, 0.0)


if __name__ == "__main__":
    unittest.main()
