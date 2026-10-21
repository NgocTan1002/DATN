"""Batch sampler xác định có thể tiếp tục giữa epoch."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from typing import Any

import torch
from torch.utils.data import Sampler


class ResumableBatchSampler(Sampler[list[int]]):
    """Sinh batch xác định và lưu cursor của batch kế tiếp."""

    def __init__(
        self,
        *,
        dataset_size: int,
        batch_size: int,
        seed: int,
        epoch: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        if dataset_size <= 0:
            raise ValueError("dataset_size phải lớn hơn 0.")
        if batch_size <= 0:
            raise ValueError("batch_size phải lớn hơn 0.")
        if epoch < 0:
            raise ValueError("epoch không được âm.")
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.batch_cursor = 0

    def _total_batches(self) -> int:
        if self.drop_last:
            return self.dataset_size // self.batch_size
        return math.ceil(self.dataset_size / self.batch_size)

    def _indices(self) -> list[int]:
        if not self.shuffle:
            return list(range(self.dataset_size))
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return torch.randperm(self.dataset_size, generator=generator).tolist()

    def __iter__(self) -> Iterator[list[int]]:
        indices = self._indices()
        for batch_index in range(self.batch_cursor, self._total_batches()):
            start = batch_index * self.batch_size
            batch = indices[start : start + self.batch_size]
            if len(batch) < self.batch_size and self.drop_last:
                break
            self.batch_cursor = batch_index + 1
            yield batch

    def __len__(self) -> int:
        return self._total_batches() - self.batch_cursor

    def set_epoch(self, epoch: int) -> None:
        """Chuyển sang epoch mới và đặt cursor về batch đầu tiên."""

        if epoch < 0:
            raise ValueError("epoch không được âm.")
        self.epoch = int(epoch)
        self.batch_cursor = 0

    def state_dict(self) -> dict[str, int | bool]:
        """Trả trạng thái đủ để tiếp tục đúng batch kế tiếp."""

        return {
            "dataset_size": self.dataset_size,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "epoch": self.epoch,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
            "batch_cursor": self.batch_cursor,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Nạp state nếu cấu hình sampler không thay đổi."""

        required = {
            "dataset_size",
            "batch_size",
            "seed",
            "epoch",
            "shuffle",
            "drop_last",
            "batch_cursor",
        }
        missing = sorted(required.difference(state))
        if missing:
            raise ValueError(f"Trạng thái sampler thiếu trường bắt buộc: {missing}")
        expected = {
            "dataset_size": self.dataset_size,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
        }
        mismatched = sorted(
            key for key, value in expected.items() if state[key] != value
        )
        if mismatched:
            raise ValueError(
                f"Cấu hình sampler không tương thích tại: {mismatched}"
            )
        epoch = int(state["epoch"])
        cursor = int(state["batch_cursor"])
        if epoch < 0 or not 0 <= cursor <= self._total_batches():
            raise ValueError("Epoch hoặc batch cursor trong sampler không hợp lệ.")
        self.epoch = epoch
        self.batch_cursor = cursor
