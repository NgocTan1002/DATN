from __future__ import annotations

import unittest

from scripts.verify_gpu_runtime import collect_runtime_info, validate_runtime


class VerifyGpuRuntimeTests(unittest.TestCase):
    def test_runtime_report_has_reproducibility_fields(self) -> None:
        report = collect_runtime_info()

        self.assertIn("git_commit", report)
        self.assertIn("python", report)
        self.assertIn("pytorch", report)
        self.assertIn("pytorch_cuda", report)
        self.assertIn("cuda_available", report)
        self.assertIn("devices", report)

    def test_require_cuda_rejects_cpu_only_report(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "không nhìn thấy CUDA"):
            validate_runtime(
                {"cuda_available": False, "device_count": 0},
                require_cuda=True,
            )

    def test_optional_cuda_accepts_cpu_only_report(self) -> None:
        validate_runtime(
            {"cuda_available": False, "device_count": 0},
            require_cuda=False,
        )


if __name__ == "__main__":
    unittest.main()
