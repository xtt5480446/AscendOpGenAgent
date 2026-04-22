"""branch_precision parity test (snapshot 占位).

真正的 parity (对 archive_tasks/ 下的真实 baseline) 在 Stream L 集成联调时做。
本文件仅确保 gates.branch_precision 可 import 且 PrecisionBranch 签名正确。
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[2]
        / "skills" / "ascendc" / "ascendc-debug" / "scripts"
    ),
)


class ImportableOnly(unittest.TestCase):
    def test_branch_precision_importable(self):
        from gates import branch_precision  # noqa: F401
        self.assertTrue(hasattr(branch_precision, "PrecisionBranch"))

    def test_precision_branch_signatures(self):
        from gates.branch_precision import PrecisionBranch
        b = PrecisionBranch("dummy_op")
        for method in ("run_gate_f", "run_gate_a", "run_gate_v"):
            self.assertTrue(callable(getattr(b, method, None)), f"missing {method}")


if __name__ == "__main__":
    unittest.main()
