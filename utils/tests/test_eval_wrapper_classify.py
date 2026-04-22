"""Unit tests for classify_failure — 对 findings.md §1.4 判定顺序做回归。"""
import signal
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.eval_wrapper import classify_failure  # type: ignore  # noqa: E402


def S(**kw):
    base = {
        "stdout_tail": "",
        "stderr_tail": "",
        "exit_code": 1,
        "timeout_marker_present": False,
        "log_path": "/tmp/fake.log",
    }
    base.update(kw)
    return base


class ClassifyTests(unittest.TestCase):
    def test_timeout_marker_wins(self):
        r = classify_failure(
            S(timeout_marker_present=True, exit_code=124, stderr_tail="whatever"),
            None,
        )
        self.assertEqual(r["failure_type"], "timeout")
        self.assertEqual(r["failed_step"], "execute")

    def test_success_zero_exit_with_pass(self):
        r = classify_failure(S(exit_code=0, stdout_tail="all cases passed"), None)
        self.assertEqual(r["failure_type"], "success")

    def test_success_zero_exit_without_pass(self):
        r = classify_failure(S(exit_code=0, stdout_tail="done"), None)
        self.assertEqual(r["failure_type"], "success")

    def test_build_failed(self):
        r = classify_failure(
            S(exit_code=1, stderr_tail="ascendc build failed\nfatal error: xxx"),
            None,
        )
        self.assertEqual(r["failure_type"], "build_failed")
        self.assertEqual(r["failed_step"], "compile")
        self.assertEqual(r["compile"]["status"], "failed")

    def test_import_kernel_side(self):
        r = classify_failure(
            S(
                exit_code=1,
                stderr_tail=(
                    "ImportError: cannot import _elu_ext\n"
                    "undefined symbol: elu_do"
                ),
            ),
            None,
        )
        self.assertEqual(r["failure_type"], "import_failed")
        self.assertEqual(r["import_subtype"], "import_kernel_side")

    def test_import_env_side(self):
        r = classify_failure(
            S(
                exit_code=1,
                stderr_tail=(
                    "ImportError: libascend_hal.so: "
                    "cannot open shared object file"
                ),
            ),
            None,
        )
        self.assertEqual(r["failure_type"], "import_failed")
        self.assertEqual(r["import_subtype"], "import_env_side")

    def test_runtime_crash_sigsegv(self):
        r = classify_failure(
            S(exit_code=-signal.SIGSEGV, stderr_tail="segfault"), None
        )
        self.assertEqual(r["failure_type"], "runtime_error")
        self.assertEqual(r["execute"]["crash_signal"], "SIGSEGV")

    def test_runtime_crash_sigabrt(self):
        r = classify_failure(
            S(exit_code=-signal.SIGABRT, stderr_tail="abort"), None
        )
        self.assertEqual(r["failure_type"], "runtime_error")
        self.assertEqual(r["execute"]["crash_signal"], "SIGABRT")

    def test_precision_failed(self):
        r = classify_failure(
            S(
                exit_code=1,
                stdout_tail="mismatch_ratio=2.30% max_abs_diff=1e-2",
            ),
            None,
        )
        self.assertEqual(r["failure_type"], "precision_failed")
        self.assertEqual(r["failed_step"], "verify")

    def test_ssh_disconnect_return_255(self):
        r = classify_failure(
            S(exit_code=255, stderr_tail="ssh: connect to host: Connection refused"),
            None,
        )
        self.assertEqual(r["failure_type"], "execution_aborted")
        self.assertEqual(r["abort_subtype"], "ssh_disconnected")

    def test_docker_unreachable(self):
        r = classify_failure(
            S(
                exit_code=1,
                stderr_tail=(
                    "Error response from daemon: "
                    "container cjm_cann1 not running"
                ),
            ),
            None,
        )
        self.assertEqual(r["failure_type"], "execution_aborted")
        self.assertEqual(r["abort_subtype"], "docker_unreachable")

    def test_unknown_fallback(self):
        r = classify_failure(
            S(exit_code=42, stderr_tail="some mysterious failure"), None
        )
        self.assertEqual(r["failure_type"], "execution_aborted")
        self.assertEqual(r["abort_subtype"], "unknown")

    def test_killed_by_outer_harness(self):
        r = classify_failure(
            S(exit_code=-signal.SIGTERM, stderr_tail="got killed"), None
        )
        self.assertEqual(r["failure_type"], "execution_aborted")
        self.assertEqual(r["abort_subtype"], "killed_by_outer_harness")


if __name__ == "__main__":
    unittest.main()
