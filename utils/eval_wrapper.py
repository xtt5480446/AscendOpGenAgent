#!/usr/bin/env python3
"""eval_wrapper.py — 机器可读包装器，驱动 evaluate_ascendc.sh 并落盘 eval_status.json。

对外约定详见 `.planning/findings.md` §1.2–1.4。下游（trace-recorder / Phase 8 / subagent
Step 0.3 / Gate-V）唯一消费此脚本的输出，不再解析 evaluate_ascendc.sh stdout。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

SCHEMA_VERSION = 1
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
EVALUATE_SH = REPO_ROOT / "skills" / "ascendc" / "ascendc-translator" / "references" / "evaluate_ascendc.sh"


def _utcnow_iso() -> str:
    return dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_dirs(task_dir: Path) -> tuple[Path, Path]:
    status_dir = task_dir / ".eval_status"
    logs_dir = task_dir / ".eval_logs"
    status_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return status_dir, logs_dir


def _tail(text: str, n: int = 50) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    return "\n".join(lines[-n:])


# -------- failure_type 分类（findings.md §1.4）---------------------------------

_COMPILE_ERR_PATTERNS = [
    r"error: ", r"fatal error:", r"undefined reference",
    r"compile .*failed", r"ascendc.*build.*fail",
]
_IMPORT_ENV_PATTERNS = [
    r"lib(ascend_hal|runtime|torch|torch_npu|torch_cpu|opapi)[^\s]*\.so",
    r"ASCEND_TOOLKIT_HOME", r"LD_LIBRARY_PATH",
    r"cannot open shared object file",
]
_IMPORT_KERNEL_PATTERNS = [
    r"pybind11", r"_ext\.so", r"undefined symbol.*(kernel|Kernel|_do)",
    r"TORCH_EXTENSION_NAME", r"ModuleNotFoundError: No module named .*_ext",
]
_PRECISION_PATTERNS = [r"mismatch_ratio=", r"max_abs_diff="]
_CRASH_SIGNALS = {
    -signal.SIGSEGV: "SIGSEGV",
    -signal.SIGABRT: "SIGABRT",
    -signal.SIGBUS: "SIGBUS",
    -signal.SIGFPE: "SIGFPE",
}
_SSH_DISCONNECT = [
    r"ssh: connect to host", r"Connection (refused|reset|timed out)",
    r"ssh_exchange_identification", r"port 22: Connection",
]
_DOCKER_UNREACHABLE = [
    r"container .* not running", r"Cannot connect to the Docker daemon",
    r"Error response from daemon", r"docker: Error",
]


def _match_any(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text or "") for p in patterns)


def classify_failure(status: dict, proc) -> dict:
    """只改 failure_type / failed_step / import_subtype / abort_subtype / exit_signal /
    各阶段子状态。返回 dict 用于 status.update()。"""
    stdout = status.get("stdout_tail", "")
    stderr = status.get("stderr_tail", "")
    combined = f"{stdout}\n{stderr}"
    out: dict = {}

    # (a) wrapper 自触发 timeout（最高优先级）
    if status.get("timeout_marker_present"):
        out["failure_type"] = "timeout"
        out["failed_step"] = "execute"
        out["execute"] = {"status": "timeout", "crash_signal": None}
        return out

    rc = status.get("exit_code")

    # (b) success
    if rc == 0 and _match_any([r"PASS", r"all cases passed"], combined):
        out["failure_type"] = "success"
        out["verify"] = {
            "status": "passed", "total_cases": None,
            "passed_cases": None, "failed_cases": [],
        }
        return out
    if rc == 0:
        # evaluate 退出 0 但未打出 PASS：谨慎作为 success（evaluate_ascendc.sh 的典型行为）
        out["failure_type"] = "success"
        return out

    # (c) build_failed
    if _match_any(_COMPILE_ERR_PATTERNS, combined):
        out["failure_type"] = "build_failed"
        out["failed_step"] = "compile"
        out["compile"] = {"status": "failed", "error_summary": _tail(stderr, 10)}
        return out

    # (d) import_failed + 子类
    if _match_any([r"ImportError", r"ModuleNotFoundError", r"OSError: cannot open shared object"], combined):
        out["failure_type"] = "import_failed"
        out["failed_step"] = "import"
        if _match_any(_IMPORT_ENV_PATTERNS, combined):
            out["import_subtype"] = "import_env_side"
        else:
            # 默认归为 kernel_side（只要不明显是环境库问题）
            out["import_subtype"] = "import_kernel_side"
        out["import"] = {"status": "failed", "traceback_path": status.get("log_path")}
        return out

    # (e) runtime_error：明确 crash signal 且未触发 wrapper timeout
    if rc in _CRASH_SIGNALS:
        out["failure_type"] = "runtime_error"
        out["failed_step"] = "execute"
        out["exit_signal"] = _CRASH_SIGNALS[rc]
        out["execute"] = {"status": "crashed", "crash_signal": _CRASH_SIGNALS[rc]}
        return out

    # (f) precision_failed：verify 阶段正常退出但输出对比失败
    if _match_any(_PRECISION_PATTERNS, combined) and rc != 0:
        out["failure_type"] = "precision_failed"
        out["failed_step"] = "verify"
        out["verify"] = {
            "status": "failed", "total_cases": None,
            "passed_cases": None, "failed_cases": [],
        }
        return out

    # (g) execution_aborted 兜底
    out["failure_type"] = "execution_aborted"
    if rc == 255 or _match_any(_SSH_DISCONNECT, combined):
        out["abort_subtype"] = "ssh_disconnected"
    elif _match_any(_DOCKER_UNREACHABLE, combined):
        out["abort_subtype"] = "docker_unreachable"
    elif rc is not None and rc < 0:
        out["abort_subtype"] = "killed_by_outer_harness"
        out["exit_signal"] = f"SIGNAL_{-rc}"
    else:
        out["abort_subtype"] = "unknown"
    return out


def run_once(phase: int, attempt: int, task_dir: Path, timeout_sec: int) -> dict:
    """单次调用。不做 ssh/docker 重试（由 run() 包裹）。"""
    status_dir, logs_dir = _ensure_dirs(task_dir)
    started = _utcnow_iso()
    log_path = logs_dir / f"phase{phase}_attempt{attempt}_{dt.datetime.utcnow():%Y%m%d_%H%M%S}.log"
    timeout_marker = status_dir / "timeout_marker"
    if timeout_marker.exists():
        timeout_marker.unlink()

    task_name = task_dir.name
    cmd = ["bash", str(EVALUATE_SH), task_name]
    env = os.environ.copy()
    env["WORKDIR"] = str(REPO_ROOT)

    status: dict = {
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "attempt": attempt,
        "started_at": started,
        "ended_at": None,
        "duration_sec": None,
        "exit_code": None,
        "exit_signal": None,
        "failure_type": None,
        "failed_step": None,
        "log_path": str(log_path),
        "stdout_tail": "",
        "stderr_tail": "",
        "timeout_marker_present": False,
        "import_subtype": None,
        "abort_subtype": None,
        "compile": {"status": "skipped", "error_summary": None},
        "import": {"status": "skipped", "traceback_path": None},
        "execute": {"status": "skipped", "crash_signal": None},
        "verify": {
            "status": "skipped", "total_cases": None,
            "passed_cases": None, "failed_cases": [],
        },
    }

    proc = None
    try:
        proc = subprocess.run(
            cmd, timeout=timeout_sec, env=env, capture_output=True, text=True,
        )
        status["exit_code"] = proc.returncode
        status["stdout_tail"] = _tail(proc.stdout)
        status["stderr_tail"] = _tail(proc.stderr)
        log_path.write_text(
            f"--- STDOUT ---\n{proc.stdout}\n--- STDERR ---\n{proc.stderr}\n"
        )
    except subprocess.TimeoutExpired as exc:
        timeout_marker.write_text(
            f"triggered_at={_utcnow_iso()} timeout_sec={timeout_sec}\n"
        )
        status["exit_code"] = 124
        status["exit_signal"] = "SIGTERM_BY_WRAPPER"
        status["stdout_tail"] = _tail(exc.stdout or "")
        status["stderr_tail"] = _tail(exc.stderr or "")
        log_path.write_text(
            f"--- TIMEOUT triggered by eval_wrapper (timeout_sec={timeout_sec}) ---\n"
            f"--- STDOUT ---\n{exc.stdout or ''}\n"
            f"--- STDERR ---\n{exc.stderr or ''}\n"
        )

    status["timeout_marker_present"] = timeout_marker.exists()
    status["ended_at"] = _utcnow_iso()
    status["duration_sec"] = round(
        (
            dt.datetime.strptime(status["ended_at"], "%Y-%m-%dT%H:%M:%SZ")
            - dt.datetime.strptime(status["started_at"], "%Y-%m-%dT%H:%M:%SZ")
        ).total_seconds(),
        2,
    )

    status.update(classify_failure(status, proc))

    status_path = status_dir / f"phase{phase}_attempt{attempt}.json"
    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False))
    latest = status_dir / "latest.json"
    latest.write_text(json.dumps(status, indent=2, ensure_ascii=False))
    return status


def run(phase: int, attempt: int, task_dir: Path, timeout_sec: int) -> dict:
    """对外入口。对 ssh_disconnected / docker_unreachable 做内部 1 次重试。"""
    result = run_once(phase, attempt, task_dir, timeout_sec)
    if result.get("abort_subtype") in ("ssh_disconnected", "docker_unreachable"):
        return run_once(phase, attempt, task_dir, timeout_sec)
    return result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", type=int, required=True, help="4 / 6 / 8")
    p.add_argument("--attempt", type=int, default=0)
    p.add_argument("--task-dir", type=Path, required=True)
    p.add_argument("--timeout-sec", type=int, default=1800)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    status = run(args.phase, args.attempt, args.task_dir.resolve(), args.timeout_sec)
    print(json.dumps(status, indent=2, ensure_ascii=False))
    ft = status.get("failure_type")
    if ft == "success":
        return 0
    if ft == "execution_aborted" and status.get("abort_subtype") == "unknown":
        return 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
