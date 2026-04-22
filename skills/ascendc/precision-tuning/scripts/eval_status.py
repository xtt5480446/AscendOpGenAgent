"""eval_status.py — eval_wrapper 产物的读取 / 校验 helper.

对外 API:
    load_eval_status(task_dir: Path, phase: int, attempt: int) -> dict
    load_latest_status(task_dir: Path) -> dict
    summarize_for_trace(status: dict) -> dict

由 Gate 路由器、trace-recorder、subagent Step 0.3 共用，确保对
`.eval_status/phase{N}_attempt{M}.json` 与 `latest.json` 的读取行为一致。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
REQUIRED_KEYS = {
    "schema_version",
    "phase",
    "attempt",
    "failure_type",
    "duration_sec",
    "exit_code",
    "log_path",
}


def _validate(status: dict) -> dict:
    missing = REQUIRED_KEYS - set(status.keys())
    if missing:
        raise ValueError(f"eval_status missing keys: {sorted(missing)}")
    if status["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"unexpected schema_version {status['schema_version']}, "
            f"expected {SCHEMA_VERSION}"
        )
    return status


def _status_dir(task_dir: Path) -> Path:
    return task_dir / ".eval_status"


def load_eval_status(task_dir: Path, phase: int, attempt: int) -> dict:
    path = _status_dir(task_dir) / f"phase{phase}_attempt{attempt}.json"
    if not path.exists():
        raise FileNotFoundError(str(path))
    return _validate(json.loads(path.read_text()))


def load_latest_status(task_dir: Path) -> dict:
    path = _status_dir(task_dir) / "latest.json"
    if not path.exists():
        raise FileNotFoundError(str(path))
    return _validate(json.loads(path.read_text()))


def summarize_for_trace(status: dict) -> dict[str, Any]:
    """提取给 trace-recorder final_status 使用的字段子集。"""
    return {
        "failure_type": status.get("failure_type"),
        "import_subtype": status.get("import_subtype"),
        "abort_subtype": status.get("abort_subtype"),
        "last_evaluate_phase": status.get("phase"),
        "last_evaluate_attempt": status.get("attempt"),
        "duration_sec": status.get("duration_sec"),
        "exit_code": status.get("exit_code"),
        "failed_step": status.get("failed_step"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task-dir", type=Path, required=True)
    ap.add_argument("--phase", type=int, default=None)
    ap.add_argument("--attempt", type=int, default=0)
    ap.add_argument(
        "--summarize",
        action="store_true",
        help="Output only the summarize_for_trace() subset",
    )
    args = ap.parse_args()

    status = (
        load_latest_status(args.task_dir)
        if args.phase is None
        else load_eval_status(args.task_dir, args.phase, args.attempt)
    )
    if args.summarize:
        status = summarize_for_trace(status)
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
