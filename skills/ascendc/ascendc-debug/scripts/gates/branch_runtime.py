"""branch_runtime.py — 运行时段错误分支 Gate.

契约（findings.md §3.3 ⑥）:
  - Gate-F: 运行时 stderr / core dump 路径存在，crash_signal 已记录
  - Gate-A: audit 含 [RUNTIME_ERROR_CITATION] [ROOT_CAUSE] [FIX_PLAN]
  - Gate-V: 新一轮不再 crash 或 crash 位置不同（进步）
"""
from __future__ import annotations

import json
from pathlib import Path

from .common import GateOutcome, MAX_ATTEMPTS


REQUIRED_SECTIONS = ("[RUNTIME_ERROR_CITATION]", "[ROOT_CAUSE]", "[FIX_PLAN]")


class RuntimeBranch:

    def run_gate_f(self, task_dir, attempt: int) -> GateOutcome:
        task_dir = Path(task_dir)
        latest = task_dir / ".verify_status" / "latest.json"
        checks = {"latest_present": latest.exists()}
        if latest.exists():
            try:
                s = json.loads(latest.read_text())
            except (OSError, json.JSONDecodeError):
                s = {}
            checks["failure_type_is_runtime"] = s.get("failure_type") == "runtime_error"
            checks["crash_signal_recorded"] = bool(
                s.get("execute", {}).get("crash_signal")
            )
            log_raw = s.get("log_path", "")
            checks["log_path_exists"] = bool(log_raw) and Path(log_raw).exists()
        ok = all(v for v in checks.values() if isinstance(v, bool))
        return GateOutcome("GATE-RUNTIME-F", ok, checks)

    def run_gate_a(self, task_dir, attempt: int) -> GateOutcome:
        task_dir = Path(task_dir)
        audit = task_dir / "precision_tuning" / f"precision_audit_{attempt}.md"
        checks = {"audit_exists": audit.exists()}
        if audit.exists():
            try:
                content = audit.read_text()
            except OSError:
                content = ""
            for sec in REQUIRED_SECTIONS:
                checks[f"has_{sec.strip('[]').lower()}"] = sec in content
        ok = all(v for v in checks.values() if isinstance(v, bool))
        return GateOutcome("GATE-RUNTIME-A", ok, checks)

    def run_gate_v(self, task_dir, attempt: int) -> GateOutcome:
        task_dir = Path(task_dir)
        curr = task_dir / ".verify_status" / f"phase8_attempt{attempt}.json"
        prev = (
            task_dir / ".verify_status" / f"phase8_attempt{attempt - 1}.json"
            if attempt > 0
            else None
        )
        checks = {"curr_present": curr.exists()}
        loop_signal = "CONTINUE"
        if curr.exists():
            try:
                c = json.loads(curr.read_text())
            except (OSError, json.JSONDecodeError):
                c = {}
            curr_crash = c.get("execute", {}).get("crash_signal")
            checks["no_longer_crashed"] = curr_crash is None
            if c.get("failure_type") == "success":
                loop_signal = "PASS"
            elif checks["no_longer_crashed"]:
                loop_signal = "CONTINUE"
            elif prev and prev.exists():
                try:
                    prev_crash = json.loads(prev.read_text()).get("execute", {}).get("crash_signal")
                except (OSError, json.JSONDecodeError):
                    prev_crash = None
                checks["crash_signal_changed"] = prev_crash != curr_crash
                loop_signal = "CONTINUE" if checks["crash_signal_changed"] else "STOP"
            else:
                loop_signal = "CONTINUE"
            # attempt cap 兜底：即使 crash_signal 每轮在变也不能无限循环
            if loop_signal == "CONTINUE" and attempt >= MAX_ATTEMPTS - 1:
                loop_signal = "STOP"
                checks["max_attempts_reached"] = True
        return GateOutcome(
            "GATE-RUNTIME-V",
            loop_signal != "STOP",
            checks,
            loop_signal=loop_signal,
            reason="runtime crash progression",
        )
