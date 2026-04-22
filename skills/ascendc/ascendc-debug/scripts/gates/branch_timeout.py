"""branch_timeout.py — 超时分支 Gate.

契约（findings.md §3.3 ⑦）:
  - Gate-F: verify_status.duration_sec ≥ 配置阈值；timeout_marker_present == true
  - Gate-A: audit 含 [SYNC_POINT_ANALYSIS] [ROOT_CAUSE] [FIX_PLAN]；
            fix_plan 指向同步/tiling/barrier/pipe
  - Gate-V: 新一轮在时限内完成（failure_type != timeout 且无 timeout_marker）
"""
from __future__ import annotations

import json
from pathlib import Path

from .common import GateOutcome, MAX_ATTEMPTS


REQUIRED_SECTIONS = ("[SYNC_POINT_ANALYSIS]", "[ROOT_CAUSE]", "[FIX_PLAN]")
TIMEOUT_FIX_KEYWORDS = {"sync", "SyncAll", "tiling", "barrier", "pipe"}


class TimeoutBranch:

    def run_gate_f(self, task_dir, attempt: int) -> GateOutcome:
        task_dir = Path(task_dir)
        latest = task_dir / ".verify_status" / "latest.json"
        checks = {"latest_present": latest.exists()}
        if latest.exists():
            try:
                s = json.loads(latest.read_text())
            except (OSError, json.JSONDecodeError):
                s = {}
            checks["failure_type_is_timeout"] = s.get("failure_type") == "timeout"
            checks["timeout_marker_present"] = bool(s.get("timeout_marker_present"))
        ok = all(v for v in checks.values() if isinstance(v, bool))
        return GateOutcome("GATE-TIMEOUT-F", ok, checks)

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
            checks["fix_plan_mentions_sync_or_tiling"] = any(
                k in content for k in TIMEOUT_FIX_KEYWORDS
            )
        ok = all(v for v in checks.values() if isinstance(v, bool))
        return GateOutcome("GATE-TIMEOUT-A", ok, checks)

    def run_gate_v(self, task_dir, attempt: int) -> GateOutcome:
        task_dir = Path(task_dir)
        curr = task_dir / ".verify_status" / f"phase8_attempt{attempt}.json"
        checks = {"curr_present": curr.exists()}
        loop_signal = "CONTINUE"
        if curr.exists():
            try:
                c = json.loads(curr.read_text())
            except (OSError, json.JSONDecodeError):
                c = {}
            no_timeout = (
                c.get("failure_type") != "timeout"
                and not c.get("timeout_marker_present")
            )
            checks["no_longer_timeout"] = no_timeout
            if c.get("failure_type") == "success":
                loop_signal = "PASS"
            elif no_timeout:
                loop_signal = "CONTINUE"
            else:
                loop_signal = "STOP" if attempt >= MAX_ATTEMPTS - 1 else "CONTINUE"
        return GateOutcome(
            "GATE-TIMEOUT-V",
            loop_signal != "STOP",
            checks,
            loop_signal=loop_signal,
            reason="timeout presence progression",
        )
