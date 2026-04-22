"""branch_import.py — import 失败分支 Gate（仅 kernel_side）。

契约（findings.md §3.3 ⑤）:
  - Gate-F: import traceback log 存在且 import_subtype == import_kernel_side
  - Gate-A: audit 含 [IMPORT_TRACEBACK_CITATION] [ROOT_CAUSE] [FIX_PLAN]
            fix_type ∈ import_kernel_fix_whitelist 且不在 env_side 黑名单
  - Gate-V: 新一轮 import.status == passed

env_side (ld_path / abi / toolkit_env / cmakelists / setup.py / build_ascendc)
由主 agent Phase 7 已过滤（debug_eligible=false）；若异常进入，Gate-F 直接 reject。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .common import GateOutcome


IMPORT_KERNEL_FIX_WHITELIST = {
    "pybind_symbol_fix",
    "kernel_ext_name_fix",
    "kernel_export_fix",
}
BLOCKED_FIX_TYPES = {
    "ld_path_fix",
    "abi_fix",
    "toolkit_env_fix",
    "cmakelists_fix",
    "setup_py_fix",
    "build_ascendc_fix",
}
REQUIRED_SECTIONS = ("[IMPORT_TRACEBACK_CITATION]", "[ROOT_CAUSE]", "[FIX_PLAN]")


class ImportBranch:

    def run_gate_f(self, task_dir, attempt: int) -> GateOutcome:
        task_dir = Path(task_dir)
        latest = task_dir / ".eval_status" / "latest.json"
        checks = {"latest_present": latest.exists()}
        if latest.exists():
            try:
                status = json.loads(latest.read_text())
            except (OSError, json.JSONDecodeError):
                status = {}
            checks["failure_type_is_import"] = status.get("failure_type") == "import_failed"
            checks["import_subtype_is_kernel"] = (
                status.get("import_subtype") == "import_kernel_side"
            )
            log_raw = status.get("log_path", "")
            log = Path(log_raw) if log_raw else None
            checks["traceback_log_present"] = bool(log and log.exists())
        ok = all(v for v in checks.values() if isinstance(v, bool))
        return GateOutcome("GATE-IMPORT-F", ok, checks)

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
            m = re.search(r"\[FIX_TYPE\]\s*:?\s*(\w+)", content)
            fix_type = m.group(1) if m else None
            checks["fix_type_in_kernel_whitelist"] = bool(
                fix_type and fix_type in IMPORT_KERNEL_FIX_WHITELIST
            )
            checks["fix_type_not_env"] = fix_type not in BLOCKED_FIX_TYPES
        ok = all(v for v in checks.values() if isinstance(v, bool))
        return GateOutcome("GATE-IMPORT-A", ok, checks)

    def run_gate_v(self, task_dir, attempt: int) -> GateOutcome:
        task_dir = Path(task_dir)
        curr = task_dir / ".eval_status" / f"phase8_attempt{attempt}.json"
        checks = {"curr_present": curr.exists()}
        loop_signal = "CONTINUE"
        if curr.exists():
            try:
                c = json.loads(curr.read_text())
            except (OSError, json.JSONDecodeError):
                c = {}
            import_passed = c.get("import", {}).get("status") == "passed"
            checks["import_passed"] = import_passed
            if c.get("failure_type") == "success":
                loop_signal = "PASS"
            elif import_passed:
                loop_signal = "CONTINUE"
            else:
                loop_signal = "STOP" if attempt >= 1 else "CONTINUE"
        return GateOutcome(
            "GATE-IMPORT-V",
            loop_signal != "STOP",
            checks,
            loop_signal=loop_signal,
            reason="import.status transitioned to passed",
        )
