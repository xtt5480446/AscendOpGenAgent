"""branch_build.py — 编译失败分支 Gate.

契约（findings.md §3.3 ④）:
  - Gate-F: .verify_logs/ 内最新 build log 存在，含 compile 错误块
  - Gate-A: audit 含 [COMPILE_ERROR_CITATION] [ROOT_CAUSE] [FIX_PLAN]
            fix_type ∈ build_fix_whitelist
  - Gate-V: 新一轮 verify_status.failed_step 从 compile 推进到 execute/verify
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .common import GateOutcome, MAX_ATTEMPTS


BUILD_FIX_WHITELIST = {
    "api_usage_fix",           # AscendC API 名称/签名错误
    "template_arg_fix",        # 模板实参错
    "include_fix",             # 缺 include 头文件
    "signature_align_fix",     # kernel/pybind 签名对齐
    "pipe_queue_fix",          # TPipe / TQue 生命周期
    "tilingdata_field_fix",    # tiling 字段名 / 类型错
}
REQUIRED_SECTIONS = ("[COMPILE_ERROR_CITATION]", "[ROOT_CAUSE]", "[FIX_PLAN]")


class BuildBranch:

    def run_gate_f(self, task_dir, attempt: int) -> GateOutcome:
        task_dir = Path(task_dir)
        latest = task_dir / ".verify_status" / "latest.json"
        checks = {"latest_present": latest.exists()}
        if latest.exists():
            try:
                status = json.loads(latest.read_text())
            except (OSError, json.JSONDecodeError):
                status = {}
            checks["failure_type_is_build"] = status.get("failure_type") == "build_failed"
            checks["failed_step_is_compile"] = status.get("failed_step") == "compile"
            log_path = Path(status.get("log_path", "")) if status.get("log_path") else None
            checks["build_log_present"] = bool(log_path and log_path.exists())
            if log_path and log_path.exists():
                try:
                    checks["has_error_block"] = bool(
                        re.search(r"error:|fatal error", log_path.read_text(errors="replace"))
                    )
                except OSError:
                    checks["has_error_block"] = False
        ok = all(v for v in checks.values() if isinstance(v, bool))
        return GateOutcome("GATE-BUILD-F", ok, checks)

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
            checks["fix_type_in_whitelist"] = bool(m and m.group(1) in BUILD_FIX_WHITELIST)
        ok = all(v for v in checks.values() if isinstance(v, bool))
        return GateOutcome("GATE-BUILD-A", ok, checks)

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
            curr_failed_step = c.get("failed_step")
            curr_failure_type = c.get("failure_type")
            checks["curr_failed_step"] = curr_failed_step
            progressed = (
                curr_failed_step in ("execute", "verify", None)
                and curr_failure_type != "build_failed"
            )
            checks["progressed_past_compile"] = progressed
            if curr_failure_type == "success":
                loop_signal = "PASS"
            elif progressed:
                # 进展了但还没最终通过 → session 结束（跨分支不切换）
                loop_signal = "CONTINUE"
            else:
                loop_signal = "STOP" if attempt >= MAX_ATTEMPTS - 1 else "CONTINUE"
        ok = loop_signal in ("PASS", "CONTINUE")
        return GateOutcome(
            "GATE-BUILD-V",
            ok,
            checks,
            loop_signal=loop_signal,
            reason="build progress tracked via failed_step transition",
        )
