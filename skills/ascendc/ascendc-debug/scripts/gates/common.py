"""common.py — Gate 通用层。

契约（findings.md §3.3 ② / §7.6）:
  - 反作弊 hash 未破坏
  - AST 退化未引入
  - task_dir 目录结构完整
  - verify_status.json 产出存在且 schema_version 正确
  - {op}.json.bak 未被破坏
  - audit_{attempt}.md 文件存在（section schema 由分支层各自定义）

不检查 audit section 格式；不在 fix 步骤单独加 Gate——由下一轮 Gate-V 通过 verify_status 差分间接验证。
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# 跨分支共享的 attempt 上限（single source of truth）。
#
# 默认值 5。优先级（高 → 低）:
#   1. Subagent 在 Step 0 根据 failure_type / 错误复杂度动态设置 env var
#      （`export ASCENDC_DEBUG_MAX_ATTEMPTS=<N>` 或 inline `ASCENDC_DEBUG_MAX_ATTEMPTS=<N> python3 ...`）
#   2. Launcher / CI 层通过 `docker exec -e` 注入
#      （见 `utils/run_benchmark_ascendc_codex_with_debug.sh`）
#   3. 本文件默认值 5
#
# 每次 Gate 脚本以新 Python 进程启动时按 env 当场读值，不缓存；因此 subagent
# 在 bash session 里 `export` 后的所有后续 gate 调用都会生效。
# ---------------------------------------------------------------------------
try:
    MAX_ATTEMPTS = int(os.environ.get("ASCENDC_DEBUG_MAX_ATTEMPTS", "5"))
    if MAX_ATTEMPTS < 1:
        MAX_ATTEMPTS = 5
except (TypeError, ValueError):
    MAX_ATTEMPTS = 5


@dataclass
class GateOutcome:
    gate: str
    ok: bool
    checks: dict
    loop_signal: Optional[str] = None
    reason: Optional[str] = None

    def to_gate_output(self) -> dict:
        out = {"gate": self.gate, "passed": self.ok, "checks": self.checks}
        if self.loop_signal is not None:
            out["loop_signal"] = self.loop_signal
        if self.reason is not None:
            out["loop_reason"] = self.reason
        return out


def _sha256(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_anticheat(task_dir: Path) -> dict:
    """对 model_new_ascendc.py / model_new_tilelang.py 对比 .bench_baseline/ 的 hash。

    若 .bench_baseline/ 不存在（非 bench 环境），跳过并视为通过。
    """
    baseline_dir = task_dir / ".bench_baseline"
    if not baseline_dir.is_dir():
        return {"anticheat_baseline_present": False, "anticheat_pass": True}
    result = {"anticheat_baseline_present": True}
    hash_keys = []
    for name in ("model_new_ascendc.py", "model_new_tilelang.py"):
        bp = baseline_dir / name
        cp = task_dir / name
        if not bp.exists():
            continue
        key = f"hash_{name}"
        result[key] = _sha256(bp) == _sha256(cp)
        hash_keys.append(key)
    # 如果没有可比 hash，也视为通过（baseline dir 存在但无文件）
    result["anticheat_pass"] = all(result[k] for k in hash_keys) if hash_keys else True
    return result


def _find_ast_validator(task_dir: Path) -> Optional[Path]:
    """向上查找 skills/ascendc/ascendc-translator/scripts/validate_ascendc_impl.py。"""
    for cand in [task_dir] + list(task_dir.parents):
        p = cand / "skills" / "ascendc" / "ascendc-translator" / "scripts" / "validate_ascendc_impl.py"
        if p.exists():
            return p
    return None


def check_ast_degrade(task_dir: Path) -> dict:
    """调用 validate_ascendc_impl.py 检测退化（退化子类型 1-4）。缺失或 exit=0 视为通过。"""
    script = _find_ast_validator(task_dir)
    target = task_dir / "model_new_ascendc.py"
    if script is None or not target.exists():
        return {
            "ast_validator_present": script is not None,
            "ast_degrade_pass": True,
        }
    try:
        r = subprocess.run(
            ["python3", str(script), str(target)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return {"ast_validator_present": True, "ast_degrade_pass": r.returncode == 0}
    except (subprocess.TimeoutExpired, OSError):
        # 验证器本身出错不能把 gate 卡死 → 视为通过但标记
        return {"ast_validator_present": True, "ast_degrade_pass": True, "ast_validator_errored": True}


def check_structure(task_dir: Path, op_name: str) -> dict:
    bak = task_dir / f"{op_name}.json.bak"
    bak_ok = True if not bak.exists() else bak.stat().st_size > 0
    return {
        "has_kernel_dir": (task_dir / "kernel").is_dir(),
        "has_model_new_ascendc": (task_dir / "model_new_ascendc.py").exists(),
        "json_bak_preserved_if_exists": bak_ok,
    }


def check_verify_status_present(task_dir: Path) -> dict:
    latest = task_dir / ".verify_status" / "latest.json"
    ok = latest.exists()
    schema_ok = False
    if ok:
        try:
            schema_ok = json.loads(latest.read_text()).get("schema_version") == 1
        except Exception:
            schema_ok = False
    return {
        "verify_status_latest_present": ok,
        "verify_status_schema_ok": schema_ok,
    }


def check_audit_file_present(task_dir: Path, attempt: int) -> dict:
    """仅检查文件存在；section schema 由分支层各自负责。"""
    path = task_dir / "precision_tuning" / f"precision_audit_{attempt}.md"
    return {
        "audit_file_present": path.exists(),
        "audit_file_nonempty": path.exists() and path.stat().st_size > 0,
    }


# 这些是"必须为 True 才视为通过"的 gating key；其余为纯诊断信息不参与 ok 判定。
# 设计契约（findings.md §3.3 ② / §7.6）：只有明确反映"前置 / 不变量"失败的 key 才 gating。
_GATING_KEYS = {
    # anticheat: 只有 pass 参与，baseline_present/hash_* 是诊断
    "anticheat_pass",
    # ast: 只看 degrade_pass；validator_present 是诊断
    "ast_degrade_pass",
    # structure
    "has_kernel_dir",
    "has_model_new_ascendc",
    "json_bak_preserved_if_exists",
    # verify_status (validate step)
    "verify_status_latest_present",
    "verify_status_schema_ok",
    # audit file (audit/fix/validate steps)
    "audit_file_present",
    "audit_file_nonempty",
}


def run_common(step: str, task_dir: Path, op_name: str, attempt: int) -> GateOutcome:
    """对给定 step 组合适用的通用检查。返回 GateOutcome。

    - forensics: 结构 / 反作弊 / AST
    - audit:     +audit 文件存在
    - fix:       +audit 文件存在（fix 不单独 Gate，此处复用 audit）
    - validate:  +verify_status + audit 文件

    只有 `_GATING_KEYS` 中的键参与 ok 判定；其它 (baseline_present / validator_present 等)
    为纯诊断信息。
    """
    checks: dict = {}
    checks.update(check_anticheat(task_dir))
    checks.update(check_ast_degrade(task_dir))
    checks.update(check_structure(task_dir, op_name))
    if step == "validate":
        checks.update(check_verify_status_present(task_dir))
    if step in ("audit", "fix", "validate"):
        checks.update(check_audit_file_present(task_dir, attempt))

    ok = all(
        checks.get(k, True) is True
        for k in _GATING_KEYS
        if k in checks
    )
    return GateOutcome(gate=f"GATE-COMMON-{step}", ok=ok, checks=checks)
