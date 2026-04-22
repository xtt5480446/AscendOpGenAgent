# Ascend Kernel Developer + AscendC Debug Subagent 集成实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans / superpowers:subagent-driven-development / superpowers:dispatching-parallel-agents 执行本计划。

**Goal:** 把 AscendC 所有可自动修复的失败类型（build/import/runtime/timeout/precision）的 debug 工作从主 agent 循环抽出来交给 subagent 处理；主 agent 负责生成 + 初筛 + 归档。

**Architecture:** 新增 `utils/eval_wrapper.py` 作为 evaluate_ascendc 唯一机器可读事实源；把 `skills/ascendc/precision-tuning/scripts/precision_gate.py` 从精度硬编码重构成"通用层 + 分支层"双层 Gate；新建 `agents/ascend-kernel-developer-with-ascendc-debug.md` 主 agent，在 Phase 8 条件性 spawn `precision-tuning-discovery` subagent；subagent SKILL.md 扩成 5 条 Step 1 分支（P/B/I/R/T），结束时自包含地产出 `debug_trace.md` + `debug_status.json`。

**Tech Stack:** Python 3（eval_wrapper、eval_status、gates、tests）、Bash、Markdown（agent / skill / trace 文档）、`subprocess` / `signal` / `json` / `re` 标准库。

**Source of Truth:** `.planning/findings.md` v3（本计划所有决策以该文档为准；本计划只负责翻译成 bite-sized tasks）。

---

## 并行执行拓扑

```
Stream F (Foundation, 必须先跑)
   F1  utils/eval_wrapper.py 骨架
   F2  failure_type 分类器 + 单测
   F3  eval_status.py loader
      ↓
  ┌─────────┬───────────┬────────────┐
  ↓         ↓           ↓            ↓
Stream G  Stream H   Stream I     Stream J
(Gates)   (SKILL.md) (Subagent    (Trace-
                     agent file)  recorder)
  ↓         ↓           ↓            ↓
  └─────────┴───────────┴────────────┘
                        ↓
                   Stream K  (Main agent 新文件)
                        ↓
                   Stream L  (集成联调 + fixtures)
```

- F 串行（依赖链 F1 → F2 → F3）
- G / H / I / J 可并行（只要 F 完成）
- K 串行（依赖 G / H / I / J 全部完成）
- L 最后

单次并行 dispatch 最多 4 个 agent（G/H/I/J），每个 agent 单独一个 stream，避免同一文件多写。

---

## Common Conventions

- **工作目录**: 本仓库根 `/Users/junming/code/operator/AscendOpGenAgent`（本地编辑），远端 `/home/c00959374/AscendOpGenAgent`（通过 `ssh npu_server` + 容器 `cjm_cann1` 执行）。
- **编辑规则**（来自 CLAUDE.md Code Sync）：**只在本地编辑**，`git commit && git push`，然后在远端 `git fetch && git pull` 后再执行。
- **测试约定**: 本仓库无 pytest 框架；沿用现有 `skills/ascendc/precision-tuning/scripts/test_executor_parity.py` 的 stdlib unittest 风格。新测试放到 `skills/ascendc/precision-tuning/scripts/tests/`，均可 `python3 -m unittest <file>` 独立运行。
- **commit 粒度**: 每个 Task 结束后单独 commit（除非 Task 明确合并）。message 用 `feat: / fix: / refactor: / docs: / test:` 前缀。
- **分支**: 当前 `cjm/debug-v2`，所有改动都在此分支。
- **反作弊不变量**: 任何时候 `model_new_ascendc.py` / `model_new_tilelang.py` / `model.py` 都禁止改。新增 subagent 分支 Gate 同样硬约束只改 `kernel/`。
- **findings.md 去引用**: 本计划每一步在描述理由 / 契约时引用 findings.md 的 Section 编号，便于追溯。

---

# STREAM F — Foundation (串行)

## Task F1: 新建 `utils/eval_wrapper.py` 骨架

**Files:**
- Create: `utils/eval_wrapper.py`
- Create: `utils/__init__.py`（若不存在）

**Context:** 新建一个 Python 包装器统一驱动 `skills/ascendc/ascendc-translator/references/evaluate_ascendc.sh`，用结构化 JSON 记录每次调用的阶段状态。下游（trace-recorder / Phase 8 / subagent Step 0.3 / Gate-V）统一消费它。（findings.md §1.2–1.3）

**Step 1: 写骨架（不含 failure_type 判定，留给 F2）**

```python
#!/usr/bin/env python3
"""eval_wrapper.py — 机器可读包装器，驱动 evaluate_ascendc.sh 并落盘 eval_status.json。

对外约定详见 `.planning/findings.md` §1.2–1.4。下游（trace-recorder / Phase 8 / subagent
Step 0.3 / Gate-V）唯一消费此脚本的输出，不再解析 evaluate_ascendc.sh stdout。
"""
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


def run_once(phase: int, attempt: int, task_dir: Path, timeout_sec: int) -> dict:
    """单次调用。不做 ssh/docker 重试（由 run() 包裹）。"""
    status_dir, logs_dir = _ensure_dirs(task_dir)
    started = _utcnow_iso()
    log_path = logs_dir / f"phase{phase}_attempt{attempt}_{dt.datetime.utcnow():%Y%m%d_%H%M%S}.log"
    timeout_marker = status_dir / "timeout_marker"
    if timeout_marker.exists():
        timeout_marker.unlink()

    # task name: 评测脚本要求以 task 目录名为参数
    task_name = task_dir.name
    cmd = ["bash", str(EVALUATE_SH), task_name]
    env = os.environ.copy()
    # evaluate_ascendc.sh 通过 WORKDIR 查找仓库根
    env["WORKDIR"] = str(REPO_ROOT)

    status = {
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
        "verify": {"status": "skipped", "total_cases": None, "passed_cases": None, "failed_cases": []},
    }

    proc = None
    try:
        proc = subprocess.run(
            cmd, timeout=timeout_sec, env=env, capture_output=True, text=True,
        )
        status["exit_code"] = proc.returncode
        status["stdout_tail"] = _tail(proc.stdout)
        status["stderr_tail"] = _tail(proc.stderr)
        log_path.write_text(f"--- STDOUT ---\n{proc.stdout}\n--- STDERR ---\n{proc.stderr}\n")
    except subprocess.TimeoutExpired as exc:
        timeout_marker.write_text(f"triggered_at={_utcnow_iso()} timeout_sec={timeout_sec}\n")
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
        (dt.datetime.strptime(status["ended_at"], "%Y-%m-%dT%H:%M:%SZ")
         - dt.datetime.strptime(status["started_at"], "%Y-%m-%dT%H:%M:%SZ")).total_seconds(), 2,
    )

    # 解析阶段状态：F2 会填 classify_failure()
    from_classifier = classify_failure(status, proc)  # noqa: F821 - defined in F2
    status.update(from_classifier)

    # 落盘
    status_path = status_dir / f"phase{phase}_attempt{attempt}.json"
    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False))
    latest = status_dir / "latest.json"
    latest.write_text(json.dumps(status, indent=2, ensure_ascii=False))
    return status


def run(phase: int, attempt: int, task_dir: Path, timeout_sec: int) -> dict:
    """对外入口。对 ssh_disconnected / docker_unreachable 做内部 1 次重试。"""
    result = run_once(phase, attempt, task_dir, timeout_sec)
    if result.get("abort_subtype") in ("ssh_disconnected", "docker_unreachable"):
        # 同 attempt 重跑 1 次，覆盖同一份 json
        retry = run_once(phase, attempt, task_dir, timeout_sec)
        if retry.get("abort_subtype") in ("ssh_disconnected", "docker_unreachable"):
            return retry
        return retry
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
    # 退出码：success=0；其它 failure_type=1；致命错误=2
    ft = status.get("failure_type")
    if ft == "success":
        return 0
    if ft in ("execution_aborted",) and status.get("abort_subtype") == "unknown":
        return 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
```

**Step 2: 本地语法检查（不跑，只确认 import）**

Run:
```bash
cd /Users/junming/code/operator/AscendOpGenAgent && python3 -c "import ast; ast.parse(open('utils/eval_wrapper.py').read()); print('ok')"
```
Expected: `ok`

**Step 3: Commit**
```bash
git add utils/eval_wrapper.py
git commit -m "feat(eval_wrapper): add skeleton + schema + timeout_marker + ssh/docker retry"
```

---

## Task F2: 实现 `classify_failure()` + 单测

**Files:**
- Modify: `utils/eval_wrapper.py`（追加函数 `classify_failure`）
- Create: `utils/tests/__init__.py`
- Create: `utils/tests/test_eval_wrapper_classify.py`

**Context:** 按 findings.md §1.4 的判定顺序实现 failure_type / import_subtype / abort_subtype 的纯函数分类器。**只认明确证据，拒绝"signal + 时长近似"推测**。

**Step 1: 追加 `classify_failure`**

在 `utils/eval_wrapper.py` 的 `run_once` 定义之前插入：

```python
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
_SSH_DISCONNECT = [r"ssh: connect to host", r"Connection (refused|reset|timed out)",
                   r"ssh_exchange_identification", r"port 22: Connection"]
_DOCKER_UNREACHABLE = [r"container .* not running", r"Cannot connect to the Docker daemon",
                       r"Error response from daemon", r"docker: Error"]


def _match_any(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text or "") for p in patterns)


def classify_failure(status: dict, proc) -> dict:
    """只改 failure_type / failed_step / import_subtype / abort_subtype / exit_signal / 各阶段子状态。"""
    stdout = status.get("stdout_tail", "")
    stderr = status.get("stderr_tail", "")
    combined = f"{stdout}\n{stderr}"
    out: dict = {}

    # (a) wrapper 自触发 timeout
    if status.get("timeout_marker_present"):
        out["failure_type"] = "timeout"
        out["failed_step"] = "execute"
        out["execute"] = {"status": "timeout", "crash_signal": None}
        return out

    rc = status.get("exit_code")

    # (b) success
    if rc == 0 and _match_any([r"PASS", r"all cases passed"], combined):
        out["failure_type"] = "success"
        out["verify"] = {"status": "passed", "total_cases": None, "passed_cases": None, "failed_cases": []}
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
        out["verify"] = {"status": "failed", "total_cases": None, "passed_cases": None, "failed_cases": []}
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
```

**Step 2: 写单测 `utils/tests/test_eval_wrapper_classify.py`**

```python
"""Unit tests for classify_failure — 对 findings.md §1.4 判定顺序做回归。"""
import unittest
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.eval_wrapper import classify_failure  # type: ignore


def S(**kw):
    base = {
        "stdout_tail": "", "stderr_tail": "", "exit_code": 1,
        "timeout_marker_present": False, "log_path": "/tmp/fake.log",
    }
    base.update(kw)
    return base


class ClassifyTests(unittest.TestCase):
    def test_timeout_marker_wins(self):
        r = classify_failure(S(timeout_marker_present=True, exit_code=124, stderr_tail="whatever"), None)
        self.assertEqual(r["failure_type"], "timeout")

    def test_success_zero_exit(self):
        r = classify_failure(S(exit_code=0, stdout_tail="all cases passed"), None)
        self.assertEqual(r["failure_type"], "success")

    def test_build_failed(self):
        r = classify_failure(S(exit_code=1, stderr_tail="ascendc build failed\nfatal error: xxx"), None)
        self.assertEqual(r["failure_type"], "build_failed")

    def test_import_kernel_side(self):
        r = classify_failure(S(exit_code=1, stderr_tail="ImportError: cannot import _elu_ext\nundefined symbol: elu_do"), None)
        self.assertEqual(r["failure_type"], "import_failed")
        self.assertEqual(r["import_subtype"], "import_kernel_side")

    def test_import_env_side(self):
        r = classify_failure(S(exit_code=1, stderr_tail="ImportError: libascend_hal.so: cannot open shared object file"), None)
        self.assertEqual(r["failure_type"], "import_failed")
        self.assertEqual(r["import_subtype"], "import_env_side")

    def test_runtime_crash(self):
        r = classify_failure(S(exit_code=-signal.SIGSEGV, stderr_tail="..."), None)
        self.assertEqual(r["failure_type"], "runtime_error")
        self.assertEqual(r["execute"]["crash_signal"], "SIGSEGV")

    def test_precision_failed(self):
        r = classify_failure(S(exit_code=1, stdout_tail="mismatch_ratio=2.30% max_abs_diff=1e-2"), None)
        self.assertEqual(r["failure_type"], "precision_failed")

    def test_ssh_disconnect(self):
        r = classify_failure(S(exit_code=255, stderr_tail="ssh: connect to host: Connection refused"), None)
        self.assertEqual(r["failure_type"], "execution_aborted")
        self.assertEqual(r["abort_subtype"], "ssh_disconnected")

    def test_docker_unreachable(self):
        r = classify_failure(S(exit_code=1, stderr_tail="Error response from daemon: container cjm_cann1 not running"), None)
        self.assertEqual(r["failure_type"], "execution_aborted")
        self.assertEqual(r["abort_subtype"], "docker_unreachable")

    def test_unknown_fallback(self):
        r = classify_failure(S(exit_code=42, stderr_tail="some unknown failure"), None)
        self.assertEqual(r["failure_type"], "execution_aborted")
        self.assertEqual(r["abort_subtype"], "unknown")


if __name__ == "__main__":
    unittest.main()
```

**Step 3: 运行单测**

Run:
```bash
cd /Users/junming/code/operator/AscendOpGenAgent && python3 -m unittest utils.tests.test_eval_wrapper_classify -v
```
Expected: 所有 10 个 test PASS。

**Step 4: Commit**
```bash
git add utils/eval_wrapper.py utils/tests/__init__.py utils/tests/test_eval_wrapper_classify.py
git commit -m "feat(eval_wrapper): implement classify_failure with 10-case unit test"
```

---

## Task F3: 新建 `eval_status.py` loader

**Files:**
- Create: `skills/ascendc/precision-tuning/scripts/eval_status.py`

**Context:** 给下游 Gate / trace-recorder / subagent 提供统一读取接口。（findings.md §3.4）

**Step 1: 实现**

```python
"""eval_status.py — eval_wrapper 产物的读取 / 校验 helper.

对外 API:
  load_eval_status(task_dir: Path, phase: int, attempt: int) -> dict
  load_latest_status(task_dir: Path) -> dict
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
REQUIRED_KEYS = {
    "schema_version", "phase", "attempt", "failure_type",
    "duration_sec", "exit_code", "log_path",
}


def _validate(status: dict) -> dict:
    missing = REQUIRED_KEYS - set(status.keys())
    if missing:
        raise ValueError(f"eval_status missing keys: {sorted(missing)}")
    if status["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unexpected schema_version {status['schema_version']}")
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


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", type=Path, required=True)
    ap.add_argument("--phase", type=int, default=None)
    ap.add_argument("--attempt", type=int, default=0)
    args = ap.parse_args()
    if args.phase is None:
        print(json.dumps(load_latest_status(args.task_dir), indent=2))
    else:
        print(json.dumps(load_eval_status(args.task_dir, args.phase, args.attempt), indent=2))
```

**Step 2: 语法检查 + Commit**

Run:
```bash
cd /Users/junming/code/operator/AscendOpGenAgent && python3 -c "import ast; ast.parse(open('skills/ascendc/precision-tuning/scripts/eval_status.py').read()); print('ok')"
git add skills/ascendc/precision-tuning/scripts/eval_status.py
git commit -m "feat(precision-tuning): add eval_status.py loader/validator"
```

---

# STREAM G — Gate 2 层重构 (并行，依赖 F3)

## Task G1: 建立 `gates/` 包

**Files:**
- Create: `skills/ascendc/precision-tuning/scripts/gates/__init__.py`

**Step 1: 写入 `__init__.py`**

```python
"""Gate 层包。

- common.py:        所有分支共享的通用层校验（反作弊、文件存在、baseline、eval_status）
- branch_precision: 精度（原 precision_gate.py 精度逻辑抽离）
- branch_build:     编译失败
- branch_import:    import 失败（仅 kernel_side）
- branch_runtime:   运行时段错误
- branch_timeout:   超时
"""
```

**Step 2: Commit**
```bash
git add skills/ascendc/precision-tuning/scripts/gates/__init__.py
git commit -m "refactor(gates): create gates package"
```

---

## Task G2: `gates/common.py` — 通用层

**Files:**
- Create: `skills/ascendc/precision-tuning/scripts/gates/common.py`

**Context:** 通用层只做**所有分支都成立的不变量**：反作弊 hash、AST 退化、文件存在、`eval_status` 产出、baseline、目录完整性。**不强制 audit section schema**（避免和精度现有 `[FORENSICS_SUMMARY]+[REFERENCE_IMPL_SPEC]` 打架——findings.md §3.3 ②, §7.6）。

**Step 1: 实现**

```python
"""common.py — Gate 通用层。

契约（findings.md §3.3 ② / §7.6）:
  - 反作弊 hash 未破坏
  - AST 退化未引入
  - task_dir 目录结构完整
  - eval_status.json 产出存在且 schema_version 正确
  - {op}.json.bak 未被破坏
  - audit_{attempt}.md 文件存在（section schema 由分支层各自定义）

不检查 audit section 格式；不在 fix 步骤单独加 Gate——由下一轮 Gate-V 通过 eval_status 差分间接验证。
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GateOutcome:
    gate: str
    ok: bool
    checks: dict
    loop_signal: str | None = None
    reason: str | None = None

    def to_gate_output(self) -> dict:
        out = {"gate": self.gate, "passed": self.ok, "checks": self.checks}
        if self.loop_signal is not None:
            out["loop_signal"] = self.loop_signal
        if self.reason is not None:
            out["loop_reason"] = self.reason
        return out


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_anticheat(task_dir: Path) -> dict:
    """对 model_new_ascendc.py / model_new_tilelang.py 对比 .bench_baseline/ 的 hash。
    若 .bench_baseline/ 不存在（非 bench 环境），跳过。"""
    baseline_dir = task_dir / ".bench_baseline"
    if not baseline_dir.is_dir():
        return {"anticheat_baseline_present": False, "anticheat_pass": True}
    result = {"anticheat_baseline_present": True}
    for name in ("model_new_ascendc.py", "model_new_tilelang.py"):
        bp = baseline_dir / name
        cp = task_dir / name
        if not bp.exists():
            continue
        result[f"hash_{name}"] = _sha256(bp) == _sha256(cp)
    result["anticheat_pass"] = all(v for k, v in result.items() if k.startswith("hash_"))
    return result


def check_ast_degrade(task_dir: Path) -> dict:
    """调用 validate_ascendc_impl.py 检测退化（退化子类型 1-4）。缺失或 exit=0 视为通过。"""
    import subprocess
    script = task_dir.parents[0] / "skills" / "ascendc" / "ascendc-translator" / "scripts" / "validate_ascendc_impl.py"
    # 兼容 REPO_ROOT 推断
    for cand in (task_dir.parents[0], task_dir.parents[1], task_dir.parents[2]):
        p = cand / "skills" / "ascendc" / "ascendc-translator" / "scripts" / "validate_ascendc_impl.py"
        if p.exists():
            script = p
            break
    target = task_dir / "model_new_ascendc.py"
    if not script.exists() or not target.exists():
        return {"ast_validator_present": script.exists(), "ast_degrade_pass": True}
    r = subprocess.run(["python3", str(script), str(target)], capture_output=True, text=True)
    return {"ast_validator_present": True, "ast_degrade_pass": r.returncode == 0}


def check_structure(task_dir: Path, op_name: str) -> dict:
    return {
        "has_kernel_dir": (task_dir / "kernel").is_dir(),
        "has_model_new_ascendc": (task_dir / "model_new_ascendc.py").exists(),
        "json_bak_preserved_if_exists": True if not (task_dir / f"{op_name}.json.bak").exists()
                                        else (task_dir / f"{op_name}.json.bak").stat().st_size > 0,
    }


def check_eval_status_present(task_dir: Path) -> dict:
    latest = task_dir / ".eval_status" / "latest.json"
    ok = latest.exists()
    schema_ok = False
    if ok:
        try:
            schema_ok = json.loads(latest.read_text()).get("schema_version") == 1
        except Exception:
            schema_ok = False
    return {"eval_status_latest_present": ok, "eval_status_schema_ok": schema_ok}


def check_audit_file_present(task_dir: Path, attempt: int) -> dict:
    """仅检查文件存在；section schema 由分支层各自负责。"""
    path = task_dir / "precision_tuning" / f"precision_audit_{attempt}.md"
    return {"audit_file_present": path.exists(), "audit_file_nonempty": path.exists() and path.stat().st_size > 0}


def run_common(step: str, task_dir: Path, op_name: str, attempt: int) -> GateOutcome:
    """对给定 step 组合适用的通用检查。返回 GateOutcome。"""
    checks: dict = {}
    checks.update(check_anticheat(task_dir))
    checks.update(check_ast_degrade(task_dir))
    checks.update(check_structure(task_dir, op_name))
    if step in ("validate",):
        checks.update(check_eval_status_present(task_dir))
    if step in ("audit", "fix", "validate"):
        checks.update(check_audit_file_present(task_dir, attempt))
    ok = all(v if isinstance(v, bool) else True for v in checks.values())
    return GateOutcome(gate=f"GATE-COMMON-{step}", ok=ok, checks=checks)
```

**Step 2: Commit**
```bash
git add skills/ascendc/precision-tuning/scripts/gates/common.py
git commit -m "feat(gates): add common layer (anticheat/ast/structure/eval_status/audit-file)"
```

---

## Task G3: `gates/branch_precision.py` — 从原 `precision_gate.py` 抽精度逻辑

**Files:**
- Modify: 读 `skills/ascendc/precision-tuning/scripts/precision_gate.py`（不改，抽走精度语义）
- Create: `skills/ascendc/precision-tuning/scripts/gates/branch_precision.py`

**Context:** findings.md §3.3 ③。精度分支原有 audit section（`[FORENSICS_SUMMARY]` / `[REFERENCE_IMPL_SPEC]` / `[ROOT_CAUSE]` / `[FIX_PLAN]` 等 7 个）不改，`mismatch_ratio` / `max_abs_diff` 趋势、`loop_signal` 计算照搬。本 Task 的目标是**语义 1:1 搬移**，不做功能新增；原 `precision_gate.py` 的方法会在 Task G9（Router 重构）改成"通用 + 精度分支"的派发方式。

**Step 1: 分析原 `precision_gate.py` 里哪些方法属于精度语义（必须抽走）**

从 `precision_gate.py:39-1018` `class GateChecker`，按 findings 要求拆：
- 属于通用层（已在 G2）：anticheat、AST、file-exists 层面检查
- 属于精度分支（本 Task 抽走）：
  - `check_forensics()` / `_write_baseline_from_forensics()`
  - `check_audit()`（7 个 section 检查）
  - `check_validate()` + `_compute_loop_signal()` + `_count_stagnant()` + `_detect_harmful_regression()` + `_compute_improvement_ratio()` + `_write_round_summary()` + `_write_tuning_directions()` + `_extract_direction_*()` + `_check_direction_assessment()` + `_extract_section()` + `_extract_fix_type()` + `_extract_changed_locations()` + `_write_audit_index()` + `_get_baseline_match_rate()` + `_kernel_dir()` + `_check_import_name_match()` + `_result()`
- `check_fix()`：通用层已覆盖"文件存在" → Gate 合并到 common；原 `check_fix` 里对 kernel 目录的完整性检查也并入 common

**Step 2: 新建 `gates/branch_precision.py`**

把上述精度相关方法**原样**复制到新文件，包装成一个 `PrecisionBranch` class，入口方法：
- `run_gate_f(task_dir, op_name, attempt) -> GateOutcome` — 前向 `check_forensics`
- `run_gate_a(task_dir, op_name, attempt) -> GateOutcome` — 前向 `check_audit` 的 section 部分（通用文件存在已在 common）
- `run_gate_v(task_dir, op_name, attempt) -> GateOutcome` — 前向 `check_validate` + loop_signal

保持以下兼容：
- 写文件路径不变（`baseline_state.json` / `round_summary_{attempt}.json` / `tuning_directions.json`）
- 7 个 audit section 名不变
- `loop_signal` 取值集不变（`PASS` / `CONTINUE` / `STOP`）

**原则：只搬不改。** 如在搬移过程中发现 bug，记录下来但**不修**，留给后续 PR。

**Step 3: 写回归测试 `utils/tests/test_branch_precision_parity.py`**

对 2 个真实 baseline（选 2 个 `archive_tasks/` 下含 precision_tuning/ 的任务），分别跑旧 `precision_gate.py --step forensics|audit|validate` 和新 `branch_precision.run_gate_*`，对比 JSON 输出。写成 `assertEqual` 的 snapshot test。若当前 repo 无此类 baseline，**本步骤降级为**：编写一份 stub test，给两条空路径确保 importable，标 `@unittest.skip("needs live baseline for full parity")`。

```python
"""branch_precision parity test (snapshot 占位). 真正的 parity 在 Stream L 集成联调时做。"""
import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "skills" / "ascendc" / "precision-tuning" / "scripts"))


class ImportableOnly(unittest.TestCase):
    def test_branch_precision_importable(self):
        from gates import branch_precision  # noqa: F401
        self.assertTrue(hasattr(branch_precision, "PrecisionBranch"))


if __name__ == "__main__":
    unittest.main()
```

**Step 4: Commit**
```bash
git add skills/ascendc/precision-tuning/scripts/gates/branch_precision.py utils/tests/test_branch_precision_parity.py
git commit -m "feat(gates): extract precision logic into gates/branch_precision (no behavior change)"
```

---

## Task G4: `gates/branch_build.py` — 新增

**Files:**
- Create: `skills/ascendc/precision-tuning/scripts/gates/branch_build.py`

**Context:** findings.md §3.3 ④。新分支，处理 `failure_type == build_failed`。

**Step 1: 实现**

```python
"""branch_build.py — 编译失败分支 Gate.

契约（findings.md §3.3 ④）:
  - Gate-F: .eval_logs/ 内最新 build log 存在，含 compile 错误块
  - Gate-A: audit 含 [COMPILE_ERROR_CITATION] [ROOT_CAUSE] [FIX_PLAN]
            fix_type ∈ build_fix_whitelist
  - Gate-V: 新一轮 eval_status.failed_step 从 compile 推进到 execute/verify
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from .common import GateOutcome

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

    def run_gate_f(self, task_dir: Path, attempt: int) -> GateOutcome:
        latest = task_dir / ".eval_status" / "latest.json"
        checks = {"latest_present": latest.exists()}
        if latest.exists():
            status = json.loads(latest.read_text())
            checks["failure_type_is_build"] = status.get("failure_type") == "build_failed"
            checks["failed_step_is_compile"] = status.get("failed_step") == "compile"
            log_path = Path(status.get("log_path", ""))
            checks["build_log_present"] = log_path.exists()
            if log_path.exists():
                checks["has_error_block"] = bool(re.search(r"error:|fatal error", log_path.read_text()))
        return GateOutcome("GATE-BUILD-F", all(checks.values()), checks)

    def run_gate_a(self, task_dir: Path, attempt: int) -> GateOutcome:
        audit = task_dir / "precision_tuning" / f"precision_audit_{attempt}.md"
        checks = {"audit_exists": audit.exists()}
        if audit.exists():
            content = audit.read_text()
            for sec in REQUIRED_SECTIONS:
                checks[f"has_{sec.strip('[]').lower()}"] = sec in content
            m = re.search(r"\[FIX_TYPE\]\s*:?\s*(\w+)", content)
            checks["fix_type_in_whitelist"] = bool(m and m.group(1) in BUILD_FIX_WHITELIST)
        return GateOutcome("GATE-BUILD-A", all(checks.values()), checks)

    def run_gate_v(self, task_dir: Path, attempt: int) -> GateOutcome:
        # 比较最近两次 eval_status
        prev = task_dir / ".eval_status" / f"phase8_attempt{attempt-1}.json" if attempt > 0 else None
        curr = task_dir / ".eval_status" / f"phase8_attempt{attempt}.json"
        checks = {"curr_present": curr.exists()}
        if curr.exists():
            c = json.loads(curr.read_text())
            checks["curr_failed_step"] = c.get("failed_step")
            progressed = c.get("failed_step") in ("execute", "verify", None) and c.get("failure_type") != "build_failed"
            checks["progressed_past_compile"] = progressed
        loop_signal = "CONTINUE"
        if checks.get("progressed_past_compile"):
            if checks.get("curr_failed_step") is None and json.loads(curr.read_text()).get("failure_type") == "success":
                loop_signal = "PASS"
            else:
                loop_signal = "CONTINUE"  # 进展了但还没最终通过 → 结束本 session（跨分支不切换）
        else:
            loop_signal = "STOP" if attempt >= 1 else "CONTINUE"
        return GateOutcome("GATE-BUILD-V", True if loop_signal in ("PASS", "CONTINUE") else False,
                           checks, loop_signal=loop_signal,
                           reason="build progress tracked via failed_step transition")
```

**Step 2: Commit**
```bash
git add skills/ascendc/precision-tuning/scripts/gates/branch_build.py
git commit -m "feat(gates): add build failure branch (Gate-F/A/V)"
```

---

## Task G5: `gates/branch_import.py` — kernel_side only

**Files:**
- Create: `skills/ascendc/precision-tuning/scripts/gates/branch_import.py`

**Context:** findings.md §3.3 ⑤。只处理 `import_kernel_side`。env_side 在主 agent Phase 7 已过滤掉；若异常进入，Gate-F 直接 reject。

**Step 1: 实现**

```python
"""branch_import.py — import 失败分支 Gate（仅 kernel_side）。"""
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
    "ld_path_fix", "abi_fix", "toolkit_env_fix",
    "cmakelists_fix", "setup_py_fix", "build_ascendc_fix",
}
REQUIRED_SECTIONS = ("[IMPORT_TRACEBACK_CITATION]", "[ROOT_CAUSE]", "[FIX_PLAN]")


class ImportBranch:

    def run_gate_f(self, task_dir: Path, attempt: int) -> GateOutcome:
        latest = task_dir / ".eval_status" / "latest.json"
        checks = {"latest_present": latest.exists()}
        if latest.exists():
            status = json.loads(latest.read_text())
            checks["failure_type_is_import"] = status.get("failure_type") == "import_failed"
            checks["import_subtype_is_kernel"] = status.get("import_subtype") == "import_kernel_side"
            log = Path(status.get("log_path", ""))
            checks["traceback_log_present"] = log.exists()
        return GateOutcome("GATE-IMPORT-F", all(checks.values()), checks)

    def run_gate_a(self, task_dir: Path, attempt: int) -> GateOutcome:
        audit = task_dir / "precision_tuning" / f"precision_audit_{attempt}.md"
        checks = {"audit_exists": audit.exists()}
        if audit.exists():
            content = audit.read_text()
            for sec in REQUIRED_SECTIONS:
                checks[f"has_{sec.strip('[]').lower()}"] = sec in content
            m = re.search(r"\[FIX_TYPE\]\s*:?\s*(\w+)", content)
            fix_type = m.group(1) if m else None
            checks["fix_type_in_kernel_whitelist"] = bool(fix_type and fix_type in IMPORT_KERNEL_FIX_WHITELIST)
            checks["fix_type_not_env"] = fix_type not in BLOCKED_FIX_TYPES
        return GateOutcome("GATE-IMPORT-A", all(checks.values()), checks)

    def run_gate_v(self, task_dir: Path, attempt: int) -> GateOutcome:
        curr = task_dir / ".eval_status" / f"phase8_attempt{attempt}.json"
        checks = {"curr_present": curr.exists()}
        loop_signal = "CONTINUE"
        if curr.exists():
            c = json.loads(curr.read_text())
            checks["import_passed"] = c.get("import", {}).get("status") == "passed"
            if c.get("failure_type") == "success":
                loop_signal = "PASS"
            elif checks["import_passed"]:
                loop_signal = "CONTINUE"
            else:
                loop_signal = "STOP" if attempt >= 1 else "CONTINUE"
        return GateOutcome("GATE-IMPORT-V", loop_signal != "STOP", checks,
                           loop_signal=loop_signal,
                           reason="import.status transitioned to passed")
```

**Step 2: Commit**
```bash
git add skills/ascendc/precision-tuning/scripts/gates/branch_import.py
git commit -m "feat(gates): add import failure branch (kernel_side only)"
```

---

## Task G6: `gates/branch_runtime.py`

**Files:**
- Create: `skills/ascendc/precision-tuning/scripts/gates/branch_runtime.py`

**Step 1: 实现**

```python
"""branch_runtime.py — 运行时段错误分支 Gate."""
from __future__ import annotations

import json
from pathlib import Path
from .common import GateOutcome

REQUIRED_SECTIONS = ("[RUNTIME_ERROR_CITATION]", "[ROOT_CAUSE]", "[FIX_PLAN]")


class RuntimeBranch:

    def run_gate_f(self, task_dir: Path, attempt: int) -> GateOutcome:
        latest = task_dir / ".eval_status" / "latest.json"
        checks = {"latest_present": latest.exists()}
        if latest.exists():
            s = json.loads(latest.read_text())
            checks["failure_type_is_runtime"] = s.get("failure_type") == "runtime_error"
            checks["crash_signal_recorded"] = bool(s.get("execute", {}).get("crash_signal"))
            checks["log_path_exists"] = Path(s.get("log_path", "")).exists()
        return GateOutcome("GATE-RUNTIME-F", all(checks.values()), checks)

    def run_gate_a(self, task_dir: Path, attempt: int) -> GateOutcome:
        audit = task_dir / "precision_tuning" / f"precision_audit_{attempt}.md"
        checks = {"audit_exists": audit.exists()}
        if audit.exists():
            content = audit.read_text()
            for sec in REQUIRED_SECTIONS:
                checks[f"has_{sec.strip('[]').lower()}"] = sec in content
        return GateOutcome("GATE-RUNTIME-A", all(checks.values()), checks)

    def run_gate_v(self, task_dir: Path, attempt: int) -> GateOutcome:
        curr = task_dir / ".eval_status" / f"phase8_attempt{attempt}.json"
        prev = task_dir / ".eval_status" / f"phase8_attempt{attempt-1}.json" if attempt > 0 else None
        checks = {"curr_present": curr.exists()}
        loop_signal = "CONTINUE"
        if curr.exists():
            c = json.loads(curr.read_text())
            curr_crash = c.get("execute", {}).get("crash_signal")
            checks["no_longer_crashed"] = curr_crash is None
            if c.get("failure_type") == "success":
                loop_signal = "PASS"
            elif checks["no_longer_crashed"]:
                loop_signal = "CONTINUE"
            elif prev and prev.exists():
                prev_crash = json.loads(prev.read_text()).get("execute", {}).get("crash_signal")
                checks["crash_signal_changed"] = prev_crash != curr_crash
                loop_signal = "CONTINUE" if checks["crash_signal_changed"] else "STOP"
            else:
                loop_signal = "CONTINUE"
        return GateOutcome("GATE-RUNTIME-V", loop_signal != "STOP", checks,
                           loop_signal=loop_signal,
                           reason="runtime crash progression")
```

**Step 2: Commit**
```bash
git add skills/ascendc/precision-tuning/scripts/gates/branch_runtime.py
git commit -m "feat(gates): add runtime crash branch"
```

---

## Task G7: `gates/branch_timeout.py`

**Files:**
- Create: `skills/ascendc/precision-tuning/scripts/gates/branch_timeout.py`

**Step 1: 实现**

```python
"""branch_timeout.py — 超时分支 Gate."""
from __future__ import annotations

import json
from pathlib import Path
from .common import GateOutcome

REQUIRED_SECTIONS = ("[SYNC_POINT_ANALYSIS]", "[ROOT_CAUSE]", "[FIX_PLAN]")
TIMEOUT_FIX_KEYWORDS = {"sync", "SyncAll", "tiling", "barrier", "pipe"}


class TimeoutBranch:

    def run_gate_f(self, task_dir: Path, attempt: int) -> GateOutcome:
        latest = task_dir / ".eval_status" / "latest.json"
        checks = {"latest_present": latest.exists()}
        if latest.exists():
            s = json.loads(latest.read_text())
            checks["failure_type_is_timeout"] = s.get("failure_type") == "timeout"
            checks["timeout_marker_present"] = bool(s.get("timeout_marker_present"))
        return GateOutcome("GATE-TIMEOUT-F", all(checks.values()), checks)

    def run_gate_a(self, task_dir: Path, attempt: int) -> GateOutcome:
        audit = task_dir / "precision_tuning" / f"precision_audit_{attempt}.md"
        checks = {"audit_exists": audit.exists()}
        if audit.exists():
            content = audit.read_text()
            for sec in REQUIRED_SECTIONS:
                checks[f"has_{sec.strip('[]').lower()}"] = sec in content
            checks["fix_plan_mentions_sync_or_tiling"] = any(k in content for k in TIMEOUT_FIX_KEYWORDS)
        return GateOutcome("GATE-TIMEOUT-A", all(checks.values()), checks)

    def run_gate_v(self, task_dir: Path, attempt: int) -> GateOutcome:
        curr = task_dir / ".eval_status" / f"phase8_attempt{attempt}.json"
        checks = {"curr_present": curr.exists()}
        loop_signal = "CONTINUE"
        if curr.exists():
            c = json.loads(curr.read_text())
            no_timeout = c.get("failure_type") != "timeout" and not c.get("timeout_marker_present")
            checks["no_longer_timeout"] = no_timeout
            if c.get("failure_type") == "success":
                loop_signal = "PASS"
            elif no_timeout:
                loop_signal = "CONTINUE"
            else:
                loop_signal = "STOP" if attempt >= 1 else "CONTINUE"
        return GateOutcome("GATE-TIMEOUT-V", loop_signal != "STOP", checks,
                           loop_signal=loop_signal,
                           reason="timeout presence progression")
```

**Step 2: Commit**
```bash
git add skills/ascendc/precision-tuning/scripts/gates/branch_timeout.py
git commit -m "feat(gates): add timeout branch"
```

---

## Task G8: 把 `precision_gate.py` 改成路由器

**Files:**
- Modify: `skills/ascendc/precision-tuning/scripts/precision_gate.py`

**Context:** findings.md §3.3 ①。保持原有 CLI（`--step forensics|audit|fix|validate --op-name --task-name --attempt`）向后兼容——对"未带 `failure_type`"的旧调用默认按 `precision_failed` 分支处理。

**Step 1: 在 `main()` 前插入派发逻辑（保留 `class GateChecker` 原样留到 Task G9 做最终清理；本 step 先加一层派发）**

替换 `main()` 函数为：

```python
from gates import common as _gate_common
from gates.branch_precision import PrecisionBranch
from gates.branch_build import BuildBranch
from gates.branch_import import ImportBranch
from gates.branch_runtime import RuntimeBranch
from gates.branch_timeout import TimeoutBranch

try:
    from eval_status import load_latest_status  # type: ignore
except ImportError:
    def load_latest_status(task_dir):  # fallback
        return {}


_BRANCHES = {
    "precision_failed": PrecisionBranch,
    "build_failed":     BuildBranch,
    "import_failed":    ImportBranch,
    "runtime_error":    RuntimeBranch,
    "timeout":          TimeoutBranch,
}


def _select_branch(failure_type: str | None):
    return _BRANCHES.get(failure_type or "precision_failed", PrecisionBranch)()


def _dispatch(step: str, task_dir, op_name: str, attempt: int) -> dict:
    # 先跑通用层
    common_outcome = _gate_common.run_common(step, task_dir, op_name, attempt)
    if not common_outcome.ok:
        return common_outcome.to_gate_output()
    # 再根据 failure_type 派发
    try:
        latest = load_latest_status(task_dir)
        failure_type = latest.get("failure_type")
    except Exception:
        failure_type = None
    branch = _select_branch(failure_type)
    method = {
        "forensics": getattr(branch, "run_gate_f", None),
        "audit":     getattr(branch, "run_gate_a", None),
        "fix":       getattr(branch, "run_gate_a", None),   # fix 步骤不再单独 gate（findings §3.3）
        "validate":  getattr(branch, "run_gate_v", None),
    }[step]
    if method is None:
        return common_outcome.to_gate_output()
    branch_outcome = method(task_dir=task_dir, attempt=attempt) if step != "forensics" \
                     else method(task_dir=task_dir, attempt=attempt)
    return branch_outcome.to_gate_output()


def main():
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description="精度调优 Gate 验证 (2 层路由)")
    parser.add_argument("--step", required=True, choices=["forensics", "audit", "fix", "validate"])
    parser.add_argument("--op-name", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--task-dir", default=None)
    parser.add_argument("--attempt", type=int, default=0)
    args = parser.parse_args()
    task_dir = Path(args.task_dir) if args.task_dir else (REPO_ROOT / args.task_name)
    result = _dispatch(args.step, task_dir, args.op_name, args.attempt)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    sys.exit(0 if result.get("passed") else 1)
```

**Step 2: 注意事项**
- 旧的 `class GateChecker` 暂时保留（供 PrecisionBranch 内部 wrapping 复用），**不删**；最终清理放到 Task G9。
- 精度分支 Gate 方法签名要和 BuildBranch 对齐（`run_gate_f(task_dir, attempt)` / `run_gate_a(task_dir, attempt)` / `run_gate_v(task_dir, attempt)`）。如签名不一致，在 PrecisionBranch 内部做适配。

**Step 3: 手动冒烟**（本地，不依赖 NPU）

Run:
```bash
cd /Users/junming/code/operator/AscendOpGenAgent
python3 -c "from skills.ascendc.precision-tuning.scripts.gates import common, branch_precision, branch_build, branch_import, branch_runtime, branch_timeout; print('imports ok')" 2>/dev/null || \
  PYTHONPATH=skills/ascendc/precision-tuning/scripts python3 -c "from gates import common, branch_precision, branch_build, branch_import, branch_runtime, branch_timeout; print('imports ok')"
```
Expected: `imports ok`

```bash
cd /Users/junming/code/operator/AscendOpGenAgent
python3 skills/ascendc/precision-tuning/scripts/precision_gate.py --help
```
Expected: 显示 argparse help，不报 import error

**Step 4: Commit**
```bash
git add skills/ascendc/precision-tuning/scripts/precision_gate.py
git commit -m "refactor(precision_gate): convert into 2-layer router (common + branch_*)"
```

---

## Task G9: PrecisionBranch 适配 + 清理

**Files:**
- Modify: `skills/ascendc/precision-tuning/scripts/gates/branch_precision.py`
- Modify: `skills/ascendc/precision-tuning/scripts/precision_gate.py`（只清理死代码）

**Step 1:** 在 PrecisionBranch 里包装 legacy `GateChecker`，确保 `run_gate_f/a/v` 签名统一。

**Step 2:** 删除 `precision_gate.py` 中已迁移到 `branch_precision.py` 的死代码（谨慎，留 `REPO_ROOT` 常量给其它脚本 import 使用）。

**Step 3:** 重跑 Task G3 的 snapshot parity 测试（若有真实 baseline 即可验证；否则仅 import 级别）。

**Step 4: Commit**
```bash
git add -u
git commit -m "refactor(precision_gate): drop migrated legacy code, keep REPO_ROOT export"
```

---

# STREAM H — precision-tuning SKILL.md 重写 (并行，依赖 F3)

## Task H1: frontmatter 改 timeout 3600 → 5400

**Files:**
- Modify: `skills/ascendc/precision-tuning/SKILL.md:6-14`

```diff
 subagent:
   enabled: true
-  agent_type: general
-  reason: >
-    精度调优涉及取证→深度分析→修复→验证的多步循环,
-    需要 Agent 结合数值证据和代码理解做深度推理。
-    使用 subagent 允许自主多步执行和错误恢复。
-  timeout: 3600
+  agent_type: general
+  reason: >
+    覆盖 AscendC build/import/runtime/timeout/precision 五类失败。
+    每类失败都涉及取证→深度分析→修复→验证的多步循环,
+    subagent 内部锁定一条分支, 不跨分支跳转。
+  timeout: 5400
   max_iterations: 60
```

Commit: `docs(precision-tuning): bump subagent timeout to 5400s + update reason`

---

## Task H2: "What I do" / "When to use me" / "Prerequisites"

**Files:**
- Modify: `skills/ascendc/precision-tuning/SKILL.md:17-41`

重写成：

```markdown
## What I do

修复 AscendC 算子的 build / import / runtime / timeout / precision 五类失败。流程:
1. 读取 `.eval_status/latest.json` 确定 failure_type, 分流到对应 Step 1 分支
2. Agent 结合上下文 + 代码 + 知识库做深度分析, 定位根因并制定修复计划
3. Agent 修复代码
4. 重新编译 + 验证
5. 根据 Gate 的循环控制信号决定继续或停止

## When to use me

当主 agent Phase 7 判定 `debug_eligible == true`（含 failure_type 白名单判断）时。

## Prerequisites（通用，所有分支共享）

- `{task_dir}/kernel/` 下至少一个 `.cpp` 文件
- `{task_dir}/model_new_ascendc.py` 未 AST 退化
- `{task_dir}/trace.md` 末尾含 `final_status` JSON block
- `{task_dir}/.eval_status/latest.json` 存在
- `{task_dir}/{op_name}.json`（及可选 `.json.bak`）存在

其中 `task_dir = {repo_root}/{task_name}`。
```

Commit: `docs(precision-tuning): broaden scope to 5 failure categories`

---

## Task H3: 新增 Step 0.3（分流 + session_branch 锁定）

**Files:**
- Modify: `skills/ascendc/precision-tuning/SKILL.md`（在 Step 0.2 之后、Step 1 之前插入 Step 0.3）

新段落内容（原文复制 findings.md §3.2.4）：

```markdown
### Step 0.3: 读 final_status + eval_status，锁定 session_branch

```bash
python3 skills/ascendc/precision-tuning/scripts/eval_status.py \
    --task-dir {task_dir} | jq '.failure_type,.import_subtype'
awk '/^```json/,/^```$/' "{task_dir}/trace.md" | jq '.debug_eligible, .failure_type'
```

**分支选择规则（本 session 唯一锁定，不可切换）**:
- 入口按 `final_status.failure_type` 选定 `session_branch`
- `import_failed` 还要读 `final_status.import_subtype`：
  - `import_kernel_side` → 进入 Step 1-I
  - `import_env_side` → 异常情况（主 agent 已过滤），直接写 `debug_trace.md` + `debug_status.json` 标 `skipped_env_issue` 后退出

**跨分支跳转禁止**:
- 若某轮修复后 `eval_status.failure_type` 变化（如 `build_failed` → `precision_failed`），视为"本分支 Gate-V 取得进展"
- **不切换分支**，本次 session 结束；`debug_trace.md` / `debug_status.json` 标 `session_outcome: progressed_to_new_failure_type`
- 主 agent 本版本不自动二次 spawn（人工判断）

**根据 `session_branch` 选择 Step 1 分支**:
| session_branch | failure_type | 进入 |
|---|---|---|
| 1-P | precision_failed | Step 1-P（现有路径） |
| 1-B | build_failed | Step 1-B |
| 1-I | import_failed + import_kernel_side | Step 1-I |
| 1-R | runtime_error | Step 1-R |
| 1-T | timeout | Step 1-T |
| — | 其他 | 写 debug_status skipped_unsupported_type, 退出 |
```

Commit: `docs(precision-tuning): add Step 0.3 branch dispatch with session_branch lock`

---

## Task H4: 新增 Step 1-B (Build)

**Files:**
- Modify: `skills/ascendc/precision-tuning/SKILL.md`（在 Step 1 之后，作为 Step 1 的平级分支；原 Step 1 重命名为 Step 1-P）

**Step 1: 原 Step 1 重命名**（只改标题）：`### Step 1: 精度取证` → `### Step 1-P: 精度取证（precision_failed 分支）`

**Step 2: 在 Step 1-P 后追加 Step 1-B / 1-I / 1-R / 1-T（每条 ~30 行，参考 findings.md §3.2.5 模板）**

Step 1-B 示例：

```markdown
### Step 1-B: Build Error Analysis（build_failed 分支）

**输入**:
- `{task_dir}/.eval_status/latest.json`  — 结构化状态 + build log 路径
- `{task_dir}/.eval_logs/phase{N}_attempt{M}.log`  — 原始 build log
- `{task_dir}/kernel/*.cpp` / `*.h`
- `{task_dir}/trace.md`  — Phase 1-7 上下文

**Agent 任务**:
1. 读 build log，提取 compile error / fatal error / undefined reference 块（最多 10 行）
2. 对照 `skills/ascendc/ascendc-translator/references/dsl2Ascendc_compute_*.md`、`dsl2Ascendc_host.md` 找 API 用法差异
3. 写 `{task_dir}/precision_tuning/precision_audit_{attempt}.md` 含:
   - `[COMPILE_ERROR_CITATION]` — 原文摘录 error 块 + 代码行号
   - `[ROOT_CAUSE]` — 根因
   - `[FIX_PLAN]` — 文件/函数/行号级修改列表
   - `[FIX_TYPE]` — 必须 ∈ `{api_usage_fix, template_arg_fix, include_fix, signature_align_fix, pipe_queue_fix, tilingdata_field_fix}`
4. 修复 `kernel/*.cpp` / `*.h`
5. 通过 Gate-通用 + Gate-BUILD-A 验证：
   ```bash
   python3 skills/ascendc/precision-tuning/scripts/precision_gate.py \
       --step audit --op-name {op_name} --task-name {task_name} --attempt {attempt}
   ```

**推荐参考资料**: `skills/ascendc/ascendc-translator/references/dsl2Ascendc*.md`、`TileLang-AscendC-API-Mapping.md`、`AscendC_knowledge/api_reference/`

**Step 4（共用）**: 调 `utils/eval_wrapper.py --phase 8 --attempt {attempt} --task-dir {task_dir}` 重跑，然后走 `Gate-BUILD-V`。`failed_step` 从 `compile` 推进到 `execute`/`verify`/无 = 进步（跨分支不切换）。
```

**Step 3: Step 1-I / 1-R / 1-T 按同样模板写入**（findings.md §3.2.5 给出分支专属 section 名称：`[IMPORT_TRACEBACK_CITATION]` / `[RUNTIME_ERROR_CITATION]` / `[SYNC_POINT_ANALYSIS]`）。

Commit: `docs(precision-tuning): add Step 1-B/I/R/T branches (build/import/runtime/timeout)`

---

## Task H5: Step 5 结束时写 `debug_trace.md` + `debug_status.json`

**Files:**
- Modify: `skills/ascendc/precision-tuning/SKILL.md`（在 Step 5 "成功收尾"之前插入 Step 5-pre：subagent 退出前必写两份产物；在 Step 6 "失败报告" 同样处插入）

**内容**（原文复制 findings.md §3.2.6 + §2.4/§8.4 中 `debug_status.json` 的 schema）：

```markdown
### Step 5-pre / Step 6-pre: subagent 退出前必须产出

**所有分支、所有 session_outcome 情况**退出前都必须写入：

#### `{task_dir}/debug_trace.md`（详细叙事，4 节强制 + 可选附录）

（4 节模板见 findings.md §3.2.6，这里略）

#### `{task_dir}/debug_status.json`（机器可读 verdict）

```json
{
  "schema_version": 1,
  "phase8_outcome": "success | failed | stopped_by_gate | stopped_by_loop_limit | progressed_to_new_failure_type | timeout | skipped_env_issue | skipped_unsupported_type",
  "session_branch": "1-P | 1-B | 1-I | 1-R | 1-T",
  "started_at": "<ISO>", "ended_at": "<ISO>",
  "attempts_used": <int>,
  "final_failure_type": "<从最后一次 .eval_status/phase8_attempt_{N}.json 读取>",
  "final_eval_status_path": "{task_dir}/.eval_status/phase8_attempt_{N}.json",
  "entry_failure_type": "<进入时的 final_status.failure_type>",
  "notes": "<一句话>"
}
```

> **不 append `trace.md`**。`trace.md` 在 Phase 7 写完后全程只读（findings.md §2.4 / §7.3）。
```

Commit: `docs(precision-tuning): require debug_trace.md + debug_status.json at exit`

---

## Task H6: 更新 README.md / STRUCTURE.md 反映 gates/ 目录

**Files:**
- Modify: `skills/ascendc/precision-tuning/README.md`
- Modify: `skills/ascendc/precision-tuning/STRUCTURE.md`

补充 `gates/` 子目录的说明（参考 findings.md §3.1 的文件结构图）。

Commit: `docs(precision-tuning): update README/STRUCTURE to reflect gates/ package`

---

# STREAM I — Subagent agent file (并行，依赖 F3)

## Task I1: 修改 `agents/precision-tuning-discovery.md`

**Files:**
- Modify: `agents/precision-tuning-discovery.md:3`（description）
- Modify: `agents/precision-tuning-discovery.md:16-22`（argument-hint / 前提）
- Modify: `agents/precision-tuning-discovery.md:33-40`（Role Definition）

按 findings.md §4 的 3 处 diff：

**① description**
```diff
-description: AscendC 算子精度调优 Agent（发现式审计）— 依赖 agent 自身 AscendC 知识从取证数据直接推理根因，不强制预读参考示例
+description: AscendC kernel debug Agent（发现式审计，覆盖 build/import/runtime/timeout/precision 五类失败）
```

**② argument-hint**
```diff
 argument-hint: >
-  输入格式: "precision tune {task_name} [npu={NPU_ID}]"
+  输入格式: "debug ascendc {task_name} [npu={NPU_ID}] [failure_type=<类型>]"
   参数:
     - task_name: task 目录名（相对于 repo root，如 avg_pool3_d）
     - npu: NPU 设备 ID（默认 0）
+    - failure_type: 可选，若主 agent 已填好 debug_eligible，这里只作冗余确认
   前提:
-    task_name 目录下已有 model.py、model_new_ascendc.py、kernel/ 目录，
-    且 evaluate_ascendc.sh 已报告 Numerical 失败（非 Build/Import 失败）。
+    {task_dir} 下已有 model.py、model_new_ascendc.py、kernel/、trace.md，
+    且 trace.md.final_status.debug_eligible == true。
```

**③ Role Definition**
```diff
-- **精度诊断专家**: 基于数值取证数据和代码分析, 定位精度问题根因
+- **kernel debug 专家**: 根据 failure_type 选择诊断路径（session 内锁定一条分支）
+  - precision_failed: 数值取证 + 精度反模式匹配
+  - build_failed: 编译错误定位 + API 用法核对
+  - import_failed (kernel_side): pybind 符号 / kernel 导出修复（env_side 不处理）
+  - runtime_error: 运行时错误 / 段错误 / stack trace 分析
+  - timeout: 死锁 / 同步缺失 / tiling 配置异常分析
```

④ 反作弊约束保持不变（L96-122）。

Commit: `docs(agent): extend precision-tuning-discovery scope to 5 failure types`

---

# STREAM J — trace-recorder skill (并行，依赖 F3)

## Task J1: 重写 `skills/ascendc/trace-recorder/SKILL.md`

**Files:**
- Modify: `skills/ascendc/trace-recorder/SKILL.md`

**Context:** findings.md §5。目标：Phase 7 结束时不再事后猜 failure_type，而是读 `.eval_status/latest.json` + Phase 4 迭代历史 + 目录状态，产出 `trace.md` + 末尾 `final_status` fenced JSON block。

**Step 1: 在原 SKILL.md 末尾追加新章节**

```markdown
## 新增步骤：产出 final_status JSON block

执行完原有 Trace 记录后，**追加**一段 fenced JSON 作为 `final_status` block。

### 输入
- `{task_dir}/.eval_status/latest.json`（eval_wrapper 产出）
- Phase 4 迭代历史（主 agent 通过 stdin JSON 传入，或主 agent 写好 `{task_dir}/.phase4_history.json` 供读）
- 目录状态：`kernel/` 是否空、`model_new_ascendc.py` 是否 AST 退化

### 判定优先级（原文摘录 findings.md §5.3）

1. `kernel/` 目录为空 → `no_kernel` + `debug_eligible=false`
2. `model_new_ascendc.py` AST 退化 → `degraded` + `debug_eligible=false`
3. Phase 3 失败且未进到 Phase 4 → `tilelang_only_failed` + `debug_eligible=false`
4. `eval_status.latest.json.failure_type == execution_aborted` → 照搬 + `debug_eligible=false`
5. 否则 → 读 `eval_status.latest.json.failure_type`，照搬
6. `debug_eligible`:
   - `failure_type ∈ {precision_failed, build_failed, import_failed, runtime_error, timeout}` 且 `has_kernel && !has_degradation`
   - 若 `failure_type == import_failed`，还要 `import_subtype == import_kernel_side`

### final_status JSON 结构（findings.md §2.4）

```json
{
  "schema_version": 2,
  "failure_type": "...",
  "import_subtype": "...|null",
  "abort_subtype": "...|null",
  "has_kernel": true,
  "has_compiled_kernel": true,
  "has_degradation": false,
  "last_evaluate_phase": 6,
  "last_evaluate_status_path": "{task_dir}/.eval_status/phase6_attempt0.json",
  "tl_iterations_used": 3,
  "ac_iterations": [
    {"attempt": 0, "verifier_error": "...", "conductor_suggestion": "...",
     "eval_status_path": "...", "ended_at": "..."}
  ],
  "debug_eligible": true,
  "debug_eligible_reason": "..."
}
```

### 输出位置

将整个 JSON block 追加到 `{output_dir}/trace.md` 末尾，用 fenced code block 包裹：
`````
```json
{ ... final_status ... }
```
`````
**禁止**：写独立 `final_status.json` / `ac_history.json` 文件。Phase 4 迭代历史**内嵌**在 `final_status.ac_iterations` 数组里。
```

**Step 2: Commit**
```bash
git add skills/ascendc/trace-recorder/SKILL.md
git commit -m "docs(trace-recorder): read eval_status.json, emit final_status JSON block"
```

---

# STREAM K — 主 agent (依赖 G+H+I+J 全部完成)

## Task K1: 复制 `ascend-kernel-developer.md` → `ascend-kernel-developer-with-ascendc-debug.md`

**Files:**
- Create: `agents/ascend-kernel-developer-with-ascendc-debug.md`（从 `ascend-kernel-developer.md` 整体复制）

Run:
```bash
cp agents/ascend-kernel-developer.md agents/ascend-kernel-developer-with-ascendc-debug.md
```

**原 `ascend-kernel-developer.md` 保持不变**（findings.md 决策摘要 v2 表：原 agent 文件不动）。

Commit: `feat(agent): fork ascend-kernel-developer-with-ascendc-debug from main agent`

---

## Task K2: 新 agent 的 Phase 4 改动

**Files:**
- Modify: `agents/ascend-kernel-developer-with-ascendc-debug.md:304-393`

**① `max_ac_iterations = 3 → 2`**（L317）

**② 走 eval_wrapper**

把 L362-366 的 `bash skills/ascendc/ascendc-translator/references/evaluate_ascendc.sh {output_dir}` 替换为：

```bash
python3 utils/eval_wrapper.py \
    --phase 4 --attempt ${ac_iteration} --task-dir ${output_dir}
```

**③ `execution_aborted` 处理**

在 4.3 功能验证后增加：
- 读 `.eval_status/phase4_attempt{N}.json`
- 若 `failure_type == execution_aborted` 且 `abort_subtype ∈ {ssh_disconnected, docker_unreachable}` → eval_wrapper 已内部重试 1 次；仍 aborted 则本 attempt 直接判失败，**不消耗 ac_iteration 额度**（额外 1 轮重试不算进 2 次迭代）
- 若 `failure_type ∈ 其他` → 正常走 Conductor

Commit: `feat(agent-debug): Phase 4 uses eval_wrapper + iter limit 2 + execution_aborted retry`

---

## Task K3: Phase 6 改成只读

**Files:**
- Modify: `agents/ascend-kernel-developer-with-ascendc-debug.md:465-470`

替换为：

```markdown
## Phase 6: 全量用例验证（只读）

将 `{output_dir}/<op_name>.json.bak` 恢复为 `{output_dir}/<op_name>.json`，通过
`utils/eval_wrapper.py --phase 6` 运行一次全量验证。

```bash
cp {output_dir}/<op_name>.json.bak {output_dir}/<op_name>.json
python3 utils/eval_wrapper.py --phase 6 --attempt 0 --task-dir {output_dir}
```

无论成功失败，**不做任何修复**，直接进入 Phase 7。
失败用例的修复由 Phase 8 的 debug subagent 接手。
```

Commit: `feat(agent-debug): Phase 6 becomes read-only, no retry`

---

## Task K4: Phase 7 传 ac_iterations 给 trace-recorder

**Files:**
- Modify: `agents/ascend-kernel-developer-with-ascendc-debug.md`（Phase 7 节）

修改 trace-recorder 调用约定：
- 主 agent 在内存中维护 `ac_history_attempts`
- 调用 trace-recorder 前，把 `ac_history_attempts` 序列化为 JSON 写到 `{output_dir}/.phase4_history.json`
- trace-recorder 读此文件后嵌入 `final_status.ac_iterations`

文末加一段：

```markdown
### Phase 7 前置：写入 Phase 4 迭代历史

```bash
cat > {output_dir}/.phase4_history.json <<EOF
[
  {"attempt": 0, "verifier_error": "...", "conductor_suggestion": "...", "eval_status_path": "{output_dir}/.eval_status/phase4_attempt0.json", "ended_at": "..."},
  ...
]
EOF
```

然后调 trace-recorder，它会读取并内嵌到 `final_status.ac_iterations`。
```

Commit: `feat(agent-debug): hand Phase 4 history to trace-recorder via .phase4_history.json`

---

## Task K5: 新增 Phase 8（条件性 spawn）

**Files:**
- Modify: `agents/ascend-kernel-developer-with-ascendc-debug.md`（在 Phase 7 章节后插入 Phase 8）

**内容（原文复制 findings.md §2.5）**：

```markdown
## Phase 8: AscendC Debug Subagent（条件性 spawn）

### 8.1 读取 final_status
从 `{output_dir}/trace.md` 末尾解析 `final_status` fenced JSON block 的 `debug_eligible` 字段。

### 8.2 Spawn 决策

只在 `debug_eligible == true` 时 spawn。

白名单 failure_type（trace-recorder 的 `debug_eligible` 规则已编码）：
- `precision_failed`
- `build_failed`
- `import_failed && import_subtype == import_kernel_side`
- `runtime_error`
- `timeout`

不 spawn 场景：
- `success` / `degraded` / `no_kernel` / `tilelang_only_failed`
- `execution_aborted`
- `import_failed && import_subtype == import_env_side`

### 8.3 Spawn 调用

- subagent_type: `precision-tuning-discovery`
- 传入: `{output_dir}` 绝对路径 + npu ID + `failure_type`（冗余确认）
- timeout: 5400 秒

### 8.4 Spawn 后处理

**关键**：`trace.md` Phase 7 写完后**全程只读**。所有 Phase 8 信息都落到 subagent 自己的产物。

**正常退出**（subagent 返回 0 且两个文件都存在）：
- 校验 `{output_dir}/debug_trace.md` + `{output_dir}/debug_status.json`
- 输出一行：`Phase 8 结束，详见 debug_trace.md / debug_status.json`

**超时 / 异常**（subagent 未产出两个文件或超时）：
- 主 agent 兜底写 `{output_dir}/debug_status.json`：
  ```json
  {
    "schema_version": 1,
    "phase8_outcome": "timeout | crashed | missing_artifacts",
    "started_at": "...", "ended_at": "...",
    "crash_reason": "...",
    "final_failure_type": "<继承 Phase 7 final_status.failure_type>",
    "notes": "主 agent 兜底产出，subagent 未完成"
  }
  ```
- **不 append trace.md**，不重跑 evaluate

### 8.5 不 spawn 场景

主 agent 直接写 `debug_status.json`：`phase8_outcome: skipped`，`skip_reason` 填 `final_status.debug_eligible_reason` 的反面。

### 8.6 下游消费约定

最终状态读取顺序：
1. 优先 `{output_dir}/debug_status.json`（如存在）
2. 只有 `final_status`（Phase 8 没跑）→ 用 final_status
3. final_status 是 Phase 7 快照，`debug_status` 才是任务最终 verdict
```

Commit: `feat(agent-debug): add Phase 8 conditional debug subagent spawn`

---

## Task K6: 更新工作流图 + 约束表

**Files:**
- Modify: `agents/ascend-kernel-developer-with-ascendc-debug.md:42-51`（工作流图）
- Modify: `agents/ascend-kernel-developer-with-ascendc-debug.md:548`（约束表）

**工作流图**：
```
Phase 0: 参数确认
Phase 1: 环境准备
Phase 2: INPUT_CASES 精简
Phase 3: TileLang 设计表达   (max 5)
Phase 4: AscendC 转译与验证  (max 2)
Phase 5: 性能分析
Phase 6: 全量用例验证（只读）
Phase 7: Trace 记录 + final_status
Phase 8: Debug Subagent（条件性 spawn）
```

**约束表**：
- "Phase 4 最大迭代" 3 → 2
- 新增 "Phase 6 修复 | 禁止"
- 新增 "Phase 8 subagent timeout | 5400s，超时/异常不重试"

Commit: `docs(agent-debug): update workflow diagram + constraints for Phase 6-8`

---

# STREAM L — 集成联调 + 兜底 (最后)

## Task L1: 准备 5 类失败 fixture

**Files:**
- Create: `tests/fixtures/ascendc_debug/build_failed/` — 一个故意用错 API 的 kernel
- Create: `tests/fixtures/ascendc_debug/import_kernel_side/` — pybind 符号不匹配
- Create: `tests/fixtures/ascendc_debug/runtime_error/` — 故意越界写 GM
- Create: `tests/fixtures/ascendc_debug/timeout/` — 缺 SyncAll 的跨核 kernel
- Create: `tests/fixtures/ascendc_debug/precision_failed/` — ReduceSum count 错算

每个 fixture 含最小化 `model.py` + `model_new_ascendc.py` + `kernel/` + `{op}.json`（5 case），能复现对应 failure_type。

**Step 1: 选 2 个 archive_tasks 里已通过任务（如 `avg_pool3_d`, `rms_norm`），派生出故意坏掉的 fixture**

**Step 2: 每个 fixture 跑一次 `python3 utils/eval_wrapper.py --phase 8 --attempt 0 --task-dir ...`**

验证：`.eval_status/latest.json.failure_type` 等于预期类型。

**Step 3: Commit**
```bash
git add tests/fixtures/ascendc_debug/
git commit -m "test(ascendc_debug): add 5 failure-type fixtures for eval_wrapper E2E"
```

---

## Task L2: 端到端 dry-run（精度分支不改变行为）

**目标**: 对已有的 archive_tasks/ 任务跑 `precision_gate.py --step forensics`，确认 refactor 不破坏精度路径。

Run（远端）:
```bash
ssh npu_server "cd /home/c00959374/AscendOpGenAgent && git fetch origin && git checkout cjm/debug-v2 && git pull origin cjm/debug-v2"
ssh npu_server 'docker exec cjm_cann1 bash -c "cd /home/c00959374/AscendOpGenAgent && \
  python3 skills/ascendc/precision-tuning/scripts/precision_gate.py \
    --step forensics --op-name rms_norm --task-name archive_tasks/rms_norm --attempt 0"'
```

Expected: 输出 `"gate": "GATE-..."`, 不报 import 错误。

Commit（若有小改动）: `fix(precision_gate): xxx (discovered in L2 smoke)`

---

## Task L3: 端到端 spawn 演练（1 个 fixture）

对 `tests/fixtures/ascendc_debug/build_failed/` 跑完整链路：
1. 运行新 agent `ascend-kernel-developer-with-ascendc-debug`（Phase 0-7）
2. 验证 `trace.md` 末尾有 `final_status` block 且 `debug_eligible == true`
3. 运行 Phase 8 spawn subagent
4. 验证 `debug_trace.md` + `debug_status.json` 均产出
5. 验证 `trace.md` Phase 7 后未被修改（`md5sum` 对比）

若链路未通，记录问题分 PR 修。

---

## Task L4: 最终文档 sweep

**Files:**
- Modify: `agents/ascend-kernel-developer-with-ascendc-debug.md` 顶部 summary
- Modify: 各 SKILL.md 顶部 description
- Modify: README.en.md / README.md（简要提及新 agent 的存在，不展开）

Commit: `docs: final sweep after ascendc-debug integration`

---

# 总结表

| Stream | Tasks | Must-before | 时间（人日） |
|---|---|---|---|
| F | F1-F3 | — | 2.5 |
| G | G1-G9 | F3 | 3-4 |
| H | H1-H6 | F3 | 1-1.5 |
| I | I1 | F3 | 0.3 |
| J | J1 | F3 | 0.5 |
| K | K1-K6 | G+H+I+J | 1.5-2 |
| L | L1-L4 | K | 2-4 |
| **合计** | | | **11-14.8** |

与 findings.md §9 的 12-16 人日估算一致（本计划少 1-2 人日主要因为 L2/L3 端到端联调时间估得偏乐观，实际上若 fixture 构造困难会补足到 16）。

---

## 风险点（findings.md §9 + 本计划补充）

1. `eval_wrapper` 的 `failure_type` 分类边界（timeout_marker 实际效果、import_subtype 模式匹配覆盖率）
2. `precision_gate.py` 拆包后的精度分支回归
3. 5 类失败 fixture 构造（build/import/runtime/timeout 各要造一个可复现的坏 kernel）
4. **新增**：Stream K 在所有上游完成前无法启动；若 Stream G 延期会直接延后 K/L
5. **新增**：远端 NPU 容器不稳定时，Stream L 会被拖慢
