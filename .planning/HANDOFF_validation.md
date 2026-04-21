# Precision Forensics Rewrite — Validation Handoff

> **Purpose**: Phase 4 (remote NPU validation) 交接文档。上一个 session 已完成 Phase 1-3 代码 + commit + push, 但**未完成 NPU 上的行为验证**。本文档给出新 session 需要跑的 check、期望结果、以及根据结果的下一步动作。

**相关文档**:
- 设计文档: `.planning/DESIGN_forensics_rewrite.md` (v2, post-codex-review — 架构、schema、边界、codex 的 12 条 finding 解决方案)
- 任务计划: `.planning/task_plan.md` (TDD 步骤, Phase 1-6 划分)

**相关 commit**（都已 push 到 `origin/cjm/debug`）:
```
7aac509 docs: add precision-tuning README
6724e44 docs: restore STRUCTURE.md and agent Core Capabilities
7561ad8 ⚠ forensics: rewrite PrecisionForensics.run with full L1-L4 + per-case aggregate
485ebc3 ⚠ forensics: rewrite OperatorExecutor to spawn _forensics_child.py subprocess
872197c forensics: enrich OperatorTypeDetector with attributes + reduction_axis
a7b9403 forensics: add parity test scaffold
15d5002 forensics: add _forensics_child.py subprocess executor
e69e157 forensics: add OutputFlattener
db33d87 forensics: add MemoryLayoutAnalyzer
fc28ec8 forensics: add DiffAnalyzer
e06a46a plan: design doc + task plan
```
⚠ 标记的两个是 runtime-critical 改动（切执行路径到新 subprocess + 新 schema 组装）。其他是 dead code 增量，revert 不影响现状。

---

## 1. 验证前提

**远端状态** (`ssh npu_server`):
- 代码已 pull 到 `/home/c00959374/AscendOpGenAgent` 的 `cjm/debug` 分支（最新 commit `7aac509`）
- 验证用 fixture: `/home/c00959374/AscendOpGenAgent/outputs/codex_batch_20260420_1755/3_Add_debug`
  - 上一 session 已 `cp` 自 `3_Add`，strip 掉 `precision_tuning/` / `__pycache__/` / `_codex_last.txt` / `kernel/build/` / `.bench_baseline/`
  - 已用 `build_ascendc.py --clean` 重新编译, `_add_ext.so` 就位

**已知遗留进程**（上一 session 结束时）:
- PID `582314`：在 `outputs/codex_batch_20260421_1232/7_Sum/` 跑 `_forensics_child.py`，**不是上一 session 启动的**，是用户自己并行跑的 precision-tuning
- 这个进程用的就是未验证的新 `_forensics_child.py` 代码；如果新代码有 bug，它可能 hang 或产生错误的 forensics_report。验证完再决定要不要让它继续

**container**: 用 `cjm_cann2` 或其他空闲容器 (cjm_cann1 可能被 7_Sum 占着); `cd /home/c00959374/AscendOpGenAgent` 后执行。

---

## 2. 三个 Check（按顺序跑）

### Check 1 — Parity Test（最关键）

**目的**: 验证 `_forensics_child.py` 复制的 loader 与 `utils/verification_ascendc.py` 产出的 per-case mismatch_ratio / pass-fail 完全一致。

**做法**:

```bash
# 1. 把 3_Add_debug 加入 FIXTURES（编辑 test_executor_parity.py 第 ~47 行 FIXTURES 列表）
#    或直接先跑默认 3 fixtures 看覆盖情况

# 2. 运行
docker exec cjm_cann2 bash -c '
  cd /home/c00959374/AscendOpGenAgent
  python3 skills/ascendc/precision-tuning/scripts/test_executor_parity.py
'
```

**期望输出**:
```
[parity] running avg_pool3_d...
[parity] SKIP: avg_pool3_d: kernel/build missing (run evaluate_ascendc.sh first)
[parity] running rms_norm...
[parity] SKIP: rms_norm: kernel/build missing
[parity] running 3_Add_pt1...
[parity] PASS: 3_Add_pt1: N cases, parity verified
=== PARITY SUMMARY ===
...
```

或对 `3_Add_debug`（如果加入 fixture）:
```
[parity] PASS: 3_Add_debug: 10 cases, parity verified
```

**失败模式**:
- `FAIL: ... child executor exit rc=...` → `_forensics_child.py` 崩了，看 stderr tail
- `FAIL: ... case count mismatch` → input_groups 解析不一致
- `FAIL: ... case[i] matched: child=... verif=...` → comparison 语义漂移（atol/rtol/nan_to_num 有差异）
- `FAIL: ... case[i] mismatch_ratio: child=X verif=Y` (diff > 1e-4) → 数值路径差异

**如果 SKIP 了所有 fixture**: 本地 parity 无从验证, 跳到 Check 2 + Check 3 用 `3_Add_debug` 间接验证。

---

### Check 2 — 新版 Forensics 端到端（schema 填充是否真实）

**目的**: 运行完整 `precision_forensics.py`, 验证 `forensics_report_0.json` 里之前 null 的字段现在都被填满了。

**做法**:

```bash
docker exec cjm_cann2 bash -c '
  cd /home/c00959374/AscendOpGenAgent
  python3 skills/ascendc/precision-tuning/scripts/precision_forensics.py \
    3_Add_debug \
    --task-dir /home/c00959374/AscendOpGenAgent/outputs/codex_batch_20260420_1755/3_Add_debug \
    --attempt 0
'
```

**期望行为**:
- 进程 < 120s 返回 (简单算子), 不 hang
- stdout 末尾有形如:
  ```
  [FORENSICS] ✅ 精度取证完成 (attempt=0)
    op_type: unknown (source=name_heuristic)
    primary_hint: <某个 pattern>
    evidence: ...
    mismatch: X.XX% (N/M)
    max_diff: X.XXXXXX
    num_outputs: 1, num_cases: 10, int8_active: False
    report: /home/c00959374/AscendOpGenAgent/outputs/codex_batch_20260420_1755/3_Add_debug/precision_tuning/forensics_report_0.json
  ```

**期望 JSON 字段**（用 `python3 -c "import json; d=json.load(open('...report_0.json')); ..."` 逐个验证）:

| 路径 | 期望 |
|------|------|
| `status` | `"completed"` |
| `attempt` | `0` |
| `primary_hint` | 非 null 字符串 |
| `outputs` | 非空列表 |
| `outputs[0].basic_stats.mismatch_ratio` | float |
| `outputs[0].basic_stats.match_rate` | float |
| **`outputs[0].basic_stats.median_abs_diff`** | **非 null**（旧版为 null） |
| **`outputs[0].basic_stats.p99_abs_diff`** | **非 null** |
| **`outputs[0].error_distribution.sign_analysis.bias_direction`** | `"positive"` \| `"negative"` \| `"balanced"`（旧版 null） |
| **`outputs[0].worst_elements`** | 列表, 含 ≥ 3 条（旧版 []） |
| `outputs[0].worst_elements[0].index` | list of int |
| `outputs[0].worst_elements[0].golden_value` | float |
| `outputs[0].worst_elements[0].actual_value` | float |
| **`outputs[0].tail_analysis`** | dict, 有 `tile_32` / `tile_64` 等键或 `note` 字段（旧版 {}） |
| **`outputs[0].dimension_analysis`** | 列表, 每项含 `dim`/`size`/`mismatch_rate_*`（旧版 []） |
| **`outputs[0].value_range`** | dict, 含 `golden_min/max/has_nan/nan_count` 等（旧版 null） |
| **`outputs[0].per_case`** | 列表, 长度 = num_cases |
| `outputs[0].case_aggregate.mismatch_ratio_max` | float |
| `outputs[0].case_aggregate.all_cases_same_pattern` | bool |
| `outputs[0].case_aggregate.shape_conditional` | bool |
| **`L6_memory_layout.inputs`** | 非空列表, 每 input 含 `shape`/`stride`/`is_contiguous`/`last_dim_alignment`（旧版 `data.get('inputs_meta', [])` 经常为 []） |
| `L8_operator.op_type` | 字符串 |
| `L8_operator.attributes` | dict（对 3_Add 可能为 {}，对 avg_pool 等应该非空） |
| `L8_operator.reduction_axis` | dict or null |
| `int8_path_active` | bool |
| `nan_inf_detected.ref.has_nan` | bool |
| `executor_parity_hash` | 字符串（16 char sha256 prefix） |

**快速全字段检查**:
```bash
docker exec cjm_cann2 bash -c '
  python3 -c "
import json
r = json.load(open(\"/home/c00959374/AscendOpGenAgent/outputs/codex_batch_20260420_1755/3_Add_debug/precision_tuning/forensics_report_0.json\"))
print(\"status:\", r.get(\"status\"))
print(\"primary_hint:\", r.get(\"primary_hint\"))
o = r[\"outputs\"][0]
print(\"  basic_stats keys:\", sorted(o[\"basic_stats\"].keys()))
print(\"  error_distribution:\", type(o[\"error_distribution\"]).__name__, \"(keys:\",
      sorted(o[\"error_distribution\"].keys()) if o[\"error_distribution\"] else None, \")\")
print(\"  worst_elements len:\", len(o[\"worst_elements\"]))
print(\"  tail_analysis keys:\", list(o[\"tail_analysis\"].keys()))
print(\"  dimension_analysis len:\", len(o[\"dimension_analysis\"]))
print(\"  per_case len:\", len(o[\"per_case\"]))
print(\"  case_aggregate:\", o[\"case_aggregate\"])
print(\"L6 inputs len:\", len(r[\"L6_memory_layout\"][\"inputs\"]))
print(\"L8 attributes:\", r[\"L8_operator\"].get(\"attributes\"))
print(\"L8 reduction_axis:\", r[\"L8_operator\"].get(\"reduction_axis\"))
print(\"int8:\", r.get(\"int8_path_active\"))
print(\"nan_inf:\", r.get(\"nan_inf_detected\"))
print(\"parity_hash:\", r.get(\"executor_parity_hash\"))
"'
```

**失败模式**:
- **Hang > 2 min**: `_forensics_child.py` 卡死。排查:
  ```bash
  # 1. 看 child 自己跑会不会 hang:
  docker exec cjm_cann2 bash -c "
    cd /home/c00959374/AscendOpGenAgent
    rm -rf /tmp/dbg_child && mkdir -p /tmp/dbg_child
    timeout 60 python3 skills/ascendc/precision-tuning/scripts/_forensics_child.py \
      /home/c00959374/AscendOpGenAgent/outputs/codex_batch_20260420_1755/3_Add_debug \
      /tmp/dbg_child
    echo rc=$?
    ls -la /tmp/dbg_child/
  "
  # 2. 如果 child 也 hang 但 bench verify 通过, 说明是 child 的 _move_to_device 或 model 构造问题
  # 3. 比较 child 跟 verification_ascendc.py 的 _run_verification 差异: 主要是 cand_module.get_init_inputs() 优先级是否传对
  ```
- **`status: error` + traceback**: 看 traceback 定位。常见: pickle 不支持某种 tensor subclass, nan_to_num 在 int8 上报错
- **某些字段仍为 null**: diff 逻辑 bug, 看具体哪个字段, 对照 `DiffAnalyzer.analyze()` 实现

---

### Check 3 — Gate-F 向后兼容

**目的**: 确认新 schema 没打破 `precision_gate.py` 的字段依赖。

**做法**:
```bash
docker exec cjm_cann2 bash -c '
  cd /home/c00959374/AscendOpGenAgent
  python3 skills/ascendc/precision-tuning/scripts/precision_gate.py \
    --step forensics \
    --op-name 3_Add_debug \
    --task-name outputs/codex_batch_20260420_1755/3_Add_debug \
    --attempt 0
'
```

**期望**: exit 0; stdout 末尾显示 `GATE-F PASSED` 或 JSON `"passed": true`; 5 个 `checks.*` 全 True: `report_exists`, `report_parseable`, `status_completed`, `has_primary_hint`, `has_outputs`, `has_basic_stats`, `attempt_matches`.

**另外**: Gate-F 通过后会自动写 `baseline_state.json`（幂等，已存在则不覆盖）。可顺便查:
```bash
cat /home/c00959374/AscendOpGenAgent/outputs/codex_batch_20260420_1755/3_Add_debug/precision_tuning/baseline_state.json
```

期望含 `match_rate`、`mismatch_ratio`、`max_abs_diff`、`mean_abs_diff`、`primary_hint`。

**失败模式**:
- `prerequisite_error` → 路径拼错, 检查 `--task-name` 是否相对 repo_root
- `has_basic_stats: false` → report 里 `outputs[0].basic_stats` 缺某字段, 调 PrecisionForensics.run schema
- `attempt_matches: false` → report 里 `attempt` 字段和 `--attempt 0` 不一致

---

## 3. 根据结果的下一步

| 场景 | 行动 |
|------|------|
| ✅ 三条全 PASS | 跟上一 session 的我说 "全过了"，我会: (1) 删 `.planning/DESIGN_forensics_rewrite.md` + `task_plan.md` + 本文档 (2) 给 final 状态报告 (3) 收尾 commit |
| ❌ Check 1 FAIL | 给我 FAIL 的 case + `child=X verif=Y` 对比, 我对照 verification_ascendc 修复 `_forensics_child.py` 的 loader 或 compare 语义 |
| ❌ Check 2 hang | 把 "排查 hang" 的 debug 命令输出给我（看 child 单独跑会不会 hang） |
| ❌ Check 2 字段 null | 给我具体哪个字段 null, 大多数是 `PrecisionForensics._analyze_path_across_cases` 里路径处理 bug |
| ❌ Check 3 FAIL | 给我 `checks` dict 输出, 我改 `PrecisionForensics.run` 的 JSON 生成逻辑 |
| 只想先回退 | 告诉我 "revert"，我 `git revert 7561ad8 485ebc3` 把 Phase 3.1+3.2 回退；Phase 1-2 的 DiffAnalyzer / MemoryLayoutAnalyzer / OutputFlattener / _forensics_child.py 作为 dead code 留着 |

---

## 4. 已知 outstanding 风险

1. **PID 582314 / 7_Sum**：用户自己启动的 precision-tuning 进程，已用上未验证的新 forensics 代码。建议先看它的 `outputs/codex_batch_20260421_1232/7_Sum/precision_tuning/forensics_report_0.json` 是否正常产出 + 有无 traceback，判断是否要终止 (`kill -9 582314`) + revert Phase 3。

2. **verification_ascendc.py 在上一 session 末尾出现过 "rc=0 但 stdout 空" 的诡异现象**。可能是我的 bash 管道截断，也可能是真的静默返回。Check 2 里如果 forensics 也出现这种情况，需要单独确认 verification 的 stdout 去向（直接 docker exec 手敲应该能看到全部）。

3. **Phase 1 的 numpy/torch import 在主进程**：新 `precision_forensics.py` 顶层加了 `import numpy as np` + `import torch`，意味着 `precision_gate.py`/`precision_knowledge.py` 如果 import precision_forensics 的话会被迫也加载 torch。目前 gate 和 knowledge 都 **不 import** precision_forensics, 所以没影响; 但任何未来 glue script 要注意这个入口成本。

4. **跨 case aggregation 的 `shape_conditional` 启发式**（相关系数阈值 0.7）是纯新增功能, 没有已知基线数据校准, 可能误报。不阻塞，但后续观察。

---

## 5. 快速参考：所有相关文件

**新增**:
- `skills/ascendc/precision-tuning/scripts/_forensics_child.py` (337 行)
- `skills/ascendc/precision-tuning/scripts/test_executor_parity.py` (279 行)
- `skills/ascendc/precision-tuning/README.md` (265 行)
- `skills/ascendc/precision-tuning/STRUCTURE.md` (600+ 行)

**改动**:
- `skills/ascendc/precision-tuning/scripts/precision_forensics.py` (主文件, +700 / -200)
- `agents/precision-tuning.md` (Core Capabilities + Operational Guidelines 补齐)
- `agents/precision-tuning-discovery.md` (Core Capabilities dtype 阈值增强)

**设计参考**:
- `.planning/DESIGN_forensics_rewrite.md` § 2 Architecture + § 2.2 Schema
- `.planning/task_plan.md` Phase 1-4 逐步

**未动**:
- `utils/verification_ascendc.py` (read-only 对照 oracle)
- `skills/ascendc/precision-tuning/SKILL.md` (schema 兼容, 不需要改)
- `skills/ascendc/precision-tuning/scripts/precision_gate.py`
- `skills/ascendc/precision-tuning/scripts/precision_knowledge.py`
- `skills/ascendc/precision-tuning/scripts/anticheat.py`
