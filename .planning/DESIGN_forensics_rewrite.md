# Precision Forensics 重构设计文档 (v2, post-codex-review)

**Scope**: 把 `skills/ascendc/precision-tuning/scripts/precision_forensics.py` 从"stdout 解析简化版"升级为"完整数值分析版"，兼容现有 SKILL.md Sub-step 2.1 `[FORENSICS_SUMMARY]` 全部字段要求。

**决策约束**（用户指定 + codex review 修订）：
1. 只针对 AscendC 评测路径，不考虑 DSL / TileLang
2. Tensor 加载逻辑**复制**一份到 forensics（不 import `utils/verification_ascendc.py`），独立演化
3. 评测语义（model 加载、input_groups、atol/rtol、int8 特判、NaN/Inf 处理）**必须**与 `utils/verification_ascendc.py` bit-for-bit 一致
4. **必须**用 child-process 隔离执行，避免 in-process 污染（sys.path、torch state、extension 加载）— codex CRITICAL 修订
5. **必须**保留现有 JSON top-level schema（`status` / `attempt` / `primary_hint` / `outputs` / `L6_memory_layout` (top-level) / `L8_operator` / `history_trend`）— codex CRITICAL 修订
6. Output 结构化：按 canonical path 扁平化（`output[0]`, `output[1].foo`），不假设 `List[List[Tensor]]` — codex CRITICAL 修订

---

## 1. 背景与问题

### 1.1 现状

迁移到 AscendOpGenAgent 时，`precision_forensics.py` 被重构为"轻量版"。三个核心分析类（`DiffAnalyzer` / `MemoryLayoutAnalyzer` / `MultiCaseForensics`）被删除，`OperatorExecutor` 只从 `verification_ascendc.py` 的 stdout 解析简化字段。

### 1.2 影响

- `outputs[i].error_distribution` / `worst_elements` / `tail_analysis` / `dimension_analysis` / `value_range` / `median_abs_diff` / `p99_abs_diff` — 永远 `null`/空集
- `L6_memory_layout` — 无实际内容
- SKILL.md Sub-step 2.1 的 `[FORENSICS_SUMMARY]` 模板强制要求这些字段；agent 只能填"无"
- 根因诊断降级：`sign_analysis.bias_direction`、`worst_elements` top-3 位置、`tail_analysis` 尾块 mismatch、`dimension_analysis` 各维度 mismatch 率 — 都是定位 padding 污染 / tail spike / overflow 等精度问题的核心证据

### 1.3 "复制 + 独立演化"重新诠释

用户要求"复制加独立演化"，codex 警告"duplicating the executor is bad without parity tests"。两者的正确折中：

- **复制**: `verification_ascendc.py` 里的**纯数据加载辅助函数**（`_load_module` / `_clone_value` / `_move_to_device` / `_normalize_output` / `_contains_int8_tensor` / `_get_input_groups` / `_get_device`）复制到 forensics 的 child-process helper，未来可独立增强
- **独立演化的是分析层**，不是执行层。DiffAnalyzer / MemoryLayoutAnalyzer 属于 forensics 专属
- **执行仍然隔离**: forensics 不在主进程 import `model.py`/`model_new_ascendc.py`；用 subprocess（spawn 一个 Python child）跑模型，tensor 通过磁盘 pickle 交换
- **强制添加 parity 测试**: 证明 copied loader 产出的 `ref_out`/`cand_out` 与 `verification_ascendc.py` 在相同输入下**结构完全相同**（同一 nested tree、同 shape、同 dtype、同数值容差内），捕获"复制后两侧漂移"

### 1.4 为何不 import verification_ascendc 统一执行

| 方案 | 评估 | 选择 |
|------|------|------|
| A-1 forensics `import verification_ascendc`，主进程复用 | 主进程污染风险（sys.path / torch state / extension load）; verification_ascendc 演化破坏 forensics | 否 |
| A-2 forensics 调 `verification_ascendc.py` 作子进程，让它 dump tensor 到磁盘 | 需要给 verification_ascendc 加 `--dump-tensors` flag，侵入评测链路 | 否 |
| **A-3 forensics 复制 loader 到自己的 child-process helper，跑自己的 subprocess** | 执行隔离 + 独立演化满足；只需 parity tests 保证语义不漂移 | ✅ **采用** |
| B 放弃字段，砍 SKILL.md 要求 | 永久削弱诊断能力 | 否 |

---

## 2. 架构设计

### 2.1 模块划分（三进程模型）

```
[Parent process: precision_forensics main]
  ├── OperatorTypeDetector      → L8_operator（op_type + attributes + reduction_axis）
  ├── subprocess spawn: python3 _forensics_child.py <task_dir>
  │     → child 加载 model.py + model_new_ascendc.py + 跑 input_groups
  │     → dump: ref_outputs, cand_outputs, inputs, atol/rtol, int8_triggered 到 /tmp/*.pkl
  │     → child exit 0 / 非 0
  │
  ├── OutputFlattener           → 把任意 nested Tensor/list/tuple/dict 展平成 {path: TensorRef}
  ├── DiffAnalyzer              → 对每个 (path, ref, cand) 跑完整 L1-L4 分析
  ├── MemoryLayoutAnalyzer      → 对每个 case 的 inputs 跑 L6（top-level，非 per-output）
  ├── IntermediateProbe (stub)  → L5 = None
  ├── CodeMapper (stub)         → L7 = None（Agent 在 Sub-step 2.3 手动完成）
  ├── HistoryComparator         → history_trend
  └── PrecisionForensics.run    → 组装 top-level JSON（schema 兼容）

[Child process: _forensics_child.py (new file)]
  ├── 复制来的 loader helpers:
  │     _load_module, _find_model_class, _clone_value, _move_to_device,
  │     _normalize_output, _contains_int8_tensor, _get_input_groups, _get_device
  │
  │  注：复制自 utils/verification_ascendc.py，语义必须一致（parity test 保证）
  │
  ├── 跑 model + model_new_ascendc 对 get_input_groups() 所有 case
  ├── 每个 case 跑完立即：
  │     - pickle dump 到 tmp
  │     - del tensors + torch.cuda.empty_cache() / torch_npu.npu.empty_cache()
  │     - 回收内存，避免累积
  └── 退出；所有 tensor 留在磁盘 /tmp/precision_forensics_<pid>/

[Parity test process: test_executor_parity.py]
  对一组固定 fixture 任务：
  1. 调 verification_ascendc.py 拿 comparisons（pass/fail + diff summary）
  2. 调 _forensics_child.py 拿 dumped tensors，用 verification 同款 compare 逻辑算 pass/fail
  3. 断言两者 all_passed 一致、每个 path 的 mismatch 数一致
```

### 2.2 Top-level JSON Schema（兼容 + 增强）

**完整保留**（consumers: `precision_gate.py` 通过第 52-74、353-380、501-509、940-988 行读取）:

```python
{
  "version": "2.0",                           # 保留
  "op_name": str,                             # 保留
  "attempt": int,                             # 保留（Gate-F attempt_matches 检查）
  "status": "completed" | "error",            # 保留（Gate-F status_completed 检查）
  "num_outputs": int,                         # 保留
  "L0_pass": bool,                            # 保留
  "all_passed": bool,                         # 保留（与 L0_pass 语义一致，冗余为兼容）

  "outputs": [                                # 保留（Gate-F has_outputs 检查）
    {
      "output_index": int,                    # 扩展: 不再假设单维度，是 canonical path 的稳定序号
      "output_path": str,                     # 新增: "output[0]" / "output[1].foo.bar"
      "output_kind": "tensor"|"scalar"|"none",# 新增
      "output_shape": list[int] | null,       # 保留
      "output_dtype": str | null,             # 保留
      "pass_fail": bool,                      # 保留

      "basic_stats": {                        # 保留（Gate-F has_basic_stats 检查）
        "max_abs_diff": float,
        "mean_abs_diff": float,
        "median_abs_diff": float,             # 新填充（之前 null）
        "p99_abs_diff": float,                # 新填充
        "num_mismatched": int,
        "total_elements": int,
        "mismatch_ratio": float,              # 0~1 比例
        "match_rate": float,                  # 0~1 比例
        "int8_special_tolerance": bool,       # 新增: atol=1.5 rtol=0 是否生效
      },

      "error_distribution": { ... } | null,   # 新填充（sign_analysis、percentiles、rel_error）
      "value_range": { ... } | null,          # 新填充（min/max/mean/std + NaN/Inf counts）
      "pattern_hint": { ... },                # 保留（primary/all_hints）
      "worst_elements": [ ... ],              # 新填充（top-10）
      "tail_analysis": { ... },               # 新填充（per tile_size: 8/16/32/64/128/256）
      "dimension_analysis": [ ... ],          # 新填充（per axis）

      # Per-output cross-case aggregation (NEW，替代"pick worst case 丢弃其余"):
      "per_case": [                           # 每个 case 对该 path 的完整诊断
        {"case_idx": int, "basic_stats": {...}, "pattern_hint": {...},
         "worst_elements": [...], "tail_analysis": {...}, "dimension_analysis": [...],
         "error_distribution": {...}, "value_range": {...}}
      ],
      "representative_case_idx": int,         # 选中作为展示的 case（max mismatch_ratio）
      "case_aggregate": {                     # 跨 case 聚合（min/max/mean mismatch_ratio 等）
        "num_cases": int,
        "mismatch_ratio_min": float, "mismatch_ratio_max": float, "mismatch_ratio_mean": float,
        "max_abs_diff_min": float, "max_abs_diff_max": float,
        "pass_case_count": int, "fail_case_count": int,
        "all_cases_same_pattern": bool,
        "shape_conditional": bool,            # True if failure rate correlates with case shape
      },
    },
  ],

  # Top-level L6（codex 修订，非 per-output）
  "L6_memory_layout": {
    "inputs": [ {name, shape, stride, dtype, is_contiguous, last_dim_alignment, ...} ],
    "outputs": [ ... ],                       # ref_outputs 的 layout
  },

  # Top-level L5/L7/L8（保留）
  "L5_intermediate": null,                    # stub, L5 决策不实现
  "L7_code_mapping": null,                    # Agent 手动
  "L8_operator": {                            # 丰富（codex HIGH 修订）
    "op_type": str,
    "source": str,
    "confidence": float,
    "pattern_priority": list[str],
    "attributes": dict,                       # 新增（SKILL.md 2.1 要求）
    "reduction_axis": {axis_length: int, ...} | null,  # 新增
  },

  # Top-level 来自 representative output（worst）
  "primary_hint": str,                        # 保留（Gate-F has_primary_hint 检查）
  "primary_confidence": float,                # 保留
  "primary_evidence": str,                    # 保留
  "all_hints": list,                          # 保留

  # 历史比较
  "history_trend": { trend: [...], mismatch_improving: bool } | null,  # 保留
  "multi_case_analysis": null,                # 保留占位
  "num_test_cases": int,                      # 保留（现为真实 case 数）

  # 可用文件
  "available_files": {                        # 保留
    "reference": bool,
    "custom": bool,
  },

  # Parity / diagnostic meta（新增）
  "executor_parity_hash": str | null,         # child process 运行指纹，供 parity test 对账
  "int8_path_active": bool,                   # 整份 report 是否 int8 路径（任意 output 为 int8）
  "nan_inf_detected": {                       # 对照 verification_ascendc 的 nan_to_num 前数据
    "ref": {"has_nan": bool, "has_inf": bool, "nan_count": int, "inf_count": int},
    "cand": {"has_nan": bool, "has_inf": bool, "nan_count": int, "inf_count": int},
  },
}
```

### 2.3 Output Flattening（codex CRITICAL 修订）

**问题**: `verification_ascendc.py` 的 `_compare_values` 递归处理任意 `Tensor / list / tuple / dict / scalar` 结构；我原设计把 outputs 扁成 `List[Tensor]` 按位置索引，会把 `(tensor, (tensor, tensor))` 这种元组结构扭曲。

**方案**: 新增 `OutputFlattener.flatten(obj, root="output")` → `dict[path_str, {ref: Tensor|scalar|None, kind: str, ...}]`

```python
class OutputFlattener:
    """
    把 ref_out / cand_out 这种任意嵌套 (Tensor|list|tuple|dict|scalar) 结构扁平化，
    产出 canonical path → {ref, cand, kind, shape, dtype, status}。

    path 语法（对齐 verification_ascendc _compare_values 的 path 约定）:
      - "output[0]"         : 顶层 tensor
      - "output[1].foo"     : dict key
      - "output[2][3]"      : list/tuple 索引

    kind: "tensor" | "scalar" | "none" | "type_mismatch" | "shape_mismatch" | "len_mismatch"
    """
    def flatten(self, ref, cand, root="output") -> dict[str, dict]:
        ...
```

每个 path 独立跑 DiffAnalyzer。`outputs` 列表按 path 字典序排序，`output_index` 稳定编号。

### 2.4 Child-Process Executor（codex CRITICAL 修订）

新增文件 `skills/ascendc/precision-tuning/scripts/_forensics_child.py`：

```python
#!/usr/bin/env python3
"""
子进程 executor，由 precision_forensics.py 主进程 spawn。
只做一件事：加载 model.py / model_new_ascendc.py，对 get_input_groups()
所有 case 跑 ref 和 cand，把 tensor 通过 pickle 存到指定目录。

本模块的 loader helpers **复制自** utils/verification_ascendc.py（不 import），
有意允许独立演化；parity 测试保证两侧语义一致。
"""

# ── 复制区（keep in sync with utils/verification_ascendc.py）──
def _load_module(module_path, module_name): ...
def _find_model_class(module, preferred): ...
def _clone_value(value): ...
def _move_to_device(value, device): ...
def _normalize_output(value): ...
def _contains_int8_tensor(value): ...
def _get_input_groups(module): ...
def _get_device(): ...
# ── 结束复制区 ──


def run(task_dir: str, dump_dir: str) -> int:
    """
    为每个 case 跑 ref/cand，dump 到 {dump_dir}/case_{N}_{ref,cand}.pkl
    + {dump_dir}/metadata.json。
    内存控制: 每个 case 跑完立即 pickle dump，del tensor，清空 NPU cache。
    退出码: 0 成功, 非 0 失败。
    """
    ...

if __name__ == "__main__":
    import sys
    sys.exit(run(sys.argv[1], sys.argv[2]))
```

主进程通过 `subprocess.run([sys.executable, _forensics_child.py, task_dir, dump_dir], check=False)` 调用。超时、stderr 重定向、异常序列化都在 child 内处理。

### 2.5 Execution / Comparison Semantic Parity（codex CRITICAL 修订）

**Comparison mask 与 verification_ascendc 完全一致**：

- 使用 verification 的 `_tensor_diff_summary` 逻辑计算 mismatch mask（含 `torch.nan_to_num` 预处理），这就是 `basic_stats.num_mismatched` / `mismatch_ratio` / `match_rate` 的来源
- **atol/rtol 触发**: `_contains_int8_tensor(ref_out_tree) and _contains_int8_tensor(cand_out_tree)` 时切 `atol=1.5, rtol=0.0`（与 verification 完全一致）；forensics 的 `int8_special_tolerance` 字段记录该状态

**Diagnostic stats**（DiffAnalyzer 独有）:

- `error_distribution` / `worst_elements` / `tail_analysis` 等在 **nan_to_num 之前的原始 tensor** 上做，额外在 `value_range` 里记 `golden_has_nan` / `golden_has_inf` / `actual_has_nan` / `actual_has_inf` 真实计数（不是处理后的 0）
- 这样两层分离：**comparison_mask 与 bench 一致**（pass/fail 绝不漂移），**diagnostic_stats 是额外叙事**（包含 NaN/Inf 原信号）

### 2.6 Primary Hint（codex HIGH 修订）

primary hint 来自**跨 path 的 worst output**（按 mismatch_ratio 降序 tiebreak max_abs_diff 降序取第一），不是 `outputs[0]`：

```python
worst_output = max(
    output_reports,
    key=lambda o: (o["case_aggregate"]["mismatch_ratio_max"],
                   o["case_aggregate"]["max_abs_diff_max"])
)
ph = worst_output["pattern_hint"]
report["primary_hint"]       = ph["primary_hint"]
report["primary_confidence"] = ph["primary_confidence"]
report["primary_evidence"]   = ph["primary_evidence"]
report["all_hints"]          = ph["all_hints"]
```

其中 `worst_output.pattern_hint` 是该 output 的 `representative_case` 的 pattern_hint（这个 case 是该 output 里 mismatch_ratio 最大的）。

### 2.7 L8_operator 丰富（codex HIGH 修订）

现有 `OperatorTypeDetector.detect(op_name, task_dir)` 返回 `{op_type, source, confidence, pattern_priority}`，缺 `attributes` 和 `reduction_axis`。

修改方案（P2 里做）：

```python
class OperatorTypeDetector:
    def detect(self, op_name, task_dir):
        base = self._detect_type(op_name)  # 现有逻辑
        # 新增: 从 model.py 提取 attributes（静态解析，不运行代码）
        base["attributes"] = self._extract_attributes(task_dir)
        # 新增: 对 reduction 类算子提取 reduction_axis
        if base["op_type"] in ("reduction", "normalization", "pooling"):
            base["reduction_axis"] = self._infer_reduction_axis(task_dir, base["attributes"])
        else:
            base["reduction_axis"] = None
        return base

    def _extract_attributes(self, task_dir):
        """静态读 model.py 的 Model.__init__ / forward 签名，提取 kernel_size / dim / axis / 
        normalized_shape 等常见属性。失败返回 {}。"""
        ...

    def _infer_reduction_axis(self, task_dir, attrs):
        """从 attributes + input_groups 第一组 inputs 的 shape 推断归约轴长度。"""
        ...
```

这两个新方法是 **static 解析**（ast.parse），不需要跑模型；在主进程即可完成。

### 2.8 Cross-case Aggregation（codex HIGH 修订）

**每个 output path 都跑所有 case**，产出 `per_case` 列表 + `case_aggregate` 聚合 + `representative_case_idx`。

Representative case 选择：`argmax(mismatch_ratio)`，tiebreak `max_abs_diff`。

`shape_conditional` 启发式: 如果 `per_case[i].basic_stats.mismatch_ratio` 与 `input_groups[i]` 的 last_dim 单调相关（Spearman 相关系数 > 0.7），标 True。

### 2.9 Int8 语义（codex HIGH 修订）

**Trigger 语义完全继承**: `_contains_int8_tensor(ref_out) and _contains_int8_tensor(cand_out)` → 进入 int8 path，`atol=1.5, rtol=0.0`。这个判断以**整份 normalized output tree**（不是单个 tensor）为粒度，与 verification_ascendc 一致。

**Int8 时 diagnostic 补充**: 在 `basic_stats` 里额外记 `int8_special_tolerance: true`；`error_distribution` 里增加 `int_delta_histogram`（整数直方图），但**不覆盖** `max_abs_diff`/`mean_abs_diff` 的浮点语义（这些继续用 `abs(ref - cand)` 浮点算）。

### 2.10 NaN/Inf 处理（codex MEDIUM 修订）

- **comparison mask**: 走 verification 同款 `torch.nan_to_num(tensor, nan=0.0, posinf=1e9, neginf=-1e9)` 后算，保证 pass/fail 与 bench 一致
- **diagnostic stats**: DiffAnalyzer 用 **原始 tensor**（nan_to_num 之前）计算 `value_range.golden_has_nan` 等真实计数；top-level `nan_inf_detected` 字段汇总

---

## 3. 文件改动清单

| 文件 | 动作 | 行数估计 |
|------|------|---------|
| `skills/ascendc/precision-tuning/scripts/precision_forensics.py` | 重写 `OperatorExecutor`（改为 subprocess 调用）、新增 `DiffAnalyzer`、`MemoryLayoutAnalyzer`、`OutputFlattener`；丰富 `OperatorTypeDetector`；重写 `PrecisionForensics.run` 组装逻辑 | -400 / +700 |
| `skills/ascendc/precision-tuning/scripts/_forensics_child.py` | 新建；复制 loader helpers；跑模型 dump tensor | +250 |
| `skills/ascendc/precision-tuning/scripts/test_executor_parity.py` | 新建 parity 测试；覆盖 nested output、shape/type mismatch、int8、NaN/Inf、missing kernel/build | +300 |
| `skills/ascendc/precision-tuning/SKILL.md` | 无需改；新 schema 对 Sub-step 2.1 字段要求**完全兼容**（dtype 从 `L8_operator` 或 `outputs[0].output_dtype` 取，`L6_memory_layout.inputs` 在 top-level） | 0 |
| `skills/ascendc/precision-tuning/scripts/precision_gate.py` | 无需改；schema 向后兼容 | 0 |
| `skills/ascendc/precision-tuning/scripts/precision_knowledge.py` | 无需改 | 0 |

---

## 4. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|-----|------|------|------|
| 复制的 loader 与 verification_ascendc 漂移 | 中 | 语义不一致，诊断与 bench pass/fail 矛盾 | Parity test `test_executor_parity.py` 作为 CI 必跑项；loader 文件顶部强 comment 指向 verification_ascendc 对应函数 |
| nested output 扁平化路径与 verification path 不一致 | 低 | path 名字对不上，log 无法追溯 | parity test 对固定 fixture 断言 `flatten()` 的 path 集合 == `_compare_values` 产生的 path 集合 |
| 超大 tensor numpy 拷贝 OOM（e.g. shape=[2,16,32,32,32]） | 中 | 诊断失败 | Child 端每个 case dump 完立即释放；主进程分析完单个 path 立即 `del`；`DiffAnalyzer` 设 `MAX_ELEMS=10_000_000`，超过时按 `worst_elements + random sample` 降采样 |
| Child process crash（NPU OOM / 模型 bug） | 中 | 诊断失败 | 捕获非 0 exit，写 `status: "error"` + `traceback` 到 report；parent 不跟着崩 |
| sys.path 修改影响主进程 | 低 | 隐蔽 bug | child process 里做，不污染 parent |
| parity test 本身维护负担 | 低 | CI 时间 | fixture 用最小算子（e.g. `3_Add`），单次 < 30s；只在改 loader 时跑 |
| L8_operator.attributes 静态解析失败 | 中 | attributes={} 降级 | 解析失败不阻断；report 仍有 op_type |
| history_trend 跨版本 schema 不兼容 | 低 | 旧 forensics_report 读不出 | HistoryComparator 用 `.get()` 容错；缺字段按 null 处理 |

---

## 5. Rollout / 实施阶段

**三阶段，每阶段独立可 revert**：

### Phase 1: 纯新增分析类（零风险）
- 新增 `DiffAnalyzer`（~280 行，纯 numpy）
- 新增 `MemoryLayoutAnalyzer`（~50 行）
- 新增 `OutputFlattener`（~80 行）
- 配套单元测试 `test_analyzers.py`：对 synthetic golden/actual numpy 数据验证字段完整性（**不需要 NPU**）
- `PrecisionForensics.run` 保持不动，这些类还没被调用 → **零行为改变**
- 验证: `python -m pytest test_analyzers.py` pass
- commit 独立

### Phase 2: 新增 child executor + parity test
- 新增 `_forensics_child.py`（复制 loader + 跑模型 + pickle dump）
- 新增 `test_executor_parity.py`（对 `archive_tasks/avg_pool3_d` 等 fixture 跑 child，和 verification_ascendc.py subprocess 对比）
- 丰富 `OperatorTypeDetector`：加 `attributes` + `reduction_axis` 静态解析
- `PrecisionForensics.run` 仍走旧路径 → **零行为改变**
- 验证: `python test_executor_parity.py` 全部 pass（fixture 覆盖 nested output、int8、shape mismatch、missing kernel/build）
- commit 独立

### Phase 3: 切 `PrecisionForensics.run` 到新路径
- 改 `PrecisionForensics.run` 组装新 JSON（新 schema 字段 + 保留旧 top-level）
- 新旧 schema 并存期：增加 `--legacy` flag 让调用方按需回退
- 跑一次完整 precision tuning flow 对 `3_Add_pt1` 或 `4_Abs_pt1`
- 验证:
  - `precision_gate.py check_forensics` PASS
  - JSON 里 `outputs[0].error_distribution.sign_analysis.bias_direction` 非 null
  - `outputs[0].worst_elements` 至少 3 条，每条含 index/golden_val/actual_val/abs_diff
  - `L6_memory_layout.inputs` 非空
  - `primary_hint` 来自 worst output（不是 output[0]）
- commit 独立，可 revert

**回滚策略**: 任何 Phase 失败，revert 该 commit；前面 Phase 的新类作为 dead code 留着不影响现有调用路径。

---

## 6. 成功标准

### 必须（blocking）
- ✅ `precision_gate.py check_forensics` 继续 PASS
- ✅ `precision_gate.py _get_baseline_match_rate()` 三层回退都能正确读到
- ✅ parity test 对 ≥ 3 个 fixture pass（含一个 int8 / 一个 NaN/Inf / 一个 nested output）
- ✅ JSON 里所有原 null/空字段被真实数值填充
- ✅ primary_hint 来源: worst output（跨 case aggregate 后的 `max(mismatch_ratio)`）

### 应该（non-blocking, 但强烈期望）
- ✅ `L8_operator.attributes` 和 `reduction_axis` 对至少 3 种算子类型（pooling / reduction / normalization）正确提取
- ✅ `shape_conditional` 检测对已知 broadcast 问题（如 3_Add 的 broadcast mismatch）返回 True
- ✅ 单次 forensics 时间 < 2× 旧版（估算 30-60s vs 15-30s）

### Nice-to-have
- `representative_case_idx` 指向最 informative 的 case
- parity test 能捕获 `torch.nan_to_num` 替换为其他 sentinel 的语义漂移

---

## 7. Open Questions（留给实现）

1. **pickle 是否需要跨版本**: child 和 parent 必然同一 Python 环境；用 pickle 直接 round-trip 即可，不用 torch.save/load（后者对 numpy array 元数据丢失）。
2. **parity test fixture 选取**: 初版 `archive_tasks/avg_pool3_d` + `archive_tasks/rms_norm` + `outputs/.../3_Add_pt1`（broadcast）+ 一个合成 int8 算子；后续按需扩充。
3. **`shape_conditional` 的相关系数阈值**: 初版 0.7；观察误报后调整。

---

## 8. Non-Goals

- ❌ 不重构 `precision_gate.py`
- ❌ 不优化 forensics 运行速度（除非超过 2× 旧版）
- ❌ 不实现 L5 `IntermediateProbe`（保留 stub 接口）
- ❌ 不实现 L7 `CodeMapper`（保留 stub 接口）
- ❌ 不动 `precision_knowledge_base.json` 内容
- ❌ 不涉及 DSL / TileLang 链路
- ❌ 不修改 `utils/verification_ascendc.py`（读 only 当作 oracle）

---

## 9. Codex Review 已解决项清单

| Codex 级别 | 问题 | 解决位置 |
|-----------|------|---------|
| CRITICAL | JSON schema 向后不兼容 | §2.2 严格保留 top-level，所有字段只增不删不改结构 |
| CRITICAL | copy + 独立演化无 parity | §1.3 / §2.5 / Phase 2 加 parity test 作为 gate |
| CRITICAL | 硬编码 `List[List[Tensor]]` 数据模型错 | §2.3 `OutputFlattener` 支持任意 nested 结构 |
| CRITICAL | 移除 subprocess 隔离 | §2.1 / §2.4 主-子进程架构，执行留在 child |
| HIGH | 跨 case 抛弃信息 | §2.8 保留 `per_case` + `case_aggregate` + `representative_case_idx` |
| HIGH | `primary_hint` 来自 `outputs[0]` | §2.6 改为 worst output |
| HIGH | int8 路径语义含糊 | §2.9 完全继承 verification trigger，`int_delta` 作为追加视图 |
| HIGH | SKILL.md 字段不匹配 | §2.7 enrich `L8_operator`（attributes + reduction_axis） |
| MEDIUM | `_get_device` 设备语义误描述 | §2.4 / §1.3 文档澄清（不读 ASCEND_RT_VISIBLE_DEVICES） |
| MEDIUM | NaN/Inf 处理与 verification 不一致 | §2.10 comparison_mask 完全继承，diagnostic 额外记录 |
| MEDIUM | 性能：tensor 全缓存 | §2.4 Child 端每 case pickle dump + 立即释放 |
| MEDIUM | P1 "零风险"不成立 | §5 Phase 1 缩窄到纯 numpy 类 + 单测；P2 独立加 parity |
