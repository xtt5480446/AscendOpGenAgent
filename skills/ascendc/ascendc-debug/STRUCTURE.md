# ascendc-debug 目录结构

```
skills/ascendc/ascendc-debug/
├── README.md                          # Skill 说明文档（设计概览、双 Subagent 架构、知识库结构、gates/ 2 层架构）
├── SKILL.md                           # Agent 执行手册（Step 0 ~ Step 7，含 Step 0.3 分流 + Step 1-P/B/I/R/T + Step 7 退出前产物）
├── STRUCTURE.md                       # 本文件：目录结构示意图
│
├── references/                        # 静态参考资料
│   ├── precision_knowledge_base.json  # 精度问题知识库（问题模式 + 算子 CHECKLIST）
│   │
│   └── decomposition_examples/        # 算子计算分解示例（Sub-step 2.2 参考）
│       ├── README.md                  # 示例格式说明与模式分类索引
│       ├── average_pooling2d.md       # 滑窗累加模式
│       ├── cumsum.md                  # 前缀累加模式
│       ├── layer_norm.md              # Normalization 模式
│       ├── matmul.md                  # 分块累加模式
│       ├── mse_loss.md                # Loss 计算模式
│       ├── reduce_sum.md              # Reduction 模式
│       └── softmax.md                 # Softmax（Reduction + Normalization）
│
└── scripts/                           # 精度 / debug 工具脚本（Subagent 共用）
    ├── precision_forensics.py         # 取证脚本：运行算子、采集误差数据（1-P 分支专用）
    ├── precision_gate.py              # Gate 入口路由器：先跑通用层，再按 failure_type 派发到 gates/ 分支层
    ├── eval_status.py                 # eval_status.json loader / validator（读取 utils/eval_wrapper.py 产出）
    ├── precision_knowledge.py         # 知识库管理：load / search / dump
    ├── anticheat.py                   # 反作弊检测：sha256 hash + Python AST + C++ 源码扫描（snapshot/verify/restore 子命令）
    │
    └── gates/                         # 2 层 Gate 包（通用层 + 分支层）
        ├── __init__.py
        ├── common.py                  # 通用层：反作弊 hash / AST 退化 / baseline / eval_status 产出 / 目录完整性
        ├── branch_precision.py        # 1-P 分支：F/A/V 精度 Gate（原 precision_gate.py 精度逻辑抽离）
        ├── branch_build.py            # 1-B 分支：编译错误 Gate（COMPILE_ERROR_CITATION + FIX_TYPE 白名单）
        ├── branch_import.py           # 1-I 分支：import kernel_side 符号 Gate（IMPORT_TRACEBACK_CITATION + 拒绝 env_side fix_type）
        ├── branch_runtime.py          # 1-R 分支：runtime crash Gate（RUNTIME_ERROR_CITATION）
        └── branch_timeout.py          # 1-T 分支：死锁 / 死循环 Gate（SYNC_POINT_ANALYSIS）
```

### `gates/` 2 层 Gate 协议

| 层 | 负责 | 通过条件 |
|---|---|---|
| 通用层 `common.py` | 反作弊 / AST / baseline / `eval_status` / 目录完整性 | 所有不变量成立 |
| 分支层 `branch_*.py` | 对应失败类型的 F / A / V 语义 | 取证产物 / audit section schema / 验证推进判据通过 |

通用层**不**检查 audit section 格式（避免和精度分支现有 `[FORENSICS_SUMMARY] + [REFERENCE_IMPL_SPEC]` 打架，findings §3.3）。每个分支层自定义 audit schema：

| 分支 | 必填 audit section |
|---|---|
| `branch_precision.py` | `[FORENSICS_SUMMARY]` + `[REFERENCE_IMPL_SPEC]` + `[ROOT_CAUSE]` + `[FIX_PLAN]`（兼容旧精度路径 9 section 方案） |
| `branch_build.py` | `[COMPILE_ERROR_CITATION]` + `[ROOT_CAUSE]` + `[FIX_PLAN]` + `[FIX_TYPE]` ∈ whitelist |
| `branch_import.py` | `[IMPORT_TRACEBACK_CITATION]` + `[ROOT_CAUSE]` + `[FIX_PLAN]` + `[FIX_TYPE]` ∈ kernel-side whitelist（显式拒绝 env-side） |
| `branch_runtime.py` | `[RUNTIME_ERROR_CITATION]` + `[ROOT_CAUSE]` + `[FIX_PLAN]` |
| `branch_timeout.py` | `[SYNC_POINT_ANALYSIS]` + `[ROOT_CAUSE]` + `[FIX_PLAN]` |

## Subagent 架构

本 Skill 被两个 Subagent 共用，分别采用不同的审计策略：

| Subagent | 文件位置 | 审计策略 | 特点 |
|----------|----------|----------|------|
| **发现式** | `agents/ascendc-debug-agent-discovery.md` | 发现式审计 | 直接运用 AscendC 领域知识推理根因，不强制预读参考示例，依赖 Agent 自身知识储备快速诊断 |
| **构建式** | `agents/ascendc-debug-agent.md` | 构建式审计 | 严格遵循 Phase A→B→C：先建规范（强制读取 lowering 示例），再看代码，最后结构化对照 |

### 共用组件

两个 Subagent 共享以下基础设施：

| 组件 | 说明 |
|------|------|
| `precision_forensics.py` | L0-L8 数值取证，输出结构化报告（1-P 分支专用） |
| `precision_gate.py` | Gate 入口路由器（派发到 gates/ 分支层）+ 循环控制 |
| `eval_status.py` | `eval_status.json` loader / schema 校验器，消费 `utils/eval_wrapper.py` 产出 |
| `gates/common.py` | 通用层：反作弊 / AST / baseline / eval_status / 目录完整性 |
| `gates/branch_*.py` | 分支层：precision / build / import / runtime / timeout 各自的 F/A/V 语义 |
| `precision_knowledge.py` | 知识库 RAG 检索与管理 |
| `anticheat.py` | 反作弊检测：snapshot / verify / restore，覆盖 Python wrapper 改动与 C++ kernel 偷调 ATen 的场景 |
| `precision_knowledge_base.json` | 精度问题模式库 + 算子 CHECKLIST |
| `decomposition_examples/` | 算子计算分解参考 |

### 策略差异

| 维度 | 发现式 | 构建式 |
|------|--------|--------|
| **Phase A** | 可选查阅参考资料，依 Agent 经验判断 | 强制读取 lowering 示例，产出 `[REFERENCE_IMPL_SPEC]` |
| **分析起点** | 直接从数值取证数据出发 | 先建立规范基准，再对照实际代码 |
| **适用场景** | Agent 对 AscendC API 规范已有充分了解 | 需要严格参照规范进行结构化审计 |
| **Gate-A 要求** | 仍需 `[REFERENCE_IMPL_SPEC]` section | 强制验证 `[REFERENCE_IMPL_SPEC]` 完整性 |

## 文件职责速查

| 文件 | 阶段 | 职责 | 使用方 |
|------|------|------|--------|
| `SKILL.md` | 全流程 | Agent 执行手册，定义 Step 0 ~ Step 7（含 Step 0.3 分流 + Step 1-P/B/I/R/T + Step 7 退出前强制产物） | 双 Subagent |
| `README.md` | 参考 | 设计文档，含双 Subagent 架构、gates/ 2 层架构、知识库结构 | 开发者 |
| `precision_forensics.py` | Step 1-P 取证 | 运行算子取证，输出误差统计与 worst element 数据 | 双 Subagent（1-P 分支专用） |
| `precision_gate.py` | 每步 Gate 末尾 | Gate 入口路由器，派发到 gates/ 分支层；通用层先跑 | 双 Subagent |
| `eval_status.py` | Step 0.3 / 每轮 Step 4 | 读取 `utils/eval_wrapper.py` 产出的 `eval_status.json`，schema 校验 | 双 Subagent |
| `gates/common.py` | 每次 Gate | 反作弊 / AST / baseline / eval_status / 目录完整性，通用不变量 | 双 Subagent |
| `gates/branch_*.py` | 每次 Gate | 对应分支的 F/A/V 语义 + audit section schema | 双 Subagent |
| `precision_knowledge.py` | Sub-step 2.1 / 2.4 / Step 5 | 知识库 RAG 检索、加载、写入 | 双 Subagent |
| `anticheat.py` | Step 0.1 / 每轮编译前 / 验收 | 检测 Python wrapper 被偷改、C++ kernel 偷调 ATen 等退化路径 | 双 Subagent |
| `precision_knowledge_base.json` | Sub-step 2.4 | 已知精度问题模式 + 算子 CHECKLIST | 双 Subagent |
| `decomposition_examples/*.md` | Sub-step 2.2 | 按算子类型提供计算分解示例 | 双 Subagent（构建式强制、发现式可选） |

## 中间文件完整说明

> 本节详述单次调优任务在 `{task_dir}/precision_tuning/` 下生成的所有中间文件，每个文件均注明创建者、创建时机、schema 与用途。

### 一、目录总览

```
{task_dir}/precision_tuning/
│
│── [Step 1]      forensics_report.json              # 取证报告（Python 脚本生成）
│── [Gate-F→]     baseline_state.json                # 初始精度基线（Gate-F 自动写入）
│── [Step 2]      precision_audit.md                 # 深度审计全文（Agent 写入）
│── [Step 2.1/4]  knowledge_search_log_{N}.json      # 知识库检索日志（precision_knowledge.py 写入）
│── [Step 4.2]    compilation_log_{N}.json           # 编译失败日志（仅编译出错时出现）
│── [Step 4.4]    validation_result_attempt_{N}.json # 精度验证结果（Agent 写入）
│── [Gate-A→V]    round_summary_{N}.json             # 本轮综合摘要（Gate-A + Gate-V 分两次写入）
│── [Gate-V→]     tuning_directions.json             # 跨轮方向学习表（Gate-V 每轮追加）
│── [Step 5.1]    candidate_kb_entry.json            # 候选知识库条目（Agent 写入，精度通过后）
│── [Step 5.4]    {op_name}_precision_tuning_process.md  # 完整过程记录（Agent 写入）
│
└── history/
    ├── baseline/code_snapshot/                      # [Step 0.1] 不可变基线代码（Agent cp）
    ├── attempt_{N}/
    │   ├── code_snapshot/                           # [Step 0.2/归档] 本轮起始代码（Agent cp）
    │   ├── sections/                                # [Gate-A→] 各 section 独立文件（Gate-A 自动提取）
    │   ├── forensics_report.json                    # [归档步骤] 取证报告副本（Agent cp）
    │   └── precision_audit.md                       # [归档步骤] 审计报告副本（Agent cp）
    ├── current_best/code_snapshot/                  # [归档步骤] 当前最佳代码（Agent 按 match_rate 更新）
    └── success/code_snapshot/                       # [Step 5.3] 最终成功代码（Agent cp，永久不覆盖）
```

---

### 二、顶层文件详解

#### `forensics_report.json` — Step 1 生成

**创建者**：`precision_forensics.py`（Python 脚本，不可跳过）

**创建时机**：每轮 Step 1 运行后覆盖写入。文件始终反映**当前轮**的取证结果（非历史归档版本）。

**内容**：L0-L8 全层次数值取证结果，是所有后续分析的原始数据源。

```json
{
  "version": "2.0",
  "op_name": "cumsum",
  "attempt": 0,
  "status": "completed",
  "L0_pass": false,
  "outputs": [{
    "basic_stats": {
      "mismatch_ratio": 0.989227,   // 0~1 比例，非百分比
      "match_rate": 0.010773,       // 0~1 比例，非百分比
      "max_abs_diff": 35.73,
      "mean_abs_diff": 10.91
    },
    "error_distribution": { ... },  // 误差分布、符号分析
    "tail_analysis": { ... },       // 尾块 mismatch 率分析
    "dimension_analysis": { ... },  // 各维度 mismatch 率
    "worst_elements": [ ... ]       // top-3 最大误差元素及其位置
  }],
  "primary_hint": "all_wrong",      // 误差模式分类（Gate 和知识库检索均依赖此字段）
  "primary_confidence": 0.90,
  "L6_memory_layout": { ... },      // 输入/输出内存 shape/stride/对齐情况
  "L8_operator": {                  // 算子语义
    "op_type": "reduction",
    "attributes": { "dim": 2 },
    "reduction_axis": { "axis_length": 64 }
  },
  "history_trend": null             // attempt 0 时为 null；attempt ≥ 1 时由取证脚本填入历史 mismatch 趋势
}
```

> **注意**：`match_rate` 和 `mismatch_ratio` 在 forensics 中的单位是 **0\~1 比例**（而非百分比）。Gate 和 `baseline_state.json` 写入时会乘以 100 转换为百分比。

---

#### `baseline_state.json` — Gate-F 通过后自动写入

**创建者**：`precision_gate.py` 的 `_write_baseline_from_forensics()`

**创建时机**：Gate-F 验证通过 **且 attempt == 0** 时立即写入。此时代码尚未被任何修复操作修改，是真正的初始精度基线。

**幂等性**：文件一旦存在则不覆盖，保证 baseline 始终记录第一次取证时的原始精度。

```json
{
  "match_rate": 1.0773,
  "mismatch_ratio": 0.989227,
  "max_abs_diff": 35.73,
  "mean_abs_diff": 10.91,
  "primary_hint": "all_wrong",
  "source": "forensics_report.json/outputs[0]/basic_stats",
  "note": "Initial precision captured at Gate-F before any code modification"
}
```

> **为何在 Gate-F 而非 Gate-V 写入**：Gate-V 在代码修复、编译、验证之后才运行，此时 `forensics_report.json` 已被 attempt 0 的取证结果覆盖，`history_trend` 仍为 null（attempt 0 无历史），从 forensics 读 baseline 是唯一可靠来源。

---

#### `precision_audit.md` — Step 2 写入

**创建者**：Agent（Step 2 各 Sub-step 的产出合并写入单一文件）

**创建时机**：Step 2 深度分析过程中 Agent 逐步追加，Gate-A 通过时文件已完整。每轮覆盖写入（历史版本在归档步骤复制到 `history/attempt_N/`）。

**内容**：包含 9 个结构化 section，Gate-A 逐项校验其存在性和完整性：

| Section | 对应 Sub-step | 内容 |
|---------|--------------|------|
| `[FORENSICS_SUMMARY]` | 2.1 | 取证数据逐字段摘录，含 L6/L8/dtype 判断 |
| `[COMPUTATION_DECOMPOSITION]` | 2.2 | 参考实现的逐步计算链，含精度风险点 |
| `[REFERENCE_IMPL_SPEC]` | 2.3 Phase A | TQue/TBuf 规范、关键 API 签名、非对齐处理规范 |
| `[KERNEL_STEP_TRACE]` | 2.3 Phase B/C | Kernel Compute() 逐步对照，含 L7 手动映射 |
| `[KNOWLEDGE_MATCH]` | 2.4 | 知识库检索结果及 CHECKLIST 逐项核查 |
| `[ROOT_CAUSE]` | 2.4 | 根因判断 + 证据链（数值/布局/代码/逻辑） |
| `[FIX_PLAN]` | 2.4 | 修复类型、修改文件、修改点、预期效果 |
| `[TARGET_FILES]` | 2.4 | 需要修改的文件列表 |
| `[DIRECTION_ASSESSMENT]` | 2.4 | attempt > 0 时填写：是否延续上一轮方向 + 换方向理由 |

Gate-A 通过后，`precision_gate.py` 自动将上述每个 section 提取为独立的 `.md` 小文件保存到 `history/attempt_N/sections/`，供后续轮次按需读取。

---

#### `knowledge_search_log_{N}.json` — Sub-step 2.1 / 2.4 写入

**创建者**：`precision_knowledge.py search` 命令（每次检索追加一条记录，覆盖写入同一文件）

**创建时机**：每次 Agent 执行 `python3 precision_knowledge.py search ...` 时追加一条记录。每轮最多两次检索（Sub-step 2.1 基础检索 + Sub-step 2.4 精化检索）。

```json
[
  {
    "attempt": null,
    "call_index": 0,           // 0=第一次检索（2.1），1=第二次检索（2.4）
    "timestamp": "...",
    "query": {
      "op_type": "reduction",
      "pattern": "all_wrong",
      "position": null,
      "top_k": 3
    },
    "matched_count": 3,
    "checklist_count": 1,
    "fallback_to_full_load": false,
    "top_titles": [
      "归约轴切分破坏导致局部归约错误 (Reduction Axis Split Error)",
      "..."
    ]
  }
]
```

> 此文件供事后回溯知识库检索的命中质量，不参与 Gate 验证。

---

#### `compilation_log_{N}.json` — Step 4.2 写入（仅编译失败时存在）

**创建者**：Agent（Step 4.2 编译失败时写入，编译通过则不创建此文件）

**创建时机**：每次 AscendC kernel 重编译失败后 Agent 追加写入，最多 3 次重试。

```json
{
  "attempt": 0,
  "entries": [
    {
      "compile_retry": 0,
      "error_category": "undefined_api",
      "error_snippet": "error: 'Vmax' was not declared in this scope",
      "fix_applied": "将 Vmax 改为 Max（AscendC 正确 API 名称）"
    }
  ]
}
```

> 首轮编译通过时该文件不存在。`round_summary_0.json` 的 `index.compilation_log` 字段为 `null`，这是正常情况。

---

#### `validation_result_attempt_{N}.json` — Step 4.4 写入

**创建者**：Agent（Step 4.4 从 `utils/verification_ascendc.py` 的 stdout 解析后写入）

**创建时机**：每轮 Step 4.3 精度验证完成后，Agent 立即写入。Gate-V 读取此文件判断精度是否通过。

```json
{
  "attempt": 0,
  "correctness_passed": true,
  "evaluate_stdout": "INFO - Evaluation correctness: [PASS]\nOutput 0: shape=[16, 32, 64], match_rate=100.00% (32768/32768), max_diff=0.00000e+00, ...",
  "match_rate": "100.00",   // 字符串，百分比数值（不带 % 号）
  "max_diff": "0.0"
}
```

> `match_rate` 在此文件中是**百分比字符串**（如 `"100.00"`），与 `forensics_report.json` 中的 0\~1 比例不同。`_write_round_summary()` 读取时用 `float(mr_str)` 直接得到百分比数值。

---

#### `round_summary_{N}.json` — Gate-A + Gate-V 两阶段写入

**创建者**：`precision_gate.py`（两次写入，分别由 `check_audit()` 和 `check_validate()` 触发）

**第一次写入（Gate-A 通过后）**：`_write_audit_index()` 写入 `diagnostics` + `index` 字段，`metrics` 各字段初始化为 `null`。

**第二次写入（Gate-V 后）**：`_write_round_summary()` 读取 Gate-A 已写内容，合并后补充 `metrics` 数值字段，同时补充 `diagnostics.forensics_hint`、`diagnostics.op_type` 和 `index.compilation_log`。

```json
{
  "attempt": 0,
  "metrics": {
    "match_rate": 100.0,             // 来自 validation_result_attempt_0.json
    "mismatch_ratio": 0.0,
    "improvement_ratio": null,       // attempt 0 且 baseline_state.json 不存在时为 null
    "absolute_improvement": null,    // 同上
    "stop_reason_code": "precision_passed"  // 永不为 null
  },
  "diagnostics": {
    "forensics_hint": "all_wrong",   // Gate-V 从 forensics_report.json 补充
    "op_type": "reduction",          // 同上
    "fix_type": "FIX_PRECISION_LOGIC",          // Gate-A 从 [FIX_PLAN] section 提取
    "changed_locations": ["kernel/cumsum.cpp"], // Gate-A 从 [TARGET_FILES] 提取
    "direction_verdict": null                   // Gate-A 从 [DIRECTION_ASSESSMENT] 提取，attempt 0 可为 null
  },
  "index": {
    "forensics": "precision_tuning/history/attempt_0/forensics_report.json",
    "audit_full": "precision_tuning/history/attempt_0/precision_audit.md",
    "sections": {
      "forensics_summary": "precision_tuning/history/attempt_0/sections/forensics_summary.md",
      "root_cause": "precision_tuning/history/attempt_0/sections/root_cause.md",
      "fix_plan": "precision_tuning/history/attempt_0/sections/fix_plan.md",
      // ... 其余 section 路径
    },
    "code_snapshot": "precision_tuning/history/attempt_0/code_snapshot/",
    "validation": "precision_tuning/validation_result_attempt_0.json",
    "compilation_log": null,
    "tuning_directions": "precision_tuning/tuning_directions.json",
    "forensics_used": "precision_tuning/forensics_report.json"
  }
}
```

**Agent 使用方式**：下一轮 attempt > 0 时，先读 `tuning_directions.json` 获取全局方向概览；若需某轮的具体根因或修复计划，通过 `round_summary_N.index.sections.root_cause` / `.fix_plan` 路径直接定位 section 小文件，而非全量读取 `precision_audit.md`。

---

#### `tuning_directions.json` — Gate-V 每轮追加写入

**创建者**：`precision_gate.py` 的 `_write_tuning_directions()`

**创建时机**：每轮 Gate-V 运行结束时追加当前轮 entry；精度通过时回溯填写所有 entry 的 `contributed` 字段。

**作用**：整个调优过程的**统一方向学习入口**，Agent 每轮优先读此文件获取历史概览，无需逐轮读取 round_summary。

```json
{
  "op_name": "cumsum",
  "final_status": "success",       // "in_progress" | "success" | "failed"
  "entries": [{
    "attempt": 0,
    "fix_type": "FIX_PRECISION_LOGIC",
    "forensics_hint": "all_wrong",
    "direction_verdict": null,      // attempt 0 无上一轮，为 null
    "direction_reason": "首轮分析，根因已定位",  // 从 [DIRECTION_ASSESSMENT] 提取
    "improvement_ratio": null,      // attempt 0 且 baseline 缺失时为 null
    "absolute_improvement": null,   // 同上
    "outcome": "passed",            // "passed" | "improved" | "stagnant" | "regressed"
    "evidence": {
      "forensics_ref": "precision_tuning/history/attempt_0/forensics_report.json",
      "audit_ref":     "precision_tuning/history/attempt_0/precision_audit.md",
      "match_rate":    100.0,
      "mismatch_ratio": 0.0
    },
    "contributed": true             // 仅 final_status=success 时存在
  }]
}
```

---

#### `candidate_kb_entry.json` — Step 5.1 写入

**创建者**：Agent（Step 5.1，精度通过后手动生成）

**创建时机**：Gate-V 返回 `PASS` 后，Agent 在 Step 5.1 基于本轮 `[ROOT_CAUSE]` 和 `[FIX_PLAN]` 生成泛化的知识条目。Step 5.2 的 `precision_knowledge.py dump` 命令读取此文件写入全局知识库。

```json
{
  "title": "Cumsum Host Tiling 与 Python 转置逻辑不一致导致输出全零 (Cumsum Host Tiling Mismatch)",
  "feature": "all_wrong 模式，actual_value 全为 0，Kernel 未写入数据，或部分不匹配且数值偏离",
  "reason": "Python 层的 transpose 逻辑与 Host Tiling 函数对 scan 轴的假设不一致...",
  "fix": "统一 Python 层与 Host Tiling 的维度转置约定...",
  "type": "FIX_PRECISION_LOGIC"
}
```

---

#### `{op_name}_precision_tuning_process.md` — Step 5.4 写入

**创建者**：Agent（Step 5.4，精度通过后写入完整过程记录）

**创建时机**：精度调优成功收尾时一次性写入，是整个调优过程的人类可读总结文档。

**内容**：调优概览（算子名、调优状态、最终 match_rate）、问题现象、取证数据关键发现、根因分析（证据链）、修复计划与代码变更对比、编译验证结果、知识库条目、文件归档说明。

> 此文件是**人工复盘和知识沉淀**的主要阅读对象，不参与任何 Gate 验证。

---

### 三、`history/` 子目录详解

#### `history/baseline/code_snapshot/` — Step 0.1 创建

**创建者**：Agent（SKILL.md Step 0.1 的 shell 命令）

**创建时机**：整个调优流程**首次**执行时（`if [ ! -d history/baseline/code_snapshot ]`），将算子原始代码一次性复制保存。**全程不覆盖**。

**内容**：AscendOpGenAgent 的 task 结构以 `kernel/` 子目录 + Python wrapper 组成，baseline 保存以下内容：

| 条目 | 来源 |
|------|------|
| `kernel/` 目录整拷贝 | `{task_dir}/kernel/`（内含 `*.cpp`、`*_kernel.h`、`*_tiling.h`、`pybind11.cpp`） |
| `model_new_ascendc.py` | `{task_dir}/model_new_ascendc.py` |

**用途**：任何时候可从此恢复到最初基线，是精度回溯和问题复现的最终参照。配合 `anticheat.py snapshot / restore` 子命令可在检测到 Python wrapper 被偷改时自动恢复。

---

#### `history/attempt_{N}/code_snapshot/` — Step 0.2 / 归档步骤创建

**创建者**：Agent（shell 命令）

**创建时机**：
- **attempt 0**：Step 0.2 在首轮开始时复制（与 baseline 相同，记录本轮分析前的代码状态）
- **attempt N+1（N ≥ 0）**：CONTINUE 归档步骤在当前轮结束后复制下一轮起始代码

**内容**：同 baseline/code_snapshot/（整份 `kernel/` 子目录 + `model_new_ascendc.py`），保存该轮修复**前**的代码状态（即本轮分析时读到的代码）。

---

#### `history/attempt_{N}/sections/` — Gate-A 通过后自动提取

**创建者**：`precision_gate.py` 的 `_write_audit_index()`

**创建时机**：每轮 Gate-A 验证通过后立即提取，无需 Agent 手动操作。

**内容**：将 `precision_audit.md` 中的每个 section 提取为独立 `.md` 文件：

| 文件 | 对应 section | 主要内容 |
|------|-------------|---------|
| `forensics_summary.md` | `[FORENSICS_SUMMARY]` | L0-L8 取证摘要 |
| `computation_decomposition.md` | `[COMPUTATION_DECOMPOSITION]` | 算子计算链分解 |
| `reference_impl_spec.md` | `[REFERENCE_IMPL_SPEC]` | AscendC API 规范对照 |
| `kernel_step_trace.md` | `[KERNEL_STEP_TRACE]` | Kernel 步骤逐行追踪 |
| `knowledge_match.md` | `[KNOWLEDGE_MATCH]` | 知识库命中条目 |
| `root_cause.md` | `[ROOT_CAUSE]` | 根因判断 + 证据链 |
| `fix_plan.md` | `[FIX_PLAN]` | 修复计划详情 |
| `target_files.md` | `[TARGET_FILES]` | 修改文件清单 |
| `direction_assessment.md` | `[DIRECTION_ASSESSMENT]` | 方向延续/切换判断 |

**设计意图**：避免下一轮 attempt 读取整个 `precision_audit.md`（几百行），Agent 按需通过 `round_summary_N.index.sections.*` 的路径直接读取对应的单个 section 文件。

> **提取失败处理**：若某 section 未找到，对应 `index.sections.{name}` 置为 `null`，Gate-A 不因此阻断，Agent fallback 读取 `index.audit_full`（完整审计文件）。

---

#### `history/attempt_{N}/forensics_report.json` 和 `history/attempt_{N}/precision_audit.md` — 归档步骤创建

**创建者**：Agent（CONTINUE 时的归档步骤，或 PASS 时的 Step 5.0）

**创建时机**：本轮 Gate-V 返回信号后，Agent 执行 `cp` 命令将顶层的 `forensics_report.json` 和 `precision_audit.md` 归档至 `history/attempt_N/`。

> **注意**：`sections/` 由 Gate-A 自动生成，是可靠的；而 `forensics_report.json` 和 `precision_audit.md` 的副本依赖 Agent 手动执行，存在遗漏风险。若 `history/attempt_N/` 下缺失这两份副本，Agent fallback 读取顶层最新版本（但顶层文件每轮都会被覆盖，无法追溯历史轮次）。

---

#### `history/current_best/code_snapshot/` — 归档步骤动态更新

**创建者**：Agent（CONTINUE 归档步骤，通过比较当前 match_rate 与 `match_rate.txt` 决定是否更新）

**创建时机**：每轮 CONTINUE 时检查当前 match_rate 是否优于历史最优，是则覆盖更新；PASS 时在 Step 5.0 无条件更新（match_rate = 100.0）。

**附属文件**：`current_best/match_rate.txt`，记录当前最优 match_rate 数值，供下一轮归档步骤比较。

**用途**：调优失败（Gate-V 返回 STOP）时，可从此处恢复精度最高的代码继续探索，而无需从头开始。

---

#### `history/success/code_snapshot/` — Step 5.3 创建

**创建者**：Agent（Step 5.3，精度 100% 通过后执行）

**创建时机**：精度验证通过（Gate-V 返回 `PASS`）后，Step 5.3 将最终成功代码复制至此。**不覆盖**（与 baseline/code_snapshot/ 同为不可变存档）。

**与 current_best 的区别**：

| 目录 | 更新时机 | 是否覆盖 | 用途 |
|------|---------|---------|------|
| `current_best/` | 每轮 match_rate 改善时 | 覆盖 | 失败时的最优恢复点 |
| `success/` | 仅在精度 100% 通过时 | 不覆盖 | 成功代码的永久存档 |

---

### 四、文件生成时序图

```
Step 0.1  →  history/baseline/code_snapshot/        (Agent, 首次执行)
          →  anticheat.py snapshot                   (保存 wrapper sha256 到 .bench_baseline/)
Step 0.2  →  history/attempt_0/code_snapshot/       (Agent)
Step 1    →  forensics_report.json                  (precision_forensics.py)
Gate-F    →  baseline_state.json                    (precision_gate.py, attempt 0 自动写入)
Step 2    →  precision_audit.md                     (Agent, 逐 Sub-step 追加)
Step 2.1  →  knowledge_search_log_0.json            (precision_knowledge.py)
Step 2.4  →  knowledge_search_log_0.json            (precision_knowledge.py, 追加第二条)
Gate-A    →  history/attempt_0/sections/*.md        (precision_gate.py, 自动提取)
          →  round_summary_0.json (diagnostics+index)(precision_gate.py)
Step 4.1  →  anticheat.py verify                    (编译前确认 wrapper 未被偷改、kernel 未偷调 ATen)
Step 4.2  →  compilation_log_0.json                 (Agent, 仅编译失败时)
Step 4.4  →  validation_result_attempt_0.json       (Agent)
Gate-V    →  round_summary_0.json (metrics 补充)    (precision_gate.py)
          →  tuning_directions.json                 (precision_gate.py)
            ├─ loop_signal=CONTINUE → 归档步骤:
            │    cp forensics_report.json → history/attempt_0/
            │    cp precision_audit.md   → history/attempt_0/
            │    更新 history/current_best/ (若 match_rate 改善)
            │    cp 起始代码 → history/attempt_1/code_snapshot/
            │    → 回到 Step 1 (attempt+1)
            └─ loop_signal=PASS → Step 5:
                 Step 5.0: 归档 + 更新 current_best (match_rate=100.0)
                 Step 5.1: candidate_kb_entry.json  (Agent)
                 Step 5.2: precision_knowledge.py dump → 写入知识库
                 Step 5.3: history/success/code_snapshot/ (Agent)
                 Step 5.4: {op_name}_precision_tuning_process.md (Agent)
```
