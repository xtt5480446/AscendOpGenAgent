# Precision Tuning Skill — 设计说明

## 概述

修复 AscendC 算子的 **build / import / runtime / timeout / precision** 五类失败。作为 AscendOpGenAgent 中主 agent Phase 8 条件性 spawn 的 debug subagent：主 agent 生成 + 初筛（Phase 0-6）→ Phase 7 `trace-recorder` 判定 `debug_eligible == true` → 进入本 Skill 做深度 debug。

**历史兼容**：原 Skill 只覆盖精度失败，现扩展为五类失败统一入口；精度分支路径（Step 1-P）保持与旧版本兼容。

本 Skill 采用**双 Subagent 架构**，共用同一套基础设施，但提供两种不同的审计策略。

## 文件结构

```
AscendOpGenAgent/
├── agents/
│   ├── ascendc-debug-agent-discovery.md      # 发现式 Subagent（直接从取证 / 日志数据推理）
│   └── ascendc-debug-agent.md                # 构建式 Subagent（Phase A→B→C 规范化审计）
└── skills/ascendc/ascendc-debug/
    ├── SKILL.md                           # Skill 执行手册（Step 0 ~ Step 7，含 Step 1-P/B/I/R/T）
    ├── README.md                          # 本文件：设计文档
    ├── STRUCTURE.md                       # 目录结构说明
    ├── scripts/                           # 共用脚本
    │   ├── precision_forensics.py         # 数值取证 (L0-L4 + L6 + L8 + available_files)
    │   ├── precision_gate.py              # Gate 入口路由器（派发到 gates/ 分支层）
    │   ├── eval_status.py                 # eval_status.json loader / validator
    │   ├── precision_knowledge.py         # 知识库管理: load / search / dump
    │   ├── anticheat.py                   # 反作弊: wrapper hash + AST + C++ 源码扫描
    │   └── gates/                         # 2 层 Gate 包：通用层 + 分支层
    │       ├── __init__.py
    │       ├── common.py                  # 通用层: 反作弊 / AST / baseline / eval_status / 目录完整性
    │       ├── branch_precision.py        # 1-P 分支: F/A/V 精度 Gate (抽自原 precision_gate.py)
    │       ├── branch_build.py            # 1-B 分支: 编译错误 Gate
    │       ├── branch_import.py           # 1-I 分支: import kernel_side 符号 Gate
    │       ├── branch_runtime.py          # 1-R 分支: runtime crash Gate
    │       └── branch_timeout.py          # 1-T 分支: 死锁 / 死循环 Gate
    └── references/                        # 共用参考资料
        ├── precision_knowledge_base.json  # 精度问题知识库
        └── decomposition_examples/        # 算子计算分解示例
            ├── README.md                  # 分解示例索引
            ├── softmax.md                 # Softmax: 单行归约 (5 步)
            ├── layer_norm.md              # LayerNorm: 单行归约 3-pass (7 步)
            ├── reduce_sum.md              # ReduceSum: 单步归约 + 跨步访存 (2 步)
            ├── mse_loss.md                # MSELoss: 跨核两阶段归约 (6 步)
            ├── matmul.md                  # MatMul: 分块累加 (4 步)
            ├── average_pooling2d.md       # AvgPool2d: 滑窗累加 (3 步)
            └── cumsum.md                  # CumSum: 前缀累加 (2 步)
```

**上下游依赖**：
- 上游：
  - 主 agent 产出 `{task_dir}/kernel/`、`model.py`、`model_new_ascendc.py`、`trace.md`（含 Phase 7 写入的 `final_status` JSON block）
  - `utils/eval_wrapper.py` 产出 `{task_dir}/.eval_status/latest.json`（及 `phase{N}_attempt{M}.json` / `.eval_logs/phase{N}_attempt{M}.log`）
- 下游：
  - 主 agent Phase 8 spawn 后处理读取 `{task_dir}/debug_trace.md` + `{task_dir}/debug_status.json`（本 skill 退出前强制产物）
  - `utils/verification_ascendc.py`（评测）、`utils/run_ascendc_debug.sh`（批量调度 + 反作弊）

## 2 层 Gate 架构

| 层 | 文件 | 职责 |
|---|---|---|
| 通用层 | `scripts/gates/common.py` | 所有分支共享的不变量：反作弊 hash / AST 退化 / baseline 存在 / 目录完整性 / `eval_status` 产出 / `.json.bak` 未被破坏 |
| 分支层 | `scripts/gates/branch_*.py` | 每类失败的专属 F/A/V 语义；audit section schema 由各分支自定义（通用层不强制统一 schema） |

`precision_gate.py` 作为入口路由器：先跑通用层 → 通过后按 `eval_status.failure_type` 派发对应 `branch_*.py`。

## 双 Subagent 架构

### 发现式 Subagent (`ascendc-debug-agent-discovery`)

**审计策略**: 直接从数值取证数据出发，运用 AscendC 领域知识推理根因。

**特点**:
- 不强制预读参考示例，依赖 Agent 自身的 AscendC 知识储备
- 快速从 diff 模式锁定嫌疑区域
- 适用场景：Agent 对 AscendC API 规范已有充分了解

### 构建式 Subagent (`ascendc-debug-agent`)

**审计策略**: 严格遵循 Phase A→B→C 的构建式流程。

**特点**:
- Phase A: 先建规范，再看代码（强制读取 `archive_tasks/` 对应案例 kernel + `ascendc-translator` references）
- Phase B: 读取当前 `{task_dir}/kernel/` 实现
- Phase C: 以 `[REFERENCE_IMPL_SPEC]` 为基准进行结构化对照
- 适用场景：需要严格参照规范进行审计

### 共用基础设施

| 组件 | 类型 | 说明 |
|------|------|------|
| `precision_forensics.py` | 脚本 | L0-L8 数值取证 |
| `precision_gate.py` | 脚本 | 链式 Gate 验证 + 循环控制 |
| `precision_knowledge.py` | 脚本 | 知识库管理 |
| `anticheat.py` | 脚本 | 反作弊: 禁改 wrapper + 扫 C++ 禁调 `at::<op>` |
| `precision_knowledge_base.json` | 数据 | 精度问题模式库 |
| `decomposition_examples/` | 文档 | 算子计算分解示例 |

### 策略对比

| 维度 | 发现式 | 构建式 |
|------|--------|--------|
| **分析起点** | 直接从数值取证数据出发 | 先建立规范基准 |
| **Phase A** | 可选查阅 | 强制读取 lowering 示例 |
| **Reference** | 按需查阅 | 必须产出 `[REFERENCE_IMPL_SPEC]` |
| **优势** | 快速、灵活 | 严谨、系统化 |
| **适用** | 经验丰富的 Agent | 规范要求高的场景 |

## 反作弊机制

精度调优的"作弊"指：通过修改 Python wrapper 或在 kernel C++ 里调用 libtorch 计算 API 来掩盖精度失败，而不是真正修复 AscendC kernel。

**三层检测**（由 `anticheat.py` 实现，`utils/run_ascendc_debug.sh` 在每个任务前后自动调用）：

1. **wrapper hash 对比**：调优前 snapshot `model_new_ascendc.py` / `model_new_tilelang.py` 的 sha256，调优后对比；变化 = 作弊（wrapper 本应保持不变）
2. **Python AST 退化检测**：通过 `skills/ascendc/ascendc-translator/scripts/validate_ascendc_impl.py` 检测 wrapper 4 类退化（无扩展导入 / 未调用 kernel / 混用 torch / 标量 for 循环）
3. **C++ 源码扫描**：扫 `kernel/**/*.cpp|h`，捕获 `at::<非白名单 op>` / `torch::<op>` / tensor 计算方法（`x.cumsum()`/`x.histc()` 等）/ `#include <ATen/ops/*.h>` / 缺失 kernel launch

**违规处理**：恢复 wrapper baseline + 标记 🚨 CHEAT（不重跑，进入人工审查队列）。

**agent 文件**（`agents/ascendc-debug-agent{,-discovery}.md`）有"反作弊约束"硬规则章节，明确唯一可改目录为 `kernel/`。

## 信息层级 (L0-L8)

| 层级 | 信息类型 | 状态 | 实现位置 |
|------|---------|------|---------|
| L0 | PASS/FAIL | ✅ 已实现 | precision_forensics.py |
| L1 | 统计值 + 误差分布 | ✅ 已实现 | DiffAnalyzer |
| L2 | 位置特征 (尾块/维度/边界) | ✅ 已实现 | DiffAnalyzer._tail_analysis / _dimension_analysis |
| L3 | 数值特征 (幅值/NaN/符号) | ✅ 已实现 | DiffAnalyzer._check_magnitude_correlation 等 |
| L4 | 张量切片 (per-index) | ✅ 已实现 | DiffAnalyzer._worst_elements |
| L5 | 中间结果探测 | ❌ 不实现，由 L7 替代 | 见下方 L5 设计决策 |
| L6 | 内存布局分析 | ✅ 已实现 | MemoryLayoutAnalyzer |
| L7 | 代码位置映射 | ✅ Agent 手动完成 | Sub-step 2.3 L7 手动映射 (静态推算) |
| L8 | 算子语义 | ✅ 部分实现 | OperatorTypeDetector + 知识库 CHECKLIST |

## 分工原则

| 操作 | 执行者 | 原因 |
|------|--------|------|
| L0-L4 数值取证 | Python | 确定性计算 |
| L6 内存布局 | Python | tensor 属性读取 |
| L8 算子类型检测 + 属性提取 | Python | 规则推断，含 dim/reduction_axis |
| 可用文件检测 | Python | 纯文件 IO |
| Pattern hint 分类 + 语义加权 | Python | 规则化，作为建议 |
| Gate 验证 + 循环控制 | Python | 结构化检查，Agent 不可覆盖 |
| 知识库 IO + RAG 检索 | Python | 结构化评分，纯文件操作 |
| 反作弊 hash / AST / C++ 扫描 | Python | 确定性检测 |
| 算子计算分解 (Sub-step 2.2) | Agent | 需要理解参考实现语义 |
| AscendC 逐步对照 (Sub-step 2.3) | Agent | 需要理解 C++ 代码结构 |
| 根因诊断 (Sub-step 2.4) | Agent | 需要推理 |
| 修复计划 + 代码修复 | Agent | 创造性工作 |

## 链式 Gate 设计

每个 Gate 不仅检查当前步骤产物，还验证前序步骤是否完成：

```
Gate-F (forensics) → 无前置依赖，检查 attempt 号匹配
Gate-A (audit)     → 前置: forensics 存在且 attempt 匹配
                     检查 7 个必填 section: FORENSICS_SUMMARY, COMPUTATION_DECOMPOSITION,
                     REFERENCE_IMPL_SPEC, KERNEL_STEP_TRACE, ROOT_CAUSE, FIX_PLAN, TARGET_FILES
                     attempt > 0 时额外检查: DIRECTION_ASSESSMENT（严格二值"是/否"）
Gate-X (fix)       → 前置: audit 存在
Gate-V (validate)  → 前置: 代码文件存在
```

返回码区分：
- 0: 通过
- 1: 产物不完整（可重试）
- 2: 前置依赖缺失（必须回退补完前序步骤）

## 循环控制

Gate-V 输出 `loop_signal`，Agent **必须遵守**：

| 信号 | 条件 | Agent 操作 |
|------|------|-----------|
| PASS | 精度通过 | 成功收尾 + 知识库跃迁 |
| CONTINUE | 未通过但 mismatch 有改善 | 归档本轮，回到 Step 1 |
| STOP | 达到 `MAX_ATTEMPTS` 轮上限 或 连续 `MAX_STAGNANT_ROUNDS` 轮无改善 | 失败报告 |

## 知识库

扁平五字段结构（title / feature / reason / fix / type），RAG-ready。包含两类条目：

1. **问题模式**（9 条）: 具体精度问题的 feature/reason/fix
2. **算子 CHECKLIST**（5 条）: 按算子类型的精度检查清单（reduction/pooling/loss/matmul/normalization）

成功修复后自动追加跃迁条目（仅成功时写入，避免污染）。

### RAG 检索

**方案**：结构化关键词筛选 + 评分排序（不使用向量嵌入）

**设计决策**：
1. aarch64 环境下 FAISS / sentence-transformers 依赖重
2. 当前规模（18 条，预期增长到 100-200 条）不需要向量检索
3. 知识库已有结构化标签（`type`、`feature` 中的 `pattern=xxx` / `op_type=xxx`），精确匹配比语义相似度更可靠
4. Fallback 到全量 load 保底，不会漏掉任何条目

**实现**：`precision_knowledge.py search` 命令

```bash
# 第一次检索 (Sub-step 2.1 完成后)
python3 precision_knowledge.py search \
    --kb-path <path> --op-type <type> --pattern <hint> --top-k 3 \
    --log-path <tuning_dir> --attempt N --call-index 0

# 第二次精化检索 (Sub-step 2.4 开始前, 增加位置特征)
python3 precision_knowledge.py search \
    --kb-path <path> --op-type <type> --pattern <hint> \
    --position <tail/boundary/scattered> --top-k 3 \
    --log-path <tuning_dir> --attempt N --call-index 1
```

**评分逻辑**：
- pattern 匹配 feature 中的 `pattern=xxx` → 权重 3
- op_type 匹配 → 权重 2
- type 字段与 pattern→type 亲和性映射匹配 → 权重 1
- position 与 pattern 亲和性映射匹配（仅第二次检索）→ 权重 1

**返回结构**：
- `matched_entries`: top-K 普通条目（按 score 降序）
- `checklists`: op_type 匹配的 CHECKLIST 条目（不占 K 配额，始终附带）
- `fallback_to_full_load`: 无任何命中时自动全量返回

## 计算分解示例 (`decomposition_examples/`)

Step 2 的 Sub-step 2.2 要求 Agent 将算子的参考实现分解为逐步计算链。`references/decomposition_examples/` 提供 7 个算子的分解示例，覆盖 5 种计算模式：

| 计算模式 | 示例算子 | 关键审计点 |
|---------|---------|-----------|
| 单行归约 | softmax, layer_norm | padding 值、count 对齐、归约维完整性 |
| 跨核归约 | mse_loss | workspace 同步、Phase 2 正确性、分母计算 |
| 分块累加 | matmul | 累加器初始化、分块边界、精度累积 |
| 滑窗累加 | average_pooling2d | 有效面积计算、边界/padding 处理 |
| 前缀累加 | cumsum | 跨 tile 累加器传递、顺序正确性 |

Agent 在 Sub-step 2.2 中：
1. 先读取 `decomposition_examples/README.md` 了解格式和模式分类
2. 查找与当前算子最匹配的示例文件
3. 按示例的粒度标准完成计算分解
4. 每步标注精度风险点和知识库关联

> 发现式 Subagent 可选查阅分解示例，构建式 Subagent 则强制参考。

## 评测链路一致性

精度调优的 tensor 获取方式与 bench 评测 (`utils/verification_ascendc.py`) 保持**语义一致**，但**代码独立演化**：

- 模型加载：`model.py` 的 `Model` + `model_new_ascendc.py` 的 `ModelNew`（通过 `importlib`）
- 输入：`get_input_groups()` 优先，fallback 到 `get_inputs()`
- 初始化：`ModelNew.get_init_inputs()` 优先（candidate 可覆盖 ref 的 init 参数）
- 容差：`atol=1e-2, rtol=1e-2`；int8 特判 `atol=1.5, rtol=0.0`
- 设备：`ASCEND_RT_VISIBLE_DEVICES` 环境变量

`precision_forensics.py` **复制**（不 import）了 `utils/verification_ascendc.py` 的 tensor 加载辅助函数到 `OperatorExecutor`，以便独立增强（如直接 dump tensor 做 L1-L4 深度分析）而不破坏 bench。语义升级时需同步两处。

## L5 设计决策：不实现，由 L7 Agent 手动映射替代

**背景**: 早期设计希望通过 Python 脚本自动探测 Kernel 内部每个计算步骤的中间输出，定位误差引入步骤。

**技术评估**:

| 维度 | 评估结果 |
|------|---------|
| 实现路径 | 需在 kernel 代码中插入额外 GlobalTensor 输出、修改 host TilingFunc 分配 workspace、重新完整编译 |
| 通用性 | 每个算子的中间步骤不同（ReduceMax/ReduceSum/Exp 各自需要独立探针），无法通用 |
| 侵入风险 | 探针修改可能影响 buffer 对齐，改变精度问题表现（Heisenbug 效应） |
| 工程代价 | 相当于为每个算子维护一个 debug 版本，成本高、不可复用 |

**替代方案**: Sub-step 2.3 的 L7 手动映射已覆盖 L5 的核心价值:
- `worst element index [row, col]` → 静态推算落在哪个 Core、main/tail block
- 对照 tiling 参数（`tileLength`、`rowsPerCore`）直接定位到 Compute() 中的 K-Step
- 对单行归约、逐元素算子完全够用
- 对跨核归约（Phase1→Phase2），L7 同样可通过 workspace 地址偏移推算

**结论**: L5 Python 实现不可行，已由 L7 Agent 手动映射替代。`IntermediateProbe` 类保留为存根，不再标记为 TODO。

## 详细中间文件说明

见 `STRUCTURE.md`。

## TODO 接口清单

| 接口 | 类 | 位置 | 状态 | 说明 |
|------|-----|------|-------|------|
| 中间结果探测 | IntermediateProbe | precision_forensics.py | ❌ 不实现（见 L5 决策） | 接口存根，不调用 |
| 代码位置映射 | CodeMapper | precision_forensics.py | ❌ 不实现，由 Agent 手动完成 | Sub-step 2.3 L7 手动映射 |
