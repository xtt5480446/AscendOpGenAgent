---
name: precision-tuning
description: >
  修复编译通过但精度测试失败的 AscendC 算子。
  通过数值取证 + Agent 深度分析 + 代码修复 + 重新验证的循环实现精度调优。
subagent:
  enabled: true
  agent_type: general
  reason: >
    精度调优涉及取证→深度分析→修复→验证的多步循环,
    需要 Agent 结合数值证据和代码理解做深度推理。
    使用 subagent 允许自主多步执行和错误恢复。
  timeout: 3600
  max_iterations: 60
---

## What I do

修复精度测试失败的 AscendC 算子。流程:
1. Python 脚本收集数值取证数据 (确定性, 不可绕过)
2. Agent 结合取证数据 + 代码 + 知识库做深度分析, 定位根因并制定修复计划
3. Agent 修复代码
4. 重新编译 + 验证
5. 根据 Gate 的循环控制信号决定继续或停止

## When to use me

当 ascendc-evaluation 的 evaluate.py 报告 correctness 失败时。
前提: 算子已通过编译 (generate_pybind.py 成功), 可以运行但精度不通过。

## Prerequisites

- `{output_path}/{op_name}_reference.py` — 含 Model, get_inputs, get_init_inputs
- `{output_path}/{op_name}_custom.py` — 含 ModelNew
- `{output_path}/{OpName}Custom/` — 已编译的 AscendC 项目
- evaluate.py 已报告 correctness 失败

## Workflow

**所有思考、分析、推理必须使用中文。**

**核心原则: Python 脚本做确定性操作 (取证、Gate、知识库), Agent 做需要推理的工作 (分析、修复)。**

---

### Step 0: 初始化

设置轮次计数器 `attempt = 0`。

**0.1 保存不可变基线快照（原始代码，仅首次执行）:**
```bash
if [ ! -d "{output_path}/precision_tuning/history/baseline/code_snapshot" ]; then
    mkdir -p "{output_path}/precision_tuning/history/baseline/code_snapshot"
    cp "{output_path}/{OpName}Custom/op_kernel/{op_name}_custom.cpp" \
       "{output_path}/precision_tuning/history/baseline/code_snapshot/op_kernel.cpp"
    cp "{output_path}/{OpName}Custom/op_host/{op_name}_custom.cpp" \
       "{output_path}/precision_tuning/history/baseline/code_snapshot/op_host.cpp"
    cp "{output_path}/{OpName}Custom/op_host/{op_name}_custom_tiling.h" \
       "{output_path}/precision_tuning/history/baseline/code_snapshot/op_tiling.h"
    cp "{output_path}/{op_name}.cpp" \
       "{output_path}/precision_tuning/history/baseline/code_snapshot/op_pybind.cpp"
    echo "基线快照已保存，后续可从 baseline 恢复"
fi
```

> 基线快照保存在 `history/baseline/code_snapshot/`，整个调优过程中**不覆盖**。如需恢复到最初始状态，使用以下命令：
> ```bash
> cp "{output_path}/precision_tuning/history/baseline/code_snapshot/op_kernel.cpp" \
>    "{output_path}/{OpName}Custom/op_kernel/{op_name}_custom.cpp"
> cp "{output_path}/precision_tuning/history/baseline/code_snapshot/op_host.cpp" \
>    "{output_path}/{OpName}Custom/op_host/{op_name}_custom.cpp"
> cp "{output_path}/precision_tuning/history/baseline/code_snapshot/op_tiling.h" \
>    "{output_path}/{OpName}Custom/op_host/{op_name}_custom_tiling.h"
> cp "{output_path}/precision_tuning/history/baseline/code_snapshot/op_pybind.cpp" \
>    "{output_path}/{op_name}.cpp"
> ```

**0.2 保存本轮起始快照:**
```bash
mkdir -p "{output_path}/precision_tuning/history/attempt_0/code_snapshot"
cp "{output_path}/{OpName}Custom/op_kernel/{op_name}_custom.cpp" \
   "{output_path}/precision_tuning/history/attempt_0/code_snapshot/op_kernel.cpp"
cp "{output_path}/{OpName}Custom/op_host/{op_name}_custom.cpp" \
   "{output_path}/precision_tuning/history/attempt_0/code_snapshot/op_host.cpp"
cp "{output_path}/{OpName}Custom/op_host/{op_name}_custom_tiling.h" \
   "{output_path}/precision_tuning/history/attempt_0/code_snapshot/op_tiling.h"
cp "{output_path}/{op_name}.cpp" \
   "{output_path}/precision_tuning/history/attempt_0/code_snapshot/op_pybind.cpp"
```

> 知识库将在 Sub-step 2.1 完成后通过 `search` 命令按需检索, 无需在此全量加载。

---

### Step 1: 编译 + 安装 + 精度取证

#### 1.0 编译并安装算子运行环境（取证前必须执行）

精度取证脚本需要加载已编译的算子 `.so`，必须先完成编译和安装。

**1.0.1 安装算子包:**
```bash
bash "{output_path}/{OpName}Custom/build_out/custom_opp_ubuntu_aarch64.run" \
    --install-path="{output_path}"
```

**1.0.2 编译并生成 Python 绑定:**
```bash
python3 .opencode/skills/ascendc-evaluation/scripts/generate_pybind.py \
    {op_name} --output-path "{output_path}"
```

**编译失败处理：**
- **attempt=0（初始代码）**：初始代码本不应有编译错误（前置条件要求已通过编译），检查环境变量和路径后停止，报告错误。不进行编译重试，这里不应该失败。
- **attempt>0（修复后代码）**：理论上 Step 4 已编译成功过，此处编译失败属于异常，检查 Step 4 的编译是否真正成功，停止并报告。

> ⚠️ **此处不进行编译重试**。编译重试机制在 Step 4（代码修改后的编译验证）中执行，最多允许 3 次重试。Step 1.0 仅做一次编译——如失败说明环境或代码状态不符合预期。

---

#### 1.1 精度取证 (Python 脚本, 不可跳过)

```bash
python3 .opencode/skills/precision-tuning/scripts/precision_forensics.py \
    {op_name} --output-path "{output_path}" --attempt {attempt}
```

Gate 验证:
```bash
python3 .opencode/skills/precision-tuning/scripts/precision_gate.py \
    --step forensics --op-name {op_name} --output-path "{output_path}" --attempt {attempt}
```

⛔ **Gate-F 未通过 → 停止, 检查错误输出。不要在没有取证数据的情况下分析代码。**
如果报错含 `ImportError` / `ModuleNotFoundError` / `FileNotFoundError`，先确认 Step 1.0 编译和安装是否真正成功，再排查 NPU 环境变量（ASCEND_CUSTOM_OPP_PATH）。

---

### Step 2: 深度分析 + 修复计划 (Agent 推理, 核心步骤)

**本步骤分为 4 个 Sub-step, 每个 Sub-step 有明确的输入文件和产出 section, 不可跳过或合并。**

将全部分析结果写入 `{output_path}/precision_tuning/precision_audit_{attempt}.md`。

**历史扫描（attempt > 0 时必须执行，首轮跳过）：**

**第一步：读方向学习表（一次读完，直接获得跨轮全貌）**
```bash
cat "{output_path}/precision_tuning/tuning_directions.json"
```

从 `tuning_directions.json` 一次性获得：
- 每轮的 `fix_type`（哪些修复类型已被尝试）
- `outcome`（passed / improved / stagnant / regressed）— 快速判断方向是否有效
- `improvement_ratio`（数值趋势一览）
- `direction_verdict`（是否曾切换方向）
- `forensics_hint`（每轮取证信号）
- `final_status`（in_progress / success / failed）

> ⚠️ **禁止重复已证实无效的方向**：outcome 为 regressed 或连续 stagnant 的 fix_type，本轮不得再用。

**第二步：按需深入（仅在确实需要时通过 round_summary 的 index 定位）**
```bash
# 读某轮的 round_summary 获取文件路径索引
cat "{output_path}/precision_tuning/round_summary_{N}.json"
# 再按 index.sections.* 路径读对应的 section 小文件
```

- 想了解某轮根因细节 → 读 `round_summary_N.index.sections.root_cause` 指向的文件
- 想了解某轮修复计划 → 读 `round_summary_N.index.sections.fix_plan` 指向的文件
- 想查看完整审计 → 读 `round_summary_N.index.audit_full` 指向的文件

**禁止**：不得跳过 `tuning_directions.json` 直接全量读 `history/attempt_*/precision_audit.md`。

---

#### Sub-step 2.1: 取证数据解读

**读取**: `{output_path}/precision_tuning/forensics_report_{attempt}.json`

**产出**: `[FORENSICS_SUMMARY]` section

逐字段摘录取证报告中的关键数值, 不允许跳过任何字段:

```
=== PRECISION AUDIT REPORT ===

[FORENSICS_SUMMARY]
  取证数据摘要 (L0-L4):
    - primary_hint: <来自取证 primary_hint>
    - primary_confidence: <来自取证 primary_confidence>
    - primary_evidence: <来自取证 primary_evidence>
    - mismatch_ratio: <来自取证 outputs[0].basic_stats.mismatch_ratio>
    - max_abs_diff: <来自取证 outputs[0].basic_stats.max_abs_diff>
    - mean_abs_diff: <来自取证 outputs[0].basic_stats.mean_abs_diff>
    - error_distribution: <来自取证 outputs[0].error_distribution, 特别关注 sign_analysis.bias_direction>
    - worst 元素位置: <来自取证 outputs[0].worst_elements, 列出 top 3>
    - 尾块分析: <来自取证 outputs[0].tail_analysis, 标注各 tile_size 下的 tail/body mismatch rate>
    - 维度分析: <来自取证 outputs[0].dimension_analysis, 标注各维度的 mismatch_rate 范围>
  L6 内存布局:
    - 输入 tensor layout: <来自取证 L6_memory_layout.inputs, 标注 shape/stride/对齐>
    - 最后一维对齐情况: <是否对齐 8/16/32/64/128/256>
  L8 算子类型:
    - op_type: <来自取证 L8_operator.op_type>
    - source: <来自取证 L8_operator.source>
    - attributes: <来自取证 L8_operator.attributes, 特别关注 dim/reduction/kernel_size 等>
    - reduction_axis: <来自取证 L8_operator.reduction_axis, 如果有>
    - 该类型的 checklist: <将在下方 search 命令中自动返回>
  可用文件:
    - reference: <来自取证 available_files.reference>
    - dsl: <来自取证 available_files.dsl>
    - op_desc: <来自取证 available_files.op_desc>
  dtype 精度级别判断:
    - dtype: <来自取证 outputs[0] 或 L8_operator, 如 float32/float16/bfloat16>
    - max_abs_diff (来自取证): <值>
    - 精度阈值参考 (来自 ascend-torch-comparison/precision_config.py AbsoluteThreshConfig):
      * float32 rtol=1e-4: max_diff > 1e-4 → 逻辑错误; ≤ 1e-4 → 精度达标
      * float16 rtol=1e-3: max_diff > 1e-2 → 逻辑错误; 1e-3~1e-2 → float16 精度损失(可能可接受); ≤ 1e-3 → 精度达标
      * bfloat16 rtol=5e-3: max_diff > 5e-2 → 逻辑错误; 5e-3~5e-2 → bfloat16 精度损失; ≤ 5e-3 → 精度达标
    - 判断: <逻辑错误(实现缺陷, 必须修复) / float16精度损失(检查 float32 下是否通过) / 精度达标>
    - 对分析方向的影响: <逻辑错误→重点查实现缺陷; float16精度损失→检查归约是否需要 upcast>
  我对取证 hint 的初步判断:
    - 取证给出的 hint 是否合理? <结合数值证据判断, 不要在此步做代码分析>
    - 是否有数值异常未被 hint 覆盖? <如 sign_analysis 显示偏向但 hint 未提及>
```

**知识库检索 (第一次 — 基于取证 hint + 算子类型):**

从 `[FORENSICS_SUMMARY]` 中提取 `primary_hint` 和 `op_type`, 检索相关知识条目:
```bash
python3 .opencode/skills/precision-tuning/scripts/precision_knowledge.py search \
    --kb-path .opencode/skills/precision-tuning/references/precision_knowledge_base.json \
    --op-type <L8_operator.op_type> \
    --pattern <primary_hint> \
    --top-k 3 \
    --log-path "{output_path}/precision_tuning" \
    --attempt {attempt} \
    --call-index 0
```

记住检索到的 `matched_entries` 和 `checklists`, 后续分析时参考。
如果输出 `fallback_to_full_load: true`, 说明无精确匹配, 已返回全量条目。

---

#### Sub-step 2.2: 算子计算流程分解

**读取** (按顺序):
1. `{output_path}/{op_name}_reference.py` — 参考实现的 forward() 逻辑
2. `{output_path}/{op_name}_dsl.py` — DSL tiling 策略和 kernel 计算步骤 (如果 available_files.dsl=true)
3. `{output_path}/{op_name}_op_desc.json` — 算子属性 (如果 available_files.op_desc=true)
4. `.opencode/skills/precision-tuning/references/decomposition_examples/README.md` — 分解示例索引
5. 查找与当前算子最匹配的分解示例文件 (如 `softmax.md`, `layer_norm.md` 等)

**产出**: `[COMPUTATION_DECOMPOSITION]` section

**要求**:
- 参考 `decomposition_examples/` 中最匹配的示例, 按其格式和粒度分解
- 如果有完全匹配的示例, 引用其步骤结构并根据当前算子的具体参数填充
- 如果没有匹配的示例, 按 README.md 的粒度标准自行分解
- 每步必须包含: 操作名、输入来源、输出 shape、数值范围预期、精度风险点
- 如果 DSL 存在, 每步标注 DSL 对应代码
- 标注算子计算模式: 单行归约 / 跨核归约 / 分块累加 / 滑窗累加 / 前缀累加 / 逐元素

```
[COMPUTATION_DECOMPOSITION]
  算子: {op_name}
  计算模式: <单行归约 / 跨核归约 / 分块累加 / 滑窗累加 / 前缀累加 / 逐元素>
  参考分解示例: <使用的示例文件名, 如 softmax.md, 或 "无匹配, 自行分解">
  归约维度: <dim={dim}, axis={axis}, 归约轴长度={length}> (如适用)
  数据类型: <dtype>

  计算链:
    Step 0: 输入
      - shape: <input_shape>
      - 数值范围: <来自取证 value_range>

    Step 1: <operation_name>
      - 来源: reference.py 中的 <具体函数/表达式>
      - DSL 对应: dsl.py 中的 <具体 tl.xxx 调用> (如 DSL 不存在则标注 "DSL 不可用")
      - 输入: <上一步输出 / 原始输入>
      - 输出 shape: <shape>
      - 数值范围预期: <基于输入范围推断>
      - 精度风险点: <该步可能引入误差的原因>
      - 知识库关联: <匹配的条目编号和标题, 或 "无">

    Step 2: <operation_name>
      ... (同上格式)

    Step N: 最终输出
      - 与取证报告的 golden output 统计对照

  DSL 文件状态: 存在 / 不存在
  DSL tiling 策略摘要: (如 DSL 存在)
    - n_cores: <值>
    - tiling 维度和参数: <如 tile_length, rows_per_core 等>
    - 归约维是否在单核内完整处理: 是 / 否
    - UB buffer 数量和大小: <列出>

  跨核通信: (仅跨核归约模式)
    - workspace buffer: <是否存在, 大小>
    - Phase 1 → Phase 2 的同步机制: <描述>
```

---

#### Sub-step 2.3: AscendC 实现逐步对照

**Phase A: 构建参考实现规范 (强制执行, 不可跳过)**

⚠️ **在读取任何 Kernel 代码之前, 必须先完成此 Phase, 建立正确实现的参考规范。** 参考规范是后续 Phase C 结构化对照的基准。

**Phase A 读取**:

1. 根据 `L8_operator.op_type` 定位对应的 lowering 示例 (必须读取完整文件):
   - `softmax` → `.opencode/skills/dsl-lowering/references/lowering_examples/softmax/softmax_process_example_for_lowering.md`
   - `mse_loss` / loss 类 → `.opencode/skills/dsl-lowering/references/lowering_examples/mse_loss/mse_loss_process_example_for_lowering.md`
   - `average_pooling2d` / pooling 类 → `.opencode/skills/dsl-lowering/references/lowering_examples/pooling/pooling_process_example_for_lowering.md`
   - `leaky_relu` / 逐元素类 → `.opencode/skills/dsl-lowering/references/lowering_examples/leaky_relu/leaky_relu_process_example_for_lowering.md`
   - 无精确匹配 → 读取最接近的示例 (优先选同类计算模式)
2. **必须读取**: `.opencode/skills/dsl-lowering/references/error_correction/error_correction_examples.md`
3. **必须读取**: `.opencode/skills/dsl-lowering/references/lowering_examples/non_aligned/non_aligned_example_for_lowering.md`
4. **必须读取**: `.opencode/skills/dsl-lowering/references/ascend_api/tl_asc_routing.md` — AscendC API 权威参考

**Phase A 产出**: `[REFERENCE_IMPL_SPEC]` section (写入 precision_audit_{attempt}.md, Gate-A 必填)

从上述参考文件中提取并填写如下规范:

```
[REFERENCE_IMPL_SPEC]
  参考实现来源: <lowering example 文件路径>

  TQue/TBuf 分配规范 (来自参考实现):
    - inQueue (TQue<VECIN>): <用途, DataCopy GM→UB 的目标 buffer>
    - outQueue (TQue<VECOUT>): <用途, DataCopy UB→GM 的源 buffer, 必须经 EnQue/DeQue>
    - TBuf (VECCALC): <用途, 中间计算 buffer, 不可直接作为 DataCopy src/dst>
    - TBuf→GM 正确路径: TBuf计算结果 → outQueue.AllocTensor → 写入outLocal → EnQue → DeQue → DataCopy

  关键 API 规范 (来自参考实现):
    - ReduceMax: <调用签名; work_buf 是否需要 Duplicate(-3.402823466e+38f, count) 初始化>
    - ReduceSum: <count 对齐要求 (64的倍数); work_buf 是否需要 Duplicate(0.0f, count) 初始化>
    - SyncAll: <是否需要, 插入位置 (跨核写GM后/读GM前)>

  非对齐处理规范 (来自 non_aligned_example):
    - 触发条件: count × sizeof(dtype) 不是 32 的倍数
    - GM→UB: DataCopyPad(dst, src, {1, count*sizeof(T), 0, 0}, {false, 0, 0, 0})
    - UB→GM: DataCopyPad(dst, src, {1, count*sizeof(T), 0, 0})
    - 本算子 tileLength 的对齐状况: <tileLength × sizeof(dtype) = ? 字节, 是否32字节对齐>

  error_correction 禁用模式 (来自 error_correction_examples.md):
    - 禁止 float↔uint 强制类型转换 (改用 float n = (float)int_val)
    - 禁止标量上下文调用向量 Log API (改用 AscendC::Log(tmp,tmp,1); tmp.GetValue(0))
    - 禁止向量上下文调用标量 AscendC::Sqrt (改用 sqrt(val))
    - 本算子代码中是否出现上述模式: <逐一检查>
```

---

**Phase B: 读取当前实现**

**读取**:
1. `{output_path}/{OpName}Custom/op_kernel/{op_name}_custom.cpp` — Kernel 代码
2. `{output_path}/{OpName}Custom/op_host/{op_name}_custom.cpp` — Host 代码 (TilingFunc)
3. `{output_path}/{OpName}Custom/op_host/{op_name}_custom_tiling.h` — TilingData 结构体定义 (字段名、类型、大小)

**产出**: `[REFERENCE_IMPL_SPEC]` + `[KERNEL_STEP_TRACE]` sections

**Phase C 要求 (结构化对照)**:
- 将 Kernel 的 Compute() 函数拆成与 Sub-step 2.2 对应的步骤
- 每步标注: AscendC API 名称、count 参数值、buffer 来源、代码行号
- 逐步与 2.2 的计算链对齐, 用 ✅/⚠️/❌ 标注匹配状态
- **对照 `[REFERENCE_IMPL_SPEC]` 逐项检查以下 5 个维度**:
  1. TQue/TBuf 数据流是否与规范一致 (特别: TBuf 是否绕过 outQueue 直接 DataCopy)
  2. ReduceMax/ReduceSum work_buf 是否按规范初始化 (Duplicate 到 -INF 或 0)
  3. DataCopy 对齐是否满足规范 (count × sizeof(dtype) 不是 32 倍数时是否换用 DataCopyPad)
  4. SyncAll 同步点是否与规范一致 (跨核场景是否遗漏)
  5. error_correction_examples 中列出的禁用模式是否在代码中出现
- 遇到不确定的 API 名称时，查阅 `tl_asc_routing.md` 确认（如 Max vs Vmax、Subs 是否存在、负无穷常量写法等）

```
[KERNEL_STEP_TRACE]
  Kernel 计算步骤 (从 Compute() 函数提取):
    K-Step 1: <AscendC API 名称>
      - 代码位置: op_kernel.cpp 第 <line> 行
      - 参数: count=<value>, src=<buffer>, dst=<buffer>
      - 对应计算链: Step <N> (<operation_name>)
      - 匹配状态: ✅ 匹配 / ⚠️ 参数偏差: <描述> / ❌ 缺失或多余

    K-Step 2: ...
    ...

  Host tiling 参数:
    - TilingData 结构体字段: <从 _tiling.h 中列出所有字段名和类型>
    - tileLength = <值> (来源: op_host.cpp TilingFunc 第 <line> 行)
    - 其他 TilingData 字段: <列出 field=value>
    - DSL 中对应参数: <tile_length=值, rows_per_core=值 等>
    - Host vs DSL 是否一致: 是 / 否 (差异: ...)
    - 归约维度完整性: tileLength <>=<> 归约轴长度 <length> → 完整 / 被切分

  跨核通信验证: (仅跨核归约模式)
    - workspace buffer: GM 中是否分配, 大小是否 = n_cores
    - 各核写入: DataCopy 后是否有同步
    - Core 0 读取: 是否在所有核完成后才读取
    - 全局归约: ReduceSum 的 count 是否正确 (= n_cores, 而非 tile_size)
    - 最终除法: 分母是否 = total_elems

  算子类型专项检查 (根据 L8 op_type 选择对应项):

    [Pooling 类] DataCopy 维度一致性:
      - 输入内存布局: <NCDHW / NCHW / NHWC, 来自 L6>
      - tileC 含义: <沿 C 维度的 tile 大小>
      - DataCopy count=tileC 读取的是: <C 维度 tileC 个通道 还是 W 维度 tileC 个元素?>
      - C 维度在内存中的 stride: <C_stride = D*H*W (NCDHW) / H*W (NCHW)>
      - ⚠️ 检查: tileC 个连续地址是否真的对应 tileC 个通道? 若 C_stride > 1, 连续地址实为沿 W/空间维读取
      - input base offset 公式: <写出 b/c0/d/h/w 各维度的 offset 计算, 标出 c0 的系数是否为 C_stride>
      - output base offset 公式: <写出 b/c0/od/oh/ow 各维度的 offset 计算, 标出 c0 的系数>
      - ⚠️ 检查: outBase 中 c0 的系数是否为 outD*outH*outW (正确) 而非 1 (错误)

    [Reduction / Normalization 类] 工作 Buffer 初始化:
      - ReduceMax work buffer: <调用前是否 Duplicate(work, -INF, count) 初始化?>
      - ReduceSum work buffer: <调用前是否 Duplicate(work, 0, count) 初始化?>
      - ⚠️ 检查: work buffer 是否从上一步骤残留了非零数据 (如 ReduceMax work buffer 含有上一步 maxVal 残留)
      - 负无穷写法: <代码中使用 -3.402823466e+38f / (float)(-INFINITY) / -65504.0f (float16 错误!)>

    [MatMul / 分块累加类] 累加器初始化:
      - 累加器 (acc buffer) 初始化位置: <在外层循环前 Duplicate(0) / 未初始化>
      - ⚠️ 检查: 多个 tile 间累加器是否在每个输出位置开始时被正确重置

    [TQue / TBuf 数据流] 同步验证 (所有算子类型必填):
      - inQueue 流程: AllocTensor → DataCopy(GM→UB) → EnQue → DeQue → (计算) → FreeTensor ✅/❌
      - outQueue 流程: AllocTensor → (计算写入) → EnQue → DeQue → DataCopy(UB→GM) → FreeTensor ✅/❌
      - TBuf 用途: <VECCALC 中间计算, 不参与 DMA 传输>
      - ⚠️ 严重: TBuf.Get() 直接作为 DataCopy dst 写 GM = 绕过 outQueue 同步 = 数据未写出 = 输出全零
      - ⚠️ 检查: CopyOut 函数中 maxLocal/accLocal 等 TBuf 变量是否直接用于 DataCopy(outputGm[], ...)

  步骤对齐结论:
    - 全部匹配: 是 / 否
    - 缺失步骤: <列出, 或 "无">
    - 参数偏差: <列出, 或 "无">
    - 新增/多余步骤: <列出, 或 "无">

  L7 代码位置映射 (手动):
    - worst element index=<index> → 对应 kernel 中的 <函数/代码块>
    - 该元素位于 main block / tail block?
    - 对应的 K-Step: <编号>
```

---

#### Sub-step 2.4: 知识库匹配 + 根因判断 + 修复计划

**读取**: Sub-step 2.1 检索到的知识库条目 + Sub-step 2.1~2.3 的全部分析结果

**知识库检索 (第二次 — 精化, 增加位置特征):**

> ⚠️ **在开始匹配前，用完整取证数据做第二次精化检索**（避免长上下文遗忘, 并利用 2.1~2.3 分析中发现的位置特征）。

从取证报告中提取 `--position` 参数:
- 若 worst_elements 集中在尾部区域 或 tail_analysis 显示尾块 mismatch 率显著偏高 → `--position tail`
- 若 worst_elements 集中在边界/起始区域 → `--position boundary`
- 若 worst_elements 分散 → `--position scattered`
- 若无明显位置特征 → 不传 `--position`

```bash
python3 .opencode/skills/precision-tuning/scripts/precision_knowledge.py search \
    --kb-path .opencode/skills/precision-tuning/references/precision_knowledge_base.json \
    --op-type <L8_operator.op_type> \
    --pattern <primary_hint> \
    --position <tail/boundary/scattered 或不传> \
    --top-k 3 \
    --log-path "{output_path}/precision_tuning" \
    --attempt {attempt} \
    --call-index 1
```

记住检索到的条目, 用于下方的 `[KNOWLEDGE_MATCH]`。

**产出**: `[KNOWLEDGE_MATCH]` + `[ROOT_CAUSE]` + `[FIX_PLAN]` + `[TARGET_FILES]` + `[DIRECTION_ASSESSMENT]` sections

**要求**: 根因判断必须基于 2.1~2.3 的具体发现, 不允许"凭直觉"给出根因。证据链中必须引用具体的 K-Step 编号和取证数据字段。

> ⚠️ **写 [FIX_PLAN] 前必须查阅 `tl_asc_routing.md`，核实所有将要使用的 AscendC API 名称**：
> - 逐元素向量最大值：`Max`（不是 `Vmax`，该 API 不存在）
> - 逐元素减法：`Sub`（无 `Subs`），逐元素除法：`Div`（无 `Divs`）
> - float32 负无穷：`-3.402823466e+38f` 或 `(float)(-INFINITY)`（不是 `AscendC::INFINITY`，该常量不存在）
> - DataCopy 写 GM 必须从 VECOUT TQue DeQue 后的 tensor，不能直接用 TBuf.Get() 的结果

```
[KNOWLEDGE_MATCH]
  知识库匹配:
    - 匹配的知识条目: <title> / 无匹配
    - 匹配度: 完全匹配 / 部分匹配 / 不匹配
    - 如何借鉴: <参考知识条目的 fix 内容>
  算子类型 checklist 检查:
    - <checklist 项 1>: 通过 / 未通过 / 不适用 (证据: <引用 K-Step 或取证数据>)
    - <checklist 项 2>: ...

[ROOT_CAUSE]
  根因判断: <综合 2.1 取证数据 + 2.3 步骤对齐结论 + 知识库匹配>
  置信度: HIGH / MEDIUM / LOW
  证据链:
    1. 数值证据: <取证 L1-L4 中哪些现象支持此判断, 引用具体字段值>
    2. 布局证据: <L6 内存布局是否有异常>
    3. 代码证据: <引用 K-Step 编号, 哪行代码有什么问题>
    4. 分解对照: <2.2 的哪个 Step 与 2.3 的哪个 K-Step 不一致>
    5. 逻辑推导: <为什么此代码问题会产生取证中观察到的 diff 模式>

[FIX_PLAN]
  修复方向: <具体描述, 引用变量名和行号>
  修复类型: <对应知识库 type, 如 FIX_PRECISION_TAIL>
  修改文件: <file1, file2>
  修改点:
    1. 文件: <文件名>, 位置: <行号或函数名>, 操作: <修改/新增/删除>
       当前代码: <现在是什么>
       修改为: <改成什么>
       对应 K-Step: <编号>
    2. ...
  预期效果: <修复后应该改善什么, 如 "尾块 mismatch 应消除">

[TARGET_FILES]
  <需要修改的文件列表, 逗号分隔>

[DIRECTION_ASSESSMENT]
  上一轮 (attempt={attempt-1}) 修复方向: <从 history/attempt_{{attempt-1}}/precision_audit.md 的 [FIX_PLAN] 中提取, 一句话描述>
  上一轮修复后 mismatch 变化: <从 forensics_report_{attempt}.json 的 history_trend 中读取, 如 "0.25→0.12 (改善)" 或 "0.25→0.28 (恶化)">
  本轮是否延续上一轮方向: <严格填写 "是" 或 "否"，不得填写其他任何文字>
  延续理由 / 换方向理由: <一句话>

=== END AUDIT ===
```

**重试轮次注意事项 (attempt > 0):**
- 取证报告中的 `history_trend` 显示了历史变化
- 你**必须**先读 `tuning_directions.json` 获取跨轮方向全貌，再按需通过 `round_summary_N.json` 的 index 路径深入具体 section 小文件
- **禁止重复 outcome 为 regressed 或连续 stagnant 的 fix_type**
- 如某轮 `index.sections.root_cause` 为 null，fallback 读 `index.audit_full`

Gate 验证:
```bash
python3 .opencode/skills/precision-tuning/scripts/precision_gate.py \
    --step audit --op-name {op_name} --output-path "{output_path}" --attempt {attempt}
```

⛔ **Gate-A 未通过 → 补全缺失的 section, 不计入轮次。Gate-A 现在检查 7 个必填 section: FORENSICS_SUMMARY, COMPUTATION_DECOMPOSITION, REFERENCE_IMPL_SPEC, KERNEL_STEP_TRACE, ROOT_CAUSE, FIX_PLAN, TARGET_FILES。**

> Gate-A 通过后，脚本自动提取 sections 小文件并写入 `round_summary_{attempt}.json` 初始字段（diagnostics + index）。**Agent 无需手动写 round_summary。**

---

### Step 3: 代码修复 (Agent 执行)

根据审计报告 [FIX_PLAN] 中的修改点, 逐一修复代码。

**修复原则:**
1. **严格遵循 FIX_PLAN**: 不要自行扩大修改范围
2. **完整文件**: 写入修改后的完整文件, 不要截断
3. **真实变量名**: 使用代码中实际存在的变量名
4. **禁止逃避**: 不得缩小 shape、添加 if 跳过、放大 tolerance、删除功能

修复完成后, Gate 验证代码文件完整性:
```bash
python3 .opencode/skills/precision-tuning/scripts/precision_gate.py \
    --step fix --op-name {op_name} --output-path "{output_path}" --attempt {attempt}
```

⛔ **Gate-X 未通过 → 检查文件是否正确保存。**

---

### Step 4: 重新编译 + 精度验证

**4.1 安装算子运行环境:**
```bash
bash "{output_path}/{OpName}Custom/build_out/custom_opp_ubuntu_aarch64.run" \
    --install-path="{output_path}"
```

**4.2 重新编译:**
```bash
python3 .opencode/skills/ascendc-evaluation/scripts/generate_pybind.py \
    {op_name} --output-path "{output_path}"
```

如果编译失败:
- 分析编译错误, 修复代码 (最多 3 次编译重试, 对齐 dsl-lowering 模式)
- **每次编译失败后, 更新 `{output_path}/precision_tuning/compilation_log_{attempt}.json`** (读取已有文件追加 entry，不存在则初始化，整体覆盖写入):
  ```json
  {
    "attempt": <N>,
    "entries": [
      {
        "compile_retry": <0/1/2>,
        "error_category": "<undefined_api|type_mismatch|count_alignment|other_compile>",
        "error_snippet": "<编译器报错的核心行, 最多3行>",
        "fix_applied": "<本次修复的简要描述>"
      }
    ]
  }
  ```
  `error_category` 选择规则:
  - `undefined_api`: 含 `error: 'Vmax' was not declared` / `undeclared identifier` 等 API 不存在错误
  - `type_mismatch`: 含 `cannot convert` / `no matching function` 等类型不匹配
  - `count_alignment`: 含 `count must be multiple of` / `alignment` 等对齐要求
  - `other_compile`: 其他编译错误
- 如果 3 次仍编译失败 → 从 `code_snapshot` 恢复, 回到 Step 2 重新审计

**4.3 精度验证:**
```bash
python3 .opencode/skills/ascendc-evaluation/scripts/evaluate.py \
    {op_name} --output-path "{output_path}"
```

**4.4 保存验证结果:**

解析 evaluate.py 的 stdout, 提取 correctness 结果。
evaluate.py 通过时输出形如 `[PASS]\nOutput 0: shape=..., match_rate=100.00% (N/N), max_diff=..., mean_diff=...`。
evaluate.py 失败时输出形如 `[FAIL] Output 0 mismatch\nMatch rate: 87.50% (3500/4000)\nExample mismatch at index ...`。

将结果写入 `{output_path}/precision_tuning/validation_result_attempt_{attempt}.json`:
```json
{
  "attempt": <N>,
  "correctness_passed": true/false,
  "evaluate_stdout": "<evaluate.py 完整输出>",
  "match_rate": "<从 stdout 用正则提取, 形如 87.50 或 100.00 的百分比数值>",
  "max_diff": "<从 stdout 用正则提取, 形如 1.23e-04 的科学记数法数值>"
}
```

提取规则:
- `match_rate`: 用正则 `r"match_rate[=:]?\s*(\d+\.?\d*)%"` 提取百分比数值（不带百分号）；如果是 PASS 输出则写 `100.0`
- `max_diff`: 用正则 `r"max_diff[=:]?\s*([0-9.eE+-]+)"` 提取数值；如果是 PASS 输出则写 `0.0`

**4.5 Gate 验证 + 循环控制:**
```bash
python3 .opencode/skills/precision-tuning/scripts/precision_gate.py \
    --step validate --op-name {op_name} --output-path "{output_path}" --attempt {attempt}
```

Gate-V 输出包含 **loop_signal**, 你**必须遵守**:

| loop_signal | 含义 | 你的操作 |
|-------------|------|---------|
| **PASS** | 精度通过 | → 跳到 Step 5 (成功收尾) |
| **CONTINUE** | 未通过但有改善 | → 归档本轮, 回到 Step 1 (attempt + 1) |
| **STOP** | 未通过且无改善/达上限 | → 跳到 Step 6 (失败报告) |

⚠️ **你不能自行决定继续或停止。loop_signal 由 Gate 脚本根据数值趋势决定, Agent 必须遵守。**

---

### 归档当前轮次 (CONTINUE 时执行)

**每次归档时，比较当前轮 match_rate 与历史最佳，决定是否更新最佳代码。**

```bash
# 1. 保存本轮取证报告和审计报告（数字命名文件已是持久化的，cp 到 history 供统一目录访问）
mkdir -p "{output_path}/precision_tuning/history/attempt_{attempt}"
cp "{output_path}/precision_tuning/forensics_report_{attempt}.json" \
   "{output_path}/precision_tuning/history/attempt_{attempt}/forensics_report.json"
cp "{output_path}/precision_tuning/precision_audit_{attempt}.md" \
   "{output_path}/precision_tuning/history/attempt_{attempt}/precision_audit.md"

# 2. 更新最佳代码（比较当前轮 match_rate 与 history/current_best/match_rate.txt）
current_mr=$(python3 -c "import json; r=json.load(open('{output_path}/precision_tuning/validation_result_attempt_{attempt}.json')); print(r.get('match_rate', '0'))")
best_mr=0
if [ -f "{output_path}/precision_tuning/history/current_best/match_rate.txt" ]; then
    best_mr=$(cat "{output_path}/precision_tuning/history/current_best/match_rate.txt")
fi

is_better=$(python3 -c "print('yes' if float('$current_mr') >= float('$best_mr') else 'no')")
if [ "$is_better" = "yes" ]; then
    mkdir -p "{output_path}/precision_tuning/history/current_best/code_snapshot"
    cp "{output_path}/{OpName}Custom/op_kernel/{op_name}_custom.cpp" \
       "{output_path}/precision_tuning/history/current_best/code_snapshot/op_kernel.cpp"
    cp "{output_path}/{OpName}Custom/op_host/{op_name}_custom.cpp" \
       "{output_path}/precision_tuning/history/current_best/code_snapshot/op_host.cpp"
    cp "{output_path}/{OpName}Custom/op_host/{op_name}_custom_tiling.h" \
       "{output_path}/precision_tuning/history/current_best/code_snapshot/op_tiling.h"
    cp "{output_path}/{op_name}.cpp" \
       "{output_path}/precision_tuning/history/current_best/code_snapshot/op_pybind.cpp"
    echo "$current_mr" > "{output_path}/precision_tuning/history/current_best/match_rate.txt"
    echo "精度改善: $best_mr → $current_mr，已更新最佳代码"
fi

# 3. 保存下一轮的起始快照
mkdir -p "{output_path}/precision_tuning/history/attempt_{next_attempt}/code_snapshot"
cp "{output_path}/{OpName}Custom/op_kernel/{op_name}_custom.cpp" \
   "{output_path}/precision_tuning/history/attempt_{next_attempt}/code_snapshot/op_kernel.cpp"
cp "{output_path}/{OpName}Custom/op_host/{op_name}_custom.cpp" \
   "{output_path}/precision_tuning/history/attempt_{next_attempt}/code_snapshot/op_host.cpp"
cp "{output_path}/{OpName}Custom/op_host/{op_name}_custom_tiling.h" \
   "{output_path}/precision_tuning/history/attempt_{next_attempt}/code_snapshot/op_tiling.h"
cp "{output_path}/{op_name}.cpp" \
   "{output_path}/precision_tuning/history/attempt_{next_attempt}/code_snapshot/op_pybind.cpp"
```

然后 `attempt += 1`, 回到 Step 1。

---

### Step 5: 成功收尾

精度通过后:

**5.0 归档当前轮次 + 更新 current_best（PASS 时必须执行）:**
```bash
# 归档本轮取证报告和审计报告（与 CONTINUE 时的归档逻辑一致）
mkdir -p "{output_path}/precision_tuning/history/attempt_{attempt}"
cp "{output_path}/precision_tuning/forensics_report_{attempt}.json" \
   "{output_path}/precision_tuning/history/attempt_{attempt}/forensics_report.json"
cp "{output_path}/precision_tuning/precision_audit_{attempt}.md" \
   "{output_path}/precision_tuning/history/attempt_{attempt}/precision_audit.md"

# 更新 current_best 为最终通过的代码（match_rate = 100.0）
mkdir -p "{output_path}/precision_tuning/history/current_best/code_snapshot"
cp "{output_path}/{OpName}Custom/op_kernel/{op_name}_custom.cpp" \
   "{output_path}/precision_tuning/history/current_best/code_snapshot/op_kernel.cpp"
cp "{output_path}/{OpName}Custom/op_host/{op_name}_custom.cpp" \
   "{output_path}/precision_tuning/history/current_best/code_snapshot/op_host.cpp"
cp "{output_path}/{OpName}Custom/op_host/{op_name}_custom_tiling.h" \
   "{output_path}/precision_tuning/history/current_best/code_snapshot/op_tiling.h"
cp "{output_path}/{op_name}.cpp" \
   "{output_path}/precision_tuning/history/current_best/code_snapshot/op_pybind.cpp"
echo "100.0" > "{output_path}/precision_tuning/history/current_best/match_rate.txt"
echo "精度通过，current_best 已更新为 100.0"
```

**5.1 生成候选知识库条目 (Agent 执行):**

基于 [ROOT_CAUSE] 和 [FIX_PLAN]，生成一条知识库候选条目，写入：
`{output_path}/precision_tuning/candidate_kb_entry.json`

格式要求:
```json
{
  "title": "<标准化中文标题，含英文关键词，如：LayerNorm 尾块 Padding 污染精度>",
  "feature": "<错误特征签名，泛化表达，不要写死具体 shape 或 tile size，如：tail_spike 模式，尾块 mismatch 率显著高于主体>",
  "reason": "<深层原因，50-200字，描述为什么会出现此问题>",
  "fix": "<通用修复指南，50-200字，描述应该如何修复，不要包含具体行号>",
  "type": "<FIX_PRECISION_XXX 枚举值，与 [FIX_PLAN] 中的修复类型一致>"
}
```

注意:
- `title` 必须含英文关键词（供 RAG 检索），格式为"中文描述 (English Keywords)"
- `feature` 要泛化，不要写 `last_dim=37` 或 `tile_size=128` 这种具体值
- `fix` 要通用，不要引用具体代码行号或变量名
- `type` 必须从以下枚举中选择：FIX_PRECISION_PADDING / FIX_PRECISION_TAIL / FIX_PRECISION_REDUCTION / FIX_PRECISION_TYPECAST / FIX_PRECISION_LAYOUT / FIX_PRECISION_SYNC / FIX_PRECISION_OVERFLOW / FIX_PRECISION_LOGIC / FIX_PRECISION_OTHER

**5.2 写入知识库 (Python 执行):**
```bash
python3 .opencode/skills/precision-tuning/scripts/precision_knowledge.py dump \
    --kb-path .opencode/skills/precision-tuning/references/precision_knowledge_base.json \
    --output-path "{output_path}" \
    --op-name {op_name}
```

**5.3 保存成功代码快照:**
```bash
# 将最终通过代码保存到 history/success/（永久保留，不覆盖）
mkdir -p "{output_path}/precision_tuning/history/success/code_snapshot"
cp "{output_path}/{OpName}Custom/op_kernel/{op_name}_custom.cpp" \
   "{output_path}/precision_tuning/history/success/code_snapshot/op_kernel.cpp"
cp "{output_path}/{OpName}Custom/op_host/{op_name}_custom.cpp" \
   "{output_path}/precision_tuning/history/success/code_snapshot/op_host.cpp"
cp "{output_path}/{OpName}Custom/op_host/{op_name}_custom_tiling.h" \
   "{output_path}/precision_tuning/history/success/code_snapshot/op_tiling.h"
cp "{output_path}/{op_name}.cpp" \
   "{output_path}/precision_tuning/history/success/code_snapshot/op_pybind.cpp"
echo "成功代码已保存到 history/success/code_snapshot/"
```

> **从最佳代码恢复（如需重新调优）：**
> ```bash
> cp "{output_path}/precision_tuning/history/current_best/code_snapshot/op_kernel.cpp" \
>    "{output_path}/{OpName}Custom/op_kernel/{op_name}_custom.cpp"
> cp "{output_path}/precision_tuning/history/current_best/code_snapshot/op_host.cpp" \
>    "{output_path}/{OpName}Custom/op_host/{op_name}_custom.cpp"
> cp "{output_path}/precision_tuning/history/current_best/code_snapshot/op_tiling.h" \
>    "{output_path}/{OpName}Custom/op_host/{op_name}_custom_tiling.h"
> cp "{output_path}/precision_tuning/history/current_best/code_snapshot/op_pybind.cpp" \
>    "{output_path}/{op_name}.cpp"
> ```

**5.4 输出成功报告:**
```
[PRECISION_TUNING_RESULT]
  status: SUCCESS
  attempts: <总轮次>
  final_match_rate: <最终 match rate>
  final_max_diff: <最终 max diff>
  root_cause_summary: <一句话总结根因>
  fix_summary: <一句话总结修复内容>
```

---

### Step 6: 失败报告

如果 Gate-V 返回 STOP:

输出失败报告, 包含所有轮次的历史:
```
[PRECISION_TUNING_RESULT]
  status: FAILED
  attempts: <总轮次>
  loop_stop_reason: <Gate 给出的停止原因>
  history:
    attempt 0: hint=<pattern>, mismatch=<ratio>, fix=<一句话>
    attempt 1: hint=<pattern>, mismatch=<ratio>, fix=<一句话>
    ...
  remaining_issue: <当前仍存在的问题描述>
  suggestion: <给人工分析的建议>
```

> **注意:** 失败时 `history/current_best/` 中保存了精度最好的代码。如需以此为基础重新调优，恢复方法：
> ```bash
> cp "{output_path}/precision_tuning/history/current_best/code_snapshot/op_kernel.cpp" \
>    "{output_path}/{OpName}Custom/op_kernel/{op_name}_custom.cpp"
> cp "{output_path}/precision_tuning/history/current_best/code_snapshot/op_host.cpp" \
>    "{output_path}/{OpName}Custom/op_host/{op_name}_custom.cpp"
> cp "{output_path}/precision_tuning/history/current_best/code_snapshot/op_tiling.h" \
>    "{output_path}/{OpName}Custom/op_host/{op_name}_custom_tiling.h"
> cp "{output_path}/precision_tuning/history/current_best/code_snapshot/op_pybind.cpp" \
>    "{output_path}/{op_name}.cpp"
> ```
> 如需恢复到最初基线：
> ```bash
> cp "{output_path}/precision_tuning/history/baseline/code_snapshot/op_kernel.cpp" \
>    "{output_path}/{OpName}Custom/op_kernel/{op_name}_custom.cpp"
> cp "{output_path}/precision_tuning/history/baseline/code_snapshot/op_host.cpp" \
>    "{output_path}/{OpName}Custom/op_host/{op_name}_custom.cpp"
> cp "{output_path}/precision_tuning/history/baseline/code_snapshot/op_tiling.h" \
>    "{output_path}/{OpName}Custom/op_host/{op_name}_custom_tiling.h"
> cp "{output_path}/precision_tuning/history/baseline/code_snapshot/op_pybind.cpp" \
>    "{output_path}/{op_name}.cpp"
> ```

---

## Note

- **每步 Gate 验证不可跳过** — Gate 是流程稳定性的保证
- **loop_signal 由 Gate 脚本决定, Agent 必须遵守** — 防止钻牛角尖
- **取证数据是分析的基础** — 不要在没有取证的情况下分析代码
- **知识库条目只在精度通过时写入** — 避免失败经验污染知识库
- **编译失败不计入精度调优轮次** — 编译问题就地修复 (最多 3 次)
- 修复后代码直接写回 AscendC 项目目录 (覆盖原文件)
- 参考 `references/precision_knowledge_base.json` 中的已知精度问题模式
