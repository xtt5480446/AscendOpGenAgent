---
name: trace-recorder
description: >
  执行 trace 记录员 Skill。在算子任务完成后，回顾整个执行过程，
  生成结构化的 trace 记录供 meta-agent 优化使用。
argument-hint: >
  输入：output_dir 目录路径、各阶段执行结果信息。
  输出：{output_dir}/trace.md 结构化执行记录。
---

# Trace 记录 Skill

你是一名执行 trace 记录员。你的目标是在算子任务完成后，回顾整个执行过程，生成结构化的 trace 记录。

## 关键限制
- 只允许在 `{output_dir}/` 目录下创建 `trace.md` 文件
- 不要修改 `{output_dir}/` 中的任何其他文件
- 只记录事实，不做改进建议（那是 meta-agent 的工作）

## 信息来源

回顾本次会话中的以下信息：
- 各阶段的执行结果（成功/失败）
- 评测脚本的输出（`@references/evaluate_tilelang.sh`、`@references/evaluate_ascendc.sh` 的返回）
- performance-analyzer 的性能测试结果
- Agent 的迭代过程（尝试了什么、失败了几次、最终如何解决）
- 遇到的错误信息
- 是否发生违规路径：外部 web 检索、直接整体复制参考实现、PyTorch / torch_npu 语义回退

## 输出格式

将以下内容写入 `{output_dir}/trace.md`：

```markdown
# Trace: {算子名称}

- 时间: {当前日期时间}
- 算子: {output_dir 对应的算子名}
- 最终结果: PASS / FAIL (tilelang) | PASS / FAIL (ascendc)

## 阶段零: Case 精简

- 结果: 通过 / 失败 / 跳过
- 原始 case 数: {n}
- 精简后 case 数: {n}
- 备注: {如有异常情况}

## 阶段一: TileLang

- 结果: 通过 / 失败
- evaluate_tilelang.sh 执行次数: {n}
- 关键错误信息: {评测脚本返回的错误，原文引用}
- Agent 行为记录:
  - 第 1 轮: {agent 做了什么，结果如何}
  - 第 2 轮: {修改了什么，结果如何}
  - ...
- 走偏点: {agent 做了哪些无效/错误/冗余的尝试，以及可能的原因}

## 阶段二: AscendC

- 结果: 通过 / 失败
- evaluate_ascendc.sh 执行次数: {n}
- 关键错误信息: {评测脚本返回的错误，原文引用}
- Agent 行为记录:
  - 第 1 轮: {agent 做了什么，结果如何}
  - 第 2 轮: {修改了什么，结果如何}
  - ...
- 走偏点: {agent 做了哪些无效/错误/冗余的尝试，以及可能的原因}

## 阶段三: 性能分析（如执行）

- 结果: {性能数据}
- performance-analyzer 执行详情:
  - 测试的实现: {reference/tilelang/ascendc}
  - 各实现平均耗时: {reference: Xms, tilelang: Yms, ascendc: Zms}
  - 加速比: {tilelang vs reference: X.x, ascendc vs reference: Y.y}

## 评测输出摘要

{粘贴最后一次 evaluate 脚本的关键输出片段，包括 PASS/FAIL 状态和错误详情}

## 违规路径记录

- 外部 web 检索: 是 / 否
- 直接整体复制参考实现: 是 / 否
- PyTorch / torch_npu 语义回退: 是 / 否
- 说明: {如发生，精确记录触发阶段与具体行为}
```

### 记录原则

1. **精确引用**: 错误信息、评测输出使用原文，不要改写或总结
2. **行为序列**: 每轮迭代记录 agent 的实际操作（改了什么文件、改了什么逻辑），而非笼统的"修复了 bug"
3. **走偏分析**: 重点记录 agent 做了哪些最终被证明无效的尝试，这是 meta-agent 优化 harness 的核心输入
4. **省略成功**: 如果某阶段一次通过且无异常，简要记录即可，不需要展开

---

## 新增步骤：产出 final_status JSON block

执行完上述 Trace 记录后，**追加**一段 fenced JSON 作为 `final_status` block，作为 Phase 7 收尾时机器可读的 verdict 快照。下游（Phase 8 spawn 决策、自动化报告）直接消费此 block，不再事后推断 `failure_type`。

### 硬约束

- 只允许在 `{output_dir}/trace.md` **末尾 append**；禁止改写已写入的任何内容。
- 禁止写独立的 `final_status.json` / `ac_history.json` 文件。Phase 4 迭代历史**内嵌**在 `final_status.ac_iterations` 数组里。
- 整个 JSON 必须用 fenced ` ```json ` 代码块包裹。

### 输入来源

1. **`{task_dir}/.eval_status/latest.json`**（eval_wrapper 产出，唯一事实源）
   - 读取方式：调用 `skills/ascendc/ascendc-debug/scripts/eval_status.py`
     ```bash
     python3 skills/ascendc/ascendc-debug/scripts/eval_status.py \
         --task-dir {task_dir} --summarize
     ```
   - 该脚本对外 API：`load_latest_status(task_dir)` 返回已校验的 dict；`summarize_for_trace(status)` 返回 `failure_type` / `import_subtype` / `abort_subtype` / `last_evaluate_phase` / `last_evaluate_attempt` / `failed_step` 等字段子集。

2. **Phase 4 迭代历史**：主 agent 在调用 trace-recorder 前写入 `{output_dir}/.phase4_history.json`（数组，每项形如 `{attempt, verifier_error, conductor_suggestion, eval_status_path, ended_at}`）。trace-recorder 读取后原样嵌入 `ac_iterations`。若文件不存在，用空数组 `[]` 兜底。

3. **目录状态**
   - `kernel/` 目录是否为空（glob `*.cpp` / `*.h` 均无 → 为空）
   - `model_new_ascendc.py` 是否 AST 退化 —— 可调：
     ```bash
     python3 skills/ascendc/ascendc-translator/scripts/validate_ascendc_impl.py \
         {output_dir}/model_new_ascendc.py
     ```
     （非零退出码或报告中出现退化 marker → `has_degradation = true`）

### 判定优先级（**严格按此顺序，原文摘录自 findings.md §5.3，顺序不可变**）

1. `kernel/` 目录为空 → `failure_type = "no_kernel"`，`debug_eligible = false`
2. `model_new_ascendc.py` AST 退化 → `failure_type = "degraded"`，`debug_eligible = false`
3. Phase 3 失败且未进到 Phase 4 → `failure_type = "tilelang_only_failed"`，`debug_eligible = false`
4. `eval_status.latest.json.failure_type == "execution_aborted"` → 照搬 `execution_aborted` + `debug_eligible = false`（环境/harness 问题，subagent 无法处理）
5. 否则 → 照搬 `eval_status.latest.json.failure_type`（`success` / `precision_failed` / `build_failed` / `import_failed` / `runtime_error` / `timeout`）

### `debug_eligible` 计算规则

- **先看**：`failure_type ∈ {precision_failed, build_failed, import_failed, runtime_error, timeout}` 且 `has_kernel == true` 且 `has_degradation == false`。三项同时成立才可能为 true；否则 `debug_eligible = false`。
- **再看**：若 `failure_type == "import_failed"`，额外要求 `import_subtype == "import_kernel_side"`；若是 `import_env_side` 或 `null`，则 `debug_eligible = false`（环境库/LD_LIBRARY_PATH 问题不在 kernel/ scope）。
- 其他分支（`success` / `degraded` / `no_kernel` / `tilelang_only_failed` / `execution_aborted`）一律 `debug_eligible = false`。
- `debug_eligible_reason` 填写得出该结论的关键判据（例如 `"failure_type=import_failed but import_subtype=import_env_side; env issue out of scope"`）。

### final_status JSON schema（schema_version = 2，findings.md §2.4）

```json
{
  "schema_version": 2,
  "failure_type": "success | precision_failed | build_failed | import_failed | runtime_error | timeout | execution_aborted | degraded | no_kernel | tilelang_only_failed",
  "import_subtype": "import_kernel_side | import_env_side | null",
  "abort_subtype": "killed_by_outer_harness | ssh_disconnected | docker_unreachable | unknown | null",
  "has_kernel": true,
  "has_compiled_kernel": true,
  "has_degradation": false,
  "last_evaluate_phase": 6,
  "last_evaluate_status_path": "{task_dir}/.eval_status/phase6_attempt0.json",
  "tl_iterations_used": 3,
  "ac_iterations": [
    {
      "attempt": 0,
      "verifier_error": "A-AscendCFallback-Type3: ...",
      "conductor_suggestion": "...",
      "eval_status_path": "{task_dir}/.eval_status/phase4_attempt0.json",
      "ended_at": "2026-04-22T10:15:58Z"
    }
  ],
  "debug_eligible": true,
  "debug_eligible_reason": "failure_type=precision_failed, has_kernel=true, has_degradation=false"
}
```

字段来源速查：

| 字段 | 来源 |
|------|------|
| `failure_type` | 判定优先级规则（见上） |
| `import_subtype` / `abort_subtype` | `eval_status.latest.json` 同名字段；若 `failure_type` 非对应类别则 `null` |
| `has_kernel` | `kernel/` 目录是否含 `.cpp`/`.h` |
| `has_compiled_kernel` | `kernel/` 下是否存在 `.so` / 编译产物 |
| `has_degradation` | `validate_ascendc_impl.py` 退出码 |
| `last_evaluate_phase` / `last_evaluate_status_path` | `eval_status.latest.json.phase` / 对应 `phase{N}_attempt{M}.json` 路径 |
| `tl_iterations_used` | Phase 3 实际迭代次数（从会话信息汇总） |
| `ac_iterations` | 直接内嵌 `{output_dir}/.phase4_history.json` 数组内容（不存在则 `[]`） |
| `debug_eligible` / `debug_eligible_reason` | 按上述规则计算 |

### 输出位置与格式

将整个 JSON 用 fenced ` ```json ` 代码块包裹，**追加**到 `{output_dir}/trace.md` 末尾：

````markdown
## final_status

```json
{
  "schema_version": 2,
  "failure_type": "...",
  ...
  "debug_eligible": true,
  "debug_eligible_reason": "..."
}
```
````

- **禁止**写独立文件（如 `final_status.json`、`ac_history.json`）。
- **禁止**对 `trace.md` 之前的任何内容做修改；只做 append。
- `trace.md` 在 Phase 7 写完后即**全程只读**（Phase 8 的 debug subagent 产物落到 `debug_trace.md` / `debug_status.json`，与本 block 互不干扰）。
