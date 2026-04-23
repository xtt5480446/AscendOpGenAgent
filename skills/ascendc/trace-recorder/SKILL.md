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

> 定量字段（次数 / 路径 / 阶段结果）的字段名与取值来源以 `agents/ascend-kernel-developer-with-ascendc-debug.md` 的 `Trace 字段契约` 表为准。

## 输出格式

向 `{output_dir}/trace.md` 写入两段内容，顺序执行：

### 1. Markdown 正文（阶段记录 + 违规路径）

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

- phase3_final_result: {PASS / FAIL / SKIPPED}
- phase3_tl_iterations: {n}   # 来源见契约表；禁止 agent 回忆估数
- tl_last_error_summary: |
    {最后一次 TileLang 失败的 stdout/stderr tail，原文引用；PASS 时留空}
- Agent 行为记录:
  - 第 1 轮: {agent 做了什么，结果如何}
  - 第 2 轮: {修改了什么，结果如何}
  - ...
- 走偏点: {agent 做了哪些无效/错误/冗余的尝试，以及可能的原因}

## 阶段二: AscendC

- phase4_final_result: {PASS / FAIL / N/A}
- phase4_ac_iterations: {n}   # 以 .verify_status/phase4_attempt*.json 文件数为准
- phase4_ac_cap_violated: {true / false}   # 若 true，必须同时在 final_status.trace_drift_warnings 中声明
- phase6_full_verify_runs: {n}
- phase6_final_result: {PASS / FAIL / N/A}
- ac_last_error_summary: |
    {latest.json.stdout_tail + stderr_tail 原文引用；success 时留空}
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

{粘贴 latest.json / 最后一次 phase{3,4,6} 验证的关键 stdout/stderr 片段，包括 PASS/FAIL 状态和错误详情；原文引用，不做改写}

## 违规路径记录

- 外部 web 检索: 是 / 否
- 直接整体复制参考实现: 是 / 否
- PyTorch / torch_npu 语义回退: 是 / 否
- 说明: {如发生，精确记录触发阶段与具体行为}
```

#### 记录原则

1. **精确引用**: 错误信息、评测输出使用原文，不要改写或总结
2. **行为序列**: 每轮迭代记录 agent 的实际操作（改了什么文件、改了什么逻辑），而非笼统的"修复了 bug"
3. **走偏分析**: 重点记录 agent 做了哪些最终被证明无效的尝试，这是 meta-agent 优化 harness 的核心输入
4. **省略成功**: 如果某阶段一次通过且无异常，简要记录即可，不需要展开

### 2. 末尾追加 final_status JSON block

作为 Phase 7 verdict 快照，供 Phase 8 spawn 决策直接消费。**只追加、不改写既有内容**；不得写独立的 `final_status.json` / `ac_history.json`，Phase 4 历史内嵌在 `ac_iterations` 字段里。

#### 输入来源

1. `{task_dir}/.verify_status/latest.json`（`classify_verify_result` 产出）：
   ```bash
   python3 skills/ascendc/ascendc-debug/scripts/verify_status.py --task-dir {task_dir} --summarize
   ```
2. `{output_dir}/.phase4_history.json`（主 agent 在调用 trace-recorder 前写入的数组；缺失则用 `[]`），原样嵌入 `ac_iterations`。
3. 目录状态：`kernel/` 是否含 `*.cpp`/`*.h`；`model_new_ascendc.py` 是否 AST 退化（`python3 skills/ascendc/ascendc-translator/scripts/validate_ascendc_impl.py {output_dir}/model_new_ascendc.py` 非零退出即 `has_degradation=true`）。

#### `failure_type` 判定优先级（顺序不可变，原文 findings.md §5.3）

1. `kernel/` 为空 → `no_kernel`
2. `model_new_ascendc.py` AST 退化 → `degraded`
3. Phase 3 失败且未进 Phase 4 → `tilelang_only_failed`
4. `verify_status.latest.json.failure_type == "execution_aborted"` → 照搬 `execution_aborted`（**专指**外部原因：SSH 断线 / Docker daemon 不可达 / outer harness kill。NPU `aicore exception` / `MTE illegal` / `ACL stream synchronize failed (5xxxxx)` 已在 `classify_verify_result.py` 归为 `runtime_error` + `exit_signal=NPU_AICORE_EXCEPTION`，不落本 bucket）
5. 否则照搬 `verify_status.latest.json.failure_type`（`success` / `precision_failed` / `build_failed` / `import_failed` / `runtime_error` / `timeout`）

前三条一律 `debug_eligible = false`。

#### `debug_eligible` 计算规则

同时满足下列三条才可能为 `true`，否则 `false`：

- `failure_type ∈ {precision_failed, build_failed, import_failed, runtime_error, timeout}`
- `has_kernel == true` 且 `has_degradation == false`
- 若 `failure_type == "import_failed"`，额外要求 `import_subtype == "import_kernel_side"`（`import_env_side` 或 `null` 属环境问题，不在 kernel/ scope）

`debug_eligible_reason` 填判据原文（例：`"failure_type=import_failed but import_subtype=import_env_side; env issue out of scope"`）。

#### Schema（schema_version = 2，findings.md §2.4）

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
  "last_verify_status_path": "{task_dir}/.verify_status/phase6_attempt0.json",
  "tl_iterations_used": 3,
  "ac_iterations": [
    {
      "attempt": 0,
      "verifier_error": "A-AscendCFallback-Type3: ...",
      "conductor_suggestion": "...",
      "verify_status_path": "{task_dir}/.verify_status/phase4_attempt0.json",
      "ended_at": "2026-04-22T10:15:58Z"
    }
  ],
  "debug_eligible": true,
  "debug_eligible_reason": "failure_type=precision_failed, has_kernel=true, has_degradation=false"
}
```

#### 字段来源速查

| 字段 | 来源 |
|------|------|
| `failure_type` | 判定优先级规则 |
| `import_subtype` / `abort_subtype` | `verify_status.latest.json` 同名字段；类别不匹配则 `null` |
| `has_kernel` / `has_compiled_kernel` | `kernel/` 下是否含 `.cpp`/`.h` / `.so` |
| `has_degradation` | `validate_ascendc_impl.py` 退出码 |
| `last_evaluate_phase` / `last_verify_status_path` | `verify_status.latest.json.phase` / 对应 `phase{N}_attempt{M}.json` |
| `tl_iterations_used` | Phase 3 实际迭代次数 |
| `ac_iterations` | 原样嵌入 `.phase4_history.json`（缺失则 `[]`） |
| `debug_eligible` / `debug_eligible_reason` | 按上述规则计算 |
