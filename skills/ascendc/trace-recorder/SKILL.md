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

补充约定：
- TileLang 当前主要用于设计表达，不默认作为 correctness / performance gate
- 若未执行 TileLang 验证，或因 TileLang 自身 bug 明确跳过验证，应如实记录为“跳过”并写明原因

## 输出格式

将以下内容写入 `{output_dir}/trace.md`：

```markdown
# Trace: {算子名称}

- 时间: {当前日期时间}
- 算子: {output_dir 对应的算子名}
- 最终结果: SKIP / PASS / FAIL (tilelang) | PASS / FAIL (ascendc)

## 阶段零: Case 精简

- 结果: 通过 / 失败 / 跳过
- 原始 case 数: {n}
- 精简后 case 数: {n}
- 备注: {如有异常情况}

## 阶段一: TileLang

- 结果: 通过 / 失败 / 跳过
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
  - 测试的实现: {reference/ascendc，若额外测试再包含 tilelang}
  - 各实现平均耗时: {reference: Xms, ascendc: Zms, tilelang: Yms(如有)}
  - 加速比: {ascendc vs reference: Y.y, tilelang vs reference: X.x(如有)}

## 评测输出摘要

{粘贴最后一次 evaluate 脚本的关键输出片段，包括 PASS/FAIL 状态和错误详情}
```

### 记录原则

1. **精确引用**: 错误信息、评测输出使用原文，不要改写或总结
2. **行为序列**: 每轮迭代记录 agent 的实际操作（改了什么文件、改了什么逻辑），而非笼统的"修复了 bug"
3. **走偏分析**: 重点记录 agent 做了哪些最终被证明无效的尝试，这是 meta-agent 优化 harness 的核心输入
4. **省略成功**: 如果某阶段一次通过且无异常，简要记录即可，不需要展开
5. **如实跳过**: TileLang 未验证不是异常；如果流程按约定跳过，应明确记录“跳过”及原因，不要误记为失败
