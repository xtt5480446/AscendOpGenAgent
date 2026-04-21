---
name: template-reverse-engineering
description: >
  官方 AscendC 模板逆向工程 Skill。先系统分析官方实现Ascendc，补对应测试benchmark，
  再把官方 AscendC 工程改造成直调并完成远端正确性验证，随后以直调版本为核心参照设计 TileLang，
  最后完成 TileLang/AscendC/official 一致性修复与审查。
argument-hint: >
  输入：official_impl_dir、output_dir，以及可选的 official_tiling_dir、op_name、补充资料路径。
  若未显式提供 reverse benchmark 的 id，则根据 benchmarks/NPUKernelBench/reverse/ 中现有文件名的最大 id 依次递增分配。
  输出：benchmarks/NPUKernelBench/reverse/{id}_{op_name}.py、{output_dir}/model.py、
  {output_dir}/kernel/、model_new_ascendc.py、{output_dir}/design/、model_new_tilelang.py，
  以及通过验证的一致性结论与对现有 skill 的改进建议。
---

# 官方模板逆向工程 Skill

你是一名官方 AscendC 模板逆向工程专家。你的目标是将Ascendc官方实现转换成高质量的、一致性的tilelang设计与Ascendc直调代码：

- 可远端验证的直调 AscendC项目
- 可远端验证的TileLang design
- 足够薄的 `model_new_ascendc.py` 和 `model_new_tilelang.py`

这个 skill 适用于如下场景：

- 用户给出官方 AscendC 实现，希望重构成tilelang as design、ascendc as implementation
- 设计目标不是单纯过 case，而是 TileLang 和 AscendC 保持较好的一致性关系

## 关键目标

1. `official -> direct-invoke ascendc -> tilelang` 三层必须能互相对照。
2. 直调 AscendC 和 TileLang 都必须仅通过 SSH 远端验证脚本完成正确性验证。
3. 不允许使用 fallback、降级路径或绕过实现问题的替代执行路径。
4. 遇到不稳定重构时，要先回到上一个正确版本，再做最小 repro 收敛根因。

## 关键限制

- 不允许把核心语义拆成多个互相打补丁的独立算子来规避问题。
- 不允许以 fallback、reference 兜底、条件分支降级、局部 case 绕过等方式“做通过”验证。
- `model_new_tilelang.py` 与 `model_new_ascendc.py` 都应尽量薄，只保留张量创建、张量变换和算子调用。
- 逆向主流程必须固定为：先补 reverse benchmark，再完成直调 AscendC 并验证，然后再做 TileLang，最后做一致性表达修复。
- TileLang 设计必须显式参考已经通过验证的直调 AscendC 版本，优先保持 staged data movement、分支划分、host/kernel 分工和边界语义一致。
- 设计判断逻辑尽量放在 `design/tile_level/`，host 调用层不承载厚重分支。
- AscendC `pybind11.cpp` 只负责 host 侧逻辑，不要把计算偷偷留在 ATen/C++ 侧。
- 对不确定的 AscendC API，例如 `DataCopyPad`，必须查文档，不允许凭印象实现。
- 不允许用本地验证、静态审查、手工 spot check 或临时脚本替代 SSH 远端统一验证脚本。
- 遇到环境问题，例如运行时缺依赖、远端机器不可用、NPU 不可分配、SSH 异常，必须直接报错并说明阻塞点，不允许试图绕过环境问题继续交付“近似完成”结果。
- 只要存在精度问题，就必须持续与官方实现逐项比对并迭代优化；不能以放宽阈值、缩 case、屏蔽输出或保留已知误差作为完成标准。
- 每个实现阶段的“修改 -> 远端验证”连续尝试次数上限为 3 次；若 3 次后仍未通过，必须停止继续盲改并明确汇报当前状态、失败日志和下一步判断。

## 推荐输入

- `official_impl_dir`
  官方 AscendC 实现目录
- `output_dir`
  当前任务目录，例如 `archive_tasks/<op_name>`
- `op_name`
  reverse benchmark 文件名与任务目录命名使用的算子名；若未显式提供，则需先从官方实现或任务上下文中确定后再继续

## 输出物

- `benchmarks/NPUKernelBench/reverse/{id}_{op_name}.py`
- `{output_dir}/model.py`
  - 必须与 reverse benchmark 保持一致
  - reverse benchmark 准备完成后，将同一份输入语义与模型准备逻辑复制到 `output_dir`
- `{output_dir}/design/block_level/`
- `{output_dir}/design/tile_level/`
- `{output_dir}/kernel/`
- `{output_dir}/model_new_ascendc.py`
- `{output_dir}/model_new_tilelang.py`
- 一份最终一致性审查结论：
  - 直调 AscendC 与 TileLang design 是否对齐
  - 与官方实现相比还有哪些差异、取舍或遗漏
- 一份对现有 skill 的改进 proposals：
  - `skills/ascendc/tilelang-designer`
  - `skills/ascendc/ascendc-translator`
  - 必要时包括缺失 API 描述、验证经验、实现技巧和踩坑总结

## 可用SKILLS

- `skills/ascendc/tilelang-designer/SKILL.md`
  用于 TileLang design、实现与远端验证
- `skills/ascendc/ascendc-registry-invoke-to-direct-invoke/SKILL.md`
  用于把官方 AscendC 工程整理成 `<<<>>>` 直调版本
- `skills/ascendc/ascendc-translator/SKILL.md`
  初始版本ascendc与tilelang验证后、开始做一致性校验且发现需要调整实现逻辑时
- `archive_tasks/rms_norm/`
  参考单入口 `@tilelang.jit`、`kernel/` 组织方式和 `pybind11.cpp`

## 总体流程

### 阶段 1：先补 reverse benchmark 用例

在开始 TileLang/AscendC 正式实现前，先补 benchmark 用例，作为统一输入描述和回归入口。

要求：

- 在 `benchmarks/NPUKernelBench/reverse/` 下新增 `{id}_{op_name}.py`
- `id` 由 agent 扫描 `benchmarks/NPUKernelBench/reverse/` 中已有 reverse benchmark 文件名的数值前缀后，按最大已有 id 递增分配
- 组织方式参考现有 bench task，例如 `benchmarks/NPUKernelBench/level1/1_GELU.py`,不同用例组织成`INPUT_CASES`
- 用例内容要覆盖从官方实现中识别出的关键场景：
  - 主要 rank 场景
  - 关键 pad / attr 组合
  - small / big 或其他调度分支
  - 关键 dtype
- reverse benchmark 准备完成后，必须将其中的模型准备逻辑同步复制为 `{output_dir}/model.py`，保证 `model.py` 与 reverse benchmark 使用同一份输入语义，不允许两边各自维护

这一步的目标是先把“问题定义”固定下来，避免后续 design、kernel、验证三处各自维护一套输入语义。

### 阶段 2：先把官方 AscendC 工程改成直调版本并验证通过

这一阶段必须调用 `ascendc-registry-invoke-to-direct-invoke` 的方法论推进，目标不是重写算子，而是把当前待逆向的官方 AscendC 工程转换成可独立维护、可 `<<<>>>` launch、可直接远端验证的直调版本。

推荐顺序：

1. 提取 kernel / tiling / host glue 的最小依赖闭包
2. 改造成当前任务目录下可维护的 `kernel/` 与 `model_new_ascendc.py`
3. 保持 kernel 语义不变，只做直调落地所需的最小必要改写
4. 使用统一远端脚本验证直调版本，直到正确性通过

设计原则：

- 只允许把官方工程中依赖注册框架的部分收口为本地可维护的直调结构，不允许顺手改写 kernel 主体语义
- `kernel/`、tiling struct、host dispatch、launch 签名必须互相匹配，并尽量贴近官方 host 逻辑
- 直调版本一旦通过远端验证，就把它视为后续 TileLang 设计的第一参考实现，而不是完成后丢开不用

### 阶段 3：直调 AscendC 远端验证

完成直调 AscendC 后，必须立即跑远端验证，而不是先跳去写 TileLang。

执行原则：

1. 只允许跑 SSH 远端统一验证脚本
2. 优先使用直调 skill 自带的 `evaluate_ascendc.sh`，或仓内统一的 AscendC 远端验证脚本
3. 如果失败，按“编译错误 -> 链接错误 -> 运行时错误 -> 数值错误”的顺序收敛
4. 单轮连续尝试最多 3 次；3 次后仍未通过时，停止继续修改并报告当前状态
5. 只有直调 AscendC 已通过验证，才能进入 TileLang 设计阶段

### 阶段 4：以直调 AscendC 为主参照完成 TileLang design

这一阶段调用 `tilelang-designer` 的方法论来做，但必须带着“已验证直调 AscendC”的结构约束推进。

推荐顺序：

1. 先做 `design/block_level/`
   - 只确定 block 级任务划分、流水骨架、workspace 与同步关系
   - block 切分必须优先对齐直调 AscendC / 官方 tiling 的任务划分
2. 再做 `design/tile_level/`
   - 用 TileLang 完整表达 直调Ascendc实现逻辑
   - 场景划分要和直调 AscendC 一致
3. 生成足够薄的 `model_new_tilelang.py`
   - 只保留张量创建、reshape、layout 调整和调用自定义算子

设计原则：

- 必须采用“单个 `@tilelang.jit` 统一出口 + 多个 `@T.prim_func` 场景实现”的组织方式
- 每个 `@T.prim_func` 都应是一个完整场景，不要做残缺补丁
- 不同分支模板实现用 `@T.prim_func` 修饰，并通过统一 `@tilelang.jit` 入口选择，不要定义多个并列的 `@tilelang.jit`
- 如果某个重构会破坏“完整场景”边界，不要继续硬推

### 阶段 5：TileLang 远端验证

完成 TileLang 后，必须跑远端验证，而不是停留在静态审查。

执行原则：

1. 只允许跑 SSH 远端统一验证脚本
2. 优先使用 `tilelang-designer` 自带的 `evaluate_tilelang.sh`，或仓内统一的 TileLang 远端验证脚本
3. 如果失败，先定位是语义错误、场景划分错误、调用层错误，还是后端运行时问题
4. 单轮连续尝试最多 3 次；3 次后仍未通过时，停止继续修改并报告当前状态
5. 如果是环境问题，例如 NPU 忙、SSH 重置、卡不可用，直接报错说明，不允许改走本地验证或其他替代路径

这里要特别注意：

- 不要把“远端执行失败”直接等同于“设计错误”
- 更换 NPU 卡号时，只改变运行环境，不改变语义判断
- 任何“环境不通但本地能跑”的结果都不能视为完成

### 阶段 6：遇到重构卡死时，先最小 repro，再回正式实现

这是我们这次任务里最关键的经验之一。

如果某次重构出现卡死、stream sync fail、AICore exception 或结果完全异常：

1. 先回到上一个已经正确的版本
2. 不要继续在坏版本上叠改动
3. 用最小 repro 验证怀疑点

推荐缩减顺序：

1. 单 dtype
2. 单 small case
3. 只保留 middle path
4. 再加入完整场景

尤其要检查：

- intermediate buffer 是否真实落盘
- staged copy 的前后依赖是否清晰
- front/back 是否只是 GM 上的本地搬运
- runtime 失败到底来自语义、workspace、同步，还是后端限制

### 阶段 7：反思 TileLang / AscendC 在一致性表达上的不足，并逐个修复验证

TileLang 和直调 AscendC 都通过后，不要立刻结束。必须反过来检查两者在“一致性表达”上是否还存在偏差，并逐个修复、逐次验证。

重点检查：

- kernel launch 入口是否显式导出
- `pybind11.cpp` 的 dtype、shape、launch 签名是否匹配
- tiling struct 字段是否完整一致
- workspace 大小是否正确
- `blockDim`、`perCoreTaskNum`、`tailTaskNum` 是否与官方逻辑一致
- TileLang 的 block/task 切分、阶段边界、中间 buffer 语义、边界处理是否与直调 AscendC 一致
- `model_new_tilelang.py` / `model_new_ascendc.py` 是否都保持薄调用层，没有把差异偷藏在 host 侧
- 如果出现数值误差，必须回到官方实现逐阶段对比中间结果、tiling 参数、dtype 路径和边界 case，持续修正直到误差消失
- 不允许为了“先过验证”添加 reference fallback、条件绕过或特殊 case 兜底

### 阶段 8：最终一致性审查与 skill 改进 proposals

正确性通过后不要立刻结束，还要做最后一轮比对：

1. `design` 和直调 `kernel` 是否一一对应
2. 直调 `kernel` 和官方实现相比还有哪些差异
3. 哪些差异是有意识取舍
4. 哪些地方虽然结果正确，但结构上还不够贴近官方
5. `pybind11.cpp` 是否还残留不该存在的 host 计算逻辑
6. 这次过程中暴露了哪些现有 skill 的缺口

额外要求：

- 最终一致性审查里必须明确声明“是否零 fallback”“是否仅经 SSH 远端脚本验证”
- 如果因为环境问题或精度问题未达到上述标准，必须明确标记为未完成，不能模糊表述为“基本完成”或“验证通过但有例外”

审查完成后，必须补一份改进 proposals，至少覆盖：

- `skills/ascendc/tilelang-designer`
  例如是否需要更强调单一 `@tilelang.jit` 出口、完整场景 `@T.prim_func`、对官方 staged 结构的贴近要求
- `skills/ascendc/ascendc-translator`
- `skills/ascendc/ascendc-registry-invoke-to-direct-invoke`
  例如是否缺少关键 API 的说明、host/kernel 分工边界、`pybind11.cpp` 约束、常见 launch/export 问题
- 必要时补充本次任务新学到的实现技巧
  例如某类 GM 本地搬运模式、某类 runtime 故障的最小 repro 策略、某些 API 使用注意事项

## 检查表

### TileLang 设计检查

- 是否显式参考了已通过验证的直调 AscendC 版本
- 是否保持单个 `@tilelang.jit` 统一出口
- 不同场景实现是否使用 `@T.prim_func`
- `model_new_tilelang.py` 是否足够薄
- `{output_dir}/model.py` 是否与 reverse benchmark 保持一致
- `design/tile_level/` 是否承载主要判断逻辑
- 每个 `@T.prim_func` 是否对应完整场景
- TileLang 实现结构是否足够贴近官方 AscendC，而不是为了好写而改写
- 是否已用远端脚本验证通过

### AscendC 实现检查

- reverse benchmark 的 `id` 是否按已有 reverse benchmark 最大 id 递增分配
- 是否已先完成直调改造，再进入 TileLang 阶段
- `kernel/` 组织是否清晰并贴近 design
- `pybind11.cpp` 是否只做 host 侧工作
- 不确定 API 是否查阅过直调 skill 参考与官方文档
- 是否已用远端脚本验证通过

### 一致性检查

- 直调 AscendC 与 TileLang design 是否能互相映射
- 与官方实现是否仍存在结构性偏离
- 偏离原因是否已明确记录
- 是否已沉淀对现有 skill 的改进 proposals

## 常见坑

- 还没看官方 tiling 就开始决定 small/big 或 blockDim
- reverse benchmark 还没补齐就先写 kernel / TileLang
- reverse benchmark 与 `{output_dir}/model.py` 各自维护，导致输入语义漂移
- 直调 AscendC 还没验过，就开始拍脑袋写 TileLang
- 用 gather 式写法把官方 staged 搬运结构抹平
- 定义多个并列的 `@tilelang.jit` 入口，而不是保持统一出口
- 在 `model_new_tilelang.py` 中堆积厚重判断逻辑
- 在坏版本上持续重构，没先回到最后一个正确版本
- 没先做最小 repro，就直接大改正式实现
- 误解 TileLang 中 `vid/cid` 与官方 block/core 语义的对应关系
- `pybind11.cpp` 承担了不该有的计算逻辑
- 远端验证失败时，没有区分代码问题和 NPU/SSH 资源问题

## 建议汇报格式

任务完成后，建议按下面顺序汇报：

1. 官方实现核心结构总结
2. reverse benchmark 用例定义
3. 直调 AscendC kernel 与 host 组织方式
4. TileLang design 的场景划分与关键取舍
5. AscendC / TileLang 远端验证结果
6. 最终一致性审查结论
7. 对已有 skill 的改进 proposals

## 成功标准

只有同时满足下面几项，才算真正完成：

- 官方实现已经被结构化理解，而不是只会复述代码
- reverse benchmark 用例已经补齐并能作为统一回归入口
- 直调 AscendC implement 已通过远端验证
- TileLang design 已通过远端验证
- `official -> direct-invoke ascendc -> tilelang` 三层关系已经复盘清楚
- 已明确列出当前版本与官方相比的差异、取舍与可能遗漏
- 已提出对相关 skill 的可执行改进建议
