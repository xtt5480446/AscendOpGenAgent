---
name: ascendc-generate
description: >
  AscendC kernel 直接生成专家 Skill。直接参考 model.py 的算子需求，生成 AscendC kernel 实现。
argument-hint: >
  输入：output_dir 目录路径（包含 model.py）。
  输出：kernel/ 下的 AscendC 实现、model_new_ascendc.py。
---

# AscendC Kernel 直接生成 Skill

你是一名 AscendC kernel 生成专家。你的目标是直接根据 `{output_dir}/model.py` 中的 PyTorch Model 需求，设计并实现 AscendC kernel，生成 `{output_dir}/model_new_ascendc.py` 调用 AscendC kernel，最终通过验证。

## 前置条件

本阶段开始前，以下文件必须已存在：
- `{output_dir}/model.py` — 参考 PyTorch 模型（包含算子逻辑和 INPUT_CASES）

## 关键限制

- 必须将核心计算融合成单个算子实现，不要拆分成多个独立算子。
- 禁止使用任何 web/search/fetch/open 等外部检索；信息来源仅限当前工作区内文件与本 skill 提供的参考资料。
- `model_new_ascendc.py` 只允许承担薄调用层职责：参数检查、必要的 dtype/shape 适配、以及调用你实现的自定义算子；禁止在其中承载算子主体语义，禁止使用 PyTorch 或 torch_npu 直接实现算子功能。
- `kernel/pybind11.cpp` 必须保持 **host 薄封装**：只允许做参数检查、shape/dtype 合法性校验、输出/workspace 分配、tiling 填充、kernel launch 分发。**禁止在其中实现任何算子语义或语义等价的 host 预处理**，包括但不限于：
  - 根据 mode/axis/rule 生成逐元素或逐位置的 offset/index lookup table
  - 生成决定输出值来源的 mask、row map、column map、gather/scatter 索引表
  - 在 host 侧展开 padding / slicing / broadcasting / reduction / permutation 等主体逻辑
  - 任何“虽然没直接调用 torch 计算，但通过预计算表把算子语义搬到 host”的做法
  - 允许的 host 数据仅限 launch 所需的 **非语义元数据**（如 shape、tile size、block 数、workspace 大小、分支枚举）；如果某个表本身已经编码了输入到输出的逐元素映射关系，则视为语义实现，必须移入 device/kernel 侧，或直接报告该分支暂未完成。
- 在 AscendC 实现中应尽可能避免标量逐元素写法，优先使用块级或向量化操作；只有在确实无法避免时才使用标量逻辑。
- **Vector 侧 GM↔UB 数据搬运必须使用 `DataCopyPad`，禁止使用 `DataCopy`**。`DataCopyPad` 支持字节级粒度的 `blockLen` 和 `srcStride`，是生产代码的标准做法。`DataCopy` 仅在 Cube 侧（L1/L0 相关通路）允许使用。
- 只允许修改或新增 `{output_dir}/` 目录中的文件，不要改动其他目录中的文件。
- 只允许读取当前工作区目录结构内的文件与子目录；禁止读取当前工作区之外的任何路径，包括父目录、兄弟目录、用户目录、绝对路径以及系统其他目录。

## 任务目录结构

```text
.
├── {output_dir}/         # 当前活跃任务目录
│   ├── design/           # 可选：存放算子设计文档（如 block-level 分析草图）
│   ├── kernel/           # AscendC kernel 实现
│   │   ├── xxx_tiling.h
│   │   ├── pybind11.cpp
│   │   ├── xxx.cpp
│   │   ├── xxx_kernel.h
│   │   └── kernel_common.h（如需要）
│   ├── model.py          # 参考 PyTorch 模型，禁止修改
│   └── model_new_ascendc.py  # AscendC 优化实现
└── archive_tasks/        # 历史成功任务，可作为参考实现
```

## Skill 参考资料

本 skill 提供以下参考资料（位于 `@references/` 目录）：
- `@references/AscendCDesign.md` — AscendC Kernel 设计与实现指南
- `@references/AscendCHost.md` — Host 侧准备与 Kernel 入口详细参考
- `@references/AscendCCompute.md` — 计算子模块详细参考
- `@references/AscendCSync.md` — 跨核同步详细参考
- `@references/AscendCVerification.md` — AscendC 验证指南
- `@references/AscendC_knowledge/` — AscendC 知识库目录
- `@references/DataCopyPad.md` — DataCopyPad API 完整参考（Vector 侧数据搬运必读）
- `@references/evaluate_ascendc.sh` — AscendC 评测脚本

除非用户明确指定其他目录，否则默认使用传入的 `output_dir` 作为当前任务目录。
其他任务目录可以作为参考实现；允许在完成需求分析和设计后按需复用局部函数或片段，但禁止不加分析地直接整体复制参考任务源码、目录或完整实现到当前任务中。

## 流程

执行以下各步骤前，必须先阅读对应的参考文档，再开始实现、验证与迭代。

### 1. `需求分析`（必须先完成）

仔细阅读 `{output_dir}/model.py`，提取以下信息：

**算子语义**：
- `Model.forward()` 中执行了哪些计算操作（元素级、归约、矩阵乘、索引等）
- 输入张量个数、每个张量的含义和预期 shape/dtype
- `__init__` 中的属性参数（如 `eps`、`dim`、`keepdim` 等）

**数据流特征**：
- 是否需要 dtype cast（如 fp16/bf16 内部用 fp32 计算）
- 是否有 broadcast 场景
- 是否有 workspace 需求（如跨 AIC/AIV 通信、排序临时空间等）
- 是否有归约操作（sum、max、mean 等）

**分支与 tiling 策略预判**：
- 输入 shape 是否有显著的不同规模区间（如 N <= 1024 vs N > 8192）
- 是否需要按 dtype 分不同实现路径
- 是否需要多 AIV 子核

### 2. `相近样例对标`

检查 `archive_tasks/` 中是否存在类型非常相近的官方或高质量 AscendC 实现样例。如果存在，先提炼该样例的统一入口组织、不同 kernel 分支划分、数据搬运阶段和调用层薄封装方式，并将其作为当前任务的优先对标对象。

对标后必须明确当前任务自己的分支规划与实现边界；只有确认某些局部函数或片段与当前任务需求完全一致时，才允许按需复用，禁止将参考任务整体搬运为当前实现。

对标时重点确认：
- 是否采用统一的 pybind11.cpp 入口做运行时分发
- 不同 kernel 分支是否分别覆盖完整场景
- 不同分支的完备度是否和场景复杂度匹配
- `model_new_ascendc.py` 是否保持薄调用层

### 3. `AscendC Kernel 设计`

在充分理解需求和参考样例后，进行 AscendC kernel 的完整设计。设计内容应覆盖：

**Host 侧设计**：
- `xxx_tiling.h`：定义 tiling struct，包含 shape、block size、tile size、workspace 深度等参数
- `pybind11.cpp`：Python 接口、输入校验、输出与 workspace 分配、tiling 构造、kernel launch 分发

**Device 侧设计**：
- Kernel 入口 `xxx.cpp`：`__global__ __aicore__` kernel 和 `extern "C"` launch 函数
- 主 Kernel 类 `xxx_kernel.h`：`Init()` / `Process()` 主流程，管理 GM tensor、调度和流水
- 计算子模块（如需要）：按计算阶段拆分为独立头文件

**关键设计原则**：
- 禁止在 C++ 中直接调用 torch / ATen 计算接口
- pybind11.cpp 只负责参数检查、输出分配、workspace 分配、tiling 填充和 kernel launch
- pybind11.cpp 禁止承载任何算子主体语义；尤其禁止通过 host 侧生成逐元素映射表、索引表、mask、copy plan 等方式“间接实现”算子
- 若算子客观上存在多个关键分支（如不同 shape 规模、不同 dtype），必须完整实现这些分支
- 每个独立场景分支应有自己的 kernel 类、入口和 launch 函数
- Host 侧负责根据 shape、dtype 等条件选择调用哪个 launch 入口

**设计产物要求**：
- 在开始写 `kernel/` 和 `model_new_ascendc.py` 之前，必须先在 `{output_dir}/design/` 下产出需求分析与分支设计文档
- 设计文档必须列出需要覆盖的 mode / dtype / shape 分支，以及每个分支对应的 host dispatch 和 kernel 入口

参考文档：`@references/AscendCDesign.md`、`@references/AscendCHost.md`

### 4. `AscendC 实现`

按照设计完成代码编写，生成 `{output_dir}/kernel/` 下的所有文件和 `{output_dir}/model_new_ascendc.py`。

**实现顺序建议**：
1. 先写 `xxx_tiling.h`（定义常量 + tiling struct）
2. 再写主 Kernel 类 `xxx_kernel.h`（Init / Process / CopyIn / Compute / CopyOut）
3. 然后写 kernel 入口 `xxx.cpp`
4. 最后写 `pybind11.cpp`（参数校验 + tiling 填充 + launch 分发）
5. 最后写 `model_new_ascendc.py`（薄封装层）

**API 查询规范**：
- 实现 AscendC kernel 时，应查阅 `@references/AscendC_knowledge/` 目录下的具体 API 文档
- 知识库入口：`api_reference/INDEX.md`
- 查阅 `@references/AscendC_knowledge/api_reference/INDEX.md` 获取具体 API 文档

参考文档：`@references/AscendCCompute.md`、`@references/AscendCSync.md`

### 5. `AscendC 验证与迭代`

编写完成后，调用 `@references/evaluate_ascendc.sh {output_dir}` 验证 AscendC；如果结果不正确，继续迭代修改直到通过验证。

**迭代规则**：
- 迭代次数上限为 3 次
- 若 3 次迭代后仍未通过验证，停止迭代并报告当前状态
- 每次迭代前，优先自查 `model_new_ascendc.py` 是否出现语义级 PyTorch 或 torch_npu 调用；如果出现，必须先删除这类违规实现，再继续验证
- 每次迭代优先检查：tiling 参数一致性、DataCopyPad 参数正确性、PIPE barrier 完整性、AIC/AIV 同步正确性
- 若某些 mode / dtype / shape 分支在迭代上限内仍无法完成，禁止用 PyTorch 或 torch_npu 包装补齐，必须直接报告未完成状态

参考文档：`@references/AscendCVerification.md`
