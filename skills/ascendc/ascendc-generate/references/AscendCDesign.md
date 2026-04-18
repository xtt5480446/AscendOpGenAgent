## AscendC Kernel 设计关键原则

本文档讨论如何直接根据算子需求，系统地设计并实现 AscendC kernel。后文中的 tiling、绑定层、主 kernel 类、子模块拆分与同步关系，都应以算子的实际计算逻辑和数据流特征为直接来源。

1. **先查 API 文档**
   实现时，应查阅 `@references/AscendC_knowledge/` 目录下对应的具体 API 文档，确认所需 AscendC API 的接口和使用方式。
   知识库入口：`api_reference/INDEX.md`
   知识库入口：`api_reference/INDEX.md`

2. **禁止在 C++ 中直接调用 torch / ATen 计算接口**
   `pybind11.cpp` 中禁止使用 `torch::*`、`torch::nn::functional::*`、`ATen` 库或任何 `at::*` 计算接口来实现或替代核心计算，包括但不限于 `at::einsum`、`at::matmul`、`at::softmax`、`at::bmm` 等。绑定层只负责参数检查、输出与 workspace 分配、tiling 填充和 kernel launch，不允许把核心计算留在 C++/ATen 侧。
---

## AscendC Kernel 设计总览

一个完整的 AscendC 实现，通常包含 5 部分：

1. **Host 侧准备**：`xxx_tiling.h` + `pybind11.cpp`
   这两部分共同构成 AscendC 的 host 侧准备逻辑。`xxx_tiling.h` 负责定义 shape、block size、tile size、workspace 深度等参数；`pybind11.cpp` 负责 Python 接口、输入校验、输出与 workspace 分配、tiling 构造和 kernel launch。

2. **公共工具**：如 `kernel_common.h`、`workspace_queue.h`、`matmul_tile.h`
   提供调度、数据搬运、workspace 管理等通用能力。

3. **Kernel 入口**：`xxx.cpp`
   定义 `__global__ __aicore__` kernel 和 `extern "C"` launch 函数。

4. **主 Kernel 类**：一个或多个 `*.h`
   负责 `Init()` / `Process()` 主流程，管理 GM tensor、调度和流水。若算子存在多个关键场景分支（如不同 shape 规模、不同计算策略），应将对应的主 `Kernel` 类拆到多个独立头文件中，例如 `xxx_merge_n_kernel.h`、`xxx_single_row_kernel.h`。

5. **计算子模块**（如需要）：如 `matmul.h`、`leakyrelu.h`
   按计算阶段拆分；通常每个独立的计算阶段对应一个子模块。

### 多分支 Kernel 设计规则

若算子客观上存在多个关键场景分支，AscendC 侧就至少要有多少个独立的 kernel 实现单元。不要把多个场景分支折叠进同一个 `Kernel` 类里再靠运行时分支区分。

具体要求：

- 每个独立场景分支都应对应一个独立的主 `Kernel` 类。
- 每个独立场景分支都应对应至少一个独立的 `__global__ __aicore__` kernel 入口和一个匹配的 `extern "C"` launch 函数。
- 如果同一个场景分支需要按 dtype 分成多个实现，例如 fp16 / fp32 / int8，则可以在该分支之下派生出更多 `extern` 入口；但主 `Kernel` 类的个数仍应首先与场景分支的个数对齐。
- Host 侧 `pybind11.cpp` 负责根据 shape、dtype 或其他 trace-time 条件，选择调用哪个 `extern` 入口；这种选择逻辑不应反向合并掉场景分支级别的结构差异。

例如，若算子有 `merge_n`（N 较小）和 `single_row`（N 中等）两个场景分支，则 AscendC 至少应有两个主 `Kernel` 类、两个 `__global__ __aicore__` 入口和两个 `extern "C"` launch 函数；若两者还各自支持多种 dtype，则 `extern` 数量可以更多，但主 `Kernel` 类仍至少是两个。

---

## 第一章：Host 侧准备（摘要）

完整实现细节与代码示例见 `@references/AscendCHost.md`。

**要点**：

- **Tiling 参数一致性**：所有 kernel 组件（Cube/Vector/Host）必须使用同一组 baseM/baseN/baseK 常量，参数不匹配会导致错误的内存访问。
- **Tiling Struct**：在 Host 侧预计算 `nTiles`、`nTilesPerH` 等派生量写入 tiling struct，避免 kernel 里重复除法。
- **绑定层职责**：`pybind11.cpp` 负责参数检查、输出分配、workspace 分配、tiling 构造、kernel launch。绑定函数只接收算子的显式输入张量，不接收输出和 workspace。
- **模块名**：推荐 `_<op_name>_ext`，不要与任务目录同名。
- **Workspace**：只要算子实现中需要跨核/workspace 通信（如 AIC→AIV 数据传递、排序临时空间等），就必须在 pybind11.cpp 中分配 workspace。

---

## 第三章：公共工具——跨核同步（摘要）

完整实现细节与代码示例见 `@references/AscendCSync.md`。

**同步模式判断规则**：

| 场景特征 | 同步模式 | AscendC 实现 |
|:---|:---|:---|
| AIC 每处理完一个 tile 后 AIV 即可开始处理该 tile | 逐 tile 同步（WorkspaceQueue） | 环形缓冲 + 每 tile Acquire/Release |
| AIC 必须完成全部 tile 后 AIV 才能开始（如需要全局统计量） | 批量同步（Bulk Sync） | 单次 CrossCoreSetFlag/WaitFlag |

**WorkspaceQueue vs 批量同步对比**：

| 特性 | WorkspaceQueue（逐 tile 同步） | 批量同步（Bulk Sync） |
|:---|:---|:---|
| **信号次数** | 每个 tile 一次 Acquire/Release | Cube 全部完成后一次 |
| **Workspace 大小** | DEPTH × baseM × baseN × sizeof(T) | M × N × sizeof(T)（全输出） |
| **Vector 启动时机** | Cube 写完一个 tile 即可开始 | 必须等 Cube 全部写完 |
| **适用场景** | 逐 tile 独立处理（如 LeakyReLU、Scale） | 需要全局统计量（如 ReduceMax + 量化） |

**CrossCore flag 规则**：
- AIC 写完后设置 flag：`CrossCoreSetFlag<0x2, PIPE_FIX>(0x8 + idx)`，只调用一次
- AIV 等待 flag：`CrossCoreWaitFlag<0x2>(0x8 + idx)`，所有 AIV 子块共享同一 flag
- 批量同步场景用单 flag 广播，**不要**逐 AIV 子块发送不同 flag

---

## 第四章：Kernel 入口（摘要）

完整实现细节与代码示例见 `@references/AscendCHost.md`。

**KERNEL_TYPE 与 AIV 子核数对应关系**：

| AIV 子核数 | KERNEL_TYPE | 每个 block 组成 |
|:---|:---|:---|
| 1 | `KERNEL_TYPE_MIX_AIC_1_1` | 1 AIC + 1 AIV |
| 2 | `KERNEL_TYPE_MIX_AIC_1_2` | 1 AIC + 2 AIV |

**代码结构要求**：
- `Process()` 封装工作负载循环，调用 `CopyInX` / `ComputeX` / `CopyOutX`
- 每个阶段函数定义为 `__aicore__ inline`
- 多 AIV 子块时，AIV 侧用 `GetSubBlockIdx()` 区分数据偏移

---

## 第五章：主 Kernel 类（摘要）

完整实现细节与代码示例见 `@references/AscendCHost.md`。

**数据和缓冲区管理三阶段**：
- **CopyInX**：`AllocTensor` → `DataCopyPad`（GM→UB） → `EnQue`
- **ComputeX**：`DeQue` → 计算 → `EnQue`（或 `FreeTensor`）
- **CopyOutX**：`DeQue` → `DataCopyPad`（UB→GM） → `FreeTensor`

> **注意**：Vector 侧的 GM↔UB 数据搬运**默认必须使用 `DataCopyPad`**，禁止使用 `DataCopy`。
> `DataCopyPad` 支持字节级粒度的 `blockLen` 和 `srcStride`，能精确控制非连续搬运和 padding，是生产代码的标准做法。
> `DataCopy` 仅在 Cube 侧（L1/L0 相关通路）或特殊兼容场景中使用。

**Matmul Kernel 关键要点**：
- 每个 K_L1 迭代只 DeQue 一次，向内层循环传递偏移指针
- 所有内层迭代完成后只 FreeTensor 一次
- K_total 和 dstStride 推荐按调用传递而非 Init 固定

**常见陷阱速查表**：

| 问题 | 症状 | 解决方法 |
|:---|:---|:---|
| TQue depth=0 用了返回值形式 API | 编译报错 | VECIN/VECOUT 必须用引用形式 |
| TBuf DataCopy 后缺少 PipeBarrier | 结果随机错误 | 插入 `PipeBarrier<PIPE_MTE2>()` |
| Fixpipe 未同步 | 流同步超时 | CrossCoreSetFlag 使用 `PIPE_FIX` |
| 内层循环中重复 DeQue | 结果错误 | 每个外层迭代只 DeQue 一次 |
| Fixpipe dstStride = baseN | tile 间数据覆盖 | dstStride 应设为完整行宽 N |
| WholeReduceMax 参数错误 | 编译报错 | 改用 `ReduceMax(dst, src, workBuf, count)` |
| 使用不存在的 Divs | 编译报错 | 改用 `Muls(dst, src, 1.0f/scaleVal, count)` |

---

## 第六章：计算子模块（摘要）

完整实现细节与代码示例见 `@references/AscendCCompute.md`。

### Buffer/Queue 选择判断规则

| 场景 | 使用 | 理由 |
|:---|:---|:---|
| 数据走 CopyIn→Compute→CopyOut 流水线 | **TQue** + VECIN/VECOUT | 自动同步 |
| 仅用于单次计算中的临时存储 | **TBuf** + VECCALC | 无需队列操作 |
| 非流水线输入（如 scale/bias，从 GM 一次加载反复使用） | **TBuf** + 手动 `PipeBarrier<PIPE_MTE2>()` | TBuf 无自动同步 |

**TQue depth=0 API 约束**：VECIN/VECOUT 队列（depth=0）必须用引用形式 `AllocTensor<T>(localVar)` / `DeQue<T>(localVar)`。Cube 侧队列（A1/B1/A2/B2/CO1，depth=1）可用返回值形式。

### TPosition 与物理内存映射

| TPosition | 物理存储 | 说明 |
|:---|:---|:---|
| **A1** / **B1** | L1 Buffer | 大块矩阵 |
| **A2** / **B2** | L0A / L0B | 小块矩阵（MMA 输入） |
| **CO1** | L0C | 矩阵计算结果 |
| **VECIN** / **VECOUT** | Unified Buffer | 矢量计算输入/输出 |
| **VECCALC** | Unified Buffer | 矢量计算临时变量 |

### 数据搬运阶段映射

| 搬运方向 | AscendC 等价操作 | 阶段 |
|:---|:---|:---|
| GM → UB | `AllocTensor` + `DataCopyPad(local, gm, params)` + `EnQue` | CopyIn |
| UB → GM | `DeQue` + `DataCopyPad(gm, local, params)` + `FreeTensor` | CopyOut |
| UB 内部 / L1 ↔ L0 | 根据上下文用 `DeQue`/`EnQue` 或 `LoadData` | Compute/Split |

### DataCopy API 选择

| 通路场景 | 推荐 API |
|:---|:---|
| GM → UB | **`DataCopyPad(local, gm, params)`**（默认，禁止用 DataCopy） |
| UB → GM | **`DataCopyPad(gm, local, params)`**（默认，禁止用 DataCopy） |
| GM → L1（ND→NZ） | `DataCopy(l1, gm, nd2nzParams)`（Cube 侧，允许） |
| L1 → L0A/L0B | `LoadData(l0, l1, params)`（Cube 侧，允许） |
| L0C → GM（NZ→ND） | `Fixpipe(gm, l0c, params)`（Cube 侧，允许） |
| 大矩阵中读写子 tile | **`DataCopyPad`** + `DataCopyExtParams`（字节级控制，禁止用 DataCopy+DataCopyParams） |

> **强制规则**：Vector 侧所有 GM↔UB 数据搬运**必须使用 `DataCopyPad`**，`DataCopy` 被禁止。
> `DataCopyPad` 使用 `DataCopyExtParams`（字节单位），能处理任意字节对齐的 blockLen 和 stride；`DataCopy` 使用 `DataCopyParams`（32B DataBlock 单位），粒度粗且容易出错。
> 完整用法参考：`@references/DataCopyPad.md`。 |

### Vector 侧常用 API 注意事项

- **ReduceMax**：无 `WholeReduceMax` 简化形式，用 `ReduceMax(dst, src, workBuf, count)` Level 2 API
- **Divs 不存在**：用 `Muls(dst, src, 1.0f/scaleVal, count)` 代替
- **bfloat16**：与 half 同为 2 字节类型（c0=16），模板函数天然支持
- **两遍 Vector 处理**：如果 Vector 计算需要两个独立的遍历循环，说明是两遍处理模式（Pass 1 扫描统计量 + Pass 2 利用统计量处理）
