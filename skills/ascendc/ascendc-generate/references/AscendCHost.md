## Host 侧准备与 Kernel 入口详细参考

本文档包含 Host 侧 tiling/pybind11、Kernel 入口、主 Kernel 类的完整实现细节与代码示例。
概览与判断规则见 `@references/AscendCDesign.md`。

---

## 第一章：Host 侧准备 `xxx_tiling.h` + `pybind11.cpp`

### 1. Tiling 参数一致性

确保所有 kernel 组件使用一致的 tiling 参数：

```cpp
// 在一处定义，到处使用
constexpr uint32_t baseM = 64;
constexpr uint32_t baseN = 64;
constexpr uint32_t baseK = 64;

// 多 Vector 核的子块
constexpr uint32_t subBlockM = baseM;  // 或 baseM / AIV 子核数
constexpr uint32_t vecBlockN = baseN;  // 必须与 baseN 匹配！
```

**警告：** 参数不匹配（如 `vecBlockN = 256` 而 `baseN = 64`）会导致错误的内存访问模式。

---


### 2. Tiling Struct：在 Host 侧预计算运行时参数

避免在 kernel 的 `Process()` 中重复计算 `nTiles` / `nTilesPerH` 等派生量。推荐在 Host（pybind11.cpp）侧预先计算并写入 tiling struct，kernel 直接读取：

**tiling struct 推荐字段**：

```cpp
struct ReshapeMatmulQuantTiling {
    int32_t M, N, H_K;      // 基本形状
    int32_t baseM, baseN, baseK, K_L1;  // tile 大小
    int32_t nTiles;          // = N / baseN，避免 kernel 里除法
    int32_t nTilesPerH;      // = H_K / baseN，避免 kernel 里除法
};
```

**Host 侧填充（pybind11.cpp）**：

```cpp
tp->nTiles     = N   / DEFAULT_BASE_N;
tp->nTilesPerH = H_K / DEFAULT_BASE_N;
```

**Kernel 侧使用**：

```cpp
for (int by = 0; by < tiling_.nTiles; by++) {
    int groupId    = by / tiling_.nTilesPerH;
    int colInGroup = by % tiling_.nTilesPerH;
    ...
}
```

> 虽然 `N / baseN` 在 kernel 里做也能正确，但将派生量存入 tiling struct 是生产代码的标准做法，便于调试和验证。

---


### 3. 绑定层职责

`kernel/pybind11.cpp` 的职责是把 AscendC kernel 包装成 Python 可调用扩展。

#### 3.1 模块名

模块名由

```cpp
PYBIND11_MODULE(<name>, m)
```

决定，`model_new_ascendc.py` 必须 import 同一个 `<name>`。

推荐格式：
- 任务目录：`<op_name>`
- 扩展模块：`_<op_name>_ext`
- Python 导入：`import _<op_name>_ext as _ext`

不要让扩展模块名与任务目录同名，否则 Python 可能先命中同名目录而不是扩展模块。

#### 3.2 绑定函数

绑定函数只接收算子的显式输入张量，不接收输出张量和 workspace。

函数内部负责：
- 检查输入 shape 和 dtype
- 从输入 shape 推导运行时参数，如 `M/N/K`
- 分配输出张量
- 如有需要，分配 workspace
- 构造 tiling tensor
- 调用 `extern "C"` kernel launch 函数
- 返回输出张量

#### 3. Workspace

规则：只要算子实现中需要跨核/workspace 通信（如 AIC→AIV 数据传递、排序临时空间等），`pybind11.cpp` 就必须分配 workspace。

实践要求：
- workspace 的字节数必须和 kernel 中的 block 组织、累加 dtype、并行度一致
- workspace 在 `pybind11.cpp` 中可以分配为一维 `Byte` tensor，只要总字节数正确即可



## 第四章：Kernel 入口 `xxx.cpp`

### 1. 整体代码结构和控制流

AscendC 代码必须将算子的整体工作负载执行模型映射到 AI Core 的执行环境。

| 执行概念 | AscendC 实现 | 原则 |
| :--- | :--- | :--- |
| **工作负载循环**（遍历分片数据/tile） | 封装在 **`Process()`** 方法中。 | `Process()` 管理分配给该核的数据单元的执行循环。 |
| **计算阶段** | `Process()` 必须调用专用函数：**`CopyInX`**、**`ComputeX`** 和 **`CopyOutX`**。 | 如果核心逻辑涉及多个不同的处理过程或数据移动步骤，使用编号（如 `CopyIn1`、`Compute2`）。 |
| **函数定义** | 每个阶段函数必须定义为 `__aicore__ inline`，如果需要计算全局内存（GM）地址，则接受当前循环索引（`uint32_t idx`）。 | 标准 kernel 函数属性和结构。 |

---


### 2. Kernel 类型配置

```cpp
// Kernel 入口点 - 指定 AIC + AIV 混合模式（每个 block 包含 1 个 AIC + 1 个 AIV）
extern "C" __global__ __aicore__ void kernel_custom(GM_ADDR ...inputs..., GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);  // 1 AIC + 1 AIV = 1 block
    AscendC::TPipe pipe;
    KernelClass<...> kernel;
    kernel.Init(..., workspace, tiling, &pipe);
    kernel.Process();
}
```

> **⚠️ KERNEL_TYPE 与 AIV 子核数对应关系**
>
> 当算子设计使用多 AIV 子核并行处理一个 block 的数据时，KERNEL_TYPE 的选择必须与 AIV 子核数一致：
>
> | AIV 子核数 | KERNEL_TYPE | 每个 block 组成 | GetSubBlockNum() |
> |:---|:---|:---|:---|
> | 1 | `KERNEL_TYPE_MIX_AIC_1_1` | 1 AIC + 1 AIV | 2 |
> | 2 | `KERNEL_TYPE_MIX_AIC_1_2` | 1 AIC + 2 AIV | 3 |
>
> **多 AIV 时的子块分配**：
> ```cpp
> // Init 中：
> int coreIdx = GetBlockIdx() / GetSubBlockNum();  // 物理核索引（跨 AIC/AIV）
> // AIV 侧：
> int subTileM = baseM / GetSubBlockNum();          // ❌ 错误！GetSubBlockNum 包含 AIC
> int subTileM = baseM / (GetSubBlockNum() - 1);    // ❌ 也不对
>
> // ✅ 正确做法（参考 matmul-leakyrelu）：
> // AIV 子核数=2 时，GetSubBlockNum()=3（1 AIC + 2 AIV）
> // AIV 内部子块数由 AIV 子核数决定，直接用 GetSubBlockNum() 即可
> // 因为 AIC 不进入 AIV 分支
> if ASCEND_IS_AIV {
>     subTileM_ = baseM / GetSubBlockNum();  // 对于 MIX_AIC_1_2: 128/3 ≈ 42
>     // 实际上 matmul-leakyrelu 中 subTileM = baseM / GetSubBlockNum() = 128/3
>     // 但这不能整除！正确的理解是：
>     // GetSubBlockNum() 返回 AIV 子核数（不含 AIC），在 MIX_AIC_1_2 模式下为 2
>     subTileM_ = baseM / GetSubBlockNum();  // 128 / 2 = 64
>     int rowOffset = GetSubBlockIdx() * subTileM_;  // AIV 0: 0, AIV 1: 64
> }
> ```
>
> **多 AIV 子块映射**：
> ```python
> # AIV 子核分配示例：
> # AIV 子核数 = 2
> sub_block_M = block_M // 2  # 128 // 2 = 64
> # 从 workspace 读取到 UB
> ```
> ```cpp
> // AscendC:
> if ASCEND_IS_AIV {
>     int subTileM = baseM / GetSubBlockNum();  // 64
>     int rowOffset = GetSubBlockIdx() * subTileM;
>     auto subSlot = slot[rowOffset * baseN];
>     auto cBlock = cGM_[(mIdx * baseM + rowOffset) * N + nIdx * baseN];
> }
> ```


## 第五章：主 Kernel 类 `*.h`

- 每个独立的场景分支对应一个独立的主 `Kernel` 类。
- 每个主 `Kernel` 类放到自己的头文件中，例如 `xxx_merge_n_kernel.h`、`xxx_single_row_kernel.h`。

### 1. 数据和缓冲区管理

本节定义算子中的内存分配和移动如何映射到 AscendC 的队列和张量缓冲区（`TBuf`）系统，用于管理统一缓冲区（UB）。

#### A. 数据加载（`CopyInX` 函数）

1.  **分配 UB 空间：** 使用相应输入队列的 **`AllocTensor<T>()`** 方法在 UB 中预留一个本地张量。
2.  **移动数据（GM 到 UB）：** 使用 **`AscendC::DataCopy`** 将数据 tile 从全局内存（算子的输入张量）传输到本地 UB 张量。
3.  **传输到计算阶段：** 使用输入队列的 **`EnQue()`** 方法将加载的本地张量传递到下一阶段。

#### B. 计算（`ComputeX` 函数）

1.  **获取所有张量（必须的开始步骤）：** 在每个 `ComputeX` 函数的**最开始**，获取该计算阶段所需的所有张量：
    * **输入张量：** 从输入队列中出队（`inQueue.DeQue<T>()`）。
    * **工作缓冲区：** 对于内部临时缓冲区，使用预定义成员张量缓冲区（`TBuf`）的 **`Get<T>()`** 方法（如 `sharedBuf.Get<float>()`）。
2.  **执行逻辑：** 将算子的计算操作转换为 AscendC API。
3.  **流程控制：**
    * 如果后续阶段（`Compute` 或 `CopyOut`）需要该结果，使用输出队列的 **`EnQue()`** 方法。
    * 一旦输入张量处理完毕且不再需要，在其来源队列上使用 **`FreeTensor()`**。
4.  **全局同步**
当不同 kernel 操作同一块全局内存且可能出现数据依赖问题时，插入同步语句 `AscendC::SyncAll()`。

#### C. 数据存储（`CopyOutX` 函数）

1.  **获取结果：** 使用输出队列的 **`DeQue<T>()`** 方法从上一阶段获取最终结果张量。
2.  **移动数据（UB 到 GM）：** 使用 **`AscendC::DataCopy`** 将结果从本地 UB 张量传输到算子的输出全局张量。
3.  **释放 UB 空间：** 在输出队列上使用 **`FreeTensor()`** 释放本地张量缓冲区。

---


### 2. AIC（Cube 核）的 Matmul Kernel

Cube kernel 将 A/B 矩阵从 GM 经过 L1 加载到 L0，执行 MMAD，并通过 Fixpipe 将结果写入 workspace：

```cpp
template <typename aType, typename bType, typename cType>
__aicore__ inline void MatmulKernel::ComputeBlock(
    const GlobalTensor<aType> &aBlock,
    const GlobalTensor<bType> &bBlock,
    const GlobalTensor<cType> &cBlock)  // cBlock 是 workspace 槽位
{
    LocalTensor<cType> c1Local = outQueueCO1.AllocTensor<cType>();

    uint32_t loop_k = (k_ + K_L1 - 1) / K_L1;
    uint32_t loop_kk = (K_L1 + baseK - 1) / baseK;

    for (uint32_t k_outer = 0; k_outer < loop_k; ++k_outer) {
        // 加载 A/B 到 L1（ND -> Nz 格式）
        CopyA(aBlock[k_outer * K_L1]);
        CopyB(bBlock[k_outer * K_L1 * bDValue_]);

        // 关键：每个 K_L1 迭代只 DeQue 一次，不是每个 baseK 迭代
        LocalTensor<aType> a1Local = inQueueA1.DeQue<aType>();
        LocalTensor<bType> b1Local = inQueueB1.DeQue<bType>();

        for (uint32_t kk = 0; kk < loop_kk; ++kk) {
            bool cmatrixInitVal = (k_outer == 0 && kk == 0);
            // 向内层循环传递 L1 缓冲区的偏移指针
            SplitA(a1Local[baseMK * kk]);
            SplitB(b1Local[baseKN * kk]);
            Compute(c1Local, cmatrixInitVal);
        }

        // 内层循环完成后只 Free 一次
        inQueueA1.FreeTensor(a1Local);
        inQueueB1.FreeTensor(b1Local);
    }

    outQueueCO1.EnQue(c1Local);
    CopyOut(cBlock);  // Fixpipe 到 workspace
}
```

**关键要点：** 当加载跨多个 baseK 迭代的 L1 缓冲区时：
- 每个 K_L1 迭代开始时只 **DeQue 一次**
- 向内层循环传递**偏移指针**（`a1Local[baseMK * kk]`）
- 所有内层迭代完成后只 **FreeTensor 一次**
- 这避免了重复的 DeQue/EnQue 循环，防止流水线停顿

#### A. AIV（Vector 核）的 Vector Kernel

Vector kernel 从 workspace 读取数据，应用逐元素运算，并写入输出 GM：

```cpp
template <typename accType, typename outType>
__aicore__ inline void ScaleKernel::ProcessBlock(const GlobalTensor<accType> &slot)
{
    // 1. 从 workspace 槽位加载
    LocalTensor<accType> accLocal = inQueue.AllocTensor<accType>();
    DataCopyExtParams copyParams = {1, subBlockM * blockN_ * sizeof(accType), 0, 0, 0};
    DataCopyPad(accLocal, slot, copyParams, padParms);
    inQueue.EnQue(accLocal);

    // 2. 处理（类型转换、缩放、类型转换）
    auto accIn = inQueue.DeQue<accType>();
    LocalTensor<float> fp32Local = rowBuf_.Get<float>();
    Cast(fp32Local, accIn, RoundMode::CAST_NONE, subBlockM * blockN_);

    // 逐行应用缩放
    for (uint32_t i = 0; i < subBlockM; ++i) {
        Mul(fp32Local[i * blockN_], fp32Local[i * blockN_], scaleLocal, blockN_);
    }

    LocalTensor<outType> outLocal = outQueue.AllocTensor<outType>();
    Cast(outLocal, fp32Local, RoundMode::CAST_NONE, subBlockM * blockN_);
    outQueue.EnQue(outLocal);

    inQueue.FreeTensor(accIn);
}

__aicore__ inline void ScaleKernel::CopyOut(const GlobalTensor<outType> &cBlock, uint32_t nDim)
{
    auto outLocal = outQueue.DeQue<outType>();
    DataCopyExtParams copyParams = {1, subBlockM * blockN_ * sizeof(outType), 0, 0, 0};
    DataCopyPad(cBlock, outLocal, copyParams);
    outQueue.FreeTensor(outLocal);
}
```

#### B. 带有 AIC/AIV 分支的 Process 循环

```cpp
template <typename...>
__aicore__ inline void KernelClass::Process()
{
    int mIdx, nIdx;
    while (sched_.HasNext()) {
        sched_.Next(mIdx, nIdx);

        if ASCEND_IS_AIC {
            auto slot = wsQueue_.ProducerAcquire();
            auto aBlock = aGM_[mIdx * baseM * Ka];
            auto bBlock = bGM_[nIdx * baseN];
            mm_.ComputeBlock(aBlock, bBlock, slot);
            wsQueue_.ProducerRelease();
        }

        if ASCEND_IS_AIV {
            auto slot = wsQueue_.ConsumerAcquire();
            scaleKernel_.ProcessBlock(slot);
            wsQueue_.ConsumerRelease();
            auto cBlock = cGM_[mIdx * baseM * N + nIdx * baseN];
            scaleKernel_.CopyOut(cBlock, N);
        }
    }
}
```

#### C. 常见陷阱

| 问题 | 症状 | 解决方法 |
|-------|---------|-----|
| **TQue depth=0 用了返回值形式 API** | 编译报错 `must use AllocTensor<LocalTensor&> api while tque's depth is zero` | VECIN/VECOUT 队列（depth=0）必须用引用形式：`queue.AllocTensor<T>(localVar)`、`queue.DeQue<T>(localVar)` |
| **TBuf DataCopy 后缺少 PipeBarrier** | 结果随机错误、部分数据为 0 或垃圾值 | 从 GM DataCopy 到 TBuf 后必须插入 `PipeBarrier<PIPE_MTE2>()` |
| **Fixpipe 未同步** | 流同步超时（507046） | 确保 CrossCoreSetFlag 使用正确的 PIPE 类型（AIC 使用 `PIPE_FIX`）|
| **内层循环中重复 DeQue** | 结果错误、队列下溢 | 每个外层迭代只 DeQue 一次，向内层循环传递偏移 |
| **缓冲区大小错误** | 内存溢出、NaN 结果 | 缓冲区大小与实际 tile 维度对齐（baseM × baseN）|
| **KERNEL_TASK_TYPE 错误** | Block 索引不匹配 | 每个 block 包含 1 AIC + 1 AIV 时使用 `KERNEL_TYPE_MIX_AIC_1_1`；1 AIC + 2 AIV 时使用 `KERNEL_TYPE_MIX_AIC_1_2` |
| **Depth 太小** | 流水线停顿 | 使用 `DEPTH >= 2` 实现双缓冲 |
| **float16 输出溢出** | 测试中 inf/-inf 导致 assertRtolEqual 失败 | int8 matmul 结果可能很大，乘 scale 后 cast 到 fp16 易溢出；测试时用小范围输入或小 scale |
| **WholeReduceMax 参数错误** | 编译报错 `requires at least 7 arguments` | AscendC 无简化 `WholeReduceMax(dst, src, count)` 形式；改用 `ReduceMax(dst, src, workBuf, count)` Level 2 API |
| **使用不存在的 Divs** | 编译报错 `no member named 'Divs' in namespace 'AscendC'` | AscendC 无标量除法 `Divs`；改用 `Muls(dst, src, 1.0f/scaleVal, count)` 实现 |
| **Fixpipe dstStride = baseN** | 写入大矩阵 workspace 时 tile 间数据覆盖 | 写入 M×N 大矩阵中的子 tile 时，dstStride 应设为完整行宽 N |

---


### 3. MatmulKernel：K_total 和 dstStride 按调用传递

当算子的 K 维度可能在不同调用中变化（如 reshape 后的分组矩阵乘法），推荐将 `K_total` 和 `dstStride` 通过 `ComputeBlock()` 参数传入，而非在 `Init()` 中固定：

```cpp
// ✅ 推荐：灵活传入 K_total 和 dstStride
class MatmulKernel {
    void Init(uint32_t ldA, uint32_t ldB, TPipe &pipe);  // 只需 lda, ldb
    void ComputeBlock(aBlock, bBlock, wsBlock, K_total, dstStride);  // 动态
};

// 调用示例（K 固定时 K_total=H_K，dstStride=N）
mm_.ComputeBlock(aBlock, bBlock, wsBlock, tiling_.H_K, tiling_.N);
```

> `ldA` 和 `ldB` 是矩阵的行跨度（memory layout），在整个 kernel 执行期间不变，适合在 Init 固定。`K_total` 和 `dstStride` 可能随输入动态变化，按调用传递更灵活。

