## 计算子模块详细参考：Buffer/Queue、数据搬运映射、DataCopyPad API、Vector API

本文档包含计算子模块的完整实现细节与代码示例，包括 Buffer/Queue 初始化、数据搬运映射规则、DataCopyPad API 家族、Vector 侧常用 API 注意事项。
概览与判断规则见 `@references/AscendCDesign.md`。

---

## 第六章：计算子模块

### 1. Buffer 和 Queue 初始化

必须根据数据流中的用途，严格选择 TBuf（计算缓冲区）或 TQue（数据队列）类型。这是 AscendC kernel 设计的核心要求。

#### A. Vector 侧缓冲区 (UB)

**L1 Buffer 映射规则** (`T.alloc_ub`):

| 内存分配语义 | AscendC 实现 | TPosition | 物理存储 | 用途 |
| :--- | :--- | :--- | :--- | :--- |
| `x_ub = T.alloc_ub((tileM, tileN), dtype)` | **数据流缓冲区**: `TQue<TPosition::VECIN, 0> inQueue;` | **VECIN** | Unified Buffer | 输入数据缓冲区，需 EnQue/DeQue 操作 |
| `y_ub = T.alloc_ub((tileM, tileN), dtype)` | **数据流缓冲区**: `TQue<TPosition::VECOUT, 0> outQueue;` | **VECOUT** | Unified Buffer | 输出数据缓冲区，需 EnQue/DeQue 操作 |
| `tmp_ub = T.alloc_ub((tileM, tileN), "uint8")` | **临时缓冲区**: `TBuf<TPosition::VECCALC> tmpBuf;` | **VECCALC** | Unified Buffer | 中间计算临时存储，无需 EnQue/DeQue |

**判断规则**：
- 如果该 UB buffer 需要在 `CopyIn` → `Compute` → `CopyOut` 流水线中传递数据 → 使用 **TQue + VECIN/VECOUT**
- 如果该 UB buffer 仅用于单次计算中的临时存储（如 reduce workspace、cast 中间结果）→ 使用 **TBuf + VECCALC**

**初始化方式**:

```cpp
// ============ TQue 方式（数据流缓冲区）============

// 算子语义：分配 UB 缓冲区 ((tileM, tileN), dtype)  // 用于 CopyIn 阶段
// AscendC 等价实现:
AscendC::TQue<AscendC::TPosition::VECIN, 0> inQueue;  // 第二个模板参数0表示队列深度占位（实际深度由InitBuffer指定）
pipe.InitBuffer(inQueue, 2, tileM * tileN * sizeof(dtype));  // 2表示双缓冲

// 算子语义：分配 UB 缓冲区 ((tileM, tileN), dtype)  // 用于 CopyOut 阶段
// AscendC 等价实现:
AscendC::TQue<AscendC::TPosition::VECOUT, 0> outQueue;
pipe.InitBuffer(outQueue, 2, tileM * tileN * sizeof(dtype));

// 使用流程:
// CopyIn: AllocTensor → DataCopy → EnQue
// Compute: DeQue → 计算 → EnQue
// CopyOut: DeQue → DataCopy → FreeTensor

// ============ TBuf 方式（临时缓冲区）============

// 算子语义：分配 UB 临时缓冲区 ((tileM, tileN), "uint8")  // 用于临时存储
// AscendC 等价实现:
AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
pipe.InitBuffer(tmpBuf, tileM * tileN * sizeof(uint8_t));

// 使用流程:
// auto tmpLocal = tmpBuf.Get<dtype>();  // 直接获取，无需 AllocTensor/FreeTensor
// 然后在计算中使用 tmpLocal
```

**TQue vs TBuf 关键差异**:

| 特性 | TQue（队列） | TBuf（缓冲区） |
| :--- | :--- | :--- |
| **入队/出队** | 支持 EnQue/DeQue | 不支持 |
| **内存申请** | 需要 AllocTensor/FreeTensor | 直接 Get，无需释放 |
| **双缓冲** | 支持（InitBuffer num=2） | 不支持（只有一块内存） |
| **适用场景** | 流水线数据传递 | 临时变量、中间结果 |
| **一次初始化** | 可分配多块内存（num参数） | 只分配一块内存 |

> **⚠️ TQue 第二个模板参数 depth=0 时的 API 约束**
>
> `TQue<TPosition::VECIN, 0>` 的第二个模板参数 `0` 表示编译期 depth 占位（实际深度由 `InitBuffer` 的 num 参数决定）。
> **当 depth=0 时，必须使用引用形式 API**，否则编译报错 `must use AllocTensor<LocalTensor&> api while tque's depth is zero`。
>
> ```cpp
> // ✅ 正确：引用形式（depth=0 时必须使用）
> AscendC::TQue<AscendC::TPosition::VECIN, 0> inQueue_;
> AscendC::LocalTensor<int32_t> inLocal_;  // 成员变量
>
> inQueue_.AllocTensor<int32_t>(inLocal_);  // 引用形式
> inQueue_.DeQue<int32_t>(inLocal_);        // 引用形式
>
> // ❌ 错误：返回值形式（depth=0 时编译失败）
> auto inLocal = inQueue_.AllocTensor<int32_t>();  // static_assert 失败
> auto inLocal = inQueue_.DeQue<int32_t>();         // static_assert 失败
>
> // ✅ 正确：返回值形式（depth≥1 时可用）
> AscendC::TQue<AscendC::TPosition::A1, 1> inQueueA1;  // depth=1
> auto a1Local = inQueueA1.AllocTensor<half>();     // OK
> auto a1Local = inQueueA1.DeQue<half>();           // OK
> ```
>
> **经验规则**：VECIN/VECOUT 队列通常使用 depth=0 + 引用形式，Cube 侧队列（A1/B1/A2/B2/CO1）使用 depth=1 + 返回值形式。

---

#### D. TBuf 从 GM 加载数据（非流水线输入）

当算子中存在**非流水线输入**（如 scale 向量、bias 向量、常量表等），这些数据不走 CopyIn→Compute→CopyOut 流水线，而是每次 ProcessBlock 前从 GM 加载到 UB 直接使用。此时应使用 **TBuf + 手动 PipeBarrier** 而非 TQue。

**典型场景**：
```python
# 算子语义：加载 scale 向量（每个 block 只需一次，不走流水线）
scale_row_ub = T.alloc_ub((1, block_N), "float32")
// 从 GM 搬运 scale 到 UB
# 后续在 for 循环中反复使用 scale_row_ub
```

**AscendC 翻译**：
```cpp
// 声明为 TBuf（非流水线数据，不需要 EnQue/DeQue）
AscendC::TBuf<AscendC::TPosition::VECCALC> scaleBuf_;
pipe->InitBuffer(scaleBuf_, blockN * sizeof(float));

// 在 ProcessBlock 中加载
AscendC::LocalTensor<float> scaleLocal = scaleBuf_.Get<float>();
DataCopyExtParams scaleParams = {1, blockN * sizeof(float), 0, 0, 0};
DataCopyPad(scaleLocal, scaleGM, scaleParams, padParms);
PipeBarrier<PIPE_MTE2>();  // ⚠️ 必须手动同步！TBuf 无 EnQue 自动同步

// 现在可以安全使用 scaleLocal
for (int i = 0; i < subBlockM; i++) {
    AscendC::Mul(fp32Local[i * blockN], fp32Local[i * blockN], scaleLocal, blockN);
}
```

> **⚠️ 关键：TBuf 从 GM DataCopy 后必须插入 `PipeBarrier<PIPE_MTE2>()`**
>
> - TQue 的 `EnQue()` 内部自动插入 MTE2 同步屏障
> - TBuf 没有队列机制，DataCopy（走 MTE2 管线）完成前数据不可用
> - 如果省略 `PipeBarrier<PIPE_MTE2>()`，后续 Vector 计算可能读到未就绪的数据，导致**结果随机错误**
>
> **判断规则**：
> - 数据走 CopyIn→Compute→CopyOut 流水线 → **TQue**（自动同步）
> - 数据从 GM 一次性加载后反复使用 → **TBuf + PipeBarrier**（手动同步）

**实际案例**（来自 LeakyKernel）:

```cpp
// LeakyKernel 使用 TQue 作为数据流缓冲区
template <typename inType, typename outType>
class LeakyKernel {
    AscendC::TQue<AscendC::TPosition::VECIN, 0> reluInQueue_;   // 输入队列
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> reluOutQueue_; // 输出队列
    AscendC::TBuf<AscendC::TPosition::VECCALC> castBuf_;        // 类型转换临时buffer

    __aicore__ inline void Init(int tileM, int tileN, AscendC::TPipe *pipe) {
        this->tileSize = tileM * tileN;
        pipe->InitBuffer(reluInQueue_, 1, tileSize * sizeof(inType));
        pipe->InitBuffer(reluOutQueue_, 1, tileSize * sizeof(outType));
        if constexpr (!std::is_same_v<inType, outType>) {
            pipe->InitBuffer(castBuf_, tileSize * sizeof(outType));  // TBuf 只需指定大小
        }
    }

    __aicore__ inline void ProcessBlock(GlobalTensor<inType> &blockGM) {
        // TQue 使用: AllocTensor → DataCopyPad → EnQue → DeQue → 计算 → EnQue
        reluInQueue_.AllocTensor<inType>(reluInLocal);
        DataCopyExtParams copyParams = {1, tileSize * sizeof(inType), 0, 0, 0};
        DataCopyPad(reluInLocal, blockGM, copyParams, padParms);
        reluInQueue_.EnQue(reluInLocal);

        reluOutQueue_.AllocTensor<outType>(reluOutLocal);
        reluInQueue_.DeQue<inType>(reluInLocal);

        // TBuf 使用: 直接 Get
        if constexpr (!std::is_same_v<inType, outType>) {
            reluCastLocal = castBuf_.Get<outType>();  // 无需 AllocTensor
            Cast(reluCastLocal, reluInLocal, RoundMode::CAST_ROUND, tileSize);
        }

        LeakyRelu(reluOutLocal, reluCastLocal, (outType)0.001, tileSize);
        reluInQueue_.FreeTensor(reluInLocal);  // TQue 需要释放
        reluOutQueue_.EnQue(reluOutLocal);
        // TBuf 的 castBuf_ 无需释放
    }
};
```

| 缓冲区语义 | AscendC 对象类型 | VEC TPosition | 初始化和用途 |
| :--- | :--- | :--- | :--- |
| **输入缓冲区**（如 copyin 中使用的 ub） | **`TQue`**（张量队列） | **VECIN**（向量输入） | **用途：** 用于同步数据传输到 `Compute` 函数。**初始化：** `pipe.InitBuffer(TENSOR_QUEUE, slot_count, SIZE_IN_BYTES)`。 |
| **输出缓冲区**（如 copyout 中使用的 ub） | **`TQue`**（张量队列） | **VECOUT**（向量输出） | **用途：** 用于将数据从 `Compute` 函数同步传输到 `CopyOut`。**初始化：** `pipe.InitBuffer(TENSOR_QUEUE, slot_count, SIZE_IN_BYTES)`。 |
| **中间/工作缓冲区**（`tmp_ub`、`shared_ub`） | **`TBuf`**（张量缓冲区） | **VECCALC**（向量计算） | **用途：** 用于单次 `Compute` 阶段的临时短期存储（如归约工作区、中间结果）。**初始化：** `pipe.InitBuffer(TENSOR_BUFFER, SIZE_IN_BYTES)`。 |

#### B. Cube 侧缓冲区 (L1/L0)

**L1 Buffer 映射规则** (`T.alloc_L1`):

| 内存分配语义 | AscendC 实现 | TPosition | 物理存储 | 判断规则 |
| :--- | :--- | :--- | :--- | :--- |
| `A_L1 = T.alloc_L1((blockM, KL1), dtype)` | `TQue<TPosition::A1, 1> inQueueA1;` | **A1** | L1 Buffer | 该 L1 buffer 通过 LoadData 搬运到 L0A |
| `B_L1 = T.alloc_L1((KL1, blockN), dtype)` | `TQue<TPosition::B1, 1> inQueueB1;` | **B1** | L1 Buffer | 该 L1 buffer 通过 LoadData 搬运到 L0B |

**初始化方式**:
```cpp
// 算子语义：分配 L1 缓冲区 ((blockM, KL1), dtype)
// AscendC 等价实现:
AscendC::TQue<AscendC::TPosition::A1, 1> inQueueA1;  // 声明队列
pipe.InitBuffer(inQueueA1, 2, blockM * KL1 * sizeof(dtype));  // 初始化，2表示双缓冲

// 算子语义：分配 L1 缓冲区 ((KL1, blockN), dtype)
// AscendC 等价实现:
AscendC::TQue<AscendC::TPosition::B1, 1> inQueueB1;
pipe.InitBuffer(inQueueB1, 2, KL1 * blockN * sizeof(dtype));
```

**L0 Buffer 映射规则** (`T.alloc_L0A/L0B/L0C`):

| 内存分配语义 | AscendC 实现 | TPosition | 物理存储 | 用途 |
| :--- | :--- | :--- | :--- | :--- |
| `A_L0 = T.alloc_L0A((blockM, blockK), dtype)` | `TQue<TPosition::A2, 1> inQueueA2;` | **A2** | L0A Buffer | 存放小块左矩阵（MMA输入A） |
| `B_L0 = T.alloc_L0B((blockK, blockN), dtype)` | `TQue<TPosition::B2, 1> inQueueB2;` | **B2** | L0B Buffer | 存放小块右矩阵（MMA输入B） |
| `C_L0 = T.alloc_L0C((blockM, blockN), dtype)` | `TQue<TPosition::CO1, 1> outQueueCO1;` | **CO1** | L0C Buffer | 存放小块矩阵计算结果（MMA输出/累积） |

**初始化方式**:
```cpp
// 算子语义：分配 L0A 缓冲区 ((blockM, blockK), dtype)
// AscendC 等价实现:
AscendC::TQue<AscendC::TPosition::A2, 1> inQueueA2;
pipe.InitBuffer(inQueueA2, 2, blockM * blockK * sizeof(dtype));

// 算子语义：分配 L0B 缓冲区 ((blockK, blockN), dtype)
// AscendC 等价实现:
AscendC::TQue<AscendC::TPosition::B2, 1> inQueueB2;
pipe.InitBuffer(inQueueB2, 2, blockK * blockN * sizeof(dtype));

// 算子语义：分配 L0C 缓冲区 ((blockM, blockN), accum_dtype)
// AscendC 等价实现:
AscendC::TQue<AscendC::TPosition::CO1, 1> outQueueCO1;
pipe.InitBuffer(outQueueCO1, 1, blockM * blockN * sizeof(accum_dtype));  // CO1 通常不需要双缓冲
```

#### C. TPosition 与物理内存映射表

| TPosition | 物理存储 | 说明 |
| :--- | :--- | :--- |
| **A1** | L1 Buffer | 存放左矩阵（大块） |
| **B1** | L1 Buffer | 存放右矩阵（大块） |
| **A2** | L0A Buffer | 存放小块左矩阵（MMA输入A） |
| **B2** | L0B Buffer | 存放小块右矩阵（MMA输入B） |
| **CO1** | L0C Buffer | 存放小块矩阵计算结果 |
| **VECIN** | Unified Buffer | 矢量计算输入数据 |
| **VECCALC** | Unified Buffer | 矢量计算临时变量 |
| **VECOUT** | Unified Buffer | 矢量计算输出数据 |

---


### 2. 数据移动映射规则

数据搬运的 AscendC 实现取决于参数位置和数据流方向。

#### A. 数据搬运映射规则表

| 数据搬运方向 | AscendC 等价实现 | 操作阶段 | 说明 |
| :--- | :--- | :--- | :--- |
| GM → UB | `DataCopyPad(local, gm, params) + EnQue(local)` | **CopyIn** | Vector 侧 GM→UB 搬运。**必须使用 `DataCopyPad`，禁止用 `DataCopy`** |
| UB → GM | `DeQue(local) + DataCopyPad(gm, local, params) + FreeTensor(local)` | **CopyOut** | Vector 侧 UB→GM 搬运。**必须使用 `DataCopyPad`，禁止用 `DataCopy`** |
| UB 内部搬运 | `DeQue(src) + ... + EnQue(dst)` 或直接计算 | **Compute** | 本地到本地，根据上下文决定是否需要队列操作 |

#### B. 详细映射规则

**规则1: 第一参数是 GM tensor（输入数据搬运）**

```python
# 算子语义：从 GM 搬运 A 矩阵 tile 到 L1
```

```cpp
// AscendC 等价（CopyIn 阶段）:
auto a1Local = inQueueA1.AllocTensor<half>();  // 先 AllocTensor
DataCopy(a1Local, aGlobal, blockM * KL1);      // DataCopy: GM → L1
inQueueA1.EnQue(a1Local);                       // EnQue: 放入队列供后续使用
```

**规则2: 第二参数是 GM tensor（输出数据搬运）**

```python
# 算子语义：将 L0C 结果写入 workspace
```

```cpp
// AscendC 等价（CopyOut 阶段）:
auto c1Local = outQueueCO1.DeQue<float>();     // DeQue: 从队列取出
FixpipeNzL0cToNdGm(workspaceGlobal, c1Local, blockM, blockN);  // DataCopy: L0C → GM
outQueueCO1.FreeTensor(c1Local);                // FreeTensor: 释放缓冲区
```

**规则3: 本地缓冲区之间的 copy**

```python
# 算子语义：从 L1 搬运 A tile 到 L0A
```

```cpp
// AscendC 等价（Split 阶段，L1 → L0）:
auto a1Local = inQueueA1.DeQue<half>();   // DeQue: 从 A1 队列取出
auto a2Local = inQueueA2.AllocTensor<half>();  // AllocTensor: 为 A2 分配
LoadNzL1ToZzL0A(a2Local, a1Local[kk * blockK * blockM], blockM, blockK);  // DataCopy: L1 → L0A
inQueueA2.EnQue(a2Local);                  // EnQue: 放入 A2 队列
inQueueA1.FreeTensor(a1Local);             // FreeTensor: 释放 A1 缓冲区
```

#### C. 完整流程映射示例

**典型代码片段**:
```python
// Vector 侧计算:
    // 等待 AIC 完成信号
    // 从 workspace 读取到 UB
    LeakyRelu(c_ub, c_ub, negative_slope_const)  // 计算
    // 从 UB 写入输出 GM
```

**AscendC 等价实现**:
```cpp
// Vector 侧 ProcessBlock
__aicore__ inline void ProcessBlock(GlobalTensor<float> &slotGM) {
    // 数据搬运：workspace → UB → DataCopyPad + EnQue (CopyIn)
    reluInQueue_.AllocTensor<float>(reluInLocal);
    DataCopyExtParams copyInParams = {1, tileSize * sizeof(float), 0, 0, 0};
    DataCopyPad(reluInLocal, slotGM, copyInParams, padParms);
    reluInQueue_.EnQue(reluInLocal);

    // T.tile.leaky_relu(c_ub, c_ub, ...) → Compute
    reluOutQueue_.AllocTensor<float>(reluOutLocal);
    auto reluInLocal = reluInQueue_.DeQue<float>();
    LeakyRelu(reluOutLocal, reluInLocal, (float)0.001, tileSize);
    reluInQueue_.FreeTensor(reluInLocal);
    reluOutQueue_.EnQue(reluOutLocal);
}

// 数据搬运：UB → GM → DeQue + DataCopyPad + FreeTensor (CopyOut)
__aicore__ inline void CopyOut(GlobalTensor<float> &cGM, uint32_t nDim) {
    auto reluOutLocal = reluOutQueue_.DeQue<float>();
    DataCopyExtParams copyOutParams = {1, tileSize * sizeof(float), 0, 0, 0};
    DataCopyPad(cGM, reluOutLocal, copyOutParams);
    reluOutQueue_.FreeTensor(reluOutLocal);
}
```

---


### 3. DataCopy / DataCopyPad API 知识要点

`DataCopyPad` 是 Vector 侧 GM↔UB 数据搬运的**默认且唯一推荐** API。`DataCopy` 仅在 Cube 侧（L1/L0 相关通路）允许使用，Vector 侧**禁止使用**。

完整 `DataCopyPad` 用法参考：`@references/DataCopyPad.md`。

#### A. API 家族概览

| 功能 | API 形式 | 通路支持 | 说明 |
| :--- | :--- | :--- | :--- |
| **Vector 连续/非连续搬运** | **`DataCopyPad(dst, src, params)`** | **GM↔UB** | **默认唯一推荐**。字节级 `blockLen`/`stride` 控制，支持 padding |
| ~~Vector 连续搬运~~ | ~~`DataCopy(dst, src, count)`~~ | ~~GM↔UB~~ | ~~**禁止在 Vector 侧使用**~~ |
| ~~Vector 非连续搬运~~ | ~~`DataCopy(dst, src, params)`~~ | ~~GM↔UB~~ | ~~**禁止在 Vector 侧使用**。DataBlock 粒度粗~~ |
| **ND→NZ 格式转换** | `DataCopy(dst, src, nd2nzParams)` | GM→L1 | Cube 矩阵加载（允许） |
| **NZ→ND 格式转换** | `Fixpipe(dst, src, params)` | L0C→GM | Cube 输出（允许） |

#### B. DataCopyPad 用法（Vector 侧默认）

```cpp
// 核心结构体（字节粒度）
struct DataCopyExtParams {
    uint16_t blockCount;   // 连续传输的数据块个数（行数），[1, 4095]
    uint32_t blockLen;     // 每个数据块长度，单位：字节，[1, 2097151]
    uint32_t srcStride;    // 源端相邻块间隔
    uint32_t dstStride;    // 目的端相邻块间隔
    uint32_t rsv;          // 保留，填 0
};

// stride 单位规则：
// - GM (Global Memory) 端：字节
// - UB (VECIN/VECOUT/VECCALC) 端：DataBlock = 32 字节

// 示例: 搬运 3 行数据，每行 16 字节（4 个 float），GM 紧密连续，UB 紧密连续
DataCopyExtParams params = {3, 16, 0, 0, 0};
DataCopyPad(inLocal, xGM_, params, padParms);  // GM → UB

// 示例: 搬运 3 行数据到 workspace，UB 紧密连续，GM 每行间隔 64 字节
DataCopyExtParams paramsOut = {3, 32, 0, 64 / 32, 0};  // dstStride 单位是 DataBlock
DataCopyPad(workspaceGM_, inLocal, paramsOut);  // UB → GM
```

**注意**：`DataCopyPad` 的 `blockLen` 单位是**字节**，而 `srcStride`/`dstStride` 在 UB 端的单位是**DataBlock (32B)**，在 GM 端的单位是**字节**。这与 `DataCopy` 的 `DataCopyParams`（全部用 DataBlock 单位）有本质区别。

#### C. 常用数据通路与 API 选择

| 通路场景 | 推荐 API | 说明 |
| :--- | :--- | :--- |
| **GM → UB (VECIN)** | **`DataCopyPad(local, gm, params)`** | **必须使用 DataCopyPad**，禁止用 DataCopy |
| **UB → GM (VECOUT)** | **`DataCopyPad(gm, local, params)`** | **必须使用 DataCopyPad**，禁止用 DataCopy |
| **GM → L1 (A1/B1)** | `DataCopy(l1, gm, nd2nzParams)` | Cube 侧，允许 |
| **L1 → L0A (A2)** | `LoadData(l0a, l1, params)` | Cube 侧，允许 |
| **L1 → L0B (B2)** | `LoadData(l0b, l1, params)` | Cube 侧，允许 |
| **L0C (CO1) → GM** | `Fixpipe(gm, l0c, params)` | Cube 侧，允许 |

#### D. 关键约束

**地址对齐要求**:
```cpp
// LocalTensor 地址对齐
// - C2: 64B 对齐
// - C2PIPE2GM: 128B 对齐
// - 其他位置: 32B 对齐

// GlobalTensor 地址对齐
// - 按数据类型大小对齐（如 half 需要 2B 对齐，float 需要 4B 对齐）
```

**DataCopyExtParams 单位说明**:
```cpp
// blockLen 单位：字节
// srcStride/dstStride 单位：GM 端为字节，UB 端为 DataBlock (32字节)

// 示例: 搬运 128 个 float (512字节)
DataCopyExtParams params = {1, 512, 0, 0, 0};  // 1块，512字节
// 对比 DataCopyParams (已废弃于 Vector 侧):
// DataCopyParams params = {1, 16, 0, 0};  // 1块，16×32=512字节
```

#### E. ND→NZ 格式转换（GM→L1）

矩阵数据从 Global Memory 加载到 L1 Buffer 时，需要从 ND 格式转换为 NZ 格式（分块格式）:

```cpp
// GM(ND) → L1(Nz) 格式转换参数
struct Nd2NzParams {
    uint32_t ndNum;           // ND矩阵个数（通常为1）
    uint32_t nValue;          // N轴长度
    uint32_t dValue;          // D轴长度（K轴）
    uint32_t srcNdMatrixStride;  // 源矩阵stride
    uint32_t srcDValue;       // 源D轴实际长度
    uint32_t dstNzC0Stride;   // 目标C0 stride
    uint32_t dstNzNStride;    // 目标N stride
    uint32_t dstNzMatrixStride;  // 目标矩阵stride
};

// 使用示例（来自 matmul_tile.h）:
AscendC::Nd2NzParams params;
params.ndNum = 1;
params.nValue = baseM;    // N轴 = blockM
params.dValue = baseK;    // D轴 = blockK
params.srcNdMatrixStride = 0;
params.srcDValue = lda;   // 原始矩阵的列数
params.dstNzC0Stride = baseM;
params.dstNzNStride = 1;
params.dstNzMatrixStride = 0;
DataCopy(dstL1, srcGM, params);  // 执行 ND→NZ 转换搬运
```

#### F. NZ→ND 格式转换（L0C→GM）

矩阵计算结果从 L0C 输出到 Global Memory 时，需要从 NZ 格式转换回 ND 格式:

```cpp
// L0C(Nz) → GM(ND) 格式转换（使用 Fixpipe）
struct FixpipeParamsV220 {
    uint32_t nSize;      // N轴长度
    uint32_t mSize;      // M轴长度
    uint32_t srcStride;  // 源stride
    uint32_t dstStride;  // 目标stride
    uint32_t ndNum;      // ND矩阵个数
    uint32_t srcNdStride;
    uint32_t dstNdStride;
};

// 使用示例（来自 matmul_tile.h）:
AscendC::FixpipeParamsV220 params;
params.nSize = blockN;
params.mSize = blockM;
params.srcStride = blockM;
params.dstStride = blockN;
params.ndNum = 1;
params.srcNdStride = 0;
params.dstNdStride = 0;
Fixpipe(dstGM, srcL0C, params);  // 执行 NZ→ND 转换输出
```

> **⚠️ Fixpipe dstStride 参数化：写入大矩阵中的子 tile**
>
> 当 Fixpipe 的目标不是独立的 baseM×baseN 缓冲区，而是嵌入在更大 M×N 矩阵（如 workspace）中的子 tile 时，
> `dstStride` 必须设置为**完整行宽 N**，而不是 baseN：
>
> ```cpp
> // 写入 workspace[bx*baseM, by*baseN]（workspace 宽度为 N）
> FixpipeParamsV220 params;
> params.nSize = baseN;       // tile 列数
> params.mSize = baseM;       // tile 行数
> params.srcStride = baseM;   // L0C Nz 格式 stride（固定）
> params.dstStride = N;       // ⚠️ 目标矩阵完整行宽，不是 baseN！
> params.ndNum = 1;
> Fixpipe(wsGM[bx * baseM * N + by * baseN], srcL0C, params);
> ```
>
> **错误对比**：
> - `dstStride = baseN`（❌）：tile 内部行正确，但行间偏移错误，后续 tile 数据覆盖前一 tile
> - `dstStride = N`（✅）：每行跳过完整行宽，正确嵌入大矩阵

#### G. 跨步 DataCopyPad：从大矩阵中读写子 tile

当 Vector 侧需要从 workspace（M×N 大矩阵）中读取 subBlockM×baseN 的子 tile，或向 GM 输出写入子 tile 时，必须使用 **DataCopyExtParams** 处理非连续行：

```cpp
// 从 workspace 读取子 tile（workspace 行宽 N，读 baseN 列）
DataCopyExtParams readParams;
readParams.blockCount = (uint16_t)subBlockM;            // 行数
readParams.blockLen = baseN * sizeof(float);            // 每行字节数
readParams.srcStride = (N - baseN) * sizeof(float);     // GM 端：字节
readParams.dstStride = 0;                               // UB 端：DataBlock=0，紧密连续
readParams.rsv = 0;
DataCopyPad(ubLocal, wsGM[rowOffset * N + colOffset], readParams, padParms);

// 向 GM 输出写入子 tile（输出行宽 N，写 baseN 列）
DataCopyExtParams writeParams;
writeParams.blockCount = (uint16_t)subBlockM;
writeParams.blockLen = baseN * sizeof(int8_t);          // 每行字节数
writeParams.srcStride = 0;                              // UB 端：紧密连续
writeParams.dstStride = (N - baseN) * sizeof(int8_t);   // GM 端：字节
writeParams.rsv = 0;
DataCopyPad(yGM[rowOffset * N + colOffset], ubLocal, writeParams);
```

> **⚠️ 注意**：`DataCopyExtParams` 与已废弃的 `DataCopyParams` 单位不同：
> - `blockLen`：字节（不是 DataBlock）
> - `srcStride`/`dstStride`：GM 端是字节，UB 端是 DataBlock (32B)
> - `srcStride`/`dstStride` 是**相邻块之间的间隔**（不含当前块长度）

#### H. 典型完整流程示例

**Vector 侧 CopyIn → Compute → CopyOut**:
```cpp
// CopyIn: GM → UB
void CopyIn() {
    auto xLocal = inQueue.AllocTensor<half>();  // 申请 UB
    DataCopyExtParams copyParams = {1, tileLength * sizeof(half), 0, 0, 0};
    DataCopyPad(xLocal, xGlobal, copyParams, padParms);  // GM → UB (必须使用 DataCopyPad)
    inQueue.EnQue(xLocal);                       // 入队
}

// Compute: UB 内计算
void Compute() {
    auto xLocal = inQueue.DeQue<half>();        // 出队获取输入
    auto yLocal = outQueue.AllocTensor<half>(); // 申请输出 UB

    // 计算操作...
    Abs(yLocal, xLocal, tileLength);            // UB 内计算

    inQueue.FreeTensor(xLocal);                 // 释放输入 UB
    outQueue.EnQue(yLocal);                     // 输出入队
}

// CopyOut: UB → GM
void CopyOut() {
    auto yLocal = outQueue.DeQue<half>();       // 出队获取结果
    DataCopyExtParams copyParams = {1, tileLength * sizeof(half), 0, 0, 0};
    DataCopyPad(yGlobal, yLocal, copyParams);   // UB → GM (必须使用 DataCopyPad)
    outQueue.FreeTensor(yLocal);                // 释放 UB
}
```

**Cube 侧 矩阵搬运流程**:
```cpp
// GM → L1 (ND→NZ)
void CopyA(GlobalTensor<half> &aBlock) {
    auto a1Local = inQueueA1.AllocTensor<half>();
    LoadNdGmToNzL1(a1Local, aBlock, baseM, baseK, lda);  // ND→NZ 格式转换
    inQueueA1.EnQue(a1Local);
}

// L1 → L0A (NZ→Zz)
void SplitA() {
    auto a1Local = inQueueA1.DeQue<half>();
    auto a2Local = inQueueA2.AllocTensor<half>();
    LoadNzL1ToZzL0A(a2Local, a1Local, baseM, baseK);  // NZ→Zz 格式转换
    inQueueA2.EnQue(a2Local);
    inQueueA1.FreeTensor(a1Local);
}

// L0C → GM (NZ→ND)
void CopyOut(GlobalTensor<float> &cBlock) {
    auto c1Local = outQueueCO1.DeQue<float>();
    FixpipeNzL0cToNdGm(cBlock, c1Local, baseM, baseN);  // NZ→ND 格式转换
    outQueueCO1.FreeTensor(c1Local);
}
```

---


### 4. Vector 侧常用 API 注意事项

#### A. ReduceMax：per-row 归约

AscendC 提供两种 ReduceMax 形式，**没有简化的 `WholeReduceMax(dst, src, count)` 形式**：

```cpp
// ✅ Level 2 简化形式（推荐用于 per-row 归约）
// 将 count 个元素归约为 1 个最大值，结果存放在 dst[0]
ReduceMax(dst, src, sharedTmpBuffer, count);

// Level 0/1 带 repeat 形式（高级）
ReduceMax(dst, src, sharedTmpBuffer, mask, repeatTime, srcRepStride, calIndex);
```

**per-row 归约典型用法**（逐行调用 ReduceMax）：
```cpp
// 对 (subBlockM × blockN) 的矩阵，求每行最大值
AscendC::LocalTensor<float> reduceWork = reduceBuf_.Get<float>();  // 需要 blockN * sizeof(float) 大小

for (int i = 0; i < subBlockM; i++) {
    AscendC::ReduceMax(rowMaxTile, absLocal[i * blockN], reduceWork, blockN);
    PipeBarrier<PIPE_V>();
    // 结果在 rowMaxTile[0]，需要手动提取
    float tileMax = rowMaxTile.GetValue(0);
    float curMax = rowAbsmax.GetValue(i);
    if (tileMax > curMax) {
        rowAbsmax.SetValue(i, tileMax);
    }
}
```

#### B. Muls 代替 Divs：per-row 标量除法

AscendC **没有 `Divs`（标量除法）API**。per-row scalar division 需要：
1. 用 `GetValue()` 从 LocalTensor 提取标量
2. 计算倒数 `1.0f / scale`
3. 用 `Muls()` 乘以倒数

```cpp
// 算子语义：
// for i in range(sub_block_M):
//     T.tile.div(out[i, :], out[i, :], row_scale[i])

// AscendC:
for (int i = 0; i < subBlockM; i++) {
    float scaleVal = rowScale.GetValue(i);  // 提取标量
    if (scaleVal > 0.0f) {
        float invScale = 1.0f / scaleVal;
        AscendC::Muls(fp32Local[i * blockN], fp32Local[i * blockN], invScale, blockN);
    }
}
PipeBarrier<PIPE_V>();
```

> **⚠️ `GetValue()`/`SetValue()` 注意事项**：
> - 这些 API 执行 UB 标量读写，性能不如向量操作
> - 适用于少量标量操作（如 subBlockM=64 次循环）
> - 如果可能，优先使用向量化方案（如 `Mul` 广播操作）

#### C. bfloat16 的 Cube 侧支持

bfloat16 与 half 同为 2 字节类型，c0=16。在 matmul_tile.h 中的格式转换函数需注意：

```cpp
// LoadNdGmToNzL1 和 LoadNzL1ToZzL0A 是模板函数，天然支持 bfloat16_t
// LoadNzL1ToZnL0B 对于 2 字节类型使用 LoadData3DParamsV2（与 half 相同算法）
// LoadNzL1ToZnL0B 对于 int8 使用不同的 LoadData2dTransposeParams（因为 c0=32）

// 2 字节类型（half, bfloat16_t）通用版本：
template<typename T>  // T = half 或 bfloat16_t
__aicore__ inline void LoadNzL1ToZnL0B_2B(
    const LocalTensor<T> &dst, const LocalTensor<T> &src,
    uint32_t k, uint32_t n, uint32_t colC0Stride)
{
    LoadData3DParamsV2<T> params;
    params.l1H = 1;  params.l1W = colC0Stride;
    params.channelSize = n;  params.kExtension = n;  params.mExtension = k;
    params.strideH = 1;  params.strideW = 1;
    params.filterH = 1;  params.filterW = 1;
    params.dilationFilterH = 1;  params.dilationFilterW = 1;
    LoadData(dst, src, params);
}
```

> **c0 大小与数据类型**：
> - `half` / `bfloat16_t`：c0 = 32 / 2 = 16
> - `int8_t`：c0 = 32 / 1 = 32（需要不同的转置加载算法）

#### D. 两遍 Vector 处理模式

当后处理需要**全局统计量**（如 per-row absmax 用于量化），Vector 必须分两遍处理：

```
Pass 1（扫描）：遍历所有 n_tile，累积统计量（如 per-row 最大绝对值）
           ↓ 计算 scale = absmax / 127
Pass 2（处理）：遍历所有 n_tile，利用统计量进行量化（div by scale → cast → write）
```

**与单遍处理的对比**：

| 模式 | 适用场景 | Workspace 同步 | Vector 遍历次数 |
|:---|:---|:---|:---|
| **单遍** | 逐 tile 独立处理（LeakyReLU、Scale） | WorkspaceQueue 逐 tile | 1 次 |
| **两遍** | 需要全局统计量（row-wise 量化） | 批量同步 | 2 次 |

**两遍处理模式识别**：如果 Vector 核对同一 workspace 区域有**两个独立的遍历循环**（第一遍扫描统计量，第二遍利用统计量处理），说明是两遍处理模式。

---


### 5. TQue vs TBuf：数据流与计算缓冲的区别

| 用途 | 推荐方式 | 算子语义 |
|:---|:---|:---|
| GM → UB 数据搬入（读输入） | `TQue<VECIN, depth>` | GM → UB 搬运 |
| UB → GM 数据搬出（写输出） | `TQue<VECOUT, depth>` | UB → GM 搬运 |
| 中间计算临时 buffer | `TBuf<VECCALC>` | `T.alloc_ub`（无显式搬运） |

**TQue<VECIN/VECOUT> 标准用法**（数据流 buffer）：

```cpp
// 初始化（depth=0 表示 reference-form API，无自动双缓冲）
pipe->InitBuffer(inQueue_, 1, tileSize * sizeof(float));

// 读数据：AllocTensor → DataCopyPad → EnQue → DeQue → 使用 → FreeTensor
inQueue_.AllocTensor<float>(inLocal_);
DataCopyPad(inLocal_, gmSrc[offset], copyParams, padParms);
inQueue_.EnQue(inLocal_);
inQueue_.DeQue<float>(inLocal_);
// ... 使用 inLocal_ ...
inQueue_.FreeTensor(inLocal_);
```

**TBuf<VECCALC> 标准用法**（计算临时 buffer）：

```cpp
// 初始化
pipe->InitBuffer(absBuf_, tileSize * sizeof(float));

// 使用：Get → 直接操作，无需 EnQue/DeQue
LocalTensor<float> absLocal = absBuf_.Get<float>();
Abs(absLocal, inLocal_, tileSize);
```

> **常见错误**：将 TBuf 用于数据流（省略 TQue），会导致 MTE2/MTE3 与 V 管线之间的同步问题，需要额外 `PipeBarrier<PIPE_MTE2>()` 保证正确性。TQue 内置了 barrier 语义，代码更清晰安全。
