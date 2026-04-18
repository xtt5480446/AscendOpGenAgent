# DataCopyPad API 完整参考手册

> 基于 circular_pad 算子源码与 AscendC 官方文档整理，涵盖函数签名、所有使用场景、同步指令与常见陷阱。

---

## 一、核心常量与单位体系

```
1 DataBlock = 32 字节 = 256 bit
```

| 数据类型 | sizeof(T) | 每 DataBlock 元素数 |
|:---|:---|:---|
| float32 / int32 | 4 B | 8 |
| float16 / bf16 / int16 | 2 B | 16 |
| int8 | 1 B | 32 |

**两套 API 的单位差异（必须牢记）**：

| API | 结构体 | blockLen 单位 | 间隔单位（GM 端）| 间隔单位（UB 端）|
|:---|:---|:---|:---|:---|
| `DataCopy` 基础版 | `DataCopyParams` | **DataBlock** (32B) | DataBlock | DataBlock |
| `DataCopyPad` 增强版 | `DataCopyExtParams` | **字节** | **字节** | **DataBlock** |

> **circular_pad 只用 DataCopyPad，不用 DataCopy。** 原因是 padding 值需要精确到字节级别的偏移控制。

---

## 二、函数签名

### 2.1 核心结构体

```cpp
// DataCopyPad 专用参数（字节粒度）
struct DataCopyExtParams {
    uint16_t blockCount;   // 连续传输的数据块个数（行数），[1, 4095]
    uint32_t blockLen;     // 每个数据块长度，单位：字节，[1, 2097151]
    uint32_t srcStride;    // 源端相邻块间隔
    uint32_t dstStride;    // 目的端相邻块间隔
    uint32_t rsv;          // 保留，填 0
};

// Padding 控制参数（GM→UB 时有效）
struct DataCopyPadExtParams<T> {
    bool     isPad;        // 是否自定义填充值（circular_pad 中设为 false）
    uint16_t leftPadding;  // 左侧填充元素个数（字节≤32）
    uint16_t rightPadding; // 右侧填充元素个数（字节≤32）
    T        padValue;     // 填充值
};
```

### 2.2 三种函数原型

```cpp
// ========== 通路 1: GM → UB ==========
template<typename T>
__aicore__ inline void DataCopyPad(
    const LocalTensor<T>& dst,           // 目的: UB (VECIN)
    const GlobalTensor<T>& src,          // 源: GM
    const DataCopyExtParams& dataCopyParams,
    const DataCopyPadExtParams<T>& padParams
);

// ========== 通路 2: UB → GM ==========
template<typename T>
__aicore__ inline void DataCopyPad(
    const GlobalTensor<T>& dst,          // 目的: GM
    const LocalTensor<T>& src,           // 源: UB (VECOUT)
    const DataCopyExtParams& dataCopyParams
);

// ========== 通路 3: UB → UB（实际路径: VECIN/VECOUT → GM → TSCM）==========
// circular_pad 中未使用此通路
template<typename T>
__aicore__ inline void DataCopyPad(
    const LocalTensor<T>& dst,
    const LocalTensor<T>& src,
    const DataCopyExtParams& dataCopyParams,
    const Nd2NzParams& nd2nzParams
);
```

**stride 单位规则**：

| 操作数位置 | srcStride / dstStride 单位 |
|:---|:---|
| GM (Global Memory) | **字节** |
| UB (VECIN/VECOUT/VECCALC) | **DataBlock = 32 字节** |

---

## 三、circular_pad 中的 DataCopyPad 使用场景

circular_pad 是**纯数据搬运型 Vector 算子**，没有数学计算，核心就是 DataCopyPad 的各种组合。以 2D Small Shape 为例（input `3×4`, `left=1, right=2, top=1, bottom=1`, float32）。

### 前置参数

```
inputH=3, inputW=4      → inputLen = 12
outputH=5, outputW=7    → outputLen = 35
TSize = 4 (float)
leftAlign=8, rightAlign=8, inOutputWAlign=8   (32B 对齐)
workspaceLen = 3 × (8 + 8 + 8) = 72 元素 = 288 字节
```

---

### 场景 1: GM → UB（读原始数据，负 stride 实现循环回绕）

**代码位置**: `circular_pad.h:58` (PadLeftAndRightSmallShape)

```cpp
DataCopyExtParams paramsIn;
paramsIn.blockCount = inOutputH_;           // 3 行
paramsIn.blockLen   = inOutputW_ * TSize_;  // 4 × 4 = 16 字节
paramsIn.srcStride  = (-nLeft_ - nRight_) * TSize_;  // (-0 - 0) × 4 = 0
paramsIn.dstStride  = 0;

int64_t offsetIn = -nTop_ * inputW_ - nLeft_;  // 0

DataCopyPad(inLocal, xGM_[offsetIn], paramsIn, padParms);
```

**参数拆解**：

| 参数 | 值 | 含义 |
|:---|:---|:---|
| `blockCount` | 3 | 读 3 行 |
| `blockLen` | 16 | 每行读 16 字节（4 个 float）|
| `srcStride` | 0 | GM 上紧密连续 |
| `dstStride` | 0 | UB 上紧密连续 |

**数据流**：

```
xGM (GM, 紧密排列):
[A B C D][E F G H][I J K L]
              ↓
UB (inLocal):
[A B C D]  ← 第0行, 16字节, 框架补 16B dummy
[E F G H]  ← 第1行
[I J K L]  ← 第2行
```

> **注意**：`blockLen=16B < 32B (DataBlock)`，框架自动在每行末尾填充 dummy（值为该行首元素）。UB 实际每行占 32B。

---

### 场景 2: UB → GM（写 data 区到 workspace，dstStride 控制行间隔）

**代码位置**: `circular_pad.h:61` (PadLeftAndRightSmallShape)

```cpp
DataCopyExtParams paramsOut;
paramsOut.blockCount = inOutputH_;                          // 3 行
paramsOut.blockLen   = inOutputWAlign_ * TSize_;            // 8 × 4 = 32 字节
paramsOut.srcStride  = 0;                                    // UB 紧密连续
paramsOut.dstStride  = (leftAlign_ + rightAlign_) * TSize_; // 16 × 4 = 64 字节

int64_t offsetOut = leftAlign_;  // 8 (跳过 left pad)

DataCopyPad(workspaceGM_[offsetOut], inLocal, paramsOut);
```

**参数拆解**：

| 参数 | 值 | 含义 |
|:---|:---|:---|
| `blockCount` | 3 | 写 3 行 |
| `blockLen` | 32 | 每行写 32 字节（8 个 float，含 dummy）|
| `srcStride` | 0 | UB 上紧密连续（0 DataBlock）|
| `dstStride` | 64 | GM 上每行间隔 64 字节（16 元素）|

**数据流**：

```
UB (inLocal, 32B/行):
[A B C D A B C D]    ← 第0行, 32字节(含dummy)
[E F G H E F G H]    ← 第1行
[I J K L I J K L]    ← 第2行
         ↓
workspaceGM (每行 96B = 24float):
[________] [A B C D A B C D] [________]   ← 第0行, offset=8
[________] [E F G H E F G H] [________]   ← 第1行, +64B
[________] [I J K L I J K L] [________]   ← 第2行, +64B
   left       data(8元素)        right
```

> **核心技巧**：`dstStride=64B` 控制 workspace 每行总宽度（leftAlign + data + rightAlign = 24 元素 = 96B）。实际上 `blockLen=32B` 只写了 8 个元素到 data 区，其余 left/right 区由后续步骤填充。

---

### 场景 3: UB → GM（写 right pad 区，覆盖实现循环）

**代码位置**: `circular_pad.h:65` (PadLeftAndRightSmallShape)

```cpp
DataCopyExtParams paramsRight;
paramsRight.blockCount = inOutputH_;                            // 3 行
paramsRight.blockLen   = rightAlign_ * TSize_;                  // 8 × 4 = 32 字节
paramsRight.srcStride  = 0;
paramsRight.dstStride  = (leftAlign_ + inOutputWAlign_) * TSize_; // 16 × 4 = 64 字节

int64_t offsetRight = leftAlign_ + inOutputW_;  // 8 + 4 = 12

DataCopyPad(workspaceGM_[offsetRight], inLocal, paramsRight);
PipeBarrier<PIPE_MTE3>();  // 确保场景2的写入完成后，再执行场景3
```

**参数拆解**：

| 参数 | 值 | 含义 |
|:---|:---|:---|
| `blockLen` | 32 | 写 32 字节（8 个 float）到 right 区 |
| `offset` | 12 | 从 data 区第 4 个元素（offset=12）开始写 |
| `dstStride` | 64 | 每行间隔 64 字节（leftAlign + inOutputWAlign）|

**数据流**：

```
写入前 workspace 行0:
[________] [A B C D ~ ~ ~ ~] [________]

写入后（从 offset=12 写 32 字节）：
[________] [A B C D] [A B C D A B C D] [__]
                    ↑12
                    right pad 区被填入 [A B C D ...]
                    有效 right pad = [A, B]（前2个）
```

> **循环回绕原理**：right pad 需要输入最左边的元素 [A, B]。通过把 UB 中同样的数据写到 workspace 的 right 区（覆盖 data 区末尾），自然实现了循环。

**同步指令**：

```cpp
PipeBarrier<PIPE_MTE3>();
```

**原因**：场景 2（写 data 区）和场景 3（写 right 区）的目的地址有重叠（都写到 workspaceGM 的同一块区域）。`PipeBarrier<PIPE_MTE3>()` 确保 MTE3（数据搬出到 GM）管线完成前一条指令后，再执行下一条，防止**写后写（WAW）冲突**。

---

### 场景 4: GM → UB → GM（写 left pad，两阶段搬运）

**代码位置**: `circular_pad.h:84-87` (PadLeftAndRightSmallShape)

```cpp
// Stage 4a: workspaceGM → UB
DataCopyExtParams paramsIn;
paramsIn.blockCount = inOutputH_;                                    // 3 行
paramsIn.blockLen   = leftAlign_ * TSize_;                          // 8 × 4 = 32 字节
paramsIn.srcStride  = (rightAlign_ + inOutputWAlign_) * TSize_;     // 16 × 4 = 64 字节
paramsIn.dstStride  = 0;

int64_t offsetIn = inOutputW_;  // 4

DataCopyPad(inLocal, workspaceGM_[offsetIn], paramsIn, padParms);

// Stage 4b: UB → workspaceGM
DataCopyExtParams paramsOut;
paramsOut.blockCount = inOutputH_;                                  // 3 行
paramsOut.blockLen   = leftAlign_ * TSize_;                        // 32 字节
paramsOut.srcStride  = 0;
paramsOut.dstStride  = (rightAlign_ + inOutputWAlign_) * TSize_;   // 64 字节

int64_t offsetOut = 0;

DataCopyPad(workspaceGM_[offsetOut], inLocal, paramsOut);
```

**参数拆解**：

| 阶段 | 方向 | offset | blockLen | stride | 作用 |
|:---|:---|:---|:---|:---|:---|
| 4a | GM→UB | 4 | 32B | srcStride=64B | 从 data 区最右列开始读 |
| 4b | UB→GM | 0 | 32B | dstStride=64B | 写到 left pad 区开头 |

**数据流**：

```
读 workspace[4..11]（每行）:
[D A B C D A B C]  ← 从 data 区第4个元素（D）开始读8元素

写回 workspace[0..7]（每行）:
[D A B C D A B C]  ← 第0个元素 D 就是 left pad！
```

> **循环回绕原理**：left pad 需要输入最右边的元素 [D]。通过从 workspace data 区末尾（offset=4）读数据，再写到 left pad 区开头（offset=0），实现了循环。

---

### 场景 5: workspaceGM → UB → yGM（组装完整输出行 + top/bottom pad）

**代码位置**: `circular_pad.h:100-115` (CopyToOutSmallShapeOnePage)

```cpp
// Stage 5a: workspaceGM → UB
DataCopyExtParams copyParamsIn;
copyParamsIn.blockCount = inOutputH_;   // 3
copyParamsIn.blockLen   = outputW_ * TSize_;  // 7 × 4 = 28 字节
copyParamsIn.srcStride  = (inOutputWAlign_ + leftAlign_ + rightAlign_ - outputW_) * TSize_;
                          // (8 + 8 + 8 - 7) × 4 = 68 字节
copyParamsIn.dstStride  = 0;

int64_t offset = leftAlign_ - pLeft_;  // 8 - 1 = 7

DataCopyPad(inLocal, workspaceGM_[offset], copyParamsIn, padParms);

// Stage 5b: UB → yGM (主体)
DataCopyExtParams copyParamsOut;
copyParamsOut.blockCount = inOutputH_;   // 3
copyParamsOut.blockLen   = outputW_ * TSize_;  // 28 字节
copyParamsOut.srcStride  = 0;
copyParamsOut.dstStride  = 0;

int64_t outOffset = pTop_ * outputW_;  // 1 × 7 = 7

DataCopyPad(yGM_[outOffset], inLocal, copyParamsOut);

// Stage 5c: top pad (UB → yGM)
if (top_ > 0) {
    copyParamsOut.blockCount = top_;  // 1
    DataCopyPad(yGM_[pageIdxOut * outputLen_],
                inLocal[(inOutputH_ - top_) * outputWAlign_],
                copyParamsOut);
}

// Stage 5d: bottom pad (UB → yGM)
if (bottom_ > 0) {
    copyParamsOut.blockCount = bottom_;  // 1
    DataCopyPad(yGM_[pageIdxOut * outputLen_ + (outputH_ - bottom_) * outputW_],
                inLocal[0],
                copyParamsOut);
}
```

**参数拆解**：

| 阶段 | 方向 | offset | blockCount | blockLen | 作用 |
|:---|:---|:---|:---|:---|:---|
| 5a | GM→UB | 7 | 3 | 28B | 从 workspace 读完整输出行（含 left pad）|
| 5b | UB→GM | 7 | 3 | 28B | 写 yGM 主体（第1-3行）|
| 5c | UB→GM | 0 | 1 | 28B | 写 top pad（yGM 第0行）|
| 5d | UB→GM | 28 | 1 | 28B | 写 bottom pad（yGM 第4行）|

**数据流**：

```
UB (从 workspace 读入):
[C D A B C D A B]  ← 行0, 从 offset=7 读 (left pad + data + right)
[G H E F G H E F]  ← 行1
[L I J K L I J K]  ← 行2

         ↓

yGM (output 5×7):
[L I J K L I J]  ← top=1, 来自 UB 行2 (倒数行)
[D A B C D A B]  ← 行0 (主体)
[H E F G H E F]  ← 行1
[L I J K L I J]  ← 行2
[D A B C D A B]  ← bottom=1, 来自 UB 行0 (首行)
```

> **top/bottom 循环原理**：top pad 复制输入的倒数行，bottom pad 复制输入的首行。通过 `inLocal[offset]` 精确指定 UB 中的源行，实现循环填充。

---

### 场景 6: GM → UB → GM（Big Shape 的 CopyGmToGm，UB 中转）

**代码位置**: `circular_pad.h:346-375`

```cpp
__aicore__ inline void CopyGmToGm(
    int64_t pages, int64_t taskNum, int64_t offsetIn, int64_t offsetOut, int64_t stride)
{
    int64_t loop = (outputLen_ * pages * TSize_) / (UB_SIZE / BUFFER_NUM);
    uint32_t tail = (outputLen_ * pages * TSize_) % (UB_SIZE / BUFFER_NUM);

    for (int64_t i = 0; i < taskNum; i++) {
        DataCopyExtParams paramsFront = {1, UB_SIZE / BUFFER_NUM, 0, 0, 0};

        for (int64_t j = 0; j < loop; j++) {
            auto inLocal = queBind_.AllocTensor<T>();

            // Step A: yGM → UB (读源数据)
            DataCopyPad(inLocal, yGM_[offsetIn], paramsFront, padParms);
            queBind_.EnQue(inLocal);

            inLocal = queBind_.DeQue<T>();

            // Step B: UB → yGM (写到目的位置)
            DataCopyPad(yGM_[offsetOut], inLocal, paramsFront);
            queBind_.FreeTensor(inLocal);

            offsetIn  += (UB_SIZE / BUFFER_NUM / TSize_);
            offsetOut += (UB_SIZE / BUFFER_NUM / TSize_);
        }

        // 处理尾部不足一整块的数据
        if (tail > 0) {
            paramsFront.blockLen = tail;
            auto inLocal = queBind_.AllocTensor<T>();
            DataCopyPad(inLocal, yGM_[offsetIn], paramsFront, padParms);
            queBind_.EnQue(inLocal);
            inLocal = queBind_.DeQue<T>();
            DataCopyPad(yGM_[offsetOut], inLocal, paramsFront);
            queBind_.FreeTensor(inLocal);
        }

        offsetIn  += (stride - loop * (UB_SIZE / BUFFER_NUM / TSize_));
        offsetOut += (stride - loop * (UB_SIZE / BUFFER_NUM / TSize_));
    }
}
```

**用途**：3D CircularPad 的 front/back 填充。由于 front/back 是在 2D 平面填充完成后进行的，需要把 yGM 中已填充好的 2D 平面复制到 front/back 区域。

**数据流**：

```
yGM (源位置, 已填充的 2D 平面)
    ↓  Step A: DataCopyPad(yGM → UB)
UB (inLocal, 中转 buffer)
    ↓  Step B: DataCopyPad(UB → yGM)
yGM (目的位置, front/back 区域)
```

> **为什么用 UB 中转**：GM 不支持直接的 GM→GM 搬运，必须借助 UB 做中转。数据从源 GM 地址读到 UB，再从 UB 写到目的 GM 地址。

---

## 四、同步指令详解

### 4.1 PipeBarrier<PIPE_MTE3>()

**代码位置**: `circular_pad.h:64`

```cpp
// 场景2写 data 区 → 场景3写 right 区
DataCopyPad(workspaceGM_[offsetOut], inLocal, paramsOut);     // 写 data
PipeBarrier<PIPE_MTE3>();                                      // 同步！
DataCopyPad(workspaceGM_[offsetRight], inLocal, paramsRight);  // 写 right
```

**原因**：两条 DataCopyPad 的目的地址都是 workspaceGM，且**有重叠**（data 区和 right 区相邻甚至部分重叠）。如果不加同步，MTE3 管线可能**并行执行**两条搬运指令，导致后写入的数据覆盖先写入的数据，产生竞争条件。

```
workspaceGM (无同步时的风险):
时间0: 写 data 区 [8..15] = [A,B,C,D,A,B,C,D]  ← 指令A开始
时间1: 写 right 区 [12..19] = [A,B,C,D,A,B,C,D] ← 指令B开始(与A并行)
结果: [8..11]=[A,B,C,D], [12..15]=[?,?,?,?]       ← 数据混乱！

workspaceGM (有 PipeBarrier<PIPE_MTE3>):
时间0: 写 data 区完成
时间1: PipeBarrier 等待 MTE3 完成
时间2: 写 right 区开始
结果: 确定性的写入顺序
```

### 4.2 MTE3ToMTE2Sync()（自定义硬件事务同步）

**代码位置**: `circular_pad_common.h:101-106`

```cpp
__aicore__ inline void MTE3ToMTE2Sync()
{
    event_t eventId3To2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventId3To2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventId3To2);
}
```

**使用位置**: `circular_pad_2d.h:49` (ProcessSmallShape)

```cpp
void ProcessSmallShape() {
    // Step 1-4: 各种 DataCopyPad 写 workspace
    PadLeftAndRightSmallShape();   // 涉及 MTE3 (UB→GM)
    MTE3ToMTE2Sync();              // 等待 MTE3 完成！
    CopyToOutSmallShape();         // 涉及 MTE2 (GM→UB)
}
```

**原因**：`PadLeftAndRightSmallShape` 中的 DataCopyPad 大部分是 **UB→GM 方向**（走 MTE3 管线），而 `CopyToOutSmallShape` 中的 DataCopyPad 是 **GM→UB 方向**（走 MTE2 管线）。必须确保 MTE3 的写入完成后，MTE2 才能读取，否则读到的是旧数据。

```
PadLeftAndRight:  UB → workspaceGM   (MTE3: 搬出)
MTE3ToMTE2Sync:   等待 MTE3 完成
CopyToOut:        workspaceGM → UB   (MTE2: 搬入)
```

### 4.3 EnQue / DeQue（队列隐式同步）

**代码位置**: `circular_pad.h:57-68`

```cpp
auto inLocal = queBind_.AllocTensor<T>();   // 从 VECIN 申请 UB
DataCopyPad(inLocal, xGM_[offset], params);  // GM → UB(VECIN)
queBind_.EnQue(inLocal);                     // 标记为"待搬入完成"

// ... 其他代码 ...

inLocal = queBind_.DeQue<T>();               // 从 VECOUT 取出（等搬入完成）
DataCopyPad(yGM_[offset], inLocal, params);  // UB(VECOUT) → GM
queBind_.FreeTensor(inLocal);
```

**同步机制**：

- `EnQue`：将 UB buffer 标记为 VECIN 队列的"生产者"状态，内部自动插入 **MTE2 同步屏障**，确保搬入完成后后续操作才能继续。
- `DeQue`：从 VECOUT 队列"消费"buffer，如果 MTE2 尚未完成，会**阻塞等待**。

**为什么不用显式 PipeBarrier**：TQue/TQueBind 内部通过队列状态机管理同步，比手动 `PipeBarrier` 更安全。但 `TBuf`（非队列 buffer）从 GM DataCopy 后**必须手动插入** `PipeBarrier<PIPE_MTE2>()`。

### 4.4 TBuf 场景的 PipeBarrier<PIPE_MTE2>()（circular_pad 未使用，但需了解）

```cpp
// 假设用 TBuf 代替 TQue 做 GM→UB 搬运
TBuf<QuePosition::VECCALC> calcBuf;
pipe.InitBuffer(calcBuf, 256 * sizeof(float));

LocalTensor<float> tmpLocal = calcBuf.Get<float>();
DataCopyExtParams params = {1, 64 * sizeof(float), 0, 0, 0};
DataCopyPad(tmpLocal, xGM_, params, padParms);  // GM → UB (TBuf，必须使用 DataCopyPad)
PipeBarrier<PIPE_MTE2>();            // 必须手动同步！TBuf 无队列机制

// 现在 tmpLocal 数据已就绪，可以安全使用
T.tile.add(tmpLocal, tmpLocal, scalar);
```

**原因**：`TBuf` 没有 `EnQue/DeQue` 的队列同步机制，`DataCopyPad`（走 MTE2 管线）完成后数据可能尚未真正写入 UB。不加 `PipeBarrier` 就使用，可能读到未就绪的数据，导致**结果随机错误**。

---

## 五、circular_pad 中的同步全景图

```
ProcessSmallShape():
├─ PadLeftAndRightSmallShape()
│   ├─ DataCopyPad(xGM → UB)         [MTE2: 搬入]
│   ├─ DataCopyPad(UB → workspaceGM) [MTE3: 搬出] ← 写 data
│   ├─ PipeBarrier<PIPE_MTE3>()      [同步! WAW 竞争]
│   ├─ DataCopyPad(UB → workspaceGM) [MTE3: 搬出] ← 写 right
│   ├─ DataCopyPad(workspaceGM → UB) [MTE2: 搬入]
│   └─ DataCopyPad(UB → workspaceGM) [MTE3: 搬出] ← 写 left
│
├─ MTE3ToMTE2Sync()                  [同步! MTE3→MTE2 方向切换]
│
└─ CopyToOutSmallShape()
    ├─ DataCopyPad(workspaceGM → UB) [MTE2: 搬入] ← 读完整行
    ├─ DataCopyPad(UB → yGM)         [MTE3: 搬出] ← 写主体
    ├─ DataCopyPad(UB → yGM)         [MTE3: 搬出] ← 写 top
    └─ DataCopyPad(UB → yGM)         [MTE3: 搬出] ← 写 bottom
```

---

## 六、常见陷阱

| 陷阱 | 表现 | 解决 |
|:---|:---|:---|
| `blockLen` 单位搞混 | DataCopy 用 DataBlock，DataCopyPad 用字节 | 牢记：DataCopyPad 的 blockLen 永远是**字节** |
| stride 单位搞混 | GM 端是字节，UB 端是 DataBlock | DataCopyPad 的 srcStride/dstStride：**GM=字节，UB=DataBlock** |
| 省略 PipeBarrier<PIPE_MTE3> | WAW 竞争，数据随机错误 | 目的地址重叠的连续 DataCopyPad 之间必须加 |
| 省略 MTE3ToMTE2Sync | MTE3 写未完成后 MTE2 就读，读到旧数据 | UB→GM 和 GM→UB 交替时加 |
| TBuf 后省略 PipeBarrier<PIPE_MTE2> | 读到未就绪数据 | TBuf + DataCopy 后必须手动加 |
| `count * sizeof(T)` 不对齐 | 静默截断，丢数据 | 确保是 32 的倍数 |
| 负 stride 理解错误 | 以为是从后往前读 | 负 stride 是**回退**，不是反向读取 |
