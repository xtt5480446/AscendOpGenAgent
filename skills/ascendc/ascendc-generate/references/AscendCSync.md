## 公共工具：跨核同步与 WorkspaceQueue 详细参考

本文档包含 WorkspaceQueue 环形缓冲区、批量同步模式、CrossCore flag 的完整实现细节与代码示例。
概览与判断规则见 `@references/AscendCDesign.md`。

---

## 第三章：公共工具

### 1. 通过 Workspace Queue 实现跨核同步

使用基于 workspace GM 的环形缓冲区进行 AIC → AIV 数据传输，配合 `CrossCoreSetFlag/WaitFlag` 同步：

```cpp
template <typename T, uint32_t DEPTH>
class WorkspaceQueue {
public:
    // AIV 初始化：将所有槽位标记为空闲（生产者可以写入）
    __aicore__ inline void InitFreeSlots() {
        for (uint32_t i = 0; i < DEPTH; ++i) {
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(vecNotifyCubeId_);
        }
    }

    // AIC（生产者）：等待空闲槽位，然后通过 Fixpipe 写入
    __aicore__ inline GlobalTensor<T> ProducerAcquire() {
        AscendC::CrossCoreWaitFlag<0x2>(vecNotifyCubeId_);  // 等待"槽位空闲"
        return workspace_[head_ % DEPTH * slotSize_];
    }
    __aicore__ inline void ProducerRelease() {
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeNotifyVecId_);  // 发送"数据就绪"信号
        head_++;
    }

    // AIV（消费者）：等待数据就绪，然后通过 MTE2 读取
    __aicore__ inline GlobalTensor<T> ConsumerAcquire() {
        AscendC::CrossCoreWaitFlag<0x2>(cubeNotifyVecId_);  // 等待"数据就绪"
        return workspace_[tail_ % DEPTH * slotSize_];
    }
    __aicore__ inline void ConsumerRelease() {
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(vecNotifyCubeId_);  // 发送"槽位空闲"信号
        tail_++;
    }
};
```

**关键同步流程：**
```
初始化阶段：
  AIV: InitFreeSlots() → 设置 DEPTH 个"槽位空闲"标志

循环阶段（每个 tile）：
  AIC: ProducerAcquire() → 等待"槽位空闲"
  AIC: Fixpipe(slot, ...) → 写入 workspace GM
  AIC: ProducerRelease() → 设置"数据就绪"

  AIV: ConsumerAcquire() → 等待"数据就绪"
  AIV: DataCopyPad(local, slot, params) → 从 workspace GM 读取（必须使用 DataCopyPad）
  AIV: Process() → 计算
  AIV: ConsumerRelease() → 设置"槽位空闲"
```

#### A. 批量同步模式（Bulk Sync，无环形缓冲）

当 Cube 需要**先完成所有 tile 写入 workspace 后**，Vector 才能开始处理时（如需要全局统计量的两遍处理），不使用 WorkspaceQueue 环形缓冲，而是使用**单次 CrossCoreSetFlag/WaitFlag 批量同步**：

**典型场景**：
```python
# 算子语义：Cube 完成所有 tile 后一次性通知 Vector
// Cube 侧计算:
    for by in range(n_num):
        // ... matmul all tiles, write to workspace ...
        // 将 L0C 结果写入 workspace
    CrossCoreSetFlag<0x2, PIPE_FIX>(0x8)  // 所有 tile 完成后才发一次信号

// Vector 侧计算:
    CrossCoreWaitFlag<0x2>(0x8)  // 等待所有 tile 就绪
    # Pass 1: 扫描所有 tile 计算统计量
    for by in range(n_num):
        // 从 workspace 读取到 UB
        # ... 累积 per-row absmax ...
    # Pass 2: 利用统计量处理
    for by in range(n_num):
        // 从 workspace 读取到 UB
        # ... quantize ...
```

**AscendC 翻译**：
```cpp
// Cube 侧：循环所有 n_tile，全部写完后一次性信号
if ASCEND_IS_AIC {
    for (int by = 0; by < nTiles; by++) {
        auto wsBlock = wsGM_[bx * baseM * N + by * baseN];
        mm_.ComputeBlock(aBlock, bBlock, wsBlock, H_K, N);  // dstStride=N
    }
    CrossCoreSetFlag<0x2, PIPE_FIX>(CUBE_NOTIFY_VECTOR_ID);  // 只发一次信号
}

// Vector 侧：等待一次信号，然后两遍处理
if ASCEND_IS_AIV {
    CrossCoreWaitFlag<0x2>(CUBE_NOTIFY_VECTOR_ID);  // 只等一次

    // Pass 1: 全局扫描
    for (int by = 0; by < nTiles; by++) { /* ... accumulate stats ... */ }
    // Pass 2: 利用统计量处理
    for (int by = 0; by < nTiles; by++) { /* ... quantize ... */ }
}
```

**与 WorkspaceQueue 环形缓冲的对比**：

| 特性 | WorkspaceQueue（逐 tile 同步） | 批量同步（Bulk Sync） |
|:---|:---|:---|
| **信号次数** | 每个 tile 一次 Acquire/Release | Cube 全部完成后一次 |
| **Workspace 大小** | DEPTH × baseM × baseN × sizeof(T) | M × N × sizeof(T)（全输出） |
| **Vector 启动时机** | Cube 写完一个 tile 即可开始 | 必须等 Cube 全部写完 |
| **适用场景** | 逐 tile 独立处理（如 LeakyReLU、Scale） | 需要全局统计量（如 ReduceMax + 量化） |
| **同步特征** | 每 tile 发一次同步信号 | 全部 tile 完成后发一次信号 |
| **核分配** | BlockScheduler 分配 mBlocks×nBlocks | 每核一个 m_block，Cube 内循环 n_tiles |

> **判断规则**：如果 AIC 完成全部 tile 后才发一次同步信号，说明是批量同步；如果每处理完一个 tile 就发一次信号，则是逐 tile 同步（WorkspaceQueue）。


### 2. CrossCore 同步：单 flag 广播模式

#### 规则：单 flag 广播给所有 AIV 子块

AscendC 中使用 `CrossCoreSetFlag<0x2, PIPE_FIX>(flagId)` / `CrossCoreWaitFlag<0x2>(flagId)` 实现**单个 flag ID** 对所有 AIV 子块广播：

```cpp
// ✅ 正确：单 flag 广播给所有 AIV 子块（KERNEL_TYPE_MIX_AIC_1_2）
#define CUBE_NOTIFY_VECTOR_ID 0x8

if ASCEND_IS_AIC {
    // AIC 完成所有 tile 后发一次信号
    CrossCoreSetFlag<0x2, PIPE_FIX>(CUBE_NOTIFY_VECTOR_ID);
}

if ASCEND_IS_AIV {
    // 所有 AIV 子块（vid=0, vid=1）都等待同一个 flag
    CrossCoreWaitFlag<0x2>(CUBE_NOTIFY_VECTOR_ID);
    // vid 由 GetSubBlockIdx() 区分各自的数据偏移
    int rowOffset = AscendC::GetSubBlockIdx() * subTileM;
}
```

**❌ 错误写法**（逐 AIV 子块发送不同 flag）：

```cpp
// 错误：AIC 发送 0x8 给 vid=0，发送 0x9 给 vid=1
for (int i = 0; i < VEC_NUM; i++) {
    CrossCoreSetFlag<0x2, PIPE_FIX>(0x8 + i);
}
// AIV: CrossCoreWaitFlag<0x2>(0x8 + vid_);
```

> **为什么会出错**：per-subblock flag 是逐 tile 同步（ring buffer）模式的写法，适用于 `matmul_leakyrelu` 中的 WorkspaceQueue。批量同步场景（two-pass 量化）只需一次信号，用单 flag 广播即可。

**识别口诀**：
- AIC 设置同步信号：`CrossCoreSetFlag<0x2, PIPE_FIX>(0x8 + idx)`，只调用一次
- AIV 等待同步信号：`CrossCoreWaitFlag<0x2>(0x8 + idx)`，所有 AIV 子块共享同一 flag

