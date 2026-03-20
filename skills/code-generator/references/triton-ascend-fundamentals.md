# Triton Ascend 基础知识参考手册

本文档汇集 Triton Ascend 编程的基础知识，包括 API 参考、编程模式、优化策略和调试技巧。

---

## 1. 编程基础

### 1.1 核心概念

Triton Ascend 采用与 Triton GPU 类似的执行模型：

- **Kernel（内核）**：用 `@triton.jit` 装饰的函数，在 NPU 上并行执行
- **Grid（网格）**：启动的并行程序总数，由 `kernel[grid](args)` 指定
- **Block（块）**：每个程序处理的数据块，由 `tl.program_id(0)` 区分

### 1.2 标准 Kernel 结构（五步模式）

```python
@triton.jit
def triton_better_kernel(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    # Step 1: 计算当前核的起始偏移（核间切分）
    xoffset = tl.program_id(0) * XBLOCK

    # Step 2: 核内循环（核内切分）
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        # Step 3: 构造数据索引
        x_index = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]

        # Step 4: 边界处理（mask）
        xmask = x_index < xnumel

        # Step 5: 加载 → 计算 → 存储
        x = tl.load(in_ptr0 + x_index, xmask)
        ret = x * 0.5 * (1.0 + tl.erf(x / tl.sqrt(2.0)))
        tl.store(out_ptr0 + x_index, ret, xmask)
```

### 1.3 边界处理（Mask）

使用 mask 防止越界访问，确保只处理合法范围内的数据：

```python
xmask = x_index < xnumel
x = tl.load(in_ptr0 + x_index, mask=xmask, other=0.0)
tl.store(out_ptr0 + x_index, ret, mask=xmask)
```

### 1.4 Autotune 使用

Triton 支持 `autotune` 自动调优，但 **Ascend NPU 不支持 `num_warps`、`num_ctas`、`num_stages` 等 CUDA 专用调优参数**。在 Ascend 上主要调优：

- `BLOCK_SIZE` / `XBLOCK` / `XBLOCK_SUB` 等分块参数
- `grid` 维度（核数）

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ],
    key=['n_elements'],
)
@triton.jit
def kernel(...):
    ...
```

---

## 2. API 参考

### 2.1 装饰器与启动

| API | 说明 |
|-----|------|
| `@triton.jit` | 将 Python 函数编译为 Triton 内核 |
| `tl.program_id(axis)` | 当前程序在 grid 指定维度上的 ID |
| `tl.num_programs(axis)` | grid 在指定维度上的程序总数 |
| `triton.cdiv(x, y)` | 向上取整除法：`(x + y - 1) // y` |

### 2.2 内存访问

| API | 说明 |
|-----|------|
| `tl.load(ptr, mask=None, other=0.0)` | 从全局内存加载数据 |
| `tl.store(ptr, value, mask=None)` | 将数据写入全局内存 |
| `tl.make_block_ptr(base, shape, strides, offsets, block_shape, order)` | 创建块指针，用于 2D/高维块访问 |
| `tl.advance(block_ptr, dim, delta)` | 沿指定维度推进块指针 |

**tl.make_block_ptr 示例**：

```python
p_q = tl.make_block_ptr(
    base=q + (bos * HQ + i_hq) * K,
    shape=(T, K),
    strides=(HQ*K, 1),
    offsets=(i_t * BT, 0),
    block_shape=(BT, BK),
    order=(1, 0)
)
val = tl.load(p_q, boundary_check=(0, 1), padding="zero")
```

### 2.3 张量构造与索引

| API | 说明 |
|-----|------|
| `tl.arange(start, end)` | 创建一维索引张量 |
| `tl.zeros(shape, dtype)` | 创建全零张量 |
| `tl.full(shape, value, dtype)` | 创建填充指定值的张量 |
| `tl.cast(tensor, dtype)` | 类型转换 |

### 2.4 数学与归约

| API | 说明 |
|-----|------|
| `tl.cdiv(x, y)` | 向上取整除法 |
| `tl.dot(a, b)` | 矩阵乘法 / 点积 |
| `tl.sum(tensor, axis)` | 沿轴求和 |
| `tl.max(tensor, axis)` | 沿轴求最大值 |
| `tl.where(condition, x, y)` | 条件选择 |

### 2.5 扫描与原子操作

| API | 说明 |
|-----|------|
| `tl.cumsum(tensor, axis)` | 沿轴累积和 |
| `tl.cumprod(tensor, axis)` | 沿轴累积积 |
| `tl.atomic_add(ptr, value)` | 原子加 |
| `tl.atomic_max(ptr, value)` | 原子最大值 |

### 2.6 编译时常量

| API | 说明 |
|-----|------|
| `tl.constexpr` | 标记参数为编译时常量，用于 kernel 参数 |

```python
def kernel(ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    ...
```

### 2.7 布局与分块

| API | 说明 |
|-----|------|
| `tl.swizzle2d(i, j, size_i, size_j, size_g)` | 2D 分块时的 swizzle 布局优化 |

---

## 3. Grid 配置策略

### 3.1 Grid 维度限制

- **Grid 各维度最大值**：65535
- **BLOCK_SIZE 限制**：单块元素数 < 65536

### 3.2 大 Shape 处理

当 `grid = (triton.cdiv(n, BLOCK_SIZE),)` 超过 65535 时，需采用以下策略之一。

#### 方法一：交错循环（Interleaved Loop）

每个核心循环处理多个块：

```python
@triton.jit
def kernel(ptr, n_elements, num_cores: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)

    for block_idx in range(pid, num_blocks, num_cores):
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        data = tl.load(ptr + offsets, mask=mask, other=0.0)
        # compute...
        tl.store(ptr + offsets, result, mask=mask)
```

#### 方法二：连续块法（Continuous Block Method）

将一维 grid 映射为多维，或使用 `(grid_x, grid_y)` 形式：

```python
MAX_GRID = 65535
grid_x = min(triton.cdiv(n, BLOCK_SIZE), MAX_GRID)
grid_y = triton.cdiv(triton.cdiv(n, BLOCK_SIZE), MAX_GRID)
grid = (grid_x, grid_y)

# kernel 内: pid = tl.program_id(0) * grid_y + tl.program_id(1)
```

### 3.3 动态核心数获取

根据算子类型选择核心类型：

| 算子类型 | 核心类型 | 获取方式 |
|----------|----------|----------|
| Element-wise、Reduce | **VEC_CORE_NUM**（向量核心） | `torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)` |
| MatMul | **CUBE_CORE_NUM**（矩阵核心） | `torch_npu.npu.npu_config.get_device_limit(0).get("cube_core_num", 20)` |

**关键规则**：必须在 `__init__` 中获取核心数，**不能在 `forward` 中获取**，否则每次前向都会查询，影响性能。

```python
import torch_npu

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        try:
            self.VEC_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)
            self.CUBE_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("cube_core_num", 20)
        except Exception:
            self.VEC_CORE_NUM = 40
            self.CUBE_CORE_NUM = 20

    def forward(self, x):
        grid = (self.VEC_CORE_NUM,)
        kernel[grid](...)
```

---

## 4. 内存访问优化

### 4.1 Block Size 选择

| 算子类型 | 推荐 BLOCK_SIZE | 说明 |
|----------|-----------------|------|
| Element-wise | 1024–2048 | 平衡并行度与资源占用 |
| Reduce | 256 / 128 | 归约需要更多寄存器 |
| MatMul | M=128, K=256, N=256（或 128/128/32） | 满足 512B 行宽对齐 |

### 4.2 2D 数据访问：tl.make_block_ptr

对于 2D 张量，使用 `tl.make_block_ptr` 可简化索引并提升访存效率：

```python
# 2D 块访问示例
ptr = tl.make_block_ptr(
    base=a_ptr,
    shape=(M, K),
    strides=(K, 1),
    offsets=(block_m * BLOCK_M, 0),
    block_shape=(BLOCK_M, BLOCK_K),
    order=(1, 0)
)
a = tl.load(ptr, boundary_check=(0, 1), padding="zero")
```

### 4.3 1D 连续内存优化

**优先使用 `.contiguous()` + 1D 访问**，避免 stride 访问带来的额外开销：

```python
class ModelNew(torch.nn.Module):
    def forward(self, input_tensor):
        if not input_tensor.is_contiguous():
            input_tensor = input_tensor.contiguous()

        output_tensor = torch.empty_like(input_tensor)
        n_elements = input_tensor.numel()
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        elementwise_kernel[grid](input_tensor, output_tensor, n_elements, BLOCK_SIZE)
        return output_tensor

@triton.jit
def elementwise_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    data = tl.load(input_ptr + offsets, mask=mask)
    result = compute(data)
    tl.store(output_ptr + offsets, result, mask=mask)
```

### 4.4 对齐规则

- **256B**：基础对齐单位
- **512B**：MatMul 等带宽敏感算子推荐（256 个 fp16/bf16 元素）

---

## 5. 性能优化

### 5.1 1D Grid 优先

优先使用一维 grid `(n,)`，便于与 Ascend 核心映射一致。

### 5.2 核内循环优化

编译器会对核内 `for` 循环做自动流水线优化，实现存算并行：

```python
@triton.jit
def optimized_elementwise(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    for i in range(4):
        offsets = (pid * 4 + i) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        data = tl.load(input_ptr + offsets, mask=mask)
        result = compute(data)
        tl.store(output_ptr + offsets, result, mask=mask)
```

### 5.3 算子拆分策略

复杂算子可拆分为多个 kernel，减少单 kernel 复杂度，便于调优。

### 5.4 NPU 核心配置

- Element-wise / Reduce：使用 VEC 核心，`grid=(VEC_CORE_NUM,)`
- MatMul：使用 CUBE 核心，`grid=(CUBE_CORE_NUM,)`

### 5.5 数据传输单位

Ascend NPU 数据传输以 **256B** 为单位，设计 block 大小时需考虑对齐。

### 5.6 数值稳定性

- **Softmax 防溢出**：先减最大值再 `exp`
- **sqrt 非负检查**：`tl.maximum(variance, 0.0)` 再开方
- **累加用 float32**：reduce 累加建议用 `tl.float32` 避免精度损失

```python
# 错误：直接 exp 可能溢出
scores = tl.exp(x)

# 正确：减去最大值
max_val = tl.max(x, axis=0)
scores = tl.exp(x - max_val)

# 方差开方前确保非负
variance = tl.maximum(variance, 0.0)
rstd = 1.0 / tl.sqrt(variance + eps)
```

### 5.7 API 限制（Ascend 特有）

| 限制 | 说明 |
|------|------|
| 无 return/break/continue | 使用变量和条件控制流程 |
| 无 lambda | 使用普通函数或内联逻辑 |
| 无 while 循环 | 用 `for` + `if` 替代 |
| 无 Python 切片 | 用 `tl.get_element` / `tl.extract_slice` / `tl.insert_slice` |
| constexpr 仅用于 kernel 参数 | 编译时常量通过参数传入 |
| 避免 tl.where 做内存偏移 | 易导致低效，改用 boundary_check 或 mask |
| 标量类型转换 | 用 `.to(tl.float32)` 等，不用 `tl.float16()` |

```python
# 不推荐：tl.where 用于偏移
# ptr = base + tl.where(mask, idx, 0)

# 推荐：mask 用于 load/store
data = tl.load(ptr + idx, mask=mask, other=0.0)
```

---

## 6. 调试清单

### 6.1 内存访问清单

- [ ] 所有 load/store 是否有正确的 mask 或 boundary_check
- [ ] 指针偏移是否可能越界
- [ ] 是否使用 `.contiguous()` 保证连续访问
- [ ] Block 大小是否满足 256B/512B 对齐

### 6.2 控制流清单

- [ ] 是否使用了 `return` / `break` / `continue`（Ascend 不支持）
- [ ] 是否使用了 `while`（需改为 `for` + `if`）
- [ ] 是否使用了 lambda
- [ ] 是否使用了 Python 切片（需改为 tl 切片 API）

### 6.3 Grid/Block 清单

- [ ] Grid 各维度是否 ≤ 65535
- [ ] BLOCK_SIZE 是否 < 65536
- [ ] 大 shape 是否采用交错循环或连续块法
- [ ] 核心数是否在 `__init__` 中获取

### 6.4 并发清单

- [ ] 多 kernel 调用时中间张量是否正确传递
- [ ] 原子操作是否正确使用（如有）

### 6.5 切片清单

- [ ] 多维索引是否正确计算
- [ ] 是否使用 `tl.extract_slice` / `tl.insert_slice` 替代 Python 切片

### 6.6 性能清单

- [ ] 是否优先 1D grid
- [ ] Element-wise 是否使用 VEC 核心
- [ ] MatMul 是否使用 CUBE 核心
- [ ] 是否利用核内循环实现存算并行

### 6.7 常见错误速查表

| 错误现象 | 可能原因 | 排查方向 |
|----------|----------|----------|
| 编译失败：unsupported op | 使用了 Ascend 不支持的 API | 检查 return/break/while/lambda/切片 |
| 数值不一致 | 精度、reduce 维度、边界处理 | 检查 mask、axis、float32 累加 |
| 越界 / 崩溃 | 指针或索引计算错误 | 检查 offset 公式、mask 覆盖 |
| Grid 超限 | grid 维度 > 65535 | 使用交错循环或拆分 grid |
| 性能差 | 访存不连续、block 过小 | 使用 contiguous、调整 BLOCK_SIZE |
| tl.where 相关错误 | Ascend 上 tl.where 用于偏移易出问题 | 改用 mask + other 或 boundary_check |

---

## 附录：完整代码示例

### A.1 Vector Add（向量加法）

```python
@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b

    tl.store(c_ptr + offsets, c, mask=mask)

class ModelNew(torch.nn.Module):
    def forward(self, a, b):
        c = torch.empty_like(a)
        n_elements = a.numel()
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        vector_add_kernel[grid](a, b, c, n_elements, BLOCK_SIZE=1024)
        return c
```

### A.2 Reduce 算子（块内归约 + 原子操作）

```python
@triton.jit
def reduction_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    block_sum = tl.sum(data, axis=0)
    tl.atomic_add(output_ptr, block_sum)
```

### A.3 MatMul（固定核心数启动）

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    num_cores: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    NUM_BLOCKS_M = triton.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = triton.cdiv(N, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N

    for block_idx in range(pid, NUM_BLOCKS, num_cores):
        block_m = block_idx // NUM_BLOCKS_N
        block_n = block_idx % NUM_BLOCKS_N

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, K, BLOCK_K):
            a_offset = (block_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] * K + \
                       (k + tl.arange(0, BLOCK_K))[None, :]
            a_mask = (block_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] < M
            a = tl.load(a_ptr + a_offset, mask=a_mask, other=0.0)

            b_offset = (k + tl.arange(0, BLOCK_K))[:, None] * N + \
                       (block_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
            b_mask = (block_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :] < N
            b = tl.load(b_ptr + b_offset, mask=b_mask, other=0.0)

            accumulator += tl.dot(a, b)

        c_offset = (block_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] * N + \
                   (block_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
        c_mask = ((block_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] < M) & \
                 ((block_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :] < N)
        tl.store(c_ptr + c_offset, accumulator, mask=c_mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        M, K = a.shape
        K2, N = b.shape
        assert K == K2
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)

        num_cores = 20
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 256

        matmul_kernel[(num_cores,)](a, b, c, M, N, K, num_cores,
                                    BLOCK_M, BLOCK_N, BLOCK_K)
        return c
```

### A.4 LayerNorm 核心逻辑（归一化）

```python
# 1. 计算均值
mean = tl.sum(x, axis=0) / n_cols

# 2. 计算方差
x_centered = x - mean
variance = tl.sum(x_centered * x_centered, axis=0) / n_cols

# 3. 归一化
variance = tl.maximum(variance, 0.0)
rstd = 1.0 / tl.sqrt(variance + eps)
normalized = x_centered * rstd

# 4. 应用 weight 和 bias
output = normalized * weight + bias
```

### A.5 Double Kernel（多内核调用）

```python
class ModelNew(torch.nn.Module):
    def forward(self, x):
        intermediate = torch.empty_like(x)
        kernel1[grid](x, intermediate, ...)

        output = torch.empty_like(x)
        kernel2[grid](intermediate, output, ...)

        return output
```

### A.6 完整 GELU 示例

```python
import triton
import triton.language as tl
import torch

@triton.jit
def triton_easy_kernel(in_ptr0, out_ptr0, NUMEL: tl.constexpr):
    idx_block = tl.arange(0, NUMEL)
    x = tl.load(in_ptr0 + idx_block)
    ret = x * 0.5 * (1.0 + tl.erf(x / tl.sqrt(2.0)))
    tl.store(out_ptr0 + idx_block, ret)

@triton.jit
def triton_better_kernel(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        x_index = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = x_index < xnumel
        x = tl.load(in_ptr0 + x_index, xmask)
        ret = x * 0.5 * (1.0 + tl.erf(x / tl.sqrt(2.0)))
        tl.store(out_ptr0 + x_index, ret, xmask)

# 调用示例
ncore = 32
xblock = 32768
xblock_sub = 8192
triton_better_kernel[ncore, 1, 1](x0, out1, x0.numel(), xblock, xblock_sub)
```
