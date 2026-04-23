# AscendOpGenAgent

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

中文 | [English](README.en.md)

**AscendOpGenAgent** 是一个面向 Ascend NPU 的自动化算子生成与评测框架。本项目基于 Triton/AscendC 自动生成并验证高性能算子代码，旨在大幅提升 Ascend 架构下的算子开发效率与质量。

## 目录

- [AscendOpGenAgent](#ascendopgenagent)
  - [目录](#目录)
  - [核心功能](#核心功能)
  - [快速开始](#快速开始)
    - [1. 环境要求](#1-环境要求)
    - [2. 安装与配置](#2-安装与配置)
    - [3. 使用场景指南](#3-使用场景指南)
      - [**3.1 Triton**](#31-triton)
      - [场景一：单算子生成](#场景一单算子生成)
      - [场景二：Benchmark 批量评测](#场景二benchmark-批量评测)
      - [**3.2 AscendC**](#32-ascendc)
      - [场景一：单算子生成 (Lingxi-code Agent)](#场景一单算子生成-lingxi-code-agent)
      - [场景二：Benchmark 批量评测 (Ascend-Benchmark-Evaluator)](#场景二benchmark-批量评测-ascend-benchmark-evaluator)
    - [评测基线](#评测基线)
      - [Triton](#triton)
      - [AscendC](#ascendc)
  - [项目结构](#项目结构)
  - [单用例多 Shape 支持](#单用例多-shape-支持)
    - [输入规格（算子描述文件）](#输入规格算子描述文件)
      - [单 Shape 格式（向后兼容）](#单-shape-格式向后兼容)
      - [多 Shape 格式](#多-shape-格式)
    - [输出规格（性能报告）](#输出规格性能报告)
      - [单 Shape 性能报告](#单-shape-性能报告)
      - [多 Shape 性能报告](#多-shape-性能报告)
    - [适用场景](#适用场景)
  - [许可证](#许可证)

## 核心功能

| 算子类型 | 模块 | 定位 | 核心能力 |
|------|------|------|----------|
| **Triton** | **AKG-Triton Agent** | 单算子交互式生成 | 任务提取 → 代码生成 → 评测验证（精度对齐与性能测试） |
| **Triton**  | **Benchmark-Evaluator** | 一键批量评测 | 执行指定 Benchmark 评测，自动总结并生成详细报告 |
| **AscendC** | **Lingxi_code Agent** | AscendC 单算子交互式生成 | 代码生成 → 评测验证（精度对齐与性能测试） |
| **AscendC** | **Ascend-Benchmark-Evaluator** | AscendC 算子一键批量评测 | 执行指定 Benchmark 评测，自动总结并生成详细报告 |

>  **共享内核**：AKG-Triton Agent、Benchmark-Evaluator两者底层共用代码生成 Agent，统一处理“代码生成 → 验证 → 性能测试”的核心工作流，确保生成逻辑的一致性与高复用性。

##  快速开始

### 1. 环境要求

在运行本项目之前，请确保您的环境满足以下要求：
- Python 3.8+
- Ascend CANN 8.0+
- Triton Ascend
- PyTorch 2.0+
- Claude Code CLI (请确保已正确安装并配置)
- tilelang-ascend (参考https://github.com/tile-ai/tilelang-ascend/blob/ascendc_pto/README.md#method-3-compile-and-install-from-source 安装)

### 2. 安装与配置

克隆本项目并配置 Claude Code 环境：

```bash
# 1. 克隆项目并进入目录
git clone https://github.com/your-repo/AscendOpGenAgent.git
cd AscendOpGenAgent

# 2. 配置 Claude Code（可选，如需自定义配置）
# Claude Code 会自动识别项目中的 .claude/CLAUDE.md 配置文件
```

完成后，即可在项目目录中使用 Claude Code 进行开发。

### 3. 使用场景指南

本项目主要提供两个核心使用场景，请根据需求选择对应的 Agent 或 Skill。
#### **3.1 Triton**

#### 场景一：单算子生成

适用于开发者需要快速生成、验证某个特定算子的 Triton 实现。

**操作步骤**：

1. 在 AscendOpGenAgent 目录下配置 Agent和skills：
```bash
mkdir -p .claude
mkdir -p .claude/skills
mv agents/triton-ascend-coder.md .claude/CLAUDE.md
mv skills/triton/* .claude/skills/
```

2. 进入 AscendOpGenAgent 目录，启动 claude：
```bash
claude
```

3. 输入算子生成 Prompt：
```text
生成一个基于 Triton-Ascend 框架的 softmax 算子实现。目标设备架构为 ascend910b1，请将生成的代码文件输出至 /path/to/output/ 目录下。
```

**执行流程**：Agent 自动执行 Phase 0-5：参数确认 → 任务构建 → 算法设计 → 代码生成与验证（迭代） → 性能优化与验证（迭代） → 输出报告。

---

#### 场景二：Benchmark 批量评测

适用于批量评测算子的生成效果，支持单 NPU 串行或多 NPU 并行执行。

**操作步骤**：

1. 在 AscendOpGenAgent 目录下创建 `.claude` 目录并配置 Agent：
```bash
mkdir -p .claude
mkdir -p .claude/skills
mv agents/triton-ascend-coder.md .claude/CLAUDE.md
mv skills/triton/* .claude/skills/
```

2. 进入 AscendOpGenAgent 目录，执行批量调度脚本：

**单 NPU 串行模式**：
```bash
cd /path/to/AscendOpGenAgent
bash utils/run_benchmark_triton.sh \
    --benchmark-dir /path/to/KernelBench \
    --level 1 \
    --range 1-30 \
    --npu 0 \
    --output /path/to/output
```

**多 NPU 并行模式**（推荐）：
```bash
cd /path/to/AscendOpGenAgent
bash utils/run_benchmark_triton.sh \
    --benchmark-dir /path/to/KernelBench \
    --level 1 \
    --range 1-30 \
    --npu-list "0,1,2,3,4,5" \
    --output /path/to/output
```

**参数说明**：
- `--benchmark-dir`: Benchmark 根目录路径（必填）
- `--level`: Level 编号，如 1, 2, 3, 4（必填）
- `--range`: 算子范围，如 `1-30`（与 `--ids` 二选一）
- `--ids`: 指定算子编号列表，逗号分隔，如 `3,7,15`（与 `--range` 二选一）
- `--npu`: 单 NPU 设备 ID，如 0（默认 0，与 `--npu-list` 互斥）
- `--npu-list`: 多 NPU 列表，逗号分隔，如 `0,1,2,3,4,5`（与 `--npu` 互斥，优先级更高）
- `--output`: 输出目录（必填）


#### **3.2 AscendC**

#### 场景一：单算子生成 (Lingxi-code Agent)

适用于开发者需要快速生成、验证某个特定算子的 AscendC 实现。

**操作步骤**：

1. 在 AscendOpGenAgent 目录下配置 Agent 和 skills：
```bash
mkdir -p .claude
mkdir -p .claude/skills
mv agents/ascend-kernel-developer.md .claude/CLAUDE.md
mv skills/ascendc/* .claude/skills/
```

2. 进入 AscendOpGenAgent 目录，启动 claude：
```bash
claude
```

3. 输入算子生成 Prompt：
```text
生成一个基于 AscendC 框架的 softmax 算子实现。目标设备架构为 ascend910b2，请将生成的代码文件输出至 /path/to/output/ 目录下。
```

**执行流程**：Agent 自动执行：确认参数 → 提取任务描述 → 生成代码 → 验证精度与性能 → 输出最终报告。

---

#### 场景二：Benchmark 批量评测 (Ascend-Benchmark-Evaluator)

适用于批量评测算子的生成效果，支持单 NPU 串行或多 NPU 并行执行。

**操作步骤**：

1. 在 AscendOpGenAgent 目录下创建 `.claude` 目录并配置 Agent：
```bash
mkdir -p .claude
mkdir -p .claude/skills
mv agents/ascend-kernel-developer.md .claude/CLAUDE.md
mv skills/ascendc/* .claude/skills/
```

2. 进入 AscendOpGenAgent 目录，执行批量调度脚本：

**单 NPU 串行模式**：
```bash
cd /path/to/AscendOpGenAgent
bash utils/run_benchmark_ascendc.sh \
    --benchmark-dir /path/to/NPUKernelBench \
    --level 1 \
    --range 1-30 \
    --npu 0 \
    --output /path/to/output
```

**多 NPU 并行模式**（推荐）：
```bash
cd /path/to/AscendOpGenAgent
bash utils/run_benchmark_ascendc.sh \
    --benchmark-dir /path/to/NPUKernelBench \
    --level 1 \
    --range 1-30 \
    --npu-list "0,1,2,3,4,5" \
    --output /path/to/output
```

**参数说明**：
- `--benchmark-dir`: Benchmark 根目录路径（必填）
- `--level`: Level 编号，如 1, 2, 3（必填）
- `--range`: 算子范围，如 `1-30`（与 `--ids` 二选一）
- `--ids`: 指定算子编号列表，逗号分隔，如 `3,7,15`（与 `--range` 二选一）
- `--npu`: 单 NPU 设备 ID，如 0（默认 0，与 `--npu-list` 互斥）
- `--npu-list`: 多 NPU 列表，逗号分隔，如 `0,1,2,3,4,5`（与 `--npu` 互斥，优先级更高）
- `--output`: 输出目录（必填）

### 评测基线

#### Triton
关于 Triton 的相关数据，请参阅[`benchmarks/BASELINE_0408.md`](benchmarks/BASELINE_0408.md)

#### AscendC
关于 AscendC 的相关数据，请参阅[`benchmarks/BASELINE_0408.md`](benchmarks/BASELINE_0415.md) 



## 项目结构

```text
AscendOpGenAgent/
├── .gitignore
├── LICENSE
├── README.en.md
├── README.md
├── agents/                     # Agent 定义目录
│   ├── AKG-triton.md           # 主编排 Agent
│   ├── benchmark-scheduler.md
│   ├── kernelgen-workflow.md   # 子 Agent（代码生成工作流）
│   ├── ascend-kernel-developer.md                # 主编排 Agent（Phase 0-7）
│   ├── ascendc-debug-agent-discovery.md               # Debug subagent（发现式ascendc debug agent）
│   └── performance-optimizer.md
├── benchmarks/                 # 评测数据集存放目录
│   ├── KernelBench/
│   │   ├── level1/             # Level 1 测试用例 (100个)
│   │   ├── level2/             # Level 2 测试用例 (99个)
│   │   ├── level3/             # Level 3 测试用例 (52个)
│   │   └── level4/             # Level 4 测试用例 (20个)
│   └── NPUKernelBench/
│       └── level1/             # NPU KernelBench Level 1 测试用例 (31个)
└── skills/                     # Skill 实现目录
    ├── ascendc_evalution/
    ├── ascend_benchmark_evaluator/
    ├── ascendc/
    ├── benchmark-evaluator/    # 批量评测 Skill
    ├── dsl_baseline_generation/
    ├── dsl_lowering/
    ├── functional_conversion/
    ├── kernel-designer/
    ├── kernel-generator/       # 代码生成 Skill
    ├── kernel-verifier/        # 验证与性能测试 Skill
    ├── latency-optimizer/
    ├── op-task-extractor/      # 任务提取 Skill
    ├── op_desc_generation/
    └── reference_generation/

```


## 单用例多 Shape 支持

本框架支持在一个算子用例中定义多个 Shape 配置进行批量验证和性能评测，适用于需要测试算子在不同规模输入下的性能表现的场景。

### 输入规格（算子描述文件）

#### 单 Shape 格式（向后兼容）

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(x)

def get_inputs():
    """返回单组输入，形式为 List[Tensor/...]"""
    return [torch.randn(128, 128, dtype=torch.float16)]

def get_init_inputs():
    """返回初始化参数列表"""
    return []
```

**规格说明**：
- `get_inputs()`: 返回 `List[Tensor/...]`，代表单组输入
- 适用于单一 Shape 场景
- `get_init_inputs()`: 返回 `__init__` 的初始化参数列表

#### 多 Shape 格式

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
    def forward(self, x: torch.Tensor, approximate='none') -> torch.Tensor:
        return torch.nn.functional.gelu(x, approximate=approximate)

# 多 Shape 配置列表
INPUT_CASES = [
    {'inputs': [{'dtype': 'float32', 'name': 'x', 'shape': [128, 128], 'type': 'tensor'},
                 {'dtype': 'str', 'name': 'approximate', 'type': 'attr', 'value': 'none'}]},
    {'inputs': [{'dtype': 'float32', 'name': 'x', 'shape': [256, 256], 'type': 'tensor'},
                 {'dtype': 'str', 'name': 'approximate', 'type': 'attr', 'value': 'tanh'}]},
    {'inputs': [{'dtype': 'float16', 'name': 'x', 'shape': [1024, 1024], 'type': 'tensor'},
                 {'dtype': 'str', 'name': 'approximate', 'type': 'attr', 'value': 'none'}]},
]

# 必须实现，返回 List[List[Tensor/...]]
def get_input_groups():
    """返回多组输入列表，每组对应一个 Shape 配置"""
    input_groups = []
    for case in INPUT_CASES:
        group = []
        for spec in case['inputs']:
            if spec['type'] == 'tensor':
                dtype = {'float16': torch.float16, 'float32': torch.float32}[spec['dtype']]
                group.append(torch.randn(*spec['shape'], dtype=dtype))
            elif spec['type'] == 'attr':
                group.append(spec['value'])
        input_groups.append(group)
    return input_groups

# 可选实现，用于向后兼容
def get_inputs():
    """返回单组输入，取第一组"""
    return get_input_groups()[0]

def get_init_inputs():
    """返回初始化参数列表"""
    return []
```

**输入规格说明**：

| 函数 | 返回类型 | 用途 | 必需 |
|------|---------|------|------|
| `get_input_groups()` | `List[List[Tensor/...]]` | 多 Shape 入口，每组对应一个测试配置 | ✅ 多 Shape 场景必需 |
| `get_inputs()` | `List[Tensor/...]` | 单 Shape 入口，返回第一组或单组输入 | 建议实现（向后兼容） |
| `get_init_inputs()` | `List[Any]` | `Model.__init__` 的初始化参数 | ✅ 必需 |

**输入配置字段说明**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `dtype` | `str` | 数据类型：float16/float32/float64/bfloat16/int8/int16/int32/int64/bool |
| `shape` | `List[int]` | 张量形状，如 `[128, 256]` |
| `name` | `str` | 参数名称 |
| `type` | `str` | 类型："tensor"（张量）、"attr"（属性值）、"tensor_list"（张量列表） |
| `value` | `Any` | 当 `type="attr"` 时，属性值 |

### 输出规格（性能报告）

#### 单 Shape 性能报告

```json
{
  "op_name": "gelu",
  "warmup": 5,
  "repeats": 50,
  "total_cases": 1,
  "framework": {
    "avg_latency_ms": 0.2345,
    "peak_memory_mb": 2.50
  },
  "implementation": {
    "avg_latency_ms": 0.1567,
    "peak_memory_mb": 1.25
  },
  "speedup_vs_torch": 1.50,
  "perf_method": "profiler",
  "skill_path": "/path/to/.claude/skills/kernel-verifier"
}
```

#### 多 Shape 性能报告

```json
{
  "op_name": "gelu",
  "warmup": 5,
  "repeats": 50,
  "total_cases": 3,
  "framework": {
    "avg_latency_ms": 0.4567,
    "peak_memory_mb": 8.50
  },
  "implementation": {
    "avg_latency_ms": 0.3123,
    "peak_memory_mb": 4.25
  },
  "speedup_vs_torch": 1.46,
  "perf_method": "profiler",
  "skill_path": "/path/to/.claude/skills/kernel-verifier",
  "per_shape_results": [
    {
      "shape": [128, 128],
      "framework": {
        "avg_latency_ms": 0.0234,
        "peak_memory_mb": 0.50
      },
      "implementation": {
        "avg_latency_ms": 0.0156,
        "peak_memory_mb": 0.25
      },
      "speedup_vs_torch": 1.50
    },
    {
      "shape": [256, 256],
      "framework": {
        "avg_latency_ms": 0.0891,
        "peak_memory_mb": 2.00
      },
      "implementation": {
        "avg_latency_ms": 0.0588,
        "peak_memory_mb": 1.00
      },
      "speedup_vs_torch": 1.52
    },
    {
      "shape": [1024, 1024],
      "framework": {
        "avg_latency_ms": 1.2577,
        "peak_memory_mb": 8.00
      },
      "implementation": {
        "avg_latency_ms": 0.8625,
        "peak_memory_mb": 12.50
      },
      "speedup_vs_torch": 1.46
    }
  ]
}
```

**输出字段说明**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `op_name` | `str` | 算子名称 |
| `warmup` | `int` | 预热次数 |
| `repeats` | `int` | 正式测试次数 |
| `total_cases` | `int` | 测试的 Shape 数量（单 Shape 为 1，多 Shape ≥2） |
| `framework.avg_latency_ms` | `float` | PyTorch 实现平均延迟（毫秒）各 Shape 平均 |
| `framework.peak_memory_mb` | `float` | PyTorch 峰值内存（MB）各 Shape 平均 |
| `implementation.avg_latency_ms` | `float` | 实现平均延迟（毫秒）各 Shape 平均 |
| `implementation.peak_memory_mb` | `float` | 实现峰值内存（MB）各 Shape 平均 |
| `speedup_vs_torch` | `float` | 相比 PyTorch 的加速比（各 Shape 加速比的平均值） |
| `perf_method` | `str` | 评测方式："profiler"（torch_npu.profiler）或 "fallback"（time.perf_counter 兜底） |
| `skill_path` | `str` | 使用的 benchmark skill 路径 |
| `per_shape_results` | `List[Dict]` | 多 Shape 明细数据（当 `total_cases > 1` 时出现） |

**per_shape_results 元素说明**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `shape` | `List[int]` | 主要输入张量的形状 |
| `framework.avg_latency_ms` | `float` | 该 Shape 的 PyTorch 延迟 |
| `implementation.avg_latency_ms` | `float` | 该 Shape 的实现延迟 |
| `speedup_vs_torch` | `float` | 该 Shape 的加速比 |

### 适用场景

1. **算子泛化性测试**：验证生成的 Triton 算子在多种输入规模下的正确性和稳定性
2. **性能趋势分析**：通过对比不同 Shape 的加速比，识别算子的优势和局限性
3. **AI 模型场景复现**：模拟真实模型中的典型输入 Shape 分布（如 LLM 的多种序列长度）
4. **自动 Benchmark 评测**：批量评测时自动覆盖多种 Shape，减少重复工作量

## 许可证

本项目采用 [Apache 2.0 License](LICENSE) 开源许可证。