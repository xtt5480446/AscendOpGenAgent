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
      - [场景一：单算子生成 (AKG-Triton Agent)](#场景一单算子生成-akg-triton-agent)
      - [场景二：Benchmark 批量评测 (Benchmark-Evaluator)](#场景二benchmark-批量评测-benchmark-evaluator)
      - [**3.2 AscendC**](#32-ascendc)
      - [场景一：单算子生成 (Lingxi-code Agent)](#场景一单算子生成-lingxi-code-agent)
      - [场景二：Benchmark 批量评测 (Ascend-Benchmark-Evaluator)](#场景二benchmark-批量评测-ascend-benchmark-evaluator)
    - [评测基线](#评测基线)
      - [Triton（更新于 2026-03-20）](#triton更新于-2026-03-20)
      - [AscendC（更新于 2026-03-27）](#ascendc更新于-2026-03-27)
  - [项目结构](#项目结构)
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
- [OpenCode](https://opencode.ai/) (请确保已正确安装并配置)

### 2. 安装与配置

首先，克隆本项目并将其配置到 OpenCode 的工作环境中：

```bash
# 1. 克隆项目并进入目录
git clone https://github.com/your-repo/AscendOpGenAgent.git
cd AscendOpGenAgent

# 2. 部署 Agent 和 Skills 到 OpenCode 默认配置路径
mkdir -p ~/.config/opencode/
cp -r agents/ ~/.config/opencode/
cp -r skills/ ~/.config/opencode/
```

完成后，启动 OpenCode，即可在界面或命令行中选择对应的 Agents 和 Skills。

### 3. 使用场景指南

本项目主要提供两个核心使用场景，请根据需求选择对应的 Agent 或 Skill。
#### **3.1 Triton**

#### 场景一：单算子生成 (AKG-Triton Agent)
适用于开发者需要快速生成、验证某个特定算子的 Triton 实现。

**操作步骤**：
1. 在 OpenCode 中，通过 `/agents` 命令切换至 `AKG-Triton`。
2. 输入算子生成 Prompt。

**Prompt 示例**：
```text
/AKG-Triton
生成一个基于 Triton-Ascend 框架的 softmax_mat 算子实现。目标设备架构为 ascend910b2，请将生成的代码文件输出至 /path/to/output/ 目录下。
```

**执行流程**：
Agent 接收到指令后，将自动执行以下流程：确认参数 → 提取任务描述 → 生成代码 → 验证精度与性能 → 输出最终报告。

#### 场景二：Benchmark 批量评测 (Benchmark-Evaluator)
适用于评估 Agent 在标准数据集（如 KernelBench）上的整体代码生成能力。

**操作步骤**：
1. 在 OpenCode 中，通过 `/agents` 命令切换至 `benchmark-scheduler`。
2. 输入评测 Prompt。

**Prompt 示例 1：基础评测**（仅指定目标与测试范围）
```text
评测KernelBench中level1的[20,30]任务,agent_workspace是<path/to/your/AscendOpGenAgent>
```

**Prompt 示例 2：全量评测**（覆盖基线任务集，指定输出路径与设备）
```text
评测KernelBench中Level 1的2, 4, 10, 11, 12, 13, 14, 15, 16, 17, 33, 34, 35, 36, 41, 42, 43, 44, 45, 46, 48, 50, 51, 53, 54, 57, 61, 63, 64, 67, 82, 87, 99, 100和Level 2的6, 12, 17, 23, 30, 94的任务,
agent_workspace是<path/to/your/AscendOpGenAgent>。
请将生成的代码和评测结果输出到 /path/to/output 目录下。
执行期间默认同意所有权限，并指定设备 ASCEND_RT_VISIBLE_DEVICES=10。
```
#### **3.2 AscendC**
#### 场景一：单算子生成 (Lingxi-code Agent)
适用于开发者需要快速生成、验证某个特定算子的 AscendC 实现。

**操作步骤**：
1. 在 OpenCode 中，通过 `/agents` 命令切换至 `Lingxi-code`。
2. 输入算子生成 Prompt。

**Prompt 示例**：
```text
/Lingxi-code
生成一个基于 AscendC 框架的 softmax_mat 算子实现。目标设备架构为 ascend910b2，请将生成的代码文件输出至 /path/to/output/ 目录下。
```

**执行流程**：
Agent 接收到指令后，将自动执行以下流程：确认参数 → 提取任务描述 → 生成代码 → 验证精度与性能 → 输出最终报告。

#### 场景二：Benchmark 批量评测 (Ascend-Benchmark-Evaluator)
适用于评估 Agent 在标准数据集（如 NPUKernelBench）上的整体代码生成能力。

**操作步骤**：
1. 在 OpenCode 中，通过 `/skills` 命令切换至 `ascend-benchmark-evaluator`。
2. 输入评测 Prompt。

**Prompt 示例 1：基础评测**（仅指定目标与测试范围）
```text
串行生成NPUKernelBench中level1的任务,agent_workspace是<path/to/your/AscendOpGenAgent>,使用<Lingxi-code> agent
```

**参数说明**：
- `<agent_path>`: 本项目的工作目录路径（需包含 `agents/` 和 `skills/`）。
- `<benchmark_path>`: 评测数据集（如 KernelBench）的本地路径。
- `<output_path>`: **[可选]** 评测结果与生成代码的输出目录。
- `ASCEND_RT_VISIBLE_DEVICES`: **[可选]** 指定使用的 NPU 设备 ID。

### 评测基线
关于 Triton 的相关数据，请参阅[`benchmarks/BASELINE.md`](benchmarks/BASELINE.md) 

#### AscendC（更新于 2026-03-27）
- **测试设备**：Ascend 910B2
- **总任务数**：23

| Level | Problem ID | 算子名称 | 编译通过 | 精度正确 | PyTorch 延迟 | 生成代码延迟 | 加速比 | 最终状态 |
|:---:|:---:|---|:---:|:---:|---:|---:|---:|:---:|
| 1 | 1 | `CrossV2` | ✅ | ✅ | 0.022 ms | 0.024 ms | 0.91x | 成功 |
| 1 | 2 | `FatreluMul` | ✅ | ✅ | 0.042 ms | 0.027 ms | 1.55x | 成功 |
| 1 | 3 | `ForeachLerpList` | ✅ | ✅ | 0.063 ms | 0.058 ms | 1.63x | 成功 |
| 1 | 4 | `ForeachPowList` | ✅ | ✅ | 0.029 ms | 0.014 ms | 2.1x | 成功 |
| 1 | 5 | `ForeachPowScalarList` | ✅ | ✅ | 0.0117 ms | 0.0195 ms | 0.6x | 成功 |
| 1 | 6 | `MulAddn` | ✅ | ✅ | 0.049 ms | 0.044 ms | 1.11x | 成功 |
| 1 | 7 | `LayerNormV4` | ✅ | ✅ | 0.71 ms | 0.539 ms | 1.32x | 成功 |
| 1 | 8 | `Logit` | ✅ | ✅ | 0.022 ms | 0.031 ms | 1.38x | 成功 |
| 1 | 9 | `LogitGrad` | ✅ | ✅ | 0.108 ms | 0.028 ms | 3.89x | 成功 |
| 1 | 10 | `MaxPool3DWithArgmaxV2` | ✅ | ✅ | 0.0154 ms | 0.0171 ms | 0.9x | 成功 |
| 1 | 11 | `QuantizedBatchNorm` | ✅ | ✅ | 0.571 ms | 0.235 ms | 2.43x | 成功 |
| 1 | 12 | `AdaptiveAvgPool3d` | ✅ | ❌ | ❌ | ❌ | ❌ | 失败 |
| 1 | 13 | `AdaptiveAvgPool3dGrad` | ✅ | ❌ | ❌ | ❌ | ❌ | 失败 |
| 1 | 14 | `AdaptiveMaxPool3DGrad` | ✅ | ❌ | ❌ | ❌ | ❌ | 失败 |
| 1 | 15 | `TransformBiasRescaleQkv` | ✅ | ❌ | ❌ | ❌ | ❌ | 失败 |
| 1 | 16 | `AddRmsNormDynamicQuantV2` | ✅ | ❌ | ❌ | ❌ | ❌ | 失败 |
| 1 | 17 | `STFT` | ✅ | ❌ | ❌ | ❌ | ❌ | 失败 |
| 1 | 18 | `ApplyTopKTopPWithSorted` | ✅ | ❌ | ❌ | ❌ | ❌ | 失败 |
| 1 | 19 | `AvgPool3D` | ✅ | ❌ | ❌ | ❌ | ❌ | 失败 |
| 1 | 20 | `AvgPool3DGrad` | ✅ | ❌ | ❌ | ❌ | ❌ | 失败 |
| 1 | 21 | `BatchNormV3` | ✅ | ❌ | ❌ | ❌ | ❌ | 失败 |
| 1 | 22 | `ChamferDistanceGrad` | ✅ | ❌ | ❌ | ❌ | ❌ | 失败 |
| 1 | 23 | `CTCLossV3` | ✅ | ❌ | ❌ | ❌ | ❌ | 失败 |



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
│   ├── lingxi_code.md
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
    ├── ascend_call_generation/
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


## 许可证

本项目采用 [Apache 2.0 License](LICENSE) 开源许可证。