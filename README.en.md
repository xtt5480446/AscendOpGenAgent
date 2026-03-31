# AscendOpGenAgent

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

[中文](README.md) | English

**AscendOpGenAgent** is an automated operator generation and evaluation framework for Ascend NPUs. Based on Triton/AscendC, this project automatically generates and verifies high-performance operator code, aiming to significantly improve the efficiency and quality of operator development on the Ascend architecture.

## Table of Contents

- [AscendOpGenAgent](#ascendopgenagent)
  - [Table of Contents](#table-of-contents)
  - [Core Features](#core-features)
  - [Quick Start](#quick-start)
    - [1. Prerequisites](#1-prerequisites)
    - [2. Installation \& Configuration](#2-installation--configuration)
    - [3. Usage Scenarios](#3-usage-scenarios)
      - [**3.1 Triton**](#31-triton)
      - [Scenario 1: Single Operator Generation (AKG-Triton Agent)](#scenario-1-single-operator-generation-akg-triton-agent)
      - [Scenario 2: Batch Benchmark Evaluation (Benchmark-Evaluator)](#scenario-2-batch-benchmark-evaluation-benchmark-evaluator)
      - [**3.2 AscendC**](#32-ascendc)
      - [Scenario 1: Single Operator Generation (Lingxi-code Agent)](#scenario-1-single-operator-generation-lingxi-code-agent)
      - [Scenario 2: Batch Benchmark Evaluation (Ascend-Benchmark-Evaluator)](#scenario-2-batch-benchmark-evaluation-ascend-benchmark-evaluator)
    - [Evaluation Baseline](#evaluation-baseline)
      - [Triton(Updated 2026-03-20)](#tritonupdated-2026-03-20)
      - [AscendC(Updated 2026-03-27)](#ascendcupdated-2026-03-27)
  - [Project Structure](#project-structure)
  - [License](#license)

## Core Features

| Operator Type | Module | Positioning | Core Capabilities |
|------|------|------|----------|
| **Triton** | **AKG-Triton Agent** | Single operator interactive generation | Task extraction → Code generation → Evaluation & Verification (Accuracy alignment & Performance testing) |
| **Triton** | **Benchmark-Evaluator** | One-click batch evaluation | Execute specified Benchmark evaluation, automatically summarize and generate detailed reports |
| **AscendC** | **Lingxi_code Agent** | AscendC single operator interactive generation | Code generation → Evaluation & Verification (Accuracy alignment & Performance testing) |
| **AscendC** | **Ascend-Benchmark-Evaluator** | AscendC operator one-click batch evaluation | Execute specified Benchmark evaluation, automatically summarize and generate detailed reports |

> **Shared Kernel**: AKG-Triton Agent and Benchmark-Evaluator share the underlying code generation Agent, uniformly handling the core workflow of "Code Generation → Verification → Performance Testing" to ensure consistency and high reusability of the generation logic.

## Quick Start

### 1. Prerequisites

Before running this project, please ensure your environment meets the following requirements:
- Python 3.8+
- Ascend CANN 8.0+
- Triton Ascend
- PyTorch 2.0+
- [OpenCode](https://opencode.ai/) (Please ensure it is correctly installed and configured)

### 2. Installation & Configuration

First, clone this project and configure it into your OpenCode workspace:

```bash
# 1. Clone the project and enter the directory
git clone https://github.com/your-repo/AscendOpGenAgent.git
cd AscendOpGenAgent

# 2. Deploy Agents and Skills to the default OpenCode configuration path
mkdir -p ~/.config/opencode/
cp -r agents/ ~/.config/opencode/
cp -r skills/ ~/.config/opencode/
```

After completion, start OpenCode, and you can select the corresponding Agents and Skills in the UI or command line.

### 3. Usage Scenarios

This project mainly provides two core usage scenarios. Please select the corresponding Agent or Skill according to your needs.

#### **3.1 Triton**

#### Scenario 1: Single Operator Generation (AKG-Triton Agent)
Suitable for developers who need to quickly generate and verify the Triton implementation of a specific operator.

**Steps**:
1. In OpenCode, switch to `AKG-Triton` via the `/agents` command.
2. Enter the operator generation Prompt.

**Prompt Example**:
```text
/AKG-Triton
Generate a softmax_mat operator implementation based on the Triton-Ascend framework. The target device architecture is ascend910b2. Please output the generated code files to the /path/to/output/ directory.
```

**Execution Flow**:
After receiving the instruction, the Agent will automatically execute the following workflow: Confirm parameters → Extract task description → Generate code → Verify accuracy and performance → Output final report.

#### Scenario 2: Batch Benchmark Evaluation (Benchmark-Evaluator)
Suitable for evaluating the overall code generation capability of the Agent on standard datasets (e.g., KernelBench).

**Steps**:
1. In OpenCode, switch to `benchmark-evaluator` via the `/skills` command.
2. Enter the evaluation Prompt.

**Prompt Example 1: Basic Evaluation** (Only specify target and test scope)
```text
Evaluate tasks [20,30] of level 1 in KernelBench, with agent_workspace set to <path/to/your/AscendOpGenAgent>, using the <AKG-triton> agent.
```

**Prompt Example 2: Advanced Evaluation** (Specify output path, running device, and permissions)
```text
Run KernelBench evaluation with the <AKG-triton> agent (workspace: <path/to/your/AscendOpGenAgent>). Target Level 1 problem_id=[6] and Level 2 problem_id=[2]. Save the generated code and results to /path/to/output. Automatically approve all permissions during execution, and specify the device ASCEND_RT_VISIBLE_DEVICES=10.
```

#### **3.2 AscendC**

#### Scenario 1: Single Operator Generation (Lingxi-code Agent)
Suitable for developers who need to quickly generate and verify the AscendC implementation of a specific operator.

**Steps**:
1. In OpenCode, switch to `Lingxi-code` via the `/agents` command.
2. Enter the operator generation Prompt.

**Prompt Example**:
```text
/Lingxi-code
Generate a softmax_mat operator implementation based on the AscendC framework. The target device architecture is ascend910b2. Please output the generated code files to the /path/to/output/ directory.
```

**Execution Flow**:
After receiving the instruction, the Agent will automatically execute the following workflow: Confirm parameters → Extract task description → Generate code → Verify accuracy and performance → Output final report.

#### Scenario 2: Batch Benchmark Evaluation (Ascend-Benchmark-Evaluator)
Suitable for evaluating the overall code generation capability of the Agent on standard datasets (e.g., NPUKernelBench).

**Steps**:
1. In OpenCode, switch to `ascend-benchmark-evaluator` via the `/skills` command.
2. Enter the evaluation Prompt.

**Prompt Example 1: Basic Evaluation** (Only specify target and test scope)
```text
Serially generate tasks of level 1 in NPUKernelBench, with agent_workspace set to <path/to/your/AscendOpGenAgent>, using the <Lingxi-code> agent.
```

**Parameter Description**:
- `<agent_path>`: The working directory path of this project (must contain `agents/` and `skills/`).
- `<benchmark_path>`: The local path of the evaluation dataset (e.g., KernelBench).
- `<output_path>`: **[Optional]** Output directory for evaluation results and generated code.
- `ASCEND_RT_VISIBLE_DEVICES`: **[Optional]** Specify the NPU device ID to use.

### Evaluation Baseline 
#### Triton(Updated 2026-03-27)

- **Test Device**: Ascend 910B2
- **Total Tasks**: 40

| Level | Problem ID | Operator Name | Compilation | Accuracy | PyTorch Latency | Generated Code Latency | Speedup | Final Status |
|:---:|:---:|---|:---:|:---:|---:|---:|---:|:---:|
| 1 | 2 | `Standard matrix multiplication` | ✅ | ✅ | 1.65 ms | 1.70 ms | 0.97x | success |
| 1 | 4 | `Matrix vector multiplication` | ✅ | ✅ | 36.94 ms | 9.54 ms | 3.87x | success |
| 1 | 10 | `3D tensor matrix multiplication` | ✅ | ✅ | 0.62 ms | 0.81 ms | 0.76x | success |
| 1 | 11 | `4D tensor matrix multiplication` | ✅ | ✅ | 5.67 ms | 6.08 ms | 0.93x | success |
| 1 | 12 | `Matmul with diagonal matrices` | ✅ | ✅ | 0.07 ms | 0.09 ms | 0.75x | success |
| 1 | 13 | `Matmul for symmetric matrices` | ✅ | ✅ | 1.65 ms | 1.68 ms | 0.98x | success |
| 1 | 14 | `Matmul for upper triangular matrices` | ✅ | ✅ | 1.66 ms | 12.78 ms | 0.13x | success |
| 1 | 15 | `Matmul for lower triangular matrices` | ✅ | ✅ | 1.71 ms | 1.76 ms | 0.98x | success |
| 1 | 16 | `Matmul with transposed A` | ✅ | ✅ | 1.65 ms | 1054.70 ms | 0.00x | success |
| 1 | 17 | `Matmul with transposed B` | ✅ | ✅ |    1.65 ms | 4983.07 ms | 0.00x | success |
| 1 | 33 | `BatchNorm` | ✅ | ✅ | 10.09 ms | 2653.64 ms | 0.00x | success |
| 1 | 34 | `InstanceNorm` | ✅ | ✅ | 16.68 ms | 128.41 ms | 0.13x | success |
| 1 | 35 | `GroupNorm` | ✅ | ✅ | 17.96 ms | 93.06 ms | 0.19x | success |
| 1 | 36 | `RMSNorm` | ✅ | ✅ | 33.60 ms | 1146.41 ms | 0.03x | success |
| 1 | 41 | `Max_Pooling_1D` | ✅ | ✅ | 22.64 ms | 6583.55 ms | 0.00x | success |
| 1 | 42 | `Max_Pooling_2D` | ✅ | ✅ | - | - | - | success |
| 1 | 43 | `Max_Pooling_3D` | ✅ | ✅ | - | 8.34 ms | - | success |
| 1 | 44 | `Average_Pooling_1D` | ✅ | ✅ | 20.52 ms | 63.92 ms | 0.32x | success |
| 1 | 45 | `Average_Pooling_2D` | ✅ | ✅ | 3.89 ms | 2.67 ms | 1.46x | success |
| 1 | 46 | `Average_Pooling_3D` | ✅ | ✅ | 0.17 ms | 0.08 ms | - | success |
| 1 | 48 | `Mean reduction over a dimension` | ✅ | ✅ | 7.69 ms | 248.97 ms | 0.03x | success |
| 1 | 50 | `conv standard 2D square input square kernel` | ✅ | ✅ | - | - | - | success |
| 1 | 51 | `Argmax over a dimension` | ✅ | ✅ | - | - | - | success |
| 1 | 53 | `Min reduction over a dimension` | ✅ | ✅ | 18.13 ms | 22.97 ms | 0.79x | success |
| 1 | 54 | `conv standard 3D square input square kernel` | ✅ | ✅ | - | - | - | success |
| 1 | 57 | `conv transposed 2D square input square kernel` | ❌ | ❌ | - | - | - | failure |
| 1 | 61 | `conv transposed 3D square input square kernel` | ✅ | ✅ | - | - | - | success |
| 1 | 63 | `conv standard 2D square input square kernel` | ❌ | ❌ | 23.59 ms | 166.72 ms | - | failure |
| 1 | 64 | `conv transposed 1D` | ❌ | ❌ | - | - | - | failure |
| 1 | 67 | `conv standard 1D` | ✅ | ✅ | - | - | - | success |
| 1 | 82 | `conv depthwise 2D square input square kernel` | ✅ | ✅ | - | - | - | success |
| 1 | 87 | `conv pointwise 2D` | ✅ | ✅ | 31.63 ms | 4304.04 ms | 0.01x | success |
| 1 | 99 | `TripletMarginLoss` | ✅ | ✅ | 10.68 ms | 4.60 ms | 2.32x | success |
| 1 | 100 | `HingeLoss` | ✅ | ✅ | 31.67 ms | 1283.66 ms | 0.02x | success |
| 2 | 6 | `Conv3d Softmax MaxPool MaxPool` | ✅ | ✅ | 0.48 ms | - | - | success |
| 2 | 12 | `Gemm Multiply LeakyReLU` | ✅ | ✅ | 0.61 ms | 0.62 ms | 0.98x | success |
| 2 | 17 | `Conv2d InstanceNorm Divide` | ✅ | ✅ | 3.03 ms | 3.25 ms | 0.93x | success |
| 2 | 23 | `Conv3d GroupNorm Mean` | ✅ | ✅ | 0.65 ms | 0.65 ms | 1.00x | success |
| 2 | 30 | `Gemm GroupNorm Hardtanh` | ✅ | ✅ | 0.66 ms | 0.63 ms | 1.04x | success |
| 2 | 94 | `Gemm BiasAdd Hardtanh Mish GroupNorm` | ❌ | ❌ | 0.66 ms | 0.66 ms | - | failure |

#### AscendC(Updated 2026-03-27)
- **Test Device**: Ascend 910B2
- **Total Tasks**: 11

| Level | Problem ID | Operator Name | Compilation | Accuracy | PyTorch Latency | Generated Code Latency | Speedup | Final Status |
|:---:|:---:|---|:---:|:---:|---:|---:|---:|:---:|
| 1 | 1 | `CrossV2` | ✅ | ✅ | 0.022 ms | 0.024 ms | 0.91x | success |
| 1 | 2 | `FatreluMul` | ✅ | ✅ | 0.042 ms | 0.027 ms | 1.55x | success |
| 1 | 3 | `ForeachLerpList` | ✅ | ✅ | 0.063 ms | 0.058 ms | 1.63x | success |
| 1 | 4 | `ForeachPowList` | ✅ | ✅ | 0.029 ms | 0.014 ms | 2.1x | success |
| 1 | 5 | `ForeachPowScalarList` | ✅ | ✅ | 0.0117 ms | 0.0195 ms | 0.6x | success |
| 1 | 6 | `MulAddn` | ✅ | ✅ | 0.049 ms | 0.044 ms | 1.11x | success |
| 1 | 7 | `LayerNormV4` | ✅ | ✅ | 0.71 ms | 0.539 ms | 1.32x | success |
| 1 | 8 | `Logit` | ✅ | ✅ | 0.022 ms | 0.031 ms | 1.38x | success |
| 1 | 9 | `LogitGrad` | ✅ | ✅ | 0.108 ms | 0.028 ms | 3.89x | success |
| 1 | 10 | `MaxPool3DWithArgmaxV2` | ✅ | ✅ | 0.0154 ms | 0.0171 ms | 0.9x | success |
| 1 | 11 | `QuantizedBatchNorm` | ✅ | ✅ | 0.571 ms | 0.235 ms | 2.43x | success |
| 1 | 12 | `AdaptiveAvgPool3d` | ✅ | ❌ | ❌ | ❌ | ❌ | failure |
| 1 | 13 | `AdaptiveAvgPool3dGrad` | ✅ | ❌ | ❌ | ❌ | ❌ | failure |
| 1 | 14 | `AdaptiveMaxPool3DGrad` | ✅ | ❌ | ❌ | ❌ | ❌ | failure |
| 1 | 15 | `TransformBiasRescaleQkv` | ✅ | ❌ | ❌ | ❌ | ❌ | failure |
| 1 | 16 | `AddRmsNormDynamicQuantV2` | ✅ | ❌ | ❌ | ❌ | ❌ | failure |
| 1 | 17 | `STFT` | ✅ | ❌ | ❌ | ❌ | ❌ | failure |
| 1 | 18 | `ApplyTopKTopPWithSorted` | ✅ | ❌ | ❌ | ❌ | ❌ | failure |
| 1 | 19 | `AvgPool3D` | ✅ | ❌ | ❌ | ❌ | ❌ | failure |
| 1 | 20 | `AvgPool3DGrad` | ✅ | ❌ | ❌ | ❌ | ❌ | failure |
| 1 | 21 | `BatchNormV3` | ✅ | ❌ | ❌ | ❌ | ❌ | failure |
| 1 | 22 | `ChamferDistanceGrad` | ✅ | ❌ | ❌ | ❌ | ❌ | failure |
| 1 | 23 | `CTCLossV3` | ✅ | ❌ | ❌ | ❌ | ❌ | failure |

## Project Structure

```text
AscendOpGenAgent/
├── .gitignore
├── LICENSE
├── README.en.md
├── README.md
├── agents/                     # Agent definition directory
│   ├── AKG-triton.md           # Main orchestration Agent
│   ├── benchmark-scheduler.md
│   ├── kernelgen-workflow.md   # Sub-Agent (Code generation workflow)
│   ├── lingxi_code.md
│   └── performance-optimizer.md
├── benchmarks/                 # Evaluation dataset storage directory
│   ├── KernelBench/
│   │   ├── level1/             # Level 1 test cases (100 tasks)
│   │   ├── level2/             # Level 2 test cases (99 tasks)
│   │   ├── level3/             # Level 3 test cases (52 tasks)
│   │   └── level4/             # Level 4 test cases (20 tasks)
│   └── NPUKernelBench/
│       └── level1/             # NPU KernelBench Level 1 test cases (31 tasks)
└── skills/                     # Skill implementation directory
    ├── ascendc_evalution/
    ├── ascend_benchmark_evaluator/
    ├── ascend_call_generation/
    ├── benchmark-evaluator/    # Batch evaluation Skill
    ├── dsl_baseline_generation/
    ├── dsl_lowering/
    ├── functional_conversion/
    ├── kernel-designer/
    ├── kernel-generator/       # Code generation Skill
    ├── kernel-verifier/        # Verification and performance testing Skill
    ├── latency-optimizer/
    ├── op-task-extractor/      # Task extraction Skill
    ├── op_desc_generation/
    └── reference_generation/
```


## License

This project is licensed under the [Apache 2.0 License](LICENSE).
