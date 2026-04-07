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

#### Scenario 1: Single Operator Generation

Suitable for developers who need to quickly generate and verify the Triton implementation of a specific operator.

**Steps**:

1. Create `.claude` directory in AscendOpGenAgent and configure the Agent:
```bash
mkdir -p .claude
mv agents/triton-ascend-coder.md .claude/CLAUDE.md
```

2. Enter the AscendOpGenAgent directory and start claude:
```bash
cd /path/to/AscendOpGenAgent
claude
```

3. Enter the operator generation Prompt:
```text
Generate a softmax operator implementation based on the Triton-Ascend framework. The target device architecture is ascend910b1. Please output the generated code files to the /path/to/output/ directory.
```

**Execution Flow**: Agent automatically executes Phase 0-5: Parameter confirmation → Task construction → Algorithm design → Code generation & verification (iterative) → Performance optimization & verification (iterative) → Output report.

---

#### Scenario 2: Batch Benchmark Evaluation

Suitable for batch generation and evaluation of multiple operators with serial execution to avoid NPU conflicts.

**Steps**:

1. Create `.claude` directory in AscendOpGenAgent and configure the Agent:
```bash
mkdir -p .claude
mv agents/triton-ascend-coder.md .claude/CLAUDE.md
```

2. Enter the AscendOpGenAgent directory and execute the batch scheduling script:
```bash
cd /path/to/AscendOpGenAgent
bash utils/run_benchmark.sh \
    --benchmark-dir /path/to/KernelBench \
    --level 1 \
    --range 1-10 \
    --npu 0 \
    --output /path/to/output
```

**Parameter Description**:
- `--benchmark-dir`: Path to KernelBench root directory
- `--level`: Level number (1, 2, 3, 4)
- `--range`: Operator range, e.g., `41-53` (mutually exclusive with `--ids`)
- `--ids`: Comma-separated operator IDs, e.g., `3,7,15` (mutually exclusive with `--range`)
- `--npu`: NPU device ID (default 0)
- `--output`: Output directory

**Execution Flow**: The script launches an independent claude session for each operator and executes serially. Each operator completes Phase 0-5 fully. Results are automatically summarized in `batch_report.md`.

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

Please refer to [`benchmarks/BASELINE.md`](benchmarks/BASELINE.md)  for Triton-related data.

#### AscendC(Updated 2026-03-27)
Please refer to [`benchmarks/BASELINE_0327.md`](benchmarks/BASELINE_0327.md)  for AscendC-related data.

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
