# precision-tuning Agent 迁移设计

## 背景

将 OpenOps-debug 中成熟的 `precision-tuning` agent 迁移到 AscendOpGenAgent，作为独立的 post-generation debug agent。当算子生成流程（ascend-kernel-developer）产出的 AscendC 实现通过了编译但未通过精度/正确性测试时，由该 agent 接管多轮修复。

**当前范围**：仅支持 AscendC 测试失败（过了 TileLang 编译，未过 `evaluate_ascendc.sh`）。TileLang 编译失败的支持留待后续。

**触发方式**：手动触发，传入失败产物的 task 名（对应 `{repo_root}/{task_name}/` 目录）。

---

## 方案：Fork + 深度适配

从 OpenOps-debug fork `precision-tuning` skill，在 AscendOpGenAgent 内独立演进。

**注意**：不是简单路径替换。两个仓库的 kernel 结构存在语义差异（见 Section 2），gate.py 的检查逻辑需重写，forensics.py 的数据来源需重新设计。

---

## Section 0：触发契约

agent 入口接受以下参数：

| 参数 | 说明 | 是否必须 |
|---|---|---|
| `task_name` | task 目录名（相对于 repo root） | 必须 |
| `npu` | NPU 设备 ID | 默认 0 |

**前提条件**（缺一不可，agent 入口校验）：

```
{repo_root}/{task_name}/
├── model.py                    ← 参考实现（必须）
├── model_new_ascendc.py        ← AscendC wrapper（必须）
├── kernel/
│   ├── pybind11.cpp            ← host launch + pybind（必须）
│   ├── {op_name}_tiling.h      ← TilingData 定义（必须）
│   └── *.cpp / *_kernel.h      ← 至少一个非 pybind 的 kernel 文件
└── {op_name}.json              ← 测试用例（必须）
```

工作目录限制：agent 只允许读写 `{repo_root}/` 内的路径，不得访问父目录或绝对路径外的位置。

---

## Section 1：目录结构

```
AscendOpGenAgent/
├── agents/
│   ├── ascend-kernel-developer.md
│   ├── triton-ascend-coder.md
│   └── precision-tuning.md                    ← 新增 agent 入口
└── skills/ascendc/
    └── precision-tuning/                       ← 新增 skill 目录
        ├── SKILL.md                            ← 主流程（fork + 适配）
        ├── references/
        │   └── precision_knowledge_base.json   ← 直接搬过来
        └── scripts/
            ├── precision_forensics.py          ← 重写数据来源
            ├── precision_gate.py               ← 重写检查逻辑
            └── precision_knowledge.py          ← 基本不变
```

`decomposition_examples/` 不搬。Phase A 参考改为指向 `archive_tasks/`（见 Section 3）。

---

## Section 2：路径映射与语义差异

### 2.1 文件路径映射

| OpenOps 约定 | AscendOpGenAgent 约定 | 影响范围 |
|---|---|---|
| `{output_path}/{op_name}_reference.py` | `{repo_root}/{task_name}/model.py` | forensics.py, SKILL.md |
| `{output_path}/{op_name}_custom.py` | `{repo_root}/{task_name}/model_new_ascendc.py` | forensics.py, SKILL.md |
| `{output_path}/{op_name}_dsl.py` | 删除 | SKILL.md Sub-step 2.2 |
| `{output_path}/{op_name}_op_desc.json` | 删除（见 2.3） | forensics.py |
| `{output_path}/{OpName}Custom/op_kernel/{op_name}_custom.cpp` | `kernel/*.cpp` + `kernel/*_kernel.h`（全部读） | SKILL.md Phase B |
| `{output_path}/{OpName}Custom/op_host/{op_name}_custom_tiling.h` | `kernel/{op_name}_tiling.h` | gate.py, SKILL.md Phase B |
| `{output_path}/{OpName}Custom/op_host/{op_name}_custom.cpp`（host 逻辑） | `kernel/pybind11.cpp`（host + pybind 合并） | SKILL.md Phase B |
| `{output_path}/{op_name}.cpp`（pybind 入口） | `kernel/pybind11.cpp`（同一文件） | gate.py |

### 2.2 pybind11.cpp 语义合并

OpenOps 的 `op_host/{op_name}_custom.cpp`（tiling 计算、launch 逻辑）和 `{op_name}.cpp`（pybind 绑定）是两个文件，分别维护。

AscendOpGenAgent 把这两部分合并到 `kernel/pybind11.cpp` 里：参数校验、tiling 构造、workspace 分配、variant dispatch、kernel launch、pybind 绑定全在此文件。

**影响**：gate.py 不能沿用旧的"host 文件 + kernel 文件分别检查"逻辑，需改为：
- `kernel/pybind11.cpp` 存在且包含合法的 `PYBIND11_MODULE(...)` 宏
- `model_new_ascendc.py` 的 `import _xxx_ext` 名称与 `PYBIND11_MODULE` 宏第一个参数一致
- `kernel/` 下至少有一个非 `pybind11.cpp` 的 `.cpp` 文件

### 2.3 kernel 多文件处理

Phase B 读取代码时，读全部：
- `kernel/{op_name}_tiling.h`
- `kernel/*_kernel.h`（所有 kernel 类定义）
- `kernel/*.cpp`（所有 kernel entry，排除 pybind11.cpp 单独列出）
- `kernel/kernel_common.h`、`kernel/vector_tile.h`、`kernel/matmul_tile.h`、`kernel/workspace_queue.h`（若存在）
- `kernel/pybind11.cpp`（host 逻辑）

这些 helper 文件在现有 archive tasks 里承担实质逻辑，不是附件。

### 2.4 model_new_ascendc.py 适配要求

数值错误可能来自 wrapper，不只来自 kernel。Phase B 分析必须包括：
- forward 签名是否与 `model.py` 的 `forward()` 一致（输入 tensor 数量、顺序）
- 顶部是否正确导入编译产物（`import _xxx_ext`，模块名与 `PYBIND11_MODULE` 一致）
- forward 中是否有残留的 PyTorch 计算（退化检测）
- tiling 参数传递是否与 `pybind11.cpp` 的 launch 函数签名对齐

### 2.5 L8 op_type 推断限制

OpenOps 有 `{op_name}_op_desc.json` 提供结构化算子描述；AscendOpGenAgent 没有等价文件。

`model.py` 里的 `forward()` 和 `get_input_groups()` 可作为推断来源，但对 fused op、model-dependent 输入，推断结果不稳定。

**处理方式**：forensics.py 保留基于算子名做关键词匹配的简单推断（如 `pool`、`norm`、`matmul`），当置信度低时 L8 标注 `source: "name_heuristic"`，Phase A 路由退化为"读最近似案例 + 让 agent 自行判断是否适用"，不强制匹配。

---

## Section 3：Phase A 参考文件替换

### 3.1 文件替换表

| OpenOps 原文件 | 替换为 |
|---|---|
| `dsl-lowering/references/lowering_examples/{op_type}/` | `archive_tasks/` 中有完整 `kernel/` 的成功案例（见路由表） |
| `error_correction_examples.md` | `skills/ascendc/ascendc-translator/references/dsl2Ascendc.md` |
| `non_aligned_example_for_lowering.md` | `skills/ascendc/ascendc-translator/references/dsl2Ascendc_compute_vector.md` |
| `tl_asc_routing.md` | `skills/ascendc/ascendc-translator/references/TileLang-AscendC-API-Mapping.md` + `AscendC_knowledge/api_reference/` |

### 3.2 archive_tasks 路由表（仅含有完整 kernel/ 的案例）

| op_type 类别 | 参考案例 | 说明 |
|---|---|---|
| pooling | `archive_tasks/avg_pool3_d/` | 多变体 pooling，不适用于 generic reduction |
| normalization / rmsnorm / layernorm | `archive_tasks/rms_norm/` | 含 `vector_tile.h`，多变体 |
| matmul / gemm / linear | `archive_tasks/matmul_leakyrelu/` 或 `quant_matmul/` | matmul_leakyrelu 是 C/V 融合，quant_matmul 更接近 pure matmul |
| gather / scatter / index | `archive_tasks/gather_elements_v2/` | 多 dtype 变体 |
| concat / memory layout | `archive_tasks/concat_dv2/` | 多 dim 变体 |
| quantized matmul | `archive_tasks/reshape_matmul_rowwise_quant_int8/` 或 `quant_matmul/` | |
| **attention / softmax** | **无有效 AscendC 案例** | `flash_attention/` 和 `sparse_attention/` 均无 `kernel/`，跳过路由，Phase A 只读 API 文档 |
| **纯 elementwise / activation** | **无纯向量案例** | `matmul_leakyrelu` 是 fused，不宜作为纯 elementwise 参考；退化为只读 `dsl2Ascendc_compute_vector.md` |
| 无精确匹配 | 按计算模式选最近似（pooling→avg_pool3_d，normalization→rms_norm，matmul→matmul_leakyrelu），并在 [REFERENCE_IMPL_SPEC] 中标注"参考案例非精确匹配" | |

`[REFERENCE_IMPL_SPEC]` 结构不变，仍检查同样 5 个维度：TQue/TBuf 数据流、work_buf 初始化、DataCopy 对齐、SyncAll、禁用模式。

---

## Section 4：evaluate_ascendc.sh 执行模型与失败分类

### 4.1 执行模型

`evaluate_ascendc.sh {task_name}` 不是本地单步命令。其实际行为：
1. 检测本地是否有 `torch_npu` 环境
   - 若有：本地 `build_ascendc.py` 编译 + `verification_ascendc.py` 验证
   - 若无：打包整个 repo → scp 到远端 → docker exec 跑编译 + 验证

SKILL.md Step 4 中的"重新编译 + 精度验证"合并为一步：
```bash
bash skills/ascendc/ascendc-translator/references/evaluate_ascendc.sh {task_name}
```
不再拆分 install + compile + evaluate 三步。

### 4.2 失败分类

`evaluate_ascendc.sh` 的失败必须按类别处理，不同类别对应不同修复策略：

| 类别 | 特征 | 修复策略 |
|---|---|---|
| **Infra 失败** | SSH 超时、docker exec 失败、scp 错误 | 停止，报告环境问题，不进入修复循环 |
| **Build 失败** | `build_ascendc.py` 报编译错误（undefined symbol、type mismatch） | 修复 kernel `.cpp`/`.h`，最多 3 次 |
| **Import 失败** | `verification_ascendc.py` import 阶段报错（模块名不一致、pybind 注册失败） | 检查 `model_new_ascendc.py` import 名 vs `PYBIND11_MODULE` 名 |
| **Numerical 失败** | `verification_ascendc.py` 报 mismatch（match_rate < 100%） | 进入 precision_forensics → 审计 → 修复循环 |

Gate-F（取证前置检查）仅对 Numerical 失败有意义；Build/Import 失败直接进对应修复路径。

---

## Section 5：Failure Ingestion — forensics.py 数据来源

OpenOps 的 `precision_forensics.py` 通过直接 import `{op_name}_custom.py` 跑推理获得数值差异。

AscendOpGenAgent 的验证由 `utils/verification_ascendc.py` 负责，有两种对接方式：

**方案 A（推荐）：解析 verification_ascendc.py stdout**

`verification_ascendc.py` 已经输出 per-case 的 mismatch 信息（match_rate、max_diff、worst index 等）。`precision_forensics.py` 在 evaluate_ascendc.sh 完成后解析其 stdout，提取 L0-L4 层的数值信息，写入 `forensics_report_{attempt}.json`。

- 不需要重跑推理
- stdout 格式需对齐（检查 verification_ascendc.py 的实际输出格式，如有不足需 patch 加 `--verbose` 选项）

**方案 B（备选）：直接调用 verification_ascendc.py 加 `--report` flag**

如果 stdout 解析不够精确，patch `verification_ascendc.py` 增加 `--report {path}` 参数，直接输出 machine-readable JSON。

**当前选择方案 A**，如果 stdout 字段不够，再升级到方案 B。

---

## Section 6：Gate-X 重写

原 Gate-X 检查 `{OpName}Custom/op_kernel/*.cpp` 存在性，需完全重写：

```python
# Gate-X 检查项（AscendOpGenAgent 版）
checks = {
    "pybind11_cpp_exists": os.path.exists(f"{task_dir}/kernel/pybind11.cpp"),
    "pybind11_has_module": "PYBIND11_MODULE" in open(f"{task_dir}/kernel/pybind11.cpp").read(),
    "has_non_pybind_cpp": any(
        f.endswith(".cpp") and f != "pybind11.cpp"
        for f in os.listdir(f"{task_dir}/kernel/")
    ),
    "model_new_ascendc_exists": os.path.exists(f"{task_dir}/model_new_ascendc.py"),
    "import_name_consistent": _check_import_name_match(task_dir),  # 比对 PYBIND11_MODULE 名 vs import _xxx_ext
}
```

---

## Section 7：DSL 删除清单

**Sub-step 2.2 `[COMPUTATION_DECOMPOSITION]`**：
- 删除：`{op_name}_dsl.py` 读取步骤
- 删除：DSL tiling 策略摘要（n_cores、tile_length 等字段）
- 删除：`DSL 文件状态` / `DSL 对应代码` 标注
- 保留：计算链分解（从 `model.py` forward() 推断）、计算模式标注、跨核通信分析

**Sub-step 2.3 Phase B**：
- 删除：`Host vs DSL 是否一致` 对比
- 保留：读 `kernel/{op_name}_tiling.h` 的 TilingData 结构体定义

**`[KERNEL_STEP_TRACE]` Host tiling 参数块**：
- 删除：`DSL 中对应参数` 行、`Host vs DSL 是否一致` 行
- 保留：其余所有字段

**取证报告 `available_files`**：
- 删除：`dsl` 字段
- 删除：`op_desc` 字段

---

## Section 8：trace-recorder 对齐

修复迭代结果（每轮的 attempt、match_rate、fix_type、loop_signal）需写入 `{task_name}/trace.md`，与现有 `trace-recorder` skill 的格式对齐。

具体做法：precision-tuning 的 Step 5/6 收尾时，调用 `trace-recorder` skill 追加一个 `## 精度调优` section，包含：
- 调优轮次总数
- 每轮 fix_type + outcome
- 最终 match_rate + max_diff
- loop_stop_reason（若 STOP）

---

## 未来扩展点

- TileLang 编译失败的 debug 支持（新增 Phase 0 检测分支，触发前先判断是 TL 还是 AscendC 失败）
- archive_tasks 路由表随新成功案例持续扩充（特别是 attention/softmax 类，等 flash_attention_gqa 有 kernel/ 后补充）
- verification_ascendc.py `--report` flag（方案 B），提升 forensics 数值精度
