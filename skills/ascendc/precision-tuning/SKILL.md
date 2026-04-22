---
name: precision-tuning
description: >
  修复编译通过但精度测试失败的 AscendC 算子。
  通过数值取证 + Agent 深度分析 + 代码修复 + 重新验证的循环实现精度调优。
subagent:
  enabled: true
  agent_type: general
  reason: >
    覆盖 AscendC build / import / runtime / timeout / precision 五类失败。
    每类失败都涉及取证→深度分析→修复→验证的多步循环,
    需要 Agent 结合数值/日志证据和代码理解做深度推理。
    subagent 内部按 final_status.failure_type 锁定一条分支, 不跨分支跳转。
  timeout: 5400
  max_iterations: 60
---

## What I do

修复 AscendC 算子的 **build / import / runtime / timeout / precision** 五类失败。流程:
1. 读取 `{task_dir}/.eval_status/latest.json` 确定 `failure_type`, 分流到对应 Step 1 分支（一次 session 锁定一条分支）
2. Agent 结合上下文 + 代码 + 日志 / 数值证据 + 知识库做深度分析, 定位根因并制定修复计划
3. Agent 修复代码（只能改 `{task_dir}/kernel/` 下文件）
4. 重新编译 + 验证（通过 `utils/eval_wrapper.py --phase 8`）
5. 根据 Gate 循环控制信号决定继续或停止

## When to use me

当主 agent Phase 7 `trace-recorder` 判定 `debug_eligible == true`（含 `failure_type` 白名单判断）时。
主 agent Phase 8 会在此条件下 spawn 本 subagent；不满足条件的任务（`success` / `degraded` / `no_kernel` /
`tilelang_only_failed` / `execution_aborted` / `import_failed` 的 `import_env_side` 子类）**不应**进入本 skill。

## Prerequisites（通用，所有分支共享）

- `{task_dir}/kernel/` 下至少一个 `.cpp` 文件（保证有 kernel 可修）
- `{task_dir}/model_new_ascendc.py` 未 AST 退化（反作弊前置条件）
- `{task_dir}/trace.md` 末尾含 `final_status` fenced JSON block（Phase 7 产出）
- `{task_dir}/.eval_status/latest.json` 存在（`utils/eval_wrapper.py` 产出）
- `{task_dir}/{op_name}.json`（及可选 `.json.bak`）存在

其中 `task_dir = {repo_root}/{task_name}`，`repo_root` 为 AscendOpGenAgent 仓库根目录。

> **分支专属前提**（进入 Step 1-X 后由各分支自校验）：
> - **1-P（precision_failed）**：`model.py` 参考实现、`kernel/pybind11.cpp` 编译通过且能运行
> - **1-B（build_failed）**：`.eval_logs/phase{N}_attempt{M}.log` 含 compile error 块
> - **1-I（import_failed + import_kernel_side）**：import traceback 指向 pybind 符号 / ext module；`import_env_side` 不进入本 skill
> - **1-R（runtime_error）**：execute 阶段有明确 crash signal (SIGSEGV/SIGABRT/SIGBUS/SIGFPE)
> - **1-T（timeout）**：`.eval_status` 含 `timeout_marker_present == true`

## Workflow

**所有思考、分析、推理必须使用中文。**

**核心原则: Python 脚本做确定性操作 (取证、Gate、知识库), Agent 做需要推理的工作 (分析、修复)。**

---

### Step 0: 初始化

设置轮次计数器 `attempt = 0`。

**0.1 保存不可变基线快照（原始代码，仅首次执行）:**
```bash
if [ ! -d "{task_dir}/precision_tuning/history/baseline/code_snapshot" ]; then
    mkdir -p "{task_dir}/precision_tuning/history/baseline/code_snapshot"
    cp -r "{task_dir}/kernel/" \
       "{task_dir}/precision_tuning/history/baseline/code_snapshot/kernel/"
    cp "{task_dir}/model_new_ascendc.py" \
       "{task_dir}/precision_tuning/history/baseline/code_snapshot/model_new_ascendc.py"
    echo "基线快照已保存，后续可从 baseline 恢复"
fi
```

> 基线快照保存在 `history/baseline/code_snapshot/`，整个调优过程中**不覆盖**。如需恢复到最初始状态，使用以下命令：
> ```bash
> cp -r "{task_dir}/precision_tuning/history/baseline/code_snapshot/kernel/" \
>    "{task_dir}/kernel/"
> cp "{task_dir}/precision_tuning/history/baseline/code_snapshot/model_new_ascendc.py" \
>    "{task_dir}/model_new_ascendc.py"
> ```

**0.2 保存本轮起始快照:**
```bash
mkdir -p "{task_dir}/precision_tuning/history/attempt_0/code_snapshot"
cp -r "{task_dir}/kernel/" \
   "{task_dir}/precision_tuning/history/attempt_0/code_snapshot/kernel/"
cp "{task_dir}/model_new_ascendc.py" \
   "{task_dir}/precision_tuning/history/attempt_0/code_snapshot/model_new_ascendc.py"
```

> 知识库将在 Sub-step 2.1 完成后通过 `search` 命令按需检索, 无需在此全量加载。

---

### Step 0.3: 读 final_status + eval_status，锁定 session_branch

```bash
# 读结构化 eval_status（最后一次 evaluate 的 failure_type / failed_step / log 路径）
python3 skills/ascendc/precision-tuning/scripts/eval_status.py \
    --task-dir {task_dir} | jq '.failure_type, .import_subtype, .timeout_marker_present'

# 读 trace.md 末尾的 final_status JSON block（Phase 7 写入, 本 session 期间只读）
awk '/^```json/,/^```$/' "{task_dir}/trace.md" | jq '.debug_eligible, .failure_type, .import_subtype'
```

**分支选择规则（本 session 唯一锁定，不可切换）**：

- 入口按 `final_status.failure_type` 选定 `session_branch`
- `session_branch` 在整个 subagent session 中**只锁定一次**，之后 Step 1 / 2 / 3 / 4 / 5 / 6 都基于这条分支的 Gate 语义执行
- `import_failed` 还要读 `final_status.import_subtype`：
  - `import_kernel_side` → 进入 Step 1-I
  - `import_env_side` → 异常情况（主 agent 已过滤）；直接写 `debug_trace.md` + `debug_status.json` 标 `phase8_outcome: skipped_env_issue` 后退出

**跨分支跳转禁止（v3 硬约束）**：

- 若某轮修复后 `eval_status.failure_type` 变化（如 `build_failed` → `precision_failed`），视为"本分支 Gate-V 取得进展"
- **不切换分支**，本次 session 结束；`debug_trace.md` / `debug_status.json` 标 `phase8_outcome: progressed_to_new_failure_type`
- 主 agent 本版本**不自动二次 spawn**（留给人工判断）
- 原因：跨分支会导致 audit schema、Gate 语义、`debug_trace.md` 模板同时漂移，风险远大于收益

**根据 `session_branch` 选择 Step 1 分支**：

| session_branch | failure_type | 进入 |
|---|---|---|
| `1-P` | `precision_failed` | Step 1-P（现有精度取证路径） |
| `1-B` | `build_failed` | Step 1-B |
| `1-I` | `import_failed` + `import_kernel_side` | Step 1-I |
| `1-R` | `runtime_error` | Step 1-R |
| `1-T` | `timeout` | Step 1-T |
| — | 其他（`success` / `degraded` / `no_kernel` / `tilelang_only_failed` / `execution_aborted` / `import_env_side`） | 写 `debug_status.json` 标 `skipped_unsupported_type`，退出 |

> **硬约束重申**：`trace.md` 在 Phase 7 写完后**全程只读**。本 skill 所有产出（`debug_trace.md` / `debug_status.json` / `.eval_status/phase8_*` / `precision_tuning/*`）都写到 `{task_dir}` 下独立文件，**禁止 append `trace.md`**（findings §7.3）。

---

### Step 1-P: 精度取证（precision_failed 分支）

#### 1.1 精度取证 (Python 脚本, 不可跳过)

```bash
python3 skills/ascendc/precision-tuning/scripts/precision_forensics.py \
    {task_name} --attempt {attempt}
```

Gate 验证:
```bash
python3 skills/ascendc/precision-tuning/scripts/precision_gate.py \
    --step forensics --op-name {op_name} --task-name {task_name} --attempt {attempt}
```

⛔ **Gate-F 未通过 → 停止, 检查错误输出。不要在没有取证数据的情况下分析代码。**
如果报错含 `FileNotFoundError`，先确认 `{task_dir}/kernel/pybind11.cpp` 存在，再检查 `utils/verification_ascendc.py` 路径。

> 1-P 分支继续走 Step 2（精度深度分析 4 Sub-step）→ Step 3（修复）→ Step 4（重编译+验证，走 Gate-V 的精度语义）→ Step 5/6。

---

### Step 1-B: Build Error Analysis（build_failed 分支）

**输入**:
- `{task_dir}/.eval_status/latest.json` — 结构化状态 + `log_path` + `compile.error_summary`
- `{task_dir}/.eval_logs/phase{N}_attempt{M}.log` — 原始 build log（compile 阶段 stderr 全文）
- `{task_dir}/kernel/*.cpp` / `*.h` — 当前 kernel 源码
- `{task_dir}/trace.md` — Phase 1-7 上下文（Phase 4 ac_iterations 里记录了历次 build 尝试）

**Agent 任务**:
1. 读 build log，提取 compile error / fatal error / undefined reference / template instantiation error 块（每块最多 10 行，按 stderr 顺序）
2. 对照 `skills/ascendc/ascendc-translator/references/dsl2Ascendc_compute_*.md`、`dsl2Ascendc_host.md`、`TileLang-AscendC-API-Mapping.md` 找 API 用法差异（签名、模板参数、include 依赖）
3. 写 `{task_dir}/precision_tuning/precision_audit_{attempt}.md` 含（Gate-BUILD-A 必填）：
   - `[COMPILE_ERROR_CITATION]` — 原文摘录 error 块 + 对应 `kernel/*.cpp` 行号（引用不少于 1 处 error）
   - `[ROOT_CAUSE]` — 根因（签名不匹配 / 模板参数错 / include 缺失 / pipe-queue 协议违反 等）
   - `[FIX_PLAN]` — 文件 / 函数 / 行号级修改列表
   - `[FIX_TYPE]` — 必须 ∈ `{api_usage_fix, template_arg_fix, include_fix, signature_align_fix, pipe_queue_fix, tilingdata_field_fix}`；不在白名单的类型 Gate-A 直接 reject
4. 修改 `{task_dir}/kernel/*.cpp` / `*.h`（**绝对不动** `utils/build_ascendc.py` / `CMakeLists.txt` / `setup.py` / `utils/` 下任何文件）
5. 通过 Gate-通用 + Gate-BUILD-A 验证：
   ```bash
   python3 skills/ascendc/precision-tuning/scripts/precision_gate.py \
       --step audit --op-name {op_name} --task-name {task_name} --attempt {attempt}
   ```

**推荐参考资料**:
- `skills/ascendc/ascendc-translator/references/dsl2Ascendc_compute_vector.md`（向量 API）
- `skills/ascendc/ascendc-translator/references/dsl2Ascendc_compute_scalar.md`（标量 API）
- `skills/ascendc/ascendc-translator/references/dsl2Ascendc_host.md`（host 侧 tiling / workspace）
- `skills/ascendc/ascendc-translator/references/TileLang-AscendC-API-Mapping.md`（API 权威参考）
- `skills/ascendc/ascendc-translator/references/AscendC_knowledge/api_reference/`（API 详细文档）

**Step 4（共用）**: 修复后调 `utils/eval_wrapper.py --phase 8 --attempt {attempt} --task-dir {task_dir}` 重跑，然后走 `Gate-BUILD-V`：
- `eval_status.failed_step` 从 `compile` 推进到 `import`/`execute`/`verify`/`null` = 进步（跨分支语义下仍算 `progressed_to_new_failure_type`，本 session 结束）
- 仍卡在 `compile` 且 error 行未变 = 停滞
- `compile` 阶段 passed 且 `failure_type != build_failed` = 本分支完成（不切分支，写 `debug_status.json` 后退出）

---

### Step 1-I: Import Error Analysis（import_failed + import_kernel_side 分支）

**输入**:
- `{task_dir}/.eval_status/latest.json` — 结构化状态，确认 `import_subtype == import_kernel_side`（若是 `import_env_side` 应已被 Step 0.3 过滤）
- `{task_dir}/.eval_status/import_traceback.log` 或 `.eval_logs/phase{N}_attempt{M}.log` — 原始 import traceback
- `{task_dir}/kernel/pybind11.cpp` — pybind 注册入口（`PYBIND11_MODULE` 名、导出符号）
- `{task_dir}/kernel/*_kernel.h` / `*.cpp` — 被 pybind 引用的 kernel 符号
- `{task_dir}/model_new_ascendc.py` — import 的 ext module 名（只读！不可改）
- `{task_dir}/trace.md` — 上下文

**Agent 任务**:
1. 读 traceback，定位缺失的符号 / 模块名 / pybind 入口
2. 对照 `skills/ascendc/ascendc-translator/references/dsl2Ascendc_host.md`（pybind 章节）核对：
   - `PYBIND11_MODULE` 的第一个参数（模块名）是否与 `model_new_ascendc.py` 中 `import` 的名字一致
   - kernel ext 的 `.so` 文件命名与 import 名是否匹配
   - 导出的函数符号是否与 `pybind11.cpp` 中 `m.def(...)` 注册的名字一致
3. 写 `{task_dir}/precision_tuning/precision_audit_{attempt}.md` 含（Gate-IMPORT-A 必填）：
   - `[IMPORT_TRACEBACK_CITATION]` — 原文摘录 traceback（至少 `ImportError` / `ModuleNotFoundError` / `OSError: cannot open shared object` 的关键行）
   - `[ROOT_CAUSE]` — 根因（pybind 模块名不一致 / kernel ext 名称错 / 符号未导出）
   - `[FIX_PLAN]` — 修改点（限定 `pybind11.cpp` 的 `PYBIND11_MODULE` / `m.def` 注册行，或 kernel 侧 `extern "C"` / 导出符号名）
   - `[FIX_TYPE]` — 必须 ∈ `{pybind_symbol_fix, kernel_ext_name_fix, kernel_export_fix}`；**明确拒绝** `ld_path_fix` / `abi_fix` / `toolkit_env_fix` / `cmakelists_fix` / `setup_py_fix` / `build_ascendc_fix`（这些属于 env_side，不在本 subagent 的 scope）
4. 修改 `{task_dir}/kernel/pybind11.cpp` 或 kernel 符号导出处（**不动** `model_new_ascendc.py`、`utils/build_ascendc.py`、`CMakeLists.txt`、`setup.py`）
5. 通过 Gate-通用 + Gate-IMPORT-A 验证（命令同 Step 1-B）

**推荐参考资料**:
- `skills/ascendc/ascendc-translator/references/dsl2Ascendc_host.md`（pybind 绑定规范）
- `skills/ascendc/precision-tuning/references/`（若有环境变量 / pybind 相关条目）
- `skills/ascendc/ascendc-translator/references/TileLang-AscendC-API-Mapping.md`（`extern "C"` 导出规范）

**Step 4（共用）**: 修复后调 `utils/eval_wrapper.py --phase 8 --attempt {attempt} --task-dir {task_dir}` 重跑，然后走 `Gate-IMPORT-V`：
- `eval_status.import.status == passed` = 本分支完成
- 仍卡在 `import` 且 traceback 未变 = 停滞
- `import` 通过但 `failure_type` 变为 `build_failed` / `runtime_error` / `precision_failed` = 进步但跨分支，本 session 结束

---

### Step 1-R: Runtime Error Analysis（runtime_error 分支）

**输入**:
- `{task_dir}/.eval_status/latest.json` — 结构化状态 + `execute.crash_signal`（SIGSEGV / SIGABRT / SIGBUS / SIGFPE）
- `{task_dir}/.eval_logs/phase{N}_attempt{M}.log` — stderr / stack trace / core dump 信息
- `{task_dir}/kernel/*.cpp` / `*.h` — kernel 源码
- `{task_dir}/trace.md` — 上下文

**Agent 任务**:
1. 读 stderr / stack trace，提取 crash 位置（函数名 / 行号 / 同步点）；若 log 中只有 signal 编号没有 stack trace，结合 `crash_signal` 类型定位可能原因：
   - `SIGSEGV` → 越界访存（UB / GM 访存越界、Tensor 未分配就读取、TQue 协议违反）
   - `SIGABRT` → assertion 失败 / 运行时检查失败（AscendC runtime 内部 check）
   - `SIGBUS` → 内存对齐错（未满足 32/64/128 字节对齐）
   - `SIGFPE` → 除零 / 浮点异常（tiling 参数为 0、分母未保护）
2. 对照 `skills/ascendc/ascendc-translator/references/dsl2Ascendc_cross_core_sync.md`、`AscendCVerification.md`、`dsl2Ascendc_compute_*.md` 找 API 约束 / 同步点 / 对齐要求
3. 写 `{task_dir}/precision_tuning/precision_audit_{attempt}.md` 含（Gate-RUNTIME-A 必填）：
   - `[RUNTIME_ERROR_CITATION]` — 原文摘录 stderr / stack trace（含 crash_signal、函数名、行号）
   - `[ROOT_CAUSE]` — 根因（越界 / 对齐 / 同步缺失 / TQue 协议违反 / 除零）
   - `[FIX_PLAN]` — 文件 / 函数 / 行号级修改列表
4. 修改 `{task_dir}/kernel/*.cpp` / `*.h`
5. 通过 Gate-通用 + Gate-RUNTIME-A 验证

**推荐参考资料**:
- `skills/ascendc/ascendc-translator/references/dsl2Ascendc_cross_core_sync.md`（跨核同步 / SyncAll）
- `skills/ascendc/ascendc-translator/references/AscendCVerification.md`（runtime 语义与验证）
- `skills/ascendc/ascendc-translator/references/dsl2Ascendc_compute_vector.md`（对齐与 DataCopyPad）

**Step 4（共用）**: 修复后调 `utils/eval_wrapper.py --phase 8 --attempt {attempt} --task-dir {task_dir}` 重跑，然后走 `Gate-RUNTIME-V`：
- `eval_status.failure_type != runtime_error` 或 crash 位置 / signal 变化 = 进步（若仍是 runtime_error 但位置变则视为 `progressed`）
- `failure_type` 变为 `precision_failed` = 进步但跨分支，本 session 结束

---

### Step 1-T: Timeout Analysis（timeout 分支）

**输入**:
- `{task_dir}/.eval_status/latest.json` — 结构化状态；必须满足 `failure_type == timeout` 且 `timeout_marker_present == true`（否则视为 `execution_aborted`，不应进本分支）
- `{task_dir}/.eval_logs/phase{N}_attempt{M}.log` — 超时前的 stdout/stderr 尾部（最后一条日志提示死锁 / 死循环位置）
- `{task_dir}/kernel/*.cpp` / `*.h` — kernel 源码（重点看 `SyncAll` / `WaitFlag` / `SetFlag` / `for` 循环边界）
- `{task_dir}/kernel/{op_name}_tiling.h` + `kernel/pybind11.cpp` — tiling 配置（block_dim / tile_size）
- `{task_dir}/trace.md` — 上下文

**Agent 任务**:
1. 读 log 尾部，定位超时时 kernel 执行到哪一步（若能判断）；结合 `duration_sec` 与预期耗时量级判断是死锁（duration ≈ timeout 阈值且无输出推进）还是性能降级
2. 对照 `skills/ascendc/ascendc-translator/references/dsl2Ascendc_cross_core_sync.md` 分析：
   - `SyncAll` 是否遗漏或多余（多余的 SyncAll 在部分核未到达时死锁）
   - `SetFlag` / `WaitFlag` 配对是否一致
   - Tiling 参数（`block_dim`、`tile_size`、归约轴切分）是否导致循环不收敛
3. 写 `{task_dir}/precision_tuning/precision_audit_{attempt}.md` 含（Gate-TIMEOUT-A 必填）：
   - `[SYNC_POINT_ANALYSIS]` — 枚举 kernel 中所有同步点（`SyncAll` / `SetFlag` / `WaitFlag`）及其配对关系，标出疑似死锁点
   - `[ROOT_CAUSE]` — 根因（同步缺失 / 同步多余 / 死循环 / tiling 死锁）
   - `[FIX_PLAN]` — 文件 / 函数 / 行号级修改列表
4. 修改 `{task_dir}/kernel/*.cpp` / `*.h`（**不动** tiling host 逻辑若超出 `kernel/pybind11.cpp` 的 TilingFunc）
5. 通过 Gate-通用 + Gate-TIMEOUT-A 验证

**推荐参考资料**:
- `skills/ascendc/ascendc-translator/references/dsl2Ascendc_cross_core_sync.md`（同步原语与死锁反模式）
- `skills/ascendc/ascendc-translator/references/dsl2Ascendc_host.md`（tiling / workspace 分配）
- `skills/ascendc/ascendc-translator/references/AscendCVerification.md`（runtime 约束）

**Step 4（共用）**: 修复后调 `utils/eval_wrapper.py --phase 8 --attempt {attempt} --task-dir {task_dir}` 重跑，然后走 `Gate-TIMEOUT-V`：
- `eval_status.duration_sec < timeout_threshold` 且 `failure_type != timeout` = 本分支完成（无论对错 — 精度对错由后续判断，但 timeout 语义已解除）
- 仍超时且 duration 基本不变 = 停滞
- 不再超时但转 `runtime_error` / `precision_failed` = 进步但跨分支，本 session 结束

---

### Step 2: 深度分析 + 修复计划 (Agent 推理, 核心步骤)

**本步骤分为 4 个 Sub-step, 每个 Sub-step 有明确的输入文件和产出 section, 不可跳过或合并。**

将全部分析结果写入 `{task_dir}/precision_tuning/precision_audit_{attempt}.md`。

**历史扫描（attempt > 0 时必须执行，首轮跳过）：**

**第一步：读方向学习表（一次读完，直接获得跨轮全貌）**
```bash
cat "{task_dir}/precision_tuning/tuning_directions.json"
```

从 `tuning_directions.json` 一次性获得：
- 每轮的 `fix_type`（哪些修复类型已被尝试）
- `outcome`（passed / improved / stagnant / regressed）— 快速判断方向是否有效
- `improvement_ratio`（数值趋势一览）
- `direction_verdict`（是否曾切换方向）
- `forensics_hint`（每轮取证信号）
- `final_status`（in_progress / success / failed）

> ⚠️ **禁止重复已证实无效的方向**：outcome 为 regressed 或连续 stagnant 的 fix_type，本轮不得再用。

**第二步：按需深入（仅在确实需要时通过 round_summary 的 index 定位）**
```bash
# 读某轮的 round_summary 获取文件路径索引
cat "{task_dir}/precision_tuning/round_summary_{N}.json"
# 再按 index.sections.* 路径读对应的 section 小文件
```

- 想了解某轮根因细节 → 读 `round_summary_N.index.sections.root_cause` 指向的文件
- 想了解某轮修复计划 → 读 `round_summary_N.index.sections.fix_plan` 指向的文件
- 想查看完整审计 → 读 `round_summary_N.index.audit_full` 指向的文件

**禁止**：不得跳过 `tuning_directions.json` 直接全量读 `history/attempt_*/precision_audit.md`。

---

#### Sub-step 2.1: 取证数据解读

**读取**: `{task_dir}/precision_tuning/forensics_report_{attempt}.json`

**可选前置读取（仅 attempt == 0 且文件存在）**: `{task_dir}/trace.md`

> trace.md 由 `ascend-kernel-developer` 在生成阶段产出，记录 Phase 4 AscendC 转译的迭代历史、走偏点、已知平台/API 限制、kernel 结构意图。读取它可以**避免重蹈生成阶段已走偏的方向**，并补全 kernel 设计背景。文件不存在时跳过，Gate-A 不强制。

**产出**: `[FORENSICS_SUMMARY]` section + `[PRIOR_TRACE_CONTEXT]`（可选，仅首轮 + trace.md 存在时）

逐字段摘录取证报告中的关键数值, 不允许跳过任何字段:

```
=== PRECISION AUDIT REPORT ===

[FORENSICS_SUMMARY]
  取证数据摘要 (L0-L4):
    - primary_hint: <来自取证 primary_hint>
    - primary_confidence: <来自取证 primary_confidence>
    - primary_evidence: <来自取证 primary_evidence>
    - mismatch_ratio: <来自取证 outputs[0].basic_stats.mismatch_ratio>
    - max_abs_diff: <来自取证 outputs[0].basic_stats.max_abs_diff>
    - mean_abs_diff: <来自取证 outputs[0].basic_stats.mean_abs_diff>
    - error_distribution: <来自取证 outputs[0].error_distribution, 特别关注 sign_analysis.bias_direction>
    - worst 元素位置: <来自取证 outputs[0].worst_elements, 列出 top 3>
    - 尾块分析: <来自取证 outputs[0].tail_analysis, 标注各 tile_size 下的 tail/body mismatch rate>
    - 维度分析: <来自取证 outputs[0].dimension_analysis, 标注各维度的 mismatch_rate 范围>
  L6 内存布局:
    - 输入 tensor layout: <来自取证 L6_memory_layout.inputs, 标注 shape/stride/对齐>
    - 最后一维对齐情况: <是否对齐 8/16/32/64/128/256>
  L8 算子类型:
    - op_type: <来自取证 L8_operator.op_type>
    - source: <来自取证 L8_operator.source>
    - attributes: <来自取证 L8_operator.attributes, 特别关注 dim/reduction/kernel_size 等>
    - reduction_axis: <来自取证 L8_operator.reduction_axis, 如果有>
    - 该类型的 checklist: <将在下方 search 命令中自动返回>
  可用文件:
    - reference: <来自取证 available_files.reference>
    - custom: <来自取证 available_files.custom>
  dtype 精度级别判断:
    - dtype: <来自取证 outputs[0] 或 L8_operator, 如 float32/float16/bfloat16>
    - max_abs_diff (来自取证): <值>
    - 精度阈值参考 (来自 ascend-torch-comparison/precision_config.py AbsoluteThreshConfig):
      * float32 rtol=1e-4: max_diff > 1e-4 → 逻辑错误; ≤ 1e-4 → 精度达标
      * float16 rtol=1e-3: max_diff > 1e-2 → 逻辑错误; 1e-3~1e-2 → float16 精度损失(可能可接受); ≤ 1e-3 → 精度达标
      * bfloat16 rtol=5e-3: max_diff > 5e-2 → 逻辑错误; 5e-3~5e-2 → bfloat16 精度损失; ≤ 5e-3 → 精度达标
    - 判断: <逻辑错误(实现缺陷, 必须修复) / float16精度损失(检查 float32 下是否通过) / 精度达标>
    - 对分析方向的影响: <逻辑错误→重点查实现缺陷; float16精度损失→检查归约是否需要 upcast>
  我对取证 hint 的初步判断:
    - 取证给出的 hint 是否合理? <结合数值证据判断, 不要在此步做代码分析>
    - 是否有数值异常未被 hint 覆盖? <如 sign_analysis 显示偏向但 hint 未提及>
```

**可选段（仅当 attempt == 0 且 `{task_dir}/trace.md` 存在时写入）**:

```
[PRIOR_TRACE_CONTEXT]
  来源: {task_dir}/trace.md (ascend-kernel-developer 生成阶段产出)
  最终结果: <如 "SKIP (tilelang) | FAIL (ascendc)" 或 "PASS">
  Phase 4 AscendC 迭代次数: <evaluate_ascendc.sh 执行次数>

  已尝试方向（本轮修复时避免重复）:
    - 第 N 轮: <一句话总结做了什么、结果如何>
    - ...

  走偏点记录（trace.md "走偏点" 章节原文提炼）:
    - <如 "把 device kernel 写成模板入口导致 host stub 找不到实际符号">
    - <如 "Muls 在 bfloat16 下不支持, 当前平台 API 限制">

  剩余未解决的平台/API 限制:
    - <trace.md 揭示的硬性约束, 如 "当前平台 Muls 不支持 __bf16">

  kernel 结构要点（若 trace.md 提及）:
    - <如 "分 fp32/fp16/bf16 三个独立入口">
```

> 如 trace.md 不存在, 省略此 section, 不影响 Gate-A。写入时只摘录关键点, **不要**粘贴 trace.md 全文或长代码块。

**知识库检索 (第一次 — 基于取证 hint + 算子类型):**

从 `[FORENSICS_SUMMARY]` 中提取 `primary_hint` 和 `op_type`, 检索相关知识条目:
```bash
python3 skills/ascendc/precision-tuning/scripts/precision_knowledge.py search \
    --kb-path skills/ascendc/precision-tuning/references/precision_knowledge_base.json \
    --op-type <L8_operator.op_type> \
    --pattern <primary_hint> \
    --top-k 3 \
    --log-path "{task_dir}/precision_tuning" \
    --attempt {attempt} \
    --call-index 0
```

记住检索到的 `matched_entries` 和 `checklists`, 后续分析时参考。
如果输出 `fallback_to_full_load: true`, 说明无精确匹配, 已返回全量条目。

---

#### Sub-step 2.2: 算子计算流程分解

**读取** (按顺序):
1. `{task_dir}/model.py` — 参考实现的 forward() 逻辑
2. `archive_tasks/` 中最近似案例的 `model.py`（可选，用于对比计算链）

**产出**: `[COMPUTATION_DECOMPOSITION]` section

**要求**:
- 参考 `decomposition_examples/` 中最匹配的示例, 按其格式和粒度分解
- 如果有完全匹配的示例, 引用其步骤结构并根据当前算子的具体参数填充
- 如果没有匹配的示例, 按 README.md 的粒度标准自行分解
- 每步必须包含: 操作名、输入来源、输出 shape、数值范围预期、精度风险点
- 如果 DSL 存在, 每步标注 DSL 对应代码
- 标注算子计算模式: 单行归约 / 跨核归约 / 分块累加 / 滑窗累加 / 前缀累加 / 逐元素

```
[COMPUTATION_DECOMPOSITION]
  算子: {op_name}
  计算模式: <单行归约 / 跨核归约 / 分块累加 / 滑窗累加 / 前缀累加 / 逐元素>
  参考分解示例: <使用的示例文件名, 如 softmax.md, 或 "无匹配, 自行分解">
  归约维度: <dim={dim}, axis={axis}, 归约轴长度={length}> (如适用)
  数据类型: <dtype>

  计算链:
    Step 0: 输入
      - shape: <input_shape>
      - 数值范围: <来自取证 value_range>

    Step 1: <operation_name>
      - 来源: reference.py 中的 <具体函数/表达式>
      - 输入: <上一步输出 / 原始输入>
      - 输出 shape: <shape>
      - 数值范围预期: <基于输入范围推断>
      - 精度风险点: <该步可能引入误差的原因>
      - 知识库关联: <匹配的条目编号和标题, 或 "无">

    Step 2: <operation_name>
      ... (同上格式)

    Step N: 最终输出
      - 与取证报告的 golden output 统计对照

  跨核通信: (仅跨核归约模式)
    - workspace buffer: <是否存在, 大小>
    - Phase 1 → Phase 2 的同步机制: <描述>
```

---

#### Sub-step 2.3: AscendC 实现逐步对照

**Phase A: 构建参考实现规范 (强制执行, 不可跳过)**

⚠️ **在读取任何 Kernel 代码之前, 必须先完成此 Phase, 建立正确实现的参考规范。** 参考规范是后续 Phase C 结构化对照的基准。

**Phase A 读取**:

1. 根据 `L8_operator.op_type` 从 `archive_tasks/` 路由，读取对应案例的 `kernel/` 目录（仅含有完整 kernel/ 的案例）：
   - pooling → `archive_tasks/avg_pool3_d/kernel/`
   - normalization / rmsnorm / layernorm → `archive_tasks/rms_norm/kernel/`（含 vector_tile.h）
   - matmul / gemm / linear → `archive_tasks/matmul_leakyrelu/kernel/` 或 `archive_tasks/quant_matmul/kernel/`
   - gather / scatter / index → `archive_tasks/gather_elements_v2/kernel/`
   - concat / memory layout → `archive_tasks/concat_dv2/kernel/`
   - **attention / softmax** → 无有效 AscendC 案例，跳过案例读取，仅读 API 文档
   - **纯 elementwise / activation** → 无纯向量案例，仅读 `dsl2Ascendc_compute_vector.md`
   - 无精确匹配 → 选最近似案例，在 [REFERENCE_IMPL_SPEC] 中标注"参考案例非精确匹配"

2. **必须读取**: `skills/ascendc/ascendc-translator/references/dsl2Ascendc.md`
   （替代 error_correction_examples.md，含禁用 API 模式和常见错误）

3. **必须读取**: `skills/ascendc/ascendc-translator/references/dsl2Ascendc_compute_vector.md`
   （含 DataCopyPad 触发条件和非对齐处理）

4. **必须读取**: `skills/ascendc/ascendc-translator/references/TileLang-AscendC-API-Mapping.md`
   （AscendC API 权威参考）
   API 详细文档：`skills/ascendc/ascendc-translator/references/AscendC_knowledge/api_reference/`

**Phase A 产出**: `[REFERENCE_IMPL_SPEC]` section (写入 precision_audit_{attempt}.md, Gate-A 必填)

从上述参考文件中提取并填写如下规范:

```
[REFERENCE_IMPL_SPEC]
  参考实现来源: <lowering example 文件路径>

  TQue/TBuf 分配规范 (来自参考实现):
    - inQueue (TQue<VECIN>): <用途, DataCopy GM→UB 的目标 buffer>
    - outQueue (TQue<VECOUT>): <用途, DataCopy UB→GM 的源 buffer, 必须经 EnQue/DeQue>
    - TBuf (VECCALC): <用途, 中间计算 buffer, 不可直接作为 DataCopy src/dst>
    - TBuf→GM 正确路径: TBuf计算结果 → outQueue.AllocTensor → 写入outLocal → EnQue → DeQue → DataCopy

  关键 API 规范 (来自参考实现):
    - ReduceMax: <调用签名; work_buf 是否需要 Duplicate(-3.402823466e+38f, count) 初始化>
    - ReduceSum: <count 对齐要求 (64的倍数); work_buf 是否需要 Duplicate(0.0f, count) 初始化>
    - SyncAll: <是否需要, 插入位置 (跨核写GM后/读GM前)>

  非对齐处理规范 (来自 non_aligned_example):
    - 触发条件: count × sizeof(dtype) 不是 32 的倍数
    - GM→UB: DataCopyPad(dst, src, {1, count*sizeof(T), 0, 0}, {false, 0, 0, 0})
    - UB→GM: DataCopyPad(dst, src, {1, count*sizeof(T), 0, 0})
    - 本算子 tileLength 的对齐状况: <tileLength × sizeof(dtype) = ? 字节, 是否32字节对齐>

  error_correction 禁用模式 (来自 error_correction_examples.md):
    - 禁止 float↔uint 强制类型转换 (改用 float n = (float)int_val)
    - 禁止标量上下文调用向量 Log API (改用 AscendC::Log(tmp,tmp,1); tmp.GetValue(0))
    - 禁止向量上下文调用标量 AscendC::Sqrt (改用 sqrt(val))
    - 本算子代码中是否出现上述模式: <逐一检查>
```

---

**Phase B: 读取当前实现**

**Phase B 读取** (全部在 `{task_dir}/kernel/` 下):
1. `{op_name}_tiling.h` — TilingData 结构体定义
2. `*_kernel.h` — 所有 kernel 类定义（可能有多个）
3. `*.cpp`（排除 pybind11.cpp）— 所有 kernel entry 文件
4. `kernel_common.h`、`vector_tile.h`、`matmul_tile.h`（若存在）— helper 逻辑
5. `pybind11.cpp` — host tiling 计算、workspace 分配、launch 逻辑

注意：AscendOpGenAgent 中 host 逻辑（TilingFunc）在 pybind11.cpp 内，不是单独的 op_host.cpp。

**产出**: `[REFERENCE_IMPL_SPEC]` + `[KERNEL_STEP_TRACE]` sections

**Phase C 要求 (结构化对照)**:
- 将 Kernel 的 Compute() 函数拆成与 Sub-step 2.2 对应的步骤
- 每步标注: AscendC API 名称、count 参数值、buffer 来源、代码行号
- 逐步与 2.2 的计算链对齐, 用 ✅/⚠️/❌ 标注匹配状态
- **对照 `[REFERENCE_IMPL_SPEC]` 逐项检查以下 5 个维度**:
  1. TQue/TBuf 数据流是否与规范一致 (特别: TBuf 是否绕过 outQueue 直接 DataCopy)
  2. ReduceMax/ReduceSum work_buf 是否按规范初始化 (Duplicate 到 -INF 或 0)
  3. DataCopy 对齐是否满足规范 (count × sizeof(dtype) 不是 32 倍数时是否换用 DataCopyPad)
  4. SyncAll 同步点是否与规范一致 (跨核场景是否遗漏)
  5. error_correction_examples 中列出的禁用模式是否在代码中出现
- 遇到不确定的 API 名称时，查阅 `tl_asc_routing.md` 确认（如 Max vs Vmax、Subs 是否存在、负无穷常量写法等）

```
[KERNEL_STEP_TRACE]
  Kernel 计算步骤 (从 Compute() 函数提取):
    K-Step 1: <AscendC API 名称>
      - 代码位置: <kernel 文件名>.cpp 第 <line> 行
      - 参数: count=<value>, src=<buffer>, dst=<buffer>
      - 对应计算链: Step <N> (<operation_name>)
      - 匹配状态: ✅ 匹配 / ⚠️ 参数偏差: <描述> / ❌ 缺失或多余

    K-Step 2: ...
    ...

  Host tiling 参数:
    - TilingData 结构体字段: <从 _tiling.h 中列出所有字段名和类型>
    - tileLength = <值> (来源: pybind11.cpp TilingFunc 第 <line> 行)
    - 其他 TilingData 字段: <列出 field=value>
    - 归约维度完整性: tileLength <>=<> 归约轴长度 <length> → 完整 / 被切分

  跨核通信验证: (仅跨核归约模式)
    - workspace buffer: GM 中是否分配, 大小是否 = n_cores
    - 各核写入: DataCopy 后是否有同步
    - Core 0 读取: 是否在所有核完成后才读取
    - 全局归约: ReduceSum 的 count 是否正确 (= n_cores, 而非 tile_size)
    - 最终除法: 分母是否 = total_elems

  算子类型专项检查 (根据 L8 op_type 选择对应项):

    [Pooling 类] DataCopy 维度一致性:
      - 输入内存布局: <NCDHW / NCHW / NHWC, 来自 L6>
      - tileC 含义: <沿 C 维度的 tile 大小>
      - DataCopy count=tileC 读取的是: <C 维度 tileC 个通道 还是 W 维度 tileC 个元素?>
      - C 维度在内存中的 stride: <C_stride = D*H*W (NCDHW) / H*W (NCHW)>
      - ⚠️ 检查: tileC 个连续地址是否真的对应 tileC 个通道? 若 C_stride > 1, 连续地址实为沿 W/空间维读取
      - input base offset 公式: <写出 b/c0/d/h/w 各维度的 offset 计算, 标出 c0 的系数是否为 C_stride>
      - output base offset 公式: <写出 b/c0/od/oh/ow 各维度的 offset 计算, 标出 c0 的系数>
      - ⚠️ 检查: outBase 中 c0 的系数是否为 outD*outH*outW (正确) 而非 1 (错误)

    [Reduction / Normalization 类] 工作 Buffer 初始化:
      - ReduceMax work buffer: <调用前是否 Duplicate(work, -INF, count) 初始化?>
      - ReduceSum work buffer: <调用前是否 Duplicate(work, 0, count) 初始化?>
      - ⚠️ 检查: work buffer 是否从上一步骤残留了非零数据 (如 ReduceMax work buffer 含有上一步 maxVal 残留)
      - 负无穷写法: <代码中使用 -3.402823466e+38f / (float)(-INFINITY) / -65504.0f (float16 错误!)>

    [MatMul / 分块累加类] 累加器初始化:
      - 累加器 (acc buffer) 初始化位置: <在外层循环前 Duplicate(0) / 未初始化>
      - ⚠️ 检查: 多个 tile 间累加器是否在每个输出位置开始时被正确重置

    [TQue / TBuf 数据流] 同步验证 (所有算子类型必填):
      - inQueue 流程: AllocTensor → DataCopy(GM→UB) → EnQue → DeQue → (计算) → FreeTensor ✅/❌
      - outQueue 流程: AllocTensor → (计算写入) → EnQue → DeQue → DataCopy(UB→GM) → FreeTensor ✅/❌
      - TBuf 用途: <VECCALC 中间计算, 不参与 DMA 传输>
      - ⚠️ 严重: TBuf.Get() 直接作为 DataCopy dst 写 GM = 绕过 outQueue 同步 = 数据未写出 = 输出全零
      - ⚠️ 检查: CopyOut 函数中 maxLocal/accLocal 等 TBuf 变量是否直接用于 DataCopy(outputGm[], ...)

  步骤对齐结论:
    - 全部匹配: 是 / 否
    - 缺失步骤: <列出, 或 "无">
    - 参数偏差: <列出, 或 "无">
    - 新增/多余步骤: <列出, 或 "无">

  L7 代码位置映射 (手动):
    - worst element index=<index> → 对应 kernel 中的 <函数/代码块>
    - 该元素位于 main block / tail block?
    - 对应的 K-Step: <编号>
```

---

#### Sub-step 2.4: 知识库匹配 + 根因判断 + 修复计划

**读取**: Sub-step 2.1 检索到的知识库条目 + Sub-step 2.1~2.3 的全部分析结果

**知识库检索 (第二次 — 精化, 增加位置特征):**

> ⚠️ **在开始匹配前，用完整取证数据做第二次精化检索**（避免长上下文遗忘, 并利用 2.1~2.3 分析中发现的位置特征）。

从取证报告中提取 `--position` 参数:
- 若 worst_elements 集中在尾部区域 或 tail_analysis 显示尾块 mismatch 率显著偏高 → `--position tail`
- 若 worst_elements 集中在边界/起始区域 → `--position boundary`
- 若 worst_elements 分散 → `--position scattered`
- 若无明显位置特征 → 不传 `--position`

```bash
python3 skills/ascendc/precision-tuning/scripts/precision_knowledge.py search \
    --kb-path skills/ascendc/precision-tuning/references/precision_knowledge_base.json \
    --op-type <L8_operator.op_type> \
    --pattern <primary_hint> \
    --position <tail/boundary/scattered 或不传> \
    --top-k 3 \
    --log-path "{task_dir}/precision_tuning" \
    --attempt {attempt} \
    --call-index 1
```

记住检索到的条目, 用于下方的 `[KNOWLEDGE_MATCH]`。

**产出**: `[KNOWLEDGE_MATCH]` + `[ROOT_CAUSE]` + `[FIX_PLAN]` + `[TARGET_FILES]` + `[DIRECTION_ASSESSMENT]` sections

**要求**: 根因判断必须基于 2.1~2.3 的具体发现, 不允许"凭直觉"给出根因。证据链中必须引用具体的 K-Step 编号和取证数据字段。

> ⚠️ **若 Sub-step 2.1 产出了 `[PRIOR_TRACE_CONTEXT]`**，`[FIX_PLAN]` 的修复方向**不得**与其"已尝试方向"里的失败路径重复，也**不得**违反"剩余未解决的平台/API 限制"（如 trace.md 明确记录"当前平台 Muls 不支持 __bf16"，则不得在修复中使用 `Muls` 处理 bf16 dtype）。在 `[ROOT_CAUSE].证据链` 里显式引用 trace 中对应的走偏点或平台限制条目。

> ⚠️ **写 [FIX_PLAN] 前必须查阅 `TileLang-AscendC-API-Mapping.md`，核实所有将要使用的 AscendC API 名称**：
> - 逐元素向量最大值：`Max`（不是 `Vmax`，该 API 不存在）
> - 逐元素减法：`Sub`（无 `Subs`），逐元素除法：`Div`（无 `Divs`）
> - float32 负无穷：`-3.402823466e+38f` 或 `(float)(-INFINITY)`（不是 `AscendC::INFINITY`，该常量不存在）
> - DataCopy 写 GM 必须从 VECOUT TQue DeQue 后的 tensor，不能直接用 TBuf.Get() 的结果

```
[KNOWLEDGE_MATCH]
  知识库匹配:
    - 匹配的知识条目: <title> / 无匹配
    - 匹配度: 完全匹配 / 部分匹配 / 不匹配
    - 如何借鉴: <参考知识条目的 fix 内容>
  算子类型 checklist 检查:
    - <checklist 项 1>: 通过 / 未通过 / 不适用 (证据: <引用 K-Step 或取证数据>)
    - <checklist 项 2>: ...

[ROOT_CAUSE]
  根因判断: <综合 2.1 取证数据 + 2.3 步骤对齐结论 + 知识库匹配>
  置信度: HIGH / MEDIUM / LOW
  证据链:
    1. 数值证据: <取证 L1-L4 中哪些现象支持此判断, 引用具体字段值>
    2. 布局证据: <L6 内存布局是否有异常>
    3. 代码证据: <引用 K-Step 编号, 哪行代码有什么问题>
    4. 分解对照: <2.2 的哪个 Step 与 2.3 的哪个 K-Step 不一致>
    5. 逻辑推导: <为什么此代码问题会产生取证中观察到的 diff 模式>

[FIX_PLAN]
  修复方向: <具体描述, 引用变量名和行号>
  修复类型: <对应知识库 type, 如 FIX_PRECISION_TAIL>
  修改文件: <file1, file2>
  修改点:
    1. 文件: <文件名>, 位置: <行号或函数名>, 操作: <修改/新增/删除>
       当前代码: <现在是什么>
       修改为: <改成什么>
       对应 K-Step: <编号>
    2. ...
  预期效果: <修复后应该改善什么, 如 "尾块 mismatch 应消除">

[TARGET_FILES]
  <需要修改的文件列表, 逗号分隔>

[DIRECTION_ASSESSMENT]
  上一轮 (attempt={attempt-1}) 修复方向: <从 history/attempt_{{attempt-1}}/precision_audit.md 的 [FIX_PLAN] 中提取, 一句话描述>
  上一轮修复后 mismatch 变化: <从 forensics_report_{attempt}.json 的 history_trend 中读取, 如 "0.25→0.12 (改善)" 或 "0.25→0.28 (恶化)">
  本轮是否延续上一轮方向: <严格填写 "是" 或 "否"，不得填写其他任何文字>
  延续理由 / 换方向理由: <一句话>

=== END AUDIT ===
```

**重试轮次注意事项 (attempt > 0):**
- 取证报告中的 `history_trend` 显示了历史变化
- 你**必须**先读 `tuning_directions.json` 获取跨轮方向全貌，再按需通过 `round_summary_N.json` 的 index 路径深入具体 section 小文件
- **禁止重复 outcome 为 regressed 或连续 stagnant 的 fix_type**
- 如某轮 `index.sections.root_cause` 为 null，fallback 读 `index.audit_full`

Gate 验证:
```bash
python3 skills/ascendc/precision-tuning/scripts/precision_gate.py \
    --step audit --op-name {op_name} --task-name {task_name} --attempt {attempt}
```

⛔ **Gate-A 未通过 → 补全缺失的 section, 不计入轮次。Gate-A 现在检查 7 个必填 section: FORENSICS_SUMMARY, COMPUTATION_DECOMPOSITION, REFERENCE_IMPL_SPEC, KERNEL_STEP_TRACE, ROOT_CAUSE, FIX_PLAN, TARGET_FILES。**

> Gate-A 通过后，脚本自动提取 sections 小文件并写入 `round_summary_{attempt}.json` 初始字段（diagnostics + index）。**Agent 无需手动写 round_summary。**

---

### Step 3: 代码修复 (Agent 执行)

根据审计报告 [FIX_PLAN] 中的修改点, 逐一修复代码。

**修复原则:**
1. **严格遵循 FIX_PLAN**: 不要自行扩大修改范围
2. **完整文件**: 写入修改后的完整文件, 不要截断
3. **真实变量名**: 使用代码中实际存在的变量名
4. **禁止逃避**: 不得缩小 shape、添加 if 跳过、放大 tolerance、删除功能

修复完成后, Gate 验证代码文件完整性:
```bash
python3 skills/ascendc/precision-tuning/scripts/precision_gate.py \
    --step fix --op-name {op_name} --task-name {task_name} --attempt {attempt}
```

⛔ **Gate-X 未通过 → 检查文件是否正确保存。**

---

### Step 4: 重新编译 + 精度验证

```bash
bash skills/ascendc/ascendc-translator/references/evaluate_ascendc.sh {task_name}
```

**失败分类处理**（根据 stdout 判断）：

| 失败类型 | 特征 | 处理 |
|---|---|---|
| Infra 失败 | SSH 超时、docker exec 失败 | 停止，报告环境问题，不进入修复循环 |
| Build 失败 | `build_ascendc.py` 报编译错误 | 修复 kernel .cpp/.h，最多 3 次 |
| Import 失败 | import 阶段报 ModuleNotFoundError 或 PYBIND11_MODULE 名不一致 | 检查 model_new_ascendc.py import 名 vs pybind11.cpp |
| Numerical 失败 | verification_ascendc.py 报 mismatch | 进入 precision_forensics → 审计 → 修复循环 |

**每次编译失败后，更新 `{task_dir}/precision_tuning/compilation_log_{attempt}.json`**（追加 entry）：
```json
{
  "attempt": <N>,
  "entries": [
    {
      "compile_retry": <0/1/2>,
      "error_category": "<undefined_api|type_mismatch|count_alignment|other_compile>",
      "error_snippet": "<编译器报错核心行，最多3行>",
      "fix_applied": "<本次修复简述>"
    }
  ]
}
```

**保存验证结果**（从 stdout 解析，写入 `{task_dir}/precision_tuning/validation_result_attempt_{attempt}.json`）：
```json
{
  "attempt": <N>,
  "correctness_passed": true/false,
  "evaluate_stdout": "<evaluate_ascendc.sh 完整输出>",
  "match_rate": "<从 stdout 提取，如 87.50 或 100.00>",
  "max_diff": "<从 stdout 提取，如 1.23e-04>"
}
```

提取规则（`verification_ascendc.py` 输出格式）：
- `match_rate`: 用正则 `r"mismatch_ratio=([0-9.]+)%"` 取所有 case 平均，转换为 match_rate = 100 - avg_mismatch；若无 mismatch 行则写 `100.0`
- `max_diff`: 用正则 `r"max_abs_diff=([0-9.eE+\-g]+)"`；若无 mismatch 行则写 `0.0`

**Gate 验证 + 循环控制:**
```bash
python3 skills/ascendc/precision-tuning/scripts/precision_gate.py \
    --step validate --op-name {op_name} --task-name {task_name} --attempt {attempt}
```

Gate-V 输出包含 **loop_signal**, 你**必须遵守**:

| loop_signal | 含义 | 你的操作 |
|-------------|------|---------|
| **PASS** | 精度通过 | → 跳到 Step 5 (成功收尾) |
| **CONTINUE** | 未通过但有改善 | → 归档本轮, 回到 Step 1 (attempt + 1) |
| **STOP** | 未通过且无改善/达上限 | → 跳到 Step 6 (失败报告) |

⚠️ **你不能自行决定继续或停止。loop_signal 由 Gate 脚本根据数值趋势决定, Agent 必须遵守。**

> 注意：这里的 Gate-V 只校验“当前 `{op_name}.json`”对应的验证结果。若任务目录还存在 `{op_name}.json.bak`，则这通常意味着当前 `.json` 是精简用例，**还不能直接宣布最终成功**；必须继续执行 Step 5 中的全量用例验证。

---

### 归档当前轮次 (CONTINUE 时执行)

**每次归档时，比较当前轮 match_rate 与历史最佳，决定是否更新最佳代码。**

```bash
# 1. 保存本轮取证报告和审计报告
mkdir -p "{task_dir}/precision_tuning/history/attempt_{attempt}"
cp "{task_dir}/precision_tuning/forensics_report_{attempt}.json" \
   "{task_dir}/precision_tuning/history/attempt_{attempt}/forensics_report.json"
cp "{task_dir}/precision_tuning/precision_audit_{attempt}.md" \
   "{task_dir}/precision_tuning/history/attempt_{attempt}/precision_audit.md"

# 2. 更新最佳代码
current_mr=$(python3 -c "import json; r=json.load(open('{task_dir}/precision_tuning/validation_result_attempt_{attempt}.json')); print(r.get('match_rate', '0'))")
best_mr=0
if [ -f "{task_dir}/precision_tuning/history/current_best/match_rate.txt" ]; then
    best_mr=$(cat "{task_dir}/precision_tuning/history/current_best/match_rate.txt")
fi

is_better=$(python3 -c "print('yes' if float('$current_mr') >= float('$best_mr') else 'no')")
if [ "$is_better" = "yes" ]; then
    mkdir -p "{task_dir}/precision_tuning/history/current_best/code_snapshot"
    cp -r "{task_dir}/kernel/" \
       "{task_dir}/precision_tuning/history/current_best/code_snapshot/kernel/"
    cp "{task_dir}/model_new_ascendc.py" \
       "{task_dir}/precision_tuning/history/current_best/code_snapshot/model_new_ascendc.py"
    echo "$current_mr" > "{task_dir}/precision_tuning/history/current_best/match_rate.txt"
    echo "精度改善: $best_mr → $current_mr，已更新最佳代码"
fi

# 3. 保存下一轮的起始快照
mkdir -p "{task_dir}/precision_tuning/history/attempt_{next_attempt}/code_snapshot"
cp -r "{task_dir}/kernel/" \
   "{task_dir}/precision_tuning/history/attempt_{next_attempt}/code_snapshot/kernel/"
cp "{task_dir}/model_new_ascendc.py" \
   "{task_dir}/precision_tuning/history/attempt_{next_attempt}/code_snapshot/model_new_ascendc.py"
```

然后 `attempt += 1`, 回到 Step 1。

---

### Step 5: 成功收尾

精度通过后:

**5.0 全量用例验证（若存在 `.json.bak`，则为强制步骤）**

若 `{task_dir}/{op_name}.json.bak` 存在，说明当前 `{op_name}.json` 为精简用例；此时必须先恢复全量用例，再做一次最终验证：

```bash
if [ -f "{task_dir}/{op_name}.json.bak" ]; then
    cp "{task_dir}/{op_name}.json.bak" "{task_dir}/{op_name}.json"
    bash skills/ascendc/ascendc-translator/references/evaluate_ascendc.sh {task_name}
fi
```

处理规则：
- 若 `.json.bak` 不存在：跳过本步骤，直接进入 5.1
- 若全量验证通过：继续进入 5.1
- 若全量验证失败：**不得**宣布成功；仅允许继续修改 `{task_dir}/kernel/` 下文件，并重新执行全量验证
- 全量验证失败后的补救次数最多 3 次（含首次全量验证）；超过次数仍失败，则转 Step 6 失败报告
- 若做过全量验证，最终成功报告中的 `final_match_rate` / `final_max_diff` 应以全量验证结果为准，而不是精简验证结果

建议额外保存全量验证结果：

```json
{task_dir}/precision_tuning/full_validation_result_attempt_{attempt}.json
{
  "attempt": <N>,
  "used_full_cases": true,
  "correctness_passed": true/false,
  "evaluate_stdout": "<全量 evaluate_ascendc.sh 完整输出>",
  "match_rate": "<从 stdout 提取>",
  "max_diff": "<从 stdout 提取>"
}
```

**5.1 归档当前轮次 + 更新 current_best（最终 PASS 时必须执行）:**
```bash
# 归档本轮取证报告和审计报告
mkdir -p "{task_dir}/precision_tuning/history/attempt_{attempt}"
cp "{task_dir}/precision_tuning/forensics_report_{attempt}.json" \
   "{task_dir}/precision_tuning/history/attempt_{attempt}/forensics_report.json"
cp "{task_dir}/precision_tuning/precision_audit_{attempt}.md" \
   "{task_dir}/precision_tuning/history/attempt_{attempt}/precision_audit.md"

# 更新 current_best 为最终通过的代码
mkdir -p "{task_dir}/precision_tuning/history/current_best/code_snapshot"
cp -r "{task_dir}/kernel/" \
   "{task_dir}/precision_tuning/history/current_best/code_snapshot/kernel/"
cp "{task_dir}/model_new_ascendc.py" \
   "{task_dir}/precision_tuning/history/current_best/code_snapshot/model_new_ascendc.py"
echo "100.0" > "{task_dir}/precision_tuning/history/current_best/match_rate.txt"
echo "精度通过，current_best 已更新为 100.0"
```

**5.2 生成候选知识库条目 (Agent 执行):**

基于 [ROOT_CAUSE] 和 [FIX_PLAN]，生成一条知识库候选条目，写入：
`{task_dir}/precision_tuning/candidate_kb_entry.json`

格式要求:
```json
{
  "title": "<标准化中文标题，含英文关键词，如：LayerNorm 尾块 Padding 污染精度>",
  "feature": "<错误特征签名，泛化表达，不要写死具体 shape 或 tile size，如：tail_spike 模式，尾块 mismatch 率显著高于主体>",
  "reason": "<深层原因，50-200字，描述为什么会出现此问题>",
  "fix": "<通用修复指南，50-200字，描述应该如何修复，不要包含具体行号>",
  "type": "<FIX_PRECISION_XXX 枚举值，与 [FIX_PLAN] 中的修复类型一致>"
}
```

注意:
- `title` 必须含英文关键词（供 RAG 检索），格式为"中文描述 (English Keywords)"
- `feature` 要泛化，不要写 `last_dim=37` 或 `tile_size=128` 这种具体值
- `fix` 要通用，不要引用具体代码行号或变量名
- `type` 必须从以下枚举中选择：FIX_PRECISION_PADDING / FIX_PRECISION_TAIL / FIX_PRECISION_REDUCTION / FIX_PRECISION_TYPECAST / FIX_PRECISION_LAYOUT / FIX_PRECISION_SYNC / FIX_PRECISION_OVERFLOW / FIX_PRECISION_LOGIC / FIX_PRECISION_OTHER

**5.3 写入知识库 (Python 执行):**
```bash
python3 skills/ascendc/precision-tuning/scripts/precision_knowledge.py dump \
    --kb-path skills/ascendc/precision-tuning/references/precision_knowledge_base.json \
    --task-name {task_name} \
    --op-name {op_name}
```

**5.4 保存成功代码快照:**
```bash
# 将最终通过代码保存到 history/success/（永久保留，不覆盖）
mkdir -p "{task_dir}/precision_tuning/history/success/code_snapshot"
cp -r "{task_dir}/kernel/" \
   "{task_dir}/precision_tuning/history/success/code_snapshot/kernel/"
cp "{task_dir}/model_new_ascendc.py" \
   "{task_dir}/precision_tuning/history/success/code_snapshot/model_new_ascendc.py"
echo "成功代码已保存到 history/success/code_snapshot/"
```

> **从最佳代码恢复（如需重新调优）：**
> ```bash
> cp -r "{task_dir}/precision_tuning/history/current_best/code_snapshot/kernel/" \
>    "{task_dir}/kernel/"
> cp "{task_dir}/precision_tuning/history/current_best/code_snapshot/model_new_ascendc.py" \
>    "{task_dir}/model_new_ascendc.py"
> ```

**5.5 输出成功报告:**
```
[PRECISION_TUNING_RESULT]
  status: SUCCESS
  attempts: <总轮次>
  final_match_rate: <最终 match rate，若跑过全量则取全量结果>
  final_max_diff: <最终 max diff，若跑过全量则取全量结果>
  root_cause_summary: <一句话总结根因>
  fix_summary: <一句话总结修复内容>
```

---

### Step 6: 失败报告

如果 Gate-V 返回 STOP:

输出失败报告, 包含所有轮次的历史:
```
[PRECISION_TUNING_RESULT]
  status: FAILED
  attempts: <总轮次>
  loop_stop_reason: <Gate 给出的停止原因>
  history:
    attempt 0: hint=<pattern>, mismatch=<ratio>, fix=<一句话>
    attempt 1: hint=<pattern>, mismatch=<ratio>, fix=<一句话>
    ...
  remaining_issue: <当前仍存在的问题描述>
  suggestion: <给人工分析的建议>
```

> **注意:** 失败时 `history/current_best/` 中保存了精度最好的代码。如需以此为基础重新调优，恢复方法：
> ```bash
> cp -r "{task_dir}/precision_tuning/history/current_best/code_snapshot/kernel/" \
>    "{task_dir}/kernel/"
> cp "{task_dir}/precision_tuning/history/current_best/code_snapshot/model_new_ascendc.py" \
>    "{task_dir}/model_new_ascendc.py"
> ```
> 如需恢复到最初基线：
> ```bash
> cp -r "{task_dir}/precision_tuning/history/baseline/code_snapshot/kernel/" \
>    "{task_dir}/kernel/"
> cp "{task_dir}/precision_tuning/history/baseline/code_snapshot/model_new_ascendc.py" \
>    "{task_dir}/model_new_ascendc.py"
> ```

---

## Note

- **每步 Gate 验证不可跳过** — Gate 是流程稳定性的保证
- **loop_signal 由 Gate 脚本决定, Agent 必须遵守** — 防止钻牛角尖
- **取证数据是分析的基础** — 不要在没有取证的情况下分析代码
- **知识库条目只在精度通过时写入** — 避免失败经验污染知识库
- **编译失败不计入精度调优轮次** — 编译问题就地修复 (最多 3 次)
- 修复后代码直接写回 AscendC 项目目录 (覆盖原文件)
- 参考 `references/precision_knowledge_base.json` 中的已知精度问题模式
