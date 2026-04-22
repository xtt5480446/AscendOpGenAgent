# Ascend Kernel Developer + AscendC Debug Subagent 集成设计（v3）

> v3 基于 codex review v2 再次修订。核心修正：
> - v2 的 `SIGKILL/SIGTERM + duration≈limit => timeout` 误判率高（外层 harness 已用 `timeout --signal=TERM` 杀进程）→ 改为只认 wrapper 自触发的 TimeoutExpired + 明确 marker
> - v2 Gate 2 层 audit schema 自相矛盾（通用层要求 `[DIAGNOSIS]+[FIX_PLAN]`，精度分支沿用 `[FORENSICS_SUMMARY]`）→ 钉死契约：通用层只管文件存在 + 必备 section **之一**存在，不强制统一 schema
> - v2 Phase 8 异常退出要 append `trace.md`，但又声明"不 append" → 改为所有 Phase 8 结果都进 `debug_trace.md`，`trace.md` 全程只读
> - v2 没有 Phase 8 后机器可读 final verdict → 新增 `debug_status.json`
> - v2 `import_failed` scope 与"只改 kernel/"硬约束矛盾 → 拆成 kernel/pybind 子类（可修）和环境/链接子类（不可自动修）
> - v2 `ac_history.json` 与 `trace.md` / `PRIOR_TRACE_CONTEXT` 冗余 → 删除，Phase 4 迭代历史并入 `trace.md` 末尾 JSON
> - v2 `debug_trace.md` 8 节强制中多项无数据源 → 降级到 4 节强制，其余改可选附录
> - v2 工作量估 10 人日过于乐观 → 修正到 12-16 人日
>
> v2 已修正的 v1 问题仍然有效（eval_status.json 基础设施、Gate 2 层架构方向、条件性 spawn、import_failed 引入）

## 背景

当前 `agents/ascend-kernel-developer.md` 的 Phase 4 (AscendC) 维护迭代 debug 循环（max 3 次），Phase 6 允许最多 3 次 kernel 修复。

`agents/ascendc-debug-agent-discovery.md` + `skills/ascendc/ascendc-debug/` 是独立的**精度调优** subagent/skill，通过 `precision_forensics.py` 做数值取证驱动 debug。

目标：把 **AscendC 所有可自动修复的失败类型**（build/import/runtime/timeout/precision）的 debug 工作从主 agent 循环抽出来交给 subagent 处理；主 agent 负责生成 + 初筛 + 归档，不再负责深度 debug。

## 决策摘要（v2 终版）

| 决策项 | v1 方案 | v2 方案 | 变更原因 |
|-------|--------|--------|---------|
| 主 agent 文件 | 新建 `ascend-kernel-developer-with-ascendc-debug.md` | **不变** | — |
| 原 agent 文件 | 恢复 main 分支状态 | **不变** | — |
| Phase 4 迭代上限 | 3 → 2 | **不变** | — |
| Phase 6 修复重试 | 删除，只读一遍 | **不变** | — |
| **eval_status.json 基础设施** | 未考虑 | **新增（必做先行）** | codex: trace-recorder 事后猜 failure_type 不可靠 |
| **Phase 7 trace-recorder** | 事后推断 failure_type | **读 eval_status.json 填充** | 同上 |
| **Phase 8 spawn 策略** | 无条件 | **条件性 + 白名单** | codex: 文档自相矛盾，degraded 无法自救 |
| **Phase 8 结束后** | 主 agent 不做任何事 | **subagent 写 `debug_trace.md`**，主 agent 只报阶段结束 | 明确 subagent 自包含 |
| `failure_type` 集合 | 7 种 | **9 种** | codex: 漏 `import_failed`，`tilelang_only_failed` 补分流；`infra_error` 不加 |
| subagent scope | 精度 + build/runtime/timeout | **同 v1** | — |
| **Gate 重构** | "不变" | **2 层架构（通用层 + 分支层）** | codex: `precision_gate.py` 硬编码精度语义 |
| **ascendc-debug-agent skill 修改面** | 4 处 | **5 类改动**（含 `precision_gate.py` 拆包） | codex: 工作量估错 |
| 反作弊硬约束 | 不变 | **不变**（提升到 Gate-通用层） | — |
| subagent timeout | 1.5h 不重试 | **同 v1**（`infra_error` 另算） | — |

---

## 架构总览

```
┌───────────────────────────────────────────────────────────────┐
│ agents/ascend-kernel-developer-with-ascendc-debug.md (主)    │
├───────────────────────────────────────────────────────────────┤
│ Phase 0-2: 参数 / 环境 / case 精简      (不变)                │
│ Phase 3:   TileLang 设计 (max 5)        (不变)                │
│ Phase 4:   AscendC 迭代 (max 2)         (AC 循环缩短)         │
│ Phase 5:   性能分析                      (不变)                │
│ Phase 6:   全量验证 (只读一遍)          (去掉修复重试)        │
│ Phase 7:   Trace 记录 + eval_status.json (消费结构化状态)     │
│ Phase 8:   条件性 Spawn Debug Subagent  (新增)                │
└───────────────────────────────────────────────────────────────┘
                      │
                      │ (白名单 failure_type 才 spawn)
                      ▼
┌───────────────────────────────────────────────────────────────┐
│ agents/ascendc-debug-agent-discovery.md (subagent, 复用文件)    │
├───────────────────────────────────────────────────────────────┤
│ 读取 trace.md + eval_status.json                              │
│ Step 0.3: failure_type 分流                                   │
│   ├─ precision_failed → Step 1-P (现有取证路径)                │
│   ├─ build_failed     → Step 1-B (新增，读 build log)         │
│   ├─ import_failed    → Step 1-I (新增，读 import traceback)  │
│   ├─ runtime_error    → Step 1-R (新增，读 stderr/stacktrace) │
│   └─ timeout          → Step 1-T (新增，查同步/tiling)        │
│ Step 2/3: 分析 + 修复 (走 2 层 Gate)                          │
│ 结束: 写 debug_trace.md                                       │
└───────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────────────────────────┐
│ skills/ascendc/ascendc-debug/ (skill, 重构)                │
├───────────────────────────────────────────────────────────────┤
│ scripts/                                                       │
│   eval_status.py           (新增，解析 eval_status.json)       │
│   precision_gate.py        (重构，只做路由)                    │
│   gates/                                                       │
│     common.py              (通用层：反作弊/结构/eval 输出)    │
│     branch_precision.py    (精度分支 F/A/V)                   │
│     branch_build.py        (新增)                             │
│     branch_import.py       (新增)                             │
│     branch_runtime.py      (新增)                             │
│     branch_timeout.py      (新增)                             │
└───────────────────────────────────────────────────────────────┘
```

---

## Section 1: eval_status.json 基础设施（先行）

### 1.1 目的

提供**机器可读的 evaluate_ascendc 结果**，作为下游（trace-recorder / Phase 8 spawn 决策 / subagent Step 0.3 / Gate）的唯一事实源。

取代当前"靠 bash exit code + stdout 正则"推断 failure_type 的不可靠方案。

### 1.2 Schema

```json
{
  "schema_version": 1,
  "phase": 4,
  "attempt": 2,
  "started_at": "2026-04-22T10:15:23Z",
  "ended_at": "2026-04-22T10:15:58Z",
  "duration_sec": 35,
  "exit_code": 124,
  "exit_signal": "SIGTERM",
  "failure_type": "timeout",
  "failed_step": "compile | import | execute | verify | null",
  "log_path": "{task_dir}/.eval_logs/phase4_attempt2_20260422_101523.log",
  "stdout_tail": "... last 50 lines ...",
  "stderr_tail": "... last 50 lines ...",
  "compile": {
    "status": "passed | failed | skipped",
    "error_summary": "..." // 若 failed
  },
  "import": {
    "status": "passed | failed | skipped",
    "traceback_path": "..." // 若 failed
  },
  "execute": {
    "status": "passed | failed | timeout | crashed | skipped",
    "crash_signal": "SIGSEGV" // 若 crashed
  },
  "verify": {
    "status": "passed | failed | skipped",
    "total_cases": 10,
    "passed_cases": 7,
    "failed_cases": [
      {"case_id": "1", "status": "numerical_failed", "max_abs_diff": 2.3e-2}
    ]
  }
}
```

### 1.3 实现方案

**位置**：`utils/eval_wrapper.py`（新建，Python 包装）

**职责**：
1. 接收调用参数（phase、attempt、task_dir）
2. `subprocess.run()` 调 `evaluate_ascendc.sh`，捕获 stdout/stderr、exit_code、signal
3. 根据 exit_code / stderr 模式识别各阶段状态（compile / import / execute / verify）
4. 分类 `failure_type`（见 §1.4 判定规则）
5. 落盘 `{task_dir}/.eval_status/phase{N}_attempt{M}.json`
6. 同时维护 `{task_dir}/.eval_status/latest.json`（symlink or copy）供下游方便读取
7. 保留原始 log 到 `{task_dir}/.eval_logs/`

**调用方**：
- Phase 4 每次迭代的 functional 验证
- Phase 6 全量验证
- subagent 内部每次 Gate-V 重跑

**不改 `evaluate_ascendc.sh` 本身**（保持向后兼容），只在调用链上加 wrapper。

### 1.4 `failure_type` 判定规则（v3 重写）

**核心原则**：只认 wrapper 自己能明确证据的 failure_type，拒绝靠 "signal + 时长近似" 推测。

**判定顺序（优先级从高到低）**：

1. **`build_failed`** — compile 阶段 stderr 含 compile error pattern，且 exit_code 由 compile 子进程直接传播（wrapper 能区分哪一步炸的）

2. **`import_failed`** — import 阶段捕获到 `ImportError` / `ModuleNotFoundError` / `OSError: cannot open shared object`
   - **子分类**（wrapper 同时落盘 `import_subtype`）：
     - `import_kernel_side` — traceback 指向 pybind 符号未找到、kernel ext module 名字不对、`_xxx_ext.so` 导出符号问题 → **可修（进 subagent）**
     - `import_env_side` — traceback 指向 `libascend_hal.so`、`libruntime.so`、`libtorch_*.so` 等环境库未找到；或 `LD_LIBRARY_PATH` / `ASCEND_TOOLKIT_HOME` 等环境变量缺失 → **不可自动修**，subagent 识别后直接退出并写 `debug_trace.md` 说明

3. **`runtime_error`** — execute 阶段子进程返回 SIGSEGV / SIGABRT / SIGBUS / SIGFPE（明确 crash signal），且 wrapper 没有触发过自己的 timeout 杀进程

4. **`timeout`** — 必须满足**全部**三条：
   - wrapper 自己用 `subprocess.run(timeout=...)` 捕获 `TimeoutExpired` 异常
   - wrapper 在杀子进程前写了 `.eval_status/timeout_marker`（一个 sentinel 文件）
   - 触发 wrapper timeout 的阈值来自 wrapper 自身配置（不是外层 harness 的总任务 timeout）

5. **`precision_failed`** — verify 阶段正常退出但输出数值对比失败

6. **`execution_aborted`** — **v3 新增**。兜底类型，满足以下任一：
   - 子进程被外层 signal 杀掉（SIGTERM/SIGKILL），且没有 wrapper 自己的 timeout_marker
   - 远端 SSH 断连、`docker exec` 报 container not running
   - 其他 wrapper 无法明确归因的非零退出
   - 含 `abort_subtype`：`killed_by_outer_harness` / `ssh_disconnected` / `docker_unreachable` / `unknown`

7. **`success`** — 所有阶段 passed

**`execution_aborted` 的处理**：
- Phase 4/6 内部识别到 `abort_subtype ∈ {ssh_disconnected, docker_unreachable}` → 重试 1 次；仍失败则按 `execution_aborted` 落盘
- trace-recorder 见到 `execution_aborted` → **`debug_eligible = false`**（subagent 无法 debug 环境问题）
- 主 agent Phase 8 跳过，任务以 execution_aborted 终止

> `degraded` 由 `validate_ascendc_impl.py` 独立产出，不走 eval_wrapper。
> `no_kernel` 由 Phase 7 trace-recorder 基于 `kernel/` 目录为空独立判定。
> `tilelang_only_failed` 同理由 Phase 7 判定。
>
> **总 failure_type 集合（v3，10 种）**：
> `success / precision_failed / build_failed / import_failed / runtime_error / timeout / execution_aborted / degraded / no_kernel / tilelang_only_failed`

**eval_wrapper 的补充字段（v3 新增）**：

```json
{
  ...
  "timeout_marker_present": true | false,
  "import_subtype": "import_kernel_side | import_env_side | null",
  "abort_subtype": "killed_by_outer_harness | ssh_disconnected | docker_unreachable | unknown | null"
}
```

---

## Section 2: 主 agent 改动

新建 `agents/ascend-kernel-developer-with-ascendc-debug.md`（从当前 `ascend-kernel-developer.md` 复制后修改）。

### 2.1 Phase 4 改动

**① 迭代上限 2**
```diff
- max_ac_iterations = 3
+ max_ac_iterations = 2
```

**② 走 eval_wrapper**

原直接调 `evaluate_ascendc.sh` 的地方替换为：
```bash
python utils/eval_wrapper.py --phase 4 --attempt ${ac_iteration} --task-dir ${output_dir}
```

**③ `infra_error` 重试机制**

eval_wrapper 若检测到 SSH/docker 抖动类错误（SSH return 255、`Connection refused`、`container not running`），**内部重试 1 次**后再落盘。不暴露给主 agent 循环。

### 2.2 Phase 6 改动

原 L465-470 改为：

```markdown
## Phase 6: 全量用例验证（只读）

将 `{output_dir}/<op_name>.json.bak` 恢复为 `{output_dir}/<op_name>.json`，通过
`utils/eval_wrapper.py --phase 6` 运行一次全量验证。

无论成功失败，**不做任何修复**，直接进入 Phase 7。
失败用例的修复由 Phase 8 的 debug subagent 接手。
```

### 2.3 Phase 7 改动

**trace-recorder skill 改为消费 `eval_status.json`，不再事后推断**。

Phase 7 结束时 trace-recorder 产出：
- `trace.md`（原有，记录 Phase 1-6 执行路径）
- `trace.md` 末尾追加结构化 `final_status` fenced JSON block（见 §2.4）

### 2.4 `final_status` 块结构（v3 修订：含 Phase 4 迭代历史）

v3 删除独立的 `ac_history.json`，把 Phase 4 迭代历史并入 `final_status` JSON（避免与 `trace.md` / `PRIOR_TRACE_CONTEXT` 三份重复记录 Phase 4 历史）。

```json
{
  "schema_version": 2,
  "failure_type": "success | precision_failed | build_failed | import_failed | runtime_error | timeout | execution_aborted | degraded | no_kernel | tilelang_only_failed",
  "import_subtype": "import_kernel_side | import_env_side | null",
  "abort_subtype": "killed_by_outer_harness | ssh_disconnected | docker_unreachable | unknown | null",
  "has_kernel": true,
  "has_compiled_kernel": true,
  "has_degradation": false,
  "last_evaluate_phase": 6,
  "last_evaluate_status_path": "{task_dir}/.eval_status/phase6_attempt0.json",
  "tl_iterations_used": 3,
  "ac_iterations": [
    {
      "attempt": 0,
      "verifier_error": "A-AscendCFallback-Type3: ...",
      "conductor_suggestion": "...",
      "eval_status_path": "{task_dir}/.eval_status/phase4_attempt0.json",
      "ended_at": "..."
    }
  ],
  "debug_eligible": true,
  "debug_eligible_reason": "..."
}
```

`debug_eligible` 字段：直接编码 Phase 8 spawn 决策，trace-recorder 根据白名单规则填。

`debug_eligible = false` 的情况（v3 明确枚举）：
- `failure_type ∉ whitelist`（whitelist 见 §2.5）
- `has_kernel == false` 或 `has_degradation == true`
- `failure_type == import_failed && import_subtype == import_env_side`（环境问题 subagent 无法处理）
- `failure_type == execution_aborted`（环境/harness 问题）

### 2.5 Phase 8 新增（条件性 spawn，v3 修订：trace.md 全程只读）

```markdown
## Phase 8: AscendC Debug Subagent（条件性 spawn）

### 8.1 读取 final_status
从 `{output_dir}/trace.md` 末尾解析 `final_status` JSON block 的 `debug_eligible` 字段。

### 8.2 Spawn 决策

只在 `debug_eligible == true` 时 spawn。
白名单 failure_type（whitelist 编码进 trace-recorder 的 `debug_eligible` 规则）：
  - `precision_failed`
  - `build_failed`
  - `import_failed && import_subtype == import_kernel_side`  ← v3 拆分
  - `runtime_error`
  - `timeout`

不 spawn 场景：
  - `success` / `degraded` / `no_kernel` / `tilelang_only_failed`
  - `execution_aborted`（环境/harness 问题，subagent 无法修）
  - `import_failed && import_subtype == import_env_side`（环境库/LD_LIBRARY_PATH，不在 kernel/ scope 内）

### 8.3 Spawn 调用
- subagent_type: `ascendc-debug-agent-discovery`
- 传入: `{output_dir}` 绝对路径 + npu ID + failure_type
- timeout: 5400 秒（1.5h）

### 8.4 Spawn 后处理（v3 修订）

**关键原则**：`trace.md` 在 Phase 7 写完后**全程只读**，主 agent 绝不 append。所有 Phase 8 相关信息（正常退出、超时、crash）统一落到 subagent 自己的产物里。

**正常退出**（subagent 自主完成，返回 0）：
- 主 agent 校验必须产出两个文件：
  - `{output_dir}/debug_trace.md`
  - `{output_dir}/debug_status.json`
- 若任一缺失 → 视为异常退出
- 输出一行："Phase 8 结束，详见 `debug_trace.md` / `debug_status.json`"

**超时 / 异常退出**（subagent 未产出 debug_trace.md 或 debug_status.json 或超时）：
- 主 agent 自己写一个 `{output_dir}/debug_status.json` 作为兜底，标明：
  ```json
  {
    "schema_version": 1,
    "phase8_outcome": "timeout | crashed | missing_artifacts",
    "started_at": "...",
    "ended_at": "...",
    "crash_reason": "subagent timeout after 5400s" | "subagent exit code N" | "debug_trace.md missing",
    "final_failure_type": "<继承 Phase 7 final_status.failure_type>",
    "notes": "主 agent 兜底产出，subagent 未完成"
  }
  ```
- **不 append trace.md**，不重跑 evaluate
- 任务以 `final_status.failure_type`（来自 Phase 7）+ `debug_status.phase8_outcome` 作为最终状态

### 8.5 不 spawn 场景
- 主 agent 自己写一个 `debug_status.json`，标 `phase8_outcome: skipped`，`skip_reason` 填 `final_status.debug_eligible_reason` 的反面
- 任务以 `final_status.failure_type` 作为最终状态

### 8.6 下游自动化消费约定（v3 新增）

任何自动化消费（benchmark 报告、ideapool 统计）的最终状态读取顺序：
1. 优先读 `{output_dir}/debug_status.json`（如存在）
2. `debug_status` 里的 `phase8_outcome` ∈ {`success`, `failed`, `stopped_by_gate`, `timeout`, `crashed`, `skipped`, `missing_artifacts`}
3. 若只有 `trace.md.final_status`（Phase 8 没跑/没落盘）→ 用 final_status
4. final_status 是 Phase 7 时刻快照，`debug_status` 才是任务最终 verdict
```

### 2.6 工作流图更新（L42-51）

```
Phase 0: 参数确认
Phase 1: 环境准备
Phase 2: INPUT_CASES 精简
Phase 3: TileLang 设计表达   (max 5)
Phase 4: AscendC 转译与验证  (max 2)
Phase 5: 性能分析
Phase 6: 全量用例验证（只读）
Phase 7: Trace 记录 + final_status
Phase 8: Debug Subagent（条件性 spawn）
```

### 2.7 约束表更新（L548）

- "Phase 4 最大迭代" 3 → 2
- 新增 "Phase 6 修复 | 禁止"
- 新增 "Phase 8 subagent timeout | 5400s，超时/异常不重试"

---

## Section 3: subagent skill 重构

### 3.1 `ascendc-debug-agent` skill 文件结构（重构后）

```
skills/ascendc/ascendc-debug/
├── SKILL.md                    (大改：Step 0.3 分流 + 5 条 Step 1 分支)
├── STRUCTURE.md                (更新目录说明)
├── README.md                   (更新 scope)
├── scripts/
│   ├── eval_status.py          (新增：eval_status.json 解析器)
│   ├── precision_gate.py       (重构：只做路由，派发到 gates/*)
│   ├── precision_forensics.py  (不变)
│   ├── precision_knowledge.py  (不变)
│   ├── anticheat.py            (不变)
│   ├── _forensics_child.py     (不变)
│   ├── test_executor_parity.py (不变)
│   └── gates/
│       ├── __init__.py
│       ├── common.py           (新增：通用层 F/A/V)
│       ├── branch_precision.py (抽离原 precision_gate 的精度逻辑)
│       ├── branch_build.py     (新增)
│       ├── branch_import.py    (新增)
│       ├── branch_runtime.py   (新增)
│       └── branch_timeout.py   (新增)
└── references/
    └── ... (不变)
```

### 3.2 `SKILL.md` 改动

#### 3.2.1 frontmatter

```diff
  subagent:
    enabled: true
-   timeout: 3600
+   timeout: 5400
    max_iterations: 60
```

#### 3.2.2 "What I do"

```diff
- 修复精度测试失败的 AscendC 算子。
+ 修复 AscendC 算子的 build / import / runtime / timeout / precision 五类失败。
  流程:
- 1. Python 脚本收集数值取证数据 (确定性, 不可绕过)
+ 1. 读取 eval_status.json 确定 failure_type, 分流到对应 Step 1 分支
  2. Agent 结合上下文 + 代码 + 知识库做深度分析...
  3. Agent 修复代码
  4. 重新编译 + 验证
  5. 根据 Gate 循环控制信号决定继续或停止
```

#### 3.2.3 "When to use me"

```diff
- 当 evaluate_ascendc.sh 报告 Numerical 失败时（非 Build/Import 失败）。
- 前提: {task_dir}/kernel/pybind11.cpp 已存在且编译通过, 运行但精度不通过。
+ 当主 agent Phase 7 判定 debug_eligible == true（含 failure_type 白名单判断）。
+
+ 前提（通用）:
+ - {task_dir}/kernel/ 下至少一个 .cpp 文件
+ - {task_dir}/model_new_ascendc.py 未 AST 退化
+ - {task_dir}/trace.md 末尾含 final_status JSON block
+ - {task_dir}/.eval_status/latest.json 存在
```

#### 3.2.4 新增 Step 0.3（分流，v3 修订：禁止跨分支跳转）

```markdown
### Step 0.3: 读 final_status + eval_status，锁定分支

```bash
cat "{task_dir}/trace.md" | awk '/^```json/,/^```$/' | jq '...'
cat "{task_dir}/.eval_status/latest.json"
```

**分支选择规则**（v3 明确）：
- 入口按 `final_status.failure_type` 选定分支 — 即 `session_branch`
- `session_branch` 在整个 subagent session **只锁定一次**，之后不再切换
- `import_failed` 还要读 `final_status.import_subtype`：
  - `import_kernel_side` → 进入 Step 1-I
  - `import_env_side` → 本就不应进入 subagent（主 agent 已过滤）；若异常进入，直接写 `debug_trace.md` + `debug_status.json` 标 `skipped_env_issue` 后退出

**跨分支跳转规则（v3）**：
- 某轮修复后，`eval_status` 显示 failure_type 变化（如 `build_failed` → `precision_failed`）：
  - 视为"本分支 Gate-V 取得进展"（build 分支：failed_step 从 compile 推进到 verify = 进步）
  - **但不切换分支**，subagent 结束本次 session，`debug_trace.md` / `debug_status.json` 标 `session_outcome: progressed_to_new_failure_type`
  - 由主 agent 根据新 failure_type 决定是否再次 spawn（本版本主 agent 不自动二次 spawn，人工判断）
- 原因：跨分支会导致 audit schema、Gate 语义、debug_trace 模板同时漂移，风险远大于收益

根据 `session_branch` 选择 Step 1 分支：
- `precision_failed` → Step 1-P（现有取证路径，保持不变）
- `build_failed`    → Step 1-B
- `import_failed` + `import_kernel_side` → Step 1-I
- `runtime_error`   → Step 1-R
- `timeout`         → Step 1-T

其他值若异常进入，写 `debug_trace.md` + `debug_status.json` 标 `skipped_unsupported_type` 后退出。
```

#### 3.2.5 新增 Step 1-B / I / R / T（每条 ~30 行）

每条分支结构相同：

```markdown
### Step 1-X: <Build/Import/Runtime/Timeout> Error Analysis

**输入**:
- {task_dir}/.eval_status/latest.json  (结构化状态 + log 路径)
- {task_dir}/.eval_logs/phase{N}_attempt{M}.log  (原始 log)
- {task_dir}/kernel/*.cpp / *.h
- {task_dir}/trace.md  (Phase 1-7 上下文)

**Agent 任务**:
1. 读 log，提取 error 摘要 / stack trace / 关键提示
2. 对照对应 references 找 API 用法差异
3. 写 {task_dir}/precision_tuning/audit_{attempt}.md 含:
   - [DIAGNOSIS] (通用 section)
   - [FIX_PLAN] (通用 section)
   - [X_SPECIFIC_CITATION] (分支专属 section)
4. 修复 kernel/
5. 通过 Gate-通用 + Gate-X 验证

**推荐参考资料**:
- Step 1-B: ascendc-translator/references/dsl2Ascendc_compute_*.md, dsl2Ascendc_host.md
- Step 1-I: ascendc-translator/references/dsl2Ascendc_host.md (pybind),
            skills/ascendc/ascendc-debug/references/（环境变量相关）
- Step 1-R: ascendc-translator/references/dsl2Ascendc_cross_core_sync.md,
            AscendCVerification.md
- Step 1-T: ascendc-translator/references/dsl2Ascendc_cross_core_sync.md
```

#### 3.2.6 结束时写 `debug_trace.md`（v3 降级：4 节强制 + 可选附录）

**原则**：`debug_trace.md` 详细程度对标 `trace.md`，但**只强制有可靠数据源的 section**。v2 的 8 节里有两节（知识库检索、Step 级细分耗时）没有统一数据源（build/runtime/timeout 分支无知识库检索机制；eval_wrapper 只覆盖 evaluate，不覆盖分析/修复），v3 降级为可选附录。

subagent 退出前（无论成功 / 失败 / stopped）必须写 `{task_dir}/debug_trace.md`：

```markdown
# AscendC Debug Trace

## 1. Phase 8 入口快照（强制）
- 调用时间: <ISO timestamp>
- task_dir: {task_dir}
- session_branch: <1-P / 1-B / 1-I / 1-R / 1-T>
- 上游 trace.md.final_status 的完整 JSON 快照（含 ac_iterations 历史）
- 上游 .eval_status/latest.json 完整快照
- 进入时 kernel/ 基线快照路径（baseline/code_snapshot）

## 2. 迭代历史（强制，每轮一节）

### Attempt 0
- 进入时 eval_status 关键字段: failure_type, failed_step, duration_sec, exit_code
- 诊断摘要: 引用 audit_0.md (或对应分支 audit 文件) 的摘要 section
- 修复代码改动: 修改文件列表 + 函数/行号级 diff 摘要（不贴全文）
- Gate-通用: PASS/FAIL + 未通过项
- Gate-分支 (F/A/V): PASS/FAIL + 关键数值
  - 精度: mismatch_ratio / max_abs_diff 变化
  - 构建: failed_step 推进情况
  - 运行时: crash 是否消除 / 位置变化
  - 超时: duration 变化
  - import: import.status 变化
- 本轮退出 eval_status 快照
- outcome: passed / improved / stagnant / regressed

### Attempt 1...N（同上）

## 3. 最终 Verdict（强制）
- session_outcome: success / failed / stopped_by_gate / stopped_by_loop_limit / progressed_to_new_failure_type / timeout / skipped_*
- 退出时 eval_status 快照
- 若 success: 确认全量 .json.bak 恢复后 verify 通过
- 若 failed / stopped: 明确原因
- 若 progressed_to_new_failure_type: 新 failure_type 是什么（主 agent 读取后自行决定是否二次 spawn）

## 4. 产物清单（强制）
- 各轮 audit 文件相对 {task_dir} 路径
- tuning_directions.json（精度分支）或对应分支的方向记录文件
- history/baseline/code_snapshot/ 和各轮 attempt_N/code_snapshot/
- .eval_status/phase8_attempt_*.json
- debug_status.json 路径

## 附录 A: 走偏点（可选但推荐）
- 尝试失败的修复方向摘要（对应 tuning_directions.json outcome ∈ {stagnant, regressed}）
- 平台 / API 限制 workaround
- 反作弊触发记录

## 附录 B: 知识库检索记录（仅精度分支 + 其他分支若有）
- search 调用次数 + 主要关键词
- 命中的 knowledge entries

## 附录 C: 耗时细分（可选）
- 总 wall clock（eval_wrapper 各轮 JSON 的 started_at/ended_at 差值之和 + subagent 整体运行时间）
- 若能区分 Step 级耗时则列出，不能则写"未精细记录"
```

**强制要求**：
- 只强制 4 节（入口快照 / 迭代历史 / Verdict / 产物清单），其余为附录
- 第 2 节每一轮都不可省略
- 第 4 节产物清单必须是 `{task_dir}` 下的相对路径

**语言 / 格式**：
- 中文为主，代码 / 路径 / 识别符用英文
- Markdown 层级严格 `## 1. ... ## 2. ...`
- JSON 快照用 fenced code block 内嵌

**与 `trace.md` / `debug_status.json` 的关系**：
- `trace.md` 末尾 `final_status` 是 Phase 7 历史快照，**Phase 8 中不被修改**
- `debug_trace.md` 是 Phase 8 的完整叙事记录（给人读）
- `debug_status.json` 是 Phase 8 的机器可读 verdict（给自动化读）
- 三份文件按时间顺序：Phase 7 写 trace.md/final_status → Phase 8 写 debug_trace.md + debug_status.json

### 3.3 `precision_gate.py` 重构

**原 `precision_gate.py`（~600 行，精度语义硬编码）拆成**：

**① `precision_gate.py`（重构为入口路由器）**
```python
def main(args):
    eval_status = load_eval_status(args.task_dir)
    failure_type = eval_status["failure_type"]

    # 先跑通用层
    common_result = gates.common.check(args.step, eval_status, args.attempt, args.task_dir)
    if not common_result.ok:
        return common_result.to_gate_output()

    # 再派发分支层
    branch = select_branch(failure_type)  # precision/build/import/runtime/timeout
    branch_result = branch.check(args.step, eval_status, args.attempt, args.task_dir)
    return branch_result.to_gate_output()
```

**② `gates/common.py`（通用层，v3 修订：不强制统一 audit schema）**

每个阶段 F/A/V 各一个函数，检查**所有分支都成立的不变量**：
- `task_dir` 目录结构完整
- 反作弊 hash 未破坏（提升到通用层）
- AST 退化未引入（同上）
- `eval_status.json` 产出完整
- `{op}.json.bak` 未被破坏
- **audit 文件存在**（文件级检查，section schema 由分支层各自负责 — v2 里要求通用层统一检查 `[DIAGNOSIS]+[FIX_PLAN]` 会与精度分支现有 `[FORENSICS_SUMMARY]+[REFERENCE_IMPL_SPEC]` 冲突，v3 让分支层各自定义 section）

**v3 明确契约**：
- 通用层**只检查**反作弊 / 文件存在 / `eval_status` / baseline / 目录完整性
- 通用层**不检查** audit section 格式（避免和精度分支现有 schema 打架）
- `fix` 步骤（Step 3 代码修改）不单独设 Gate — 由 Gate-V 在下一轮开始时通过 `eval_status` 差分间接验证（否则会变成多设一层无必要的 gate）

**③ `gates/branch_precision.py`**（抽离原 `precision_gate.py` 的精度逻辑，**schema 完全沿用现有**）

- Gate-F: `forensics_report_{attempt}.json` 存在+可解析，`baseline_state.json`
- Gate-A: audit 含 `[FORENSICS_SUMMARY]` `[REFERENCE_IMPL_SPEC]` `[ROOT_CAUSE]` `[FIX_PLAN]`（精度分支原有 section，不改）
- Gate-V: `mismatch_ratio` / `max_abs_diff` 趋势、loop_signal

**④ `gates/branch_build.py`（新增）**

- Gate-F: `.eval_logs/` 内最新 build log 存在，含 compile 错误块
- Gate-A: audit 含 `[COMPILE_ERROR_CITATION]` `[ROOT_CAUSE]` `[FIX_PLAN]`，fix_type ∈ build_fix_whitelist
- Gate-V: 新一轮 `eval_status.failed_step` 从 `compile` 推进到 `execute` 或更后

**⑤ `gates/branch_import.py`（新增，v3 修订：只处理 kernel_side）**

仅在 `session_branch == import_kernel_side` 时生效。`import_env_side` 不应进到此分支（主 agent 已过滤）。

- Gate-F: import traceback log 存在且 `import_subtype == import_kernel_side`
- Gate-A: audit 含 `[IMPORT_TRACEBACK_CITATION]` `[ROOT_CAUSE]` `[FIX_PLAN]`，fix_type ∈ import_kernel_fix_whitelist（如 `pybind_symbol_fix` / `kernel_ext_name_fix` / `kernel_export_fix`）
- Gate-V: 新一轮 `import.status == passed`

**不在白名单的 import_fix_type（环境层问题）**：
`ld_path_fix` / `abi_fix` / `toolkit_env_fix` / 任何需要改 `CMakeLists.txt` / `setup.py` / `utils/build_ascendc.py` 的修复 — 这些不由本 subagent 处理，因为硬约束只允许改 `kernel/`。Gate-A 识别到这类 fix_type 会直接 reject。

**⑥ `gates/branch_runtime.py`（新增）**

- Gate-F: 运行时 stderr / core dump 路径存在
- Gate-A: audit 含 `[RUNTIME_ERROR_CITATION]` `[ROOT_CAUSE]` `[FIX_PLAN]`
- Gate-V: 新一轮不再 crash 或 crash 位置不同（进步）

**⑦ `gates/branch_timeout.py`（新增）

- Gate-F: `eval_status.duration_sec` ≥ 配置阈值
- Gate-A: audit 含 `[SYNC_POINT_ANALYSIS]`，fix_type 指向同步/tiling
- Gate-V: 新一轮在时限内完成（无论对错）

### 3.4 `eval_status.py`（新增）

~80 行的小工具：
- `load_eval_status(task_dir) -> dict`
- `load_latest_status(task_dir) -> dict`
- `load_phase_status(task_dir, phase, attempt) -> dict`
- schema 校验

---

## Section 4: subagent agent 文件改动（`agents/ascendc-debug-agent-discovery.md`）

三处最小修改：

**① description**
```diff
- AscendC 算子精度调优 Agent（发现式审计）
+ AscendC kernel debug Agent（发现式审计，覆盖 build/import/runtime/timeout/precision 五类失败）
```

**② argument-hint 前提**
与 SKILL.md §3.2.3 一致。

**③ Role Definition 扩写**
```diff
- 精度诊断专家: 基于数值取证数据和代码分析, 定位精度问题根因
+ kernel debug 专家: 根据 failure_type 选择诊断路径（session 内锁定一条分支）
+   - precision_failed: 数值取证 + 精度反模式匹配
+   - build_failed: 编译错误定位 + API 用法核对
+   - import_failed (kernel_side): pybind 符号 / kernel 导出修复（env_side 不处理）
+   - runtime_error: 运行时错误 / 段错误 / stack trace 分析
+   - timeout: 死锁 / 同步缺失 / tiling 配置异常分析
```

**④ 反作弊约束完全不变**（L96-122）。

---

## Section 5: trace-recorder skill 改动

### 5.1 目标

Phase 7 收尾时，不再"事后猜 failure_type"，直接读 `eval_status.json` 填 `final_status` block。

### 5.2 改动（v3 修订）

- `SKILL.md` 加一个 Step：读 `{task_dir}/.eval_status/latest.json` + **Phase 4 迭代历史（来自主 agent in-memory）** + 目录状态（kernel 是否空、是否 degraded），按 §1.4 判定规则组装 `final_status`
- 结束时将 `final_status` fenced JSON block 追加到 `trace.md` 末尾
- 计算 `debug_eligible` 字段
- **Phase 4 迭代历史作为 `final_status.ac_iterations` 字段直接内嵌**（v3 取消独立 `ac_history.json` 文件，避免与 `trace.md` / `PRIOR_TRACE_CONTEXT` 三份冗余）

### 5.3 判定优先级

1. `kernel/` 目录为空 → `no_kernel` + `debug_eligible=false`
2. `model_new_ascendc.py` AST 退化 → `degraded` + `debug_eligible=false`
3. Phase 3 失败且未进到 Phase 4 → `tilelang_only_failed` + `debug_eligible=false`
4. `eval_status.latest.json.failure_type == execution_aborted` → 照搬 + `debug_eligible=false`（环境问题 subagent 无法处理）
5. 否则 → 读 `eval_status.latest.json.failure_type`，照搬
6. `debug_eligible` 计算规则：
   - 先看 `failure_type ∈ {precision_failed, build_failed, import_failed, runtime_error, timeout}` 且 `has_kernel && !has_degradation`
   - 若 `failure_type == import_failed`，还要额外判定 `import_subtype == import_kernel_side`，否则 false

---

## Section 6: 删除（v3）

**v2 的 "Section 6: `ac_history.json`" 整节在 v3 删除。**

原因：
- 与 `trace.md` / `PRIOR_TRACE_CONTEXT` 机制重叠
- 增加了一份跨主/子 agent 边界的文件，维护成本高
- Phase 4 迭代历史直接放 `final_status.ac_iterations` 字段即可（主 agent 在 Phase 7 调 trace-recorder 时传入）

---

## Section 7: 关键不变量（v3 修订）

1. **反作弊约束不动** — 从 subagent agent 移到 Gate-通用层，所有分支共享；`kernel/` 是唯一可修改目录
2. **主 agent 不兜底 subagent** — Phase 8 结束后主 agent 不做任何 verify / fix
3. **`trace.md` 全程只读**（v3）— Phase 7 写完后，主 agent 绝不 append；所有 Phase 8 相关信息（成功/失败/timeout/crash/skipped）都落到 `debug_trace.md` + `debug_status.json`
4. **subagent 自包含** — subagent 正常退出必须产出 `debug_trace.md` + `debug_status.json` 两个文件；任一缺失视为异常退出，由主 agent 兜底写 `debug_status.json`
5. **eval_status.json 是唯一事实源** — 下游（trace-recorder / Phase 8 spawn / subagent Step 0.3 / Gate）全部消费它，不重复推断
6. **Gate 通用层单点防守** — 反作弊、结构校验、AST 退化、文件存在性只在通用层做一次；audit section schema 由分支层各自定义（v3 不再在通用层强制 `[DIAGNOSIS]+[FIX_PLAN]`）
7. **session 分支锁定**（v3）— subagent 一次 session 只走一条 Step 1 分支；若修复后 failure_type 变化，视为"进步"并退出，不跨分支跳转
8. **`import_failed` 环境子类不进 subagent**（v3）— `import_env_side` 由主 agent 在 trace-recorder 阶段过滤掉（`debug_eligible=false`）
9. **Phase 4 内部 `execution_aborted` 子类 ssh_disconnected / docker_unreachable 重试 1 次** — 不暴露给主循环

---

## Section 8: 不做（明确排除，v3 修订）

- 不新增 `infra_error` 为独立 `failure_type`（统一归到 `execution_aborted` 兜底类型）
- 不对 subagent timeout/crash 重试
- 不修改 `utils/verification_ascendc.py` / `utils/build_ascendc.py`（仅加 `utils/eval_wrapper.py`）
- 不改 Phase 3 TileLang 循环（scope 外）
- 不引入新 MCP 工具
- 不造 build/runtime/timeout 专属的独立 subagent（codex 的 B 方案）— 选 A 方案：统一 subagent + 分支 Gate
- 不跨分支跳转（v3 新增）— subagent 一次 session 固定在入口分支
- 主 agent Phase 8 结束后**不 append trace.md**（v3 统一规则，包括异常退出）
- 不在通用层强制统一 audit schema（v3 — 避免和精度分支现有 section 打架）
- 不处理 `import_env_side` / `execution_aborted`（超出 subagent scope，写进 `debug_status.json.skip_reason` 后退出）
- 主 agent 不自动二次 spawn（即使 subagent 报 `progressed_to_new_failure_type`）— 本版本留给人工判断
- 不实现独立 `ac_history.json`（v3 删除，并入 final_status.ac_iterations）

---

## Section 9: 工作量估算（v3 修订，范围化）

工作量会因"POC 通跑"还是"production 质量"差异较大，给一个范围。

| 工作项 | 估时（人日） | 先后 |
|-------|------------|------|
| `utils/eval_wrapper.py` + failure_type 分类 + timeout_marker + abort_subtype + import_subtype + 单测 | 2 | 先行 |
| `skills/ascendc/ascendc-debug/scripts/eval_status.py` | 0.5 | 先行 |
| `precision_gate.py` 拆包 + `gates/common.py` + `gates/branch_precision.py` + 回归 | 2.5 | 依赖 eval_wrapper |
| `gates/branch_build.py` / `branch_import.py` / `branch_runtime.py` / `branch_timeout.py` | 2-3 | 依赖 common + precision |
| `SKILL.md` Step 0.3 + Step 1-B/I/R/T + Step 5 debug_trace 模板 + debug_status.json 规范 | 1 | 依赖 gates |
| `trace-recorder` skill 改读 `eval_status.json` + 产 `final_status`（含 ac_iterations 内嵌） | 0.5 | 依赖 eval_wrapper |
| 新建 `ascend-kernel-developer-with-ascendc-debug.md` + Phase 4/6/7/8 改动 + Phase 4 `execution_aborted` 重试逻辑 | 1-1.5 | 依赖上述 |
| `agents/ascendc-debug-agent-discovery.md` description / Role / 前提扩写 | 0.3 | 并行 |
| 原 `ascend-kernel-developer.md` 恢复 main | 0.1 | 任意 |
| 集成联调：5 类失败 fixture + 端到端跑通 + 边界验证 | 2-4 | 最后 |
| **合计** | **~12-16 人日** | — |

**主要风险点**（最可能超预期的三个）：
1. `eval_wrapper` 的 `failure_type` 分类边界（timeout_marker 实际效果、import_subtype 模式匹配覆盖率）
2. `precision_gate.py` 拆包后的精度分支回归（保证现有精度流水不坏）
3. 5 类失败的端到端 fixture 构造（build_failed / import_failed / runtime_error / timeout 各要造一个可复现的坏 kernel）

---

## Section 10: 待确认事项（v3 修订）

1. eval_wrapper 的 `timeout_marker` 实现细节（sentinel 文件的命名 / 原子写 / wrapper 崩溃时的清理）
2. `abort_subtype` 的识别规则 — 哪些 return code / stderr 模式算 `ssh_disconnected`（SSH return 255？），哪些算 `docker_unreachable`（`container not running` 提示？）
3. `import_subtype` 的 traceback 模式匹配规则 — 如何从 traceback 字符串区分 `kernel_side` vs `env_side`（基于 `ImportError` 文本 / 丢失符号名 / 丢失的 .so 路径）
4. `gates/branch_import.py` 的 `import_kernel_fix_whitelist` 具体包含哪些 fix_type
5. main 分支 `agents/ascend-kernel-developer.md` 的恢复边界（先 diff 再定）
6. `debug_trace.md` 和 `debug_status.json` 的具体 schema 样例（最好起草一份 reference doc）
7. Phase 4 `execution_aborted` 子类的重试阈值（重试 1 次？2 次？立即 fail？）
8. trace-recorder 接收主 agent Phase 4 迭代历史的接口（通过命令行参数 / 环境变量 / stdin JSON？）

这些都是实施阶段才需要敲定的细节，不影响当前设计方向。
