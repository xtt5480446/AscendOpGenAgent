#!/bin/bash
# 批量调度 codex CLI 跨多个 docker 容器 + 多 NPU 执行 AscendC 算子精度调优
#
# 动态工作队列：容器间并行（每容器绑定一张 NPU），容器内串行；
# 谁先完成当前任务，就从共享队列拉下一个，不做预分配。
#
# 典型用法:
#   bash utils/run_ascendc_debug.sh \
#        --task-list tasks/precision_tasks.txt \
#        --agent agents/ascendc-debug-agent-discovery.md \
#        --containers cjm_cann1,cjm_cann2 \
#        --npus 0,1 \
#        --output /home/c00959374/AscendOpGenAgent/outputs/precision_$(date +%Y%m%d_%H%M)
#
# task-list 格式: 每行一个任务目录的绝对路径（支持 # 注释行），例如:
#   /home/c00959374/AscendOpGenAgent/outputs/codex_batch_20260420_1755/3_Add
#   /home/c00959374/AscendOpGenAgent/outputs/codex_batch_20260420_1755/5_Mul
#
# 也可用 --task-dirs 传逗号分隔的目录列表（适合少量任务）:
#   --task-dirs /path/to/3_Add,/path/to/5_Mul

set -euo pipefail

# ── 默认值 ──
TASK_LIST=""
TASK_DIRS=""
AGENT_FILE="agents/ascendc-debug-agent-discovery.md"
CONTAINERS=""
NPUS=""
OUTPUT_DIR=""
TIMEOUT_SEC="3600"          # 单任务超时（秒），默认 1 小时
WORKDIR_IN_CONTAINER="/home/c00959374/AscendOpGenAgent"
TILELANG_ENV_SH="/home/c00959374/tilelang/tilelang-ascend/set_env.sh"
PROMPT_TEMPLATE='严格按照 __AGENT_FILE__ 中定义的 agent 规范执行算子精度调优。

任务参数:
- task_name: __TASK_NAME__
- task_dir (绝对路径): __TASK_DIR__
- npu: __NPU__

执行要求:
1. 首先 Read 并完整理解 agent 规范文件: __AGENT_FILE__
2. 按该规范中定义的精度调优流程逐步执行（数值取证 → 代码审计 → 修复 → Gate 验证循环）
3. 【反作弊硬约束】只允许修改 __TASK_DIR__/kernel/ 目录下的 AscendC 源文件（.cpp / .h / pybind11.cpp）；
   严禁修改 __TASK_DIR__/model_new_ascendc.py、__TASK_DIR__/model_new_tilelang.py、__TASK_DIR__/model.py；
   任何通过改写 wrapper 引入 PyTorch 退化路径、绕过 kernel 调用、或用 torch.* / F.* 计算掩盖精度失败的行为均视为作弊；
   bench 在任务前后会对 wrapper 文件做 sha256 hash 对比 + validate_ascendc_impl.py AST 退化检测，命中即自动恢复 baseline 并将任务标记为 🚨 CHEAT。
4. 精度调优脚本路径: __WORKDIR__/skills/ascendc/ascendc-debug/scripts/
5. NPU 设备已通过环境变量 ASCEND_RT_VISIBLE_DEVICES=__NPU__ 暴露
6. 若 precision_forensics.py / precision_gate.py 需要 --task-dir 参数，使用 __TASK_DIR__
7. 全程不要向用户询问或等待交互；遇到分支/决策均按 agent 规范定义的默认路径处理
8. 结束时输出精度调优结果摘要（是否通过、最终 max_abs_diff、修改了哪些 kernel/ 文件）'

ANTICHEAT_SCRIPT="skills/ascendc/ascendc-debug/scripts/anticheat.py"

# ── 参数解析 ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --task-list)    TASK_LIST="$2"; shift 2 ;;
        --task-dirs)    TASK_DIRS="$2"; shift 2 ;;
        --agent)        AGENT_FILE="$2"; shift 2 ;;
        --containers)   CONTAINERS="$2"; shift 2 ;;
        --npus)         NPUS="$2"; shift 2 ;;
        --output)       OUTPUT_DIR="$2"; shift 2 ;;
        --timeout)      TIMEOUT_SEC="$2"; shift 2 ;;
        --workdir)      WORKDIR_IN_CONTAINER="$2"; shift 2 ;;
        --tilelang-env) TILELANG_ENV_SH="$2"; shift 2 ;;
        -h|--help)
            sed -n '1,28p' "$0"
            exit 0
            ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# ── 校验 ──
[[ -z "$TASK_LIST" && -z "$TASK_DIRS" ]] && { echo "错误: 必须指定 --task-list 或 --task-dirs"; exit 1; }
[[ -z "$CONTAINERS" ]]  && { echo "错误: 必须 --containers (逗号分隔)"; exit 1; }
[[ -z "$NPUS" ]]        && { echo "错误: 必须 --npus (逗号分隔，与 containers 一一对应)"; exit 1; }
[[ -z "$OUTPUT_DIR" ]]  && { echo "错误: 必须 --output"; exit 1; }

IFS=',' read -ra CONTAINER_ARR <<< "$CONTAINERS"
IFS=',' read -ra NPU_ARR <<< "$NPUS"
(( ${#CONTAINER_ARR[@]} == ${#NPU_ARR[@]} )) \
    || { echo "错误: containers 与 npus 数量不一致"; exit 1; }

# ── 构造任务目录列表 ──
TASK_DIR_LIST=()
if [[ -n "$TASK_LIST" ]]; then
    [[ -f "$TASK_LIST" ]] || { echo "错误: task-list 文件不存在: $TASK_LIST"; exit 1; }
    while IFS= read -r line; do
        line="${line%%#*}"           # 去行内注释
        line="$(echo "$line" | xargs)" # trim 首尾空白
        [[ -n "$line" ]] && TASK_DIR_LIST+=("$line")
    done < "$TASK_LIST"
else
    IFS=',' read -ra TASK_DIR_LIST <<< "$TASK_DIRS"
fi

(( ${#TASK_DIR_LIST[@]} > 0 )) || { echo "错误: 无可用任务目录"; exit 1; }

mkdir -p "$OUTPUT_DIR"
QUEUE="$OUTPUT_DIR/.queue"
LOCK="$OUTPUT_DIR/.lock"
REPORT="$OUTPUT_DIR/batch_report.md"

# ── 初始化队列（每行一个 task_dir 绝对路径） ──
: > "$QUEUE"
for td in "${TASK_DIR_LIST[@]}"; do
    echo "$td" >> "$QUEUE"
done
: > "$LOCK"

# ── 初始化报告 ──
{
    echo "# Precision Tuning 批量执行报告"
    echo
    echo "- agent: $AGENT_FILE"
    echo "- 任务数: ${#TASK_DIR_LIST[@]}"
    echo "- containers: $CONTAINERS"
    echo "- npus: $NPUS"
    echo "- timeout: ${TIMEOUT_SEC}s/task"
    echo "- start: $(date '+%F %T')"
    echo
    echo "| task_name | 状态 | 耗时(s) | 容器@NPU |"
    echo "|-----------|------|---------|----------|"
} > "$REPORT"

TOTAL=${#TASK_DIR_LIST[@]}
echo "================================================================"
echo "总任务数: $TOTAL    workers: ${#CONTAINER_ARR[@]}    timeout: ${TIMEOUT_SEC}s"
echo "agent: $AGENT_FILE"
for i in "${!CONTAINER_ARR[@]}"; do
    echo "  worker[$i]: ${CONTAINER_ARR[$i]} → npu=${NPU_ARR[$i]}"
done
echo "================================================================"

# ── 反作弊：调用独立 Python 脚本（容器内执行）──
# 用法: anticheat_run <container> <subcmd> <task_dir> [extra_args...]
anticheat_run() {
    local container="$1" subcmd="$2" task_dir="$3"; shift 3
    docker exec "$container" bash -lc "
        cd '$WORKDIR_IN_CONTAINER'
        python3 '$ANTICHEAT_SCRIPT' $subcmd '$task_dir' $*
    "
}

# ── 全量验证后置检查：若存在 <op>.json.bak，则要求产出 full_validation_result_attempt_*.json 且通过 ──
fullcase_check() {
    local task_dir="$1"
    python3 - "$task_dir" <<'PY'
import json
import os
import re
import sys

task_dir = sys.argv[1]
result = {
    "required": False,
    "passed": True,
    "reason": "",
    "result_path": None,
    "case_backup": None,
}

try:
    bak_files = sorted(
        name for name in os.listdir(task_dir)
        if name.endswith(".json.bak")
    )
except OSError:
    result["passed"] = False
    result["reason"] = "task_dir_unreadable"
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit

if not bak_files:
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit

result["required"] = True
result["case_backup"] = bak_files[0]

tuning_dir = os.path.join(task_dir, "precision_tuning")
pattern = re.compile(r"full_validation_result_attempt_(\d+)\.json$")
candidates = []
if os.path.isdir(tuning_dir):
    for name in os.listdir(tuning_dir):
        match = pattern.fullmatch(name)
        if match:
            candidates.append((int(match.group(1)), os.path.join(tuning_dir, name)))

if not candidates:
    result["passed"] = False
    result["reason"] = "missing_full_validation_result"
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit

_, latest_path = max(candidates, key=lambda item: item[0])
result["result_path"] = latest_path

try:
    with open(latest_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
except Exception:
    result["passed"] = False
    result["reason"] = "full_validation_result_parse_error"
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit

if not payload.get("used_full_cases", False):
    result["passed"] = False
    result["reason"] = "full_validation_not_marked_full_cases"
elif not payload.get("correctness_passed", False):
    result["passed"] = False
    result["reason"] = "full_validation_failed"

print(json.dumps(result, ensure_ascii=False))
PY
}

# ── worker：从队列拉任务，跨 docker 执行 codex ──
run_worker() {
    local container="$1" npu="$2"
    local wlog="$OUTPUT_DIR/worker_${container}.log"
    : > "$wlog"

    while true; do
        local task_dir=""
        # 原子出队
        exec 9>"$LOCK"
        flock -x 9
        if [[ -s "$QUEUE" ]]; then
            task_dir=$(head -n1 "$QUEUE")
            sed -i '1d' "$QUEUE"
        fi
        flock -u 9
        exec 9>&-

        [[ -z "$task_dir" ]] && break

        local task_name; task_name=$(basename "$task_dir")
        mkdir -p "$task_dir/precision_tuning"
        local task_log="$task_dir/precision_tuning/agent_output.log"
        local exec_log="$task_dir/precision_tuning/exec.log"

        # 替换 prompt 占位符
        local prompt="${PROMPT_TEMPLATE//__AGENT_FILE__/$WORKDIR_IN_CONTAINER/$AGENT_FILE}"
        prompt="${prompt//__TASK_NAME__/$task_name}"
        prompt="${prompt//__TASK_DIR__/$task_dir}"
        prompt="${prompt//__NPU__/$npu}"
        prompt="${prompt//__WORKDIR__/$WORKDIR_IN_CONTAINER}"

        echo "[${container}@npu${npu}] 开始: $task_name" | tee -a "$wlog"

        # 反作弊前置：保存 wrapper 基线
        anticheat_run "$container" snapshot "$task_dir" >/dev/null 2>&1 || true

        local start end elapsed status
        start=$(date +%s)

        set +eo pipefail
        timeout --signal=TERM --kill-after=30 "$TIMEOUT_SEC" \
            docker exec \
                -e "ASCEND_RT_VISIBLE_DEVICES=$npu" \
                -e "CODEX_PROMPT=$prompt" \
                "$container" bash -lc "
                    set -e
                    [ -f '$TILELANG_ENV_SH' ] && source '$TILELANG_ENV_SH'
                    cd '$WORKDIR_IN_CONTAINER'
                    codex exec \
                        --dangerously-bypass-approvals-and-sandbox \
                        --skip-git-repo-check \
                        --output-last-message '$task_log' \
                        -- \"\$CODEX_PROMPT\"
                " 2>&1 | tee -a "$wlog" > "$exec_log"
        status=${PIPESTATUS[0]}
        set -eo pipefail

        end=$(date +%s); elapsed=$((end - start))

        # 反作弊后置：hash 对比 + AST 退化检测
        local cheat_json; cheat_json=$(anticheat_run "$container" verify "$task_dir" --json 2>/dev/null || true)
        local cheat_verdict cheat_reasons
        cheat_verdict=$(echo "$cheat_json" | python3 -c "
import sys, json
try:
    print(json.loads(sys.stdin.read()).get('verdict', 'UNKNOWN'))
except Exception:
    print('UNKNOWN')
" 2>/dev/null || echo "UNKNOWN")
        cheat_reasons=$(echo "$cheat_json" | python3 -c "
import sys, json
try:
    print(';'.join(json.loads(sys.stdin.read()).get('reasons', [])))
except Exception:
    print('')
" 2>/dev/null || echo "")

        if [[ "$cheat_verdict" == "CHEAT" ]]; then
            anticheat_run "$container" restore "$task_dir" >/dev/null 2>&1 || true
            echo "[${container}@npu${npu}] 🚨 CHEAT detected for $task_name: $cheat_reasons — baseline restored" | tee -a "$wlog"
        fi

        local fullcase_json fullcase_required fullcase_passed fullcase_reason
        fullcase_json=$(fullcase_check "$task_dir" 2>/dev/null || true)
        fullcase_required=$(echo "$fullcase_json" | python3 -c "
import sys, json
try:
    print('yes' if json.loads(sys.stdin.read()).get('required') else 'no')
except Exception:
    print('no')
" 2>/dev/null || echo "no")
        fullcase_passed=$(echo "$fullcase_json" | python3 -c "
import sys, json
try:
    print('yes' if json.loads(sys.stdin.read()).get('passed') else 'no')
except Exception:
    print('no')
" 2>/dev/null || echo "no")
        fullcase_reason=$(echo "$fullcase_json" | python3 -c "
import sys, json
try:
    print(json.loads(sys.stdin.read()).get('reason', ''))
except Exception:
    print('')
" 2>/dev/null || echo "")

        local icon
        if [[ "$cheat_verdict" == "CHEAT" ]]; then
            icon="🚨 CHEAT${cheat_note}"
            echo "[${container}@npu${npu}] 🚨 $task_name CHEAT (${elapsed}s)"
        elif [[ $status -eq 0 && "$fullcase_required" == "yes" && "$fullcase_passed" != "yes" ]]; then
            icon="❌ 失败(${fullcase_reason:-fullcase_missing})"
            echo "[${container}@npu${npu}] ❌ $task_name missing/failed full-case verification (${fullcase_reason:-unknown}, ${elapsed}s)" | tee -a "$wlog"
        elif [[ $status -eq 0 ]]; then
            icon="✅ 成功"
            echo "[${container}@npu${npu}] ✅ $task_name (${elapsed}s)"
        elif [[ $status -eq 124 ]]; then
            icon="⏱ 超时"
            echo "[${container}@npu${npu}] ⏱ $task_name TIMEOUT (${elapsed}s)"
        else
            icon="❌ 失败(rc=$status)"
            echo "[${container}@npu${npu}] ❌ $task_name rc=$status (${elapsed}s)"
        fi

        exec 9>"$LOCK"; flock -x 9
        echo "| $task_name | $icon | $elapsed | ${container}@npu${npu} |" >> "$REPORT"
        flock -u 9; exec 9>&-
    done
}

# ── 并行启动 workers ──
pids=()
for i in "${!CONTAINER_ARR[@]}"; do
    run_worker "${CONTAINER_ARR[$i]}" "${NPU_ARR[$i]}" &
    pids+=("$!")
done
for p in "${pids[@]}"; do wait "$p" || true; done

# ── 汇总 ──
SUCCESS=$(grep -c "✅ 成功" "$REPORT" || echo 0)
TIMEOUT_CNT=$(grep -c "⏱ 超时" "$REPORT" || echo 0)
FAIL=$(grep -c "❌ 失败" "$REPORT" || echo 0)
CHEAT=$(grep -c "🚨 CHEAT" "$REPORT" || echo 0)

{
    echo
    echo "## 汇总"
    echo
    echo "- 总数: $TOTAL"
    echo "- 成功: $SUCCESS"
    echo "- 超时: $TIMEOUT_CNT"
    echo "- 失败: $FAIL"
    echo "- 作弊: $CHEAT"
    echo "- 结束: $(date '+%F %T')"
} >> "$REPORT"

echo "================================================================"
echo "完成: ✅$SUCCESS  ⏱$TIMEOUT_CNT  ❌$FAIL  🚨$CHEAT  /  共 $TOTAL"
echo "报告: $REPORT"
echo "每 worker 日志: $OUTPUT_DIR/worker_<container>.log"
echo "每 task 日志:   <task_dir>/precision_tuning/exec.log + agent_output.log"
echo "================================================================"
