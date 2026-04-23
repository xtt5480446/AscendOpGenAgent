#!/bin/bash
# 批量调度 codex CLI 跨多个 docker 容器 + 多 NPU 执行 AscendC debug。
#
# 输入：一组"主 agent 已产出的算子目录"（由 ascend-kernel-developer-anti-cheat
#       生成；每个目录应含 model.py / model_new_ascendc.py / kernel/ / trace.md /
#       <op_name>.json(.bak)）。
# 行为：为每个 task_dir 独立调用 agents/ascendc-debug-agent-discovery.md（通过
#       codex exec），按 ASCENDC_DEBUG_MAX_ATTEMPTS 上限迭代修复。
#
# 动态工作队列：容器间并行（每容器绑定一张 NPU），容器内串行；
# 谁先完成当前任务，就从共享队列拉下一个，不做预分配。
#
# 典型用法:
#   # 方式 A：显式传入逗号分隔的 task_dir 列表
#   bash utils/run_ascendc_debug_batch.sh \
#        --task-dirs /home/c00959374/AscendOpGenAgent/outputs/run_20260422_1900/31_ELU,/home/c00959374/AscendOpGenAgent/outputs/run_20260422_1900/32_GELU \
#        --containers cjm_cann1,cjm_cann2 --npus 1,6 \
#        --output /home/c00959374/AscendOpGenAgent/outputs/debug_$(date +%Y%m%d_%H%M)
#
#   # 方式 B：从文件读，每行一个 task_dir
#   bash utils/run_ascendc_debug_batch.sh \
#        --task-dirs-file debug_targets.txt \
#        --containers cjm_cann1,cjm_cann2,cjm_cann3 --npus 1,6,7 \
#        --output /home/c00959374/AscendOpGenAgent/outputs/debug_run_01 \
#        --max-attempts 7

set -euo pipefail

# ── 默认值 ──
TASK_DIRS=""
TASK_DIRS_FILE=""
CONTAINERS=""
NPUS=""
OUTPUT_DIR=""
MODEL=""
EFFORT="high"               # thinking effort: minimal / low / medium / high
TIMEOUT_SEC="5400"          # 单任务超时（秒），默认 1.5 小时（对齐 subagent timeout）
MAX_ATTEMPTS="5"            # ASCENDC_DEBUG_MAX_ATTEMPTS 默认值（可被 --max-attempts 覆盖）
WORKDIR_IN_CONTAINER="/home/c00959374/AscendOpGenAgent"
TILELANG_ENV_SH="/home/c00959374/tilelang/tilelang-ascend/set_env.sh"
AGENT_SPEC_PATH="/home/c00959374/AscendOpGenAgent/agents/ascendc-debug-agent-discovery.md"
PROMPT_TEMPLATE=""
ANTICHEAT_SCRIPT="skills/ascendc/ascendc-debug/scripts/anticheat.py"

# ── 参数解析 ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --task-dirs)       TASK_DIRS="$2"; shift 2 ;;
        --task-dirs-file)  TASK_DIRS_FILE="$2"; shift 2 ;;
        --containers)      CONTAINERS="$2"; shift 2 ;;
        --npus)            NPUS="$2"; shift 2 ;;
        --output)          OUTPUT_DIR="$2"; shift 2 ;;
        --model)           MODEL="$2"; shift 2 ;;
        --effort)          EFFORT="$2"; shift 2 ;;
        --timeout)         TIMEOUT_SEC="$2"; shift 2 ;;
        --max-attempts)    MAX_ATTEMPTS="$2"; shift 2 ;;
        --workdir)         WORKDIR_IN_CONTAINER="$2"; shift 2 ;;
        --tilelang-env)    TILELANG_ENV_SH="$2"; shift 2 ;;
        --agent-spec)      AGENT_SPEC_PATH="$2"; shift 2 ;;
        --prompt)          PROMPT_TEMPLATE="$2"; shift 2 ;;
        -h|--help)
            sed -n '1,30p' "$0"
            exit 0
            ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# ── 校验 ──
[[ -z "$TASK_DIRS" && -z "$TASK_DIRS_FILE" ]] && {
    echo "错误: 必须指定 --task-dirs 或 --task-dirs-file"; exit 1;
}
[[ -n "$TASK_DIRS" && -n "$TASK_DIRS_FILE" ]] && {
    echo "错误: --task-dirs 与 --task-dirs-file 互斥"; exit 1;
}
[[ -z "$CONTAINERS" ]] && { echo "错误: 必须 --containers (逗号分隔)"; exit 1; }
[[ -z "$NPUS" ]]       && { echo "错误: 必须 --npus (逗号分隔，与 containers 一一对应)"; exit 1; }
[[ -z "$OUTPUT_DIR" ]] && { echo "错误: 必须 --output"; exit 1; }
[[ "$MAX_ATTEMPTS" =~ ^[0-9]+$ ]] || { echo "错误: --max-attempts 必须是正整数"; exit 1; }

IFS=',' read -ra CONTAINER_ARR <<< "$CONTAINERS"
IFS=',' read -ra NPU_ARR <<< "$NPUS"
(( ${#CONTAINER_ARR[@]} == ${#NPU_ARR[@]} )) \
    || { echo "错误: containers 与 npus 数量不一致"; exit 1; }

# ── 构造 task_dir 列表 ──
TASK_LIST=()
if [[ -n "$TASK_DIRS_FILE" ]]; then
    [[ -f "$TASK_DIRS_FILE" ]] || { echo "错误: 不存在文件 $TASK_DIRS_FILE"; exit 1; }
    while IFS= read -r line; do
        line="${line%%#*}"                    # 去注释
        line="$(echo "$line" | xargs)"        # 去首尾空白
        [[ -z "$line" ]] && continue
        TASK_LIST+=("$line")
    done < "$TASK_DIRS_FILE"
else
    IFS=',' read -ra TASK_LIST <<< "$TASK_DIRS"
fi
(( ${#TASK_LIST[@]} > 0 )) || { echo "错误: 无可用 task_dir"; exit 1; }

# ── 默认 prompt 模板（如用户未通过 --prompt 覆盖） ──
if [[ -z "$PROMPT_TEMPLATE" ]]; then
    PROMPT_TEMPLATE="读取文件 ${AGENT_SPEC_PATH}。把 YAML --- frontmatter 之后的 System Prompt 整段作为你的 developer instructions，严格按其执行（含 Initialization Protocol 三步、Step 0.3 分支锁定、Gate 协议与反作弊约束）。

**独立调用预授权（非交互，一次跑完）**：
本脚本批量调度即视为用户对本 agent 的显式调用授权；遇到分支 / 决策按 spec 默认路径处理，不得停机等待确认。必填产物缺失或不可恢复错误时直接终止并写 {task_dir}/debug_trace.md + {task_dir}/debug_status.json（session_outcome=crashed + crash_reason=<原因>）。

参数:
  task_dir: __TASK_DIR__
  npu: __NPU__

attempt 上限由环境变量 ASCENDC_DEBUG_MAX_ATTEMPTS 决定（本次 = __MAX_ATTEMPTS__）。"
fi

mkdir -p "$OUTPUT_DIR"
QUEUE="$OUTPUT_DIR/.queue"
LOCK="$OUTPUT_DIR/.lock"
REPORT="$OUTPUT_DIR/batch_report.md"

# ── 初始化队列（按原始顺序写入） ──
: > "$QUEUE"
for td in "${TASK_LIST[@]}"; do
    echo "$td" >> "$QUEUE"
done

: > "$LOCK"

# ── 初始化报告 ──
{
    echo "# Codex AscendC Debug 批量执行报告"
    echo
    echo "- containers: $CONTAINERS"
    echo "- npus: $NPUS"
    echo "- task_dirs: ${#TASK_LIST[@]} 个"
    echo "- max_attempts: $MAX_ATTEMPTS"
    echo "- agent spec: $AGENT_SPEC_PATH"
    echo "- model: ${MODEL:-<config default>}"
    echo "- reasoning_effort: $EFFORT"
    echo "- tilelang env: $TILELANG_ENV_SH"
    echo "- timeout: ${TIMEOUT_SEC}s/task"
    echo "- start: $(date '+%F %T')"
    echo
    echo "| # | task_dir | session_outcome | 耗时(s) | 容器@NPU |"
    echo "|---|----------|-----------------|---------|----------|"
} > "$REPORT"

TOTAL=${#TASK_LIST[@]}
echo "================================================================"
echo "总 debug 任务数: $TOTAL    workers: ${#CONTAINER_ARR[@]}    timeout: ${TIMEOUT_SEC}s    max_attempts: $MAX_ATTEMPTS"
for i in "${!CONTAINER_ARR[@]}"; do
    echo "  worker[$i]: ${CONTAINER_ARR[$i]} → npu=${NPU_ARR[$i]}"
done
echo "================================================================"

# ── worker：从队列拉 task_dir，跨 docker 执行 codex debug ──
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
        local op_name; op_name=$(basename "$task_dir")

        local prompt="${PROMPT_TEMPLATE//__TASK_DIR__/$task_dir}"
        prompt="${prompt//__NPU__/$npu}"
        prompt="${prompt//__MAX_ATTEMPTS__/$MAX_ATTEMPTS}"

        local start end elapsed status
        start=$(date +%s)

        set +e
        timeout --signal=TERM --kill-after=30 "$TIMEOUT_SEC" \
            docker exec \
                -e "ASCEND_RT_VISIBLE_DEVICES=$npu" \
                -e "ASCENDC_DEBUG_MAX_ATTEMPTS=$MAX_ATTEMPTS" \
                -e "CODEX_PROMPT=$prompt" \
                "$container" bash -lc "
                    set -e
                    [ -f '$TILELANG_ENV_SH' ] && source '$TILELANG_ENV_SH'
                    cd '$WORKDIR_IN_CONTAINER'
                    codex exec \
                        --dangerously-bypass-approvals-and-sandbox \
                        --skip-git-repo-check \
                        -c model_reasoning_effort='$EFFORT' \
                        ${MODEL:+-m '$MODEL'} \
                        --output-last-message '$task_dir/_codex_last.txt' \
                        -- \"\$CODEX_PROMPT\"
                " >> "$wlog" 2>&1
        status=$?
        set -e

        end=$(date +%s); elapsed=$((end - start))

        # 反作弊后置检测：AST + C++ 源码扫描（debug agent 禁改 wrapper，命中即作弊）
        local cheat_json cheat_verdict cheat_reasons cheat_mark
        cheat_json=$(docker exec "$container" bash -lc "
            cd '$WORKDIR_IN_CONTAINER'
            python3 '$ANTICHEAT_SCRIPT' verify '$task_dir' --json 2>/dev/null
        " 2>/dev/null || true)
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
        [[ -n "$cheat_json" ]] && echo "$cheat_json" > "$task_dir/_anticheat.json"

        cheat_mark=""
        if [[ "$cheat_verdict" == "CHEAT" ]]; then
            cheat_mark=" / 🚨 CHEAT"
            echo "[${container}@npu${npu}] 🚨 ${op_name} CHEAT: $cheat_reasons"
        fi

        # 从 debug_status.json 读 session_outcome（debug agent 产物）
        local session_outcome="unknown"
        if [[ -f "$task_dir/debug_status.json" ]]; then
            session_outcome=$(python3 -c "
import json, sys
try:
    d = json.load(open('$task_dir/debug_status.json'))
    print(d.get('session_outcome', 'unknown'))
except Exception:
    print('unknown')
" 2>/dev/null || echo "unknown")
        fi

        local icon
        if [[ $status -eq 0 ]]; then
            case "$session_outcome" in
                success)                       icon="✅ $session_outcome${cheat_mark}" ;;
                progressed_to_new_failure_type) icon="↗ $session_outcome${cheat_mark}" ;;
                skipped_*)                     icon="⊘ $session_outcome${cheat_mark}" ;;
                failed|stopped_*|crashed|timeout) icon="❌ $session_outcome${cheat_mark}" ;;
                *)                             icon="⚠ $session_outcome${cheat_mark}" ;;
            esac
            echo "[${container}@npu${npu}] ✅ ${op_name} session_outcome=${session_outcome} (${elapsed}s)"
        elif [[ $status -eq 124 ]]; then
            icon="⏱ 超时(codex)${cheat_mark}"
            echo "[${container}@npu${npu}] ⏱ ${op_name} CODEX_TIMEOUT (${elapsed}s)"
        else
            icon="❌ codex_rc=$status${cheat_mark}"
            echo "[${container}@npu${npu}] ❌ ${op_name} codex_rc=$status (${elapsed}s)"
        fi

        local idx row
        exec 9>"$LOCK"; flock -x 9
        # 序号 = 已写入 report 的 debug 任务数 + 1（不含表头行）
        idx=$(grep -c '^| [0-9]' "$REPORT" 2>/dev/null || echo 0)
        idx=$((idx + 1))
        row="| $idx | $op_name | $icon | $elapsed | ${container}@npu${npu} |"
        echo "$row" >> "$REPORT"
        # 增量生成汇总报告（若工具存在）
        GEN_REPORT="$(dirname "$0")/generate_report_dynamic.py"
        if [[ -f "$GEN_REPORT" ]]; then
            python3 "$GEN_REPORT" -i "$OUTPUT_DIR" -o "$OUTPUT_DIR/final_batch_report.md" >>"$OUTPUT_DIR/report_gen.log" 2>&1 || true
        fi
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
SUCCESS=$(grep -c "✅ success" "$REPORT" || echo 0)
PROGRESSED=$(grep -c "↗ progressed" "$REPORT" || echo 0)
SKIPPED=$(grep -c "⊘ skipped" "$REPORT" || echo 0)
TIMEOUT_CNT=$(grep -c "⏱ 超时" "$REPORT" || echo 0)
FAIL=$(grep -c "❌ " "$REPORT" || echo 0)
CHEAT=$(grep -c "🚨 CHEAT" "$REPORT" || echo 0)

{
    echo
    echo "## 汇总"
    echo
    echo "- 总数: $TOTAL"
    echo "- success: $SUCCESS"
    echo "- progressed_to_new_failure_type: $PROGRESSED"
    echo "- skipped_*: $SKIPPED"
    echo "- codex timeout: $TIMEOUT_CNT"
    echo "- failed / crashed / stopped_* / codex_rc!=0: $FAIL"
    echo "- 作弊 (🚨 CHEAT, 与 outcome 正交): $CHEAT"
    echo "- 结束: $(date '+%F %T')"
} >> "$REPORT"

echo "================================================================"
echo "完成: ✅$SUCCESS  ↗$PROGRESSED  ⊘$SKIPPED  ⏱$TIMEOUT_CNT  ❌$FAIL  🚨$CHEAT  /  共 $TOTAL"
echo "报告: $REPORT"
echo "每 worker 日志: $OUTPUT_DIR/worker_<container>.log"
echo "================================================================"

# ── 若存在生成详细报告的工具则调用 ──
GEN="$(dirname "$0")/generate_report_dynamic.py"
if [[ -f "$GEN" ]]; then
    echo "正在调用 generate_report_dynamic.py..."
    python3 "$GEN" -i "$OUTPUT_DIR" -o "$OUTPUT_DIR/final_batch_report.md" || true
fi
