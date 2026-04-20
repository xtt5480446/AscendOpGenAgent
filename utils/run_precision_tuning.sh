#!/bin/bash
# 批量调度 codex CLI 跨多个 docker 容器 + 多 NPU 执行 AscendC 算子精度调优
#
# 动态工作队列：容器间并行（每容器绑定一张 NPU），容器内串行；
# 谁先完成当前任务，就从共享队列拉下一个，不做预分配。
#
# 典型用法:
#   bash utils/run_precision_tuning.sh \
#        --task-list tasks/precision_tasks.txt \
#        --agent agents/precision-tuning-discovery.md \
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
AGENT_FILE="agents/precision-tuning-discovery.md"
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
3. 硬约束：只允许修改 __TASK_DIR__/kernel/ 和 __TASK_DIR__/model_new_ascendc.py，禁止改动其他目录内的任何文件
4. 精度调优脚本路径: __WORKDIR__/skills/ascendc/precision-tuning/scripts/
5. NPU 设备已通过环境变量 ASCEND_RT_VISIBLE_DEVICES=__NPU__ 暴露
6. 若 precision_forensics.py / precision_gate.py 需要 --task-dir 参数，使用 __TASK_DIR__
7. 全程不要向用户询问或等待交互；遇到分支/决策均按 agent 规范定义的默认路径处理
8. 结束时输出精度调优结果摘要（是否通过、最终 max_abs_diff、修改了哪些文件）'

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
        local task_log="$OUTPUT_DIR/${task_name}.log"

        # 替换 prompt 占位符
        local prompt="${PROMPT_TEMPLATE//__AGENT_FILE__/$WORKDIR_IN_CONTAINER/$AGENT_FILE}"
        prompt="${prompt//__TASK_NAME__/$task_name}"
        prompt="${prompt//__TASK_DIR__/$task_dir}"
        prompt="${prompt//__NPU__/$npu}"
        prompt="${prompt//__WORKDIR__/$WORKDIR_IN_CONTAINER}"

        echo "[${container}@npu${npu}] 开始: $task_name" | tee -a "$wlog"

        local start end elapsed status
        start=$(date +%s)

        set +e
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
                " >> "$wlog" 2>&1
        status=$?
        set -e

        end=$(date +%s); elapsed=$((end - start))

        local icon
        if [[ $status -eq 0 ]]; then
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

{
    echo
    echo "## 汇总"
    echo
    echo "- 总数: $TOTAL"
    echo "- 成功: $SUCCESS"
    echo "- 超时: $TIMEOUT_CNT"
    echo "- 失败: $FAIL"
    echo "- 结束: $(date '+%F %T')"
} >> "$REPORT"

echo "================================================================"
echo "完成: ✅$SUCCESS  ⏱$TIMEOUT_CNT  ❌$FAIL  /  共 $TOTAL"
echo "报告: $REPORT"
echo "每 worker 日志: $OUTPUT_DIR/worker_<container>.log"
echo "每 task 日志:   $OUTPUT_DIR/<task_name>.log"
echo "================================================================"
