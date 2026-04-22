#!/bin/bash
# 批量调度 codex CLI 跨多个 docker 容器 + 多 NPU 执行 AscendC 算子生成
#
# 动态工作队列：容器间并行（每容器绑定一张 NPU），容器内串行；
# 谁先完成当前任务，就从共享队列拉下一个，不做预分配。
#
# 典型用法:
#   bash utils/run_benchmark_ascendc_codex.sh \
#        --benchmark-dir /home/c00959374/AscendOpGenAgent/benchmarks/NPUKernelBench \
#        --level 1 --ids 1,2,3,4,5,6 \
#        --containers cjm_cann1,cjm_cann2,cjm_cann3 \
#        --npus 1,6,7 \
#        --output /home/c00959374/AscendOpGenAgent/outputs/run_$(date +%Y%m%d_%H%M)

set -euo pipefail

# ── 默认值 ──
BENCHMARK_DIR=""
LEVEL=""
RANGE=""
IDS=""
CONTAINERS=""
NPUS=""
OUTPUT_DIR=""
MODEL=""
EFFORT="high"               # thinking effort: minimal / low / medium / high
TIMEOUT_SEC="7200"          # 单任务超时（秒），默认 2 小时
WORKDIR_IN_CONTAINER="/home/c00959374/AscendOpGenAgent"
TILELANG_ENV_SH="/home/c00959374/tilelang/tilelang-ascend/set_env.sh"   # 设 TL_ROOT / PYTHONPATH / ACL_OP_INIT_MODE
PROMPT_TEMPLATE='严格按照 /home/c00959374/AscendOpGenAgent/agents/ascend-kernel-developer.md 中定义的 ascend-kernel-developer agent 规范和 Phase 0-7 流程执行。

任务参数:
- npu: __NPU__
- 算子描述文件 (op_file): __FILE__
- 输出目录 (output_dir): __TARGET__

执行要求:
1. 首先 Read 并完整理解 agent 规范文件: /home/c00959374/AscendOpGenAgent/agents/ascend-kernel-developer.md
2. 按该规范中定义的 Phase 0-7 逐阶段执行；在需要 skill 时 Read 对应目录 /home/c00959374/AscendOpGenAgent/skills/ascendc/<skill-name>/ 下的 SKILL.md 或 README 文件获取该 skill 的具体 step 与脚本
3. 硬约束：只允许修改或新增 __TARGET__ 目录下的文件，禁止改动其他目录内的任何文件
4. Phase 1 先将 __FILE__ 复制到 __TARGET__/ 作为工作基准
5. NPU 设备已通过环境变量 ASCEND_RT_VISIBLE_DEVICES=__NPU__ 暴露，生成的 Python / 验证脚本应直接使用该环境变量，不要自行覆盖
6. 产物至少包括: __TARGET__/model_new_tilelang.py 和 __TARGET__/model_new_ascendc.py，以及 agent 规范中列出的各 Phase 输出
7. 【反作弊硬约束】核心计算必须在 AscendC kernel 内完成，严禁以下任何"绕过 kernel"行为：
   (a) Python wrapper (model_new_ascendc.py) 的 forward() 中调用 torch.* / F.* 计算算子，或用 tensor 计算方法 (x.sum/x.matmul/x.cumsum 等)；
   (b) kernel/*.cpp 或 *.h 中调用 at::<算子>(...) 或 torch::<算子>(...) 这类 libtorch 计算 API；
   (c) kernel/ 中使用 tensor 计算方法 (如 x.cumsum() / x.histc() / .sum() / .matmul()) 绕过 AscendC 实现；
   (d) #include <ATen/ops/*.h> 引入 ATen 算子头文件；
   (e) 仅写一个"空壳 __global__ __aicore__ 函数"但 pybind 层直接返回 at::xxx/x.xxx() 结果；
   只允许在 pybind11.cpp 里用 at::empty / at::empty_like / at::zeros 这类 allocator 和 TensorOptions 构造 (at::device(...)/at::dtype(...))，
   且必须通过 <<<...>>> 或 aclrtLaunchKernel / *_do() stub launcher 真正触发 AscendC kernel 计算。
   生成结束后 bench 会自动调用 anticheat.py verify 做 AST + C++ 源码双重扫描，命中即在报告标 🚨 CHEAT (不重跑, 但会记录违规点)。
8. 全程不要向用户询问或等待交互；遇到分支/决策均按 agent 规范定义的默认路径处理
9. 结束时在 __TARGET__/ 输出一份简短的 trace/final 报告（Phase 7），说明各阶段成功/失败与最终产物清单'

ANTICHEAT_SCRIPT="skills/ascendc/ascendc-debug/scripts/anticheat.py"

# ── 参数解析 ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --benchmark-dir) BENCHMARK_DIR="$2"; shift 2 ;;
        --level)         LEVEL="$2"; shift 2 ;;
        --range)         RANGE="$2"; shift 2 ;;
        --ids)           IDS="$2"; shift 2 ;;
        --containers)    CONTAINERS="$2"; shift 2 ;;
        --npus)          NPUS="$2"; shift 2 ;;
        --output)        OUTPUT_DIR="$2"; shift 2 ;;
        --model)         MODEL="$2"; shift 2 ;;
        --effort)        EFFORT="$2"; shift 2 ;;
        --timeout)       TIMEOUT_SEC="$2"; shift 2 ;;
        --workdir)       WORKDIR_IN_CONTAINER="$2"; shift 2 ;;
        --tilelang-env)  TILELANG_ENV_SH="$2"; shift 2 ;;
        --prompt)        PROMPT_TEMPLATE="$2"; shift 2 ;;
        -h|--help)
            sed -n '1,15p' "$0"
            exit 0
            ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# ── 校验 ──
[[ -z "$BENCHMARK_DIR" ]] && { echo "错误: 必须 --benchmark-dir"; exit 1; }
[[ -z "$LEVEL" ]]         && { echo "错误: 必须 --level"; exit 1; }
[[ -z "$RANGE" && -z "$IDS" ]] && { echo "错误: 必须 --range 或 --ids"; exit 1; }
[[ -z "$CONTAINERS" ]]    && { echo "错误: 必须 --containers (逗号分隔)"; exit 1; }
[[ -z "$NPUS" ]]          && { echo "错误: 必须 --npus (逗号分隔，与 containers 一一对应)"; exit 1; }
[[ -z "$OUTPUT_DIR" ]]    && { echo "错误: 必须 --output"; exit 1; }

LEVEL_DIR="${BENCHMARK_DIR}/level${LEVEL}"
[[ -d "$LEVEL_DIR" ]] || { echo "错误: 不存在 $LEVEL_DIR"; exit 1; }

IFS=',' read -ra CONTAINER_ARR <<< "$CONTAINERS"
IFS=',' read -ra NPU_ARR <<< "$NPUS"
(( ${#CONTAINER_ARR[@]} == ${#NPU_ARR[@]} )) \
    || { echo "错误: containers 与 npus 数量不一致"; exit 1; }

# ── 构造算子 ID 列表 ──
OP_IDS=()
if [[ -n "$RANGE" ]]; then
    S=$(echo "$RANGE" | cut -d'-' -f1); E=$(echo "$RANGE" | cut -d'-' -f2)
    for i in $(seq "$S" "$E"); do OP_IDS+=("$i"); done
else
    IFS=',' read -ra OP_IDS <<< "$IDS"
fi

# ── 扫描文件 ──
declare -A OP_FILES
for id in "${OP_IDS[@]}"; do
    matched=$(find "$LEVEL_DIR" -maxdepth 1 -name "${id}_*.py" -type f 2>/dev/null | head -1)
    if [[ -n "$matched" ]]; then
        OP_FILES[$id]="$matched"
    else
        echo "警告: 未找到算子 ${id}，跳过"
    fi
done
(( ${#OP_FILES[@]} > 0 )) || { echo "错误: 无可用算子"; exit 1; }

mkdir -p "$OUTPUT_DIR"
QUEUE="$OUTPUT_DIR/.queue"
LOCK="$OUTPUT_DIR/.lock"
REPORT="$OUTPUT_DIR/batch_report.md"

# ── 初始化队列（按原始顺序写入） ──
: > "$QUEUE"
for id in "${OP_IDS[@]}"; do
    [[ -v OP_FILES[$id] ]] && echo "$id" >> "$QUEUE"
done

: > "$LOCK"

# ── 初始化报告 ──
{
    echo "# Codex 批量执行报告"
    echo
    echo "- benchmark: $BENCHMARK_DIR"
    echo "- level: $LEVEL"
    echo "- containers: $CONTAINERS"
    echo "- npus: $NPUS"
    echo "- tasks: ${OP_IDS[*]}"
    echo "- model: ${MODEL:-<config default>}"
    echo "- reasoning_effort: $EFFORT"
    echo "- tilelang env: $TILELANG_ENV_SH"
    echo "- timeout: ${TIMEOUT_SEC}s/task"
    echo "- start: $(date '+%F %T')"
    echo
    echo "| id | file | 状态 | 耗时(s) | 容器@NPU |"
    echo "|----|------|------|---------|----------|"
} > "$REPORT"

TOTAL=${#OP_FILES[@]}
echo "================================================================"
echo "总任务数: $TOTAL    workers: ${#CONTAINER_ARR[@]}    timeout: ${TIMEOUT_SEC}s"
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
        local id=""
        # 原子出队
        exec 9>"$LOCK"
        flock -x 9
        if [[ -s "$QUEUE" ]]; then
            id=$(head -n1 "$QUEUE")
            sed -i '1d' "$QUEUE"
        fi
        flock -u 9
        exec 9>&-

        [[ -z "$id" ]] && break
        local file="${OP_FILES[$id]}"
        local filename; filename=$(basename "$file")
        local op_name="${filename%.*}"
        local target_dir="$OUTPUT_DIR/$op_name"
        mkdir -p "$target_dir"

        local prompt="${PROMPT_TEMPLATE//__NPU__/$npu}"
        prompt="${prompt//__FILE__/$file}"
        prompt="${prompt//__TARGET__/$target_dir}"

        local start end elapsed status
        start=$(date +%s)

        # 注意：通过 bash -lc 进入 login shell 以加载 ~/.bashrc (OPENAI_API_KEY 等)
        # ASCEND_RT_VISIBLE_DEVICES 硬绑到当前 worker 的 NPU
        # codex exec 使用 bypass flag 跳过一切审批
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
                        -c model_reasoning_effort='$EFFORT' \
                        ${MODEL:+-m '$MODEL'} \
                        --output-last-message '$target_dir/_codex_last.txt' \
                        -- \"\$CODEX_PROMPT\"
                " >> "$wlog" 2>&1
        status=$?
        set -e

        end=$(date +%s); elapsed=$((end - start))

        # 反作弊后置检测：AST + C++ 源码扫描（生成场景无 baseline, 纯检测）
        local cheat_json cheat_verdict cheat_reasons cheat_mark
        cheat_json=$(docker exec "$container" bash -lc "
            cd '$WORKDIR_IN_CONTAINER'
            python3 '$ANTICHEAT_SCRIPT' verify '$target_dir' --json 2>/dev/null
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
        # 保存详细结果供事后审查
        [[ -n "$cheat_json" ]] && echo "$cheat_json" > "$target_dir/_anticheat.json"

        cheat_mark=""
        if [[ "$cheat_verdict" == "CHEAT" ]]; then
            cheat_mark=" / 🚨 CHEAT"
            echo "[${container}@npu${npu}] 🚨 id=$id ${filename} CHEAT: $cheat_reasons"
        fi

        local row icon
        if [[ $status -eq 0 ]]; then
            icon="✅ 成功${cheat_mark}"
            echo "[${container}@npu${npu}] ✅ id=$id ${filename} (${elapsed}s)"
        elif [[ $status -eq 124 ]]; then
            icon="⏱ 超时${cheat_mark}"
            echo "[${container}@npu${npu}] ⏱ id=$id ${filename} TIMEOUT (${elapsed}s)"
        else
            icon="❌ 失败(rc=$status)${cheat_mark}"
            echo "[${container}@npu${npu}] ❌ id=$id ${filename} rc=$status (${elapsed}s)"
        fi
        row="| $id | $filename | $icon | $elapsed | ${container}@npu${npu} |"

        exec 9>"$LOCK"; flock -x 9
        echo "$row" >> "$REPORT"
        # 任务完成后立刻增量生成汇总报告（依赖 generate_report_dynamic.py 扫描 trace.md）
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
    echo "- 作弊 (🚨 CHEAT, 与成功/失败正交，不重跑): $CHEAT"
    echo "- 结束: $(date '+%F %T')"
} >> "$REPORT"

echo "================================================================"
echo "完成: ✅$SUCCESS  ⏱$TIMEOUT_CNT  ❌$FAIL  🚨$CHEAT  /  共 $TOTAL"
echo "报告: $REPORT"
echo "每 worker 日志: $OUTPUT_DIR/worker_<container>.log"
echo "================================================================"

# ── 若存在生成详细报告的工具则调用 ──
GEN="$(dirname "$0")/generate_report_dynamic.py"
if [[ -f "$GEN" ]]; then
    echo "正在调用 generate_report_dynamic.py..."
    python3 "$GEN" -i "$OUTPUT_DIR" -o "$OUTPUT_DIR/final_batch_report.md" || true
fi
