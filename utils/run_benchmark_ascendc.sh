#!/bin/bash
# run_benchmark _ascendc.sh — 批量调度 ascendc-coder 串行生成算子
#
# 每个算子在独立的 claude session 中执行，context 互不污染。
# claude -p 是同步阻塞的，前一个完成后才启动下一个，不会 NPU 冲突。
#
# 用法:
#   bash .claude/run_benchmark _ascendc.sh --benchmark-dir /path/to/KernelBench --level 1 --range 41-53 --npu 6 --output /path/to/output
#   bash .claude/run_benchmark _ascendc.sh --benchmark-dir /path/to/KernelBench --level 2 --range 1-10 --npu 0 --output /path/to/output
#   bash .claude/run_benchmark _ascendc.sh --benchmark-dir /path/to/KernelBench --level 1 --ids "3,7,15,22" --npu 6 --output /path/to/output

# ── 环境变量 ──
export ANTHROPIC_AUTH_TOKEN=sk-  # 替换为您的实际令牌
export ANTHROPIC_BASE_URL=https://yunwu.ai
export API_TIMEOUT_MS=300000  # 设置为 300 秒超时

set -euo pipefail

# ── 默认值 ──
BENCHMARK_DIR=""
LEVEL=""
RANGE=""
IDS=""
NPU_ID=0
OUTPUT_DIR=""

# ── 参数解析 ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --benchmark-dir) BENCHMARK_DIR="$2"; shift 2 ;;
        --level)         LEVEL="$2"; shift 2 ;;
        --range)         RANGE="$2"; shift 2 ;;
        --ids)           IDS="$2"; shift 2 ;;
        --npu)           NPU_ID="$2"; shift 2 ;;
        --output)        OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "用法: bash .claude/run_benchmark _ascendc.sh --benchmark-dir <path> --level <N> --range <start-end> --npu <id> --output <path>"
            echo ""
            echo "参数:"
            echo "  --benchmark-dir  KernelBench 根目录路径 (必填)"
            echo "  --level          Level 编号，如 1, 2, 3 (必填)"
            echo "  --range          算子范围，如 41-53 (与 --ids 二选一)"
            echo "  --ids            指定算子编号列表，逗号分隔，如 3,7,15 (与 --range 二选一)"
            echo "  --npu            NPU 设备 ID (默认 0)"
            echo "  --output         输出目录 (必填)"
            exit 0
            ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# ── 参数校验 ──
if [[ -z "$BENCHMARK_DIR" ]]; then
    echo "错误: 必须指定 --benchmark-dir"
    exit 1
fi

if [[ -z "$LEVEL" ]]; then
    echo "错误: 必须指定 --level"
    exit 1
fi

if [[ -z "$RANGE" && -z "$IDS" ]]; then
    echo "错误: 必须指定 --range 或 --ids"
    exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    echo "错误: 必须指定 --output"
    exit 1
fi

LEVEL_DIR="${BENCHMARK_DIR}/level${LEVEL}"
if [[ ! -d "$LEVEL_DIR" ]]; then
    echo "错误: 目录不存在: ${LEVEL_DIR}"
    exit 1
fi

# ── 构建算子 ID 列表 ──
OP_IDS=()
if [[ -n "$RANGE" ]]; then
    START=$(echo "$RANGE" | cut -d'-' -f1)
    END=$(echo "$RANGE" | cut -d'-' -f2)
    for i in $(seq "$START" "$END"); do
        OP_IDS+=("$i")
    done
elif [[ -n "$IDS" ]]; then
    IFS=',' read -ra OP_IDS <<< "$IDS"
fi

# ── 扫描算子文件 ──
declare -A OP_FILES
for id in "${OP_IDS[@]}"; do
    # 匹配 {id}_{name}.py 格式
    matched=$(find "$LEVEL_DIR" -maxdepth 1 -name "${id}_*.py" -type f 2>/dev/null | head -1)
    if [[ -n "$matched" ]]; then
        OP_FILES[$id]="$matched"
    else
        echo "警告: 未找到算子 ${id} 的文件，跳过"
    fi
done

if [[ ${#OP_FILES[@]} -eq 0 ]]; then
    echo "错误: 未找到任何算子文件"
    exit 1
fi

# ── 创建输出目录 ──
mkdir -p "$OUTPUT_DIR"

# ── 结果记录 ──
REPORT_FILE="${OUTPUT_DIR}/batch_report.md"
echo "# 批量执行报告" > "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "- benchmark: ${BENCHMARK_DIR}" >> "$REPORT_FILE"
echo "- level: ${LEVEL}" >> "$REPORT_FILE"
echo "- npu: ${NPU_ID}" >> "$REPORT_FILE"
echo "- 开始时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "| 算子ID | 文件 | 状态 | 耗时(s) |" >> "$REPORT_FILE"
echo "|--------|------|------|---------|" >> "$REPORT_FILE"

TOTAL=${#OP_FILES[@]}
CURRENT=0
SUCCESS=0
FAIL=0

# ── 串行执行 ──
for id in $(echo "${!OP_FILES[@]}" | tr ' ' '\n' | sort -n); do
    file="${OP_FILES[$id]}"
    filename=$(basename "$file")
    
    # === 新增/修改的逻辑开始 ===
    # 提取不带后缀的文件名，例如 "31_ELU.py" 变成 "31_ELU"
    op_name="${filename%.*}"
    
    # 构建当前算子的专属输出子目录
    TARGET_OP_DIR="${OUTPUT_DIR}/${op_name}"
    
    # 创建该子目录
    mkdir -p "$TARGET_OP_DIR"
    # === 新增/修改的逻辑结束 ===

    CURRENT=$((CURRENT + 1))

    echo ""
    echo "================================================================"
    echo "[${CURRENT}/${TOTAL}] 算子 ${id}: ${filename} (输出至: ${op_name}/)"
    echo "================================================================"

    START_TIME=$(date +%s)

    # 修改 PROMPT：将输出路径从 ${OUTPUT_DIR}/ 改为 ${TARGET_OP_DIR}/
    PROMPT="使用当前agent生成ascendC算子，npu=${NPU_ID}，算子描述文件为 ${file}，输出到 ${TARGET_OP_DIR}/"

    if claude -p "$PROMPT" \
        --allowedTools 'Bash(*)' 'Read(*)' 'Write(*)' 'Edit(*)' 'Glob(*)' 'Grep(*)' 'Skill(*)'; then
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        echo "| ${id} | ${filename} | ✅ 成功 | ${ELAPSED} |" >> "$REPORT_FILE"
        SUCCESS=$((SUCCESS + 1))
        echo "[${CURRENT}/${TOTAL}] ✅ 算子 ${id} 完成 (${ELAPSED}s)"
    else
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        echo "| ${id} | ${filename} | ❌ 失败 | ${ELAPSED} |" >> "$REPORT_FILE"
        FAIL=$((FAIL + 1))
        echo "[${CURRENT}/${TOTAL}] ❌ 算子 ${id} 失败 (${ELAPSED}s)"
    fi
done

# ── 写入汇总 ──
echo "" >> "$REPORT_FILE"
echo "## 汇总" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "- 总数: ${TOTAL}" >> "$REPORT_FILE"
echo "- 成功: ${SUCCESS}" >> "$REPORT_FILE"
echo "- 失败: ${FAIL}" >> "$REPORT_FILE"
echo "- 结束时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$REPORT_FILE"

echo ""
echo "================================================================"
echo "批量执行完成: 成功 ${SUCCESS}/${TOTAL}, 失败 ${FAIL}/${TOTAL}"
echo "报告: ${REPORT_FILE}"
echo "================================================================"
