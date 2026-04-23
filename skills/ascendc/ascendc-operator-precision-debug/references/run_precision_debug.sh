#!/usr/bin/env bash
#
# AscendC 精度调试脚本执行入口。
#
# 以 skills/ascendc/ascendc-translator/references/evaluate_ascendc.sh 为蓝本，
# 保留打包 / 上传 / Docker / set_env.sh / build_ascendc.py 等环节；仅将最后一步
# 由 `python utils/verification_ascendc.py "${TASK}"` 改为
# `python "${TASK}/debug_${OP_NAME}_precision.py"`。
#
# 两个脚本刻意不抽公共执行器，手工同步即可。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

find_workdir() {
  if [[ -n "${WORKDIR:-}" ]]; then
    echo "${WORKDIR}"
    return 0
  fi

  local candidate="${SCRIPT_DIR}"
  while [[ "${candidate}" != "/" ]]; do
    if [[ -f "${candidate}/utils/verification_ascendc.py" ]]; then
      echo "${candidate}"
      return 0
    fi
    candidate="$(cd "${candidate}/.." && pwd)"
  done

  return 1
}

WORKDIR="$(find_workdir)" || {
  echo "Unable to locate repository root containing utils/verification_ascendc.py" >&2
  exit 1
}

SSH_TARGET="${SSH_TARGET:-ascend-box}"
REMOTE_PORT="${REMOTE_PORT:-}"
SSH_KEY="${SSH_KEY:-}"

REMOTE_BASE_DIR="${REMOTE_BASE_DIR:-/root/tilelang_eval}"
CONTAINER_NAME="${CONTAINER_NAME:-zyy_cann}"
CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-/home/z00893531/tilelang-ascend}"
REMOTE_EVAL_WORKDIR="${REMOTE_EVAL_WORKDIR:-workdir_remote_eval}"
ASCENDC_SOC_VERSION="${ASCENDC_SOC_VERSION:-Ascend910B3}"
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-3}"
ASCENDC_CLEAN_BUILD="${ASCENDC_CLEAN_BUILD:-1}"

usage() {
  cat <<'EOF'
Usage: bash <path-to-precision-debug>/references/run_precision_debug.sh <task> <op_name>

Arguments:
  task     Task directory (relative to repo root or absolute) containing
           model.py / model_new_ascendc.py / kernel/ /
           debug_<op_name>_precision.py
  op_name  Operator name (used to locate debug_<op_name>_precision.py)

Environment overrides: identical to evaluate_ascendc.sh
  SSH_TARGET / REMOTE_PORT / SSH_KEY
  REMOTE_BASE_DIR / CONTAINER_NAME / CONTAINER_WORKDIR / REMOTE_EVAL_WORKDIR
  ASCENDC_SOC_VERSION / ASCEND_RT_VISIBLE_DEVICES / ASCENDC_CLEAN_BUILD

Example:
  bash skills/ascendc/ascendc-operator-precision-debug/references/run_precision_debug.sh 31_ELU elu
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

TASK="$1"
OP_NAME="$2"
DEBUG_SCRIPT="${TASK}/debug_${OP_NAME}_precision.py"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE_NAME="workdir_${TIMESTAMP}.tar.gz"
LOCAL_ARCHIVE="/tmp/${ARCHIVE_NAME}"
REMOTE_ARCHIVE="/tmp/${ARCHIVE_NAME}"
REMOTE_SESSION_DIR="${REMOTE_BASE_DIR}/${TIMESTAMP}"

PYTHONPATH_PREFIX="${WORKDIR}"
if [[ -d "${WORKDIR}/archive_tasks" ]]; then
  PYTHONPATH_PREFIX="${WORKDIR}/archive_tasks:${PYTHONPATH_PREFIX}"
fi

if [[ ! -f "${WORKDIR}/utils/verification_ascendc.py" ]]; then
  echo "Missing verification script: ${WORKDIR}/utils/verification_ascendc.py" >&2
  exit 1
fi

if [[ ! -f "${WORKDIR}/utils/build_ascendc.py" ]]; then
  echo "Missing build script: ${WORKDIR}/utils/build_ascendc.py" >&2
  exit 1
fi

if [[ ! -d "${WORKDIR}/${TASK}" ]]; then
  echo "Task directory not found: ${WORKDIR}/${TASK}" >&2
  exit 1
fi

if [[ ! -d "${WORKDIR}/${TASK}/kernel" ]]; then
  echo "Task kernel directory not found: ${WORKDIR}/${TASK}/kernel" >&2
  exit 1
fi

if [[ ! -f "${WORKDIR}/${DEBUG_SCRIPT}" ]]; then
  echo "Missing debug script: ${WORKDIR}/${DEBUG_SCRIPT}" >&2
  echo "Expected to find ${TASK}/debug_${OP_NAME}_precision.py (derived from scripts/debug_precision_template.py)" >&2
  exit 1
fi

if python -c 'import torch; import torch_npu' >/dev/null 2>&1; then
  echo "Detected local Ascend environment, building kernel and running debug script"
  cd "${WORKDIR}"
  if [[ "${ASCENDC_CLEAN_BUILD}" == "1" ]]; then
    PYTHONPATH="${PYTHONPATH_PREFIX}${PYTHONPATH:+:${PYTHONPATH}}" \
      python utils/build_ascendc.py "${TASK}" -v "${ASCENDC_SOC_VERSION}" --clean
  else
    PYTHONPATH="${PYTHONPATH_PREFIX}${PYTHONPATH:+:${PYTHONPATH}}" \
      python utils/build_ascendc.py "${TASK}" -v "${ASCENDC_SOC_VERSION}"
  fi
  PYTHONPATH="${PYTHONPATH_PREFIX}${PYTHONPATH:+:${PYTHONPATH}}" \
    ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" \
    WORKDIR="${WORKDIR}" \
    python "${DEBUG_SCRIPT}"
  exit 0
fi

SSH_OPTS=()
SCP_OPTS=()
if [[ -n "${REMOTE_PORT}" ]]; then
  SSH_OPTS+=(-p "${REMOTE_PORT}")
  SCP_OPTS+=(-P "${REMOTE_PORT}")
fi

if [[ -n "${SSH_KEY}" ]]; then
  if [[ ! -f "${SSH_KEY}" ]]; then
    echo "SSH key not found: ${SSH_KEY}" >&2
    exit 1
  fi
  SSH_OPTS+=(-i "${SSH_KEY}")
  SCP_OPTS+=(-i "${SSH_KEY}")
fi

SSH_CMD=(ssh)
SCP_CMD=(scp)
if [[ ${#SSH_OPTS[@]} -gt 0 ]]; then
  SSH_CMD+=("${SSH_OPTS[@]}")
fi
if [[ ${#SCP_OPTS[@]} -gt 0 ]]; then
  SCP_CMD+=("${SCP_OPTS[@]}")
fi

cleanup() {
  rm -f "${LOCAL_ARCHIVE}"
}
trap cleanup EXIT

echo "[1/4] Packaging ${WORKDIR}"
tar \
  --exclude=".git" \
  --exclude="__pycache__" \
  --exclude=".DS_Store" \
  --exclude=".pytest_cache" \
  --exclude=".mypy_cache" \
  --exclude=".ruff_cache" \
  -C "${WORKDIR}" \
  -czf "${LOCAL_ARCHIVE}" \
  .

echo "[2/4] Uploading archive to ${SSH_TARGET}:${REMOTE_ARCHIVE}"
"${SCP_CMD[@]}" \
  "${LOCAL_ARCHIVE}" \
  "${SSH_TARGET}:${REMOTE_ARCHIVE}"

read -r -d '' REMOTE_SCRIPT <<EOF || true
set -euo pipefail
cleanup_remote() {
  rm -rf "${REMOTE_SESSION_DIR}"
}
trap cleanup_remote EXIT

mkdir -p "${REMOTE_SESSION_DIR}"
tar -xzf "${REMOTE_ARCHIVE}" -C "${REMOTE_SESSION_DIR}"
rm -f "${REMOTE_ARCHIVE}"

if [[ "${ASCENDC_CLEAN_BUILD}" == "1" ]]; then
  docker exec "${CONTAINER_NAME}" /bin/bash -lc '
set -euo pipefail
rm -rf "${CONTAINER_WORKDIR}/${REMOTE_EVAL_WORKDIR}"
mkdir -p "${CONTAINER_WORKDIR}"
'
else
  docker exec "${CONTAINER_NAME}" /bin/bash -lc 'mkdir -p "${CONTAINER_WORKDIR}/${REMOTE_EVAL_WORKDIR}"'
fi

docker cp "${REMOTE_SESSION_DIR}/." "${CONTAINER_NAME}:${CONTAINER_WORKDIR}/${REMOTE_EVAL_WORKDIR}"

docker exec "${CONTAINER_NAME}" /bin/bash -lc '
set -euo pipefail
cd "${CONTAINER_WORKDIR}"
source set_env.sh
cd "${CONTAINER_WORKDIR}/${REMOTE_EVAL_WORKDIR}"
if [[ "${ASCENDC_CLEAN_BUILD}" == "1" ]]; then
  PYTHONPATH="${CONTAINER_WORKDIR}/${REMOTE_EVAL_WORKDIR}/archive_tasks:${CONTAINER_WORKDIR}/${REMOTE_EVAL_WORKDIR}\${PYTHONPATH:+:\${PYTHONPATH}}" \
  python utils/build_ascendc.py "${TASK}" -v "${ASCENDC_SOC_VERSION}" --clean
else
  PYTHONPATH="${CONTAINER_WORKDIR}/${REMOTE_EVAL_WORKDIR}/archive_tasks:${CONTAINER_WORKDIR}/${REMOTE_EVAL_WORKDIR}\${PYTHONPATH:+:\${PYTHONPATH}}" \
  python utils/build_ascendc.py "${TASK}" -v "${ASCENDC_SOC_VERSION}"
fi
PYTHONPATH="${CONTAINER_WORKDIR}/${REMOTE_EVAL_WORKDIR}/archive_tasks:${CONTAINER_WORKDIR}/${REMOTE_EVAL_WORKDIR}\${PYTHONPATH:+:\${PYTHONPATH}}" \
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" \
WORKDIR="${CONTAINER_WORKDIR}/${REMOTE_EVAL_WORKDIR}" \
python "${DEBUG_SCRIPT}"
'
EOF

echo "[3/4] Running precision debug script inside container ${CONTAINER_NAME}"
"${SSH_CMD[@]}" \
  "${SSH_TARGET}" \
  "${REMOTE_SCRIPT}"

echo "[4/4] Precision debug run completed"
