#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG="${1:-${PROJECT_ROOT}/configs/memory_simulation.json}"
SHOTS_OVERRIDE="${2:-}"
BATCH_OVERRIDE="${3:-}"
WORKERS_OVERRIDE="${4:-}"
QUIET_FLAG="${5:-}"

# Use currently activated environment (e.g. conda). Optional override:
#   CONCATBP_PYTHON=/path/to/python bash run_concatbp_threshold.sh ...
PYTHON_CMD="${CONCATBP_PYTHON:-python}"

CMD=("${PYTHON_CMD}" -m concatbp.experiment_cli --config "${CONFIG}")
if [[ -n "${SHOTS_OVERRIDE}" ]]; then
  CMD+=(--shots "${SHOTS_OVERRIDE}")
fi
if [[ -n "${BATCH_OVERRIDE}" ]]; then
  CMD+=(--shots-per-batch "${BATCH_OVERRIDE}")
fi
if [[ -n "${WORKERS_OVERRIDE}" ]]; then
  CMD+=(--workers "${WORKERS_OVERRIDE}")
fi
if [[ "${QUIET_FLAG}" == "quiet" ]]; then
  CMD+=(--quiet)
fi

echo "[concatbp] running memory experiment"
echo "  project: ${PROJECT_ROOT}"
echo "  config : ${CONFIG}"
echo "  python : ${PYTHON_CMD}"
if [[ -n "${SHOTS_OVERRIDE}" ]]; then
  echo "  shots override: ${SHOTS_OVERRIDE}"
fi
if [[ -n "${BATCH_OVERRIDE}" ]]; then
  echo "  batch override: ${BATCH_OVERRIDE}"
fi
if [[ -n "${WORKERS_OVERRIDE}" ]]; then
  echo "  workers override: ${WORKERS_OVERRIDE}"
fi
if [[ "${QUIET_FLAG}" == "quiet" ]]; then
  echo "  quiet mode: enabled"
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${CMD[@]}"

echo "[concatbp] run done"
