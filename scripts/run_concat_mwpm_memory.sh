#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG="${1:-}"
QUIET_FLAG="${2:-}"

PYTHON_CMD="${CONCATBP_PYTHON:-python}"

if [[ -z "${CONFIG}" ]]; then
  echo "usage: bash scripts/run_concat_mwpm_memory.sh <config.json> [quiet]" >&2
  exit 2
fi

CMD=("${PYTHON_CMD}" -m concatbp.concat_mwpm.cli --config "${CONFIG}")
if [[ "${QUIET_FLAG}" == "quiet" ]]; then
  CMD+=(--quiet)
fi

echo "[concat_mwpm] running memory experiment"
echo "  project: ${PROJECT_ROOT}"
echo "  config : ${CONFIG}"
echo "  python : ${PYTHON_CMD}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${CMD[@]}"

echo "[concat_mwpm] run done"
