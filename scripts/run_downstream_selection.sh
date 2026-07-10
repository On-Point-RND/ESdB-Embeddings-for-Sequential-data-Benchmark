#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-twitter}"
if [ "$#" -gt 0 ]; then
  shift
fi
METHOD="${METHOD:-jepa}"
EXPERIMENT="${EXPERIMENT:-train_best_dstop}"
SPECIFY="${SPECIFY:-}"
EXTRA_CONFIG="${EXTRA_CONFIG:-downstream_selection}"
GPU="${GPU:-cuda:0}"
VALIDATOR="${VALIDATOR:-universal_validator/configs/validator/logreg.yaml}"

export DOWNSTREAM_EVERY="${DOWNSTREAM_EVERY:-1}"
export DOWNSTREAM_PATIENCE="${DOWNSTREAM_PATIENCE:-3}"

cmd=(
  python main.py
  -d "${DATASET}"
  -m "${METHOD}"
  -e "${EXPERIMENT}"
  -g "${GPU}"
  -dv "${VALIDATOR}"
  --extra-config "${EXTRA_CONFIG}"
)

if [ -n "${SPECIFY}" ]; then
  cmd+=(-s "${SPECIFY}")
fi

"${cmd[@]}" "$@"
