#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

DATASET="${DATASET:-age}"
METHOD="${METHOD:-jepa_clean}"
SPECIFY="${SPECIFY:-manual_1}"
TASK="${TASK:-jepa_manual_1}"
OUT="${OUT:-${TASK}_logreg}"
VALIDATOR="${VALIDATOR:-universal_validator/configs/validator/logreg.yaml}"
GPU="${GPU:-cuda:0}"
SEED_DIR="${SEED_DIR:-seed_0}"
EPOCHS="${EPOCHS:-5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100}"
FORCE="${FORCE:-0}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CKPT_DIR="${ROOT}/log/${DATASET}/${METHOD}/tests/${TASK}/${SEED_DIR}/ckpt"

cd "${ROOT}"

for epoch in ${EPOCHS}; do
  ep="$(printf "%04d" "${epoch}")"
  ckpts=("${CKPT_DIR}/epoch__${ep}"*.ckpt)
  [[ ${#ckpts[@]} -eq 1 ]] || { echo "bad ckpt match for epoch ${ep}: ${CKPT_DIR}"; exit 1; }

  task_name="${OUT}_epoch_${ep}"
  results=("${ROOT}/log/${DATASET}/${METHOD}/tests/${task_name}"/results.csv)
  results+=("${ROOT}/log/${DATASET}/${METHOD}/tests/${task_name}"\(*\)/results.csv)
  if [[ "${FORCE}" != "1" && ${#results[@]} -gt 0 ]]; then
    echo "skip epoch ${ep}: ${results[0]}"
    continue
  fi

  PTH="${ckpts[0]}" TASK_NAME="${task_name}" python main.py \
    -d "${DATASET}" \
    -m "${METHOD}" \
    -e inference \
    -s "${SPECIFY}" \
    -g "${GPU}" \
    -dv "${VALIDATOR}"
done
