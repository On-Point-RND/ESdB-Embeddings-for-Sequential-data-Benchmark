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
EPOCHS="${EPOCHS:-1 $(seq 5 5 1000)}"
FORCE="${FORCE:-0}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CKPT_DIR="${ROOT}/log/${DATASET}/${METHOD}/tests/${TASK}/${SEED_DIR}/ckpt"

cd "${ROOT}"

ckpt_files=("${CKPT_DIR}"/epoch__*.ckpt)
if [[ ${#ckpt_files[@]} -eq 0 ]]; then
  echo "no checkpoints in ${CKPT_DIR}"
  exit 1
fi

max_epoch=0
for ckpt in "${ckpt_files[@]}"; do
  name="$(basename "${ckpt}")"
  if [[ "${name}" =~ epoch__([0-9]+) ]]; then
    epoch_num=$((10#${BASH_REMATCH[1]}))
    (( epoch_num > max_epoch )) && max_epoch="${epoch_num}"
  fi
done

for epoch in ${EPOCHS}; do
  if (( epoch > max_epoch )); then
    echo "stop: epoch ${epoch} is after last checkpoint epoch ${max_epoch}"
    break
  fi

  ep="$(printf "%04d" "${epoch}")"
  task_name="${OUT}_epoch_${ep}"
  result="${ROOT}/log/${DATASET}/${METHOD}/tests/${task_name}/results.csv"
  results=()
  [[ -f "${result}" ]] && results+=("${result}")
  results+=("${ROOT}/log/${DATASET}/${METHOD}/tests/${task_name}"\(*\)/results.csv)
  if [[ "${FORCE}" != "1" && ${#results[@]} -gt 0 ]]; then
    echo "skip epoch ${ep}: ${results[0]}"
    continue
  fi

  ckpts=("${CKPT_DIR}/epoch__${ep}"*.ckpt)
  if [[ ${#ckpts[@]} -eq 0 ]]; then
    echo "skip epoch ${ep}: no checkpoint"
    continue
  fi
  [[ ${#ckpts[@]} -eq 1 ]] || { echo "bad ckpt match for epoch ${ep}: ${CKPT_DIR}"; exit 1; }

  PTH="${ckpts[0]}" TASK_NAME="${task_name}" python main.py \
    -d "${DATASET}" \
    -m "${METHOD}" \
    -e inference \
    -s "${SPECIFY}" \
    -g "${GPU}" \
    -dv "${VALIDATOR}"
done
