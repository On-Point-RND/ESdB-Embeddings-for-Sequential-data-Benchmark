#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

DATASET="${DATASET:-age}"
METHOD="${METHOD:-jepa_optuna}"
SPECIFY="${SPECIFY-classification}"
TASK="${TASK:-reeval_clf}"
VALIDATOR="${VALIDATOR:-}"
GPU="${GPU:-cuda:0}"
EXTRA_CONFIGS="${EXTRA_CONFIGS:-${EXTRA_CONFIG:-}}"
SEED_DIR="${SEED_DIR:-seed_0}"
EPOCHS="${EPOCHS:-1 $(seq 5 5 100)}"
FORCE="${FORCE:-0}"
REVAL_DIR="${REVAL_DIR:-${TASK}/revalidation}"
KEEP_RAW_EMBEDDINGS="${KEEP_RAW_EMBEDDINGS:-0}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CKPT_DIR="${ROOT}/log/${DATASET}/${METHOD}/${TASK}/${SEED_DIR}/ckpt"

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
  task_name="${REVAL_DIR}/epoch_${ep}"
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

  validator_args=()
  [[ -n "${VALIDATOR}" ]] && validator_args=(-dv "${VALIDATOR}")
  specify_args=()
  [[ -n "${SPECIFY}" ]] && specify_args=(-s "${SPECIFY}")
  extra_config_args=()
  if [[ -n "${EXTRA_CONFIGS}" ]]; then
    read -r -a extra_configs <<< "${EXTRA_CONFIGS}"
    extra_config_args=(--extra-config "${extra_configs[@]}")
  fi

  PTH="${ckpts[0]}" TASK_NAME="${task_name}" python main.py \
    -d "${DATASET}" \
    -m "${METHOD}" \
    -e inference \
    "${specify_args[@]}" \
    -g "${GPU}" \
    "${extra_config_args[@]}" \
    "${validator_args[@]}"

  if [[ "${KEEP_RAW_EMBEDDINGS}" != "1" ]]; then
    rm -rf "${ROOT}/log/${DATASET}/${METHOD}/tests/${task_name}/${SEED_DIR}/embeddings/train"
    rm -rf "${ROOT}/log/${DATASET}/${METHOD}/tests/${task_name}/${SEED_DIR}/embeddings/test"
  fi
done
