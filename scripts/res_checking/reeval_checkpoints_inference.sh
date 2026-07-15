#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

DATASET="${DATASET:-age}"
METHOD="${METHOD:-jepa_optuna}"
SPECIFY="${SPECIFY:-classification}"
TASK="${TASK:-reeval_clf}"
VALIDATOR="${VALIDATOR:-}"
GPU="${GPU:-cuda:0}"
EXTRA_CONFIG="${EXTRA_CONFIG:-}"
SEED_DIR="${SEED_DIR:-seed_0}"
EPOCHS="${EPOCHS:-}"
FORCE="${FORCE:-0}"
REVAL_DIR="${REVAL_DIR:-${TASK}/revalidation}"
KEEP_RAW_EMBEDDINGS="${KEEP_RAW_EMBEDDINGS:-0}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_METHOD="${METHOD}"
case "${METHOD}" in
  ntp_gru) LOG_METHOD="NTP_GRU" ;;
  ntp_gpt) LOG_METHOD="NTP_GPT" ;;
esac
RUN_ROOT="${RUN_ROOT:-${ROOT}/log/${DATASET}/${LOG_METHOD}/tests}"
if [[ -z "${CKPT_DIR:-}" ]]; then
  CKPT_DIR="${RUN_ROOT}/${TASK}/${SEED_DIR}/ckpt"
  if [[ ! -d "${CKPT_DIR}" ]]; then
    PRETRAIN_CKPT_DIR="${RUN_ROOT}/${TASK}/${SEED_DIR}/pretrain/ckpt"
    if [[ -d "${PRETRAIN_CKPT_DIR}" ]]; then
      CKPT_DIR="${PRETRAIN_CKPT_DIR}"
    fi
  fi
fi

cd "${ROOT}"

ckpt_files=("${CKPT_DIR}"/epoch__*.ckpt)
if [[ ${#ckpt_files[@]} -eq 0 ]]; then
  echo "no checkpoints in ${CKPT_DIR}"
  exit 1
fi

if [[ -z "${EPOCHS}" ]]; then
  if [[ ${#ckpt_files[@]} -ne 1 ]]; then
    echo "EPOCHS is not set and ${CKPT_DIR} contains ${#ckpt_files[@]} checkpoints."
    echo "Set EPOCHS explicitly or leave exactly one checkpoint in the directory."
    exit 1
  fi
  name="$(basename "${ckpt_files[0]}")"
  if [[ ! "${name}" =~ epoch__([0-9]+) ]]; then
    echo "failed to parse epoch from checkpoint: ${ckpt_files[0]}"
    exit 1
  fi
  EPOCHS="$((10#${BASH_REMATCH[1]}))"
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
  out_dir="${RUN_ROOT}/${task_name}"
  result="${out_dir}/results.csv"
  results=()
  [[ -f "${result}" ]] && results+=("${result}")
  results+=("${out_dir}"\(*\)/results.csv)
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
  extra_config_args=()
  [[ -n "${EXTRA_CONFIG}" ]] && extra_config_args=(--extra-config "${EXTRA_CONFIG}")

  echo "run epoch ${ep}: ${ckpts[0]}"
  PTH="${ckpts[0]}" TASK_NAME="${task_name}" python main.py \
    -d "${DATASET}" \
    -m "${METHOD}" \
    -e inference \
    -s "${SPECIFY}" \
    -g "${GPU}" \
    "${extra_config_args[@]}" \
    "${validator_args[@]}"

  if [[ "${KEEP_RAW_EMBEDDINGS}" != "1" ]]; then
    rm -rf "${out_dir}/${SEED_DIR}/embeddings/train"
    rm -rf "${out_dir}/${SEED_DIR}/embeddings/test"
  fi
done
