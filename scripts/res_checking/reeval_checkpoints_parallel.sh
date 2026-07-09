#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

DATASET="${DATASET:-age}"
METHOD="${METHOD:-jepa_optuna}"
SPECIFY="${SPECIFY:-classification}"
TASK="${TASK:-${SPECIFY}}"
VALIDATOR="${VALIDATOR:-}"
GPU="${GPU:-cuda:0}"
EXTRA_CONFIG="${EXTRA_CONFIG:-}"
SEED_DIR="${SEED_DIR:-seed_0}"
EPOCHS="${EPOCHS:-1 $(seq 5 5 100)}"
FORCE="${FORCE:-0}"
REVAL_DIR="${TASK}/${REVAL_DIR:-revalidation}"
KEEP_RAW_EMBEDDINGS="${KEEP_RAW_EMBEDDINGS:-0}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ROOT="${RUN_ROOT:-${ROOT}/log/${DATASET}/${METHOD}/tests}"
CKPT_DIR="${CKPT_DIR:-${RUN_ROOT}/${TASK}/${SEED_DIR}/pretrain/ckpt}"

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
  out_dir="${RUN_ROOT}/${task_name}"
  done_marker="${out_dir}/DONE"
  lock_dir="${out_dir}.lock"
  result="${out_dir}/results.csv"

  results=()
  [[ -f "${result}" ]] && results+=("${result}")
  results+=("${out_dir}"\(*\)/results.csv)
  if [[ "${FORCE}" != "1" && -f "${done_marker}" ]]; then
    echo "skip epoch ${ep}: done (${done_marker})"
    continue
  fi
  if [[ "${FORCE}" != "1" && ${#results[@]} -gt 0 ]]; then
    echo "skip epoch ${ep}: ${results[0]}"
    continue
  fi

  mkdir -p "$(dirname "${out_dir}")"
  if ! mkdir "${lock_dir}" 2>/dev/null; then
    echo "skip epoch ${ep}: locked (${lock_dir})"
    continue
  fi
  {
    echo "pid=$$"
    echo "host=$(hostname)"
    echo "started_at=$(date -Iseconds)"
    echo "epoch=${epoch}"
  } > "${lock_dir}/info"

  if [[ "${FORCE}" == "1" ]]; then
    rm -rf "${out_dir}"
  fi

  ckpts=("${CKPT_DIR}/epoch__${ep}"*.ckpt)
  if [[ ${#ckpts[@]} -eq 0 ]]; then
    echo "skip epoch ${ep}: no checkpoint"
    rm -rf "${lock_dir}"
    continue
  fi
  if [[ ${#ckpts[@]} -ne 1 ]]; then
    echo "bad ckpt match for epoch ${ep}: ${CKPT_DIR}"
    rm -rf "${lock_dir}"
    exit 1
  fi

  validator_args=()
  [[ -n "${VALIDATOR}" ]] && validator_args=(-dv "${VALIDATOR}")
  extra_config_args=()
  [[ -n "${EXTRA_CONFIG}" ]] && extra_config_args=(--extra-config "${EXTRA_CONFIG}")

  echo "run epoch ${ep}: ${ckpts[0]}"
  if PTH="${ckpts[0]}" TASK_NAME="${task_name}" python main.py \
    -d "${DATASET}" \
    -m "${METHOD}" \
    -e inference \
    -s "${SPECIFY}" \
    -g "${GPU}" \
    "${extra_config_args[@]}" \
    "${validator_args[@]}"; then
    touch "${done_marker}"
    if [[ "${KEEP_RAW_EMBEDDINGS}" != "1" ]]; then
      rm -rf "${out_dir}/${SEED_DIR}/embeddings/train"
      rm -rf "${out_dir}/${SEED_DIR}/embeddings/test"
    fi
    rm -rf "${lock_dir}"
  else
    status=$?
    rm -rf "${lock_dir}"
    exit "${status}"
  fi
done
