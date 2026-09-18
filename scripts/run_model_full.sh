#!/usr/bin/env bash
# Usage: CUDA_VISIBLE_DEVICES=2 bash scripts/run_model_full.sh MODEL [DATASET ...]
set -euo pipefail
shopt -s nullglob

cd "$(dirname "${BASH_SOURCE[0]}")/.."
if (( $# == 0 )); then
    echo "Usage: bash scripts/run_model_full.sh MODEL [DATASET ...]" >&2
    exit 1
fi
method=$1
shift
if [[ ! -f "configs/methods/${method}.yaml" ]]; then
    echo "Unknown model: ${method}" >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

mkdir -p log/batch_runs
log_dir=$(mktemp -d "log/batch_runs/${method}_full.XXXXXXXX")
status_log="${log_dir}/status.tsv"
printf 'time\tdataset\ttask\tstatus\texit_code\n' > "${status_log}"
echo "Logs: ${log_dir}"

record() {
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Iseconds)" "$dataset" "$task" "$1" "$2" \
        | tee -a "${status_log}"
}

datasets=("$@")
if (( ${#datasets[@]} == 0 )); then
    for config_dir in configs/specify/full/*/"${method}"; do
        dataset_dir=${config_dir%/*}
        datasets+=("${dataset_dir##*/}")
    done
fi

failed=0
for dataset in "${datasets[@]}"; do
    configs=(configs/specify/full/"${dataset}"/"${method}"/*.yaml)
    if (( ${#configs[@]} == 0 )); then
        task=-
        record NO_CONFIGS -
        failed=1
        continue
    fi
    for config in "${configs[@]}"; do
        task=${config##*/}
        task=${task%.yaml}

        record STARTED -
        echo "Output: ${log_dir}/${dataset}_${task}.log"
        if TASK_NAME="$task" python -u main.py \
            -d "full/${dataset}" -m "$method" -e train_best -s "$task" \
            -dv universal_validator/configs/validator/logreg_3seed.yaml \
            2>&1 | tee "${log_dir}/${dataset}_${task}.log"; then
            record COMPLETED 0
        else
            code=$?
            record FAILED "$code"
            failed=1
        fi
    done
done

echo "Finished. Status: ${status_log}"
exit "$failed"
