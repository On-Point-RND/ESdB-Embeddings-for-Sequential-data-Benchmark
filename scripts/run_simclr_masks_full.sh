#!/usr/bin/env bash
# Usage: CUDA_VISIBLE_DEVICES=2 bash scripts/run_simclr_masks_full.sh [age alpha ...]
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

mkdir -p log/batch_runs
log_dir=$(mktemp -d "log/batch_runs/simclr_masks_full.XXXXXXXX")
status_log="${log_dir}/status.tsv"
printf 'time\tdataset\ttask\tstatus\texit_code\n' > "${status_log}"
echo "Logs: ${log_dir}"

record() {
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Iseconds)" "$dataset" "$task" "$1" "$2" \
        | tee -a "${status_log}"
}

datasets=("$@")
if (( ${#datasets[@]} == 0 )); then
    for config_dir in configs/specify/full/*/SimCLR_masks; do
        dataset_dir=${config_dir%/SimCLR_masks}
        datasets+=("${dataset_dir##*/}")
    done
fi

failed=0
for dataset in "${datasets[@]}"; do
    for task in best_classification best_regression best_anomaly best_forecasting; do
        if [[ ! -f "configs/specify/full/${dataset}/SimCLR_masks/${task}.yaml" ]]; then
            record SKIPPED_MISSING_CONFIG -
            continue
        fi

        record STARTED -
        if TASK_NAME="$task" python -u main.py \
            -d "full/${dataset}" -m SimCLR_masks -e train_best -s "$task" \
            -dv universal_validator/configs/validator/logreg_3seed.yaml \
            > "${log_dir}/${dataset}_${task}.log" 2>&1; then
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
