#!/usr/bin/env bash
# Usage: bash scripts/res_checking/revalidate_saved_embeddings.sh zvuk 30music
# Or pass a single PATH/TO/seed_0.
set -euo pipefail
shopt -s nullglob
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if (( $# == 0 )); then
    echo "Usage: bash $0 DATASET [DATASET ...] | PATH/TO/seed_0" >&2
    exit 1
fi
if [[ ! -f "$1/config.yaml" ]]; then
    mkdir -p log/batch_runs
    log_dir=$(mktemp -d log/batch_runs/revalidate_saved.XXXXXXXX)
    printf 'run\tstatus\texit_code\n' > "$log_dir/status.tsv"
    echo "Logs: $log_dir"
    failed=0
    number=0
    for dataset in "$@"; do
        configs=(log/full/"$dataset"/SimCLR_masks/tests/best*/reeval/*/seed_0/config.yaml)
        if (( ${#configs[@]} == 0 )); then
            printf '%s\tNO_RUNS\t-\n' "$dataset" | tee -a "$log_dir/status.tsv"
            failed=1
        fi
        for config in "${configs[@]}"; do
            run=${config%/config.yaml}
            completed=("$run"/revalidation_saved.*/metrics.csv)
            if [[ -f "${run%/seed_0}/results.csv" ]] || (( ${#completed[@]} > 0 )); then
                printf '%s\tSKIPPED_COMPLETED\t-\n' "$run" | tee -a "$log_dir/status.tsv"
                continue
            fi
            if [[ ! -e "$run/embeddings/train" || ! -e "$run/embeddings/test" ]]; then
                printf '%s\tSKIPPED_NO_EMBEDDINGS\t-\n' "$run" | tee -a "$log_dir/status.tsv"
                continue
            fi
            number=$((number + 1))
            printf '%s\tSTARTED\t-\n' "$run" | tee -a "$log_dir/status.tsv"
            echo "Output: $log_dir/$number.log"
            if bash scripts/res_checking/revalidate_saved_embeddings.sh "$run" 2>&1 | tee "$log_dir/$number.log"; then
                printf '%s\tCOMPLETED\t0\n' "$run" | tee -a "$log_dir/status.tsv"
            else
                code=$?
                printf '%s\tFAILED\t%s\n' "$run" "$code" | tee -a "$log_dir/status.tsv"
                failed=1
            fi
        done
    done
    echo "Finished. Status: $log_dir/status.tsv"
    exit "$failed"
fi
if (( $# != 1 )); then
    echo "Pass only one seed directory, or a list of datasets." >&2
    exit 1
fi
python -u - "$1" <<'PY'
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

from omegaconf import OmegaConf
from ebes.pipeline.data_retrieve.downstreams import (
    aggregate_seed_metrics, create_postproc_spark_session,
    extract_downstream_metrics, post_processing, run_downstream_with_seed,
    run_with_paths, save_seed_metrics,
)

run = Path(sys.argv[1]).resolve()
config = OmegaConf.to_container(OmegaConf.load(run / 'config.yaml'), resolve=True)
train, test = (run / 'embeddings' / split for split in ('train', 'test'))
for path in (train, test):
    if not path.exists():
        raise FileNotFoundError(f'Saved raw embeddings not found: {path}')
output = Path(tempfile.mkdtemp(prefix='revalidation_saved.', dir=run))
print(f'Results: {output}', flush=True)
spark = create_postproc_spark_session()
try:
    for path, split in ((train, 'train'), (test, 'test')):
        post_processing(config, path, split, spark=spark)
finally:
    spark.stop()

validator = config['universal_validator']
train_path, test_path = str(train) + '_postproc', str(test) + '_postproc'
metrics = {}
seeds = validator.get('validator_seeds')
if seeds is None:
    metrics = extract_downstream_metrics(run_with_paths(validator, train_path, test_path))
else:
    if validator.get('embedding_metrics', {}).get('enabled', False):
        geometry = deepcopy(validator)
        geometry.pop('validator_seeds', None)
        geometry['models'] = {}
        geometry['task_names'] = []
        metrics.update(extract_downstream_metrics(run_with_paths(geometry, train_path, test_path)))
    results = []
    for seed in seeds:
        seed_metrics = run_downstream_with_seed(validator, train_path, test_path, seed)
        save_seed_metrics(output / f'downstream_validator_seed_{seed}.csv', seed_metrics)
        results.append((seed, seed_metrics))
    metrics.update(aggregate_seed_metrics(results))
save_seed_metrics(output / 'metrics.csv', metrics)
print(f'Completed: {output / "metrics.csv"}', flush=True)
PY
