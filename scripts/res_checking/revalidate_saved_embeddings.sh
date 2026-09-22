#!/usr/bin/env bash
# Usage: bash scripts/res_checking/revalidate_saved_embeddings.sh PATH/TO/seed_0
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if (( $# != 1 )); then
    echo "Usage: bash $0 PATH/TO/seed_0" >&2
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
