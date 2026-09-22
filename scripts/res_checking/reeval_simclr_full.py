#!/usr/bin/env python3
"""Revalidate saved SimCLR runs; use --dry-run to inspect checkpoint selection."""

import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ['alpha', 'x5-retail', 'twitter', 'zvuk', '30music']


def prepare(seed_dir, dataset, output_name, method="SimCLR"):
    config = OmegaConf.load(seed_dir / 'config.yaml')
    metric = config.unsupervised_trainer.ckpt_track_metric
    checkpoints = list((seed_dir / 'pretrain/ckpt').glob('*.ckpt'))
    if not checkpoints:
        raise ValueError(f'No checkpoints: {seed_dir / "pretrain/ckpt"}')

    def score(path):
        values = dict(part.split('__') for part in path.stem.split('_-_'))
        return float(values[metric])

    # Matches Trainer.best_checkpoint(), including its loss sign convention.
    checkpoint = max(checkpoints, key=score)
    config = OmegaConf.merge(config, OmegaConf.load('configs/experiments/inference.yaml'))
    if dataset in ('zvuk', '30music'):
        config = OmegaConf.merge(
            config, OmegaConf.load(f'configs/specify/full/{dataset}/eval_batch1.yaml')
        )
    else:
        config = OmegaConf.merge(
            config, OmegaConf.load(f'configs/specify/full/{dataset}/rsample.yaml')
        )
    config.run_name = f'{method}/tests/{output_name}'
    config.device = 'cuda:0'
    config.unsupervised_trainer.ckpt_resume = str(checkpoint.resolve())
    return config, checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('datasets', nargs='*', default=DATASETS, choices=None)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--method', choices=['SimCLR', 'SimCLR_masks'], default='SimCLR')
    parser.add_argument('--validator', default=(
        'universal_validator/configs/validator/logreg_3seed_embedding_metrics.yaml'
    ))
    args = parser.parse_args()
    if any(dataset not in DATASETS for dataset in args.datasets):
        parser.error(f'Datasets: {", ".join(DATASETS)}')
    os.chdir(ROOT)
    batch_root = ROOT / 'log/batch_runs'
    batch_root.mkdir(parents=True, exist_ok=True)
    log_dir = Path(tempfile.mkdtemp(prefix=f'{args.method}_reeval.', dir=batch_root))
    print(f'Logs: {log_dir}', flush=True)
    failed = False
    with (log_dir / 'status.tsv').open('w') as status_file:
        status = csv.writer(status_file, delimiter='\t')
        status.writerow(['dataset', 'run', 'status', 'checkpoint', 'exit_code'])
        for dataset in args.datasets:
            run_root = Path(f'log/full/{dataset}/{args.method}/tests')
            runs = sorted(run_root.glob('best*/seed_0/config.yaml'))
            if not runs:
                print(f'{dataset}: NO_RUNS; searched {run_root.resolve()}/best*/seed_0/config.yaml', flush=True)
                status.writerow([dataset, '-', 'NO_RUNS', '-', '-'])
                failed = True
            for saved_config in runs:
                seed_dir = saved_config.parent
                run = seed_dir.parent.name
                temporary = None
                try:
                    output_name = f'{run}/reeval/{log_dir.name}'
                    config, checkpoint = prepare(seed_dir, dataset, output_name, args.method)
                    print(f'{dataset}/{run}: {checkpoint}', flush=True)
                    if args.dry_run:
                        status.writerow([dataset, run, 'PLANNED', checkpoint, '-'])
                        continue
                    specify_dir = Path(f'configs/specify/full/{dataset}/{args.method}')
                    specify_dir.mkdir(parents=True, exist_ok=True)
                    with tempfile.NamedTemporaryFile(
                        prefix='__reeval_', suffix='.yaml', dir=specify_dir, delete=False
                    ) as file:
                        temporary = Path(file.name)
                    OmegaConf.save(config, temporary)
                    command = [sys.executable, '-u', 'main.py', '-d', f'full/{dataset}',
                               '-m', args.method, '-e', 'inference', '-s', temporary.stem,
                               '-g', 'cuda:0', '-dv', args.validator]
                    status.writerow([dataset, run, 'STARTED', checkpoint, '-'])
                    status_file.flush()
                    with (log_dir / f'{dataset}_{run}.log').open('w') as output:
                        with subprocess.Popen(command, stdout=subprocess.PIPE,
                                              stderr=subprocess.STDOUT, text=True) as process:
                            for line in process.stdout:
                                print(line, end='', flush=True)
                                output.write(line)
                                output.flush()
                            code = process.wait()
                    state = 'COMPLETED' if code == 0 else 'FAILED'
                    status.writerow([dataset, run, state, checkpoint, code])
                    failed |= code != 0
                except (OSError, ValueError, KeyError) as error:
                    print(f'{dataset}/{run}: {error}', flush=True)
                    status.writerow([dataset, run, 'ERROR', str(error), '-'])
                    failed = True
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
                    status_file.flush()
    return int(failed)


if __name__ == '__main__':
    sys.exit(main())
