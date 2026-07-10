#!/usr/bin/env python3
"""Unified script: generate embeddings, fuse with rank sweep, run downstream validation."""
import resource
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Fusion pipeline")
    parser.add_argument("--root", default="./log/full", help="Root path for checkpoints")
    parser.add_argument("--mirror-root", default="./mirror_log/full", help="Root path for generated/fused embeddings")
    parser.add_argument("--d", default="age", help="Dataset name")
    parser.add_argument("--m1", default="coles", help="Model 1")
    parser.add_argument("--s1", default="best_regression", help="Task 1")
    parser.add_argument("--m2", default="ntp_gru", help="Model 2")
    parser.add_argument("--s2", default="best_forecast", help="Task 2")
    parser.add_argument("--config", default="./universal_validator/configs/validator/logreg_lgbm_3seed_embedding_metrics.yaml")
    parser.add_argument("--rank-step", type=int, default=128)
    parser.add_argument("--gpu", type=str, default=0, help="GPU id for inference")
    parser.add_argument("--cleanup", action="store_true", help="Remove fused embeddings after validation")
    parser.add_argument("--test", action="store_true", help="Run only on the first rank for quick testing")
    parser.add_argument("--resample", action="store_true", help="Run only on the small dataset for quick testing")
    args = parser.parse_args()
    return args

args = parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import logging
import subprocess
import csv
import glob
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime
from typing import Any, cast
from copy import deepcopy
from pathlib import Path
from functools import partial
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from cca_zoo.linear import CCA, rCCA, PLS, SCCA_ADMM, PLS_ALS
from scipy.sparse.linalg import eigsh
from joblib import Parallel, delayed

# ------------------------------------------------------------------
# Helper for model directory names (uppercase for NTP_GRU/NTP_GPT)
# ------------------------------------------------------------------
def model_dir_name(m: str) -> str:
    return m.upper() if m.lower() in {'ntp_gru', 'ntp_gpt'} else m

# ------------------------------------------------------------------
# Suppress noisy logs
# ------------------------------------------------------------------
def _suppress_noisy_py4j_logs() -> None:
    logging.getLogger("py4j.clientserver").setLevel(logging.ERROR)
    logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)
    logging.getLogger("py4j").setLevel(logging.ERROR)

# ------------------------------------------------------------------
# Generate embeddings (output goes to original root, then moved to mirror)
# ------------------------------------------------------------------
def run_embedding_generation(root: str, mirror_root: str, d: str, m: str, s: str,
                             dv_config: str, gpu: int = 0, resample: bool = False) -> None:
    m_upper = model_dir_name(m)
    # Checkpoint
    ckpt_pattern = f"{root}/{d}/{m_upper}/tests/{s}/seed_0/pretrain/ckpt/*.ckpt"
    ckpt_files = sorted(glob.glob(ckpt_pattern))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint found: {ckpt_pattern}")
    checkpoint = ckpt_files[0]

    tmp_task = "fusion_tmp"
    # Remove any leftover temporary folder from a previous run
    tmp_dir = Path(f"{root}/{d}/{m_upper}/tests/{tmp_task}")
    if tmp_dir.exists():
        import shutil
        shutil.rmtree(tmp_dir)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TASK_NAME"] = tmp_task          # <-- forces output folder to be "fusion_tmp"
    env["ec"] = "rsample"
    env["PTH"] = checkpoint

    cmd = [
        "python", "main.py",
        "-d", f"full/{d}",
        "-m", m,
        "-e", "inference",
        "-s", s,                         # <-- real task name (e.g., best_regression)
        #s"-dv", dv_config,
        #"--extra-config", "rsample"
    ]
    if args.resample:
        cmd += ["--extra-config", "rsample"]
    print(f"Running: {' '.join(cmd)} (GPU {gpu})")
    subprocess.run(cmd, env=env, check=True)

    # The generated embeddings are now in .../tests/fusion_tmp/seed_0/embeddings
    generated_emb = tmp_dir / "seed_0" / "embeddings"
    if not generated_emb.exists():
        raise RuntimeError(f"Generation completed but {generated_emb} not found")

    # Move to mirror root with the correct task name
    mirror_emb_dir = Path(mirror_root) / d / m_upper / 'tests' / s / 'seed_0' / 'embeddings'
    mirror_emb_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Moving {generated_emb} -> {mirror_emb_dir}")
    os.rename(generated_emb, mirror_emb_dir)

    # Clean up the temporary folder (it should be empty now)
    try:
        tmp_dir.rmdir()
    except OSError:
        pass

# ------------------------------------------------------------------
# Robust parquet loading (from mirror root)
# ------------------------------------------------------------------
def safe_stack(series: pd.Series) -> np.ndarray:
    vals = []
    for x in series.values:
        if isinstance(x, tuple):
            x = x[0]
        arr = np.asarray(x.tolist() if hasattr(x, 'tolist') else x, dtype=np.float64)
        vals.append(arr)
    shapes = {a.shape for a in vals}
    if len(shapes) > 1:
        raise ValueError(f"Inconsistent shapes in column '{series.name}': {shapes}")
    return np.stack(vals)

def load_embeddings(mirror_root: str, d: str, m: str, s: str):
    m_dir = model_dir_name(m)
    train_path = f"{mirror_root}/{d}/{m_dir}/tests/{s}/seed_0/embeddings/train_postproc/"
    test_path  = f"{mirror_root}/{d}/{m_dir}/tests/{s}/seed_0/embeddings/test_postproc/"
    # Читаем всю папку, не отдельный файл
    X_train = pd.read_parquet(train_path)
    X_test  = pd.read_parquet(test_path)
    return (safe_stack(X_train['global_emb']),
            safe_stack(X_train['shift_emb']),
            safe_stack(X_test['global_emb']),
            safe_stack(X_test['shift_emb']))

# ------------------------------------------------------------------
# Fusion functions
# ------------------------------------------------------------------
def _get_latent_dim(d1, d2, n_samples, user_dim):
    max_possible = min(d1, d2, n_samples)
    if user_dim is not None:
        return min(user_dim, max_possible)
    return max_possible

def concat_fusion(v1_train, v2_train, v1_test, v2_test, rank=None):
    return np.concatenate([v1_train, v2_train], axis=1), np.concatenate([v1_test, v2_test], axis=1)

def pca_fusion(v1_train, v2_train, v1_test, v2_test, rank=None):
    concat_train = np.concatenate([v1_train, v2_train], axis=1)
    concat_test  = np.concatenate([v1_test, v2_test], axis=1)
    d = _get_latent_dim(v1_train.shape[1], v2_train.shape[1], v1_train.shape[0], rank)
    pca = PCA(n_components=d)
    return pca.fit_transform(concat_train), pca.transform(concat_test)

def cca_fusion(v1_train, v2_train, v1_test, v2_test, rank=None, model=CCA):
    ld = _get_latent_dim(v1_train.shape[1], v2_train.shape[1], v1_train.shape[0], rank)
    cca = model(latent_dimensions=ld).fit([v1_train, v2_train])
    tv = cca.transform([v1_train, v2_train])
    tev = cca.transform([v1_test, v2_test])
    return (tv[0] + tv[1]) / 2, (tev[0] + tev[1]) / 2

class EfficientTucker:
    def __init__(self, rank=None, batch_size=512, n_jobs=-1):
        self.rank = rank
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.B_ = None
        self.C_ = None

    def fit(self, v1_train, v2_train):
        F1, F2 = v1_train.shape[1], v2_train.shape[1]
        max_rank = min(F1, F2)
        if self.rank is None:
            r = max_rank
        else:
            r = min(self.rank[0] if isinstance(self.rank, tuple) else self.rank, max_rank)
        N = len(v1_train)
        batches = [(v1_train[i:i+self.batch_size], v2_train[i:i+self.batch_size])
                   for i in range(0, N, self.batch_size)]
        def process_batch(bv1, bv2):
            G1 = np.zeros((F1, F1))
            G2 = np.zeros((F2, F2))
            for x, y in zip(bv1, bv2):
                G1 += np.outer(x, x) * (y @ y)
                G2 += np.outer(y, y) * (x @ x)
            return G1, G2
        gram_parts = Parallel(n_jobs=self.n_jobs, backend='threading')(
            delayed(process_batch)(bv1, bv2) for bv1, bv2 in batches)
        G1 = np.sum(np.array([g[0] for g in gram_parts]), axis=0)
        G2 = np.sum(np.array([g[1] for g in gram_parts]), axis=0)
        _, B = eigsh(G1, k=r, which='LM')
        _, C = eigsh(G2, k=r, which='LM')
        self.B_ = B[:, ::-1]
        self.C_ = C[:, ::-1]
        self.r_ = r
        return self

    def transform(self, v1, v2):
        if self.B_ is None or self.C_ is None:
            raise RuntimeError("Fit the model first.")
        proj1 = v1 @ self.B_
        proj2 = v2 @ self.C_
        return np.concatenate([proj1, proj2], axis=1)

    def fit_transform(self, v1_train, v2_train, v1_test, v2_test):
        self.fit(v1_train, v2_train)
        return self.transform(v1_train, v2_train), self.transform(v1_test, v2_test)
    
class EfficientTuckerProduct(EfficientTucker):
    """
    EfficientTucker с поэлементным произведением проекций вместо конкатенации.
    fit() наследуется, transform() переопределён.
    """
    def transform(self, v1, v2):
        if self.B_ is None or self.C_ is None:
            raise RuntimeError("Fit the model first.")
        proj1 = v1 @ self.B_
        proj2 = v2 @ self.C_
        return proj1 * proj2

def tucker_concat_fusion(v1_train, v2_train, v1_test, v2_test, rank=None):
    return EfficientTucker(rank=rank).fit_transform(v1_train, v2_train, v1_test, v2_test)

def tucker_product_fusion(v1_train, v2_train, v1_test, v2_test, rank=None):
    return EfficientTuckerProduct(rank=rank).fit_transform(v1_train, v2_train, v1_test, v2_test)


def krossfuse_fusion(v1_train, v2_train, v1_test, v2_test, rank=None, random_state=42):
    """
    Fuse two views via random projection + Hadamard product.
    Each view is projected to `rank` dimensions using a fixed random matrix,
    then the element-wise product is computed.
    """
    F1, F2 = v1_train.shape[1], v2_train.shape[1]
    if rank is None:
        rank = min(F1, F2)
    rng = np.random.RandomState(random_state)
    # Random projection matrices with scaled uniform entries as in the original code
    proj1 = (rng.rand(F1, rank) * 2 * np.sqrt(3) - np.sqrt(3)) / np.sqrt(rank)
    proj2 = (rng.rand(F2, rank) * 2 * np.sqrt(3) - np.sqrt(3)) / np.sqrt(rank)
    # Apply projections and multiply element-wise
    train_fused = (v1_train @ proj1) * (v2_train @ proj2)
    test_fused  = (v1_test  @ proj1) * (v2_test  @ proj2)
    return train_fused, test_fused

def fuse_3d(fusion_func, v1_train, v2_train, v1_test, v2_test, name="3d"):
    B_tr, T_tr, F1 = v1_train.shape
    B_te, T_te, _ = v1_test.shape
    flat1_tr = v1_train.reshape(-1, F1)
    flat2_tr = v2_train.reshape(-1, v2_train.shape[2])
    flat1_te = v1_test.reshape(-1, F1)
    flat2_te = v2_test.reshape(-1, v2_test.shape[2])
    print(f"    [{name}] Flattened train: {flat1_tr.shape[0]} samples")
    fused_flat_tr, fused_flat_te = fusion_func(flat1_tr, flat2_tr, flat1_te, flat2_te)
    fused_dim = fused_flat_tr.shape[1]
    print(f"    [{name}] Fused dimension: {fused_dim}")
    return fused_flat_tr.reshape(B_tr, T_tr, fused_dim), fused_flat_te.reshape(B_te, T_te, fused_dim)

# ------------------------------------------------------------------
# Fusion + save (output goes to mirror_root)
# ------------------------------------------------------------------
def fuse_and_save(method_name, global_fn, shift_fn, rank,
                  g1_tr, g2_tr, g1_te, g2_te,
                  s1_tr, s2_tr, s1_te, s2_te,
                  X1_train_template, X1_test_template, 
                  mirror_root, d, m2, s2):
    rank_str = f"_rank{rank}" if rank is not None else ""
    full_name = f"{method_name}{rank_str}"
    print(f"\n=== {full_name} ===")
    try:
        g_tr, g_te = global_fn(g1_tr, g2_tr, g1_te, g2_te)
        s_tr, s_te = fuse_3d(shift_fn, s1_tr, s2_tr, s1_te, s2_te, name="shift_emb")
        out_train = deepcopy(X1_train_template)
        out_test  = deepcopy(X1_test_template)
        out_train['global_emb'] = [x.tolist() for x in g_tr]
        out_test['global_emb']  = [x.tolist() for x in g_te]
        out_train['shift_emb']  = [x.tolist() for x in s_tr]
        out_test['shift_emb']   = [x.tolist() for x in s_te]
        out_dir = Path(mirror_root) / d / model_dir_name(m2) / 'tests' / s2 / 'seed_0' / f'embeddings_{full_name}'
        (out_dir / 'train_postproc').mkdir(parents=True, exist_ok=True)
        (out_dir / 'test_postproc').mkdir(parents=True, exist_ok=True)
        out_train.to_parquet(out_dir / 'train_postproc' / 'data.parquet')
        out_test.to_parquet(out_dir / 'test_postproc' / 'data.parquet')
        print(f"  Saved to {out_dir}")
        return str(out_dir)
    except Exception as e:
        print(f"  SKIPPED: {e}")
        return None

# ------------------------------------------------------------------
# Downstream validation functions (unchanged)
# ------------------------------------------------------------------
from universal_validator.pipeline.universal_validator import UniversalValidator
from universal_validator.pipeline.utils import ValidatorConfig
from universal_validator.utils import ensure_validator_logging, run_with_config

def main(cfg: ValidatorConfig):
    ensure_validator_logging()
    _suppress_noisy_py4j_logs()
    validator = UniversalValidator(cfg)
    all_tasks = validator.get_available_tasks(verbose=True)
    if cfg.list_configs:
        return
    if cfg.task_names is None:
        tasks = all_tasks
    else:
        assert set(cfg.task_names) <= set(all_tasks)
        tasks = cfg.task_names
    reports = []
    embedding_report = validator.run_embedding_metrics()
    if embedding_report:
        reports.append(embedding_report)
    for task in tasks:
        report = validator.run_pipeline(task_name=task)
        report["task_name"] = task
        reports.append(report)
    return reports

def run_with_paths(downstream_config: Mapping[str, Any], train_path: str, test_path: str):
    raw_config = dict(downstream_config)
    data_conf_overrides = dict(raw_config.pop("data_conf", {}))
    cfg = cast(
        ValidatorConfig,
        OmegaConf.to_object(
            OmegaConf.merge(
                OmegaConf.structured(ValidatorConfig),
                OmegaConf.create(raw_config),
            )
        ),
    )
    cfg = replace(
        cfg,
        data_conf=replace(cfg.data_conf, **data_conf_overrides,
                          train_path=train_path, test_path=test_path),
    )
    return main(cfg)

def extract_downstream_metrics(reports) -> dict[str, float]:
    metrics = {}
    for report in reports:
        if "metrics" in report:          # embedding metrics report
            metrics.update(report["metrics"])
            continue
        if not report:
            continue
        task_name = report["task_name"]
        all_results = report["all_results"]

        # -- backward‑compatible: keep the best model's main metric under the task name
        _, metric_names = task_name.rsplit("__", 1)
        best_model = report.get("best_model")
        main_metric = metric_names.split("+")[0]
        if main_metric == "mse":
            main_metric = "neg_mean_squared_error"
        metrics[task_name] = float(all_results[best_model][main_metric])

        # -- additionally expose every model–metric pair
        for model_name, model_results in all_results.items():
            for key, value in model_results.items():
                if key in ("main_metric", "predictions", "model", "cv_results"):
                    continue
                flat_key = f"flat____{task_name}____{model_name}____{key}"
                metrics[flat_key] = float(value)
    return metrics

def set_validator_seed(downstream_config: dict, seed: int) -> None:
    for model_config in downstream_config.get("models", {}).values():
        shared_params = model_config.get("shared_params", {})
        if "random_state" in shared_params:
            shared_params["random_state"] = seed

def run_downstream_with_seed(downstream_config: dict, train_path: str, test_path: str, seed: int):
    seeded_config = deepcopy(downstream_config)
    seeded_config.pop("validator_seeds", None)
    if "embedding_metrics" in seeded_config:
        seeded_config["embedding_metrics"]["enabled"] = False
    set_validator_seed(seeded_config, seed)
    reports = run_with_paths(downstream_config=seeded_config, train_path=train_path, test_path=test_path)
    metrics = extract_downstream_metrics(reports)
    return reports, metrics

def save_seed_metrics(path: Path, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])

# ------------------------------------------------------------------
# Main execution
# ------------------------------------------------------------------
if __name__ == "__main__":
    
    start_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root = args.root
    mirror_root = args.mirror_root
    d, m1, s1, m2, s2 = args.d, args.m1, args.s1, args.m2, args.s2

    # Generate embeddings if not present (in mirror root)
    for m, s in [(m1, s1), (m2, s2)]:
        emb_dir = Path(f"{mirror_root}/{d}/{model_dir_name(m)}/tests/{s}/seed_0/embeddings/train_postproc/")
        if not emb_dir.exists() or not list(emb_dir.glob("*.parquet")):
            print(f"Generating embeddings for {m}/{s}...")
            run_embedding_generation(root, mirror_root, d, m, s, args.config, gpu=args.gpu, resample=args.resample)

    # Load all embeddings from mirror root
    g1_tr, s1_tr, g1_te, s1_te = load_embeddings(mirror_root, d, m1, s1)
    g2_tr, s2_tr, g2_te, s2_te = load_embeddings(mirror_root, d, m2, s2)

    # Template DataFrames for saving (from mirror root)
    X1_train = pd.read_parquet(f"{mirror_root}/{d}/{model_dir_name(m1)}/tests/{s1}/seed_0/embeddings/train_postproc/")
    X1_test  = pd.read_parquet(f"{mirror_root}/{d}/{model_dir_name(m1)}/tests/{s1}/seed_0/embeddings/test_postproc/")
    
    # ------------------------------------------------------------------
    # Validate original (non-fused) embeddings for each model
    # ------------------------------------------------------------------
    config_dict = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    seeds = config_dict.get("validator_seeds")
    if seeds is None:
        seeds = [None]
    elif isinstance(seeds, (int, float)):
        seeds = [int(seeds)]
    else:
        seeds = list(seeds)

    for m, s in [(m1, s1), (m2, s2)]:
        model_label = f"{m}_{s}"
        train_path = f"{mirror_root}/{d}/{model_dir_name(m)}/tests/{s}/seed_0/embeddings/train_postproc/"
        test_path  = f"{mirror_root}/{d}/{model_dir_name(m)}/tests/{s}/seed_0/embeddings/test_postproc/"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = Path.cwd() / f"outputs_validator_{start_timestamp}"  / f"{d}_{m1}_{s1}_{m2}_{s2}" / f"validator_output_{timestamp}_embeddings_{model_label}"
        run_dir.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            if seed is not None:
                seed_reports, seed_metrics = run_downstream_with_seed(
                    config_dict, train_path, test_path, seed
                )
                label = f"seed_{seed}"
            else:
                no_seed_config = deepcopy(config_dict)
                no_seed_config.pop("validator_seeds", None)
                if "embedding_metrics" in no_seed_config:
                    no_seed_config["embedding_metrics"]["enabled"] = False
                seed_reports = run_with_paths(
                    downstream_config=no_seed_config,
                    train_path=train_path,
                    test_path=test_path,
                )
                seed_metrics = extract_downstream_metrics(seed_reports)
                label = "no_seed"
            save_seed_metrics(run_dir / f"downstream_validator_{label}.csv", seed_metrics)
            pd.DataFrame(seed_reports).to_json(
                run_dir / f"validator_output_{label}.json",
                orient='records', indent=4, date_format='iso'
            )
        print(f"Original model {model_label} validation results saved to {run_dir}")

    # Determine ranks (based on the loaded embeddings)
    min_dim = min(g1_tr.shape[1], g2_tr.shape[1])
    max_dim = max(g1_tr.shape[1], g2_tr.shape[1])
    print('dims:', g1_tr.shape[1], g2_tr.shape[1])
    ranks = list(range(args.rank_step, max_dim + 1, args.rank_step))
    ranks.append(min_dim)
    ranks.append(max_dim)
    ranks = sorted(sorted(set(ranks)))
    
    # Fusion methods
    methods = [
        ('concatenation', concat_fusion, None),
        ('PCA', pca_fusion, None),
        ('CCA', cca_fusion, None),
        ('rCCA', partial(cca_fusion, model=rCCA), None),
        ('TuckerFactorConcat', tucker_concat_fusion, None),
        #('TuckerFactorProduct', tucker_product_fusion, None),
        ('KrossFuse', krossfuse_fusion, None),
    ]

    for method_name, global_fn, shift_fn in methods:
        if shift_fn is None:
            shift_fn = global_fn

        if method_name == 'concatenation':
            rank_list = [None]
        else:
            rank_list = [min_dim, max_dim] if args.test else ranks

        for rank in rank_list:
            out_dir = fuse_and_save(method_name, global_fn, shift_fn, rank,
                                    g1_tr, g2_tr, g1_te, g2_te,
                                    s1_tr, s2_tr, s1_te, s2_te,
                                    X1_train, X1_test, 
                                    mirror_root=mirror_root, d=d, m2=m2, s2=s2)
            if out_dir is None:
                continue

            train_path = f"{out_dir}/train_postproc/"
            test_path  = f"{out_dir}/test_postproc/"

            postfix = Path(out_dir).name
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_dir = Path.cwd() / f"outputs_validator_{start_timestamp}" / f"{d}_{m1}_{s1}_{m2}_{s2}" / f"validator_output_{timestamp}_{postfix}"
            run_dir.mkdir(parents=True, exist_ok=True)

            for seed in seeds:
                if seed is not None:
                    seed_reports, seed_metrics = run_downstream_with_seed(
                        config_dict, train_path, test_path, seed
                    )
                    label = f"seed_{seed}"
                else:
                    no_seed_config = deepcopy(config_dict)
                    no_seed_config.pop("validator_seeds", None)
                    if "embedding_metrics" in no_seed_config:
                        no_seed_config["embedding_metrics"]["enabled"] = False
                    seed_reports = run_with_paths(
                        downstream_config=no_seed_config,
                        train_path=train_path,
                        test_path=test_path,
                    )
                    seed_metrics = extract_downstream_metrics(seed_reports)
                    label = "no_seed"

                save_seed_metrics(run_dir / f"downstream_validator_{label}.csv", seed_metrics)
                pd.DataFrame(seed_reports).to_json(
                    run_dir / f"validator_output_{label}.json",
                    orient='records', indent=4, date_format='iso'
                )
                print(f"  Validation results saved to {run_dir}")

            if args.cleanup:
                import shutil
                shutil.rmtree(out_dir)
                print(f"  Removed {out_dir}")

    print("\nAll done.")
