import csv
import gc
import logging
import random
from copy import deepcopy
from dataclasses import dataclass
import shutil
from multiprocessing import current_process
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from pyarrow.parquet import ParquetDataset
from pyspark.sql import SparkSession

from ebes.data.utils import build_loaders, get_accessor, get_collator, get_loader

from validate import run_with_paths
from universal_validator.tasks.scorers import METRIC_INFO

from ..data_retrieve.auto_post_processing import post_processing
from ..data_retrieve.embeddings_gen import ResultsGetter

logger = logging.getLogger(__name__)


def create_postproc_spark_session() -> SparkSession:
    # Use dynamic ports (0 = OS picks a free port) to avoid conflicts when
    # multiple seeds run in parallel. Disable UI to eliminate that port entirely.
    proc_name = current_process().name
    return (
        SparkSession.builder.appName("JoinEmbeddings").master("local[8]")  # type: ignore
        .config("spark.sql.legacy.parquet.nanosAsLong", "true")
        .config("spark.driver.memory", "48g")
        .config("spark.driver.memoryOverhead", "8g")
        .config("spark.executor.memory", "20g")
        .getOrCreate()
    )


def extract_downstream_metrics(reports) -> dict[str, float]:
    metrics = {}
    for report in reports:
        if "metrics" in report:
            metrics.update(report["metrics"])
            continue
        if not report:
            continue

        task = report["task_name"]
        result = report["all_results"][report["best_model"]]
        first = METRIC_INFO[task.rsplit("__", 1)[1].split("+")[0]]["scorer"]
        metrics[task] = float(result[first])
        for split in ("val", "test"):
            values = result.get(f"{split}_metrics", {})
            if values:
                metrics[f"downstream/{split}/{task}"] = float(values[first])
                for name, value in values.items():
                    metrics[f"downstream/{split}/{task}/{name}"] = float(value)
    return metrics


def set_validator_seed(downstream_config: dict, seed: int) -> None:
    for model_config in downstream_config.get("models", {}).values():
        shared_params = model_config.get("shared_params", {})
        if "random_state" in shared_params:
            shared_params["random_state"] = seed


def run_downstream_with_seed(
    downstream_config: dict,
    train_path: str,
    test_path: str,
    seed: int,
) -> dict[str, float]:
    seeded_config = deepcopy(downstream_config)
    seeded_config.pop("validator_seeds", None)
    if "embedding_metrics" in seeded_config:
        seeded_config["embedding_metrics"]["enabled"] = False
    set_validator_seed(seeded_config, seed)
    reports = run_with_paths(
        downstream_config=seeded_config,
        train_path=train_path,
        test_path=test_path,
    )
    return extract_downstream_metrics(reports)


def aggregate_seed_metrics(
    metrics_by_seed: list[tuple[int, dict[str, float]]],
) -> dict[str, float]:
    if len(metrics_by_seed) == 1:
        return metrics_by_seed[0][1]

    aggregated = {}
    for seed, metrics in metrics_by_seed:
        for key, value in metrics.items():
            aggregated[f"{key}__validator_seed_{seed}"] = value
    return aggregated


def save_seed_metrics(path: Path, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])


def compute_downstreams(
    trainer,
    train_loaders,
    test_loaders,
    config,
    downstream_config,
    *,
    model=None,
    device=None,
):
    if downstream_config is not None and "global_train_filter" in config:
        downstream_config = dict(
            downstream_config, global_train_filter=config["global_train_filter"]
        )
    model = model if model is not None else trainer.model
    device = device if device is not None else trainer.device
    train_embeddings_getter = ResultsGetter(config, "train")
    keys = {"gen_train", "gen_train_val"}
    subloaders = {k: train_loaders[k] for k in keys if k in train_loaders}
    df_train = train_embeddings_getter.df_get(
        subloaders, model=model, device=device
    )
    embed_train_file = (
        Path(config["log_dir"]) / config["run_name"] / "embeddings" / "train"
    )
    embed_train_file.parent.mkdir(parents=True, exist_ok=True)
    if downstream_config and downstream_config.get("use_is_val"):
        if df_train.groupby("base_index").is_val.nunique().gt(1).any():
            raise ValueError("An entity belongs to both gen_train and gen_train_val")
    df_train.to_parquet(embed_train_file, index=False)
    del df_train, train_embeddings_getter

    embed_test_file = None
    if (
        test_loaders
        and (downstream_config or {}).get("evaluate_test", True)
        and (downstream_config is None or needs_local_test(config, downstream_config))
    ):
        test_embeddings_getter = ResultsGetter(config, "test")
        subloaders = {k: v for k, v in test_loaders.items() if k == "gen_test"}
        df_test = test_embeddings_getter.df_get(
            subloaders, model=model, device=device
        )
        embed_test_file = embed_train_file.with_name("test")
        df_test.to_parquet(embed_test_file, index=False)
        del df_test, test_embeddings_getter, subloaders
    gc.collect()

    spark = create_postproc_spark_session()
    try:
        post_processing(config, embed_train_file, "train", spark=spark)
        if embed_test_file is not None:
            post_processing(config, embed_test_file, "test", spark=spark)
    finally:
        spark.stop()

    test_path = (
        str(embed_test_file) + "_postproc" if embed_test_file is not None else ""
    )
    downstream_metrics = {}
    if downstream_config:
        validator_seeds = downstream_config.get("validator_seeds")
        if validator_seeds is None:
            reports = run_with_paths(
                downstream_config=downstream_config,
                train_path=str(embed_train_file) + "_postproc",
                test_path=test_path,
            )
            downstream_metrics = extract_downstream_metrics(reports)
            shutil.rmtree(Path(config["log_dir"]) / config["run_name"] / "embeddings")
            return downstream_metrics

        if downstream_config.get("embedding_metrics", {}).get("enabled", False):
            geometry_config = deepcopy(downstream_config)
            geometry_config.pop("validator_seeds", None)
            geometry_config["models"] = {}
            geometry_config["task_names"] = []
            reports = run_with_paths(
                downstream_config=geometry_config,
                train_path=str(embed_train_file) + "_postproc",
                test_path=test_path,
            )
            downstream_metrics.update(extract_downstream_metrics(reports))

        metrics_by_seed = []
        run_dir = Path(config["log_dir"]) / config["run_name"]
        for seed in validator_seeds:
            seed_metrics = run_downstream_with_seed(
                downstream_config=downstream_config,
                train_path=str(embed_train_file) + "_postproc",
                test_path=test_path,
                seed=seed,
            )
            metrics_by_seed.append((seed, seed_metrics))
            if len(validator_seeds) > 1:
                save_seed_metrics(
                    run_dir / f"downstream_validator_seed_{seed}.csv",
                    seed_metrics,
                )
        downstream_metrics.update(aggregate_seed_metrics(metrics_by_seed))
        shutil.rmtree(Path(config["log_dir"]) / config["run_name"] / "embeddings")
    return downstream_metrics


def _prepare_embedding_model(training_model, embedding_model, device):
    model = embedding_model if embedding_model is not None else training_model
    if model is not training_model:
        result = model.load_state_dict(training_model.state_dict(), strict=False)
        if result.missing_keys:
            raise ValueError(f"Missing embedding weights: {result.missing_keys}")
    return model.to(device).eval()


@dataclass
class _SampledAccessor:
    source: Any
    index_name: str
    fraction: float = 1.0
    seed: int = 42
    exclude_global_test: bool = False
    size: int | None = None
    filter_sampling_pool: bool = False

    def get_split(self, index):
        data = self.source.get_split(index)
        if not 0 < self.fraction <= 1:
            raise ValueError("Sample fraction must be in (0, 1]")
        if self.size is not None:
            if self.size < 1:
                raise ValueError("Sample size must be positive or null")
            eligible = data
            if self.filter_sampling_pool and "global_train" in data:
                eligible = data[data.global_train == 1]
            ids = eligible[self.index_name].drop_duplicates()
            ids = ids.sample(n=min(self.size, len(ids)), random_state=self.seed)
            selected = data[self.index_name].isin(ids)
            if (self.filter_sampling_pool and not self.exclude_global_test
                    and "global_train" in data):
                # Diagnostic group 0 must not change the selected fitting IDs.
                test_ids = data.loc[data.global_train == 0, self.index_name].drop_duplicates()
                test_ids = test_ids.sample(n=min(self.size, len(test_ids)), random_state=self.seed)
                selected |= data[self.index_name].isin(test_ids)
            data = data[selected]
        elif self.fraction < 1:
            ids = (
                data[self.index_name]
                .drop_duplicates()
                .sample(frac=self.fraction, random_state=self.seed)
            )
            data = data[data[self.index_name].isin(ids)]
        if self.exclude_global_test and "global_train" in data:
            data = data[data.global_train == 1]
        if data.empty:
            raise ValueError("Empty probe split; increase the sample fraction")
        return data


def build_validation_loaders(
    data, *, accessor=None, fractions=None, seed=42, include_global_test=True,
    global_train_filter=None, sizes=None,
):
    """Experiment-specific sampling/filtering; ebes.data remains unchanged."""
    if global_train_filter not in (None, "exclude_test"):
        raise ValueError("global_train_filter must be null or exclude_test")
    source = accessor if accessor is not None else get_accessor(**data["dataset"])
    collators = {
        key: get_collator(**value) for key, value in data["preprocessing"].items()
    }
    loaders = {}
    for name, params in data["loaders"].items():
        if fractions is not None and name not in fractions:
            continue
        accessor = _SampledAccessor(
            source,
            collators[params["preprocessing"]].index_name,
            (fractions or {}).get(name, 1.0),
            seed,
            global_train_filter == "exclude_test"
            and (not name.startswith("gen_") or not include_global_test),
            size=(sizes or {}).get(name),
            filter_sampling_pool=global_train_filter == "exclude_test",
        )
        selected = dict(params)
        if fractions is not None:
            selected.update(shuffle=False, loop=False, drop_incomplete=False)
        loaders[name] = get_loader(accessor, collators, **selected)
    return loaders



def needs_local_test(config, validator):
    tasks = validator.get("task_names")
    if tasks is None:
        tasks = ParquetDataset(config["data"]["dataset"]["parquet_path"]).schema.names
    return any(name.startswith("target__") and "__local__" in name for name in tasks)


def make_checkpoint_evaluator(config, embedding_model=None, *, accessor=None):
    """Supply downstream metrics to Trainer's existing checkpoint selection."""
    selection = config.get("downstream_selection")
    if not selection:
        return None
    every = selection["every_n_epochs"]
    if every < 1:
        raise ValueError("every_n_epochs must be positive")
    diagnostic = selection.get("evaluate_test", False)
    seed = selection.get("sample_seed", 42)
    loaders = build_validation_loaders(
        config["data"],
        accessor=accessor,
        seed=seed,
        include_global_test=diagnostic,
        global_train_filter=config.get("global_train_filter"),
        sizes={"gen_train": selection.get("train_size"),
               "gen_train_val": selection.get("val_size")},
        fractions={
            "gen_train": selection.get("train_fraction", 1.0),
            "gen_train_val": selection.get("val_fraction", 1.0),
        },
    )
    validator = deepcopy(config["universal_validator"])
    validator.update(
        use_is_val=True, evaluate_test=diagnostic,
        task_names=selection.get("task_names"),
        global_train_filter=config.get("global_train_filter"),
    )
    validator["embedding_metrics"] = {"enabled": False}
    test_loaders = {}
    if diagnostic and needs_local_test(config, validator):
        test_loaders = build_validation_loaders(
            config["test_data"],
            fractions={"gen_test": selection.get("test_fraction", 1.0)},
            seed=seed,
        )

    def evaluate(training_model, device, epoch, is_last_check):
        if epoch != 1 and epoch % every and not is_last_check:
            return None
        model = embedding_model if embedding_model is not None else training_model
        was_training = model.training
        py_state, np_state = random.getstate(), np.random.get_state()
        devices = (
            [torch.device(device).index or 0] if str(device).startswith("cuda") else []
        )
        try:
            with torch.random.fork_rng(devices=devices):
                model = _prepare_embedding_model(
                    training_model, embedding_model, device
                )
                epoch_config = dict(
                    config,
                    run_name=config["run_name"] + f"/downstream/epoch_{epoch:04d}",
                )
                metrics = compute_downstreams(
                    None,
                    loaders,
                    test_loaders,
                    epoch_config,
                    validator,
                    model=model,
                    device=device,
                )
        finally:
            model.train(was_training)
            if embedding_model is not None:
                model.to("cpu")
            random.setstate(py_state)
            np.random.set_state(np_state)
        save_seed_metrics(
            Path(config["log_dir"])
            / config["run_name"]
            / "downstream"
            / f"epoch_{epoch:04d}.csv",
            metrics,
        )
        return metrics

    return evaluate


def finish_downstreams(trainer, loaders, config, embedding_model=None):
    """Optionally run the existing final validator on the selected checkpoint(s)."""
    scores = trainer.checkpoint_scores
    if not config["downstream_selection"].get("final_evaluation", False):
        return scores
    validator = dict(
        config["universal_validator"], use_is_val=False, evaluate_test=True
    )
    test_loaders = (
        build_loaders(**config["test_data"])
        if needs_local_test(config, validator)
        else {}
    )
    for key, path in trainer.selected_checkpoints.items():
        trainer.load_ckpt(path)
        model = _prepare_embedding_model(trainer.model, embedding_model, trainer.device)
        task_validator = dict(validator)
        if key is not None:
            task_validator["task_names"] = [key.removeprefix("downstream/val/")]
        final_metrics = compute_downstreams(
            trainer,
            loaders,
            test_loaders,
            config,
            task_validator,
            model=model,
            device=trainer.device,
        )
        scores.update(
            {
                key: value
                for key, value in final_metrics.items()
                if not key.startswith("downstream/")
            }
        )
    return scores
