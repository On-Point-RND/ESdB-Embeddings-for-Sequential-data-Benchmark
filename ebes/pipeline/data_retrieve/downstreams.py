import csv
import logging
from copy import deepcopy
import gc
import shutil
from multiprocessing import current_process
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
from pyspark.sql import SparkSession

from validate import run_with_paths

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

        _, metric_names = report["task_name"].rsplit("__", 1)
        best_model = report.get("best_model")
        m = metric_names.split("+")[0]
        if m == "mse":
            m = "neg_mean_squared_error"
        all_results = report["all_results"]
        metrics[report["task_name"]] = float(all_results[best_model][m])
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
    values_by_key = {}
    for seed, metrics in metrics_by_seed:
        for key, value in metrics.items():
            aggregated[f"{key}__validator_seed_{seed}"] = value
            values_by_key.setdefault(key, []).append(value)
    for key, values in values_by_key.items():
        aggregated[key] = float(sum(values) / len(values))
    return aggregated


def save_seed_metrics(path: Path, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])


def normalize_loader_ref(ref: str) -> str:
    return ref if "." in ref else f"data.{ref}"


def resolve_loader_ref(ref: str, train_loaders, test_loaders):
    ref = normalize_loader_ref(ref)
    section, key = ref.split(".", 1)
    if section == "data":
        return train_loaders[key], "train", key
    if section == "test_data":
        return test_loaders[key], "test", key
    raise ValueError(
        "Loader reference must start with 'data.' or 'test_data.', "
        f"got {ref!r}"
    )


def _loaders_by_mode(refs: list[str], train_loaders, test_loaders) -> dict[str, Any]:
    by_mode: dict[str, dict[str, Any]] = {"train": {}, "test": {}}
    for ref in refs:
        loader, mode, key = resolve_loader_ref(ref, train_loaders, test_loaders)
        by_mode[mode][key] = loader
    return by_mode


def _generate_embeddings(trainer, loaders_by_mode, config, output_path: Path) -> None:
    dfs = []
    for mode, loaders in loaders_by_mode.items():
        if not loaders:
            continue
        getter = ResultsGetter(config, mode)
        dfs.append(getter.df_get(loaders, trainer))
    if not dfs:
        raise ValueError("No loaders were provided for downstream embeddings")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(dfs, ignore_index=True).to_parquet(output_path, index=False)


def _single_postprocess_mode(loaders_by_mode) -> str:

    modes = [mode for mode, loaders in loaders_by_mode.items() if loaders]
    if len(modes) != 1:
        raise ValueError(
            "A postprocessed downstream parquet must reference exactly one data mode, "
            f"got {modes}."
        )
    return modes[0]


def _run_validator(
    downstream_config: dict,
    train_path: str,
    test_path: str,
    *,
    task_names: list[str] | None = None,
    seed_metrics_dir: Path | None = None,
) -> dict[str, float]:
    if task_names == []:
        return {}

    downstream_config = deepcopy(downstream_config)
    if task_names is not None:
        downstream_config["task_names"] = task_names

    validator_seeds = downstream_config.get("validator_seeds")
    if validator_seeds is None:
        reports = run_with_paths(
            downstream_config=downstream_config,
            train_path=train_path,
            test_path=test_path,
        )
        return extract_downstream_metrics(reports)

    downstream_metrics = {}
    if downstream_config.get("embedding_metrics", {}).get("enabled", False):
        geometry_config = deepcopy(downstream_config)
        geometry_config.pop("validator_seeds", None)
        geometry_config["models"] = {}
        geometry_config["task_names"] = []
        reports = run_with_paths(
            downstream_config=geometry_config,
            train_path=train_path,
            test_path=test_path,
        )
        downstream_metrics.update(extract_downstream_metrics(reports))

    metrics_by_seed = []
    for seed in validator_seeds:
        seed_metrics = run_downstream_with_seed(
            downstream_config=downstream_config,
            train_path=train_path,
            test_path=test_path,
            seed=seed,
        )
        metrics_by_seed.append((seed, seed_metrics))
        if seed_metrics_dir is not None and len(validator_seeds) > 1:
            save_seed_metrics(
                seed_metrics_dir / f"downstream_validator_seed_{seed}.csv",
                seed_metrics,
            )
    downstream_metrics.update(aggregate_seed_metrics(metrics_by_seed))
    return downstream_metrics


def run_downstream_validator(
    downstream_config: dict,
    train_path: str,
    test_path: str,
    *,
    task_names: list[str] | None = None,
    seed_metrics_dir: Path | None = None,
) -> dict[str, float]:
    return _run_validator(
        downstream_config,
        train_path,
        test_path,
        task_names=task_names,
        seed_metrics_dir=seed_metrics_dir,
    )


def run_downstream_pipeline(
    *,
    trainer,
    train_loaders,
    test_loaders,
    config,
    downstream_config,
    train_loader_refs: list[str],
    test_loader_refs: list[str] | None,
    output_dir: Path,
    cleanup: bool = True,
    task_names: list[str] | None = None,
    seed_metrics_dir: Path | None = None,
) -> dict[str, float]:
    embed_train_file = output_dir / "train"
    embed_test_file = output_dir / "test"

    try:
        spark = create_postproc_spark_session()
        try:
            train_path = prepare_downstream_file(
                trainer=trainer,
                train_loaders=train_loaders,
                test_loaders=test_loaders,
                config=config,
                loader_refs=train_loader_refs,
                output_path=embed_train_file,
                spark=spark,
            )
            if test_loader_refs:
                test_path = prepare_downstream_file(
                    trainer=trainer,
                    train_loaders=train_loaders,
                    test_loaders=test_loaders,
                    config=config,
                    loader_refs=test_loader_refs,
                    output_path=embed_test_file,
                    spark=spark,
                )
            else:
                test_path = train_path
        finally:
            spark.stop()

        if not downstream_config:
            return {}

        return run_downstream_validator(
            deepcopy(dict(downstream_config)),
            train_path=train_path,
            test_path=test_path,
            task_names=task_names,
            seed_metrics_dir=seed_metrics_dir,
        )
    finally:
        if cleanup:
            shutil.rmtree(output_dir, ignore_errors=True)


def prepare_downstream_file(
    *,
    trainer,
    train_loaders,
    test_loaders,
    config,
    loader_refs: list[str],
    output_path: Path,
    spark: SparkSession | None = None,
) -> str:
    refs = [normalize_loader_ref(ref) for ref in loader_refs]
    loaders_by_mode = _loaders_by_mode(refs, train_loaders, test_loaders)
    _generate_embeddings(trainer, loaders_by_mode, config, output_path)
    gc.collect()

    mode = _single_postprocess_mode(loaders_by_mode)
    if spark is None:
        spark = create_postproc_spark_session()
        try:
            post_processing(config, output_path, mode, spark=spark)
        finally:
            spark.stop()
    else:
        post_processing(config, output_path, mode, spark=spark)

    return str(output_path) + "_postproc"


def list_downstream_tasks(
    train_path: str | Path,
    task_names: list[str] | None = None,
) -> list[str]:
    if task_names is not None:
        return list(task_names)
    cols = pq.ParquetDataset(str(train_path)).schema.names
    return [col for col in cols if col.startswith("target__")]


def compute_downstreams(
    trainer, train_loaders, test_loaders, config, downstream_config
):
    output_dir = Path(config["log_dir"]) / config["run_name"] / "embeddings"
    return run_downstream_pipeline(
        trainer=trainer,
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        config=config,
        downstream_config=downstream_config,
        train_loader_refs=["data.gen_train", "data.gen_train_val"],
        test_loader_refs=["test_data.gen_test"],
        output_dir=output_dir,
        cleanup=downstream_config is not None,
        seed_metrics_dir=Path(config["log_dir"]) / config["run_name"],
    )
