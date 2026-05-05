import csv
import logging
from copy import deepcopy
import shutil
from pathlib import Path

from py4j.protocol import Py4JError
from pyspark.sql import SparkSession

from validate import run_with_paths

from ..data_retrieve.auto_post_processing import post_processing
from ..data_retrieve.embeddings_gen import ResultsGetter

logger = logging.getLogger(__name__)


def _reset_pyspark_gateway() -> None:
    from pyspark.context import SparkContext

    SparkContext._active_spark_context = None
    SparkContext._gateway = None
    SparkContext._jvm = None


def create_postproc_spark_session() -> SparkSession:
    builder = (
        SparkSession.builder.appName("JoinEmbeddings")  # type: ignore
        .config("spark.sql.legacy.parquet.nanosAsLong", "true")
        .config("spark.driver.memory", "4g")
        .config("spark.driver.memoryOverhead", "1g")
        .config("spark.executor.memory", "2g")
    )
    try:
        return builder.getOrCreate()
    except Py4JError:
        logger.warning("PySpark gateway is stale. Restarting Spark session.")
        _reset_pyspark_gateway()
        return builder.getOrCreate()


def extract_downstream_metrics(reports) -> dict[str, float]:
    metrics = {}
    for report in reports:
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
    trainer, train_loaders, test_loaders, config, downstream_config
):
    train_embeddings_getter = ResultsGetter(config, "train")
    keys = {"gen_train", "gen_train_val"}
    subloaders = {k: train_loaders[k] for k in keys if k in train_loaders}
    df_train = train_embeddings_getter.df_get(subloaders, trainer)
    embed_train_file = (
        Path(config["log_dir"]) / config["run_name"] / "embeddings" / "train"
    )
    embed_train_file.parent.mkdir(parents=True, exist_ok=True)
    df_train.to_parquet(embed_train_file, index=False)

    test_embeddings_getter = ResultsGetter(config, "test")
    keys = {"gen_test"}
    subloaders = {k: test_loaders[k] for k in keys if k in test_loaders}
    df_test = test_embeddings_getter.df_get(subloaders, trainer)
    embed_test_file = (
        Path(config["log_dir"]) / config["run_name"] / "embeddings" / "test"
    )
    embed_test_file.parent.mkdir(parents=True, exist_ok=True)
    df_test.to_parquet(embed_test_file, index=False)

    spark = create_postproc_spark_session()
    try:
        post_processing(config, embed_train_file, "train", spark=spark)
        post_processing(config, embed_test_file, "test", spark=spark)
    finally:
        spark.stop()

    downstream_metrics = {}
    if downstream_config:
        validator_seeds = downstream_config.get("validator_seeds")
        if validator_seeds is None:
            reports = run_with_paths(
                downstream_config=downstream_config,
                train_path=str(embed_train_file) + "_postproc",
                test_path=str(embed_test_file) + "_postproc",
            )
            downstream_metrics = extract_downstream_metrics(reports)
            shutil.rmtree(Path(config["log_dir"]) / config["run_name"] / "embeddings")
            return downstream_metrics

        metrics_by_seed = []
        run_dir = Path(config["log_dir"]) / config["run_name"]
        for seed in validator_seeds:
            seed_metrics = run_downstream_with_seed(
                downstream_config=downstream_config,
                train_path=str(embed_train_file) + "_postproc",
                test_path=str(embed_test_file) + "_postproc",
                seed=seed,
            )
            metrics_by_seed.append((seed, seed_metrics))
            if len(validator_seeds) > 1:
                save_seed_metrics(
                    run_dir / f"downstream_validator_seed_{seed}.csv",
                    seed_metrics,
                )
        downstream_metrics = aggregate_seed_metrics(metrics_by_seed)
        shutil.rmtree(Path(config["log_dir"]) / config["run_name"] / "embeddings")
    return downstream_metrics
