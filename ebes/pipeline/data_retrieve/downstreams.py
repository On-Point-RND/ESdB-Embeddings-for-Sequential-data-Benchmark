import logging
from numbers import Number
import shutil
from pathlib import Path

from pyspark.sql import SparkSession

from validate import run_with_paths

from ..data_retrieve.auto_post_processing import post_processing
from ..data_retrieve.embeddings_gen import ResultsGetter

logger = logging.getLogger(__name__)


def create_postproc_spark_session() -> SparkSession:
    return (
        SparkSession.builder.appName("JoinEmbeddings") # type: ignore
        .config("spark.sql.legacy.parquet.nanosAsLong", "true")
        .config("spark.driver.memory", "4g")
        .config("spark.driver.memoryOverhead", "1g")
        .config("spark.executor.memory", "2g")
        .getOrCreate()
    )


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

        for model_name, model_results in all_results.items():
            for metric_name, value in model_results.items():
                if metric_name in {
                    "main_metric",
                    "predictions",
                    "model",
                    "cv_results",
                }:
                    continue
                if not isinstance(value, Number):
                    continue
                key = f"{report['task_name']}__{model_name}__{metric_name}"
                metrics[key] = float(value)
    return metrics


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
        reports = run_with_paths(
            downstream_config=downstream_config,
            train_path=str(embed_train_file) + "_postproc",
            test_path=str(embed_test_file) + "_postproc",
        )
        downstream_metrics = extract_downstream_metrics(reports)
        shutil.rmtree(Path(config["log_dir"]) / config["run_name"] / "embeddings")
    return downstream_metrics
