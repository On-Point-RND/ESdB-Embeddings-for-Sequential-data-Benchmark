import logging
import shutil
from multiprocessing import current_process
from pathlib import Path

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
        SparkSession.builder.appName(f"JoinEmbeddings_{proc_name}") # type: ignore
        .config("spark.sql.legacy.parquet.nanosAsLong", "true")
        .config("spark.driver.memory", "4g")
        .config("spark.driver.memoryOverhead", "1g")
        .config("spark.executor.memory", "4g")
        .config("spark.sql.parquet.enableVectorizedReader", "false")
        .getOrCreate()
    )


def extract_downstream_metrics(reports) -> dict[str, float]:
    metrics = {}
    for report in reports:
        task_name, metric_names = report["task_name"].rsplit("__", 1)
        task_name = task_name.replace("target__", "")
        best_model = report.get("best_model")
        m = metric_names.split("+")[0]
        if m == "mse":
            m = "neg_mean_squared_error"
        metrics[report["task_name"]] = float(report["all_results"][best_model][m])
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
