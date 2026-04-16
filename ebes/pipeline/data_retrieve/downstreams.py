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
    # Assign unique ports per worker process to avoid conflicts when multiple
    # seeds run in parallel. Uses the same process-name pattern as _run_with_seed.
    proc_name = current_process().name
    idx_str = proc_name.split("-")[-1]
    idx = int(idx_str) if idx_str.isdigit() else 0
    port_base = 34000 + idx * 20  # e.g. worker-1 → 34020, worker-2 → 34040

    return (
        SparkSession.builder.appName(f"JoinEmbeddings_{proc_name}") # type: ignore
        .config("spark.sql.legacy.parquet.nanosAsLong", "true")
        .config("spark.driver.memory", "2g")
        .config("spark.driver.memoryOverhead", "512m")
        .config("spark.executor.memory", "1g")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.port", str(port_base))
        .config("spark.blockManager.port", str(port_base + 1))
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
