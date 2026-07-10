#!/usr/bin/env python3
import logging
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


def post_processing(
    config,
    emb_path,
    data_mode,
    spark: SparkSession | None = None,
):
    logger.info(f'Embeddings and data postprocessing in "{data_mode}"_mode has started')

    owns_spark = spark is None

    write_mode = "overwrite"
    id_col = config["data"]["preprocessing"]["common_pipeline"]["index_name"]

    if data_mode == "train":
        data_path = Path(config["data"]["dataset"]["parquet_path"])
    elif data_mode == "test":
        if "test_data" in config:
            data_path = Path(config["test_data"]["dataset"]["parquet_path"])
        else:
            data_path = Path(config["data"]["dataset"]["parquet_path"]).parent / "test"
    else:
        raise ValueError(f"Unsupported data_mode '{data_mode}'")

    emb_df = spark.read.parquet(emb_path.as_posix())
    if "base_index" not in emb_df.columns:
        print(emb_df.columns)
        raise ValueError("Embeddings parquet must contain column 'base_index'")

    emb_df = emb_df.withColumnRenamed("base_index", id_col)
    data_df = spark.read.parquet(str(data_path))

    if id_col not in data_df.columns:
        raise ValueError(f"Data parquet must contain identifier column '{id_col}'")

    emb_df_renamed = emb_df.withColumnRenamed("shifts", "shifts_emb")
    joined_df = data_df.join(emb_df_renamed, on=id_col, how="left")

    joined_df = joined_df.drop("shifts")
    bad_cond = F.col("embeddings").isNull() | F.expr(
        "exists(embeddings, x -> x is null)"
    )

    joined_df_bad = joined_df.filter(bad_cond)
    joined_df_good = joined_df.filter(~bad_cond)
    logger.info(
        f"Postprocessing deleted {joined_df_bad.count()} strings (for better...)"
    )

    output_path = emb_path.with_name(emb_path.name + "_postproc")
    joined_df_good.coalesce(4).write.parquet(output_path.as_posix(), mode=write_mode)

    if owns_spark:
        spark.stop()
