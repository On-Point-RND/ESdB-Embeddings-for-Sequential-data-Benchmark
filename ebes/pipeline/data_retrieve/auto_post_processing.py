#!/usr/bin/env python3
import logging
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


def trim_target_by_embedding_len(df, target_col: str, emb_col: str = "shift_emb"):
    emb_size = F.size(F.col(emb_col))
    start = F.when(emb_size == 0, F.lit(1)).otherwise(-emb_size)
    trimmed_target = F.slice(F.col(target_col), start, emb_size)
    return df.withColumn(
        target_col,
        F.when(
            F.col(target_col).isNull() | F.col(emb_col).isNull(), F.col(target_col)
        ).otherwise(trimmed_target),
    )


def post_processing(config, emb_path, data_mode, partitions=10):
    logger.info(f'Embeddings and data postprocessing in "{data_mode}"_mode has started')
    try:
        spark = (
            SparkSession.builder.appName("JoinEmbeddings")
            .config("spark.sql.legacy.parquet.nanosAsLong", "true")
            .config("spark.driver.memory", "4g")  # ← was "2g"
            .config("spark.driver.memoryOverhead", "1g")  # ← add this
            .config("spark.executor.memory", "2g")
            .getOrCreate()
        )
        write_mode = "overwrite"
        id_col = config["data"]["preprocessing"]["common_pipeline"]["index_name"]

        if data_mode == "train":
            data_path = Path(config["data"]["dataset"]["parquet_path"])
        elif data_mode == "test":
            data_path = Path(config["data"]["dataset"]["parquet_path"]).parent / "test"
        # -------------------------------------------------------------------------
        # 1. Load embeddings
        # -------------------------------------------------------------------------
        emb_df = spark.read.parquet(emb_path.as_posix())
        # embeddings expected to have "index" column
        if "base_index" not in emb_df.columns:
            print(emb_df.columns)
            raise ValueError("Embeddings parquet must contain column 'base_index'")

        emb_df = emb_df.withColumnRenamed("base_index", id_col)

        # -------------------------------------------------------------------------
        # 2. Load additional data
        # -------------------------------------------------------------------------
        data_df = spark.read.parquet(str(data_path))

        if id_col not in data_df.columns:
            raise ValueError(f"Data parquet must contain identifier column '{id_col}'")

        # -------------------------------------------------------------------------
        # 3. Join
        # -------------------------------------------------------------------------
        emb_df_renamed = emb_df.withColumnRenamed("shifts", "shifts_emb")
        joined_df = data_df.join(emb_df_renamed, on=id_col, how="left")

        # Проверка совпадения shifts
        # bad = joined_df.filter(F.col("shifts") != F.col("shifts_emb"))
        # if bad.count() > 0:
        #    bad.select(id_col, "shifts", "shifts_emb").show(truncate=False)
        #    raise ValueError("Shifts mismatch detected")

        joined_df = joined_df.drop("shifts")
        # условие плохой строки
        bad_cond = F.col("embeddings").isNull() | F.expr(  # None вместо массива
            "exists(embeddings, x -> x is null)"
        )  # хотя бы один элемент None

        joined_df_bad = joined_df.filter(bad_cond)
        joined_df_good = joined_df.filter(~bad_cond)
        logger.info(
            f"Postprocessing deleted {joined_df_bad.count()} strings (for better...)"
        )

        # -------------------------------------------------------------------------
        # 4. Save result next to embeddings parquet
        # -------------------------------------------------------------------------
        output_path = emb_path.with_name(emb_path.name + "_postproc")
        joined_df_good.coalesce(4).write.parquet(
            output_path.as_posix(), mode=write_mode
        )
    finally:
        if spark is not None:
            spark.stop()
            # Give JVM time to shut down cleanly
            import time; time.sleep(1)
