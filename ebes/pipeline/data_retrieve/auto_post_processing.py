#!/usr/bin/env python3
import logging
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)

def trim_target_by_embedding_len(df, target_col: str, emb_col: str = "shift_emb"):
    trimmed_target = F.expr(
        f"slice(`{target_col}`, -size(`{emb_col}`), size(`{emb_col}`))"
    )
    return df.withColumn(
        target_col,
        F.when(F.col(target_col).isNull() | F.col(emb_col).isNull(), F.col(target_col))
        .otherwise(trimmed_target),
    )

# Transformer len cutter cause this target's trimming
def remove_skipped_target(
    df,
    emb_col: str = "shift_emb",
) :
    target_cols = [
        col
        for col in df.columns
        if col.startswith("target_") and "__local__" in col
    ]

    for target_col in target_cols:
        df = trim_target_by_embedding_len(df, target_col=target_col, emb_col=emb_col)

    return df


def post_processing(config, emb_path, data_mode, partitions=10, transformer_skipped_target_removal=False):
    logger.info(f"Embeddings and data postprocessing in \"{data_mode}\"_mode has started")
    spark = (
        SparkSession.builder
        .appName("JoinEmbeddings")
        .config("spark.driver.memory", "2g")
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
        raise ValueError(
            f"Data parquet must contain identifier column '{id_col}'"
        )

    # -------------------------------------------------------------------------
    # 3. Join
    # -------------------------------------------------------------------------
    emb_df_renamed = emb_df.withColumnRenamed("shifts", "shifts_emb")
    joined_df = data_df.join(
        emb_df_renamed,
        on=id_col,
        how="left"
    )

    # Проверка совпадения shifts
    # bad = joined_df.filter(F.col("shifts") != F.col("shifts_emb"))
    # if bad.count() > 0:
    #    bad.select(id_col, "shifts", "shifts_emb").show(truncate=False)
    #    raise ValueError("Shifts mismatch detected")

    joined_df = joined_df.drop("shifts")

    if transformer_skipped_target_removal:
        logger.info('Removing skipped targets caused by Transformer sequence trimming')
        joined_df = remove_skipped_target(joined_df)

    # -------------------------------------------------------------------------
    # 4. Save result next to embeddings parquet
    # -------------------------------------------------------------------------
    output_path = emb_path.with_name(emb_path.name + "_postproc")

    (
        joined_df
        .repartition(partitions)
        .write
        .parquet(output_path.as_posix(), mode=write_mode)
    )
