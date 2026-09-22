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

    if config.get("embedding_generation", {}).get("legacy", False):
        section = "data" if data_mode == "train" else "test_data"
        max_len = config[section]["preprocessing"]["gen_pipeline"]["max_seq_len"]
        cutoff = F.greatest(F.col("_seq_len") - F.lit(max_len), F.lit(0))
        retained_len = F.least(F.col("_seq_len"), F.lit(max_len))
        # Keep original array positions to select the corresponding targets.
        positions = F.transform(
            F.col("shifts"),
            lambda shift, i: F.struct(shift.alias("shift"), i.alias("position")),
        )
        positions = F.array_sort(F.filter(positions, lambda point: point.shift >= cutoff))
        joined_df = joined_df.withColumn("_legacy_positions", positions)
        expected_shifts = F.transform(
            F.col("_legacy_positions"),
            lambda point: F.least(F.greatest(point.shift - cutoff, F.lit(0)), retained_len),
        )
        local_targets = [name for name in data_df.columns
                         if name.startswith("target__") and "__local__" in name]
        invalid = ~expected_shifts.eqNullSafe(F.col("shifts_emb"))
        invalid = invalid | (F.size("shift_emb") != F.size("_legacy_positions"))
        for name in local_targets:
            invalid = invalid | F.col(name).isNull() | (F.size(name) != F.size("shifts"))
        if joined_df.filter(F.col("embeddings").isNotNull() & invalid).limit(1).count():
            raise ValueError("Legacy shifts/targets do not match saved embeddings")
        for name in local_targets:
            joined_df = joined_df.withColumn(
                name, F.transform(
                    F.col("_legacy_positions"),
                    lambda point: F.element_at(F.col(name), point.position + 1),
                ),
            )
        joined_df = joined_df.drop("_legacy_positions")

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
