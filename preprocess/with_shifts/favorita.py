from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import TimestampType

from ..common import cat_freq
from .common_pandas import (
    add_shift_columns,
    global_time_split,
    save_partitioned_parquet,
    filter_short,
    shift_end_by_len,
    split_num_shifts,
    global_train_column,
    trim_test,
    transform_train_test_features,
)


CAT_FEATURES = ["class_id"]
INDEX_COLUMNS = ["store_nbr"]
ORDERING_COLUMNS = ["date"]
TM = ORDERING_COLUMNS[0]
DATETIME_LOC = "2013-01-01"
DATETIME_SCALE = (30, "D")


def reg_target_row(
    row: pd.Series, sales_cols: list[str], horizon: int = 30
) -> list[float]:
    d = np.asarray(row[TM])
    out = []
    for s in row["shifts"]:
        s = int(s) - 1
        delta = (d - d[s]) / np.timedelta64(1, "D")
        mask = (delta > 0) & (delta < horizon)
        total = 0.0
        for c in sales_cols:
            arr = np.asarray(row[c])
            if len(arr) == 0:
                continue
            assert len(arr) == len(d), "sales and date arrays must have the same length"
            total += arr[mask].sum()
        out.append(float(np.log1p(total)))
    return out


def get_forecast_target(row: pd.Series, sales_cols: list[str]) -> list[float]:
    out = []
    for s in row["shifts"]:
        s = int(s)
        total = 0.0
        for c in sales_cols:
            arr = row[c]
            if len(arr) > s:
                total += float(arr[s])
        out.append(total)
    return out


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--data-path",
        help="Path to directory containing CSV files",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--save-path",
        help="Where to save preprocessed parquets",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--cat-codes-path",
        help="Path where to save codes for categorical features",
        type=Path,
    )
    parser.add_argument(
        "--split-seed",
        help="Random seed for train-test split",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--overwrite",
        help='Toggle "overwrite" mode on all spark writes',
        action="store_true",
    )
    parser.add_argument(
        "--num-shifts",
        help="How many shifts to sample per sequence",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--shift-seed",
        help="Random seed for shifts",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--ntp",
        help="Whether to use splitting for NTP",
        action="store_true",
    )
    args = parser.parse_args()
    mode = "overwrite" if args.overwrite else "error"

    if args.ntp:
        TIME_TRAIN_SPLIT = 0.5
    else:
        TIME_TRAIN_SPLIT = 0.9
    USER_TRAIN_SPLIT = 0.9

    if not (0.0 < TIME_TRAIN_SPLIT < 1.0):
        parser.error("time_train_split must be in range (0, 1)")
    if not (0.0 < USER_TRAIN_SPLIT < 1.0):
        parser.error("user_train_split must be in range (0, 1)")
    time_test_split = 1 - TIME_TRAIN_SPLIT

    spark = (
        SparkSession.builder.master("local[32]")  # type: ignore[attr-defined]
        .config("spark.driver.memory", "50g")
        .config("spark.driver.maxResultSize", "0")
        .config("spark.sql.shuffle.partitions", 1000)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.memory.fraction", "0.8")
        .config("spark.memory.storageFraction", "0.3")
        .config(
            "spark.driver.extraJavaOptions",
            "-XX:+UseG1GC "
            "-XX:InitiatingHeapOccupancyPercent=35 "
            "-XX:+ExplicitGCInvokesConcurrent"
            "-Xss16m",
        )
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    data_dir = args.data_path.as_posix()

    df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(f"{data_dir}/train.csv")
        .select(
            F.col("store_nbr").cast("int"),
            F.col("item_nbr").cast("int"),
            F.col("date").cast(TimestampType()),
            F.col("unit_sales").cast("double"),
        )
    )

    cls_df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(f"{data_dir}/items.csv")
        .select(
            F.col("item_nbr").cast("int"),
            F.col("class").cast("int").alias("class_id"),
        )
    )

    df_cls = (
        df.join(cls_df, on="item_nbr", how="inner")
        .groupBy("store_nbr", "class_id", "date")
        .agg(F.sum("unit_sales").alias("unit_sales"))
    )
    vcs = cat_freq(df_cls, CAT_FEATURES)
    for vc in vcs:
        df_cls = vc.encode(df_cls)
        if args.cat_codes_path is not None:
            vc.write(args.cat_codes_path / vc.feature_name, mode=mode)

    # ---------- DIMENSIONS ----------
    stores = df_cls.select("store_nbr").distinct()
    classes = df_cls.select("class_id").distinct()
    dates = df_cls.select("date").distinct()

    # ---------- GRID ----------
    full_grid = stores.crossJoin(classes).crossJoin(dates).repartition("store_nbr")

    # ---------- JOIN + ZERO FILL ----------
    df_full = full_grid.join(
        df_cls, on=["store_nbr", "class_id", "date"], how="left"
    ).withColumn("unit_sales", F.coalesce(F.col("unit_sales"), F.lit(0.0)))

    # ---------- TS COLLECT ----------
    sales_ts = (
        df_full.groupBy("store_nbr", "class_id")
        .agg(F.sort_array(F.collect_list(F.struct("date", "unit_sales"))).alias("tmp"))
        .select(
            "store_nbr",
            "class_id",
            F.expr("transform(tmp, x -> x.unit_sales)").alias("sales_ts"),
        )
    )

    # ---------- PIVOT ----------
    pivot_df = sales_ts.groupBy("store_nbr").pivot("class_id").agg(F.first("sales_ts"))

    for c in pivot_df.columns:
        if c != "store_nbr":
            pivot_df = pivot_df.withColumnRenamed(c, f"class_{c}_sales")

        # ---------- MATERIALIZE PIVOT ----------
    # переписываем pivot в parquet с большим числом партиций
    pivot_df.repartition(50).write.mode("overwrite").parquet("/tmp/pivot_cached")

    # читаем обратно — теперь lineage короткий, а партиций 50
    pivot_df = spark.read.parquet("/tmp/pivot_cached")

    row = dates.orderBy("date").agg(F.collect_list("date").alias("date")).first()
    assert row is not None, "dates is empty"
    date_array = row["date"]

    full_df = pivot_df.withColumn("date", F.lit(date_array))
    full_df = full_df.repartition("store_nbr").cache()
    full_df.count()

    full_df.repartition(50).write.parquet("/tmp/fav_cached", mode="overwrite")

    df = pd.read_parquet("/tmp/fav_cached")
    df["_seq_len"] = df[TM].apply(len)
    print(df["_seq_len"])
    df = filter_short(df)
    stores_df = pd.read_csv(args.data_path / "stores.csv")

    df = df.merge(
        stores_df,
        on="store_nbr",
        how="left",
    )

    df[TM] = df[TM].map(lambda x: np.asarray(x, dtype="datetime64[ns]"))

    sales_cols = [c for c in df.columns if c.endswith("_sales")]
    horizon = 30
    df["shift_end"] = shift_end_by_len(df[TM], -1 - horizon)

    type_codes = pd.Categorical(df["type"]).codes
    df["type_code"] = type_codes
    assert (
        df["type_code"] >= 0
    ).all(), "Found missing type values; fill or drop before coding"

    train_df, test_df = global_time_split(
        data=df,
        test_frac=time_test_split,
        min_shift_start=2,
        time_col=TM,
        seqlen_col="_seq_len",
    )

    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df["shift_end"] = shift_end_by_len(train_df[TM], -1 - horizon)

    assert (train_df["shift_end"] >= train_df["shift_start"]).all()
    assert (test_df["shift_end"] >= test_df["shift_start"]).all()
    train_n_shifts, test_n_shifts = split_num_shifts(args.num_shifts, time_test_split)
    test_df = add_shift_columns(test_df, test_n_shifts, args.shift_seed)
    train_df = add_shift_columns(train_df, train_n_shifts, args.shift_seed)

    if args.ntp:
        test_df = test_df.apply(trim_test, axis=1)
        test_df["_seq_len"] = test_df[TM].apply(len)

    train_df, test_df = global_train_column(
        train_df, test_df, USER_TRAIN_SPLIT, args.split_seed
    )

    anomaly_cities = stores_df.city.value_counts()[
        stores_df.city.value_counts() == 1
    ].index.tolist()

    test_df["target__store_type__global__accuracy+f1_macro"] = test_df["type_code"]
    test_df["target__anomaly__global__roc_auc+f1_macro+accuracy"] = test_df.city.apply(
        lambda x: int(x in anomaly_cities)
    )
    test_df["target__reg_amount__local__r2"] = test_df.apply(
        lambda r: reg_target_row(r, sales_cols, horizon=horizon), axis=1
    )
    test_df["target__forecast__local__r2"] = test_df.apply(
        lambda r: get_forecast_target(r, sales_cols), axis=1
    )
    train_df["target__store_type__global__accuracy+f1_macro"] = train_df["type_code"]
    train_df["target__anomaly__global__roc_auc+f1_macro+accuracy"] = (
        train_df.city.apply(lambda x: int(x in anomaly_cities))
    )
    train_df["target__reg_amount__local__r2"] = train_df.apply(
        lambda r: reg_target_row(r, sales_cols, horizon=horizon), axis=1
    )
    train_df["target__forecast__local__r2"] = train_df.apply(
        lambda r: get_forecast_target(r, sales_cols), axis=1
    )

    train_df, test_df = transform_train_test_features(
        train_df=train_df,
        test_df=test_df,
        time_col=TM,
        datetime_loc=DATETIME_LOC,
        datetime_scale=DATETIME_SCALE,
    )

    keep_cols = (
        INDEX_COLUMNS
        + ORDERING_COLUMNS
        + [
            "shifts",
            "global_train",
            "_seq_len",
            "target__store_type__global__accuracy+f1_macro",
            "target__anomaly__global__roc_auc+f1_macro+accuracy",
            "target__reg_amount__local__r2",
            "target__forecast__local__r2",
        ]
        + sales_cols
    )

    save_partitioned_parquet(
        train_df[keep_cols], args.save_path / "train", 20, mode=mode
    )
    save_partitioned_parquet(test_df[keep_cols], args.save_path / "test", 20, mode=mode)


if __name__ == "__main__":
    main()
