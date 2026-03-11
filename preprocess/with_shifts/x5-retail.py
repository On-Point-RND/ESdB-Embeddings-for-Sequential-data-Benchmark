from argparse import ArgumentParser
from pathlib import Path

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import LongType, FloatType
from pyspark.ml.feature import Bucketizer
import numpy as np

from ..common import cat_freq, collect_lists
from .common_pandas import (
    add_shift_columns,
    global_time_split,
    save_partitioned_parquet,
    filter_short,
    split_num_shifts,
    global_train_column,
    trim_test,
)

CAT_FEATURES = [
    "product_id",
    "is_own_trademark",
    "is_alcohol",
    "level_1",
    "level_2",
    "level_3",
    "level_4",
    "segment_id",
]

NUM_FEATURES = [
    "purchase_sum",
    "trn_sum_from_red",
    "trn_sum_from_iss",
    "netto",
    "regular_points_received",
    "express_points_received",
    "product_quantity",
    "regular_points_spent",
    "express_points_spent",
]
INDEX_COLUMNS = ["client_id", "age"]
ORDERING_COLUMNS = ["transaction_datetime"]
TM = ORDERING_COLUMNS[0]
HORIZON = np.timedelta64(10, "D")
AGE_BOUNDS = [10.0, 35.0, 45.0, 60.0, 90.0]


def get_reg_target(row):
    a = np.asarray(row["purchase_sum"], dtype=float)
    t = np.asarray(row["transaction_datetime"], dtype="datetime64[s]")
    out = []
    for s in row["shifts"]:
        delta = t - t[s - 1]
        mask = (delta > np.timedelta64(0, "s")) & (delta < HORIZON)
        mask = mask & (a > 0)
        out.append(np.log1p(a[mask].sum()))
    return out


def get_forecast_target(row):
    t = np.asarray(row["transaction_datetime"], dtype="datetime64[s]").astype(
        "datetime64[h]"
    )
    out = []
    for s in row["shifts"]:
        assert s > 0, "shift should be more than zero"
        out.append(np.log1p(np.sum(t[s:] == t[s - 1])))
    return out


def get_anomaly_target(row):
    r = np.asarray(row["trn_sum_from_red"])
    out = []
    for s in row["shifts"]:
        assert s > 0, "shift should be more than zero"
        out.append(1 if np.sum(r[s:]) > 10 else 0)
    return out


def compute_shift_end(arr):
    arr = np.asarray(arr, dtype="datetime64[s]")
    diff = arr[-1] - arr
    return (diff > HORIZON).sum() - 1


def trim_users(arr):
    arr = np.asarray(arr, dtype="datetime64[s]")
    if len(arr) < 2:
        return True
    total_duration = arr[-1] - arr[0]
    return total_duration < HORIZON


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
        "--which-split",
        help="Whether to preprocess train set, test set or their union",
        choices=["train", "test", "union"],
        required=True,
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
        default=10,
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
        SparkSession.builder.master("local[*]")  # type: ignore[attr-defined]
        .appName("RetailPreprocessing")
        .config("spark.driver.memory", "12g")
        .config("spark.executor.memory", "4g")
        .config("spark.driver.maxResultSize", "0")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC")
        .getOrCreate()
    )

    df = None

    if args.which_split == "train":
        df_clients = spark.read.csv(
            (args.data_path / "clients.csv").as_posix(), header=True
        )
        df_clients = df_clients.select(
            F.col("client_id"),
            F.col("age").cast(LongType()),
        )
        df_clients = df_clients.filter((F.col("age") > 10) & (F.col("age") < 90))

        df_tx = spark.read.csv(
            (args.data_path / "purchases.csv").as_posix(), header=True
        )
        df_tx = df_tx.select(
            F.col("client_id"),
            F.to_timestamp(F.col("transaction_datetime"))
            .cast(LongType())
            .alias("transaction_datetime"),
            F.col("purchase_sum").cast(FloatType()),
            F.col("product_id"),
            F.col("trn_sum_from_red").cast(FloatType()),
        )
        df = df_tx.join(df_clients, on="client_id")

    else:
        raise NotImplementedError(
            "We doesn't know what to do with test.csv for Retail hero dataset without labels."
        )

    vcs = cat_freq(df, CAT_FEATURES)
    for vc in vcs:
        df = vc.encode(df)
        if args.cat_codes_path is not None:
            vc.write(args.cat_codes_path / vc.feature_name, mode=mode)

    df = collect_lists(df, group_by=INDEX_COLUMNS, order_by=ORDERING_COLUMNS)

    # split age on buckers as in CoLES
    df = (
        Bucketizer(
            splits=AGE_BOUNDS,
            inputCol="age",
            outputCol="age_clf",
            handleInvalid="error",
        )
        .transform(df)
        .withColumn("age_clf", F.col("age_clf").cast(LongType()))
        .cache()
    )

    df = df.sort("client_id").toPandas()
    df = filter_short(df)

    df["shift_end"] = df[TM].map(compute_shift_end)

    train_df, test_df = global_time_split(
        data=df,
        test_frac=time_test_split,
        time_col=TM,
        seqlen_col="_seq_len",
    )

    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df["is_bad_user"] = train_df[TM].apply(trim_users)
    bad_indices = train_df.index[train_df["is_bad_user"]].tolist()
    train_df = train_df.drop(index=bad_indices)
    del train_df["is_bad_user"]

    train_df["shift_end"] = train_df[TM].map(compute_shift_end)

    valid_mask_train = train_df.index[train_df["shift_end"] >= train_df["shift_start"]]
    train_df = train_df.loc[valid_mask_train].copy()
    valid_mask_test = test_df.index.intersection(valid_mask_train)
    test_df = test_df.loc[valid_mask_test].copy()

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

    test_df["target__clf__global__accuracy+f1_macro"] = test_df["age_clf"]
    test_df["target__reg__local__mse+r2"] = test_df.apply(get_reg_target, axis=1)
    test_df["target__forecast__local__mse+r2"] = test_df.apply(
        get_forecast_target, axis=1
    )
    test_df["target__anomaly__local__roc_auc+f1_macro+accuracy"] = test_df.apply(
        get_anomaly_target, axis=1
    )

    train_df["target__clf__global__accuracy+f1_macro"] = train_df["age_clf"]
    train_df["target__reg__local__mse+r2"] = train_df.apply(get_reg_target, axis=1)
    train_df["target__forecast__local__mse+r2"] = train_df.apply(
        get_forecast_target, axis=1
    )
    train_df["target__anomaly__local__roc_auc+f1_macro+accuracy"] = train_df.apply(
        get_anomaly_target, axis=1
    )

    keep_cols = (
        INDEX_COLUMNS
        + ORDERING_COLUMNS
        + CAT_FEATURES
        + NUM_FEATURES
        + [
            "_seq_len",
            "shifts",
            "global_train",
            "target__clf__global__accuracy+f1_macro",
            "target__anomaly__local__roc_auc+f1_macro+accuracy",
            "target__reg__local__mse+r2",
            "target__forecast__local__mse+r2",
        ]
    )

    save_partitioned_parquet(test_df[keep_cols], args.save_path / "test", 20, mode=mode)
    save_partitioned_parquet(
        train_df[keep_cols], args.save_path / "train", 20, mode=mode
    )


if __name__ == "__main__":
    main()
