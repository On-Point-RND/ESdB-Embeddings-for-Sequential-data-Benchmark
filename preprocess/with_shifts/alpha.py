from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import LongType, FloatType
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
    "mcc_category",
    "currency",
    "operation_kind",
    "operation_type",
    "operation_type_group",
    "ecommerce_flag",
    "payment_system",
    "income_flag",
    "mcc",
    "country",
    "city",
]

NUM_FEATURES = ["amnt", "day_of_week", "hour", "weekofyear"]
INDEX_COLUMNS = ["client_id"]
ORDERING_COLUMNS = ["time_from_first_trn"]
TM = ORDERING_COLUMNS[0]
HORIZON = 30 * 24


def reg_target_row(row):
    a = np.asarray(row["amnt"], dtype=float)
    t = np.asarray(row["time_from_first_trn"])
    out = []
    for s in row["shifts"]:
        assert s > 0, "shift should be more than zero"
        delta = t - t[s - 1]
        mask = (delta > 0) & (delta < HORIZON)
        out.append(np.log1p(a[mask].sum()))
    return out


def get_forecast_target(row):
    t = np.asarray(row["time_from_first_trn"])
    out = []
    for s in row["shifts"]:
        assert s > 0, "shift should be more than zero"
        out.append(np.log1p(np.sum(t[s:] == t[s - 1])))
    return out


def hours_since_first_tx(hour_diff):
    arr = np.asarray(hour_diff, dtype=np.int64)
    return np.cumsum(np.maximum(arr, 0)).astype("float32")


def compute_shift_end(arr):
    arr = np.asarray(arr)
    return (arr[-1] - arr > HORIZON).sum() - 1 if len(arr) else -1


def trim_users(arr):
    arr = np.asarray(arr)
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
        choices=["train", "test"],
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

    if args.ntp and args.which_split != "train":
        parser.error("NTP mode supports only --which-split train.")

    spark = (
        SparkSession.builder.master("local[20]")  # type: ignore[attr-defined]
        .appName("AlphaPreprocessing")
        .config("spark.driver.memory", "100g")
        .config("spark.executor.memory", "50g")
        .config("spark.driver.maxResultSize", "80g")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "50000")
        .config(
            "spark.executor.extraJavaOptions",
            "-XX:+UseG1GC -XX:+UseStringDeduplication",
        )
        .getOrCreate()
    )
    transactions_dir = (
        "alfabattle2_train_transactions_contest/train_transactions_contest"
        if args.which_split == "train"
        else "alfabattle2_test_transactions_contest/test_transactions_contest"
    )
    df = spark.read.parquet(
        (args.data_path / transactions_dir / "*.parquet").as_posix(),
        header=True,
    )
    df = df.select(
        F.col("app_id").cast(LongType()),
        F.col("amnt").cast(FloatType()),
        F.col("mcc_category").cast(LongType()),
        F.col("currency").cast(LongType()),
        F.col("operation_kind").cast(LongType()),
        F.col("operation_type").cast(LongType()),
        F.col("operation_type_group").cast(LongType()),
        F.col("ecommerce_flag").cast(LongType()),
        F.col("payment_system").cast(LongType()),
        F.col("income_flag").cast(LongType()),
        F.col("mcc").cast(LongType()),
        F.col("country").cast(LongType()),
        F.col("city").cast(LongType()),
        F.col("days_before").cast(LongType()),
        F.col("day_of_week").cast(LongType()),
        F.col("hour").cast(LongType()),
        F.col("weekofyear").cast(LongType()),
        F.col("transaction_number").cast(LongType()),
        F.col("hour_diff").cast(LongType()),
    )

    if args.which_split == "train":
        df_target = spark.read.csv(
            (args.data_path / "alfabattle2_train_target.csv").as_posix(), header=True
        ).select(
            F.col("app_id").cast(LongType()),
            F.col("product").cast(LongType()),
            F.col("flag").cast(FloatType()),
        )
    else:
        df_target_base = spark.read.csv(
            (args.data_path / "alfabattle2_test_target_contest.csv").as_posix(),
            header=True,
        ).select(
            F.col("app_id").cast(LongType()),
            F.col("product").cast(LongType()),
        )
        df_target_sample = spark.read.csv(
            (args.data_path / "alfabattle2_alpha_sample.csv").as_posix(), header=True
        ).select(
            F.col("app_id").cast(LongType()),
            F.col("flag").cast(FloatType()),
        )
        df_target = df_target_base.join(
            df_target_sample, on="app_id", how="left"
        ).fillna({"flag": 0.0})

    df = df.join(df_target, on="app_id")
    df = df.withColumnRenamed("app_id", "client_id")
    vcs = cat_freq(df, CAT_FEATURES)
    for vc in vcs:
        df = vc.encode(df)
        if args.cat_codes_path is not None:
            vc.write(args.cat_codes_path / vc.feature_name, mode=mode)

    df = collect_lists(df, group_by=INDEX_COLUMNS, order_by="transaction_number")

    df.repartition(50).write.parquet("/tmp/alpha_cached", mode="overwrite")

    df = pd.read_parquet("/tmp/alpha_cached")

    df = df.sort_values("client_id").reset_index(drop=True)
    df[TM] = df["hour_diff"].map(hours_since_first_tx)
    df = filter_short(df)
    df["shift_end"] = df[TM].map(compute_shift_end)

    train_n_shifts, test_n_shifts = split_num_shifts(args.num_shifts, time_test_split)
    train_df, test_df = None, None

    if args.ntp:
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

        valid_mask_train = train_df.index[
            train_df["shift_end"] >= train_df["shift_start"]
        ]
        train_df = train_df.loc[valid_mask_train].copy()
        valid_mask_test = test_df.index.intersection(valid_mask_train)
        test_df = test_df.loc[valid_mask_test].copy()

        assert (train_df["shift_end"] >= train_df["shift_start"]).all()
        assert (test_df["shift_end"] >= test_df["shift_start"]).all()

        test_df = add_shift_columns(test_df, test_n_shifts, args.shift_seed)
        train_df = add_shift_columns(train_df, train_n_shifts, args.shift_seed)

        test_df = test_df.apply(trim_test, axis=1)
        test_df["_seq_len"] = test_df[TM].apply(len)

        train_df, test_df = global_train_column(
            train_df, test_df, USER_TRAIN_SPLIT, args.split_seed
        )

        test_df["target__prod__global__accuracy+f1_macro"] = test_df["product"]
        test_df["target__reg_amount__local__mse+r2"] = test_df.apply(
            reg_target_row, axis=1
        )
        test_df["target__forecast__local__mse+r2"] = test_df.apply(
            get_forecast_target, axis=1
        )
        test_df["target__anomaly__global__roc_auc+f1_macro+accuracy"] = test_df["flag"]

        train_df["target__prod__global__accuracy+f1_macro"] = train_df["product"]
        train_df["target__reg_amount__local__mse+r2"] = train_df.apply(
            reg_target_row, axis=1
        )
        train_df["target__forecast__local__mse+r2"] = train_df.apply(
            get_forecast_target, axis=1
        )
        train_df["target__anomaly__global__roc_auc+f1_macro+accuracy"] = train_df[
            "flag"
        ]
    else:
        train_df = df.copy()
        train_df["shift_start"] = 2
        train_df["is_bad_user"] = train_df[TM].apply(trim_users)
        bad_indices = train_df.index[train_df["is_bad_user"]].tolist()
        train_df = train_df.drop(index=bad_indices)
        del train_df["is_bad_user"]

        train_df["shift_end"] = train_df[TM].map(compute_shift_end)
        valid_mask_train = train_df.index[
            train_df["shift_end"] >= train_df["shift_start"]
        ]
        train_df = train_df.loc[valid_mask_train].copy()
        assert (train_df["shift_end"] >= train_df["shift_start"]).all()

        num_shifts = train_n_shifts if args.which_split == "train" else test_n_shifts
        train_df = add_shift_columns(train_df, num_shifts, args.shift_seed)
        train_df["global_train"] = 1 if args.which_split == "train" else 0

        train_df["target__prod__global__accuracy+f1_macro"] = train_df["product"]
        train_df["target__reg_amount__local__mse+r2"] = train_df.apply(
            reg_target_row, axis=1
        )
        train_df["target__forecast__local__mse+r2"] = train_df.apply(
            get_forecast_target, axis=1
        )
        train_df["target__anomaly__local__roc_auc+f1_macro+accuracy"] = train_df["flag"]

    keep_cols = (
        INDEX_COLUMNS
        + ORDERING_COLUMNS
        + CAT_FEATURES
        + NUM_FEATURES
        + [
            "_seq_len",
            "shifts",
            "global_train",
            "target__prod__global__accuracy+f1_macro",
            "target__anomaly__local__roc_auc+f1_macro+accuracy",
            "target__reg_amount__local__mse+r2",
            "target__forecast__local__mse+r2",
        ]
    )

    if test_df is not None:
        save_partitioned_parquet(
            test_df[keep_cols], args.save_path / "test", 20, mode=mode
        )
        save_partitioned_parquet(
            train_df[keep_cols], args.save_path / "train", 20, mode=mode
        )
    else:
        out_split = "train" if args.which_split == "train" else "test"
        save_partitioned_parquet(
            train_df[keep_cols], args.save_path / out_split, 20, mode=mode
        )


if __name__ == "__main__":
    main()
