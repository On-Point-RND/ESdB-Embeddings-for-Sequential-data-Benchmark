from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import FloatType, LongType

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

CAT_FEATURES = ["small_group"]
NUM_FEATURES = ["amount_rur"]
INDEX_COLUMNS = ["client_id", "age"]
ORDERING_COLUMNS = ["trans_date"]
TARGET_VALS = [0, 1, 2, 3]
TM = ORDERING_COLUMNS[0]


def get_anomaly_target(df: pd.DataFrame) -> pd.Series:
    def _cv_list(row):
        a = np.asarray(row["amount_rur"])
        mean = a.mean()
        std = a.std()
        assert mean != 0, "Mean of amount_rur shouldn't be zero even after shifts"
        return std / mean

    cv_list = df.apply(_cv_list, axis=1)

    all_cv = np.asarray(cv_list)
    q95 = np.nanquantile(all_cv, 0.95)
    return np.asarray(cv_list > q95, dtype=np.int32).tolist()


def reg_target_row(row, horizon=30):
    d = np.asarray(row["trans_date"])
    a = np.asarray(row["amount_rur"])
    out = []
    for s in row["shifts"]:
        s = int(s) - 1
        delta = d - d[s]
        mask = (delta > 0) & (delta < horizon)
        out.append(np.log1p(a[mask].sum()))
    return out


def get_forecast_target(row):
    t = np.asarray(row["trans_date"])
    out = []
    for s in row["shifts"]:
        mask = t == t[s - 1]
        mask[:s] = False
        out.append(np.log1p(np.sum(mask)))
    return out


def cut_data(row):
        start_idx = int(row["shift_start"])
        for col_name in row.index:
            if col_name in ["shifts", "shift_start", "shift_end", "client_id"]:
                continue
            val = row[col_name]
            if isinstance(val, (list ,np.ndarray)):
                row[col_name] = val[start_idx:]
        old_shifts = np.array(row["shifts"])
        new_shifts = old_shifts - start_idx
        row["shifts"] = new_shifts[new_shifts >= 0].tolist()
        if not row["shifts"]: 
            row["shifts"] = [0]
        return row


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
        SparkSession.builder
        .master("local[*]") # type: ignore[attr-defined]
        .appName("AGEPreprocessing")
        .config("spark.driver.memory", "12g")
        .config("spark.executor.memory", "4g")
        .config("spark.driver.maxResultSize", "0")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC")
        .getOrCreate()
    )
    df, df_kag_train = None, None

    if args.which_split == "train":
        df_kag_train = spark.read.csv(
            (args.data_path / "transactions_train.csv").as_posix(), header=True
        )
        df_kag_train = df_kag_train.select(
            F.col("client_id").cast(LongType()),
            F.col("trans_date").cast(LongType()),
            F.col("small_group").cast(LongType()),
            F.col("amount_rur").cast(FloatType()),
        )

        df_label = spark.read.csv(
            (args.data_path / "train_target.csv").as_posix(), header=True
        ).select(F.col("client_id").cast(LongType()), F.col("bins").cast(LongType()))

        df = df_kag_train.join(df_label, on="client_id")
        df = df.withColumnRenamed("bins", "age")
    else:
        raise NotImplementedError(
            "We doesn't know what to do with test.csv for AGE dataset without labels."
        )

    vcs = cat_freq(df, CAT_FEATURES)
    for vc in vcs:
        df = vc.encode(df)
        if args.cat_codes_path is not None:
            vc.write(args.cat_codes_path / vc.feature_name, mode=mode)

    df = collect_lists(df, group_by=INDEX_COLUMNS, order_by=ORDERING_COLUMNS)

    df = df.sort("client_id").toPandas()
    df = filter_short(df)

    def compute_shift_end(arr, horizon):
        arr = np.asarray(arr)
        return (arr[-1] - arr > horizon).sum() - 1 if len(arr) else -1

    horizon_days = 30
    df["shift_end"] = df[TM].map(lambda x: compute_shift_end(x, horizon_days))

    train_df, test_df = global_time_split(
        data=df,
        test_frac=time_test_split,
        min_shift_start=2,
        time_col=TM,
        seqlen_col="_seq_len",
    )

    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df["shift_end"] = train_df[TM].map(
        lambda x: compute_shift_end(x, horizon_days)
    )

    assert (train_df["shift_end"] >= train_df["shift_start"]).all()
    assert (test_df["shift_end"] >= test_df["shift_start"]).all()
    train_n_shifts, test_n_shifts = split_num_shifts(args.num_shifts, time_test_split)
    test_df = add_shift_columns(test_df, test_n_shifts, args.shift_seed)
    train_df = add_shift_columns(train_df, train_n_shifts, args.shift_seed)

    if args.ntp:
        test_df = test_df.apply(trim_test, axis=1)

    train_df, test_df = global_train_column(
        train_df, test_df, USER_TRAIN_SPLIT, args.split_seed
    )
    test_df["target__reg_amount__local__mse+r2"] = test_df.apply(reg_target_row, axis=1)
    test_df["target__age__global__accuracy+f1_macro"] = test_df["age"]
    test_df["target__forecast__local__mse+r2"] = test_df.apply(
        get_forecast_target, axis=1
    )
    test_df["target__anomaly__local__roc_auc+f1_macro+accuracy"] = get_anomaly_target(
        test_df
    )

    train_df["target__reg_amount__local__mse+r2"] = train_df.apply(
        reg_target_row, axis=1
    )
    train_df["target__age__global__accuracy+f1_macro"] = train_df["age"]
    train_df["target__forecast__local__mse+r2"] = train_df.apply(
        get_forecast_target, axis=1
    )
    train_df["target__anomaly__local__roc_auc+f1_macro+accuracy"] = get_anomaly_target(
        train_df
    )

    # get real part of test data 
    if args.ntp:
        test_df = test_df.apply(cut_data, axis = 1)

    keep_cols = [
        "client_id",
        "age",
        TM,
        "small_group",
        "amount_rur",
        "_seq_len",
        "shifts",
        'global_train',
        "target__reg_amount__local__mse+r2",
        "target__age__global__accuracy+f1_macro",
        "target__forecast__local__mse+r2",
        "target__anomaly__local__roc_auc+f1_macro+accuracy",
    ]

    save_partitioned_parquet(
        train_df[keep_cols], args.save_path / "train", 20, mode=mode
    )
    save_partitioned_parquet(test_df[keep_cols], args.save_path / "test", 20, mode=mode)


if __name__ == "__main__":
    main()
