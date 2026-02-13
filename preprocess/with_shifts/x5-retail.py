from argparse import ArgumentParser
from pathlib import Path
from collections import defaultdict

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import LongType, StringType, TimestampType, FloatType
from pyspark.ml.feature import Bucketizer
import numpy as np
import pandas as pd

from ..common import cat_freq, collect_lists
from .common_pandas import (
    add_shift_columns,
    add_debug_f,
    global_time_split,
    duplicate_target_by_shifts,
    save_partitioned_parquet,
    filter_short,
    shift_end_by_len,
    split_num_shifts,
)

CAT_FEATURES = [
    "client_id",
    "product_id"
]
NUM_FEATURES = [
    "purchase_sum",
    "trn_sum_from_red"
]
INDEX_COLUMNS = [
    "client_id",
    "age"
]
ORDERING_COLUMNS = [
    "transaction_datetime",
]
AGE_BOUNDS = [10.0, 35.0, 45.0, 60.0, 90.0]
TEST_FRACTION = 0.1

def get_reg_target(row, horizon=30):
    d = np.asarray(row["transaction_datetime"])
    a = np.asarray(row["purchase_sum"])
    out = []
    for s in row["shifts"]:
        s = int(s)
        delta = d - d[s]
        mask = (delta > 0) & (delta < horizon)
        out.append(np.log1p(a[mask].sum()))
    return out

def get_anomaly_target(row):
    r = np.asarray(row["trn_sum_from_red"]) 
    out = []
    for s in row["shifts"]:
        s = int(s)
        if s >= len(r):
            out.append(0)
            continue
        post_r = r[s:]
        if np.sum(post_r) > 10:
            out.append(1)
        else:
            out.append(0)
    return out
        
def get_forecast_target(row):
    t = np.asarray(row['transaction_datetime'])
    out = []
    for s in row['shifts']:
        mask = (t == t[s])
        mask[:s] = False
        out.append(np.log1p(np.sum(mask)))
    return out

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--data-path",
        help="Path CSV train user",
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
        "--overwrite",
        help='Toggle "overwrite" mode on all spark writes',
        action="store_true",
    )
    parser.add_argument(
        "--train-partitions",
        help="Number of parquet partitions for train dataset",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--test-partitions",
        help="Number of parquet partitions for test dataset",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--split-seed",
        help="Random seed for train-test split",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--num-shifts",
        help="How many shifts to sample per sequence",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--shift-seed",
        help="Random seed for shifts",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--global-split-ntp",
        help="Global split with 0.5 or 0.1 test fraction using y/n ",
        type=str,
        default='n'
    )
    args = parser.parse_args()
    mode = "overwrite" if args.overwrite else "error"

    spark = SparkSession.builder \
        .master("local[*]") \
        .appName("RetailPreprocessing") \
        .config("spark.driver.memory", "12g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.maxResultSize", "0") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC") \
        .getOrCreate()
    
    df = None

    if args.global_split_ntp == 'y':
        TEST_FRACTION = 0.5
    else:
        TEST_FRACTION = 0.1

    if args.which_split == 'train':
        df_clients = (
            spark.read.csv((args.data_path / "clients.csv").as_posix(), header=True)
        )
        df_clients = df_clients.select(
            F.col("client_id"),
            F.col("age").cast(LongType()),
        )
        df_clients = df_clients.filter(
            (F.col("age") > 10) & (F.col("age") < 90)
        )

        df_tx = (
            spark.read.csv((args.data_path / "purchases.csv").as_posix(), header=True)
        )
        df_tx = df_tx.select(
            F.col("client_id"),
            F.to_timestamp(F.col("transaction_datetime")).cast(LongType()).alias("transaction_datetime"),
            F.col("purchase_sum").cast(FloatType()),
            F.col("product_id"),
            F.col("trn_sum_from_red").cast(FloatType())
        )
    else:
        raise NotImplementedError("We doesn't know what to do with test.csv for Retail hero dataset without labels.")

    df = df_tx.join(df_clients, on="client_id")

    vcs = cat_freq(df, CAT_FEATURES)
    for vc in vcs:
        df = vc.encode(df)
        if args.cat_codes_path is not None:
            vc.write(args.cat_codes_path / vc.feature_name, mode=mode)

    df = collect_lists(
        df,
        group_by=INDEX_COLUMNS,
        order_by=ORDERING_COLUMNS,
    )

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
    df["_seq_len"] = df["transaction_datetime"].apply(len)
    df = filter_short(df)

    def compute_shift_end(arr, horizon_days):
        arr = np.asarray(arr, dtype='datetime64[s]')
        diff = arr[-1] - arr
        limit = np.timedelta64(horizon_days, 'D')
        return (diff > limit).sum() -1 if len(arr) else 0

    def trim_users(arr, horizon_days):
        arr = np.asarray(arr, dtype='datetime64[s]')
        if len(arr) < 2:
            return True
        total_duration = arr[-1] - arr[0]
        limit = np.timedelta64(horizon_days, 'D')
        return total_duration < limit

    horizon_days = 30
    df['shift_end'] = df['transaction_datetime'].apply(lambda x: compute_shift_end(x, horizon_days))

    train_df, test_df = global_time_split(
        data=df,
        test_frac=TEST_FRACTION,
        min_shift_start=2,
        time_col='transaction_datetime',
        seqlen_col='_seq_len'
    )

    train_df = train_df.copy()
    test_df = test_df.copy()

    # 90% split per users
    rng = np.random.default_rng(seed=42)
    n_train_users = int(len(train_df.index) * 0.9)
    train_indices = rng.choice(train_df.index, size=n_train_users, replace=False)
    train_df["users_in_train"] = 0
    train_df.loc[train_indices, "users_in_train"] = 1
    valid_test_indices = test_df.index.intersection(train_indices)
    test_df["users_in_train"] = 0
    test_df.loc[valid_test_indices, "users_in_train"] = 1

    train_df['is_bad_user'] = train_df['transaction_datetime'].apply(lambda x: trim_users(x, horizon_days))
    bad_indices = train_df.index[train_df['is_bad_user']].tolist()
    train_df = train_df.drop(index=bad_indices)
    del train_df['is_bad_user']

    train_df['shift_end'] = train_df['transaction_datetime'].map(lambda x: compute_shift_end(x, horizon_days))

    valid_mask_train = train_df.index[train_df["shift_end"] >= train_df["shift_start"]]

    train_df = train_df.loc[valid_mask_train].copy()

    valid_mask_test = test_df.index.intersection(valid_mask_train)
    test_df = test_df.loc[valid_mask_test].copy()

    assert (train_df["shift_end"] >= train_df["shift_start"]).all()
    assert (test_df["shift_end"] >= test_df["shift_start"]).all()

    train_n_shifts, test_n_shifts = split_num_shifts(args.num_shifts, TEST_FRACTION)

    test_df = add_shift_columns(test_df, test_n_shifts, args.shift_seed)
    train_df = add_shift_columns(train_df, train_n_shifts, args.shift_seed)

    n_train_count = train_df["transaction_datetime"].apply(len).sum()
    print(n_train_count)

    def get_test_idx(row):
        total_length = len(row["transaction_datetime"])
        if len(row["shifts"]) > 0:
            last_idx = int(row["shifts"][-1])
            return total_length - last_idx
        return 0
    
    n_test_count = test_df.apply(get_test_idx, axis=1).sum()
    total = n_train_count + n_test_count
    print(n_test_count)
    print("Train percent:", n_train_count/total)
    print("Test percent:", n_test_count/total)
    print("Test Fraction", TEST_FRACTION)

    test_df['post_target'] = duplicate_target_by_shifts(test_df, "age_clf")
    test_df['post_reg_target'] = test_df.apply(get_reg_target, axis=1)
    test_df['post_forecast_target'] = test_df.apply(get_forecast_target, axis=1)
    test_df['post_anomaly_target'] = test_df.apply(get_anomaly_target, axis=1)

    train_df['post_target'] = duplicate_target_by_shifts(train_df, "age_clf")
    train_df['post_reg_target'] = train_df.apply(get_reg_target, axis=1)
    train_df['post_forecast_target'] = train_df.apply(get_forecast_target, axis=1)
    train_df['post_anomaly_target'] = train_df.apply(get_anomaly_target, axis=1)

    test_df = add_debug_f(test_df, time_col='transaction_datetime')
    train_df = add_debug_f(train_df, time_col='transaction_datetime')

    keep_cols = [
        "client_id",
        "age",
        "transaction_datetime",
        "trn_sum_from_red",
        "product_id",
        "purchase_sum",
        "_seq_len",
        "shifts",
        "post_reg_target",
        "post_target",
        "post_forecast_target",
        "post_anomaly_target",
        "users_in_train",
        "debug_f"
    ]

    save_partitioned_parquet(test_df[keep_cols], args.save_path / "test", 20)
    save_partitioned_parquet(train_df[keep_cols], args.save_path / "train", 20)

if __name__ == "__main__":
    main()
