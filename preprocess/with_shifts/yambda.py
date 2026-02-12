from argparse import ArgumentParser
from pathlib import Path

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import LongType, StringType, TimestampType, FloatType
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

CAT_FEATURES = ["item_id", "is_organic", "event_type"]
INDEX_COLUMNS = ["client_id"]
ORDERING_COLUMNS = ["timestamp"]
TARGET_VALS = [0, 1]
TEST_FRACTION = 0.1

def get_reg_target(row, horizon=30):
    d = np.asarray(row["timestamp"])
    out = []
    for s in row["shifts"]:
        s = int(s)
        delta = d - d[s]
        mask = (delta > 0) & (delta < horizon)
        out.append(np.log1p(d[mask].sum()))
    return out

def get_ratio(row):
    events = np.asarray(row["event_type"])
    out = []     
    for s in row["shifts"]:
        s = int(s)
        post = events[s:]
        if len(post) == 0:
            out.append(0)
            continue
        n_dislike = (post == 3).sum()
        n_listens = (post == 1).sum()
        if n_listens == 0:
            out.append(0)
        else:
            out.append(float(n_dislike / n_listens))
    return out

def get_forecast_target(row):
    t = np.asarray(row['timestamp'])
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
        .appName("YambdaPreprocessing") \
        .config("spark.driver.memory", "12g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.maxResultSize", "0") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
        .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC") \
        .getOrCreate()
    df, df_kag_train = None, None

    if args.global_split_ntp == 'y':
        TEST_FRACTION = 0.5
    else:
        TEST_FRACTION = 0.1

    if args.which_split == 'train':
        df_kag_train = spark.read.parquet(
            (args.data_path / "multi_event.parquet").as_posix(), header=True
        )
        df_kag_train = df_kag_train.select(
            F.col("uid").cast(LongType()),
            F.col("timestamp").cast(LongType()),
            F.col("item_id").cast(LongType()),
            F.col("track_length_seconds").cast(LongType()),
            F.col("event_type").cast(StringType()),
            F.col("is_organic").cast(LongType())
        )

        df_kag_train = df_kag_train.withColumn(
            "event_type", 
            F.when(F.col("event_type") == "listen", 1)
            .when(F.col("event_type") == "like", 2)
            .when(F.col("event_type") == "dislike", 3)
            .when(F.col("event_type") == "undislike", 4)
            .when(F.col("event_type") == "unlike", 5)
            .otherwise(-1) 
        )

        df = df_kag_train
        df = df.withColumnRenamed("uid", "client_id")
    else:
        raise NotImplemented("We doesn't know what to do with test.parquet for YAMBDA dataset without labels.")

    vcs = cat_freq(df, CAT_FEATURES)
    for vc in vcs:
        df = vc.encode(df)
        if args.cat_codes_path is not None:
            vc.write(args.cat_codes_path / vc.feature_name, mode=mode)

    df = collect_lists(
        df,
        group_by=INDEX_COLUMNS,
        order_by=ORDERING_COLUMNS
    )

    df = df.sort("client_id").toPandas()
    df['_seq_len'] = df['timestamp'].apply(len)
    df = filter_short(df)

    def compute_shift_end(arr, horizon_days):
        arr = np.asarray(arr, dtype='datetime64[s]')
        diff = arr[-1] - arr
        limit = np.timedelta64(horizon_days, 'D')
        return (diff > limit).sum() - 1 if len(arr) else -1
    
    def trim_users(arr, horizon_days):
        arr = np.asarray(arr, dtype='datetime64[s]')
        if len(arr) < 2:
            return True
        total_duration = arr[-1] - arr[0]
        limit = np.timedelta64(horizon_days, 'D')
        return total_duration < limit

    horizon_days = 30
    df['shift_end'] = df['timestamp'].map(lambda x: compute_shift_end(x, horizon_days))

    train_df, test_df = global_time_split(
        data=df,
        test_frac=TEST_FRACTION,
        min_shift_start=2,
        time_col='timestamp',
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

    train_df['is_bad_user'] = train_df['timestamp'].apply(lambda x: trim_users(x, horizon_days))
    bad_indices = train_df.index[train_df['is_bad_user']].tolist()
    train_df = train_df.drop(index=bad_indices)
    del train_df['is_bad_user']

    train_df['shift_end'] = train_df['timestamp'].map(lambda x: compute_shift_end(x, horizon_days))

    valid_mask_train = train_df.index[train_df["shift_end"] >= train_df["shift_start"]]

    train_df = train_df.loc[valid_mask_train].copy()

    valid_mask_test = test_df.index.intersection(valid_mask_train)
    test_df = test_df.loc[valid_mask_test].copy()

    assert (train_df["shift_end"] >= train_df["shift_start"]).all()
    assert (test_df["shift_end"] >= test_df["shift_start"]).all()

    train_n_shifts, test_n_shifts = split_num_shifts(args.num_shifts, TEST_FRACTION)

    test_df = add_shift_columns(test_df, test_n_shifts, args.shift_seed)
    train_df = add_shift_columns(train_df, train_n_shifts, args.shift_seed)

    test_df['mode_is_organic'] = test_df['is_organic'].apply(lambda x: 1 if np.mean(x) > 0.5 else 0)
    train_df['mode_is_organic'] = train_df['is_organic'].apply(lambda x: 1 if np.mean(x) > 0.5 else 0)
    test_df['ratio'] = test_df.apply(get_ratio, axis=1)
    train_df['ratio'] = train_df.apply(get_ratio, axis=1)

    threshold = np.quantile(np.concatenate(train_df["ratio"].values), 0.95)

    test_df['post_target'] = duplicate_target_by_shifts(test_df, "mode_is_organic")
    test_df['post_reg_target'] = test_df.apply(get_reg_target, axis=1)
    test_df['post_forecast_target'] = test_df.apply(get_forecast_target, axis=1)
    test_df['post_anomaly_target'] = test_df["ratio"].apply(lambda x: [1 if r > threshold else 0 for r in x])

    train_df['post_target'] = duplicate_target_by_shifts(train_df, "mode_is_organic")
    train_df['post_reg_target'] = train_df.apply(get_reg_target, axis=1)
    train_df['post_forecast_target'] = train_df.apply(get_forecast_target, axis=1)
    test_df['post_anomaly_target'] = train_df["ratio"].apply(lambda x: [1 if r > threshold else 0 for r in x])

    del train_df['ratio']
    del test_df['ratio']

    test_df = add_debug_f(test_df, time_col='timestamp')
    train_df = add_debug_f(train_df, time_col='timestamp')

    keep_cols = [
        "client_id",
        "is_organic",
        "timestamp",
        "event_type",
        "item_id",
        "track_length_seconds",
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
