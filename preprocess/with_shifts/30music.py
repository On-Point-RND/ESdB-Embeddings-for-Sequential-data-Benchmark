from argparse import ArgumentParser
from pathlib import Path

import pyspark.sql.functions as F
from pyspark.sql.functions import col, from_json
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import LongType, StringType, TimestampType, FloatType, StructType, StructField, ArrayType
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

CAT_FEATURES = ["track_id"]
INDEX_COLUMNS = ["client_id"]
ORDERING_COLUMNS = ["timestamp"]
TARGET_VALS = [0, 1, 2, 3]
TEST_FRACTION = 0.1
    
def get_reg_target(row, horizon=30):
    d = np.asarray(row["timestamp"])
    a = np.asarray(row["play_duration"])
    out = []
    for s in row["shifts"]:
        s = int(s)
        if s >= len(d):
            out.append(0.0)
            continue
        start = d[s]
        post_d = d[s:]
        post_a = a[s:]
        delta = post_d - start
        mask = (delta > 0) & (delta < horizon)
        total = post_a[mask].sum()
        if total > 0:
            out.append(np.log1p(total))
        else:
            out.append(0.0)
    return out

def get_diversity(row):
    d = np.asarray(row["play_duration"], dtype=float)
    out = []     
    for s in row["shifts"]:
        s = int(s)
        post = d[s:]
        if len(post) < 2:
            assert len(post) >= 2
        post_mean = np.mean(post)
        post_std = np.std(post)
        if post_mean == 0: 
            out.append(0.0)
        else:
            cv = post_std / post_mean
            out.append(cv)
    return out

def get_mean(row):
    d = np.asarray(row["play_duration"], dtype=float)
    out = []
    for s in row["shifts"]:
        s = int(s)
        post = d[s:]
        out.append(np.mean(post))
    return out

def apply_threshold(diversity_list, mean_list, th_dv, th_mean):
    if not isinstance(diversity_list, (list, np.ndarray)): diversity_list = []
    if not isinstance(mean_list, (list, np.ndarray)): mean_list = []
    out = []
    for d, m in zip(diversity_list, mean_list):
        if (d > th_dv) and (m < th_mean):
            out.append(1)
        else:
            out.append(0)
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
        .config("spark.driver.memory", "64g") \
        .config("spark.executor.memory", "16g") \
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
        playtime_schema = StructType([
            StructField("playtime", LongType(), True)
        ])
        inner_schema = StructType([
            StructField("type", StringType(), True),
            StructField("id", LongType(), True)
        ])
        full_schema = StructType([
            StructField("subjects", ArrayType(inner_schema), True),
            StructField("objects", ArrayType(inner_schema), True)
        ])

        df_raw = spark.read.option("sep", "\t").csv((args.data_path / 'events.idomaar').as_posix())

        df_parsed = df_raw.select(
            col("_c2").cast(LongType()).alias("timestamp"),
            from_json(col("_c3"), playtime_schema).alias("props"),
            from_json(col("_c4"), full_schema).alias("entities")
        )
        df_final = df_parsed.select(
            col("timestamp"),
            col("props.playtime").alias("play_duration"),
            col("entities.subjects")[0]["id"].alias("user_id"),
            col("entities.objects")[0]["id"].alias("track_id")
        )

        df = df_final.withColumnRenamed("user_id", "client_id")
    else:
        raise NotImplementedError("We doesn't know what to do with test.csv for Retail hero dataset without labels.")

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
    df["_seq_len"] = df["timestamp"].map(len)
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

    test_df['diversity'] = test_df.apply(get_diversity, axis=1)
    test_df['mean'] = test_df.apply(get_mean, axis=1)
    train_df['diversity'] = train_df.apply(get_diversity, axis=1)
    train_df['mean'] = train_df.apply(get_mean, axis=1)

    threshold_diversity = np.quantile(np.concatenate(train_df["diversity"].values), 0.95)
    threshold_mean = np.quantile(np.concatenate(train_df["mean"].values), 0.95)

    _, bins = pd.qcut(np.concatenate(train_df["diversity"].values), q=4, retbins=True, duplicates='drop')

    test_df['post_target'] = test_df["diversity"].apply(
        lambda x: np.clip(np.digitize(x, bins) - 1, 0, 3).tolist()
    )
    test_df['post_reg_target'] = test_df.apply(get_reg_target, axis=1)
    test_df['post_forecast_target'] = test_df.apply(get_forecast_target, axis=1)
    test_df['post_anomaly_target'] = test_df.apply(
        lambda x: apply_threshold(x["diversity"], x["mean"], threshold_diversity, threshold_mean), 
        axis=1 
    )

    train_df['post_target'] = train_df["diversity"].apply(
        lambda x: np.clip(np.digitize(x, bins) - 1, 0, 3).tolist()
    )
    train_df['post_reg_target'] = train_df.apply(get_reg_target, axis=1)
    train_df['post_forecast_target'] = train_df.apply(get_forecast_target, axis=1)
    train_df['post_anomaly_target'] = train_df.apply(
        lambda x: apply_threshold(x["diversity"], x["mean"], threshold_diversity, threshold_mean), 
        axis=1
    )

    test_df = add_debug_f(test_df, time_col='timestamp')
    train_df = add_debug_f(train_df, time_col='timestamp')

    keep_cols = [
        "client_id",
        "track_id",
        "timestamp",
        "play_duration",
        "_seq_len",
        "shifts",
        "post_reg_target",
        "post_target",
        "post_forecast_target",
        "post_anomaly_target",
        "users_in_train",
        "debug_f"
    ]

    save_partitioned_parquet(test_df[keep_cols], args.save_path / "test", 20, mode=mode)
    save_partitioned_parquet(train_df[keep_cols], args.save_path / "train", 20, mode=mode)

if __name__ == "__main__":
    main()