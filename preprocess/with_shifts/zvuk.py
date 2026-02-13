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

CAT_FEATURES = ["track_id"]
INDEX_COLUMNS = ["client_id"]
ORDERING_COLUMNS = ["datetime"]
TARGET_VALS = [0, 1]
TEST_FRACTION = 0.1

def get_clf_target(row):
    c = np.asarray(row["cluster_id"])
    out = []
    for s in row["shifts"]:
        s = int(s)
        post_c = c[s+1:]
        if len(post_c) == 0:
            out.append(-1)
            continue
        vals, counts = np.unique(post_c, return_counts=True)
        idx_max_counts = np.argmax(counts)
        mode_cluster = vals[idx_max_counts]
        out.append(int(mode_cluster))
    return out

def get_reg_target(row, horizon=30):
    t = np.asarray(row["datetime"], dtype='datetime64[s]')
    p = np.asarray(row["play_duration"], dtype=float)
    out = []
    horizon_td = np.timedelta64(horizon, 'D')
    zero_td = np.timedelta64(0, 's')
    for s in row["shifts"]:
        s = int(s)
        if s + 1 >= len(t):
            out.append(0.0)
            continue
        post_t = t[s+1:]
        post_p = p[s+1:]
        start = t[s]
        deltas = post_t - start
        mask = (deltas >= zero_td) & (deltas < horizon_td)
        val_sum = np.nansum(post_p[mask])
        if val_sum < 0: val_sum = 0.0
        out.append(np.log1p(val_sum))
    return out

def get_diversity(row):
    d = np.asarray(row["play_duration"], dtype=float)
    out = []     
    for s in row["shifts"]:
        s = int(s)
        post = d[s+1:]
        if len(post) < 2:
            out.append(0)
            continue
        post_mean = np.mean(post)
        post_std = np.std(post)
        if post_mean == 0: 
            out.append(0.0)
        else:
            cv = post_std / post_mean
            out.append(cv)
    return out

def apply_threshold(ratio, threshold):
    return [1 if r > threshold else 0 for r in ratio]

def get_forecast_target(row):
    t = np.asarray(row['datetime'])
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
        .master("local[20]") \
        .appName("ZvukPreprocessing") \
        .config("spark.driver.memory", "100g") \
        .config("spark.executor.memory", "50g") \
        .config("spark.driver.maxResultSize", "80g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "50000") \
        .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC -XX:+UseStringDeduplication") \
        .getOrCreate()
    
    df = None

    if args.global_split_ntp == 'y':
        TEST_FRACTION = 0.5
    else:
        TEST_FRACTION = 0.1

    if args.which_split == 'train':

        df_interactions = (
            spark.read.parquet((args.data_path / "zvuk-interactions.parquet").as_posix(),  header=True)
        )
        df_interactions = df_interactions.select(
            F.col("user_id").cast(LongType()),
            F.col("track_id").cast(LongType()),
            F.col("play_duration").cast(FloatType()),
            F.col("datetime").cast(TimestampType()).cast(LongType()).alias("datetime")
        )

        df_interactions = df_interactions.withColumnRenamed("user_id", "client_id")

        df_interactions = df_interactions.withColumn(
            "client_id", 
            F.abs(F.hash(F.col("client_id"))).cast(LongType())
        )

        df_artist = (
            spark.read.parquet((args.data_path / "zvuk-track_artist_embedding.parquet").as_posix(),  header=True)
        )
        df_artist = df_artist.select(
            F.col("track_id").cast(LongType()),
            F.col("cluster_id").cast(FloatType())
        )
    else:
        raise NotImplementedError("We doesn't know what to do with test.csv for Zvuk hero dataset without labels.")

    df = df_interactions.join(df_artist, on="track_id")

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

    df = df.toPandas()
    df = df.sort_values("client_id")
    df["_seq_len"] = df["datetime"].map(len)
    df = filter_short(df)

    df["datetime"] = df["datetime"].map(lambda x: np.asarray(x).astype("datetime64[s]"))

    def compute_shift_end(arr, horizon_days):
        arr = np.asarray(arr, dtype='datetime64[ms]')
        diff = arr[-1] - arr
        limit = np.timedelta64(horizon_days, 'D')
        return (diff > limit).sum() - 1 if len(arr) else -1
    
    def trim_users(arr, horizon_days):
        arr = np.asarray(arr, dtype='datetime64[ms]')
        if len(arr) < 2:
            return True
        total_duration = arr[-1] - arr[0]
        limit = np.timedelta64(horizon_days, 'D')
        return total_duration < limit

    horizon_days = 30
    df['shift_end'] = df['datetime'].map(lambda x: compute_shift_end(x, horizon_days))

    train_df, test_df = global_time_split(
        data=df,
        test_frac=TEST_FRACTION,
        min_shift_start=2,
        time_col='datetime',
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

    train_df['is_bad_user'] = train_df['datetime'].apply(lambda x: trim_users(x, horizon_days))
    bad_indices = train_df.index[train_df['is_bad_user']].tolist()
    train_df = train_df.drop(index=bad_indices)
    del train_df['is_bad_user']

    train_df['shift_end'] = train_df['datetime'].map(lambda x: compute_shift_end(x, horizon_days))

    valid_mask_train = train_df.index[train_df["shift_end"] >= train_df["shift_start"]]

    train_df = train_df.loc[valid_mask_train].copy()

    valid_mask_test = test_df.index.intersection(valid_mask_train)
    test_df = test_df.loc[valid_mask_test].copy()

    assert (train_df["shift_end"] >= train_df["shift_start"]).all()
    assert (test_df["shift_end"] >= test_df["shift_start"]).all()

    train_n_shifts, test_n_shifts = split_num_shifts(args.num_shifts, TEST_FRACTION)

    test_df = add_shift_columns(test_df, test_n_shifts, args.shift_seed)
    train_df = add_shift_columns(train_df, train_n_shifts, args.shift_seed)

    train_df['cv'] = train_df.apply(get_diversity, axis=1)
    test_df['cv'] = test_df.apply(get_diversity, axis=1)

    threshold = np.quantile(np.concatenate(train_df["cv"].values), 0.95) 

    test_df['post_target'] = test_df.apply(get_clf_target, axis=1)
    test_df['post_reg_target'] = test_df.apply(get_reg_target, axis=1)
    test_df['post_forecast_target'] = test_df.apply(get_forecast_target, axis=1)
    test_df['post_anomaly_target'] = test_df["cv"].apply(lambda x: apply_threshold(x, threshold))

    train_df['post_target'] = train_df.apply(get_clf_target, axis=1)
    train_df['post_reg_target'] = train_df.apply(get_reg_target, axis=1)
    train_df['post_forecast_target'] = train_df.apply(get_forecast_target, axis=1)
    train_df['post_anomaly_target'] = train_df["cv"].apply(lambda x: apply_threshold(x, threshold))

    del test_df["cv"]
    del train_df["cv"]

    test_df = add_debug_f(test_df, time_col='datetime')
    train_df = add_debug_f(train_df, time_col='datetime')

    keep_cols = [
        "client_id",
        "datetime",
        "cluster_id",
        "track_id",
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

    save_partitioned_parquet(test_df[keep_cols], args.save_path / "test", 20)
    save_partitioned_parquet(train_df[keep_cols], args.save_path / "train", 20)

if __name__ == "__main__":
    main()
