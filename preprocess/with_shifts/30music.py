from argparse import ArgumentParser
from pathlib import Path

from pyspark.sql.functions import col, from_json
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    LongType,
    StringType,
    StructType,
    StructField,
    ArrayType,
)
import numpy as np
import pandas as pd

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

CAT_FEATURES = ["track_id"]
NUM_FEATURES = ["play_duration"]
INDEX_COLUMNS = ["client_id"]
ORDERING_COLUMNS = ["timestamp"]
TM = ORDERING_COLUMNS[0]
HORIZON = np.timedelta64(3, "D")


def get_reg_target(row):
    a = np.asarray(row["play_duration"], dtype=float)
    t = np.asarray(row["timestamp"], dtype="datetime64[s]")
    out = []
    for s in row["shifts"]:
        assert s > 0, "shift should be more than zero"
        delta = t - t[s - 1]
        mask = (delta > np.timedelta64(0, "s")) & (delta < HORIZON)
        mask = mask & (a != -1)
        out.append(np.log1p(a[mask].sum()))
    return out


def get_forecast_target(row):
    t = np.asarray(row["timestamp"], dtype="datetime64[s]").astype("datetime64[D]")
    out = []
    for s in row["shifts"]:
        assert s > 0, "shift should be more than zero"
        mask = t == t[s - 1]
        mask[:s] = False
        out.append(np.log1p(np.sum(mask)))
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
    if not isinstance(diversity_list, (list, np.ndarray)):
        diversity_list = []
    if not isinstance(mean_list, (list, np.ndarray)):
        mean_list = []
    out = []
    for d, m in zip(diversity_list, mean_list):
        if (d > th_dv) and (m < th_mean):
            out.append(1)
        else:
            out.append(0)
    return out


def compute_shift_end(arr):
    arr = np.asarray(arr, dtype="datetime64[s]").astype("datetime64[D]")
    diff = arr[-1] - arr
    return (diff > HORIZON).sum() - 1 if len(arr) else -1


def trim_users(arr):
    arr = np.asarray(arr, dtype="datetime64[s]").astype("datetime64[D]")
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
        .config("spark.driver.memory", "64g")
        .config("spark.executor.memory", "16g")
        .config("spark.driver.maxResultSize", "0")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC")
        .getOrCreate()
    )
    df = None

    if args.which_split == "train":
        playtime_schema = StructType([StructField("playtime", LongType(), True)])
        inner_schema = StructType(
            [
                StructField("type", StringType(), True),
                StructField("id", LongType(), True),
            ]
        )
        full_schema = StructType(
            [
                StructField("subjects", ArrayType(inner_schema), True),
                StructField("objects", ArrayType(inner_schema), True),
            ]
        )

        df_raw = (
            spark.read.option("sep", "\t")
            .csv((args.data_path / "events.idomaar").as_posix())
        )

        df_parsed = df_raw.select(
            col("_c2").cast(LongType()).alias("timestamp"),
            from_json(col("_c3"), playtime_schema).alias("props"),
            from_json(col("_c4"), full_schema).alias("entities"),
        )
        df_final = df_parsed.select(
            col("timestamp"),
            col("props.playtime").alias("play_duration"),
            col("entities.subjects")[0]["id"].alias("user_id"),
            col("entities.objects")[0]["id"].alias("track_id"),
        )

        df = df_final.withColumnRenamed("user_id", "client_id")
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
        test_df['_seq_len'] = test_df[TM].apply(len)

    train_df, test_df = global_train_column(
        train_df, test_df, USER_TRAIN_SPLIT, args.split_seed
    )

    test_df["diversity"] = test_df.apply(get_diversity, axis=1)
    test_df["mean"] = test_df.apply(get_mean, axis=1)
    train_df["diversity"] = train_df.apply(get_diversity, axis=1)
    train_df["mean"] = train_df.apply(get_mean, axis=1)

    threshold_diversity = np.quantile(
        np.concatenate(train_df["diversity"].values), 0.95
    )
    threshold_mean = np.quantile(np.concatenate(train_df["mean"].values), 0.95)

    _, bins = pd.qcut(
        np.concatenate(train_df["diversity"].values),
        q=4,
        retbins=True,
        duplicates="drop",
    )
    test_df["target__clf__local__accuracy+f1_macro"] = test_df["diversity"].apply(
        lambda x: np.clip(np.digitize(x, bins) - 1, 0, 3)
    )
    test_df["target__reg__local__mse+r2"] = test_df.apply(get_reg_target, axis=1)
    test_df["target__forecast__local__mse+r2"] = test_df.apply(
        get_forecast_target, axis=1
    )
    test_df["target__anomaly__local__roc_auc+f1_macro+accuracy"] = test_df.apply(
        lambda x: apply_threshold(
            x["diversity"], x["mean"], threshold_diversity, threshold_mean
        ),
        axis=1,
    )

    train_df["target__clf__local__accuracy+f1_macro"] = train_df["diversity"].apply(
        lambda x: np.clip(np.digitize(x, bins) - 1, 0, 3)
    )
    train_df["target__reg__local__mse+r2"] = train_df.apply(get_reg_target, axis=1)
    train_df["target__forecast__local__mse+r2"] = train_df.apply(
        get_forecast_target, axis=1
    )
    train_df["target__anomaly__local__roc_auc+f1_macro+accuracy"] = train_df.apply(
        lambda x: apply_threshold(
            x["diversity"], x["mean"], threshold_diversity, threshold_mean
        ),
        axis=1,
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
            "target__clf__local__accuracy+f1_macro",
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
