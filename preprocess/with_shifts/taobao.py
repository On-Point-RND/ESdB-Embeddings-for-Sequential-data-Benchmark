from argparse import ArgumentParser
from pathlib import Path

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import LongType
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

CAT_FEATURES = ["behavior_type", "item_category"]
INDEX_COLUMNS = ["client_id"]
ORDERING_COLUMNS = ["time"]
TM = ORDERING_COLUMNS[0]
HORIZON = 48


def get_reg_target(row, horizon_hours=300):
    horizon = np.timedelta64(horizon_hours * 3600, "s")
    t = np.array(row["time"])
    out = []
    for s in row["shifts"]:
        s = int(s)
        delta = t - t[s - 1]
        mask = (delta > np.timedelta64(0, "s")) & (delta < horizon)
        out.append(np.log1p(np.sum(mask)))
    return out


def get_forecast_target(row):
    t = np.asarray(row["time"])
    out = []
    for s in row["shifts"]:
        mask = t == t[s - 1]
        mask[:s] = False
        out.append(np.log1p(np.sum(mask)))
    return out


def get_anomaly_target(row):
    b = np.asarray(row["behavior_type"])
    i = np.asarray(row["item_id"])
    s = row["shift_start"]
    out = []
    post_b = b[:]
    post_i = i[:]
    item_history = {}
    for item, action in zip(post_i, post_b):
        if item not in item_history:
            item_history[item] = []
        item_history[item].append(action)

    shift_has_anomaly = 0
    for item, history in item_history.items():
        if shift_has_anomaly == 1:
            break

        collect_idx = [idx for idx, x in enumerate(history) if x == 2]
        add_idx = [idx for idx, x in enumerate(history) if x == 3]
        purchase_idx = [idx for idx, x in enumerate(history) if x == 4]
        if not collect_idx or not purchase_idx:
            continue

        for c_idx in collect_idx:
            for p_idx in purchase_idx:
                if p_idx > c_idx:
                    cart_between = any(c_idx < cart_idx < p_idx for cart_idx in add_idx)

                    if not cart_between:
                        shift_has_anomaly = 1
                        break
            if shift_has_anomaly == 1:
                break

    return shift_has_anomaly


def compute_shift_end(arr, horizon_hours):
    horizon = np.timedelta64(horizon_hours, "h")
    return (arr[-1] - arr > horizon).sum() - 1


def trim_users(arr, horizon_hours):
    if len(arr) < 2:
        return True
    total_duration = arr[-1] - arr[0]
    limit = np.timedelta64(horizon_hours, "h")
    return total_duration < limit


def get_ratio_raw(row):
    b = np.asarray(row["behavior_type"])
    shifts = row["shifts"]
    out = []
    for s in shifts:
        s = int(s)
        future_b = b[s:]
        n_views = np.sum(future_b == 1)
        n_buys = np.sum(future_b == 4)
        if n_views == 0:
            out.append(0.0)
        else:
            out.append(n_buys / n_views)
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
        .appName("TaobaoPreprocessing")
        .config("spark.driver.memory", "12g")
        .config("spark.executor.memory", "4g")
        .config("spark.driver.maxResultSize", "0")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC")
        .getOrCreate()
    )
    df = None

    if args.which_split == "train":
        df = spark.read.csv(
            (args.data_path / "tianchi_mobile_recommend_train_user.csv").as_posix(),
            header=True,
        )

        df = df.select(
            F.col("user_id").cast(LongType()),
            F.col("item_id").cast(LongType()),
            F.col("behavior_type").cast(LongType()),
            F.col("item_category").cast(LongType()),
            F.to_timestamp(F.col("time"), "yyyy-MM-dd HH")
            .cast(LongType())
            .alias("time"),
        )
        df = df.withColumnRenamed("user_id", "client_id")
    else:
        raise NotImplementedError(
            "We doesn't know what to do with test.csv for Taobao dataset without labels."
        )

    vcs = cat_freq(df, CAT_FEATURES)
    for vc in vcs:
        df = vc.encode(df)
        if args.cat_codes_path is not None:
            vc.write(args.cat_codes_path / vc.feature_name, mode=mode)

    df = collect_lists(df, group_by=INDEX_COLUMNS, order_by=ORDERING_COLUMNS)

    df = df.sort("client_id").toPandas()
    df[TM] = df[TM].map(lambda x: np.asarray(x, dtype="datetime64[s]"))
    df = filter_short(df)

    df["shift_end"] = df[TM].map(lambda x: compute_shift_end(x, HORIZON))

    train_df, test_df = global_time_split(
        data=df,
        test_frac=time_test_split,
        time_col=TM,
        seqlen_col="_seq_len",
    )

    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df["is_bad_user"] = train_df[TM].apply(lambda x: trim_users(x, HORIZON))
    bad_indices = train_df.index[train_df["is_bad_user"]].tolist()
    train_df = train_df.drop(index=bad_indices)
    del train_df["is_bad_user"]

    train_df["shift_end"] = train_df[TM].map(lambda x: compute_shift_end(x, HORIZON))

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

    train_df, test_df = global_train_column(
        train_df, test_df, USER_TRAIN_SPLIT, args.split_seed
    )

    train_ratios = train_df.apply(get_ratio_raw, axis=1)
    test_ratios = test_df.apply(get_ratio_raw, axis=1)
    all_ratio = np.asarray(train_ratios.explode().dropna(), dtype=np.float64)
    assert (
        all_ratio.size > 0
    ), "No ratio values in train after filtering; cannot compute quantiles."

    q = np.quantile(all_ratio, [0.25, 0.5, 0.75])

    def discretize_ratio(ratios_list):
        return np.searchsorted(
            q, np.asarray(ratios_list, dtype=np.float64), side="left"
        ).tolist()

    train_df["post_ratio_target"] = train_ratios.map(discretize_ratio)
    test_df["post_ratio_target"] = test_ratios.map(discretize_ratio)

    test_df["target__clf__local__accuracy+f1_macro"] = test_df["post_ratio_target"]
    test_df["target__reg__local__mse+r2"] = test_df.apply(get_reg_target, axis=1)
    test_df["target__forecast__local__mse+r2"] = test_df.apply(
        get_forecast_target, axis=1
    )
    test_df["target__anomaly__global__roc_auc+f1_macro+accuracy"] = test_df.apply(
        get_anomaly_target, axis=1
    )

    train_df["target__clf__local__accuracy+f1_macro"] = train_df["post_ratio_target"]
    train_df["target__reg__local__mse+r2"] = train_df.apply(get_reg_target, axis=1)
    train_df["target__forecast__local__mse+r2"] = train_df.apply(
        get_forecast_target, axis=1
    )
    train_df["target__anomaly__global__roc_auc+f1_macro+accuracy"] = train_df.apply(
        get_anomaly_target, axis=1
    )

    keep_cols = (
        INDEX_COLUMNS
        + ORDERING_COLUMNS
        + CAT_FEATURES
        + [
            "_seq_len",
            "shifts",
            "global_train",
            "target__clf__local__accuracy+f1_macro",
            "target__anomaly__global__roc_auc+f1_macro+accuracy",
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
