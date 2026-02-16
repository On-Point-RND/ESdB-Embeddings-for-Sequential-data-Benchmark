from argparse import ArgumentParser
from pathlib import Path

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import LongType
import numpy as np

from ..common import cat_freq, collect_lists
from .common_pandas import (
    add_shift_columns,
    add_debug_f,
    global_time_split,
    save_partitioned_parquet,
    filter_short,
    split_num_shifts,
)

CAT_FEATURES = ["item_id", "behavior_type", "item_category"]
INDEX_COLUMNS = ["client_id"]
ORDERING_COLUMNS = ["time"]
TM = ORDERING_COLUMNS[0]


def get_reg_target(row, horizon_hours=300):
    t = np.array(row["time"])
    horizon_seconds = np.timedelta64(horizon_hours * 3600, "s")
    out = []
    for s in row["shifts"]:
        s = int(s)
        start_time = t[s - 1]
        future_times = t[s:]
        hours_from_shift = future_times - start_time
        count_events = np.sum(hours_from_shift <= horizon_seconds)
        out.append(np.log1p(count_events))
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
    out = []
    for s in row["shifts"]:
        s = int(s)
        post_b = b[s:]
        post_i = i[s:]
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
                        cart_between = any(
                            c_idx < cart_idx < p_idx for cart_idx in add_idx
                        )

                        if not cart_between:
                            shift_has_anomaly = 1
                            break
                if shift_has_anomaly == 1:
                    break

        out.append(shift_has_anomaly)

    return out


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
        "--time-train-split",
        help="Train fraction for global time split from 0 to 1",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--user-train-split",
        help="User train split from 0 to 1",
        type=float,
        default=0.9,
    )
    args = parser.parse_args()
    mode = "overwrite" if args.overwrite else "error"

    if not (0.0 < args.time_train_split < 1.0):
        parser.error("--time-train-split must be in range (0, 1)")
    if not (0.0 < args.user_train_split < 1.0):
        parser.error("--user-train-split must be in range (0, 1)")
    time_test_split = 1 - args.time_train_split
    user_train_split = args.user_train_split

    spark = (
        SparkSession.builder.master("local[*]")
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

    horizon_hours = 300
    df["shift_end"] = df[TM].map(lambda x: compute_shift_end(x, horizon_hours))

    train_df, test_df = global_time_split(
        data=df,
        test_frac=time_test_split,
        min_shift_start=2,
        time_col="time",
        seqlen_col="_seq_len",
    )

    train_df = train_df.copy()
    test_df = test_df.copy()

    # User split with configurable train fraction.
    rng = np.random.default_rng(seed=args.split_seed)
    n_train_users = int(len(train_df.index) * user_train_split)
    train_indices = rng.choice(
        train_df.index, size=n_train_users, replace=False
    ).tolist()
    train_df["users_in_train"] = 0
    train_df.loc[train_indices, "users_in_train"] = 1
    valid_test_indices = test_df.index.intersection(train_indices)
    test_df["users_in_train"] = 0
    test_df.loc[valid_test_indices, "users_in_train"] = 1

    train_df["is_bad_user"] = train_df["time"].apply(
        lambda x: trim_users(x, horizon_hours)
    )
    bad_indices = train_df.index[train_df["is_bad_user"]].tolist()
    train_df = train_df.drop(index=bad_indices)
    del train_df["is_bad_user"]

    train_df["shift_end"] = train_df["time"].map(
        lambda x: compute_shift_end(x, horizon_hours)
    )

    valid_mask_train = train_df.index[train_df["shift_end"] >= train_df["shift_start"]]

    train_df = train_df.loc[valid_mask_train].copy()

    valid_mask_test = test_df.index.intersection(valid_mask_train)
    test_df = test_df.loc[valid_mask_test].copy()

    train_n_shifts, test_n_shifts = split_num_shifts(args.num_shifts, time_test_split)

    test_df = add_shift_columns(test_df, test_n_shifts, args.shift_seed)
    train_df = add_shift_columns(train_df, train_n_shifts, args.shift_seed)

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
    test_df["target__anomaly__local__roc_auc+f1_macro+accuracy"] = test_df.apply(
        get_anomaly_target, axis=1
    )

    train_df["target__clf__local__accuracy+f1_macro"] = train_df["post_ratio_target"]
    train_df["target__reg__local__mse+r2"] = train_df.apply(get_reg_target, axis=1)
    train_df["target__forecast__local__mse+r2"] = train_df.apply(
        get_forecast_target, axis=1
    )
    train_df["target__anomaly__local__roc_auc+f1_macro+accuracy"] = train_df.apply(
        get_anomaly_target, axis=1
    )

    # get real part of test data 
    if args.time_train_split == 0.5:   
        test_df = test_df.apply(cut_data, axis = 1)

    test_df = add_debug_f(test_df, time_col=TM)
    train_df = add_debug_f(train_df, time_col=TM)

    keep_cols = INDEX_COLUMNS + ORDERING_COLUMNS + CAT_FEATURES + [
        "_seq_len",
        "shifts",
        "target__clf__local__accuracy+f1_macro",
        "target__reg__local__mse+r2",
        "target__forecast__local__mse+r2",
        "target__anomaly__local__roc_auc+f1_macro+accuracy",
        "users_in_train",
        "debug_f",
    ]

    save_partitioned_parquet(test_df[keep_cols], args.save_path / "test", 20, mode=mode)
    save_partitioned_parquet(train_df[keep_cols], args.save_path / "train", 20, mode=mode)


if __name__ == "__main__":
    main()
