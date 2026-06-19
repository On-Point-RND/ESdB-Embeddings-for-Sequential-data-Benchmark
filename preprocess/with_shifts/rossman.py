from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import LongType, FloatType

from ..common import cat_freq, collect_lists
from .common_pandas import (
    add_shift_columns,
    global_time_split,
    save_partitioned_parquet,
    filter_short,
    shift_end_by_len,
    split_num_shifts,
    global_train_column,
    transform_train_test_features,
    trim_test,
)

CAT_FEATURES = ["DayOfWeek", "Open", "Promo", "StateHoliday", "SchoolHoliday"]
NUM_FEATURES = ["Sales", "Customers"]
INDEX_COLUMNS = ["Store"]
ORDERING_COLUMNS = ["Date"]
TM = ORDERING_COLUMNS[0]
LOG_FEATURES: list[str] = []
RESCALE_FEATURES = [x for x in NUM_FEATURES if x not in LOG_FEATURES] + [TM]
DATETIME_LOC = "2013-01-01"
DATETIME_SCALE = (30, "D")
INDEX = INDEX_COLUMNS[0]
HORIZON = 60


def get_reg_target(df: pd.DataFrame) -> pd.Series:
    sums_list = df.apply(lambda r: reg_sums_list(r), axis=1)
    flat_sums = np.concatenate([np.asarray(v) for v in sums_list if len(v)])
    mu = flat_sums.mean() if len(flat_sums) else 0.0
    sigma = flat_sums.std() if len(flat_sums) else 1.0
    return sums_list.apply(
        lambda v: [(x - mu) / sigma if sigma != 0 else 0.0 for x in v]
    )


def reg_sums_list(row: pd.Series) -> list[float]:
    sales = np.asarray(row["Sales"])
    out = []
    for s in row["shifts"]:
        out.append(float(sales[s : s + HORIZON].sum()))
    return out


def get_forecast_target(row):
    seq = np.asarray(row["Sales"])
    out = []
    for s in row["shifts"]:
        base = seq[s]
        out.append(float(base))
    return out


def spike_ratio_list(row: pd.Series, eps: float = 1e-6) -> list[float]:
    sales = np.asarray(row["Sales"])
    out = []
    for s in row["shifts"]:
        s = int(s)
        window = sales[s : s + HORIZON]
        if len(window) == 0:
            out.append(np.nan)
            continue
        out.append(float(window.max() / (np.median(window) + eps)))
    return out


def get_anomaly_target(df: pd.DataFrame, q: float = 0.95) -> pd.Series:
    spike_list = df.apply(lambda r: spike_ratio_list(r), axis=1)
    flat_spike = np.concatenate([np.asarray(v) for v in spike_list if len(v)])
    thr = np.nanquantile(flat_spike, q) if len(flat_spike) else np.nan
    return spike_list.apply(
        lambda v: [int(x > thr) if not np.isnan(x) else 0 for x in v]
    )


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
        default=100,
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
    df = (
        spark.read.csv((args.data_path / "train.csv").as_posix(), header=True)
        .withColumn(
            # Переставляем dd.MM.yyyy → yyyy.MM.dd сразу в regexp
            "Date_str",
            F.regexp_replace(
                TM, r"(\d{2})\.(\d{2})\.(\d{4})", r"$3.$2.$1"  # yyyy.MM.dd
            ),
        )
        .withColumn(
            # Меняем точки на дефисы
            "Date_str",
            F.when(
                F.col("Date_str").isNotNull(), F.regexp_replace("Date_str", r"\.", "-")
            ).otherwise(None),
        )
        .withColumn(TM, F.to_timestamp("Date_str", "yyyy-MM-dd"))
        .select(
            F.col(INDEX).cast(LongType()),
            F.col("DayOfWeek").cast(LongType()),
            F.col("Open").cast(LongType()),
            F.col("Promo").cast(LongType()),
            F.col("StateHoliday").cast(LongType()),
            F.col("SchoolHoliday").cast(LongType()),
            F.col(TM),
            F.col("Sales").cast(FloatType()),
            F.col("Customers").cast(FloatType()),
        )
    )
    vcs = cat_freq(df, CAT_FEATURES)
    for vc in vcs:
        df = vc.encode(df)
        if args.cat_codes_path is not None:
            vc.write(args.cat_codes_path / vc.feature_name, mode=mode)

    df = collect_lists(df, group_by=INDEX_COLUMNS, order_by=TM)

    for col_name in df.columns:
        if col_name.endswith("_list"):
            df = df.withColumnRenamed(col_name, col_name.replace("_list", ""))
    full_df = df

    full_df = full_df.toPandas()
    full_df["_seq_len"] = full_df[TM].apply(len)
    full_df = filter_short(full_df)

    store_info_df = pd.read_csv(args.data_path / "store.csv")
    store_type_map = dict(zip(store_info_df[INDEX], store_info_df["StoreType"]))
    full_df["store_type_letter"] = full_df[INDEX].map(store_type_map)
    type_codes = pd.Categorical(full_df["store_type_letter"]).codes
    assert (
        type_codes >= 0
    ).all(), "Found missing store_type values; fill or drop before coding"
    full_df["type_code"] = type_codes

    # ensure datetime64 for global time split
    full_df[TM] = full_df[TM].map(lambda x: np.asarray(x, dtype="datetime64[ns]"))

    full_df["shift_end"] = shift_end_by_len(full_df[TM], -1 - HORIZON)

    train_df, test_df = global_time_split(
        data=full_df,
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

    train_df["shift_end"] = shift_end_by_len(train_df[TM], -1 - HORIZON)

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

    test_df["target__reg_amount__local__r2"] = get_reg_target(test_df)
    test_df["target__forecast__local__r2"] = test_df.apply(get_forecast_target, axis=1)

    test_df["target__anomaly__local__roc_auc"] = get_anomaly_target(
        test_df, q=0.95
    )
    test_df["target__clf__global__accuracy+f1_macro"] = test_df["type_code"]
    train_df["target__reg_amount__local__r2"] = get_reg_target(train_df)
    train_df["target__forecast__local__r2"] = train_df.apply(
        get_forecast_target, axis=1
    )
    train_df["target__anomaly__local__roc_auc"] = get_anomaly_target(
        train_df, q=0.95
    )
    train_df["target__clf__global__accuracy+f1_macro"] = train_df["type_code"]

    train_df, test_df = transform_train_test_features(
        train_df=train_df,
        test_df=test_df,
        rescale_features=RESCALE_FEATURES,
        log_features=LOG_FEATURES,
        time_col=TM,
        datetime_loc=DATETIME_LOC,
        datetime_scale=DATETIME_SCALE,
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
            "target__anomaly__local__roc_auc",
            "target__reg_amount__local__r2",
            "target__forecast__local__r2",
        ]
    )

    save_partitioned_parquet(
        train_df[keep_cols], args.save_path / "train", 20, mode=mode
    )
    save_partitioned_parquet(test_df[keep_cols], args.save_path / "test", 20, mode=mode)


if __name__ == "__main__":
    main()
