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
    add_debug_f,
    duplicate_target_by_shifts,
    filter_short,
    global_time_split,
    save_partitioned_parquet,
    shift_end_by_len,
    split_num_shifts,
)

CAT_FEATURES = ["DayOfWeek", "Open", "Promo", "StateHoliday", "SchoolHoliday"]
NUM_FEATURES = ["Sales", "Customers"]
INDEX_COLUMNS = ["Store"]
ORDERING_COLUMNS = ["Date"]
INDEX = INDEX_COLUMNS[0]
TM = ORDERING_COLUMNS[0]
TARGET_VALS = [0, 1]
TEST_FRACTION = 0.1

def reg_sums_list(row: pd.Series, horizon: int = 30) -> list[float]:
    sales = np.asarray(row["Sales"])
    out = []
    for s in row["shifts"]:
        s = int(s)
        window = sales[s : s + horizon]
        out.append(float(window.sum()) if len(window) else 0.0)
    return out


def forecast_list(row: pd.Series) -> list[float]:
    sales = np.asarray(row["Sales"])
    out = []
    for s in row["shifts"]:
        s = int(s)
        window = sales[s:]
        out.append(float(np.log1p(np.median(window))) if len(window) else 0.0)
    return out


def spike_ratio_list(row: pd.Series, horizon: int = 60, eps: float = 1e-6) -> list[float]:
    sales = np.asarray(row["Sales"])
    out = []
    for s in row["shifts"]:
        s = int(s)
        window = sales[s : s + horizon]
        if len(window) == 0:
            out.append(np.nan)
            continue
        out.append(float(window.max() / (np.median(window) + eps)))
    return out


def get_reg_target(df: pd.DataFrame, horizon: int = 30) -> pd.Series:
    sums_list = df.apply(lambda r: reg_sums_list(r, horizon=horizon), axis=1)
    flat_sums = np.concatenate([np.asarray(v) for v in sums_list if len(v)])
    mu = flat_sums.mean() if len(flat_sums) else 0.0
    sigma = flat_sums.std() if len(flat_sums) else 1.0
    return sums_list.apply(lambda v: [(x - mu) / sigma if sigma != 0 else 0.0 for x in v])


def get_anomaly_target(df: pd.DataFrame, horizon: int = 60, q: float = 0.95) -> pd.Series:
    spike_list = df.apply(lambda r: spike_ratio_list(r, horizon=horizon), axis=1)
    flat_spike = np.concatenate([np.asarray(v) for v in spike_list if len(v)])
    thr = np.nanquantile(flat_spike, q) if len(flat_spike) else np.nan
    return spike_list.apply(lambda v: [int(x > thr) if not np.isnan(x) else 0 for x in v])


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
        help="Random seed used to split the data on train and test",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--overwrite",
        help='Toggle "overwrite" mode on all spark writes',
        action="store_true",
    )
    parser.add_argument(
        "--num-shifts",
        help="How many shifts to sample per test store",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--shift-seed",
        help="Random seed for shifts",
        type=int,
        default=0,
    )
    args = parser.parse_args()
    mode = "overwrite" if args.overwrite else "error"

    spark = SparkSession.builder.master("local[32]").getOrCreate()  # type: ignore[attr-defined]

    df = (
        spark.read.csv((args.data_path / 'data/train.csv').as_posix(), header=True)
        .withColumn(
            # Переставляем dd.MM.yyyy → yyyy.MM.dd сразу в regexp
            "Date_str",
            F.regexp_replace(
                TM,
                r"(\d{2})\.(\d{2})\.(\d{4})",
                r"$3.$2.$1"  # yyyy.MM.dd
            )
        )
        .withColumn(
            # Меняем точки на дефисы
            "Date_str",
            F.when(F.col("Date_str").isNotNull(), F.regexp_replace("Date_str", r"\.", "-"))
            .otherwise(None)
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

    rows_df = collect_lists(
        df,
        group_by=INDEX_COLUMNS,
        order_by=ORDERING_COLUMNS,
    )

    for col_name in rows_df.columns:
        if col_name.endswith("_list"):
            rows_df = rows_df.withColumnRenamed(col_name, col_name.replace("_list", ""))
    full_df = rows_df

    full_df = full_df.toPandas()
    full_df['_seq_len'] = full_df[TM].apply(len)
    full_df = filter_short(full_df)

    store_info_df = pd.read_csv(args.data_path / "data/store.csv")
    store_type_map = dict(zip(store_info_df[INDEX], store_info_df["StoreType"]))
    full_df["store_type_letter"] = full_df[INDEX].map(store_type_map)
    type_codes = pd.Categorical(full_df["store_type_letter"]).codes
    assert (type_codes >= 0).all(), "Found missing store_type values; fill or drop before coding"
    full_df["type_code"] = type_codes

    # ensure datetime64 for global time split
    full_df[TM] = full_df[TM].map(lambda x: np.asarray(x, dtype="datetime64[ns]"))

    horizon_reg = 30
    horizon_anom = 60

    full_df["shift_end"] = shift_end_by_len(full_df[TM], -1 - horizon_anom)

    train_df, test_df = global_time_split(
        data=full_df,
        test_frac=TEST_FRACTION,
        min_shift_start=2,
        time_col=TM,
        seqlen_col="_seq_len",
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

    # recompute shift_end for train after trimming
    train_df["shift_end"] = shift_end_by_len(train_df[TM], -1 - horizon_anom)

    assert (train_df["shift_end"] >= train_df["shift_start"]).all()
    assert (test_df["shift_end"] >= test_df["shift_start"]).all()

    train_n_shifts, test_n_shifts = split_num_shifts(args.num_shifts, TEST_FRACTION)

    test_df = add_shift_columns(test_df, test_n_shifts, args.shift_seed)
    train_df = add_shift_columns(train_df, train_n_shifts, args.shift_seed)

    test_df["post_amount"] = get_reg_target(test_df, horizon=horizon_reg)
    test_df["post_forecast_target"] = test_df.apply(forecast_list, axis=1)

    test_df["post_anomaly_target"] = get_anomaly_target(test_df, horizon=horizon_anom, q=0.95)
    test_df["post_target"] = duplicate_target_by_shifts(test_df, "type_code")

    train_df["post_amount"] = get_reg_target(train_df, horizon=horizon_reg)
    train_df["post_forecast_target"] = train_df.apply(forecast_list, axis=1)
    train_df["post_anomaly_target"] = get_anomaly_target(train_df, horizon=horizon_anom, q=0.95)
    train_df["post_target"] = duplicate_target_by_shifts(train_df, "type_code")

    # debug: map shifts to timestamps
    test_df = add_debug_f(test_df, time_col=TM)
    train_df = add_debug_f(train_df, time_col=TM)

    keep_cols = INDEX_COLUMNS + [
        TM,
        "shifts",
        "post_amount",
        "post_target",
        "post_forecast_target",
        "post_anomaly_target",
        "users_in_train",
        "debug_f",
    ] + CAT_FEATURES + NUM_FEATURES

    save_partitioned_parquet(train_df[keep_cols], args.save_path / "train", 20)
    save_partitioned_parquet(test_df[keep_cols], args.save_path / "test", 20)

if __name__ == "__main__":
    main()
