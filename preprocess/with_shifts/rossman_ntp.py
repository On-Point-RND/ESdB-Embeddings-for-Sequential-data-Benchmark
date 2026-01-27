from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import LongType, FloatType

from common import cat_freq, collect_lists
from .common_pandas import (
    add_shift_columns,
    duplicate_target_by_shifts,
    fill_train_targets,
    pandas_train_test_split,
    save_partitioned_parquet,
)

CAT_FEATURES = ["DayOfWeek", "Open", "Promo", "StateHoliday", "SchoolHoliday"]
NUM_FEATURES = ["Sales", "Customers"]
INDEX_COLUMNS = ["Store"]
ORDERING_COLUMNS = ["Date"]
INDEX = INDEX_COLUMNS[0]
TM = ORDERING_COLUMNS[0]
TARGET_VALS = [0, 1]
TEST_FRACTION = 0.5

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
        default=0,
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
        spark.read.csv(args.data_path.as_posix(), header=True)
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

    ###########################
    full_df = full_df.withColumn(
        "used_in_train",
        F.when(F.col(TM).isNotNull() & (F.size(TM) < 400),
               F.lit(-1)
               ).otherwise(F.lit(0))
    )
    # Здесь удалили все не подходящие начисто
    full_df = full_df.filter(F.col("used_in_train") != -1)
    MAX_LEN = 300  # или любое другое
    full_df = full_df.withColumn("_seq_len", F.lit(MAX_LEN))
    full_df = full_df.withColumn(
        "mock_target",
        (F.rand() < 0.5).cast("int")
    )

    full_df = full_df.toPandas()
    store_info_df = pd.read_csv(args.data_path / "store.csv")
    store_type_map = dict(zip(store_info_df[INDEX], store_info_df["StoreType"]))
    full_df["store_type_letter"] = full_df[INDEX].map(store_type_map)
    type_codes = pd.Categorical(full_df["store_type_letter"]).codes
    assert (type_codes >= 0).all(), "Found missing store_type values; fill or drop before coding"
    full_df["type_code"] = type_codes

    stratify_col, stratify_col_vals = None, None
    stratify_col = "mock_target"
    stratify_col_vals = TARGET_VALS

    train_df, test_df = pandas_train_test_split(
        df=full_df,
        test_frac=TEST_FRACTION,
        index_col=INDEX,
        stratify_col=stratify_col,
        stratify_col_vals=stratify_col_vals,
        random_seed=args.split_seed
    )

    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df["used_in_train"] = 1
    test_df["used_in_train"] = 0

    shift_start = MAX_LEN
    horizon_reg = 30
    horizon_anom = 60
    test_df = add_shift_columns(
        test_df,
        shift_start=shift_start,
        shift_end=test_df[TM].map(len) - 1 - horizon_anom,
        num_shifts=args.num_shifts,
        seed=args.shift_seed,
    )
    assert (test_df["shift_end"] >= test_df["shift_start"]).all(), "Some rows have shift_end < shift_start"

    test_df["post_amount"] = get_reg_target(test_df, horizon=horizon_reg)
    test_df["post_forecast_target"] = test_df.apply(forecast_list, axis=1)

    test_df["post_anomaly_target"] = get_anomaly_target(test_df, horizon=horizon_anom, q=0.95)
    test_df["post_target"] = duplicate_target_by_shifts(test_df, "type_code")

    train_df["shift_start"] = -1
    train_df["shift_end"] = -1
    train_df = fill_train_targets(
        train_df,
        [
            "shifts",
            "post_amount",
            "post_forecast_target",
            "post_anomaly_target",
            "post_target",
        ],
        value=-1,
    )

    keep_cols = INDEX_COLUMNS + [
        TM,
        "used_in_train",
        "shifts",
        "post_amount",
        "post_target",
        "post_forecast_target",
        "post_anomaly_target",
    ] + CAT_FEATURES + NUM_FEATURES

    full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    full_df = full_df[keep_cols]
    save_partitioned_parquet(full_df, args.save_path / "full_ntp", 20)

if __name__ == "__main__":
    main()
