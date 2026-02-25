from datetime import datetime, timedelta
from joblib import Parallel, delayed
from argparse import ArgumentParser
from pathlib import Path

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import LongType, FloatType
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

CAT_FEATURES = ["mcc_category"]
NUM_FEATURES = ["amnt_rur"]
INDEX_COLUMNS = ["client_id"]
ORDERING_COLUMNS = ["time_from_first_trn"]
TM = ORDERING_COLUMNS[0]


def reg_target_row(row, horizon=30):
    a = np.asarray(row["amnt"])
    t = np.asarray(row["time_from_first_trn"])
    out = []
    for s in row["shifts"]:
        s = int(s)
        delta = t - t[s - 1]
        mask = (delta > 0) & (delta < horizon)
        out.append(np.log1p(a[mask].sum()))
    return out


def get_forecast_target(row):
    t = np.asarray(row["time_from_first_trn"])
    out = []
    for s in row["shifts"]:
        mask = t == t[s - 1]
        mask[:s] = False
        out.append(np.log1p(np.sum(mask)))
    return out


def create_datetimes_single(row, reference_year=2023):
    """Process a single row"""
    weeks = row["weekofyear"]
    hours = row["hour"]
    days = row["day_of_week"]
    datetimes = []
    for week, hour, day in zip(weeks, hours, days):
        # Convert numpy ints to Python ints
        week_int = int(week)
        hour_int = int(hour)
        day_int = int(day)

        # Create first day of the reference year
        first_day = datetime(reference_year, 1, 1)

        # Adjust to first Monday (ISO weeks start with Monday)
        # weekday() returns 0=Monday, 6=Sunday
        days_to_monday = (7 - first_day.weekday()) % 7
        first_monday = first_day + timedelta(days=days_to_monday)

        # Calculate the date from week and day
        # Week 1 starts from first_monday
        target_date = first_monday + timedelta(weeks=week_int - 1, days=day_int - 1)

        # Add the hour
        target_date = target_date.replace(hour=hour_int % 24)
        print(hour_int % 24)
        datetimes.append(target_date)

    return datetimes


def process_chunk(chunk, reference_year=2023):
    """Process a chunk of rows"""
    results = []
    for _, row in chunk.iterrows():
        results.append(create_datetimes_single(row, reference_year))
    return results


def create_datetimes_chunked(df, reference_year=2023, n_jobs=8, chunk_size=1000):
    """Parallel processing with chunking for better performance"""
    # Split dataframe into chunks
    chunks = [df.iloc[i : i + chunk_size] for i in range(0, len(df), chunk_size)]

    # Process chunks in parallel
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_chunk)(chunk, reference_year) for chunk in chunks
    )

    # Flatten results
    flattened = []
    for chunk_result in results:
        flattened.extend(chunk_result)

    return flattened


def get_time_from_first_trx(users_time, t0):
    return [(t - t0) / np.timedelta64(1, "h") for t in users_time]


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
    df, df_product = None, None

    if args.which_split == "train":
        df = spark.read.parquet(
            (
                args.data_path
                / "alfabattle2_train_transactions_contest"
                / "train_transactions_contest"
                / "*.parquet"
            ).as_posix(),
            header=True,
        )

        df = df.select(
            F.col("app_id").cast(LongType()),
            F.col("amnt").cast(FloatType()),
            F.col("mcc_category").cast(LongType()),
            F.col("day_of_week").cast(LongType()),
            F.col("hour").cast(LongType()),
            F.col("weekofyear").cast(LongType()),
            F.col("transaction_number").cast(LongType()),
        )

        df_product = spark.read.csv(
            (args.data_path / "alfabattle2_train_target.csv").as_posix(), header=True
        ).select(F.col("app_id"), F.col("product"), F.col("flag"))

        df = df.join(df_product, on="app_id")
        df = df.withColumnRenamed("app_id", "client_id")
    else:
        raise NotImplementedError(
            "We doesn't know what to do with test.csv for Alpha dataset without labels."
        )

    vcs = cat_freq(df, CAT_FEATURES)
    for vc in vcs:
        df = vc.encode(df)
        if args.cat_codes_path is not None:
            vc.write(args.cat_codes_path / vc.feature_name, mode=mode)

    df = collect_lists(df, group_by=INDEX_COLUMNS, order_by="transaction_number")

    df = df.sort("client_id").toPandas()
    df["time"] = create_datetimes_chunked(df)
    global_min_time = np.concatenate(df["time"].values).min()
    df[TM] = df.apply(
        lambda x: get_time_from_first_trx(x["time"], global_min_time), axis=1
    )
    df[TM] = df[TM].map(lambda x: np.asarray(x, dtype="float32"))
    df["_seq_len"] = df[TM].apply(len)
    df = filter_short(df)

    def compute_shift_end(arr, horizon):
        arr = np.asarray(arr)
        return (arr[-1] - arr > horizon).sum() - 1 if len(arr) else -1

    def trim_users(arr, horizon_days):
        arr = np.asarray(arr)
        if len(arr) < 2:
            return True
        total_duration = arr[-1] - arr[0]
        limit = horizon_days
        return total_duration < limit

    horizon_days = 30 * 24
    df["shift_end"] = df[TM].map(lambda x: compute_shift_end(x, horizon_days))

    train_df, test_df = global_time_split(
        data=df,
        test_frac=time_test_split,
        time_col=TM,
        seqlen_col="_seq_len",
    )

    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df['is_bad_user'] = train_df[TM].apply(lambda x: trim_users(x, horizon_days))
    bad_indices = train_df.index[train_df['is_bad_user']].tolist()
    train_df = train_df.drop(index=bad_indices)
    del train_df['is_bad_user']

    train_df["shift_end"] = train_df[TM].map(
        lambda x: compute_shift_end(x, horizon_days)
    )

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
    test_df["target__prod__global__accuracy+f1_macro"] = train_df["product"]
    test_df["target__reg_amount__local__mse+r2"] = test_df.apply(reg_target_row, axis=1)
    test_df["target__forecast__local__mse+r2"] = test_df.apply(
        get_forecast_target, axis=1
    )
    test_df["target__anomaly__local__roc_auc+f1_macro+accuracy"] = test_df["flag"]

    train_df["target__prod__global__accuracy+f1_macro"] = train_df["product"]
    train_df["target__reg_amount__local__mse+r2"] = train_df.apply(
        reg_target_row, axis=1
    )
    train_df["target__forecast__local__mse+r2"] = train_df.apply(
        get_forecast_target, axis=1
    )
    train_df["target__anomaly__local__roc_auc+f1_macro+accuracy"] = train_df["flag"]


    keep_cols = INDEX_COLUMNS + ORDERING_COLUMNS + NUM_FEATURES + CAT_FEATURES + [
        "_seq_len",
        "shifts",
        "global_train",
        "target__age__global__accuracy+f1_macro",
        "target__anomaly__global__roc_auc+f1_macro+accuracy",
        "target__reg_amount__local__mse+r2",
        "target__forecast__local__mse+r2",
    ]

    save_partitioned_parquet(test_df[keep_cols], args.save_path / "test", 20, mode=mode)
    save_partitioned_parquet(
        train_df[keep_cols], args.save_path / "train", 20, mode=mode
    )


if __name__ == "__main__":
    main()
