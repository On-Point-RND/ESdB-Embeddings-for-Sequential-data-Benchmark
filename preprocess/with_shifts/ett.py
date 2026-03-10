from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import SparkSession, Window
from pyspark.sql.types import FloatType

from ..common import collect_lists
from .common_pandas import (
    add_shift_columns,
    global_time_split,
    save_partitioned_parquet,
    filter_short,
    shift_end_by_len,
    split_num_shifts,
    global_train_column,
    trim_test,
)

NUM_FEATURES = ["LUFL", "MUFL", "MULL", "LULL", "HULL", "HUFL", "OT"]
INDEX_COLUMNS = ["week_id"]
ORDERING_COLUMNS = ["time"]
TM = ORDERING_COLUMNS[0]


def load_and_aggregate(spark, path, transformer_id):
    df = spark.read.option("header", True).option("inferSchema", True).csv(path)

    exempt_cols = {"date"}
    for col_name, col_type in df.dtypes:
        if col_name not in exempt_cols and col_type == "double":
            df = df.withColumn(col_name, F.col(col_name).cast(FloatType()))

    df = df.withColumn("date", F.to_timestamp("date"))
    df = df.withColumn("week", F.date_trunc("week", F.col("date")))

    # Добавляем колонку transformer до агрегации
    df = df.withColumn("transformer", F.lit(transformer_id))

    rows_df = collect_lists(df, group_by=["transformer", "week"], order_by=["date"])

    for col_name in rows_df.columns:
        if col_name.endswith("_list"):
            rows_df = rows_df.withColumnRenamed(col_name, col_name.replace("_list", ""))

    sequence_col = "date"
    rows_df = rows_df.withColumn(TM, F.expr(f"sequence(1, size({sequence_col}))"))

    return rows_df


def get_forecast_target_row(row: pd.Series) -> list[float]:
    seq = np.asarray(row["OT"])
    out = []
    for s in row["shifts"]:
        s = int(s)
        base = seq[s - 1]
        diff = seq[s] - base
        out.append(float(diff))
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

    spark = SparkSession.builder.master("local[32]").getOrCreate()  # type: ignore[attr-defined]

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

    base_path = args.data_path.as_posix()

    df1 = load_and_aggregate(spark, f"{base_path}/ETTm1.csv", 1)
    df2 = load_and_aggregate(spark, f"{base_path}/ETTm2.csv", 2)

    full_df = df1.unionByName(df2)

    # уникальный ID недели для каждой пары (transformer, week)
    w = Window.orderBy("transformer", "week")
    full_df = full_df.withColumn("week_id", F.dense_rank().over(w))

    df = full_df.toPandas()
    df = filter_short(df)

    df["shift_end"] = shift_end_by_len(df[TM], -2)

    train_df, test_df = global_time_split(
        data=df,
        test_frac=time_test_split,
        time_col=TM,
        seqlen_col="_seq_len",
    )

    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df["shift_end"] = shift_end_by_len(train_df[TM], -2)

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
    test_df["target__forecast__local__mse+r2"] = test_df.apply(
        get_forecast_target_row, axis=1
    )
    train_df["target__forecast__local__mse+r2"] = train_df.apply(
        get_forecast_target_row, axis=1
    )

    keep_cols = (
        INDEX_COLUMNS
        + ORDERING_COLUMNS
        + NUM_FEATURES
        + [
            "_seq_len",
            "shifts",
            "target__forecast__local__mse+r2",
            "global_train",
        ]
    )

    save_partitioned_parquet(
        train_df[keep_cols], args.save_path / "train", 20, mode=mode
    )
    save_partitioned_parquet(test_df[keep_cols], args.save_path / "test", 20, mode=mode)


if __name__ == "__main__":
    main()
