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
    add_debug_f,
    filter_short,
    global_time_split,
    save_partitioned_parquet,
    shift_end_by_len,
    split_num_shifts,
)


CAT_FEATURES = []
INDEX_COLUMNS = ["week_id"]
ORDERING_COLUMNS = ["date"]
TM = ORDERING_COLUMNS[0]
TARGET_VALS = [0, 1]
TEST_FRACTION = 0.1


def get_forecast_target_row(row: pd.Series) -> list[float]:
    ot = np.asarray(row["OT"])
    out = []
    for s in row["shifts"]:
        s = int(s)
        window = ot[s:]
        out.append(float(np.nanmedian(window)) if len(window) else 0.0)
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
        help="How many shifts to sample per test week",
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

    spark = SparkSession.builder.master("local[32]").getOrCreate()  # type: ignore[attr-defined]

    if args.global_split_ntp == 'y':
        TEST_FRACTION = 0.5
    else:
        TEST_FRACTION = 0.1

    base_path = args.data_path.as_posix()

    def load_and_aggregate(path, transformer_id):
        df = spark.read.option("header", True).option("inferSchema", True).csv(path)

        # Приведение всех double к float перед агрегацией
        exempt_cols = {"date"}  # колонка date оставляем timestamp
        for col_name, col_type in df.dtypes:
            if col_name not in exempt_cols and col_type == "double":
                df = df.withColumn(col_name, F.col(col_name).cast(FloatType()))

        df = df.withColumn("date", F.to_timestamp("date"))
        df = df.withColumn("week", F.date_trunc("week", F.col("date")))

        # Добавляем колонку transformer до агрегации
        df = df.withColumn("transformer", F.lit(transformer_id))

        rows_df = collect_lists(df, group_by=["transformer", "week"], order_by=["date"])

        # Переименование колонок с _list
        for col_name in rows_df.columns:
            if col_name.endswith("_list"):
                rows_df = rows_df.withColumnRenamed(
                    col_name, col_name.replace("_list", "")
                )

        sequence_col = "date"
        # Добавление fictive_time
        rows_df = rows_df.withColumn(
            "fictive_time", F.expr(f"sequence(1, size({sequence_col}))")
        )

        return rows_df

    df1 = load_and_aggregate(f"{base_path}/ETTm1.csv", 1)
    df2 = load_and_aggregate(f"{base_path}/ETTm2.csv", 2)

    # объединяем
    full_df = df1.unionByName(df2)

    # уникальный ID недели для каждой пары (transformer, week)
    w = Window.orderBy("transformer", "week")
    full_df = full_df.withColumn("week_id", F.dense_rank().over(w))

    full_df = full_df.toPandas()

    full_df["date"] = full_df["date"].map(lambda x: np.asarray(x, dtype="datetime64[ns]"))
    full_df["_seq_len"] = full_df[TM].map(len)
    full_df = filter_short(full_df)

    full_df["shift_end"] = shift_end_by_len(full_df["date"], -2)

    train_df, test_df = global_time_split(
        data=full_df,
        test_frac=TEST_FRACTION,
        min_shift_start=2,
        time_col="date",
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
    train_df["shift_end"] = shift_end_by_len(train_df["date"], -2)

    assert (train_df["shift_end"] >= train_df["shift_start"]).all()
    assert (test_df["shift_end"] >= test_df["shift_start"]).all()

    train_n_shifts, test_n_shifts = split_num_shifts(args.num_shifts, TEST_FRACTION)

    test_df = add_shift_columns(test_df, test_n_shifts, args.shift_seed)
    train_df = add_shift_columns(train_df, train_n_shifts, args.shift_seed)

    test_df["post_forecast_target"] = test_df.apply(get_forecast_target_row, axis=1)
    train_df["post_forecast_target"] = train_df.apply(get_forecast_target_row, axis=1)

    # debug: map shifts to timestamps
    test_df = add_debug_f(test_df, time_col="date")
    train_df = add_debug_f(train_df, time_col="date")
    feature_cols = [
        "date",
        "HULL",
        "MULL",
        "OT",
        "LULL",
        "HUFL",
        "LUFL",
        "MUFL",
        "fictive_time",
    ]
    meta_cols = [
        "week_id",
        "_seq_len",
    ]
    target_cols = [
        "shifts",
        "post_forecast_target",
        "users_in_train",
        "debug_f",
    ]
    keep_cols = meta_cols + feature_cols + target_cols

    save_partitioned_parquet(train_df[keep_cols], args.save_path / "train", 20, mode=mode)
    save_partitioned_parquet(test_df[keep_cols], args.save_path / "test", 20, mode=mode)


if __name__ == "__main__":
    main()
