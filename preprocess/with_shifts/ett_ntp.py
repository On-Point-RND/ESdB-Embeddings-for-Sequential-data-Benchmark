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
    fill_train_targets,
    pandas_train_test_split,
    save_partitioned_parquet,
)


CAT_FEATURES = []
INDEX_COLUMNS = ["week_id"]
ORDERING_COLUMNS = []
TARGET_VALS = [0, 1]
TEST_FRACTION = 0.5


def get_forecast_target_row(row: pd.Series) -> list[float]:
    ot = np.asarray(row["OT"])
    out = []
    for s in row["shifts"]:
        s = int(s)
        window = ot[s:]
        out.append(float(np.log1p(np.nanmedian(window))) if len(window) else 0.0)
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

    args = parser.parse_args()
    mode = "overwrite" if args.overwrite else "error"

    spark = SparkSession.builder.master("local[32]").getOrCreate()  # type: ignore[attr-defined]

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

    ###########################
    full_df = full_df.withColumn(
        "used_in_train", F.when(F.size("date") < 650, F.lit(-1)).otherwise(F.lit(0))
    )
    # Здесь удалили все не подходящие начисто
    full_df = full_df.filter(F.col("used_in_train") != -1)
    MAX_LEN = 192  # или любое другое
    full_df = full_df.withColumn("_seq_len", F.lit(MAX_LEN))

    full_df = full_df.withColumn("mock_target", (F.rand() < 0.5).cast("int"))

    full_df = full_df.toPandas()

    stratify_col, stratify_col_vals = None, None
    stratify_col = "mock_target"
    stratify_col_vals = TARGET_VALS

    train_df, test_df = pandas_train_test_split(
        df=full_df,
        test_frac=TEST_FRACTION,
        index_col="week_id",
        stratify_col=stratify_col,
        stratify_col_vals=stratify_col_vals,
        random_seed=args.split_seed,
    )

    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df["used_in_train"] = 1
    test_df["used_in_train"] = 0

    test_df = add_shift_columns(
        test_df,
        shift_start=MAX_LEN,
        shift_end=test_df["date"].map(len) - 2,
        num_shifts=args.num_shifts,
        seed=args.shift_seed,
    )
    assert (test_df["shift_end"] >= test_df["shift_start"]).all(), "Some rows have shift_end < shift_start"

    test_df["post_forecast_target"] = test_df.apply(get_forecast_target_row, axis=1)

    train_df = fill_train_targets(train_df, ["shifts", "post_forecast_target"], value=-1)
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
        "_last_date",
        "used_in_train",
        "mock_target",
    ]
    target_cols = [
        "shifts",
        "post_forecast_target",
    ]
    keep_cols = meta_cols + feature_cols + target_cols

    full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    full_df = full_df[keep_cols]
    save_partitioned_parquet(full_df, args.save_path / "full_ntp", 20)


if __name__ == "__main__":
    main()
