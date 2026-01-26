from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import SparkSession, Window

from .common_pandas import (
    add_shift_columns,
    duplicate_target_by_shifts,
    fill_train_targets,
    pandas_train_test_split,
    save_partitioned_parquet,
)


CAT_FEATURES = []
INDEX_COLUMNS = ["sequence_id"]
ORDERING_COLUMNS = []
TEST_FRACTION = 0.5


def get_forecast_target_row(row: pd.Series) -> list[float]:
    seq = np.asarray(row["sequence"])
    out = []
    for s in row["shifts"]:
        s = int(s)
        if s >= len(seq) - 1:
            out.append(0.0)
            continue
        base = seq[s]
        diff = seq[s + 1 :] - base
        idx = np.where(diff != 0)[0]
        out.append(float(diff[idx[0]]) if len(idx) else 0.0)
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
    args = parser.parse_args()
    mode = "overwrite" if args.overwrite else "error"

    spark = SparkSession.builder.master("local[32]").getOrCreate()  # type: ignore[attr-define]

    # 1. Читаем txt построчно
    df = spark.read.text(args.data_path.as_posix())  # колонка "value"

    # 2. Парсим строку в массив чисел
    df = df.withColumn(
        "values",
        F.expr("transform(split(trim(value), '\\\\s+'), x -> cast(x as float))")
    )

    # 3. Первый элемент строки
    df = df.withColumn("first", F.col("values")[0])

    # 4. Флаг начала новой последовательности
    df = df.withColumn(
        "is_start",
        F.when(
            (F.col("first").between(1.0, 7.0)) &
            (F.col("first") == F.floor(F.col("first"))),
            1
        ).otherwise(0)
    )

    # 5. Кумулятивный ID последовательности
    w = Window.orderBy(F.monotonically_increasing_id())
    df = df.withColumn("sequence_id", F.sum("is_start").over(w))

    # 6. target — метка из стартовой строки
    df = df.withColumn(
        "target",
        F.when(F.col("is_start") == 1, F.col("first"))
    )

    # 7. Убираем первый элемент из values
    df = df.withColumn(
        "values",
        F.expr("slice(values, 2, size(values))")
    )

    # 8. Агрегация: одна строка = одна последовательность
    rows_df = (
        df
            .groupBy("sequence_id")
            .agg(
            (F.first("target", ignorenulls=True) - 1)
                .cast("int")
                .alias("target"),
            F.flatten(F.collect_list("values")).alias("sequence")
        )
    )
    full_df = rows_df
    ###########################
    full_df = full_df.withColumn(
        "used_in_train",
        F.when(F.size("sequence") < 90,
               F.lit(-1)).otherwise(F.lit(0))
    )
    # Здесь удалили все не подходящие начисто
    full_df = full_df.filter(F.col("used_in_train") != -1)
    MAX_LEN = 32  # или любое другое
    full_df = full_df.withColumn("_seq_len", F.lit(MAX_LEN))
    full_df = full_df.withColumn(
        "time",
        F.expr("transform(sequence, (x, i) -> cast(i + 1 as float))")
    )


    full_df = full_df.toPandas()

    stratify_col, stratify_col_vals = None, None
    stratify_col = "target"
    stratify_col_vals = list(range(1, 8))

    train_df, test_df = pandas_train_test_split(
        df=full_df,
        test_frac=TEST_FRACTION,
        index_col="sequence_id",
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
        shift_end=test_df["sequence"].map(len) - 2,
        num_shifts=args.num_shifts,
        seed=args.shift_seed,
    )
    assert (test_df["shift_end"] >= test_df["shift_start"]).all(), "Some rows have shift_end < shift_start"
    test_df["post_forecast_target"] = test_df.apply(get_forecast_target_row, axis=1)
    test_df["post_target"] = duplicate_target_by_shifts(test_df, "target")

    train_df["shift_start"] = -1
    train_df["shift_end"] = -1
    train_df = fill_train_targets(train_df, ["shifts", "post_forecast_target", "post_target"], value=-1)

    keep_cols = [
        "sequence_id",
        "sequence",
        "time",
        "_seq_len",
        "target",
        "used_in_train",
        "shifts",
        "post_forecast_target",
        "post_target",
    ]

    full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    full_df = full_df[keep_cols]
    save_partitioned_parquet(full_df, args.save_path / "full_ntp", 20)

if __name__ == "__main__":
    main()
