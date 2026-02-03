from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import SparkSession, Window

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


INDEX_COLUMNS = ["sequence_id"]
TEST_FRACTION = 0.1
ORDERING_COLUMNS = ["time"]
TM = ORDERING_COLUMNS[0]

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
    parser.add_argument("--data-path", required=True, type=Path)
    parser.add_argument("--save-path", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--num-shifts", type=int, default=100)
    parser.add_argument("--shift-seed", type=int, default=1)
    args = parser.parse_args()

    spark = SparkSession.builder.master("local[32]").getOrCreate()  # type: ignore[attr-define]

    df = spark.read.text(args.data_path.as_posix())
    df = df.withColumn(
        "values",
        F.expr("transform(split(trim(value), '\\\\s+'), x -> cast(x as float))"),
    )
    df = df.withColumn("first", F.col("values")[0])
    df = df.withColumn(
        "is_start",
        F.when(
            (F.col("first").between(1.0, 7.0)) & (F.col("first") == F.floor(F.col("first"))),
            1,
        ).otherwise(0),
    )
    w = Window.orderBy(F.monotonically_increasing_id())
    df = df.withColumn("sequence_id", F.sum("is_start").over(w))
    df = df.withColumn("target", F.when(F.col("is_start") == 1, F.col("first")))
    df = df.withColumn("values", F.expr("slice(values, 2, size(values))"))

    rows_df = (
        df.groupBy("sequence_id")
        .agg(
            (F.first("target", ignorenulls=True) - 1).cast("int").alias("target"),
            F.flatten(F.collect_list("values")).alias("sequence"),
        )
    )
    full_df = rows_df

    full_df = full_df.withColumn(
        "time", F.expr("transform(sequence, (x, i) -> cast(i + 1 as float))")
    )

    df = full_df.toPandas()
    full_df["_seq_len"] = full_df[TM].map(len)
    df = filter_short(df)

    # shift_end as index of last valid shift (len - 2)
    df["shift_end"] = shift_end_by_len(df["sequence"], -2)

    train_df, test_df = global_time_split(
        data=df,
        test_frac=TEST_FRACTION,
        min_shift_start=2,
        time_col="time",
        seqlen_col="_seq_len",
    )

    train_df = train_df.copy()
    test_df = test_df.copy()

    # recompute shift_end for train after trimming
    train_df["shift_end"] = shift_end_by_len(train_df["sequence"], -2)

    assert (train_df["shift_end"] >= train_df["shift_start"]).all()
    assert (test_df["shift_end"] >= test_df["shift_start"]).all()

    train_n_shifts, test_n_shifts = split_num_shifts(args.num_shifts, TEST_FRACTION)

    test_df = add_shift_columns(test_df, test_n_shifts, args.shift_seed)
    train_df = add_shift_columns(train_df, train_n_shifts, args.shift_seed)

    test_df["post_forecast_target"] = test_df.apply(get_forecast_target_row, axis=1)
    test_df["post_target"] = duplicate_target_by_shifts(test_df, "target")

    train_df["post_forecast_target"] = train_df.apply(get_forecast_target_row, axis=1)
    train_df["post_target"] = duplicate_target_by_shifts(train_df, "target")

    # debug: map shifts to timestamps
    test_df = add_debug_f(test_df, time_col="time")
    train_df = add_debug_f(train_df, time_col="time")

    keep_cols = [
        "sequence_id",
        "sequence",
        "time",
        "_seq_len",
        "target",
        "shifts",
        "post_forecast_target",
        "post_target",
        "debug_f",
    ]

    save_partitioned_parquet(train_df[keep_cols], args.save_path / "train", 20)
    save_partitioned_parquet(test_df[keep_cols], args.save_path / "test", 20)


if __name__ == "__main__":
    main()
