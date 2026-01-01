from argparse import ArgumentParser
from pathlib import Path

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import LongType, FloatType

from common import cat_freq, collect_lists, train_test_split


CAT_FEATURES = ["DayOfWeek", "Open", "Promo", "StateHoliday", "SchoolHoliday"]
NUM_FEATURES = ["Sales", "Customers"]
INDEX_COLUMNS = ["Store"]
ORDERING_COLUMNS = ["Date"]
TARGET_VALS = [0, 1]
TEST_FRACTION = 0.5


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
    args = parser.parse_args()
    mode = "overwrite" if args.overwrite else "error"

    spark = SparkSession.builder.master("local[32]").getOrCreate()  # pyright: ignore

    df = (
        spark.read.csv(args.data_path.as_posix(), header=True)
        .withColumn(
            # Переставляем dd.MM.yyyy → yyyy.MM.dd сразу в regexp
            "Date_str",
            F.regexp_replace(
                "Date",
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
        .withColumn("Date", F.to_timestamp("Date_str", "yyyy-MM-dd"))
        .select(
            F.col("Store").cast(LongType()),
            F.col("DayOfWeek").cast(LongType()),
            F.col("Open").cast(LongType()),
            F.col("Promo").cast(LongType()),
            F.col("StateHoliday").cast(LongType()),
            F.col("SchoolHoliday").cast(LongType()),
            F.col("Date"),
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
        F.when(F.col("Date").isNotNull() & (F.size("Date") < 400),
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

    array_cols = [c for c, t in full_df.dtypes if t.startswith("array")]
    cut_last_df = full_df
    cut_df = full_df

    for col_name in array_cols:
        start = F.lit(MAX_LEN + 1)
        length = F.size(F.col(col_name)) - F.lit(MAX_LEN)

        cut_last_df = cut_last_df.withColumn(
            col_name,
            F.slice(F.col(col_name), start, length)
        )
    # Post_sequences
    for col_name in array_cols:
        cut_last_df = cut_last_df.withColumnRenamed(col_name, f"post_{col_name}")
    # Pre_sequencies
    for col_name in array_cols:
        cut_df = cut_df.withColumn(
            col_name,
            F.slice(F.col(col_name), 1, MAX_LEN)
        )
    # cut_df = cut_df.withColumn("_seq_len", F.lit(MAX_LEN))
    ###########################
    # stratified splitting on train and test
    train_df, test_df = train_test_split(
        df=cut_df,
        test_frac=TEST_FRACTION,
        index_col="Store",
        stratify_col="mock_target",
        stratify_col_vals=list(range(len(TARGET_VALS))),
        random_seed=args.split_seed,
    )

    train_df = train_df.withColumn("used_in_train", F.lit(1))
    test_df = test_df.withColumn("used_in_train", F.lit(0))

    train_df.repartition(1).write.parquet(
        (args.save_path / "train").as_posix(), mode=mode
    )
    test_df.repartition(1).write.parquet(
        (args.save_path / "test").as_posix(), mode=mode
    )
    ###############################################################################
    full_base_df = train_df.unionByName(test_df)
    post_cols = [c for c, t in cut_last_df.dtypes if c.startswith("post_")]
    if post_cols:
        join_keys = ["Store"]
        post_df = cut_last_df.select(*(join_keys + post_cols))
        full_base_df = full_base_df.join(post_df, on=join_keys, how="left")
    else:
        raise ValueError("Не найдены колонки post_ в cut_last_df — проверь место их формирования")
    ###############################################################################
    # Находим post-колонки
    post_array_cols = [c for c, t in full_base_df.dtypes
                       if t.startswith("array") and c.startswith("post_")]

    if not post_array_cols:
        raise ValueError("Нет post_ колонок — проверь логику формирования cut_last_df")

    seq_post = post_array_cols[0]  # любая post_ колонка

    full_base_df = full_base_df.withColumn(
        "_seq_len",
        F.lit(MAX_LEN) + F.size(F.col(seq_post))
    )
    ################################################################################
    full_base_df.repartition(1).write.parquet(
        (args.save_path / "full_ntp").as_posix(), mode=mode
    )


if __name__ == "__main__":
    main()
