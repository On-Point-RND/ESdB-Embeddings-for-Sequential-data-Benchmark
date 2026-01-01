from argparse import ArgumentParser
from pathlib import Path

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import LongType, FloatType

from common import cat_freq, collect_lists, train_test_split


CAT_FEATURES = ["small_group"]
NUM_FEATURES = ["amount_rur"]
INDEX_COLUMNS = ["client_id", "bins"]
ORDERING_COLUMNS = ["trans_date"]
TARGET_VALS = [0, 1, 2, 3]
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
    df, df_kag_train, df_kag_test = None, None, None

    if args.which_split in ("train", "union"):
        df_kag_train = spark.read.csv(
            (args.data_path / "transactions_train.csv").as_posix(), header=True
        )
        df_kag_train = df_kag_train.select(
            F.col("client_id").cast(LongType()),
            F.col("trans_date").cast(LongType()),
            F.col("small_group").cast(LongType()),
            F.col("amount_rur").cast(FloatType()),
        )

        df_label = spark.read.csv(
            (args.data_path / "train_target.csv").as_posix(), header=True
        ).select(F.col("client_id").cast(LongType()), F.col("bins").cast(LongType()))

        df_kag_train = df_kag_train.join(df_label, on="client_id")

    if args.which_split in ("test", "union"):
        df_kag_test = spark.read.csv(
            (args.data_path / "transactions_test.csv").as_posix(), header=True
        )

        df_kag_test = df_kag_test.select(
            F.col("client_id").cast(LongType()),
            F.col("trans_date").cast(LongType()),
            F.col("small_group").cast(LongType()),
            F.col("amount_rur").cast(FloatType()),
        )

    if df_kag_train is not None and df_kag_test is not None:
        df_kag_test = df_kag_test.withColumn("bins", F.lit(None).cast(LongType()))
        df = df_kag_train.union(df_kag_test)
    elif df_kag_train is not None:
        df = df_kag_train
    elif df_kag_test is not None:
        df = df_kag_test
    else:
        raise ValueError("Something went wrong, train and test are None")

    vcs = cat_freq(df, CAT_FEATURES)
    for vc in vcs:
        df = vc.encode(df)
        if args.cat_codes_path is not None:
            vc.write(args.cat_codes_path / vc.feature_name, mode=mode)

    df = collect_lists(
        df,
        group_by=INDEX_COLUMNS,
        order_by=ORDERING_COLUMNS,
    )

    stratify_col, stratify_col_vals = None, None
    if df_kag_train is not None:  # target has non-null values
        stratify_col = "bins"
        stratify_col_vals = TARGET_VALS

    # stratified splitting on train and test

    ###########################
    df = df.withColumn(
        "used_in_train",
        F.when(F.col("trans_date").isNotNull() & (F.size("trans_date") < 400),
               F.lit(-1)
               ).otherwise(F.lit(0))
    )
    # Здесь удалили все не подходящие начисто
    df = df.filter(F.col("used_in_train") != -1)
    MAX_LEN = 300  # или любое другое

    array_cols = [c for c, t in df.dtypes if t.startswith("array")]
    cut_last_df = df
    cut_df = df

    for col_name in array_cols:
        start = F.lit(MAX_LEN + 1)
        length = F.size(F.col(col_name)) - F.lit(MAX_LEN)

        cut_last_df = cut_last_df.withColumn(
            col_name,
            F.slice(F.col(col_name), start, length)
        )

    for col_name in array_cols:
        cut_last_df = cut_last_df.withColumnRenamed(col_name, f"post_{col_name}")

    for col_name in array_cols:
        cut_df = cut_df.withColumn(
            col_name,
            F.slice(F.col(col_name), 1, MAX_LEN)
        )
    cut_df = cut_df.withColumn("_seq_len", F.lit(MAX_LEN))
    ###########################
    # stratified splitting on train and test
    train_df, test_df = train_test_split(
        df=cut_df,
        test_frac=TEST_FRACTION,
        index_col="client_id",
        stratify_col=stratify_col,
        stratify_col_vals=stratify_col_vals,
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

    cut_last_small = (
        cut_last_df
            .select(
            "client_id",
            F.col("_seq_len").alias("_seq_len_cut"),
            *[c for c in cut_last_df.columns if c.startswith("post_")]
        )
    )

    joined = full_base_df.join(cut_last_small, on="client_id", how="left")

    joined = joined.withColumn(
        "_seq_len",
        F.coalesce(F.col("_seq_len_cut"), F.col("_seq_len"))
    )

    joined = joined.drop("_seq_len_cut")

    joined.repartition(1).write.parquet(
        (args.save_path / "full_ntp").as_posix(), mode=mode
    )
    ###############################################################################


if __name__ == "__main__":
    main()

