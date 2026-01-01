from argparse import ArgumentParser
from pathlib import Path

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import LongType, StringType, TimestampType

from common import cat_freq, collect_lists, train_test_split


CAT_FEATURES = ["item_id", "behavior_type"]
INDEX_COLUMNS = ["user_id", "ratio_bucket"]
ORDERING_COLUMNS = ["time"]
TARGET_VALS = [0, 1, 2, 3]
TEST_FRACTION = 0.5

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
    args = parser.parse_args()
    mode = "overwrite" if args.overwrite else "error"

    spark = SparkSession.builder.master("local[32]").getOrCreate()  # pyright: ignore

    df = spark.read.csv(args.data_path.as_posix(), header=True)
    df = df.select(
        F.col("user_id").cast(StringType()),
        F.col("item_id").cast(LongType()),
        F.col("behavior_type").cast(LongType()),
        F.col("time").cast(TimestampType()),
    )

    agg_df = (
        df.groupBy("user_id")
            .agg(
            F.sum(F.when(F.col("behavior_type") == 1, 1).otherwise(0)).alias("purchase_view_cnt"),
            F.sum(F.when(F.col("behavior_type") == 4, 1).otherwise(0)).alias("payment_cnt"),
        )
    )
    vcs = cat_freq(df, CAT_FEATURES)
    for vc in vcs:
        df = vc.encode(df)
        if args.cat_codes_path is not None:
            vc.write(args.cat_codes_path / vc.feature_name, mode=mode)

    agg_df = agg_df.withColumn(
        "ratio",
        F.when(F.col("purchase_view_cnt") == 0, 0)
            .otherwise(F.col("payment_cnt") / F.col("purchase_view_cnt"))
    )

    quantiles = agg_df.approxQuantile("ratio", [0.25, 0.5, 0.75], 0.001)
    q1, q2, q3 = quantiles

    agg_df = agg_df.withColumn(
        "ratio_bucket",
        F.when(F.col("ratio") <= q1, 0)
            .when(F.col("ratio") <= q2, 1)
            .when(F.col("ratio") <= q3, 2)
            .otherwise(3)
    )

    rows_df = collect_lists(
        df,
        group_by=["user_id"],  # одна строка на пользователя
        order_by=["time"]  # сортировка внутри последовательности
    )

    for col_name in rows_df.columns:
        if col_name.endswith("_list"):
            rows_df = rows_df.withColumnRenamed(col_name, col_name.replace("_list", ""))

    agg_target = agg_df.select("user_id", "ratio_bucket")

    full_df = rows_df.join(agg_target, on="user_id", how="inner")

    ###########################
    full_df = full_df.withColumn(
        "used_in_train",
        F.when(F.col("time").isNotNull() & (F.size("time") < 1000),
               F.lit(-1)
               ).otherwise(F.lit(0))
    )
    # Здесь удалили все не подходящие начисто
    full_df = full_df.filter(F.col("used_in_train") != -1)
    MAX_LEN = 600  # или любое другое
    full_df = full_df.withColumn("_seq_len", F.lit(MAX_LEN))

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
    #cut_df = cut_df.withColumn("_seq_len", F.lit(MAX_LEN))
    ###########################
    # stratified splitting on train and test
    train_df, test_df = train_test_split(
        df=cut_df,
        test_frac=TEST_FRACTION,
        index_col="user_id",
        stratify_col="ratio_bucket",
        stratify_col_vals=list(range(len(TARGET_VALS))),
        random_seed=args.split_seed,
    )

    train_df = train_df.withColumn("used_in_train", F.lit(1))
    test_df = test_df.withColumn("used_in_train", F.lit(0))

    train_df.repartition(args.train_partitions).write.parquet(
        (args.save_path / "train").as_posix(), mode=mode
    )
    test_df.repartition(args.test_partitions).write.parquet(
        (args.save_path / "test").as_posix(), mode=mode
    )
    ###############################################################################
    full_base_df = train_df.unionByName(test_df)
    post_cols = [c for c, t in cut_last_df.dtypes if c.startswith("post_")]
    if post_cols:
        join_keys = ["user_id"]
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
    full_base_df.repartition(args.test_partitions).write.parquet(
        (args.save_path / "full_ntp").as_posix(), mode=mode
    )
if __name__ == "__main__":
    main()
