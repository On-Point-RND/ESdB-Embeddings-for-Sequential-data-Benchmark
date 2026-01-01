from argparse import ArgumentParser
from pathlib import Path

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql.types import LongType, StringType, TimestampType, FloatType

from common import cat_freq, collect_lists, train_test_split


CAT_FEATURES = []
INDEX_COLUMNS = ["week_id"]
ORDERING_COLUMNS = []
TARGET_VALS = [0, 1]
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

    spark = SparkSession.builder.master("local[32]").getOrCreate()

    base_path = args.data_path.as_posix()

    def load_and_aggregate(path, transformer_id):
        df = (
            spark.read
                .option("header", True)
                .option("inferSchema", True)
                .csv(path)
        )

        # Приведение всех double к float перед агрегацией
        exempt_cols = {"date"}  # колонка date оставляем timestamp
        for col_name, col_type in df.dtypes:
            if col_name not in exempt_cols and col_type == "double":
                df = df.withColumn(col_name, F.col(col_name).cast(FloatType()))

        df = df.withColumn("date", F.to_timestamp("date"))
        df = df.withColumn("week", F.date_trunc("week", F.col("date")))

        # Добавляем колонку transformer до агрегации
        df = df.withColumn("transformer", F.lit(transformer_id))

        rows_df = collect_lists(
            df,
            group_by=["transformer", "week"],
            order_by=["date"]
        )

        # Переименование колонок с _list
        for col_name in rows_df.columns:
            if col_name.endswith("_list"):
                rows_df = rows_df.withColumnRenamed(col_name, col_name.replace("_list", ""))

        sequence_col = "date"
        # Добавление fictive_time
        rows_df = rows_df.withColumn(
            "fictive_time",
            F.expr(f"sequence(1, size({sequence_col}))")
        )

        return rows_df

    df1 = load_and_aggregate(f"{base_path}/ETTm1.csv", 1)
    df2 = load_and_aggregate(f"{base_path}/ETTm2.csv", 2)

    # объединяем
    full_df = df1.unionByName(df2)

    # уникальный ID недели для каждой пары (transformer, week)
    w = Window.orderBy("transformer", "week")
    full_df = full_df.withColumn(
        "week_id",
        F.dense_rank().over(w)
    )

    ###########################
    full_df = full_df.withColumn(
        "used_in_train",
        F.when(F.size("date") < 650,
               F.lit(-1)).otherwise(F.lit(0))
    )
    # Здесь удалили все не подходящие начисто
    full_df = full_df.filter(F.col("used_in_train") != -1)
    MAX_LEN = 192  # или любое другое
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
    #cut_df = cut_df.withColumn("_seq_len", F.lit(MAX_LEN))
    ###########################
    # stratified splitting on train and test
    train_df, test_df = train_test_split(
        df=cut_df,
        test_frac=TEST_FRACTION,
        index_col="week_id",
        stratify_col="mock_target",
        stratify_col_vals=TARGET_VALS,
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
        join_keys = ["week_id"]
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
