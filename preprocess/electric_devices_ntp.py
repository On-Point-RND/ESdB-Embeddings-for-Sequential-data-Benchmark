from argparse import ArgumentParser
from pathlib import Path

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql.types import LongType, StringType, TimestampType, FloatType

from common import cat_freq, collect_lists, train_test_split


CAT_FEATURES = []
INDEX_COLUMNS = ["sequence_id"]
ORDERING_COLUMNS = []
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
        index_col="sequence_id",
        stratify_col="target",
        stratify_col_vals=list(range(1, 8)),
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
        join_keys = ["sequence_id"]
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
