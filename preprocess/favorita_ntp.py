from argparse import ArgumentParser
from pathlib import Path

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import Window
from pyspark.sql.types import LongType, StringType, TimestampType, FloatType

from common import cat_freq, collect_lists, train_test_split


CAT_FEATURES = []
INDEX_COLUMNS = ["store_nbr"]
ORDERING_COLUMNS = ["date"]
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

    spark = (
        SparkSession.builder
            .master("local[32]")
            .config("spark.driver.memory", "50g")
            .config("spark.driver.maxResultSize", "0")
            .config("spark.sql.shuffle.partitions", 1000)
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.memory.fraction", "0.8")
            .config("spark.memory.storageFraction", "0.3")
            .config(
            "spark.driver.extraJavaOptions",
            "-XX:+UseG1GC "
            "-XX:InitiatingHeapOccupancyPercent=35 "
            "-XX:+ExplicitGCInvokesConcurrent"
            "-Xss16m"
        )
            .config("spark.sql.adaptive.enabled", "true")
            .getOrCreate()
    )

    # ---------- READ ----------
    data_dir = args.data_path.as_posix()

    df = (
        spark.read
            .option("header", True)
            .option("inferSchema", True)
            .csv(f"{data_dir}/train.csv")  # <-- имя файла руками
            .select(
            F.col("store_nbr").cast("int"),
            F.col("item_nbr").cast("int"),
            F.col("date").cast(TimestampType()),
            F.col("unit_sales").cast("double"),
        )
    )

    cls_df = (
        spark.read
            .option("header", True)
            .option("inferSchema", True)
            .csv(f"{data_dir}/items.csv")  # <-- второе имя файла руками
            .select(
            F.col("item_nbr").cast("int"),
            F.col("class").cast("int").alias("class_id"),
        )
    )

    # ---------- MAP TO CLASS + AGGREGATE ----------
    df_cls = (
        df.join(cls_df, on="item_nbr", how="inner")
            .groupBy("store_nbr", "class_id", "date")
            .agg(F.sum("unit_sales").alias("unit_sales"))
    )

    CAT_FEATURES = ["class_id"]
    vcs = cat_freq(df_cls, CAT_FEATURES)
    for vc in vcs:
        df_cls = vc.encode(df_cls)
        if args.cat_codes_path is not None:
            vc.write(args.cat_codes_path / vc.feature_name, mode=mode)

    # ---------- DIMENSIONS ----------
    stores = df_cls.select("store_nbr").distinct()
    classes = df_cls.select("class_id").distinct()
    dates = df_cls.select("date").distinct()

    # ---------- GRID ----------
    full_grid = (
        stores
            .crossJoin(classes)
            .crossJoin(dates)
            .repartition("store_nbr")
    )

    # ---------- JOIN + ZERO FILL ----------
    df_full = (
        full_grid.join(
            df_cls,
            on=["store_nbr", "class_id", "date"],
            how="left"
        )
            .withColumn("unit_sales", F.coalesce(F.col("unit_sales"), F.lit(0.0)))
    )

    # ---------- TS COLLECT ----------
    sales_ts = (
        df_full
            .groupBy("store_nbr", "class_id")
            .agg(
            F.sort_array(
                F.collect_list(F.struct("date", "unit_sales"))
            ).alias("tmp")
        )
            .select(
            "store_nbr",
            "class_id",
            F.expr("transform(tmp, x -> x.unit_sales)").alias("sales_ts")
        )
    )

    # ---------- PIVOT ----------
    pivot_df = (
        sales_ts
            .groupBy("store_nbr")
            .pivot("class_id")
            .agg(F.first("sales_ts"))
    )

    for c in pivot_df.columns:
        if c != "store_nbr":
            pivot_df = pivot_df.withColumnRenamed(c, f"class_{c}_sales")

        # ---------- MATERIALIZE PIVOT ----------
    # переписываем pivot в parquet с большим числом партиций
    pivot_df.repartition(50).write.mode("overwrite").parquet("/tmp/pivot_cached")

    # читаем обратно — теперь lineage короткий, а партиций 50
    pivot_df = spark.read.parquet("/tmp/pivot_cached")
    # ---------- DATE ARRAY ----------
    date_array = (
        dates.orderBy("date")
            .agg(F.collect_list("date").alias("date"))
            .first()["date"]
    )

    full_df = pivot_df.withColumn("date", F.lit(date_array))

    full_df = pivot_df.withColumn("date", F.lit(date_array))
    full_df = full_df.repartition("store_nbr").cache()
    full_df.count()

    ###########################
    full_df = full_df.withColumn(
        "used_in_train",
        F.when(F.col("date").isNotNull() & (F.size("date") < 600),
               F.lit(-1)
               ).otherwise(F.lit(0))
    )
    # Здесь удалили все не подходящие начисто
    full_df = full_df.filter(F.col("used_in_train") != -1)
    MAX_LEN = 400  # или любое другое
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
    cut_df = cut_df.repartition("store_nbr").cache()
    cut_df.count()
    train_df, test_df = train_test_split(
        df=cut_df,
        test_frac=TEST_FRACTION,
        index_col="store_nbr",
        stratify_col="mock_target",
        stratify_col_vals=list(range(len(TARGET_VALS))),
        random_seed=args.split_seed,
    )

    train_df = train_df.withColumn("used_in_train", F.lit(1))
    test_df = test_df.withColumn("used_in_train", F.lit(0))

    train_df.repartition(50).write.parquet(
        (args.save_path / "train").as_posix(), mode=mode
    )
    test_df.repartition(50).write.parquet(
        (args.save_path / "test").as_posix(), mode=mode
    )
    ###############################################################################
    full_base_df = train_df.unionByName(test_df)
    post_cols = [c for c, t in cut_last_df.dtypes if c.startswith("post_")]
    if post_cols:
        join_keys = ["store_nbr"]
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
    full_base_df.repartition(50).write.parquet(
        (args.save_path / "full_ntp").as_posix(), mode=mode
    )
if __name__ == "__main__":
    main()
