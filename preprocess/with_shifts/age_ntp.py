from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import LongType, FloatType

from ..common import cat_freq, collect_lists
from .common_pandas import (
    add_shift_columns,
    duplicate_target_by_shifts,
    fill_train_targets,
    save_partitioned_parquet,
    pandas_train_test_split
)


CAT_FEATURES = ["small_group"]
NUM_FEATURES = ["amount_rur"]
INDEX_COLUMNS = ["client_id", "age"]
ORDERING_COLUMNS = ["trans_date"]
TARGET_VALS = [0, 1, 2, 3]
TEST_FRACTION = 0.5


def log_log_transform(y, epsilon=1e-10):
        """Apply log(log(y)) transformation for doubly exponential data"""
        # Ensure y > 0 for double log
        y_positive = y - y.min() + 1  # Shift to ensure y > 0
        return np.log(np.log(y_positive + epsilon))

def get_anomaly_target(df: pd.DataFrame) -> pd.Series:
    def _cv_list(row):
        a = np.asarray(row["amount_rur"])
        out = []
        for s in row["shifts"]:
            s = int(s)
            post = a[s:]
            assert len(post) != 0, "The column amount_rur after shifts shouldn't be empty."
            mean = post.mean()
            std = post.std()
            assert mean != 0, "Mean of amount_rur shouldn't be zero even after shifts"
            out.append(std / mean)
        return out

    cv_list = df.apply(_cv_list, axis=1)
    all_cv = np.concatenate([np.asarray(v) for v in cv_list if len(v)])
    q95 = np.nanquantile(all_cv, 0.95)
    return cv_list.apply(
        lambda v: [int(x > q95) if not np.isnan(x) else 0 for x in v]
    )

def reg_target_row(row, horizon=30):
    d = np.asarray(row["trans_date"])
    a = np.asarray(row["amount_rur"])
    out = []
    for s in row["shifts"]:
        s = int(s)
        delta = d - d[s]
        mask = (delta > 0) & (delta < horizon)
        out.append(np.log1p(a[mask].sum()))
    return out

def get_forecast_target(row):
    t = np.asarray(row['trans_date'])
    out = []
    for s in row['shifts']:
        mask = (t == t[s])
        mask[:s] = False
        out.append(np.log1p(np.sum(mask)))
    return out

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
        default=1,
        type=int,
    )
    parser.add_argument(
        "--overwrite",
        help='Toggle "overwrite" mode on all spark writes',
        action="store_true",
    )
    parser.add_argument(
        "--num-shifts",
        help="How many shifts to sample per test user",
        default=5,
        type=int,
    )
    parser.add_argument(
        "--shift-seed",
        help="Random seed for shifts",
        default=1,
        type=int,
    )
    args = parser.parse_args()
    mode = "overwrite" if args.overwrite else "error"

    spark = SparkSession.builder.master("local[32]").getOrCreate()  # pyright: ignore
    df, df_kag_train = None, None

    if args.which_split == 'train':
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

        df = df_kag_train.join(df_label, on="client_id")
        df = df.withColumnRenamed("bins", "age")
    else:
        raise NotImplementedError("We doesn't know what to do with test.csv for AGE dataset without labels.")

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
    stratify_col = "age"
    stratify_col_vals = TARGET_VALS

    # stratified splitting on train and test

    ###########################
    df = df.withColumn(
        "used_in_train", # it means just use - no literally in train.
        F.when(F.col("trans_date").isNotNull() & (F.size("trans_date") < 400),
               F.lit(-1)
               ).otherwise(F.lit(0))
    )
    # Здесь удалили все не подходящие начисто
    df = df.filter(F.col("used_in_train") != -1)
    MAX_LEN = 300  # или любое другое


    # Appendix
    df = df.sort("client_id").toPandas()

    train_df, test_df = pandas_train_test_split(
        df=df,
        test_frac=TEST_FRACTION,
        index_col="client_id",
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
        shift_end=test_df["trans_date"].apply(
            lambda x: (np.asarray(x)[-1] - np.asarray(x) > 30).sum()
        ),
        num_shifts=args.num_shifts,
        seed=args.shift_seed,
    )
    assert (test_df["shift_end"] >= test_df["shift_start"]).all(), "Some rows have shift_end < shift_start"

    test_df['post_reg_target'] = test_df.apply(reg_target_row, axis=1)
    test_df['post_target'] = duplicate_target_by_shifts(test_df, "age")
    test_df['post_forecast_target'] = test_df.apply(get_forecast_target, axis=1)
    test_df["post_anomaly_target"] = get_anomaly_target(test_df)

    train_df = fill_train_targets(
        train_df,
        [
            "shifts",
            "post_reg_target",
            "post_target",
            "post_forecast_target",
            "post_anomaly_target",
        ],
        value=-1,
    )

    keep_cols = [
        "client_id",
        "age",
        "trans_date",
        "small_group",
        "amount_rur",
        "_seq_len",
        "used_in_train",
        "shifts",
        "post_reg_target",
        "post_target",
        "post_forecast_target",
        "post_anomaly_target",
    ]

    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df = full_df[keep_cols]
    save_partitioned_parquet(full_df, args.save_path / "full_ntp", 20)


if __name__ == "__main__":
    main()
