from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .common_pandas import save_partitioned_parquet

CAT_FEATURES = ["char"]
TARGET_VALS = [0, 1]
TEST_FRACTION = 0.1
LOWER_BOUND = 80
UPPER_BOUND = 150


def get_reg_target(data: pd.DataFrame) -> pd.Series:
    return data["mentions"]


def get_anomaly_target(data: pd.DataFrame) -> pd.Series:
    data = data.copy()
    data["punct_ratio"] = data["clean_tweet"].astype(str).map(punctuation_ratio)
    return (data["punct_ratio"] > 0.1).astype(np.int64)


def get_clf_target(data: pd.DataFrame) -> pd.Series:
    return data["sentiment"]


def get_forecast_target(data: pd.DataFrame) -> pd.Series:
    return data["char"].map(lambda x: int(x[0]) if len(x) else 0)


def _stratified_split(
    df: pd.DataFrame, test_frac: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    test_idx = []
    for val in TARGET_VALS:
        idx = df.index[df["sentiment"] == val].to_numpy()
        if len(idx) == 0:
            continue
        size = int(np.ceil(len(idx) * test_frac))
        pick = rng.choice(idx, size=size, replace=False)
        test_idx.append(pick)
    if test_idx:
        test_idx = np.concatenate(test_idx)
    else:
        test_idx = np.empty(0, dtype=np.int64)
    test_idx_list = test_idx.tolist()
    test_df = df.loc[test_idx_list]
    train_df = df.drop(index=test_idx_list)
    return train_df, test_df


def punctuation_ratio(s: str) -> float:
    if not s:
        return 0.0
    punct = sum(1 for ch in s if ch in "!?.,;:")
    return punct / len(s)


def find_sobaka(text: str) -> tuple[int, str]:
    count = text.count("@")
    cleaned = text.replace("@", "")
    return count, cleaned


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
        help='Toggle "overwrite" mode on all writes',
        action="store_true",
    )
    args = parser.parse_args()

    data_path = args.data_path
    cols = ["sentiment", "tweet_id", "date", "flag", "user_name", "text"]

    df = pd.read_csv(
        data_path.as_posix(),
        engine="python",
        names=cols,
        header=None,
        encoding="latin-1",
    )

    df = df.dropna(subset=["text"])

    df["sentiment"] = df["sentiment"].astype(np.int64)
    df["sentiment"] = df["sentiment"].map({0: 0, 4: 1})
    df = df.dropna(subset=["sentiment"])
    df["sentiment"] = df["sentiment"].astype(np.int64)

    df["text"] = df["text"].astype(str)
    df[["mentions", "clean_tweet"]] = df["text"].apply(
        lambda x: pd.Series(find_sobaka(x))
    )

    df["char"] = df["clean_tweet"].map(list)
    df["_seq_len"] = df["char"].map(len)

    save_mask = (df["_seq_len"] > LOWER_BOUND) & (df["_seq_len"] < UPPER_BOUND)
    df = df[save_mask].copy()

    filtered_text = "".join(df["clean_tweet"].tolist())
    char_counts = Counter(filtered_text)
    cat_codes = pd.DataFrame(char_counts.most_common(), columns=["char", "frequency"])
    cat_codes["_code"] = np.arange(1, len(cat_codes) + 1)
    cat_codes.index = cat_codes["char"]

    if args.cat_codes_path is not None:
        (args.cat_codes_path / "char").parent.mkdir(parents=True, exist_ok=True)
        cat_codes.to_parquet(args.cat_codes_path / "char", index=False)

    code_map = cat_codes["_code"].to_dict()
    df["char"] = df["char"].map(lambda x: [int(code_map[c]) for c in x])
    df["char_number"] = df["char"].map(lambda x: np.arange(len(x), dtype=np.float32))

    train_df, test_df = _stratified_split(df, TEST_FRACTION, args.split_seed)
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["shifts"] = [[-1] for _ in range(len(train_df))]
    test_df["shifts"] = [[-1] for _ in range(len(test_df))]

    test_df["post_amount"] = get_reg_target(test_df)
    test_df["post_anomaly_target"] = get_anomaly_target(test_df)
    test_df["post_target"] = get_clf_target(test_df)
    test_df["post_forecast_target"] = get_forecast_target(test_df)



    keep_cols = [
        "tweet_id",
        "char",
        "char_number",
        "shifts",
        "_seq_len",
        "post_anomaly_target",
        "post_amount",
        "post_target",
        "post_forecast_target"
    ]

    train_df = train_df[keep_cols]
    test_df = test_df[keep_cols]

    save_partitioned_parquet(train_df, args.save_path / "train", 20)
    save_partitioned_parquet(test_df, args.save_path / "test", 20)


if __name__ == "__main__":
    main()
