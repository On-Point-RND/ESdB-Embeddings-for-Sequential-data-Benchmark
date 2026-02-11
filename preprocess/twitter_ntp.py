from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


CAT_FEATURES = ["char"]
TARGET_VALS = [0, 1, 2]
TEST_FRACTION = 0.5
MIN_LEN = 160
MAX_LEN = 128


def _resolve_data_path(data_path: Path) -> Path:
    if data_path.is_dir():
        if (data_path / "twitter_sentiment_dataset.csv").exists():
            return data_path / "twitter_sentiment_dataset.csv"
        if (data_path / "twitter_dataset.csv").exists():
            return data_path / "twitter_dataset.csv"
        raise ValueError("No known twitter CSV found in data-path directory")
    return data_path


def _map_sentiment(series: pd.Series) -> pd.Series:
    mapped = series.map(
        {
            "Negative": 0,
            "Neutral": 1,
            "Positive": 2,
        }
    )
    return mapped.fillna(pd.to_numeric(series, errors="coerce"))


def _stratified_split(df: pd.DataFrame, test_frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def _save_partitioned_parquet(df: pd.DataFrame, save_path: Path, num_shards: int) -> None:
    df = df.copy()
    df["shard"] = np.arange(len(df)) % num_shards
    save_path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(save_path, partition_cols=["shard"], engine="pyarrow")


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

    data_path = _resolve_data_path(args.data_path)
    df = pd.read_csv(data_path.as_posix(), engine="python")

    text_col = "Tweet" if "Tweet" in df.columns else "Text" if "Text" in df.columns else None
    if text_col is None:
        raise ValueError("Text column not found. Expected 'Tweet' or 'Text'")
    if "Sentiment" not in df.columns:
        raise ValueError("Sentiment column not found in twitter dataset")

    df = df.rename(columns={text_col: "text"})
    df = df.dropna(subset=["text"])

    if "Tweet_ID" in df.columns:
        df["tweet_id"] = pd.to_numeric(df["Tweet_ID"], errors="coerce")
        df = df.dropna(subset=["tweet_id"])
        df["tweet_id"] = df["tweet_id"].astype(np.int64)
    else:
        df["tweet_id"] = np.arange(len(df), dtype=np.int64)

    df["sentiment"] = _map_sentiment(df["Sentiment"])
    df = df.dropna(subset=["sentiment"])
    df["sentiment"] = df["sentiment"].astype(np.int64)

    retweets_src = df["Retweets"] if "Retweets" in df.columns else pd.Series(0, index=df.index)
    likes_src = df["Likes"] if "Likes" in df.columns else pd.Series(0, index=df.index)
    df["retweets"] = pd.to_numeric(retweets_src, errors="coerce").fillna(0).astype(np.int64)
    df["likes"] = pd.to_numeric(likes_src, errors="coerce").fillna(0).astype(np.int64)

    df["text"] = df["text"].astype(str)
    df["char"] = df["text"].map(list)
    df["_seq_len"] = df["char"].map(len)
    breakpoint()
    df = df[df["_seq_len"] >= MIN_LEN].copy()

    filtered_text = "".join(df["text"].tolist())
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

    df["used_in_train"] = 0
    df["post_char"] = df["char"].map(lambda x: x[MAX_LEN:])
    df["post_char_number"] = df["char_number"].map(lambda x: x[MAX_LEN:])
    df["char"] = df["char"].map(lambda x: x[:MAX_LEN])
    df["char_number"] = df["char_number"].map(lambda x: x[:MAX_LEN])
    df["_seq_len"] = MAX_LEN

    train_df, test_df = _stratified_split(df, TEST_FRACTION, args.split_seed)
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["used_in_train"] = 1
    test_df["used_in_train"] = 0

    keep_cols = [
        "tweet_id",
        "char",
        "char_number",
        "sentiment",
        "likes",
        "retweets",
        "_seq_len",
        "post_char",
        "post_char_number",
        "used_in_train"
    ]
    train_df = train_df[keep_cols]
    test_df = test_df[keep_cols]

    if args.which_split in ("train", "union"):
        _save_partitioned_parquet(train_df, args.save_path / "train", 20)
    if args.which_split in ("test", "union"):
        _save_partitioned_parquet(test_df, args.save_path / "test", 3)

    full_base_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    full_base_df["_seq_len"] = full_base_df["post_char"].map(len) + MAX_LEN
    _save_partitioned_parquet(full_base_df, args.save_path / "full_ntp", 3)


if __name__ == "__main__":
    main()
