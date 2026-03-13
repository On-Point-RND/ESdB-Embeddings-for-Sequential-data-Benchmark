from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .common_pandas import filter_short, save_partitioned_parquet

CAT_FEATURES = ["char"]
INDEX_COLUMNS = ["tweet_id"]
ORDERING_COLUMNS = ["char_number"]


def get_anomaly_target(data: pd.DataFrame) -> pd.Series:
    punct_ratio = data["clean_tweet"].astype(str).map(punctuation_ratio)
    return (punct_ratio > 0.1).astype(np.int64)


def punctuation_ratio(s: str) -> float:
    if not s:
        return 0.0
    punct = sum(1 for ch in s if ch in "!?.,;:")
    return punct / len(s)


def find_mentions(text: str) -> tuple[int, str]:
    count = text.count("@")
    cleaned = text.replace("@", "")
    return count, cleaned


def stratified_split_column(
    df: pd.DataFrame,
    label_col: str,
    train_ratio: float,
    seed: int,
) -> pd.Series:
    rng = np.random.default_rng(seed)
    global_train = pd.Series(0, index=df.index, dtype=np.int8)

    for _, idx in df.groupby(label_col).groups.items():
        idx = np.asarray(idx)
        size = int(np.ceil(len(idx) * train_ratio))
        global_train.loc[rng.choice(idx, size=size, replace=False)] = 1

    return global_train


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
        help="Random seed for train-test split",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--overwrite",
        help='Toggle "overwrite" mode on all spark writes',
        action="store_true",
    )
    parser.add_argument(
        "--num-shifts",
        help="How many shifts to sample per sequence",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--shift-seed",
        help="Random seed for shifts",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--ntp",
        help="Whether to use splitting for NTP",
        action="store_true",
    )
    args = parser.parse_args()
    mode = "overwrite" if args.overwrite else "error"

    if args.ntp:
        USER_TRAIN_SPLIT = 0.5
    else:
        USER_TRAIN_SPLIT = 0.9

    if not (0.0 < USER_TRAIN_SPLIT < 1.0):
        parser.error("user_train_split must be in range (0, 1)")

    cols = ["sentiment", "tweet_id", "date", "flag", "user_name", "text"]
    csv_path = args.data_path / "training.1600000.processed.noemoticon.csv"
    df = pd.read_csv(
        csv_path.as_posix(),
        engine="python",
        names=cols,
        header=None,
        encoding="latin-1",
    )

    df = df.dropna(subset=["text"])
    df["sentiment"] = df["sentiment"].astype(np.int64).map({0: 0, 4: 1})
    df = df.dropna(subset=["sentiment"])
    df["sentiment"] = df["sentiment"].astype(np.int64)

    df["text"] = df["text"].astype(str)
    df[["mentions", "clean_tweet"]] = df["text"].apply(
        lambda x: pd.Series(find_mentions(x))
    )

    df["char"] = df["clean_tweet"].map(list)
    df["_seq_len"] = df["char"].map(len)
    df = filter_short(df)

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

    df["global_train"] = stratified_split_column(
        df=df,
        label_col="sentiment",
        train_ratio=USER_TRAIN_SPLIT,
        seed=args.split_seed,
    )

    # Keep compatibility with the with_shifts data format.
    df["shifts"] = [[-1] for _ in range(len(df))]

    df["target__clf__global__accuracy+f1_macro"] = df["sentiment"]
    df["target__reg_mentions__global__r2"] = df["mentions"]
    df["target__anomaly__global__roc_auc+f1_macro+accuracy"] = get_anomaly_target(df)

    keep_cols = (
        INDEX_COLUMNS
        + ORDERING_COLUMNS
        + CAT_FEATURES
        + [
            "_seq_len",
            "shifts",
            "global_train",
            "target__clf__global__accuracy+f1_macro",
            "target__reg_mentions__global__r2",
            "target__anomaly__global__roc_auc+f1_macro+accuracy",
        ]
    )

    save_partitioned_parquet(df[keep_cols], args.save_path / "train", 20, mode=mode)


if __name__ == "__main__":
    main()
