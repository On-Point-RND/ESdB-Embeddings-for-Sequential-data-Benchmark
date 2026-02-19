from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

from .common_pandas import (
    add_shift_columns,
    add_debug_f,
    filter_short,
    global_time_split,
    save_partitioned_parquet,
    shift_end_by_len,
    split_num_shifts,
)


INDEX_COLUMNS = ["sequence_id"]
ORDERING_COLUMNS = ["time"]
NUM_FEATURES = ["sequence"]
TM = ORDERING_COLUMNS[0]


def get_forecast_target_row(row: pd.Series) -> list[float]:
    seq = np.asarray(row["sequence"])
    out = []
    for s in row["shifts"]:
        s = int(s)
        if s >= len(seq) - 1:
            out.append(0.0)
            continue
        base = seq[s]
        diff = seq[s + 1 :] - base
        idx = np.where(diff != 0)[0]
        out.append(float(diff[idx[0]]) if len(idx) else 0.0)
    return out


def load_sequences(data_path: Path) -> pd.DataFrame:
    if data_path.is_dir():
        data_path = data_path / "ElectricDevices_TRAIN.txt"
    rows = []
    with data_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            values = np.fromstring(line.strip(), sep=" ", dtype=np.float64)
            sequence = values[1:].astype(np.float64).tolist()

            rows.append(
                {
                    "sequence_id": i,
                    "target": int(values[0]) - 1,
                    "sequence": sequence,
                    "time": [float(j + 1) for j in range(len(sequence))],
                }
            )

    return pd.DataFrame(rows)

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
        "--split-seed",
        help="Random seed for train-test split",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--num-shifts",
        help="How many shifts to sample per sequence",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--shift-seed",
        help="Random seed for shifts",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--time-train-split",
        help="Train fraction for global time split from 0 to 1",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--user-train-split",
        help="User train split from 0 to 1",
        type=float,
        default=0.9,
    )
    args = parser.parse_args()
    mode = "overwrite" if args.overwrite else "error"

    if not (0.0 < args.time_train_split < 1.0):
        parser.error("--time-train-split must be in range (0, 1)")
    if not (0.0 < args.user_train_split < 1.0):
        parser.error("--user-train-split must be in range (0, 1)")
    time_test_split = 1 - args.time_train_split
    user_train_split = args.user_train_split

    df = load_sequences(args.data_path)
    df["_seq_len"] = df[TM].map(len)
    df = filter_short(df)

    df["shift_end"] = shift_end_by_len(df[TM], -2)

    train_df, test_df = global_time_split(
        data=df,
        test_frac=time_test_split,
        min_shift_start=2,
        time_col="time",
        seqlen_col="_seq_len",
    )

    train_df = train_df.copy()
    test_df = test_df.copy()

    rng = np.random.default_rng(seed=args.split_seed)
    n_train_users = int(len(train_df.index) * user_train_split)
    train_indices = (
        rng.choice(train_df.index, size=n_train_users, replace=False).tolist()
        if n_train_users > 0
        else []
    )
    train_df["users_in_train"] = 0
    train_df.loc[train_indices, "users_in_train"] = 1
    valid_test_indices = test_df.index.intersection(train_indices)
    test_df["users_in_train"] = 0
    test_df.loc[valid_test_indices, "users_in_train"] = 1

    # recompute shift_end for train after trimming
    train_df["shift_end"] = shift_end_by_len(train_df[TM], -2)

    assert (train_df["shift_end"] >= train_df["shift_start"]).all()
    assert (test_df["shift_end"] >= test_df["shift_start"]).all()

    train_n_shifts, test_n_shifts = split_num_shifts(args.num_shifts, time_test_split)

    test_df = add_shift_columns(test_df, test_n_shifts, args.shift_seed)
    train_df = add_shift_columns(train_df, train_n_shifts, args.shift_seed)

    test_df["target__forecast__local__mse+r2"] = test_df.apply(
        get_forecast_target_row, axis=1
    )
    test_df["target__clf__global__accuracy+f1_macro"] = test_df["target"]
    train_df["target__forecast__local__mse+r2"] = train_df.apply(
        get_forecast_target_row, axis=1
    )
    train_df["target__clf__global__accuracy+f1_macro"] = train_df["target"]

    test_df = add_debug_f(test_df, time_col=TM)
    train_df = add_debug_f(train_df, time_col=TM)

    keep_cols = (
        INDEX_COLUMNS
        + ORDERING_COLUMNS
        + NUM_FEATURES
        + [
            "_seq_len",
            "shifts",
            "target__forecast__local__mse+r2",
            "target__clf__global__accuracy+f1_macro",
            "users_in_train",
            "debug_f",
        ]
    )

    save_partitioned_parquet(
        train_df[keep_cols], args.save_path / "train", 20, mode=mode
    )
    save_partitioned_parquet(test_df[keep_cols], args.save_path / "test", 20, mode=mode)


if __name__ == "__main__":
    main()
