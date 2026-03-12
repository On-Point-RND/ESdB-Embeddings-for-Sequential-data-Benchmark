from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

from .common_pandas import (
    add_shift_columns,
    add_debug_f,
    global_time_split,
    save_partitioned_parquet,
    filter_short,
    shift_end_by_len,
    split_num_shifts,
    global_train_column,
    trim_test,
)

INDEX_COLUMNS = ["sequence_id"]
ORDERING_COLUMNS = ["time"]
NUM_FEATURES = ["sequence"]
TM = ORDERING_COLUMNS[0]


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
                    "time": list(range(len(sequence))),
                }
            )

    return pd.DataFrame(rows)


def get_forecast_target_row(row: pd.Series) -> list[float]:
    seq = np.asarray(row["sequence"])
    out = []
    for s in row["shifts"]:
        s = int(s)
        base = seq[s - 1]
        diff = seq[s] - base
        out.append(float(diff))
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
        default=30,
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
        TIME_TRAIN_SPLIT = 0.5
    else:
        TIME_TRAIN_SPLIT = 0.9
    USER_TRAIN_SPLIT = 0.9

    if not (0.0 < TIME_TRAIN_SPLIT < 1.0):
        parser.error("time_train_split must be in range (0, 1)")
    if not (0.0 < USER_TRAIN_SPLIT < 1.0):
        parser.error("user_train_split must be in range (0, 1)")
    time_test_split = 1 - TIME_TRAIN_SPLIT

    df = load_sequences(args.data_path)
    df["_seq_len"] = df[TM].map(len)
    df = filter_short(df)

    df["shift_end"] = shift_end_by_len(df[TM], -2)

    train_df, test_df = global_time_split(
        data=df,
        test_frac=time_test_split,
        time_col=TM,
        seqlen_col="_seq_len",
    )

    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df["shift_end"] = shift_end_by_len(train_df[TM], -2)

    assert (train_df["shift_end"] >= train_df["shift_start"]).all()
    assert (test_df["shift_end"] >= test_df["shift_start"]).all()
    train_n_shifts, test_n_shifts = split_num_shifts(args.num_shifts, time_test_split)
    test_df = add_shift_columns(test_df, test_n_shifts, args.shift_seed)
    train_df = add_shift_columns(train_df, train_n_shifts, args.shift_seed)

    if args.ntp:
        test_df = test_df.apply(trim_test, axis=1)
        test_df["_seq_len"] = test_df[TM].apply(len)

    train_df, test_df = global_train_column(
        train_df, test_df, USER_TRAIN_SPLIT, args.split_seed
    )
    test_df["target__forecast__local__r2"] = test_df.apply(
        get_forecast_target_row, axis=1
    )
    test_df["target__clf__global__accuracy+f1_macro"] = test_df["target"]
    train_df["target__forecast__local__r2"] = train_df.apply(
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
            "target__forecast__local__r2",
            "target__clf__global__accuracy+f1_macro",
            "global_train",
            "debug_f",
        ]
    )

    save_partitioned_parquet(
        train_df[keep_cols], args.save_path / "train", 20, mode=mode
    )
    save_partitioned_parquet(test_df[keep_cols], args.save_path / "test", 20, mode=mode)


if __name__ == "__main__":
    main()
