from pathlib import Path

import numpy as np
import pandas as pd


def save_partitioned_parquet(df: pd.DataFrame, save_path: Path, num_shards: int) -> None:
    df = df.copy()
    df["shard"] = np.arange(len(df)) % num_shards
    save_path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(save_path, partition_cols=["shard"], engine="pyarrow")


def pandas_train_test_split(
    df: pd.DataFrame,
    test_frac: float,
    index_col: str,
    stratify_col: str,
    stratify_col_vals: list[int],
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert stratify_col_vals, "stratify_col_vals must be a non-empty list"
    rng = np.random.default_rng(random_seed)

    base = df[[index_col, stratify_col]].drop_duplicates()
    test_ids_list = []
    for val in stratify_col_vals:
        ids = base.loc[base[stratify_col] == val, index_col].to_numpy()
        if len(ids) == 0:
            continue
        size = int(np.ceil(len(ids) * test_frac))
        pick = rng.choice(ids, size=size, replace=False)
        test_ids_list.append(pick)
    if test_ids_list:
        test_ids = np.concatenate(test_ids_list)
    else:
        test_ids = np.empty(0, dtype=np.int64)

    test_mask = df[index_col].isin(test_ids.tolist())
    test_df = df.loc[test_mask]
    train_df = df.loc[~test_mask]
    return train_df, test_df


def sample_shifts(
    row: pd.Series,
    rng: np.random.Generator,
    n: int,
    start_col: str = "shift_start",
    end_col: str = "shift_end",
) -> list[int]:
    lo = int(row[start_col])
    hi = int(row[end_col])
    if hi < lo:
        return []
    vals = np.arange(lo, hi + 1)
    replace = n > len(vals)
    shifts = rng.choice(vals, size=n, replace=replace)
    return np.sort(shifts).tolist()


def add_shift_columns(
    df: pd.DataFrame,
    shift_start,
    shift_end,
    num_shifts: int,
    seed: int,
    start_col: str = "shift_start",
    end_col: str = "shift_end",
    shifts_col: str = "shifts",
) -> pd.DataFrame:
    df = df.copy()
    df[start_col] = shift_start
    df[end_col] = shift_end
    rng = np.random.default_rng(seed)
    df[shifts_col] = df.apply(
        lambda r: sample_shifts(r, rng, num_shifts, start_col=start_col, end_col=end_col),
        axis=1,
    )
    return df


def fill_train_targets(
    df: pd.DataFrame,
    cols: list[str],
    value=-1,
) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        df[col] = [[value] for _ in range(len(df))]
    return df


def duplicate_target_by_shifts(
    df: pd.DataFrame,
    col_name: str,
    shifts_col: str = "shifts",
) -> pd.Series:
    return df.apply(lambda r: [int(r[col_name])] * len(r[shifts_col]), axis=1)
