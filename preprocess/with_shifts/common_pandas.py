from pathlib import Path
import shutil

import numpy as np
import pandas as pd

MIN_SHIFT_START = 2
MIN_SEQ_LEN = 2


def trim_test(row):
    start_idx = int(row["shift_start"])
    for col_name in row.index:
        if col_name in ["shifts", "shift_start", "shift_end"]:
            continue
        val = row[col_name]
        if isinstance(val, (list, np.ndarray)):
            row[col_name] = val[start_idx:]
    old_shifts = np.array(row["shifts"])
    new_shifts = old_shifts - start_idx
    row["shifts"] = new_shifts[new_shifts >= 0].tolist()
    if not row["shifts"]:
        row["shifts"] = [0]
    return row


def global_train_column(
    train_df, test_df, train_ratio, seed
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # User split with configurable train fraction.
    rng = np.random.default_rng(seed=seed)
    n_train_users = int(len(train_df.index) * train_ratio)
    train_indices = rng.choice(
        train_df.index, size=n_train_users, replace=False
    ).tolist()
    train_df["global_train"] = 0
    train_df.loc[train_indices, "global_train"] = 1
    valid_test_indices = test_df.index.intersection(train_indices)
    test_df["global_train"] = 0
    test_df.loc[valid_test_indices, "global_train"] = 1
    return train_df, test_df


def save_partitioned_parquet(
    df: pd.DataFrame,
    save_path: Path,
    num_shards: int,
    mode: str = "error",
) -> None:
    if mode not in {"error", "overwrite"}:
        raise ValueError(f"Unsupported mode: {mode}")

    if save_path.exists():
        if mode == "overwrite":
            if save_path.is_dir():
                shutil.rmtree(save_path)
            else:
                save_path.unlink()
        else:
            raise FileExistsError(
                f"Output path already exists: {save_path}. "
                "Use overwrite mode to replace existing data."
            )

    df = df.copy()
    df["shard"] = np.arange(len(df)) % num_shards
    save_path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(save_path, partition_cols=["shard"], engine="pyarrow", index=False)


def sample_users(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    sample_frac: float,
    seed: int,
    index_col: str = "client_id",
    split_name: str = "global_time_split",
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if not (0.0 < sample_frac <= 1.0):
        raise ValueError("sample_frac must be in range (0, 1]")
    if split_name not in {"global_time_split", "user_split"}:
        raise ValueError(f"Unknown split_name: {split_name}")
    if split_name == "global_time_split" and test_df is None:
        raise ValueError("test_df must be provided for global time split")
    if split_name == "user_split" and test_df is not None:
        raise ValueError("test_df must be None for user split")
    if sample_frac == 1.0:
        return train_df.copy(), None if test_df is None else test_df.copy()

    user_ids = np.sort(train_df[index_col].drop_duplicates().to_numpy())
    n_users = max(1, int(np.floor(len(user_ids) * sample_frac)))
    rng = np.random.default_rng(seed)
    sampled_users = rng.choice(user_ids, size=n_users, replace=False).tolist()

    train_sampled = train_df.loc[train_df[index_col].isin(sampled_users)].copy()
    if test_df is None:
        return train_sampled, None

    test_sampled = test_df.loc[test_df[index_col].isin(sampled_users)].copy()
    return train_sampled, test_sampled


def filter_short(
    df: pd.DataFrame,
    min_len: int = MIN_SEQ_LEN,
    seqlen_col: str = "_seq_len",
) -> pd.DataFrame:
    if seqlen_col not in df.columns:
        raise KeyError(f"Missing column: {seqlen_col}")
    return df[df[seqlen_col] > min_len].copy()


def split_num_shifts(num_shifts: int, test_frac: float) -> tuple[int, int]:
    test_n = int(np.ceil(test_frac * num_shifts))
    train_n = int(np.floor((1 - test_frac) * num_shifts))
    return max(1, train_n), max(1, test_n)


def shift_end_by_len(series: pd.Series, offset: int) -> pd.Series:
    return series.map(lambda x: len(x) + offset)


def add_debug_f(
    df: pd.DataFrame,
    time_col: str,
    shifts_col: str = "shifts",
    out_col: str = "debug_f",
) -> pd.DataFrame:
    df[out_col] = df.apply(
        lambda r: [r[time_col][int(s)] for s in r[shifts_col]],
        axis=1,
    )
    return df


def datetime_preproc(
    df: pd.DataFrame,
    time_col: str,
    loc: str | np.datetime64,
    scale: tuple[int, str] | np.timedelta64,
) -> pd.DataFrame:
    if isinstance(loc, str):
        loc = np.datetime64(loc)
    if isinstance(scale, (tuple, list)):
        scale = np.timedelta64(*scale)

    df[time_col] = df[time_col].apply(
        lambda x: ((np.asarray(x, dtype="datetime64[ns]") - loc) / scale).tolist()
    )
    return df


def rescale_preproc(
    df: pd.DataFrame,
    rescale_values: dict[str, float],
) -> pd.DataFrame:
    for feature, max_value in rescale_values.items():
        df[feature] = df[feature].apply(
            lambda x: (np.asarray(x, dtype=np.float32) / max_value)
            .astype(np.float32)
            .tolist()
        )
    return df


def log_preproc(
    df: pd.DataFrame,
    log_features: list[str],
) -> pd.DataFrame:
    for feature in log_features:
        df[feature] = df[feature].apply(
            lambda x: (
                np.log1p(np.abs(np.asarray(x))) * np.sign(np.asarray(x))
            ).tolist()
        )
    return df


def transform_train_test_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    rescale_features: list[str] | None = None,
    log_features: list[str] | None = None,
    time_col: str | None = None,
    datetime_loc: str | np.datetime64 | None = None,
    datetime_scale: tuple[int, str] | np.timedelta64 | None = None,
    rescale_q: float = 0.99,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.copy()
    test_df = test_df.copy()

    if time_col is not None and datetime_loc is not None and datetime_scale is not None:
        train_df = datetime_preproc(train_df, time_col, datetime_loc, datetime_scale)
        test_df = datetime_preproc(test_df, time_col, datetime_loc, datetime_scale)

    rescale_features = rescale_features or []
    if rescale_features:

        rescale_values = {}
        for feature in rescale_features:
            train_vals = train_df[feature].explode()
            test_vals = test_df[feature].explode()
            q_values = np.asarray(
                [train_vals.quantile(rescale_q), test_vals.quantile(rescale_q)],
                dtype=np.float32,
            )
            if np.isnan(q_values).all():
                raise ValueError(f"NaN rescale quantile for feature: {feature}")

            max_value = np.float32(np.nanmax(q_values))
            if max_value == 0:
                continue
            rescale_values[feature] = max_value

        train_df = rescale_preproc(train_df, rescale_values)
        test_df = rescale_preproc(test_df, rescale_values)

    log_features = log_features or []
    if log_features:
        train_df = log_preproc(train_df, log_features)
        test_df = log_preproc(test_df, log_features)

    return train_df, test_df


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
    num_shifts: int,
    start_col: str = "shift_start",
    end_col: str = "shift_end",
) -> list[int]:
    lo = int(row[start_col])
    hi = int(row[end_col])
    if hi < lo:
        return []
    vals = np.arange(lo, hi + 1)
    num_shifts = min(len(vals), num_shifts)
    shifts = rng.choice(vals, size=num_shifts, replace=False)
    return np.sort(shifts).astype(int).tolist()


def add_shift_columns(
    df: pd.DataFrame,
    num_shifts: int,
    seed: int,
    shifts_col: str = "shifts",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df[shifts_col] = df.apply(lambda r: sample_shifts(r, rng, num_shifts), axis=1)
    return df


def first_true(m):
    idx = np.flatnonzero(m)
    return int(idx[0]) if len(idx) else -1


def global_time_split(
    data: pd.DataFrame,
    test_frac: float,
    min_shift_start: int = MIN_SHIFT_START,
    time_col: str = "trans_date",
    seqlen_col: str = "_seq_len",
) -> tuple[pd.DataFrame, pd.DataFrame]:

    trans_date = data[time_col].map(np.asarray)
    shift_end_full = data["shift_end"]  # index of last valid element

    valid_times = np.concatenate(
        [a[: int(e) + 1] for a, e in zip(trans_date, shift_end_full) if e >= 0]
    )
    assert len(valid_times) > 0, "No valid times for split"

    T = np.quantile(valid_times, 1 - test_frac)
    is_test = trans_date.map(lambda x: x >= T)

    train = data.copy()
    array_cols = [
        c for c in train.columns if isinstance(train[c].iloc[0], (list, np.ndarray))
    ]

    def trim_row(row):
        mask = np.asarray(row[time_col]) < T
        for c in array_cols:
            row[c] = np.asarray(row[c])[mask]
        return row

    train = pd.DataFrame(train.apply(trim_row, axis=1, result_type="expand"))
    train = train[train[time_col].apply(lambda x: x.ndim > 0)].copy()
    train = train[train[time_col].map(len) > 0].copy()

    train["shift_start"] = min_shift_start

    test_mask = pd.Series(
        [
            np.any(m[: e + 1]) if e >= 0 else False
            for m, e in zip(is_test, shift_end_full)
        ],
        index=data.index,
    )

    test = data[test_mask].copy()

    test["shift_start"] = is_test.loc[test.index].map(
        lambda m: max(first_true(m), min_shift_start)
    )

    train[seqlen_col] = train[time_col].map(len)
    test[seqlen_col] = test[time_col].map(len)

    return train, test


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
