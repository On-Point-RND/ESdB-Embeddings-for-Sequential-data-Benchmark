import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import EfficientFCParameters
from xgboost import XGBRegressor
import json
import gc 

from argparse import ArgumentParser
from pathlib import Path
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import LongType, FloatType
from pyspark.sql.window import Window

_TIME_COL_CANDIDATES = ["transaction_datetime", "datetime", "date_time", "timestamp", "trans_date", "time_from_first_trn", "date", "time"]


def detect_time_col(sdf):
    cols = sdf.columns
    for candidate in _TIME_COL_CANDIDATES:
        if candidate in cols:
            return candidate
    raise ValueError(
        f"Error. Time column not found in {cols}. Expected one of: {_TIME_COL_CANDIDATES}"
    )


def prepare_data_for_tsfresh(sdf, mode="global"):
    tc = detect_time_col(sdf)

    if mode == "local":
        if "shifts" not in sdf.columns:
            raise ValueError(f"Column 'shifts' not found. Available columns: {sdf.columns}")
        first_shift_len = F.col("shifts").getItem(0).cast("int")
        sdf = sdf.withColumn(tc, F.slice(F.col(tc), 1, first_shift_len))

    exploded = sdf.withColumn("dt_exp", F.explode(F.col(tc))) \
                  .select(
                      F.col("client_id").cast(LongType()),
                      F.col("dt_exp").cast(LongType()).alias("date_time")
                  )

    window = Window.partitionBy("client_id").orderBy("date_time")

    final_sdf = exploded \
        .withColumn("previous_dt", F.lag("date_time", 1).over(window)) \
        .withColumn("time_delta", (F.col("date_time") - F.col("previous_dt")).cast(FloatType())) \
        .fillna(0.0, subset=["time_delta"]) \
        .drop("previous_dt")\
        .filter(F.col("client_id").isNotNull())

    return final_sdf.sort("client_id", "date_time").toPandas()


def get_embeddings(df, params, scaler=None, train_columns=None):
    X = extract_features(
        df,
        column_id="client_id",
        column_sort="date_time",
        default_fc_parameters=params,
        impute_function=impute,
        n_jobs=15
    )

    if train_columns is not None:
        X = X.reindex(columns=train_columns, fill_value=0)

    X = X.replace([np.inf, -np.inf], 0).fillna(0)

    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return pd.DataFrame(X_scaled, index=X.index, columns=X.columns), scaler, X.columns
    else:
        X_scaled = scaler.transform(X)
        return pd.DataFrame(X_scaled, index=X.index, columns=X.columns), None, None


def get_top_n_features(X, y_df, n=25):
    common_idx = X.index.intersection(y_df.index)
    X_aligned = X.loc[common_idx]
    y_aligned = y_df.loc[common_idx]

    all_targets_top_features = {}

    for col_name in y_aligned.columns:
        y_curr = y_aligned[col_name].dropna()

        curr_common_idx = X_aligned.index.intersection(y_curr.index)
        if len(curr_common_idx) == 0:
            continue

        X_curr = X_aligned.loc[curr_common_idx]
        y_curr = y_curr.loc[curr_common_idx]

        if y_curr.empty:
            continue

        if isinstance(y_curr.iloc[0], (list, np.ndarray)):
            y_curr = y_curr.apply(lambda val: val[-1] if len(val) > 0 else 0)

        model = XGBRegressor(n_estimators=100, random_state=42, n_jobs=10)
        model.fit(X_curr, y_curr.astype(float))

        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1][:n]

        parts = str(col_name).split("__")
        clean_name = parts[1] if len(parts) >= 2 else str(col_name)

        all_targets_top_features[clean_name] = {
            str(X_curr.columns[i]): float(importances[i]) for i in idx
        }

    return all_targets_top_features


def process_and_save(df_main, df_local_emb, df_global_emb, is_train=True):
    local_map = {idx: row.values.astype(np.float32) for idx, row in df_local_emb.iterrows()}
    global_map = {idx: row.values.astype(np.float32) for idx, row in df_global_emb.iterrows()}

    local_dim = df_local_emb.shape[1]
    global_dim = df_global_emb.shape[1]

    local_target_cols = [c for c in df_main.columns if "__local__" in c]
    global_target_cols = [c for c in df_main.columns if "__global__" in c]

    shift_emb_list = []
    global_emb_list = []
    truncated_locals = {col: [] for col in local_target_cols}

    for _, row in df_main.iterrows():
        cid = row['client_id']

        seq_len = None
        for col in local_target_cols:
            val = row[col]
            if isinstance(val, (list, np.ndarray)):
                col_len = len(val)
                seq_len = col_len if seq_len is None else min(seq_len, col_len)

        if seq_len is None:
            seq_len = 1

        for col in local_target_cols:
            val = row[col]
            if isinstance(val, (list, np.ndarray)):
                truncated_locals[col].append(list(val[:seq_len]))
            else:
                fill_val = 0.0
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    fill_val = val
                truncated_locals[col].append([fill_val] * seq_len)

        base_local = local_map.get(cid, np.zeros(local_dim, dtype=np.float32))
        shift_emb_list.append([base_local.tolist() for _ in range(seq_len)])

        base_global = global_map.get(cid, np.zeros(global_dim, dtype=np.float32))
        global_emb_list.append(base_global.tolist())

    result = df_main.copy()
    result['shift_emb'] = shift_emb_list
    result['global_emb'] = global_emb_list

    for col in local_target_cols:
        result[col] = truncated_locals[col]

    for col in global_target_cols:
        result[col] = result[col].apply(
            lambda val: val[0] if isinstance(val, (list, np.ndarray)) and len(val) == 1 else val
        )

    result["global_train"] = 1 if is_train else 0

    return result


def main():
    parser = ArgumentParser()
    parser.add_argument("--data-path", required=True, type=Path)
    parser.add_argument("--dataset-name", required=True, type=str)
    args = parser.parse_args()

    spark = (SparkSession.builder
             .master("local[20]")
             .appName(f"{args.dataset_name}_tsfresh_embeddings")
             .config("spark.driver.memory", "100g")
             .config("spark.driver.maxResultSize", "50g")
             .getOrCreate())

    train_sdf = spark.read.parquet((args.data_path / "train").as_posix())
    test_sdf = spark.read.parquet((args.data_path / "test").as_posix())
    ts_params = EfficientFCParameters()

    train_pd = train_sdf.toPandas()
    test_pd = test_sdf.toPandas()

    train_targets_idx = train_pd.set_index("client_id")
    target_cols = [c for c in train_pd.columns if c.startswith("target__")]

    global_targets = [c for c in target_cols if "__global__" in c]
    local_targets = [c for c in target_cols if "__local__" in c]

    print(f"Global targets: {global_targets}")
    print(f"Local targets: {local_targets}")

    train_local_pd = prepare_data_for_tsfresh(train_sdf, "local")
    train_global_pd = prepare_data_for_tsfresh(train_sdf, "global")

    emb_train_local, scaler_l, cols_l = get_embeddings(train_local_pd, ts_params)
    emb_train_global, scaler_g, cols_g = get_embeddings(train_global_pd, ts_params)

    del train_local_pd
    del train_global_pd
    gc.collect()

    test_local_pd = prepare_data_for_tsfresh(test_sdf, "local")
    test_global_pd = prepare_data_for_tsfresh(test_sdf, "global")

    emb_test_local, _, _ = get_embeddings(test_local_pd, ts_params, scaler=scaler_l, train_columns=cols_l)
    emb_test_global, _, _ = get_embeddings(test_global_pd, ts_params, scaler=scaler_g, train_columns=cols_g)

    del test_local_pd
    del test_global_pd
    gc.collect()

    top_features = {}
    if global_targets:
        top_features["global"] = get_top_n_features(
            emb_train_global, train_targets_idx[global_targets]
        )
    if local_targets:
        top_features["local"] = get_top_n_features(
            emb_train_local, train_targets_idx[local_targets]
        )

    out_dir = args.data_path / f"out_file_{args.dataset_name}_time_delta"
    out_dir.mkdir(exist_ok=True)
    name = args.dataset_name
    chunk_size = 1000

    train_dir = out_dir / "train"
    test_dir = out_dir / "test"
    combined_dir = out_dir / "combined"
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    combined_dir.mkdir(exist_ok=True)

    combined_chunk_idx = 0

    for i in range(0, len(train_pd), chunk_size):
        chunk_pd = train_pd.iloc[i:i + chunk_size].copy()
        chunk_final = process_and_save(chunk_pd, emb_train_local, emb_train_global, is_train=True)
        
        chunk_final.to_parquet(train_dir / f"{name}_tsfresh_train_{i}.parquet", index=False, engine='pyarrow')
        chunk_final.to_parquet(combined_dir / f"{name}_tsfresh_combined_{combined_chunk_idx}.parquet", index=False, engine='pyarrow')
        
        combined_chunk_idx += 1
        del chunk_pd, chunk_final
        gc.collect()

    del train_pd, emb_train_local, emb_train_global
    gc.collect()

    for i in range(0, len(test_pd), chunk_size):
        chunk_pd = test_pd.iloc[i:i + chunk_size].copy()
        chunk_final = process_and_save(chunk_pd, emb_test_local, emb_test_global, is_train=False)
        
        chunk_final.to_parquet(test_dir / f"{name}_tsfresh_test_{i}.parquet", index=False, engine='pyarrow')
        chunk_final.to_parquet(combined_dir / f"{name}_tsfresh_combined_{combined_chunk_idx}.parquet", index=False, engine='pyarrow')
        
        combined_chunk_idx += 1
        del chunk_pd, chunk_final
        gc.collect()

    del test_pd, emb_test_local, emb_test_global
    gc.collect()

    with open(out_dir / f"top_25_time_delta_tsfresh_features.json", "w") as f:
        json.dump(top_features, f, indent=4)

    print(f"Done! Files saved to {out_dir}")

if __name__ == "__main__":
    main()
