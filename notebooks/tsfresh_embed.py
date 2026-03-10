import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tsfresh import extract_features
from tsfresh.feature_selection import select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import EfficientFCParameters
from xgboost import XGBRegressor
import json

from argparse import ArgumentParser
from pathlib import Path
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import LongType, FloatType
from pyspark.sql.window import Window

def prepare_data_for_tsfresh(sdf, mode="global"):
    if mode == "local":
        first_shift_len = F.col("shifts").getItem(0).cast("int")
        sdf = sdf.withColumn("trans_date", F.slice(F.col("trans_date"), 1, first_shift_len))

    exploded = sdf.withColumn("trans_date_exp", F.explode(F.col("trans_date"))) \
                  .select(
                      F.col("client_id").cast(LongType()),
                      F.col("trans_date_exp").cast(LongType()).alias("trans_date")
                  )
    
    window = Window.partitionBy("client_id").orderBy("trans_date")

    final_sdf = exploded.withColumn("previous_date", F.lag("trans_date", 1).over(window)) \
                        .withColumn("time_delta", (F.col("trans_date") - F.col("previous_date")).cast(FloatType())) \
                        .fillna(0.0, subset=["time_delta"]) \
                        .drop("previous_date")

    return final_sdf.sort("client_id", "trans_date").toPandas()


def _select_features_for_target(X, y_series):
    common_idx = X.index.intersection(y_series.index)
    X_common = X.loc[common_idx]
    y_common = y_series.loc[common_idx].astype(float)
    
    if len(y_common.unique()) <= 1:
        return set()
        
    try:
        selected_df = select_features(X_common, y_common)
        return set(selected_df.columns)
    except Exception:
        return set()

def get_embeddings(df, y_df, params, scaler=None, train_columns=None):
    X = extract_features(
        df, column_id="client_id", column_sort="trans_date",
        default_fc_parameters=params, impute_function=impute, n_jobs=8
    )
    
    if scaler is None:
        all_selected_features = set()
        
        for col_name in y_df.columns:
            y_col = y_df[col_name].dropna()
            if y_col.empty:
                continue

            first_val = y_col.iloc[0]
            
            if isinstance(first_val, (list, np.ndarray)):
                max_len = y_col.apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0).max()
                
                for i in range(max_len):
                    y_extracted = y_col.apply(lambda x: x[i] if isinstance(x, (list, np.ndarray)) and len(x) > i else np.nan)
                    y_clean = y_extracted.dropna()
                    
                    if not y_clean.empty:
                        selected_cols = _select_features_for_target(X, y_clean)
                        all_selected_features.update(selected_cols)
            else:
                selected_cols = _select_features_for_target(X, y_col)
                all_selected_features.update(selected_cols)

        final_columns = list(all_selected_features)
        if not final_columns:
            final_columns = X.columns.tolist()

        X = X[final_columns]
        X = X.replace([np.inf, -np.inf], 0).fillna(0)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return pd.DataFrame(X_scaled, index=X.index, columns=X.columns), scaler, X.columns
    else:
        if train_columns is not None:
            X = X.reindex(columns=train_columns, fill_value=0)
        
        X = X.replace([np.inf, -np.inf], 0).fillna(0)
        X_scaled = scaler.transform(X)
        return pd.DataFrame(X_scaled, index=X.index, columns=X.columns)

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

        model = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
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

    def transform_row(row):
        cid = row['client_id']

        local_cols = [c for c in row.index if "__local__" in c]
        if local_cols:
            ref_val = row[local_cols[0]]
            seq_len = len(ref_val) if isinstance(ref_val, (list, np.ndarray)) else 1
        else:
            seq_len = 1

        base_local = local_map.get(cid, np.zeros(local_dim, dtype=np.float32))
        row['shift_emb'] = [base_local for _ in range(seq_len)]
        row['global_emb'] = global_map.get(cid, np.zeros(global_dim, dtype=np.float32))
        global_target_cols = [c for c in row.index if "__global__" in c]
        for col in global_target_cols:
            val = row[col]
            if isinstance(val, (list, np.ndarray)) and len(val) == 1:
                row[col] = val[0]
        
        return row

    df_processed = df_main.apply(transform_row, axis=1)
    
    if "global_train" not in df_processed.columns:
        df_processed["global_train"] = 1 if is_train else 0
        
    return df_processed

def main():
    parser = ArgumentParser()
    parser.add_argument("--data-path", required=True, type=Path)
    args = parser.parse_args()

    spark = (SparkSession.builder.master("local[*]")
             .appName("AGE_tsfresh_embed")
             .config("spark.driver.memory", "16g")
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

    train_local_pd = prepare_data_for_tsfresh(train_sdf, "local")
    train_global_pd = prepare_data_for_tsfresh(train_sdf, "global")
    
    emb_train_local, scaler_l, cols_l = get_embeddings(
        train_local_pd, 
        train_targets_idx[local_targets] if local_targets else pd.DataFrame(), 
        ts_params
    )
    
    emb_train_global, scaler_g, cols_g = get_embeddings(
        train_global_pd, 
        train_targets_idx[global_targets] if global_targets else pd.DataFrame(), 
        ts_params
    )

    test_local_pd = prepare_data_for_tsfresh(test_sdf, "local")
    test_global_pd = prepare_data_for_tsfresh(test_sdf, "global")
    
    emb_test_local = get_embeddings(test_local_pd, pd.DataFrame(), ts_params, scaler=scaler_l, train_columns=cols_l)
    emb_test_global = get_embeddings(test_global_pd, pd.DataFrame(), ts_params, scaler=scaler_g, train_columns=cols_g)

    top_features = {}
    if global_targets:
        top_features["global"] = get_top_n_features(emb_train_global, train_targets_idx[global_targets])
    if local_targets:
        top_features["local"] = get_top_n_features(emb_train_local, train_targets_idx[local_targets])

    train_final = process_and_save(train_pd, emb_train_local, emb_train_global, is_train=True)
    test_final = process_and_save(test_pd, emb_test_local, emb_test_global, is_train=False)

    out_dir = args.data_path / "out_file_age_time_delta"
    out_dir.mkdir(exist_ok=True)
    
    train_final.to_parquet(out_dir / "age_tsfresh_embed_time_delta_train.parquet", index=False)
    test_final.to_parquet(out_dir / "age_tsfresh_embed_time_delta_test.parquet", index=False)

    with open(out_dir / "top_25_time_delta_tsfresh_features.json", "w") as f:
        json.dump(top_features, f, indent=4)

if __name__ == "__main__":
    main()
