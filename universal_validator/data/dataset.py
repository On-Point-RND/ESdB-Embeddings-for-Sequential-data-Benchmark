from typing import Tuple

import pandas as pd
from data_types import DataConfig
import pyarrow.parquet as pq


class EmbeddingDataset:
    """AGE dataset implementation"""

    def __init__(self, data_conf: DataConfig):
        self.data_conf = data_conf

    def get_available_tasks(self, verbose=False):
        cols = pq.ParquetDataset(self.data_conf.train_path).schema.names
        target_cols = [col for col in cols if col.startswith("target__")]
        if verbose:
            print(f"Available tasks for {self.data_conf.dataset_name} dataset:")
        for col in target_cols:
            name, target_type, metric = self._parse_target_name(col)
            if verbose:
                print(f"Task Name: {name}, {target_type} type, Metric: {metric}| {col}")
        return target_cols

    def load_for_task(self, target_name) -> Tuple[pd.DataFrame, pd.DataFrame]:
        print(f"Loading {self.data_conf.dataset_name} dataset...")

        _, target_type, metric = self._parse_target_name(target_name)
        if target_type in ["local"]:
            columns = ["embeddings", target_name]
            train = pd.read_parquet(self.data_conf.train_path, columns=columns)
            test = pd.read_parquet(self.data_conf.train_path, columns=columns)
            X_train, y_train = train["embeddings"], train[target_name]
            X_test, y_test = test["embeddings"], test[target_name]
        print(f"Done!")
        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "metric": metric,
        }

    def _parse_target_name(self, target_name):
        parts = target_name.split("__")
        assert parts[0] == "target"
        assert len(parts) == 4
        name, target_type, metric = parts[1:]
        assert target_type in ["global", "local"], f"{target_type} is a wrong task type"
        return name, target_type, metric
