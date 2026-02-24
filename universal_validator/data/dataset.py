from typing import Tuple

import pandas as pd
import numpy as np
import pyarrow.parquet as pq

from dataclasses import dataclass
from ..tasks.scorers import METRIC_INFO, TaskType

@dataclass(frozen=True)
class DataConfig:
    dataset_name: str = "taobao"
    train_path: str = ""
    test_path: str = ""


@dataclass
class DataSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    metrics: list[str]
    task_name: str


class ValidatorDataset:
    """AGE dataset implementation"""

    def __init__(self, data_conf: DataConfig):
        self.data_conf = data_conf

    def get_available_tasks(self, verbose=False):
        cols = pq.ParquetDataset(self.data_conf.train_path).schema.names
        target_cols = [col for col in cols if col.startswith("target__")]
        if not verbose:
            return target_cols

        print(f"Available tasks for {self.data_conf.dataset_name} dataset:")
        for col in target_cols:
            name, target_type, metrics = self._parse_target_name(col)
            print(
                f"Task Name: {name},\t{target_type} type,\tMetrics: {metrics}\t| {col}"
            )
        return target_cols

    def load_for_task(self, target_name) -> DataSplit:
        print(f"Loading {self.data_conf.dataset_name} dataset...")
        _, target_type, metrics = self._parse_target_name(target_name)
        if target_type in ["local"]:
            columns = ["shift_emb", target_name]
            train = pd.read_parquet(self.data_conf.train_path, columns=columns)
            test = pd.read_parquet(self.data_conf.test_path, columns=columns)
            train, test = train.explode(columns), test.explode(columns)

            X_train = np.stack(train["shift_emb"].values).astype(np.float32)
            y_train = train[target_name].values

            X_test = np.stack(test["shift_emb"].values).astype(np.float32)
            y_test = test[target_name].values
        elif target_type == "global":
            columns = ["global_emb", "global_train", target_name]
            data = pd.read_parquet(self.data_conf.train_path, columns=columns)
            # TODO for user-wise split test should be loaded directly
            train, test = data[data["global_train"] == 1], data[data["global_train"] == 0]

            X_train = np.stack(train["global_emb"].values).astype(np.float32)
            y_train = train[target_name].values

            X_test = np.stack(test["global_emb"].values).astype(np.float32)
            y_test = test[target_name].values
        else:
            raise NotImplementedError(f"Target type {target_type} not supported!")
        assert metrics[0] in METRIC_INFO, f"{metrics[0]} not supported!"
        if METRIC_INFO[metrics[0]]["task"] == TaskType.REGRESSION:
            y_train, y_test = y_train.astype(np.float32), y_test.astype(np.float32)
        elif METRIC_INFO[metrics[0]]["task"] == TaskType.CLASSIFICATION:
            y_train, y_test = y_train.astype(np.int32), y_test.astype(np.int32)
        else:
            raise ValueError(METRIC_INFO[metrics[0]]["task"])

        print(f"Done!")
        return DataSplit(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            metrics=metrics,
            task_name=target_name,
        )

    @staticmethod
    def _parse_target_name(target_name):
        parts = target_name.split("__")
        assert parts[0] == "target"
        assert len(parts) == 4, parts
        name, target_type, metrics = parts[1:]
        assert target_type in ["global", "local"], f"{target_type} is a wrong task type"
        metrics = metrics.split("+")
        return name, target_type, metrics
