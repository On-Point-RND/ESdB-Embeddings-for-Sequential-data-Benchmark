import logging

import pandas as pd
import numpy as np
import pyarrow.parquet as pq

from dataclasses import dataclass
from ..tasks.scorers import METRIC_INFO, TaskType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataConfig:
    dataset_name: str = "age"
    train_path: str = ""
    test_path: str = ""


@dataclass
class DataSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray | None
    y_test: np.ndarray | None
    metrics: list[str]
    task_name: str
    X_val: np.ndarray | None = None
    y_val: np.ndarray | None = None


class ValidatorDataset:
    """AGE dataset implementation"""

    def __init__(self, data_conf: DataConfig):
        self.data_conf = data_conf

    def get_available_tasks(self, verbose=False):
        cols = pq.ParquetDataset(self.data_conf.train_path).schema.names
        target_cols = [col for col in cols if col.startswith("target__")]
        if not verbose:
            return target_cols

        logger.info("Available tasks for %s dataset:", self.data_conf.dataset_name)
        for col in target_cols:
            name, target_type, metrics = self._parse_target_name(col)
            logger.info(
                "Task Name: %s,\t%s type,\tMetrics: %s\t| %s",
                name,
                target_type,
                metrics,
                col,
            )
        return target_cols

    def load_for_task(
        self, target_name, use_is_val=False, evaluate_test=True, global_train_filter=None
    ) -> DataSplit:
        if global_train_filter not in (None, "exclude_test"):
            raise ValueError("global_train_filter must be null or exclude_test")
        exclude_global_test = global_train_filter == "exclude_test"
        _, scope, metrics = self._parse_target_name(target_name)
        emb = "shift_emb" if scope == "local" else "global_emb"
        columns = [emb, target_name]
        has_global = (
            "global_train" in pq.ParquetDataset(self.data_conf.train_path).schema.names
        )
        if scope == "global" or (exclude_global_test and has_global):
            columns.append("global_train")
        if use_is_val:
            columns.append("is_val")
        train = pd.read_parquet(self.data_conf.train_path, columns=columns)
        test = None
        if scope == "global":
            if evaluate_test:
                test = train[train.global_train == 0]
            train = train[train.global_train == 1]
        elif evaluate_test and self.data_conf.test_path:
            test = pd.read_parquet(self.data_conf.test_path, columns=[emb, target_name])
        val = None
        if exclude_global_test and has_global:
            train = train[train.global_train == 1]
        if use_is_val:
            if train.is_val.isna().any() or not train.is_val.isin([True, False]).all():
                raise ValueError("is_val must contain non-null booleans")
            val, train = train[train.is_val == True], train[train.is_val == False]

        def arrays(frame):
            if frame is None:
                return None, None
            if scope == "local":
                frame = frame[
                    (frame[emb].map(len) > 0) | (frame[target_name].map(len) > 0)
                ]
                frame = frame.explode([emb, target_name])
            if frame.empty:
                raise ValueError(f"Empty split for {target_name}; increase sample size")
            dtype = (
                np.float32
                if METRIC_INFO[metrics[0]]["task"] == TaskType.REGRESSION
                else np.int32
            )
            return np.stack(frame[emb]).astype(np.float32), frame[
                target_name
            ].to_numpy().astype(dtype)

        x, y = arrays(train)
        xv, yv = arrays(val)
        xt, yt = arrays(test)
        return DataSplit(x, y, xt, yt, metrics, target_name, xv, yv)

    @staticmethod
    def _parse_target_name(target_name):
        parts = target_name.split("__")
        assert parts[0] == "target"
        assert len(parts) == 4, parts
        name, target_type, metrics = parts[1:]
        assert target_type in ["global", "local"], f"{target_type} is a wrong task type"
        metrics = metrics.split("+")
        return name, target_type, metrics
