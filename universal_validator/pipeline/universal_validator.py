"""Main pipeline orchestrator with task routing"""

import os
from typing import Dict, Any
import pandas as pd
import numpy as np
from omegaconf import DictConfig
from ..tasks.classification_task import ClassificationTask
from ..tasks.regression_task import RegressionTask
from ..data.dataset import ValidatorDataset


class UniversalValidator:
    def __init__(self, config: DictConfig):
        self.config = config
        self.dataset = ValidatorDataset(config.data_conf)

    def get_available_tasks(self, verbose=True):
        return self.dataset.get_available_tasks(verbose)

    def run_pipeline(self, task_name) -> Dict[str, Any]:
        task_data = self.dataset.load_for_task(task_name)
        print(f"Starting {self.config.data_conf.dataset_name} pipeline, Task: {task_name}")
        # TODO init tasker
        results = task.execute(task_data)
        report = self._generate_report(dataset_name, results)
        print("Pipeline completed successfully!")
        return report

    def _generate_report(self, dataset_name, task_type, results):
        if not results:
            return {
                "dataset": dataset_name,
                "task_type": task_type.value,
                "error": "No models trained",
            }
        # TODO заменить место хранения метрик гдето
        metric = metric_map[task_type]
        best_model = max(results.keys(), key=lambda x: results[x][metric])

        return {
            "dataset": dataset_name,
            "task_type": task_type.value,
            "best_model": best_model,
            f"best_{metric}": results[best_model][metric],
            "all_results": results,
            "timestamp": pd.Timestamp.now().isoformat(),
        }
