"""Main pipeline orchestrator with task routing"""

from typing import Any

import pandas as pd
from omegaconf import DictConfig

from ..data.dataset import ValidatorDataset
from ..tasks.task_manager import TaskManager


class UniversalValidator:
    def __init__(self, config: DictConfig):
        self.config = config
        self.dataset = ValidatorDataset(config.data_conf)
        self.task = TaskManager(config)

    def get_available_tasks(self, verbose=True):
        return self.dataset.get_available_tasks(verbose)

    def run_pipeline(self, task_name) -> dict[str, Any]:
        task_data = self.dataset.load_for_task(task_name)
        print(
            f"Starting {self.config.data_conf.dataset_name} pipeline, Task: {task_name}"
        )
        results = self.task.execute(task_data)
        report = self._generate_report(results)
        return report

    def _generate_report(self, results):
        if not results:
            return {}
        first_model_results = next(iter(results.values()))
        scoring_metric = first_model_results["main_metric"]
        sorting_key = lambda model_name: results[model_name].get(scoring_metric)
        best_model = max(results.keys(), key=sorting_key)
        report = {
            "dataset": self.config.data_conf.dataset_name,
            "best_model": best_model,
            f"best_{scoring_metric}": results[best_model].get(scoring_metric),
            "all_results": results,
            "timestamp": pd.Timestamp.now().isoformat(),
        }
        return report
