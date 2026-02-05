"""Main pipeline orchestrator with task routing"""

import os
from typing import Dict, Any
import pandas as pd
import numpy as np
from omegaconf import DictConfig
from ..tasks.task_manager import TaskManager
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
        self.task = TaskManager(self.config, task_name)
        results = self.task.execute(task_data)
        report = self._generate_report(results)
        return report

    def _generate_report(self, results):
        if not results:
            return {}
        scoring_metric = self.task.scoring_function
        # double check this pattern
        if 'neg' in scoring_metric:
            scoring_metric = scoring_metric.replace('neg_','')
        best_model = max(results.keys(), key=lambda x: results[x].get(scoring_metric))
        report = {
            "dataset": self.config.data_conf.dataset_name,
            "best_model": best_model,
            f"best_{scoring_metric}": results[best_model].get(scoring_metric, None),
            "all_results": results,
            "timestamp": pd.Timestamp.now().isoformat(),
        }
        return report
