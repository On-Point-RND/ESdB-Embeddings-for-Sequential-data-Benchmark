"""Main pipeline orchestrator with task routing"""

import logging
from typing import Any

import pandas as pd

from universal_validator.pipeline.utils import ValidatorConfig

from ..data.dataset import ValidatorDataset
from ..tasks.task_manager import TaskManager

logger = logging.getLogger(__name__)


class UniversalValidator:
    def __init__(self, config: ValidatorConfig):
        self.config = config
        self.dataset = ValidatorDataset(config.data_conf)
        self.task = TaskManager(config)

    def get_available_tasks(self, verbose=True):
        return self.dataset.get_available_tasks(verbose)

    def run_pipeline(self, task_name) -> dict[str, Any]:
        task_data = self.dataset.load_for_task(task_name)
        logger.info(
            "Starting %s pipeline, task: %s",
            self.config.data_conf.dataset_name,
            task_name,
        )
        results = self.task.execute(task_data)
        report = self._generate_report(results)
        return report

    def _generate_report(self, results):
        if not results:
            return {}
        scoring_metric = next(iter(results.values()))["main_metric"]
        best_model = max(results.keys(), key=lambda x: results[x].get(scoring_metric))
        report = {
            "dataset": self.config.data_conf.dataset_name,
            "best_model": best_model,
            f"best_{scoring_metric}": results[best_model].get(scoring_metric),
            "all_results": results,
            "timestamp": pd.Timestamp.now().isoformat(),
        }
        return report
