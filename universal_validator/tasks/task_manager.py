import importlib
from typing import Any, Optional

import optuna.logging
from sklearn.base import BaseEstimator

from ..data.dataset import DataSplit
from ..pipeline.utils import ValidatorConfig
from .hpo_optimizer import HPOOptimizer
from .scorers import METRIC_INFO, TaskType, get_scorers


def import_model_class(class_path: str):
    module_name, class_name = class_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


class TaskManager:

    def __init__(self, config: ValidatorConfig):
        self.config = config
        self.use_hpo = self.config.hpo.enabled
        self.verbose = self.config.verbose
        if not self.verbose:
            optuna.logging.set_verbosity(optuna.logging.ERROR)
        if self.use_hpo:
            self.hpo_optimizer = HPOOptimizer(self.config.hpo)

    def _init_models(
        self, metrics: list[str], task_name: str
    ) -> dict[str, BaseEstimator]:
        if not metrics:
            return {}

        first_metric = metrics[0]
        task_type = METRIC_INFO[first_metric]["task"]

        models = {}
        for name, model_config in self.config.models.items():
            if task_type == TaskType.CLASSIFICATION:
                estimator_path = model_config["classifier"]
            elif task_type == TaskType.REGRESSION:
                estimator_path = model_config["regressor"]
            else:
                print(f"Warning: No estimator defined for {name} in {task_type} mode")
                continue
            params = model_config.get("shared_params", {}).copy()
            task_specific = model_config.get("task_specific", {})
            if task_specific and task_name in task_specific:
                params.update(task_specific[task_name])
            search_space = model_config.get("search_space", {})
            try:
                model_class = import_model_class(estimator_path)
                models[name] = (model_class(**params), search_space)
            except Exception as e:
                print(f"Warning: Failed to initialize {name}: {e}")

        return models

    def _check_metric(self, metrics):
        for metric in metrics:
            if metric not in METRIC_INFO.keys():
                print(f"Metric {metric} is not supported!")
                return False
        task_types = [METRIC_INFO[metric]["task"] for metric in metrics]
        if len(set(task_types)) > 1:
            print(f"Metrics {metrics} have different task types!")
            return False
        return True

    def _print_task_info(self, split_data: dict[str, Any]) -> None:
        pass

    def execute(self, split_data: DataSplit) -> Optional[dict[str, Any] | None]:
        metrics = split_data.metrics
        if not self._check_metric(metrics):
            return None
        scorers = get_scorers(metrics)
        models = self._init_models(metrics, split_data.task_name)
        self._print_task_info(split_data)
        if self.use_hpo:
            print("Using HPO optimization")
        print(f"Models: {models}")
        results = {}
        for name in models:
            print(f"\nTraining {name}...")
            base_model, search_space = models[name]
            if base_model.__class__.__module__ == "universal_validator.models.torch_mlp":
                base_model.set_params(
                    early_stopping_scorer=scorers[0],
                )
            if self.use_hpo:
                model, cv_results = self.hpo_optimizer.optimize(
                    base_model=base_model,
                    X_train=split_data.X_train,
                    y_train=split_data.y_train,
                    scorer=scorers[0],
                    search_space=search_space,
                )
            else:
                model = base_model
                model.fit(split_data.X_train, split_data.y_train)
                cv_results = None

            result_metrics = {}
            predictions = model.predict(split_data.X_test)
            for scorer in scorers:
                result_metrics[scorer.name] = scorer(
                    model, split_data.X_test, split_data.y_test
                )
            results[name] = {
                "main_metric": scorers[0].name,
                **result_metrics,
                "predictions": predictions,
                "model": model,
                "cv_results": cv_results,
            }
            self._print_model_results(name, result_metrics, cv_results)

        return results

    def _print_model_results(
        self, model_name: str, metrics: dict[str, float], cv_results: dict | None
    ):
        if cv_results and not cv_results.get("failed", False):
            print(f"  {model_name}: CV = {cv_results.get('best_score', 0):.4f}", end="")
        else:
            print(f"  {model_name}:", end="")
        for metric_name, value in metrics.items():
            print(f", {metric_name} = {value:.4f}", end="")
        print()
