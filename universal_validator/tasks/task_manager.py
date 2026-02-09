import importlib
from enum import StrEnum
from typing import Any

import optuna.logging
import sklearn.metrics as sk_metrics
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, roc_auc_score

from ..data.dataset import DataSplit
from ..pipeline.utils import ValidatorConfig
from .hpo_optimizer import HPOOptimizer


def import_model_class(class_path: str):
    module_name, class_name = class_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


class TaskType(StrEnum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"


# TODO uncomment via implementing calculate metrics
METRIC_TO_TASK = {
    # "r2": TaskType.REGRESSION,
    # "mse": TaskType.REGRESSION,
    "f1": TaskType.CLASSIFICATION,
    # "accuracy": TaskType.CLASSIFICATION,
    "roc_auc": TaskType.CLASSIFICATION,
}


class TaskManager:

    def __init__(self, config: ValidatorConfig):
        self.config = config
        self.use_hpo = self.config.hpo.enabled
        self.verbose = self.config.verbose
        if not self.verbose:
            optuna.logging.set_verbosity(optuna.logging.ERROR)
        if self.use_hpo:
            self.hpo_optimizer = HPOOptimizer(self.config.hpo, config.models)

    def _init_models(self, task_config) -> dict[str, BaseEstimator]:
        models_config = task_config.get("models")
        if not models_config:
            return {}
        models = {}
        for name, params in models_config.items():
            try:
                params_dict = params["params"]
                model_class = import_model_class(params["class"])
                models[name] = model_class(**params_dict)
            except Exception as e:
                print(f"Warning: Failed {name}: {e}")
        return models

    def _check_metric(self, metrics):
        for metric in metrics:
            if metric not in METRIC_TO_TASK.keys():
                print(f"Metric {metric} is not supported!")
                return False
        task_types = [METRIC_TO_TASK[metric] for metric in metrics]
        if len(set(task_types)) > 1:
            print(f"Metrics {metrics} have different task types!")
            return False
        return True

    def _print_task_info(self, split_data: dict[str, Any]) -> None:
        pass

    def execute(self, split_data: DataSplit) -> dict[str, Any]:
        metrics = split_data.metrics
        if not self._check_metric(metrics):
            return
        task_config = # TODO with new config format
        scorers = # TODO create scorer
        models = self._init_models(task_config)
        if not models:
            print(f"Info: No models for {self.CONFIG_SECTION}")
            return
        self._print_task_info(split_data)
        if self.use_hpo:
            print("Using HPO optimization")
        if not models:
            print("Warning: No models!")
            return {}
        print(f"Models: {models}")
        results = {}
        for name in models:
            print(f"\nTraining {name}...")
            base_model = models[name]
            if self.use_hpo:
                model, cv_results = self.hpo_optimizer.optimize(
                    model=base_model, 
                    X_train=split_data.X_train, 
                    y_train=split_data.y_train, 
                    scorer=scorers[0],
                    task_name=split_data.task_name,
                )
            else:
                model = base_model
                model.fit(split_data.X_train, split_data.y_train)
                cv_results = None
            predictions = model.predict(split_data.X_test)
            # metrics = self._calculate_metrics(model, split_data)
            for scorer in scorers:
                metrics[] = scorer(model, split_data.X_test, split_data.y_test)
            results[name] = {
                "main_metric": scorers[0].name, # TODO make correct
                **metrics,
                "predictions": predictions,
                "model": model,
                "cv_results": cv_results,
            }
            self._print_model_results(name, metrics, cv_results)

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

    # def _calculate_metrics(self, model, split_data: DataSplit):
    #     metrics = split_data.metrics
    #     X_test, y_test = split_data.X_test, split_data.y_test
    #     result = {}
    #     for metric in metrics:
    #         if metric == "roc_auc":
    #             y_pred_proba = model.predict_proba(X_test)[:, 1]
    #             result[metric] = roc_auc_score(y_test, y_pred_proba)
    #         elif metric == "f1":
    #             result[metric] = f1_score(
    #                 y_test, model.predict(X_test), average="weighted"
    #             )
    #         else:
    #             metric_func = getattr(sk_metrics, metric)
    #             result[metric] = metric_func(y_test, model.predict(X_test))
    #     return result
