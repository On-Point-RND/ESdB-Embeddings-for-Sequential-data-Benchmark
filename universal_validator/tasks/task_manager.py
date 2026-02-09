import importlib
from typing import Any

import optuna.logging
import sklearn.metrics as sk_metrics
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, roc_auc_score
from enum import Enum

from ..data.dataset import DataSplit
from ..pipeline.utils import ValidatorConfig
from .hpo_optimizer import HPOOptimizer
from .scorers import get_scorers

def import_model_class(class_path: str):
    module_name, class_name = class_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)

class TaskType(str, Enum): # for older python version
    REGRESSION = "regression"
    CLASSIFICATION = "classification"

# TODO стандартизировать имена!
METRIC_TO_TASK = {
    "r2": TaskType.REGRESSION,
    "mse": TaskType.REGRESSION,
    "f1": TaskType.CLASSIFICATION,
    "accuracy": TaskType.CLASSIFICATION,
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
    
    def _build_task_config(self, metrics: list, task_name: str) -> dict:
        """Строит task_config исходя структуры конфигов"""
        task_config = {"models": {}}

        if not metrics:
            return task_config

        first_metric = metrics[0]
        task_type = METRIC_TO_TASK.get(first_metric)
        
        if not task_type:
            return task_config

        for model_name, model_config in self.config.models.items():
            # Выбираем правильный estimator_class
            if task_type == TaskType.CLASSIFICATION:
                estimator_class = model_config.get("classifier")
            else:  # TaskType.REGRESSION
                estimator_class = model_config.get("regressor")

            params = model_config.get("shared_params", {}).copy()
            task_specific = model_config.get("task_specific", {})
            if task_specific and task_name in task_specific:
                params.update(task_specific[task_name])

            task_config["models"][model_name] = {
                "class": estimator_class,
                "params": params
            }

        return task_config

    def execute(self, split_data: DataSplit) -> dict[str, Any]:
        metrics = split_data.metrics
        
        if not self._check_metric(metrics):
            return
        
        scorers = get_scorers(metrics)
        task_config = self._build_task_config(metrics, split_data.task_name)
        models = self._init_models(task_config)
              
        self._print_task_info(split_data)
      
        if self.use_hpo:
            print("Using HPO optimization")
      
        print(f"Models: {models}")
        
        results = {}
        for name in models:
            print(f"\nTraining {name}...")
            base_model = models[name]
            
            model_config = self.config.models.get(name)
            if not model_config:
                print(f"  Warning: No config found for {name}")
                continue
            
            if self.use_hpo:
                search_space = model_config.get("search_space", {})
                model, cv_results = self.hpo_optimizer.optimize(
                    base_model=base_model, 
                    X_train=split_data.X_train, 
                    y_train=split_data.y_train, 
                    scorer=scorers[0],
                    task_name=split_data.task_name,
                    search_space=search_space
                )
            else:
                model = base_model
                model.fit(split_data.X_train, split_data.y_train)
                cv_results = None
            
            result_metrics = {}
            predictions = model.predict(split_data.X_test)
            for scorer in scorers:
                result_metrics[scorer.name] = scorer(model, split_data.X_test, split_data.y_test)       
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
