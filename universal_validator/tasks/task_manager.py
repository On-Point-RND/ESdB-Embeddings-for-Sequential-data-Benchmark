import importlib
import logging
from datetime import datetime
from typing import Any

import numpy as np
import optuna.logging
from sklearn.model_selection import train_test_split

from ..data.dataset import DataSplit
from ..pipeline.utils import ValidatorConfig
from .hpo_optimizer import Estimator, HPOOptimizer
from .scorers import METRIC_INFO, TaskType, get_scorers

logger = logging.getLogger(__name__)


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
    ) -> dict[str, tuple[Estimator, dict[str, Any]]]:
        if not metrics:
            return {}

        first_metric = metrics[0]
        task_type = METRIC_INFO[first_metric]["task"]

        models: dict[str, tuple[Estimator, dict[str, Any]]] = {}
        for name, model_config in self.config.models.items():
            if task_type == TaskType.CLASSIFICATION:
                estimator_path = model_config["classifier"]
            elif task_type == TaskType.REGRESSION:
                estimator_path = model_config["regressor"]
            else:
                logger.warning(
                    "No estimator defined for %s in %s mode", name, task_type
                )
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
                logger.warning("Failed to initialize %s: %s", name, e)

        return models

    def _check_metric(self, metrics):
        for metric in metrics:
            if metric not in METRIC_INFO.keys():
                logger.warning("Metric %s is not supported!", metric)
                return False
        task_types = [METRIC_INFO[metric]["task"] for metric in metrics]
        if len(set(task_types)) > 1:
            logger.warning("Metrics %s have different task types!", metrics)
            return False
        return True

    def _print_task_info(self, split_data: DataSplit) -> None:
        pass

    def execute(self, split_data: DataSplit) -> dict[str, Any] | None:
        metrics = split_data.metrics
        if not self._check_metric(metrics):
            return None
        scorers = get_scorers(metrics)
        models = self._init_models(metrics, split_data.task_name)
        self._print_task_info(split_data)
        if self.use_hpo:
            logger.info("Using HPO optimization")
        if not models:
            return {}
        x, y = split_data.X_train, split_data.y_train
        xv, yv = split_data.X_val, split_data.y_val
        if xv is None:
            classifier = METRIC_INFO[metrics[0]]["task"] == TaskType.CLASSIFICATION
            stratify = (
                y
                if classifier and np.unique(y, return_counts=True)[1].min() >= 2
                else None
            )
            x, xv, y, yv = train_test_split(
                x,
                y,
                test_size=self.config.validation_fraction,
                random_state=self.config.seed,
                stratify=stratify,
            )
        results = {}
        for name in models:
            logger.info("Training %s", name)
            base_model, search_space = models[name]
            if (
                base_model.__class__.__module__
                == "universal_validator.models.torch_mlp"
            ):
                base_model.set_params(
                    early_stopping_scorer=scorers[0],
                )
            if self.use_hpo:
                model, cv_results = self.hpo_optimizer.optimize(
                    base_model=base_model,
                    x_train=x,
                    y_train=y,
                    scorer=scorers[0],
                    search_space=search_space,
                    refit=False,
                )
            else:
                model = base_model
                cv_results = None

            self._fit_model(model, x, y, xv, yv, scorers[0], metrics[0])
            val_metrics = {
                scorer.name: float(scorer(model, xv, yv)) for scorer in scorers
            }
            if not all(np.isfinite(value) for value in val_metrics.values()):
                raise ValueError(
                    f"Invalid validation metrics for {split_data.task_name}"
                )
            test_metrics = {}
            predictions = None
            if split_data.X_test is not None:
                predictions = model.predict(split_data.X_test)
                test_metrics = {
                    scorer.name: float(
                        scorer(model, split_data.X_test, split_data.y_test)
                    )
                    for scorer in scorers
                }
            result_metrics = (
                val_metrics if self.config.use_is_val else (test_metrics or val_metrics)
            )
            results[name] = {
                "main_metric": scorers[0].name,
                **result_metrics,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "predictions": predictions,
                "model": model,
                "cv_results": cv_results,
            }
            self._log_model_results(name, result_metrics, cv_results)
        return results

    def _fit_model(self, model, x, y, xv, yv, scorer, metric_name):
        module = model.__class__.__module__
        if module == "universal_validator.models.torch_mlp":
            model.fit(x, y, validation_data=(xv, yv))
        elif module.startswith("lightgbm"):
            classifier = METRIC_INFO[metric_name]["task"] == TaskType.CLASSIFICATION

            def eval_metric(target, prediction):
                if classifier and metric_name != "roc_auc":
                    prediction = (
                        np.argmax(prediction, axis=1)
                        if prediction.ndim == 2
                        else (prediction >= 0.5).astype(int)
                    )
                value = scorer._sign * scorer._score_func(
                    target, prediction, **scorer._kwargs
                )
                return scorer.name, float(value), True

            # Boosting budget/early_stopping_rounds remain model config parameters.
            model.set_params(metric="None", first_metric_only=True)
            model.fit(x, y, eval_set=[(xv, yv)], eval_metric=eval_metric)
        else:
            model.fit(x, y)

    def _log_model_results(
        self, model_name: str, metrics: dict[str, float], cv_results: dict | None
    ) -> None:
        if cv_results and not cv_results.get("failed", False):
            head = f"  {model_name}: CV = {cv_results.get('best_score', 0):.4f}"
        else:
            head = f"  {model_name}:"
        tail = "".join(f", {k} = {v:.4f}" for k, v in metrics.items())
        logger.info("%s%s", head, tail)
