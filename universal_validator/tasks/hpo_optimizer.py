import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import optuna
from sklearn.base import BaseEstimator
from sklearn.metrics._scorer import _Scorer
from sklearn.model_selection import KFold, cross_val_score

warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class HPOConfig:
    enabled: bool = False
    n_trials: int = 50
    cv: int = 3
    n_jobs: int = 4
    max_threads: int = 4
    show_progress_bar: bool = True
    search_spaces: dict = field(default_factory=dict)


class HPOOptimizer:
    def __init__(self, hpo_conf: HPOConfig):
        self.hpo_config = hpo_conf

    def optimize(
        self,
        base_model: BaseEstimator,
        X_train: np.ndarray,
        y_train: np.ndarray,
        scorer: _Scorer,
        search_space: dict = None,
    ):
        model_name = base_model.__class__.__name__

        if not search_space:
            print(f"  No search space for {model_name}")
            base_model.fit(X_train, y_train)
            return base_model, None

        print(f"  Optimizing {model_name}... by {scorer.name} (maximize)")
        study = optuna.create_study(direction="maximize")
        try:
            objective = self._create_objective(
                base_model, X_train, y_train, scorer, search_space
            )
            study.optimize(
                objective,
                n_trials=self.hpo_config.n_trials,
                show_progress_bar=self.hpo_config.show_progress_bar,
            )

            best_params = study.best_params
            final_model = base_model.__class__(
                **{**base_model.get_params(), **best_params}
            )
            final_model.fit(X_train, y_train)

            cv_results = {
                "best_score": study.best_value,
                "best_params": best_params,
                "failed": False,
            }
            print(f"  Best CV: {study.best_value:.4f}")
            return final_model, cv_results

        except Exception as e:
            print(f"  Optuna failed: {e}")
            base_model.fit(X_train, y_train)
            return base_model, {"failed": True}

    def _create_objective(
        self,
        base_model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        scorer: _Scorer,
        search_space: dict,
    ):

        def objective(trial):
            params = {}
            for param_name, param_values in search_space.items():
                suggestion = self._suggest_param(trial, param_name, param_values)
                if suggestion:
                    params.update(suggestion)
            model = base_model.__class__(**{**base_model.get_params(), **params})
            cv = self.hpo_config.cv
            kf = KFold(n_splits=cv, shuffle=True, random_state=42)
            scores = cross_val_score(model, X, y, cv=kf, scoring=scorer)
            return scores.mean()

        return objective

    def _suggest_param(self, trial, param_name: str, param_values: Any):
        if isinstance(param_values, list):
            return {param_name: trial.suggest_categorical(param_name, param_values)}

        if (
            isinstance(param_values, dict)
            and "low" in param_values
            and "high" in param_values
        ):
            low, high = param_values["low"], param_values["high"]
            log = param_values.get("log", False)
            if isinstance(low, int) and isinstance(high, int):
                return {param_name: trial.suggest_int(param_name, low, high, log=log)}
            else:
                return {param_name: trial.suggest_float(param_name, low, high, log=log)}

        return {}

