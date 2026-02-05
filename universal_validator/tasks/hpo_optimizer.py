from typing import Dict, Any
import numpy as np
import optuna
import warnings
from omegaconf import DictConfig, OmegaConf
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, KFold

warnings.filterwarnings('ignore', category=FutureWarning)
import traceback

metric_mapping = {
    'accuracy': 'accuracy', 'AUCROC': 'roc_auc', 'r2': 'r2',
    'F1': 'f1_weighted', 'precision': 'precision_weighted', 'recall': 'recall_weighted',
    'mse': 'neg_mean_squared_error', 'mae': 'neg_mean_absolute_error'
}

# todo pass to config?
metric_mapping_per_task = {
    'regression': ['r2_score', 'mean_squared_error'],
    'classification': ['f1_score', 'accuracy_score', 'roc_auc_score']
}

class HPOOptimizer:
    def __init__(self, config: DictConfig, task):
        self.config = config
        self.task = task
        self.downstream_config = self.config.downstream
        self.search_spaces = self._get_search_spaces()
        self.hpo_config = self.downstream_config.get('hpo', OmegaConf.create({}))
    
    def _get_search_spaces(self) -> Dict[str, Dict[str, Any]]:
        spaces = self.downstream_config.get('search_spaces', OmegaConf.create({}))
        return OmegaConf.to_container(spaces, resolve=True) if spaces else {}
    
    def optimize(self, model_name: str, base_model: BaseEstimator, X_train: np.ndarray, y_train: np.ndarray):
        if model_name not in self.search_spaces or not self.search_spaces[model_name]:
            print(f"  No search space for {model_name}")
            base_model.fit(X_train, y_train)
            return base_model, None
        
        scoring_function = self.task.scoring_function
        direction='maximize'
        print(f"  Optimizing {model_name}... by {scoring_function} {direction}")
        
        study = optuna.create_study(
            direction=direction,
            study_name=f"{model_name}_{self.task.CONFIG_SECTION}"
        )
        
        try:
            objective = self._create_objective(model_name, base_model, X_train, y_train)
            show_progress_bar = self.hpo_config.get('show_progress_bar', True)
            study.optimize(objective, n_trials=self.hpo_config.get('n_trials', 50), show_progress_bar=show_progress_bar)
            best_params = self._clean_params(model_name, study.best_params)
            final_model = base_model.__class__(**{**base_model.get_params(), **best_params})
            final_model.fit(X_train, y_train)
            cv_results = {
                'best_score': study.best_value,
                'best_params': best_params,
                'failed': False
            }
            print(f"  Best CV: {study.best_value:.4f}")
            return final_model, cv_results
            
        except Exception as e:
            print(f"  Optuna failed: {e}")
            base_model.fit(X_train, y_train)
            return base_model, {'failed': True}

    def _create_objective(self, model_name: str, base_model: BaseEstimator, X: np.ndarray, y: np.ndarray):
        
        def objective(trial):
            params = {}
            for param_name, param_values in self.search_spaces.get(model_name, {}).items():
                suggestion = self._suggest_param(trial, model_name, param_name, param_values)
                if suggestion: 
                    params.update(suggestion)
            model = base_model.__class__(**{**base_model.get_params(), **params})
            cv = self.hpo_config.get('cv', 3)
            kf = KFold(n_splits=cv, shuffle=True, random_state=42)
            scores = cross_val_score(model, X, y, cv=kf, scoring=self.task.scoring_function)
            return scores.mean()
            
        return objective
    
    def _suggest_param(self, trial, model_name: str, param_name: str, param_values: Any):
        if isinstance(param_values, list):
            if param_name == 'hidden_layer_sizes':
                idx = trial.suggest_int(f"{model_name}_{param_name}_idx", 0, len(param_values)-1)
                return {param_name: param_values[idx]}
            else:
                return {param_name: trial.suggest_categorical(f"{model_name}_{param_name}", param_values)}
        if isinstance(param_values, dict) and 'low' in param_values and 'high' in param_values:
            low, high = param_values['low'], param_values['high']
            log = param_values.get('log', False)
            if isinstance(low, int) and isinstance(high, int):
                return {param_name: trial.suggest_int(f"{model_name}_{param_name}", low, high, log=log)}
            else:
                return {param_name: trial.suggest_float(f"{model_name}_{param_name}", low, high, log=log)}
        return {}
    
    def _clean_params(self, model_name: str, params: Dict) -> Dict:
        cleaned = {}
        for key, value in params.items():
            clean_key = key[len(f"{model_name}_"):] if key.startswith(f"{model_name}_") else key
            if clean_key.endswith('_idx'):
                param_name = clean_key[:-4]
                if param_name in self.search_spaces.get(model_name, {}):
                    param_values = self.search_spaces[model_name][param_name]
                    if isinstance(param_values, list) and 0 <= value < len(param_values):
                        cleaned[param_name] = param_values[value]
            else:
                cleaned[clean_key] = value
        return cleaned
