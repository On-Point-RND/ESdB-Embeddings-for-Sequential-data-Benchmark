"""Base task implementation"""
from typing import Dict, Any, List, Optional
from omegaconf import DictConfig, OmegaConf
import numpy as np
import optuna
import importlib
import warnings
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, KFold

from ..types import TaskType

warnings.filterwarnings('ignore', category=FutureWarning)

class BaseTask:
    TASK_TYPE = None
    CONFIG_SECTION = None
    DEFAULT_SCORING = 'accuracy'
    
    def __init__(self, config: DictConfig):
        if self.TASK_TYPE is None or self.CONFIG_SECTION is None:
            raise ValueError("TASK_TYPE and CONFIG_SECTION must be set")
        
        self.config = config
        self.downstream_config = self.config.get('downstream', OmegaConf.create({}))
        self.task_config = self._get_task_specific_config()
        self.search_spaces = self._get_search_spaces()
        self.hpo_config = self._get_hpo_config()
        self.use_hpo = self.hpo_config.get('enabled', False)
        self.scoring = self._get_scoring()
        self.models = self._init_models()
        
        if not self.models:
            print(f"Info: No models configured for {self.CONFIG_SECTION}")
    
    def _get_task_specific_config(self) -> DictConfig:
        return self.downstream_config.get(self.CONFIG_SECTION, OmegaConf.create({}))
    
    def _get_search_spaces(self) -> Dict[str, Dict[str, Any]]:
        if 'search_spaces' in self.task_config:
            spaces = self.task_config.get('search_spaces', OmegaConf.create({}))
            if spaces: return OmegaConf.to_container(spaces, resolve=True)
        
        spaces = self.downstream_config.get('search_spaces', OmegaConf.create({}))
        return OmegaConf.to_container(spaces, resolve=True) if spaces else {}
    
    def _get_hpo_config(self) -> DictConfig:
        return self.downstream_config.get('hpo', OmegaConf.create({'enabled': False}))
    
    def _get_scoring(self) -> str:
        return self.task_config.get('scoring', self.DEFAULT_SCORING)
    
    def _init_models(self) -> Dict[str, BaseEstimator]:
        models_config = self.task_config.get('models', OmegaConf.create({}))
        if not models_config: return {}
        
        models = {}
        for name, params in models_config.items():
            try:
                model_class = self._import_model_class(params['class'])
                params_dict = OmegaConf.to_container(params.get('params', {}), resolve=True)
                models[name] = model_class(**self._fix_params(params_dict))
            except Exception as e:
                print(f"Warning: Failed to initialize {name}: {e}")
                continue
        return models
    
    def _fix_params(self, params: Dict) -> Dict:
        if 'force_all_finite' in params:
            params['ensure_all_finite'] = params.pop('force_all_finite')
        return params
    
    def _import_model_class(self, class_path: str):
        module_name, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    
    def _create_objective(self, model_name: str, base_model: BaseEstimator, X: np.ndarray, y: np.ndarray):
        def objective(trial):
            params = {}
            if model_name in self.search_spaces:
                for param_name, param_values in self.search_spaces[model_name].items():
                    params.update(self._suggest_param(trial, model_name, param_name, param_values))
            
            try:
                model = base_model.__class__(**{**base_model.get_params(), **self._fix_params(params)})
                cv = self.hpo_config.get('cv', 3)
                kf = KFold(n_splits=cv, shuffle=True, random_state=42)
                scores = cross_val_score(model, X, y, cv=kf, scoring=self._get_scoring_function(), 
                                        n_jobs=self.hpo_config.get('n_jobs', 1))
                return scores.mean()
            except:
                return float('-inf') if self.scoring not in ['mse', 'mae', 'rmse'] else float('inf')
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
            step = param_values.get('step')
            
            if isinstance(low, int) and isinstance(high, int):
                if step: return {param_name: trial.suggest_int(f"{model_name}_{param_name}", low, high, step=step, log=log)}
                return {param_name: trial.suggest_int(f"{model_name}_{param_name}", low, high, log=log)}
            else:
                if step: return {param_name: trial.suggest_float(f"{model_name}_{param_name}", low, high, step=step, log=log)}
                return {param_name: trial.suggest_float(f"{model_name}_{param_name}", low, high, log=log)}
        
        return {}
    
    def _get_scoring_function(self) -> str:
        mapping = {
            'accuracy': 'accuracy', 'roc_auc': 'roc_auc', 'r2': 'r2',
            'f1': 'f1_weighted', 'precision': 'precision_weighted', 'recall': 'recall_weighted',
            'mse': 'neg_mean_squared_error', 'mae': 'neg_mean_absolute_error'
        }
        return mapping.get(self.scoring, self.scoring)
    
    def _optimize_with_optuna(self, model_name: str, base_model: BaseEstimator, X_train: np.ndarray, y_train: np.ndarray):
        n_trials = self.hpo_config.get('n_trials', 50)
        direction = 'minimize' if self.scoring in ['mse', 'mae', 'rmse'] else 'maximize'
        
        study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=42),
            study_name=f"{model_name}_{self.CONFIG_SECTION}"
        )
        
        try:
            objective = self._create_objective(model_name, base_model, X_train, y_train)
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
            best_params = self._clean_params(model_name, study.best_params)
            
            final_params = {**base_model.get_params(), **best_params}
            final_model = base_model.__class__(**self._fix_params(final_params))
            final_model.fit(X_train, y_train)
            
            return {
                'model': final_model,
                'best_params': best_params,
                'best_score': study.best_value,
                'study': study,
                'failed': False
            }
        except Exception as e:
            print(f"  Optuna failed: {e}")
            base_model.fit(X_train, y_train)
            return {
                'model': base_model,
                'best_params': base_model.get_params(),
                'best_score': 0.0,
                'study': None,
                'failed': True
            }
    
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
    
    def _print_task_info(self, split_data: Dict[str, Any]) -> None:
        pass
    
    def _get_available_models(self) -> List[str]:
        return list(self.models.keys())
    
    def execute(self, split_data: Dict[str, Any]) -> Dict[str, Any]:
        print(f"Running {self.TASK_TYPE.value} task...")
        self._print_task_info(split_data)
        
        if self.use_hpo:
            print(f"Using Optuna optimization (n_trials={self.hpo_config.get('n_trials', 50)})")
        
        available_models = self._get_available_models()
        if not available_models:
            print("Warning: No models available!")
            return {}
        
        print(f"Models: {available_models}")
        results = {}
        
        for name in available_models:
            print(f"\nTraining {name}...")
            base_model = self.models[name]
            
            if self.use_hpo and name in self.search_spaces and self.search_spaces[name]:
                optuna_result = self._optimize_with_optuna(name, base_model, split_data['X_train'], split_data['y_train'])
                model = optuna_result['model']
                cv_results = {
                    'best_score': optuna_result['best_score'],
                    'best_params': optuna_result['best_params'],
                    'failed': optuna_result['failed']
                }
                if not optuna_result['failed']:
                    print(f"  Best CV: {optuna_result['best_score']:.4f}")
            else:
                model = base_model
                model.fit(split_data['X_train'], split_data['y_train'])
                cv_results = None
            
            predictions = model.predict(split_data['X_test'])
            metrics = self._calculate_metrics(model, split_data['X_test'], split_data['y_test'])
            
            results[name] = {
                **metrics,
                'predictions': predictions,
                'model': model,
                'cv_results': cv_results
            }
            
            self._print_model_results(name, metrics, cv_results)
        
        return results
    
    def _calculate_metrics(self, model: BaseEstimator, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        raise NotImplementedError("Subclasses must implement _calculate_metrics")
    
    def _print_model_results(self, model_name: str, metrics: Dict[str, float], cv_results: Optional[Dict]):
        if cv_results and not cv_results.get('failed', False):
            print(f"  {model_name}: CV = {cv_results.get('best_score', 0):.4f}", end="")
        else:
            print(f"  {model_name}:", end="")
        
        for metric_name, value in metrics.items():
            print(f", {metric_name} = {value:.4f}", end="")
        print()
