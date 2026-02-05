from typing import Dict, Any, List, Optional
from omegaconf import DictConfig, OmegaConf
import numpy as np
import optuna.logging
from sklearn.base import BaseEstimator
from .hpo_optimizer import HPOOptimizer
from .hpo_optimizer import metric_mapping, metric_mapping_per_task
from dataclasses import asdict
import traceback
import sklearn.metrics

class TaskManager:
    
    def __init__(self, config: DictConfig, task_name):
        self.config = OmegaConf.create(asdict(config))
        self.task_name = task_name # full task name
        default_scoring = self.task_name.split('__')[-1]
        self.short_task_name = self.task_name.split('__')[1]
        if '+' in default_scoring:
            default_scoring = default_scoring.split('+')[0]
        default_scoring in metric_mapping.keys()
        self.downstream_config = config.downstream
        self.CONFIG_SECTION = self.task_name
        # short name in dataset column
        self.default_scoring = default_scoring
        # actual function name from sklearn, alway maximize
        self.scoring_function = metric_mapping.get(self.default_scoring, self.default_scoring)
        self.task_config = self.downstream_config.get(self.CONFIG_SECTION, OmegaConf.create({}))
        self.hpo_config = self.downstream_config.get('hpo', OmegaConf.create({'enabled': False}))
        self.use_hpo = self.hpo_config.get('enabled', False)
        self.models = self._init_models()
        self.verbose = self.hpo_config.get('verbose', True)
        if not self.verbose:
            optuna.logging.set_verbosity(optuna.logging.ERROR)
        if self.use_hpo:
            self.hpo_optimizer = HPOOptimizer(self.config, self)
        if not self.models:
            print(f"Info: No models for {self.CONFIG_SECTION}")
        
    
    def _init_models(self) -> Dict[str, BaseEstimator]:
        models_config = self.task_config.get('models', OmegaConf.create({}))
        if not models_config: 
            return {}
        models = {}
        for name, params in models_config.items():
            try:
                params_dict = params['params']
                model_class = self._import_model_class(params['class'])
                models[name] = model_class(**params_dict)
            except Exception as e:
                print(f"Warning: Failed {name}: {e}")
        return models
    
    def _import_model_class(self, class_path: str):
        import importlib
        module_name, class_name = class_path.rsplit('.', 1)
        return getattr(importlib.import_module(module_name), class_name)
    
    def _get_available_models(self) -> List[str]:
        return list(self.models.keys())
    
    def _print_task_info(self, split_data: Dict[str, Any]) -> None:
        pass
    
    def execute(self, split_data: Dict[str, Any]) -> Dict[str, Any]:
        self._print_task_info(split_data)
        if self.use_hpo:
            print(f"Using HPO optimization")
        available_models = self._get_available_models()
        if not available_models:
            print("Warning: No models!")
            return {}
        print(f"Models: {available_models}")
        results = {}    
        for name in available_models:
            print(f"\nTraining {name}...")
            base_model = self.models[name]
            if self.use_hpo:
                model, cv_results = self.hpo_optimizer.optimize(
                    name, base_model, split_data['X_train'], split_data['y_train']
                )
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
        
    def _print_model_results(self, model_name: str, metrics: Dict[str, float], cv_results: Optional[Dict]):
        if cv_results and not cv_results.get('failed', False):
            print(f"  {model_name}: CV = {cv_results.get('best_score', 0):.4f}", end="")
        else:
            print(f"  {model_name}:", end="")
        for metric_name, value in metrics.items():
            print(f", {metric_name} = {value:.4f}", end="")
        print()
        
    def _calculate_metrics(self, model, X_test, y_test):
        metrics = {}
        task_type = 'classification'
        if 'reg' in self.short_task_name or 'forecast' in self.short_task_name:
            task_type = 'regression'
        y_pred = model.predict(X_test)
        for metric_name in metric_mapping_per_task[task_type]:
            if metric_name == 'roc_auc_score':
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                metrics[metric_name] = roc_auc_score(y_test, y_pred_proba)
            elif metric_name == 'f1_score':
                metrics[metric_name] = f1_score(y_test, y_pred, average='weighted')
            else:
                metric_func = getattr(sklearn.metrics, metric_name)
                metrics[metric_name] = metric_func(y_test, y_pred)
        return metrics
