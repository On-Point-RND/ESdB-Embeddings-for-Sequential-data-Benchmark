from typing import Dict, Any, List, Optional
from omegaconf import DictConfig, OmegaConf
import numpy as np
import optuna.logging
from sklearn.base import BaseEstimator
from .hpo_optimizer import HPOOptimizer


class BaseTask:
    TASK_TYPE = None
    CONFIG_SECTION = None
    DEFAULT_SCORING = 'accuracy'
    
    def __init__(self, config: DictConfig):
        if self.TASK_TYPE is None or self.CONFIG_SECTION is None:
            raise ValueError("TASK_TYPE and CONFIG_SECTION must be set")
        self.config = config
        self.downstream_config = config.get('downstream', OmegaConf.create({}))
        self.task_config = self.downstream_config.get(self.CONFIG_SECTION, OmegaConf.create({}))
        self.hpo_config = self.downstream_config.get('hpo', OmegaConf.create({'enabled': False}))
        self.use_hpo = self.hpo_config.get('enabled', False)
        self.scoring = self.task_config.get('scoring', self.DEFAULT_SCORING)
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
        if not models_config: return {}
        
        models = {}
        for name, params in models_config.items():
            try:
                model_class = self._import_model_class(params['class'])
                params_dict = OmegaConf.to_container(params.get('params', {}), resolve=True)
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
        print(f"Running {self.TASK_TYPE.value} task...")
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
