"""Classification task implementation"""
from typing import Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .base_task import BaseTask
from ..types import TaskType

class ClassificationTask(BaseTask):
    """Classification task implementation"""
    
    def _get_task_name(self) -> str:
        """Get task name for config lookup"""
        return 'classification'
    
    def _get_supported_task_type(self) -> TaskType:
        return TaskType.CLASSIFICATION
    
    def _calculate_metrics(self, model, X_test, y_test) -> Dict[str, float]:
        """Calculate classification metrics"""
        predictions = model.predict(X_test)
        return {
            'accuracy': accuracy_score(y_test, predictions),
            'f1': f1_score(y_test, predictions, average='weighted'),
            'precision': precision_score(y_test, predictions, average='weighted'),
            'recall': recall_score(y_test, predictions, average='weighted')
        }
    
    def _get_optuna_params(self, model_name: str, trial) -> Dict[str, Any]:
        """Override to add specific parameters for classification"""
        # Получаем базовые параметры из родительского класса
        params = super()._get_optuna_params(model_name, trial)
        
        # Добавляем специфичные параметры для CatBoost в задаче классификации
        if model_name == 'catboost':
            # Получаем параметры для классификации CatBoost
            catboost_classification_params = self.optuna_params_config.get('catboost_classification', {})
            for param_name, param_config in catboost_classification_params.items():
                param_type = param_config.get('type', 'categorical')
                
                if param_type == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, 
                        param_config.get('choices', [])
                    )
        
        return params
