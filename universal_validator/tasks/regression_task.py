"""Regression task implementation"""
from typing import Dict, Any
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from .base_task import BaseTask
from ..types import TaskType

class RegressionTask(BaseTask):
    """Regression task implementation"""
    
    def _get_task_name(self) -> str:
        """Get task name for config lookup"""
        return 'regression'
    
    def _get_supported_task_type(self) -> TaskType:
        return TaskType.REGRESSION
    
    def _get_default_scoring(self) -> str:
        """Use R2 scoring for regression"""
        return 'r2'
    
    def _get_optimization_direction(self) -> str:
        """For regression we maximize R2 score"""
        return "maximize"
    
    def _calculate_metrics(self, model, X_test, y_test) -> Dict[str, float]:
        """Calculate regression metrics"""
        predictions = model.predict(X_test)
        return {
            'r2': r2_score(y_test, predictions),
            'mse': mean_squared_error(y_test, predictions),
            'mae': mean_absolute_error(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions))
        }
    
    def _get_optuna_params(self, model_name: str, trial) -> Dict[str, Any]:
        """Override to add specific parameters for regression"""
        # Получаем базовые параметры из родительского класса
        params = super()._get_optuna_params(model_name, trial)
        
        # Добавляем специфичные параметры для CatBoost в задаче регрессии
        if model_name == 'catboost':
            # Получаем параметры для регрессии CatBoost
            catboost_regression_params = self.optuna_params_config.get('catboost_regression', {})
            for param_name, param_config in catboost_regression_params.items():
                param_type = param_config.get('type', 'categorical')
                
                if param_type == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, 
                        param_config.get('choices', [])
                    )
        
        return params