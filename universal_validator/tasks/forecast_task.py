"""Forecast task implementation"""
from .regression_task import RegressionTask
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, Any
from omegaconf import DictConfig
import numpy as np

from ..core.base_classes import BaseTask
from ..core.types import TaskType

from catboost import CatBoostRegressor

class ForecastTask(RegressionTask):
    """Forecast task implementation - inherits completely from RegressionTask"""
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        # No changes needed - inherits all functionality from RegressionTask
        self.models = {
            # 'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'mlp': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000),
            'catboost': CatBoostRegressor(
                iterations=100,
                random_state=42, 
                verbose=False,
                thread_count=1
            )
        }
        
        # Search spaces for hyperparameter optimization
        self.search_spaces = {
            'random_forest': {
                'n_estimators': [50, 100, 150, 200],
                'max_depth': [3, 5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
            },
            'mlp': {
                'hidden_layer_sizes': [64, 128],
                'alpha': [0.0001, 0.001, 0.01, 0.1],
                'learning_rate_init': [0.001, 0.01, 0.1],
            },
            'catboost': {
                'iterations': [50, 100, 150, 200],
                'depth': [4, 6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'l2_leaf_reg': [1, 3, 5, 7, 9],
            }
        }

    
    def execute(self, split_data: Dict[str, Any], task_type: TaskType) -> Dict[str, Any]:
        """Execute forecast task"""
        if task_type != TaskType.FORECAST:
            raise ValueError("This task only supports forecasting")
        
        print("Running forecast task...")
        print("Note: Using regression models for forecasting with different target variables")
        
        # Call parent's execute method, passing REGRESSION as the task type
        # since all the models and logic are the same
        results = super().execute(split_data, TaskType.REGRESSION)
        
        # Add forecast identifier to results
        for name in results:
            results[name]['task_type'] = 'forecast'
            
        return results
