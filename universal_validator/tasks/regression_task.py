"""Regression task implementation"""
from typing import Dict, Any
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from .base_task import BaseTask
from ..types import TaskType

class RegressionTask(BaseTask):
    """Regression task implementation"""
    
    TASK_TYPE = TaskType.REGRESSION
    CONFIG_SECTION = 'regression'
    DEFAULT_SCORING = 'r2'
    
    def _calculate_metrics(self, model, X_test, y_test) -> Dict[str, float]:
        """Calculate regression metrics"""
        predictions = model.predict(X_test)
        return {
            'r2': r2_score(y_test, predictions),
            'mse': mean_squared_error(y_test, predictions),
            'mae': mean_absolute_error(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions))
        }
