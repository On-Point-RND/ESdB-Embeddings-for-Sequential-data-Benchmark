"""Forecast task implementation"""
from typing import Dict, Any
from omegaconf import DictConfig

from .regression_task import RegressionTask
from ..core.types import TaskType

class ForecastTask(RegressionTask):
    """Forecast task implementation - inherits completely from RegressionTask"""
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        # No changes needed - inherits all functionality from RegressionTask
    
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
