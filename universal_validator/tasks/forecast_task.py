"""Forecast task implementation"""
from typing import Dict, Any

from .regression_task import RegressionTask
from ..types import TaskType

class ForecastTask(RegressionTask):
    """Forecast task implementation - inherits from RegressionTask"""
    
    TASK_TYPE = TaskType.FORECAST
    CONFIG_SECTION = 'forecast'
    
    def execute(self, split_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute forecast task - add forecast identifier"""
        print("Running forecast task...")
        
        # Use parent's execute method
        results = super().execute(split_data)
        
        # Add forecast identifier
        for name in results:
            results[name]['task_type'] = 'forecast'
            
        return results
