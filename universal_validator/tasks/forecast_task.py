"""Forecast task implementation"""
from typing import Dict, Any
from omegaconf import DictConfig, OmegaConf

from .regression_task import RegressionTask
from ..types import TaskType

class ForecastTask(RegressionTask):
    """Forecast task implementation - inherits from RegressionTask"""
    
    def _get_task_name(self) -> str:
        """Get task name for config lookup"""
        return 'forecast'
    
    def _get_supported_task_type(self) -> TaskType:
        return TaskType.FORECAST
    
    def _get_task_specific_config(self) -> DictConfig:
        """Get task-specific configuration with fallback to regression"""
        task_name = self._get_task_name()
        
        # Сначала пытаемся получить из новой структуры downstream.optuna.<task_name>
        if OmegaConf.is_dict(self.downstream_config) and 'optuna' in self.downstream_config:
            task_config = self.downstream_config.optuna.get(task_name, OmegaConf.create({}))
            if task_config and OmegaConf.is_dict(task_config) and len(task_config) > 0:
                return task_config
        
        # Если нет, пытаемся получить из старой структуры (прямо в downstream)
        task_config = self.downstream_config.get(task_name, OmegaConf.create({}))
        if task_config and OmegaConf.is_dict(task_config) and len(task_config) > 0:
            return task_config
        
        # Если конфиг для forecast пустой, используем конфиг regression
        regression_config = self.downstream_config.get('regression', OmegaConf.create({}))
        if regression_config and OmegaConf.is_dict(regression_config) and len(regression_config) > 0:
            return regression_config
        
        # Если и regression конфиг пустой, возвращаем пустой
        return OmegaConf.create({})
    
    def execute(self, split_data: Dict[str, Any], task_type: TaskType) -> Dict[str, Any]:
        """Execute forecast task - add forecast identifier"""
        if task_type != TaskType.FORECAST:
            raise ValueError(f"This task only supports forecasting")
        
        print("Running forecast task...")
        
        # Use parent's execute method
        results = super().execute(split_data, TaskType.REGRESSION)
        
        # Add forecast identifier
        for name in results:
            results[name]['task_type'] = 'forecast'
            
        return results
