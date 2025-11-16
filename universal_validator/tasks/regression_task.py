"""Regression task implementation"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, Any
from omegaconf import DictConfig

from ..core.base_classes import BaseTask
from ..core.types import TaskType

class RegressionTask(BaseTask):
    """Regression task implementation"""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'mlp': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=100)
        }

    def execute(self, split_data: Dict[str, Any], task_type: TaskType) -> Dict[str, Any]:
        """Execute regression task"""
        if task_type != TaskType.REGRESSION:
            raise ValueError("This task only supports regression")

        print("Running regression task...")
        results = {}

        for name, model in self.models.items():
            print(f"  Training {name}...")
            model.fit(split_data['X_train'], split_data['y_train'])
            predictions = model.predict(split_data['X_test'])
            mse = mean_squared_error(split_data['y_test'], predictions)
            r2 = r2_score(split_data['y_test'], predictions)

            results[name] = {
                'mse': mse,
                'r2': r2,
                'predictions': predictions,
                'model': model
            }
            print(f"    {name} MSE: {mse:.4f}, R2: {r2:.4f}")

        return results
