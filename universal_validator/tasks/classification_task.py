"""Classification task implementation"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from typing import Dict, Any
from omegaconf import DictConfig

from ..core.base_classes import BaseTask
from ..core.types import TaskType

class ClassificationTask(BaseTask):
    """Classification task implementation"""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'mlp': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=100)
        }

    def execute(self, split_data: Dict[str, Any], task_type: TaskType) -> Dict[str, Any]:
        """Execute classification task"""
        if task_type != TaskType.CLASSIFICATION:
            raise ValueError("This task only supports classification")

        print("Running classification task...")
        results = {}

        for name, model in self.models.items():
            print(f"  Training {name}...")
            model.fit(split_data['X_train'], split_data['y_train'])
            predictions = model.predict(split_data['X_test'])
            accuracy = accuracy_score(split_data['y_test'], predictions)

            results[name] = {
                'accuracy': accuracy,
                'predictions': predictions,
                'model': model
            }
            print(f"    {name} accuracy: {accuracy:.4f}")

        return results
