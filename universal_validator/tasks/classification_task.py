"""Classification task implementation"""
from typing import Dict, Any
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .base_task import BaseTask

class ClassificationTask(BaseTask):
    """Classification task implementation"""    
    CONFIG_SECTION = 'classification'
    DEFAULT_SCORING = 'accuracy'
    
    def _calculate_metrics(self, model, X_test, y_test) -> Dict[str, float]:
        """Calculate classification metrics"""
        predictions = model.predict(X_test)
        return {
            'accuracy': accuracy_score(y_test, predictions),
            'f1': f1_score(y_test, predictions, average='weighted'),
            'precision': precision_score(y_test, predictions, average='weighted'),
            'recall': recall_score(y_test, predictions, average='weighted')
        }
