"""Anomaly detection task implementation"""
from typing import Dict, Any
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from .base_task import BaseTask
from ..types import TaskType

class AnomalyDetectionTask(BaseTask):
    """Anomaly detection task implementation"""
    
    TASK_TYPE = TaskType.ANOMALY_DETECTION
    CONFIG_SECTION = 'anomaly_detection'
    DEFAULT_SCORING = 'roc_auc'
    
    def _print_task_info(self, split_data: Dict[str, Any]) -> None:
        """Print anomaly class distribution"""
        train_anomaly_ratio = split_data['y_train'].mean()
        test_anomaly_ratio = split_data['y_test'].mean()
        print(f"Class distribution - Train: {train_anomaly_ratio:.2%} anomalies, Test: {test_anomaly_ratio:.2%} anomalies")
    
    def _calculate_metrics(self, model, X_test, y_test) -> Dict[str, float]:
        """Calculate anomaly detection metrics"""
        predictions = model.predict(X_test)
        
        metrics = {
            'f1': f1_score(y_test, predictions),
            'precision': precision_score(y_test, predictions),
            'recall': recall_score(y_test, predictions),
        }
        
        # Calculate AUC if model supports probability
        auc = 0.0
        if hasattr(model, 'predict_proba'):
            try:
                y_scores = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_scores)
            except:
                pass
        metrics['auc'] = auc
        
        return metrics
