"""Anomaly detection task implementation"""
from typing import Dict, Any
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from .base_task import BaseTask
from ..types import TaskType

class AnomalyDetectionTask(BaseTask):
    """Anomaly detection task implementation"""
    
    def _get_task_name(self) -> str:
        """Get task name for config lookup"""
        return 'anomaly_detection'
    
    def _get_supported_task_type(self) -> TaskType:
        return TaskType.ANOMALY_DETECTION
    
    def _get_default_scoring(self) -> str:
        """Use AUC scoring for anomaly detection"""
        return 'roc_auc'
    
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
    
    def _get_optuna_params(self, model_name: str, trial) -> Dict[str, Any]:
        """Override to add specific parameters for anomaly detection"""
        # Получаем базовые параметры из родительского класса
        params = super()._get_optuna_params(model_name, trial)
        
        # Добавляем специфичные параметры для CatBoost в задаче детекции аномалий
        if model_name == 'catboost':
            # Получаем параметры для классификации CatBoost
            catboost_classification_params = self.optuna_params_config.get('catboost_classification', {})
            for param_name, param_config in catboost_classification_params.items():
                param_type = param_config.get('type', 'categorical')
                
                if param_type == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, 
                        param_config.get('choices', [])
                    )
        
        return params
