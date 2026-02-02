from .base_task import BaseTask
from .classification_task import ClassificationTask
from .regression_task import RegressionTask
from .anomaly_detection_task import AnomalyDetectionTask
from .forecast_task import ForecastTask

__all__ = [
    'BaseTask',
    'ClassificationTask',
    'RegressionTask',
    'AnomalyDetectionTask',
    'ForecastTask'
]