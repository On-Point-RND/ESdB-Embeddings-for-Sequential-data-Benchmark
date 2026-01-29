"""Core type definitions with OmegaConf support"""
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Any, Dict
from omegaconf import DictConfig

class TaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    ANOMALY_DETECTION = "anomaly_detection"
    FORECAST = "forecast"
