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

class SplitStrategy(Enum):
    TIME_BASED = "time_based"
    USER_BASED = "user_based"
    FEATURE_BASED = "feature_based"

@dataclass
class DatasetSpec:
    name: str
    version: str
    task_type: TaskType
    features: List[str]
    target: str
    sequence_id: str
    timestamp_col: Optional[str] = None
    
    @classmethod
    def from_config(cls, config: DictConfig) -> 'DatasetSpec':
        """Create DatasetSpec from OmegaConf configuration"""
        return cls(
            name=config.name,
            version=config.version,
            task_type=TaskType(config.task_type),
            features=list(config.features),
            target=config.target,
            sequence_id=config.sequence_id,
            timestamp_col=config.get('timestamp_col')
        )
