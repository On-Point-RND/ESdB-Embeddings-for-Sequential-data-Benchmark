"""Abstract base classes with OmegaConf support"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Union
import pandas as pd
from omegaconf import DictConfig

from .types import TaskType, DatasetSpec

class BaseDataset(ABC):
    """Abstract base class for all datasets"""
    
    def __init__(self, spec: DatasetSpec):
        self.spec = spec
        self._data = None
    
    @abstractmethod
    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load dataset and return (sequences_df, targets_df)"""
        pass
    
    @abstractmethod
    def preprocess(self, sequences_df: pd.DataFrame, targets_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Preprocess the dataset"""
        pass
    
    def check(self) -> bool:
        """Check dataset integrity"""
        try:
            sequences_df, targets_df = self.load()
            assert len(sequences_df) > 0, "No sequence data"
            assert len(targets_df) > 0, "No target data"
            assert self.spec.sequence_id in sequences_df.columns, f"Missing {self.spec.sequence_id}"
            assert self.spec.target in targets_df.columns, f"Missing {self.spec.target}"
            return True
        except Exception as e:
            import logging
            logging.error(f"Dataset check failed: {e}")
            return False

class BaseEmbedder(ABC):
    """Abstract base class for all embedding methods"""
    
    def __init__(self, config: DictConfig):
        self.config = config
    
    @abstractmethod
    def fit(self, sequences_df: pd.DataFrame) -> Any:
        """Train embedding model"""
        pass
    
    @abstractmethod
    def transform(self, sequences_df: pd.DataFrame) -> pd.DataFrame:
        """Generate embeddings"""
        pass
    
    def fit_transform(self, sequences_df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step"""
        self.fit(sequences_df)
        return self.transform(sequences_df)
    
    def save_embeddings(self, embeddings_df: pd.DataFrame, path: str):
        """Save embeddings to parquet format"""
        embeddings_df.to_parquet(path, index=False)
        print(f"Embeddings saved to {path}")
    
    def load_embeddings(self, path: str) -> pd.DataFrame:
        """Load embeddings from parquet format"""
        embeddings_df = pd.read_parquet(path)
        print(f"Embeddings loaded from {path}: {len(embeddings_df)} sequences")
        return embeddings_df

class BaseSplitter(ABC):
    """Abstract base class for data splitting strategies"""
    
    def __init__(self, config: DictConfig):
        self.config = config
    
    @abstractmethod
    def split(self, embeddings: pd.DataFrame, targets: pd.DataFrame, task_type: TaskType) -> Dict[str, Any]:
        """Split data into train/test sets"""
        pass

class BaseTask(ABC):
    """Abstract base class for all downstream tasks"""
    
    def __init__(self, config: DictConfig):
        self.config = config
    
    @abstractmethod
    def execute(self, split_data: Any, task_type: TaskType) -> Dict[str, Any]:
        """Execute downstream task"""
        pass
