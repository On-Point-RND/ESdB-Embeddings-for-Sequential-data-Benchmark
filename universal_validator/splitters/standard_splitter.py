"""Standard data splitter implementation"""
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Dict, Any
from omegaconf import DictConfig
from abc import ABC, abstractmethod
from ..types import TaskType

class BaseSplitter(ABC):
    """Abstract base class for data splitting strategies"""
    
    def __init__(self, config: DictConfig):
        self.config = config
    
    @abstractmethod
    def split(self, embeddings: pd.DataFrame, targets: pd.DataFrame, task_type: TaskType) -> Dict[str, Any]:
        """Split data into train/test sets"""
        pass

class StandardSplitter(BaseSplitter):
    """Standard data splitting pipeline"""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.scaler = StandardScaler()

    def split(self, embeddings: pd.DataFrame, targets: pd.DataFrame, task_type: TaskType) -> Dict[str, Any]:
        """Split data into train/test sets"""
        print(f"Splitting data for {task_type.value} task...")
            
        if 'sequence_id' in embeddings.columns or embeddings.index.name:
            # Merge embeddings with targets
            merged_data = embeddings.merge(targets, left_on='sequence_id', right_index=True, how='inner')
            merged_data = merged_data.dropna(subset=['target'])
        else:
            merged_data = embeddings.copy()
            merged_data['target'] = targets.values
        print(f"Merged data: {len(merged_data)} samples")

        # Prepare features and target
        feature_cols = [col for col in merged_data.columns if col.startswith('embed_')]
        X = merged_data[feature_cols]
        y = merged_data['target']

        # Encode target for regression if needed
        if task_type == TaskType.REGRESSION and y.dtype == 'object':
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            print("Converted target to numeric for regression")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y if task_type == TaskType.CLASSIFICATION else None
        )

        print(f"Data split: {len(X_train)} train, {len(X_test)} test")

        # Preprocess features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_cols,
            'task_type': task_type
        }
