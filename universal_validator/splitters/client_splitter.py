"""Client data splitter implementation"""
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, Any
from omegaconf import DictConfig

from ..core.base_classes import BaseSplitter
from ..core.types import TaskType

class ClientSplitter(BaseSplitter):
    """Client-based data splitting pipeline"""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.scaler = StandardScaler()

    def split(self, embeddings: pd.DataFrame, targets: pd.DataFrame, task_type: TaskType) -> Dict[str, Any]:
        """Split data into train/test sets using client-based split"""
        print(f"Splitting data for {task_type.value} task using client split...")
            
        if 'sequence_id' in embeddings.columns or embeddings.index.name:
            # Merge embeddings with targets
            merged_data = embeddings.merge(targets, left_on='sequence_id', right_index=True, how='inner')
            merged_data = merged_data.dropna(subset=['target'])
        else:
            merged_data = embeddings.copy()
            merged_data['target'] = targets.values
        print(f"Merged data: {len(merged_data)} samples")

        # Check if client_id column exists
        if 'client_id' not in merged_data.columns:
            raise ValueError("Data must contain 'client_id' column for client splitting")

        # Prepare features and target
        feature_cols = [col for col in merged_data.columns if col.startswith('embed_')]

        # Client-based split
        unique_clients = merged_data['client_id'].unique()
        train_client_count = int(len(unique_clients) * (1 - self.config.test_size))
        
        train_clients = unique_clients[:train_client_count]
        test_clients = unique_clients[train_client_count:]
        
        train_data = merged_data[merged_data['client_id'].isin(train_clients)]
        test_data = merged_data[merged_data['client_id'].isin(test_clients)]
        
        X_train = train_data[feature_cols]
        y_train = train_data['target']
        X_test = test_data[feature_cols] 
        y_test = test_data['target']
        
        print(f"Data split: {len(X_train)} train, {len(X_test)} test")
        print(f"Client split: {len(train_clients)} train clients, {len(test_clients)} test clients")
        print(f"Samples - Train: {len(X_train)}, Test: {len(X_test)}")

        # Encode target for regression if needed
        if task_type == TaskType.REGRESSION and y_train.dtype == 'object':
            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(y_train)
            y_test = label_encoder.transform(y_test)
            print("Converted target to numeric for regression")

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