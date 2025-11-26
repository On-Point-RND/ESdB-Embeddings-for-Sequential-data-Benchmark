"""LastDate data splitter implementation"""
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, Any
from omegaconf import DictConfig

from ..core.base_classes import BaseSplitter
from ..core.types import TaskType

class LastDateSplitter(BaseSplitter):
    """Last date data splitting pipeline"""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.scaler = StandardScaler()

    def split(self, embeddings: pd.DataFrame, targets: pd.DataFrame, task_type: TaskType) -> Dict[str, Any]:
        """Split data into train/test sets using last date split"""
        print(f"Splitting data for {task_type.value} task...")
            
        if 'sequence_id' in embeddings.columns or embeddings.index.name:
            # Merge embeddings with targets
            merged_data = embeddings.merge(targets, left_on='sequence_id', right_index=True, how='inner')
            merged_data = merged_data.dropna(subset=['target'])
        else:
            merged_data = embeddings.copy()
            merged_data['target'] = targets.values
        print(f"Merged data: {len(merged_data)} samples")

        # Check if last date column exists
        if '_last_trans_date' not in merged_data.columns:
            raise ValueError("Data must contain '_last_trans_date' column for last date splitting")

        # Prepare features and target
        feature_cols = [col for col in merged_data.columns if col.startswith('embed_')]
        X = merged_data[feature_cols]
        y = merged_data['target']

        # Encode target for regression if needed
        if task_type == TaskType.REGRESSION and y.dtype == 'object':
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            print("Converted target to numeric for regression")

        # Last date split with no overlap
        merged_data_sorted = merged_data.sort_values('_last_trans_date')
        
        # Find the split date that ensures no overlap
        unique_dates = merged_data_sorted['_last_trans_date'].unique()
        split_index = int(len(unique_dates) * (1 - self.config.test_size))
        split_date = unique_dates[split_index]
        
        # Split data ensuring no date overlap
        train_data = merged_data_sorted[merged_data_sorted['_last_trans_date'] < split_date]
        test_data = merged_data_sorted[merged_data_sorted['_last_trans_date'] >= split_date]
        
        X_train = train_data[feature_cols]
        y_train = train_data['target']
        X_test = test_data[feature_cols] 
        y_test = test_data['target']
        
        print(f"Data split: {len(X_train)} train, {len(X_test)} test")
        print(f"Last date split: {len(X_train)} train, {len(X_test)} test")
        print(f"Date range - Train: {train_data['_last_trans_date'].min()}-{train_data['_last_trans_date'].max()}, "
              f"Test: {test_data['_last_trans_date'].min()}-{test_data['_last_trans_date'].max()}")

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