"""AGE dataset implementation"""
import pandas as pd
from typing import Tuple
from ..core.base_classes import BaseDataset
from ..core.types import DatasetSpec, TaskType

class AgeDataset(BaseDataset):
    """AGE dataset implementation"""
    
    def __init__(self, dataset_config):
        spec = DatasetSpec(
            name=dataset_config.name,
            version=dataset_config.version,
            task_type=TaskType(dataset_config.task_type),
            features=list(dataset_config.features),
            target=dataset_config.target,
            sequence_id=dataset_config.sequence_id,
            timestamp_col=dataset_config.get('timestamp_col')
        )
        super().__init__(spec)
        self.transactions_url = dataset_config.transactions_url
        self.targets_url = dataset_config.targets_url
    
    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        print(f"Loading {self.spec.name} dataset...")
        
        # Load sequence data
        sequences_df = pd.read_csv(self.transactions_url, compression="gzip")
        print(f"Loaded {len(sequences_df)} sequences")
        
        # Load targets
        targets_df = pd.read_csv(self.targets_url)
        targets_df = targets_df.set_index("client_id")
        targets_df.rename(columns={"bins": "target"}, inplace=True)
        print(f"Loaded {len(targets_df)} targets")
        
        return sequences_df, targets_df
    
    def preprocess(self, sequences_df: pd.DataFrame, targets_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """AGE-specific preprocessing"""
        # Convert date columns to datetime and then to numeric format
        if 'trans_date' in sequences_df.columns:
            print("Converting trans_date to numeric format...")
            sequences_df['trans_date'] = pd.to_datetime(sequences_df['trans_date'])
            # Convert to numeric (Unix timestamp) for PTLS compatibility
            sequences_df['trans_date'] = sequences_df['trans_date'].astype('int64') // 10**9  # Convert to seconds
        
        # Filter out sequences with insufficient length
        seq_lengths = sequences_df.groupby('client_id').size()
        valid_clients = seq_lengths[seq_lengths >= 25].index
        sequences_df = sequences_df[sequences_df['client_id'].isin(valid_clients)]
        
        print(f"After preprocessing: {len(sequences_df)} sequences")
        return sequences_df, targets_df
