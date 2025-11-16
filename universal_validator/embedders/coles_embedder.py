"""CoLES embedder implementation"""
import torch
import pandas as pd
import pytorch_lightning as pl
from functools import partial
from typing import Dict, Any
from omegaconf import DictConfig, OmegaConf

# PTLS imports
from ptls.nn import TrxEncoder, RnnSeqEncoder
from ptls.frames.coles import CoLESModule, ColesDataset
from ptls.frames.coles.split_strategy import SampleSlices
from ptls.data_load.datasets import MemoryMapDataset, inference_data_loader
from ptls.data_load.iterable_processing import SeqLenFilter
from ptls.preprocessing import PandasDataPreprocessor

from ..core.base_classes import BaseEmbedder

class CoLESEmbedder(BaseEmbedder):
    """CoLES embedding implementation"""
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.model = None
        self.preprocessor = None
        self.trainer = None

    def fit(self, sequences_df: pd.DataFrame) -> Any:
        """Train CoLES model"""
        print("Training CoLES embedder...")
        
        # PTLS preprocessing - use the DataFrame directly
        self.preprocessor = PandasDataPreprocessor(
            col_id='client_id',
            col_event_time='trans_date',
            event_time_transformation="none",
            cols_category=['small_group'],
            cols_numerical=['amount_rur'],
            return_records=True,
        )
        
        dataset = self.preprocessor.fit_transform(sequences_df)
        dataset = [seq for seq in dataset if len(seq['trans_date']) >= self.config.get('min_seq_len', 25)]
        print(f"Processed {len(dataset)} sequences for CoLES training")
        
        # Model setup
        trx_encoder_params = OmegaConf.to_container(self.config.trx_encoder_params)
        seq_encoder = RnnSeqEncoder(
            trx_encoder=TrxEncoder(**trx_encoder_params),
            **OmegaConf.to_container(self.config.seq_encoder_params)
        )
        
        self.model = CoLESModule(
            seq_encoder=seq_encoder,
            optimizer_partial=partial(torch.optim.Adam, lr=0.001),
            lr_scheduler_partial=partial(torch.optim.lr_scheduler.StepLR, step_size=10, gamma=0.9),
        )
        
        # Training
        train_dl = self._create_data_loader(dataset)
        self.trainer = pl.Trainer(
            max_epochs=self.config.training.max_epochs,
            accelerator="cuda" if torch.cuda.is_available() else "cpu",
            enable_progress_bar=True,
        )
        
        print("Starting CoLES training...")
        self.trainer.fit(self.model, train_dl)
        print("CoLES training completed!")
        return self.model

    def transform(self, sequences_df: pd.DataFrame) -> pd.DataFrame:
        """Generate embeddings using trained CoLES model"""
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        print("Generating embeddings...")
        dataset = self.preprocessor.transform(sequences_df)
        
        # Generate embeddings using the proper inference pipeline
        self.model.eval()
        
        # Use the same inference data loader as in the original code
        dl = inference_data_loader(dataset, num_workers=0, batch_size=256)
        
        # Use trainer for prediction if available, otherwise do manual inference
        if self.trainer:
            embeddings = torch.vstack(self.trainer.predict(self.model, dl))
        else:
            # Manual inference
            all_embeddings = []
            device = next(self.model.parameters()).device
            
            with torch.no_grad():
                for batch in dl:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    embedding = self.model(batch)
                    all_embeddings.append(embedding.cpu())
            
            embeddings = torch.vstack(all_embeddings)
        
        # Create embeddings DataFrame
        embeddings_df = pd.DataFrame(embeddings.numpy())
        embeddings_df = embeddings_df.add_prefix('embed_')
        
        # Extract sequence IDs
        sequence_ids = [seq['client_id'] for seq in dataset]
        embeddings_df['sequence_id'] = sequence_ids
        
        print(f"Generated embeddings for {len(embeddings_df)} sequences")
        return embeddings_df
    
    def _create_data_loader(self, dataset):
        """Create data loader for training"""
        train_data = ColesDataset(
            MemoryMapDataset(
                data=dataset,
                i_filters=[SeqLenFilter(min_seq_len=self.config.get('min_seq_len', 25))],
            ),
            splitter=SampleSlices(
                split_count=5,
                cnt_min=self.config.get('min_seq_len', 25),
                cnt_max=200,
            ),
        )
        
        from ptls.frames import PtlsDataModule
        return PtlsDataModule(
            train_data=train_data,
            train_num_workers=0,
            train_batch_size=self.config.training.batch_size,
        )
