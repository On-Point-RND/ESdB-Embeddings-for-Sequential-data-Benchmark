import glob
from pathlib import Path
from typing import Literal
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_utils import (
    ReconPredictor,
)
from ...types import Batch, Seq
from ...model import BaseModel
from ..preprocess import Batch2Seq
from ...model.seq2seq import GRU
from copy import deepcopy
from ...model import build_model, FrozenModel
from .gpt import GPT


class NTPModel(BaseModel):
    def __init__(
        self,
        # Preprocess:
        cat_cardinalities,
        num_features,
        cat_emb_dim=16,
        num_emb_dim=16,
        time_process="cat",
        num_norm=True,
        # Encoder:
        enc_hidden_size=128,
        enc_num_layers=1,
        use_transformer=False,
        n_head=4,
        max_len=1000,
        # Loss weights:
        mse_weight=1,
        ce_weight=1,
    ):
        super().__init__()

        self.mse_weight = mse_weight
        self.ce_weight = ce_weight
        ### PROCESSORS ###
        self.processor = Batch2Seq(
            cat_cardinalities=cat_cardinalities,
            num_features=num_features,
            cat_emb_dim=cat_emb_dim,
            num_emb_dim=num_emb_dim,
            time_process=time_process,
            num_norm=num_norm,
        )
        self.use_transformer = use_transformer
        self.input_size = self.processor.output_dim
        ### ENCODER ###
        if self.use_transformer:
            self.encoder = GPT(
                input_size=self.processor.output_dim,
                n_embd=enc_hidden_size,
                n_head=n_head,
                n_layer=enc_num_layers,
                max_len=max_len,
            )
        else:
            self.encoder = GRU(
                input_size=self.processor.output_dim,
                hidden_size=enc_hidden_size,
                num_layers=enc_num_layers,
            )
        
        ### ACTIVATION ###
        self.act = nn.GELU()

        ### LOSS ###
        self.recon_predictor = ReconPredictor(
            enc_hidden_size,
            cat_cardinalities,
            num_features,
        )

    def reconstruction_loss(self, batch: Batch):
        """
        output: Dict that is outputed from forward method
        """
        output = self.reconstruct(batch)
        ce_loss, mse_loss = self.recon_predictor.loss(output["prediction"], batch)
        total_ce_loss = sum([value for _, value in ce_loss.items()])
        total_mse_loss = sum([value for _, value in mse_loss.items()])

        losses_dict = {
            "total_mse_loss": total_mse_loss,
            "total_CE_loss": total_ce_loss,
        }

        total_loss = self.mse_weight * total_mse_loss + self.ce_weight * total_ce_loss
        losses_dict["reconstruction_loss"] = total_loss

        return losses_dict, output
    
    def encode(self, batch: Batch):
        x = self.processor(batch)
        all_hid = self.encoder(x)
        return all_hid
    
    def reconstruct(self, batch: Batch):
        global_hidden = self.encode(batch)
        
        if isinstance(global_hidden, Seq):
            enc_out = global_hidden.tokens
        else:
            enc_out = global_hidden
             
        #[0, ..., L-2]
        pred_input = enc_out[:-1, :, :] 
        
        pred = self.recon_predictor(pred_input)
        
        res_dict = {
            "prediction": pred,
            "latent": global_hidden,
        }
        return res_dict


class NTPPretrainer(NTPModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, batch: Batch):
        if self.use_transformer:
            batch = batch.tail_clamp(self.encoder.config.max_len)
        check_batch = deepcopy(batch)
        losses, output = self.reconstruction_loss(batch)
        metrics = self.recon_predictor.metrics(output["prediction"], batch)
        assert batch == check_batch
        losses["loss"] = losses["reconstruction_loss"]
        losses.update(metrics) 
        return losses


class NTPEncoder(NTPPretrainer):
    @property
    def output_dim(self):
        return self.encoder.output_dim

    def forward(self, batch: Batch):
        if self.use_transformer:
            batch = batch.tail_clamp(self.encoder.config.max_len)
        return self.encode(batch)
