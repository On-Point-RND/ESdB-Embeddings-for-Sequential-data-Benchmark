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

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, norm, dim_feedforward):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            norm=norm,
        )

    def forward(self, tgt, memory, tgt_mask):
        return self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)

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
        # Decoder:
        use_transformer=False,
        dec_hidden_size=128,
        dec_num_layers=3,
        dec_num_heads=8,
        dec_scale_hidden=2,
        max_len=1000,
        # Loss weights:
        mse_weight=1,
        reconstruction_weight=1,
        ce_weight=1,
    ):
        super().__init__()

        self.mse_weight = mse_weight
        self.ce_weight = ce_weight
        self.reconstruction_weight = reconstruction_weight
        ### PROCESSORS ###
        self.processor = Batch2Seq(
            cat_cardinalities=cat_cardinalities,
            num_features=num_features,
            cat_emb_dim=cat_emb_dim,
            num_emb_dim=num_emb_dim,
            time_process=time_process,
            num_norm=num_norm,
        )
        self.input_size = self.processor.output_dim
        ### NORMS ###
        self.post_encoder_norm = nn.LayerNorm(enc_hidden_size)

        ### ENCODER ###
        self.encoder = GRU(
            input_size=self.processor.output_dim,
            hidden_size=enc_hidden_size,
            num_layers=enc_num_layers,
        )
        ### DECODER ###
        self.use_transformer = use_transformer
        self.decoder = None
        
        if self.use_transformer:
            self.decoder = TransformerDecoder(
                d_model=enc_hidden_size,
                nhead=dec_num_heads,
                num_layers=dec_num_layers,
                norm=nn.LayerNorm(enc_hidden_size),
                dim_feedforward=enc_hidden_size * dec_scale_hidden,
            )

        ### ACTIVATION ###
        self.act = nn.GELU()

        ### LOSS ###
        self.recon_predictor = ReconPredictor(
            dec_hidden_size,
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
    
    def reconstruct(self, batch: Batch):
        global_hidden = self.encode(batch)
        
        if isinstance(global_hidden, Seq):
            dec_out = global_hidden.tokens
        else:
            dec_out = global_hidden
        
        if self.decoder is not None:
            if self.use_transformer:
                seq_len = dec_out.size(0)
                tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len).to(dec_out.device)
                dec_out = self.decoder(tgt=dec_out, memory=dec_out, tgt_mask=tgt_mask)
            else:
                raise NotImplementedError
             
        #[0, ..., L-2]
        pred_input = dec_out[:-1, :, :] 
        
        pred = self.recon_predictor(pred_input)
        
        res_dict = {
            "prediction": pred,
            "latent": global_hidden,
        }
        return res_dict

    def encode(self, batch: Batch):
        x = self.processor(batch)
        all_hid = self.encoder(x)
        return all_hid


class NTPPretrainer(NTPModel):
    def __init__(self, normalize_z: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalize_z = normalize_z

    def forward(self, batch: Batch):
        check_batch = deepcopy(batch)
        losses, output = self.reconstruction_loss(batch)
        metrics = self.recon_predictor.metrics(output["prediction"], batch)
        assert batch == check_batch
        losses["loss"] = self.reconstruction_weight * losses["reconstruction_loss"]
        losses.update(metrics) 
        return losses


class NTPEncoder(NTPPretrainer):
    @property
    def output_dim(self):
        return self.encoder.output_dim

    def forward(self, batch: Batch):
        return self.encode(batch)
