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
from ...types import Batch
from ...model import BaseModel
from ..preprocess import Batch2Seq
from ...model.seq2seq import GRU
from copy import deepcopy
from ...model import build_model, FrozenModel


class GenModel(BaseModel):
    def __init__(
        self,
        # Preprocess:
        cat_cardinalities,
        num_features,
        cat_emb_dim=16,
        num_emb_dim=16,
        time_process: Literal["cat", "diff", "none"] = "cat",
        num_norm=True,
        # Encoder:
        enc_hidden_size=128,  # get from contrastive model
        enc_num_layers=1,
        # Decoder:
        dec_hidden_size=128,
        dec_num_layers=3,
        dec_num_heads=8,
        dec_scale_hidden=2,
        max_len=1000,
        # Loss weights:
        mse_weight=1,
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

        ### HIDDEN TO X0 PROJECTION ###
        self.hidden_to_x0 = nn.Linear(enc_hidden_size, self.input_size)

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
        pred = self.decode(batch, global_hidden)
        res_dict = {
            "prediction": pred,
            "latent": global_hidden,
        }
        return res_dict

    def encode(self, batch: Batch):
        # encode + decode in fact
        x = self.processor(batch)
        all_hid = self.encoder(x)
        return all_hid


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


class MLEMPretrainer(GenModel):
    def __init__(
        self, contr_model_folder: str, normalize_z: bool = False, *args, **kwargs
    ):
        contr_model_config = OmegaConf.load(Path(contr_model_folder) / "config.yaml")[
            "unsupervised_model"
        ]  # type: ignore
        kwargs["enc_hidden_size"] = contr_model_config["encoder"]["params"][
            "hidden_size"
        ]
        super().__init__(*args, **kwargs)
        self.contrastive_model = build_model(contr_model_config)
        contr_model_checkpoint = Path(contr_model_folder) / "pretrain/ckpt"
        ckpts = list(glob.glob(f"{contr_model_checkpoint}/*.ckpt"))
        if len(ckpts) != 1:
            raise ValueError("Not 1 checkpoint in folder")
        self.contrastive_model.load_state_dict(
            torch.load(ckpts[0], map_location="cpu")["model"]
        )
        self.contrastive_model = FrozenModel(self.contrastive_model)

        self.normalize_z = normalize_z
        init_temp = torch.tensor(10.0)
        init_bias = torch.tensor(-10.0)
        self.bias = nn.Parameter(init_bias)
        print("Pretrain success")

    def forward(self, batch: Batch):
        check_batch = deepcopy(batch)
        losses, output = self.reconstruction_loss(batch)
        assert batch == check_batch
        losses["loss"] = self.reconstruction_weight * losses["reconstruction_loss"]
        return losses


class MLEMEncoder(MLEMPretrainer):
    @property
    def output_dim(self):
        return self.encoder.output_dim

    def forward(self, batch: Batch):
        return self.encode(batch)
