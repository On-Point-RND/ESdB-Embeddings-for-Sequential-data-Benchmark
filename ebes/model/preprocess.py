"""Preprocessing model."""

from collections.abc import Mapping, Sequence
from typing import Literal

import torch
from torch import nn

from .basemodel import BaseModel
from ..types import Seq, Batch
from copy import deepcopy


class SeqBatchNorm(nn.Module):
    def __init__(self, num_count: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_count)

    def forward(self, x, lengths):  # (len, bs, num), (bs,)
        # (len, bs)
        len_mask = torch.arange(x.shape[0], device=x.device)[:, None] < lengths

        bx = x[len_mask]
        is_training = self.bn.training
        if is_training:
            self.bn(bx)  # update BN stats using only valid features (drop padding)

        self.bn.eval()
        # but compute actual BN on all values to assure that padding features stay equal
        # to the last valid
        res = self.bn(x.reshape(-1, x.shape[-1]))
        self.bn.train(is_training)  # return the state
        return res.reshape(x.shape)


class NoisyEmbedding(nn.Embedding):
    """Embedding with additive Gaussian noise during training (regularization).

    Noise is sampled once per embedding dimension and broadcast across the
    batch, matching the original CoLES implementation where the same
    perturbation is applied to every sample in the batch.
    """

    def __init__(self, num_embeddings, embedding_dim, noise_scale=0.0, **kwargs):
        super().__init__(num_embeddings, embedding_dim, **kwargs)
        self.noise_scale = noise_scale

    def forward(self, x):
        out = super().forward(x)
        if self.training and self.noise_scale > 0:
            noise = torch.randn(self.embedding_dim, device=out.device) * self.noise_scale
            out = out + noise
        return out


class PeriodicTimeEncoding(nn.Module):
    """Learnable periodic encoding for a scalar time feature.

    Maps t -> [sin(w_1*t + phi_1), ..., sin(w_k*t + phi_k)], then embeds each
    component independently via a depthwise Conv1d (1 -> emb_dim per component).

    Inspired by Time2Vec (Kazemi et al., 2019).
    """

    def __init__(self, n_periodic: int, emb_dim: int):
        super().__init__()
        self.n_periodic = n_periodic
        self.emb_dim = emb_dim
        self.w = nn.Parameter(torch.randn(n_periodic))
        self.phi = nn.Parameter(torch.zeros(n_periodic))
        # depthwise: each periodic component independently mapped to emb_dim
        self._emb = nn.Conv1d(
            in_channels=n_periodic,
            out_channels=emb_dim * n_periodic,
            kernel_size=1,
            groups=n_periodic,
        )

    @property
    def output_dim(self) -> int:
        return self.emb_dim * self.n_periodic

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [T, B, 1]
        periodic = torch.sin(t * self.w + self.phi)  # [T, B, n_periodic]
        x = periodic.permute(1, 2, 0)                # [B, n_periodic, T]
        x = self._emb(x)                             # [B, emb_dim*n_periodic, T]
        return x.permute(2, 0, 1)                    # [T, B, emb_dim*n_periodic]


class Batch2Seq(BaseModel):
    def __init__(
        self,
        cat_cardinalities: Mapping[str, int],
        emb_dimensionalities: Mapping[str, int] | None = None,  # Done in a special way different from
        num_count: int | None = None,
        num_features: Sequence[str] | None = None,
        cat_emb_dim: int | Mapping[str, int] | None = None,
        num_emb_dim: int | None = None,
        time_process: Literal["cat", "cat_periodic", "diff", "none"] = "none",
        num_norm: bool = False,
        emb_noise: float = 0.0,
        n_periodic: int = 4,
    ):
        super().__init__()
        cat_cardinalities = cat_cardinalities if cat_cardinalities is not None else {}
        if num_count is None:
            if num_features is not None:
                num_count = len(num_features)
            else:
                num_count = 0
        if time_process != "none":
            assert time_process in [
                "diff",
                "cat",
                "cat_periodic",
            ], "time_process may only be cat|cat_periodic|diff|none"
            num_count += 1
        self._out_dim = 0

        self._cat_embs = nn.ModuleDict()
        cat_dims = []
        for name, card in cat_cardinalities.items():
            if cat_emb_dim is None:
                dim = int(min(600, round(1.6 * card**0.56)))
            elif isinstance(cat_emb_dim, int):
                dim = cat_emb_dim
            else:
                dim = cat_emb_dim[name]

            self._out_dim += dim
            cat_dims.append(dim)
            if emb_noise > 0:
                self._cat_embs[name] = NoisyEmbedding(
                    card, dim, noise_scale=emb_noise,
                )
            else:
                self._cat_embs[name] = nn.Embedding(card, dim)

        if num_emb_dim is None:
            if not cat_dims:
                raise ValueError(
                    "Auto dim choice for num embeddings does not work with no cat "
                    "features"
                )
            num_emb_dim = int(sum(cat_dims) / len(cat_dims))

        self._periodic_time = None
        self._num_emb = None
        self.batch_norm = None
        if num_count:
            self.batch_norm = SeqBatchNorm(num_count) if num_norm else None
            if time_process == "cat_periodic":
                # Last channel of num_features is time; encode it with learnable sines.
                # Remaining features (if any) go through the regular depthwise Conv1d.
                non_time_count = num_count - 1
                self._periodic_time = PeriodicTimeEncoding(n_periodic, num_emb_dim)
                self._out_dim += self._periodic_time.output_dim
                if non_time_count > 0:
                    self._num_emb = nn.Conv1d(
                        in_channels=non_time_count,
                        out_channels=num_emb_dim * non_time_count,
                        kernel_size=1,
                        groups=non_time_count,
                    )
                    self._out_dim += num_emb_dim * non_time_count
            else:
                self._num_emb = nn.Conv1d(
                    in_channels=num_count,
                    out_channels=num_emb_dim * num_count,
                    kernel_size=1,
                    groups=num_count,
                )
                self._out_dim += num_emb_dim * num_count
        ##################################################
        if emb_dimensionalities:
            self._emb_dims = emb_dimensionalities
            for name, emb_length in emb_dimensionalities.items():
                dim = emb_length
                self._out_dim += dim
        ##################################################
        
        


    @property
    def output_dim(self):
        return self._out_dim

    def forward(self, batch: Batch, copy=True) -> Seq:  # of shape (len, batch_size, )
        if copy:
            batch = deepcopy(batch)

        if not isinstance(batch.time, torch.Tensor):
            raise ValueError(
                "`time` field in batch must be a Tensor. "
                "Consider proper time preprocessing"
            )

        embs = []  # in embs there are many objects of shape (len, batch, string_length)
        masks = []
        if batch.cat_features_names:
            for i, cf in enumerate(batch.cat_features_names):
                embs.append(self._cat_embs[cf](batch[cf]))
                if batch.cat_mask is not None:
                    mask = batch.cat_mask[:, :, i].unsqueeze(2)
                    mask = torch.repeat_interleave(
                        mask, self._cat_embs[cf].embedding_dim, 2
                    )
                    masks.append(mask)

        if batch.num_features is not None:
            x = batch.num_features
            if self.batch_norm:
                x = self.batch_norm(x, batch.lengths)

            if self._periodic_time is not None:
                # Split: last channel is time, rest are regular numerics
                t = x[:, :, -1:]          # (len, batch, 1)
                embs.append(self._periodic_time(t))
                if batch.num_mask is not None:
                    time_mask = batch.num_mask[:, :, -1:]
                    masks.append(
                        torch.repeat_interleave(
                            time_mask, self._periodic_time.output_dim, dim=2
                        )
                    )
                if self._num_emb is not None:
                    x_rest = x[:, :, :-1].permute(1, 2, 0)  # (batch, non_time_count, len)
                    x_rest = self._num_emb(x_rest)
                    embs.append(x_rest.permute(2, 0, 1))
                    if batch.num_mask is not None:
                        rest_mask = batch.num_mask[:, :, :-1]
                        masks.append(
                            torch.repeat_interleave(
                                rest_mask,
                                self._num_emb.out_channels // self._num_emb.in_channels,
                                dim=2,
                            )
                        )
            else:
                assert self._num_emb is not None
                x = x.permute(1, 2, 0)  # batch, features, len
                x = self._num_emb(x)
                embs.append(x.permute(2, 0, 1))
                if batch.num_mask is not None:
                    masks.append(
                        torch.repeat_interleave(
                            batch.num_mask,
                            self._num_emb.out_channels // self._num_emb.in_channels,
                            dim=2,
                        )
                    )
        ######################################################################
        # batch.emb_features_names = (features)
        # batch.emb_features = {feature_name: (emb_dim, len, batch)} (is not needed here - batch[feature_name] is used)
        # batch.emb_mask = (len, batch, features)
        if batch.emb_features_names:
            for i, ef in enumerate(batch.emb_features_names):
                # batch[ef] = (emb_len, len, batch)
                emb = batch[ef].permute(1, 2, 0)  # (len, batch, emb_len)
                embs.append(emb)
                if batch.emb_mask is not None:
                    mask = batch.emb_mask[:, :, i].unsqueeze(2)
                    emb_dim = self._emb_dims[ef]
                    mask = torch.repeat_interleave(
                        mask, emb_dim, 2
                    )
                    masks.append(mask)
        ######################################################################
        tokens = torch.cat(embs, dim=2)
        masks = torch.cat(masks, dim=2) if len(masks) > 0 else None
        return Seq(tokens=tokens, lengths=batch.lengths, time=batch.time, masks=masks)
