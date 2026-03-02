import contextlib
from collections.abc import Mapping
from typing import Literal, cast

import torch
import torch.nn.functional as F
from torch import nn

from .fast_blocks import TransformerBlockFast
from .masking import build_masker
from ..basemodel import BaseModel
from ..preprocess import Batch2Seq
from ...types import Batch

AggregationMode = Literal["last", "mean", "max", "mean_last_k"]


class PositionalEmbedding(nn.Module):
    """Learnable positional embeddings."""

    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape[0], x.shape[1]
        return self.pe.weight[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)


class TransformerBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        hidden_size: int,
        attn_heads: int,
        feed_forward_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_size, attn_heads, dropout=dropout, batch_first=True
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.pff = PositionwiseFeedForward(
            d_model=hidden_size, d_ff=feed_forward_hidden, dropout=dropout
        )
        self.pff_dropout = nn.Dropout(dropout)
        self.pff_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        x_norm = self.attention_norm(x)
        attn_out, _ = self.attention(
            x_norm, x_norm, x_norm, key_padding_mask=~mask.bool(), need_weights=False
        )
        y = x + self.attention_dropout(attn_out)
        z = y + self.pff_dropout(self.pff(self.pff_norm(y)))
        return self.dropout(z)


class PositionwiseFeedForward(nn.Module):
    """Feed-forward sub-layer."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class BertEmbedding(nn.Module):
    """Feature embedding for BERT4Rec using Batch2Seq + projection + positional encoding."""

    def __init__(
        self,
        cat_cardinalities: Mapping[str, int],
        num_features: list[str] | None,
        hidden_size: int,
        max_len: int,
        dropout: float = 0.1,
        enable_positional_embedding: bool = True,
        time_process: Literal["cat", "diff", "none"] = "none",
        cat_emb_dim: int | Mapping[str, int] | None = None,
        num_emb_dim: int | None = None,
        num_norm: bool = False,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.enable_positional_embedding = enable_positional_embedding

        adjusted_cards = {name: card + 1 for name, card in cat_cardinalities.items()}

        self.processor = Batch2Seq(
            cat_cardinalities=adjusted_cards,
            num_features=[] if num_features is None else num_features,
            cat_emb_dim=cat_emb_dim,
            num_emb_dim=num_emb_dim,
            time_process=time_process,
            num_norm=num_norm,
        )

        self.sequential = nn.Sequential(
            nn.Linear(self.processor.output_dim, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),   
        )

        self.dropout = nn.Dropout(p=dropout)
        if self.enable_positional_embedding:
            self.position = PositionalEmbedding(max_len=max_len, d_model=hidden_size)

    def forward(self, batch: Batch) -> torch.Tensor:
        if batch.emb_features is not None or batch.emb_features_names is not None:
            raise ValueError("Bert4Rec does not support emb_features")
        seq = self.processor(batch, copy=False)
        x = self.sequential(seq.tokens.permute(1, 0, 2))
        if self.enable_positional_embedding:
            x = x + self.position(x)
        return self.dropout(x)


class Bert4Rec(BaseModel):
    """BERT-style encoder for masked reconstruction of all sequence features."""

    def __init__(
        self,
        cat_cardinalities: Mapping[str, int],
        num_features: list[str] | None,
        masker: Mapping,
        max_len: int = 100,
        hidden_size: int = 256,
        embedding_size: int = 128,
        num_blocks: int = 2,
        num_heads: int = 4,
        num_passes_over_block: int = 1,
        dropout: float = 0.1,
        time_process: Literal["cat", "diff", "none"] = "none",
        cat_emb_dim: int | Mapping[str, int] | None = None,
        num_emb_dim: int | None = None,
        num_norm: bool = False,
        query_aggregation: AggregationMode = "last",
        query_aggregation_k: int | None = None,
        enable_positional_embedding: bool = True,
        reconstruction_ce_weight: float = 1.0,
        reconstruction_mse_weight: float = 1.0,
        acceleration_config: Mapping | None = None,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()

        self.max_len = max_len
        self.num_passes_over_block = num_passes_over_block
        self.reconstruction_ce_weight = reconstruction_ce_weight
        self.reconstruction_mse_weight = reconstruction_mse_weight
        self.ignore_index = ignore_index

        self.query_aggregation = query_aggregation
        self.query_aggregation_k = query_aggregation_k

        self.num_feature_names = [] if num_features is None else num_features
        self.cat_features_names = list(cat_cardinalities.keys())

        self.item_embedder = BertEmbedding(
            cat_cardinalities=cat_cardinalities,
            num_features=num_features,
            hidden_size=hidden_size,
            max_len=max_len,
            dropout=dropout,
            enable_positional_embedding=enable_positional_embedding,
            time_process=time_process,
            cat_emb_dim=cat_emb_dim,
            num_emb_dim=num_emb_dim,
            num_norm=num_norm,
        )

        block_conf = (
            acceleration_config.get("transformer_block")
            if acceleration_config is not None
            else None
        )
        if block_conf is not None:
            self.transformer_blocks = nn.ModuleList(
                [
                    TransformerBlockFast(
                        hidden_size,
                        num_heads,
                        4 * hidden_size,
                        dropout,
                        acceleration_config=block_conf,
                    )
                    for _ in range(num_blocks)
                ]
            )
        else:
            self.transformer_blocks = nn.ModuleList(
                [
                    TransformerBlock(
                        hidden_size,
                        num_heads,
                        4 * hidden_size,
                        dropout,
                    )
                    for _ in range(num_blocks)
                ]
            )

        self.cat_heads = nn.ModuleDict(
            {
                name: nn.Linear(embedding_size, card)
                for name, card in cat_cardinalities.items()
            }
        )
        self.num_head = (
            nn.Linear(embedding_size, len(self.num_feature_names))
            if len(self.num_feature_names) > 0
            else None
        )

        self.masker = build_masker(
            conf=masker,
            base_params={
                "cat_cardinalities": cat_cardinalities,
                "ignore_index": self.ignore_index,
            },
        )

        self.reconstruction_stem = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, embedding_size),
        )

        self._init()

    def forward(self, batch: Batch):
        batch = batch.tail_clamp(self.max_len)
        masked_batch, targets = self._mask_inputs(batch)
        x = self._encode(masked_batch)
        return self._reconstruction_output(x, targets)


    def get_query_embeddings(
        self,
        batch: Batch,
    ) -> torch.Tensor:
        batch = batch.tail_clamp(self.max_len)
        x = self._encode(batch)
        x = self.reconstruction_stem(x)
        mode = cast(AggregationMode, self.query_aggregation)
        return self._aggregate_embeddings(x, batch.lengths, mode, self.query_aggregation_k)

    @staticmethod
    def _aggregate_embeddings(
        x: torch.Tensor,
        lengths: torch.Tensor,
        aggregation: AggregationMode,
        k: int | None = None,
    ) -> torch.Tensor:
        
        if aggregation == "last":
            batch_idx = torch.arange(x.shape[0], device=x.device)
            last_idx = lengths.clamp_min(1) - 1
            return x[batch_idx, last_idx]

        idx = torch.arange(x.shape[1], device=x.device)[None, :]   # [1, T]
        valid_2d = idx < lengths[:, None]                          # [B, T]
        valid = valid_2d.unsqueeze(-1)

        if aggregation == "mean":
            summed = (x * valid).sum(dim=1)
            denom = lengths.clamp_min(1).unsqueeze(-1)
            return summed / denom

        if aggregation == "mean_last_k":
            if k is None or k < 1:
                raise ValueError("mean_last_k requires k >= 1")
            start = (lengths - k).clamp_min(0)[:, None]            # [B, 1]
            tail_2d = (idx >= start) & valid_2d                    # [B, T]
            tail = tail_2d.unsqueeze(-1)                           # [B, T, 1]
            summed = (x * tail).sum(dim=1)
            denom = tail_2d.sum(dim=1).clamp_min(1).unsqueeze(-1)
            return summed / denom

        if aggregation == "max":
            neg_inf = torch.finfo(x.dtype).min
            masked = x.masked_fill(~valid, neg_inf)
            return masked.max(dim=1).values
        
        raise ValueError(f"Aggregation {aggregation} - is not known. Check your configs.")

    def _encode(self, batch: Batch) -> torch.Tensor:
        pad_mask = self._get_pad_mask(batch)
        x = self.item_embedder(batch)
        for transformer in self.transformer_blocks:
            for _ in range(self.num_passes_over_block):
                x = transformer(x, pad_mask)
        return x

    def _mask_inputs(self, batch: Batch):
        return self.masker(batch)

    def _reconstruction_output(
        self,
        x: torch.Tensor,
        targets: dict[str, torch.Tensor | None],
    ):
        x = self.reconstruction_stem(x)
        total_ce = x.new_tensor(0.0)
        total_mse = x.new_tensor(0.0)

        cat_target = targets["cat_target"]
        if cat_target is not None:
            assert isinstance(cat_target, torch.Tensor)
            for i, name in enumerate(self.cat_features_names):
                logits = self.cat_heads[name](x).permute(0, 2, 1)
                target = cat_target[:, :, i].permute(1, 0).long()
                total_ce = total_ce + F.cross_entropy(
                    logits, target, ignore_index=self.ignore_index
                )

        num_target = targets["num_target"]
        num_loss_mask = targets["num_loss_mask"]
        if (
            self.num_head is not None
            and num_target is not None
            and num_loss_mask is not None
        ):
            assert isinstance(num_target, torch.Tensor)
            assert isinstance(num_loss_mask, torch.Tensor)
            num_pred = self.num_head(x)
            n_target = num_target.permute(1, 0, 2)
            n_mask = num_loss_mask.permute(1, 0, 2).float()
            total_mse = total_mse + (
                (num_pred - n_target) ** 2 * n_mask
            ).sum() / n_mask.sum().clamp_min(1.0)

        loss = (
            self.reconstruction_ce_weight * total_ce
            + self.reconstruction_mse_weight * total_mse
        )
        return {
            "loss": loss,
            "total_ce_loss": total_ce,
            "total_mse_loss": total_mse,
        }

    def _get_pad_mask(self, batch: Batch) -> torch.Tensor:
        if batch.cat_features is not None:
            seq_len = batch.cat_features.shape[0]
            device = batch.cat_features.device
        elif batch.num_features is not None:
            seq_len = batch.num_features.shape[0]
            device = batch.num_features.device
        else:
            raise ValueError("Batch must contain cat_features or num_features")
        return torch.arange(seq_len, device=device)[None, :] < batch.lengths[:, None]

    def _init(self) -> None:
        for _, param in self.named_parameters():
            with contextlib.suppress(ValueError):
                torch.nn.init.xavier_normal_(param.data)
