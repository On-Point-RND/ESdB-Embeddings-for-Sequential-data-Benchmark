from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Literal, cast

import torch
import torch.nn.functional as F
from torch import nn

from .fast_blocks import TransformerBlockFast
from .masking import build_masker
from .model import AggregationMode, BertEmbedding, TransformerBlock
from ..basemodel import BaseModel
from ...types import Batch


class JEPAPredictor(nn.Module):
    """Predict target latents from encoded context and target positions."""

    def __init__(
        self,
        hidden_size: int,
        embedding_size: int,
        num_heads: int,
        max_len: int,
        dropout: float,
        predictor_hidden_size: int,
    ) -> None:
        super().__init__()
        self.position = nn.Embedding(max_len, hidden_size)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.context_norm = nn.LayerNorm(hidden_size)
        self.query_norm = nn.LayerNorm(hidden_size)
        self.cross_attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, predictor_hidden_size),
            nn.GELU(),
            nn.Linear(predictor_hidden_size, hidden_size),
        )
        self.ffn_dropout = nn.Dropout(dropout)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, predictor_hidden_size),
            nn.GELU(),
            nn.Linear(predictor_hidden_size, embedding_size),
        )

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_positions: torch.Tensor,
        context_lengths: torch.Tensor,
        target_positions: torch.Tensor,
    ) -> torch.Tensor:
        context = context_tokens + self.position(context_positions)
        query = self.mask_token.expand(
            target_positions.shape[0],
            target_positions.shape[1],
            -1,
        ) + self.position(target_positions)

        attn_out, _ = self.cross_attention(
            self.query_norm(query),
            self.context_norm(context),
            self.context_norm(context),
            key_padding_mask=~JEPA._get_pad_mask_from_lengths(
                context_lengths,
                context_tokens.shape[1],
            ),
            need_weights=False,
        )
        x = query + self.attention_dropout(attn_out)
        x = x + self.ffn_dropout(self.ffn(self.ffn_norm(x)))
        return self.output_head(x)


class JEPA(BaseModel):
    """JEPA-style encoder with EMA target network and modular objectives."""

    def __init__(
        self,
        cat_cardinalities: Mapping[str, int] | None,
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
        acceleration_config: Mapping | None = None,
        ignore_index: int = -100,
        ema_tau: float = 0.99,
        predictor_hidden_size: int | None = None,
        objectives: Mapping[str, Mapping[str, Any]] | None = None,
        jepa_weight: float = 1.0,
        mlm_weight: float = 0.0,
        contrastive_weight: float = 0.0,
    ) -> None:
        super().__init__()

        self.max_len = max_len
        self.num_passes_over_block = num_passes_over_block
        self.ignore_index = ignore_index
        self.query_aggregation = query_aggregation
        self.query_aggregation_k = query_aggregation_k
        self.ema_tau = ema_tau

        cat_cardinalities = {} if cat_cardinalities is None else dict(cat_cardinalities)
        self.masker = build_masker(
            conf=masker,
            base_params={
                "cat_cardinalities": cat_cardinalities,
                "ignore_index": self.ignore_index,
            },
        )
        self.objectives = self._build_objectives(
            objectives=objectives,
            jepa_weight=jepa_weight,
            mlm_weight=mlm_weight,
            contrastive_weight=contrastive_weight,
        )

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

        predictor_hidden_size = (
            hidden_size if predictor_hidden_size is None else predictor_hidden_size
        )
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, embedding_size),
        )
        self.predictor = JEPAPredictor(
            hidden_size=hidden_size,
            embedding_size=embedding_size,
            num_heads=num_heads,
            max_len=max_len,
            dropout=dropout,
            predictor_hidden_size=predictor_hidden_size,
        )

        self.target_item_embedder = deepcopy(self.item_embedder)
        self.target_transformer_blocks = deepcopy(self.transformer_blocks)
        self.target_projection_head = deepcopy(self.projection_head)
        self._freeze_target()

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_item_embedder.eval()
        self.target_transformer_blocks.eval()
        self.target_projection_head.eval()
        return self

    def forward(self, batch: Batch):
        batch = batch.tail_clamp(self.max_len)
        masked_batch, targets = self._mask_inputs(batch)

        total_loss = torch.zeros((), device=batch.lengths.device, dtype=torch.float32)
        output: dict[str, torch.Tensor] = {}

        if self._objective_enabled("jepa"):
            jepa = self._compute_jepa_objective(batch, masked_batch, targets)
            output.update(jepa)
            total_loss = total_loss + self._objective_weight("jepa") * jepa["jepa_loss"]

        if self._objective_enabled("mlm"):
            raise NotImplementedError(
                "MLM objective scaffold exists but is not implemented yet"
            )
        if self._objective_enabled("contrastive"):
            raise NotImplementedError(
                "Contrastive objective scaffold exists but is not implemented yet"
            )

        output["loss"] = total_loss
        output["total_loss"] = total_loss
        output["jepa_weight"] = total_loss.new_tensor(self._objective_weight("jepa"))

        with torch.no_grad():
            self._momentum_update_target()

        return output

    def _compute_jepa_objective(
        self,
        batch: Batch,
        masked_batch: Batch,
        targets: Mapping[str, torch.Tensor | None],
    ) -> dict[str, torch.Tensor]:
        del masked_batch

        context_mask, target_mask, valid_mask = self._build_context_and_target_masks(
            batch,
            targets,
        )
        context_tokens, context_positions, context_lengths = self._encode_context(
            batch,
            context_mask,
        )
        target_positions, target_lengths = self._gather_indices_by_mask(target_mask)
        pred = self.predictor(
            context_tokens=context_tokens,
            context_positions=context_positions,
            context_lengths=context_lengths,
            target_positions=target_positions,
        )

        with torch.no_grad():
            target = self._encode_target(batch)
            target = self.target_projection_head(target)
            target, _, _ = self._gather_tokens_by_mask(target, target_mask)
            target = F.layer_norm(target, (target.shape[-1],))

        target_valid_mask = self._get_pad_mask_from_lengths(
            target_lengths,
            pred.shape[1],
        )
        pred_selected = pred[target_valid_mask]
        target_selected = target[target_valid_mask]

        jepa_l2 = ((pred_selected - target_selected) ** 2).sum(dim=-1).mean()
        jepa_cosine = F.cosine_similarity(pred_selected, target_selected, dim=-1).mean()
        jepa_loss = F.smooth_l1_loss(pred_selected, target_selected)
        mask_ratio = target_mask.float().sum() / valid_mask.float().sum().clamp_min(1.0)

        return {
            "jepa_loss": jepa_loss,
            "jepa_l2": jepa_l2,
            "jepa_cosine": jepa_cosine,
            "mask_ratio": mask_ratio,
        }

    def _resolve_event_mask(
        self,
        batch: Batch,
        targets: Mapping[str, torch.Tensor | None],
    ) -> torch.Tensor:
        event_mask = targets.get("event_mask")
        if isinstance(event_mask, torch.Tensor):
            return event_mask.bool().permute(1, 0)

        combined = None
        cat_target = targets.get("cat_target")
        if isinstance(cat_target, torch.Tensor):
            combined = (cat_target != self.ignore_index).any(dim=2)
        num_loss_mask = targets.get("num_loss_mask")
        if isinstance(num_loss_mask, torch.Tensor):
            num_event_mask = num_loss_mask.any(dim=2)
            combined = (
                num_event_mask if combined is None else (combined | num_event_mask)
            )

        if combined is not None:
            return combined.permute(1, 0)

        return self._get_pad_mask(batch)

    def _build_context_and_target_masks(
        self,
        batch: Batch,
        targets: Mapping[str, torch.Tensor | None],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid_mask = self._get_pad_mask(batch)
        target_mask = self._resolve_event_mask(batch, targets) & valid_mask
        non_empty = batch.lengths > 0

        missing_target = (~target_mask.any(dim=1)) & non_empty
        if missing_target.any():
            rows = missing_target.nonzero(as_tuple=False).squeeze(1)
            last_idx = batch.lengths[rows] - 1
            target_mask[rows, last_idx] = True

        context_mask = valid_mask & (~target_mask)
        missing_context = (~context_mask.any(dim=1)) & non_empty
        if missing_context.any():
            context_mask[missing_context] = valid_mask[missing_context]

        return context_mask, target_mask, valid_mask

    def _mask_inputs(self, batch: Batch):
        return self.masker(batch)

    def _encode_online(self, batch: Batch) -> torch.Tensor:
        x = self.item_embedder(batch)
        return self._apply_transformer_blocks(
            x=x,
            lengths=batch.lengths,
            transformer_blocks=self.transformer_blocks,
        )

    def _encode_context(
        self,
        batch: Batch,
        context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.item_embedder(batch)
        x, context_positions, context_lengths = self._gather_tokens_by_mask(
            x, context_mask
        )
        x = self._apply_transformer_blocks(
            x=x,
            lengths=context_lengths,
            transformer_blocks=self.transformer_blocks,
        )
        return x, context_positions, context_lengths

    def _encode_target(self, batch: Batch) -> torch.Tensor:
        x = self.target_item_embedder(batch)
        return self._apply_transformer_blocks(
            x=x,
            lengths=batch.lengths,
            transformer_blocks=self.target_transformer_blocks,
        )

    def _apply_transformer_blocks(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        transformer_blocks: nn.ModuleList,
    ) -> torch.Tensor:
        pad_mask = self._get_pad_mask_from_lengths(lengths, x.shape[1])
        for transformer in transformer_blocks:
            for _ in range(self.num_passes_over_block):
                x = transformer(x, pad_mask)
        return x

    @staticmethod
    def _gather_indices_by_mask(
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lengths = mask.sum(dim=1).long()
        max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
        if max_len == 0:
            empty = mask.new_zeros((mask.shape[0], 0), dtype=torch.long)
            return empty, lengths

        positions = torch.arange(mask.shape[1], device=mask.device)[None, :].expand_as(
            mask
        )
        gathered = (
            positions.masked_fill(~mask, mask.shape[1]).sort(dim=1).values[:, :max_len]
        )
        valid = JEPA._get_pad_mask_from_lengths(lengths, max_len)
        gathered = gathered.clamp_max(mask.shape[1] - 1)
        return gathered.masked_fill(~valid, 0), lengths

    @staticmethod
    def _gather_tokens_by_mask(
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices, lengths = JEPA._gather_indices_by_mask(mask)
        if indices.shape[1] == 0:
            empty_tokens = x[:, :0]
            return empty_tokens, indices, lengths

        gather_index = indices.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        gathered = torch.gather(x, 1, gather_index)
        return gathered, indices, lengths

    def _momentum_update_target(self) -> None:
        tau = self.ema_tau
        self._ema_update(self.target_item_embedder, self.item_embedder, tau)
        self._ema_update(self.target_transformer_blocks, self.transformer_blocks, tau)
        self._ema_update(self.target_projection_head, self.projection_head, tau)

    @staticmethod
    def _ema_update(
        target_module: nn.Module, online_module: nn.Module, tau: float
    ) -> None:
        for target_param, online_param in zip(
            target_module.parameters(), online_module.parameters()
        ):
            target_param.data.mul_(tau).add_(online_param.data, alpha=1.0 - tau)

        for target_buffer, online_buffer in zip(
            target_module.buffers(), online_module.buffers()
        ):
            target_buffer.copy_(online_buffer)

    def _freeze_target(self) -> None:
        for module in (
            self.target_item_embedder,
            self.target_transformer_blocks,
            self.target_projection_head,
        ):
            for param in module.parameters():
                param.requires_grad = False
            module.eval()

    def _get_pad_mask(self, batch: Batch) -> torch.Tensor:
        if batch.cat_features is not None:
            seq_len = batch.cat_features.shape[0]
            device = batch.cat_features.device
        elif batch.num_features is not None:
            seq_len = batch.num_features.shape[0]
            device = batch.num_features.device
        else:
            raise ValueError("Batch must contain cat_features or num_features")
        return self._get_pad_mask_from_lengths(batch.lengths, seq_len, device=device)

    @staticmethod
    def _get_pad_mask_from_lengths(
        lengths: torch.Tensor,
        seq_len: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if device is None:
            device = lengths.device
        return torch.arange(seq_len, device=device)[None, :] < lengths[:, None]

    def _objective_enabled(self, name: str) -> bool:
        return bool(self.objectives.get(name, {}).get("enabled", False))

    def _objective_weight(self, name: str) -> float:
        return float(self.objectives.get(name, {}).get("weight", 0.0))

    @staticmethod
    def _build_objectives(
        objectives: Mapping[str, Mapping[str, Any]] | None,
        jepa_weight: float,
        mlm_weight: float,
        contrastive_weight: float,
    ) -> dict[str, dict[str, float | bool]]:
        cfg: dict[str, dict[str, float | bool]] = {
            "jepa": {"enabled": True, "weight": float(jepa_weight)},
            "mlm": {"enabled": False, "weight": float(mlm_weight)},
            "contrastive": {"enabled": False, "weight": float(contrastive_weight)},
        }
        if objectives is None:
            return cfg

        for name, obj_cfg in objectives.items():
            if name not in cfg or not isinstance(obj_cfg, Mapping):
                continue
            if "enabled" in obj_cfg:
                cfg[name]["enabled"] = bool(obj_cfg["enabled"])
            if "weight" in obj_cfg:
                cfg[name]["weight"] = float(obj_cfg["weight"])
        return cfg

    def get_query_embeddings(self, batch: Batch) -> torch.Tensor:
        batch = batch.tail_clamp(self.max_len)
        x = self._encode_online(batch)
        x = self.projection_head(x)
        mode = cast(AggregationMode, self.query_aggregation)
        return self._aggregate_embeddings(
            x, batch.lengths, mode, self.query_aggregation_k
        )

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

        idx = torch.arange(x.shape[1], device=x.device)[None, :]
        valid_2d = idx < lengths[:, None]
        valid = valid_2d.unsqueeze(-1)

        if aggregation == "mean":
            summed = (x * valid).sum(dim=1)
            denom = lengths.clamp_min(1).unsqueeze(-1)
            return summed / denom

        if aggregation == "mean_last_k":
            if k is None or k < 1:
                raise ValueError("mean_last_k requires k >= 1")
            start = (lengths - k).clamp_min(0)[:, None]
            tail_2d = (idx >= start) & valid_2d
            tail = tail_2d.unsqueeze(-1)
            summed = (x * tail).sum(dim=1)
            denom = tail_2d.sum(dim=1).clamp_min(1).unsqueeze(-1)
            return summed / denom

        if aggregation == "max":
            neg_inf = torch.finfo(x.dtype).min
            masked = x.masked_fill(~valid, neg_inf)
            return masked.max(dim=1).values

        raise ValueError(
            f"Aggregation {aggregation} - is not known. Check your configs."
        )
