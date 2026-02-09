from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Callable, Literal, TypeVar

import torch

from ...types import Batch


class MaskingStrategy:
    def __call__(self, batch: Batch):
        raise NotImplementedError


MASKER_REGISTRY: dict[str, type[MaskingStrategy]] = {}


TMasker = TypeVar("TMasker", bound=MaskingStrategy)


def register_masker(name: str) -> Callable[[type[TMasker]], type[TMasker]]:
    def decorator(cls: type[TMasker]) -> type[TMasker]:
        MASKER_REGISTRY[name] = cls
        return cls

    return decorator


def build_masker(
    conf: Mapping | MaskingStrategy,
    base_params: Mapping[str, Any],
) -> MaskingStrategy:
    if isinstance(conf, MaskingStrategy):
        return conf
    if not isinstance(conf, Mapping):
        raise TypeError("masker must be a Mapping or MaskingStrategy instance")

    name = conf.get("name")
    params = dict(base_params)
    params.update(dict(conf.get("params", {})))
    if name is None:
        raise ValueError("masker config must contain `name`")

    cls = MASKER_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown masker name: {name}")
    return cls(**params)


@dataclass
class Bert4RecMaskerBase(MaskingStrategy):
    cat_cardinalities: Mapping[str, int]
    ignore_index: int = -100
    mask_prob: float = 0.15
    span_len: int = 5
    span_share: float = 0.5
    random_token_prob: float = 0.1
    keep_token_prob: float = 0.1
    num_mask_fill: Literal["zero", "mean", "noise"] = "zero"

    def clone_with(self, **overrides: Any) -> "Bert4RecMaskerBase":
        return replace(self, **overrides)

    def __call__(self, batch: Batch):
        if batch.emb_features is not None or batch.emb_features_names is not None:
            raise ValueError("Bert4Rec masking does not support emb_features")
        seq_len, device = self._resolve_seq_meta(batch)


        pad_valid = torch.arange(seq_len, device=device)[:, None] < batch.lengths[None, :]
        event_mask = self._sample_event_mask(batch.lengths, seq_len, device)

        masked_cat = batch.cat_features
        cat_target = None
        cat_loss_mask = None

        if batch.cat_features is not None and batch.cat_features_names is not None:
            masked_cat = batch.cat_features.clone()
            cat_target = torch.full_like(masked_cat, self.ignore_index)
            cat_loss_mask = torch.zeros_like(masked_cat, dtype=torch.bool)

            for i, name in enumerate(batch.cat_features_names):
                vals = masked_cat[:, :, i]
                valid = event_mask & pad_valid
                if batch.cat_mask is not None:
                    valid = valid & batch.cat_mask[:, :, i]
                else:
                    valid = valid & (vals != 0)

                cat_target[:, :, i] = torch.where(
                    valid, batch.cat_features[:, :, i], cat_target[:, :, i]
                )
                cat_loss_mask[:, :, i] = valid

                mask_token_id = self.cat_cardinalities[name]
                if self.random_token_prob > 0.0 or self.keep_token_prob > 0.0:
                    rand = torch.rand(seq_len, batch.lengths.numel(), device=device)
                    p_mask = 1.0 - self.random_token_prob - self.keep_token_prob
                    mask_mask = (rand < p_mask) & valid
                    random_mask = (rand >= p_mask) & (rand < (1.0 - self.keep_token_prob)) & valid
                    vals[mask_mask] = mask_token_id
                    if random_mask.any():
                        random_tokens = torch.randint(
                            1,
                            max(self.cat_cardinalities[name], 2),
                            size=(seq_len, batch.lengths.numel()),
                            device=device,
                        )
                        vals[random_mask] = random_tokens[random_mask]
                else:
                    vals[valid] = mask_token_id

                masked_cat[:, :, i] = vals

        masked_num = batch.num_features
        num_target = None
        num_loss_mask = None

        if batch.num_features is not None:
            masked_num = batch.num_features.clone()
            num_target = batch.num_features.clone()
            num_loss_mask = event_mask[:, :, None].expand_as(masked_num).clone()
            if batch.num_mask is not None:
                num_loss_mask &= batch.num_mask
            else:
                num_loss_mask &= pad_valid[:, :, None]

            for i in range(masked_num.shape[2]):
                selector = num_loss_mask[:, :, i]
                valid_positions = pad_valid
                if batch.num_mask is not None:
                    valid_positions = valid_positions & batch.num_mask[:, :, i]
                vals = masked_num[:, :, i]
                self._fill_numeric(vals, selector, valid_positions)
                masked_num[:, :, i] = vals

        masked_batch = replace(
            batch,
            cat_features=masked_cat,
            num_features=masked_num,
        )

        targets = {
            "cat_target": cat_target,
            "cat_loss_mask": cat_loss_mask,
            "num_target": num_target,
            "num_loss_mask": num_loss_mask,
        }
        return masked_batch, targets

    @staticmethod
    def _resolve_seq_meta(batch: Batch) -> tuple[int, torch.device]:
        if batch.cat_features is not None:
            return batch.cat_features.shape[0], batch.cat_features.device
        if batch.num_features is not None:
            return batch.num_features.shape[0], batch.num_features.device
        raise ValueError("Batch must contain cat_features or num_features")

    def _sample_event_mask(self, lengths: torch.Tensor, seq_len: int, device: torch.device):
        raise NotImplementedError

    def _fill_numeric(self, values: torch.Tensor, selector: torch.Tensor, valid_positions: torch.Tensor):
        if self.num_mask_fill == "zero":
            values[selector] = 0.0
            return

        valid_vals = values[valid_positions & (~selector)]
        if valid_vals.numel() == 0:
            values[selector] = 0.0
            return

        mean = valid_vals.mean()
        if self.num_mask_fill == "mean":
            values[selector] = mean
            return

        std = valid_vals.std(unbiased=False).clamp_min(1e-6)
        values[selector] = torch.randn_like(values[selector]) * std + mean


@register_masker("PointMasker")
@dataclass
class PointMasker(Bert4RecMaskerBase):
    def _sample_event_mask(self, lengths: torch.Tensor, seq_len: int, device: torch.device):
        mask = torch.rand(seq_len, lengths.numel(), device=device) < self.mask_prob
        valid = torch.arange(seq_len, device=device)[:, None] < lengths[None, :]
        return mask & valid


@register_masker("SpanMasker")
@dataclass
class SpanMasker(Bert4RecMaskerBase):
    def _sample_event_mask(self, lengths: torch.Tensor, seq_len: int, device: torch.device):
        mask = torch.zeros(seq_len, lengths.numel(), device=device, dtype=torch.bool)
        for b, seq_len_t in enumerate(lengths):
            cur_len = int(seq_len_t.item())
            if cur_len <= 0:
                continue

            target_count = max(1, int(round(cur_len * self.mask_prob)))
            covered = 0
            while covered < target_count:
                cur_span = min(self.span_len, target_count - covered, cur_len)
                start = torch.randint(0, cur_len - cur_span + 1, (1,), device=device).item()
                mask[start : start + cur_span, b] = True
                covered += cur_span
        return mask


@register_masker("MixedMasker")
@dataclass
class MixedMasker(Bert4RecMaskerBase):
    def _sample_event_mask(self, lengths: torch.Tensor, seq_len: int, device: torch.device):
        mask = torch.zeros(seq_len, lengths.numel(), device=device, dtype=torch.bool)
        for b, seq_len_t in enumerate(lengths):
            cur_len = int(seq_len_t.item())
            if cur_len <= 0:
                continue

            target_count = max(1, int(round(cur_len * self.mask_prob)))
            point_count = int(round(target_count * (1.0 - self.span_share)))
            if point_count > 0:
                idx = torch.randperm(cur_len, device=device)[:point_count]
                mask[idx, b] = True
            target_count -= point_count

            covered = 0
            while covered < target_count:
                cur_span = min(self.span_len, target_count - covered, cur_len)
                start = torch.randint(0, cur_len - cur_span + 1, (1,), device=device).item()
                mask[start : start + cur_span, b] = True
                covered += cur_span
        return mask

