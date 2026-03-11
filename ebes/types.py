from dataclasses import fields, dataclass, replace
from copy import copy
from typing import Any

import numpy as np
import torch


def gather(tensor, target_ids):
    if tensor is None:
        return None
    ndim_diff = tensor.ndim - target_ids.ndim
    target_expanded = target_ids.view(*target_ids.shape, *(1,) * ndim_diff)
    target_expanded = target_expanded.expand(-1, -1, *tensor.shape[2:])
    return torch.gather(tensor, 0, target_expanded)  # [L, B, ...]


@dataclass(kw_only=True)
class Batch:
    lengths: torch.Tensor  # (batch,)
    time: np.ndarray | torch.Tensor  # (len, batch)
    index: torch.Tensor | np.ndarray | None = None  # (batch,)
    target: torch.Tensor | None = None  # (batch,), (len, batch) or (batch, n_targets)
    num_features: torch.Tensor | None = None  # (len, batch, features)
    cat_features: torch.Tensor | None = None  # (len, batch, features)
    emb_features: dict[str, torch.Tensor] | None = None  # {feature_name: (emb_dim, len, batch)}
    cat_features_names: list[str] | None = None
    num_features_names: list[str] | None = None
    emb_features_names: list[str] | None = None
    cat_mask: torch.Tensor | None = None  # (len, batch, features)
    num_mask: torch.Tensor | None = None  # (len, batch, features)
    emb_mask: torch.Tensor | None = None  # (len, batch, features)

    def to(self, device: str):
        for field in fields(self):
            f = getattr(self, field.name)
            if isinstance(f, torch.Tensor):
                setattr(self, field.name, f.to(device))
            elif field.name == "emb_features" and f is not None:
                setattr(
                    self,
                    field.name,
                    {k: v.to(device) for k, v in f.items()}
                )

        return self

    def _find_tensor_and_idx(self, feature_name):
        if self.cat_features is not None and self.cat_features_names is not None:
            try:
                cat_idx = self.cat_features_names.index(feature_name)
            except ValueError:
                pass
            else:
                return self.cat_features, cat_idx

        if self.num_features is not None and self.num_features_names is not None:
            try:
                num_idx = self.num_features_names.index(feature_name)
            except ValueError:
                pass
            else:
                return self.num_features, num_idx
        #########################################################################
        if self.emb_features_names and feature_name in self.emb_features_names:
            raise TypeError(
                f"'{feature_name}' is an embedding feature; "
                "use batch.emb_features[...]"
            )
        ##########################################################################
        raise ValueError(
            f"Cannot access feature by name {feature_name}."
            f" Known cat names are: {self.cat_features_names}."
            f" Known num names are: {self.num_features_names}."
            f" Known emb names are: {self.emb_features_names}."
        )

    def __len__(self) -> int:
        """Batch size."""

        return len(self.lengths)

    def __setitem__(self, feature_name: str, value: Any):
        if self.emb_features is not None and feature_name in self.emb_features:
            self.emb_features[feature_name] = value
        else:
            tensor, idx = self._find_tensor_and_idx(feature_name)
            tensor[:, :, idx] = value

    def __getitem__(
        self,
        feature_name: str,
    ) -> torch.Tensor:  # of shape (len, batch) or (emb_dim, len, batch)
        if self.emb_features is not None and feature_name in self.emb_features:
            return self.emb_features[feature_name]
        tensor, idx = self._find_tensor_and_idx(feature_name)
        return tensor[:, :, idx]

    def __eq__(self, other):
        assert isinstance(other, self.__class__)
        equal = True
        for field in fields(self):
            my_field, other_field = getattr(self, field.name), getattr(
                other, field.name
            )
            if isinstance(my_field, torch.Tensor | np.ndarray):
                equal = equal and (my_field == other_field).all()
        return equal

    def pop_target(self) -> torch.Tensor | None:
        target = self.target
        self.target = None
        return target

    def extract_indexes_from_batch(self):
        
        idx=None
        if self.index is not None:
            idx = self.index
            if isinstance(idx, torch.Tensor):
                idx = idx.cpu().tolist()
        
        return idx
    
    def tail_clamp(self, len_max: int):
        """Returns a new batch containing only last tail_len elements of each sequence."""
        assert self.time is not None, "Why time feature is None?"
        seq_len = self.time.shape[0]
        if seq_len <= len_max:
            return self
        new_lengths = self.lengths.clamp_max(len_max)
        start_index = (self.lengths - new_lengths).clip(0)  # [1, B]
        target_ids = (
            torch.arange(len_max, device=start_index.device)[:, None] + start_index
        )  # [target_len, B]

        return replace(
            self,
            lengths=new_lengths,
            time=gather(self.time, target_ids),
            index=copy(self.index),
            num_features=gather(self.num_features, target_ids),
            cat_features=gather(self.cat_features, target_ids),
            cat_features_names=copy(self.cat_features_names),
            num_features_names=copy(self.num_features_names),
            cat_mask=gather(self.cat_mask, target_ids),
            num_mask=gather(self.num_mask, target_ids),
        )


@dataclass(kw_only=True)
class Seq:
    tokens: torch.Tensor  # of shape (len, batch, features)
    lengths: torch.Tensor  # of shape (batch,)
    time: torch.Tensor  # of shape (len, batch)
    masks: torch.Tensor | None = None  # of shape (len, batch, features, [2])

    def to(self, device):
        self.tokens = self.tokens.to(device)
        self.lengths = self.lengths.to(device)
        self.time = self.time.to(device)
        self.masks = self.masks.to(device) if self.masks else None
        return self

    def __len__(self):
        return len(self.lengths)

    def tail_clamp(self, len_max: int):
        assert self.time is not None, "Why time feature is None?"
        seq_len = self.time.shape[0]
        if seq_len <= len_max:
            return self
        new_lengths = self.lengths.clamp_max(len_max)
        start_index = (self.lengths - new_lengths).clip(0)  # [1, B]
        target_ids = (
            torch.arange(len_max, device=start_index.device)[:, None] + start_index
        )  # [target_len, B]

        return Seq(
            lengths=new_lengths,
            tokens=gather(self.tokens, target_ids),
            time=gather(self.time, target_ids),
            masks=gather(self.masks, target_ids),
        )


@dataclass(kw_only=True)
class NHSeq(Seq):
    clustering_loss: torch.Tensor  # of shape (len, batch)
    clus_labels: torch.Tensor  # of shape (len, batch)


@dataclass
class PrimeNetReturn:
    loss: torch.Tensor
    cl_loss: torch.Tensor
    mse_loss: torch.Tensor
    correct_num: int
    total_num: int

    def to(self, device: torch.device):
        self.loss = self.loss.to(device)
        self.cl_loss = self.cl_loss.to(device)
        self.mse_loss = self.mse_loss.to(device)
        return self

    def __getitem__(self, name):
        return self.__dict__[name]


@dataclass
class NHReturn:
    pre_event_intensities_of_gt: torch.Tensor  # (len, batch)
    non_event_intensity: torch.Tensor  # (batch,)
    clustering_loss: torch.Tensor  # of shape (len, batch)
    lengths: torch.Tensor  # (batch,)
    clus_labels: torch.Tensor  # of shape (len, batch)
    pred_labels: torch.Tensor  # of shape (len, batch)

    def to(self, device: torch.device):
        self.pre_event_intensities_of_gt = self.pre_event_intensities_of_gt.to(device)
        self.non_event_intensity = self.non_event_intensity.to(device)
        self.clustering_loss = self.clustering_loss.to(device)
        self.lengths = self.lengths.to(device)
        self.clus_labels = self.clus_labels.to(device)
        self.pred_labels = self.pred_labels.to(device)
        return self
