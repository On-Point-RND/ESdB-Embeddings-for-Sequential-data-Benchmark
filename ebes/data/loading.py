from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from ..types import Batch


@dataclass(kw_only=True, frozen=True)
class SequenceCollator:
    time_name: str
    cat_cardinalities: Mapping[str, int] | None = None
    emb_dimensionalities: Mapping[str, int] | None = None
    num_names: list[str] | None = None
    index_name: str | None = None
    target_name: str | list[str] | None = None
    max_seq_len: int = 0
    batch_transforms: list[Callable[[Batch], None]] | None = None
    method_specific_batch_transforms: list[Callable[[Batch], None]] | None = None
    padding_type: str = "zeros"

    def __call__(self, seqs: Sequence[pd.Series]) -> Batch:
        ml = min(max(s["_seq_len"] for s in seqs), self.max_seq_len)  # type: ignore
        bs = len(seqs)
        ##############################################################################
        num_features = None
        num_names = deepcopy(self.num_names)
        if num_names:
            num_features = torch.zeros(ml, bs, len(num_names), dtype=torch.float32)

        cat_features = None
        cat_cardinalities = {}
        cat_names = None
        if self.cat_cardinalities is not None:
            cat_cardinalities = self.cat_cardinalities
            cat_names = list(self.cat_cardinalities.keys())
        if self.cat_cardinalities:
            cat_features = torch.zeros(
                ml, bs, len(self.cat_cardinalities), dtype=torch.long
            )

        emb_features = None
        emb_dimensionalities = {}
        emb_names = None
        if self.emb_dimensionalities is not None:
            emb_features={}
            emb_dimensionalities = self.emb_dimensionalities
            emb_names = list(self.emb_dimensionalities.keys())
        if self.emb_dimensionalities:
            for name in emb_names:
                emb_features[name] = torch.zeros(emb_dimensionalities[name],
                    ml, bs,  dtype=torch.float32
                )
                # name: (emb_dim, time_len_max, number_of_seqs)
        ##############################################################################
        indices = None
        if self.index_name:
            indices = []

        targets = None
        if self.target_name:
            targets = []

        seq_time_dtype: np.dtype = seqs[0][self.time_name].dtype  # type: ignore
        times = np.zeros((ml, bs), dtype=seq_time_dtype)

        seq_lens = torch.empty(bs, dtype=torch.long)
        ##############################################################################
        for b, s in enumerate(seqs):
            sl = min(s["_seq_len"], ml)  # type: ignore
            seq_lens[b] = sl
            ################################################################################
            if num_names is not None:
                for i, name in enumerate(num_names):
                    assert num_features is not None
                    num_features[:sl, b, i] = torch.tensor(s[name][-sl:])
                    num_features[sl:, b, i] = (
                        torch.zeros(1)
                        if self.padding_type == "zeros"
                        else torch.tensor(s[name][-1])
                    )
            #################################################################################
            for i, (name, card) in enumerate(cat_cardinalities.items()):
                assert cat_features is not None
                cat_features[:sl, b, i] = torch.tensor(s[name][-sl:]).clamp_(
                    max=card - 1
                )
                cat_features[sl:, b, i] = (
                    torch.zeros(1)
                    if self.padding_type == "zeros"
                    else torch.tensor(s[name][-1]).clamp_(max=card - 1)
                )
            ##################################################################################
            for i, (name, dim) in enumerate(emb_dimensionalities.items()):
                assert emb_features is not None
                emb_features[name][:, :sl, b] = torch.tensor(np.stack(s[name][-sl:], axis=0), dtype=torch.float32).T  # inside .T it is (sl, emb_dim), so we .T
                emb_features[name][:, sl:, b] = (
                    torch.zeros(1)
                    if self.padding_type == "zeros"
                    else torch.tensor(s[name][-1]).T
                )
            ##################################################################################
            if indices is not None:
                indices.append(s[self.index_name])

            if targets is not None:
                target = s[self.target_name]
                if isinstance(self.target_name, list):
                    vals = target.values  # pyright: ignore
                    vals = vals.astype(type(vals[0]))
                    targets.append(vals)
                elif np.isscalar(target):
                    targets.append(target)
                else:
                    target_padded = np.zeros(ml)
                    target_padded[:sl] = target[-sl:]
                    target_padded[sl:] = (
                        torch.zeros(1) if self.padding_type == "zeros" else target[-1]
                    )
                    targets.append(target_padded)

            times[:sl, b] = s[self.time_name][-sl:]
            times[sl:, b] = (
                torch.zeros(1)
                if self.padding_type == "zeros"
                else s[self.time_name][-1]
            )
        #############################################################################
        index = np.asanyarray(indices)
        if targets is not None:
            if isinstance(self.target_name, list):
                targets = np.stack(targets)
            else:
                targets = np.stack(targets).T
            target = torch.asarray(targets)
            target = (
                target if target.dtype != torch.float64 else target.to(torch.float32)
            )
        else:
            target = None

        try:
            times = torch.asarray(times)
        except TypeError:
            # keep numpy
            pass
        ##############################################################################
        batch = Batch(
            num_features=num_features,
            cat_features=cat_features,
            emb_features=emb_features,
            index=index,
            target=target,
            time=times,
            lengths=seq_lens,
            cat_features_names=cat_names,
            num_features_names=num_names,
            emb_features_names=emb_names,
        )
        if self.batch_transforms is not None:
            for tf in self.batch_transforms:
                tf(batch)
        if self.method_specific_batch_transforms is not None:
            for tf in self.method_specific_batch_transforms:
                tf(batch)
        return batch
