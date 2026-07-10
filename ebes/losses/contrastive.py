from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def outer_pairwise_distance(a, b=None):
    """
    Compute pairwise_distance of Tensors
        A (size(A) = n x d, where n - rows count, d - vector size) and
        B (size(A) = m x d, where m - rows count, d - vector size)
    return matrix C (size n x m), such as
        C_ij = distance(i-th row matrix A, j-th row matrix B)

    if only one Tensor was given, computer pairwise distance to itself (B = A)
    """

    if b is None:
        b = a

    max_size = 2**26
    n = a.size(0)
    m = b.size(0)
    d = a.size(1)

    if n * m * d <= max_size or m == 1:
        return torch.pairwise_distance(
            a[:, None].expand(n, m, d).reshape((-1, d)),
            b.expand(n, m, d).reshape((-1, d)),
        ).reshape((n, m))

    else:
        batch_size = max(1, max_size // (n * d))
        batch_results = []
        for i in range((m - 1) // batch_size + 1):
            id_left = i * batch_size
            id_rigth = min((i + 1) * batch_size, m)
            batch_results.append(outer_pairwise_distance(a, b[id_left:id_rigth]))

        return torch.cat(batch_results, dim=1)


##### PAIR SELECTORS #####


class PairSelector(ABC):
    """Strategy to sample positive and negative embedding pairs."""

    @abstractmethod
    def get_pairs(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class HardNegativePairSelector(PairSelector):
    """
    Generates all possible possitive pairs given labels and
         neg_count hardest negative example for each example
    """

    def __init__(self, neg_count=1):
        super().__init__()
        self.neg_count = neg_count

    def get_pairs(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # construct matrix x, such as x_ij == 0 <==> labels[i] == labels[j]
        n = len(labels)
        x = labels - labels[:, None]  # The same but works for numpy. I've tested!!!

        # positive pairs
        positive_pairs = torch.triu((x == 0).int(), diagonal=1).nonzero(as_tuple=False)

        # hard negative minning
        mat_distances = outer_pairwise_distance(
            embeddings.detach()
        )  # pairwise_distance

        upper_bound = int((2 * n) ** 0.5) + 1
        mat_distances = (upper_bound - mat_distances) * (x != 0).type(
            mat_distances.dtype
        )  # filter: get only negative pairs

        _, indices = mat_distances.topk(k=self.neg_count, dim=0, largest=True)
        negative_pairs = torch.stack(
            [
                torch.arange(0, n, dtype=indices.dtype, device=indices.device).repeat(
                    self.neg_count
                ),
                torch.cat(indices.unbind(dim=0)),
            ]
        ).t()

        return positive_pairs, negative_pairs


class TemporalHardNegativePairSelector(PairSelector):
    """
    Temporal variant of :class:`HardNegativePairSelector`.

    Expects ``labels`` of shape ``(N, 3)`` = ``[client_label, start_pos, rank]``
    produced by ``RandomSlices(temporal=True)``.

    - Positive pairs are **ordered** ``(past, future)``: both slices belong to the
      same client and ``rank[past] > rank[future]`` (larger rank == more past).
      Column 0 is the past slice, column 1 is the future slice. The loss uses this
      order to make the pull asymmetric (future is a detached anchor).
    - Negative pairs use only ``client_label`` and are mined exactly like in
      :class:`HardNegativePairSelector`.
    """

    def __init__(self, neg_count=1):
        super().__init__()
        self.neg_count = neg_count

    def get_pairs(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(labels, torch.Tensor):
            labels = torch.as_tensor(labels)
        labels = labels.to(embeddings.device)
        cls = labels[:, 0]
        rank = labels[:, 2]
        n = len(labels)

        # ordered positive pairs: same client and past (larger rank) -> future
        same = cls[:, None] == cls[None, :]
        pos_mask = same & (rank[:, None] > rank[None, :])
        positive_pairs = pos_mask.nonzero(as_tuple=False)  # (K, 2): [past, future]

        # hard negative mining (by client label)
        different = (cls[:, None] != cls[None, :]).type(embeddings.dtype)
        mat_distances = outer_pairwise_distance(embeddings.detach())
        upper_bound = int((2 * n) ** 0.5) + 1
        mat_distances = (upper_bound - mat_distances) * different

        _, indices = mat_distances.topk(k=self.neg_count, dim=0, largest=True)
        negative_pairs = torch.stack(
            [
                torch.arange(0, n, dtype=indices.dtype, device=indices.device).repeat(
                    self.neg_count
                ),
                torch.cat(indices.unbind(dim=0)),
            ]
        ).t()

        return positive_pairs, negative_pairs


class GraphChainPairSelector(PairSelector):
    """
    Treat each client's slices as a path graph ordered in time.

    Expects ``labels`` of shape ``(N, 3)`` = ``[client_label, start_pos, rank]``
    produced by ``RandomSlices(temporal=True)``.

    Nodes are slices; an **edge** connects two slices of the same client whose
    ranks differ by at most ``neighbor_radius`` (1 == immediate temporal
    neighbours). Positive pairs are the edges, taken once each (upper triangle).

    Negatives are mined exactly as in :class:`HardNegativePairSelector`: the
    ``neg_count`` hardest (closest) counterparts for every anchor. The only
    difference is the set they are drawn from -- here it is the **non-edges**
    (other clients, plus same-client slices further than ``neighbor_radius``
    apart in rank), whereas the original draws from other classes only.
    """

    def __init__(self, neg_count=1, neighbor_radius: int = 1):
        super().__init__()
        self.neg_count = neg_count
        self.neighbor_radius = neighbor_radius

    def get_pairs(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(labels, torch.Tensor):
            labels = torch.as_tensor(labels)
        labels = labels.to(embeddings.device)
        cls = labels[:, 0]
        rank = labels[:, 2]
        n = len(labels)

        same = cls[:, None] == cls[None, :]
        rank_gap = (rank[:, None] - rank[None, :]).abs()
        edge = same & (rank_gap >= 1) & (rank_gap <= self.neighbor_radius)

        # positive pairs: graph edges, each counted once
        positive_pairs = torch.triu(edge.int(), diagonal=1).nonzero(as_tuple=False)

        # hard negative mining over non-edges (self-pairs excluded explicitly,
        # since the diagonal is not an edge either)
        eye = torch.eye(n, dtype=torch.bool, device=embeddings.device)
        non_edge = (~edge) & (~eye)

        mat_distances = outer_pairwise_distance(
            embeddings.detach()
        )  # pairwise_distance

        upper_bound = int((2 * n) ** 0.5) + 1
        mat_distances = (upper_bound - mat_distances) * non_edge.type(
            mat_distances.dtype
        )  # filter: get only non-edge pairs

        _, indices = mat_distances.topk(k=self.neg_count, dim=0, largest=True)
        negative_pairs = torch.stack(
            [
                torch.arange(0, n, dtype=indices.dtype, device=indices.device).repeat(
                    self.neg_count
                ),
                torch.cat(indices.unbind(dim=0)),
            ]
        ).t()

        return positive_pairs, negative_pairs


##### LOSSES #####


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss

    "Signature verification using a siamese time delay neural network", NIPS 1993
    https://papers.nips.cc/paper/769-signature-verification-using-a-siamese-time-delay-neural-network.pdf
    """

    def __init__(
        self,
        margin: float,
        pair_selector: PairSelector,
    ):
        super().__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(
            embeddings, target
        )
        positive_loss = F.pairwise_distance(
            embeddings[positive_pairs[:, 0]], embeddings[positive_pairs[:, 1]]
        ).pow(2)

        negative_loss = F.relu(
            self.margin
            - F.pairwise_distance(
                embeddings[negative_pairs[:, 0]],
                embeddings[negative_pairs[:, 1]],
            )
        ).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)

        return loss.mean()


class TemporalContrastiveLoss(nn.Module):
    """
    Asymmetric temporal contrastive loss.

    Uses ordered positive pairs ``(past, future)`` from
    :class:`TemporalHardNegativePairSelector`. The positive term pulls the **past**
    slice towards the **future** slice, but the future slice is a *detached* anchor:
    it receives no extra gradient from this pair. This breaks the symmetry that would
    otherwise cancel out a directional signal (past->future and future->past pulls
    would sum to zero on the same pair). Negatives are pushed apart symmetrically,
    exactly as in :class:`ContrastiveLoss`.
    """

    def __init__(
        self,
        margin: float,
        pair_selector: PairSelector,
    ):
        super().__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(
            embeddings, target
        )
        # positive_pairs[:, 0] == past (gets gradient), [:, 1] == future (anchor)
        past = embeddings[positive_pairs[:, 0]]
        future = embeddings[positive_pairs[:, 1]].detach()
        positive_loss = F.pairwise_distance(past, future).pow(2)

        negative_loss = F.relu(
            self.margin
            - F.pairwise_distance(
                embeddings[negative_pairs[:, 0]],
                embeddings[negative_pairs[:, 1]],
            )
        ).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)

        return loss.mean()


class GraphContrastiveLoss(ContrastiveLoss):
    """
    Contrastive loss over a per-client temporal path graph.

    Mechanically identical to :class:`ContrastiveLoss` -- symmetric pull on
    positives, symmetric hinge push on mined negatives, mean over all terms. The
    graph structure lives entirely in :class:`GraphChainPairSelector`: positives
    are edges (temporally adjacent slices of one client) instead of "any two
    slices of one client", and negatives are mined from non-edges instead of
    other clients only.

    Note the built-in tension: the edge set is a chain, so pulling ``d(r2, r1)``
    and ``d(r1, r0)`` towards zero also drags ``d(r2, r0)`` down, which the
    margin resists. Equilibrium spreads a client's slices along a curve with
    adjacent spacing ~``margin/2`` -- an ordered temporal trajectory -- so the
    positive term has a non-zero floor by design.
    """


class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss https://arxiv.org/abs/1807.03748
    """

    def __init__(
        self,
        temperature: float,
        pair_selector: PairSelector,
        angular_margin: float = 0.0,  # = 0.5 ArcFace default
    ):
        super().__init__()

        self.temperature = temperature
        self.pair_selector = pair_selector
        self.angular_margin = angular_margin

    def forward(self, embeddings, target):
        #embeddings = self.project(embeddings)

        positive_pairs, _ = self.pair_selector.get_pairs(embeddings, target)
        dev = positive_pairs.device
        all_idx = torch.arange(len(positive_pairs), dtype=torch.long, device=dev)
        mask_same = torch.zeros(len(positive_pairs), len(embeddings), device=dev)
        mask_same[all_idx, positive_pairs[:, 0]] -= torch.inf

        sim = (
            F.cosine_similarity(
                embeddings[positive_pairs[:, 0], None],
                embeddings[None],
                dim=-1,
            )
            + mask_same
        )
        if self.angular_margin > 0.0:
            with torch.no_grad():
                target_sim = sim[all_idx, positive_pairs[:, 1]].clamp(0, 1)
                target_sim.arccos_()
                target_sim += self.angular_margin
                target_sim.cos_()
                sim[all_idx, positive_pairs[:, 1]] = target_sim

        sim /= self.temperature

        n_negatives = embeddings.shape[0] - 2 # here should be 2 or 3, does not really matter
        #print(embeddings.shape)
        lsm = -F.log_softmax(sim, dim=-1)
        lsm = lsm - torch.tensor(n_negatives, dtype=torch.float, device=dev).log()
        loss = torch.take_along_dim(
            lsm,
            positive_pairs[:, [1]],
            dim=1,
        ).mean()

        return loss
