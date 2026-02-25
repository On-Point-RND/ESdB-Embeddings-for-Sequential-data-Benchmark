import torch.nn as nn
import torch
from ...types import Batch


class ReconPredictor(nn.Module):
    def __init__(
        self,
        dec_hidden_size,
        cat_cardinalities,
        num_features,
    ):
        super().__init__()

        self.cat_criterion = nn.CrossEntropyLoss(reduction="none")
        self.cat_cardinalities = cat_cardinalities or dict()
        self.cat_predictors = nn.ModuleDict()
        for name, vocab_size in self.cat_cardinalities.items():
            self.cat_predictors[name] = nn.Linear(dec_hidden_size, vocab_size)

        self.mse_fn = torch.nn.MSELoss(reduction="none")
        self.num_features = num_features or []
        self.num_predictors = nn.ModuleDict()
        for name in self.num_features:
            self.num_predictors[name] = nn.Linear(dec_hidden_size, 1)

    def forward(self, x_recon):
        predictions = {}
        for name in self.cat_cardinalities:
            predictions[name] = self.cat_predictors[name](x_recon)

        for i, name in enumerate(self.num_features):
            predictions[name] = self.num_predictors[name](x_recon)

        return predictions

    def loss(self, predictions: dict[str, torch.Tensor], batch: Batch):
        ce_loss = {}
        max_len = batch.lengths.max()
        valid_mask = (torch.arange(max_len, device=batch.lengths.device)[:, None] < batch.lengths)
        target_mask = valid_mask[1:]  # (L-1, Batch) - сдвиг для таргетов

        for name in self.cat_cardinalities:
            distribution = predictions[name]  # (L-1, Batch, Vocab)
            labels = batch[name][1:].long()  # (L-1, Batch) - СДВИГ!
            labels = labels.masked_fill(~target_mask, -100)

            loss = self.cat_criterion(distribution.permute(1, 2, 0), labels.permute(1, 0))
            ce_loss[name] = loss.mean()

        mse_loss = {}
        for name in self.num_features:
            pred = predictions[name].squeeze(-1)  # (L-1, Batch)
            target = batch[name][1:]  # (L-1, Batch) - СДВИГ!

            loss = self.mse_fn(pred, target) * target_mask.float()
            mse_loss[name] = (loss.sum(0) / target_mask.sum(0).clamp(min=1)).mean()

        return ce_loss, mse_loss
