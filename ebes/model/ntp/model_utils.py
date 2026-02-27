import torch.nn as nn
import torch
import torch.nn.functional as F
from ...types import Batch


class ReconPredictor(nn.Module):
    def __init__(
        self,
        dec_hidden_size,
        cat_cardinalities,
        num_features,
    ):
        super().__init__()
        self._ignore_index = -100
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

    def metrics(self, predictions: dict[str, torch.Tensor], batch: Batch):
        metrics_dict = {}
        rng = torch.arange(batch.lengths.max(), device=batch.lengths.device)
        mask = (rng[:, None] < batch.lengths)[1:].float()
        total = mask.sum().clamp(min=1)

        if self.cat_cardinalities:
            accs = []
            perps = []
            for name in self.cat_cardinalities:
                logits, target = predictions[name], batch[name][1:].long()
                pred = logits.argmax(dim=-1)

                accs.append(((pred == target) & mask.bool()).sum().float() / total)
                loss = F.cross_entropy(
                    logits.flatten(0, 1), target.flatten(), reduction="none"
                )
                perps.append(torch.exp((loss * mask.reshape(-1)).sum() / total))

            metrics_dict["accuracy"] = torch.stack(accs).mean()
            metrics_dict["perplexity"] = torch.stack(perps).mean()

        if self.num_features:
            r2s = []
            for name in self.num_features:
                pred, target = predictions[name].squeeze(-1), batch[name][1:]
                pv, tv = pred[mask.bool()], target[mask.bool()]
                if pv.numel() == 0:
                    r2s.append(torch.tensor(0.0, device=batch.lengths.device))
                    continue
                ss_res = ((pv - tv) ** 2).sum()
                ss_tot = ((tv - tv.mean()) ** 2).sum()
                r2s.append(1 - ss_res / ss_tot.clamp(min=1e-8))
            metrics_dict["r2"] = torch.stack(r2s).mean()
        return metrics_dict

    def loss(self, predictions: dict[str, torch.Tensor], y_true: Batch):
        ce_loss = {}
        mse_loss = {}

        valid_mask = (
            torch.arange(y_true.lengths.max(), device=y_true.lengths.device)[:, None]
            < y_true.lengths
        )
        current_mask = valid_mask[1:]

        for key in self.cat_cardinalities:
            distribution = predictions[key]
            pred_cat = distribution.permute(1, 2, 0)

            true_cat = y_true[key][1:].clone()
            true_cat[~current_mask] = self._ignore_index

            ce_loss[key] = F.cross_entropy(
                pred_cat,
                true_cat.permute(1, 0),
                ignore_index=self._ignore_index,
            ).mean()

        for key in self.num_features:
            pred = predictions[key].squeeze(-1)
            target = y_true[key][1:]
            loss = self.mse_fn(pred, target) * current_mask.float()
            mse_loss[key] = (loss.sum(0) / current_mask.sum(0).clamp(min=1)).mean()

        return ce_loss, mse_loss
