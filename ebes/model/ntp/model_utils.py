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

        valid_mask = (
            torch.arange(batch.lengths.max(), device=batch.lengths.device)[:, None]
            < batch.lengths
        )
        current_mask = valid_mask[1:].float()
        total_valid = current_mask.sum().clamp(min=1)

        if self.cat_cardinalities:
            accuracies = []
            perplexities = []
            for name in self.cat_cardinalities:
                pred_logits = predictions[name]
                pred_classes = pred_logits.argmax(dim=-1)
                true_classes = batch[name][1:].long()

                correct = (pred_classes == true_classes) & current_mask.bool()
                acc = correct.sum().float() / total_valid
                accuracies.append(acc)

                ce_loss = F.cross_entropy(
                    pred_logits.permute(1, 2, 0),
                    true_classes.permute(1, 0),
                    reduction="none",
                    ignore_index=-100,
                )
                ce_loss_masked = ce_loss * current_mask.permute(1, 0).float()
                avg_ce = ce_loss_masked.sum() / total_valid
                perplexities.append(torch.exp(avg_ce))

            metrics_dict["accuracy"] = torch.stack(accuracies).mean()
            metrics_dict["perplexity"] = torch.stack(perplexities).mean()

        if self.num_features:
            r2_scores = []
            for name in self.num_features:
                pred = predictions[name].squeeze(-1)
                target = batch[name][1:]

                pred_valid = pred * current_mask
                target_valid = target * current_mask

                ss_res = ((pred_valid - target_valid) ** 2).sum()
                mean_target = target_valid.sum() / total_valid
                ss_tot = ((target_valid - mean_target) ** 2).sum()

                r2 = 1 - ss_res / ss_tot.clamp(min=1e-8)
                r2_scores.append(r2)

            metrics_dict["r2"] = torch.stack(r2_scores).mean()

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
