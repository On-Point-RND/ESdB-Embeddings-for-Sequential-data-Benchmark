from __future__ import annotations

import logging
from copy import deepcopy

import numpy as np
import torch
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.ops import MLP

logger = logging.getLogger(__name__)


class BaseMLP:
    def __init__(self, **params):
        self.params = dict(params)
        self.sync_params()

    def sync_params(self):
        self.device = self.params.get("device", "auto")
        self.random_state = self.params.get("random_state", None)
        self.validation_fraction = float(self.params.get("validation_fraction", 0.1))
        self.batch_size = int(self.params.get("batch_size", 64))
        self.dropout = float(self.params.get("dropout", 0.0))
        self.num_workers = int(self.params.get("num_workers", 4))
        activation_name = self.params.get("activation", "relu")
        activation_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
        }
        if activation_name not in activation_map:
            raise ValueError(f"Unsupported activation: {activation_name}")
        self.activation_layer = activation_map[activation_name]

    def get_params(self, deep: bool = True):
        return dict(self.params)

    def set_params(self, **params):
        self.params.update(params)
        self.sync_params()
        return self

    def resolve_early_stopping(self):
        early_stopping_scorer = self.params.get("early_stopping_scorer")
        if early_stopping_scorer is not None and not callable(early_stopping_scorer):
            raise TypeError("early_stopping_scorer must be callable.")

        early_stopping_enabled = bool(
            self.params.get("early_stopping", False) or early_stopping_scorer is not None
        )
        score_monitor_name = (
            getattr(early_stopping_scorer, "name", "val_score")
            if early_stopping_scorer is not None
            else "val_loss"
        )
        return early_stopping_enabled, early_stopping_scorer, score_monitor_name

    def set_random_seed(self):
        if self.random_state is None:
            return
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def resolve_device(self) -> torch.device:
        device = self.device
        if device.startswith("cuda:") or device == "cuda":
            return torch.device(device)
        else:
            raise RuntimeError("CUDA device requested, but CUDA is not available")

    def build_hidden_channels(self, output_dim: int) -> list[int]:
        hidden = self.params.get("hidden_layer_sizes", [100])
        if isinstance(hidden, int):
            hidden = [hidden]
        return [int(v) for v in hidden] + [output_dim]

    def build_model(self, input_dim: int, output_dim: int):
        self.model_ = MLP(
            in_channels=input_dim,
            hidden_channels=self.build_hidden_channels(output_dim),
            activation_layer=self.activation_layer,
            dropout=self.dropout,
        ).to(self.device)

    def make_loader(self, x_data: torch.Tensor, y_data: torch.Tensor):
        dataset = TensorDataset(x_data, y_data)

        generator = None
        if self.random_state is not None:
            generator = torch.Generator()
            generator.manual_seed(self.random_state)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=self.num_workers,
        )

    def split_train_val(self, x_data: np.ndarray, y_data: np.ndarray, stratify=None):
        early_stopping_enabled, _, _ = self.resolve_early_stopping()
        if not early_stopping_enabled:
            return x_data, None, y_data, None
        else:
            assert (
                self.validation_fraction > 0
            ), "Provide validation set for early stopping."
            if stratify is not None:
                _, class_counts = np.unique(stratify, return_counts=True)
                if class_counts.min() < 2:
                    stratify = None
            return train_test_split(
                x_data,
                y_data,
                test_size=self.validation_fraction,
                random_state=self.random_state,
                stratify=stratify,
            )

    def train_model(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        y_dtype: torch.dtype,
        loss_fn: nn.Module,
        x_val: np.ndarray | None,
        y_val: np.ndarray | None,
    ):
        weight_decay = float(self.params["weight_decay"])
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=float(self.params.get("learning_rate_init", 1e-3)),
            weight_decay=weight_decay,
        )

        loader = self.make_loader(
            torch.as_tensor(x_train, dtype=torch.float32),
            torch.as_tensor(y_train, dtype=y_dtype),
        )

        max_epoch = int(self.params.get("max_epoch", 200))
        patience = int(
            self.params.get("n_iter_no_change", self.params.get("patience", 20))
        )
        tol = float(self.params.get("tol", 1e-4))
        verbose = bool(self.params.get("verbose", False))

        early_stopping, early_stopper, early_stopper_name = self.resolve_early_stopping()
        use_score_monitor = early_stopper is not None
        if early_stopping and (x_val is None or y_val is None):
            raise ValueError(
                "early_stopping=True requires validation data (x_val and y_val).",
            )

        best_state = None
        best_monitor = float("-inf") if use_score_monitor else float("inf")
        stale_epochs = 0
        val_input = None
        val_target = None
        if early_stopping and not use_score_monitor:
            assert x_val is not None and y_val is not None
            val_input = torch.as_tensor(x_val, dtype=torch.float32).to(
                self.device,
            )
            val_target = torch.as_tensor(y_val, dtype=y_dtype).to(
                self.device,
            )

        for epoch in range(max_epoch):
            self.model_.train()
            total_loss = 0.0
            total_items = 0

            for x_batch, y_batch in loader:
                x_batch = x_batch.to(
                    self.device,
                )
                y_batch = y_batch.to(
                    self.device,
                    dtype=y_dtype,
                )
                optimizer.zero_grad(set_to_none=True)
                pred = self.model_(x_batch)
                loss = loss_fn(pred, y_batch)
                loss.backward()
                optimizer.step()
                batch_items = x_batch.shape[0]
                total_loss += loss.item() * batch_items
                total_items += batch_items

            train_loss = total_loss / max(total_items, 1)

            if not early_stopping:
                if verbose:
                    logger.info(
                        "Epoch %s/%s: train_loss=%.6f",
                        epoch + 1,
                        max_epoch,
                        train_loss,
                    )
                continue

            self.model_.eval()
            if use_score_monitor:
                monitor_value = float(early_stopper(self, x_val, y_val))
                improved = monitor_value > best_monitor + tol
            else:
                with torch.inference_mode():
                    val_pred = self.model_(val_input)
                    monitor_value = loss_fn(val_pred, val_target).item()
                improved = monitor_value + tol < best_monitor

            if verbose:
                logger.info(
                    "Epoch %s/%s: train_loss=%.6f, %s=%.6f",
                    epoch + 1,
                    max_epoch,
                    train_loss,
                    early_stopper_name,
                    monitor_value,
                )

            if improved:
                best_monitor = monitor_value
                best_state = deepcopy(self.model_.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    if verbose:
                        logger.info("Early stopping at epoch %s", epoch + 1)
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()


class TorchMLPClassifier(BaseMLP, ClassifierMixin):
    def fit(self, x_data: np.ndarray, y_data: np.ndarray):
        x_data = np.asarray(x_data, dtype=np.float32)
        y_data = np.asarray(y_data)

        self.classes_, y_encoded = np.unique(y_data, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("Classifier requires at least two classes")

        self.set_random_seed()
        self.device = self.resolve_device()
        self.n_features_in_ = x_data.shape[1]

        self.build_model(input_dim=self.n_features_in_, output_dim=len(self.classes_))
        x_train, x_val, y_train, y_val = self.split_train_val(
            x_data,
            y_encoded.astype(np.int64),
            stratify=y_encoded,
        )
        self.train_model(
            x_train=x_train,
            y_train=y_train,
            y_dtype=torch.long,
            loss_fn=nn.CrossEntropyLoss(),
            x_val=x_val,
            y_val=y_val,
        )
        return self

    def predict_proba(self, x_data: np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise ValueError("Model is not fitted yet")
        x_data = np.asarray(x_data, dtype=np.float32)
        with torch.inference_mode():
            x_tensor = torch.as_tensor(x_data, dtype=torch.float32).to(self.device)
            logits = self.model_(x_tensor)
            return torch.softmax(logits, dim=1).cpu().numpy()

    def predict(self, x_data: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(x_data)
        return self.classes_[np.argmax(proba, axis=1)]


class TorchMLPRegressor(BaseMLP, RegressorMixin):
    def fit(self, x_data: np.ndarray, y_data: np.ndarray):
        x_data = np.asarray(x_data, dtype=np.float32)
        y_data = np.asarray(y_data, dtype=np.float32)
        if y_data.ndim == 1:
            y_data = y_data.reshape(-1, 1)

        self.set_random_seed()
        self.device = self.resolve_device()
        self.n_features_in_ = x_data.shape[1]

        self.build_model(input_dim=self.n_features_in_, output_dim=y_data.shape[1])
        x_train, x_val, y_train, y_val = self.split_train_val(x_data, y_data)
        self.train_model(
            x_train=x_train,
            y_train=y_train,
            y_dtype=torch.float32,
            loss_fn=nn.MSELoss(),
            x_val=x_val,
            y_val=y_val,
        )
        return self

    def predict(self, x_data: np.ndarray) -> np.ndarray:
        x_data = np.asarray(x_data, dtype=np.float32)
        with torch.inference_mode():
            x_tensor = torch.as_tensor(x_data, dtype=torch.float32).to(self.device)
            pred = self.model_(x_tensor).cpu().numpy()
        if pred.shape[1] == 1:
            return pred[:, 0]
        return pred
