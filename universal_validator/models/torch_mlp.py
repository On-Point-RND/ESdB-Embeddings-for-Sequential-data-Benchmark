from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.ops import MLP as TorchvisionMLP


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

    def get_param(self, key: str, default=None):
        return self.params.get(key, default)

    def set_random_seed(self):
        if self.random_state is None:
            return
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def resolve_device(self) -> torch.device:
        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested, but CUDA is not available")
        return torch.device(device)

    def build_hidden_channels(self, output_dim: int) -> list[int]:
        hidden = self.get_param("hidden_layer_sizes", [100])
        if isinstance(hidden, int):
            hidden = [hidden]
        return [int(v) for v in hidden] + [output_dim]

    def build_model(self, input_dim: int, output_dim: int):
        self.model_ = TorchvisionMLP(
            in_channels=input_dim,
            hidden_channels=self.build_hidden_channels(output_dim),
            activation_layer=self.activation_layer,
            dropout=self.dropout,
        ).to(self.device)

    def make_loader(self, X: torch.Tensor, y: torch.Tensor):
        dataset = TensorDataset(X, y)

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

    def split_train_val(self, X: np.ndarray, y: np.ndarray, stratify=None):
        if not bool(self.get_param("early_stopping", False)):
            return X, None, y, None

        return train_test_split(
            X,
            y,
            test_size=self.validation_fraction,
            random_state=self.random_state,
            stratify=stratify,
        )

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        y_dtype: torch.dtype,
        loss_fn: nn.Module,
        X_val: np.ndarray | None,
        y_val: np.ndarray | None,
    ):
        n_train = max(len(X_train), 1)
        if "weight_decay" in self.params:
            weight_decay = float(self.params["weight_decay"])
        elif "weight_decay_init" in self.params:
            weight_decay = float(self.params["weight_decay_init"])
        else:
            # Backward compatibility with older configs.
            weight_decay = float(self.get_param("alpha", 0.0)) / n_train
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=float(self.get_param("learning_rate_init", 1e-3)),
            weight_decay=weight_decay,
        )

        X_train_tensor = torch.as_tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.as_tensor(y_train, dtype=y_dtype)
        loader = self.make_loader(X_train_tensor, y_train_tensor)
        max_epoch = int(self.get_param("max_epoch", 200))
        patience = int(self.get_param("n_iter_no_change", self.get_param("patience", 20)))
        tol = float(self.get_param("tol", 1e-4))
        verbose = bool(self.get_param("verbose", False))
        use_early_stopping = X_val is not None and y_val is not None
        early_stopping_scorer = self.get_param("early_stopping_scorer", None)
        early_stopping_scorer_name = self.get_param("early_stopping_scorer_name", "val_score")
        use_score_monitor = use_early_stopping and callable(early_stopping_scorer)

        best_state = None
        best_val_loss = float("inf")
        best_val_score = float("-inf")
        stale_epochs = 0
        val_input = None
        val_target = None
        if use_early_stopping and not use_score_monitor:
            val_input = torch.as_tensor(X_val, dtype=torch.float32).to(
                self.device,
            )
            val_target = torch.as_tensor(y_val, dtype=y_dtype).to(
                self.device,
            )
        for epoch in range(max_epoch):
            self.model_.train()
            total_loss = 0.0
            total_items = 0

            for X_batch, y_batch in loader:
                X_batch = X_batch.to(
                    self.device,
                )
                y_batch = y_batch.to(
                    self.device,
                    dtype=y_dtype,
                )
                optimizer.zero_grad(set_to_none=True)
                pred = self.model_(X_batch)
                loss = loss_fn(pred, y_batch)
                loss.backward()
                optimizer.step()
                batch_items = X_batch.shape[0]
                total_loss += loss.item() * batch_items
                total_items += batch_items

            train_loss = total_loss / max(total_items, 1)

            if not use_early_stopping:
                if verbose:
                    print(f"Epoch {epoch + 1}/{max_epoch}: train_loss={train_loss:.6f}")
                continue

            self.model_.eval()
            if use_score_monitor:
                assert early_stopping_scorer is not None
                assert X_val is not None and y_val is not None
                val_score = float(early_stopping_scorer(self, X_val, y_val))
                improved = val_score > best_val_score + tol
            else:
                with torch.inference_mode():
                    assert val_input is not None and val_target is not None
                    val_pred = self.model_(val_input)
                    val_loss = loss_fn(val_pred, val_target).item()
                improved = val_loss + tol < best_val_loss

            if verbose:
                if use_score_monitor:
                    print(
                        f"Epoch {epoch + 1}/{max_epoch}: "
                        f"train_loss={train_loss:.6f}, "
                        f"{early_stopping_scorer_name}={val_score:.6f}"
                    )
                else:
                    print(
                        f"Epoch {epoch + 1}/{max_epoch}: "
                        f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}"
                    )

            if improved:
                if use_score_monitor:
                    best_val_score = val_score
                else:
                    best_val_loss = val_loss
                best_state = deepcopy(self.model_.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()


class TorchMLPClassifier(BaseMLP, ClassifierMixin):
    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("Classifier requires at least two classes")

        self.set_random_seed()
        self.device = self.resolve_device()
        self.n_features_in_ = X.shape[1]

        self.build_model(input_dim=self.n_features_in_, output_dim=len(self.classes_))
        X_train, X_val, y_train, y_val = self.split_train_val(
            X,
            y_encoded.astype(np.int64),
            stratify=y_encoded,
        )
        self.train_model(
            X_train=X_train,
            y_train=y_train,
            y_dtype=torch.long,
            loss_fn=nn.CrossEntropyLoss(),
            X_val=X_val,
            y_val=y_val,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise ValueError("Model is not fitted yet")
        X = np.asarray(X, dtype=np.float32)
        with torch.inference_mode():
            X_tensor = torch.as_tensor(X, dtype=torch.float32).to(self.device)
            logits = self.model_(X_tensor)
            return torch.softmax(logits, dim=1).cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class TorchMLPRegressor(BaseMLP, RegressorMixin):
    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.set_random_seed()
        self.device = self.resolve_device()
        self.n_features_in_ = X.shape[1]

        self.build_model(input_dim=self.n_features_in_, output_dim=y.shape[1])
        X_train, X_val, y_train, y_val = self.split_train_val(X, y)
        self.train_model(
            X_train=X_train,
            y_train=y_train,
            y_dtype=torch.float32,
            loss_fn=nn.MSELoss(),
            X_val=X_val,
            y_val=y_val,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        with torch.inference_mode():
            X_tensor = torch.as_tensor(X, dtype=torch.float32).to(self.device)
            pred = self.model_(X_tensor).cpu().numpy()
        if pred.shape[1] == 1:
            return pred[:, 0]
        return pred
