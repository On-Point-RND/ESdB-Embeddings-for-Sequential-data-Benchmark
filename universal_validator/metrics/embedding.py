from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from universal_validator.pipeline.utils import EmbeddingMetricsConfig

logger = logging.getLogger(__name__)


def _sample_frame(df: pd.DataFrame, sample_size: int | None, seed: int) -> pd.DataFrame:
    if sample_size is None or len(df) <= sample_size:
        return df
    return df.sample(n=sample_size, random_state=seed)


def _stack_embeddings(values: pd.Series) -> np.ndarray:
    return np.stack(values.values).astype(np.float32)


def _load_embedding_sample(
    path: str,
    source: str,
    sample_size: int | None,
    seed: int,
    global_train_value: int | None = None,
) -> np.ndarray | None:
    columns = pq.ParquetDataset(path).schema.names
    if source not in columns:
        raise KeyError(source)

    read_columns = [source]
    if global_train_value is not None:
        read_columns.append("global_train")
    df = pd.read_parquet(path, columns=read_columns)
    total_rows = len(df)
    if global_train_value is not None:
        df = df[df["global_train"] == global_train_value]

    if source == "shift_emb":
        df = df.explode(source).dropna(subset=[source])
        total_vectors = len(df)
        df = _sample_frame(df, sample_size, seed)
        logger.info(
            "Embedding metrics source=%s path=%s rows=%s vectors=%s sample=%s",
            source,
            path,
            total_rows,
            total_vectors,
            len(df),
        )
    else:
        df = df.dropna(subset=[source])
        total_vectors = len(df)
        df = _sample_frame(df, sample_size, seed)
        logger.info(
            "Embedding metrics source=%s path=%s vectors=%s sample=%s",
            source,
            path,
            total_vectors,
            len(df),
        )

    if df.empty:
        logger.info("Skip embedding metrics: %s source=%s is empty", path, source)
        return None

    return _stack_embeddings(df[source])


def _parquet_columns(path: str) -> set[str]:
    if not path or not Path(path).exists():
        return set()
    return set(pq.ParquetDataset(path).schema.names)


def _effective_rank(x: np.ndarray) -> float:
    """Effective rank of embeddings from the normalized covariance spectrum."""
    centered = x - x.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0, None)
    total = eigvals.sum()
    if total <= 0:
        return 0.0

    probs = eigvals[eigvals > 0] / total
    return float(np.exp(-(probs * np.log(probs)).sum()))


def _rankme(x: np.ndarray) -> float:
    singular_values = np.linalg.svd(x, compute_uv=False)
    total = singular_values.sum()
    if total <= 0:
        return 0.0

    probs = singular_values[singular_values > 0] / total
    return float(np.exp(-(probs * np.log(probs)).sum()))


def _stable_rank(x: np.ndarray) -> float:
    centered = x - x.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    if len(singular_values) == 0 or singular_values[0] == 0:
        return 0.0
    return float((singular_values**2).sum() / singular_values[0] ** 2)


def _anisotropy(x: np.ndarray) -> float:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    normalized = x / np.clip(norms, 1e-12, None)
    summed = normalized.sum(axis=0)
    n = normalized.shape[0]
    if n < 2:
        return 0.0

    pairwise_sum = float(summed @ summed - n)
    return pairwise_sum / (n * (n - 1))


def _compute_metrics(x: np.ndarray, metric_names: list[str]) -> dict[str, float]:
    metrics = {
        "n_samples": int(x.shape[0]),
        "dim": int(x.shape[1]),
    }
    if "effective_rank" in metric_names:
        metrics["effective_rank"] = _effective_rank(x)
    if "rankme" in metric_names:
        metrics["rankme"] = _rankme(x)
    if "stable_rank" in metric_names:
        metrics["stable_rank"] = _stable_rank(x)
    if "anisotropy" in metric_names:
        metrics["anisotropy"] = _anisotropy(x)
    return metrics


def compute_embedding_metrics(
    train_path: str,
    test_path: str,
    config: EmbeddingMetricsConfig,
) -> dict[str, float]:
    train_columns = _parquet_columns(train_path)
    test_columns = _parquet_columns(test_path)
    sources = []
    if "global_emb" in train_columns:
        if "global_train" in train_columns:
            sources.extend(
                [
                    ("train", train_path, "global_emb", 1),
                    ("test", train_path, "global_emb", 0),
                ]
            )
        else:
            sources.append(("train", train_path, "global_emb", None))
            if "global_emb" in test_columns:
                sources.append(("test", test_path, "global_emb", None))
    if "shift_emb" in train_columns:
        sources.append(("train", train_path, "shift_emb", None))
    if "shift_emb" in test_columns:
        sources.append(("test", test_path, "shift_emb", None))

    metrics = {}
    for split, path, source, global_train_value in sources:
        x = _load_embedding_sample(
            path=path,
            source=source,
            sample_size=config.sample_size,
            seed=config.sample_seed,
            global_train_value=global_train_value,
        )
        if x is None:
            continue

        source_name = source.removesuffix("_emb")
        for metric_name, value in _compute_metrics(x, config.metrics).items():
            metrics[f"embedding__{split}__{source_name}__{metric_name}"] = value

    return metrics
