from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from universal_validator.metrics.embedding_geometry import (
    anisotropy,
    asmi_ksg_pca_whiten,
    effective_rank,
    effdim,
    lidar_from_views,
    rankme,
    stable_rank,
    total_compression,
)

if TYPE_CHECKING:
    from universal_validator.pipeline.utils import EmbeddingMetricsConfig

logger = logging.getLogger(__name__)

_effective_rank = effective_rank
_rankme = rankme
_stable_rank = stable_rank
_anisotropy = anisotropy

EmbeddingColumnContext = tuple[str, str, int | None]
SplitContext = tuple[str, int | None]


def _metric_column_name(column: str) -> str:
    return column.removesuffix("_emb").removesuffix("_global")


def _as_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _sample_frame(df: pd.DataFrame, sample_size: int | None, seed: int) -> pd.DataFrame:
    if sample_size is None or len(df) <= sample_size:
        return df
    return df.sample(n=sample_size, random_state=seed)


def _stack_embeddings(values: pd.Series) -> np.ndarray:
    return np.stack(values.values).astype(np.float32)


def _timed_metric(metric_name: str, context: str, metric_fn, *args, **kwargs):
    started_at = perf_counter()
    value = metric_fn(*args, **kwargs)
    logger.info(
        "Embedding metric timing metric=%s %s seconds=%.3f",
        metric_name,
        context,
        perf_counter() - started_at,
    )
    return value


def _is_shift_source(column: str) -> bool:
    return column == "shift_emb" or column.endswith("_shift_emb")


def _parquet_columns(path: str) -> set[str]:
    if not path or not Path(path).exists():
        return set()
    return set(pq.ParquetDataset(path).schema.names)


def _metric_path(
    split: str,
    train_path: str,
    test_path: str,
    global_train_value: int | None,
) -> str:
    if global_train_value is not None or split == "train":
        return train_path
    return test_path


def _load_embedding_sample(
    path: str,
    column: str,
    sample_size: int | None,
    seed: int,
    global_train_value: int | None = None,
) -> np.ndarray | None:
    columns = pq.ParquetDataset(path).schema.names
    if column not in columns:
        raise KeyError(column)

    read_columns = [column]
    if global_train_value is not None:
        read_columns.append("global_train")
    df = pd.read_parquet(path, columns=read_columns)
    total_rows = len(df)
    if global_train_value is not None:
        df = df[df["global_train"] == global_train_value]

    if _is_shift_source(column):
        df = df.explode(column).dropna(subset=[column])
        total_vectors = len(df)
    else:
        df = df.dropna(subset=[column])
        total_vectors = len(df)

    df = _sample_frame(df, sample_size, seed)
    logger.info(
        "Embedding metrics column=%s path=%s rows=%s vectors=%s sample=%s",
        column,
        path,
        total_rows,
        total_vectors,
        len(df),
    )
    if df.empty:
        logger.info("Skip embedding metrics: %s column=%s is empty", path, column)
        return None
    return _stack_embeddings(df[column])


def _load_pair_sample(
    path: str,
    source_column: str,
    target_column: str,
    sample_size: int | None,
    seed: int,
    global_train_value: int | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    columns = pq.ParquetDataset(path).schema.names
    missing = [col for col in [source_column, target_column] if col not in columns]
    if missing:
        raise KeyError(missing)

    read_columns = [source_column, target_column]
    if global_train_value is not None:
        read_columns.append("global_train")
    df = pd.read_parquet(path, columns=read_columns)
    if global_train_value is not None:
        df = df[df["global_train"] == global_train_value]
    df = df.dropna(subset=[source_column, target_column])
    df = _sample_frame(df, sample_size, seed)
    if df.empty:
        return None
    return _stack_embeddings(df[source_column]), _stack_embeddings(df[target_column])


def _load_views_sample(
    path: str,
    views: list[str],
    sample_size: int | None,
    seed: int,
    global_train_value: int | None,
) -> np.ndarray | None:
    read_columns = list(views)
    if global_train_value is not None:
        read_columns.append("global_train")
    df = pd.read_parquet(path, columns=read_columns)
    if global_train_value is not None:
        df = df[df["global_train"] == global_train_value]
    df = df.dropna(subset=views)
    df = _sample_frame(df, sample_size, seed)
    if df.empty:
        return None
    return np.stack([_stack_embeddings(df[col]) for col in views], axis=1)


def _default_embedding_columns(
    train_columns: set[str], test_columns: set[str]
) -> list[EmbeddingColumnContext]:
    embedding_columns: list[EmbeddingColumnContext] = []
    if "global_emb" in train_columns:
        if "global_train" in train_columns:
            embedding_columns.extend(
                [("train", "global_emb", 1), ("test", "global_emb", 0)]
            )
        else:
            embedding_columns.append(("train", "global_emb", None))
            if "global_emb" in test_columns:
                embedding_columns.append(("test", "global_emb", None))
    if "shift_emb" in train_columns:
        embedding_columns.append(("train", "shift_emb", None))
    if "shift_emb" in test_columns:
        embedding_columns.append(("test", "shift_emb", None))
    return embedding_columns


def _configured_embedding_columns(
    config: EmbeddingMetricsConfig,
    train_columns: set[str],
    test_columns: set[str],
) -> list[EmbeddingColumnContext]:
    if not config.sources:
        return _default_embedding_columns(train_columns, test_columns)

    embedding_columns: list[EmbeddingColumnContext] = []
    for source in config.sources:
        if source in train_columns:
            if "global_train" in train_columns:
                embedding_columns.extend([("train", source, 1), ("test", source, 0)])
                continue
            embedding_columns.append(("train", source, None))
        if source in test_columns:
            embedding_columns.append(("test", source, None))
    if config.splits:
        embedding_columns = [
            context for context in embedding_columns if context[0] in config.splits
        ]
    return embedding_columns


def _unique_contexts(
    embedding_columns: list[EmbeddingColumnContext],
) -> list[SplitContext]:
    contexts: list[SplitContext] = []
    seen = set()
    for split, _, global_train_value in embedding_columns:
        key = (split, global_train_value)
        if key not in seen:
            contexts.append((split, global_train_value))
            seen.add(key)
    return contexts or [("train", None)]


def _view_columns_from_group(group: dict, columns: set[str]) -> list[str]:
    if "views" in group:
        return [col for col in group["views"] if col in columns]
    regex = group.get("views_regex")
    if regex:
        return list(pd.Series(sorted(columns))[lambda s: s.str.match(regex)])
    return []


def _per_embedding_metrics(
    x: np.ndarray,
    metric_names: list[str],
    config: EmbeddingMetricsConfig,
    context: str,
) -> dict[str, float]:
    metrics = {"n_samples": int(x.shape[0]), "dim": int(x.shape[1])}
    metric_fns = {
        "effective_rank": effective_rank,
        "rankme": rankme,
        "stable_rank": stable_rank,
        "anisotropy": anisotropy,
        "effdim": effdim,
    }
    for metric_name, metric_fn in metric_fns.items():
        if metric_name in metric_names:
            metrics[metric_name] = _timed_metric(
                metric_name, context, metric_fn, x
            )

    if "asmi" in metric_names:
        asmi_config = config.asmi or {}
        sigmas = _as_list(asmi_config.get("sigmas", [0.01, 0.1, 1.0]))
        projection_dims = _as_list(
            asmi_config.get("projection_dims", asmi_config.get("projection_dim", 2))
        )
        n_projections_values = _as_list(asmi_config.get("n_projections", 50))
        ksg_ks = _as_list(asmi_config.get("ksg_ks", asmi_config.get("ksg_k", 1)))
        for sigma in sigmas:
            for projection_dim in projection_dims:
                for n_projections in n_projections_values:
                    for ksg_k in ksg_ks:
                        metrics[
                            "asmi_ksg_pca_whiten"
                            f"_sigma{float(sigma):g}"
                            f"_dim{int(projection_dim)}"
                            f"_k{int(ksg_k)}"
                            f"_m{int(n_projections)}"
                        ] = _timed_metric(
                            "asmi",
                            context
                            + f" sigma={float(sigma):g}"
                            + f" dim={int(projection_dim)}"
                            + f" k={int(ksg_k)}"
                            + f" projections={int(n_projections)}",
                            asmi_ksg_pca_whiten,
                            x,
                            sigma=float(sigma),
                            projection_dim=int(projection_dim),
                            n_projections=int(n_projections),
                            ksg_k=int(ksg_k),
                            seed=config.sample_seed,
                        )
    return metrics


def _tc_pairs(config: EmbeddingMetricsConfig) -> list[dict]:
    tc_config = config.total_compression or {}
    pairs = list(tc_config.get("pairs", []))

    chain = tc_config.get("chain", [])
    if chain:
        pairs = [
            {"source": source, "target": target}
            for source, target in zip(chain[:-1], chain[1:])
        ] + pairs

    target = tc_config.get("target")
    sources = tc_config.get("sources", [])
    if target and sources:
        pairs = [{"source": source, "target": target} for source in sources] + pairs
    return pairs


def _compute_embedding_column_metrics(
    train_path: str,
    test_path: str,
    embedding_columns: list[EmbeddingColumnContext],
    config: EmbeddingMetricsConfig,
) -> dict[str, float]:
    metrics = {}
    for split, column, global_train_value in embedding_columns:
        path = _metric_path(split, train_path, test_path, global_train_value)
        x = _load_embedding_sample(
            path=path,
            column=column,
            sample_size=config.sample_size,
            seed=config.sample_seed,
            global_train_value=global_train_value,
        )
        if x is None:
            continue

        column_name = _metric_column_name(column)
        for metric_name, value in _per_embedding_metrics(
            x,
            config.metrics,
            config,
            context=f"split={split} source={column}",
        ).items():
            metrics[f"embedding__{split}__{column_name}__{metric_name}"] = value
    return metrics


def _compute_tc_metrics(
    train_path: str,
    test_path: str,
    embedding_columns: list[EmbeddingColumnContext],
    config: EmbeddingMetricsConfig,
) -> dict[str, float]:
    metrics = {}
    for pair in _tc_pairs(config):
        source_column = pair["source"]
        target_column = pair["target"]
        for split, global_train_value in _unique_contexts(embedding_columns):
            path = _metric_path(split, train_path, test_path, global_train_value)
            path_columns = _parquet_columns(path)
            if source_column not in path_columns or target_column not in path_columns:
                continue
            arrays = _load_pair_sample(
                path=path,
                source_column=source_column,
                target_column=target_column,
                sample_size=config.sample_size,
                seed=config.sample_seed,
                global_train_value=global_train_value,
            )
            if arrays is None:
                continue
            source_x, target_x = arrays
            source_name = _metric_column_name(source_column)
            target_name = _metric_column_name(target_column)
            prefix = f"embedding__{split}__tc__{source_name}_to_{target_name}"
            context = f"split={split} source={source_column} target={target_column}"
            metrics[f"{prefix}__total_compression"] = _timed_metric(
                "total_compression",
                context,
                total_compression,
                source_x,
                target_x,
            )
            metrics[f"{prefix}__source_effdim"] = _timed_metric(
                "effdim", context + " role=source", effdim, source_x
            )
            metrics[f"{prefix}__target_effdim"] = _timed_metric(
                "effdim", context + " role=target", effdim, target_x
            )
    return metrics


def _compute_lidar_metrics(
    train_path: str,
    test_path: str,
    embedding_columns: list[EmbeddingColumnContext],
    config: EmbeddingMetricsConfig,
) -> dict[str, float]:
    lidar_config = config.lidar or {}
    groups = lidar_config.get("groups", [])
    if isinstance(groups, dict):
        groups = [groups]

    metrics = {}
    for group in groups:
        regs = group.get("reg", lidar_config.get("reg", [1e-4]))
        if not isinstance(regs, list):
            regs = [regs]
        group_sample_size = group.get("sample_size", config.sample_size)
        group_name = group.get("name", "views")
        for split, global_train_value in _unique_contexts(embedding_columns):
            path = _metric_path(split, train_path, test_path, global_train_value)
            views = _view_columns_from_group(group, _parquet_columns(path))
            if len(views) < 2:
                continue
            view_embeddings = _load_views_sample(
                path=path,
                views=views,
                sample_size=group_sample_size,
                seed=config.sample_seed,
                global_train_value=global_train_value,
            )
            if view_embeddings is None:
                continue
            for reg in regs:
                metrics[
                    f"embedding__{split}__lidar__{group_name}"
                    f"__views_{len(views)}__reg_{float(reg):g}"
                ] = _timed_metric(
                    "lidar",
                    f"split={split} group={group_name}"
                    f" views={len(views)} samples={len(view_embeddings)}"
                    f" reg={float(reg):g}",
                    lidar_from_views,
                    view_embeddings,
                    reg=float(reg),
                )
    return metrics


def compute_embedding_metrics(
    train_path: str,
    test_path: str,
    config: EmbeddingMetricsConfig,
) -> dict[str, float]:
    train_columns = _parquet_columns(train_path)
    test_columns = _parquet_columns(test_path)
    embedding_columns = _configured_embedding_columns(
        config, train_columns, test_columns
    )

    metrics = {}
    metrics.update(
        _compute_embedding_column_metrics(
            train_path, test_path, embedding_columns, config
        )
    )
    metrics.update(
        _compute_tc_metrics(train_path, test_path, embedding_columns, config)
    )
    metrics.update(
        _compute_lidar_metrics(train_path, test_path, embedding_columns, config)
    )
    return metrics
