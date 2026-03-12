import logging
from collections.abc import Iterable, Mapping
from typing import Any

from ..analysis.hessian import (
    estimate_efim_trace_on_loader,
    estimate_hessian_trace_on_loader,
)

logger = logging.getLogger(__name__)


def _resolve_hessian_loader(
    loader_name: str,
    train_loaders: Mapping[str, Any],
    test_loaders: Mapping[str, Any],
) -> Iterable[Any]:
    if loader_name in train_loaders:
        return train_loaders[loader_name]
    if loader_name in test_loaders:
        return test_loaders[loader_name]
    raise ValueError(f"Unknown hessian loader '{loader_name}'")


def _select_hessian_params(model: Any, params_scope: str):
    if params_scope == "all":
        return [p for p in model.parameters() if p.requires_grad]
    if params_scope == "heads_only":
        selected = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if (
                name.startswith("cat_heads.")
                or name.startswith("num_head.")
                or name.startswith("reconstruction_stem.")
            ):
                selected.append(p)
        return selected
    raise ValueError(f"Unknown hessian params_scope '{params_scope}'")


def compute_hessian_metrics(
    *,
    config: Mapping[str, Any],
    trainer: Any,
    loaders: Mapping[str, Any],
    test_loaders: Mapping[str, Any],
    loss_fn: Any,
) -> dict[str, float]:
    hessian_conf = config.get("hessian")
    if not (isinstance(hessian_conf, Mapping) and hessian_conf.get("enabled", False)):
        return {}

    model = trainer.model
    if model is None:
        raise ValueError("Trainer model is not initialized")

    loader_name = str(hessian_conf.get("loader", "train_val"))
    hessian_loader = _resolve_hessian_loader(loader_name, loaders, test_loaders)

    params_scope = str(hessian_conf.get("params_scope", "all"))
    hessian_params = _select_hessian_params(model, params_scope)
    if not hessian_params:
        raise ValueError(
            f"Hessian params selection is empty for params_scope='{params_scope}'"
        )

    def _loss_on_batch(m: Any, batch: Any):
        pred = m(batch)
        return loss_fn(pred, batch.target)

    estimator = str(hessian_conf.get("estimator", "hutchinson")).lower()
    if estimator not in {"hutchinson", "efim", "both"}:
        raise ValueError(
            "Unknown hessian estimator "
            f"'{estimator}'. Expected one of: hutchinson, efim, both."
        )

    n_params = sum(p.numel() for p in hessian_params)
    per_param = bool(hessian_conf.get("per_param", True))
    metrics: dict[str, float] = {}

    if estimator in {"hutchinson", "both"}:
        try:
            h_trace = estimate_hessian_trace_on_loader(
                model=model,
                loss_fn=_loss_on_batch,
                loader=hessian_loader,
                device=config["device"],
                K=int(hessian_conf.get("K", 2)),
                n_batches=int(hessian_conf.get("n_batches", 1)),
                params=hessian_params,
            )
            metrics["hessian_trace"] = h_trace
            if per_param:
                metrics["hessian_trace_per_param"] = h_trace / max(n_params, 1)
        except RuntimeError as err:
            if estimator == "hutchinson":
                raise
            logger.warning("Hutchinson trace failed: %s", err)

    if estimator in {"efim", "both"}:
        efim_trace = estimate_efim_trace_on_loader(
            model=model,
            loss_fn=_loss_on_batch,
            loader=hessian_loader,
            device=config["device"],
            n_batches=int(hessian_conf.get("n_batches", 1)),
            per_sample=bool(hessian_conf.get("efim_per_sample", True)),
            params=hessian_params,
        )
        metrics["efim_trace"] = efim_trace
        if per_param:
            metrics["efim_trace_per_param"] = efim_trace / max(n_params, 1)

    logger.info("Hessian probe metrics: %s", metrics)
    return metrics
