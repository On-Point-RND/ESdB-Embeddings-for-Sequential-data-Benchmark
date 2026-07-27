from __future__ import annotations

import contextlib
from dataclasses import replace
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import torch

Tensor = torch.Tensor


def _second_order_sdp_context():
    """
    Efficient/flash SDPA kernels often do not support grad-grad.
    For Hutchinson/HVP we force math SDPA locally.
    """
    nn_attention = getattr(torch.nn, "attention", None)
    sdpa_kernel = getattr(nn_attention, "sdpa_kernel", None)
    sdp_backend = getattr(nn_attention, "SDPBackend", None)
    if sdpa_kernel is not None and sdp_backend is not None:
        return sdpa_kernel(backends=[sdp_backend.MATH])

    cuda_backends = getattr(torch.backends, "cuda", None)
    sdp_kernel = getattr(cuda_backends, "sdp_kernel", None)
    if sdp_kernel is not None:
        return sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
    return contextlib.nullcontext()


def _params_of(model: Any, only_requires_grad: bool = True) -> list[Tensor]:
    if only_requires_grad:
        return [p for p in model.parameters() if p.requires_grad]
    return list(model.parameters())


def _rademacher_like(params: list[Tensor]) -> list[Tensor]:
    vec = []
    for p in params:
        v = torch.empty_like(p)
        v.bernoulli_(0.5).mul_(2).sub_(1)
        vec.append(v)
    return vec


def _hvp(loss: Tensor, params: list[Tensor], vec: list[Tensor]) -> list[Tensor]:
    grads = torch.autograd.grad(
        loss,
        params,
        create_graph=True,
        allow_unused=True,
    )

    dot_terms = [(g * v).sum() for g, v in zip(grads, vec) if g is not None]
    if not dot_terms:
        return [torch.zeros_like(p) for p in params]
    dot = sum(dot_terms)

    hv = torch.autograd.grad(
        dot,
        params,
        allow_unused=True,
    )
    return [torch.zeros_like(p) if h is None else h for h, p in zip(hv, params)]


def trace_hessian_hutchinson(
    loss_fn: Callable[[Any, Any], Tensor],
    model: Any,
    batch: Any,
    K: int = 2,
    params: list[Tensor] | None = None,
) -> float:
    if K <= 0:
        raise ValueError(f"K should be positive, got {K}")

    if params is None:
        params = _params_of(model)
    if not params:
        raise ValueError("No parameters to differentiate")

    tr = 0.0
    with _second_order_sdp_context():
        for _ in range(K):
            loss = loss_fn(model, batch)
            if loss.ndim != 0:
                loss = loss.mean()

            v = _rademacher_like(params)
            hv = _hvp(loss, params, v)
            v_h_v = sum((vi * hvi).sum() for vi, hvi in zip(v, hv))
            tr += float(v_h_v.detach().float().cpu().item())
            del loss, v, hv, v_h_v

    return tr / K


def estimate_hessian_trace_on_loader(
    *,
    model: Any,
    loss_fn: Callable[[Any, Any], Tensor],
    loader: Iterable[Any],
    device: str | torch.device,
    K: int = 2,
    n_batches: int = 1,
    params: list[Tensor] | None = None,
) -> float:
    if n_batches <= 0:
        raise ValueError(f"n_batches should be positive, got {n_batches}")

    if params is None:
        params = _params_of(model)
    if not params:
        raise ValueError("No parameters to differentiate")

    was_training = bool(model.training)
    model.eval()

    traces: list[float] = []
    try:
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            batch.to(device)
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                trace = trace_hessian_hutchinson(
                    loss_fn=loss_fn,
                    model=model,
                    batch=batch,
                    K=K,
                    params=params,
                )
            traces.append(trace)
            model.zero_grad(set_to_none=True)
    finally:
        model.train(was_training)

    if not traces:
        raise ValueError("Loader did not yield batches")
    return float(sum(traces) / len(traces))


def _slice_batch_item(batch: Any, item_idx: int) -> Any:
    """
    Return a batch with a single sample selected by batch index.
    Expects the project Batch dataclass-like object.
    """
    kwargs = {}
    batch_size = int(batch.lengths.shape[0])
    if item_idx < 0 or item_idx >= batch_size:
        raise IndexError(f"item_idx={item_idx} is out of range for batch_size={batch_size}")

    for name, value in batch.__dict__.items():
        if value is None:
            kwargs[name] = None
            continue

        if name == "lengths":
            kwargs[name] = value[item_idx : item_idx + 1]
            continue

        if name in {"cat_features", "num_features", "cat_mask", "num_mask", "emb_mask"}:
            kwargs[name] = value[:, item_idx : item_idx + 1, ...]
            continue

        if name == "time":
            if isinstance(value, torch.Tensor):
                kwargs[name] = value[:, item_idx : item_idx + 1, ...]
            elif isinstance(value, np.ndarray):
                kwargs[name] = value[:, item_idx : item_idx + 1, ...]
            else:
                kwargs[name] = value
            continue

        if name == "index":
            if isinstance(value, torch.Tensor):
                kwargs[name] = value[item_idx : item_idx + 1, ...]
            elif isinstance(value, np.ndarray):
                kwargs[name] = value[item_idx : item_idx + 1, ...]
            else:
                kwargs[name] = value
            continue

        if name == "target":
            if isinstance(value, torch.Tensor):
                if value.ndim == 1:
                    kwargs[name] = value[item_idx : item_idx + 1]
                elif value.ndim >= 2 and value.shape[1] == batch_size:
                    kwargs[name] = value[:, item_idx : item_idx + 1, ...]
                elif value.ndim >= 2 and value.shape[0] == batch_size:
                    kwargs[name] = value[item_idx : item_idx + 1, ...]
                else:
                    kwargs[name] = value
            else:
                kwargs[name] = value
            continue

        if name == "emb_features":
            kwargs[name] = {
                k: v[:, :, item_idx : item_idx + 1, ...] for k, v in value.items()
            }
            continue

        # names/meta fields: keep as-is
        kwargs[name] = value

    # Batch is a dataclass; replace keeps type and untouched defaults.
    return replace(batch, **kwargs)


def estimate_efim_trace_on_loader(
    *,
    model: Any,
    loss_fn: Callable[[Any, Any], Tensor],
    loader: Iterable[Any],
    device: str | torch.device,
    n_batches: int = 1,
    per_sample: bool = True,
    params: list[Tensor] | None = None,
) -> float:
    """
    Estimate empirical Fisher trace proxy:
      Tr(EFIM) ~= E[ ||grad log p||^2 ].
    For generic losses in this project we use squared gradient norm of loss.
    """
    if n_batches <= 0:
        raise ValueError(f"n_batches should be positive, got {n_batches}")

    if params is None:
        params = _params_of(model)
    if not params:
        raise ValueError("No parameters to differentiate")

    was_training = bool(model.training)
    model.eval()

    values: list[float] = []
    sample_count = 0
    try:
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= n_batches:
                break
            batch.to(device)

            if not per_sample:
                model.zero_grad(set_to_none=True)
                with torch.enable_grad():
                    loss = loss_fn(model, batch)
                    grads = torch.autograd.grad(
                        loss,
                        params,
                        allow_unused=True,
                    )
                grad_norm_sq = sum(
                    float((g.detach() ** 2).sum().item()) for g in grads if g is not None
                )
                values.append(grad_norm_sq)
                sample_count += 1
                model.zero_grad(set_to_none=True)
                continue

            batch_size = int(batch.lengths.shape[0])
            for item_idx in range(batch_size):
                one = _slice_batch_item(batch, item_idx)
                one.to(device)
                model.zero_grad(set_to_none=True)
                with torch.enable_grad():
                    loss = loss_fn(model, one)
                    grads = torch.autograd.grad(
                        loss,
                        params,
                        allow_unused=True,
                    )
                grad_norm_sq = sum(
                    float((g.detach() ** 2).sum().item()) for g in grads if g is not None
                )
                values.append(grad_norm_sq)
                sample_count += 1
                model.zero_grad(set_to_none=True)
    finally:
        model.train(was_training)

    if sample_count == 0:
        raise ValueError("Loader did not yield batches")
    return float(sum(values) / sample_count)
