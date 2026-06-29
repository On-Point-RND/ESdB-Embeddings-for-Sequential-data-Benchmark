from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import torch

from ...types import Batch, Seq


class RepresentationExtractor(Protocol):
    def extract(
        self,
        model: torch.nn.Module,
        batch: Batch,
        config: Mapping[str, Any],
    ) -> dict[str, torch.Tensor]: ...


def _flatten_representations(output: Any) -> dict[str, torch.Tensor]:
    if isinstance(output, torch.Tensor):
        return {"embedding": output}
    if not isinstance(output, Mapping):
        raise TypeError(f"Unsupported representation output type: {type(output)!r}")

    flat: dict[str, torch.Tensor] = {}
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            flat[key] = value
        elif isinstance(value, Mapping):
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, torch.Tensor):
                    flat[f"{key}_{nested_key}"] = nested_value
                elif isinstance(nested_value, (list, tuple)):
                    for idx, item in enumerate(nested_value):
                        if isinstance(item, torch.Tensor):
                            flat[f"{key}_{nested_key}_{idx}"] = item
        elif isinstance(value, (list, tuple)):
            for idx, item in enumerate(value):
                if isinstance(item, torch.Tensor):
                    flat[f"{key}_{idx}"] = item
    return flat


class SequentialRepresentationExtractor:
    def extract(
        self,
        model: torch.nn.Module,
        batch: Batch,
        config: Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        if not isinstance(model, torch.nn.Sequential):
            raise TypeError("Sequential extractor requires torch.nn.Sequential")

        layer_names = list(config.get("layer_names", []))
        aggregation = model[-1]
        value: Any = batch
        representations: dict[str, torch.Tensor] = {}

        for index, module in enumerate(model):
            name = (
                layer_names[index]
                if index < len(layer_names)
                else module.__class__.__name__.lower()
            )
            forward_with_representations = getattr(
                module, "forward_with_representations", None
            )
            if callable(forward_with_representations):
                value, nested = forward_with_representations(value)
                for nested_name, nested_value in nested.items():
                    representations[f"{name}_{nested_name}"] = self._as_embedding(
                        nested_value, aggregation
                    )
            else:
                value = module(value)

            if index < len(model) - 1:
                representations[name] = self._as_embedding(value, aggregation)

        representations["embedding"] = self._as_embedding(value, aggregation)
        return representations

    @staticmethod
    def _as_embedding(value: Any, aggregation: torch.nn.Module) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, Seq):
            embedding = aggregation(value)
            if isinstance(embedding, torch.Tensor):
                return embedding
        raise TypeError(f"Cannot convert {type(value)!r} to an embedding")


class DefaultRepresentationExtractor:
    def extract(
        self,
        model: torch.nn.Module,
        batch: Batch,
        config: Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        get_representations = getattr(model, "get_representations", None)
        if callable(get_representations):
            kwargs = {
                key: value
                for key, value in config.items()
                if key not in {"enabled", "extractor", "lidar", "primary", "strict"}
            }
            return _flatten_representations(get_representations(batch, **kwargs))

        get_query_embeddings = getattr(model, "get_query_embeddings", None)
        if callable(get_query_embeddings):
            return {"embedding": get_query_embeddings(batch)}
        return {"embedding": model(batch)}
