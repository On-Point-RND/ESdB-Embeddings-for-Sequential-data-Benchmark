from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import (
    DefaultRepresentationExtractor,
    RepresentationExtractor,
    SequentialRepresentationExtractor,
)


_EXTRACTORS: dict[str, type[RepresentationExtractor]] = {
    "default": DefaultRepresentationExtractor,
    "sequential": SequentialRepresentationExtractor,
}


def build_representation_extractor(
    config: Mapping[str, Any] | None,
) -> RepresentationExtractor:
    name = "default" if config is None else config.get("extractor", "default")
    if name not in _EXTRACTORS:
        raise KeyError(f"Unknown representation extractor: {name}")
    return _EXTRACTORS[name]()
