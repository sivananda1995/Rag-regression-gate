"""Vector index backends."""

from __future__ import annotations

from ..config import IndexConfig
from ..errors import RagateError
from .base import VectorIndex
from .flat import FlatIndex

__all__ = ["VectorIndex", "FlatIndex", "build_index"]


def build_index(cfg: IndexConfig) -> VectorIndex:
    if cfg.backend == "flat":
        return FlatIndex()
    if cfg.backend == "faiss":
        from .faiss_index import FaissHnswIndex

        return FaissHnswIndex()
    raise RagateError(f"unknown index backend: {cfg.backend}")
