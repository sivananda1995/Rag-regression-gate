"""Embedding providers."""

from __future__ import annotations

from ..config import EmbedderConfig
from ..errors import EmbedderError
from .base import Embedder
from .hashing import HashingEmbedder

__all__ = ["Embedder", "HashingEmbedder", "build_embedder"]


def build_embedder(cfg: EmbedderConfig) -> Embedder:
    if cfg.provider == "hashing":
        return HashingEmbedder(
            dimensions=cfg.dimensions,
            char_ngram=cfg.char_ngram,
            idf_weighting=cfg.idf_weighting,
        )
    if cfg.provider == "openai":
        from .openai_provider import OpenAIEmbedder

        return OpenAIEmbedder(
            model=cfg.model or "text-embedding-3-small", dimensions=cfg.dimensions
        )
    raise EmbedderError(f"unknown embedder provider: {cfg.provider}")
