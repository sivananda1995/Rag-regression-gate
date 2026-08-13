"""Retrieval strategies."""

from __future__ import annotations

from ..config import Config
from ..errors import ConfigError
from .base import Ranking, Retriever
from .bm25 import Bm25Retriever
from .dense import DenseRetriever
from .hybrid import HybridRetriever

__all__ = [
    "Ranking",
    "Retriever",
    "Bm25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "build_retriever",
]


def build_retriever(cfg: Config) -> Retriever:
    mode = cfg.retriever.mode
    if mode == "dense":
        return DenseRetriever(cfg.embedder, cfg.index)
    if mode == "bm25":
        return Bm25Retriever(k1=cfg.retriever.bm25_k1, b=cfg.retriever.bm25_b)
    if mode == "hybrid":
        return HybridRetriever(
            parts=[
                (Bm25Retriever(k1=cfg.retriever.bm25_k1, b=cfg.retriever.bm25_b),
                 cfg.retriever.bm25_weight),
                (DenseRetriever(cfg.embedder, cfg.index), cfg.retriever.dense_weight),
            ],
            rrf_k=cfg.retriever.rrf_k,
            candidate_multiplier=cfg.retriever.candidate_multiplier,
        )
    raise ConfigError(f"unknown retriever mode: {mode}")
