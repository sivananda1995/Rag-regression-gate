"""Dense retriever: an embedder plus a vector index, behind the Retriever interface."""

from __future__ import annotations

from collections.abc import Sequence

from ..config import EmbedderConfig, IndexConfig
from ..corpus import IndexUnit
from ..embedders import build_embedder
from ..indexes import build_index
from ..logging_setup import get_logger
from .base import Ranking

log = get_logger(__name__)


class DenseRetriever:
    name = "dense"

    def __init__(self, embedder_cfg: EmbedderConfig, index_cfg: IndexConfig) -> None:
        self._embedder = build_embedder(embedder_cfg)
        self._index = build_index(index_cfg)
        self.name = f"dense:{self._embedder.name}+{self._index.name}"

    def fit(self, units: Sequence[IndexUnit]) -> None:
        texts = [unit.text for unit in units]
        self._embedder.fit(texts)
        self._index.build(self._embedder.encode(texts))

    def search(self, queries: Sequence[str], k: int) -> list[Ranking]:
        vectors = self._embedder.encode(list(queries))
        scores, indices = self._index.search(vectors, k)
        return [
            [(int(i), float(s)) for i, s in zip(row_i, row_s, strict=True) if int(i) >= 0]
            for row_i, row_s in zip(indices, scores, strict=True)
        ]
