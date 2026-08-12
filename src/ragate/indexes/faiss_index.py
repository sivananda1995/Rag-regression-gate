"""FAISS HNSW index for the production-scale path.

Included because a golden set of a few thousand chunks is not the shape of the real
corpus this gate is meant to protect, and the harness should be able to measure the
recall cost of the approximate index itself: run the same golden set with
index.backend=flat and index.backend=faiss, and the difference is the ANN recall
loss at those build parameters, separated from any pipeline change.
"""

from __future__ import annotations

import numpy as np

from ..errors import RagateError
from ..logging_setup import get_logger

log = get_logger(__name__)

_M = 32            # graph degree
_EF_CONSTRUCTION = 128
_EF_SEARCH = 64


class FaissHnswIndex:
    name = "faiss-hnsw"

    def __init__(self, m: int = _M, ef_construction: int = _EF_CONSTRUCTION,
                 ef_search: int = _EF_SEARCH) -> None:
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RagateError("install the faiss extra: pip install '.[faiss]'") from exc
        self._faiss = faiss
        self._index = None
        self._m = m
        self._ef_construction = ef_construction
        self._ef_search = ef_search

    def build(self, vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        # Vectors are already L2 normalised, so inner product equals cosine similarity.
        index = self._faiss.IndexHNSWFlat(
            vectors.shape[1], self._m, self._faiss.METRIC_INNER_PRODUCT
        )
        index.hnsw.efConstruction = self._ef_construction
        index.hnsw.efSearch = self._ef_search
        index.add(vectors)
        self._index = index
        log.info(
            "faiss index built",
            extra={"vectors": int(vectors.shape[0]), "m": self._m,
                   "ef_construction": self._ef_construction},
        )

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self._index is None:
            raise RagateError("build() must be called before search()")
        scores, indices = self._index.search(
            np.ascontiguousarray(queries, dtype=np.float32), k
        )
        return scores, indices
