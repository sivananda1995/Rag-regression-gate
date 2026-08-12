"""Exact cosine search over a dense matrix.

Deliberately the default. See docs/adr/ADR-001-exact-search-for-evaluation.md: an
approximate index contributes its own recall error, which would be indistinguishable
from the pipeline change the gate is trying to measure.
"""

from __future__ import annotations

import numpy as np

from ..errors import RagateError


class FlatIndex:
    name = "flat"

    def __init__(self) -> None:
        self._vectors: np.ndarray | None = None

    def build(self, vectors: np.ndarray) -> None:
        if vectors.ndim != 2:
            raise RagateError("index expects a 2d array of vectors")
        self._vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self._vectors is None:
            raise RagateError("build() must be called before search()")
        if queries.shape[1] != self._vectors.shape[1]:
            raise RagateError(
                f"query dimension {queries.shape[1]} does not match "
                f"index dimension {self._vectors.shape[1]}"
            )
        k = min(k, self._vectors.shape[0])
        scores = np.asarray(queries, dtype=np.float32) @ self._vectors.T
        # argpartition finds the top k in O(n) per query, then only those k are sorted.
        top = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        ordered = np.take_along_axis(scores, top, axis=1).argsort(axis=1)[:, ::-1]
        indices = np.take_along_axis(top, ordered, axis=1)
        return np.take_along_axis(scores, indices, axis=1), indices
