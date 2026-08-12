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
        # Squared L2, not inner product, even though the goal is cosine similarity.
        # HNSW navigates by greedy descent over a proximity graph, and that descent
        # assumes a true metric; inner product provides no triangle inequality for the
        # traversal to lean on. For unit-length vectors the two rankings are
        # equivalent, because ||a - b||^2 = 2 - 2*cos(a, b), so ascending squared L2
        # is exactly descending cosine and nothing is given up by choosing the metric
        # the structure was designed for.
        #
        # Honest scope of this choice: it is correctness by construction, not a
        # measured win. On this corpus the two metrics scored 0.814 and 0.808 on the
        # retrieved-score parity check in benchmark/bench_retrieval.py, a difference
        # inside run-to-run variation. The large parity loss that prompted the
        # investigation came from duplicate vectors, not from the metric; see the war
        # story in the README.
        index = self._faiss.IndexHNSWFlat(vectors.shape[1], self._m, self._faiss.METRIC_L2)
        index.hnsw.efConstruction = self._ef_construction
        index.hnsw.efSearch = self._ef_search
        index.add(vectors)
        self._index = index
        log.info(
            "faiss index built",
            extra={"vectors": int(vectors.shape[0]), "m": self._m,
                   "ef_construction": self._ef_construction, "metric": "l2_on_unit_vectors"},
        )

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return cosine similarities, so callers cannot tell the backends apart.

        FAISS reports squared L2 distances here. They are converted back with
        cos = 1 - d/2, which is exact for unit-length vectors, keeping the score
        semantics identical to the exact backend.
        """
        if self._index is None:
            raise RagateError("build() must be called before search()")
        distances, indices = self._index.search(
            np.ascontiguousarray(queries, dtype=np.float32), k
        )
        similarities = 1.0 - (distances / 2.0)
        return similarities.astype(np.float32), indices
