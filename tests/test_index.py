from __future__ import annotations

import numpy as np
import pytest

from ragate.errors import RagateError
from ragate.indexes.flat import FlatIndex


def _unit(rows: list[list[float]]) -> np.ndarray:
    array = np.asarray(rows, dtype=np.float32)
    return array / np.linalg.norm(array, axis=1, keepdims=True)


def test_flat_index_returns_nearest_first():
    index = FlatIndex()
    index.build(_unit([[1, 0], [0.9, 0.1], [0, 1]]))
    scores, indices = index.search(_unit([[1, 0]]), k=3)
    assert list(indices[0]) == [0, 1, 2]
    assert scores[0][0] >= scores[0][1] >= scores[0][2]


def test_k_larger_than_corpus_is_clamped():
    index = FlatIndex()
    index.build(_unit([[1, 0], [0, 1]]))
    _, indices = index.search(_unit([[1, 0]]), k=10)
    assert indices.shape == (1, 2)


def test_search_before_build_is_refused():
    with pytest.raises(RagateError, match="build\\(\\) must be called"):
        FlatIndex().search(_unit([[1, 0]]), k=1)


def test_dimension_mismatch_is_reported_clearly():
    index = FlatIndex()
    index.build(_unit([[1, 0]]))
    with pytest.raises(RagateError, match="does not match"):
        index.search(_unit([[1, 0, 0]]), k=1)


def test_faiss_backend_agrees_with_exact_search_on_a_small_corpus():
    faiss_index = pytest.importorskip("ragate.indexes.faiss_index")
    vectors = _unit(np.random.default_rng(7).normal(size=(200, 32)).tolist())
    queries = vectors[:5]
    exact = FlatIndex()
    exact.build(vectors)
    _, exact_top = exact.search(queries, k=1)
    approximate = faiss_index.FaissHnswIndex()
    approximate.build(vectors)
    _, ann_top = approximate.search(queries, k=1)
    # At this size HNSW should find the exact nearest neighbour, which is itself.
    assert list(exact_top[:, 0]) == list(ann_top[:, 0])
