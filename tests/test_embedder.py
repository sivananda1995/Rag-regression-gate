from __future__ import annotations

import numpy as np
import pytest

from ragate.config import EmbedderConfig
from ragate.embedders import build_embedder
from ragate.embedders.hashing import HashingEmbedder
from ragate.errors import EmbedderError

CORPUS = ["reset the authenticator token", "clear the stuck print queue", "rotate the secret"]


def test_vectors_are_l2_normalised():
    embedder = HashingEmbedder(dimensions=128)
    embedder.fit(CORPUS)
    vectors = embedder.encode(CORPUS)
    assert vectors.shape == (3, 128)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)


def test_encoding_is_identical_across_instances():
    """Two processes must produce the same vector for the same text, otherwise a
    committed baseline is not comparable with a run in CI."""
    first = HashingEmbedder(dimensions=128)
    second = HashingEmbedder(dimensions=128)
    first.fit(CORPUS)
    second.fit(CORPUS)
    np.testing.assert_array_equal(first.encode(CORPUS), second.encode(CORPUS))


def test_similar_text_scores_above_unrelated_text():
    embedder = HashingEmbedder(dimensions=512)
    embedder.fit(CORPUS)
    vectors = embedder.encode(
        ["clear the stuck print queue", "print queue is stuck", "rotate the secret"]
    )
    assert float(vectors[0] @ vectors[1]) > float(vectors[0] @ vectors[2])


def test_idf_weighting_changes_the_representation():
    weighted = HashingEmbedder(dimensions=256, idf_weighting=True)
    unweighted = HashingEmbedder(dimensions=256, idf_weighting=False)
    weighted.fit(CORPUS)
    unweighted.fit(CORPUS)
    assert not np.allclose(weighted.encode(CORPUS), unweighted.encode(CORPUS))


def test_encode_before_fit_is_refused_when_idf_is_on():
    with pytest.raises(EmbedderError, match="fit\\(\\) must be called"):
        HashingEmbedder(dimensions=64).encode(CORPUS)


def test_fit_on_empty_corpus_is_refused():
    with pytest.raises(EmbedderError, match="empty corpus"):
        HashingEmbedder(dimensions=64).fit([])


def test_empty_text_yields_a_zero_vector_not_a_crash():
    embedder = HashingEmbedder(dimensions=64, idf_weighting=False)
    embedder.fit(CORPUS)
    assert float(np.linalg.norm(embedder.encode([""])[0])) == 0.0


def test_builder_rejects_unknown_provider():
    cfg = EmbedderConfig()
    cfg.provider = "telepathy"
    with pytest.raises(EmbedderError, match="unknown embedder provider"):
        build_embedder(cfg)
