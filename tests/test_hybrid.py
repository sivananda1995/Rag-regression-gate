"""Reciprocal rank fusion, checked against arithmetic done by hand.

With rrf_k = 60 and unit weights, a document at rank 1 contributes 1/61 = 0.0163934 and
one at rank 2 contributes 1/62 = 0.0161290. Those two numbers are the whole mechanism:
they are close together, which is the point, because fusion is meant to reward agreement
between retrievers rather than the magnitude of either one's score.
"""

from __future__ import annotations

import pytest

from ragate.corpus import IndexUnit
from ragate.errors import ConfigError
from ragate.retrievers.hybrid import HybridRetriever

UNITS = [IndexUnit(text=f"unit {i}", doc_ids=(f"d{i}",), occurrences=1) for i in range(5)]


class FakeRetriever:
    """Returns a fixed ranking, so the test measures fusion and nothing else."""

    def __init__(self, name: str, order: list[int], score: float = 1.0) -> None:
        self.name = name
        self._order = order
        self._score = score
        self.fitted = False

    def fit(self, units) -> None:
        self.fitted = True

    def search(self, queries, k: int):
        return [[(i, self._score) for i in self._order[:k]] for _ in queries]


def test_agreement_beats_a_single_strong_opinion():
    """Unit 1 is second for both retrievers; unit 0 is first for one and absent from the
    other. Two second places (1/62 + 1/62) beat one first place (1/61)."""
    a = FakeRetriever("a", [0, 1])
    b = FakeRetriever("b", [2, 1])
    fused = HybridRetriever([(a, 1.0), (b, 1.0)], rrf_k=60).search(["q"], k=3)[0]
    assert fused[0][0] == 1
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 62)
    assert fused[1][1] == pytest.approx(1 / 61)


def test_scores_ignore_component_magnitudes():
    """One retriever reporting scores a thousand times larger must not change the fusion,
    because only ranks are used."""
    small = FakeRetriever("small", [0, 1], score=0.001)
    large = FakeRetriever("large", [1, 0], score=1000.0)
    fused = HybridRetriever([(small, 1.0), (large, 1.0)], rrf_k=60).search(["q"], k=2)[0]
    assert fused[0][1] == pytest.approx(fused[1][1])


def test_weights_break_the_symmetry():
    a = FakeRetriever("a", [0, 1])
    b = FakeRetriever("b", [1, 0])
    fused = HybridRetriever([(a, 1.0), (b, 0.25)], rrf_k=60).search(["q"], k=2)[0]
    assert fused[0][0] == 0


def test_zero_weight_component_cannot_change_the_order():
    a = FakeRetriever("a", [0, 1, 2])
    b = FakeRetriever("b", [2, 1, 0])
    fused = HybridRetriever([(a, 1.0), (b, 0.0)], rrf_k=60).search(["q"], k=3)[0]
    assert [i for i, _ in fused] == [0, 1, 2]


def test_components_retrieve_deeper_than_k():
    """A candidate ranked below k by both retrievers must still be fusable, so each
    component is asked for k * candidate_multiplier results."""
    deep = FakeRetriever("deep", [4, 3, 2, 1, 0])
    hybrid = HybridRetriever([(deep, 1.0)], rrf_k=60, candidate_multiplier=4)
    assert len(hybrid.search(["q"], k=1)[0]) == 1
    assert len(deep.search(["q"], k=4)[0]) == 4


def test_fit_reaches_every_component():
    a, b = FakeRetriever("a", [0]), FakeRetriever("b", [1])
    HybridRetriever([(a, 1.0), (b, 1.0)]).fit(UNITS)
    assert a.fitted and b.fitted


def test_ties_break_deterministically_on_unit_index():
    a = FakeRetriever("a", [3, 1])
    b = FakeRetriever("b", [1, 3])
    first = HybridRetriever([(a, 1.0), (b, 1.0)], rrf_k=60).search(["q"], k=2)[0]
    second = HybridRetriever([(a, 1.0), (b, 1.0)], rrf_k=60).search(["q"], k=2)[0]
    assert first == second
    assert [i for i, _ in first] == [1, 3]


def test_name_records_the_recipe():
    a = FakeRetriever("bm25", [0])
    b = FakeRetriever("dense:hashing+flat", [1])
    name = HybridRetriever([(a, 1.0), (b, 0.5)]).name
    assert name == "hybrid:bm25@1+dense:hashing+flat@0.5"


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"parts": [], "rrf_k": 60}, "at least one component"),
        ({"parts": [(FakeRetriever("a", [0]), 1.0)], "rrf_k": 0}, "rrf_k"),
        (
            {"parts": [(FakeRetriever("a", [0]), 1.0)], "candidate_multiplier": 0},
            "candidate_multiplier",
        ),
    ],
)
def test_invalid_configuration_is_refused(kwargs, message):
    with pytest.raises(ConfigError, match=message):
        HybridRetriever(**kwargs)
