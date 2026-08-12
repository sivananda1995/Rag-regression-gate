"""Metric behaviour is verified against values computed by hand, not against the
implementation's own output, so a refactor cannot quietly redefine recall."""

from __future__ import annotations

import math

import pytest

from ragate.metrics import (
    dedupe_preserving_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_dedupe_keeps_best_rank():
    assert dedupe_preserving_rank(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_dedupe_is_stable_on_empty_input():
    assert dedupe_preserving_rank([]) == []


@pytest.mark.parametrize(
    "ranked, relevant, k, expected",
    [
        (["a", "b", "c"], ["a"], 3, 1.0),
        (["a", "b", "c"], ["c"], 2, 0.0),
        (["a", "b", "c"], ["a", "c"], 3, 1.0),
        (["a", "b", "c"], ["a", "z"], 3, 0.5),
        (["a", "b", "c", "d"], ["a", "b", "z", "y"], 4, 0.5),
    ],
)
def test_recall_at_k(ranked, relevant, k, expected):
    assert recall_at_k(ranked, relevant, k) == pytest.approx(expected)


def test_recall_ignores_documents_past_k():
    assert recall_at_k(["x", "y", "a"], ["a"], 2) == 0.0


def test_precision_divides_by_k_not_by_hits():
    assert precision_at_k(["a", "x", "y", "z"], ["a"], 4) == pytest.approx(0.25)


@pytest.mark.parametrize(
    "ranked, expected",
    [(["a", "x", "y"], 1.0), (["x", "a", "y"], 0.5), (["x", "y", "a"], 1 / 3), (["x", "y"], 0.0)],
)
def test_reciprocal_rank(ranked, expected):
    assert reciprocal_rank(ranked, ["a"], 3) == pytest.approx(expected)


def test_ndcg_perfect_ordering_is_one():
    assert ndcg_at_k(["a", "b", "x"], ["a", "b"], 3) == pytest.approx(1.0)


def test_ndcg_matches_hand_computation():
    # One relevant document at rank 2: gain = 1/log2(3), ideal = 1/log2(2) = 1.
    assert ndcg_at_k(["x", "a", "y"], ["a"], 3) == pytest.approx(1 / math.log2(3))


def test_ndcg_penalises_lower_rank():
    high = ndcg_at_k(["a", "x", "y"], ["a"], 3)
    low = ndcg_at_k(["x", "y", "a"], ["a"], 3)
    assert high > low


def test_metrics_reject_empty_relevant_set():
    with pytest.raises(ValueError):
        recall_at_k(["a"], [], 1)
    with pytest.raises(ValueError):
        ndcg_at_k(["a"], [], 1)


def test_metrics_reject_non_positive_k():
    with pytest.raises(ValueError):
        recall_at_k(["a"], ["a"], 0)
    with pytest.raises(ValueError):
        precision_at_k(["a"], ["a"], -1)
