"""The ranking rule, and the property whose absence made CI disagree with the laptop.

The failure this file guards: on this corpus 24 of the 140 golden queries have an exact tie
at the k=5 boundary, np.argpartition decided which tied unit landed inside the cut, and its
choice differs between numpy builds. The fixed-chunking profile therefore scored 0.8826 on a
GitHub runner and 0.8794 locally, on identical code and data.
"""

from __future__ import annotations

import numpy as np
import pytest

from ragate.errors import RagateError
from ragate.indexes.flat import FlatIndex
from ragate.ranking import RANK_PRECISION, rank_rows, rank_scores


def test_orders_by_descending_score():
    assert [i for i, _ in rank_scores(np.array([0.1, 0.9, 0.5]), 3)] == [1, 2, 0]


def test_exact_ties_break_on_the_lower_index():
    """The whole bug in one assertion: a tie must resolve the same way everywhere."""
    assert [i for i, _ in rank_scores(np.array([0.5, 0.9, 0.5, 0.5]), 4)] == [1, 0, 2, 3]


def test_returns_the_unrounded_score_not_the_comparison_value():
    """Quantisation decides the order; the reported score stays what was computed."""
    scores = np.array([0.1234567891234, 0.9])
    assert rank_scores(scores, 2)[1][1] == pytest.approx(0.1234567891234)


def test_differences_below_the_precision_do_not_reorder():
    base = 0.5
    scores = np.array([base, base + 10 ** -(RANK_PRECISION + 2)])
    # The second value is larger, but by less than the comparison precision, so the tie
    # break on index wins and the order is by index.
    assert [i for i, _ in rank_scores(scores, 2)] == [0, 1]


def test_differences_above_the_precision_do_reorder():
    base = 0.5
    scores = np.array([base, base + 10 ** -(RANK_PRECISION - 2)])
    assert [i for i, _ in rank_scores(scores, 2)] == [1, 0]


def test_k_larger_than_the_array_is_clamped():
    assert len(rank_scores(np.array([0.1, 0.2]), 10)) == 2


def test_non_positive_k_is_refused():
    with pytest.raises(RagateError, match="k must be positive"):
        rank_scores(np.array([0.1]), 0)


def test_row_ranking_matches_the_one_dimensional_rule():
    matrix = np.array([[0.1, 0.9, 0.5], [0.5, 0.5, 0.2]])
    rows = rank_rows(matrix, 3)
    assert list(rows[0]) == [1, 2, 0]
    assert list(rows[1]) == [0, 1, 2]


def _unit(rows: list[list[float]]) -> np.ndarray:
    array = np.asarray(rows, dtype=np.float32)
    return array / np.linalg.norm(array, axis=1, keepdims=True)


def test_a_relative_perturbation_at_machine_scale_cannot_change_a_ranking():
    """The regression test for the CI disagreement.

    1e-12 relative is the scale at which two machines' arithmetic can differ: a different
    BLAS, a different numpy build, a different summation order. Rankings must not move.
    """
    rng = np.random.default_rng(11)
    for _ in range(50):
        scores = rng.choice([0.25, 0.5, 0.75, 1.0], size=40)
        perturbed = scores * (1 + 1e-12 * rng.standard_normal(scores.shape))
        assert [i for i, _ in rank_scores(scores, 5)] == [
            i for i, _ in rank_scores(perturbed, 5)
        ]


def test_the_flat_index_ranks_ties_deterministically_too():
    index = FlatIndex()
    # Three identical vectors and one different: the identical ones tie exactly.
    index.build(_unit([[1, 0], [1, 0], [1, 0], [0, 1]]))
    _scores, indices = index.search(_unit([[1, 0]]), k=3)
    assert list(indices[0]) == [0, 1, 2]


def test_bm25_ranking_is_stable_across_repeated_calls():
    from ragate.corpus import IndexUnit
    from ragate.retrievers.bm25 import Bm25Retriever

    units = [
        IndexUnit(text="reset the token for atlas vpn", doc_ids=(f"d{i}",), occurrences=1)
        for i in range(6)
    ]
    retriever = Bm25Retriever()
    retriever.fit(units)
    first = retriever.search(["reset the token"], k=4)
    second = retriever.search(["reset the token"], k=4)
    assert first == second
    # Six identical documents tie exactly, so the order must be by index.
    assert [i for i, _ in first[0]] == [0, 1, 2, 3]
