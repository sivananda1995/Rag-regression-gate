"""BM25 is checked against arithmetic done outside the implementation.

The expected values below were computed from the published formula on paper, not by
running this code and copying the output, which would only prove the implementation
agrees with itself. Corpus for every hand-computed case:

    u0 = "the cat sat"                (3 tokens)
    u1 = "the dog sat on the mat"     (6 tokens)
    u2 = "cat"                        (1 token)

    N = 3, avgdl = 10/3 = 3.33333, k1 = 1.5, b = 0.75

    idf(t)      = ln(1 + (N - n(t) + 0.5) / (n(t) + 0.5))
    idf("cat")  = ln(1 + 1.5/2.5) = ln(1.6)     = 0.470004   (n = 2)
    idf("dog")  = ln(1 + 2.5/1.5) = ln(2.66667) = 0.980829   (n = 1)

    score(t, d) = idf(t) * f * (k1 + 1) / (f + k1 * (1 - b + b * |d| / avgdl))
    score("cat", u0) = 0.470004 * 1 * 2.5 / (1 + 1.5 * 0.925)   = 0.492150
    score("cat", u2) = 0.470004 * 1 * 2.5 / (1 + 1.5 * 0.475)   = 0.686137
    score("dog", u1) = 0.980829 * 1 * 2.5 / (1 + 1.5 * 1.6)     = 0.721198
"""

from __future__ import annotations

import pytest

from ragate.corpus import IndexUnit
from ragate.errors import RagateError
from ragate.retrievers.bm25 import Bm25Retriever, tokenize

UNITS = [
    IndexUnit(text="the cat sat", doc_ids=("d0",), occurrences=1),
    IndexUnit(text="the dog sat on the mat", doc_ids=("d1",), occurrences=1),
    IndexUnit(text="cat", doc_ids=("d2",), occurrences=1),
]


@pytest.fixture
def fitted() -> Bm25Retriever:
    retriever = Bm25Retriever()
    retriever.fit(UNITS)
    return retriever


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("Reset MFA-4021, now!") == ["reset", "mfa", "4021", "now"]


def test_single_term_scores_match_hand_computation(fitted):
    ranking = dict(fitted.search(["cat"], k=3)[0])
    assert ranking[2] == pytest.approx(0.686137, abs=1e-6)
    assert ranking[0] == pytest.approx(0.492150, abs=1e-6)
    assert ranking[1] == pytest.approx(0.0, abs=1e-12)


def test_rare_term_scores_above_common_term(fitted):
    """dog appears in one unit of three, the appears in two, so dog must weigh more."""
    dog = dict(fitted.search(["dog"], k=3)[0])[1]
    the = dict(fitted.search(["the"], k=3)[0])[1]
    assert dog == pytest.approx(0.721198, abs=1e-6)
    assert dog > the


def test_shorter_unit_wins_on_equal_term_frequency(fitted):
    """Length normalisation: one occurrence of cat in a 1-token unit beats one
    occurrence in a 3-token unit."""
    ranking = fitted.search(["cat"], k=3)[0]
    assert ranking[0][0] == 2
    assert ranking[1][0] == 0


def test_multi_term_query_sums_term_contributions(fitted):
    ranking = dict(fitted.search(["cat sat"], k=3)[0])
    assert ranking[0] == pytest.approx(0.984301, abs=1e-6)
    assert ranking[2] == pytest.approx(0.686137, abs=1e-6)
    assert ranking[1] == pytest.approx(0.345591, abs=1e-6)


def test_ranking_is_sorted_best_first(fitted):
    scores = [score for _, score in fitted.search(["cat sat"], k=3)[0]]
    assert scores == sorted(scores, reverse=True)


def test_unknown_terms_score_zero_without_raising(fitted):
    ranking = fitted.search(["nonexistentterm"], k=3)[0]
    assert all(score == pytest.approx(0.0) for _, score in ranking)


def test_k_larger_than_corpus_is_clamped(fitted):
    assert len(fitted.search(["cat"], k=99)[0]) == 3


def test_b_zero_disables_length_normalisation():
    """With b = 0 the two units containing one cat each score identically, because the
    only difference between them is length."""
    retriever = Bm25Retriever(b=0.0)
    retriever.fit(UNITS)
    ranking = dict(retriever.search(["cat"], k=3)[0])
    assert ranking[0] == pytest.approx(ranking[2])


def test_search_before_fit_is_refused():
    with pytest.raises(RagateError, match="fit\\(\\) must be called"):
        Bm25Retriever().search(["cat"], k=1)


def test_empty_corpus_is_refused():
    with pytest.raises(RagateError, match="at least one index unit"):
        Bm25Retriever().fit([])


@pytest.mark.parametrize(
    "kwargs, message",
    [({"k1": -0.1}, "k1"), ({"b": 1.5}, "b must be"), ({"b": -0.2}, "b must be")],
)
def test_invalid_parameters_are_refused(kwargs, message):
    with pytest.raises(RagateError, match=message):
        Bm25Retriever(**kwargs)


def test_results_are_identical_across_instances():
    first, second = Bm25Retriever(), Bm25Retriever()
    first.fit(UNITS)
    second.fit(UNITS)
    assert first.search(["cat sat"], k=3) == second.search(["cat sat"], k=3)
