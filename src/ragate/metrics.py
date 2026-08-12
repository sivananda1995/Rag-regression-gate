"""Retrieval metrics, computed at document level.

Every metric takes a ranked list of document ids (duplicates already collapsed,
best rank kept) and the set of labeled relevant documents for that query. Keeping
them pure functions of those two inputs is what makes them unit-testable against
hand-computed values, which tests/test_metrics.py does.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def dedupe_preserving_rank(doc_ids: Iterable[str]) -> list[str]:
    """Collapse repeated documents to their best (earliest) rank.

    Retrieval runs over chunks, so one document can occupy several of the top-k
    chunk slots. Counting those slots separately inflates every metric, because a
    single document appearing three times looks like three retrieved documents.
    """
    seen: set[str] = set()
    out: list[str] = []
    for doc_id in doc_ids:
        if doc_id not in seen:
            seen.add(doc_id)
            out.append(doc_id)
    return out


def recall_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Share of labeled relevant documents that appear in the top k."""
    relevant_set = set(relevant)
    if not relevant_set:
        raise ValueError("recall_at_k requires at least one relevant document")
    if k <= 0:
        raise ValueError("k must be positive")
    hits = len(relevant_set.intersection(ranked[:k]))
    return hits / len(relevant_set)


def precision_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    relevant_set = set(relevant)
    return len(relevant_set.intersection(ranked[:k])) / k


def reciprocal_rank(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """1/rank of the first relevant document inside the top k, else 0."""
    relevant_set = set(relevant)
    for position, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant_set:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Binary-gain nDCG@k. Ideal ordering puts every relevant document first."""
    relevant_set = set(relevant)
    if not relevant_set:
        raise ValueError("ndcg_at_k requires at least one relevant document")
    gain = sum(
        1.0 / math.log2(position + 1)
        for position, doc_id in enumerate(ranked[:k], start=1)
        if doc_id in relevant_set
    )
    ideal = sum(
        1.0 / math.log2(position + 1)
        for position in range(1, min(len(relevant_set), k) + 1)
    )
    return gain / ideal if ideal else 0.0


METRICS = {
    "recall_at_k": recall_at_k,
    "precision_at_k": precision_at_k,
    "mrr_at_k": reciprocal_rank,
    "ndcg_at_k": ndcg_at_k,
}
