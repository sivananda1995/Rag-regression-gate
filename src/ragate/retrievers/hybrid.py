"""Reciprocal rank fusion of several retrievers.

Why fusion by rank and not by score: BM25 scores are unbounded sums of idf-weighted
term contributions, cosine scores live in [-1, 1], and the two distributions differ per
query. Normalising them onto a shared scale (min-max or z-score) makes the weighting a
function of how many strong candidates a query happened to have, which is not a
property anyone intends to tune. Ranks discard magnitude on purpose, so a document that
both retrievers place near the top wins without either scale mattering.

    RRF(d) = sum over retrievers r of weight(r) / (rrf_k + rank_r(d))

rrf_k damps the contribution of top ranks; the value 60 comes from Cormack, Clarke and
Buettcher (2009) and is exposed as config. See docs/adr/ADR-004.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..corpus import IndexUnit
from ..errors import ConfigError
from ..logging_setup import get_logger
from .base import Ranking, Retriever

log = get_logger(__name__)


class HybridRetriever:
    def __init__(self, parts: list[tuple[Retriever, float]], rrf_k: int = 60,
                 candidate_multiplier: int = 4) -> None:
        if not parts:
            raise ConfigError("hybrid retrieval needs at least one component retriever")
        if rrf_k < 1:
            raise ConfigError("retriever.rrf_k must be at least 1")
        if candidate_multiplier < 1:
            raise ConfigError("retriever.candidate_multiplier must be at least 1")
        self._parts = parts
        self._rrf_k = rrf_k
        self._multiplier = candidate_multiplier
        self.name = "hybrid:" + "+".join(
            f"{retriever.name}@{weight:g}" for retriever, weight in parts
        )

    def fit(self, units: Sequence[IndexUnit]) -> None:
        for retriever, _ in self._parts:
            retriever.fit(units)

    def search(self, queries: Sequence[str], k: int) -> list[Ranking]:
        # Each component retrieves deeper than k, because a document ranked 30th by one
        # retriever and 2nd by the other should still be fusable into the top k.
        depth = k * self._multiplier
        per_retriever = [retriever.search(queries, depth) for retriever, _ in self._parts]
        weights = [weight for _, weight in self._parts]

        fused: list[Ranking] = []
        for row in range(len(queries)):
            totals: dict[int, float] = {}
            for rankings, weight in zip(per_retriever, weights, strict=True):
                for rank, (unit_index, _score) in enumerate(rankings[row], start=1):
                    totals[unit_index] = totals.get(unit_index, 0.0) + weight / (
                        self._rrf_k + rank
                    )
            # Ties break on unit index so the ordering is deterministic across runs.
            ordered = sorted(totals.items(), key=lambda item: (-item[1], item[0]))
            fused.append([(index, score) for index, score in ordered[:k]])
        return fused
