"""BM25 Okapi, implemented here rather than imported.

Written from the formula (Robertson and Zaragoza, 2009) because this repository's whole
claim is that its numbers are traceable, and a retriever whose behaviour I cannot open
up is a poor foundation for that. tests/test_bm25.py checks the scores against values
computed by hand on a three-document corpus, so the implementation is pinned to the
formula rather than to itself.

    score(q, d) = sum over terms t in q of
        idf(t) * f(t, d) * (k1 + 1) / (f(t, d) + k1 * (1 - b + b * |d| / avgdl))

    idf(t) = ln(1 + (N - n(t) + 0.5) / (n(t) + 0.5))

k1 controls term-frequency saturation, b controls length normalisation. The defaults
(1.5, 0.75) are the standard starting point and are exposed as config, not constants.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

import numpy as np

from ..corpus import IndexUnit
from ..errors import RagateError
from ..logging_setup import get_logger
from ..ranking import rank_scores
from .base import Ranking

log = get_logger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Bm25Retriever:
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 < 0:
            raise RagateError("bm25 k1 must not be negative")
        if not 0.0 <= b <= 1.0:
            raise RagateError("bm25 b must be between 0 and 1")
        self.k1 = k1
        self.b = b
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._idf: dict[str, float] = {}
        self._lengths: np.ndarray | None = None
        self._avgdl = 0.0
        self._count = 0

    def fit(self, units: Sequence[IndexUnit]) -> None:
        if not units:
            raise RagateError("bm25 needs at least one index unit")
        self._count = len(units)
        lengths = np.zeros(len(units), dtype=np.float32)
        postings: dict[str, list[tuple[int, int]]] = {}
        for position, unit in enumerate(units):
            tokens = tokenize(unit.text)
            lengths[position] = len(tokens)
            for term, frequency in Counter(tokens).items():
                postings.setdefault(term, []).append((position, frequency))
        self._postings = postings
        self._lengths = lengths
        self._avgdl = float(lengths.mean()) if len(lengths) else 0.0
        # Document frequency is the length of a term's posting list, so idf is free.
        self._idf = {
            term: math.log(
                1.0 + (self._count - len(plist) + 0.5) / (len(plist) + 0.5)
            )
            for term, plist in postings.items()
        }
        log.info(
            "bm25 index built",
            extra={"units": self._count, "vocabulary": len(postings),
                   "avg_length": round(self._avgdl, 1), "k1": self.k1, "b": self.b},
        )

    def _score_query(self, query: str, k: int) -> Ranking:
        if self._lengths is None:
            raise RagateError("fit() must be called before search()")
        scores = np.zeros(self._count, dtype=np.float32)
        # Only units that contain a query term can score, so the loop walks posting
        # lists rather than the corpus. On this corpus that touches a few percent of
        # units per query instead of all of them.
        #
        # Terms are sorted rather than taken in set order. Floating-point addition is not
        # associative, so accumulating a unit's term contributions in a different order
        # changes its score in the last bits, and set iteration order for strings depends
        # on the interpreter's per-process hash seed. Sorting fixes the order for good.
        for term in sorted(set(tokenize(query))):
            plist = self._postings.get(term)
            if plist is None:
                continue
            idf = self._idf[term]
            for position, frequency in plist:
                norm = 1.0 - self.b + self.b * (self._lengths[position] / self._avgdl)
                scores[position] += idf * frequency * (self.k1 + 1.0) / (
                    frequency + self.k1 * norm
                )
        return rank_scores(scores, k)

    def search(self, queries: Sequence[str], k: int) -> list[Ranking]:
        return [self._score_query(query, k) for query in queries]
