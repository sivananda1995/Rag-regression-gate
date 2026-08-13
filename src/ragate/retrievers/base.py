"""Retriever interface.

A retriever turns query strings into ranked index units. Everything above it (metrics,
the gate) works on document ids, and everything below it (embedding models, inverted
indexes, fusion) is a retriever's private business. That boundary is what lets the same
golden set score a sparse retriever, a dense one, and a fusion of both without the
evaluation loop knowing which is which.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ..corpus import IndexUnit

# One query's results: (unit index, score), best first.
Ranking = list[tuple[int, float]]


class Retriever(Protocol):
    name: str

    def fit(self, units: Sequence[IndexUnit]) -> None:
        """Build whatever structure this retriever needs over the index units."""

    def search(self, queries: Sequence[str], k: int) -> list[Ranking]:
        """Return one ranking of at most k units per query."""
