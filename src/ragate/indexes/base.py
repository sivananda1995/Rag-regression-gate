"""Vector index interface."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class VectorIndex(Protocol):
    name: str

    def build(self, vectors: np.ndarray) -> None:
        """Index the given (n, d) matrix of L2-normalised row vectors."""

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, indices), both shaped (len(queries), k)."""
