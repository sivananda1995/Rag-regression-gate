"""Embedder interface.

The gate's job is to detect a change in retrieval quality, so the provider must be
swappable without touching the evaluation loop. Anything implementing this protocol
can be evaluated: a local deterministic embedder, a hosted API, or a fine-tuned
model served in-house.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    name: str
    dimensions: int

    def fit(self, corpus: Sequence[str]) -> None:
        """Learn any corpus-level statistics (for example inverse document frequency).

        Providers that need no fitting implement this as a no-op.
        """

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return an L2-normalised float32 array of shape (len(texts), dimensions)."""
