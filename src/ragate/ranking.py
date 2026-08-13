"""One deterministic ranking rule, shared by every component that orders things.

This module exists because of a specific failure, and it is worth stating plainly since the
whole repository claims its numbers are measured. CI here re-measures every number in the
README on every push. It disagreed with the development machine on three of them: the
fixed-chunking candidate profile scored 0.8826 on a GitHub runner and 0.8794 locally, on
identical code, identical data, and the same committed model.

The cause was ties, not arithmetic. On this corpus 24 of the 140 golden queries have an
**exact** tie at the k=5 boundary, because templated documents produce equal BM25 scores.
np.argpartition decided which of the tied units landed inside the cut, and that choice is
unspecified: it depends on the numpy build, so it depends on the machine. A gate whose
verdict depends on the machine it runs on is not a gate.

The rule below removes the ambiguity in two steps:

  1. **Quantise.** Scores are rounded to RANK_PRECISION decimals before comparison, so
     last-bit differences from a different BLAS, a different numpy, or a different summation
     order cannot reorder anything. The rounding is orders of magnitude coarser than float
     noise and orders of magnitude finer than any score difference that carries meaning
     (BM25 term contributions here are around 0.1), so it discards nothing informative.
  2. **Break ties explicitly.** The index is a secondary sort key, so genuinely equal scores
     always resolve the same way, on every machine and in every numpy version.

tests/test_ranking.py asserts the property that was missing: perturbing every score by a
relative 1e-12, which is the scale of cross-machine arithmetic difference, must not change
the ranking.
"""

from __future__ import annotations

import numpy as np

from .errors import RagateError

# Nine decimal places. Chosen to sit well below any meaningful score difference and well
# above float32 noise, which is around 1e-7 relative.
RANK_PRECISION = 9

# One query's results: (index, score), best first.
Ranking = list[tuple[int, float]]


def quantise(scores: np.ndarray) -> np.ndarray:
    """Round scores to the comparison precision, in float64 to avoid a second rounding."""
    return np.round(np.asarray(scores, dtype=np.float64), RANK_PRECISION)


def rank_scores(scores: np.ndarray, k: int) -> Ranking:
    """Top k of a 1d score array, ordered by descending score then ascending index."""
    if k <= 0:
        raise RagateError("k must be positive")
    count = int(np.asarray(scores).shape[0])
    top = min(k, count)
    # lexsort takes the primary key last: descending score, then ascending index.
    order = np.lexsort((np.arange(count), -quantise(scores)))[:top]
    return [(int(position), float(scores[position])) for position in order]


def rank_rows(scores: np.ndarray, k: int) -> np.ndarray:
    """Top k indices for every row of a 2d score matrix, under the same total order."""
    if k <= 0:
        raise RagateError("k must be positive")
    rows, count = scores.shape
    top = min(k, count)
    quantised = quantise(scores)
    columns = np.arange(count)
    indices = np.empty((rows, top), dtype=np.int64)
    for row in range(rows):
        indices[row] = np.lexsort((columns, -quantised[row]))[:top]
    return indices
