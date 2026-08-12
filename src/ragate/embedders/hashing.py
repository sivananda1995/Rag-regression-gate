"""Deterministic local embedder built on the hashing trick.

Why this exists: the harness has to produce the same vectors on a laptop and in CI,
with no network call and no model download, or the gate's own results become a
moving target. A hashed lexical embedder is reproducible to the bit, costs nothing,
and is sensitive to exactly the pipeline changes the gate is built to catch
(chunking, normalisation, weighting).

What it is not: a semantic model. It cannot match a paraphrase that shares no
surface form. The remote providers in this package are for that; this one is the
default so the repository is runnable and testable offline.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence

import numpy as np

from ..errors import EmbedderError
from ..logging_setup import get_logger

log = get_logger(__name__)

_WORD = re.compile(r"[a-z0-9]+")
_SPACE = re.compile(r"\s+")


class HashingEmbedder:
    """Feature-hashed bag of word unigrams and character n-grams.

    Python's built-in hash() is salted per process, which would make vectors differ
    between runs of the same code. blake2b is used instead so a baseline captured
    today is comparable to a run tomorrow.
    """

    name = "hashing"

    def __init__(self, dimensions: int = 512, char_ngram: int = 4, idf_weighting: bool = True):
        if dimensions < 16:
            raise EmbedderError("dimensions must be at least 16")
        if char_ngram < 2:
            raise EmbedderError("char_ngram must be at least 2")
        self.dimensions = dimensions
        self.char_ngram = char_ngram
        self.idf_weighting = idf_weighting
        self._idf: dict[str, float] = {}
        self._default_idf = 1.0
        self._bucket_cache: dict[str, tuple[int, float]] = {}
        self._fitted = False

    # ---------------------------------------------------------------- features
    def _terms(self, text: str) -> list[str]:
        lowered = _SPACE.sub(" ", text.lower()).strip()
        terms = _WORD.findall(lowered)
        n = self.char_ngram
        if len(lowered) >= n:
            terms.extend(lowered[i : i + n] for i in range(len(lowered) - n + 1))
        return terms

    def _bucket(self, term: str) -> tuple[int, float]:
        """Map a term to (bucket index, sign).

        The sign halves the collision bias: two colliding terms with opposite signs
        partially cancel instead of always reinforcing.
        """
        cached = self._bucket_cache.get(term)
        if cached is None:
            digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            cached = (value % self.dimensions, 1.0 if (value >> 63) & 1 else -1.0)
            self._bucket_cache[term] = cached
        return cached

    # ------------------------------------------------------------------- fit
    def fit(self, corpus: Sequence[str]) -> None:
        self._fitted = True
        if not self.idf_weighting:
            self._idf = {}
            log.info("embedder fitted", extra={"embedder": self.name, "idf_weighting": False})
            return
        if not corpus:
            raise EmbedderError("cannot fit idf weights on an empty corpus")
        document_frequency: Counter[str] = Counter()
        for text in corpus:
            document_frequency.update(set(self._terms(text)))
        total = len(corpus)
        # Smoothed idf, always positive so a term never contributes negative weight.
        self._idf = {
            term: math.log((1.0 + total) / (1.0 + df)) + 1.0
            for term, df in document_frequency.items()
        }
        self._default_idf = math.log(1.0 + total) + 1.0
        log.info(
            "embedder fitted",
            extra={
                "embedder": self.name,
                "idf_weighting": True,
                "vocabulary": len(self._idf),
                "documents": total,
            },
        )

    # ---------------------------------------------------------------- encode
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self.idf_weighting and not self._fitted:
            raise EmbedderError("fit() must be called before encode() when idf_weighting is on")
        out = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            counts = Counter(self._terms(text))
            if not counts:
                continue
            for term, count in counts.items():
                bucket, sign = self._bucket(term)
                weight = 1.0 + math.log(count)
                if self.idf_weighting:
                    weight *= self._idf.get(term, self._default_idf)
                out[row, bucket] += sign * weight
            norm = float(np.linalg.norm(out[row]))
            if norm > 0.0:
                out[row] /= norm
        return out
