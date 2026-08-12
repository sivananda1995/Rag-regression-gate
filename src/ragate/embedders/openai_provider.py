"""OpenAI embeddings adapter.

Not exercised by the test suite or the benchmarks in this repository: the build
environment has no outbound access to api.openai.com, and inventing latency or cost
numbers for a call that never happened would make every other number here suspect.
The adapter is real code with real batching and retry behaviour; point
RAGATE_EMBEDDER_PROVIDER=openai at a key to use it, and the gate works identically
because the evaluation loop only knows the Embedder protocol.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence

import numpy as np

from ..errors import EmbedderError
from ..logging_setup import get_logger

log = get_logger(__name__)

_BATCH = 128
_MAX_ATTEMPTS = 5


class OpenAIEmbedder:
    name = "openai"

    def __init__(self, model: str = "text-embedding-3-small", dimensions: int = 1536):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EmbedderError(
                "OPENAI_API_KEY is not set; set it or use RAGATE_EMBEDDER_PROVIDER=hashing"
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise EmbedderError("install the openai extra: pip install '.[openai]'") from exc
        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.dimensions = dimensions

    def fit(self, corpus: Sequence[str]) -> None:
        """No corpus statistics are needed for a hosted model."""

    def _embed_batch(self, batch: Sequence[str]) -> list[list[float]]:
        delay = 0.5
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = self._client.embeddings.create(
                    model=self.model, input=list(batch), dimensions=self.dimensions
                )
                return [item.embedding for item in response.data]
            except Exception as exc:  # provider exceptions vary by SDK version
                if attempt == _MAX_ATTEMPTS:
                    raise EmbedderError(
                        f"embedding request failed after {attempt} attempts: {exc}"
                    ) from exc
                log.warning(
                    "embedding request failed, backing off",
                    extra={"attempt": attempt, "sleep_s": delay, "error": str(exc)},
                )
                time.sleep(delay)
                delay *= 2
        raise EmbedderError("unreachable")

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _BATCH):
            vectors.extend(self._embed_batch(texts[start : start + _BATCH]))
        array = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return array / norms
