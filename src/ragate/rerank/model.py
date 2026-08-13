"""The reranker at inference time: a linear model stored as reviewable JSON.

Two decisions worth defending.

Stored as JSON, not a pickle. A pickle is opaque in a pull request and executes
arbitrary code on load, and this artifact is committed to a repository whose whole point
is that changes to retrieval quality are reviewable. Coefficients in JSON mean a reviewer
can see that the weight on "unit_breadth" went more negative and ask why.

Inference in numpy, training in scikit-learn. The runtime dependency stays numpy plus
PyYAML, so adopting the gate does not drag scikit-learn into a CI image; training is a
developer-machine task under tools/.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..corpus import IndexUnit
from ..errors import RagateError
from ..logging_setup import get_logger
from .features import FEATURE_NAMES, FeatureContext, extract_documents

log = get_logger(__name__)


@dataclass
class LinearReranker:
    feature_names: tuple[str, ...]
    coefficients: list[float]
    intercept: float
    mean: list[float]
    scale: list[float]
    trained_at: str = ""
    training_notes: str = ""

    def __post_init__(self) -> None:
        if len(self.coefficients) != len(self.feature_names):
            raise RagateError("reranker has a different number of coefficients and features")
        if tuple(self.feature_names) != FEATURE_NAMES:
            raise RagateError(
                "reranker was trained on a different feature set: "
                f"{self.feature_names} != {FEATURE_NAMES}. Retrain with tools/train_reranker.py"
            )

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["feature_names"] = list(self.feature_names)
        p.write_text(json.dumps(payload, indent=2) + "\n")
        return p

    @classmethod
    def load(cls, path: str | Path) -> LinearReranker:
        p = Path(path)
        if not p.exists():
            raise RagateError(
                f"no reranker model at {p}. Train one with 'python tools/train_reranker.py' "
                "or set rerank.enabled=false"
            )
        raw = json.loads(p.read_text())
        raw["feature_names"] = tuple(raw["feature_names"])
        return cls(**raw)

    def score(self, features: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.mean, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        scale = np.where(scale == 0.0, 1.0, scale)
        standardised = (features - mean) / scale
        return standardised @ np.asarray(self.coefficients, dtype=np.float64) + self.intercept

    def rank_documents(
        self,
        query: str,
        ranking: list[tuple[int, float]],
        units: list[IndexUnit],
        ctx: FeatureContext,
    ) -> list[str]:
        """Return candidate document ids, best first, reordered by model score.

        Ties, including the degenerate case of a model that scores everything equally,
        fall back to retrieval order, so a useless model is a no-op rather than a
        shuffle.
        """
        doc_ids, features = extract_documents(query, ranking, units, ctx)
        if not doc_ids:
            return []
        scores = self.score(features)
        order = sorted(range(len(doc_ids)), key=lambda i: (-float(scores[i]), i))
        return [doc_ids[i] for i in order]
