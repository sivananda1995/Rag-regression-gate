"""The gate: decide whether a candidate run is a regression against the baseline.

A single threshold on the aggregate metric is not enough. Golden sets are small, so
the aggregate moves by a point or two between two runs that differ in nothing that
matters. A gate that fires on that noise gets switched off by the team within a week,
which is worse than having no gate.

So two conditions must both hold before the build fails:
  1. the drop exceeds the tolerance the team agreed on (an engineering decision), and
  2. the drop is larger than this golden set's own noise, established by a paired
     bootstrap over per-query scores (a statistical decision).

The pairing matters: the same queries are scored in both runs, so resampling query
ids and taking the difference of the two means removes between-query variance, which
is the dominant term. An unpaired test on 140 queries would be far less sensitive.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import GateConfig
from .errors import BaselineError
from .evaluate import EvalReport
from .logging_setup import get_logger

log = get_logger(__name__)

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"


@dataclass
class QueryDelta:
    query_id: str
    text: str
    baseline: float
    candidate: float
    delta: float
    lost_doc_ids: list[str]
    gained_doc_ids: list[str]


@dataclass
class GateVerdict:
    status: str
    metric: str
    baseline: float
    candidate: float
    delta: float
    tolerance: float
    confidence: float
    ci_low: float
    ci_high: float
    significant: bool
    reason: str
    queries_compared: int
    regressed_queries: list[QueryDelta]
    improved_queries: list[QueryDelta]

    @property
    def failed(self) -> bool:
        return self.status == FAIL

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def paired_bootstrap_ci(
    baseline: np.ndarray,
    candidate: np.ndarray,
    confidence: float,
    resamples: int,
    seed: int = 20260812,
) -> tuple[float, float]:
    """Percentile confidence interval for mean(candidate) - mean(baseline).

    Seeded on purpose: a gate that returns a different verdict on a re-run of the
    same two reports is not a gate. Callers that want to inspect stability can vary
    the seed explicitly.
    """
    if baseline.shape != candidate.shape:
        raise ValueError("paired bootstrap requires equal-length score vectors")
    if baseline.size == 0:
        raise ValueError("paired bootstrap requires at least one query")
    rng = np.random.default_rng(seed)
    differences = candidate - baseline
    picks = rng.integers(0, differences.size, size=(resamples, differences.size))
    means = differences[picks].mean(axis=1)
    alpha = 1.0 - confidence
    low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def _deltas(
    baseline_report: EvalReport, candidate_report: EvalReport, metric: str
) -> list[QueryDelta]:
    baseline_by_id = {q.query_id: q for q in baseline_report.per_query}
    out: list[QueryDelta] = []
    for candidate_query in candidate_report.per_query:
        baseline_query = baseline_by_id.get(candidate_query.query_id)
        if baseline_query is None:
            continue
        before = set(baseline_query.retrieved_doc_ids)
        after = set(candidate_query.retrieved_doc_ids)
        out.append(
            QueryDelta(
                query_id=candidate_query.query_id,
                text=candidate_query.text,
                baseline=baseline_query.scores[metric],
                candidate=candidate_query.scores[metric],
                delta=candidate_query.scores[metric] - baseline_query.scores[metric],
                lost_doc_ids=sorted(before - after),
                gained_doc_ids=sorted(after - before),
            )
        )
    return out


def evaluate_gate(
    baseline_report: EvalReport, candidate_report: EvalReport, cfg: GateConfig
) -> GateVerdict:
    cfg.validate()
    metric = cfg.primary_metric
    if metric not in candidate_report.aggregate:
        raise BaselineError(f"metric {metric!r} is not present in the candidate report")

    if baseline_report.fingerprint != candidate_report.fingerprint:
        # Comparing runs over different corpora or different k is meaningless, and
        # silently allowing it is how a gate ends up reporting a fake regression the
        # morning after someone adds golden queries.
        differing = sorted(
            key
            for key in set(baseline_report.fingerprint) | set(candidate_report.fingerprint)
            if baseline_report.fingerprint.get(key) != candidate_report.fingerprint.get(key)
        )
        raise BaselineError(
            "baseline and candidate are not comparable, these fingerprint fields differ: "
            f"{', '.join(differing)}. baseline={baseline_report.fingerprint} "
            f"candidate={candidate_report.fingerprint}. The corpus, the golden set, or k "
            "changed, so the two metrics are not measuring the same thing. Re-record the "
            "baseline with 'ragate baseline' and let the diff be reviewed."
        )

    deltas = _deltas(baseline_report, candidate_report, metric)
    if not deltas:
        raise BaselineError("no query ids are shared between baseline and candidate reports")

    baseline_scores = np.array([d.baseline for d in deltas], dtype=np.float64)
    candidate_scores = np.array([d.candidate for d in deltas], dtype=np.float64)
    baseline_mean = float(baseline_scores.mean())
    candidate_mean = float(candidate_scores.mean())
    delta = candidate_mean - baseline_mean

    ci_low, ci_high = paired_bootstrap_ci(
        baseline_scores, candidate_scores, cfg.bootstrap_confidence, cfg.bootstrap_resamples
    )
    # A drop is distinguishable from noise when the whole interval sits below zero.
    significant = ci_high < 0.0
    breached = delta <= -cfg.max_absolute_drop

    if breached and significant:
        status = FAIL
        reason = (
            f"{metric} fell {abs(delta):.4f} (tolerance {cfg.max_absolute_drop:.4f}) and the "
            f"{int(cfg.bootstrap_confidence * 100)}% interval "
            f"[{ci_low:.4f}, {ci_high:.4f}] excludes zero"
        )
    elif breached and not significant:
        status = WARN
        reason = (
            f"{metric} fell {abs(delta):.4f}, past the tolerance, but the "
            f"{int(cfg.bootstrap_confidence * 100)}% interval [{ci_low:.4f}, {ci_high:.4f}] "
            "includes zero, so this golden set cannot separate it from run-to-run noise. "
            "Add golden queries before tightening the tolerance"
        )
    elif significant and delta < 0:
        status = PASS
        reason = (
            f"{metric} fell {abs(delta):.4f}, which is measurable but inside the agreed "
            f"tolerance of {cfg.max_absolute_drop:.4f}"
        )
    else:
        status = PASS
        reason = f"{metric} moved {delta:+.4f}, no regression beyond tolerance"

    regressed = sorted(
        [d for d in deltas if d.delta <= -cfg.blame_threshold], key=lambda d: d.delta
    )
    improved = sorted(
        [d for d in deltas if d.delta >= cfg.blame_threshold], key=lambda d: -d.delta
    )

    verdict = GateVerdict(
        status=status,
        metric=metric,
        baseline=baseline_mean,
        candidate=candidate_mean,
        delta=delta,
        tolerance=cfg.max_absolute_drop,
        confidence=cfg.bootstrap_confidence,
        ci_low=ci_low,
        ci_high=ci_high,
        significant=significant,
        reason=reason,
        queries_compared=len(deltas),
        regressed_queries=regressed,
        improved_queries=improved,
    )
    log.info(
        "gate verdict",
        extra={
            "status": status,
            "metric": metric,
            "baseline": round(baseline_mean, 4),
            "candidate": round(candidate_mean, 4),
            "delta": round(delta, 4),
            "ci": [round(ci_low, 4), round(ci_high, 4)],
            "regressed_queries": len(regressed),
        },
    )
    return verdict


def load_baseline(path: str | Path) -> EvalReport:
    p = Path(path)
    if not p.exists():
        raise BaselineError(
            f"no baseline at {p}. Record one on a known-good commit with "
            "'ragate baseline' and commit it, so the gate has something to compare against."
        )
    return EvalReport.load(p)
