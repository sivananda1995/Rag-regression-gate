from __future__ import annotations

import numpy as np
import pytest

from ragate.config import GateConfig
from ragate.errors import BaselineError
from ragate.evaluate import EvalReport, QueryResult
from ragate.gate import FAIL, PASS, WARN, evaluate_gate, load_baseline, paired_bootstrap_ci


def _report(scores: dict[str, float], retrieved: dict[str, list[str]] | None = None) -> EvalReport:
    retrieved = retrieved or {}
    per_query = [
        QueryResult(
            query_id=query_id,
            text=f"query {query_id}",
            relevant_doc_ids=["gold"],
            retrieved_doc_ids=retrieved.get(query_id, ["gold"]),
            scores={"recall_at_k": value},
        )
        for query_id, value in scores.items()
    ]
    mean = sum(scores.values()) / len(scores)
    return EvalReport(
        ragate_version="test",
        generated_at="2026-08-12T00:00:00+00:00",
        k=5,
        aggregate={"recall_at_k": mean},
        by_split={},
        per_query=per_query,
        corpus_stats={"documents": 4, "chunks": 8, "chunks_per_document": 2.0,
                      "queries": len(scores), "mean_relevant_per_query": 1.0,
                      "recall_at_k_ceiling": 1.0},
        timings_ms={"total": 1.0},
        config={"chunking": {"strategy": "fixed", "target_chars": 10, "overlap_chars": 0},
                "embedder": {"provider": "hashing", "dimensions": 8, "idf_weighting": True},
                "index": {"backend": "flat"}},
        fingerprint={"corpus_sha256": "abc123", "queries_sha256": "def456", "k": 5},
    )


def _cfg(**overrides) -> GateConfig:
    cfg = GateConfig(bootstrap_resamples=2000)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_identical_runs_produce_a_zero_width_interval():
    scores = np.array([1.0, 0.0, 1.0, 0.5])
    low, high = paired_bootstrap_ci(scores, scores, 0.95, 500)
    assert (low, high) == (0.0, 0.0)


def test_bootstrap_interval_brackets_the_observed_difference():
    rng = np.random.default_rng(3)
    baseline = rng.random(200)
    candidate = baseline - 0.1 + rng.normal(0.0, 0.03, size=200)
    observed = float((candidate - baseline).mean())
    low, high = paired_bootstrap_ci(baseline, candidate, 0.95, 2000)
    assert low <= observed <= high
    assert high < 0.0


def test_bootstrap_requires_paired_vectors():
    with pytest.raises(ValueError, match="equal-length"):
        paired_bootstrap_ci(np.array([1.0]), np.array([1.0, 1.0]), 0.95, 200)


def test_large_consistent_drop_fails_the_gate():
    baseline = _report({f"q{i}": 1.0 for i in range(40)})
    candidate = _report({f"q{i}": (1.0 if i % 5 == 0 else 0.0) for i in range(40)})
    verdict = evaluate_gate(baseline, candidate, _cfg())
    assert verdict.status == FAIL
    assert verdict.failed
    assert verdict.significant
    assert verdict.ci_high < 0


def test_tiny_noisy_drop_warns_instead_of_failing():
    """Two queries out of forty flip. That is past a 2 point tolerance but a paired
    bootstrap cannot separate it from noise, so blocking the build would be wrong."""
    baseline = _report({f"q{i}": 1.0 if i < 20 else 0.0 for i in range(40)})
    candidate = _report(
        {f"q{i}": (0.0 if i in (0, 1) else (1.0 if i < 20 else 0.0)) for i in range(40)}
    )
    verdict = evaluate_gate(baseline, candidate, _cfg(max_absolute_drop=0.02))
    assert verdict.status == WARN
    assert not verdict.failed
    assert "includes zero" in verdict.reason


def test_measurable_drop_inside_tolerance_passes():
    """Ten queries out of a hundred lose their document. That is a real, measurable
    drop, but the team's agreed tolerance is wider, so the build is not blocked."""
    baseline = _report({f"q{i}": 1.0 for i in range(100)})
    candidate = _report({f"q{i}": (0.0 if i < 10 else 1.0) for i in range(100)})
    verdict = evaluate_gate(baseline, candidate, _cfg(max_absolute_drop=0.15))
    assert verdict.status == PASS
    assert verdict.significant is True
    assert "inside the agreed tolerance" in verdict.reason


def test_single_query_drop_is_not_called_significant():
    """One query out of a hundred is exactly the case a naive threshold gate gets
    wrong: over a third of bootstrap resamples never draw that query, so the
    interval touches zero and the drop is not distinguishable from noise."""
    baseline = _report({f"q{i}": 1.0 for i in range(100)})
    candidate = _report({f"q{i}": (0.0 if i < 1 else 1.0) for i in range(100)})
    verdict = evaluate_gate(baseline, candidate, _cfg(max_absolute_drop=0.005))
    assert verdict.significant is False
    assert verdict.status == WARN


def test_improvement_passes_and_is_listed_as_improved():
    baseline = _report({f"q{i}": 0.0 for i in range(30)})
    candidate = _report({f"q{i}": 1.0 for i in range(30)})
    verdict = evaluate_gate(baseline, candidate, _cfg())
    assert verdict.status == PASS
    assert verdict.delta > 0
    assert len(verdict.improved_queries) == 30
    assert verdict.regressed_queries == []


def test_blame_lists_documents_that_left_the_top_k():
    baseline = _report({"q0": 1.0, "q1": 1.0}, retrieved={"q0": ["gold", "a"], "q1": ["gold"]})
    candidate = _report({"q0": 0.0, "q1": 1.0}, retrieved={"q0": ["a", "b"], "q1": ["gold"]})
    verdict = evaluate_gate(baseline, candidate, _cfg(blame_threshold=0.5))
    assert [d.query_id for d in verdict.regressed_queries] == ["q0"]
    assert verdict.regressed_queries[0].lost_doc_ids == ["gold"]
    assert verdict.regressed_queries[0].gained_doc_ids == ["b"]


def test_mismatched_fingerprint_is_refused():
    baseline = _report({"q0": 1.0})
    candidate = _report({"q0": 1.0})
    candidate.fingerprint = {"corpus_sha256": "abc123", "queries_sha256": "999999", "k": 5}
    with pytest.raises(BaselineError, match="queries_sha256"):
        evaluate_gate(baseline, candidate, _cfg())


def test_disjoint_query_ids_are_refused():
    baseline = _report({"q0": 1.0})
    candidate = _report({"q9": 1.0})
    with pytest.raises(BaselineError, match="no query ids are shared"):
        evaluate_gate(baseline, candidate, _cfg())


def test_unknown_primary_metric_is_refused():
    baseline = _report({"q0": 1.0})
    candidate = _report({"q0": 1.0})
    with pytest.raises(BaselineError, match="not present"):
        evaluate_gate(baseline, candidate, _cfg(primary_metric="vibes_at_k"))


def test_verdict_is_reproducible_for_the_same_inputs():
    baseline = _report({f"q{i}": 1.0 for i in range(30)})
    candidate = _report({f"q{i}": float(i % 2) for i in range(30)})
    first = evaluate_gate(baseline, candidate, _cfg())
    second = evaluate_gate(baseline, candidate, _cfg())
    assert (first.ci_low, first.ci_high) == (second.ci_low, second.ci_high)


def test_missing_baseline_explains_how_to_create_one(tmp_path):
    with pytest.raises(BaselineError, match="ragate baseline"):
        load_baseline(tmp_path / "nope.json")


def test_a_query_that_loses_its_last_correct_document_is_counted_apart_from_the_blame_list():
    """The aggregate hides the difference between a rank slipping and an answer vanishing."""
    baseline = _report({"a": 1.0, "b": 1.0, "c": 0.0, "d": 0.6})
    candidate = _report({"a": 0.0, "b": 0.5, "c": 0.4, "d": 0.6})
    cfg = GateConfig(primary_metric="recall_at_k", max_absolute_drop=0.02,
                     bootstrap_confidence=0.95, bootstrap_resamples=200, blame_threshold=0.5)

    verdict = evaluate_gate(baseline, candidate, cfg)

    assert [d.query_id for d in verdict.blanked_queries] == ["a"]
    assert [d.query_id for d in verdict.recovered_queries] == ["c"]
    assert verdict.net_blanked == 0
    # b lost half a point without going to zero, so it is blame-listed but not blanked.
    assert "b" in [d.query_id for d in verdict.regressed_queries]
    assert "b" not in [d.query_id for d in verdict.blanked_queries]


def test_the_candidates_own_zero_count_is_not_what_the_gate_reports():
    """Three queries score zero on the candidate, but two of them scored zero already.

    This is the arithmetic the README got wrong: quoting the candidate's total charges a
    change for breakage that predates it.
    """
    baseline = _report({"a": 1.0, "b": 0.0, "c": 0.0, "d": 1.0})
    candidate = _report({"a": 0.0, "b": 0.0, "c": 0.0, "d": 1.0})
    cfg = GateConfig(primary_metric="recall_at_k", max_absolute_drop=0.02,
                     bootstrap_confidence=0.95, bootstrap_resamples=200, blame_threshold=0.5)

    verdict = evaluate_gate(baseline, candidate, cfg)

    candidate_zeros = sum(1 for q in candidate.per_query if q.scores["recall_at_k"] == 0.0)
    assert candidate_zeros == 3
    assert len(verdict.blanked_queries) == 1
    assert verdict.net_blanked == 1


def test_blanked_queries_are_ordered_by_id_so_two_runs_agree():
    baseline = _report({f"q{i}": 1.0 for i in range(6)})
    candidate = _report({f"q{i}": 0.0 for i in range(6)})
    cfg = GateConfig(primary_metric="recall_at_k", max_absolute_drop=0.02,
                     bootstrap_confidence=0.95, bootstrap_resamples=200, blame_threshold=0.5)

    verdict = evaluate_gate(baseline, candidate, cfg)

    ids = [d.query_id for d in verdict.blanked_queries]
    assert ids == sorted(ids)
    assert verdict.net_blanked == 6
