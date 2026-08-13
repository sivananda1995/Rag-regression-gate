"""Tests for the document-level reranker.

The behaviour these lock down is the one that was wrong the first time: features and
labels have to describe documents, because recall@k counts documents. A chunk-level
reranker scored well in training and lost 6 points of held-out recall, and the tests
below encode the properties that made the difference.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ragate.corpus import IndexUnit
from ragate.errors import RagateError
from ragate.rerank import FEATURE_NAMES, FeatureContext, LinearReranker, extract_documents

UNITS = [
    # Unit 0 belongs to two documents, which is what happens to shared boilerplate.
    IndexUnit(text="Reset multi factor authentication for Atlas VPN. Remove the stale token.",
              doc_ids=("shared-a", "shared-b"), occurrences=2),
    IndexUnit(text="Clear a stuck print queue for Harbor Print on Windows 11. "
                   "Restart the spooler service.",
              doc_ids=("print-win",), occurrences=1),
    IndexUnit(text="Reset multi factor authentication for Atlas VPN on Android 15. "
                   "Remove the stale token and error MFA-4021 stops.",
              doc_ids=("mfa-android",), occurrences=1),
]


@pytest.fixture
def ctx() -> FeatureContext:
    return FeatureContext.build(UNITS)


def _index(name: str) -> int:
    return FEATURE_NAMES.index(name)


def test_context_computes_idf_over_units(ctx):
    # "authentication" appears in two of three units, "spooler" in one, so the rarer
    # term must carry more weight.
    assert ctx.idf["spooler"] > ctx.idf["authentication"]


def test_candidates_expand_to_documents_in_retrieval_order(ctx):
    doc_ids, features = extract_documents("reset mfa", [(0, 1.0), (2, 0.5)], UNITS, ctx)
    # Unit 0 carries two documents, so three candidates come out of two chunks, and the
    # order follows the chunk ranking.
    assert doc_ids == ["shared-a", "shared-b", "mfa-android"]
    assert features.shape == (3, len(FEATURE_NAMES))


def test_documents_are_not_duplicated_when_several_chunks_hit(ctx):
    doc_ids, features = extract_documents(
        "reset mfa", [(0, 1.0), (0, 0.9), (2, 0.5)], UNITS, ctx
    )
    assert doc_ids.count("shared-a") == 1
    # chunks_hit is log1p of the number of hits, so two hits gives log(3).
    assert features[0][_index("chunks_hit")] == pytest.approx(np.log1p(2))


def test_evidence_from_several_chunks_is_aggregated(ctx):
    _doc_ids, features = extract_documents("reset mfa", [(0, 0.4), (0, 0.6)], UNITS, ctx)
    assert features[0][_index("max_fused_score")] == pytest.approx(0.6)
    assert features[0][_index("sum_fused_score")] == pytest.approx(1.0)


def test_best_rank_is_kept_not_the_worst(ctx):
    _doc_ids, features = extract_documents(
        "print queue", [(1, 1.0), (0, 0.5)], UNITS, ctx
    )
    assert features[0][_index("best_reciprocal_rank")] == pytest.approx(1.0)
    assert features[1][_index("best_reciprocal_rank")] == pytest.approx(0.5)


def test_code_match_fires_only_on_an_exact_code(ctx):
    _ids, with_code = extract_documents("MFA-4021 keeps appearing", [(2, 1.0)], UNITS, ctx)
    _ids, wrong_code = extract_documents("STG-8815 keeps appearing", [(2, 1.0)], UNITS, ctx)
    assert with_code[0][_index("code_token_match")] == 1.0
    assert wrong_code[0][_index("code_token_match")] == 0.0


def test_sibling_similarity_to_the_top_hit_is_measured(ctx):
    """The mechanism behind the reranker's gain: unit 2 is a platform variant of unit 0,
    so when unit 0 leads, unit 2's document scores high on lead similarity while an
    unrelated article scores low."""
    _ids, features = extract_documents(
        "reset mfa for atlas vpn", [(0, 1.0), (2, 0.9), (1, 0.8)], UNITS, ctx
    )
    sibling = features[2][_index("top1_lead_jaccard")]
    unrelated = features[3][_index("top1_lead_jaccard")]
    assert sibling > unrelated


def test_no_features_for_an_empty_ranking(ctx):
    doc_ids, features = extract_documents("anything", [], UNITS, ctx)
    assert doc_ids == []
    assert features.shape == (0, len(FEATURE_NAMES))


def _model(**overrides) -> LinearReranker:
    payload = {
        "feature_names": FEATURE_NAMES,
        "coefficients": [0.0] * len(FEATURE_NAMES),
        "intercept": 0.0,
        "mean": [0.0] * len(FEATURE_NAMES),
        "scale": [1.0] * len(FEATURE_NAMES),
    }
    payload.update(overrides)
    return LinearReranker(**payload)


def test_a_zero_model_is_a_no_op(ctx):
    """An untrained or useless model must leave retrieval order alone rather than
    shuffle it, so switching reranking on can never be worse than a tie."""
    ranking = [(0, 1.0), (2, 0.9), (1, 0.8)]
    baseline, _ = extract_documents("reset mfa", ranking, UNITS, ctx)
    assert _model().rank_documents("reset mfa", ranking, UNITS, ctx) == baseline


def test_a_model_can_promote_a_lower_ranked_document(ctx):
    coefficients = [0.0] * len(FEATURE_NAMES)
    coefficients[_index("code_token_match")] = 5.0
    ranked = _model(coefficients=coefficients).rank_documents(
        "MFA-4021 error", [(1, 1.0), (2, 0.5)], UNITS, ctx
    )
    # print-win leads retrieval, but only mfa-android matches the code.
    assert ranked[0] == "mfa-android"


def test_scaling_is_applied_before_the_dot_product():
    model = _model(
        coefficients=[1.0] + [0.0] * (len(FEATURE_NAMES) - 1),
        mean=[2.0] + [0.0] * (len(FEATURE_NAMES) - 1),
        scale=[4.0] + [1.0] * (len(FEATURE_NAMES) - 1),
        intercept=0.5,
    )
    features = np.zeros((1, len(FEATURE_NAMES)))
    features[0][0] = 6.0
    # (6 - 2) / 4 = 1.0, times coefficient 1.0, plus intercept 0.5
    assert float(model.score(features)[0]) == pytest.approx(1.5)


def test_zero_scale_does_not_divide_by_zero():
    model = _model(scale=[0.0] * len(FEATURE_NAMES))
    assert np.isfinite(model.score(np.ones((1, len(FEATURE_NAMES))))).all()


def test_model_round_trips_through_json(tmp_path):
    original = _model(coefficients=[0.5] * len(FEATURE_NAMES), intercept=-1.25,
                      training_notes="unit test")
    path = original.save(tmp_path / "reranker.json")
    restored = LinearReranker.load(path)
    assert restored.coefficients == original.coefficients
    assert restored.intercept == pytest.approx(-1.25)
    assert restored.training_notes == "unit test"
    # Stored as reviewable JSON, never as a pickle.
    assert "coefficients" in json.loads(path.read_text())


def test_a_model_trained_on_other_features_is_refused(tmp_path):
    path = tmp_path / "stale.json"
    path.write_text(json.dumps({
        "feature_names": ["fused_score", "reciprocal_rank"],
        "coefficients": [1.0, 1.0],
        "intercept": 0.0,
        "mean": [0.0, 0.0],
        "scale": [1.0, 1.0],
    }))
    with pytest.raises(RagateError, match="different feature set"):
        LinearReranker.load(path)


def test_coefficient_count_must_match_the_feature_count():
    with pytest.raises(RagateError, match="different number of coefficients"):
        _model(coefficients=[1.0])


def test_missing_model_file_names_the_training_command(tmp_path):
    with pytest.raises(RagateError, match="tools/train_reranker.py"):
        LinearReranker.load(tmp_path / "absent.json")


def test_shipped_model_matches_the_current_feature_set():
    """Guards the exact staleness that a feature-set change would introduce: the model in
    the repository must still describe the features the code computes."""
    model = LinearReranker.load("models/reranker.json")
    assert tuple(model.feature_names) == FEATURE_NAMES
    assert model.trained_at
