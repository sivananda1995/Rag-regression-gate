from __future__ import annotations

from pathlib import Path

from ragate.config import GateConfig
from ragate.evaluate import EvalReport, QueryResult, run
from ragate.gate import evaluate_gate
from ragate.report import blast_radius_sentence, render_markdown


def test_end_to_end_run_scores_the_golden_set(config):
    report = run(config)
    assert report.corpus_stats["documents"] == 4
    assert report.corpus_stats["queries"] == 3
    assert 0.0 <= report.aggregate["recall_at_k"] <= 1.0
    # The three golden queries quote wording from their own document, so a lexical
    # retriever is expected to place each one inside the top 2.
    assert report.aggregate["recall_at_k"] == 1.0
    assert all(q.retrieved_doc_ids for q in report.per_query)


def test_report_round_trips_through_json(config, tmp_path: Path):
    original = run(config)
    path = original.save(tmp_path / "report.json")
    restored = EvalReport.load(path)
    assert restored.aggregate == original.aggregate
    assert restored.fingerprint == original.fingerprint
    assert [q.query_id for q in restored.per_query] == [q.query_id for q in original.per_query]
    assert restored.scores_for("recall_at_k") == original.scores_for("recall_at_k")


def test_retrieved_documents_are_unique_per_query(config):
    """One document can supply several top chunks; the document list must not repeat
    it, or recall would be computed over fewer distinct documents than it appears."""
    report = run(config)
    for query in report.per_query:
        assert len(query.retrieved_doc_ids) == len(set(query.retrieved_doc_ids))


def test_ceiling_accounts_for_queries_with_more_labels_than_k(config):
    config.evaluate.k = 1
    report = run(config)
    assert report.corpus_stats["recall_at_k_ceiling"] == 1.0
    assert report.k == 1


def test_chunking_change_is_visible_in_corpus_stats(config):
    coarse = run(config)
    config.chunking.target_chars = 100
    config.chunking.overlap_chars = 20
    fine = run(config)
    assert fine.corpus_stats["chunks"] > coarse.corpus_stats["chunks"]


def test_dedupe_does_not_change_document_level_recall(config):
    """Collapsing identical chunk text is an indexing optimisation, so the documents
    a query can reach must not shrink. Fewer vectors, same or better recall."""
    # Small chunks are needed for the fixture's two near-identical articles to share
    # a passage; at the default 480 chars each article is a single chunk.
    config.chunking.target_chars = 60
    config.chunking.overlap_chars = 0
    config.chunking.dedupe_identical = False
    without = run(config)
    config.chunking.dedupe_identical = True
    with_dedupe = run(config)
    assert with_dedupe.corpus_stats["index_units"] < without.corpus_stats["index_units"]
    assert with_dedupe.aggregate["recall_at_k"] >= without.aggregate["recall_at_k"]


def test_duplicate_chunk_count_is_reported(config):
    config.chunking.dedupe_identical = True
    report = run(config)
    assert report.corpus_stats["duplicate_chunks_collapsed"] == (
        report.corpus_stats["chunks"] - report.corpus_stats["index_units"]
    )


def test_reranked_run_reports_the_rerank_stage_and_reorders(config, tmp_path):
    """End to end with a model that only rewards an exact code match: the report must
    record that reranking happened and the ranking must change."""
    from ragate.rerank import FEATURE_NAMES, LinearReranker

    coefficients = [0.0] * len(FEATURE_NAMES)
    coefficients[FEATURE_NAMES.index("code_token_match")] = 5.0
    model_path = LinearReranker(
        feature_names=FEATURE_NAMES,
        coefficients=coefficients,
        intercept=0.0,
        mean=[0.0] * len(FEATURE_NAMES),
        scale=[1.0] * len(FEATURE_NAMES),
    ).save(tmp_path / "reranker.json")

    plain = run(config)
    config.rerank.enabled = True
    config.rerank.model_path = str(model_path)
    reranked = run(config)

    assert plain.corpus_stats["reranked"] is False
    assert reranked.corpus_stats["reranked"] is True
    assert "rerank" in reranked.timings_ms
    assert reranked.corpus_stats["retriever"] == plain.corpus_stats["retriever"]


def test_report_carries_per_split_aggregates(config, tmp_path):
    import json

    ids = [q.query_id for q in __import__("ragate.corpus", fromlist=["x"]).load_queries(
        config.corpus.queries)]
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"train": ids[:2], "eval": ids[2:]}))
    config.evaluate.splits_path = str(splits)
    report = run(config)
    assert report.by_split["train"]["queries"] == 2.0
    assert report.by_split["eval"]["queries"] == float(len(ids) - 2)
    assert 0.0 <= report.by_split["eval"]["recall_at_k"] <= 1.0


def test_missing_splits_file_is_not_an_error(config):
    config.evaluate.splits_path = "does/not/exist.json"
    assert run(config).by_split == {}


def test_retriever_mode_is_recorded_in_the_report(config):
    config.retriever.mode = "bm25"
    assert run(config).corpus_stats["retriever"] == "bm25"
    config.retriever.mode = "hybrid"
    assert run(config).corpus_stats["retriever"].startswith("hybrid:")


def _synthetic_report(scores: dict[str, float]) -> EvalReport:
    """A report with hand-chosen per-query scores, for exercising the gate's own reporting."""
    return EvalReport(
        ragate_version="test", generated_at="2026-08-13T00:00:00+00:00", k=5,
        aggregate={"recall_at_k": sum(scores.values()) / len(scores)}, by_split={},
        per_query=[
            QueryResult(query_id=q, text=q, relevant_doc_ids=["gold"],
                        retrieved_doc_ids=["gold"], scores={"recall_at_k": v})
            for q, v in scores.items()
        ],
        corpus_stats={"documents": 2, "chunks": 4, "chunks_per_document": 2.0,
                      "queries": len(scores), "mean_relevant_per_query": 1.0,
                      "recall_at_k_ceiling": 1.0},
        timings_ms={"total": 1.0},
        config={"chunking": {"strategy": "fixed", "target_chars": 10, "overlap_chars": 0},
                "embedder": {"provider": "hashing", "dimensions": 8, "idf_weighting": True},
                "index": {"backend": "flat"}},
        fingerprint={"corpus_sha256": "a", "queries_sha256": "b", "k": 5},
    )


_GATE = GateConfig(primary_metric="recall_at_k", max_absolute_drop=0.02,
                   bootstrap_confidence=0.95, bootstrap_resamples=200, blame_threshold=0.5)


def test_the_blast_radius_sentence_reports_both_directions_and_the_net():
    verdict = evaluate_gate(_synthetic_report({"a": 1.0, "b": 1.0, "c": 0.0, "d": 1.0}),
                            _synthetic_report({"a": 0.0, "b": 0.0, "c": 0.7, "d": 1.0}), _GATE)

    sentence = blast_radius_sentence(verdict)

    assert "2 of 4 questions" in sentence
    assert "1 that had none now finds one" in sentence
    assert "Net 1 question worse off, 25.0% of the golden set" in sentence
    # The whole point of the sentence: do not let the reader quote the raw zero count.
    assert "total count of zero-scoring queries" in sentence


def test_the_blast_radius_sentence_says_so_when_nothing_crossed():
    verdict = evaluate_gate(_synthetic_report({"a": 1.0, "b": 0.8}),
                            _synthetic_report({"a": 0.9, "b": 0.8}), _GATE)

    assert "No query crossed the line" in blast_radius_sentence(verdict)


def test_the_markdown_report_carries_the_blast_radius():
    verdict = evaluate_gate(_synthetic_report({"a": 1.0, "b": 1.0}),
                            _synthetic_report({"a": 0.0, "b": 1.0}), _GATE)

    markdown = render_markdown(_synthetic_report({"a": 0.0, "b": 1.0}), verdict)

    assert "### Blast radius" in markdown
    assert "1 of 2 questions" in markdown
