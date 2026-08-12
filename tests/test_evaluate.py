from __future__ import annotations

from pathlib import Path

from ragate.evaluate import EvalReport, run


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
