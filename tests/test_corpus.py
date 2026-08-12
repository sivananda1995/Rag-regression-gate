from __future__ import annotations

import pytest

from ragate.config import ChunkingConfig
from ragate.corpus import (
    build_index_units,
    chunk_documents,
    collapse_duplicate_chunks,
    load_documents,
    load_queries,
)
from ragate.errors import ConfigError, CorpusError


def test_load_documents_and_queries(corpus_files):
    corpus, queries = corpus_files
    docs = load_documents(corpus)
    assert [d.doc_id for d in docs] == ["d1", "d2", "d3", "d4"]
    loaded = load_queries(queries, {d.doc_id for d in docs})
    assert len(loaded) == 3
    assert loaded[0].relevant_doc_ids == ("d1",)


def test_label_pointing_at_missing_document_is_rejected(tmp_path, write_jsonl, corpus_files):
    corpus, _ = corpus_files
    bad = write_jsonl(
        tmp_path / "bad_queries.jsonl",
        [{"query_id": "q9", "text": "anything", "relevant_doc_ids": ["does-not-exist"]}],
    )
    with pytest.raises(CorpusError, match="absent from the corpus"):
        load_queries(bad, {d.doc_id for d in load_documents(corpus)})


def test_duplicate_doc_id_is_rejected(tmp_path, write_jsonl):
    path = write_jsonl(
        tmp_path / "dupes.jsonl",
        [
            {"doc_id": "same", "title": "a", "text": "one"},
            {"doc_id": "same", "title": "b", "text": "two"},
        ],
    )
    with pytest.raises(CorpusError, match="duplicate doc_id"):
        load_documents(path)


def test_malformed_json_names_the_line(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"doc_id": "a", "title": "t", "text": "x"}\nnot json\n')
    with pytest.raises(CorpusError, match=":2"):
        load_documents(path)


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(CorpusError, match="file not found"):
        load_documents(tmp_path / "nope.jsonl")


def test_sentence_window_respects_target_size(corpus_files):
    corpus, _ = corpus_files
    docs = load_documents(corpus)
    cfg = ChunkingConfig(strategy="sentence_window", target_chars=90, overlap_chars=30)
    chunks = chunk_documents(docs, cfg)
    assert len(chunks) > len(docs)
    assert all(chunk.text.strip() for chunk in chunks)
    # Whole sentences are kept, so a chunk may exceed the target by one sentence,
    # but it must never be wildly larger.
    assert max(len(chunk.text) for chunk in chunks) < 90 * 3


def test_chunk_ids_carry_their_document(corpus_files):
    corpus, _ = corpus_files
    chunks = chunk_documents(
        load_documents(corpus), ChunkingConfig(target_chars=80, overlap_chars=20)
    )
    assert all(chunk.chunk_id.startswith(chunk.doc_id + "#") for chunk in chunks)


def test_fixed_chunker_covers_the_whole_document(corpus_files):
    corpus, _ = corpus_files
    docs = load_documents(corpus)
    chunks = chunk_documents(
        docs, ChunkingConfig(strategy="fixed", target_chars=60, overlap_chars=0)
    )
    rebuilt = "".join(c.text for c in chunks if c.doc_id == "d1")
    assert rebuilt == docs[0].full_text


def test_overlap_at_or_above_target_is_rejected():
    with pytest.raises(ConfigError, match="overlap_chars must be smaller"):
        ChunkingConfig(target_chars=100, overlap_chars=100).validate()


def test_identical_chunk_text_is_indexed_once_with_every_owner(corpus_files):
    """d1 and d2 differ only in platform, so their middle sentences are identical.
    One vector must be indexed for that text, carrying both document ids."""
    corpus, _ = corpus_files
    docs = load_documents(corpus)
    chunks = chunk_documents(docs, ChunkingConfig(target_chars=60, overlap_chars=0))
    units = collapse_duplicate_chunks(chunks)
    assert len(units) < len(chunks)
    shared = [u for u in units if len(u.doc_ids) > 1]
    assert shared, "expected at least one passage shared between d1 and d2"
    assert all(u.occurrences == len(u.doc_ids) for u in units)
    assert sum(u.occurrences for u in units) == len(chunks)


def test_collapse_preserves_first_seen_order(corpus_files):
    corpus, _ = corpus_files
    chunks = chunk_documents(
        load_documents(corpus), ChunkingConfig(target_chars=60, overlap_chars=0)
    )
    units = collapse_duplicate_chunks(chunks)
    assert units[0].text == chunks[0].text


def test_build_index_units_can_be_switched_off(corpus_files):
    corpus, _ = corpus_files
    chunks = chunk_documents(
        load_documents(corpus), ChunkingConfig(target_chars=60, overlap_chars=0)
    )
    kept = build_index_units(chunks, dedupe=False)
    collapsed = build_index_units(chunks, dedupe=True)
    assert len(kept) == len(chunks)
    assert all(len(u.doc_ids) == 1 for u in kept)
    assert len(collapsed) < len(kept)
