"""Coverage for the wiring that the happy path never touches: provider selection,
the log formatter's contract, and error surfaces of optional dependencies."""

from __future__ import annotations

import json
import logging

import pytest

from ragate.config import EmbedderConfig, IndexConfig
from ragate.embedders import build_embedder
from ragate.errors import EmbedderError, RagateError
from ragate.indexes import build_index
from ragate.logging_setup import JsonFormatter, configure


def test_index_builder_selects_backends():
    assert build_index(IndexConfig(backend="flat")).name == "flat"
    faiss_backend = build_index(IndexConfig(backend="faiss"))
    assert faiss_backend.name == "faiss-hnsw"


def test_index_builder_rejects_unknown_backend():
    cfg = IndexConfig()
    cfg.backend = "annoy"
    with pytest.raises(RagateError, match="unknown index backend"):
        build_index(cfg)


def test_embedder_builder_selects_hashing():
    assert build_embedder(EmbedderConfig(provider="hashing")).name == "hashing"


def test_openai_provider_refuses_to_start_without_a_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(EmbedderError, match="OPENAI_API_KEY is not set"):
        build_embedder(EmbedderConfig(provider="openai"))


def test_json_formatter_emits_one_parseable_line_with_extras():
    record = logging.LogRecord(
        name="ragate.test", level=logging.INFO, pathname=__file__, lineno=1,
        msg="something happened", args=(), exc_info=None,
    )
    record.recall_at_k = 0.86
    payload = json.loads(JsonFormatter().format(record))
    assert payload["msg"] == "something happened"
    assert payload["level"] == "INFO"
    assert payload["recall_at_k"] == 0.86
    assert payload["run_id"]
    assert "\n" not in JsonFormatter().format(record)


def test_json_formatter_includes_the_traceback_when_present():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = logging.LogRecord(
            name="ragate.test", level=logging.ERROR, pathname=__file__, lineno=1,
            msg="failed", args=(), exc_info=sys.exc_info(),
        )
    payload = json.loads(JsonFormatter().format(record))
    assert "ValueError: boom" in payload["exc"]


def test_configure_is_idempotent():
    configure("WARNING", "text")
    configure("WARNING", "json")
    assert len(logging.getLogger().handlers) == 1
