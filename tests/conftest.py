from __future__ import annotations

import json
from pathlib import Path

import pytest

DOCS = [
    {"doc_id": "d1", "title": "Reset multi-factor authentication for Atlas VPN on iOS 18",
     "text": "The authenticator never receives a challenge. Confirm the enrollment record. "
             "Remove the stale token. Ask the user to sign in again from iOS 18."},
    {"doc_id": "d2", "title": "Reset multi-factor authentication for Atlas VPN on Android 15",
     "text": "The authenticator never receives a challenge. Confirm the enrollment record. "
             "Remove the stale token. Ask the user to sign in again from Android 15."},
    {"doc_id": "d3", "title": "Clear a stuck print queue for Harbor Print on Windows 11",
     "text": "Jobs stay queued and cannot be cleared. Restart the spooler service. "
             "Check for a driver mismatch. Release the held job."},
    {"doc_id": "d4", "title": "Rotate an expiring service credential for Granite Vault",
     "text": "An integration fails authentication after rotation. Inspect the secret version. "
             "Extend the grace period. Use a dual-write window during the change."},
]

QUERIES = [
    {"query_id": "q1", "text": "atlas vpn authenticator challenge never arrives on iOS 18",
     "relevant_doc_ids": ["d1"]},
    {"query_id": "q2", "text": "harbor print jobs stuck in the queue spooler",
     "relevant_doc_ids": ["d3"]},
    {"query_id": "q3", "text": "granite vault secret version rotation grace period",
     "relevant_doc_ids": ["d4"]},
]


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


@pytest.fixture
def corpus_files(tmp_path: Path) -> tuple[Path, Path]:
    return (
        _write_jsonl(tmp_path / "corpus.jsonl", DOCS),
        _write_jsonl(tmp_path / "queries.jsonl", QUERIES),
    )


@pytest.fixture
def config(corpus_files):
    from ragate.config import Config

    corpus, queries = corpus_files
    cfg = Config()
    cfg.corpus.path = str(corpus)
    cfg.corpus.queries = str(queries)
    cfg.evaluate.k = 2
    cfg.evaluate.splits_path = ""
    cfg.embedder.dimensions = 256
    cfg.gate.bootstrap_resamples = 500
    cfg.logging.level = "WARNING"
    # Unit tests must not depend on a trained model living in the repository root, and
    # the reranker has its own tests that build a model explicitly.
    cfg.rerank.enabled = False
    return cfg.validate()


@pytest.fixture
def write_jsonl():
    return _write_jsonl


# A three-document corpus is enough to test wiring, but not to test the gate: with
# three golden queries the paired bootstrap can never reach significance, which is
# the gate behaving correctly. Gate-level CLI tests need a golden set with enough
# queries for a real drop to be separable from noise.
def _bulk(count: int = 60) -> tuple[list[dict], list[dict]]:
    platforms = ["Windows 11", "macOS 15", "iOS 18", "Android 15"]
    docs, queries = [], []
    for i in range(count):
        platform = platforms[i % len(platforms)]
        code = f"ERR-{4000 + i}"
        component = f"quantile{i}"
        docs.append(
            {
                "doc_id": f"b{i:03d}",
                "title": f"Resolve {component} failures for Service {i} on {platform}",
                "text": (
                    f"This article applies to Service {i} on {platform}. Operators report "
                    f"that the {component} handler stops acknowledging work and the log "
                    f"records error {code}. Confirm the {component} lease first, because a "
                    f"stale lease left by the previous release is the usual cause. Open the "
                    f"console, select the affected tenant, and review the {component} audit "
                    f"entry for the last day. If it is absent the request never reached the "
                    f"{component} service. Apply the fix by draining the tenant, deleting the "
                    f"stale lease, then replaying the queue so a fresh {component} record is "
                    f"written. Verify that error {code} has stopped appearing."
                ),
            }
        )
        queries.append(
            {
                "query_id": f"bq{i:03d}",
                "text": f"error {code} stale {component} lease on {platform} service {i}",
                "relevant_doc_ids": [f"b{i:03d}"],
            }
        )
    return docs, queries


@pytest.fixture
def bulk_corpus_files(tmp_path: Path) -> tuple[Path, Path]:
    docs, queries = _bulk()
    return (
        _write_jsonl(tmp_path / "bulk_corpus.jsonl", docs),
        _write_jsonl(tmp_path / "bulk_queries.jsonl", queries),
    )
