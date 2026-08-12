"""Run the golden set through a retrieval pipeline and produce a report.

The report is the unit of currency in this repository: `ragate baseline` stores one
in git, `ragate gate` compares a fresh one against it, and `ragate report` renders
one for humans. It carries per-query scores, not just aggregates, because an
aggregate cannot tell you which query broke.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .config import Config
from .corpus import chunk_documents, load_documents, load_queries
from .embedders import build_embedder
from .errors import RagateError
from .indexes import build_index
from .logging_setup import get_logger
from .metrics import METRICS, dedupe_preserving_rank

log = get_logger(__name__)


@dataclass
class QueryResult:
    query_id: str
    text: str
    relevant_doc_ids: list[str]
    retrieved_doc_ids: list[str]
    scores: dict[str, float]


@dataclass
class EvalReport:
    ragate_version: str
    generated_at: str
    k: int
    aggregate: dict[str, float]
    per_query: list[QueryResult]
    corpus_stats: dict[str, Any]
    timings_ms: dict[str, float]
    config: dict[str, Any]
    fingerprint: dict[str, Any]
    environment: dict[str, str] = field(default_factory=dict)

    def scores_for(self, metric: str) -> dict[str, float]:
        return {q.query_id: q.scores[metric] for q in self.per_query}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=False) + "\n")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "EvalReport":
        raw = json.loads(Path(path).read_text())
        raw["per_query"] = [QueryResult(**q) for q in raw["per_query"]]
        raw.setdefault("environment", {})
        return cls(**raw)


def _ceiling(relevant_counts: list[int], k: int) -> float:
    """The best recall@k any retriever could reach on this golden set.

    A query with more labeled relevant documents than k cannot score 1.0, so the
    theoretical maximum is below 1.0 and reporting the raw score without it invites
    the wrong conclusion that the pipeline is leaving recall on the table.
    """
    return sum(min(count, k) / count for count in relevant_counts) / len(relevant_counts)


def run(cfg: Config) -> EvalReport:
    t0 = time.perf_counter()
    documents = load_documents(cfg.corpus.path)
    queries = load_queries(cfg.corpus.queries, {d.doc_id for d in documents})
    t_load = time.perf_counter()

    chunks = chunk_documents(documents, cfg.chunking)
    t_chunk = time.perf_counter()

    embedder = build_embedder(cfg.embedder)
    chunk_texts = [c.text for c in chunks]
    embedder.fit(chunk_texts)
    chunk_vectors = embedder.encode(chunk_texts)
    t_embed_corpus = time.perf_counter()

    index = build_index(cfg.index)
    index.build(chunk_vectors)
    t_build = time.perf_counter()

    query_vectors = embedder.encode([q.text for q in queries])
    t_embed_queries = time.perf_counter()

    k = cfg.evaluate.k
    # Retrieval is over chunks, so more than k chunks are pulled to leave room for
    # several chunks of the same document collapsing into one document slot.
    chunk_k = min(len(chunks), max(k * 4, k + 10))
    _, indices = index.search(query_vectors, chunk_k)
    t_search = time.perf_counter()

    results: list[QueryResult] = []
    for row, query in enumerate(queries):
        ranked_doc_ids = dedupe_preserving_rank(
            chunks[int(i)].doc_id for i in indices[row] if int(i) >= 0
        )[:k]
        scores = {
            name: float(fn(ranked_doc_ids, query.relevant_doc_ids, k))
            for name, fn in METRICS.items()
        }
        results.append(
            QueryResult(
                query_id=query.query_id,
                text=query.text,
                relevant_doc_ids=list(query.relevant_doc_ids),
                retrieved_doc_ids=ranked_doc_ids,
                scores=scores,
            )
        )

    if not results:
        raise RagateError("golden set produced no results")

    aggregate = {
        name: sum(r.scores[name] for r in results) / len(results) for name in METRICS
    }
    relevant_counts = [len(r.relevant_doc_ids) for r in results]
    report = EvalReport(
        ragate_version=__version__,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        k=k,
        aggregate=aggregate,
        per_query=results,
        corpus_stats={
            "documents": len(documents),
            "chunks": len(chunks),
            "chunks_per_document": round(len(chunks) / len(documents), 3),
            "queries": len(results),
            "mean_relevant_per_query": round(sum(relevant_counts) / len(relevant_counts), 3),
            "recall_at_k_ceiling": round(_ceiling(relevant_counts, k), 4),
        },
        timings_ms={
            "load": round((t_load - t0) * 1000, 1),
            "chunk": round((t_chunk - t_load) * 1000, 1),
            "embed_corpus": round((t_embed_corpus - t_chunk) * 1000, 1),
            "index_build": round((t_build - t_embed_corpus) * 1000, 1),
            "embed_queries": round((t_embed_queries - t_build) * 1000, 1),
            "search": round((t_search - t_embed_queries) * 1000, 1),
            "total": round((time.perf_counter() - t0) * 1000, 1),
        },
        config=cfg.as_dict(),
        fingerprint=cfg.fingerprint(),
        environment={
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    )
    log.info(
        "evaluation complete",
        extra={
            "recall_at_k": round(aggregate["recall_at_k"], 4),
            "ndcg_at_k": round(aggregate["ndcg_at_k"], 4),
            "ceiling": report.corpus_stats["recall_at_k_ceiling"],
            "queries": len(results),
            "total_ms": report.timings_ms["total"],
        },
    )
    return report
