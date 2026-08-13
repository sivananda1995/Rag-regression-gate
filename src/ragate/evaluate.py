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
from .corpus import build_index_units, chunk_documents, load_documents, load_queries
from .errors import RagateError
from .logging_setup import get_logger
from .metrics import METRICS, dedupe_preserving_rank
from .rerank import FeatureContext, LinearReranker
from .retrievers import build_retriever

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
    by_split: dict[str, dict[str, float]]
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
    def load(cls, path: str | Path) -> EvalReport:
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


def _split_aggregates(
    results: list[QueryResult], splits_path: str
) -> dict[str, dict[str, float]]:
    """Per-split metric means, so a gain can be quoted on queries nothing was fitted on.

    A missing splits file is not an error: the split is an optional discipline, and a
    consumer pointing the gate at their own golden set has no reason to have one.
    """
    p = Path(splits_path) if splits_path else None
    if p is None or not p.exists():
        return {}
    payload = json.loads(p.read_text())
    by_id = {r.query_id: r for r in results}
    out: dict[str, dict[str, float]] = {}
    for split_name in ("train", "eval"):
        ids = [qid for qid in payload.get(split_name, []) if qid in by_id]
        if not ids:
            continue
        out[split_name] = {
            "queries": float(len(ids)),
            **{
                name: sum(by_id[qid].scores[name] for qid in ids) / len(ids)
                for name in METRICS
            },
        }
    return out


def run(cfg: Config) -> EvalReport:
    t0 = time.perf_counter()
    documents = load_documents(cfg.corpus.path)
    queries = load_queries(cfg.corpus.queries, {d.doc_id for d in documents})
    t_load = time.perf_counter()

    chunks = chunk_documents(documents, cfg.chunking)
    units = build_index_units(chunks, cfg.chunking.dedupe_identical)
    t_chunk = time.perf_counter()

    retriever = build_retriever(cfg)
    retriever.fit(units)
    t_build = time.perf_counter()

    k = cfg.evaluate.k
    # Retrieval runs over chunks, so more than k units are pulled: several units of the
    # same document collapse into one document slot, and the reranker needs candidates
    # below the cut to be able to promote anything.
    unit_k = min(len(units), max(k * 4, k + 10, cfg.rerank.depth if cfg.rerank.enabled else 0))
    rankings = retriever.search([q.text for q in queries], unit_k)
    t_search = time.perf_counter()

    reranker = None
    feature_ctx = None
    if cfg.rerank.enabled:
        reranker = LinearReranker.load(cfg.rerank.model_path)
        feature_ctx = FeatureContext.build(units)
    t_rerank_start = time.perf_counter()

    document_rankings: list[list[str]] = []
    for row in range(len(queries)):
        if reranker is not None and feature_ctx is not None:
            document_rankings.append(
                reranker.rank_documents(
                    queries[row].text,
                    rankings[row][: cfg.rerank.depth],
                    units,
                    feature_ctx,
                )
            )
        else:
            document_rankings.append(
                dedupe_preserving_rank(
                    doc_id
                    for unit_index, _score in rankings[row]
                    for doc_id in units[unit_index].doc_ids
                )
            )
    t_rerank = time.perf_counter()

    results: list[QueryResult] = []
    for row, query in enumerate(queries):
        ranked_doc_ids = document_rankings[row][:k]
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
    by_split = _split_aggregates(results, cfg.evaluate.splits_path)
    relevant_counts = [len(r.relevant_doc_ids) for r in results]
    report = EvalReport(
        ragate_version=__version__,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        k=k,
        aggregate=aggregate,
        by_split=by_split,
        per_query=results,
        corpus_stats={
            "retriever": retriever.name,
            "reranked": bool(reranker),
            "documents": len(documents),
            "chunks": len(chunks),
            "index_units": len(units),
            "duplicate_chunks_collapsed": len(chunks) - len(units),
            "chunks_per_document": round(len(chunks) / len(documents), 3),
            "queries": len(results),
            "mean_relevant_per_query": round(sum(relevant_counts) / len(relevant_counts), 3),
            "recall_at_k_ceiling": round(_ceiling(relevant_counts, k), 4),
        },
        timings_ms={
            "load": round((t_load - t0) * 1000, 1),
            "chunk": round((t_chunk - t_load) * 1000, 1),
            "retriever_build": round((t_build - t_chunk) * 1000, 1),
            "search": round((t_search - t_build) * 1000, 1),
            "rerank": round((t_rerank - t_rerank_start) * 1000, 1),
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
            "retriever": retriever.name,
            "reranked": bool(reranker),
            "recall_at_k": round(aggregate["recall_at_k"], 4),
            "ndcg_at_k": round(aggregate["ndcg_at_k"], 4),
            "ceiling": report.corpus_stats["recall_at_k_ceiling"],
            "queries": len(results),
            "total_ms": report.timings_ms["total"],
        },
    )
    return report
