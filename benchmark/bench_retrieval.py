"""Measure how retrieval latency and index choice behave as the corpus grows.

Two questions are being answered, both of which the README quotes:

  1. Where does exact (flat) search stop being viable? The gate defaults to exact
     search, and that default is only defensible if the cost is known.
  2. How much does the approximate index cost in quality? Two label-free measures are
     reported, because the first one alone is misleading:
       * top-k document set overlap with exact search, which is easy to read but
         punishes an index for swapping two documents of equal similarity, and
       * retrieved-score parity: the true cosine similarity of what the approximate
         index returned, divided by the true cosine similarity of the exact top-k.
         1.0 means no quality was lost. This is the measure that exposed the
         duplicate-vector problem described in the README war story.

Scaling method: the labeled corpus is 420 articles. To measure latency at larger sizes
it is padded with tenant-variant articles, with the tenant name woven into every
sentence rather than only the title. That detail matters: an earlier version prefixed
the title alone, which left the remaining chunks byte-identical across tenants, and the
resulting duplicate vectors, not the corpus size, dominated the measurement.
Padding is used for latency and index-agreement only. Every quality number in this
repository comes from the unpadded, labeled corpus.

Run: python benchmark/bench_retrieval.py --out benchmark/results
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ragate.config import load as load_config
from ragate.corpus import (
    Document,
    build_index_units,
    chunk_documents,
    load_documents,
    load_queries,
)
from ragate.embedders import build_embedder
from ragate.indexes.flat import FlatIndex
from ragate.logging_setup import configure, get_logger
from ragate.metrics import dedupe_preserving_rank

log = get_logger("benchmark")

TENANTS = [
    "northwind", "acme", "globex", "initech", "umbrella", "hooli", "vandelay",
    "wonka", "stark", "wayne", "cyberdyne", "tyrell", "soylent", "gringotts",
    "duff", "monarch", "oscorp", "abstergo", "aperture", "blackmesa", "encom",
    "massive", "sirius", "weyland",
]


def pad_corpus(base: list[Document], multiple: int) -> list[Document]:
    """Return the base corpus plus (multiple - 1) tenant variants of every document.

    The tenant name is woven into every sentence so that no chunk of a variant is
    byte-identical to a chunk of the original. Prefixing the title alone is not enough:
    it leaves every chunk after the first identical across tenants, which turns a
    scaling benchmark into a duplicate-vector benchmark.
    """
    out = list(base)
    for copy_index in range(1, multiple):
        tenant = TENANTS[(copy_index - 1) % len(TENANTS)]
        for doc in base:
            sentences = [s.strip() for s in doc.text.split(". ") if s.strip()]
            body = ". ".join(f"{s} for the {tenant} tenant" for s in sentences)
            out.append(
                Document(
                    doc_id=f"{tenant}-{copy_index}-{doc.doc_id}",
                    title=f"[{tenant} tenant] {doc.title}",
                    text=f"Tenant {tenant} deployment note {copy_index}. {body}",
                )
            )
    return out


def score_parity(
    exact_scores: np.ndarray, ann_indices: np.ndarray, true_scores: np.ndarray
) -> tuple[float, float]:
    """Mean and worst-case ratio of returned similarity mass to the exact optimum.

    true_scores is the full (units, queries) similarity matrix, so the similarity of
    whatever the approximate index returned is looked up rather than trusted from the
    index itself.
    """
    ratios = []
    for row in range(ann_indices.shape[0]):
        optimum = float(exact_scores[row].sum())
        if optimum <= 0.0:
            continue
        returned = float(
            sum(true_scores[int(i), row] for i in ann_indices[row] if int(i) >= 0)
        )
        ratios.append(returned / optimum)
    return round(statistics.fmean(ratios), 5), round(min(ratios), 5)


def percentiles(samples_ms: list[float]) -> dict[str, float]:
    ordered = sorted(samples_ms)
    return {
        "p50": round(statistics.median(ordered), 3),
        "p95": round(ordered[max(0, int(0.95 * len(ordered)) - 1)], 3),
        "p99": round(ordered[max(0, int(0.99 * len(ordered)) - 1)], 3),
        "max": round(ordered[-1], 3),
        "mean": round(statistics.fmean(ordered), 3),
    }


def time_queries(index, query_vectors: np.ndarray, k: int, passes: int) -> list[float]:
    """One timing sample per query per pass, single query at a time.

    Batched search is faster per query but hides the latency a caller actually sees,
    and this gate runs one query at a time inside a CI step.
    """
    samples: list[float] = []
    for _ in range(passes):
        for row in range(query_vectors.shape[0]):
            single = query_vectors[row : row + 1]
            start = time.perf_counter()
            index.search(single, k)
            samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def top_docs(indices: np.ndarray, units, k: int) -> list[list[str]]:
    return [
        dedupe_preserving_rank(
            doc_id for i in row if int(i) >= 0 for doc_id in units[int(i)].doc_ids
        )[:k]
        for row in indices
    ]


def agreement(a: list[list[str]], b: list[list[str]]) -> float:
    """Mean overlap of two top-k document lists, as a fraction of k."""
    scores = [len(set(x) & set(y)) / max(len(x), 1) for x, y in zip(a, b, strict=True)]
    return round(statistics.fmean(scores), 4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="ragate.yaml")
    parser.add_argument("--out", default="benchmark/results")
    parser.add_argument("--multiples", default="1,3,6,12,24",
                        help="corpus size multiples of the labeled corpus")
    parser.add_argument("--passes", type=int, default=3)
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure("WARNING", "json")
    base_docs = load_documents(cfg.corpus.path)
    queries = load_queries(cfg.corpus.queries, {d.doc_id for d in base_docs})
    query_texts = [q.text for q in queries]
    k = cfg.evaluate.k
    chunk_k = max(k * 4, k + 10)

    try:
        from ragate.indexes.faiss_index import FaissHnswIndex
    except Exception as exc:  # pragma: no cover - optional dependency
        FaissHnswIndex = None
        log.warning("faiss unavailable, exact search only", extra={"error": str(exc)})

    rows = []
    for multiple in [int(m) for m in args.multiples.split(",")]:
        docs = pad_corpus(base_docs, multiple)
        chunks = chunk_documents(docs, cfg.chunking)
        units = build_index_units(chunks, cfg.chunking.dedupe_identical)
        texts = [u.text for u in units]

        embedder = build_embedder(cfg.embedder)
        t0 = time.perf_counter()
        embedder.fit(texts)
        vectors = embedder.encode(texts)
        embed_s = time.perf_counter() - t0
        query_vectors = embedder.encode(query_texts)

        flat = FlatIndex()
        t0 = time.perf_counter()
        flat.build(vectors)
        flat_build_ms = (time.perf_counter() - t0) * 1000.0
        flat_samples = time_queries(flat, query_vectors, chunk_k, args.passes)
        flat_scores, flat_indices = flat.search(query_vectors, chunk_k)
        flat_docs = top_docs(flat_indices, units, k)
        true_scores = vectors @ query_vectors.T

        row = {
            "corpus_multiple": multiple,
            "documents": len(docs),
            "chunks": len(chunks),
            "index_units": len(units),
            "duplicates_collapsed": len(chunks) - len(units),
            "embed_corpus_s": round(embed_s, 2),
            "flat": {
                "build_ms": round(flat_build_ms, 1),
                "query_ms": percentiles(flat_samples),
                "index_mb": round(vectors.nbytes / 1e6, 1),
            },
        }

        if FaissHnswIndex is not None:
            ann = FaissHnswIndex()
            t0 = time.perf_counter()
            ann.build(vectors)
            ann_build_ms = (time.perf_counter() - t0) * 1000.0
            ann_samples = time_queries(ann, query_vectors, chunk_k, args.passes)
            _, ann_indices = ann.search(query_vectors, chunk_k)
            ann_docs = top_docs(ann_indices, units, k)
            parity_mean, parity_worst = score_parity(flat_scores, ann_indices, true_scores)
            row["faiss_hnsw"] = {
                "build_ms": round(ann_build_ms, 1),
                "query_ms": percentiles(ann_samples),
                "top_k_agreement_with_exact": agreement(flat_docs, ann_docs),
                "retrieved_score_parity_mean": parity_mean,
                "retrieved_score_parity_worst": parity_worst,
            }

        rows.append(row)
        print(
            f"multiple={multiple:>3} docs={len(docs):>6} chunks={len(chunks):>6} "
            f"flat p95={row['flat']['query_ms']['p95']:>7.3f} ms"
            + (
                f"  hnsw p95={row['faiss_hnsw']['query_ms']['p95']:>7.3f} ms"
                f"  agreement={row['faiss_hnsw']['top_k_agreement_with_exact']}"
                f"  parity={row['faiss_hnsw']['retrieved_score_parity_mean']}"
                if "faiss_hnsw" in row
                else ""
            )
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "conditions": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": __import__("os").cpu_count(),
            "queries_per_pass": len(queries),
            "passes": args.passes,
            "k_documents": k,
            "k_chunks_retrieved": chunk_k,
            "embedder": cfg.embedder.provider,
            "dimensions": cfg.embedder.dimensions,
            "chunking": f"{cfg.chunking.strategy} {cfg.chunking.target_chars}"
                        f"/{cfg.chunking.overlap_chars}",
            "note": "single-query search, one at a time; padded corpora are used for "
                    "latency and agreement only",
        },
        "rows": rows,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "retrieval_scaling.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "| corpus multiple | documents | chunks | index units | exact p50 ms | exact p95 ms "
        "| exact p99 ms | hnsw p95 ms | hnsw score parity | hnsw top-5 doc overlap "
        "| exact index MB |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        ann = row.get("faiss_hnsw")
        lines.append(
            f"| {row['corpus_multiple']}x | {row['documents']} | {row['chunks']} "
            f"| {row['index_units']} "
            f"| {row['flat']['query_ms']['p50']} | {row['flat']['query_ms']['p95']} "
            f"| {row['flat']['query_ms']['p99']} "
            f"| {ann['query_ms']['p95'] if ann else 'n/a'} "
            f"| {ann['retrieved_score_parity_mean'] if ann else 'n/a'} "
            f"| {ann['top_k_agreement_with_exact'] if ann else 'n/a'} "
            f"| {row['flat']['index_mb']} |"
        )
    (out_dir / "retrieval_scaling.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out_dir}/retrieval_scaling.json and .md")


if __name__ == "__main__":
    main()
