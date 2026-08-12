"""Measure what collapsing byte-identical chunks does to index size, latency, and
approximate-search quality.

This script exists because a wrong conclusion was nearly shipped. The approximate
index looked fine on the base corpus and appeared to fall apart as the corpus grew,
which reads as "HNSW does not work here". The real cause was duplicate vectors. The
numbers below are the evidence, regenerated on demand rather than quoted from memory.

Run from the repository root: python benchmark/bench_dedupe_effect.py
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# The sibling benchmark module is reused rather than copied, so both scripts pad and
# time the same way. Run from the repository root; the data paths in ragate.yaml are
# relative to it.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_retrieval import pad_corpus, percentiles, time_queries  # noqa: E402

from ragate.config import load as load_config  # noqa: E402
from ragate.corpus import (  # noqa: E402
    build_index_units,
    chunk_documents,
    load_documents,
    load_queries,
)
from ragate.embedders import build_embedder  # noqa: E402
from ragate.evaluate import run  # noqa: E402
from ragate.indexes.flat import FlatIndex  # noqa: E402
from ragate.logging_setup import configure  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="ragate.yaml")
    parser.add_argument("--out", default="benchmark/results")
    parser.add_argument("--multiples", default="1,3,12")
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure("WARNING", "json")
    base_docs = load_documents(cfg.corpus.path)
    queries = load_queries(cfg.corpus.queries, {d.doc_id for d in base_docs})
    query_texts = [q.text for q in queries]
    chunk_k = max(cfg.evaluate.k * 4, cfg.evaluate.k + 10)

    from ragate.indexes.faiss_index import FaissHnswIndex

    rows = []
    for multiple in [int(m) for m in args.multiples.split(",")]:
        docs = pad_corpus(base_docs, multiple)
        chunks = chunk_documents(docs, cfg.chunking)
        for dedupe in (False, True):
            units = build_index_units(chunks, dedupe)
            texts = [u.text for u in units]
            embedder = build_embedder(cfg.embedder)
            embedder.fit(texts)
            vectors = np.ascontiguousarray(embedder.encode(texts))
            query_vectors = embedder.encode(query_texts)

            flat = FlatIndex()
            flat.build(vectors)
            exact_scores, _ = flat.search(query_vectors, chunk_k)
            exact_samples = time_queries(flat, query_vectors, chunk_k, passes=3)

            ann = FaissHnswIndex()
            t0 = time.perf_counter()
            ann.build(vectors)
            ann_build_ms = (time.perf_counter() - t0) * 1000.0
            _, ann_indices = ann.search(query_vectors, chunk_k)
            true_scores = vectors @ query_vectors.T
            ratios = [
                float(sum(true_scores[int(i), r] for i in ann_indices[r] if int(i) >= 0))
                / float(exact_scores[r].sum())
                for r in range(len(queries))
            ]

            rows.append(
                {
                    "corpus_multiple": multiple,
                    "dedupe_identical": dedupe,
                    "chunks": len(chunks),
                    "index_units": len(units),
                    "vectors_saved_pct": round(
                        100.0 * (len(chunks) - len(units)) / len(chunks), 1
                    ),
                    "index_mb": round(vectors.nbytes / 1e6, 2),
                    "exact_query_ms": percentiles(exact_samples),
                    "hnsw_build_ms": round(ann_build_ms, 1),
                    "hnsw_score_parity_mean": round(statistics.fmean(ratios), 5),
                    "hnsw_score_parity_worst": round(min(ratios), 5),
                }
            )
            print(
                f"multiple={multiple:>2} dedupe={str(dedupe):<5} "
                f"units={len(units):>6} exact p95={rows[-1]['exact_query_ms']['p95']:>6.3f} ms "
                f"hnsw parity mean={rows[-1]['hnsw_score_parity_mean']:.5f} "
                f"worst={rows[-1]['hnsw_score_parity_worst']:.4f}"
            )

    # Quality on the labeled corpus, which is the only place a recall number is valid.
    quality = {}
    for dedupe in (False, True):
        cfg.chunking.dedupe_identical = dedupe
        report = run(cfg)
        quality[f"dedupe_{str(dedupe).lower()}"] = {
            "recall_at_k": round(report.aggregate["recall_at_k"], 4),
            "ndcg_at_k": round(report.aggregate["ndcg_at_k"], 4),
            "index_units": report.corpus_stats["index_units"],
            "total_ms": report.timings_ms["total"],
        }
    print("\nlabeled corpus quality:", json.dumps(quality, indent=2))

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "conditions": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "queries": len(queries),
            "k_chunks_retrieved": chunk_k,
            "note": "identical embedder and chunker in every row; the only variable is "
                    "whether byte-identical chunk text is indexed once or many times",
        },
        "rows": rows,
        "labeled_corpus_quality": quality,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dedupe_effect.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "| corpus multiple | dedupe | chunks | vectors indexed | vectors saved | "
        "exact p95 ms | hnsw parity mean | hnsw parity worst case |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['corpus_multiple']}x | {'on' if row['dedupe_identical'] else 'off'} "
            f"| {row['chunks']} | {row['index_units']} | {row['vectors_saved_pct']}% "
            f"| {row['exact_query_ms']['p95']} | {row['hnsw_score_parity_mean']} "
            f"| {row['hnsw_score_parity_worst']} |"
        )
    (out_dir / "dedupe_effect.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out_dir}/dedupe_effect.json and .md")


if __name__ == "__main__":
    main()
