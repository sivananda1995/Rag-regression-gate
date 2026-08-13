"""Reproduce the reranker variant that failed, so the war story is checkable.

The README explains that the first reranker ranked chunks instead of documents and lost
held-out recall. That claim is only worth reading if the number behind it can be
regenerated, so the failed design lives here rather than in a commit message.

What this script rebuilds, deliberately unchanged from the version that was thrown away:

  * features per retrieved chunk, including "unit_breadth", the number of documents a
    chunk belongs to,
  * a label of 1 when the chunk belongs to any relevant document,
  * a logistic regression fitted on the train split, exactly as the shipped model is,
  * the chunk ranking reordered by model score, then collapsed to documents for scoring.

The flaw the numbers expose: a chunk shared by thirty articles has thirty chances of
touching one of the labeled documents, so the label rewards promoting boilerplate. The
model learns exactly that, and its largest positive coefficient sits on chunk breadth,
while recall@5 counts distinct documents and is unimpressed.

Run from the repository root: python experiments/chunk_level_reranker.py
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from ragate.config import load as load_config
from ragate.corpus import build_index_units, chunk_documents, load_documents, load_queries
from ragate.logging_setup import configure
from ragate.metrics import dedupe_preserving_rank, recall_at_k
from ragate.rerank.features import FeatureContext, bigrams, jaccard, tokens
from ragate.retrievers import build_retriever

CHUNK_FEATURE_NAMES = (
    "fused_score",
    "reciprocal_rank",
    "token_coverage",
    "idf_weighted_coverage",
    "bigram_coverage",
    "lead_overlap",
    "code_token_match",
    "unit_breadth",     # the culprit
    "unit_length",
    "top1_jaccard",
    "top1_lead_jaccard",
)


def chunk_features(query, ranking, units, ctx: FeatureContext) -> np.ndarray:
    """The discarded per-chunk feature extractor, preserved verbatim in behaviour."""
    from ragate.rerank.features import _CODE  # noqa: PLC0415 - experiment-only import

    query_tokens = tokens(query)
    query_set = set(query_tokens)
    query_bigrams = bigrams(query_tokens)
    query_codes = set(_CODE.findall(query))
    query_idf_total = sum(ctx.idf.get(t, 1.0) for t in query_set) or 1.0

    top1 = ranking[0][0] if ranking else None
    top1_tokens = set(ctx.unit_tokens[top1]) if top1 is not None else set()
    top1_lead = ctx.unit_lead[top1] if top1 is not None else set()

    rows = np.zeros((len(ranking), len(CHUNK_FEATURE_NAMES)), dtype=np.float64)
    for row, (unit_index, fused_score) in enumerate(ranking):
        unit_set = set(ctx.unit_tokens[unit_index])
        matched = query_set & unit_set
        rows[row] = (
            fused_score,
            1.0 / (row + 1),
            len(matched) / max(len(query_set), 1),
            sum(ctx.idf.get(t, 1.0) for t in matched) / query_idf_total,
            len(query_bigrams & ctx.unit_bigrams[unit_index]) / max(len(query_bigrams), 1),
            len(query_set & ctx.unit_lead[unit_index]) / max(len(query_set), 1),
            1.0 if (query_codes and query_codes & ctx.unit_codes[unit_index]) else 0.0,
            math.log1p(len(units[unit_index].doc_ids)),
            math.log1p(len(ctx.unit_tokens[unit_index])),
            jaccard(unit_set, top1_tokens),
            jaccard(ctx.unit_lead[unit_index], top1_lead),
        )
    return rows


def recall_for(queries, rankings, units, scores_per_query, k: int) -> float:
    values = []
    for query, ranking, scores in zip(queries, rankings, scores_per_query, strict=True):
        order = sorted(range(len(ranking)), key=lambda i: (-float(scores[i]), i))
        ranked_docs = dedupe_preserving_rank(
            doc_id for i in order for doc_id in units[ranking[i][0]].doc_ids
        )
        values.append(recall_at_k(ranked_docs, query.relevant_doc_ids, k))
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="ragate.yaml")
    parser.add_argument("--out", default="docs/experiments/chunk_level_reranker.json")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    configure("ERROR", "json")

    cfg = load_config(args.config)
    cfg.rerank.enabled = False
    k = cfg.evaluate.k
    splits = json.loads(Path(cfg.evaluate.splits_path).read_text())

    documents = load_documents(cfg.corpus.path)
    all_queries = load_queries(cfg.corpus.queries, {d.doc_id for d in documents})
    units = build_index_units(
        chunk_documents(documents, cfg.chunking), cfg.chunking.dedupe_identical
    )
    retriever = build_retriever(cfg)
    retriever.fit(units)
    ctx = FeatureContext.build(units)

    def prepare(query_ids: set[str]):
        selected = [q for q in all_queries if q.query_id in query_ids]
        rankings = retriever.search([q.text for q in selected], cfg.rerank.depth)
        matrices = [chunk_features(q.text, r, units, ctx)
                    for q, r in zip(selected, rankings, strict=True)]
        labels = [
            np.array([1 if set(q.relevant_doc_ids) & set(units[u].doc_ids) else 0
                      for u, _ in r], dtype=np.int8)
            for q, r in zip(selected, rankings, strict=True)
        ]
        return selected, rankings, matrices, labels

    train_q, train_r, train_X, train_y = prepare(set(splits["train"]))
    eval_q, eval_r, eval_X, eval_y = prepare(set(splits["eval"]))

    X = np.vstack(train_X)
    y = np.concatenate(train_y)
    groups = np.concatenate(
        [np.full(len(m), i) for i, m in enumerate(train_X)]
    )

    retrieval_order = [(-np.arange(len(r), dtype=float)) for r in train_r]
    train_baseline = recall_for(train_q, train_r, units, retrieval_order, k)
    eval_baseline = recall_for(
        eval_q, eval_r, units, [(-np.arange(len(r), dtype=float)) for r in eval_r], k
    )

    out_of_fold = np.zeros(X.shape[0])
    for fit_idx, test_idx in GroupKFold(n_splits=args.folds).split(X, y, groups):
        scaler = StandardScaler().fit(X[fit_idx])
        model = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        model.fit(scaler.transform(X[fit_idx]), y[fit_idx])
        out_of_fold[test_idx] = model.decision_function(scaler.transform(X[test_idx]))

    cursor, oof_per_query = 0, []
    for matrix in train_X:
        oof_per_query.append(out_of_fold[cursor : cursor + len(matrix)])
        cursor += len(matrix)
    train_oof = recall_for(train_q, train_r, units, oof_per_query, k)

    scaler = StandardScaler().fit(X)
    final = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    final.fit(scaler.transform(X), y)
    train_in_sample = recall_for(
        train_q, train_r, units,
        [final.decision_function(scaler.transform(m)) for m in train_X], k,
    )
    eval_scored = recall_for(
        eval_q, eval_r, units,
        [final.decision_function(scaler.transform(m)) for m in eval_X], k,
    )

    coefficients = dict(
        zip(CHUNK_FEATURE_NAMES, [round(float(c), 4) for c in final.coef_[0]], strict=True)
    )
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {"python": platform.python_version()},
        "what_this_is": (
            "The rejected chunk-level reranker, reproduced so the README's war story can "
            "be checked rather than believed. The shipped reranker ranks documents; see "
            "docs/adr/ADR-005."
        ),
        "unit_of_ranking": "chunk",
        "training": {
            "rows": int(X.shape[0]),
            "queries": len(train_q),
            "positive_rate": round(float(y.mean()), 4),
            "folds": args.folds,
        },
        f"recall_at_{k}": {
            "train_retrieval_order": round(train_baseline, 4),
            "train_reranked_in_sample": round(train_in_sample, 4),
            "train_reranked_out_of_fold": round(train_oof, 4),
            "eval_retrieval_order": round(eval_baseline, 4),
            "eval_reranked": round(eval_scored, 4),
            "eval_delta": round(eval_scored - eval_baseline, 4),
        },
        "coefficients": coefficients,
        "diagnosis": (
            "unit_breadth carries a large positive weight, so the model promotes chunks "
            "that many documents share. Under a chunk-level label that is rewarded, "
            "because a widely shared chunk is more likely to touch some relevant "
            "document, but recall@k counts distinct documents and the promoted "
            "boilerplate displaces the specific answer."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"chunk-level reranker, recall@{k}")
    print(f"  train retrieval order   {train_baseline:.4f}")
    print(f"  train in sample         {train_in_sample:.4f}")
    print(f"  train out of fold       {train_oof:.4f}")
    print(f"  eval retrieval order    {eval_baseline:.4f}")
    print(f"  eval reranked           {eval_scored:.4f}  "
          f"(delta {eval_scored - eval_baseline:+.4f})")
    print(f"  unit_breadth coefficient {coefficients['unit_breadth']:+.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
