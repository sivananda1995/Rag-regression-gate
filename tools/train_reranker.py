"""Train the linear reranker on the train split, and report it out of fold.

Protocol, which matters more than the model:

  * Training pairs come only from queries in data/splits.json train. The eval split is
    never seen by the fit, the feature scaler, or the hyperparameter choice.
  * Generalisation inside train is estimated with GroupKFold grouped by query, so a
    query's candidates never straddle a fold boundary. Without the grouping, a fold
    would contain other candidates from the same query and the score would be optimistic.
  * Three numbers are reported and all three go in the README: the in-sample train score
    (labeled as optimistic), the out-of-fold train score, and the eval-split score. The
    claimed gain is the eval one.
  * The shipped model is refit on all train queries after the folds are measured, which
    is standard practice: the folds estimate the error, the final fit uses all the data
    that is allowed to be used.

The model is deliberately a logistic regression. A gradient-boosted tree would likely
score a little higher and would be impossible to review in a pull request diff; the
coefficients of a linear model on eleven named features can be read and argued with.

Run from the repository root: python tools/train_reranker.py
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from ragate.config import load as load_config
from ragate.corpus import build_index_units, chunk_documents, load_documents, load_queries
from ragate.evaluate import run
from ragate.logging_setup import configure
from ragate.metrics import dedupe_preserving_rank, recall_at_k
from ragate.rerank import FEATURE_NAMES, FeatureContext, LinearReranker, extract_documents
from ragate.retrievers import build_retriever


def build_training_data(cfg, query_ids: set[str]):
    documents = load_documents(cfg.corpus.path)
    queries = [
        q for q in load_queries(cfg.corpus.queries, {d.doc_id for d in documents})
        if q.query_id in query_ids
    ]
    units = build_index_units(
        chunk_documents(documents, cfg.chunking), cfg.chunking.dedupe_identical
    )
    retriever = build_retriever(cfg)
    retriever.fit(units)
    ctx = FeatureContext.build(units)
    rankings = retriever.search([q.text for q in queries], cfg.rerank.depth)

    features, labels, groups, per_query = [], [], [], []
    for position, (query, ranking) in enumerate(zip(queries, rankings, strict=True)):
        if not ranking:
            continue
        relevant = set(query.relevant_doc_ids)
        doc_ids, matrix = extract_documents(query.text, ranking, units, ctx)
        if not doc_ids:
            continue
        # One row per candidate document, labeled by whether that document is relevant.
        # This is the change that made the reranker work: the label now means exactly
        # what recall@k counts.
        target = np.array([1 if doc_id in relevant else 0 for doc_id in doc_ids],
                          dtype=np.int8)
        features.append(matrix)
        labels.append(target)
        groups.append(np.full(len(doc_ids), position))
        per_query.append((query, doc_ids))
    return (
        np.vstack(features), np.concatenate(labels), np.concatenate(groups),
        per_query, units, ctx,
    )


def recall_after_scores(per_query, _units, scores_by_row, k: int) -> float:
    """Recall@k after reordering each query's candidate documents by the given scores."""
    values, cursor = [], 0
    for query, doc_ids in per_query:
        window = scores_by_row[cursor : cursor + len(doc_ids)]
        cursor += len(doc_ids)
        order = sorted(range(len(doc_ids)), key=lambda i: (-float(window[i]), i))
        ranked_docs = dedupe_preserving_rank(doc_ids[i] for i in order)
        values.append(recall_at_k(ranked_docs, query.relevant_doc_ids, k))
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="ragate.yaml")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--out-model", default="models/reranker.json")
    parser.add_argument("--out-report", default="docs/reranker_training_report.json")
    args = parser.parse_args()
    configure("ERROR", "json")

    cfg = load_config(args.config)
    cfg.rerank.enabled = False   # candidates must come from retrieval, unreranked
    splits = json.loads(Path(cfg.evaluate.splits_path).read_text())
    train_ids = set(splits["train"])

    X, y, groups, per_query, units, _ctx = build_training_data(cfg, train_ids)
    k = cfg.evaluate.k
    print(f"training rows: {X.shape[0]} candidate documents from {len(per_query)} train "
          f"queries, {int(y.sum())} positive ({y.mean():.1%})")

    # Retrieval order is the reference point: score each candidate by its negative
    # position so the original ranking is reproduced exactly.
    positions = []
    for _query, doc_ids in per_query:
        positions.extend(range(len(doc_ids)))
    retrieval_scores = -np.array(positions, dtype=float)
    baseline_recall = recall_after_scores(per_query, units, retrieval_scores, k)

    folds = GroupKFold(n_splits=args.folds)
    out_of_fold = np.zeros(X.shape[0], dtype=float)
    for fold, (fit_idx, test_idx) in enumerate(folds.split(X, y, groups), start=1):
        scaler = StandardScaler().fit(X[fit_idx])
        model = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        model.fit(scaler.transform(X[fit_idx]), y[fit_idx])
        out_of_fold[test_idx] = model.decision_function(scaler.transform(X[test_idx]))
        print(f"  fold {fold}: fit on {len(fit_idx)} pairs, scored {len(test_idx)}")

    oof_recall = recall_after_scores(per_query, units, out_of_fold, k)

    scaler = StandardScaler().fit(X)
    final = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    final.fit(scaler.transform(X), y)
    in_sample_recall = recall_after_scores(
        per_query, units, final.decision_function(scaler.transform(X)), k
    )

    reranker = LinearReranker(
        feature_names=FEATURE_NAMES,
        coefficients=[float(c) for c in final.coef_[0]],
        intercept=float(final.intercept_[0]),
        mean=[float(m) for m in scaler.mean_],
        scale=[float(s) for s in scaler.scale_],
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        training_notes=(
            f"logistic regression, class_weight=balanced, C=1.0, fitted on "
            f"{X.shape[0]} candidate pairs from {len(per_query)} train-split queries, "
            f"rerank depth {cfg.rerank.depth}"
        ),
    )
    model_path = reranker.save(args.out_model)

    # The honest headline: run the real pipeline with the shipped model and read the
    # eval-split number, which nothing above has touched.
    cfg_eval = load_config(args.config)
    cfg_eval.rerank.enabled = True
    cfg_eval.rerank.model_path = str(model_path)
    cfg_eval.validate()
    with_rerank = run(cfg_eval)
    cfg_plain = load_config(args.config)
    cfg_plain.rerank.enabled = False
    cfg_plain.validate()
    without_rerank = run(cfg_plain)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {"python": platform.python_version()},
        "protocol": (
            "Pairs come only from train-split queries. Out-of-fold scores use GroupKFold "
            "grouped by query id. The shipped model is refit on all train pairs. The "
            "eval-split numbers come from running the full pipeline with the shipped "
            "model and were not used for any fitting or selection decision."
        ),
        "training": {
            "candidate_documents": int(X.shape[0]),
            "queries": len(per_query),
            "positive_rate": round(float(y.mean()), 4),
            "folds": args.folds,
            "rerank_depth": cfg.rerank.depth,
        },
        f"recall_at_{k}_on_train_candidates": {
            "retrieval_order": round(baseline_recall, 4),
            "reranked_in_sample_optimistic": round(in_sample_recall, 4),
            "reranked_out_of_fold": round(oof_recall, 4),
        },
        f"recall_at_{k}_full_pipeline": {
            "all_queries_without_rerank": round(without_rerank.aggregate["recall_at_k"], 4),
            "all_queries_with_rerank": round(with_rerank.aggregate["recall_at_k"], 4),
            "train_split_with_rerank": round(with_rerank.by_split["train"]["recall_at_k"], 4),
            "eval_split_without_rerank": round(
                without_rerank.by_split["eval"]["recall_at_k"], 4),
            "eval_split_with_rerank": round(with_rerank.by_split["eval"]["recall_at_k"], 4),
        },
        "coefficients": dict(
            zip(FEATURE_NAMES, [round(float(c), 4) for c in final.coef_[0]], strict=True)
        ),
    }
    out = Path(args.out_report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"\nretrieval order          recall@{k} = {baseline_recall:.4f}")
    print(f"reranked, in sample      recall@{k} = {in_sample_recall:.4f}  (optimistic)")
    print(f"reranked, out of fold    recall@{k} = {oof_recall:.4f}")
    print("\nfull pipeline, eval split (held out):")
    pipeline = report[f"recall_at_{k}_full_pipeline"]
    print(f"  without rerank {pipeline['eval_split_without_rerank']:.4f}"
          f"   with rerank {pipeline['eval_split_with_rerank']:.4f}")
    print("\ncoefficients:")
    for name, value in report["coefficients"].items():
        print(f"  {name:<22} {value:+.4f}")
    print(f"\nwrote {model_path} and {out}")


if __name__ == "__main__":
    main()
