"""Choose fusion weights on the train split, then report the choice on the held-out half.

Naive expectation going in was that hybrid retrieval beats both of its components. On
this corpus with equal weights it does not: BM25 alone scores 0.9133 recall@5 and the
equal-weight fusion scores 0.9000, because the dense component here is a reproducible
hashing embedder rather than a semantic model, and reciprocal rank fusion with equal
weights lets the weaker retriever pull good candidates down. That is worth measuring
rather than asserting, and worth fixing by weighting rather than by dropping the
component, since the weight is the knob the design already exposes.

Protocol: every configuration is scored on the train split only. The single best
configuration by train recall is then scored once on the eval split, which nothing in
this script has looked at. Both numbers are reported, because the gap between them is
the honest estimate of how much of the gain is selection noise.

Run from the repository root: python tools/tune_retrieval.py
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from ragate.config import load as load_config
from ragate.evaluate import run
from ragate.logging_setup import configure

DENSE_WEIGHTS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5)
RRF_K = (10, 30, 60, 120)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="ragate.yaml")
    parser.add_argument("--out", default="docs/tuning_report.json")
    parser.add_argument("--metric", default="recall_at_k")
    args = parser.parse_args()
    configure("ERROR", "json")

    rows = []
    for dense_weight in DENSE_WEIGHTS:
        for rrf_k in RRF_K:
            cfg = load_config(args.config)
            cfg.retriever.mode = "hybrid"
            cfg.retriever.bm25_weight = 1.0
            cfg.retriever.dense_weight = dense_weight
            cfg.retriever.rrf_k = rrf_k
            cfg.rerank.enabled = False
            cfg.validate()
            report = run(cfg)
            rows.append({
                "dense_weight": dense_weight,
                "rrf_k": rrf_k,
                "train": round(report.by_split["train"][args.metric], 4),
                "eval": round(report.by_split["eval"][args.metric], 4),
                "all": round(report.aggregate[args.metric], 4),
            })
            print(f"dense_weight={dense_weight:<5} rrf_k={rrf_k:<4} "
                  f"train={rows[-1]['train']:.4f}  (eval={rows[-1]['eval']:.4f} not used "
                  f"for selection)")

    # Selection looks at train only. Ties break toward the simpler configuration, which
    # here means the smaller dense weight and the standard rrf_k.
    best = max(rows, key=lambda r: (r["train"], -r["dense_weight"], -abs(r["rrf_k"] - 60)))

    baselines = {}
    for mode in ("bm25", "dense"):
        cfg = load_config(args.config)
        cfg.retriever.mode = mode
        cfg.rerank.enabled = False
        cfg.validate()
        report = run(cfg)
        baselines[mode] = {
            "train": round(report.by_split["train"][args.metric], 4),
            "eval": round(report.by_split["eval"][args.metric], 4),
            "all": round(report.aggregate[args.metric], 4),
        }

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metric": args.metric,
        "protocol": (
            "Configurations are scored on the train split; the best by train score is "
            "reported on the eval split, which was not used for selection. Component "
            "baselines are shown on both splits for reference."
        ),
        "environment": {"python": platform.python_version()},
        "grid": rows,
        "component_baselines": baselines,
        "selected": best,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"\nselected on train: dense_weight={best['dense_weight']} rrf_k={best['rrf_k']}")
    print(f"  train {best['train']:.4f}   eval {best['eval']:.4f}   all {best['all']:.4f}")
    for mode in ("bm25", "dense"):
        print(f"  {mode} alone: train {baselines[mode]['train']:.4f}  "
              f"eval {baselines[mode]['eval']:.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
