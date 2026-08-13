"""Split the golden set into a tuning half and a held-out half, once, deterministically.

Why this file exists at all: every tunable in this pipeline (fusion weights, rrf_k, the
reranker's coefficients) could be fitted directly against the 140 golden queries, and the
resulting numbers would be higher and meaningless. Anything chosen by looking at a query
must not be used to make a claim about that query.

The split is stratified by task family, so both halves contain every kind of question,
and it is written to disk rather than recomputed, so a future change to this script
cannot silently move the boundary under a published number.

Run: python tools/make_splits.py --eval-fraction 0.3 --seed 20260813
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", default="data/golden_queries.jsonl")
    parser.add_argument("--out", default="data/splits.json")
    parser.add_argument("--eval-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.queries).read_text().splitlines()
        if line.strip()
    ]
    by_family: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_family[row.get("task_key", "unknown")].append(row["query_id"])

    rng = random.Random(args.seed)
    train: list[str] = []
    held_out: list[str] = []
    for _family, query_ids in sorted(by_family.items()):
        ids = sorted(query_ids)
        rng.shuffle(ids)
        cut = max(1, round(len(ids) * args.eval_fraction)) if len(ids) > 1 else 0
        held_out.extend(ids[:cut])
        train.extend(ids[cut:])

    payload = {
        "seed": args.seed,
        "eval_fraction": args.eval_fraction,
        "stratified_by": "task_key",
        "note": (
            "train is the only split any tunable may look at: fusion weights, rrf_k, and "
            "the reranker coefficients are all fitted on train. Numbers quoted as a gain "
            "come from eval, which nothing was fitted on."
        ),
        "train": sorted(train),
        "eval": sorted(held_out),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}: {len(train)} train, {len(held_out)} eval, "
          f"{len(by_family)} task families")


if __name__ == "__main__":
    main()
