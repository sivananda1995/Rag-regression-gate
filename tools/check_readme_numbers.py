"""Fail if the README quotes a number that the code no longer produces.

Reads docs/metrics.json, which tools/collect_metrics.py regenerates by running the real
pipeline, benchmarks, and test suite, then checks that the README still contains each
current value. A number that has moved shows up as a missing display string, which is the
signal that the document and the code have diverged.

This is deliberately a text search rather than a template renderer. Templating the README
would keep it consistent and unreadable; a check keeps the prose hand-written and the
numbers honest, and it fails loudly the moment those two goals conflict.

Run from the repository root: python tools/check_readme_numbers.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", default="docs/metrics.json")
    parser.add_argument("--readme", default="README.md")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        print(f"{metrics_path} is missing. Run: python tools/collect_metrics.py")
        return 2
    payload = json.loads(metrics_path.read_text())
    readme = Path(args.readme).read_text()

    missing, checked = [], 0
    for name, record in payload["metrics"].items():
        variants = record["display"]
        checked += 1
        if not any(variant in readme for variant in variants):
            missing.append((name, variants, record["value"], record["source"]))

    print(f"checked {checked} measured values from {metrics_path} against {args.readme}")
    if not missing:
        print("every number in the readme matches a value this build measured")
        return 0

    print(f"\n{len(missing)} value(s) in the readme are stale or absent:\n")
    for name, variants, value, source in missing:
        shown = " or ".join(repr(v) for v in variants)
        print(f"  {name}")
        print(f"    current value : {value}  (expected the readme to contain {shown})")
        print(f"    measured by   : {source}")
    print("\nEither the readme needs updating, or the change that moved these numbers "
          "needs explaining.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
