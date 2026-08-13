#!/usr/bin/env bash
# Reproduce every gate outcome shown in the README, from the committed baselines.
# Nothing here is hand-edited: reports/ is written by these runs.
set -uo pipefail
mkdir -p reports

run() {
  local name="$1" profile="$2" baseline="${3:-}"
  local args=(-c "$profile" gate --markdown "reports/${name}.md" --html "reports/${name}.html")
  [ -n "$baseline" ] && args+=(--baseline "$baseline")
  echo "=== $name ($profile${baseline:+ vs $baseline}) ==="
  ragate "${args[@]}" > "reports/${name}.stdout.txt" 2> "reports/${name}.stderr.log"
  local code=$?
  echo "exit code: $code"
  head -n 6 "reports/${name}.stdout.txt"
  echo
  return 0
}

# A regression the gate must block.
run fail-regression  configs/candidate-fixed-chunking.yaml
# A drop past a tightened tolerance that this golden set cannot separate from noise.
run warn-borderline  configs/candidate-borderline.yaml
# Removing the reranker is itself a regression, which proves it is load-bearing.
run fail-no-reranker configs/candidate-no-reranker.yaml
# Read the other way: the current pipeline against the pre-reranker baseline, which is
# how the reranker's gain was confirmed rather than argued.
run pass-reranker-gain ragate.yaml baselines/baseline-no-rerank.json
echo "reports written to reports/"
