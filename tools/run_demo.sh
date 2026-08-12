#!/usr/bin/env bash
# Reproduce the three gate outcomes shown in the README, from the committed baseline.
# Every artifact under reports/ is produced by this script; nothing is hand-edited.
set -uo pipefail
mkdir -p reports

run() {
  local name="$1" profile="$2"
  echo "=== $name ($profile) ==="
  ragate -c "$profile" gate \
    --markdown "reports/${name}.md" \
    --html "reports/${name}.html" > "reports/${name}.stdout.txt" 2> "reports/${name}.stderr.log"
  local code=$?
  echo "exit code: $code"
  tail -n 14 "reports/${name}.stdout.txt"
  echo
  return 0
}

run pass-improvement configs/candidate-trigram.yaml
run warn-borderline  configs/candidate-borderline.yaml
run fail-regression  configs/candidate-fixed-chunking.yaml
echo "reports written to reports/"
