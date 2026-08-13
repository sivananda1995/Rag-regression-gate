# Every target here regenerates something the README claims. Nothing in this file
# post-processes a number; each target runs the real thing and writes its output.
.PHONY: help install test lint verify baseline gate demo prove-reranker tune train bench charts shots video receipts clean

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | column -t -s "$$(printf '\t')"

install: ## install the package with dev, faiss, and training extras
	pip install -e ".[dev,faiss,train]"

lint: ## ruff
	ruff check .

test: ## run the suite with coverage
	pytest --cov=ragate --cov-report=term --cov-report=xml -q

baseline: ## re-record the committed baseline (review the diff before committing it)
	ragate baseline

gate: ## run the gate against the committed baseline
	ragate gate

demo: ## reproduce the three gate outcomes in the readme
	bash tools/run_demo.sh

prove-reranker: ## show the gate confirming the reranker is a real improvement, not noise
	ragate gate --baseline baselines/baseline-no-rerank.json

tune: ## re-run the fusion weight sweep (selects on train, reports on eval)
	python tools/tune_retrieval.py

train: ## retrain the document reranker on the train split only
	python tools/train_reranker.py

bench: ## re-run both benchmarks
	python benchmark/bench_retrieval.py
	python benchmark/bench_dedupe_effect.py

charts: ## regenerate benchmark charts from results json
	python benchmark/plot_results.py

shots: ## regenerate readme screenshots from real reports
	python tools/capture_screenshots.py

video: ## record the demo video from a real terminal session
	python tools/record_demo.py

receipts: ## regenerate the metrics registry and check every readme number against it
	python tools/collect_metrics.py
	python tools/check_readme_numbers.py

verify: lint test receipts ## the full check: lint, tests, and every readme number re-measured

clean:
	rm -rf reports .pytest_cache .ruff_cache .coverage coverage.xml htmlcov
