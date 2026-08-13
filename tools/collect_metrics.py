"""Regenerate every number the README claims, into one machine-readable registry.

This exists because of a specific failure mode in engineering writing: a README quotes a
measurement, the code changes, and the number stays. Six months later the document is
confidently wrong and nobody can tell which parts still hold.

So the numbers are not typed into the README by hand and left there. Each one is produced
here by running the real thing, written to docs/metrics.json with the command that
produced it, and then tools/check_readme_numbers.py asserts that the README contains the
current value. CI runs both, so a stale number fails the build exactly like a stale test.

Run from the repository root: python tools/collect_metrics.py
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from ragate.config import load as load_config
from ragate.evaluate import EvalReport, run
from ragate.gate import evaluate_gate
from ragate.logging_setup import configure


def entry(value, display, source: str, note: str = "", reproducible: bool = True) -> dict:
    return {
        "value": value,
        "display": display if isinstance(display, list) else [display],
        "source": source,
        "note": note,
        "reproducible": reproducible,
    }


def fmt(value: float, places: int = 4) -> str:
    return f"{value:.{places}f}"


def collect_pipeline(metrics: dict) -> None:
    configure("ERROR", "json")
    cfg = load_config("ragate.yaml")
    report = run(cfg)
    metrics["baseline_recall_at_5"] = entry(
        round(report.aggregate["recall_at_k"], 4), fmt(report.aggregate["recall_at_k"]),
        "ragate eval", "default pipeline, all 140 golden queries")
    metrics["baseline_ndcg_at_5"] = entry(
        round(report.aggregate["ndcg_at_k"], 4),
        [fmt(report.aggregate["ndcg_at_k"]), fmt(report.aggregate["ndcg_at_k"], 3)],
        "ragate eval")
    metrics["baseline_mrr_at_5"] = entry(
        round(report.aggregate["mrr_at_k"], 4), fmt(report.aggregate["mrr_at_k"]),
        "ragate eval")
    metrics["recall_ceiling"] = entry(
        report.corpus_stats["recall_at_k_ceiling"], str(report.corpus_stats["recall_at_k_ceiling"]),
        "ragate eval", "queries with more labels than k cannot reach 1.0")
    metrics["baseline_recall_train_split"] = entry(
        round(report.by_split["train"]["recall_at_k"], 4),
        fmt(report.by_split["train"]["recall_at_k"]), "ragate eval",
        f"{int(report.by_split['train']['queries'])} queries the reranker was fitted on")
    metrics["baseline_recall_eval_split"] = entry(
        round(report.by_split["eval"]["recall_at_k"], 4),
        fmt(report.by_split["eval"]["recall_at_k"]), "ragate eval",
        f"{int(report.by_split['eval']['queries'])} held-out queries")
    metrics["index_units"] = entry(
        report.corpus_stats["index_units"], str(report.corpus_stats["index_units"]),
        "ragate eval", "chunks after collapsing byte-identical text")
    metrics["chunks"] = entry(
        report.corpus_stats["chunks"], str(report.corpus_stats["chunks"]), "ragate eval")
    metrics["documents"] = entry(
        report.corpus_stats["documents"], str(report.corpus_stats["documents"]), "ragate eval")
    metrics["golden_queries"] = entry(
        report.corpus_stats["queries"], str(report.corpus_stats["queries"]), "ragate eval")


def collect_gate_outcomes(metrics: dict) -> None:
    baseline = EvalReport.load("baselines/baseline.json")
    scenarios = {
        "fail_regression": "configs/candidate-fixed-chunking.yaml",
        "warn_borderline": "configs/candidate-borderline.yaml",
        "fail_no_reranker": "configs/candidate-no-reranker.yaml",
    }
    for name, profile in scenarios.items():
        cfg = load_config(profile)
        verdict = evaluate_gate(baseline, run(cfg), cfg.gate)
        metrics[f"{name}_recall"] = entry(
            round(verdict.candidate, 4), fmt(verdict.candidate), f"ragate -c {profile} gate")
        metrics[f"{name}_delta"] = entry(
            round(verdict.delta, 4), f"{verdict.delta:+.4f}".replace("+", ""),
            f"ragate -c {profile} gate")
        metrics[f"{name}_interval"] = entry(
            [round(verdict.ci_low, 4), round(verdict.ci_high, 4)],
            f"[{fmt(verdict.ci_low)}, {fmt(verdict.ci_high)}]",
            f"ragate -c {profile} gate", "95% paired bootstrap, 5000 resamples")
        metrics[f"{name}_status"] = entry(
            verdict.status, verdict.status, f"ragate -c {profile} gate")
        metrics[f"{name}_regressed_queries"] = entry(
            len(verdict.regressed_queries), str(len(verdict.regressed_queries)),
            f"ragate -c {profile} gate")

    # The reranker's gain, read in the other direction against the pre-reranker baseline.
    cfg = load_config("ragate.yaml")
    gain = evaluate_gate(EvalReport.load("baselines/baseline-no-rerank.json"), run(cfg), cfg.gate)
    metrics["reranker_gain_all_queries"] = entry(
        round(gain.delta, 4), fmt(gain.delta),
        "ragate gate --baseline baselines/baseline-no-rerank.json")
    metrics["reranker_gain_interval"] = entry(
        [round(gain.ci_low, 4), round(gain.ci_high, 4)],
        f"[{fmt(gain.ci_low)}, {fmt(gain.ci_high)}]",
        "ragate gate --baseline baselines/baseline-no-rerank.json")


def collect_reports(metrics: dict) -> None:
    tuning = json.loads(Path("docs/tuning_report.json").read_text())
    metrics["bm25_only_all"] = entry(
        tuning["component_baselines"]["bm25"]["all"],
        str(tuning["component_baselines"]["bm25"]["all"]), "python tools/tune_retrieval.py")
    metrics["bm25_only_eval"] = entry(
        tuning["component_baselines"]["bm25"]["eval"],
        str(tuning["component_baselines"]["bm25"]["eval"]), "python tools/tune_retrieval.py")
    metrics["dense_only_all"] = entry(
        tuning["component_baselines"]["dense"]["all"],
        str(tuning["component_baselines"]["dense"]["all"]), "python tools/tune_retrieval.py")
    metrics["selected_dense_weight"] = entry(
        tuning["selected"]["dense_weight"], str(tuning["selected"]["dense_weight"]),
        "python tools/tune_retrieval.py", "zero means bm25 alone won the sweep")

    training = json.loads(Path("docs/reranker_training_report.json").read_text())
    candidates = training["recall_at_5_on_train_candidates"]
    pipeline = training["recall_at_5_full_pipeline"]
    metrics["reranker_train_in_sample"] = entry(
        candidates["reranked_in_sample_optimistic"],
        str(candidates["reranked_in_sample_optimistic"]), "python tools/train_reranker.py")
    metrics["reranker_train_out_of_fold"] = entry(
        candidates["reranked_out_of_fold"], str(candidates["reranked_out_of_fold"]),
        "python tools/train_reranker.py", "GroupKFold by query id, 5 folds")
    metrics["reranker_eval_without"] = entry(
        pipeline["eval_split_without_rerank"], str(pipeline["eval_split_without_rerank"]),
        "python tools/train_reranker.py")
    metrics["reranker_eval_with"] = entry(
        pipeline["eval_split_with_rerank"], str(pipeline["eval_split_with_rerank"]),
        "python tools/train_reranker.py")
    metrics["reranker_candidate_rows"] = entry(
        training["training"]["candidate_documents"],
        str(training["training"]["candidate_documents"]), "python tools/train_reranker.py")

    failed = json.loads(Path("docs/experiments/chunk_level_reranker.json").read_text())
    scores = failed["recall_at_5"]
    metrics["chunk_reranker_eval"] = entry(
        scores["eval_reranked"], str(scores["eval_reranked"]),
        "python experiments/chunk_level_reranker.py", "the rejected chunk-level variant")
    metrics["chunk_reranker_eval_delta"] = entry(
        scores["eval_delta"], str(abs(scores["eval_delta"])),
        "python experiments/chunk_level_reranker.py")
    metrics["chunk_reranker_breadth_coefficient"] = entry(
        failed["coefficients"]["unit_breadth"],
        f"+{failed['coefficients']['unit_breadth']}",
        "python experiments/chunk_level_reranker.py", "the sign that gave the flaw away")

    dedupe = json.loads(Path("benchmark/results/dedupe_effect.json").read_text())
    one_x = {row["dedupe_identical"]: row for row in dedupe["rows"]
             if row["corpus_multiple"] == 1}
    twelve_x = {row["dedupe_identical"]: row for row in dedupe["rows"]
                if row["corpus_multiple"] == 12}
    metrics["dedupe_vectors_saved_pct"] = entry(
        one_x[True]["vectors_saved_pct"], f"{one_x[True]['vectors_saved_pct']}%",
        "python benchmark/bench_dedupe_effect.py")
    metrics["parity_with_duplicates_3x"] = entry(
        next(r["hnsw_score_parity_mean"] for r in dedupe["rows"]
             if r["corpus_multiple"] == 3 and not r["dedupe_identical"]),
        str(next(r["hnsw_score_parity_mean"] for r in dedupe["rows"]
                 if r["corpus_multiple"] == 3 and not r["dedupe_identical"])),
        "python benchmark/bench_dedupe_effect.py")
    metrics["parity_deduped_3x"] = entry(
        next(r["hnsw_score_parity_mean"] for r in dedupe["rows"]
             if r["corpus_multiple"] == 3 and r["dedupe_identical"]),
        str(next(r["hnsw_score_parity_mean"] for r in dedupe["rows"]
                 if r["corpus_multiple"] == 3 and r["dedupe_identical"])),
        "python benchmark/bench_dedupe_effect.py")
    metrics["parity_with_duplicates_12x"] = entry(
        twelve_x[False]["hnsw_score_parity_mean"],
        str(twelve_x[False]["hnsw_score_parity_mean"]),
        "python benchmark/bench_dedupe_effect.py")
    metrics["parity_deduped_12x"] = entry(
        twelve_x[True]["hnsw_score_parity_mean"], str(twelve_x[True]["hnsw_score_parity_mean"]),
        "python benchmark/bench_dedupe_effect.py")

    scaling = json.loads(Path("benchmark/results/retrieval_scaling.json").read_text())
    smallest, largest = scaling["rows"][0], scaling["rows"][-1]
    metrics["exact_p95_smallest"] = entry(
        smallest["flat"]["query_ms"]["p95"], str(smallest["flat"]["query_ms"]["p95"]),
        "python benchmark/bench_retrieval.py")
    metrics["exact_p95_largest"] = entry(
        largest["flat"]["query_ms"]["p95"], str(largest["flat"]["query_ms"]["p95"]),
        "python benchmark/bench_retrieval.py")
    metrics["exact_p99_largest"] = entry(
        largest["flat"]["query_ms"]["p99"], str(largest["flat"]["query_ms"]["p99"]),
        "python benchmark/bench_retrieval.py")
    metrics["largest_index_vectors"] = entry(
        largest["index_units"], f"{largest['index_units']:,}",
        "python benchmark/bench_retrieval.py")


def collect_test_health(metrics: dict, skip: bool) -> None:
    if skip:
        return
    junit = Path("build/junit.xml")
    junit.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--cov=ragate", "--cov-report=xml",
         f"--junitxml={junit}"],
        check=True, capture_output=True,
    )
    coverage = float(ET.parse("coverage.xml").getroot().attrib["line-rate"])
    # pytest writes <testsuites><testsuite tests="N">, so the count is on the child.
    root = ET.parse(junit).getroot()
    suites = root.findall("testsuite") or [root]
    tests = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
    metrics["test_count"] = entry(tests, str(tests), "pytest --junitxml")
    metrics["line_coverage_pct"] = entry(
        round(coverage * 100, 1),
        [f"{coverage * 100:.0f}%", f"{coverage * 100:.1f}%"],
        "pytest --cov=ragate --cov-report=xml")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="docs/metrics.json")
    parser.add_argument("--skip-tests", action="store_true",
                        help="skip the pytest run that measures coverage and test count")
    args = parser.parse_args()

    metrics: dict = {}
    collect_pipeline(metrics)
    collect_gate_outcomes(metrics)
    collect_reports(metrics)
    collect_test_health(metrics, args.skip_tests)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "how_to_use": (
            "Every value here was produced by the command in its source field. "
            "tools/check_readme_numbers.py asserts the README still contains each display "
            "string, so a stale claim fails CI."
        ),
        "metrics": metrics,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out} with {len(metrics)} measured values")


if __name__ == "__main__":
    main()
