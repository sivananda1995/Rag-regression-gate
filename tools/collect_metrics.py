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

# Every file that writes a measured value down in prose. The first version of this tool
# checked the README only, and the numbers duly rotted everywhere else: a config comment
# still quoted a pre-fix recall figure and a module docstring quoted a held-out score that
# had moved by four points. A number is either checked or it drifts, so the check follows
# the numbers instead of the file. See docs/adr/ADR-007-anchored-receipts.md.
CHECKED_DOCUMENTS = [
    "README.md",
    "ragate.yaml",
    "configs/candidate-fixed-chunking.yaml",
    "src/ragate/rerank/features.py",
]

# anchor phrases, as templates with one {} where the measured value belongs. The checker
# substitutes the value and requires the resulting literal string to appear in one of the
# documents above, so the claim is pinned to the sentence that makes it rather than to the
# document as a whole. A metric with a short display string and no anchor is an error:
# asserting that "11" appears somewhere in a long README proves nothing.
ANCHORS: dict[str, list[str]] = {
    "baseline_ndcg_at_5": ["nDCG@5 of {}"],
    "index_units": ["to {} vectors", "chunks down to {}"],
    "chunks": ["{} chunks to", "{} chunks down to"],
    "documents": ["--docs {}"],
    "golden_queries": ["--queries {}", "of the {} golden queries"],
    "baseline_zero_hit": ["{} queries were already scoring zero"],
    "fail_regression_status": ["status {}, with"],
    "fail_regression_regressed_queries": ["{} queries named in the blame table"],
    "fail_regression_blanked_queries": [
        "{} of 140 questions from answered to unanswered",
        "{} questions that had a correct document in the top 5 now have none",
    ],
    "fail_regression_recovered_queries": ["{} that had none now finds one"],
    "fail_regression_candidate_zero_hit": ["{} questions score zero after the refactor"],
    "fail_regression_net_blanked_share": ["a net {} of the golden set"],
    "fail_regression_delta_points": ["costs {} points of recall@5", "%20{}pt%20recall"],
    "warn_borderline_status": ["reported as {} with an instruction"],
    "warn_borderline_regressed_queries": ["{} queries affected. The interval contains zero"],
    "fail_no_reranker_status": ["status {} again"],
    "fail_no_reranker_regressed_queries": ["with {} queries affected"],
    "selected_dense_weight": ["dense weight of {}"],
    "reranker_candidate_rows": ["{} candidate documents"],
    "reranker_eval_without": ["recall@5 {} without it"],
    "reranker_eval_with": ["without it, {} with it"],
    "chunk_reranker_eval": ["down to {}"],
    "dedupe_vectors_saved_pct": ["removed {} of the vectors"],
    "exact_p95_smallest": ["| {} ms |", "p95 rising from {} ms"],
    "exact_p95_largest": ["| {} ms |", "to {} ms at 12,673"],
    "exact_p99_largest": ["| {} ms |"],
    "p99_over_p50_largest": ["p99 is now {}x p50"],
    "test_count": ["badge/tests-{}-", "{} tests, 92% line coverage", "lint, {} tests, and"],
    "line_coverage_pct": ["badge/coverage-{}25-", "{} line coverage"],
}


def entry(value, display, source: str, note: str = "", reproducible: bool = True) -> dict:
    """One measured value, with the command that produced it.

    Anchors are attached centrally by :func:`attach_anchors` from the ANCHORS table above,
    so this stays a description of the measurement and the table stays a single readable
    list of which sentence in which document each number belongs to.
    """
    return {
        "value": value,
        "display": display if isinstance(display, list) else [display],
        "source": source,
        "note": note,
        "reproducible": reproducible,
    }


def attach_anchors(metrics: dict) -> None:
    """Copy the anchor templates onto each metric, and refuse to invent one.

    An anchor naming a metric that no longer exists is a stale check pretending to be a
    live one, so it fails here rather than being ignored.
    """
    unknown = sorted(set(ANCHORS) - set(metrics))
    if unknown:
        raise SystemExit(
            "ANCHORS names metrics that this run did not measure: "
            f"{', '.join(unknown)}. Remove them or fix the metric name."
        )
    for name, record in metrics.items():
        record["anchors"] = ANCHORS.get(name)


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
    # How many queries the default pipeline already answers with nothing correct at all. The
    # baseline's own floor: without it, a candidate's zero count reads as damage the
    # candidate did.
    baseline_zero = sum(1 for q in report.per_query if q.scores["recall_at_k"] == 0.0)
    metrics["baseline_zero_hit"] = entry(
        baseline_zero, str(baseline_zero), "ragate eval",
        "golden queries with no correct document in the top k on the default pipeline")


def collect_gate_outcomes(metrics: dict) -> None:
    baseline = EvalReport.load("baselines/baseline.json")
    scenarios = {
        "fail_regression": "configs/candidate-fixed-chunking.yaml",
        "warn_borderline": "configs/candidate-borderline.yaml",
        "fail_no_reranker": "configs/candidate-no-reranker.yaml",
    }
    for name, profile in scenarios.items():
        cfg = load_config(profile)
        candidate = run(cfg)
        verdict = evaluate_gate(baseline, candidate, cfg.gate)
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
        if name != "fail_regression":
            continue
        # The two halves of an honest impact estimate, registered for the one scenario the
        # README quantifies: questions this profile pushed from some correct document to
        # none, and the ones it accidentally fixed. Quoting only the first number would
        # charge the refactor for queries that were already broken before it landed.
        metrics[f"{name}_blanked_queries"] = entry(
            len(verdict.blanked_queries), str(len(verdict.blanked_queries)),
            f"ragate -c {profile} gate",
            "questions that had a correct document in the top k and now have none")
        metrics[f"{name}_recovered_queries"] = entry(
            len(verdict.recovered_queries), str(len(verdict.recovered_queries)),
            f"ragate -c {profile} gate",
            "questions that had nothing correct and now do")
        zero_hit = sum(
            1 for q in candidate.per_query if q.scores[cfg.gate.primary_metric] == 0.0
        )
        metrics[f"{name}_candidate_zero_hit"] = entry(
            zero_hit, str(zero_hit),
            f"ragate -c {profile} gate",
            "total zero-scoring queries on the candidate, including ones already broken")
        metrics[f"{name}_net_blanked_share"] = entry(
            round(verdict.net_blanked / verdict.queries_compared, 4),
            f"{verdict.net_blanked / verdict.queries_compared:.1%}",
            f"ragate -c {profile} gate",
            "net questions left with no correct document, as a share of the golden set")
        # The headline badge and the summary bullets quote the drop in points rather than
        # in units, so that rounding is measured here too instead of done by hand.
        metrics[f"{name}_delta_points"] = entry(
            round(abs(verdict.delta) * 100, 1), f"{abs(verdict.delta) * 100:.1f}",
            f"ragate -c {profile} gate", "the same delta expressed in points of recall@5")

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
        fmt(tuning["component_baselines"]["bm25"]["all"]), "python tools/tune_retrieval.py")
    metrics["bm25_only_eval"] = entry(
        tuning["component_baselines"]["bm25"]["eval"],
        fmt(tuning["component_baselines"]["bm25"]["eval"]), "python tools/tune_retrieval.py")
    metrics["dense_only_all"] = entry(
        tuning["component_baselines"]["dense"]["all"],
        fmt(tuning["component_baselines"]["dense"]["all"]), "python tools/tune_retrieval.py")
    metrics["selected_dense_weight"] = entry(
        tuning["selected"]["dense_weight"], str(tuning["selected"]["dense_weight"]),
        "python tools/tune_retrieval.py", "zero means bm25 alone won the sweep")

    training = json.loads(Path("docs/reranker_training_report.json").read_text())
    candidates = training["recall_at_5_on_train_candidates"]
    pipeline = training["recall_at_5_full_pipeline"]
    metrics["reranker_train_in_sample"] = entry(
        candidates["reranked_in_sample_optimistic"],
        fmt(candidates["reranked_in_sample_optimistic"]), "python tools/train_reranker.py")
    metrics["reranker_train_out_of_fold"] = entry(
        candidates["reranked_out_of_fold"], fmt(candidates["reranked_out_of_fold"]),
        "python tools/train_reranker.py", "GroupKFold by query id, 5 folds")
    metrics["reranker_eval_without"] = entry(
        pipeline["eval_split_without_rerank"], fmt(pipeline["eval_split_without_rerank"]),
        "python tools/train_reranker.py")
    metrics["reranker_eval_with"] = entry(
        pipeline["eval_split_with_rerank"], fmt(pipeline["eval_split_with_rerank"]),
        "python tools/train_reranker.py")
    metrics["reranker_candidate_rows"] = entry(
        training["training"]["candidate_documents"],
        str(training["training"]["candidate_documents"]), "python tools/train_reranker.py")

    failed = json.loads(Path("docs/experiments/chunk_level_reranker.json").read_text())
    scores = failed["recall_at_5"]
    metrics["chunk_reranker_eval"] = entry(
        scores["eval_reranked"], fmt(scores["eval_reranked"]),
        "python experiments/chunk_level_reranker.py", "the rejected chunk-level variant")
    metrics["chunk_reranker_eval_delta"] = entry(
        scores["eval_delta"], fmt(abs(scores["eval_delta"])),
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
    # The tail-to-median ratio is a claim about the latency distribution's shape, so it is
    # divided here rather than in prose. A full sort has none of the branch-dependent
    # variance a partition does, and this is the number that shows it.
    tail_ratio = largest["flat"]["query_ms"]["p99"] / largest["flat"]["query_ms"]["p50"]
    metrics["p99_over_p50_largest"] = entry(
        round(tail_ratio, 2), f"{tail_ratio:.2f}", "python benchmark/bench_retrieval.py",
        "how much heavier the tail is than the median at the largest index size")


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
    attach_anchors(metrics)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "how_to_use": (
            "Every value here was produced by the command in its source field. "
            "tools/check_readme_numbers.py asserts that each value still appears in the "
            "documents listed under checked_documents, and that values with an anchor "
            "appear inside that exact phrase, so a stale claim fails CI."
        ),
        "checked_documents": CHECKED_DOCUMENTS,
        "metrics": metrics,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out} with {len(metrics)} measured values")


if __name__ == "__main__":
    main()
