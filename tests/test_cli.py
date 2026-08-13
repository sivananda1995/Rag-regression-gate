"""The CLI's exit codes are the contract with CI, so they are asserted directly."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from ragate.cli import EXIT_BASELINE, EXIT_OK, EXIT_REGRESSION, EXIT_USAGE, main


def _profile(tmp_path: Path, corpus: Path, queries: Path, **overrides) -> Path:
    body = {
        "corpus": {"path": str(corpus), "queries": str(queries)},
        "evaluate": {"k": 2, "splits_path": ""},
        "embedder": {"dimensions": 256},
        "rerank": {"enabled": False},
        "gate": {"bootstrap_resamples": 500, "baseline_path": str(tmp_path / "baseline.json")},
        "logging": {"level": "WARNING"},
    }
    for section, values in overrides.items():
        body.setdefault(section, {}).update(values)
    path = tmp_path / "ragate.yaml"
    path.write_text(yaml.safe_dump(body))
    return path


def test_eval_writes_a_report(tmp_path, corpus_files, capsys):
    corpus, queries = corpus_files
    profile = _profile(tmp_path, corpus, queries)
    out = tmp_path / "candidate.json"
    assert main(["-c", str(profile), "eval", "-o", str(out)]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["report"] == str(out)
    assert out.exists()


def test_gate_passes_against_its_own_baseline(tmp_path, corpus_files):
    corpus, queries = corpus_files
    profile = _profile(tmp_path, corpus, queries)
    assert main(["-c", str(profile), "baseline"]) == EXIT_OK
    assert main(["-c", str(profile), "gate"]) == EXIT_OK


def test_gate_fails_when_the_pipeline_degrades(tmp_path, bulk_corpus_files):
    """Recorded with the default BM25 pipeline, then re-run after swapping in an
    sixteen-dimension dense retriever, which is a plausible-looking change that destroys
    retrieval quality."""
    corpus, queries = bulk_corpus_files
    baseline_profile = _profile(tmp_path, corpus, queries)
    assert main(["-c", str(baseline_profile), "baseline"]) == EXIT_OK
    broken = _profile(
        tmp_path,
        corpus,
        queries,
        retriever={"mode": "dense"},
        embedder={"idf_weighting": False, "dimensions": 16},
        gate={"max_absolute_drop": 0.01},
    )
    assert main(["-c", str(broken), "gate", "--html", str(tmp_path / "r.html")]) == EXIT_REGRESSION
    assert (tmp_path / "r.html").exists()


def test_warn_only_reports_the_regression_but_exits_zero(tmp_path, bulk_corpus_files):
    corpus, queries = bulk_corpus_files
    assert main(["-c", str(_profile(tmp_path, corpus, queries)), "baseline"]) == EXIT_OK
    broken = _profile(
        tmp_path, corpus, queries,
        retriever={"mode": "dense"},
        embedder={"idf_weighting": False, "dimensions": 16},
        gate={"max_absolute_drop": 0.01},
    )
    assert main(["-c", str(broken), "gate", "--warn-only"]) == EXIT_OK


def test_missing_baseline_exits_with_the_baseline_code(tmp_path, corpus_files):
    corpus, queries = corpus_files
    profile = _profile(tmp_path, corpus, queries)
    assert main(["-c", str(profile), "gate"]) == EXIT_BASELINE


def test_invalid_config_exits_with_the_usage_code(tmp_path, corpus_files):
    corpus, queries = corpus_files
    profile = _profile(tmp_path, corpus, queries, evaluate={"k": 0})
    assert main(["-c", str(profile), "eval"]) == EXIT_USAGE


def test_report_command_renders_html_from_a_saved_report(tmp_path, corpus_files):
    corpus, queries = corpus_files
    profile = _profile(tmp_path, corpus, queries)
    report_path = tmp_path / "candidate.json"
    assert main(["-c", str(profile), "eval", "-o", str(report_path)]) == EXIT_OK
    html_path = tmp_path / "report.html"
    assert main(["-c", str(profile), "report", "--report", str(report_path),
                 "--html", str(html_path)]) == EXIT_OK
    body = html_path.read_text()
    assert "Retrieval quality report" in body
    assert "<script" not in body
