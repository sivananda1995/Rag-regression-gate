"""Command line entry point.

Exit codes are the contract with CI, so they are explicit and stable:
  0  gate passed (or a non-gate command succeeded)
  1  gate failed: a regression beyond tolerance that is larger than measured noise
  2  usage or configuration error
  3  baseline missing or not comparable
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import load as load_config
from .errors import BaselineError, ConfigError, CorpusError, RagateError
from .evaluate import EvalReport, run
from .gate import FAIL, WARN, evaluate_gate, load_baseline
from .logging_setup import configure, get_logger
from .report import render_html, render_markdown

log = get_logger("ragate.cli")

EXIT_OK = 0
EXIT_REGRESSION = 1
EXIT_USAGE = 2
EXIT_BASELINE = 3


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragate",
        description="Fail a build when RAG retrieval quality regresses on a labeled golden set.",
    )
    parser.add_argument("--version", action="version", version=f"ragate {__version__}")
    parser.add_argument("-c", "--config", default="ragate.yaml", help="YAML profile path")
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("eval", help="evaluate the golden set and write a report")
    p_eval.add_argument("-o", "--out", default="reports/candidate.json")

    p_base = sub.add_parser("baseline", help="record a report as the committed baseline")
    p_base.add_argument("-o", "--out", default=None, help="defaults to gate.baseline_path")

    p_gate = sub.add_parser("gate", help="compare a candidate against the baseline")
    p_gate.add_argument("--candidate", default=None,
                        help="existing report json; if omitted, an evaluation is run now")
    p_gate.add_argument("--baseline", default=None, help="defaults to gate.baseline_path")
    p_gate.add_argument("--markdown", default=None, help="write a markdown summary here")
    p_gate.add_argument("--html", default=None, help="write a self-contained HTML report here")
    p_gate.add_argument("--warn-only", action="store_true",
                        help="report a regression but exit 0 (for a soft rollout of the gate)")

    p_report = sub.add_parser("report", help="render an existing report")
    p_report.add_argument("--report", required=True)
    p_report.add_argument("--html", default=None)
    p_report.add_argument("--markdown", default=None)
    return parser


def _write(path: str | None, content: str) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    log.info("wrote artifact", extra={"path": str(p), "bytes": len(content)})


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        configure()
        log.error("configuration rejected", extra={"error": str(exc)})
        return EXIT_USAGE
    configure(cfg.logging.level, cfg.logging.format)

    try:
        if args.command == "eval":
            report = run(cfg)
            path = report.save(args.out)
            print(json.dumps({"report": str(path), **report.aggregate}, indent=2))
            return EXIT_OK

        if args.command == "baseline":
            out = args.out or cfg.gate.baseline_path
            report = run(cfg)
            path = report.save(out)
            print(f"baseline recorded at {path}")
            print(f"  {cfg.gate.primary_metric} = {report.aggregate[cfg.gate.primary_metric]:.4f}")
            print("  commit this file so the gate has a reference point")
            return EXIT_OK

        if args.command == "gate":
            baseline_report = load_baseline(args.baseline or cfg.gate.baseline_path)
            candidate_report = (
                EvalReport.load(args.candidate) if args.candidate else run(cfg)
            )
            verdict = evaluate_gate(baseline_report, candidate_report, cfg.gate)
            _write(args.markdown, render_markdown(candidate_report, verdict))
            _write(args.html, render_html(candidate_report, verdict))
            print(render_markdown(candidate_report, verdict))
            if verdict.status == FAIL and not args.warn_only:
                return EXIT_REGRESSION
            if verdict.status == FAIL:
                log.warning("regression detected but --warn-only was set, exiting 0")
            elif verdict.status == WARN:
                log.warning(
                    "drop past tolerance but inside noise", extra={"reason": verdict.reason}
                )
            return EXIT_OK

        if args.command == "report":
            report = EvalReport.load(args.report)
            _write(args.html, render_html(report))
            _write(args.markdown, render_markdown(report))
            if not args.html and not args.markdown:
                print(render_markdown(report))
            return EXIT_OK

    except BaselineError as exc:
        log.error("baseline problem", extra={"error": str(exc)})
        return EXIT_BASELINE
    except (ConfigError, CorpusError) as exc:
        log.error("input rejected", extra={"error": str(exc)})
        return EXIT_USAGE
    except RagateError as exc:
        log.error("run failed", extra={"error": str(exc)})
        return EXIT_USAGE

    return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
