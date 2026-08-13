"""Capture the screenshots used in the README from artifacts this repository produced.

Nothing here is a mock-up. The HTML pages are the real reports written by
`ragate gate`, and the terminal image renders the exact bytes that the gate wrote to
stdout and stderr during `tools/run_demo.sh`, read back from reports/.

Prerequisites: python -m pip install playwright && python -m playwright install chromium
Run from the repository root, after tools/run_demo.sh:
    python tools/capture_screenshots.py
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

TERMINAL_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><style>
  body {{ margin: 0; padding: 26px; background: #f1f0ec;
          font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  .window {{ background: #14161a; border-radius: 8px; overflow: hidden;
             box-shadow: 0 6px 22px rgba(0,0,0,0.18); }}
  .bar {{ background: #22262c; padding: 8px 12px; color: #b9bec7; font-size: 11.5px;
          letter-spacing: 0.02em; }}
  .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%;
          margin-right: 6px; vertical-align: -1px; }}
  pre {{ margin: 0; padding: 16px 18px 20px; color: #dfe3ea; font-size: 11.6px;
         line-height: 1.55; white-space: pre-wrap; word-break: break-word; }}
  .cmd {{ color: #7fd18d; }}
  .key {{ color: #8ab4f8; }}
  .warn {{ color: #f0b429; }}
  .bad {{ color: #ff7b72; }}
  .dim {{ color: #8b93a1; }}
</style></head><body>
  <div class="window">
    <div class="bar"><span class="dot" style="background:#ff5f57"></span>
      <span class="dot" style="background:#febc2e"></span>
      <span class="dot" style="background:#28c840"></span>
      {title}</div>
    <pre>{body}</pre>
  </div>
</body></html>
"""


def _terminal_html(title: str, lines: list[str]) -> str:
    rendered = []
    for line in lines:
        escaped = html.escape(line)
        if line.startswith("$"):
            escaped = f'<span class="cmd">{escaped}</span>'
        elif '"level":"ERROR"' in line or "FAIL" in line:
            escaped = f'<span class="bad">{escaped}</span>'
        elif '"level":"WARNING"' in line or "WARN" in line:
            escaped = f'<span class="warn">{escaped}</span>'
        elif line.startswith("{"):
            escaped = f'<span class="dim">{escaped}</span>'
        rendered.append(escaped)
    return TERMINAL_TEMPLATE.format(title=html.escape(title), body="\n".join(rendered))


def _gate_lines(reports: Path, name: str, command: str) -> list[str]:
    stdout = (reports / f"{name}.stdout.txt").read_text().splitlines()
    stderr = (reports / f"{name}.stderr.log").read_text().splitlines()
    verdict_lines = [line for line in stderr if "gate verdict" in line]
    lines = [f"$ {command}"]
    # Keep the structured log lines that carry the decision, then the markdown summary.
    lines += [line[:190] for line in stderr if '"msg":"evaluation complete"' in line]
    lines += [line[:190] for line in verdict_lines]
    lines += stdout[:26]
    if verdict_lines:
        status = json.loads(verdict_lines[-1]).get("status", "")
        lines += ["", "$ echo $?", "1" if status == "FAIL" else "0"]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--out", default="docs/screenshots")
    args = parser.parse_args()
    reports = Path(args.reports)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pages = [
        ("fail-regression.html", "gate_fail_report.png", 1360, True),
        ("warn-borderline.html", "gate_warn_report.png", 1360, False),
        ("pass-reranker-gain.html", "gate_pass_report.png", 1360, False),
        ("fail-no-reranker.html", "gate_no_reranker_report.png", 1360, False),
    ]
    terminal = _terminal_html(
        "ci: retrieval gate step",
        _gate_lines(reports, "fail-regression",
                    "ragate -c configs/candidate-fixed-chunking.yaml gate"),
    )
    terminal_path = out_dir / "_terminal.html"
    terminal_path.write_text(terminal)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            for source, target, width, full in pages:
                path = reports / source
                if not path.exists():
                    print(f"skipping {source}: run tools/run_demo.sh first")
                    continue
                page = browser.new_page(viewport={"width": width, "height": 900},
                                        device_scale_factor=2)
                page.goto(path.resolve().as_uri())
                page.wait_for_load_state("load")
                page.screenshot(path=out_dir / target, full_page=full)
                page.close()
                print(f"wrote {out_dir / target}")

            page = browser.new_page(viewport={"width": 1180, "height": 900},
                                    device_scale_factor=2)
            page.goto(terminal_path.resolve().as_uri())
            page.wait_for_load_state("load")
            page.screenshot(path=out_dir / "ci_gate_run.png", full_page=True)
            page.close()
            print(f"wrote {out_dir / 'ci_gate_run.png'}")
        finally:
            browser.close()
    terminal_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
