"""Human-facing renderings of an evaluation report and a gate verdict.

Two audiences, two formats. Markdown goes into the pull request as a job summary,
because that is where the reviewer already is. HTML is a single self-contained file
with no external assets, so it survives being uploaded as a CI artifact and opened
from a browser with no server.
"""

from __future__ import annotations

import html
from typing import Iterable

from .evaluate import EvalReport
from .gate import FAIL, PASS, WARN, GateVerdict, QueryDelta

_STATUS_NOTE = {
    PASS: "no regression beyond tolerance",
    WARN: "drop exceeds tolerance but is inside this golden set's noise",
    FAIL: "regression confirmed, build blocked",
}


def _fmt(value: float, places: int = 4) -> str:
    return f"{value:.{places}f}"


def render_markdown(report: EvalReport, verdict: GateVerdict | None = None) -> str:
    lines: list[str] = []
    if verdict is not None:
        lines += [
            f"## Retrieval gate: {verdict.status}",
            "",
            f"{verdict.reason}.",
            "",
            f"| {verdict.metric} | value |",
            "| --- | --- |",
            f"| baseline | {_fmt(verdict.baseline)} |",
            f"| candidate | {_fmt(verdict.candidate)} |",
            f"| delta | {verdict.delta:+.4f} |",
            f"| tolerance | {_fmt(verdict.tolerance)} |",
            f"| {int(verdict.confidence * 100)}% paired-bootstrap interval | "
            f"[{_fmt(verdict.ci_low)}, {_fmt(verdict.ci_high)}] |",
            f"| queries compared | {verdict.queries_compared} |",
            "",
        ]
    lines += [
        "### Metrics on this run",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for name, value in report.aggregate.items():
        lines.append(f"| {name} | {_fmt(value)} |")
    lines += [
        f"| recall_at_k ceiling for this golden set | "
        f"{report.corpus_stats['recall_at_k_ceiling']} |",
        "",
        f"Corpus: {report.corpus_stats['documents']} documents, "
        f"{report.corpus_stats['chunks']} chunks "
        f"({report.corpus_stats['chunks_per_document']} per document), "
        f"{report.corpus_stats['queries']} golden queries. "
        f"Wall time {report.timings_ms['total']} ms.",
        "",
    ]
    if verdict is not None and verdict.regressed_queries:
        lines += [
            f"### Queries that lost ground ({len(verdict.regressed_queries)})",
            "",
            "| query | before | after | documents that fell out of top-k |",
            "| --- | --- | --- | --- |",
        ]
        for delta in verdict.regressed_queries[:15]:
            lost = ", ".join(delta.lost_doc_ids[:4]) or "none"
            lines.append(
                f"| `{delta.query_id}` {delta.text[:70]} | {_fmt(delta.baseline, 2)} "
                f"| {_fmt(delta.candidate, 2)} | {lost} |"
            )
        lines.append("")
    return "\n".join(lines)


def _rows(deltas: Iterable[QueryDelta], limit: int) -> str:
    out: list[str] = []
    for delta in list(deltas)[:limit]:
        out.append(
            "<tr>"
            f"<td class='mono'>{html.escape(delta.query_id)}</td>"
            f"<td>{html.escape(delta.text)}</td>"
            f"<td class='num'>{_fmt(delta.baseline, 2)}</td>"
            f"<td class='num'>{_fmt(delta.candidate, 2)}</td>"
            f"<td class='num neg'>{delta.delta:+.2f}</td>"
            f"<td class='mono small'>{html.escape(', '.join(delta.lost_doc_ids[:5])) or '-'}</td>"
            f"<td class='mono small'>{html.escape(', '.join(delta.gained_doc_ids[:5])) or '-'}</td>"
            "</tr>"
        )
    return "\n".join(out)


def render_html(report: EvalReport, verdict: GateVerdict | None = None) -> str:
    """Self-contained HTML report. No external CSS, fonts, or scripts."""
    status = verdict.status if verdict else "REPORT"
    status_class = {PASS: "ok", WARN: "warn", FAIL: "bad"}.get(status, "ok")
    metric_rows = "\n".join(
        f"<tr><td>{html.escape(name)}</td><td class='num'>{_fmt(value)}</td></tr>"
        for name, value in report.aggregate.items()
    )
    timing_rows = "\n".join(
        f"<tr><td>{html.escape(stage)}</td><td class='num'>{value:.1f}</td></tr>"
        for stage, value in report.timings_ms.items()
    )

    if verdict is not None:
        headline = (
            f"<div class='tiles'>"
            f"<div class='tile'><div class='label'>baseline {html.escape(verdict.metric)}</div>"
            f"<div class='value'>{_fmt(verdict.baseline, 3)}</div></div>"
            f"<div class='tile'><div class='label'>candidate</div>"
            f"<div class='value'>{_fmt(verdict.candidate, 3)}</div></div>"
            f"<div class='tile'><div class='label'>delta</div>"
            f"<div class='value {'neg' if verdict.delta < 0 else 'pos'}'>"
            f"{verdict.delta:+.3f}</div></div>"
            f"<div class='tile'><div class='label'>"
            f"{int(verdict.confidence * 100)}% paired bootstrap</div>"
            f"<div class='value small'>[{_fmt(verdict.ci_low, 3)}, "
            f"{_fmt(verdict.ci_high, 3)}]</div></div>"
            f"<div class='tile'><div class='label'>queries compared</div>"
            f"<div class='value'>{verdict.queries_compared}</div></div>"
            f"</div>"
            f"<p class='reason'>{html.escape(verdict.reason)}.</p>"
        )
        blame = (
            "<h2>Queries that lost ground</h2>"
            "<p class='hint'>Per-query blame. The last two columns are the documents that "
            "left and entered the top-k, which is what turns a number into a fix.</p>"
            "<table><thead><tr><th>query id</th><th>query</th><th>before</th><th>after</th>"
            "<th>delta</th><th>fell out of top-k</th><th>entered top-k</th></tr></thead>"
            f"<tbody>{_rows(verdict.regressed_queries, 25)}</tbody></table>"
            if verdict.regressed_queries
            else "<h2>Queries that lost ground</h2><p class='hint'>None past the blame "
            "threshold.</p>"
        )
    else:
        headline = ""
        blame = ""

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>ragate retrieval report {html.escape(status)}</title>
<style>
  :root {{
    --surface: #fcfcfb; --panel: #ffffff; --line: #e4e3df;
    --ink: #0b0b0b; --ink2: #52514e; --muted: #7c7a75;
    --blue: #2a78d6; --red: #e34948; --amber: #eda100; --green: #008300;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 32px 36px 44px; background: var(--surface); color: var(--ink);
    font: 14px/1.5 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
  h1 {{ font-size: 20px; margin: 0 0 2px; letter-spacing: -0.01em; }}
  h2 {{ font-size: 15px; margin: 30px 0 8px; }}
  .sub {{ color: var(--ink2); font-size: 12.5px; margin: 0 0 20px; }}
  .badge {{ display: inline-block; padding: 3px 10px; border-radius: 4px; font-weight: 650;
    font-size: 12px; letter-spacing: 0.04em; color: #fff; margin-left: 10px;
    vertical-align: 3px; }}
  .badge.ok {{ background: var(--green); }}
  .badge.warn {{ background: var(--amber); color: #241a00; }}
  .badge.bad {{ background: var(--red); }}
  .tiles {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 18px 0 6px; }}
  .tile {{ background: var(--panel); border: 1px solid var(--line); border-radius: 6px;
    padding: 12px 14px; min-width: 148px; }}
  .tile .label {{ color: var(--muted); font-size: 11.5px; text-transform: uppercase;
    letter-spacing: 0.05em; }}
  .tile .value {{ font-size: 22px; font-weight: 600; margin-top: 4px;
    font-variant-numeric: tabular-nums; }}
  .tile .value.small {{ font-size: 15px; padding-top: 6px; }}
  .neg {{ color: var(--red); }} .pos {{ color: var(--green); }}
  .reason {{ color: var(--ink2); margin: 14px 0 0; max-width: 78ch; }}
  .hint {{ color: var(--muted); font-size: 12.5px; margin: 0 0 10px; max-width: 86ch; }}
  table {{ border-collapse: collapse; width: 100%; background: var(--panel);
    border: 1px solid var(--line); border-radius: 6px; overflow: hidden; }}
  th, td {{ text-align: left; padding: 7px 11px; border-bottom: 1px solid var(--line);
    font-size: 12.5px; }}
  th {{ background: #f6f5f2; color: var(--ink2); font-weight: 600; font-size: 11.5px;
    text-transform: uppercase; letter-spacing: 0.04em; }}
  tr:last-child td {{ border-bottom: none; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; width: 88px; }}
  td.mono, .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  .small {{ font-size: 11.5px; color: var(--ink2); }}
  .cols {{ display: flex; gap: 22px; align-items: flex-start; }}
  .cols > div {{ flex: 1; }}
  footer {{ color: var(--muted); font-size: 11.5px; margin-top: 28px; }}
</style></head>
<body>
  <h1>Retrieval quality report<span class="badge {status_class}">{html.escape(status)}</span></h1>
  <p class="sub">{html.escape(_STATUS_NOTE.get(status, 'evaluation only, no baseline comparison'))}
    &nbsp;&middot;&nbsp; k = {report.k}
    &nbsp;&middot;&nbsp; {report.corpus_stats['documents']} documents,
    {report.corpus_stats['chunks']} chunks,
    {report.corpus_stats['queries']} golden queries
    &nbsp;&middot;&nbsp; ragate {html.escape(report.ragate_version)}
    &nbsp;&middot;&nbsp; {html.escape(report.generated_at)}</p>
  {headline}
  <div class="cols">
    <div>
      <h2>Metrics</h2>
      <table><thead><tr><th>metric</th><th class="num">value</th></tr></thead>
      <tbody>{metric_rows}
      <tr><td>recall_at_k ceiling</td>
      <td class="num">{report.corpus_stats['recall_at_k_ceiling']}</td></tr></tbody></table>
    </div>
    <div>
      <h2>Stage timings (ms)</h2>
      <table><thead><tr><th>stage</th><th class="num">ms</th></tr></thead>
      <tbody>{timing_rows}</tbody></table>
    </div>
  </div>
  {blame}
  <footer>Chunking {html.escape(str(report.config['chunking']['strategy']))}
    {report.config['chunking']['target_chars']}/{report.config['chunking']['overlap_chars']} chars
    &nbsp;&middot;&nbsp; embedder {html.escape(str(report.config['embedder']['provider']))}
    d={report.config['embedder']['dimensions']}
    idf={str(report.config['embedder']['idf_weighting']).lower()}
    &nbsp;&middot;&nbsp; index {html.escape(str(report.config['index']['backend']))}
    &nbsp;&middot;&nbsp; python {html.escape(report.environment.get('python', 'unknown'))}</footer>
</body></html>
"""
