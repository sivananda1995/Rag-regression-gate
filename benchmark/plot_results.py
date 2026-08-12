"""Render the benchmark result files as labeled PNG charts for the README.

Every value plotted is read from benchmark/results/*.json, which are written by the
two benchmark scripts. Nothing here computes or adjusts a measurement.

Run from the repository root: python benchmark/plot_results.py
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

SURFACE = "#fcfcfb"
PANEL = "#ffffff"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#7c7a75"
GRID = "#e4e3df"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": PANEL,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.titleweight": "600",
        "axes.labelsize": 9.5,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 160,
    }
)


def _caption(fig, text: str, width: int = 118) -> None:
    """Place the test conditions under the plot, wrapped so nothing runs off the page.

    Every chart in this repository carries its conditions; a latency number without
    them is not a measurement, it is a rumour.
    """
    wrapped = textwrap.fill(text, width=width)
    fig.text(0.008, 0.012, wrapped, fontsize=7.4, color=MUTED, va="bottom")


def _thousands(value, _pos):
    return f"{value / 1000:.0f}k" if value >= 1000 else f"{value:.0f}"


def latency_chart(results_dir: Path, out_dir: Path) -> Path:
    payload = json.loads((results_dir / "retrieval_scaling.json").read_text())
    rows = payload["rows"]
    units = [row["index_units"] for row in rows]
    conditions = payload["conditions"]

    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    for label, key, color in (
        ("p50", "p50", BLUE),
        ("p95", "p95", ORANGE),
        ("p99", "p99", AQUA),
    ):
        values = [row["flat"]["query_ms"][key] for row in rows]
        ax.plot(units, values, color=color, linewidth=2.0, marker="o", markersize=5.5,
                markeredgecolor=PANEL, markeredgewidth=1.4, label=f"exact search {label}",
                zorder=3)
        ax.annotate(
            f"{values[-1]:.2f} ms",
            xy=(units[-1], values[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=color,
            fontsize=8.5,
            va="center",
        )
    hnsw = [row.get("faiss_hnsw", {}).get("query_ms", {}).get("p95") for row in rows]
    if all(value is not None for value in hnsw):
        ax.plot(units, hnsw, color=MUTED, linewidth=1.6, linestyle=(0, (4, 3)),
                marker="s", markersize=4.5, label="hnsw p95", zorder=2)

    ax.set_title("Single-query retrieval latency as the index grows")
    ax.set_xlabel("vectors in the index (deduplicated chunks)")
    ax.set_ylabel("latency per query (milliseconds)")
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(FuncFormatter(_thousands))
    ax.set_xlim(0, max(units) * 1.12)
    ax.set_ylim(0, max(row["flat"]["query_ms"]["p99"] for row in rows) * 1.18)
    ax.legend(loc="upper left", ncols=2)
    _caption(
        fig,
        f"{conditions['queries_per_pass']} golden queries x {conditions['passes']} passes, "
        f"one query at a time, k={conditions['k_chunks_retrieved']} chunks, "
        f"{conditions['dimensions']}-dim vectors, {conditions['cpu_count']} vCPU, "
        f"python {conditions['python']}. Corpus padded with tenant variants to reach each "
        f"size; quality numbers elsewhere use the unpadded labeled corpus.",
        width=104,
    )
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    path = out_dir / "latency_scaling.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def dedupe_chart(results_dir: Path, out_dir: Path) -> Path:
    payload = json.loads((results_dir / "dedupe_effect.json").read_text())
    rows = payload["rows"]
    multiples = sorted({row["corpus_multiple"] for row in rows})
    labels = [f"{m}x corpus" for m in multiples]

    def series(dedupe: bool, field):
        return [
            field(next(r for r in rows if r["corpus_multiple"] == m
                       and r["dedupe_identical"] is dedupe))
            for m in multiples
        ]

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    positions = range(len(multiples))
    width = 0.36

    off_units = series(False, lambda r: r["index_units"])
    on_units = series(True, lambda r: r["index_units"])
    left.bar([p - width / 2 for p in positions], off_units, width, color=ORANGE,
             label="every chunk indexed", zorder=3)
    left.bar([p + width / 2 for p in positions], on_units, width, color=BLUE,
             label="identical text indexed once", zorder=3)
    for position, value in zip(positions, off_units, strict=True):
        left.annotate(f"{value:,}", (position - width / 2, value), ha="center",
                      va="bottom", fontsize=8, color=INK2, xytext=(0, 2),
                      textcoords="offset points")
    for position, value in zip(positions, on_units, strict=True):
        left.annotate(f"{value:,}", (position + width / 2, value), ha="center",
                      va="bottom", fontsize=8, color=INK2, xytext=(0, 2),
                      textcoords="offset points")
    left.set_title("Vectors in the index")
    left.set_ylabel("vectors")
    left.set_xticks(list(positions), labels)
    left.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    left.set_axisbelow(True)
    left.set_ylim(0, max(off_units) * 1.16)
    left.legend(loc="upper left")

    off_loss = [100 * (1 - v) for v in series(False, lambda r: r["hnsw_score_parity_mean"])]
    on_loss = [100 * (1 - v) for v in series(True, lambda r: r["hnsw_score_parity_mean"])]
    right.bar([p - width / 2 for p in positions], off_loss, width, color=ORANGE, zorder=3)
    right.bar([p + width / 2 for p in positions], on_loss, width, color=BLUE, zorder=3)
    for position, value in zip(positions, off_loss, strict=True):
        right.annotate(f"{value:.1f}%", (position - width / 2, value), ha="center",
                       va="bottom", fontsize=8, color=INK2, xytext=(0, 2),
                       textcoords="offset points")
    for position, value in zip(positions, on_loss, strict=True):
        right.annotate(f"{value:.2f}%", (position + width / 2, value), ha="center",
                       va="bottom", fontsize=8, color=INK2, xytext=(0, 2),
                       textcoords="offset points")
    right.set_title("Quality lost by the approximate index")
    right.set_ylabel("similarity mass missed vs exact (%)")
    right.set_xticks(list(positions), labels)
    right.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    right.set_axisbelow(True)
    right.set_ylim(0, max(off_loss) * 1.2)

    _caption(
        fig,
        "Same chunker, embedder, and hnsw parameters in every bar; the only variable is "
        "whether byte-identical chunk text is indexed once or many times. "
        f"{payload['conditions']['queries']} golden queries, "
        f"k={payload['conditions']['k_chunks_retrieved']} chunks retrieved. Quality lost is "
        "1 minus retrieved-score parity: the true cosine similarity of what the "
        "approximate index returned, against the exact top-k optimum.",
        width=130,
    )
    fig.tight_layout(rect=(0, 0.115, 1, 1))
    path = out_dir / "dedupe_effect.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="benchmark/results")
    parser.add_argument("--out", default="docs/screenshots")
    args = parser.parse_args()
    results_dir = Path(args.results)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in (latency_chart(results_dir, out_dir), dedupe_chart(results_dir, out_dir)):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
