"""Record the demo video and GIF from real command output.

How this stays honest. Every line of terminal text in the video is the actual stdout and
stderr of the command shown above it, captured by running that command here, and the
reveal speed of each segment is paced by the command's measured wall-clock time. What the
video is not is a live screen recording: it is a replay, the same way an asciinema cast is
a replay. The README says so next to the embedded file.

Pipeline: run the commands and capture output, generate a self-contained HTML player that
replays the capture, let Playwright record that page to webm, then transcode with ffmpeg to
an MP4 for sharing and an optimised GIF that autoplays inline on GitHub.

Prerequisites: playwright with chromium, and ffmpeg on PATH.
Run from the repository root: python tools/record_demo.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

# Each segment is one caption plus one real command. max_lines trims long output to the
# part worth reading on screen; the trim is shown as an explicit marker, never silently.
SEGMENTS = [
    {
        "caption": "A RAG pipeline regresses silently. Unit tests pass, the service still "
                   "returns 200, and users stop finding answers.",
        "command": "ragate eval -o reports/video-candidate.json",
        "grep": None,
        "max_lines": 9,
        "hold_ms": 900,
    },
    {
        "caption": "Here is a plausible pull request: replace sentence-window chunking with "
                   "fixed 240-character slices. Six lines, inherits everything else.",
        "command": "cat configs/candidate-fixed-chunking.yaml",
        "grep": None,
        "max_lines": 14,
        "hold_ms": 1400,
    },
    {
        "caption": "The gate scores 140 labeled queries, blames the queries that lost their "
                   "document, and blocks the merge. Exit code 1.",
        "command": "ragate -c configs/candidate-fixed-chunking.yaml gate; echo \"exit=$?\"",
        "grep": None,
        "max_lines": 22,
        "hold_ms": 2600,
    },
    {
        "caption": "It also refuses to cry wolf. This drop is past the tolerance, but the "
                   "bootstrap interval contains zero, so the build is not blocked.",
        "command": "ragate -c configs/candidate-borderline.yaml gate | head -6; "
                   "echo \"exit=${PIPESTATUS[0]}\"",
        "grep": None,
        "max_lines": 10,
        "hold_ms": 2400,
    },
    {
        "caption": "And it confirms improvements instead of arguing about them. This is how "
                   "the reranker earned its place in the default config.",
        "command": "ragate gate --baseline baselines/baseline-no-rerank.json | head -5",
        "grep": None,
        "max_lines": 8,
        "hold_ms": 2400,
    },
    {
        "caption": "Every number in the README is re-measured on every push, and CI fails if "
                   "the document and the measurements disagree.",
        "command": "python tools/check_readme_numbers.py",
        "grep": None,
        "max_lines": 6,
        "hold_ms": 2200,
    },
]

PLAYER = """<!doctype html>
<html><head><meta charset="utf-8"><style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; width: 1280px; height: 720px; background: #eceae4; overflow: hidden;
         font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
  .stage {{ padding: 22px 26px 0; height: 100%; display: flex; flex-direction: column; }}
  .title {{ display: flex; align-items: baseline; gap: 12px; margin-bottom: 10px; }}
  .title b {{ font-size: 19px; color: #0b0b0b; letter-spacing: -0.01em; }}
  .title span {{ font-size: 13px; color: #52514e; }}
  .caption {{ min-height: 52px; font-size: 16.5px; line-height: 1.45; color: #0b0b0b;
              background: #fff; border-left: 3px solid #2a78d6; padding: 9px 14px;
              margin-bottom: 12px; opacity: 0; transition: opacity 220ms ease; }}
  .caption.on {{ opacity: 1; }}
  .term {{ flex: 1; background: #14161a; border-radius: 8px; overflow: hidden;
           box-shadow: 0 8px 26px rgba(0,0,0,0.16); display: flex; flex-direction: column; }}
  .bar {{ background: #22262c; padding: 7px 12px; color: #b9bec7; font-size: 11.5px;
          flex: none; }}
  .dot {{ display: inline-block; width: 9px; height: 9px; border-radius: 50%;
          margin-right: 5px; vertical-align: -1px; }}
  pre {{ margin: 0; padding: 13px 16px; color: #dfe3ea; font-size: 13.2px; line-height: 1.5;
         font-family: ui-monospace, SFMono-Regular, Menlo, monospace; white-space: pre-wrap;
         word-break: break-word; flex: 1; overflow: hidden; }}
  .cmd {{ color: #7fd18d; }}
  .bad {{ color: #ff7b72; }} .warn {{ color: #f0b429; }} .ok {{ color: #7fd18d; }}
  .dim {{ color: #8b93a1; }}
  .cursor {{ display: inline-block; width: 8px; height: 15px; background: #dfe3ea;
             vertical-align: -2px; animation: blink 1s steps(1) infinite; }}
  @keyframes blink {{ 50% {{ opacity: 0; }} }}
  .foot {{ flex: none; padding: 8px 2px 10px; font-size: 11.5px; color: #7c7a75; }}
</style></head>
<body><div class="stage">
  <div class="title"><b>rag-regression-gate</b>
    <span>CI gate for RAG retrieval quality &middot; replay of a real captured session</span></div>
  <div class="caption" id="cap"></div>
  <div class="term">
    <div class="bar"><span class="dot" style="background:#ff5f57"></span>
      <span class="dot" style="background:#febc2e"></span>
      <span class="dot" style="background:#28c840"></span> bash</div>
    <pre id="out"></pre>
  </div>
  <div class="foot" id="foot"></div>
</div>
<script>
const SEGMENTS = {segments};
const out = document.getElementById('out');
const cap = document.getElementById('cap');
const foot = document.getElementById('foot');
const sleep = ms => new Promise(r => setTimeout(r, ms));
function classify(line) {{
  if (line.startsWith('$ ')) return 'cmd';
  if (line.includes('FAIL') || line.includes('exit=1')) return 'bad';
  if (line.includes('WARN')) return 'warn';
  if (line.includes('PASS') || line.includes('matches a value')) return 'ok';
  if (line.startsWith('{{')) return 'dim';
  return '';
}}
function append(line) {{
  const span = document.createElement('span');
  const cls = classify(line);
  if (cls) span.className = cls;
  span.textContent = line + '\\n';
  out.appendChild(span);
  // Keep the newest output visible in a fixed-height terminal.
  while (out.scrollHeight > out.clientHeight && out.firstChild) out.removeChild(out.firstChild);
}}
(async () => {{
  for (const seg of SEGMENTS) {{
    cap.classList.remove('on');
    await sleep(160);
    cap.textContent = seg.caption;
    cap.classList.add('on');
    foot.textContent = 'measured wall time ' + seg.duration_s.toFixed(2) + ' s';
    await sleep(560);
    // Type the command.
    let typed = '$ ';
    const cursor = '<span class="cursor"></span>';
    const holder = document.createElement('span');
    holder.className = 'cmd';
    out.appendChild(holder);
    for (const ch of seg.command) {{
      typed += ch;
      holder.innerHTML = typed + cursor;
      await sleep(16);
    }}
    holder.innerHTML = typed + '\\n';
    await sleep(320);
    // Reveal the captured output, paced by how long the command really took.
    const lineCount = Math.max(seg.lines.length, 1);
    const perLine = Math.max(26, Math.min(150, (seg.duration_s * 1000) / lineCount));
    for (const line of seg.lines) {{ append(line); await sleep(perLine); }}
    await sleep(seg.hold_ms);
    out.textContent = '';
  }}
  cap.classList.remove('on');
  await sleep(400);
  cap.textContent = 'github.com/USERNAME/rag-regression-gate';
  cap.classList.add('on');
  foot.textContent = 'every number in the README is re-measured by CI';
  await sleep(1800);
  window.__done = true;
}})();
</script></body></html>
"""


def capture(segment: dict) -> dict:
    """Run one command for real, keeping its output and how long it took."""
    started = time.perf_counter()
    completed = subprocess.run(
        ["bash", "-c", segment["command"]], capture_output=True, text=True
    )
    duration = time.perf_counter() - started
    stream = (completed.stdout or "") + (completed.stderr or "")
    lines = [line.rstrip() for line in stream.splitlines() if line.strip()]
    limit = segment["max_lines"]
    if len(lines) > limit:
        trimmed = len(lines) - limit
        lines = lines[:limit] + [f"... {trimmed} more lines, full output in reports/"]
    return {
        "caption": segment["caption"],
        "command": segment["command"],
        "lines": lines,
        "duration_s": round(duration, 3),
        "hold_ms": segment["hold_ms"],
        "exit_code": completed.returncode,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="docs/video")
    parser.add_argument("--gif-width", type=int, default=900)
    parser.add_argument("--keep-webm", action="store_true")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required: apt-get install ffmpeg")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    captured = []
    for segment in SEGMENTS:
        result = capture(segment)
        captured.append(result)
        print(f"captured: {result['command'][:64]:<64} "
              f"{result['duration_s']:>6.2f}s  exit={result['exit_code']}  "
              f"{len(result['lines'])} lines")

    player = out_dir / "_player.html"
    player.write_text(PLAYER.format(segments=json.dumps(captured, indent=1)))

    raw_dir = out_dir / "_raw"
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(
            viewport={"width": 1280, "height": 720},
            record_video_dir=str(raw_dir),
            record_video_size={"width": 1280, "height": 720},
        )
        page = context.new_page()
        page.goto(player.resolve().as_uri())
        page.wait_for_function("window.__done === true", timeout=180_000)
        context.close()
        browser.close()

    webm = next(raw_dir.glob("*.webm"))
    mp4 = out_dir / "gate-demo.mp4"
    gif = out_dir / "gate-demo.gif"
    palette = out_dir / "_palette.png"

    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(webm),
         "-c:v", "libx264", "-crf", "23", "-preset", "slow", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(mp4)],
        check=True,
    )
    # Two-pass GIF: a shared palette keeps the terminal text readable at 256 colours.
    common = f"fps=10,scale={args.gif_width}:-1:flags=lanczos"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(webm),
         "-vf", f"{common},palettegen=stats_mode=diff", str(palette)],
        check=True,
    )
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(webm),
         "-i", str(palette), "-lavfi",
         f"{common}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle",
         str(gif)],
        check=True,
    )

    palette.unlink(missing_ok=True)
    player.unlink(missing_ok=True)
    if not args.keep_webm:
        shutil.rmtree(raw_dir)

    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps({
        "note": "Terminal text is the real stdout and stderr of each command, captured by "
                "tools/record_demo.py. Reveal speed is paced by each command's measured "
                "wall time. This is a replay, not a live screen recording.",
        "segments": [
            {"command": s["command"], "exit_code": s["exit_code"],
             "measured_wall_time_s": s["duration_s"]}
            for s in captured
        ],
    }, indent=2) + "\n")

    for path in (mp4, gif):
        print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
