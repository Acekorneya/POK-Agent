"""Charts and a summary of Windows Agent Arena runs for the README.

Reads the runs under ~/arena/runs (or $POKAI_ARENA_RUNS) and writes:

    docs/assets/arena-success.svg     tasks passed per run
    docs/assets/arena-calls.svg       planner LLM calls per task per run
    docs/assets/arena-system1.svg     share of on-screen actions done by System 1
    docs/assets/muscle-memory.svg     planner calls, first vs second attempt, by replay
    docs/results/arena-summary.json   the numbers behind the charts

    python3 scripts/arena/make_report.py

The SVGs follow GitHub's light and dark themes. Only full 154-task passes are
charted, each over every task of that pass.
"""

from __future__ import annotations

import collections
import glob
import json
import os
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = Path(os.environ.get("POKAI_ARENA_RUNS", Path.home() / "arena" / "runs"))
ASSETS = REPO / "docs" / "assets"
RESULTS = REPO / "docs" / "results"

# (tag, pass, label, note)
FULL_RUNS = [
    ("base1", 1, "Baseline", "no step cap"),
    ("v2", 2, "v2", "skills as advice"),
    ("v4", 1, "v4", "skill replay"),
    ("v5", 1, "v5 · 1st", "first look, System 1 plans"),
    ("v5", 2, "v5 · 2nd", "muscle memory"),
]
PLANNER_ACTIONS = {"click_target", "click_localized", "type_text", "simulate_input",
                   "execute_action_batch", "scroll_view", "drag_target", "hover_target"}


def rows(tag: str, pass_index: int) -> list[dict]:
    path = RUNS / tag / "results.jsonl"
    return [row for row in map(json.loads, path.open(encoding="utf-8")) if row["pass"] == pass_index]


def trace(tag: str, run_id: str | None) -> Path | None:
    if not run_id:
        return None
    found = glob.glob(str(RUNS / tag / "*" / "share" / "out" / run_id / "trace.jsonl"))
    return Path(found[0]) if found else None


def run_metrics(tag: str, pass_index: int) -> dict:
    data = rows(tag, pass_index)
    system1 = planner = 0
    for row in data:
        path = trace(tag, row["run_id"])
        if path is None:
            continue
        for line in path.open(encoding="utf-8", errors="ignore"):
            if '"kind":"fast_actions_step"' in line:
                system1 += 1
            elif '"kind":"tool_result"' in line:
                if json.loads(line)["payload"].get("name") in PLANNER_ACTIONS:
                    planner += 1
    calls = [row["llm_requests"] for row in data if row.get("llm_requests") is not None]
    return {
        "tasks": len(data),
        "passed": round(sum(row["score"] or 0 for row in data), 1),
        "calls_mean": round(statistics.mean(calls), 1),
        "calls_median": statistics.median(calls),
        "system1_share": round(100 * system1 / max(1, system1 + planner)),
        "seconds_mean": round(statistics.mean(row.get("seconds") or 0 for row in data)),
    }


def muscle_memory(tag: str = "v5") -> dict:
    """Second attempt vs first on the same tasks, grouped by what System 1
    replayed on the second attempt."""
    first = {(r["domain"], r["task"]): r for r in rows(tag, 1)}
    second = {(r["domain"], r["task"]): r for r in rows(tag, 2)}
    groups: dict[str, list] = collections.defaultdict(list)
    for key, row in second.items():
        if key not in first:
            continue
        replay = None
        path = trace(tag, row["run_id"])
        if path is not None:
            for line in path.open(encoding="utf-8", errors="ignore"):
                if '"kind":"skill_replay"' in line:
                    replay = json.loads(line)["payload"]
                    break
        if replay and replay.get("mode") == "program" and replay["completed"] == replay["steps"]:
            group = "whole program replayed"
        elif replay and replay.get("mode") == "program":
            group = "program, partly replayed"
        elif replay:
            group = "opening steps only"
        else:
            group = "no replay"
        groups[group].append(key)
    summary = {}
    for group, keys in groups.items():
        calls_first = [first[k]["llm_requests"] or 0 for k in keys]
        calls_second = [second[k]["llm_requests"] or 0 for k in keys]
        summary[group] = {
            "tasks": len(keys),
            "passed_first": round(sum(first[k]["score"] or 0 for k in keys), 1),
            "passed_second": round(sum(second[k]["score"] or 0 for k in keys), 1),
            "calls_first": round(statistics.mean(calls_first), 1),
            "calls_second": round(statistics.mean(calls_second), 1),
            "median_first": statistics.median(calls_first),
            "median_second": statistics.median(calls_second),
        }
    return summary


STYLE = """<style>
  .bg { fill: #ffffff; }
  .ink { fill: #1f2328; }
  .muted { fill: #59636e; }
  .grid { stroke: #d1d9e0; }
  .accent { fill: #2a78d6; }
  .base { fill: #9aa4ae; }
  text { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }
  @media (prefers-color-scheme: dark) {
    .bg { fill: #0d1117; }
    .ink { fill: #e6edf3; }
    .muted { fill: #9198a1; }
    .grid { stroke: #30363d; }
    .accent { fill: #4493f8; }
    .base { fill: #6b7580; }
  }
</style>"""


def escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def bar_chart(title: str, subtitle: str, bars: list[tuple[str, str, float, str]], unit: str,
              maximum: float, highlight: int) -> str:
    """Horizontal bars: (label, note, value, value text); `highlight` is the
    index drawn in the accent colour, the rest in a neutral tone."""
    width, left, right, top, row = 640, 150, 90, 64, 34
    height = top + row * len(bars) + 16
    span = width - left - right
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" '
             f'aria-label="{escape(title)}">', STYLE,
             f'<rect class="bg" width="{width}" height="{height}" rx="8"/>',
             f'<text class="ink" x="16" y="26" font-size="16" font-weight="600">{escape(title)}</text>',
             f'<text class="muted" x="16" y="46" font-size="12">{escape(subtitle)}</text>']
    for tick in (0.25, 0.5, 0.75, 1.0):
        x = left + span * tick
        parts.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{top - 6}" y2="{height - 12}" stroke-width="1"/>')
    for index, (label, note, value, text) in enumerate(bars):
        y = top + row * index
        length = max(2.0, span * value / maximum)
        cls = "accent" if index == highlight else "base"
        parts.append(f'<text class="ink" x="{left - 10}" y="{y + 13}" font-size="13" text-anchor="end">{escape(label)}</text>')
        parts.append(f'<text class="muted" x="{left - 10}" y="{y + 26}" font-size="10" text-anchor="end">{escape(note)}</text>')
        parts.append(f'<rect class="{cls}" x="{left}" y="{y + 2}" width="{length:.1f}" height="20" rx="3"/>')
        parts.append(f'<text class="ink" x="{left + length + 8:.1f}" y="{y + 17}" font-size="12" font-weight="600">{escape(text)}{unit}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def paired_chart(title: str, subtitle: str, groups: list[tuple[str, float, float, str]]) -> str:
    """Per group: a neutral bar for the first attempt and an accent bar for the
    second, with the change written beside the second."""
    width, left, right, top, row = 640, 190, 110, 84, 58
    height = top + row * len(groups) + 10
    span = width - left - right
    maximum = max(max(a, b) for _, a, b, _ in groups) * 1.05
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" '
             f'aria-label="{escape(title)}">', STYLE,
             f'<rect class="bg" width="{width}" height="{height}" rx="8"/>',
             f'<text class="ink" x="16" y="26" font-size="16" font-weight="600">{escape(title)}</text>',
             f'<text class="muted" x="16" y="46" font-size="12">{escape(subtitle)}</text>',
             f'<rect class="base" x="16" y="58" width="12" height="10" rx="2"/>',
             '<text class="muted" x="34" y="67" font-size="11">first attempt</text>',
             f'<rect class="accent" x="124" y="58" width="12" height="10" rx="2"/>',
             '<text class="muted" x="142" y="67" font-size="11">second attempt</text>']
    for index, (label, first, second, note) in enumerate(groups):
        y = top + row * index
        parts.append(f'<text class="ink" x="{left - 10}" y="{y + 16}" font-size="13" text-anchor="end">{escape(label)}</text>')
        parts.append(f'<text class="muted" x="{left - 10}" y="{y + 30}" font-size="10" text-anchor="end">{escape(note)}</text>')
        for offset, value, cls in ((0, first, "base"), (22, second, "accent")):
            length = max(2.0, span * value / maximum)
            parts.append(f'<rect class="{cls}" x="{left}" y="{y + offset}" width="{length:.1f}" height="18" rx="3"/>')
            parts.append(f'<text class="ink" x="{left + length + 6:.1f}" y="{y + offset + 13}" font-size="11">{value:.1f}</text>')
        change = (second - first) / first * 100 if first else 0
        parts.append(f'<text class="ink" x="{width - 16}" y="{y + 35}" font-size="13" font-weight="600" '
                     f'text-anchor="end">{change:+.0f}%</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    runs = []
    for tag, pass_index, label, note in FULL_RUNS:
        metrics = run_metrics(tag, pass_index)
        runs.append({"tag": tag, "pass": pass_index, "label": label, "note": note, **metrics})
        print(label, metrics)
    memory = muscle_memory("v5")
    latest = len(runs) - 1
    (ASSETS / "arena-success.svg").write_text(bar_chart(
        "Tasks passed", "Windows Agent Arena, 154 tasks per pass; stealth planner with JEV as System 1",
        [(r["label"], r["note"], r["passed"] / r["tasks"] * 100, f'{r["passed"] / r["tasks"] * 100:.1f}') for r in runs],
        "%", 100, latest), encoding="utf-8")
    (ASSETS / "arena-calls.svg").write_text(bar_chart(
        "Planner LLM calls per task", "Mean over all 154 tasks; lower means the large model is asked less",
        [(r["label"], r["note"], r["calls_mean"], f'{r["calls_mean"]:.1f}') for r in runs],
        "", max(r["calls_mean"] for r in runs) * 1.1, latest), encoding="utf-8")
    (ASSETS / "arena-system1.svg").write_text(bar_chart(
        "Actions done by System 1", "Share of on-screen actions carried out by the fast decision model",
        [(r["label"], r["note"], r["system1_share"], str(r["system1_share"])) for r in runs],
        "%", 100, latest), encoding="utf-8")
    order = [("whole program replayed", "System 1 ran the recorded program"),
             ("no replay", "no program for the task yet")]
    groups = [(name, memory[name]["calls_first"], memory[name]["calls_second"],
               f'{memory[name]["tasks"]} tasks') for name, _ in order if name in memory]
    (ASSETS / "muscle-memory.svg").write_text(paired_chart(
        "Muscle memory: planner calls per task",
        "v5, the same tasks attempted twice; the second attempt replays programs recorded on the first",
        groups), encoding="utf-8")
    (RESULTS / "arena-summary.json").write_text(json.dumps(
        {"runs": runs, "muscle_memory_v5": memory}, indent=1) + "\n", encoding="utf-8")
    print("wrote", ", ".join(p.name for p in sorted(ASSETS.glob("*.svg"))), "and", RESULTS / "arena-summary.json")


if __name__ == "__main__":
    main()
