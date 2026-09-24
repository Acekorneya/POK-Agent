"""Extract candidate states for the router question suite from bench traces.

Scans diagnostics/bench/sessions (ignored by git) for desktop observations
and managed-browser snapshots, and prints bounded, sanitized states grouped
by window or page so benchmark cases can be written and labelled by hand in
diagnostics/router_question_suite.json. Nothing here is committed
automatically: states can contain private screen text, so review each one.

Sanitization: labels containing an email address or a long digit run are
dropped, and only windows whose title matches --title-filter are kept.

Usage:
    python scripts/build_router_suite.py --title-filter "Settings|File Explorer" > states.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

SESSIONS = os.path.join(os.path.dirname(__file__), "..", "diagnostics", "bench", "sessions")
PRIVATE = re.compile(r"@|\d{6,}")


def clean(labels: list[str], limit: int) -> list[str]:
    seen: list[str] = []
    for label in labels:
        label = " ".join(label.split())[:120]
        if label and not PRIVATE.search(label) and label not in seen:
            seen.append(label)
        if len(seen) >= limit:
            break
    return seen


def desktop_states(title_filter: re.Pattern[str], limit: int) -> dict[str, dict]:
    states: dict[str, dict] = {}
    for path in glob.glob(os.path.join(SESSIONS, "*", "observation-*.json")):
        if path.endswith("-targets.json"):
            continue
        try:
            observation = json.load(open(path, encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(observation, dict):
            continue
        observation = observation.get("observation", observation)
        window = observation.get("foreground_window") or {}
        title = window.get("title", "")
        if not title_filter.search(title):
            continue
        targets = observation.get("targets") or []
        labels = clean([f"{t.get('name', '')} ({t.get('control_type', '')})" for t in targets], limit)
        key = f"{title}|{len(labels)}"
        states.setdefault(
            key,
            {
                "source": os.path.relpath(path, SESSIONS),
                "window": {"application": window.get("process_name"), "title": title},
                "visible": labels,
            },
        )
    return states


def page_states(url_filter: re.Pattern[str]) -> dict[str, dict]:
    states: dict[str, dict] = {}
    for trace in glob.glob(os.path.join(SESSIONS, "*", "trace.jsonl")):
        for line in open(trace, encoding="utf-8", errors="replace"):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            payload = record.get("payload") or {}
            if record.get("kind") != "tool_result" or not str(payload.get("name", "")).startswith("managed_browser"):
                continue
            result = payload.get("result") or {}
            snapshot = result.get("snapshot", result) if isinstance(result, dict) else {}
            url = snapshot.get("url") or ""
            if not url or not url_filter.search(url) or url in states:
                continue
            states[url] = {
                "source": os.path.relpath(trace, SESSIONS),
                "page": {
                    "url": url[:300],
                    "title": (snapshot.get("title") or "")[:200],
                    "text_start": " ".join((snapshot.get("text") or "").split())[:200],
                },
                "links": clean([e.get("name", "") for e in snapshot.get("elements") or [] if e.get("role") == "link"], 30),
            }
    return states


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--title-filter", default="Settings|File Explorer|Program Files|This PC")
    parser.add_argument("--url-filter", default="doc.rust-lang.org")
    parser.add_argument("--labels", type=int, default=40)
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "desktop": desktop_states(re.compile(args.title_filter), args.labels),
                "pages": page_states(re.compile(args.url_filter)),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
