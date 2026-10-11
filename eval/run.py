#!/usr/bin/env python3
"""Evaluate the harness on fixed tasks, with models of different sizes, with and without System 1.

Each model and task gets three runs:

  s2         System 2 only (no decision router), fresh memory.
  s1         System 1 (JEV) plus System 2, fresh memory. This run learns.
  s1_replay  The same task again in the s1 data folder, so System 1 can replay
             what s1 learned.

Every run is scored by checking the outcome, not by the model's own report:
the final answer for the Wikipedia and shop tasks, and the form fields and the
unsent state of the pages for the job-application task.

Needs OPENROUTER_API_KEY, plus TYPESAFE_API_KEY for the s1 modes, in the
environment. The keys are passed to the CLI and never written to a file.
The user-browser tasks need Chrome with --remote-debugging-port=9333 (see
eval/README.md). Run from the repository root:

  python3 eval/run.py --out ~/pok-eval
"""

import argparse
import json
import os
import re
import select
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE_DIR = ROOT / "eval" / "sites"
RESUME = ROOT / "eval" / "fixtures" / "resume.md"
PROBE = ROOT / "eval" / "probes" / "tabs.mjs"

# Model tiers: a capable free model, then smaller ones. The small tiers are
# closer to what runs on a local machine (a few billion active parameters).
DEFAULT_MODELS = [
    ("capable", "nvidia/nemotron-3-super-120b-a12b:free"),
    ("small", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"),
    ("tiny", "liquid/lfm-2.5-2.6b:free"),
]
MODES = ["s2", "s1", "s1_replay"]
# Opt-in: System 1 answered by a decision model on OpenRouter's Decisions API
# instead of JEV. Pass --mode s1_decision to include it.
EXTRA_MODES = ["s1_decision"]
DECISION_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
DECISION_MODEL = "microsoft/microsoft-decision-1"

WIKI_URL = "https://en.wikipedia.org/wiki/Rust_(programming_language)"

JOB_FIELDS = {
    "name": "Jordan Avery",
    "email": "jordan.avery@example.com",
    "phone": "+1 555 0142",
    "linkedin": "linkedin.com/in/jordan-avery-example",
    "years": "6-9",
    "start": "2026-11-16",
}
JOB_PAGES = ["/jobs/rust-backend.html", "/jobs/devops.html"]


def squeeze(text):
    """Model answers use many kinds of space (narrow no-break, no-break), so
    every whitespace run counts as one plain space before matching."""
    return re.sub(r"\s+", " ", text or "")


def check_wiki(answer, _pages, _base):
    ok = re.search(r"November 9,? 2004|9 November 2004", squeeze(answer))
    return (1.0 if ok else 0.0), {"expected": "November 9, 2004"}


def check_shop(answer, _pages, _base):
    text = squeeze(answer)
    ok = re.search(r"aero ?14", text, re.I) and "899" in text
    return (1.0 if ok else 0.0), {"expected": "Aero 14, $899"}


def check_jobs(_answer, pages, base):
    """Score the two application forms in the user's browser (20 checks).

    A form can be open in more than one tab (the orchestrator opens a page the
    worker already filled), so each form is scored by its best-matching tab."""
    passed = 0
    total = 0
    detail = {}
    for path in JOB_PAGES:
        candidates = [p for p in pages if p["url"] == base + path]
        best = {}
        for page in candidates:
            results = job_form_checks(page)
            if sum(results.values()) > sum(best.values()):
                best = results
        if not best:
            # A form that is not open cannot be checked: every check on it fails.
            best = {name: False for name in [*JOB_FIELDS, "focus", "cover", "remote", "not submitted"]}
        passed += sum(1 for ok in best.values() if ok)
        total += len(best)
        detail[path] = best
    return passed / total, detail


def job_form_checks(page):
    results = {}
    fields = {}
    for field in page["fields"]:
        fields.setdefault(field["name"], field)
    for name, expected in JOB_FIELDS.items():
        results[name] = (fields.get(name) or {}).get("value") == expected
    results["focus"] = len(((fields.get("focus") or {}).get("value") or "").strip()) >= 40
    results["cover"] = len(((fields.get("cover") or {}).get("value") or "").strip()) >= 40
    results["remote"] = bool((fields.get("remote") or {}).get("checked"))
    submitted = page["title"].startswith("Application sent") or "Thank you" in page["text"]
    results["not submitted"] = not submitted
    return results


CHECKS = {
    "wiki_firefox": check_wiki,
    "shop_cheapest": check_shop,
    "job_forms": check_jobs,
}

TASKS = {
    "wiki_firefox": {
        "browser": "user",
        "prompt": (
            f"In my browser, open {WIKI_URL}. Navigating only by clicking links on the pages "
            "(do not type any other address), go to the article on Mozilla, then from there to "
            "the article on Firefox, and tell me Firefox's initial release date."
        ),
    },
    "job_forms": {
        "browser": "user",
        "prompt": (
            "Using my resume in resume.md in this folder, fill out the job applications for two "
            "roles at {base}/jobs/index.html in my browser: (1) Senior Rust Backend Engineer and "
            "(2) Platform / DevOps Engineer. Fill every field from the resume, write a short "
            "tailored answer for the text questions, and do not submit - I will review them first."
        ),
    },
    "shop_cheapest": {
        "browser": "managed",
        "prompt": (
            "On the Corner Shop at {base}/shop/index.html, find the cheapest laptop and tell me "
            "its model name and price."
        ),
    },
}


def http_ok(url):
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            return response.status == 200
    except OSError:
        return False


def ensure_site(port):
    """Serve eval/sites on the port, unless something already does."""
    base = f"http://127.0.0.1:{port}"
    if http_ok(base + "/jobs/index.html"):
        return None, base
    server = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
        cwd=SITE_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1)
    return server, base


def devtools_pages(port):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=5) as response:
        return json.load(response)


def close_fixture_tabs(port, bases):
    """Close tabs left by earlier runs. The user's own tabs are never touched."""
    for tab in devtools_pages(port):
        if tab.get("type") == "page" and any(tab.get("url", "").startswith(b) for b in bases):
            urllib.request.urlopen(f"http://127.0.0.1:{port}/json/close/{tab['id']}", timeout=5).close()


def probe_pages(port):
    result = subprocess.run(
        ["node", str(PROBE), str(port)], capture_output=True, text=True, timeout=60
    )
    if result.returncode != 0:
        return []
    return json.loads(result.stdout or "[]")


def drive_cli(cmd, env, cwd, log_path, timeout):
    """Run the CLI. Answers the external-provider questions with y and every
    other [y/N] question with n, so an approval (such as a Send) is never given."""
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = ""
    pending = ""
    outcome = "exited"
    deadline = time.time() + timeout
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        while True:
            if time.time() > deadline:
                proc.kill()
                outcome = "timeout"
                break
            ready, _, _ = select.select([proc.stdout], [], [], 1.0)
            if not ready:
                if proc.poll() is not None:
                    break
                continue
            chunk = os.read(proc.stdout.fileno(), 4096)
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace")
            log.write(text)
            log.flush()
            output += text
            pending += text
            if pending.endswith("[y/N] "):
                question = pending.strip()
                external = "external" in question.lower() or "jev" in question.lower()
                try:
                    proc.stdin.write(b"y\n" if external else b"n\n")
                    proc.stdin.flush()
                except BrokenPipeError:
                    break
                pending = ""
    try:
        proc.stdin.close()
    except OSError:
        pass
    proc.wait()
    if outcome == "exited" and proc.returncode != 0:
        outcome = f"exit {proc.returncode}"
    return outcome, output


def session_traces(data_dir):
    return set(Path(data_dir, "diagnostics", "sessions").glob("**/trace.jsonl"))


def trace_metrics(data_dir, before):
    """Planner calls, tokens, tool calls and final answer, from the traces this run wrote."""
    new = sorted(session_traces(data_dir) - before)
    planner = prompt_tokens = completion_tokens = tool_calls = 0
    main = None
    for path in new:
        if len(path.relative_to(Path(data_dir, "diagnostics", "sessions")).parts) == 2:
            main = path
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            payload = event.get("payload", {})
            if event.get("kind") == "model_response":
                planner += 1
                prompt_tokens += payload.get("prompt_tokens") or 0
                completion_tokens += payload.get("completion_tokens") or 0
            elif event.get("kind") == "tool_result":
                tool_calls += 1
    answer, status, extras = "", "no_result", {}
    if main is not None:
        for line in main.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("kind") == "run_completed":
                answer = str(event["payload"].get("answer") or "")
                status = "completed"
                extras = event["payload"].get("metrics", {}).get("extras", {})
            elif event.get("kind") == "run_failed":
                answer = str(event["payload"].get("error") or "")
                status = "failed"
    return {
        "status": status,
        "answer": answer,
        "planner_calls": planner,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "tool_calls": tool_calls,
        "system1_steps": extras.get("fast_actions_steps", 0),
        "skill_replays": extras.get("skill_replays", 0),
    }


def router_config(mode):
    """The [decision_router] block for a mode: JEV, a decision model, or none."""
    if mode == "s2":
        return ""
    if mode == "s1_decision":
        return (
            "\n[decision_router]\nenabled = true\nbackend = \"jev\"\nmode = \"delegated\"\n"
            f'endpoint = "{DECISION_ENDPOINT}"\n'
            f'model = "{DECISION_MODEL}"\n'
            'api_key_env = "OPENROUTER_API_KEY"\n'
        )
    return '\n[decision_router]\nenabled = true\nbackend = "jev"\nmode = "delegated"\n'


def run_one(args, env, out, tier, model, task, mode, base):
    spec = TASKS[task]
    if mode == "s1_replay":
        data_dir = out / tier / task / "s1" / "data"
        if not data_dir.exists():
            return None
    else:
        data_dir = out / tier / task / mode / "data"
    run_dir = out / tier / task / mode
    ws = run_dir / "ws"
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    shutil.copy(RESUME, ws / "resume.md")
    data_dir.mkdir(parents=True, exist_ok=True)
    config = run_dir / "config.toml"
    config.write_text(
        f'data_dir = "{data_dir.as_posix()}"\n\n[user_browser]\nenabled = true\n'
        f"port = {args.user_browser_port}\n"
        + router_config(mode),
        encoding="utf-8",
    )
    prompt = spec["prompt"].format(base=base)
    cmd = [
        str(args.binary),
        "--config",
        str(config),
        "run",
        "--provider",
        "openrouter",
        "--model",
        model,
        "--workspace",
        str(ws),
        prompt,
    ]
    if shutil.which("xvfb-run"):
        cmd = ["xvfb-run", "-a"] + cmd
    if spec["browser"] == "user":
        close_fixture_tabs(args.user_browser_port, [base, "https://en.wikipedia.org/"])
    before = session_traces(data_dir)
    started = time.time()
    outcome, _ = drive_cli(cmd, env, ws, run_dir / "run.log", args.timeout)
    wall = round(time.time() - started, 1)
    metrics = trace_metrics(data_dir, before)
    pages = []
    if spec["browser"] == "user":
        pages = probe_pages(args.user_browser_port)
        (run_dir / "tabs.json").write_text(json.dumps(pages, indent=1), encoding="utf-8")
    score, detail = CHECKS[task](metrics["answer"], pages, base)
    record = {
        "tier": tier,
        "model": model,
        "task": task,
        "mode": mode,
        "outcome": outcome if outcome != "exited" else metrics["status"],
        "score": round(score, 3),
        "detail": detail,
        "wall_s": wall,
        **metrics,
    }
    record.pop("status", None)
    return record


def summarize(records):
    lines = [
        "| tier | task | mode | score | ok | planner calls | tool calls | prompt tokens | system 1 steps | wall s | outcome |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        ok = "yes" if r["score"] == 1.0 else "no"
        lines.append(
            f"| {r['tier']} | {r['task']} | {r['mode']} | {r['score']:.2f} | {ok} | "
            f"{r['planner_calls']} | {r['tool_calls']} | {r['prompt_tokens']:,} | "
            f"{r['system1_steps']} | {r['wall_s']} | {r['outcome']} |"
        )
    return "\n".join(lines)


def rescore(out, site_port):
    """Score every saved run again from its answer and page state, with the
    current checks. Runs are not repeated."""
    path = out / "results.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    base = f"http://127.0.0.1:{site_port}"
    for record in records:
        run_dir = out / record["tier"] / record["task"] / record["mode"]
        tabs = run_dir / "tabs.json"
        pages = json.loads(tabs.read_text(encoding="utf-8")) if tabs.exists() else []
        score, detail = CHECKS[record["task"]](record["answer"], pages, base)
        record["score"] = round(score, 3)
        record["detail"] = detail
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    summary = summarize(records)
    (out / "summary.md").write_text(summary + "\n", encoding="utf-8")
    print(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="folder for runs and results (keep it out of the repo)")
    parser.add_argument("--binary", type=Path, default=ROOT / "target" / "debug" / "pok-ai")
    parser.add_argument("--model", action="append", metavar="TIER=MODEL_ID", help="repeatable; default: capable, small and tiny free models")
    parser.add_argument("--task", action="append", choices=sorted(TASKS), help="repeatable; default: all")
    parser.add_argument("--mode", action="append", choices=MODES + EXTRA_MODES, help="repeatable; default: s2, s1, s1_replay")
    parser.add_argument("--timeout", type=int, default=900, help="seconds per run")
    parser.add_argument("--site-port", type=int, default=8765)
    parser.add_argument("--user-browser-port", type=int, default=9333)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rescore", action="store_true", help="score the saved runs in --out again, without running anything")
    args = parser.parse_args()

    if args.rescore:
        rescore(args.out, args.site_port)
        return

    models = [tuple(m.split("=", 1)) for m in args.model] if args.model else DEFAULT_MODELS
    tasks = args.task or list(TASKS)
    modes = args.mode or MODES
    needs_jev = any(mode in ("s1", "s1_replay") for mode in modes)

    env = dict(os.environ)
    missing = [k for k in ["OPENROUTER_API_KEY"] + (["TYPESAFE_API_KEY"] if needs_jev else []) if not env.get(k)]
    if missing:
        sys.exit(f"missing environment variables: {', '.join(missing)}")
    if not args.binary.exists():
        sys.exit(f"no binary at {args.binary}; run cargo build -p pok-ai-cli")

    plan = [(t, m, task, mode) for t, m in models for task in tasks for mode in modes]
    if args.dry_run:
        for tier, model, task, mode in plan:
            print(f"{tier:8} {task:15} {mode:10} {model}")
        return

    if any(TASKS[t]["browser"] == "user" for t in tasks):
        if not http_ok(f"http://127.0.0.1:{args.user_browser_port}/json/version"):
            sys.exit(f"Chrome is not listening on port {args.user_browser_port}; see eval/README.md")
    server, base = ensure_site(args.site_port)

    args.out.mkdir(parents=True, exist_ok=True)
    results_path = args.out / "results.jsonl"
    records = []
    try:
        for tier, model, task, mode in plan:
            record = run_one(args, env, args.out, tier, model, task, mode, base)
            if record is None:
                print(f"skip {tier} {task} {mode}: no s1 run to replay", flush=True)
                continue
            records.append(record)
            with open(results_path, "a", encoding="utf-8") as results:
                results.write(json.dumps(record) + "\n")
            print(f"{tier:8} {task:15} {mode:10} score={record['score']:.2f} planner={record['planner_calls']} "
                  f"tokens={record['prompt_tokens']} wall={record['wall_s']}s {record['outcome']}", flush=True)
    finally:
        if any(TASKS[t]["browser"] == "user" for t in tasks):
            close_fixture_tabs(args.user_browser_port, [base, "https://en.wikipedia.org/"])
        if server is not None:
            server.terminate()

    summary = summarize(records)
    (args.out / "summary.md").write_text(summary + "\n", encoding="utf-8")
    print()
    print(summary)


if __name__ == "__main__":
    main()
