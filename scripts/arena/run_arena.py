"""Run POK-Ai on Windows Agent Arena and measure it (run from WSL or Linux).

    python3 scripts/arena/run_arena.py run --llm bunny --arms none,jev \\
        --tasks settings,notepad,windows_calc,clock --passes 2 --tag waa1
    python3 scripts/arena/run_arena.py summary --tag waa1
    python3 scripts/arena/run_arena.py dataset --tag waa1 --output dataset/

Each (planner, arm) gets its own arena share, holding POK-Ai's memory, and
runs the task list `--passes` times. Every pass starts from a fresh copy of
the golden VM, so tasks cannot see each other's leftovers across passes, but
with `--memory persist` (default) the agent keeps the skills and facts it
learned: pass 2 onward measures self-improvement. `--memory fresh` gives
every task empty memory, the no-learning control.

Outputs go to ~/arena/runs/<tag>/ (outside the repository): results.jsonl
(one row per task and pass), per-task logs and traces, router training
records, and the memory itself. The arena VM holds no personal data, so its
router records can seed an open System 1 training set (`dataset`).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import setup_waa  # noqa: E402
from live_bench import ARMS, LLMS, SIDECARS, set_key  # noqa: E402

RUNS = Path(os.environ.get("POKAI_ARENA_RUNS", Path.home() / "arena" / "runs"))
TASKS_DIR = setup_waa.container_dir() / "client" / "evaluation_examples_windows"
WAA_ACTION_SPACE, WAA_OBSERVATION = "pyautogui", "a11y_tree"  # run.py defaults, used in result paths


# Windows interop is not always on the WSL PATH.
POWERSHELL = shutil.which("powershell.exe") or "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"


def win_path(path: Path) -> str:
    return subprocess.run(["wslpath", "-w", str(path)], capture_output=True, text=True, check=True).stdout.strip()


def powershell(*args: str) -> str:
    return subprocess.run(
        [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", *args],
        capture_output=True, text=True, check=True, cwd="/mnt/c",
    ).stdout


def read_credential(name: str) -> str:
    return powershell("-File", win_path(REPO / "scripts" / "read-credential.ps1"), "-Name", name).strip()


def build_exe() -> Path:
    output = powershell("-File", win_path(HERE / "build-cli.ps1")).strip().splitlines()
    exe = Path(subprocess.run(["wslpath", "-u", output[-1].strip()], capture_output=True, text=True, check=True).stdout.strip())
    if not exe.exists():
        sys.exit("\n".join(output[-6:]))
    return exe


def provider_key_env(provider: str) -> str | None:
    section = None
    for line in (REPO / "pok-ai.toml").read_text(encoding="utf-8").splitlines():
        if line.startswith("["):
            section = line.strip()
        elif section == f"[providers.{provider}]" and line.split("=")[0].strip() == "api_key_env":
            return json.loads(line.split("=", 1)[1].strip())
    return None


def arena_config(llm: str, arm: str) -> str:
    """pok-ai.toml for the VM: VM paths, the planner, the arm's router, and
    the router training log on."""
    spec, model = ARMS[arm], LLMS[llm]
    sections: list[list[str]] = [[]]
    for line in (REPO / "pok-ai.toml").read_text(encoding="utf-8").splitlines():
        if line.startswith("["):
            sections.append([])
        sections[-1].append(line)
    out = []
    for position, section in enumerate(sections):
        header = section[0].strip() if position and section else ""
        if position == 0:
            section = set_key(section, "data_dir", '"C:/pokai/data"')
            section = set_key(section, "diagnostics_dir", '"C:/pokai/diag"')
            section = set_key(section, "default_provider", json.dumps(model["provider"]))
            section = set_key(section, "default_model", json.dumps(model["model"]))
        elif header == "[decision_router]":
            section = set_key(section, "enabled", "true" if spec.get("enabled", True) else "false")
            section = set_key(section, "backend", json.dumps(spec.get("backend", "jev")))
            section = set_key(section, "mode", '"delegated"')
            section = set_key(section, "training_log", "true")
            if "timeout_ms" in spec:
                section = set_key(section, "timeout_ms", str(spec["timeout_ms"]))
            if "trust" in spec:
                section = set_key(section, "trust_model_targets", "true" if spec["trust"] else "false")
                section = set_key(section, "trust_model_conditions", "true" if spec["trust"] else "false")
        elif header == "[decision_router.judge]":
            section = set_key(section, "enabled", "false")
        out.extend(section)
    return "\n".join(out) + "\n"


def select_tasks(spec: str) -> dict[str, list[str]]:
    """`all`, domain names, or domain/id entries, comma separated."""
    everything = json.loads((TASKS_DIR / "test_all.json").read_text(encoding="utf-8"))
    chosen: dict[str, list[str]] = collections.defaultdict(list)
    for item in spec.split(","):
        item = item.strip()
        if item == "all":
            for domain, ids in everything.items():
                chosen[domain].extend(ids)
        elif "/" in item:
            domain, task_id = item.split("/", 1)
            chosen[domain].append(task_id)
        elif item in everything:
            chosen[item].extend(everything[item])
        else:
            sys.exit(f"unknown task or domain: {item} (domains: {', '.join(everything)})")
    return {domain: list(dict.fromkeys(ids)) for domain, ids in chosen.items()}


def parse_score(text: str | None) -> float | None:
    """WAA writes a number, or sometimes a Python bool, to result.txt."""
    if text in (None, ""):
        return None
    lowered = text.strip().lower()
    if lowered in ("true", "false"):
        return 1.0 if lowered == "true" else 0.0
    try:
        return float(text)
    except ValueError:
        return None


def split_workers(tasks: dict[str, list[str]], workers: int, split_apps: bool = False) -> list[dict[str, list[str]]]:
    """Whole applications per worker (largest first onto the least loaded),
    so each application's memory is only ever used by one VM at a time.

    With `split_apps`, the largest applications are divided until every
    worker has work; each piece then runs on its own copy of that
    application's memory (see `start_worker`)."""
    units = [(domain, ids) for domain, ids in tasks.items()]
    if split_apps:
        while len(units) < workers:
            units.sort(key=lambda unit: -len(unit[1]))
            domain, ids = units[0]
            if len(ids) < 2:
                break
            half = (len(ids) + 1) // 2
            units[0:1] = [(domain, ids[:half]), (domain, ids[half:])]
    groups: list[dict[str, list[str]]] = [{} for _ in range(max(1, min(workers, len(units))))]
    for domain, ids in sorted(units, key=lambda unit: -len(unit[1])):
        # A worker takes at most one piece of an application.
        candidates = [group for group in groups if domain not in group] or groups
        target = min(candidates, key=lambda group: sum(map(len, group.values())))
        target.setdefault(domain, []).extend(ids)
    return groups


def memory_stats(memory_root: Path) -> dict:
    """Skills across every memory scope of a share."""
    totals = {"skills": 0, "skill_tasks": 0, "skills_disabled": 0}
    for database in memory_root.glob("*/memory.db"):
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
            skills, tasks, disabled = connection.execute(
                "SELECT COUNT(*), COUNT(DISTINCT kind || task_signature || applications), "
                "COALESCE(SUM(enabled = 0), 0) FROM procedures WHERE kind = 'workflow'"
            ).fetchone()
        totals["skills"] += skills
        totals["skill_tasks"] += tasks
        totals["skills_disabled"] += disabled
    return totals


def start_worker(args, llm: str, label: str, pass_index: int, worker: int, tasks: dict[str, list[str]],
                 share: Path, env_keys: dict, result_rel: str, pass_dir: Path) -> tuple[str, Path]:
    storage = pass_dir / f"storage-w{worker}"
    shutil.rmtree(storage, ignore_errors=True)
    subprocess.run(["cp", "-r", "--sparse=always", str(setup_waa.golden_storage()), str(storage)], check=True)
    task_file = f"pokai-{args.tag}-{label}-p{pass_index}-w{worker}.json"
    # A piece of a split application works on its own copy of that
    # application's memory, so parallel pieces never overwrite each other.
    suffix = f"-w{worker}" if getattr(args, "split_apps", False) else ""
    if suffix and args.memory == "persist":
        for domain in tasks:
            source, copy = share / "memory" / domain, share / "memory" / f"{domain}{suffix}"
            if source.exists() and not copy.exists():
                # Memory folders are written from the VM as root; copy as root.
                subprocess.run(
                    ["docker", "run", "--rm", "-v", f"{share / 'memory'}:/memory", "--entrypoint", "cp",
                     setup_waa.WAA_IMAGE, "-a", f"/memory/{domain}", f"/memory/{domain}{suffix}"],
                    check=True, capture_output=True,
                )
    (TASKS_DIR / task_file).write_text(json.dumps(tasks, indent=1), encoding="utf-8")
    env_values = {
        "POKAI_TASK_TIMEOUT": str(args.timeout),
        # Persistent memory is kept per application ("@domain").
        "POKAI_MEMORY_SCOPE": "@domain" if args.memory == "persist" else "",
        "POKAI_SCOPE_SUFFIX": suffix,
        "POKAI_KEY_NAMES": ",".join(env_keys),
        **env_keys,
    }
    # Keys reach the container through a private, short-lived env file.
    with tempfile.NamedTemporaryFile("w", delete=False, prefix="pokai-arena-", suffix=".env") as env_file:
        env_file.write("\n".join(f"{name}={value}" for name, value in env_values.items()) + "\n")
    os.chmod(env_file.name, 0o600)
    name = f"pokai-{args.tag}-{label}-p{pass_index}-w{worker}".replace("_", "-")
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    command = [
        # No --rm: a crashed worker's log must survive to be saved.
        "docker", "run", "-d", "--name", name,
        "-p", f"{args.viewer_port + worker}:8006",
        "--device=/dev/kvm", "-e", f"RAM_SIZE={args.ram}", "-e", f"CPU_CORES={args.cpus}",
        "-v", f"{storage}:/storage",
        "-v", f"{setup_waa.container_dir() / 'vm' / 'setup'}:/shared",
        "-v", f"{share}:/shared/pokai",
        "-v", f"{setup_waa.container_dir() / 'client'}:/client",
        "--env-file", env_file.name,
        "--cap-add", "NET_ADMIN", "--stop-timeout", "120", "--entrypoint", "/bin/bash",
        setup_waa.WAA_IMAGE, "-c",
        f"./entry.sh --start-client true --agent pokai --model {llm} --clean-results false "
        f"--json-name evaluation_examples_windows/{task_file} --result-dir /client/{result_rel}",
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
    finally:
        os.unlink(env_file.name)
    return name, storage


def run_pass(args, llm: str, arm: str, pass_index: int, tasks: dict[str, list[str]], share: Path, env_keys: dict) -> list[dict]:
    label = f"{llm}-{arm}"
    pass_dir = RUNS / args.tag / label / f"pass{pass_index}"
    pass_dir.mkdir(parents=True, exist_ok=True)
    result_rel = f"results/pokai/{args.tag}/{label}/pass{pass_index}"
    result_host = setup_waa.container_dir() / "client" / result_rel
    task_root = result_host / WAA_ACTION_SPACE / WAA_OBSERVATION / llm / "0"

    def scored(domain: str, task_id: str) -> bool:
        return (task_root / domain / task_id / "result.txt").exists()

    if args.resume:
        # Keep earlier results; run only the tasks without a score.
        tasks = {domain: [task_id for task_id in ids if not scored(domain, task_id)] for domain, ids in tasks.items()}
        tasks = {domain: ids for domain, ids in tasks.items() if ids}
        if not tasks:
            print(f"[{label} pass {pass_index}] nothing to resume", flush=True)
            return []
    else:
        shutil.rmtree(result_host, ignore_errors=True)

    groups = split_workers(tasks, args.workers, getattr(args, "split_apps", False))
    print(f"[{label} pass {pass_index}] starting {len(groups)} VM(s) for {sum(map(len, tasks.values()))} tasks", flush=True)
    workers = []
    for worker, group in enumerate(groups):
        name, storage = start_worker(args, llm, label, pass_index, worker, group, share, env_keys, result_rel, pass_dir)
        workers.append((name, storage, group))
        print(f"  worker {worker}: {', '.join(group)} ({sum(map(len, group.values()))} tasks); "
              f"watch http://localhost:{args.viewer_port + worker}", flush=True)

    expected = [(domain, task_id) for domain, ids in tasks.items() for task_id in ids]
    longest = max(sum(map(len, group.values())) for group in groups)
    restarts = [0] * len(workers)
    deadline = time.monotonic() + longest * (args.timeout + 300) + 1800
    try:
        reported = 0
        while time.monotonic() < deadline:
            finished = sum(scored(domain, task_id) for domain, task_id in expected)
            if finished != reported:
                reported = finished
                print(f"[{label} pass {pass_index}] {finished}/{len(expected)} tasks scored", flush=True)
            running = [
                name for name, _, _ in workers
                if subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", name],
                                  capture_output=True, text=True).stdout.strip() == "true"
            ]
            for index, (name, storage, group) in enumerate(workers):
                left = {domain: [task_id for task_id in ids if not scored(domain, task_id)] for domain, ids in group.items()}
                left = {domain: ids for domain, ids in left.items() if ids}
                if not left and name in running:
                    # All of this worker's tasks are scored: stop it early.
                    subprocess.run(["docker", "stop", name], capture_output=True)
                elif left and name not in running and restarts[index] < 2:
                    # The WAA client died with tasks left (for example a task
                    # setup crashed): start a fresh VM for the rest.
                    restarts[index] += 1
                    with open(pass_dir / f"container-w{index}-crash{restarts[index]}.log", "w", encoding="utf-8") as log:
                        subprocess.run(["docker", "logs", name], stdout=log, stderr=subprocess.STDOUT)
                    shutil.rmtree(storage, ignore_errors=True)
                    print(f"[{label} pass {pass_index}] worker {index} stopped with "
                          f"{sum(map(len, left.values()))} tasks left; restarting it", flush=True)
                    name, storage = start_worker(args, llm, label, pass_index, index, left, share, env_keys,
                                                 result_rel, pass_dir)
                    workers[index] = (name, storage, group)
                    running.append(name)
            if finished == len(expected) or not running:
                break
            time.sleep(30)
    finally:
        for index, (name, storage, _) in enumerate(workers):
            with open(pass_dir / f"container-w{index}.log", "w", encoding="utf-8") as log:
                subprocess.run(["docker", "logs", name], stdout=log, stderr=subprocess.STDOUT)
            subprocess.run(["docker", "stop", name], capture_output=True)
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
            if not args.keep_storage:
                shutil.rmtree(storage, ignore_errors=True)

    rows = []
    stats = memory_stats(share / "memory")
    for domain, task_id in expected:
        task_dir = task_root / domain / task_id
        result = (task_dir / "result.txt").read_text().strip() if (task_dir / "result.txt").exists() else None
        outcome = json.loads((task_dir / "pokai.json").read_text()) if (task_dir / "pokai.json").exists() else {}
        rows.append({
            "tag": args.tag, "llm": llm, "arm": arm, "pass": pass_index, "memory": args.memory,
            "domain": domain, "task": task_id,
            "score": parse_score(result),
            **{key: outcome.get(key) for key in (
                "run_id", "session_id", "status", "seconds", "llm_requests", "prompt_tokens",
                "skills_loaded", "skill_outcomes", "fast_statuses", "infeasible", "run_failed")},
            **stats,
        })
    return rows


def write_manifest(args: argparse.Namespace, tasks: dict[str, list[str]], exe: Path) -> None:
    """What produced this run, so results stay interpretable later (for
    example once an anonymous planner model is identified)."""
    def git(*command: str, cwd: Path = REPO) -> str:
        return subprocess.run(["git", "-C", str(cwd), *command], capture_output=True, text=True).stdout.strip()

    manifest = {
        "tag": args.tag,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "planners": {llm: {key: LLMS[llm][key] for key in ("provider", "model")} for llm in args.llm.split(",")},
        "arms": {arm: ARMS[arm] for arm in args.arms.split(",")},
        "passes": args.passes, "memory": args.memory, "timeout": args.timeout, "workers": args.workers,
        "tasks": sum(map(len, tasks.values())), "domains": sorted(tasks),
        "harness_commit": git("rev-parse", "HEAD"),
        "harness_uncommitted_changes": bool(git("status", "--porcelain", "--untracked-files=no")),
        "waa_commit": git("rev-parse", "HEAD", cwd=setup_waa.waa_dir()),
        "exe_sha256": subprocess.run(["sha256sum", str(exe)], capture_output=True, text=True).stdout.split(" ")[0],
    }
    path = RUNS / args.tag / "manifest.json"
    if getattr(args, "resume", False) and path.exists():
        # Keep the original record; note what the resume ran with.
        original = json.loads(path.read_text(encoding="utf-8"))
        original.setdefault("resumes", []).append(manifest)
        manifest = original
    path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    if not (setup_waa.golden_storage().exists() and any(setup_waa.golden_storage().iterdir())):
        sys.exit("no golden VM image; see: python3 scripts/arena/setup_waa.py status")
    tasks = select_tasks(args.tasks)
    exe = Path(args.exe) if args.exe else build_exe()
    results = RUNS / args.tag / "results.jsonl"
    results.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(args, tasks, exe)
    for llm in args.llm.split(","):
        for arm in args.arms.split(","):
            spec = ARMS[arm]
            if arm in SIDECARS or "llm_endpoint" in spec or "laya_endpoint" in spec:
                sys.exit(f"arm {arm} needs a local decision-model sidecar, which the arena VM cannot reach yet")
            key_names = [provider_key_env(LLMS[llm]["provider"])]
            if spec.get("enabled", True) and spec.get("backend", "jev") == "jev":
                key_names.append("TYPESAFE_API_KEY")
            env_keys = {name: read_credential(name) for name in key_names if name}
            share = RUNS / args.tag / f"{llm}-{arm}" / "share"
            (share / "bin").mkdir(parents=True, exist_ok=True)
            shutil.copy2(exe, share / "bin" / "pok-ai.exe")
            # A clean VM may lack the Visual C++ runtime the MSVC build links
            # to; ship it next to the exe (app-local deployment).
            for dll in ("vcruntime140.dll", "vcruntime140_1.dll"):
                source = Path("/mnt/c/Windows/System32") / dll
                if source.exists():
                    # System32 copies are read-only; replace rather than overwrite.
                    (share / "bin" / dll).unlink(missing_ok=True)
                    shutil.copyfile(source, share / "bin" / dll)
            (share / "bin" / "arena.toml").write_text(arena_config(llm, arm), encoding="utf-8")
            shutil.copy2(HERE / "vm" / "run-task.ps1", share / "run-task.ps1")
            for pass_index in range(1, args.passes + 1):
                rows = run_pass(args, llm, arm, pass_index, tasks, share, env_keys)
                with open(results, "a", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row) + "\n")
                scored = [row["score"] for row in rows if row["score"] is not None]
                print(json.dumps({
                    "llm": llm, "arm": arm, "pass": pass_index, "tasks": len(rows),
                    "success": round(sum(scored) / len(rows), 3) if rows else None,
                    "skills": rows[-1]["skills"] if rows else None,
                }), flush=True)


def load_rows(tag: str) -> list[dict]:
    """One row per (planner, arm, memory, pass, task): a resumed task's row
    replaces the unscored one written by the interrupted run."""
    path = RUNS / tag / "results.jsonl"
    rows: dict[tuple, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["llm"], row["arm"], row["memory"], row["pass"], row["domain"], row["task"])
        if key not in rows or row.get("score") is not None:
            rows[key] = row
    return list(rows.values())


def summary(args: argparse.Namespace) -> None:
    rows = load_rows(args.tag)
    groups: dict[tuple, list[dict]] = collections.defaultdict(list)
    for row in rows:
        groups[(row["llm"], row["arm"], row["memory"], row["pass"])].append(row)
    print(f"{'llm':8}{'arm':8}{'memory':9}{'pass':>5}{'n':>5}{'success':>9}{'llm_req':>9}{'sec':>7}{'skill_use':>10}{'skills':>8}{'dups':>6}")
    for (llm, arm, memory, pass_index), group in sorted(groups.items()):
        requests = [row["llm_requests"] for row in group if row.get("llm_requests") is not None]
        seconds = [row["seconds"] for row in group if row.get("seconds") is not None]
        used = sum(bool(row.get("skills_loaded")) for row in group)
        last = group[-1]
        print(
            f"{llm:8}{arm:8}{memory:9}{pass_index:>5}{len(group):>5}"
            f"{sum(row['score'] or 0 for row in group) / len(group):>9.1%}"
            f"{statistics.mean(requests) if requests else float('nan'):>9.1f}"
            f"{statistics.mean(seconds) if seconds else float('nan'):>7.0f}"
            f"{used / len(group):>10.0%}{last.get('skills', 0):>8}"
            f"{(last.get('skills', 0) or 0) - (last.get('skill_tasks', 0) or 0):>6}"
        )
    # Self-improvement: the same tasks, first pass against the last.
    by_task: dict[tuple, dict[int, dict]] = collections.defaultdict(dict)
    for row in rows:
        by_task[(row["llm"], row["arm"], row["memory"], row["domain"], row["task"])][row["pass"]] = row
    print("\nfirst pass -> last pass on the same tasks")
    deltas: dict[tuple, list[tuple]] = collections.defaultdict(list)
    for (llm, arm, memory, domain, _), passes in by_task.items():
        if len(passes) < 2:
            continue
        first, last = passes[min(passes)], passes[max(passes)]
        deltas[(llm, arm, memory)].append((
            (last["score"] or 0) - (first["score"] or 0),
            (last.get("llm_requests") or 0) - (first.get("llm_requests") or 0),
            domain,
        ))
    for key, values in sorted(deltas.items()):
        gained = sum(value[0] > 0 for value in values)
        lost = sum(value[0] < 0 for value in values)
        print(
            f"{'/'.join(key):28} tasks {len(values):>4}  gained {gained:>3}  lost {lost:>3}  "
            f"mean LLM calls change {statistics.mean(value[1] for value in values):+.1f}"
        )


def dataset(args: argparse.Namespace) -> None:
    """Router training records from every arena task in the tag, with each
    session's arena score, turned into a Laya training set."""
    outcomes = RUNS / args.tag / "outcomes.jsonl"
    with open(outcomes, "w", encoding="utf-8") as handle:
        for row in load_rows(args.tag):
            if row.get("session_id") and row.get("score") is not None:
                handle.write(json.dumps({"session_id": row["session_id"], "score": row["score"]}) + "\n")
    command = [
        sys.executable, str(REPO / "scripts" / "build_router_dataset.py"),
        "--input", str(RUNS / args.tag), "--output", args.output,
        "--outcomes", str(outcomes), "--teacher", "jev", "--teacher-successful-only",
    ]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--llm", default="bunny", help="comma list of: " + ",".join(LLMS))
    run_parser.add_argument("--arms", default="none,jev", help="comma list of: " + ",".join(ARMS))
    run_parser.add_argument("--tasks", default="all", help="all, domains, or domain/id, comma separated")
    run_parser.add_argument("--passes", type=int, default=2)
    run_parser.add_argument("--memory", choices=["persist", "fresh"], default="persist")
    run_parser.add_argument("--timeout", type=int, default=900, help="seconds per task")
    run_parser.add_argument("--tag", default=time.strftime("waa%m%d%H%M"))
    run_parser.add_argument("--exe", help="pok-ai.exe to use (default: build a release CLI)")
    run_parser.add_argument("--ram", default="8G")
    run_parser.add_argument("--cpus", default="6")
    run_parser.add_argument("--viewer-port", type=int, default=8006, help="first worker's viewer; the next ones count up")
    run_parser.add_argument("--workers", type=int, default=1, help="VMs in parallel (applications are split between them)")
    run_parser.add_argument("--keep-storage", action="store_true", help="keep each pass's VM disk")
    run_parser.add_argument("--resume", action="store_true", help="keep existing scores and run only unscored tasks")
    run_parser.add_argument("--split-apps", action="store_true",
                            help="divide large applications across VMs (each piece gets its own copy of that app's memory)")
    summary_parser = sub.add_parser("summary")
    summary_parser.add_argument("--tag", required=True)
    dataset_parser = sub.add_parser("dataset")
    dataset_parser.add_argument("--tag", required=True)
    dataset_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    {"run": run, "summary": summary, "dataset": dataset}[args.command](args)


if __name__ == "__main__":
    main()
