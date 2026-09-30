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
import re
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
WAA_ACTION_SPACE = "pyautogui"  # WAA's run.py default, part of its result paths

# WAA's reference agent can answer FAIL for an impossible task. POK-Ai gets
# the same option as a standing instruction, so task text stays unmodified;
# the arena agent maps the INFEASIBLE: answer to WAA's FAIL action.
INFEASIBLE_CONVENTION = (
    "If a task is impossible on this computer (the requested feature, file, or content does not exist here), "
    "do not attempt a workaround: start your final answer with INFEASIBLE: and say why."
)


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


def arena_config(llm: str, arm: str, step_budget: int = 30) -> str:
    """pok-ai.toml for the VM: VM paths, the planner, the arm's router, the
    router training log on, the step budget, and the infeasible-task
    convention."""
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
            if step_budget > 0:
                section = set_key(section, "action_step_budget", str(step_budget))
            section = set_key(section, "standing_instructions", json.dumps(INFEASIBLE_CONVENTION))
            # No one works at the VM: handing focus back to the launching
            # console after each input only closes the app's open menus.
            section = set_key(section, "background_desktop_work", "false")
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


def plan_units(tasks: dict[str, list[str]], workers: int, split_apps: bool = False) -> list[tuple[str, list[str]]]:
    """The work queue: one unit per application, largest first. A unit runs
    on one VM at a time, so an application's memory is never written by two
    VMs at once.

    With `split_apps`, the largest applications are divided until there is a
    unit for every worker; each piece then runs on its own copy of that
    application's memory."""
    units = [(domain, list(ids)) for domain, ids in tasks.items() if ids]
    if split_apps and len(units) < workers:
        # Give each application VMs in proportion to its tasks, then cut it
        # into near-equal pieces, so the VMs finish at about the same time.
        shares = {domain: 1 for domain, _ in units}
        sizes = dict(units)
        for _ in range(workers - len(units)):
            domain = max(shares, key=lambda name: len(sizes[name]) / shares[name])
            if len(sizes[domain]) <= shares[domain]:
                break
            shares[domain] += 1
        pieces = []
        for domain, ids in units:
            count = shares[domain]
            start = 0
            for index in range(count):
                size = len(ids) // count + (1 if index < len(ids) % count else 0)
                pieces.append((domain, ids[start:start + size]))
                start += size
        units = pieces
    return sorted(units, key=lambda unit: -len(unit[1]))


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


def memory_suffix(args, worker: int) -> str:
    """Each VM learns into its own copy of an application's memory; copies are
    merged back when the VM finishes a piece of work and at the end of a pass
    (see merge_memory), so learning is shared as it would be by one agent."""
    return f"-w{worker}" if args.memory == "persist" else ""


def merge_memory(share: Path, copies: list[str]) -> None:
    """Merge VM copies (`<app>-w<n>`) into their application's memory. The
    folders are written from the VMs as root, so the merge runs as root in
    the WAA container."""
    for name in copies:
        match = re.fullmatch(r"(.+)-w\d+", name)
        if not match or not (share / "memory" / name).exists():
            continue
        result = subprocess.run(
            ["docker", "run", "--rm", "-v", f"{share}:/share", "--entrypoint", "python3", setup_waa.WAA_IMAGE,
             "/share/merge_memory.py", f"/share/memory/{match.group(1)}", f"/share/memory/{name}"],
            capture_output=True, text=True,
        )
        detail = (result.stdout.strip() or result.stderr.strip()[-300:]).replace("\n", "; ")
        print(f"  memory {name} -> {match.group(1)}: {detail}", flush=True)


def merge_all_memory(share: Path) -> None:
    folder = share / "memory"
    if folder.exists():
        merge_memory(share, sorted(path.name for path in folder.iterdir() if re.fullmatch(r".+-w\d+", path.name)))


def seed_memory_copy(args, share: Path, domain: str, suffix: str) -> None:
    """Start a split piece's memory from the application's shared memory."""
    if not suffix or args.memory != "persist":
        return
    source, copy = share / "memory" / domain, share / "memory" / f"{domain}{suffix}"
    if source.exists() and not copy.exists():
        # Memory folders are written from the VM as root; copy as root.
        subprocess.run(
            ["docker", "run", "--rm", "-v", f"{share / 'memory'}:/memory", "--entrypoint", "cp",
             setup_waa.WAA_IMAGE, "-a", f"/memory/{domain}", f"/memory/{domain}{suffix}"],
            check=True, capture_output=True,
        )


def write_unit_file(args, label: str, pass_index: int, worker: int, serial: int, unit: tuple[str, list[str]]) -> str:
    task_file = f"pokai-{args.tag}-{label}-p{pass_index}-w{worker}-{serial}.json"
    (TASKS_DIR / task_file).write_text(json.dumps({unit[0]: unit[1]}, indent=1), encoding="utf-8")
    return task_file


def client_command(llm: str, task_file: str, result_rel: str) -> str:
    return (f"--agent pokai --model {llm} --clean-results false "
            f"--json-name evaluation_examples_windows/{task_file} --result-dir /client/{result_rel}")


def start_worker(args, llm: str, label: str, pass_index: int, worker: int, unit: tuple[str, list[str]],
                 share: Path, env_keys: dict, result_rel: str, pass_dir: Path, serial: int = 0,
                 own_memory: bool = False) -> tuple[str, Path]:
    """Boot a fresh VM from the golden image and run `unit` on it."""
    storage = pass_dir / f"storage-w{worker}"
    shutil.rmtree(storage, ignore_errors=True)
    subprocess.run(["cp", "-r", "--sparse=always", str(setup_waa.golden_storage()), str(storage)], check=True)
    suffix = f"-w{worker}" if own_memory else memory_suffix(args, worker)
    seed_memory_copy(args, share, unit[0], suffix)
    task_file = write_unit_file(args, label, pass_index, worker, serial, unit)
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
        f"./entry.sh --start-client true {client_command(llm, task_file, result_rel)}",
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
    finally:
        os.unlink(env_file.name)
    return name, storage


def container_running(name: str) -> bool:
    return subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", name],
                          capture_output=True, text=True).stdout.strip() == "true"


def take_waiting_tasks(workers: list[dict], thief: int, scored) -> tuple[int, str, list[str]] | None:
    """Tasks another VM has not started, for a VM that ran out of work: the
    back half of the longest waiting list. A VM's first unscored task is the
    one it is running, so it is never taken."""
    best = None
    for index, worker in enumerate(workers):
        if index == thief:
            continue
        domain, ids = worker["unit"]
        waiting = [task_id for task_id in ids if not scored(domain, task_id)][1:]
        if waiting and (best is None or len(waiting) > len(best[2])):
            best = (index, domain, waiting)
    if best is None:
        return None
    index, domain, waiting = best
    return index, domain, waiting[len(waiting) // 2:]


def client_running(name: str) -> bool:
    return subprocess.run(["docker", "exec", name, "pgrep", "-f", "python run.py"],
                          capture_output=True).returncode == 0


def find_task_dir(result_host: Path, llm: str, domain: str, task_id: str) -> Path:
    """A task's WAA result folder. WAA's result path includes the observation
    type, which changed between runs; find the task wherever it finished."""
    for match in result_host.glob(f"*/*/{llm}/0/{domain}/{task_id}"):
        if (match / "result.txt").exists() or (match / "pokai.json").exists():
            return match
    return result_host / WAA_ACTION_SPACE / "screenshot" / llm / "0" / domain / task_id


def pass_rows(args, llm: str, arm: str, pass_index: int, expected: list[tuple[str, str]], share: Path) -> list[dict]:
    """One results row per task of a pass, read from WAA's result folders."""
    label = f"{llm}-{arm}"
    result_host = setup_waa.container_dir() / "client" / f"results/pokai/{args.tag}/{label}/pass{pass_index}"
    rows = []
    stats = memory_stats(share / "memory")
    for domain, task_id in expected:
        folder = find_task_dir(result_host, llm, domain, task_id)
        result = (folder / "result.txt").read_text().strip() if (folder / "result.txt").exists() else None
        outcome = json.loads((folder / "pokai.json").read_text()) if (folder / "pokai.json").exists() else {}
        rows.append({
            "tag": args.tag, "llm": llm, "arm": arm, "pass": pass_index, "memory": args.memory,
            "domain": domain, "task": task_id,
            "score": parse_score(result),
            # The agent finished but WAA's checker crashed: a failed attempt.
            "eval_error": result is None and bool(outcome),
            **{key: outcome.get(key) for key in (
                "run_id", "session_id", "status", "seconds", "llm_requests", "prompt_tokens",
                "skills_loaded", "skill_outcomes", "fast_statuses", "infeasible", "run_failed")},
            **stats,
        })
    return rows


def finalize(args: argparse.Namespace) -> None:
    """Rebuild results.jsonl from the result folders of a run that was
    stopped before it wrote them. Runs nothing and keeps the manifest."""
    tasks = select_tasks(args.tasks)
    expected = [(domain, task_id) for domain, ids in tasks.items() for task_id in ids]
    results = RUNS / args.tag / "results.jsonl"
    rows = []
    for llm in args.llm.split(","):
        for arm in args.arms.split(","):
            share = RUNS / args.tag / f"{llm}-{arm}" / "share"
            for pass_index in range(1, args.passes + 1):
                rows.extend(pass_rows(args, llm, arm, pass_index, expected, share))
    results.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    finished = [row for row in rows if row["score"] is not None or row["eval_error"]]
    print(f"wrote {len(rows)} rows ({len(finished)} finished) to {results}")


def run_pass(args, llm: str, arm: str, pass_index: int, tasks: dict[str, list[str]], share: Path, env_keys: dict) -> list[dict]:
    label = f"{llm}-{arm}"
    pass_dir = RUNS / args.tag / label / f"pass{pass_index}"
    pass_dir.mkdir(parents=True, exist_ok=True)
    result_rel = f"results/pokai/{args.tag}/{label}/pass{pass_index}"
    result_host = setup_waa.container_dir() / "client" / result_rel

    def task_dir(domain: str, task_id: str) -> Path:
        return find_task_dir(result_host, llm, domain, task_id)

    def scored(domain: str, task_id: str) -> bool:
        # The agent's outcome (pokai.json) without a score means WAA's own
        # checker crashed on the result: the attempt counts as a failure and
        # is never run again, since one attempt per task is the protocol.
        folder = task_dir(domain, task_id)
        return (folder / "result.txt").exists() or (folder / "pokai.json").exists()

    def evaluated(domain: str, task_id: str) -> bool:
        return (task_dir(domain, task_id) / "result.txt").exists()

    if args.resume:
        # Keep earlier results; run only the tasks without a score.
        tasks = {domain: [task_id for task_id in ids if not scored(domain, task_id)] for domain, ids in tasks.items()}
        tasks = {domain: ids for domain, ids in tasks.items() if ids}
        if not tasks:
            print(f"[{label} pass {pass_index}] nothing to resume", flush=True)
            return []
    else:
        shutil.rmtree(result_host, ignore_errors=True)

    expected = [(domain, task_id) for domain, ids in tasks.items() for task_id in ids]
    queue = plan_units(tasks, args.workers, getattr(args, "split_apps", False))
    print(f"[{label} pass {pass_index}] {len(expected)} tasks in {len(queue)} units on "
          f"{min(args.workers, len(queue))} VM(s); a VM takes the next unit when it finishes one", flush=True)
    workers = []
    for worker in range(min(args.workers, len(queue))):
        unit = queue.pop(0)
        name, storage = start_worker(args, llm, label, pass_index, worker, unit, share, env_keys, result_rel, pass_dir)
        workers.append({"name": name, "storage": storage, "unit": unit, "serial": 0, "restarts": 0,
                        "client_seen": False, "started": time.monotonic()})
        print(f"  worker {worker}: {unit[0]} ({len(unit[1])} tasks); watch http://localhost:{args.viewer_port + worker}",
              flush=True)

    deadline = time.monotonic() + (len(expected) / max(len(workers), 1) + 5) * (args.timeout + 300) + 3600
    reported = 0
    try:
        while time.monotonic() < deadline:
            finished = sum(scored(domain, task_id) for domain, task_id in expected)
            if finished != reported:
                reported = finished
                print(f"[{label} pass {pass_index}] {finished}/{len(expected)} tasks scored", flush=True)
            active = 0
            clients_running = False
            for index, worker in enumerate(workers):
                domain, ids = worker["unit"]
                alive = container_running(worker["name"])
                client = alive and client_running(worker["name"])
                clients_running |= client
                # WAA scores a task after the agent's outcome is written: while
                # its client runs, only a score means done. Once the client has
                # exited, an outcome without a score is a checker crash.
                left = [task_id for task_id in ids
                        if not (evaluated(domain, task_id) or (not client and scored(domain, task_id)))]
                worker["client_seen"] |= client
                # WAA's own task setup can hang (a web page that never finishes
                # loading while it opens a task's tabs). The agent stops itself
                # at the time limit, so a task still without an outcome long
                # after that is stuck before the agent ever ran: stop the
                # client, and the restart below retries it on a fresh VM.
                if client and left:
                    # Timed from when this runner first saw the task as current,
                    # so a retried task is not judged by its earlier attempt.
                    if worker.get("current") != (domain, left[0]):
                        worker["current"], worker["current_since"] = (domain, left[0]), time.monotonic()
                    current = task_dir(domain, left[0])
                    if (not (current / "pokai.json").exists()
                            and time.monotonic() - worker["current_since"] > args.timeout + 900):
                        print(f"[{label} pass {pass_index}] worker {index}: {left[0][:12]} stuck in task setup; "
                              f"restarting the VM", flush=True)
                        subprocess.run(["docker", "exec", worker["name"], "pkill", "-f", "python run.py"],
                                       capture_output=True)
                        active += 1
                        continue
                    # WAA's checker can hang too: the agent's outcome is written
                    # but no score follows. Stop the client; the outcome without
                    # a score then counts as a checker failure (one attempt).
                    outcome = current / "pokai.json"
                    if (outcome.exists() and not (current / "result.txt").exists()
                            and time.time() - outcome.stat().st_mtime > 600):
                        print(f"[{label} pass {pass_index}] worker {index}: {left[0][:12]} checker hung; "
                              f"stopping the client", flush=True)
                        subprocess.run(["docker", "exec", worker["name"], "pkill", "-f", "python run.py"],
                                       capture_output=True)
                        active += 1
                        continue
                if not left:
                    # This VM finished its piece: share what it learned before
                    # it (or any VM) starts the next one.
                    if args.memory == "persist" and not worker.get("merged"):
                        merge_memory(share, [f"{domain}{memory_suffix(args, index)}"])
                        worker["merged"] = True
                    # Next work for this VM: the next unit, or else tasks a
                    # busy VM has not started (it re-reads its task file
                    # before each task, so shortening the file hands them over).
                    unit, own_memory, note = None, False, ""
                    if queue:
                        unit = queue.pop(0)
                    else:
                        taken = take_waiting_tasks(workers, index, scored)
                        if taken is not None:
                            busy_index, busy_domain, moved = taken
                            busy = workers[busy_index]
                            busy["unit"] = (busy_domain, [t for t in busy["unit"][1] if t not in moved])
                            write_unit_file(args, label, pass_index, busy_index, busy["serial"], busy["unit"])
                            unit, own_memory = (busy_domain, moved), True
                            note = f" taken from worker {busy_index}"
                    if unit is None:
                        if alive:
                            subprocess.run(["docker", "stop", worker["name"]], capture_output=True)
                        continue
                    worker["unit"] = unit
                    worker["own_memory"] = own_memory
                    worker["merged"] = False
                    worker["serial"] += 1
                    worker["client_seen"] = False
                    worker["started"] = time.monotonic()
                    worker.pop("current", None)
                    # The container stops as soon as its client finishes, so a
                    # client started inside it would die with it: boot a fresh
                    # VM, which also gives the next application a clean Windows.
                    with open(pass_dir / f"container-w{index}-unit{worker['serial'] - 1}.log", "w",
                              encoding="utf-8") as log:
                        subprocess.run(["docker", "logs", worker["name"]], stdout=log, stderr=subprocess.STDOUT)
                    subprocess.run(["docker", "rm", "-f", worker["name"]], capture_output=True)
                    shutil.rmtree(worker["storage"], ignore_errors=True)
                    worker["name"], worker["storage"] = start_worker(
                        args, llm, label, pass_index, index, unit, share, env_keys, result_rel,
                        pass_dir, worker["serial"], own_memory)
                    print(f"  worker {index} -> {unit[0]} ({len(unit[1])} tasks){note}", flush=True)
                    active += 1
                    continue
                active += 1
                booting = not worker["client_seen"] and time.monotonic() - worker["started"] < 1200
                if client or booting or worker["restarts"] >= 2:
                    continue
                # The WAA client stopped with tasks left (for example a task
                # setup crashed): run the rest again, on a fresh VM if the
                # container itself is gone.
                worker["restarts"] += 1
                with open(pass_dir / f"container-w{index}-crash{worker['restarts']}.log", "w", encoding="utf-8") as log:
                    subprocess.run(["docker", "logs", worker["name"]], stdout=log, stderr=subprocess.STDOUT)
                worker["unit"] = (domain, left)
                worker["serial"] += 1
                worker["client_seen"] = False
                worker["started"] = time.monotonic()
                worker.pop("current", None)
                print(f"[{label} pass {pass_index}] worker {index} client stopped with {len(left)} tasks left; "
                      f"restarting it", flush=True)
                # Always on a fresh VM: a container whose client stopped is
                # shutting down.
                subprocess.run(["docker", "rm", "-f", worker["name"]], capture_output=True)
                shutil.rmtree(worker["storage"], ignore_errors=True)
                worker["name"], worker["storage"] = start_worker(
                    args, llm, label, pass_index, index, worker["unit"], share, env_keys, result_rel,
                    pass_dir, worker["serial"], worker.get("own_memory", False))
            # Done when every task is scored, or when the remaining outcomes
            # are checker crashes (no client is still scoring them).
            all_scored = all(evaluated(domain, task_id) for domain, task_id in expected)
            if all_scored or (finished == len(expected) and not clients_running) or (not active and not queue):
                break
            time.sleep(30)
    finally:
        for index, worker in enumerate(workers):
            with open(pass_dir / f"container-w{index}.log", "w", encoding="utf-8") as log:
                subprocess.run(["docker", "logs", worker["name"]], stdout=log, stderr=subprocess.STDOUT)
            subprocess.run(["docker", "stop", worker["name"]], capture_output=True)
            subprocess.run(["docker", "rm", "-f", worker["name"]], capture_output=True)
            if not args.keep_storage:
                shutil.rmtree(worker["storage"], ignore_errors=True)
        # Everything learned in this pass, in one memory per application.
        if args.memory == "persist":
            merge_all_memory(share)

    return pass_rows(args, llm, arm, pass_index, expected, share)


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
        "step_budget": getattr(args, "step_budget", 0),
        "seed_memory": getattr(args, "seed_memory", None),
        "instruction": "unmodified; infeasible-task convention given as a standing instruction",
        "tasks": sum(map(len, tasks.values())), "domains": sorted(tasks),
        "harness_commit": git("rev-parse", "HEAD"),
        "harness_uncommitted_changes": bool(git("status", "--porcelain", "--untracked-files=no")),
        "waa_commit": git("rev-parse", "HEAD", cwd=setup_waa.waa_dir()),
        "exe_sha256": subprocess.run(["sha256sum", str(exe)], capture_output=True, text=True).stdout.split(" ")[0],
        # Changes made to the golden image after WAA prepared it (for
        # example a newer LibreOffice), so runs on different images are
        # never compared unknowingly.
        "golden_image": setup_waa.golden_notes(),
    }
    path = RUNS / args.tag / "manifest.json"
    if getattr(args, "resume", False) and path.exists():
        # Keep the original record; note what the resume ran with.
        original = json.loads(path.read_text(encoding="utf-8"))
        original.setdefault("resumes", []).append(manifest)
        manifest = original
    path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")


def seed_from_run(args, share: Path, llm: str, arm: str, domains) -> None:
    """Start this run's memory from another run's final memory (`--seed-memory
    TAG`), so learned skills are in use from the first task. Compare the result
    with that run's later passes, which started from the same memory."""
    source = RUNS / args.seed_memory / f"{llm}-{arm}" / "share" / "memory"
    if not source.exists():
        sys.exit(f"--seed-memory: no memory at {source}")
    (share / "memory").mkdir(parents=True, exist_ok=True)
    for domain in sorted(domains):
        if not (source / domain).exists() or (share / "memory" / domain).exists():
            continue
        # Copied inside the container: files the VM wrote may be root-owned.
        subprocess.run(
            ["docker", "run", "--rm", "-v", f"{source}:/source:ro", "-v", f"{share / 'memory'}:/memory",
             "--entrypoint", "cp", setup_waa.WAA_IMAGE, "-a", f"/source/{domain}", f"/memory/{domain}"],
            check=True)
        print(f"  memory {domain} seeded from {args.seed_memory}", flush=True)


def run(args: argparse.Namespace) -> None:
    if not (setup_waa.golden_storage().exists() and any(setup_waa.golden_storage().iterdir())):
        sys.exit("no golden VM image; see: python3 scripts/arena/setup_waa.py status")
    tasks = select_tasks(args.tasks)
    # Keep the disk clean: an earlier stopped or crashed run's VM disks go
    # before this run makes new ones.
    remove_leftover_disks()
    stale = setup_waa.container_dir() / "client" / "results" / "pokai" / args.tag
    if not args.resume and stale.exists():
        # Old scores would count as done and their tasks would never run.
        sys.exit(f"results for tag {args.tag} already exist ({stale}); use a new tag or --resume")
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
            (share / "bin" / "arena.toml").write_text(arena_config(llm, arm, args.step_budget), encoding="utf-8")
            shutil.copy2(HERE / "vm" / "run-task.ps1", share / "run-task.ps1")
            shutil.copy2(HERE / "merge_memory.py", share / "merge_memory.py")
            if args.seed_memory and args.memory == "persist":
                seed_from_run(args, share, llm, arm, tasks)
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
    # Success over scored tasks, and over all tasks (unscored count as failed).
    print(f"{'llm':8}{'arm':8}{'memory':9}{'pass':>5}{'n':>5}{'scored':>7}{'success':>9}{'of_all':>8}{'llm_req':>9}{'sec':>7}{'skill_use':>10}{'skills':>8}{'dups':>6}")
    for (llm, arm, memory, pass_index), group in sorted(groups.items()):
        requests = [row["llm_requests"] for row in group if row.get("llm_requests") is not None]
        seconds = [row["seconds"] for row in group if row.get("seconds") is not None]
        used = sum(bool(row.get("skills_loaded")) for row in group)
        last = group[-1]
        scored = [row for row in group if row.get("score") is not None]
        passed = sum(row["score"] for row in scored)
        print(
            f"{llm:8}{arm:8}{memory:9}{pass_index:>5}{len(group):>5}{len(scored):>7}"
            f"{passed / max(len(scored), 1):>9.1%}{passed / len(group):>8.1%}"
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
    teacher = args.teacher
    if not teacher:
        # The decision backend the run itself used supplies the soft labels.
        manifest = json.loads((RUNS / args.tag / "manifest.json").read_text(encoding="utf-8"))
        backends = [arm.get("backend") for arm in manifest.get("arms", {}).values() if arm.get("backend")]
        teacher = backends[0] if backends else ""
    with open(outcomes, "w", encoding="utf-8") as handle:
        for row in load_rows(args.tag):
            if row.get("session_id") and row.get("score") is not None:
                handle.write(json.dumps({"session_id": row["session_id"], "score": row["score"]}) + "\n")
    command = [
        sys.executable, str(REPO / "scripts" / "build_router_dataset.py"),
        "--input", str(RUNS / args.tag), "--output", args.output,
        "--outcomes", str(outcomes), "--teacher-successful-only",
    ]
    if teacher:
        command += ["--teacher", teacher]
    subprocess.run(command, check=True)


def remove_leftover_disks() -> int:
    """Delete VM disks a stopped or crashed run left behind (about 20 GB per
    VM). Results, traces, training data, and memory are kept. Does nothing
    while any arena VM is running, so a live run is never touched. Returns how
    many disk folders were removed, or -1 when a VM is running."""
    running = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True).stdout
    if any(name.startswith("pokai-") for name in running.split()):
        return -1
    leftovers = sorted(RUNS.glob("*/*/pass*/storage-w*"))
    if leftovers:
        # The VM writes these as root; remove them from inside a container.
        subprocess.run(
            ["docker", "run", "--rm", "-v", f"{RUNS}:/runs", "--entrypoint", "rm", setup_waa.WAA_IMAGE, "-rf",
             *[f"/runs/{path.relative_to(RUNS)}" for path in leftovers]],
            check=True)
        for path in leftovers:
            print(f"removed leftover VM disk {path.relative_to(RUNS)}", flush=True)
    return len(leftovers)


def clean(_: argparse.Namespace) -> None:
    removed = remove_leftover_disks()
    if removed < 0:
        sys.exit("an arena VM is running; stop the run first")
    print(f"removed {removed} leftover VM disk folder(s)")


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
    run_parser.add_argument("--workers", type=int, default=5, help="VMs in parallel; each takes the next application when it finishes one")
    run_parser.add_argument("--step-budget", type=int, default=30,
                            help="acting planner turns per task (a fast_actions plan is one); 0 = unlimited")
    run_parser.add_argument("--seed-memory", metavar="TAG",
                            help="start from the final memory of run TAG (same planner and arm) instead of empty")
    run_parser.add_argument("--keep-storage", action="store_true", help="keep each pass's VM disk")
    run_parser.add_argument("--resume", action="store_true", help="keep existing scores and run only unscored tasks")
    run_parser.add_argument("--split-apps", action=argparse.BooleanOptionalAction, default=True,
                            help="divide applications across VMs so every VM has work (each piece gets its own "
                                 "copy of that app's memory); --no-split-apps keeps one VM per application")
    finalize_parser = sub.add_parser("finalize", help="rebuild results.jsonl of a stopped run")
    finalize_parser.add_argument("--tag", required=True)
    finalize_parser.add_argument("--llm", default="bunny")
    finalize_parser.add_argument("--arms", default="jev")
    finalize_parser.add_argument("--tasks", default="all")
    finalize_parser.add_argument("--passes", type=int, default=1)
    finalize_parser.add_argument("--memory", choices=["persist", "fresh"], default="persist")
    sub.add_parser("clean", help="delete VM disks left behind by a stopped or crashed run")
    summary_parser = sub.add_parser("summary")
    summary_parser.add_argument("--tag", required=True)
    dataset_parser = sub.add_parser("dataset")
    dataset_parser.add_argument("--tag", required=True)
    dataset_parser.add_argument("--output", required=True)
    dataset_parser.add_argument("--teacher", help="decision backend whose confident answers become soft labels "
                                "(default: the backend the run used)")
    args = parser.parse_args()
    {"run": run, "summary": summary, "dataset": dataset, "finalize": finalize, "clean": clean}[args.command](args)


if __name__ == "__main__":
    main()
