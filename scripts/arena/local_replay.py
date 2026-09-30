"""Replay Windows Agent Arena tasks on this Windows PC (run with Windows Python).

Useful for watching POK-Ai work on a benchmark task in real applications
(LibreOffice, VLC, ...) and telling harness problems from planner mistakes.
It does what WAA's setup does for "download" and "open" steps: the task's own
sample files are downloaded into a dedicated test folder and opened. It then
runs pok-ai.exe on the unmodified instruction.

Everything stays in the test folder (default %USERPROFILE%\\POK-Ai-Arena-Tests):
the task files, POK-Ai's memory for these runs, and the logs. Your normal
POK-Ai memory and documents are not used. Tasks whose setup needs more than
downloading and opening files are reported and skipped; replay those in the
arena instead.

    python scripts\\arena\\local_replay.py --waa-dir \\\\wsl.localhost\\Ubuntu\\home\\<you>\\arena\\WindowsAgentArena ^
        --task libreoffice_calc/7a4e4bc8-d9f5-4ad0-9b6b-05c1b4f8cda1-WOS --exe <pok-ai.exe>

Scoring still happens in the arena; here you check the result yourself.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

VM_HOME = "C:\\Users\\Docker\\"
OFFICE_EXTENSIONS = {".xlsx", ".xls", ".ods", ".csv", ".docx", ".doc", ".odt", ".rtf", ".pptx", ".odp"}
SOFFICE = Path(os.environ.get("ProgramFiles", "C:\\Program Files")) / "LibreOffice" / "program" / "soffice.exe"


def local_path(vm_path: str, test_dir: Path) -> Path:
    """The test-folder equivalent of a path in WAA's VM (C:\\Users\\Docker\\...)."""
    if vm_path.lower().startswith(VM_HOME.lower()):
        return test_dir / vm_path[len(VM_HOME):]
    return test_dir / Path(vm_path).name


def read_credential(name: str) -> str:
    return subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
         str(REPO / "scripts" / "read-credential.ps1"), "-Name", name],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def setup_task(task: dict, test_dir: Path) -> list[str]:
    """Apply the task's download and open steps; return steps that were skipped."""
    skipped = []
    for step in task.get("config", []):
        kind, params = step.get("type"), step.get("parameters", {})
        if kind == "download":
            for item in params.get("files", []):
                target = local_path(item["path"], test_dir)
                target.parent.mkdir(parents=True, exist_ok=True)
                print(f"  download {item['url']} -> {target}")
                urllib.request.urlretrieve(item["url"], target)
        elif kind == "open":
            target = local_path(params["path"], test_dir)
            print(f"  open {target}")
            if target.suffix.lower() in OFFICE_EXTENSIONS and SOFFICE.exists():
                subprocess.Popen([str(SOFFICE), str(target)])
            else:
                os.startfile(target)  # noqa: S606 - the task's own file, in the test folder
        elif kind == "sleep":
            time.sleep(float(params.get("seconds", 1)))
        else:
            skipped.append(kind)
    return skipped


def local_config(llm: str, arm: str, test_dir: Path, step_budget: int) -> Path:
    import run_arena
    from live_bench import set_key

    lines = run_arena.arena_config(llm, arm, step_budget).splitlines()
    # Top-level keys come before the first [section].
    split = next((index for index, line in enumerate(lines) if line.startswith("[")), len(lines))
    head, rest = lines[:split], lines[split:]
    head = set_key(head, "data_dir", json.dumps(str(test_dir / "pokai-data").replace("\\", "/")))
    head = set_key(head, "diagnostics_dir", json.dumps(str(test_dir / "pokai-diag").replace("\\", "/")))
    path = test_dir / f"pokai-{llm}-{arm}.toml"
    path.write_text("\n".join(head + rest) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--waa-dir", required=True, help="WindowsAgentArena checkout (a \\\\wsl.localhost path works)")
    parser.add_argument("--task", action="append", required=True, help="domain/id (repeatable)")
    parser.add_argument("--exe", required=True, help="pok-ai.exe to run")
    parser.add_argument("--llm", default="bunny")
    parser.add_argument("--arm", default="jev")
    parser.add_argument("--step-budget", type=int, default=0, help="0 = unlimited")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--test-dir", default=str(Path.home() / "POK-Ai-Arena-Tests"))
    args = parser.parse_args()

    import run_arena
    from live_bench import ARMS, LLMS

    test_dir = Path(args.test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    config = local_config(args.llm, args.arm, test_dir, args.step_budget)
    key_names = [run_arena.provider_key_env(LLMS[args.llm]["provider"])]
    if ARMS[args.arm].get("enabled", True) and ARMS[args.arm].get("backend", "jev") == "jev":
        key_names.append("TYPESAFE_API_KEY")
    env = dict(os.environ, **{name: read_credential(name) for name in key_names if name})

    examples = Path(args.waa_dir) / "src" / "win-arena-container" / "client" / "evaluation_examples_windows" / "examples"
    for spec in args.task:
        domain, task_id = spec.split("/", 1)
        task = json.loads((examples / domain / f"{task_id}.json").read_text(encoding="utf-8"))
        print(f"\n=== {spec}\n{task['instruction']}")
        skipped = setup_task(task, test_dir)
        if skipped:
            print(f"  skipped: setup steps {skipped} need the arena VM")
            continue
        time.sleep(8)  # let the document finish opening, as WAA's setup does
        log = test_dir / "logs" / f"{domain}-{task_id[:12]}-{time.strftime('%H%M%S')}.log"
        log.parent.mkdir(exist_ok=True)
        started = time.monotonic()
        with open(log, "w", encoding="utf-8", errors="replace") as handle:
            process = subprocess.Popen(
                [args.exe, "--config", str(config), "run", task["instruction"], "--yes"],
                cwd=test_dir, env=env, stdout=handle, stderr=subprocess.STDOUT,
            )
            try:
                process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                print("  stopped at the time limit")
        text = log.read_text(encoding="utf-8", errors="replace")
        artifacts = re.search(r"Artifacts: (.+)", text)
        print(f"  finished in {round(time.monotonic() - started)} s; log {log}")
        if artifacts:
            print(f"  trace {artifacts.group(1).strip()}")
        print("  answer:", text.split("External JEV warning")[-1].strip()[-600:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
