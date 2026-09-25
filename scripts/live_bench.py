"""Live computer-use benchmark for router architectures (run on native Windows).

Runs the read-only tasks in scripts/live-bench-tasks.json against a matrix of
primary LLMs and router arms, then summarises success, primary-model
requests, tokens, time and fast-action outcomes from each run's trace.

It is Windows-native on purpose: the agent's output goes to files, never to a
pipe, so processes the agent launches cannot keep a WSL interop pipe open,
and each run has a hard timeout that kills the whole process tree.

Before each run the LM Studio server is set to hold exactly the models that
run needs (planner and, for llm_choice arms, the router model), unloading
everything else first.

Usage (Windows Python, from the repository root):
    python scripts\\live_bench.py run --llm qwen --arms none,grounding,laya,laya_judge,jev --tasks book,display --reps 3
    python scripts\\live_bench.py summary
Arguments are documented in --help. Results go to diagnostics/bench/live/
(ignored by git; traces can contain private screen text).

Runs never touch the user's own memory, skills, USER.md, or browser profile:
each gets a data folder under diagnostics/bench/live/data/. With
`--memory arm` (the default) the runs of one arm share a folder, so later
repetitions can reuse skills the first ones learned; `--memory run` gives
every run a fresh folder, a no-learning baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import statistics
import subprocess
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lmstudio_models  # noqa: E402

LIVE_DIR = REPO / "diagnostics" / "bench" / "live"
RESULTS = LIVE_DIR / "results.jsonl"
LM_SERVER = os.environ.get("POK_LM_SERVER", "http://169.254.83.107:512")

LLMS = {
    "qwen": {
        "provider": "lan_lm_studio",
        "model": "qwen3.8-27b@q2_k_xl",
        "lm_models": ["qwen3.8-27b@q2_k_xl"],
    },
    "deepseek": {"provider": "openrouter", "model": "deepseek/deepseek-v4.1-flash", "lm_models": []},
    # Free OpenRouter planner with vision and tools, for when the GPU is busy
    # with a local decision model.
    "nexpro": {"provider": "openrouter", "model": "nex-agi/nex-n2.5-pro:free", "lm_models": []},
    # Free and ~20x faster than nex-pro in a tool-call probe; stealth models may
    # log prompts, so use it only on non-private tasks.
    "bunny": {"provider": "openrouter", "model": "stealth/space-bunny-alpha", "lm_models": []},
    "luna": {"provider": "openrouter", "model": "openai/gpt-6-luna", "lm_models": []},
}

# Router arms. "trust" sets trust_model_targets and trust_model_conditions.
ARMS = {
    "none": {"enabled": False},
    "grounding": {"backend": "laya", "trust": False},
    "laya": {"backend": "laya", "trust": True},
    "laya_judge": {"backend": "laya", "trust": True, "judge": True},
    # A hosted backend needs room for network latency; local backends
    # already get at least 12 s, so give it the 10 s hosted maximum.
    "jev": {"backend": "jev", "timeout_ms": 10000},
    "lfm": {"backend": "llm_choice", "trust": True, "llm_endpoint": "http://127.0.0.1:43211/v1/systemone",
            "llm_model": "lfm2.5-8b-a1b"},
    "minicpm": {"backend": "llm_choice", "trust": True, "llm_endpoint": "http://127.0.0.1:43212/v1/systemone",
                "llm_model": "minicpm5-2b@q8_0"},
    # Local typed-contract servers reached through the local-backend path.
    "kev": {"backend": "laya", "trust": True, "laya_endpoint": "http://127.0.0.1:43216/v1/systemone",
            "laya_model": "jaredpalmer/kev-4b", "laya_token_env": "POK_KEV_TOKEN"},
    "zeiger": {"backend": "laya", "trust": True, "laya_endpoint": "http://127.0.0.1:43217/v1/systemone",
               "laya_model": "php-ai/zeiger-0.6b@r17", "laya_token_env": "POK_JUDGE_TOKEN"},
}

DATA = Path(os.environ.get("LOCALAPPDATA", "")) / "POK-Ai" / "POK-Ai" / "data"
BENCH_PY = DATA / "openjev-bench" / "venv" / "Scripts" / "python.exe"
# The local decision-model process each arm needs: (command, working dir,
# extra environment, health URL, token). Started before the arm's runs and
# stopped after, so only one decision model holds GPU memory at a time.
SIDECARS = {
    "grounding": "laya", "laya": "laya", "laya_judge": "laya",
    "kev": "kev", "zeiger": "zeiger",
}
SIDECAR_SPECS = {
    "laya": ([str(DATA / "laya" / "runtime" / "Scripts" / "python.exe"), str(REPO / "scripts" / "laya_sidecar.py"),
              "--model-dir", str(DATA / "laya" / "models" / "typed-decisions"), "--port", "43127", "--device", "auto"],
             None, {"POK_LAYA_TOKEN": "dev-laya-token"}, "http://127.0.0.1:43127/health", "dev-laya-token"),
    "kev": ([str(BENCH_PY), "-m", "kev.serve", "--run", "jaredpalmer/kev-4b", "--port", "43216"],
            str(DATA / "kev" / "kev-repo"), {"KEV_API_KEY": "dev-kev-token"}, "http://127.0.0.1:43216/v1/models",
            "dev-kev-token"),
    "zeiger": ([str(DATA / "judge" / "runtime" / "Scripts" / "python.exe"), str(REPO / "scripts" / "zeiger_sidecar.py"),
                "--model-dir", str(DATA / "judge" / "models" / "zeiger-0.6b-r17"), "--port", "43217", "--device", "auto"],
               None, {"POK_JUDGE_TOKEN": "dev-judge-token", "PYTHONPATH": str(DATA / "judge" / "zeiger")},
               "http://127.0.0.1:43217/health", "dev-judge-token"),
}


def start_sidecar(name: str) -> subprocess.Popen:
    command, cwd, extra, health, token = SIDECAR_SPECS[name]
    log = open(LIVE_DIR / "logs" / f"sidecar-{name}.log", "a", encoding="utf-8")
    process = subprocess.Popen(command, cwd=cwd, env={**os.environ, **extra}, stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        try:
            request = urllib.request.Request(health, headers={"Authorization": f"Bearer {token}"})
            urllib.request.urlopen(request, timeout=3).read()
            print(json.dumps({"sidecar": name, "status": "ready"}), flush=True)
            return process
        except OSError:
            if process.poll() is not None:
                raise SystemExit(f"{name} sidecar exited during startup; see logs/sidecar-{name}.log")
            time.sleep(5)
    raise SystemExit(f"{name} sidecar did not become ready")


def stop_sidecar(process: subprocess.Popen | None) -> None:
    if process is not None and process.poll() is None:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)], capture_output=True)


def set_key(section: list[str], key: str, value: str) -> list[str]:
    pattern = re.compile(rf"^#?\s*{re.escape(key)}\s*=")
    for index, line in enumerate(section):
        if pattern.match(line):
            section[index] = f"{key} = {value}"
            return section
    return section[:1] + [f"{key} = {value}"] + section[1:]


def arm_config(llm: str, arm: str, data_dir: Path) -> str:
    """pok-ai.toml with the arm's router settings, the LLM provider, and an
    isolated data folder."""
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
            section = set_key(section, "data_dir", json.dumps(str(data_dir).replace("\\", "/")))
            section = set_key(section, "diagnostics_dir", json.dumps(str(LIVE_DIR).replace("\\", "/")))
            section = set_key(section, "default_provider", json.dumps(model["provider"]))
            section = set_key(section, "default_model", json.dumps(model["model"]))
        elif header == "[decision_router]":
            section = set_key(section, "enabled", "true" if spec.get("enabled", True) else "false")
            section = set_key(section, "backend", json.dumps(spec.get("backend", "jev")))
            section = set_key(section, "mode", '"delegated"')
            if "timeout_ms" in spec:
                section = set_key(section, "timeout_ms", str(spec["timeout_ms"]))
            if "trust" in spec:
                section = set_key(section, "trust_model_targets", "true" if spec["trust"] else "false")
                section = set_key(section, "trust_model_conditions", "true" if spec["trust"] else "false")
        elif header == "[decision_router.judge]":
            section = set_key(section, "enabled", "true" if spec.get("judge") else "false")
        elif header == "[decision_router.laya]" and "laya_endpoint" in spec:
            section = set_key(section, "endpoint", json.dumps(spec["laya_endpoint"]))
            section = set_key(section, "model", json.dumps(spec["laya_model"]))
            section = set_key(section, "token_env", json.dumps(spec["laya_token_env"]))
        elif header == "[decision_router.llm_choice]" and "llm_endpoint" in spec:
            section = set_key(section, "endpoint", json.dumps(spec["llm_endpoint"]))
            section = set_key(section, "model", json.dumps(spec["llm_model"]))
        out.extend(section)
    if llm == "qwen":
        out += [
            "",
            "[providers.lan_lm_studio]",
            'protocol = "openai_compatible"',
            f'base_url = "{LM_SERVER}/v1"',
            'data_boundary = "local_device"',
            'api_key_env = "LM_STUDIO_API_KEY"',
        ]
    return "\n".join(out) + "\n"


def powershell(script: str, timeout: float = 60) -> str:
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL,
    )
    return result.stdout.strip()


def read_credential(name: str) -> str:
    return powershell(f"& '{REPO / 'scripts' / 'read-credential.ps1'}' -Name {name}")


def process_running(name: str) -> bool:
    return bool(powershell(f"Get-Process {name} -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Id"))


# Top-level windows are listed as "handle<TAB>process<TAB>title". Used to
# close only windows that appeared during testing (never the user's own).
LIST_WINDOWS = (
    "Add-Type -Name Top -Namespace PokList -MemberDefinition '"
    "public delegate bool EnumProc(System.IntPtr h, System.IntPtr l); "
    "[DllImport(\"user32.dll\")] public static extern bool EnumWindows(EnumProc f, System.IntPtr l); "
    "[DllImport(\"user32.dll\", CharSet=CharSet.Unicode)] public static extern int GetWindowText(System.IntPtr h, System.Text.StringBuilder s, int n); "
    "[DllImport(\"user32.dll\")] public static extern bool IsWindowVisible(System.IntPtr h); "
    "[DllImport(\"user32.dll\")] public static extern uint GetWindowThreadProcessId(System.IntPtr h, out uint p);'; "
    "$rows = New-Object System.Collections.Generic.List[string]; "
    "[PokList.Top]::EnumWindows({ param($h, $l) if ([PokList.Top]::IsWindowVisible($h)) { "
    "$sb = New-Object System.Text.StringBuilder 512; [void][PokList.Top]::GetWindowText($h, $sb, 512); "
    "if ($sb.Length -gt 0) { $p = 0; [void][PokList.Top]::GetWindowThreadProcessId($h, [ref]$p); "
    "$name = (Get-Process -Id $p -ErrorAction SilentlyContinue).ProcessName; "
    "$rows.Add(\"$([int64]$h)`t$name`t$($sb.ToString())\") } }; $true }, [System.IntPtr]::Zero) | Out-Null; "
    "$rows -join \"`n\""
)
# Processes whose new windows are test leftovers (apps the tasks open, shell
# dialogs and consoles the agent may spawn).
TEST_PROCESSES = {"explorer", "applicationframehost", "systemsettings", "cmd", "conhost", "notepad"}


def running_processes() -> set[str]:
    out = powershell("(Get-Process | Select-Object -ExpandProperty ProcessName | Sort-Object -Unique) -join ','")
    return {name.strip().lower() for name in out.split(",") if name.strip()}


def expand_path(path: str) -> Path:
    return Path(os.path.expandvars(path))


def document_text(path: Path) -> str:
    """Plain text of a file; for .docx, its paragraphs (runs joined)."""
    if path.suffix.lower() == ".docx":
        import zipfile
        xml = zipfile.ZipFile(path).read("word/document.xml").decode("utf-8", "replace")
        paragraphs = re.findall(r"<w:p[ >].*?</w:p>", xml, flags=re.S)
        return "\n".join("".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", p, flags=re.S)) for p in paragraphs)
    return path.read_text(encoding="utf-8", errors="replace")


def check_file(path: Path, expected: list[str]) -> str:
    if not path.exists():
        return "missing"
    try:
        text = document_text(path)
    except Exception as error:  # noqa: BLE001 - reported, not raised
        return f"unreadable: {error}"
    missing = [item for item in expected if item not in text]
    return "ok" if not missing else f"missing text: {missing}"


def list_windows() -> list[tuple[int, str, str]]:
    rows = []
    for line in powershell(LIST_WINDOWS).splitlines():
        parts = line.split("\t", 2)
        if len(parts) == 3 and parts[0].lstrip("-").isdigit():
            rows.append((int(parts[0]), parts[1].lower(), parts[2]))
    return rows


def close_new_test_windows(baseline: set[int]) -> list[str]:
    """Close windows that appeared since the matrix started and belong to the
    apps the tasks use; the user's own windows are in the baseline."""
    closed = []
    for handle, process, title in list_windows():
        if handle in baseline or process not in TEST_PROCESSES:
            continue
        # Never close the desktop shell itself (taskbar/desktop windows).
        if process == "explorer" and title in ("", "Program Manager"):
            continue
        powershell(
            "Add-Type -Name Post -Namespace PokClose -MemberDefinition "
            "'[DllImport(\"user32.dll\")] public static extern bool PostMessage(System.IntPtr h, uint m, System.IntPtr w, System.IntPtr l);'; "
            f"[void][PokClose.Post]::PostMessage([System.IntPtr]{handle}, 0x0010, [System.IntPtr]::Zero, [System.IntPtr]::Zero)"
        )
        closed.append(f"{process}: {title[:60]}")
    if closed:
        time.sleep(1.5)
    return closed


RESET = {
    "close_settings": "Get-Process SystemSettings -ErrorAction SilentlyContinue | Stop-Process -Force",
    # Shell.Application().Windows().Quit() misses Windows 11 tabbed Explorer
    # windows, which left Program Files open for the next run. Close every
    # top-level "File Explorer" window with a normal WM_CLOSE instead.
    "close_explorer": (
        "Add-Type -Name Wnd -Namespace PokReset -MemberDefinition '"
        "public delegate bool EnumProc(System.IntPtr h, System.IntPtr l); "
        "[DllImport(\"user32.dll\")] public static extern bool EnumWindows(EnumProc f, System.IntPtr l); "
        "[DllImport(\"user32.dll\", CharSet=CharSet.Unicode)] public static extern int GetWindowText(System.IntPtr h, System.Text.StringBuilder s, int n); "
        "[DllImport(\"user32.dll\")] public static extern bool IsWindowVisible(System.IntPtr h); "
        "[DllImport(\"user32.dll\")] public static extern bool PostMessage(System.IntPtr h, uint m, System.IntPtr w, System.IntPtr l);'; "
        "$handles = New-Object System.Collections.Generic.List[System.IntPtr]; "
        "[PokReset.Wnd]::EnumWindows({ param($h, $l) $sb = New-Object System.Text.StringBuilder 512; "
        "[void][PokReset.Wnd]::GetWindowText($h, $sb, 512); "
        "if ([PokReset.Wnd]::IsWindowVisible($h) -and $sb.ToString() -like '*File Explorer*') { $handles.Add($h) }; $true }, "
        "[System.IntPtr]::Zero) | Out-Null; "
        "foreach ($h in $handles) { [void][PokReset.Wnd]::PostMessage($h, 0x0010, [System.IntPtr]::Zero, [System.IntPtr]::Zero) }; "
        "Start-Sleep -Milliseconds 800; $handles.Count"
    ),
}
# A Start or Search panel left open by an earlier run steals the foreground.
DISMISS_SHELL = (
    "Add-Type -Name Fg -Namespace Pok -MemberDefinition '[DllImport(\"user32.dll\")] public static extern System.IntPtr GetForegroundWindow(); "
    "[DllImport(\"user32.dll\")] public static extern uint GetWindowThreadProcessId(System.IntPtr h, out uint p);'; "
    "$p = 0; [Pok.Fg]::GetWindowThreadProcessId([Pok.Fg]::GetForegroundWindow(), [ref]$p) | Out-Null; "
    "if ((Get-Process -Id $p).ProcessName -match 'SearchHost|StartMenuExperienceHost') "
    "{ (New-Object -ComObject WScript.Shell).SendKeys('{ESC}') }"
)


def trace_metrics(artifact_dir: Path) -> dict:
    metrics: dict = {}
    statuses: list[str] = []
    for line in (artifact_dir / "trace.jsonl").read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind, payload = record.get("kind"), record.get("payload") or {}
        if kind == "run_completed":
            metrics["answer"] = payload.get("answer") or ""
            metrics["run_metrics"] = payload.get("metrics") or {}
        elif kind == "run_failed":
            metrics["run_failed"] = payload.get("error")
        elif kind == "fast_actions_finished":
            statuses.append(payload.get("status"))
    metrics["fast_statuses"] = statuses
    return metrics


def skill_counts(data_dir: Path) -> dict:
    """Learned skills in a bench data folder: how many, and how many tasks
    they cover (more skills than tasks means duplicates)."""
    database = data_dir / "memory" / "memory.db"
    if not database.exists():
        return {"skills": 0, "skill_tasks": 0}
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        skills, tasks, successes = connection.execute(
            "SELECT COUNT(*), COUNT(DISTINCT kind || task_signature || applications), COALESCE(SUM(success_count), 0) "
            "FROM procedures WHERE kind = 'workflow'"
        ).fetchone()
    return {"skills": skills, "skill_tasks": tasks, "skill_successes": successes}


def trace_memory(artifact_dir: Path) -> dict:
    """Skills given to the model and what the run learned."""
    loaded, learned = [], []
    for line in (artifact_dir / "trace.jsonl").read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind, payload = record.get("kind"), record.get("payload") or {}
        if kind == "skill_auto_loaded":
            loaded.append(payload.get("title"))
        elif kind in {"procedure_learned", "procedure_reinforced", "procedure_not_learned", "procedure_rejected"}:
            learned.append(payload.get("outcome") or kind.removeprefix("procedure_"))
    return {"skills_loaded": loaded, "skill_outcomes": learned}


def run(args: argparse.Namespace) -> None:
    tasks = {task["id"]: task for task in json.loads((REPO / "scripts" / "live-bench-tasks.json").read_text())["tasks"]}
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    (LIVE_DIR / "configs").mkdir(exist_ok=True)
    (LIVE_DIR / "logs").mkdir(exist_ok=True)
    # Windows open before testing (the user's own) are never closed.
    args.window_baseline = {handle for handle, _, _ in list_windows()}
    # Nor are applications that were already running.
    args.initial_processes = running_processes()
    env = dict(os.environ)
    env["OPENROUTER_API_KEY"] = read_credential("OPENROUTER_API_KEY")
    env["TYPESAFE_API_KEY"] = read_credential("TYPESAFE_API_KEY")
    env.setdefault("POK_LAYA_TOKEN", "dev-laya-token")
    env.setdefault("POK_JUDGE_TOKEN", "dev-judge-token")
    env.setdefault("POK_LLM_CHOICE_TOKEN", "dev-llm-token")
    env.setdefault("POK_KEV_TOKEN", "dev-kev-token")
    for llm in args.llm.split(","):
        for arm in args.arms.split(","):
            config_path = LIVE_DIR / "configs" / f"{llm}-{arm}.toml"
            config_path.write_text(arm_config(llm, arm, LIVE_DIR / "data" / f"{args.tag}-{llm}-{arm}"), encoding="utf-8")
            wanted = LLMS[llm]["lm_models"] + ([ARMS[arm]["llm_model"]] if "llm_model" in ARMS[arm] else [])
            if wanted or args.unload_idle:
                lmstudio_models.ensure(LM_SERVER, wanted)
            sidecar = start_sidecar(SIDECARS[arm]) if arm in SIDECARS else None
            try:
                run_arm(args, env, tasks, llm, arm, config_path)
            finally:
                stop_sidecar(sidecar)


def run_arm(args, env, tasks, llm, arm, config_path) -> None:
    for task_id in args.tasks.split(","):
        task = tasks[task_id]
        for rep in range(1, args.reps + 1):
            label = f"{args.tag}-{llm}-{arm}-{task_id}-{rep}"
            record = {"label": label, "llm": llm, "arm": arm, "task": task_id, "rep": rep}
            # Tasks about the user's own accounts (for example which chat
            # servers to read) take those names from the environment, so they
            # never live in the repository.
            names = {key: os.environ.get(variable, "") for key, variable in task.get("requires_env", {}).items()}
            if not all(names.values()):
                record["skipped"] = "set " + ", ".join(task["requires_env"].values())
                print(json.dumps(record), flush=True)
                continue
            if names:
                task = dict(task)
                for key, value in names.items():
                    task["prompt"] = task["prompt"].replace("{" + key + "}", value)
                    task["answer_regex"] = task["answer_regex"].replace("{" + key + "}", re.escape(value))
            if task.get("requires_process") and not process_running(task["requires_process"]):
                record["skipped"] = f"{task['requires_process']} is not running"
                print(json.dumps(record), flush=True)
                continue
            for action in task.get("reset", []):
                powershell(RESET[action])
            verify_path = expand_path(task["verify_file"]) if task.get("verify_file") else None
            if verify_path and verify_path.exists():
                verify_path.unlink()
            closed = close_new_test_windows(args.window_baseline)
            if closed:
                print(json.dumps({"reset_closed": closed}), flush=True)
            powershell(DISMISS_SHELL)
            time.sleep(3)
            data_dir = LIVE_DIR / "data" / f"{args.tag}-{llm}-{arm}"
            if args.memory == "run":
                data_dir = LIVE_DIR / "data" / label
                config_path.write_text(arm_config(llm, arm, data_dir), encoding="utf-8")
            log_path = LIVE_DIR / "logs" / f"{label}.log"
            started = time.monotonic()
            with open(log_path, "w", encoding="utf-8", errors="replace") as log:
                process = subprocess.Popen(
                    [args.exe, "--config", str(config_path), "run", task["prompt"],
                     "--provider", LLMS[llm]["provider"], "--model", LLMS[llm]["model"], "--yes"],
                    cwd=REPO, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                )
                try:
                    record["exit"] = process.wait(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)], capture_output=True)
                    record["timeout"] = True
            record["seconds"] = round(time.monotonic() - started)
            text = log_path.read_text(encoding="utf-8", errors="replace")
            artifacts = re.search(r"Artifacts: (.+)", text)
            if artifacts:
                record["artifact_dir"] = artifacts.group(1).strip()
                record.update(trace_metrics(Path(record["artifact_dir"])))
                record.update(trace_memory(Path(record["artifact_dir"])))
            else:
                record["log_tail"] = text[-400:]
            record["success"] = bool(re.search(task["answer_regex"], record.get("answer", "")))
            if verify_path:
                record["file_check"] = check_file(verify_path, task.get("verify_contains", []))
                record["success"] = record["success"] and record["file_check"] == "ok"
            for process_name in task.get("cleanup_processes", []):
                # Only an application this task opened; never one the user had open.
                if process_name.lower() not in args.initial_processes:
                    subprocess.run(["taskkill", "/IM", f"{process_name}.exe", "/F"], capture_output=True)
            record.update(skill_counts(data_dir))
            with open(RESULTS, "a", encoding="utf-8") as results:
                results.write(json.dumps(record) + "\n")
            extras = (record.get("run_metrics") or {}).get("extras") or {}
            print(json.dumps({
                "label": label, "success": record["success"], "seconds": record["seconds"],
                "file_check": record.get("file_check"),
                "llm_requests": extras.get("primary_model_requests"),
                "fast": record.get("fast_statuses"), "timeout": record.get("timeout", False),
                "skills_loaded": record.get("skills_loaded"), "skill_outcomes": record.get("skill_outcomes"),
                "skills": record.get("skills"), "skill_tasks": record.get("skill_tasks"),
            }), flush=True)


def summary(args: argparse.Namespace) -> None:
    rows = [json.loads(line) for line in RESULTS.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.tag:
        rows = [row for row in rows if row["label"].startswith(f"{args.tag}-")]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        if not row.get("skipped"):
            groups[(row["llm"], row["task"], row["arm"])].append(row)
    print(f"{'llm':9}{'task':11}{'arm':12}{'n':>3}{'ok':>4}{'req_mean':>9}{'req_min':>8}{'req_max':>8}{'ptok_k':>8}{'sec':>6}{'judge':>6}  fast_statuses")
    for (llm, task, arm), group in sorted(groups.items()):
        extras = [(row.get("run_metrics") or {}).get("extras") or {} for row in group]
        requests = [extra.get("primary_model_requests") or 0 for extra in extras]
        tokens = [(extra.get("primary_model_prompt_tokens") or 0) / 1000 for extra in extras]
        judge = sum(extra.get("fast_actions_judge_calls", 0) for extra in extras)
        statuses = Counter(status for row in group for status in row.get("fast_statuses") or [])
        print(
            f"{llm:9}{task:11}{arm:12}{len(group):>3}{sum(row['success'] for row in group):>4}"
            f"{statistics.mean(requests):>9.1f}{min(requests):>8}{max(requests):>8}"
            f"{statistics.mean(tokens):>8.0f}{statistics.mean(row['seconds'] for row in group):>6.0f}{judge:>6}  "
            + ",".join(f"{key}:{value}" for key, value in statuses.most_common())
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--llm", default="qwen", help="comma list of: " + ",".join(LLMS))
    run_parser.add_argument("--arms", default="none,grounding,laya,laya_judge,jev", help="comma list of: " + ",".join(ARMS))
    run_parser.add_argument("--tasks", default="book,display,progfiles,discord")
    run_parser.add_argument("--reps", type=int, default=3)
    run_parser.add_argument("--timeout", type=int, default=600)
    run_parser.add_argument("--tag", default=time.strftime("live%m%d%H%M"))
    run_parser.add_argument("--memory", choices=["arm", "run"], default="arm",
                            help="share one isolated data folder per arm (learning) or use a fresh one per run")
    run_parser.add_argument("--unload-idle", action="store_true", help="unload LM Studio models for arms that need none")
    run_parser.add_argument("--exe", default=str(Path(os.environ.get("LOCALAPPDATA", "")) / "Temp" / "pokbench" / "target" / "debug" / "pok-ai.exe"))
    summary_parser = sub.add_parser("summary")
    summary_parser.add_argument("--tag", default="")
    args = parser.parse_args()
    run(args) if args.command == "run" else summary(args)


if __name__ == "__main__":
    main()
