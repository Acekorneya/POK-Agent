"""POK-Ai as a Windows Agent Arena (WAA) agent.

`scripts/arena/setup_waa.py` installs this folder into the WAA client as
`mm_agents/pokai`. WAA keeps doing what it does for every agent: it prepares
each task (`env.reset`) and scores it (`env.evaluate`). In between, this agent
does not stream pyautogui actions; it runs the whole POK-Ai harness inside the
Windows VM, where POK-Ai observes and acts through UI Automation itself, and
then hands control back with `DONE` (or `FAIL` when POK-Ai reports the task
cannot be done, which is the expected answer for WAA's infeasible tasks).

Files cross between the client container and the VM through the arena share,
mounted at `/shared/pokai` here and `\\\\host.lan\\Data\\pokai` in the VM:

- `bin/`            pok-ai.exe and arena.toml (written by run_arena.py)
- `run-task.ps1`    the in-VM wrapper that runs one task
- `memory/<scope>/` POK-Ai memory (skills and facts) kept between tasks/passes
- `out/<run id>/`   per-task log, trace, and router training records

API keys are uploaded per task with `/setup/upload` (the VM server does not
log upload contents) and deleted by the wrapper as soon as it has read them.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid

import requests

logger = logging.getLogger("desktopenv.agent.pokai")

SHARE_IN_CONTAINER = "/shared/pokai"
SHARE_IN_VM = "\\\\host.lan\\Data\\pokai"
VM_ROOT = "C:\\pokai"

# Appended to every instruction. WAA's reference agent can answer FAIL for an
# impossible task; this is the same option expressed for POK-Ai.
INFEASIBLE_NOTE = (
    "\n\n(If this task is impossible on this computer, do not attempt a "
    "workaround: start your final answer with INFEASIBLE: and say why.)"
)


class PokAiAgent:
    # WAA only needs this to configure its environment; POK-Ai acts itself.
    action_space = "code_block"

    def __init__(self, emulator_ip: str = "20.20.20.21", port: int = 5000):
        self.server = f"http://{emulator_ip}:{port}"
        self.timeout = int(os.environ.get("POKAI_TASK_TIMEOUT", "900"))
        # Empty: fresh memory for every task. Otherwise tasks share the named
        # memory ("@domain": one per application), so later tasks and passes
        # can reuse what earlier ones learned.
        self.memory_scope = os.environ.get("POKAI_MEMORY_SCOPE", "")
        self.key_names = [name for name in os.environ.get("POKAI_KEY_NAMES", "").split(",") if name]
        self.task: tuple[str, str, str] | None = None
        self.finished = False

    # Called by run.py (patched by setup_waa.py) before each example.
    def set_task(self, domain: str, example_id: str, result_dir: str) -> None:
        self.task = (domain, example_id, result_dir)

    def reset(self) -> None:
        """Called before WAA prepares each task. Close browsers and any POK-Ai
        left running by the previous task (for example one stopped at the time
        limit): WAA's local runs cannot revert a VM snapshot, and a leftover
        browser without a debugging port makes the next browser task's setup
        fail."""
        self.finished = False
        for image in ("pok-ai.exe", "chrome.exe", "msedge.exe"):
            try:
                self._post_json("/setup/execute", {"command": ["taskkill", "/IM", image, "/F", "/T"]})
            except requests.RequestException:
                pass

    def predict(self, instruction: str, obs: dict):
        if self.finished:
            return "", ["DONE"], {}, None
        self.finished = True
        outcome = self._run(instruction)
        if self.task:
            with open(os.path.join(self.task[2], "pokai.json"), "w", encoding="utf-8") as handle:
                json.dump(outcome, handle, indent=1)
        action = "FAIL" if outcome.get("infeasible") else "DONE"
        return outcome.get("answer", ""), [action], {"pokai_status": outcome.get("status", "")}, None

    def _run(self, instruction: str) -> dict:
        domain, example_id, _ = self.task or ("unknown", "unknown", "")
        run_id = f"{domain}-{example_id[:12]}-{uuid.uuid4().hex[:8]}"
        vm_run = f"{VM_ROOT}\\runs\\{run_id}"
        out_dir = os.path.join(SHARE_IN_CONTAINER, "out", run_id)
        started = time.monotonic()
        try:
            self._post_json("/setup/create_folder", {"path": vm_run})
            self._upload(f"{vm_run}\\task.txt", instruction + INFEASIBLE_NOTE)
            # "@domain" keeps one memory per application.
            scope = (
                domain + os.environ.get("POKAI_SCOPE_SUFFIX", "")
                if self.memory_scope == "@domain"
                else self.memory_scope
            )
            self._upload(f"{vm_run}\\memory_scope.txt", scope)
            keys = "\n".join(f"{name}={os.environ[name]}" for name in self.key_names if os.environ.get(name))
            self._upload(f"{vm_run}\\keys.env", keys)
            # Start detached: the server's command timeout (60 s) is far shorter
            # than a task, so the wrapper runs on its own and signals when done.
            launch = (
                "Start-Process powershell -WindowStyle Hidden -ArgumentList "
                f"'-NoProfile','-ExecutionPolicy','Bypass','-File','{SHARE_IN_VM}\\run-task.ps1','-RunId','{run_id}'"
            )
            self._post_json("/setup/execute", {"command": ["powershell", "-NoProfile", "-Command", launch]})
        except requests.RequestException as error:
            return {"run_id": run_id, "status": "launch_failed", "error": str(error)}

        exit_file = os.path.join(out_dir, "exit.txt")
        deadline = time.monotonic() + self.timeout
        timed_out = False
        while not os.path.exists(exit_file):
            if time.monotonic() > deadline and not timed_out:
                timed_out = True
                logger.warning("POK-Ai exceeded %s s; stopping it", self.timeout)
                try:
                    self._post_json("/setup/execute", {"command": ["taskkill", "/IM", "pok-ai.exe", "/F"]})
                except requests.RequestException:
                    pass
                # The wrapper still saves memory and outputs after the kill.
                deadline = time.monotonic() + 120
            elif time.monotonic() > deadline:
                return {"run_id": run_id, "status": "lost", "seconds": round(time.monotonic() - started)}
            time.sleep(5)

        outcome = {
            "run_id": run_id,
            "status": "timeout" if timed_out else "finished",
            "exit_code": _read(exit_file).strip(),
            "seconds": round(time.monotonic() - started),
        }
        outcome.update(_trace_summary(os.path.join(out_dir, "trace.jsonl")))
        answer = outcome.get("answer", "")
        outcome["infeasible"] = bool(re.match(r"\s*\**INFEASIBLE\**\s*:", answer, re.IGNORECASE))
        return outcome

    def _post_json(self, path: str, body: dict) -> None:
        response = requests.post(self.server + path, json=body, timeout=60)
        response.raise_for_status()

    def _upload(self, vm_path: str, text: str) -> None:
        response = requests.post(
            self.server + "/setup/upload",
            data={"file_path": vm_path},
            files={"file_data": ("file", text.encode("utf-8"))},
            timeout=60,
        )
        response.raise_for_status()


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return ""


def _trace_summary(path: str) -> dict:
    """The answer, run metrics, and skill activity from a POK-Ai trace."""
    summary: dict = {"skills_loaded": [], "skill_outcomes": [], "fast_statuses": []}
    for line in _read(path).splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind, payload = record.get("kind"), record.get("payload") or {}
        if kind == "run_started":
            summary["session_id"] = record.get("session_id")
        elif kind == "run_completed":
            summary["answer"] = payload.get("answer") or ""
            extras = (payload.get("metrics") or {}).get("extras") or {}
            summary["llm_requests"] = extras.get("primary_model_requests")
            summary["prompt_tokens"] = extras.get("primary_model_prompt_tokens")
        elif kind == "run_failed":
            summary["run_failed"] = payload.get("error")
        elif kind == "skill_auto_loaded":
            summary["skills_loaded"].append(payload.get("title"))
        elif kind in {"procedure_learned", "procedure_reinforced", "procedure_not_learned", "procedure_rejected"}:
            summary["skill_outcomes"].append(payload.get("outcome") or kind.removeprefix("procedure_"))
        elif kind == "skill_failed":
            summary["skill_outcomes"].append("skill_failed")
        elif kind == "fast_actions_finished":
            summary["fast_statuses"].append(payload.get("status"))
    return summary
