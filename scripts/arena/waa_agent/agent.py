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
        # WAA's local runs do not revert the VM between tasks, so close the
        # applications a previous task left open (its documents would
        # otherwise sit beside the next task's).
        for image in ("pok-ai.exe", "chrome.exe", "msedge.exe", "soffice.bin", "soffice.exe", "vlc.exe"):
            try:
                self._post_json("/setup/execute", {"command": ["taskkill", "/IM", image, "/F", "/T"]})
            except requests.RequestException:
                pass
        # A LibreOffice process that survives a forced kill (seen after the
        # planner scripted LibreOffice) keeps the profile: every later start
        # then only flashes its splash and hands the document to the stuck,
        # invisible instance. This VM can no longer run LibreOffice tasks, so
        # end this client: the runner restarts the worker on a fresh VM and the
        # task, which has no outcome yet, runs there.
        try:
            survivors = self._powershell_output(
                "Start-Sleep 2; @(Get-Process soffice,soffice.bin -ErrorAction SilentlyContinue).Count"
            ).strip()
        except (requests.RequestException, ValueError):
            survivors = "0"
        if survivors.isdigit() and int(survivors) > 0:
            print(f"POK-Ai: {survivors} LibreOffice process(es) survived a forced kill; "
                  "ending this client so the task runs on a fresh VM", flush=True)
            os._exit(3)
        # Keep LibreOffice at its installed version: its online updater
        # otherwise upgrades it in the middle of a pass, which leaves the
        # installation crashing and on a different version than the tasks'.
        freeze_libreoffice = (
            "$hosts = \"$env:SystemRoot\\System32\\drivers\\etc\\hosts\"; "
            "foreach ($name in 'update.libreoffice.org','update-mar.libreoffice.org','update-pool.libreoffice.org') "
            "{ if (-not (Select-String -Path $hosts -Pattern $name -SimpleMatch -Quiet)) "
            "{ Add-Content -Path $hosts -Value \"`r`n127.0.0.1 $name\" } }; "
            "Stop-Service LibreOfficeMaintenance -Force -ErrorAction SilentlyContinue; "
            "Set-Service LibreOfficeMaintenance -StartupType Disabled -ErrorAction SilentlyContinue; "
            "Get-Process updater, update_service -ErrorAction SilentlyContinue | Stop-Process -Force; "
            "Remove-Item \"$env:APPDATA\\LibreOffice\\4\\updates\" -Recurse -Force -ErrorAction SilentlyContinue; "
            # Blocking the update servers is not enough: once a day LibreOffice
            # still starts its updater at launch, which holds or restarts it
            # without the task's document. Without the updater it cannot.
            "$lo = \"$env:ProgramFiles\\LibreOffice\"; "
            "foreach ($f in 'program\\updater.exe','program\\update_service.exe','update-settings.ini') "
            "{ $p = Join-Path $lo $f; if (Test-Path $p) { Rename-Item $p ((Split-Path $p -Leaf) + '.disabled') -Force -ErrorAction SilentlyContinue } }; "
            # LibreOffice was force-closed above: drop its crash-recovery list
            # so the next start does not offer to restore those documents, and
            # let Ctrl+S keep a document's own format without asking. WAA's
            # checkers press Ctrl+S themselves and read the file half a second
            # later, which the "Keep current format?" dialog would block.
            "$xcu = \"$env:APPDATA\\LibreOffice\\4\\user\\registrymodifications.xcu\"; "
            "if (Test-Path $xcu) { "
            "$lines = Get-Content $xcu -Encoding UTF8 | Where-Object { $_ -notmatch 'Office.Recovery/RecoveryList' -and $_ -notmatch 'WarnAlienFormat' -and $_ -notmatch 'ShowTipOfTheDay' }; "
            "$lines = $lines | ForEach-Object { if ($_ -eq '</oor:items>') { "
            "'<item oor:path=\"/org.openoffice.Office.Common/Save/Document\"><prop oor:name=\"WarnAlienFormat\" oor:op=\"fuse\"><value>false</value></prop></item>'; "
            # No "Tip of the Day" dialog in front of the task's document.
            "'<item oor:path=\"/org.openoffice.Office.Common/Misc\"><prop oor:name=\"ShowTipOfTheDay\" oor:op=\"fuse\"><value>false</value></prop></item>' }; $_ }; "
            "[IO.File]::WriteAllLines($xcu, [string[]]$lines) }; "
            "Remove-Item \"$env:APPDATA\\LibreOffice\\4\\user\\backup\\*\" -Force -ErrorAction SilentlyContinue; "
            "Get-ChildItem \"$env:USERPROFILE\" -Recurse -Depth 2 -Force -Filter '.~lock.*#' -ErrorAction SilentlyContinue | Remove-Item -Force; "
            # File Explorer windows from a previous task (the shell stays).
            "(New-Object -ComObject Shell.Application).Windows() | ForEach-Object { $_.Quit() }"
        )
        try:
            self._post_json("/setup/execute", {"command": ["powershell", "-NoProfile", "-Command", freeze_libreoffice]})
        except requests.RequestException:
            pass
        # Store apps in this image (Clock) insist on an update before they run.
        # Blocking automatic Store updates left Clock at "Ready to install" for
        # good, so make sure no such policy is set and let the update finish.
        # (The lasting fix is updating the Store apps in the golden image.)
        allow_store_updates = (
            "Remove-ItemProperty -Path 'HKLM:\\SOFTWARE\\Policies\\Microsoft\\WindowsStore' "
            "-Name AutoDownload -ErrorAction SilentlyContinue"
        )
        try:
            self._post_json("/setup/execute", {"command": ["powershell", "-NoProfile", "-Command", allow_store_updates]})
        except requests.RequestException:
            pass
        # OneDrive and Windows' backup prompts ("Back up your folders", "Manage
        # backup reminders") covered task windows; no arena task uses OneDrive.
        # The golden image has this too (setup_waa.py update-apps); repeated
        # here in case anything turns it back on.
        quiet_onedrive = "Get-Process OneDrive -ErrorAction SilentlyContinue | Stop-Process -Force; $od = 'HKLM:\\SOFTWARE\\Policies\\Microsoft\\Windows\\OneDrive'; New-Item -Path $od -Force | Out-Null; Set-ItemProperty -Path $od -Name DisableFileSyncNGSC -Value 1 -Type DWord; Set-ItemProperty -Path $od -Name KFMBlockOptIn -Value 1 -Type DWord; Remove-ItemProperty -Path 'HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Run' -Name OneDrive -ErrorAction SilentlyContinue; Set-ItemProperty -Path 'HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\Advanced' -Name ShowSyncProviderNotifications -Value 0 -Type DWord; $cdm = 'HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\ContentDeliveryManager'; New-Item -Path $cdm -Force | Out-Null; foreach ($n in 'SubscribedContent-338389Enabled','SubscribedContent-338393Enabled','SoftLandingEnabled','SystemPaneSuggestionsEnabled') { Set-ItemProperty -Path $cdm -Name $n -Value 0 -Type DWord }"
        try:
            self._post_json("/setup/execute", {"command": ["powershell", "-NoProfile", "-Command", quiet_onedrive]})
        except requests.RequestException:
            pass
        # Keep Windows' screen-reader flag off: applications such as VS Code
        # change behavior under a screen reader, and LibreOffice 26.8 exposes
        # its UI Automation tree without it (24.8 needed the flag, and its
        # Calc then froze while cells were edited; see setup_waa.py
        # update-libreoffice).
        screen_reader = (
            "Add-Type -Namespace PokAi -Name Spi -MemberDefinition "
            "'[DllImport(\"user32.dll\")] public static extern bool SystemParametersInfo(uint a, uint b, System.IntPtr c, uint d);'; "
            "[void][PokAi.Spi]::SystemParametersInfo(0x47, 0, [System.IntPtr]::Zero, 3)"
        )
        try:
            self._post_json("/setup/execute", {"command": ["powershell", "-NoProfile", "-Command", screen_reader]})
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

    def _window_titles(self) -> list[str]:
        response = requests.post(
            self.server + "/setup/execute",
            json={"command": ["powershell", "-NoProfile", "-Command",
                              "Get-Process | Where-Object MainWindowTitle | ForEach-Object MainWindowTitle"],
                  "shell": False},
            timeout=60,
        )
        return sorted(line.strip() for line in (response.json().get("output") or "").splitlines() if line.strip())

    def _wait_for_settled_desktop(self, domain: str, example_id: str) -> dict:
        """Wait until the task's setup has finished on screen: every file WAA
        opened shows in a window title, and the window list has stopped
        changing. WAA only launches those applications; without this wait the
        agent could start while a document is still loading (a person would
        start from the loaded document)."""
        opened: list[str] = []
        config_path = os.path.join("evaluation_examples_windows", "examples", domain, f"{example_id}.json")
        try:
            with open(config_path, encoding="utf-8") as handle:
                for step in json.load(handle).get("config", []):
                    if step.get("type") == "open":
                        opened.append(re.split(r"[\\/]", step["parameters"]["path"])[-1])
        except (OSError, ValueError, KeyError):
            pass
        started = time.monotonic()
        previous, stable = None, 0
        while time.monotonic() - started < 120:
            try:
                titles = self._window_titles()
            except (requests.RequestException, ValueError):
                titles = []
            ready = all(any(name.lower() in title.lower() for title in titles) for name in opened)
            stable = stable + 1 if titles == previous else 0
            previous = titles
            if ready and stable >= 2:
                return {"settle_seconds": round(time.monotonic() - started), "settled": True}
            time.sleep(3)
        return {"settle_seconds": round(time.monotonic() - started), "settled": False, "waiting_for": opened}

    def _run(self, instruction: str) -> dict:
        domain, example_id, _ = self.task or ("unknown", "unknown", "")
        settle = self._wait_for_settled_desktop(domain, example_id)
        run_id = f"{domain}-{example_id[:12]}-{uuid.uuid4().hex[:8]}"
        vm_run = f"{VM_ROOT}\\runs\\{run_id}"
        out_dir = os.path.join(SHARE_IN_CONTAINER, "out", run_id)
        started = time.monotonic()
        try:
            self._post_json("/setup/create_folder", {"path": vm_run})
            # The instruction is passed unmodified; the infeasible-task
            # convention is a standing instruction in the arena config.
            self._upload(f"{vm_run}\\task.txt", instruction)
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
            **settle,
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

    def _powershell_output(self, command: str) -> str:
        response = requests.post(
            self.server + "/setup/execute",
            json={"command": ["powershell", "-NoProfile", "-Command", command]},
            timeout=60,
        )
        response.raise_for_status()
        return str(response.json().get("output") or "")

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
