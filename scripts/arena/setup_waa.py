"""Set up Windows Agent Arena (WAA) for POK-Agent (run from WSL or Linux).

WAA runs a Windows 11 VM in a Docker container (QEMU/KVM). POK-Agent runs
inside that VM as a WAA agent; see scripts/arena/README.md for the design.

    python3 scripts/arena/setup_waa.py status     # what is ready, what is missing
    python3 scripts/arena/setup_waa.py install    # clone WAA, install the POK-Agent agent
    python3 scripts/arena/setup_waa.py images     # pull the WAA container image
    python3 scripts/arena/setup_waa.py prepare    # build the golden Windows image (~20 min)
    python3 scripts/arena/setup_waa.py update-libreoffice --libreoffice 26.8.0
    python3 scripts/arena/setup_waa.py update-apps   # Clock's Store update; OneDrive prompts off
                                                  # install another LibreOffice into the golden image

`prepare` needs a Windows 11 Enterprise evaluation ISO saved as
<waa>/src/win-arena-container/vm/image/setup.iso (download it yourself from
the Microsoft Evaluation Center; it is licensed to you, not bundled here).

WAA lives outside this repository (default ~/arena/WindowsAgentArena, or
POKAI_WAA_DIR). It is pinned to the commit below so results stay comparable.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

WAA_REPO = "https://github.com/microsoft/WindowsAgentArena.git"
WAA_COMMIT = "6d39ed88c545a0d40a7a02e39b928e278df7332b"
WAA_IMAGE = "windowsarena/winarena:latest"
HERE = Path(__file__).resolve().parent

PATCH_MARKER = "# POK-Ai arena agent"
AGENT_BRANCH = f"""    elif cfg_args["agent_name"] == "pokai":  {PATCH_MARKER}
        from mm_agents.pokai.agent import PokAiAgent
        agent = PokAiAgent(emulator_ip=args.emulator_ip)
        # POK-Agent observes the screen itself; skip WAA's accessibility-tree
        # capture, which takes minutes per task.
        args.observation_type = "screenshot"
"""
SET_TASK = f"""                if hasattr(agent, "set_task"):  {PATCH_MARKER}
                    agent.set_task(domain, example_id, example_result_dir)
"""

# The runner hands tasks a busy VM has not started yet to an idle one by
# rewriting the busy VM's task file; re-read it before each task.
TASK_LOOP = '        for example_id in tqdm(test_all_meta[domain], desc="Example", leave=False):\n'
TASK_RELOAD = f"""            try:  {PATCH_MARKER}: skip tasks moved to another VM
                with open(args.test_all_meta_path, "r", encoding="utf-8") as current:
                    if example_id not in json.load(current).get(domain, []):
                        continue
            except (OSError, ValueError):
                pass
"""


# Chrome 136+ ignores --remote-debugging-port on its default profile folder,
# so WAA's Chrome task setup could no longer attach. Launching through a
# directory junction to that same folder is allowed, and the data stays where
# WAA's checkers read it.
CHROME_SPLIT = "            command = command.split()\n\n"
CHROME_PAYLOAD = "        payload = json.dumps({\"command\": command, \"shell\": shell})\n"
CHROME_ANCHOR = CHROME_SPLIT + CHROME_PAYLOAD
CHROME_PATCH = f"""        if (  {PATCH_MARKER}: Chrome debugging through a profile junction
            isinstance(command, list) and command
            and "chrome" in str(command[0]).lower()
            and any(str(part).startswith("--remote-debugging-port") for part in command)
            and not any(str(part).startswith("--user-data-dir") for part in command)
        ):
            junction = "if (-not (Test-Path C:\\\\chrome-debug)) {{ cmd /c mklink /J C:\\\\chrome-debug \\"%LOCALAPPDATA%\\\\Google\\\\Chrome\\\\User Data\\" }}"
            requests.post(self.http_server + "/setup/execute",
                          json={{"command": ["powershell", "-NoProfile", "-Command", junction], "shell": False}})
            # Open the existing Default profile directly: through the junction
            # Chrome can otherwise show its profile picker or first-run page,
            # and WAA's setup then waits for a tab that never opens.
            command = list(command) + ["--user-data-dir=C:\\\\chrome-debug", "--profile-directory=Default",
                                       "--no-first-run", "--no-default-browser-check"]
"""


def waa_dir() -> Path:
    return Path(os.environ.get("POKAI_WAA_DIR", Path.home() / "arena" / "WindowsAgentArena"))


def container_dir() -> Path:
    return waa_dir() / "src" / "win-arena-container"


def golden_storage() -> Path:
    return container_dir() / "vm" / "storage"


def iso_path() -> Path:
    return container_dir() / "vm" / "image" / "setup.iso"


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, check=True, **kwargs)


def install(_: argparse.Namespace) -> None:
    root = waa_dir()
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", WAA_REPO, str(root)])
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != WAA_COMMIT:
        run(["git", "-C", str(root), "fetch", "--depth", "1", "origin", WAA_COMMIT])
        run(["git", "-C", str(root), "checkout", "--quiet", WAA_COMMIT])

    # The client directory is mounted into the container, so the agent is
    # copied (a symlink out of the mount would not resolve there).
    target = container_dir() / "client" / "mm_agents" / "pokai"
    # Overwrite in place: the container leaves a root-owned __pycache__ here.
    shutil.copytree(HERE / "waa_agent", target, ignore=shutil.ignore_patterns("__pycache__"), dirs_exist_ok=True)
    print(f"installed agent -> {target}")

    # Patches are applied to pristine files, so updating them is idempotent.
    for patched in ("src/win-arena-container/client/run.py",
                    "src/win-arena-container/client/desktop_env/controllers/setup.py",
                    "src/win-arena-container/client/desktop_env/evaluators/getters/file.py"):
        run(["git", "-C", str(root), "checkout", "--", patched])
    run_py = container_dir() / "client" / "run.py"
    text = run_py.read_text(encoding="utf-8")
    if PATCH_MARKER not in text:
        anchor = '    else:\n        raise ValueError(f"Unknown agent name'
        call = "                lib_run_single.run_single_example(agent, env, example"
        if text.count(anchor) != 1 or text.count(call) != 1 or text.count(TASK_LOOP) != 1:
            sys.exit("WAA run.py changed shape; update the patch in setup_waa.py")
        text = text.replace(anchor, AGENT_BRANCH + anchor)
        text = text.replace(call, SET_TASK + call)
        text = text.replace(TASK_LOOP, TASK_LOOP + TASK_RELOAD)
        run_py.write_text(text, encoding="utf-8")
        print(f"patched {run_py}")
    else:
        print("run.py already patched")

    setup_py = container_dir() / "client" / "desktop_env" / "controllers" / "setup.py"
    text = setup_py.read_text(encoding="utf-8")
    if PATCH_MARKER not in text:
        if text.count(CHROME_ANCHOR) != 1:
            sys.exit("WAA setup.py changed shape; update the Chrome patch in setup_waa.py")
        text = text.replace(CHROME_ANCHOR, CHROME_SPLIT + CHROME_PATCH + CHROME_PAYLOAD)
        setup_py.write_text(text, encoding="utf-8")
        print(f"patched {setup_py}")
    else:
        print("setup.py already patched")

    # WAA's checker calls the VM and GitHub without a timeout. Reopening the
    # task's file (os.startfile behind a Windows prompt) or a stalled download
    # then hung a checker for good, and the task was lost although the agent
    # had finished it. With a timeout the checker goes on to read the result.
    client = container_dir() / "client" / "desktop_env"
    for path, old, new in CHECKER_TIMEOUTS:
        target = client / path
        text = target.read_text(encoding="utf-8")
        if text.count(old) != 1:
            sys.exit(f"WAA {path} changed shape; update CHECKER_TIMEOUTS in setup_waa.py")
        target.write_text(text.replace(old, new), encoding="utf-8")
        print(f"patched {target} (timeout)")


# (file under client/desktop_env, exact call, the same call with a timeout)
CHECKER_TIMEOUTS = [
    ("controllers/setup.py",
     'requests.post(self.http_server + "/setup" + "/open_file", headers=headers, data=payload)',
     'requests.post(self.http_server + "/setup" + "/open_file", headers=headers, data=payload, timeout=120)'),
    ("controllers/setup.py",
     'requests.post(self.http_server + "/setup" + "/activate_window", headers=headers, data=payload)',
     'requests.post(self.http_server + "/setup" + "/activate_window", headers=headers, data=payload, timeout=120)'),
    ("evaluators/getters/file.py",
     "response = requests.get(url, stream=True)",
     "response = requests.get(url, stream=True, timeout=120)"),
]


def images(_: argparse.Namespace) -> None:
    # The image is public. A Docker config left by Docker Desktop can name a
    # credential helper that WSL cannot run, so pull with an empty config.
    with tempfile.TemporaryDirectory() as config:
        Path(config, "config.json").write_text("{}", encoding="utf-8")
        run(["docker", "pull", WAA_IMAGE], env=dict(os.environ, DOCKER_CONFIG=config))


# The LibreOffice build WAA's tasks were written for, from the official
# archive (WAA's pinned mirrors no longer carry it).
LIBREOFFICE_ARCHIVE = (
    "https://downloadarchive.documentfoundation.org/libreoffice/old/24.8.2.1/win/x86_64/"
    "LibreOffice_24.8.2.1_Win_x86-64.msi"
)


def patched_oem() -> Path:
    """WAA's in-VM setup files (the image's /oem), with two fixes:

    - LibreOffice downloads from the official archive first, and GIMP from
      its official host first.
    - The Visual C++ runtime is installed first (WAA's server needs it).
    - LibreOffice is installed without its online updater.
    - `Start-Process -Wait` waits for every process an installer touched; on
      current Windows 11 that can block setup forever after an installer has
      finished. Each installer is now awaited on its own process, for at most
      15 minutes.
    """
    target = waa_dir() / "pokai-oem"
    shutil.rmtree(target, ignore_errors=True)
    name = "pokai-oem-extract"
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    run(["docker", "create", "--name", name, WAA_IMAGE])
    try:
        run(["docker", "cp", f"{name}:/oem", str(target)])
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)

    tools = json.loads((target / "tools_config.json").read_text(encoding="utf-8"))
    mirrors = tools["LibreOffice"]["mirrors"]
    if LIBREOFFICE_ARCHIVE not in mirrors:
        mirrors.insert(0, LIBREOFFICE_ARCHIVE)
    # WAA's first GIMP mirror serves about 30 KB/s (hours for 320 MB); the
    # official host carries the same file and is far faster.
    tools["GIMP"]["mirrors"].sort(key=lambda url: "download.gimp.org" not in url)
    (target / "tools_config.json").write_text(json.dumps(tools, indent=4), encoding="utf-8")

    setup = target / "setup.ps1"
    lines = []
    for line in setup.read_text(encoding="utf-8").splitlines():
        if 'Start-Process "msiexec.exe"' in line and "libreOffice" in line:
            # Install LibreOffice without its online updater, which would
            # otherwise upgrade it in the middle of benchmark runs.
            line = line.replace("/quiet", "/quiet ISCHECKFORPRODUCTUPDATES=0 REMOVE=gm_o_Onlineupdate")
        if line.startswith("[Net.ServicePointManager]::SecurityProtocol"):
            # The Visual C++ 2015-2022 runtime (with MFC): the latest pywin32's
            # win32ui, which WAA's server imports through pywinauto, fails to
            # load without it on a clean Windows 11.
            lines += [
                line,
                '$vcRedist = "$env:TEMP\\vc_redist.x64.exe"',
                'Invoke-WebRequest -Uri "https://aka.ms/vs/17/release/vc_redist.x64.exe" -OutFile $vcRedist -UseBasicParsing',
                '$installer = Start-Process -FilePath $vcRedist -ArgumentList "/install /quiet /norestart" -PassThru; '
                "if ($installer) { $installer.WaitForExit(600000) | Out-Null }",
            ]
            continue
        match = re.match(r"^(\s*)Start-Process (.*) -Wait(.*)$", line)
        if match:
            indent, before, after = match.groups()
            line = (
                f"{indent}$installer = Start-Process {before}{after} -PassThru; "
                "if ($installer -and -not $installer.WaitForExit(900000)) "
                '{ Write-Host "Installer still running after 15 minutes; continuing." }'
            )
        lines.append(line)
    setup.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")
    return target


def prepare(args: argparse.Namespace) -> None:
    if not iso_path().exists():
        sys.exit(f"missing {iso_path()} (Windows 11 Enterprise evaluation ISO renamed to setup.iso)")
    golden_storage().mkdir(parents=True, exist_ok=True)
    if any(golden_storage().iterdir()):
        sys.exit(f"{golden_storage()} is not empty; move it away to rebuild the golden image")
    oem = patched_oem()
    # The same container WAA's run.sh starts for --prepare-image, with the
    # patched setup files mounted as /oem.
    command = [
        "docker", "run", "--rm", "--name", "winarena-prepare",
        "-p", "8006:8006", "--device=/dev/kvm",
        "-e", f"RAM_SIZE={args.ram}", "-e", f"CPU_CORES={args.cpus}",
        "--mount", f"type=bind,source={iso_path()},target=/custom.iso",
        "-v", f"{golden_storage()}:/storage",
        "-v", f"{container_dir() / 'vm' / 'setup'}:/shared",
        "-v", f"{oem}:/oem",
        "-v", f"{container_dir() / 'client'}:/client",
        "--cap-add", "NET_ADMIN", "--stop-timeout", "120", "--entrypoint", "/bin/bash",
        WAA_IMAGE, "-c", "./entry.sh --prepare-image true --start-client false",
    ]
    print("+", " ".join(command), flush=True)
    subprocess.run(command)
    # WAA's entry script exits non-zero after a successful shutdown; the
    # marker Windows leaves on a completed install is what counts.
    if not (golden_storage() / "windows.boot").exists():
        sys.exit("image preparation did not complete; see the log above")
    print(f"golden image ready: {golden_storage()} (back it up; rebuilding takes about an hour)")


def golden_notes() -> dict:
    """What was changed in the golden image after it was prepared (recorded
    in every run's manifest)."""
    path = golden_storage().parent / "golden.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def vm_execute(name: str, command: str, timeout: int = 900) -> str:
    """Run a PowerShell command inside the VM of container `name` through
    WAA's in-VM server; returns its output and error text."""
    body = json.dumps({"command": ["powershell", "-NoProfile", "-Command", command]})
    result = subprocess.run(
        ["docker", "exec", name, "curl", "-s", "-m", str(timeout), "-X", "POST",
         "http://20.20.20.21:5000/setup/execute", "-H", "Content-Type: application/json", "-d", body],
        capture_output=True, text=True,
    )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return result.stdout + result.stderr
    return (payload.get("output") or "") + (payload.get("error") or "")


def wait_for_shutdown(name: str, limit: int = 900) -> None:
    """Wait until Windows in container `name` has shut down. The container
    itself stays alive after that ("Keeping container alive"), so waiting for
    it to exit only ended at the time limit, with an error before the image
    was swapped."""
    started = time.monotonic()
    while time.monotonic() - started < limit:
        logs = subprocess.run(["docker", "logs", "--tail", "20", name], capture_output=True, text=True)
        if "Shutdown completed" in logs.stdout + logs.stderr:
            return
        if subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", name],
                          capture_output=True, text=True).stdout.strip() != "true":
            return
        time.sleep(5)
    sys.exit("Windows did not shut down in time; the golden image is unchanged")


def update_libreoffice(args: argparse.Namespace) -> None:
    """Install another LibreOffice version into the golden image, keeping the
    previous image. LibreOffice 24.8 exposes its accessibility tree only when
    Windows reports a screen reader, and Calc then froze while cells were
    edited; newer versions expose it on request. The update is verified
    (version, UI Automation tree, Calc responsive while typing) before the
    new image replaces the old one."""
    version = args.libreoffice
    golden = golden_storage()
    if not (golden / "windows.boot").exists():
        sys.exit("no golden image; run prepare first")
    work = golden.parent / "storage-update"
    backup = golden.parent / f"storage-before-libreoffice-{version}"
    if backup.exists():
        sys.exit(f"{backup} already exists; move it away first")
    run(["docker", "run", "--rm", "-v", f"{golden.parent}:/vm", "busybox", "rm", "-rf", "/vm/storage-update"])
    run(["cp", "-r", "--sparse=always", str(golden), str(work)])
    name = "winarena-update"
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    run([
        "docker", "run", "-d", "--name", name, "--device=/dev/kvm",
        "-e", f"RAM_SIZE={args.ram}", "-e", f"CPU_CORES={args.cpus}",
        "-v", f"{work}:/storage", "-v", f"{container_dir() / 'vm' / 'setup'}:/shared",
        "-v", f"{container_dir() / 'client'}:/client",
        "--cap-add", "NET_ADMIN", "--stop-timeout", "120", "--entrypoint", "/bin/bash",
        WAA_IMAGE, "-c", "./entry.sh --start-client false",
    ])
    try:
        for _ in range(90):
            probe = subprocess.run(["docker", "exec", name, "curl", "-s", "-m", "5",
                                    "http://20.20.20.21:5000/probe"], capture_output=True, text=True)
            if probe.returncode == 0 and probe.stdout.strip():
                break
            time.sleep(10)
        else:
            sys.exit("the VM did not come up")
        print("VM is up; installing LibreOffice", version, flush=True)
        url = (f"https://download.documentfoundation.org/libreoffice/stable/{version}/win/x86_64/"
               f"LibreOffice_{version}_Win_x86-64.msi")
        # WAA's server stops any command after 120 seconds, and the download
        # and install take longer: run them detached and poll their log.
        script = container_dir() / "vm" / "setup" / "libreoffice-install.ps1"
        script.write_text(
            "$ErrorActionPreference = 'Stop'; $ProgressPreference = 'SilentlyContinue'\r\n"
            "$log = 'C:\\Windows\\Temp\\libreoffice-install.log'\r\n"
            "try {\r\n"
            "  Get-Process soffice* -ErrorAction SilentlyContinue | Stop-Process -Force\r\n"
            "  $msi = Join-Path $env:TEMP 'libreoffice.msi'\r\n"
            f"  Invoke-WebRequest -UseBasicParsing -Uri '{url}' -OutFile $msi\r\n"
            "  'downloaded' | Out-File $log\r\n"
            "  $p = Start-Process msiexec.exe -Wait -PassThru -ArgumentList '/i', $msi, '/qn', '/norestart', "
            "'ISCHECKFORPRODUCTUPDATES=0', 'RebootYesNo=No'\r\n"
            "  Remove-Item $msi -Force\r\n"
            "  ('done ' + $p.ExitCode) | Out-File $log -Append\r\n"
            "} catch { ('failed ' + $_) | Out-File $log -Append }\r\n",
            encoding="utf-8",
        )
        print(vm_execute(name, (
            "Remove-Item C:\\Windows\\Temp\\libreoffice-install.log -ErrorAction SilentlyContinue; "
            "Start-Process powershell -WindowStyle Hidden -ArgumentList "
            "'-NoProfile','-ExecutionPolicy','Bypass','-File','\\\\host.lan\\Data\\libreoffice-install.ps1'"
        ), timeout=60))
        state = ""
        for _ in range(120):
            time.sleep(20)
            state = vm_execute(name, "Get-Content C:\\Windows\\Temp\\libreoffice-install.log -ErrorAction SilentlyContinue", timeout=60)
            if "done" in state or "failed" in state:
                break
        script.unlink(missing_ok=True)
        print("install log:", " ".join(state.split()))
        installed = vm_execute(name, "(Get-Item 'C:\\Program Files\\LibreOffice\\program\\soffice.exe').VersionInfo.ProductVersion").strip()
        print("installed:", installed)
        if not installed.startswith(version):
            sys.exit(f"LibreOffice {version} was not installed (found {installed!r}); the golden image is unchanged")
        if args.probe and Path(args.probe).exists():
            # The probe is read-only; it brings Calc to the front, then keys
            # are typed into cells while Calc is watched for a freeze.
            shutil.copy2(args.probe, container_dir() / "vm" / "setup" / "dialog_probe.exe")
            report = vm_execute(name, (
                "Start-Process 'C:\\Program Files\\LibreOffice\\program\\soffice.exe' -ArgumentList '--norestore','--calc'; "
                "Start-Sleep 25; & '\\\\host.lan\\Data\\dialog_probe.exe' 0 'LibreOffice Calc' | Select-Object -First 4; "
                "$ws = New-Object -ComObject WScript.Shell; Start-Sleep 1; "
                "$ws.SendKeys('^{HOME}'); $ws.SendKeys('Net Income{ENTER}'); $ws.SendKeys('=1+2{ENTER}'); "
                "$ws.SendKeys('{RIGHT}Total{TAB}42{ENTER}'); "
                "foreach ($s in 3,8,15) { Start-Sleep 4; Get-Process soffice.bin | ForEach-Object { 'typing +' + $s + 's responding=' + $_.Responding } }; "
                "& '\\\\host.lan\\Data\\dialog_probe.exe' 0 'LibreOffice Calc' | Select-Object -First 3; "
                "Get-Process soffice* -ErrorAction SilentlyContinue | Stop-Process -Force"
            ), timeout=600)
            print(report)
            (container_dir() / "vm" / "setup" / "dialog_probe.exe").unlink(missing_ok=True)
            if "responding=False" in report or "UI Automation elements" not in report:
                sys.exit("the new LibreOffice failed the UI Automation or responsiveness check; the golden image is unchanged")
        print("shutting Windows down", flush=True)
        vm_execute(name, "shutdown /s /t 5", timeout=60)
        wait_for_shutdown(name)
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    golden.rename(backup)
    work.rename(golden)
    # The container creates some image files readable by root only; runs
    # copy the image as a regular user.
    run(["docker", "run", "--rm", "-v", f"{golden}:/s", "busybox", "chmod", "-R", "a+rX", "/s"])
    notes = golden_notes()
    notes.update({"libreoffice": version, "libreoffice_updated": time.strftime("%Y-%m-%d")})
    (golden.parent / "golden.json").write_text(json.dumps(notes, indent=1) + "\n", encoding="utf-8")
    print(f"golden image now has LibreOffice {version}; the previous image is in {backup}")


# OneDrive and Windows' backup prompts ("Back up your folders", "Manage backup
# reminders") cover the task's window mid-run. No arena task uses OneDrive.
# Also run by the agent's reset before every task.
QUIET_ONEDRIVE = (
    "Get-Process OneDrive -ErrorAction SilentlyContinue | Stop-Process -Force; "
    "$od = 'HKLM:\\SOFTWARE\\Policies\\Microsoft\\Windows\\OneDrive'; "
    "New-Item -Path $od -Force | Out-Null; "
    "Set-ItemProperty -Path $od -Name DisableFileSyncNGSC -Value 1 -Type DWord; "
    "Set-ItemProperty -Path $od -Name KFMBlockOptIn -Value 1 -Type DWord; "
    "Remove-ItemProperty -Path 'HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Run' -Name OneDrive -ErrorAction SilentlyContinue; "
    "Set-ItemProperty -Path 'HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\Advanced' "
    "-Name ShowSyncProviderNotifications -Value 0 -Type DWord; "
    "$cdm = 'HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\ContentDeliveryManager'; "
    "New-Item -Path $cdm -Force | Out-Null; "
    "foreach ($n in 'SubscribedContent-338389Enabled','SubscribedContent-338393Enabled','SoftLandingEnabled','SystemPaneSuggestionsEnabled') "
    "{ Set-ItemProperty -Path $cdm -Name $n -Value 0 -Type DWord }"
)

# Counts the Clock window's UI Automation elements and whether it shows its
# "needs an update" screen instead of its controls.
CLOCK_CHECK = (
    "Add-Type -AssemblyName UIAutomationClient; "
    "Get-Process Time -ErrorAction SilentlyContinue | Stop-Process -Force; "
    "Start-Process 'ms-clock:'; Start-Sleep 12; "
    "$root = [Windows.Automation.AutomationElement]::RootElement; "
    "$name = New-Object Windows.Automation.PropertyCondition([Windows.Automation.AutomationElement]::NameProperty, 'Clock'); "
    "$w = $root.FindFirst([Windows.Automation.TreeScope]::Children, $name); "
    "if (-not $w) { 'clock window: missing' } else { "
    "$all = $w.FindAll([Windows.Automation.TreeScope]::Descendants, [Windows.Automation.Condition]::TrueCondition); "
    "$names = @($all | ForEach-Object { $_.Current.Name }); "
    "'clock elements: ' + $all.Count; "
    "'clock update screen: ' + [bool]($names | Where-Object { $_ -match 'needs an update|Install Clock|Downloading updates' }) }; "
    "Get-Process Time -ErrorAction SilentlyContinue | Stop-Process -Force"
)


def update_apps(args: argparse.Namespace) -> None:
    """Update the Store's Clock app in the golden image and turn off OneDrive
    and backup prompts, keeping the previous image. Clock in the image insists
    on an update before it runs, so every fresh VM showed "Clock needs an
    update" for its first Clock task; OneDrive's backup prompts covered task
    windows. The new image is checked (Clock shows its controls, not the update
    screen) before it replaces the old one."""
    golden = golden_storage()
    if not (golden / "windows.boot").exists():
        sys.exit("no golden image; run prepare first")
    work = golden.parent / "storage-update"
    backup = golden.parent / f"storage-before-apps-{time.strftime('%Y%m%d')}"
    if backup.exists():
        sys.exit(f"{backup} already exists; move it away first")
    run(["docker", "run", "--rm", "-v", f"{golden.parent}:/vm", "busybox", "rm", "-rf", "/vm/storage-update"])
    run(["cp", "-r", "--sparse=always", str(golden), str(work)])
    name = "winarena-update"
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    run([
        "docker", "run", "-d", "--name", name, "--device=/dev/kvm",
        "-e", f"RAM_SIZE={args.ram}", "-e", f"CPU_CORES={args.cpus}",
        "-v", f"{work}:/storage", "-v", f"{container_dir() / 'vm' / 'setup'}:/shared",
        "-v", f"{container_dir() / 'client'}:/client",
        "--cap-add", "NET_ADMIN", "--stop-timeout", "120", "--entrypoint", "/bin/bash",
        WAA_IMAGE, "-c", "./entry.sh --start-client false",
    ])
    try:
        for _ in range(90):
            probe = subprocess.run(["docker", "exec", name, "curl", "-s", "-m", "5",
                                    "http://20.20.20.21:5000/probe"], capture_output=True, text=True)
            if probe.returncode == 0 and probe.stdout.strip():
                break
            time.sleep(10)
        else:
            sys.exit("the VM did not come up")
        print("VM is up", flush=True)
        print(vm_execute(name, QUIET_ONEDRIVE, timeout=120) or "OneDrive and backup prompts turned off")
        print("before:", vm_execute(name, CLOCK_CHECK, timeout=120).strip())
        # The Store's own update and Clock's required update take minutes;
        # WAA's server stops a command after 120 s, so run it detached.
        script = container_dir() / "vm" / "setup" / "apps-update.ps1"
        script.write_text(
            "$log = 'C:\\Windows\\Temp\\apps-update.log'\r\n"
            "try {\r\n"
            "  $before = (Get-AppxPackage Microsoft.WindowsAlarms).Version\r\n"
            "  ('clock before ' + $before) | Out-File $log\r\n"
            "  try { Get-CimInstance -Namespace 'Root\\cimv2\\mdm\\dmmap' -ClassName 'MDM_EnterpriseModernAppManagement_AppManagement01' "
            "| Invoke-CimMethod -MethodName UpdateScanMethod | Out-Null; 'store scan requested' | Out-File $log -Append } "
            "catch { ('store scan unavailable ' + $_) | Out-File $log -Append }\r\n"
            "  Start-Process 'ms-clock:'\r\n"
            "  for ($i = 0; $i -lt 60; $i++) { Start-Sleep 20; if ((Get-AppxPackage Microsoft.WindowsAlarms).Version -ne $before) { break } }\r\n"
            "  ('clock after ' + (Get-AppxPackage Microsoft.WindowsAlarms).Version) | Out-File $log -Append\r\n"
            "  'done' | Out-File $log -Append\r\n"
            "} catch { ('failed ' + $_) | Out-File $log -Append }\r\n",
            encoding="utf-8",
        )
        vm_execute(name, (
            "Remove-Item C:\\Windows\\Temp\\apps-update.log -ErrorAction SilentlyContinue; "
            "Start-Process powershell -WindowStyle Hidden -ArgumentList "
            "'-NoProfile','-ExecutionPolicy','Bypass','-File','\\\\host.lan\\Data\\apps-update.ps1'"
        ), timeout=60)
        state = ""
        for _ in range(75):
            time.sleep(20)
            state = vm_execute(name, "Get-Content C:\\Windows\\Temp\\apps-update.log -ErrorAction SilentlyContinue", timeout=60)
            if "done" in state or "failed" in state:
                break
        script.unlink(missing_ok=True)
        print("update log:", " ".join(state.split()))
        after = vm_execute(name, CLOCK_CHECK, timeout=120).strip()
        print("after:", after)
        if "clock update screen: False" not in after:
            sys.exit("Clock still shows its update screen (or did not open); the golden image is unchanged")
        print("shutting Windows down", flush=True)
        vm_execute(name, "shutdown /s /t 5", timeout=60)
        wait_for_shutdown(name)
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    golden.rename(backup)
    work.rename(golden)
    run(["docker", "run", "--rm", "-v", f"{golden}:/s", "busybox", "chmod", "-R", "a+rX", "/s"])
    notes = golden_notes()
    clock = next((line.split()[-1] for line in state.splitlines() if line.startswith("clock after")), "updated")
    notes.update({"clock": clock, "onedrive": "sync and backup prompts disabled",
                  "apps_updated": time.strftime("%Y-%m-%d")})
    (golden.parent / "golden.json").write_text(json.dumps(notes, indent=1) + "\n", encoding="utf-8")
    print(f"golden image updated; the previous image is in {backup}")


def status(_: argparse.Namespace) -> None:
    def check(label: str, ok: bool, hint: str = "") -> None:
        print(f"[{'ok' if ok else '--'}] {label}" + ("" if ok or not hint else f"  ({hint})"))

    docker = shutil.which("docker") is not None
    check("docker", docker, "install Docker")
    check("/dev/kvm", os.path.exists("/dev/kvm"), "enable nested virtualization for WSL")
    check("WAA checkout", (container_dir() / "client" / "run.py").exists(), "setup_waa.py install")
    check("POK-Agent agent installed", (container_dir() / "client" / "mm_agents" / "pokai" / "agent.py").exists(), "setup_waa.py install")
    if docker:
        have = subprocess.run(["docker", "image", "inspect", WAA_IMAGE], capture_output=True).returncode == 0
        check("WAA image", have, "setup_waa.py images")
    check("setup.iso", iso_path().exists(), f"save the evaluation ISO as {iso_path()}")
    ready = golden_storage().exists() and any(golden_storage().iterdir())
    check("golden image", ready, "setup_waa.py prepare")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["status", "install", "images", "prepare", "update-libreoffice",
                                            "update-apps"])
    parser.add_argument("--ram", default="8G", help="VM memory for prepare")
    parser.add_argument("--cpus", default="8", help="VM cores for prepare")
    parser.add_argument("--libreoffice", default="26.8.0", help="LibreOffice version for update-libreoffice")
    parser.add_argument("--probe", help="dialog_probe.exe (pok-ai-windows example) to verify the update with")
    args = parser.parse_args()
    {"status": status, "install": install, "images": images, "prepare": prepare,
     "update-libreoffice": update_libreoffice, "update-apps": update_apps}[args.command](args)


if __name__ == "__main__":
    main()
