"""Set up Windows Agent Arena (WAA) for POK-Ai (run from WSL or Linux).

WAA runs a Windows 11 VM in a Docker container (QEMU/KVM). POK-Ai runs
inside that VM as a WAA agent; see scripts/arena/README.md for the design.

    python3 scripts/arena/setup_waa.py status     # what is ready, what is missing
    python3 scripts/arena/setup_waa.py install    # clone WAA, install the POK-Ai agent
    python3 scripts/arena/setup_waa.py images     # pull the WAA container image
    python3 scripts/arena/setup_waa.py prepare    # build the golden Windows image (~20 min)

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
from pathlib import Path

WAA_REPO = "https://github.com/microsoft/WindowsAgentArena.git"
WAA_COMMIT = "6d39ed88c545a0d40a7a02e39b928e278df7332b"
WAA_IMAGE = "windowsarena/winarena:latest"
HERE = Path(__file__).resolve().parent

PATCH_MARKER = "# POK-Ai arena agent"
AGENT_BRANCH = f"""    elif cfg_args["agent_name"] == "pokai":  {PATCH_MARKER}
        from mm_agents.pokai.agent import PokAiAgent
        agent = PokAiAgent(emulator_ip=args.emulator_ip)
"""
SET_TASK = f"""                if hasattr(agent, "set_task"):  {PATCH_MARKER}
                    agent.set_task(domain, example_id, example_result_dir)
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

    run_py = container_dir() / "client" / "run.py"
    text = run_py.read_text(encoding="utf-8")
    if PATCH_MARKER not in text:
        anchor = '    else:\n        raise ValueError(f"Unknown agent name'
        call = "                lib_run_single.run_single_example(agent, env, example"
        if text.count(anchor) != 1 or text.count(call) != 1:
            sys.exit("WAA run.py changed shape; update the patch in setup_waa.py")
        text = text.replace(anchor, AGENT_BRANCH + anchor)
        text = text.replace(call, SET_TASK + call)
        run_py.write_text(text, encoding="utf-8")
        print(f"patched {run_py}")
    else:
        print("run.py already patched")


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


def status(_: argparse.Namespace) -> None:
    def check(label: str, ok: bool, hint: str = "") -> None:
        print(f"[{'ok' if ok else '--'}] {label}" + ("" if ok or not hint else f"  ({hint})"))

    docker = shutil.which("docker") is not None
    check("docker", docker, "install Docker")
    check("/dev/kvm", os.path.exists("/dev/kvm"), "enable nested virtualization for WSL")
    check("WAA checkout", (container_dir() / "client" / "run.py").exists(), "setup_waa.py install")
    check("POK-Ai agent installed", (container_dir() / "client" / "mm_agents" / "pokai" / "agent.py").exists(), "setup_waa.py install")
    if docker:
        have = subprocess.run(["docker", "image", "inspect", WAA_IMAGE], capture_output=True).returncode == 0
        check("WAA image", have, "setup_waa.py images")
    check("setup.iso", iso_path().exists(), f"save the evaluation ISO as {iso_path()}")
    ready = golden_storage().exists() and any(golden_storage().iterdir())
    check("golden image", ready, "setup_waa.py prepare")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["status", "install", "images", "prepare"])
    parser.add_argument("--ram", default="8G", help="VM memory for prepare")
    parser.add_argument("--cpus", default="8", help="VM cores for prepare")
    args = parser.parse_args()
    {"status": status, "install": install, "images": images, "prepare": prepare}[args.command](args)


if __name__ == "__main__":
    main()
