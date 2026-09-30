"""Record demo videos of POK-Ai working in the arena VMs.

While `run_arena.py run --tag <tag>` is running, this polls the screenshot
endpoint of every VM whose container belongs to that run and keeps the frames,
then turns each task's frames into an MP4 and a small GIF for the README.

    python3 scripts/arena/record_demo.py record --tag demo            # until the run ends
    python3 scripts/arena/record_demo.py render --tag demo            # one clip per task

Only frames from while POK-Ai was running a task are kept in the clips: a
task's window runs from its run folder appearing on the arena share to its
exit.txt. The VMs hold no personal data.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

RUNS = Path(os.environ.get("POKAI_ARENA_RUNS", Path.home() / "arena" / "runs"))
VM_SCREENSHOT = "http://20.20.20.21:5000/screenshot"


def containers(tag: str) -> list[str]:
    out = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True).stdout
    prefix = f"pokai-{tag}-".replace("_", "-")
    return [name for name in out.split() if name.startswith(prefix)]


def frames_dir(tag: str) -> Path:
    return RUNS / tag / "demo-frames"


def record(args: argparse.Namespace) -> None:
    """Grab frames from every running VM of the run until none is left."""
    root = frames_dir(args.tag)
    root.mkdir(parents=True, exist_ok=True)
    seen_any = False
    idle_since = time.time()
    while True:
        names = containers(args.tag)
        if names:
            seen_any, idle_since = True, time.time()
        elif seen_any and time.time() - idle_since > args.idle_stop:
            break
        started = time.time()
        for name in names:
            png = subprocess.run(["docker", "exec", name, "curl", "-s", "-m", "5", VM_SCREENSHOT],
                                 capture_output=True).stdout
            if png[:4] == b"\x89PNG":
                folder = root / name
                folder.mkdir(exist_ok=True)
                (folder / f"{time.time():.2f}.png").write_bytes(png)
        time.sleep(max(0.0, args.interval - (time.time() - started)))
    print("recording stopped; frames in", root)


def task_windows(tag: str) -> list[dict]:
    """Each task's start and end on the host clock, from its run folder on the share."""
    windows = []
    for out in (RUNS / tag).glob("*/share/out/*"):
        exit_file = out / "exit.txt"
        trace = out / "trace.jsonl"
        if not exit_file.exists() or not trace.exists():
            continue
        stamps = [json.loads(line)["timestamp"] for line in trace.open(encoding="utf-8", errors="ignore")
                  if line.startswith("{")]
        if not stamps:
            continue
        # The trace uses the VM's clock, which need not match the host's;
        # exit.txt is written on the host share right after the trace ends.
        first, last = vm_time(stamps[0]), vm_time(stamps[-1])
        end = exit_file.stat().st_mtime
        windows.append({"run_id": out.name, "start": end - (last - first), "end": end})
    return windows


def vm_time(stamp: str) -> float:
    return dt.datetime.fromisoformat(stamp[:26].rstrip("Z") + "+00:00").timestamp()


def ran_task(tag: str, container: str, row: dict) -> bool:
    """Whether the worker behind `container` (…-p<pass>-w<n>) ran this task,
    from the container logs the runner saves per pass."""
    worker = container.rsplit("-w", 1)[-1]
    for log in (RUNS / tag).glob(f'*/pass{row["pass"]}/container-w{worker}*.log'):
        if row["task"] in log.read_text(encoding="utf-8", errors="ignore"):
            return True
    return False


def render(args: argparse.Namespace) -> None:
    root = frames_dir(args.tag)
    results = {}
    for line in (RUNS / args.tag / "results.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        results[row["run_id"]] = row
    out_dir = RUNS / args.tag / "demo-clips"
    out_dir.mkdir(exist_ok=True)
    frames = sorted((float(p.stem), p) for p in root.glob("*/*.png"))
    for window in task_windows(args.tag):
        row = results.get(window["run_id"])
        if row is None:
            continue
        clip = [p for t, p in frames if window["start"] - 2 <= t <= window["end"] + 2]
        if len(clip) < 5:
            continue
        # One container per worker; keep the one that has this task's frames.
        by_container: dict[str, list[Path]] = {}
        for p in clip:
            by_container.setdefault(p.parent.name, []).append(p)
        name = f'{row["domain"]}-{row["task"][:8]}-p{row["pass"]}'
        container = next((c for c in by_container if ran_task(args.tag, c, row)), None)
        if container is None:
            print(f"{name}: no worker log names this task; skipped")
            continue
        chosen = by_container[container]
        staging = out_dir / f"{name}-frames"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir()
        for index, p in enumerate(sorted(chosen)):
            os.link(p, staging / f"{index:05d}.png")
        seconds = window["end"] - window["start"]
        # Frames where nothing changed (the planner thinking) are dropped, so
        # the clip shows only the screen changing, played at a steady rate.
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "2",
                        "-i", str(staging / "%05d.png"),
                        "-vf", f"mpdecimate=hi=768:lo=320:frac=0.1,setpts=N/({args.fps}*TB),"
                               "scale=1280:-2,format=yuv420p",
                        "-r", str(args.fps), "-c:v", "libx264", "-crf", "26", str(out_dir / f"{name}.mp4")],
                       check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out_dir / f"{name}.mp4"),
                        "-vf", f"fps={args.fps},scale=720:-1:flags=lanczos,split[a][b];"
                               "[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer",
                        str(out_dir / f"{name}.gif")], check=True)
        length = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                 str(out_dir / f"{name}.mp4")], capture_output=True, text=True).stdout.strip()
        print(f'{name}: score {row["score"]}, {row["llm_requests"]} planner calls, task {seconds:.0f}s, '
              f'clip {float(length or 0):.0f}s -> {name}.mp4/.gif')
        shutil.rmtree(staging)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("--tag", required=True)
    rec.add_argument("--interval", type=float, default=0.5, help="seconds between frames per VM")
    rec.add_argument("--idle-stop", type=float, default=300, help="stop after this long with no VM running")
    ren = sub.add_parser("render")
    ren.add_argument("--tag", required=True)
    ren.add_argument("--fps", type=int, default=2, help="clip frame rate after unchanged frames are dropped")
    args = parser.parse_args()
    record(args) if args.command == "record" else render(args)


if __name__ == "__main__":
    main()
