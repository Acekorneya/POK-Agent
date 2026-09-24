"""Make an LM Studio server hold exactly the models a test needs.

Benchmarks switch between several models on one GPU. Relying on LM Studio's
just-in-time loading leaves earlier models resident or silently evicts the
planner, so every test first unloads what it does not need and then loads
what it does, using LM Studio's REST API (`/api/v1/models`, `/load`,
`/unload`).

Usage:
    python scripts/lmstudio_models.py --server http://169.254.83.107:512 --ensure qwen3.8-27b@q2_k_xl
    python scripts/lmstudio_models.py --server http://169.254.83.107:512 --ensure qwen3.8-27b@q2_k_xl,lfm2.5-8b-a1b
    python scripts/lmstudio_models.py --server http://169.254.83.107:512 --unload-all
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request


def call(server: str, path: str, body: dict | None = None, timeout: float = 30) -> dict:
    request = urllib.request.Request(
        f"{server.rstrip('/')}{path}",
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read() or b"{}")


def loaded_instances(server: str) -> list[tuple[str, str]]:
    models = call(server, "/api/v1/models").get("models", [])
    return [
        (model["key"], instance["id"])
        for model in models
        for instance in model.get("loaded_instances") or []
    ]


def ensure(server: str, wanted: list[str]) -> None:
    for key, instance in loaded_instances(server):
        if key not in wanted:
            call(server, "/api/v1/models/unload", {"instance_id": instance})
            print(f"unloaded {instance}")
    resident = {key for key, _ in loaded_instances(server)}
    for key in wanted:
        if key in resident:
            print(f"kept {key}")
            continue
        result = call(server, "/api/v1/models/load", {"model": key}, timeout=900)
        print(f"loaded {key} in {result.get('load_time_seconds', '?')} s")
    final = sorted(key for key, _ in loaded_instances(server))
    if final != sorted(wanted):
        sys.exit(f"expected exactly {sorted(wanted)} loaded, found {final}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True, help="LM Studio server root, e.g. http://host:port")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ensure", help="comma-separated model keys that must be the only loaded models")
    group.add_argument("--unload-all", action="store_true")
    args = parser.parse_args()
    ensure(args.server, [] if args.unload_all else [key.strip() for key in args.ensure.split(",") if key.strip()])


if __name__ == "__main__":
    main()
