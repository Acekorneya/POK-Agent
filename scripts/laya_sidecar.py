"""Loopback-only Laya typed-decision sidecar for POK-Agent.

The process is intentionally small: Rust still owns candidate construction,
redaction, confidence gates, policy, and action execution.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import time
import uuid
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from laya import Agent
from pydantic import BaseModel


class DecisionRequest(BaseModel):
    model: str
    state: Any
    questions: dict[str, dict[str, Any]]


def choose_device(preference: str, min_free_vram_mb: int) -> tuple[str, str]:
    if preference == "cpu":
        return "cpu", "CPU was selected in POK-Agent"
    if not torch.cuda.is_available():
        return "cpu", f"CUDA is unavailable in this PyTorch runtime (requested: {preference})"
    try:
        free_bytes, _ = torch.cuda.mem_get_info()
        free_mb = free_bytes // (1024 * 1024)
        if preference == "auto" and free_mb < min_free_vram_mb:
            return "cpu", f"CUDA has {free_mb} MiB free; {min_free_vram_mb} MiB is required"
        return "cuda", f"CUDA selected with {free_mb} MiB free ({preference})"
    except Exception as error:  # device probing must never prevent CPU fallback
        return "cpu", f"CUDA probe failed: {error}"


def create_app(
    model_dir: str, token: str, device: str, min_free_vram_mb: int, log_dir: str | None = None
) -> FastAPI:
    requested_device, device_reason = choose_device(device, min_free_vram_mb)
    loaded_at = time.monotonic()
    agent = Agent(model_dir, device=requested_device)
    # Expand token budget to support high-cardinality candidate lists without option token starvation
    agent.cfg["head_max_len"] = max(int(agent.cfg.get("head_max_len", 256)), 512)
    agent.cfg["max_len"] = max(int(agent.cfg.get("max_len", 512)), 1024)
    loaded_device = str(agent.device)
    if loaded_device != requested_device:
        device_reason = f"Laya fell back from {requested_device} to {loaded_device}"

    # Warm up CUDA kernels to avoid first-request latency spikes
    if "cuda" in loaded_device and torch.cuda.is_available():
        try:
            with torch.inference_mode():
                agent.predict(
                    {"task": "warmup", "current_step": "warmup"},
                    {"warmup": {"type": "choice", "instructions": "warmup", "criteria": {"a": "warmup"}}},
                )
        except Exception:
            pass

    def device_status() -> tuple[str, str]:
        actual_device = str(agent.device)
        if actual_device != loaded_device:
            return actual_device, f"Laya fell back from {loaded_device} to {actual_device} during inference"
        return actual_device, device_reason

    app = FastAPI(title="POK-Agent Laya sidecar", docs_url=None, redoc_url=None)

    def authenticate(authorization: str | None = Header(default=None)) -> None:
        expected = f"Bearer {token}"
        if not authorization or not secrets.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="invalid sidecar token")

    @app.get("/health")
    def health(_: None = Depends(authenticate)) -> dict[str, Any]:
        actual_device, actual_reason = device_status()
        display_device = actual_device
        if actual_device == "cuda" and torch.cuda.is_available():
            display_device = f"cuda ({torch.cuda.get_device_name(0)})"
        checkpoint_name = agent.cfg.get("model_name", "laya-typed-decisions")
        return {
            "status": "ready",
            "model": "laya",
            "checkpoint": checkpoint_name,
            "device": display_device,
            "device_reason": actual_reason,
            "context_tokens": int(agent.cfg.get("max_len", 1024)),
            "startup_ms": round((time.monotonic() - loaded_at) * 1000),
        }

    @app.post("/v1/systemone")
    def system_one(request: DecisionRequest, _: None = Depends(authenticate)) -> dict[str, Any]:
        started = time.monotonic()
        num_candidates = len(request.state.get("candidates", [])) if isinstance(request.state, dict) else 0
        agent.cfg["head_max_len"] = max(512, num_candidates * 24)
        agent.cfg["max_len"] = max(1024, num_candidates * 48)
        with torch.inference_mode():
            result = agent.predict(request.state, request.questions)
        actual_device, _ = device_status()
        result["model"] = request.model
        result["sidecar"] = {
            "backend": "laya",
            "device": actual_device,
            "context_tokens": int(agent.cfg.get("max_len", 512)),
            "elapsed_ms": round((time.monotonic() - started) * 1000),
        }
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            (Path(log_dir) / f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}.json").write_text(
                json.dumps({"request": request.model_dump(), "response": result}, indent=2)
            )
        return result

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--min-free-vram-mb", type=int, default=2048)
    parser.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    parser.add_argument("--log-dir", default=None, help="directory for raw request/response JSON dumps")
    args = parser.parse_args()
    token = os.environ.get("POK_LAYA_TOKEN", "")
    if not token:
        raise SystemExit("POK_LAYA_TOKEN is required")
    uvicorn.run(
        create_app(args.model_dir, token, args.device, args.min_free_vram_mb, args.log_dir),
        host="127.0.0.1",
        port=args.port,
        access_log=False,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
