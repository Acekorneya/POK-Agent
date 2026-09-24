"""Loopback-only Zeiger typed-decision sidecar for POK-Agent.

Mirrors laya_sidecar.py's contract exactly (same /health and /v1/systemone
shape, same bearer-token auth) so TypeSafeDecisionRouter can talk to either
backend unmodified. This is the reference example for
docs/decision-router-backends.md: adding a new decision model to the harness
means writing one small file like this one, not touching Rust.

Requires the `zeiger` package on PYTHONPATH (clone github.com/PurHur/zeiger
and either `pip install -e .` if it gains packaging, or run this script with
that repo's root added to sys.path / PYTHONPATH) and a checkpoint directory
such as php-ai/zeiger-0.6b downloaded locally.
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
from pydantic import BaseModel
from zeiger import Engine


class DecisionRequest(BaseModel):
    model: str
    state: Any
    questions: dict[str, dict[str, Any]]


def create_app(model_dir: str, token: str, device: str, log_dir: str | None = None) -> FastAPI:
    loaded_at = time.monotonic()
    engine = Engine(model_dir, device=device, warmup=True)

    app = FastAPI(title="POK-Agent Zeiger sidecar", docs_url=None, redoc_url=None)

    def authenticate(authorization: str | None = Header(default=None)) -> None:
        expected = f"Bearer {token}"
        if not authorization or not secrets.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="invalid sidecar token")

    @app.get("/health")
    def health(_: None = Depends(authenticate)) -> dict[str, Any]:
        info = engine.info()
        display_device = str(engine.device)
        if engine.device.type == "cuda" and torch.cuda.is_available():
            display_device = f"cuda ({torch.cuda.get_device_name(0)})"
        return {
            "status": "ready",
            "model": "zeiger",
            "checkpoint": os.path.basename(os.path.normpath(model_dir)),
            "device": display_device,
            "device_reason": f"dtype={info.get('dtype')}",
            "context_tokens": int(engine.max_len),
            "startup_ms": round((time.monotonic() - loaded_at) * 1000),
        }

    @app.post("/v1/systemone")
    def system_one(request: DecisionRequest, _: None = Depends(authenticate)) -> dict[str, Any]:
        started = time.monotonic()
        answers = engine.decide(request.state, request.questions)
        display_device = str(engine.device)
        if engine.device.type == "cuda" and torch.cuda.is_available():
            display_device = f"cuda ({torch.cuda.get_device_name(0)})"
        response = {
            "model": request.model,
            "answers": answers,
            "sidecar": {
                "backend": "zeiger",
                "device": display_device,
                "context_tokens": int(engine.max_len),
                "elapsed_ms": round((time.monotonic() - started) * 1000),
            },
        }
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            (Path(log_dir) / f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}.json").write_text(
                json.dumps({"request": request.model_dump(), "response": response}, indent=2)
            )
        return response

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--log-dir", default=None, help="directory for raw request/response JSON dumps")
    args = parser.parse_args()
    # POK_JUDGE_TOKEN takes priority so this sidecar can run alongside a
    # separate Laya process (the dual-cascade case: each needs its own
    # bearer token). POK_LAYA_TOKEN remains a fallback for the single-backend
    # swap-in case documented in this file's module docstring.
    token = os.environ.get("POK_JUDGE_TOKEN") or os.environ.get("POK_LAYA_TOKEN", "")
    if not token:
        raise SystemExit("POK_JUDGE_TOKEN (or POK_LAYA_TOKEN) is required")
    uvicorn.run(
        create_app(args.model_dir, token, args.device, args.log_dir),
        host="127.0.0.1",
        port=args.port,
        access_log=False,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
