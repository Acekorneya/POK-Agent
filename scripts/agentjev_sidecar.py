"""Loopback adapter from POK-Ai's /v1/systemone contract to an AgentJev server.

AgentJev-0.6B (https://github.com/malevrigns/agent-jev, Apache-2.0; weights at
https://huggingface.co/aimeigaoshou/agent-jev) is a typed decision model with
its own loopback service (`python -m jev_service.server`, POST /api/evaluate).
This adapter only translates request and response shapes:

  choice question  -> AgentJev `choice` with options {option_id: description}
  noul question    -> AgentJev `boolean`; the probability of true is the noul

AgentJev rejects duplicate option descriptions, so a repeated description is
suffixed with its option id before sending. The model name from the request
is echoed back, as the contract requires.

Usage:
    POK_AGENTJEV_TOKEN=dev-agentjev-token python scripts/agentjev_sidecar.py \
        --upstream http://127.0.0.1:8149 --port 43213
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import time
import urllib.error
import urllib.request
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel


class DecisionRequest(BaseModel):
    model: str
    state: Any
    questions: dict[str, dict[str, Any]]


def distinct(criteria: dict[str, str]) -> dict[str, str]:
    seen: set[str] = set()
    options = {}
    for key, text in criteria.items():
        text = str(text) or key
        if text in seen:
            text = f"{text} [{key}]"
        seen.add(text)
        options[key] = text
    return options


def create_app(upstream: str, token: str, timeout: float) -> FastAPI:
    app = FastAPI(title="POK-Ai AgentJev adapter", docs_url=None, redoc_url=None)

    def authenticate(authorization: str | None = Header(default=None)) -> None:
        expected = f"Bearer {token}"
        if not authorization or not secrets.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="invalid sidecar token")

    def call(path: str, body: dict | None = None) -> dict:
        request = urllib.request.Request(
            f"{upstream.rstrip('/')}{path}",
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())

    @app.get("/health")
    def health(_: None = Depends(authenticate)) -> dict[str, Any]:
        try:
            info = call("/health")
        except OSError as error:
            raise HTTPException(status_code=503, detail=f"AgentJev upstream unavailable: {error}") from error
        return {"status": "ready", "model": "agentjev", "checkpoint": info.get("checkpoint"), "device": upstream}

    @app.post("/v1/systemone")
    def system_one(request: DecisionRequest, _: None = Depends(authenticate)) -> dict[str, Any]:
        started = time.monotonic()
        questions = []
        answers: dict[str, Any] = {}
        for key, question in request.questions.items():
            text = str(question.get("instructions") or key)
            if question.get("type") == "noul":
                questions.append({"id": key, "type": "boolean", "question": text})
                continue
            criteria = question.get("criteria") or {}
            if len(criteria) < 2:
                # AgentJev requires 2..255 options; a single option (for
                # example a forced operation) needs no model call.
                only = next(iter(criteria), None)
                answers[key] = {"type": "choice", "choice": only, "confidence": 1.0 if only else 0.0,
                                "probabilities": {only: 1.0} if only else {}}
                continue
            questions.append({"id": key, "type": "choice", "question": text, "options": distinct(criteria)})
        result = {"results": [{"answers": []}], "usage": {}}
        if questions:
            try:
                result = call("/api/evaluate", {"state": request.state, "questions": questions})
            except urllib.error.HTTPError as error:
                detail = error.read().decode(errors="replace")[:300]
                raise HTTPException(status_code=502, detail=f"AgentJev rejected the request: {detail}") from error
            except OSError as error:
                raise HTTPException(status_code=502, detail=f"AgentJev upstream failed: {error}") from error
        for item in result["results"][0]["answers"]:
            if item["type"] == "boolean":
                answers[item["id"]] = {"type": "noul", "noul": float(item["probability"])}
            else:
                probabilities = {key: float(p) for key, p in item["distribution"].items()}
                answers[item["id"]] = {
                    "type": "choice",
                    "choice": item["value"],
                    "confidence": probabilities[item["value"]],
                    "probabilities": probabilities,
                }
        return {
            "model": request.model,
            "answers": answers,
            "sidecar": {
                "backend": "agentjev",
                "upstream_wall_ms": result.get("usage", {}).get("wall_ms"),
                "elapsed_ms": round((time.monotonic() - started) * 1000),
            },
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", default="http://127.0.0.1:8149")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    token = os.environ.get("POK_AGENTJEV_TOKEN", "")
    if not token:
        raise SystemExit("POK_AGENTJEV_TOKEN is required")
    uvicorn.run(create_app(args.upstream, token, args.timeout), host="127.0.0.1", port=args.port,
                access_log=False, log_level="warning")


if __name__ == "__main__":
    main()
