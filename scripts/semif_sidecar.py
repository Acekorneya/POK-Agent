"""Loopback SemIf decision sidecar speaking POK-Agent's /v1/systemone contract.

SemIf (https://github.com/TheoLeeCJ/SemIf-OpenJev, MIT) turns a frozen
general LLM into a typed decision model with one forward pass: it renders the
state, question and lettered options through the model's chat template, reads
the next-token logits of the option letters only, and softmaxes over them.
Nothing is generated, so there is no free text to parse and no thinking
tokens; the probabilities are the model's own.

This sidecar imports SemIf's `load_causal_model` and `direct.score` unchanged
(put SemIf's `src` on PYTHONPATH). SemIf allows at most 16 options, so larger
option sets run as a tournament: each group of up to 16 is scored, then the
group winners are scored against each other. Probabilities outside the final
round are reported as 0.

Usage:
    POK_SEMIF_TOKEN=dev-semif-token PYTHONPATH=<SemIf>/src python scripts/semif_sidecar.py \
        --model-dir <Qwen3.5-4B dir> --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --port 43214
"""

from __future__ import annotations

import argparse
import os
import secrets
import time
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel
from semif_phase1.core import LETTERS, load_causal_model
from semif_phase1.direct import score

MAX_OPTIONS = len(LETTERS)


class DecisionRequest(BaseModel):
    model: str
    state: Any
    questions: dict[str, dict[str, Any]]


def create_app(model_dir: str, revision: str, token: str, device: str) -> FastAPI:
    loaded_at = time.monotonic()
    model, tokenizer, metadata = load_causal_model(model_dir, revision, device)
    app = FastAPI(title="POK-Agent SemIf sidecar", docs_url=None, redoc_url=None)

    def authenticate(authorization: str | None = Header(default=None)) -> None:
        expected = f"Bearer {token}"
        if not authorization or not secrets.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="invalid sidecar token")

    def probabilities(state: Any, question: str, options: list[tuple[str, str]]) -> dict[str, float]:
        row = {
            "id": "q",
            "state": state,
            "question": question,
            "options": [{"id": key, "description": text} for key, text in options],
        }
        result = score(model, tokenizer, row, metadata)
        return dict(zip(result["option_ids"], result["probabilities"]))

    def choice(state: Any, question: str, options: list[tuple[str, str]]) -> dict[str, float]:
        if len(options) <= MAX_OPTIONS:
            return probabilities(state, question, options)
        winners = []
        for start in range(0, len(options), MAX_OPTIONS):
            group = options[start : start + MAX_OPTIONS]
            if len(group) == 1:
                winners.extend(group)
                continue
            scores = probabilities(state, question, group)
            winners.append(next(option for option in group if option[0] == max(scores, key=scores.get)))
        final = choice(state, question, winners)
        return {key: final.get(key, 0.0) for key, _ in options}

    @app.get("/health")
    def health(_: None = Depends(authenticate)) -> dict[str, Any]:
        return {
            "status": "ready",
            "model": "semif",
            "checkpoint": os.path.basename(os.path.normpath(model_dir)),
            "device": metadata.get("device"),
            "startup_ms": round((time.monotonic() - loaded_at) * 1000),
        }

    @app.post("/v1/systemone")
    def system_one(request: DecisionRequest, _: None = Depends(authenticate)) -> dict[str, Any]:
        started = time.monotonic()
        answers: dict[str, Any] = {}
        for key, question in request.questions.items():
            text = str(question.get("instructions") or key)
            if question.get("type") == "noul":
                scores = probabilities(request.state, text, [("yes", "Yes"), ("no", "No")])
                answers[key] = {"type": "noul", "noul": float(scores["yes"])}
                continue
            criteria = question.get("criteria") or {}
            options = [(str(option), str(description)) for option, description in criteria.items()]
            if len(options) < 2:
                only = options[0][0] if options else None
                answers[key] = {"type": "choice", "choice": only, "confidence": 1.0 if only else 0.0,
                                "probabilities": {only: 1.0} if only else {}}
                continue
            scores = choice(request.state, text, options)
            best = max(scores, key=scores.get)
            answers[key] = {"type": "choice", "choice": best, "confidence": float(scores[best]),
                            "probabilities": {option: float(p) for option, p in scores.items()}}
        return {
            "model": request.model,
            "answers": answers,
            "sidecar": {"backend": "semif", "device": metadata.get("device"),
                        "elapsed_ms": round((time.monotonic() - started) * 1000)},
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--revision", required=True, help="pinned revision string recorded with every answer")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    token = os.environ.get("POK_SEMIF_TOKEN", "")
    if not token:
        raise SystemExit("POK_SEMIF_TOKEN is required")
    uvicorn.run(create_app(args.model_dir, args.revision, token, args.device), host="127.0.0.1",
                port=args.port, access_log=False, log_level="warning")


if __name__ == "__main__":
    main()
