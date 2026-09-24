"""Loopback-only OpenJev typed-decision sidecar for POK-Agent.

OpenJev (https://huggingface.co/AlexWortega/openjev) is an NLI cross-encoder
(Qwen3.5 backbone, contradiction/entailment/neutral classification head), not
a typed-choice model like Laya/Zeiger. This adapter is the translation layer
docs/decision-router-backends.md describes as real, model-specific work: each
`choice` question becomes one `predict_hypotheses` call (one shared-premise,
many-hypotheses NLI pass), with per-option P(entailment) taken as the raw
score and renormalized across options into a `probabilities` map, since
per-hypothesis entailment probabilities are independent 3-way softmaxes, not
already comparable across options.

`--format v5` (the default) uses the author's own typed-decision adapter,
`code/openjev_decide.py` (`OpenJev.decide`): the state is JSON, and each
option becomes the hypothesis `The answer to "<question>" is <id>: <rubric>`,
the exact format the 4B v5 checkpoint was trained and JevBench-scored on.
Each option is scored on its own, so answers do not depend on option order.
`--format legacy` keeps the original hand-written hypothesis wording used for
the earlier 0.8B measurement.

Requires `modeling_openjev.py` (downloaded alongside the checkpoint) on
PYTHONPATH, plus transformers/torch/sentencepiece/tiktoken. Mirrors
laya_sidecar.py's/zeiger_sidecar.py's contract exactly (same /health and
/v1/systemone shape, same bearer-token auth).
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import time
from typing import Any

import torch
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from modeling_openjev import ENT, OpenJevCrossEncoder
from pydantic import BaseModel

try:  # the checkpoint's own typed-decision adapter (code/openjev_decide.py)
    from openjev_decide import RUBRIC_MARK, OpenJev
except ImportError:  # older checkpoints ship without it
    OpenJev = None
    RUBRIC_MARK = ""


class DecisionRequest(BaseModel):
    model: str
    state: Any
    questions: dict[str, dict[str, Any]]


def answer_choice(ce: OpenJevCrossEncoder, premise: str, question: dict[str, Any]) -> dict[str, Any]:
    criteria: dict[str, str] = question["criteria"]
    option_ids = list(criteria.keys())
    hypotheses = [f"The correct answer is: {opt_id}: {criteria[opt_id]}" for opt_id in option_ids]
    probs = ce.predict_hypotheses(premise, hypotheses)
    entailment = probs[:, ENT]
    total = float(entailment.sum())
    normalized = (entailment / total) if total > 0 else (entailment * 0 + 1.0 / len(entailment))
    probabilities = {opt_id: round(float(p), 5) for opt_id, p in zip(option_ids, normalized)}
    best_idx = int(entailment.argmax())
    return {
        "type": "choice",
        "choice": option_ids[best_idx],
        "confidence": round(float(normalized[best_idx]), 5),
        "probabilities": probabilities,
    }


def answer_noul(ce: OpenJevCrossEncoder, premise: str, question: dict[str, Any]) -> dict[str, Any]:
    probs = ce.predict_hypotheses(premise, [question["instructions"]])
    p_true = float(probs[0, ENT])
    return {
        "type": "noul",
        "noul": round(p_true, 5),
        "confidence": round(max(p_true, 1.0 - p_true), 5),
    }


def answer_v5(jev: Any, state: Any, question: dict[str, Any]) -> dict[str, Any]:
    """Typed answer through the checkpoint's own `OpenJev.decide`."""
    if question.get("type") == "noul":
        result = jev.decide(state, [{"type": "noul", "instructions": question.get("instructions", ""), "options": ["no", "yes"]}])[0]
        p_true = float(result["noul"])
        return {"type": "noul", "noul": round(p_true, 5), "confidence": round(max(p_true, 1.0 - p_true), 5)}
    criteria: dict[str, str] = question["criteria"]
    options = list(criteria.keys())
    instructions = str(question.get("instructions", "")) + RUBRIC_MARK + json.dumps(criteria, ensure_ascii=False)
    result = jev.decide(state, [{"type": "choice", "instructions": instructions, "options": options}])[0]
    probabilities = {option: round(float(p), 5) for option, p in result["probabilities"].items()}
    best = max(probabilities, key=probabilities.get)
    return {"type": "choice", "choice": best, "confidence": probabilities[best], "probabilities": probabilities}


def create_app(model_dir: str, subfolder: str | None, token: str, device: str, answer_format: str) -> FastAPI:
    loaded_at = time.monotonic()
    ce = OpenJevCrossEncoder(model_dir, subfolder=subfolder, device=None if device == "auto" else device)
    if answer_format == "v5" and OpenJev is None:
        raise SystemExit("--format v5 needs code/openjev_decide.py on PYTHONPATH")
    jev = OpenJev(ce) if answer_format == "v5" else None

    app = FastAPI(title="POK-Agent OpenJev sidecar", docs_url=None, redoc_url=None)

    def authenticate(authorization: str | None = Header(default=None)) -> None:
        expected = f"Bearer {token}"
        if not authorization or not secrets.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="invalid sidecar token")

    @app.get("/health")
    def health(_: None = Depends(authenticate)) -> dict[str, Any]:
        display_device = str(ce.device)
        if ce.device == "cuda" and torch.cuda.is_available():
            display_device = f"cuda ({torch.cuda.get_device_name(0)})"
        return {
            "status": "ready",
            "model": "openjev",
            "checkpoint": subfolder or os.path.basename(os.path.normpath(model_dir)),
            "device": display_device,
            "device_reason": "",
            "context_tokens": int(ce.max_len),
            "startup_ms": round((time.monotonic() - loaded_at) * 1000),
        }

    @app.post("/v1/systemone")
    def system_one(request: DecisionRequest, _: None = Depends(authenticate)) -> dict[str, Any]:
        started = time.monotonic()
        premise = f"{request.state}"
        answers: dict[str, Any] = {}
        for question_id, question in request.questions.items():
            if jev is not None:
                answers[question_id] = answer_v5(jev, request.state, question)
            elif question.get("type") == "noul":
                answers[question_id] = answer_noul(ce, premise, question)
            else:
                answers[question_id] = answer_choice(ce, premise, question)
        display_device = str(ce.device)
        if ce.device == "cuda" and torch.cuda.is_available():
            display_device = f"cuda ({torch.cuda.get_device_name(0)})"
        return {
            "model": request.model,
            "answers": answers,
            "sidecar": {
                "backend": "openjev",
                "format": answer_format,
                "device": display_device,
                "context_tokens": int(ce.max_len),
                "elapsed_ms": round((time.monotonic() - started) * 1000),
            },
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--subfolder", default=None, help="e.g. qwen3.5-0.8b-nli-v2s-long")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--format", choices=["v5", "legacy"], default="v5")
    args = parser.parse_args()
    token = os.environ.get("POK_LAYA_TOKEN", "")
    if not token:
        raise SystemExit("POK_LAYA_TOKEN is required")
    uvicorn.run(
        create_app(args.model_dir, args.subfolder, token, args.device, args.format),
        host="127.0.0.1",
        port=args.port,
        access_log=False,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
