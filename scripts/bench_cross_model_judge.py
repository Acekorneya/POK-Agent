"""Cross-model "picker + judge" cascade test against diagnostics/decision_suite_cases.json.

Tests the architecture idea surfaced from today's four-backend comparison
(docs/decision-router-backends.md, decision-router plan): rather than one
backend both picking the candidate and judging its own pick, one backend
(the "picker") makes the two-stage operation/target choice, and a *different*
backend (the "judge") answers a 3-question committee scoped to that specific
picked candidate. This mirrors the "semantic cascade" pattern from recent
model-cascade research -- cross-model agreement as the deferral signal,
rather than a single model's own self-reported confidence -- which reported
matching or beating single-model quality at a fraction of the cost/latency.

Unlike bench_decision_models.py (which tests one backend's own two-stage
pick plus its own committee about its own pick), this script:
  1. Queries the picker backend for its two-stage operation/target answer.
  2. Builds a committee question set scoped to that SPECIFIC candidate
     (not a generic per-operation-family question), and sends it to the
     judge backend.
  3. Ground truth is simply: does the picker's chosen candidate match the
     case's `expected` operation+target? This works for all 20 suite cases,
     not just the 12 `clustering_band` ones, since it doesn't depend on the
     static clustering_band labels at all -- it's evaluated fresh against
     whatever the picker actually chose in this run.

Usage:
    python scripts/bench_cross_model_judge.py \
        --picker-endpoint http://127.0.0.1:43199/v1/systemone --picker-token TOK1 \
        --judge-endpoint http://127.0.0.1:43399/v1/systemone --judge-token TOK2 \
        --picker-label laya --judge-label zeiger
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

from bench_decision_models import (
    COMMITTEE_OPTIONS,
    build_body,
    candidate_operation,
    operation_groups,
)

SUITE_PATH = Path(__file__).resolve().parent.parent / "diagnostics" / "decision_suite_cases.json"

ACTION_CLASS = (
    "in-page navigation: changes only the currently displayed page or scroll position "
    "and can be undone with the browser Back action"
)
# Mirrors decision.rs judge_action_class: browser interactions only, so the
# offline suite (which contains no managed_browser_click candidates) is
# structurally unaffected.
ACTION_CLASS_TOOLS = {
    "managed_browser_click",
    "managed_browser_scroll",
    "managed_browser_hover",
}

CANDIDATE_COMMITTEE_SUBQUESTIONS = [
    (
        "reversible",
        "Is taking this specific action safe and reversible right now, given the current "
        "structured state: {description!r}? Judge only the supplied criteria.",
    ),
    (
        "cheap",
        "Is taking this specific action cheap and low-risk to try now (no risk of wasted, "
        "destructive, or hard-to-undo side effects): {description!r}? Judge only the supplied criteria.",
    ),
    (
        "evidenced",
        "Is this specific action directly supported by the current structured state, not "
        "inferred or left over from a different task: {description!r}? Judge only the supplied criteria.",
    ),
]

# Comparative 4th question, added per user request to target the specific
# failure mode the absolute reversible/cheap/evidenced questions miss:
# genuine multi-way ties (case_16/case_17-style) where every individual
# question can look fine for a plausible-but-wrong candidate. Research on
# LLM-as-judge protocols (pairwise vs. pointwise) finds pairwise comparison
# resists the scoring pathologies (central-tendency compression, scale
# drift) that hurt absolute ratings -- but is also more vulnerable to
# distraction/manipulation (~35% flip rate vs. ~9% for absolute scoring in
# published results), so this is tested as an ADDITION to the existing
# three questions, not a replacement.
COMPARATIVE_INSTRUCTIONS = (
    "Given the active task, which of these two candidate actions is the better fit right now? "
    "Judge only the two supplied options; never invent a third."
)


def find_runner_up(case: dict, pick: dict) -> dict | None:
    candidate = pick["candidate"]
    if candidate is None:
        return None
    same_operation = [
        c
        for c in case["candidates"]
        if candidate_operation(c) == pick["operation"] and c["id"] != candidate["id"]
    ]
    if not same_operation:
        return None
    same_operation.sort(key=lambda c: c.get("local_score", 0.0), reverse=True)
    return same_operation[0]


def get_pick(endpoint: str, token: str, case: dict, model: str) -> dict:
    body, groups = build_body(case, model)
    headers = {"Authorization": f"Bearer {token}"}
    started = time.monotonic()
    response = requests.post(endpoint, json=body, headers=headers, timeout=30)
    elapsed_ms = (time.monotonic() - started) * 1000
    response.raise_for_status()
    answers = response.json()["answers"]
    operation = answers["operation"]["choice"]
    target_answer = answers.get(f"target_{operation}", {})
    target = target_answer.get("choice")
    candidate = next((c for c in case["candidates"] if c["id"] == target), None)
    return {
        "operation": operation,
        "target": target,
        "candidate": candidate,
        "elapsed_ms": elapsed_ms,
    }


def judge_pick(endpoint: str, token: str, case: dict, pick: dict, model: str, include_state: bool, action_class: bool = False) -> dict:
    candidate = pick["candidate"]
    if candidate is None:
        return {
            "votes": {"reversible": None, "cheap": None, "evidenced": None},
            "comparative_choice": None,
            "runner_up_id": None,
            "elapsed_ms": 0.0,
        }
    questions = {
        suffix: {
            "type": "choice",
            "instructions": instructions.format(description=candidate["description"][:300]),
            "criteria": COMMITTEE_OPTIONS,
        }
        for suffix, instructions in CANDIDATE_COMMITTEE_SUBQUESTIONS
    }
    runner_up = find_runner_up(case, pick)
    if runner_up is not None:
        questions["comparative"] = {
            "type": "choice",
            "instructions": COMPARATIVE_INSTRUCTIONS,
            "criteria": {
                "a": candidate["description"][:300],
                "b": runner_up["description"][:300],
            },
        }
    state = {
        "task": case["task"],
        "current_step": case["current_step"],
        "candidate": {"id": candidate["id"], "description": candidate["description"]},
    }
    if action_class and candidate.get("tool") in ACTION_CLASS_TOOLS:
        state["candidate"]["action_class"] = ACTION_CLASS
    if include_state:
        # Mirrors bounded_judge_state in decision.rs: the NextAction keys the
        # picker saw, string-bounded.
        raw_state = case.get("state") or {}
        context = {
            key: raw_state[key]
            for key in ("window", "browser", "last_action", "previous_outcome", "state_revision")
            if key in raw_state
        }
        state["context"] = context
    body = {
        "model": model,
        "state": state,
        "questions": questions,
    }
    headers = {"Authorization": f"Bearer {token}"}
    started = time.monotonic()
    response = requests.post(endpoint, json=body, headers=headers, timeout=30)
    elapsed_ms = (time.monotonic() - started) * 1000
    response.raise_for_status()
    answers = response.json()["answers"]
    votes = {
        suffix: (answers.get(suffix) or {}).get("choice")
        for suffix, _ in CANDIDATE_COMMITTEE_SUBQUESTIONS
    }
    comparative_choice = (answers.get("comparative") or {}).get("choice") if runner_up else None
    return {
        "votes": votes,
        "comparative_choice": comparative_choice,
        "runner_up_id": runner_up["id"] if runner_up else None,
        "elapsed_ms": elapsed_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--picker-endpoint", required=True)
    parser.add_argument("--picker-token", required=True)
    parser.add_argument("--picker-model", default="bench")
    parser.add_argument("--picker-label", default="picker")
    parser.add_argument("--judge-endpoint", required=True)
    parser.add_argument("--judge-token", required=True)
    parser.add_argument("--judge-model", default="bench")
    parser.add_argument("--judge-label", default="judge")
    parser.add_argument(
        "--judge-state",
        action="store_true",
        help="include the case's bounded NextAction state in the judge payload (mirrors production)",
    )
    parser.add_argument(
        "--judge-action-class",
        action="store_true",
        help="include the factual action class in the candidate state (mirrors production)",
    )
    args = parser.parse_args()

    suite = json.loads(SUITE_PATH.read_text())
    cases = suite["cases"]

    print("=" * 100)
    print(f"  Cross-model cascade: {args.picker_label} picks -> {args.judge_label} judges  ({len(cases)} cases)")
    print("=" * 100)

    agree_3q = 0
    agree_4q = 0
    picker_correct = 0
    comparative_evaluable = 0
    comparative_agrees_with_correctness = 0
    for case in cases:
        pick = get_pick(args.picker_endpoint, args.picker_token, case, args.picker_model)
        pick_correct = (
            pick["operation"] == case["expected"]["operation"]
            and pick["target"] == case["expected"]["target"]
        )
        picker_correct += pick_correct

        judged = judge_pick(args.judge_endpoint, args.judge_token, case, pick, args.judge_model, args.judge_state, args.judge_action_class)
        votes = judged["votes"]
        unanimous_yes = all(v == "yes" for v in votes.values())

        would_promote_3q = unanimous_yes
        matches_3q = would_promote_3q == pick_correct
        agree_3q += matches_3q

        # 4-question version: a runner-up existed and the judge preferred it
        # ("b") vetoes promotion even if the absolute questions were unanimous
        # yes -- this is the one new failure mode the comparative question can
        # catch that the three absolute questions structurally cannot.
        comparative_choice = judged["comparative_choice"]
        would_promote_4q = unanimous_yes and comparative_choice != "b"
        matches_4q = would_promote_4q == pick_correct
        agree_4q += matches_4q

        comparative_note = ""
        if comparative_choice is not None:
            runner_up_is_correct = (
                not pick_correct and judged["runner_up_id"] == case["expected"]["target"]
            )
            if pick_correct or runner_up_is_correct:
                comparative_evaluable += 1
                expected_choice = "a" if pick_correct else "b"
                comparative_agrees_with_correctness += comparative_choice == expected_choice
            comparative_note = f" comparative={comparative_choice}"

        status = "PICK-OK  " if pick_correct else "PICK-BAD "
        judge_status = "OK  " if matches_4q else "MISS"
        print(
            f"  [{status}][{judge_status}] {case['id']:<48} "
            f"pick={pick['operation']}/{pick['target']!s:<20} "
            f"votes={votes['reversible']}/{votes['cheap']}/{votes['evidenced']}{comparative_note} "
            f"would_promote_3q={would_promote_3q} would_promote_4q={would_promote_4q} "
            f"pick_correct={pick_correct} "
            f"pick_lat={pick['elapsed_ms']:.0f}ms judge_lat={judged['elapsed_ms']:.0f}ms"
        )

    n = len(cases)
    print("=" * 100)
    print(f"  {args.picker_label} raw pick accuracy: {picker_correct}/{n} ({100 * picker_correct / n:.1f}%)")
    print(
        f"  {args.judge_label} judging {args.picker_label}'s picks (3Q: reversible/cheap/evidenced) -- "
        f"correlation: {agree_3q}/{n} ({100 * agree_3q / n:.1f}%)"
    )
    print(
        f"  {args.judge_label} judging {args.picker_label}'s picks (4Q: + comparative veto) -- "
        f"correlation: {agree_4q}/{n} ({100 * agree_4q / n:.1f}%)"
    )
    if comparative_evaluable:
        print(
            f"  comparative question accuracy (picks the objectively-correct option, "
            f"where determinable): {comparative_agrees_with_correctness}/{comparative_evaluable} "
            f"({100 * comparative_agrees_with_correctness / comparative_evaluable:.1f}%)"
        )
    print(
        "  Compare 3Q vs 4Q above: the comparative question should only help if 4Q's "
        "correlation is >= 3Q's without materially hurting recall on correct picks."
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
