"""Benchmark a running decision-router sidecar against diagnostics/decision_suite_cases.json.

Works against any sidecar speaking the shared /v1/systemone contract
(laya_sidecar.py, zeiger_sidecar.py, or any future adapter following
docs/decision-router-backends.md) — this script doesn't know or care which
model is behind the endpoint. Point it at a running sidecar and it reports
accuracy against each case's expected operation/target plus per-case and
average latency, so different checkpoints/models can be compared on our own
task instead of their model cards' generic benchmarks.

The request shape mirrors the real two-stage decomposition built by
`decide_once` in crates/pok-ai-core/src/decision.rs: one `operation` choice
question plus one `target_<OPERATION>` choice question per operation group —
not a single flat "target" question. Earlier benchmark runs used a flat
question and never exercised the actual operation-probability gate that
`session.rs` checks in production, which is why checkpoint comparisons kept
missing the real bottleneck (see docs/decision-router-backends.md).

For cases carrying a `clustering_band` block (added to test the speculative
committee-promotion idea borrowed from jev-x-kit's "BELKİ" gatekeeper — see
the plan this script supports), this also asks a 3-question committee per
operation group, mirroring jev-x-kit's `speculate()` function
(src/core/gatekeeper.ts) as closely as our contract allows: independent
yes/no/unknown choice questions asking whether the operation is (1) safe and
reversible, (2) cheap/low-risk to try now, and (3) directly evidenced by the
current state (not inferred or left over from a different task). Promotion
requires unanimous "yes" across all three, exactly like jev-x-kit's gate
(`agree === subChoices.length`) — not a single blended confidence score,
which is what the first version of this script tried and which came back at
chance-level correlation (50%, n=4) with a false negative worse than a coin
flip. This version tests whether committee *structure* (multiple independent
checks, unanimous vote) makes a difference where a single question didn't.

Everything still goes in the same HTTP request as `operation`/`target_<OP>`
— zero extra round trips, cheaper than jev-x-kit's own two-call design (see
docs/decision-router-backends.md for why that's possible here).

Usage:
    python scripts/bench_decision_models.py --endpoint http://127.0.0.1:43199/v1/systemone \
        --token test-token-1234 --label "laya-typed-decisions"
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

SUITE_PATH = Path(__file__).resolve().parent.parent / "diagnostics" / "decision_suite_cases.json"

# One (question_suffix, instructions) pair per jev-x-kit's speculate() sub-question.
COMMITTEE_SUBQUESTIONS = [
    (
        "reversible",
        "Is choosing operation {operation} safe and reversible right now, given the "
        "current structured state? Judge only the supplied criteria.",
    ),
    (
        "cheap",
        "Is choosing operation {operation} cheap and low-risk to try now (no risk of "
        "wasted, destructive, or hard-to-undo side effects)? Judge only the supplied criteria.",
    ),
    (
        "evidenced",
        "Is choosing operation {operation} directly supported by the current structured "
        "state, not inferred or left over from a different task? Judge only the supplied criteria.",
    ),
]
COMMITTEE_OPTIONS = {
    "yes": "The sub-question is clearly true",
    "no": "The sub-question is clearly false",
    "unknown": "There is not enough evidence to decide either way",
}


def operation_id(tool: str) -> str:
    return "".join(ch.upper() if ch.isalnum() else "_" for ch in tool)


def candidate_operation(candidate: dict) -> str:
    if candidate["id"] == "search_web_for_request":
        return "SEARCH_WEB"
    tool = candidate["tool"]
    mapping = {
        "__done__": "DONE",
        "__blocked__": "BLOCKED",
        "managed_browser_open": "OPEN_URL",
        "browser_navigate": "OPEN_URL",
        "managed_browser_snapshot": "CAPTURE",
        "capture_screen": "CAPTURE",
        "managed_browser_click": "CLICK",
        "click_target": "CLICK",
        "managed_browser_type": "TYPE_TEXT",
        "managed_browser_select": "SELECT",
        "managed_browser_hover": "HOVER",
        "scroll_down": "SCROLL_DOWN",
        "scroll_up": "SCROLL_UP",
        "activate_window": "ACTIVATE_WINDOW",
        "list_windows": "LIST_WINDOWS",
    }
    return mapping.get(tool, operation_id(tool))


def operation_groups(candidates: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for candidate in candidates:
        groups.setdefault(candidate_operation(candidate), []).append(candidate)
    return groups


def build_body(case: dict, model: str = "bench") -> tuple[dict, dict[str, list[dict]]]:
    groups = operation_groups(case["candidates"])
    operation_criteria = {
        operation: "; ".join(c["description"][:160] for c in members[:4])
        for operation, members in groups.items()
    }
    questions: dict = {
        "operation": {
            "type": "choice",
            "instructions": "Choose the single operation that best advances the active task from the current structured state.",
            "criteria": operation_criteria,
        }
    }
    for operation, members in groups.items():
        questions[f"target_{operation}"] = {
            "type": "choice",
            "instructions": f"Choose the exact bounded target for operation {operation}. Judge only the supplied criteria; never invent a target.",
            "criteria": {c["id"]: c["description"][:400] for c in members},
        }
        for suffix, instructions in COMMITTEE_SUBQUESTIONS:
            questions[f"committee_{suffix}_{operation}"] = {
                "type": "choice",
                "instructions": instructions.format(operation=operation),
                "criteria": COMMITTEE_OPTIONS,
            }
    body = {
        "model": model,
        "state": {
            "task": case["task"],
            "current_step": case["current_step"],
            "candidates": [
                {"id": c["id"], "description": c["description"]} for c in case["candidates"]
            ],
        },
        "questions": questions,
    }
    return body, groups


def resolve_should_promote(case: dict, result: dict) -> bool:
    band = case["clustering_band"]
    raw_label = band["should_promote"]
    if raw_label == "conditional_on_selected_target":
        return result["target"] == band["correct_target"]
    if raw_label == "conditional_on_selected_operation":
        return result["operation"] == band["correct_operation"]
    return raw_label


# Candidate combined operation+target probability gates, tested against the
# same real (operation_probability, target_probability) pairs the committee
# was tested against. Laya's operation stage runs conservative (0.35-0.69 in
# observed data) while its target stage is often near-certain (1.0) on the
# same correct decisions -- these formulas test whether letting a confident
# target compensate for a merely-clustering-band operation score beats both
# the current independent AND-gate and the committee (58.3% on n=12).
COMBINED_GATE_FORMULAS = {
    "geometric_mean>=0.55": lambda op, tgt: (op * tgt) ** 0.5 >= 0.55,
    "min(op,tgt)>=0.40": lambda op, tgt: min(op, tgt) >= 0.40,
    "op>=0.35 and tgt>=0.90": lambda op, tgt: op >= 0.35 and tgt >= 0.90,
    "weighted(0.3*op+0.7*tgt)>=0.70": lambda op, tgt: 0.3 * op + 0.7 * tgt >= 0.70,
}


def run_case(endpoint: str, token: str, case: dict, model: str = "bench") -> dict:
    body, groups = build_body(case, model)
    headers = {"Authorization": f"Bearer {token}"}
    started = time.monotonic()
    response = requests.post(endpoint, json=body, headers=headers, timeout=30)
    elapsed_ms = (time.monotonic() - started) * 1000
    response.raise_for_status()
    answers = response.json()["answers"]

    operation_answer = answers["operation"]
    operation = operation_answer["choice"]
    operation_probability = operation_answer["probabilities"].get(operation, 0.0)

    target_key = f"target_{operation}"
    target_answer = answers.get(target_key, {})
    target = target_answer.get("choice")
    target_probability = (target_answer.get("probabilities") or {}).get(target, 0.0)

    committee_votes: dict[str, str | None] = {}
    for suffix, _ in COMMITTEE_SUBQUESTIONS:
        answer = answers.get(f"committee_{suffix}_{operation}") or {}
        committee_votes[suffix] = answer.get("choice")
    unanimous_yes = all(vote == "yes" for vote in committee_votes.values())
    unanimous_no = all(vote == "no" for vote in committee_votes.values())
    yes_count = sum(1 for vote in committee_votes.values() if vote == "yes")

    return {
        "operation": operation,
        "target": target,
        "operation_probability": operation_probability,
        "target_probability": target_probability,
        "committee_votes": committee_votes,
        "committee_unanimous_yes": unanimous_yes,
        "committee_unanimous_no": unanimous_no,
        "committee_yes_count": yes_count,
        "elapsed_ms": elapsed_ms,
        "operation_count": len(groups),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True, help="Full /v1/systemone URL")
    parser.add_argument("--token", required=True)
    parser.add_argument("--label", default="model")
    parser.add_argument(
        "--model",
        default="bench",
        help="Value sent in the request body's 'model' field. Local sidecars ignore this; "
        "the hosted JEV API validates it strictly (e.g. 'jev-1.13.0').",
    )
    args = parser.parse_args()

    suite = json.loads(SUITE_PATH.read_text())
    cases = suite["cases"]

    print("=" * 96)
    print(f"  {args.label} — {len(cases)} cases — {args.endpoint}")
    print("=" * 96)

    correct = 0
    total_ms = 0.0
    committee_rows: list[tuple[dict, dict]] = []
    for case in cases:
        result = run_case(args.endpoint, args.token, case, args.model)
        ok = (
            result["operation"] == case["expected"]["operation"]
            and result["target"] == case["expected"]["target"]
        )
        correct += ok
        total_ms += result["elapsed_ms"]
        status = "PASS" if ok else "FAIL"
        committee_note = ""
        if "clustering_band" in case:
            committee_rows.append((case, result))
            votes = result["committee_votes"]
            committee_note = f" votes={votes['reversible']}/{votes['cheap']}/{votes['evidenced']}"
        print(
            f"  [{status}] {case['id']:<48} op={result['operation']:<16} "
            f"target={str(result['target']):<20} op_prob={result['operation_probability']:.3f} "
            f"tgt_prob={result['target_probability']:.3f} "
            f"lat={result['elapsed_ms']:.0f}ms{committee_note}"
        )

    n = len(cases)
    print("=" * 96)
    print(f"  {args.label}: {correct}/{n} correct ({100 * correct / n:.1f}%), avg latency {total_ms / n:.0f}ms")

    if committee_rows:
        print("-" * 96)
        print("  Speculative-committee correlation check (clustering_band cases only)")
        print("  Promotion rule: unanimous 'yes' across reversible/cheap/evidenced (jev-x-kit's own rule)")
        print("-" * 96)
        agree = 0
        for case, result in committee_rows:
            band = case["clustering_band"]
            would_promote = result["committee_unanimous_yes"]
            should_promote = resolve_should_promote(case, result)
            label_note = ""
            if isinstance(band["should_promote"], str):
                key = "target" if "target" in band["should_promote"] else "operation"
                label_note = f"(resolved: selected={result[key]}, correct={band[f'correct_{key}']})"
            matches = would_promote == should_promote
            agree += matches
            votes = result["committee_votes"]
            print(
                f"  {'OK  ' if matches else 'MISS'} {case['id']:<48} "
                f"should_promote={should_promote!s:<6}{label_note} "
                f"votes(reversible/cheap/evidenced)={votes['reversible']}/{votes['cheap']}/{votes['evidenced']} "
                f"yes_count={result['committee_yes_count']}/3 would_promote={would_promote}"
            )
        total = len(committee_rows)
        print("-" * 96)
        print(
            f"  committee correlation: {agree}/{total} matched expected promote/hold "
            f"({100 * agree / total:.1f}%)"
        )
        print(
            "  Go/no-go: this must clearly beat chance and must not just rubber-stamp "
            "whatever the operation answer already picked — see the plan's Phase 1 criteria."
        )

        by_group_size: dict[int, list[bool]] = {}
        for case, result in committee_rows:
            band = case["clustering_band"]
            size = band.get("operation_group_size")
            if size is None:
                continue
            should_promote = resolve_should_promote(case, result)
            matches = result["committee_unanimous_yes"] == should_promote
            by_group_size.setdefault(size, []).append(matches)
        if by_group_size:
            print("-" * 96)
            print("  Correlation by operation-group size (answers the single-candidate-bypass question)")
            for size in sorted(by_group_size):
                matches = by_group_size[size]
                print(
                    f"    group_size={size:<3} n={len(matches):<2} "
                    f"correct={sum(matches)}/{len(matches)} ({100 * sum(matches) / len(matches):.0f}%)"
                )

        print("-" * 96)
        print("  Combined operation+target probability gate check (no committee questions needed)")
        print("  Tests whether a confident target can compensate for a clustering-band operation score")
        print("-" * 96)
        for name, formula in COMBINED_GATE_FORMULAS.items():
            agree = 0
            details = []
            for case, result in committee_rows:
                should_promote = resolve_should_promote(case, result)
                would_promote = formula(result["operation_probability"], result["target_probability"])
                matches = would_promote == should_promote
                agree += matches
                details.append((case["id"], matches, would_promote, should_promote))
            total = len(committee_rows)
            print(f"  {name:<32} {agree}/{total} correct ({100 * agree / total:.1f}%)")
            for case_id, matches, would_promote, should_promote in details:
                if not matches:
                    print(f"      MISS {case_id:<48} should_promote={should_promote} would_promote={would_promote}")
    print("=" * 96)


if __name__ == "__main__":
    main()
