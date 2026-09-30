"""Build a Laya typed-decisions training set from POK-Ai router training logs.

The desktop app (Settings > Decision router > Training data) and the CLI
write one `router-training/<session>.jsonl` per conversation when
`decision_router.training_log` is on. Each file holds:

- `question` records: the exact `/v1/systemone` body the router was (or, for
  decisions local grounding made, would have been) asked: `{state, questions}`
  plus the backend's answers when it was asked;
- `label` records: the proven answer for one question of a record, with its
  source (`grounding`, `quoted_label_match`, `local_evidence`).

This script joins them across any number of sessions (and machines: pass
several `--input` folders), keeps questions that have a proven label whose
value is one of the offered options, removes duplicates, splits train/test by
conversation so near-identical screens never straddle the split, and writes
JSONL rows with the fields Laya's fine-tuning notebook reads:

    {"id", "workflow", "state", "questions", "gold"}

where `state`, `questions`, and `gold` are JSON strings, as in the
`LocalLLaMA/typed-decisions` dataset. Load them with
`datasets.load_dataset("json", data_files={"train": ..., "test": ...})`.

Usage:
    python scripts/build_router_dataset.py --output dataset/ \\
        [--input DIR ...] [--test-fraction 0.15] [--exclude-source SOURCE ...] \\
        [--teacher BACKEND --teacher-min 0.9]

`--teacher BACKEND` adds soft labels: where no proven label exists, a
confident answer recorded from that decision backend becomes a soft target.
Proven labels always take precedence. Only backends whose terms allow it can
be teachers: hosted TypeSafe models (JEV) are refused, because TypeSafe's
Master Customer Agreement (section 2.3(b)) forbids using their output to
distill or train a model that imitates it.

For the same reason, sessions that used hosted JEV are left out of the
dataset by default, proven labels included: the agreement also forbids using
the service to develop a similar or competing product.
`--include-hosted-sessions` keeps them, for use only with TypeSafe's written
permission. Collect Laya training data with a local backend (Laya, kev,
llm_choice) or without a router.

`--outcomes` joins each session's task result (JSONL of {session_id, score},
written by `scripts/arena/run_arena.py dataset`), and
`--teacher-successful-only` keeps teacher answers only from sessions whose
task passed. Folders are searched recursively, so an arena run folder works
as an input directly.

The default input is the app's data folder (`%LOCALAPPDATA%\\POK-Ai\\POK-Ai\\
data\\router-training` on Windows). The logs stay local; review a sample
before sharing a dataset anywhere.
"""

from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

WORKFLOW = "pok_ai_desktop"

# Hosted decision models whose terms forbid training on their output
# (TypeSafe Master Customer Agreement, section 2.3(b)).
NO_TEACHING = ("jev",)


def default_input() -> list[str]:
    root = os.environ.get("LOCALAPPDATA")
    if root:
        return [str(Path(root) / "POK-Ai" / "POK-Ai" / "data" / "router-training")]
    return [str(Path.home() / ".local" / "share" / "pok-ai" / "router-training")]


def read_records(folders: list[str]) -> tuple[dict[str, dict], list[dict], list[dict]]:
    questions: dict[str, dict] = {}
    labels: list[dict] = []
    handbacks: list[dict] = []
    for folder in folders:
        for path in sorted(glob.glob(os.path.join(folder, "**", "*.jsonl"), recursive=True)):
            # Session traces and result tables share the extension; skip them.
            if os.path.basename(path) in {"trace.jsonl", "results.jsonl", "outcomes.jsonl"}:
                continue
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if record.get("kind") == "question" and record.get("id"):
                        questions[record["id"]] = record
                    elif record.get("kind") == "label" and record.get("id"):
                        labels.append(record)
                    elif record.get("kind") == "handback":
                        handbacks.append(record)
    return questions, labels, handbacks


def words(text: str) -> str:
    return " ".join("".join(c.lower() if c.isalnum() else " " for c in text).split())


def handback_examples(
    handbacks: list[dict], questions: dict[str, dict], stats: collections.Counter
) -> tuple[list[dict], dict[str, dict]]:
    """Training data from steps System 1 handed back and the primary model
    then did. When System 1 had asked a target question, the resolving
    control answers it (as a label). When it had no candidates, a choice
    question over the labels it could see is built, answered the same way.
    A resolving label System 1 could not see is a perception miss: counted,
    never trained on."""
    labels: list[dict] = []
    built: dict[str, dict] = {}
    for handback in handbacks:
        resolution = handback.get("resolution") or {}
        label = str(resolution.get("label") or "").strip()
        if not label:
            stats["handback: resolving action has no label"] += 1
            continue
        wanted = words(label)
        question = questions.get(str(handback.get("question_id") or ""))
        target = ((question or {}).get("questions") or {}).get("target_click")
        if target:
            matches = [option for option, text in (target.get("criteria") or {}).items()
                       if wanted and wanted in words(str(text))]
            if len(matches) == 1:
                labels.append({"id": question["id"], "question": "target_click", "label": matches[0], "source": "handback"})
            else:
                stats["handback: resolving control not among System 1's options"] += 1
            continue
        visible = []
        for text in handback.get("visible") or []:
            text = str(text).strip()
            if text and text not in visible:
                visible.append(text)
        options = visible[:16]
        chosen = [text for text in options if words(text) == wanted]
        if len(chosen) != 1:
            stats["handback: resolving control not in System 1's evidence (perception miss)"] += 1
            continue
        record_id = "handback-" + hashlib.sha256(json.dumps(handback, sort_keys=True).encode()).hexdigest()[:24]
        criteria = {f"option_{index}": text for index, text in enumerate(options)}
        built[record_id] = {
            "id": record_id,
            "session_id": handback.get("session_id"),
            "source": "handback",
            "state": {"window": handback.get("window"), "goal": handback.get("goal"),
                      "target_hint": handback.get("target_hint"), "visible": options},
            "questions": {"target_click": {
                "type": "choice",
                "instructions": "Which visible control should be activated next to accomplish the step?",
                "criteria": criteria,
            }},
        }
        labels.append({"id": record_id, "question": "target_click",
                       "label": next(key for key, text in criteria.items() if text == chosen[0]),
                       "source": "handback"})
    return labels, built


def gold_for(question: dict, label: str) -> dict | None:
    """One-hot gold probabilities over the question's options, or None when
    the label is not one of them."""
    kind = question.get("type")
    if kind == "noul":
        value = label.lower() in {"true", "yes", "1"}
        return {"label": str(value).lower(), "probabilities": {"true": float(value), "false": float(not value)}}
    options = list((question.get("criteria") or {}).keys())
    if label not in options:
        return None
    return {"label": label, "probabilities": {option: float(option == label) for option in options}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", action="append", help="folder of router-training/*.jsonl (repeatable)")
    parser.add_argument("--output", required=True, help="folder for train.jsonl and test.jsonl")
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--exclude-source", action="append", default=[], help="drop labels from this source")
    parser.add_argument(
        "--teacher",
        action="append",
        default=[],
        help="also learn from this backend's confident answers (model name prefix, e.g. laya) where no proven label exists",
    )
    parser.add_argument("--teacher-min", type=float, default=0.9, help="minimum teacher probability for its choice")
    parser.add_argument("--outcomes", help="JSONL of {session_id, score} task results (from the arena)")
    parser.add_argument(
        "--teacher-successful-only",
        action="store_true",
        help="use teacher answers only from sessions whose task passed (needs --outcomes)",
    )
    parser.add_argument(
        "--include-hosted-sessions",
        action="store_true",
        help="keep sessions that used hosted JEV (only with TypeSafe's written permission; "
        "their Master Customer Agreement 2.3(b) forbids using the service to develop a competing model)",
    )
    args = parser.parse_args()
    refused = [teacher for teacher in args.teacher if teacher.lower().startswith(NO_TEACHING)]
    if refused:
        parser.error(
            f"--teacher {', '.join(refused)}: TypeSafe's terms (Master Customer Agreement 2.3(b)) "
            "forbid training a model on its output; use proven labels or a local backend instead"
        )
    scores: dict[str, float] = {}
    if args.outcomes:
        with open(args.outcomes, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    scores[str(row["session_id"])] = float(row["score"])

    questions, labels, handbacks = read_records(args.input or default_input())
    # Sessions a hosted model with no-training terms took part in are left
    # out entirely unless you have the provider's permission.
    hosted = {
        str(record.get("session_id"))
        for record in questions.values()
        if str(record.get("model") or "").lower().startswith(NO_TEACHING)
    }
    if hosted and not args.include_hosted_sessions:
        kept = {key: record for key, record in questions.items() if str(record.get("session_id")) not in hosted}
        dropped = set(questions) - set(kept)
        questions = kept
        labels = [label for label in labels if label.get("id") not in dropped]
        handbacks = [item for item in handbacks if str(item.get("session_id")) not in hosted]
        print(f"left out {len(hosted)} sessions that used a hosted JEV backend (see --include-hosted-sessions)")
    handback_stats: collections.Counter = collections.Counter()
    handback_labels, handback_questions = handback_examples(handbacks, questions, handback_stats)
    questions.update(handback_questions)
    labels.extend(handback_labels)
    # First proven label per (record, question) wins; later ones are ignored.
    gold_by_record: dict[str, dict[str, tuple[str, str]]] = collections.defaultdict(dict)
    for label in labels:
        if label.get("source") in args.exclude_source:
            continue
        gold_by_record[label["id"]].setdefault(label.get("question", ""), (str(label.get("label")), label.get("source", "")))

    # Soft labels: a backend's confident answers fill in questions no proven
    # label covers. Proven labels always win; the teacher's full
    # probability spread is kept as a soft target.
    teacher_gold: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    if args.teacher:
        for record_id, record in questions.items():
            model = str(record.get("model") or "").lower()
            if record.get("source") != "router" or not any(model.startswith(t.lower()) for t in args.teacher):
                continue
            if args.teacher_successful_only and scores.get(str(record.get("session_id")), 0.0) <= 0.0:
                continue
            for key, answer in (record.get("answers") or {}).items():
                if key in gold_by_record.get(record_id, {}) or not isinstance(answer, dict):
                    continue
                probabilities = answer.get("probabilities") or {}
                choice = answer.get("choice")
                options = list(((record.get("questions") or {}).get(key) or {}).get("criteria", {}).keys())
                if choice not in options or probabilities.get(choice, 0.0) < args.teacher_min:
                    continue
                total = sum(float(probabilities.get(option, 0.0)) for option in options) or 1.0
                teacher_gold[record_id][key] = {
                    "label": choice,
                    "probabilities": {option: float(probabilities.get(option, 0.0)) / total for option in options},
                }
                gold_by_record.setdefault(record_id, {})

    rows: dict[str, list[dict]] = {"train": [], "test": []}
    seen: set[str] = set()
    stats = collections.Counter(handback_stats)
    for record_id, gold in gold_by_record.items():
        record = questions.get(record_id)
        if not record:
            stats["label without question"] += 1
            continue
        asked = record.get("questions") or {}
        kept_questions, kept_gold = {}, {}
        for key, (label, source) in gold.items():
            question = asked.get(key)
            gold_value = gold_for(question, label) if question else None
            if gold_value is None:
                stats["label not among options"] += 1
                continue
            kept_questions[key] = question
            kept_gold[key] = gold_value
            stats[f"question {key.split('_')[0]} / {source}"] += 1
        for key, soft in teacher_gold.get(record_id, {}).items():
            if key in kept_questions:
                continue
            kept_questions[key] = asked[key]
            kept_gold[key] = soft
            stats[f"question {key.split('_')[0]} / teacher {record.get('model')}"] += 1
        if not kept_questions:
            continue
        state = record.get("state") or {}
        fingerprint = hashlib.sha256(
            json.dumps([state, kept_questions, kept_gold], sort_keys=True).encode()
        ).hexdigest()
        if fingerprint in seen:
            stats["duplicate"] += 1
            continue
        seen.add(fingerprint)
        # Split by conversation: a session's screens go entirely to one side.
        session_hash = int(hashlib.sha256(str(record.get("session_id")).encode()).hexdigest(), 16)
        split = "test" if (session_hash % 10_000) / 10_000 < args.test_fraction else "train"
        rows[split].append({
            "id": record_id,
            "workflow": WORKFLOW,
            "state": json.dumps(state, ensure_ascii=False),
            "questions": json.dumps(kept_questions, ensure_ascii=False),
            "gold": json.dumps(kept_gold, ensure_ascii=False),
            # The task's result when known (arena runs); None otherwise.
            "outcome": scores.get(str(record.get("session_id"))),
        })

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for split, items in rows.items():
        with open(output / f"{split}.jsonl", "w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    decisions = sum(len(json.loads(row["gold"])) for items in rows.values() for row in items)
    print(f"{len(questions)} questions read, {len(labels)} labels, {len(handbacks)} hand-backs")
    print(f"train: {len(rows['train'])} states, test: {len(rows['test'])} states, {decisions} labeled decisions")
    for key, count in sorted(stats.items()):
        print(f"  {count:6}  {key}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
