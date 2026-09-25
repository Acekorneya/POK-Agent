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
        [--teacher jev --teacher-min 0.9]

`--teacher` adds distillation: where no proven label exists, a confident
answer from that backend (JEV's calibrated probabilities) becomes a soft
target. Proven labels always take precedence.

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


def default_input() -> list[str]:
    root = os.environ.get("LOCALAPPDATA")
    if root:
        return [str(Path(root) / "POK-Ai" / "POK-Ai" / "data" / "router-training")]
    return [str(Path.home() / ".local" / "share" / "pok-ai" / "router-training")]


def read_records(folders: list[str]) -> tuple[dict[str, dict], list[dict]]:
    questions: dict[str, dict] = {}
    labels: list[dict] = []
    for folder in folders:
        for path in sorted(glob.glob(os.path.join(folder, "*.jsonl"))):
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
    return questions, labels


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
        help="also learn from this backend's confident answers (model name prefix, e.g. jev) where no proven label exists",
    )
    parser.add_argument("--teacher-min", type=float, default=0.9, help="minimum teacher probability for its choice")
    args = parser.parse_args()

    questions, labels = read_records(args.input or default_input())
    # First proven label per (record, question) wins; later ones are ignored.
    gold_by_record: dict[str, dict[str, tuple[str, str]]] = collections.defaultdict(dict)
    for label in labels:
        if label.get("source") in args.exclude_source:
            continue
        gold_by_record[label["id"]].setdefault(label.get("question", ""), (str(label.get("label")), label.get("source", "")))

    # Distillation: a strong backend's confident answers fill in questions no
    # proven label covers. Proven labels always win; the teacher's full
    # probability spread is kept as a soft target.
    teacher_gold: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    if args.teacher:
        for record_id, record in questions.items():
            model = str(record.get("model") or "").lower()
            if record.get("source") != "router" or not any(model.startswith(t.lower()) for t in args.teacher):
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
    stats = collections.Counter()
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
        })

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for split, items in rows.items():
        with open(output / f"{split}.jsonl", "w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    decisions = sum(len(json.loads(row["gold"])) for items in rows.values() for row in items)
    print(f"{len(questions)} questions read, {len(labels)} labels")
    print(f"train: {len(rows['train'])} states, test: {len(rows['test'])} states, {decisions} labeled decisions")
    for key, count in sorted(stats.items()):
        print(f"  {count:6}  {key}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
