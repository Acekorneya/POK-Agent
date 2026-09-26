# System 1 + System 2 for General Computer Use: Thesis, Evidence, Roadmap

This is the working record of POK-Ai's central research question: what has been
built, what the measurements say, and what remains. Update it after every arena
run or architectural change.

## The thesis

A general computer-use agent should work the way people do:

- **System 2** (a large reasoning LLM, "the planner") thinks. It decomposes the
  task, writes plans with checkable conditions, handles anything new or
  surprising, writes code for bulk content work, and turns experience into
  skills.
- **System 1** (grounding plus a small, fast, non-generative decision model:
  Laya locally, JEV hosted) acts. It finds targets, clicks, types, checks
  "did it work?", reacts to popups, and escalates to System 2 only on surprise.

With practice, work should move from System 2 to System 1, as human skills
move from deliberate to automatic ("muscle memory"). The measurable prediction
is that **planner calls per task fall with practice, following a power law of
practice**, while success holds or rises:

$$C(N) = C_{\min} + (C_1 - C_{\min}) \cdot N^{-\beta}$$

Here $C(N)$ is the planner calls on the $N$-th encounter with a task (or task
family), $C_1$ is the first-time cost, $C_{\min}$ is the floor (ideally 1–2
calls: send a plan, confirm the result), and $\beta$ is the learning rate.
Newell and Rosenbloom explained this law by *chunking*; a POK-Ai skill is a
chunk, a verified multi-step plan that later runs as one unit through
System 1.

## Why this is different

Most computer-use harnesses send every step through one large model (Navi,
Agent S2/S3), or cut calls by batching the large model's own predictions (UFO²
speculative execution). Research cascades route easy steps to a smaller
*generative* policy model. Skill-library work (Agent Workflow Memory and
others) reuses successful trajectories as prompts. MCP servers and CLI coding
agents give models tools, but no fast decision layer and no desktop
perception-action loop.

We found no open harness that combines:

1. an LLM planner that writes **checkable plan trees** (quoted labels,
   branches, interrupts, reads);
2. **local grounding first** (UI Automation + OCR), with no model call when a
   quoted label settles a decision;
3. a **non-autoregressive typed-decision model** as the executor and verifier,
   answering small questions with calibrated probabilities and handing back
   when unsure;
4. **skills and memory** learned only from the agent's own verified
   experience, deduplicated, and rewritten by the planner;
5. a **training loop** that distills a hosted System 1 (JEV) and the planner's
   own resolutions into a local one (Laya), on the user's own software.

Absence of evidence is not proof; this claim should be rechecked before any
publication.

## What is built

| Area | Status | Where |
| --- | --- | --- |
| Plan trees for System 1 (`fast_actions`: steps, branches, interrupts, reads, avoid) | done | `session/fast_actions.rs`, `builtins/fast_actions_tool.rs` |
| Quoted-label grounding and local completion checks | done | `decision.rs`, `grounding.rs` |
| Typed-decision router (JEV, Laya, LLM choice) with thresholds and hand-back | done | `decision.rs`, `session/decision_router.rs` |
| Cursor-free input (UI Automation patterns first) and background work | done | `builtins/input.rs`, `agent_window.rs` |
| Skills: learn, deduplicate (3 layers), rewrite, load, reinforce, fail and disable | done | `memory.rs`, `session/curation.rs`, `session/run_loop.rs` |
| Router training log (questions, proven labels, teacher answers) | done | `router_training.rs`, `scripts/build_router_dataset.py` |
| Windows Agent Arena suite: parallel VMs, memory modes, per-task data, manifests | done | `scripts/arena/` |
| Hand-back labels (planner resolutions become System 1 labels) | **todo** | `router_training.rs` |
| Direct skill execution (System 1 runs a matched skill without re-planning) | **todo** | `session/run_loop.rs`, `session/fast_actions.rs` |
| Perception in System 1 (actions return a state summary; no "look" turns) | **todo** | `session/fast_actions.rs`, `builtins/capture.rs` |
| Target discovery for grids, unnamed icons, OCR-only labels | **todo** | `grounding.rs`, `builtins/capture.rs` |
| Document-aware code editing and app automation (UNO, COM) | **todo** | `builtins/`, `coding.rs` |
| Protocol mode: 30-step cap counting System 1 actions; no instruction edits | **todo** | `session/run_loop.rs`, `scripts/arena/` |
| Laya fine-tuned on arena data; VM-to-GPU bridge for Laya arms | **todo** | `scripts/arena/`, Laya notebook |
| Arena: WAA Chrome setup for current Chrome (separate debug profile), per-app work-stealing, 5 workers, screenshot-only observations | **todo** | `scripts/arena/` |

## Evidence so far

### Live PC benchmark (before the arena)

Delegated `fast_actions` with a router cut planner calls by about 16–24%
against the no-router arm. Grounding did most of the deciding; Laya was
unreliable at picking 1 of about 30 noisy targets but good at small yes/no and
choice questions. Skills cut repeated Explorer tasks from 17 planner calls to
5–7. Details: `docs/decision-router-backends.md`.

### Windows Agent Arena baseline (`base1`, complete)

The run used an anonymous stealth planner (identity to be recorded when
known), JEV as System 1, memory kept per application, no step cap, a
15-minute limit per task, and 4 VMs. Its full configuration is in the run's
`manifest.json`.

| Measure | Value |
| --- | --- |
| Success | **94/145 runnable tasks (64.8%)**; 60.9% of all 154 if the 9 unrunnable tasks count as failures |
| Estimated success with a 30-step cap | about 59% of runnable tasks (12 passes needed more than 30 acting steps) |
| Unrunnable tasks | 9 Chrome tasks: WAA's setup cannot attach to current Chrome (v136+ ignores remote debugging on the default profile) |
| Planner calls per task | mean 42 |
| Actions performed by System 1 | **15%** |
| `fast_actions` outcomes (669 plans) | done 32%, unverified 21%, no_candidates 25%, uncertain 12%, stalled 5%, budget 4% |
| Skills learned | 62, one per task, zero duplicates, 47 rewritten by the planner, 0 disabled; 97 facts |
| Router training data | 1,819 labeled decisions (1,115 train / 263 test states) |

Per application:

| App | Passed | Median planner calls |
| --- | --- | --- |
| Settings | 5/5 | 11 |
| Paint | 3/3 | 23 |
| Calculator | 3/3 | 32 |
| VS Code | 18/24 | 23 |
| Clock | 3/4 | 19 |
| File Explorer | 14/19 | 5 |
| VLC | 15/21 | 31 |
| Writer | 12/19 | 38 |
| Chrome | 5/8 (+9 unrunnable) | 24 |
| Edge | 8/13 | 13.5 |
| Notepad | 1/2 | 28.5 |
| **Calc** | **7/24** | 50.5 |

Published WAA results for context: human 74.5%, Agent S3 50.2% (single
attempt) and 56.6% (best of 3), Agent S2 about 29.8%, UFO² 27.9%, Navi 19.5%.
They are **not directly comparable** yet. They used different planners, a
step cap, and 2024-era software, and our baseline had memory on and no cap.

**The honest reading:** the harness succeeds often, but System 1 does not yet
carry most of the acting. System 2 still performs 85% of actions, and many
plans hand back before acting. The architecture is right; System 1's
perception and confidence are the bottleneck on unfamiliar applications.
LibreOffice Calc alone accounts for 17 of 51 failures.

## Failure analysis (baseline, first 111 tasks)

| Cause | Tasks | Fix |
| --- | --- | --- |
| Timeout from long UI grinding (mostly spreadsheets) | 12 | grid and keyboard navigation; code for bulk content |
| Claimed success, checker disagreed (edits lost behind an open document, task knowledge) | 11 | "the open document is the truth": edit in the app, or close, edit and reopen |
| Impossible task attempted anyway | 6 | clearer standing instruction to check feasibility first |
| Many target or System 1 hand-backs | 4 | target discovery; trained Laya |
| Wrongly declared impossible | 1 | model |

## The metrics that decide the thesis

Track these on every arena run, against a fresh-memory control:

1. **System 1 action share**: actions done inside `fast_actions` over all
   actions. Target: 15% → 50% and above.
2. **Planner calls per task**, fitted to the power law across passes. Target:
   mean 40 → single digits for practiced tasks. Report $\beta$ and $C_{\min}$.
3. **Success rate** under the standard protocol, which must not fall as
   System 1 takes over.
4. **Hand-back rate by cause** (`no_candidates`, `uncertain`, `stalled`).
5. **System 1 accuracy and calibration by question kind** (target, completion,
   operation, condition), offline on held-out arena questions.
6. **Latency and cost per task.**

## Experiments, in order

Change one thing at a time, so every result has a single explanation.

1. **Finish the baseline**, then build the first dataset
   (`run_arena.py dataset --tag base1`).
2. **Harness fixes** from the failure analysis and the System 1 gaps:
   spreadsheet grids, document-aware code, target discovery, hand-back labels,
   perception in System 1. Add a 30-step budget the agent knows about (with
   steps remaining visible, and a `fast_actions` plan counting as one step).
   Iterate locally with LibreOffice and sample files, then on the arena's 43
   LibreOffice tasks (about an hour per run) until they improve.
3. **v2, a clean run**: the fixed harness, a 30-step cap, empty memory at the
   start with online learning, JEV. It is the new baseline and the
   comparison with published results. Do not reuse `base1` skills; they came
   from the old harness. A fresh-memory-per-task control can run alongside.
4. **Pass 2 (and 3) with v2's skills**: the same harness and JEV, only memory
   changes. The difference is pure self-improvement; fit $C(N)$.
5. **Laya, clean**: fine-tune Laya on `base1` + v2 data with applications held
   out; run a clean arena pass with the same harness and protocol (held-out
   applications first). Compared with v2 this isolates the trained System 1.
   Needs the VM-to-GPU bridge.
6. **Laya + skills**: the full system, compared with pass 2.
7. **Small planner transfer**: a local ~27B planner with fresh memory, with the
   large planner's skills, and with skills plus a trained Laya. Measures how
   much of the planner-size gap skills and System 1 close.
8. **Same-planner harness comparison**: WAA's Navi agent with the same planner,
   to isolate the harness's own contribution.

## Rules that keep the results valid

- Never add rules for a specific task, website, or benchmark fixture; fixes
  must be general (see AGENTS.md).
- Learning uses only the agent's own verification, never the benchmark's
  checker or answers.
- Hold out applications when training System 1 on arena data; never train on
  the test set.
- Change one thing at a time; hold the planner fixed while measuring harness or
  System 1 changes; keep each run's manifest.
- Report pass@1 and the step budget; disclose environment differences (Windows
  build, setup fixes, planner identity).

## Open-source and sponsorship notes

The artifacts others can reuse are the strongest case for recognition:

- **the harness**, with its System 1 + System 2 architecture;
- **the arena suite**, with fixes that make WAA build on current Windows;
- **an open typed-decision dataset for computer use**, generated in clean VMs
  with no personal data;
- **a fine-tuned local System 1 model**;
- **the learning-curve result.**

A one-page pitch should lead with the power-law chart and System 1 action
share, and with the result that a local System 1 matches a hosted one.
