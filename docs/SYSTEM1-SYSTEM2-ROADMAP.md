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
5. a **training loop** that turns the harness's own labeled decisions (proven
   outcomes, the planner's resolutions of hand-backs, and confident answers from
   whichever decision backend was configured) into a local System 1 (Laya), on
   the user's own software.

Absence of evidence is not proof; this claim should be rechecked before any
publication.

## System 1 as the default mode, not a tool

In the brain, System 1 runs continuously and handles almost everything; System
2 is recruited on surprise. POK-Ai so far used System 1 as a tool the planner
calls now and then. The design below maps established models of human skill
onto concrete harness mechanisms.

| Brain principle | Model | Today | To build |
| --- | --- | --- | --- |
| System 1 runs whole motor sequences; System 2 sets goals | chunking, motor programs | System 1 clicks and scrolls; the planner does keys, typing, and most actions (85%) | keyboard and typing steps in `fast_actions` |
| Predictive processing: escalate only on prediction error | error = observed − predicted | any `done_when` mismatch hands back | local System 1 recovery (retry, dismiss popup, scroll into view, re-find target) before escalating |
| Evidence accumulation to a threshold | drift-diffusion | one-shot probability vs 0.90 / 0.80 | accumulate grounding, model, and post-action evidence before deciding |
| Habit strength with practice | $C(N) = C_{\min} + (C_1 - C_{\min}) N^{-\beta}$ | skills replayed through the planner | direct skill execution by System 1; the planner confirms |
| Trust learned from outcomes | Rescorla–Wagner / delta rule, $V \leftarrow V + \alpha(\lambda - V)$ | success/failure counts, fixed thresholds | per-skill trust $V$ that sets how much checking a skill needs |
| Error-driven motor learning (cerebellum) | supervised correction | JEV answers logged | hand-back labels: every planner correction becomes a System 1 example |
| Attention bottleneck | compact summaries | ~10 planner turns per task spent looking | actions return a compact "what changed" summary |

Build order, by expected cut in planner calls: keys and typing in
`fast_actions`; local recovery on prediction errors; compact state summaries;
direct skill execution with trust; hand-back labels; evidence accumulation.
Target: System 1 action share from 15% toward 70–90%.

## What is built

| Area | Status | Where |
| --- | --- | --- |
| Plan trees for System 1 (`fast_actions`: steps, branches, interrupts, reads, avoid) | done | `session/fast_actions.rs`, `builtins/fast_actions_tool.rs` |
| Quoted-label grounding and local completion checks | done | `decision.rs`, `grounding.rs` |
| Typed-decision router (JEV, Laya, LLM choice) with thresholds and hand-back | done | `decision.rs`, `session/decision_router.rs` |
| Cursor-free input (UI Automation patterns first) and background work | done | `builtins/input.rs`, `agent_window.rs` |
| Skills: learn, deduplicate (3 layers), rewrite, load, reinforce, fail and disable | done | `memory.rs`, `session/curation.rs`, `session/run_loop.rs` |
| Router training log (questions, proven labels, local backend answers as soft labels; JEV sessions excluded per TypeSafe's terms) | done | `router_training.rs`, `scripts/build_router_dataset.py` |
| Windows Agent Arena suite: parallel VMs, memory modes, per-task data, manifests | done | `scripts/arena/` |
| System 1 motor skills: typing and key chains the planner writes, right-click, hover, drag between quoted labels | done | `session/fast_actions.rs`, `builtins/targeting.rs` |
| Popup reflex: an unexpected dialog is read (title, text, buttons) and handed back as text | done | `session/fast_actions.rs` |
| Typed-text read-back from the screen when a field does not expose its text | done | `decision.rs`, `session/fast_actions.rs` |
| OCR-only labels (e.g. LibreOffice menus) clickable by an exact quoted label seen once | done | `decision.rs` |
| Hand-back labels (planner resolutions become System 1 labels; unseen targets counted as perception misses) | done | `router_training.rs`, `session/fast_actions.rs`, `scripts/build_router_dataset.py` |
| UI Automation for owned dialogs (LibreOffice dialogs were blind: 14% of `lo7` screens) | done | `pok-ai-windows/src/lib.rs` |
| Arena memory shared like one agent (VM copies merged after each piece and each pass) | done | `scripts/arena/merge_memory.py`, `scripts/arena/run_arena.py` |
| Perception fixes from `v3`: Windows 11 control types (app bar button, toggle button/switch, option, data item, date pickers, header items) were not treated as clickable, so File Explorer's View/New/Sort never reached System 1; labels match without a keyboard-shortcut suffix (\"Extensions\" = \"Extensions (Ctrl+Shift+X)\"); a UI Automation walk that outlasted one capture no longer blinds the next ones. 12 of `v3`'s first 47 quoted-label misses resolve locally with these rules | done | `grounding.rs`, `decision.rs`, `pok-ai-windows/src/lib.rs` |
| Direct skill execution: before the planner's first call, System 1 replays the top skill's motor program when the skill is verified and its trust (s+1)/(s+f+2) is at least 0.6; the planner gets the replayed steps and the screen after them. Replaying only a skill's opening steps (without a program) was tried in v4-v5 and removed: it cost more calls than it saved | done | `session/skill_replay.rs`, `session/run_loop.rs` |
| Motor programs (muscle memory): a verified run with a clean result check saves its exact actions (clicks by label, keys, text the agent wrote; request text only as a fingerprint plus its surrounding words) on its skill; the same or a similar request (40% shared words) replays it end to end in one plan of up to 64 steps, with the new request's values filled into the recorded slots; a leaner program replaces a longer one, one that stops partway is replaced by the run that finished; skills that keep failing lose trust and are switched off. v5: -28% planner calls (median -42%) where a whole program ran | done | `session/motor_program.rs`, `session/skill_replay.rs`, `memory.rs` |
| Learning from failures: a run that did not succeed leaves one general lesson in memory (approved when unattended, a draft for review otherwise) | done | `session/curation.rs` |
| Perception in System 1 (actions return a state summary; no "look" turns) | **todo** | `session/fast_actions.rs`, `builtins/capture.rs` |
| Target discovery for grids and unnamed icons (OCR-only labels done) | **todo** | `grounding.rs`, `builtins/capture.rs` |
| Document-aware code editing and app automation (UNO, COM) | **todo** | `builtins/`, `coding.rs` |
| Protocol mode: 30-step cap counting System 1 actions; no instruction edits | done | `session/run_loop.rs`, `scripts/arena/` |
| Laya fine-tuned on arena data; VM-to-GPU bridge for Laya arms | **todo** | `scripts/arena/`, Laya notebook |
| Arena: 5 workers, even per-app split, work-stealing, clean reset between tasks, scorer crashes counted as failed attempts | done | `scripts/arena/` |
| Arena: WAA Chrome setup for current Chrome (separate debug profile) | **todo** | `scripts/arena/` |

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

### LibreOffice iteration (`lo4`–`lo6`, 43 Calc and Writer tasks)

Same stealth planner and JEV, fresh memory per run, one change set per run.
`lo5` and `lo6` use the 30-step cap; `base1` and `lo4` had none.

| Run | Calc (24) | Writer (19) | Total | Planner calls / task |
| --- | --- | --- | --- | --- |
| `base1` (no cap) | 7 | 12 | 19 (44.2%) | – |
| `lo4` (no cap) | 11 | 7 | 18 (41.9%) | 45.3 |
| `lo5` (cap 30, System 1 typing) | 7 | 9 | 16 (37.2%) | 42.3 |
| `lo6` (cap 30, fixes below) | 8 | 15 | **23 (53.5%)** | 40.1 |

What changed from `lo5` to `lo6`: OCR-only menu labels, the popup reflex,
keeping the application's own dialogs instead of switching back, foreground
mode in the VM, System 1 read-back after typing, hover/right-click/drag,
working in the file the user has open, and a fresh look when the repeat-input
guard blocks System 1.

Harness and environment faults these runs exposed (each fixed):

- LibreOffice's updater starts once a day even with its servers blocked,
  holding or restarting LibreOffice without the task's file; the reset now
  removes it. Runs after the golden image's first day would otherwise be
  invalid.
- "Replace existing" typing fell back to Ctrl+A and Delete, which in a
  spreadsheet erases the whole sheet (285 such requests in `lo6` Calc). It now
  only selects all in controls that expose their text. Expected to lift Calc
  in `lo7`.
- A same-application window (a dialog the agent opened) was treated as an
  outside change and switched away from, discarding a planner turn each time.
- Background mode returned focus to the launching console after each input,
  which closes LibreOffice's open menus.
- WAA's scorer can crash on an agent's result (cell colors); that attempt now
  counts as a failure and is never rerun.

### System 1 perception in LibreOffice (found after `lo8`)

System 1 has no vision; it sees UI Automation labels and OCR. Probing the
arena VMs showed that LibreOffice 24.8 exposed only its title bar to UI
Automation (6 elements, against 185 on a desktop running 26.8), and every
dialog was blind (0 elements: owned windows were only searched at the top
level). In `lo4`–`lo8` System 1 therefore worked from OCR alone in
LibreOffice. Fixes:

- Owned dialogs are found under their owner window (Find and Replace now 35–54
  elements, Format Cells 83).
- UI Automation is walked level by level and never enters a table or data
  grid, and typing verification never reads a focused table's text: a
  spreadsheet sheet exposed as one table otherwise hangs the application.
- LibreOffice 24.8 builds its tree only under Windows' screen-reader flag, and
  its Calc then froze while cells were edited. The golden image was updated to
  LibreOffice 26.8.0 (`setup_waa.py update-libreoffice`), which exposes the
  tree without the flag; the previous image is kept. Runs record the image's
  LibreOffice version in `manifest.json` (`golden_image`). This is an
  environment change against published WAA results, which used an older
  LibreOffice; WAA's checkers read the saved files, so scoring is unchanged.

First test on 26.8 (3 tasks): no freezes, no blind dialogs, System 1 action
share 37%, "no candidates" hand-backs 19% of plans (40–58% before).

### Full clean run (`v2`, 154 tasks, 2 passes)

Protocol: the fixed harness, a 30-step cap, memory empty at the start with
online learning (VM copies merged as one agent), JEV, LibreOffice 26.8 image,
5 VMs. The stealth planner is unchanged.

| Measure | `base1` | `v2` pass 1 | `v2` pass 2 |
| --- | --- | --- | --- |
| Success (of 154) | 95 (61.7%, 145 runnable, no cap) | **93 (60.4%, all runnable, cap 30)** | 91 (59.1%) |
| Planner calls per task (same tasks) | 28.1 | **22.7** | 24.5 (of 25.2 in pass 1) |
| Time per task | 236 s | 308 s | 313 s (of 344 s) |

- Pass 1 matches the baseline's success under a 30-step cap with 19% fewer
  planner calls, and runs all 154 tasks (the 9 Chrome tasks WAA could not set
  up before now run; 4 pass).
- Three VS Code failures are WAA checker crashes: the agent wrote
  `settings.json` with Windows PowerShell 5.1, whose UTF-8 output carries a
  byte-order mark the checker's `json.load` rejects. Fixed for the next run:
  `run_command` tells the planner to write text files as UTF-8 without a BOM.
- Pass 2 barely improved (10 tasks gained, 12 lost; calls -2%, time -9%).
  Tasks that loaded a learned skill passed slightly more (54 -> 56) but used
  no fewer calls: skills are advisory text, and the planner still plans every
  step. Failed pass-1 tasks teach nothing. The small cross-app check (short
  tasks) showed -12% calls and -47% time; at full scale advisory skills are
  not enough. **Next: direct skill execution by System 1** (replay a matched
  skill's steps, hand back only on a mismatch, trust per skill), and learning
  from failures.
- Why v2 took longer per task (monotonic timings, 108 paired tasks; VM clock
  jumps make wall-clock differences unreliable): of +54 s, about 40 s is the
  planner model itself generating more slowly (12.2 -> 15.3 ms per output token,
  median first token 2.5 -> 3.3 s; same model and reasoning setting, fewer and
  smaller calls). The other ~13 s is the harness doing more per call: batches of
  4.35 steps instead of 3 with 3.5x the key presses, System 1 plans, and a UI
  Automation walk of ~0.5 s per capture instead of ~0.2 s. Compare time only
  within one run, or with provider speed factored out.

### Skill replay at full scale (`v4`, 154 tasks, 1 pass from v2's memory)

v4 starts from v2's final memory and adds skill replay by System 1, Windows 11
control types, label matching without keyboard-shortcut suffixes, replay that
clicks only saved labels, and waiting for an application System 1 opened. It is
compared with v2 pass 2, which also started from learned memory.

| Measure | v2 pass 2 | v4 |
| --- | --- | --- |
| Success (of 154) | 89.7 (58.3%) | 88.7 (57.6%): 11 lost, 10 gained |
| Planner calls per task (mean / median) | 24.5 / 21 | 27.3 / 26.5 |
| System 1 action share | 36% | 38% |
| System 1 plans with no target | 17% | 12% |

- System 1 got more accurate (exact-label clicks up about 55%, no-target
  hand-backs down), and Calc improved (7 vs 5), but planner calls rose. The
  extra calls are looks: after a `fast_actions` result without the new screen
  (a plan whose conditions already held), the planner captured again 67% of
  the time, 91% after reads without an image, and 24% when the screen came back.
  56 plans were rejected for format mistakes, and 46% of plans had one step.
- Fixed for the next run: every System 1 result returns the current screen
  (a no-op says so), plan-format repairs (lists in lists, missing goal or
  done_when, empty entries, plain strings as keys), input sequences up to 40
  entries, and planner guidance to send every foreseeable step in one plan.
- Smoke tests on File Explorer, Settings and Clock (28 tasks, same memory):
  24 vs 22 passed; replays finished whole sequences (3/3, 3/3, 5/5 steps).
  Windows Defender blocked new unsigned builds inside the test VM; the VM
  launcher now excludes the agent's folder.

### Muscle memory at full scale (`v5`, 154 tasks, 2 passes from v2's memory)

Verified runs save their exact actions as a program on the skill; in pass 2
System 1 replays the program before the planner is asked. Golden image: Clock
updated, OneDrive prompts off.

| Measure | v2 pass 2 | v4 | v5 pass 1 | v5 pass 2 |
| --- | --- | --- | --- | --- |
| Success (of 154) | 89.7 | 88.7 | 87.8 | 88.7 |
| Planner calls per task (mean / median) | 24.5 / 21 | 27.3 / 26.5 | 23.0 / 22 | 23.0 / 22 |
| System 1 action share | 36% | 38% | 45% | 49% |

Pass 2 by what System 1 did (same tasks, pass 1 -> pass 2):

| Group | Tasks | Success | Calls (mean) | Calls (median) |
| --- | --- | --- | --- | --- |
| Whole program replayed | 26 | 18 -> 17 | 20.6 -> 14.8 (-28%) | 15.5 -> 9 (-42%) |
| Any program replay | 48 | 33 -> 31 | 19.6 -> 16.6 | 14 -> 14.5 |
| No replay | 88 | 41.8 -> 45.8 | 19.7 -> 20.7 | 16.5 -> 20 |
| Opening-step replay (no program) | 18 | 13 -> 11.9 | 24.1 -> 28.3 | 22 -> 33.5 |

- Muscle memory works where a whole program exists; too few tasks had one
  (41 of 154 matched no skill, 30 matched a skill without a program).
- Opening-step replays cost calls: turned off after v5; only programs replay.
- Similar requests now replay too (40% word overlap), with the new request's
  values filled into the recorded slots by their surrounding words.
- Of 52 programs saved in pass 1, 15 came from runs the checker failed; only 4
  of those showed a warning. Programs are now saved only when the result
  check passed without recovery or accepted warnings; per-skill trust covers
  the rest.

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
5. **Laya, clean**: fine-tune Laya on data from runs with a local System 1
   (not the JEV runs: TypeSafe's terms, section 2.3(b), forbid using JEV to
   develop a competing model) with applications held out; run a clean arena pass with the same harness and protocol (held-out
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
