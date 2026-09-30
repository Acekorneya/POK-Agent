# Adding a decision-router backend

The decision router (`crates/pok-ai-core/src/decision.rs`) is provider-neutral
by design: `TypeSafeDecisionRouter` speaks one bounded HTTP/JSON contract —
`DecisionRequest` in, `ApiResponse` out, against a single `POST
/v1/systemone` endpoint (plus `GET /health`) — regardless of what model is
actually running behind it. Both existing backends, hosted JEV and local
Laya, are this same contract; Laya's side of it is a small Python process
(`scripts/laya_sidecar.py`) that translates the contract into calls against
the `laya` package.

New small "System 1" decision models keep appearing (Laya itself, Zeiger,
OpenJev, and others). Adding one to this harness should mean writing one
small adapter file, not touching session/policy/tool code.

## Delegated mode: the primary model plans, the router executes

`decision_router.mode` selects how the router is used. The default,
`delegated`, gives the primary model a `fast_actions` tool:

```json
{
  "goal": "open System settings",
  "target_hint": "list item \"System\"",
  "done_when": "the System page is shown",
  "allowed_operations": ["click"],
  "then": [{"goal": "open Display", "target_hint": "\"Display\"", "done_when": "the Display page is shown"}]
}
```

The primary model owns planning (which operation, in what order, what
"done" looks like). Each node of the plan runs in `run_fast_subgoal`
(`session/fast_actions.rs`):

1. **Completion check.** `DecisionRouter::check_condition` asks one
   yes/no/unknown `choice` question about `done_when` against evidence from
   only the surface being acted on (browser URL/title/text, or desktop labels
   ranked by the condition), because small routers have short contexts.
2. **Candidates.** Only the allowed operations. The default is click-only,
   so the router is never asked the weak "which operation" planning question.
   Candidates are re-ranked by `hint_relevance` against the target hint.
3. **Pick.** In order: a single candidate is executed directly; an exact
   label the planner quoted (`exact_quoted_match`) or an unambiguous hint
   match is executed directly; otherwise the router picks, and the pick must
   clear the target gate **and** not be contradicted by local label evidence
   (a pick scoring under half the best local match is handed back).
4. **Execution** goes through the ordinary registry (policy, approval,
   freshness) and the continuity ledger. After the named target is clicked,
   the node returns `unverified` instead of letting the router pick more
   targets.

The whole plan returns as one tool result with per-node status (`done`,
`unverified`, `uncertain`, `stalled`, `no_candidates`, `budget_exhausted`),
the executed steps, and the newest observation. When the router is enabled,
raw navigation tools (`click_target`, `scroll_view`, `activate_window`,
`managed_browser_click`, `managed_browser_scroll`) are withheld until a node
hands back without completing, and then only for two primary-model turns.
`router_first` keeps the original loop for comparison.

Every run now records `primary_model_requests`, `primary_model_prompt_tokens`
and `primary_model_completion_tokens`, which count every primary-model call
(agent turns, bounded helpers, compaction). That is the number the router is
meant to reduce.

### Live comparison (2026-09-22/23, native Windows, n=1 per cell)

The primary model was `deepseek/deepseek-v4.1-flash` on OpenRouter in every
arm. The tasks were click-only navigation: the Rust Book table of contents in
the managed browser, and File Explorer This PC > C: > Program Files. The
table shows primary-model requests, with prompt tokens in parentheses.

| Arm | Router | Book | Program Files |
|---|---|---|---|
| A | none | failed (looped past 170 turns) | 31 (624k) |
| B | router_first, Laya + judge | 7 (163k) | 13 (251k) |
| C | router_first, JEV | 2 (39k) | 17 (308k) |
| D | delegated, Laya | 8 (180k) | 22 (458k) |
| E | delegated, Laya + judge | 4 (84k) | 14 (306k) |
| F | delegated, JEV | 4 (84k) | 31 (660k) |

A Settings > Display > Advanced display task was also run. It is excluded
because the B, D, E and F runs aborted when the Windows Search panel took the
foreground and environment recovery failed fatally (see "Known issues").

What the traces showed, beyond the totals:

- **Laya's own target picks were unreliable on real screens.** On Settings
  it gave "Windows spotlight, dynamic images" 0.87 and the named "System"
  item 0.07. On the Rust Book it picked the site-title link at 1.0 over and
  over. In delegated mode the successful clicks came from the planner's
  quoted labels and the local agreement check, not from Laya.
- **The judge was almost never needed in delegated mode.** Arm E made no
  judge calls on either task. Exact-label and unambiguous matches cover the
  picks the judge used to rescue, so it can stay off by default.
- **The router clearly helps on the browser task.** Without it the primary
  model looped. Every router arm finished in 2–8 requests.
- **Explorer results were mixed.** n=1 is noisy, and the gap between arms is
  within what one retry by the primary model can cause.

### Bug found during live testing: router click candidates never worked

`desktop_candidates_for_task` built `observation_id` as `obs:<uuid>`, but
input tools accept only `obs_<first 8 hex>`. As a result every router-selected
desktop click failed as a stale observation, in `router_first` mode as well,
which accounts for much of the fallback and handoff volume in earlier live
traces. Candidates now use `builtins::observation_id_for`, and a regression
assertion covers it.

### Known issues not addressed here

- A Start/Search panel that takes the foreground mid-run can make environment
  recovery fail and end the run with `run_failed`, instead of recovering or
  handing back.
- The primary model occasionally emits tool-call markup as plain text with
  `finish_reason: stop`, and the session accepts it as the final answer.

## The contract

Request body:

```json
{
  "model": "<the caller's expected model name, echoed back>",
  "state": { "...": "free-form JSON context" },
  "questions": {
    "<question_id>": {
      "type": "choice",
      "instructions": "...",
      "criteria": { "<option_id>": "<option description>", "...": "..." }
    }
  }
}
```

Response body:

```json
{
  "model": "<the model name that actually answered>",
  "answers": {
    "<question_id>": {
      "type": "choice",
      "choice": "<option_id>",
      "confidence": 0.0,
      "probabilities": { "<option_id>": 0.0 }
    }
  },
  "sidecar": { "...": "optional, informational only" }
}
```

`model` is required and is checked against the configured model name
(the `model_mismatch` gate in `session/decision_router.rs`) — the adapter must echo back a model
identifier that matches what's configured in `pok-ai.toml`, not a hardcoded
literal, or every decision will be silently rejected as ineligible. See
`crates/pok-ai-core/src/config.rs::LayaRouterConfig.model` for the pattern:
thread the configured value through rather than hardcoding a string in the
adapter.

## Steps to add a backend

1. **Write the adapter.** A small FastAPI (or any HTTP framework) process
   exposing `/health` and `/v1/systemone` exactly as above, with the same
   loopback-only bearer-token auth pattern (`scripts/laya_sidecar.py` and
   `scripts/zeiger_sidecar.py` are both complete, working reference
   implementations — the Zeiger one is close to a template: same imports,
   same `authenticate()`/`/health`/`/v1/systemone` shape, only the model
   load call and the `decide`/`predict` call differ). This is the only
   piece of work that's genuinely model-specific:
   - If the model already answers typed `choice`/`noul`/`score` questions
     over `{option_id: description}` criteria (Laya, Zeiger, and likely any
     future model in that family), the adapter is a near-direct passthrough.
   - If the model solves a structurally different task — e.g.
     `AlexWortega/openjev` is an NLI cross-encoder over premise/hypothesis
     pairs, not a typed-choice model — the adapter must translate each
     `choice` question into that model's native shape (e.g. one
     entailment pass per candidate option, normalized into a
     `probabilities` map) before it can satisfy this contract. That's real,
     separate design work, not a config change. `scripts/openjev_sidecar.py`
     is a complete example of this harder case: each `choice` question
     becomes one `OpenJevCrossEncoder.predict_hypotheses(premise,
     hypotheses)` call (one shared-premise, many-hypotheses NLI pass), with
     per-option P(entailment) renormalized across options into
     `probabilities` (raw entailment scores are independent per-hypothesis
     3-way softmaxes, not already comparable). OpenJev's checkpoint ships
     `modeling_openjev.py` alongside the weights rather than as a pip
     package, so the adapter needs it on `PYTHONPATH` (or copied next to the
     checkpoint) rather than just `pip install`-ing something.
2. **Add a config section.** Mirror `LayaRouterConfig` in `config.rs`: an
   `endpoint`, `model`, `token_env`, and whatever backend-specific knobs
   matter (device preference, VRAM floor, etc.).
3. **Add a `DecisionRouterBackend` enum variant** and wire it into
   `DecisionRouterConfig::active_endpoint`/`active_model`/`active_key_env`
   (`config.rs`) so `TypeSafeDecisionRouter` picks the right sub-config.
4. **Wire process lifecycle in the Tauri app**, if it needs one (spawn,
   health-poll, stop) — mirror `ensure_laya_started`/`stop_laya` in
   `apps/desktop/src-tauri/src/lib.rs`.

No changes to the session's decision loop, eligibility gates, or policy are
needed — those already work identically regardless of backend.

## Benchmarking a candidate before adopting it

`scripts/bench_decision_models.py` runs every case in
`diagnostics/decision_suite_cases.json` against any running sidecar's
`/v1/systemone` endpoint and reports accuracy against each case's expected
target plus latency. It doesn't know or care which model is behind the
endpoint — point it at a new adapter the same way you would the existing
ones:

```bash
python scripts/bench_decision_models.py \
    --endpoint http://127.0.0.1:PORT/v1/systemone \
    --token <the sidecar's bearer token> \
    --label "candidate-model-name"
```

This sends the real two-stage decomposition `decide_once` builds for live
traffic — one `operation` choice question plus one `target_<OPERATION>`
choice question per operation group — not a flat single question. An earlier
version used a flat question and never exercised the actual
operation-probability gate the session checks in production, which is why
early checkpoint comparisons missed the real clustering-band bottleneck (see
below). For cases carrying a `clustering_band` block, it also asks a
3-question committee per operation group and tests several combined
operation+target probability gate formulas — see "Decision eligibility"
below for what came out of that.

### Reference results (8-case suite, RTX 3090, 2026-09-21)

| Model | License | Accuracy | Avg latency |
|---|---|---|---|
| `Luni/laya-grounded` (current default) | **CC-BY-NC-4.0 (non-commercial)** | 8/8 (100%) | 61ms |
| `convaiinnovations/laya-typed-decisions` (prior default) | Apache-2.0 | 7/8 (87.5%) | 72ms |
| `php-ai/zeiger-0.6b` (via `zeiger_sidecar.py`) | CC-BY-SA-4.0 (weights) | 7/8 (87.5%) | 79ms |
| `AlexWortega/openjev` 0.8B (`qwen3.5-0.8b-nli-v2s-long`, via `openjev_sidecar.py`) | MIT | 6/8 (75.0%) | 283ms |

OpenJev is included as the reference implementation of the harder "translate
a structurally different model into our contract" case (see above), not as a
recommendation — on this harness's own benchmark it's both the least
accurate and ~4-5x slower than every typed-choice model tested, running
without the architecture's optimized kernels (`causal_conv1d`,
`flash-linear-attention`) installed. It's a decoder-based NLI cross-encoder
doing one full forward pass per candidate option rather than one shared
forward pass scoring all options at once, which is the more fundamental
reason it's slower, not just a missing-kernel gap. Worth re-testing if a
smaller OpenJev checkpoint appears or the missing kernels get installed, but
it doesn't beat the current default today.

`laya-grounded` beat the typed-decisions specialist checkpoint on this
harness's own benchmark, including the hardest case (a 26-candidate click
list) that the prior default got wrong, which is why it's the default now.
POK-Ai is open source and this checkpoint's non-commercial license is
accepted for that use; anyone building a commercial product on top of this
harness needs to swap it for `convaiinnovations/laya-typed-decisions` or
another permissively-licensed checkpoint first. 8 cases is directionally
useful, not statistically definitive; re-run and grow
`decision_suite_cases.json` before treating a close result as decisive.

## Decision eligibility: structural bypass and the combined probability gate

`try_decision_router_action` (`session/decision_router.rs`) gates every ordinary (non-DONE/
BLOCKED) decision before letting it execute directly. Two changes came out
of testing the router's calibration on real and synthetic clustering-band
cases (`decision_suite_cases.json`'s `case_09`-`case_20`, `should_promote`
labels, `operation_group_size`) rather than tuning thresholds by feel:

1. **Structural bypass** (`structural_bypass_candidate` in `session/decision_router.rs`).
   When exactly one non-terminal candidate is offered, no judgment call
   exists — whatever generated the candidate set already decided it's the
   only reversible option — so the router is skipped entirely: no HTTP call,
   no probability gate, execute directly. This is safe by construction for
   genuinely-forced cases (e.g. `list_windows` before any observation
   exists), but its safety depends on candidate-generation precision, not
   router calibration: if the generator itself offers a single wrong
   candidate, the bypass executes it without any second check
   (`case_15_synthetic_forced_single_evidence_wrong` in the suite is exactly
   this scenario, kept as a documented, accepted limitation rather than a
   bug — it's also a pre-existing gap in the old independent gate, not a new
   risk this bypass introduces).

2. **Combined operation+target probability gate**
   (`LayaRouterConfig::min_operation_probability`/`min_target_probability`,
   now 0.35/0.90). The old independent gate (0.45/0.50, both required
   separately) deferred essentially 100% of real clustering-band decisions
   to the LLM even when they were later confirmed correct, because Laya's
   operation stage runs conservative (0.35-0.69 observed) while its target
   stage is often near-certain (1.0) on the exact same correct decisions.
   Before changing the numbers, a **3-question speculative-committee
   escalation** (borrowed from the `jev-x-kit` project's "BELKİ" gatekeeper —
   independent yes/no/unknown sub-questions on reversibility/cost/evidence,
   promoting only on unanimous agreement) was tried first and rejected: on a
   12-case real+synthetic suite it scored 58.3%, barely above chance, and
   *worse than chance (40%, n=5)* specifically on single-candidate scenes —
   the opposite of what a judgment mechanism should do on its easiest cases.
   The combined gate tested instead reuses `operation_probability` and
   `target_probability` that `decide_once` already returns — no new
   questions, no extra HTTP round trip — and scored 75% (9/12) on the same
   suite, with defensible failure modes (a documented pre-existing blind
   spot on single-wrong-candidate scenes, and appropriately cautious misses
   on genuine 3-way ties with low target confidence). The threshold shape
   itself didn't change (`operation_probability >= X AND target_probability
   >= Y`), only the calibration constants — `min_operation_probability`
   lowered so target confidence can compensate a clustering-band operation
   score, `min_target_probability` raised correspondingly since it now
   carries more of the discriminative weight.

Both changes are Laya-specific; JEV's independently well-calibrated
`operation_confidence`/`target_confidence` outputs did not show the same
clustering pattern in earlier live A/B testing and were not touched.

n=12 is still a small sample. Re-validate with `bench_decision_models.py`'s
`COMBINED_GATE_FORMULAS` section against a larger case set before trusting a
close result, the same way the committee's initial n=4 pass (75%) reversed
to a clear fail at n=12 (58.3%, worse than chance on the largest real
category).

## Cross-model judge: a different backend judging Laya's pick

Comparing four backends on the same 20-case suite (Laya, `AlexWortega/openjev`
0.8B, hosted JEV, `php-ai/zeiger-0.6b`) surfaced a pattern: raw two-stage
pick accuracy and self-judgment quality are independent axes, not the same
competency measured twice. Zeiger has the worst raw pick accuracy of the
four (65%) but the best self-committee correlation (91.7%); Laya is the
opposite (75% raw accuracy, 58.3% self-committee, worse than chance on
single-candidate scenes). This matches recent "semantic cascade" research
(cross-model agreement is a more reliable escalation signal than a single
model's own self-reported confidence) closely enough that it was tested
directly with `scripts/bench_cross_model_judge.py`: one backend (the
picker) makes the two-stage choice, a *different* backend (the judge)
answers a 3-question committee scoped to that specific picked candidate.

### Live-run failure and its root cause: un-pinned pre-alpha weights

A live Windows session with the judge enabled produced **37 judge attempts
and 0 promotions** (every attempt `error: null`, every vote withheld). The
plumbing was fine; the judge payloads were the standard
reversible/cheap/evidenced questions with the candidate description (see
`judge_candidate_once` in `decision.rs`). Re-running the identical
`bench_cross_model_judge.py` against the *installed* sidecar collapsed from
the documented 90% to **40%**, with zeiger answering `unknown` to the
reversible and cheap questions on almost every case.

Root cause: `php-ai/zeiger-0.6b` is **pre-alpha weights "live on the Hub"**
and the Hub's `main` was updated to a "Round 15: better calibrated on every
kind of question" commit on 2026-09-21 — hours after the 90% validation ran
against the previous revision. The Round-15 weights behave very
differently on this harness's yes/no/unknown safety questions (near-universal
`unknown`), which made the judge inert in production. The pre-Round-15
weights (commit `441dcf64aef0`) reproduce the validated behavior:

| Cascade (20-case suite, 2026-09-21 replay) | Pick acc. | Judge 3Q | Judge 4Q |
|---|---|---|---|
| Laya → zeiger **Round 15** (unpinned `main`) | 60% | 40% | 40% |
| Laya → zeiger pre-Round-15 (`441dcf64aef0`) | 60% | 80% | 85% |
| **laya-typed-decisions (Apache) → zeiger pre-Round-15** | **75%** | **90%** | **90%** |
| Laya → hosted JEV | 60% | 90% | 90% |
| JEV → zeiger pre-Round-15 | 90% | 85% | 85% |
| JEV → JEV (self) | 90% | 83% | 83% |

(Note: the same replay also found raw pick accuracy varies sharply across
Laya's available checkpoints — 75% for the Apache-2.0 typed-decisions
specialist (now the default), 60% for `laya-grounded`, 55% for the app's
ModernBERT `english` fallback — while hosted JEV still picks 90%. The
picker, not the judge, is the current bottleneck for the "fewer LLM tool
calls" goal, and the picker choice is now documented in the comparison
table below.)

**The fix is checkpoint pinning.** `setup-zeiger.ps1` now takes a
`-Revision` parameter and defaults to `441dcf64aef0` (the validated
pre-Round-15 weights); the installed revision is recorded in
`manifest.json`. Reinstall the judge with the pinned revision before
trusting it: `powershell -File scripts/setup-zeiger.ps1 -DataDir <dir> -Revision 441dcf64aef0`.
The desktop app's `install_judge` command passes the parameter through.
`setup-laya.ps1` gained the same `-Revision` parameter (default `main`) for
consistency. If a future validation says a newer revision is better, change
the default *and* re-run `bench_cross_model_judge.py` before shipping it.

### Comparative 4th question (wired into production)

`bench_cross_model_judge.py` also tests a 4th **comparative** question
(picked candidate vs. the best runner-up in the same operation family).
It is the local judges' strongest competency (7/7, 100% on the bench's
objectively determinable cases) and it vetoes plausible-but-wrong picks
that the three absolute questions miss (e.g. activating the wrong window
in a same-operation group): it raised zeiger's correlation 80% → 85%.

Production now mirrors this: `judge_candidate` receives the full candidate
list, `judge_runner_up` picks the highest-local-score alternative in the
same operation family, and when one exists a `comparative` question
(`a` = picked, `b` = runner-up) is asked alongside the absolute three.
Promotion requires unanimous "yes" on the absolute questions **and** the
judge not preferring the runner-up (`comparative != "b"`). The question is
only ever a veto; it is never asked when no runner-up exists. The verdict
(`JudgeVerdict`: `promoted`, per-question `votes`, `confidences`,
`probabilities`) is logged under `decision_router_judge_attempt` together
with the bounded task/current_step/candidate description, so any live
session is replayable offline via `scripts/replay_live_judge_payloads.py`.

Replaying the original 37 live candidates against the pinned zeiger:
**4/37 would have promoted** (the `Example Guild, Voice call active`
evidence candidates and a channel click) — the mechanism works on real
desktop candidates; the unanimous bar stays conservative by design.

### Production must match the validated judge formatting

The judge question templates substitute the candidate description the same way
`bench_cross_model_judge.py` does (Python's `{description!r}`): single-quoted
text with embedded double quotes untouched (`quote_description` in
`decision.rs`). Two earlier production/bench mismatches were found and fixed
while chasing live browser-click holds:

1. Unquoted substitution ran the raw description into the question sentence.
2. Rust's `{:?}` quoting produced backslash-escaped inner quotes where the
   bench produced a clean single-quoted string.

Zeiger is deterministic but hypersensitive to this formatting: the identical
payload voted `reversible: unknown` with the escaped form and `yes/yes/yes`
with the bench form in direct sidecar A/B tests. The regression test
`judge_quotes_the_candidate_description_in_questions` locks the bench format in.

Browser interaction candidates (`managed_browser_click`/`scroll`/`hover`) also
carry a factual `action_class` note ("in-page navigation: ... can be undone
with the browser Back action"), because live traces showed browser link clicks
held solely on `reversible: unknown` while cheap/evidenced/comparative were all
positive. The note is scoped to browser interactions only — the offline suite
contains no such candidates, so its validated 90% correlation is structurally
unaffected (re-verified). Even so, the model remains conservative on long
ambiguous card links; when the judge holds, the primary model performs the
in-page navigation itself and the run still completes correctly.

### Live head-to-head: local stack vs hosted JEV (2026-09-22)

The same real Discord task ("who is in the Example Guild Voice Channels")
run natively on the same Windows desktop, three configurations:

| Configuration | Turns | Tool calls | Prompt tokens | Router handoffs |
|---|---|---|---|---|
| Local stack before loop fixes (laya-typed picker + pinned zeiger judge) | 24 | 44 | 314k | 24 |
| Hosted JEV (`jev-1.13.0`, same config shape) | 3 | 8 | 65k | 3 |
| **Local stack after loop fixes** (2 runs) | **2–3** | **4** | **35–36k** | **2–3** |

The pre-fix local run was dominated by loop waste: 42 router evaluations,
17 judge calls (the same `click_8` candidate re-judged six times with
identical votes), 9 environment recoveries, and 24 handoffs. Two changes
closed most of the gap to JEV without touching the picker:

1. **Judge verdict memoization.** `judge_candidate` verdicts are cached per
   (task, current_step, candidate fingerprint) in the session. A repeated
   candidate across re-observations — the same window or channel re-proposed
   after a capture that did not change the state — is answered from the
   cache instead of burning another sidecar round trip; the trace logs
   `reused: true`. The promoted verdict also means the second proposal
   executes without re-judging.
2. **Judge state enrichment.** The judge payload now carries the same
   bounded NextAction state the picker saw (`window`, `browser`,
   `last_action`, `previous_outcome`, `state_revision`, string-bounded via
   `bounded_judge_state`), so the reversible/cheap/evidenced questions are
   answerable against real context instead of only the candidate
   description. Live, the promoted picks are the same ones the bench
   predicts; the judge still holds window activations it cannot verify as
   reversible — the safe failure mode.

Two caveats from testing: the bench suite's stored `state` fields are
pre-observation snapshots (e.g. "locate the Discord window" with
`foreground_window: explorer.exe`), so feeding them to the judge *hurts*
bench correlation (90% → 80%) — context only helps when it is consistent
with the candidate, which production state is by construction. And the
elapsed-time comparison is noisy (provider latency varies run to run); the
stable signal is turns/tool calls/tokens, where the fixed local stack is
now at or slightly better than hosted JEV on this task. JEV remains the
higher-accuracy picker (90% vs 75% raw on the 20-case suite); the local
stack's edge here comes from the judge rescuing Laya's borderline picks
and the loop fixes removing redundant work. n=1-2 runs per configuration;
directionally useful, not statistical proof.

Configured under `[decision_router.judge]` (`enabled`, `endpoint`, `model`,
`token_env`, `timeout_ms`), disabled by default and validated to a loopback
HTTP endpoint the same way Laya's is. The desktop app installs and manages
the judge sidecar (`install_judge`/`ensure_judge_started` in `lib.rs` via
`setup-zeiger.ps1` + `zeiger_sidecar.py`); the CLI and the bench scripts
talk to a manually started sidecar. n=20 is still a modest sample; treat
the exact correlation numbers as directionally useful, not statistical
proof.

## Context-selection questions use `choice`, not `noul`

`decide_once`'s generic per-candidate branch (used by
`DecisionPurpose::ContextSelection` and `EnvironmentRecovery`, e.g.
`route_optional_context` in `session/decision_router.rs` for workspace-code/session-archive/
memory context selection) asks a `choice` (relevant/irrelevant) question per
candidate rather than a raw `noul` probability. This changed after testing
found `noul` degenerate across every backend tried on this kind of
independent per-item relevance judgment (Laya and zeiger either approved
everything or approved nothing), while the identical judgment as a `choice`
question recovered real, well-calibrated behavior. The "relevant" option's
probability is parsed back into the same per-candidate score shape existing
consumers already expect (falling back to a plain 1.0/0.0 from the bare
choice if a backend omits per-option probabilities), so no downstream
threshold/ranking logic needed to change.

This was tested and validated specifically for a *new*, not-yet-wired-in
use case (filtering the desktop-observation target list before it reaches
the model — see below), but applies to the already-shipped context-selection
call sites too, since `noul` underperformed `choice` for every backend
tested, not just on that one experiment.

### Desktop-observation target filtering: validated idea, not yet wired in

Comparing the current static heuristic (`grounding.rs`'s `task_terms`/
`target_rank`, literal keyword-substring overlap) against Laya, zeiger, their
combination, and JEV on 5 synthetic busy scenes (Discord's 26-option
channel list, a checkout page, etc. — `diagnostics/context_gating_suite_cases.json`)
found JEV's `choice`-format relevance judgment clearly ahead: 0.80 precision
and 0.77 recall vs. the heuristic's 0.31/0.70, at half the token cost (9%
of targets kept vs. 18%). Laya and zeiger did not beat the heuristic (Laya's
`choice` answers degenerated to approving everything; zeiger's recall was
worse than the free heuristic).

This is a real result but was **not wired into `grounding.rs`/`builtins/`**.
Tracing the actual render path (`model_observation_value`,
`relevant_targets`, `sparse_annotation_target_ids`) surfaced real complexity
(screenshot-annotation sparsity and click-target id resolution both depend
on the same target list) and a behavioral risk the 5 single-step synthetic
scenes cannot rule out: a relevance filter tuned to *this* step could hide a
target a later step of the same multi-step task needs, with no fallback
path today. This needs live-capture, multi-step validation — and probably a
re-observe/expand fallback design — before it's safe to ship, not just a
same-day synthetic pass.

## Collecting training data from real use

Fine-tuning a Laya-type model on the harness's own questions is the most
direct way to improve System 1 decisions: the Laya authors report the base
checkpoint at ~0.36 on a new question type and ~0.77 after fine-tuning on
in-domain questions. The harness can build that dataset while you use it.

Turn on **Settings > Decision router > Training data** (or
`decision_router.training_log = true`). Each conversation then appends to
`<data_dir>/router-training/<session>.jsonl`:

- `question` records: the exact `/v1/systemone` body in Laya's typed shape,
  `{state, questions}`, after the router's outbound sanitization. Real router
  questions include the backend's answers. Decisions local grounding made are
  recorded as the question the router *would* have been asked
  (`source: "grounding"`).
- `label` records: the proven answer for one question of a record:
  - `grounding`: a quoted `done_when` checked against screen evidence;
  - `quoted_label_match`: the target a quoted hint named exactly;
  - `local_evidence`: the grounded best target when a router pick
    contradicted clear label evidence.

Only on-screen decisions are recorded (target picks, completion checks,
branch choices), not intent or context routing, which carry prompts and
history. Windows matching `training_exclude` (title or process) and any
record containing an email address or a run of six or more digits are
skipped. Nothing leaves the machine.

Build a dataset from any number of sessions and machines:

```bash
python scripts/build_router_dataset.py --output dataset/ \
  --input "%LOCALAPPDATA%/POK-Ai/POK-Ai/data/router-training" --input other-pc/
```

Windows Agent Arena runs (`scripts/arena/`) are the other source. Their
tasks run in a clean VM, so the records hold no personal data. Each session
is joined with its task's automatic score (`run_arena.py dataset`), and
backend answers can be limited to tasks that passed
(`--outcomes … --teacher-successful-only`).

**Terms.** TypeSafe's Master Customer Agreement (section 2.3(b)) forbids using
hosted JEV or its output to distill, train a model that imitates it, or develop
a competing product. The dataset builder therefore refuses JEV as a teacher and
leaves out sessions that used JEV (`--include-hosted-sessions` keeps them, only
with TypeSafe's written permission). Collect training data with a local backend
(Laya, kev, llm_choice) or without a router.

Soft labels: with `--teacher <backend> --teacher-min 0.9`, where no proven
label exists, a confident answer recorded from a local backend becomes
a soft target (its full probability spread). Proven labels always take
precedence, so the trained model can still improve on any backend where
grounding knows better.

It keeps questions with a proven label that is one of the offered options,
drops duplicates, splits train/test by conversation, and writes
`{id, workflow, state, questions, gold}` rows (JSON strings, one-hot `gold`
probabilities), the format of `LocalLLaMA/typed-decisions`. Train with the
Laya repository's `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`
(load with `datasets.load_dataset("json", ...)`), mixing in the public set so
general skills are kept. Compare base and fine-tuned checkpoints with
`pok-ai router-bench` on `diagnostics/router_question_suite.json` and the live
bench before switching `[decision_router.laya]` to the new model folder.

## Swapping the installed Laya checkpoint

`scripts/setup-laya.ps1` takes a `-Repo` parameter (default
`convaiinnovations/laya-typed-decisions`) and a `-Revision` parameter
(default `843893f92cf9`, the measured fine-tune commit; all revisions of
that repo currently share one weights file — the pin protects against a
future Hub-side update like the zeiger regression). Pass a different
Hugging Face repo id to install an alternative checkpoint instead — it
always installs to the same `models\typed-decisions` directory the app
expects (`laya_checkpoint()` in `apps/desktop/src-tauri/src/lib.rs`), so
only one checkpoint is resident at a time. Update
`decision_router.laya.model` in

### Picker comparison (20-case suite, 2026-09-22 replay, local RTX 3090)

The default moved from `Luni/laya-grounded` to the Apache-2.0 typed-decisions
checkpoint because the 20-case suite (which includes the 12 clustering-band
cases the 8-case suite misses) ranks the local pickers differently than the
original 8-case pass:

| Local picker | 20-case pick | License | Notes |
|---|---|---|---|
| `convaiinnovations/laya-typed-decisions` (now default) | **75%** (15/20) | Apache-2.0 | best local picker measured; commercial-safe |
| `php-ai/zeiger-0.6b` pre-Round-15 (judge sidecar) | 65% | CC-BY-SA (weights) | excellent judge, mediocre picker |
| `Luni/laya-grounded` (prior default) | 60% | CC-BY-NC-4.0 | non-commercial license |
| `models/english` ("rl-agent", ModernBERT fallback the app serves when typed-decisions lacks `model.safetensors`) | 55% | — | the app was silently serving this |

The app's `laya_checkpoint()` serves `models/typed-decisions` only when it
contains `model.safetensors`; an ONNX-only install falls back to `english`.
Installing the Apache checkpoint with the default `setup-laya.ps1` puts
`model.safetensors` in place, so the app serves the 75% picker. With the
pinned zeiger judge, the local cascade reaches the documented 90% with
zero false positives (see the table above). The residual gap to hosted JEV
(90% raw pick) is the two conservative judge holds plus five wrong picks
that are correctly held; closing it further means fine-tuning the picker on
harness data (the 20-case suite plus live session traces), not tuning
gates — every gate formula caps at the picker's raw accuracy.
`pok-ai.toml` to match whatever is installed.
