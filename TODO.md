# POK-Ai Roadmap: A Local Windows “Jarvis”

POK-Ai’s long-term goal is one local-first agent that can collaborate on code,
operate the user’s Windows desktop, answer through text or voice, participate in
approved communication channels, and run scheduled tasks. Better local models
should improve the experience without requiring model-specific application
flows.

This file is the implementation roadmap. Completed behavior belongs in the
README and architecture documentation; proposed work remains here until it is
implemented and verified.

## Product principles

- Keep computer use and coding in the same session, memory, policy, and event
  model. They are capabilities of one agent, not separate products.
- Optimize for capable local models that fit on one consumer GPU, while keeping
  provider and model behavior generic.
- Keep Windows capture, grounding, input, process control, and secrets in native
  Rust. React remains a presentation layer.
- Make the benchmark a developer CLI workflow. Do not add benchmark controls or
  model leaderboards to the desktop UI.
- Prefer deterministic verification over asking a model whether it succeeded.
- Require explicit workspaces, identities, and approval scopes. Voice or a
  remote message is never sufficient authorization for a dangerous action.
- Treat user interruption and focus changes as normal desktop state, not as
  task failure.
- Allow the agent to improve through versioned, tested procedures and helpers;
  never permit uncontrolled rewriting or deployment of its own source.

## Current foundation

- [x] Native Windows capture, OCR/UIA grounding, target-based input, focus
  recovery, multi-monitor support, and post-action verification.
- [x] Local/provider model discovery, vision and text-only operation, tool-call
  recovery, strict-role compatibility, configurable temperature, context
  compaction, and transient-provider retry handling.
- [x] Workspace-scoped file reading, search, editing, undo, command execution,
  generated helpers, and a restricted coding subagent.
- [x] Risk-based approvals, emergency stop, password/elevation protection,
  diagnostics, and visible command output.
- [x] SQLite FTS5 memory, approval-gated facts, and verified reusable desktop
  procedures.
- [x] Deterministic coding and desktop exams with isolated fixtures and JSON/Markdown
  reports.

## Phase 1 — Measure before expanding

Extend the existing `exam` implementation into a repeatable benchmark system.
Keep `exam` as a compatible alias or migrate it with a documented transition;
do not build a second runner with different session behavior.

- [ ] Add a CLI command shaped like:

  ```text
  pok-ai benchmark run <suite> --models <model,...> --repeat 3
  pok-ai benchmark compare <run-a> <run-b>
  ```

- [ ] Define versioned suite manifests containing scenario ID, category,
  prompt, fixture, platform requirement, maximum turns, timeout, weight,
  deterministic assertions, and cleanup behavior.
- [ ] Support coding, shell/file, Windows desktop, browser, focus-interruption,
  long-context/compaction, tool-schema compatibility, memory, and orchestration
  suites.
- [ ] Copy or reset every fixture before every attempt. Run GPU models
  sequentially by default so they do not compete for memory.
- [ ] Record success and partial assertion score plus:
  - wall-clock and model latency;
  - turns and total tool calls, grouped by tool;
  - malformed/recovered calls, retries, repeated actions, and provider errors;
  - prompt, completion, and cumulative tokens;
  - captures, OCR/UIA time, compactions, approvals, cancellations, and command
    failures.
- [ ] Store each attempt’s trace, screenshots, command output, workspace diff,
  and final answer under a dedicated benchmark run directory.
- [ ] Produce machine-readable JSON and CSV plus a concise Markdown ranking with
  success rate, median, and p95 results across repeats.
- [ ] Rank deterministic task completion first, then malformed calls, tool
  count, turns, and latency. An optional model judge may explain a result but
  must not replace deterministic assertions.
- [ ] Add comparison output that highlights regressions in success, turns, tool
  calls, latency, and tokens between two harness versions or model runs.
- [ ] Make coding suites runnable cross-platform and clearly skip Windows-only
  scenarios outside a native Windows build.
- [ ] Add a small checked-in smoke suite for CI and a broader local suite for
  model evaluation.

Definition of done: the same suite can compare several models and harness
commits without manual cleanup, and its report makes it obvious whether a change
improved task completion or merely changed the model’s prose.

## Phase 2 — Full coding collaboration

Bring repository work closer to mature coding harnesses without weakening
POK-Ai’s Windows computer-use specialization.

- [ ] Add atomic multi-file patch support with create, edit, rename, and delete
  operations plus exact rollback.
- [ ] Add structured Git tools for status, diff, log, branches, worktrees,
  staging, and commits. Keep destructive or publishing actions approval-gated.
- [ ] Build a bounded repository map from tracked files, project instructions,
  symbols, and relevant documentation so small models receive focused context.
- [ ] Add optional language-service adapters for symbols, references, type
  errors, and diagnostics without making any language server mandatory.
- [ ] Detect project test, lint, format, and build commands and return structured
  output with failures summarized and raw output retained.
- [ ] Add session checkpoints that include file state, generated helpers, task
  state, and the commands required to reproduce or undo a change.
- [ ] Run risky coding work and subagents in disposable worktrees or an
  equivalent restricted workspace.
- [ ] Add review mode for explaining diffs, risks, validation performed, and
  unresolved issues before a change is accepted.
- [ ] Add benchmark cases for multi-file edits, unfamiliar repositories,
  failing tests, merge conflicts, and user interruption during a coding task.

Definition of done: a user can ask POK-Ai to investigate, implement, test,
review, and safely undo a realistic multi-file change without switching to a
different harness.

## Phase 3 — Durable automation

Build scheduling before always-listening voice so unattended work has a durable,
auditable execution model.

- [ ] Add a SQLite-backed job service supporting one-time tasks, recurring
  schedules, time zones, disabled jobs, and durable run history.
- [ ] Define missed-run, retry, overlap, concurrency, and machine-sleep behavior.
- [ ] Give every job an explicit workspace, model/provider, tool allowlist,
  approval scope, timeout, and result-delivery destination.
- [ ] Support useful initial tasks such as alarms, reminders, status checks,
  local reports, and approved maintenance.
- [ ] Add CLI management first; later add a small desktop UI for viewing,
  creating, pausing, and cancelling jobs.
- [ ] Reuse the normal session loop, policy, diagnostics, memory, model
  load/unload lifecycle, and emergency cancellation path.
- [ ] Deliver results through the desktop initially, then through approved voice
  and Discord channels when those adapters exist.

Definition of done: scheduled work survives application restarts, cannot overlap
unexpectedly, records exactly what happened, and can always be stopped.

## Phase 4 — Discord project assistant

Implement Discord as a separate gateway using the same agent core, rather than
embedding bot lifecycle logic in the React UI.

- [ ] Add a bot/gateway CLI or background-service mode with secrets stored in
  the OS credential store and never exposed to the model.
- [ ] Configure an allowlist of servers, channels, roles, and administrators,
  with rate limits and a full audit trail.
- [ ] Map each channel or thread to an isolated agent session and an explicitly
  configured project workspace.
- [ ] Ground answers in approved project files, project instructions, indexed
  memory, release notes, and attached logs. Include source paths or links when
  useful.
- [ ] Support replies, threads, attachments, status, new-session, stop, and
  administrator model-selection commands.
- [ ] Separate “answer questions about this project” from “change the project.”
  Writes, commands, deployments, and external side effects require narrower
  permissions and, where appropriate, local approval.
- [ ] Disable arbitrary remote desktop control by default. Remote users must not
  inherit the local operator’s computer-use authority.
- [ ] Add prompt-injection tests for Discord content, attachments, quoted
  messages, and untrusted project files.

Definition of done: approved Discord users can obtain reliable, project-grounded
help while secrets, unrelated workspaces, and the local desktop remain
inaccessible.

## Phase 5 — Voice and a configurable wake name

Voice should be another input/output adapter around the existing session, not a
second agent implementation.

- [ ] Start with push-to-talk and a provider-neutral speech-to-text interface;
  include a fully local backend.
- [ ] Add a provider-neutral text-to-speech interface, local voice output, and
  interruption/barge-in support.
- [ ] Send the transcript through the same tool, memory, policy, task-status,
  and diagnostics pipeline as typed prompts.
- [ ] Add voice activity detection and visible states for idle, listening,
  transcribing, thinking, acting, and speaking.
- [ ] Add an optional local wake-word engine with a configurable name only after
  push-to-talk is reliable.
- [ ] Provide a hardware/software mute, clear privacy indicators, configurable
  transcript retention, and no raw-audio retention by default.
- [ ] Ask for explicit confirmation before destructive, financial, publishing,
  credential, or other high-risk actions. Voice identity alone is not
  authentication.
- [ ] Integrate alarms and reminders with voice output without requiring the
  main model to remain loaded.

Definition of done: the user can say a wake phrase or press push-to-talk, issue
a normal POK-Ai task, interrupt the response, and receive spoken results with the
same safety guarantees as text.

## Phase 6 — Safe self-improvement

Expand the existing verified-procedure and generated-helper systems into a
controlled improvement loop.

- [ ] Record success/failure statistics and provenance for procedures and
  helpers, scoped by application, project, and harness version.
- [ ] Detect repeated successful flows and propose a generalized skill or typed
  helper with task-specific text, recipients, paths, coordinates, and target
  IDs removed.
- [ ] Validate every proposal in a sandbox or disposable worktree using unit
  tests, exams, and relevant benchmark cases.
- [ ] Present the code, evidence, permissions, and rollback plan for human
  review before promotion.
- [ ] Version promoted skills/tools, support disable and rollback, and retire
  candidates that regress or repeatedly fail.
- [ ] Permit reviewed proposals for prompts, routing, or configuration; never
  silently change production behavior based only on the agent’s own judgment.
- [ ] Prohibit autonomous source rewriting, release publishing, permission
  expansion, secret access, and benchmark tampering.

Definition of done: POK-Ai can propose and validate an improvement, but a human
can inspect, reject, disable, or roll it back and the original harness remains
recoverable.

## Phase 7 — Always-available product

- [ ] Add a quiet Windows tray/background mode with optional startup, crash
  recovery, and clear model/listening/job state.
- [ ] Separate the lightweight gateway/scheduler from GPU-heavy inference so
  alarms and incoming requests do not require a model to stay loaded.
- [ ] Add profiles for workspaces, identities, models, voices, communication
  channels, permissions, and memory boundaries.
- [ ] Add encrypted backup/export and deletion controls for configuration,
  memory, jobs, skills, transcripts, and diagnostics.
- [ ] Add signed releases, migrations, update rollback, health checks, and a
  support bundle with automatic secret redaction.
- [ ] Evaluate a generic extension protocol for optional integrations only after
  the core policy and isolation model is stable.

## Full-harness acceptance scenarios

- [ ] “Fix this failing project, run its tests, explain the diff, and let me
  undo it.”
- [ ] “Open this Windows application, complete the task, and recover if I click
  into another window.”
- [ ] “Compare these local models on the same coding and desktop suites and show
  success, turns, tool calls, latency, and tokens.”
- [ ] “Answer a Discord user’s question using only the approved manager-project
  workspace, without exposing unrelated local data.”
- [ ] “Hey &lt;name&gt;, what time is it?” works locally without loading the main
  model when a deterministic tool can answer.
- [ ] “Wake me at 7:00 AM” creates a visible, durable, cancellable local alarm.
- [ ] A repeated successful workflow can become a tested, versioned proposal,
  but cannot promote or deploy itself without the configured approval.

## Explicit non-goals and guardrails

- No model-specific hardcoded application flows in production.
- No benchmark controls in the end-user desktop UI.
- No unlimited remote control of the local Windows session.
- No UAC, secure-desktop, password-field, or hidden credential automation.
- No unverified memory presented as fact and no unverified procedure promoted
  as a skill.
- No autonomous self-modification or release publishing.
- No requirement that one provider, speech engine, messaging service, or model
  be permanently built into the core.

