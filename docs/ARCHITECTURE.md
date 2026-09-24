# POK-Agent Architecture

## Runtime boundaries

POK-Agent (internally POK-Ai) has one provider-neutral Rust core. The CLI and Tauri app construct the
same `Session`, `ToolRegistry`, `Policy`, `MemoryStore`, and `DesktopPlatform`.
No agent decisions are implemented in React.

```text
CLI / Tauri
    │
    ▼
Session actor ──► Brain adapter ──► LM Studio / Ollama / Anthropic Messages
    │                   │
    │                   └── normalized text, reasoning, tool calls, usage
    ▼
Policy ──► Approval broker ──► Typed tool registry
                                  │
                 ┌────────────────┼────────────────┐
                 ▼                ▼                ▼
           Windows desktop   Coding tools    Memory / child agents
```

## Local-model recovery

The OpenAI-compatible adapter assembles streamed tool-call fragments. Invalid
JSON becomes a structured `MalformedToolCall` event rather than terminating the
session. The session feeds a short correction back to the model and increments
the benchmark error count. It also recognizes LM Studio's fallback
`[TOOL_REQUEST]...` format when a model does not emit native tool-call objects.

Only the newest screenshot remains in active context. Older images are replaced
with compact markers while their full PNGs remain in the artifact directory.
After six tool results, older complete tool cycles are micro-compacted into one
bounded continuity ledger. The original request and recent cycles remain
verbatim; the ledger retains tool counts, recent earlier actions, arguments,
and success/failure state. Tool-call arguments are included in token estimates,
and the complete unabridged execution remains available in `trace.jsonl`.

## Desktop safety

Desktop input requires a fresh observation UUID. POK-Ai rejects stale actions,
coordinates outside the selected window/monitor, elevated windows, and typing while a
password element is exposed. Exam mode additionally checks the process and
window title. The Windows adapter uses `screenshots`, Windows OCR, UI Automation,
and Enigo's Windows `SendInput` backend.

The dashboard and the native `Ctrl+Alt+Esc` global shortcut both cancel the
active session's cancellation token. This does not depend on the dashboard
having keyboard focus.

Windows OCR runs on a persistent WinRT worker and returns word rectangles from
the exact source frame used for the model capture. UI Automation uses one cached
bulk descendant query. The grounding layer treats named interactive UIA controls
as authoritative and uses overlapping OCR to enrich their labels. Unmatched OCR
regions are readable spatial anchors but do not independently authorize clicks.
When accessible actionable coverage is sparse or misses task terms, a bounded
self-contained visual pass detects generic high-contrast slider/track affordances
and contributes actionable geometry. Tiny OCR fragments remain readable content
rather than click authority. Failure of either
source is recorded as a warning while the other source and vision remain usable.

Repeated capture of the exact same target image and foreground window reuses
the prior OCR/UIA enrichment. The model receives an `unchanged` state with the
current target list instead of repeated ordered text, while diagnostics retain
the complete observation. Any input, navigation, scrolling, activation, focus
change, or image difference forces fresh grounding.

The original task is available to the tool context for deterministic ranking.
Task-label matches, high-quality actionable controls, focus state, and spatial
diversity determine the bounded action registry. Vision-capable models receive a
clean image followed by an annotated image whose ids exactly match that registry.
The model contract includes `id`, `label`, `role`, local `box`, `source`,
`actionable`, and `grounding_quality`; full ranking reasons stay in diagnostics.

The complete native UIA result is fused before the model target cap is applied.
For monitor, region, and all-screen captures, the Windows adapter queries primary and
secondary taskbar UIA roots first and reserves a bounded share of the native element
budget for them. This prevents a large foreground accessibility tree from hiding Shell
controls before grounding can rank them.
Task-matching text, editors, actionable controls, and spatially diverse controls
receive reserved capacity, preventing late tree elements from disappearing in
large Electron applications. Adjacent OCR words are also exposed as compact lines.
Model-facing input results use short observation handles and model-image points;
mapped physical desktop coordinates remain diagnostic-only.

Interactive capture is hierarchical and target-scoped. `observe_desktop` provides
an orientation-only monitor map plus compact task-ranked windows and invalidates
old input authority. `list_windows` is the larger compact fallback,
`activate_window` establishes a foreground application, and `capture_screen`
returns one detailed numbered monitor/window image with model-local target bounds.
When native foreground activation is rejected by Windows, the tool captures the relevant
monitor, clicks only a unique high-confidence taskbar identity match, and then tries a
bounded verified Alt+Tab sequence. An exhausted attempt returns an enlarged live recovery
observation instead of encouraging another blind activation call. `inspect_screen_region`
can enlarge either a fresh target id or a bounded rectangle derived from the latest
observation. Derived crops retain the source observation's input authority, carry a
short-lived `view_id`, and are invalidated by every real capture or input. For unnumbered
controls, `locate_visual_target` records the model's labeled box, checks it against UIA/OCR,
and returns a confirmation crop. A separate `click_localized` call consumes the one-use token
and clicks the resolved center. Model-only boxes remain valid for games and canvases.
The vision model is the primary perception source; deterministic controls corroborate by overlap
and center proximity, with labels used only to break close geometric ties.
Each localization persists an annotated full-source PNG, an annotated confirmation-crop PNG,
and bounded JSON metadata keyed by the same localization id for diagnostic correlation.
Hover and drag tools map view-local coordinates through the source capture before policy
validation. `click_target` checks the model's expected visible
label against the selected id. A single stronger semantic match may correct the id;
zero or multiple matches execute no input and require re-grounding. Raw coordinate clicks through
`simulate_input` are rejected. A same-process foreground child opened by
input is returned as a related transient so the next turn continues in that popup
instead of reactivating its parent. Native window
coordinates remain internal for policy checks. Monitor capture authorizes its
monitor boundary so taskbar and tray targets can be used. All-monitor capture and
desktop overview cannot authorize input.

### Working alongside the user

There is one mouse and one keyboard, and the agent runs in the user's own
session, so it is designed to disturb the user as little as possible:

- **Cursor-free actions.** `click_target` first tries a UI Automation pattern
  whose meaning matches the mouse action exactly: Invoke for buttons, links,
  menu items and split buttons; Toggle for check boxes; SelectionItem for radio
  buttons and tabs; and, for a double click on a list, tree or data item, Invoke
  or the item's default action (open). The control must match the observed
  target's name and bounds inside the authorized window, and password fields are
  refused. Anything else falls back to physical input: a single click on a list
  or tree item (which selects in some apps and opens in others), right clicks,
  canvas controls, and OCR-only or visual targets. The input result reports
  `input_method` (`physical` or `ui_automation:<pattern>`).
- **Yielding to the user.** Before any physical input, the agent checks for
  mouse or keyboard activity from the user in the last 1.5 s. If there is some,
  it holds the input until the user has been idle for 2.5 s, then continues on
  its own. The dashboard shows "Waiting for you" meanwhile
  (`user_activity_wait` events). The Windows adapter separates the user's input
  from the agent's own: input that `GetLastInputInfo` records after the agent's
  last injection finished can only be the user's.
- **Background work** (`background_desktop_work`, on by default).
  `AgentWindowDesktop` (`agent_window.rs`) wraps the platform and tracks the
  window the agent works in. While it is open and not minimized, the runtime
  treats it as the foreground window, so every observation, freshness and input
  check applies to the agent's window rather than to the window the user has in
  front. `activate_window` only moves the agent's attention. Captures of a
  covered window are rendered with PrintWindow (full-content rendering, via
  `xcap`), controls inside it are found through its own accessibility tree, and
  switching windows is not treated as a desktop change. Only physical input
  brings the agent's window forward, after the user is idle. If the window
  closes or is minimized, the wrapper reverts to the real foreground and normal
  desktop-change recovery takes over.

A disconnected remote-desktop session has no interactive display: screen
capture and physical input fail. PrintWindow rendering and accessibility
actions may still reach the agent's window, but any step that needs a real
click or typing waits for a live session.

## Coding and subagents

All paths are canonicalized with Windows-safe `dunce::canonicalize`. Interactive
and Exam sessions remain workspace/sandbox-contained; Autonomous sessions may
use normal local-drive paths outside the workspace. Autonomous sessions run
ordinary commands—including user-data, network, publishing, messaging,
credential, and external actions—without approval. Argument-aware command
classification still requires approval for OS-critical files, boot and disk
state, machine-wide security, accounts, services, drivers, and system policy.
Edits require one exact match, use atomic writes, and record a hash-guarded undo
entry. Commands use a conversation-scoped manager with progressive output,
bounded full logs, foreground-to-background handoff, paginated reads,
wait/kill/stdin/PTY controls, process-tree cancellation, and explicit working
directories. A desktop policy change updates the live session and its
authorization context before the next prompt while retaining conversation
history and clearing approvals cached under the previous mode.

Artifact completion is capability-neutral. Missing, unreadable, corrupt, or
unopenable deliverables are blocking; formatting, layout, chart construction,
and other quality heuristics are advisory. A usable result is returned as
`completed_with_warnings` instead of being discarded when advisory inspection
notes remain, and clean application evidence can resolve conflicting heuristics.

Browser navigation uses bounded adaptive settling instead of a fixed screenshot
delay. A changed address bar alone is not loaded: the destination must be paired
with a changed title or confirmed changed content. A timed-out observation is
returned as `loading` and cannot ground a live factual answer.

Run completion includes authoritative structured deliverables sourced from
artifact evidence. Preexisting command inputs remain observations rather than
new outputs. Optional application-view verification has a bounded recovery
budget and cannot keep a usable artifact in an open-ended loop.

History compaction never stores a raw reasoning channel. Visible model summaries
are screened for planning scaffolding and echoed instructions. Compaction has no
small fixed output or time ceiling: reasoning models may finish their internal
work before emitting the visible summary. Invalid or unavailable summaries are
retried through the LLM; if all attempts fail, canonical history remains intact
and compaction is reported as failed rather than replaced heuristically. Read-only
live browsing is not learned as a reusable procedure from UI changes alone.

Coding children require a clean Git repository. Each child receives only coding
tools, cannot spawn grandchildren, and works on a temporary branch/worktree.
POK-Ai returns its commit and diff but never merges it automatically.

## Optional JEV decisions

The core owns one optional bounded decision interface for action routing,
structured-evidence focus, optional retrieved context, and inferred-memory
verification. Candidate generation, relevance prefiltering, deduplication,
budgets, thresholds, policy, execution, and outcome verification remain local.
For action routing JEV chooses one operation and then one exact target identifier
already supplied by the harness. The session may continue this structured loop
through structured desktop or managed-browser actions until DONE, BLOCKED, a
safety boundary, ambiguity, or the configured action budget. Missing text uses a
no-tools primary-model helper, and consequential commits require a state-bound
primary-model review before JEV executes them. High-confidence DONE disables
tools persistently for final synthesis, including output-repair retries; BLOCKED restores the normal
full-tool model. Unknown
IDs, non-finite scores, and model mismatches use the primary-model handoff. Low-confidence
actions and BLOCKED selections made while viable reversible actions remain enter an adaptive
refinement loop: refresh once, temporarily withhold repeated uncertain semantic targets, narrow the
action set, and retry while confidence improves. Operation and exact-target confidence use separate
thresholds. One failed target does not disable its entire tool class. After the
bounded refinement budget is exhausted, a tool-free visual assistant may choose only an existing
candidate or request a refresh, once per unchanged state; visible text is parsed first and reasoning
is a structured-output fallback. The selected action returns through normal policy and then back to
JEV. Current external-information work is routed to the managed browser, while desktop orientation
is offered only for explicit UI/application work. Context routing reorders a bounded local shortlist and fills any unused slots from the local
ranking, so JEV cannot remove the complete baseline.
Three consecutive provider failures suspend JEV for the run.

Candidate generation covers the whole environment, not just the visible window list: when the task
names a known application that is not already on screen, the harness offers an explicit
`open_application` candidate (bounded known-app list, token-boundary matching, launch-gated as
process execution) so the fast loop launches the application, lists the new window, and activates it
instead of clicking around looking for something that was never open. Terminal DONE is offered only
after the current run has produced fresh evidence (`qualifies_as_fresh_evidence` on a router or tool
action), so the router can never select a finish the terminal gate must reject; the refinement
budget and the withheld-candidate set persist across re-observations rather than resetting on every
state-revision bump, which bounds repeated uncertain candidates; and evidence from a previous prompt
never authorizes DONE for a new request.

Optional retrieval is local-first. A workspace-scoped SQLite FTS5 index is stored under the data
directory and refreshed from changed file fingerprints. BM25 content/path and symbol rankings are
combined with reciprocal-rank fusion; implementation searches reserve source-code capacity and
overlapping excerpts are suppressed. Memory and archive FTS searches produce larger bounded pools
for the same reranker. JEV may first classify implementation, explanation, or general intent and may
make one bounded recovery attempt when no candidate clears the relevance threshold.

Mandatory instructions, the current request, required tool call/result pairs,
unresolved failures, approvals, and action observations never pass through JEV
selection. Outbound fields are centrally allowlisted and bounded. Candidate
labels and memory/context snippets are explicitly marked as untrusted data;
sensitive-looking payloads stay local, and serialized requests are capped at
32 KiB. Counts, token arithmetic, dates,
versions, and score comparisons are always computed in Rust because they are
deterministic harness work, not semantic classification.

The conversation stores the requested decision-router backend independently from current key
availability. The backend is Off, hosted JEV, or local Laya. Laya runs in an authenticated,
loopback-only sidecar managed by the Rust desktop runtime; both backends share the same bounded
candidate, policy, verification, caching, and diagnostics path. Older conversations restore the
legacy JEV setting when present and otherwise use Off. The desktop composer
shows Ready, Evaluating, local fallback, unavailable, or off and persists
bounded lifecycle summaries. The CLI reports the same lifecycle and requests
the same external-data acknowledgement. Neither entry point requires JEV or a
TypeSafe credential for normal operation.
When the router is disabled or unavailable, the fast path is not entered and the primary model
uses the unchanged full agent loop.

Laya's installed checkpoint is a setup-time choice, not a hardcoded one: `scripts/setup-laya.ps1`
takes a `-Repo` parameter naming any compatible Hugging Face checkpoint (default
`convaiinnovations/laya-typed-decisions` — Apache-2.0, and the best local picker measured on the
20-case harness suite at 75% raw two-stage accuracy; `Luni/laya-grounded` (CC-BY-NC-4.0) and the
ModernBERT `english` fallback measure 60% and 55% respectively — see
`docs/decision-router-backends.md`). Both local checkpoints are pinned to validated revisions
(`-Revision`): pre-alpha Hub `main` weights have regressed silently before (zeiger's "Round 15"
update collapsed judge correlation from 90% to 40% mid-day), so the setup scripts default to the
exact measured commits and record them in `manifest.json`. Adding a structurally different backend
model, rather than swapping Laya's checkpoint, means writing a small adapter process speaking the
same bounded HTTP contract, also documented there.

### Cross-model judge

When the combined probability gate rejects the picker's candidate, the session may ask a **second
local model** (pinned `php-ai/zeiger-0.6b`, via `judge_candidate` in `decision.rs`) to double-check
that specific pick before falling back to the primary model. The judge payload carries the task,
current step, the candidate description, and the same bounded NextAction state the picker saw
(`bounded_judge_state`: window/browser/last-action/outcome — so the evidence question is
answerable). It answers four `choice` questions: three absolute ones (safe/reversible, cheap/low-
risk, evidenced by the current state) and a **comparative** fourth (picked candidate vs. the best
runner-up in the same operation family). Promotion requires unanimous "yes" and the pick preferred
over the runner-up; the comparative question is only ever a veto and is only asked when a runner-up
exists. Verdicts (`JudgeVerdict`: votes, confidences, probabilities) are logged under
`decision_router_judge_attempt` with the bounded inputs, **memoized** per (task, current step,
candidate fingerprint) so a repeated candidate after an unchanged re-capture is answered from cache
(`reused: true`) instead of re-judged, and replayed offline via
`scripts/replay_live_judge_payloads.py`.

The measured effect on the same real Windows task: the local fast loop (picker + gate + judge)
dropped the run from 24 turns / 44 tool calls / ~314k prompt tokens to 2-3 turns / 4 tool calls /
~35k prompt tokens, at parity with hosted JEV on that task (3 turns / 8 calls / ~65k). JEV remains
the higher-accuracy picker (90% vs. 75% raw on the 20-case suite); the local edge comes from the
judge rescuing borderline picks and the loop fixes removing redundant work. The judge is wired for
the Laya backend only (cross-model agreement is the escalation signal; a model judging its own pick
shares its blind spots — JEV judging itself re-endorsed wrong picks in the same suite).

## Persistence

The data root contains SQLite FTS5 memory, structured verified procedures,
project-scoped generated tools and promotion candidates, undo entries,
worktrees, and exam reports. Exam runs use separate memory and copied
fixtures, preventing training or state leakage between compared models.

Conversation checkpoints retain the provider-neutral message sequence. The desktop initially reads
only the newest bounded page, reconstructs tool calls at their original sequence positions, and
loads older pages when the user scrolls upward. Runtime restoration still reads the complete
checkpoint only when the user continues that conversation.

Interactive desktop input maintains a session-local submission ledger. A typed
payload followed by Enter is verified only when it appears as new non-editable
UIA/OCR content. Verified exact repeats to the same destination are suppressed.
At accepted task completion, verified desktop trajectories and successful command
templates are compiled into task-aware procedure records. The compiler preserves
the reusable action pattern across window navigation, files, browsers, settings,
forms, creative tools, and communication applications while omitting payloads,
recipients, paths, coordinates, ephemeral target ids, and secrets. Matching
fingerprints reinforce one record instead of producing duplicates. Retrieval is
balanced across workflows, commands, and facts and is injected as guidance only;
every new run must still obtain fresh observations and pass current safety checks.
Unverified flows never become procedures, and inferred facts remain
approval-gated curation drafts. Ordinary fact deletion retains only an exact
fingerprint. The separate semantic-forget operation discloses and stores a
normalized local copy, allowing similar inferred facts to be suppressed without
silently retaining deleted text during ordinary deletion.

A successfully executed workspace helper becomes a project-scoped promotion
candidate. Candidates retain only the helper path, source hash, runtime, and
verified invocation; they are not copied, enabled, or executable as generated
tools. Installation still requires `promote_helper_tool`, source inspection,
a declared JSON contract/capability set, and the existing approval policy.
Promotion builds a tool-local Python virtual environment, Node package tree, or
PowerShell module path, writes locked environment metadata, and atomically
installs only after all declared smoke tests and semantic JSON assertions pass.
Helpers consume one JSON value on stdin and produce one JSON value on stdout.
Promotion validates RFC 6901 assertion pointers before environment creation,
rejects undeclared packages installed during smoke tests, and returns
phase-specific bounded diagnostics. Invocation results over 64 KiB are retained
as session artifacts and exposed through opaque, session-bound paged reads.
The desktop can delete an installed project tool after confirmation, removing
its manifest, copied entrypoint, dependency lock, and private environment while
leaving the workspace helper source untouched.

Provider configuration declares a `local_device` or `external_service` data
boundary. Local models receive full-fidelity task context. Before inference with
an external service, the desktop requires conversation-scoped confirmation that
screenshots, OCR, workspace paths, documents, prompts, and tool results may
leave the device. This data boundary does not weaken execution approvals or
permit the agent to discover unrelated secrets.

Each interactive session directory is an LLM-development diagnostic bundle. Its
`trace.jsonl` records the prompt, request shape, per-turn latency, reasoning,
response text, finish reason, token usage, tool calls/results, and failures.
Successful observations also write the source/model PNG, numbered model PNG,
word OCR text, fused target JSON, full observation JSON with stage timings, and
an SVG overlay with OCR and UIA rectangles.
For checked-in development configuration these bundles live in the repository's
`diagnostics/sessions` directory and are not exposed through the application UI.
Duplicated image base64 is omitted from JSON and traces because the exact bytes
are saved as separate full-fidelity PNG artifacts.

Session construction adds a Windows-local date, weekday, timezone abbreviation,
and UTC offset to the cached system prefix. The date is checked before every model
turn; a midnight rollover is appended at the conversation tail instead of mutating
the prefix. Exact local and UTC timestamps are available through a read-only clock
tool, keeping minute-level volatility out of ordinary prompts.

Before every model call, stale screenshots and older tool payloads are
deterministically compacted toward the configured 12K prompt target. LM Studio
usage updates calibrate the estimator after each turn. The dashboard reports
the latest request separately from session-cumulative token usage.

The Tauri runtime owns a live `Session` keyed by provider, model, and canonical workspace.
Successive user prompts reuse its messages, compacted tool history, safety ledger,
and diagnostic session id. At safe history boundaries the core transactionally checkpoints
the provider-neutral transcript and resumable ledgers to the central conversation SQLite
store. Tool results for one assistant tool-call batch are kept contiguous and in declared
call order; restore also repairs displaced, duplicate, orphaned, or interrupted historical
results without replaying a tool. Resume reconstructs the saved provider/model/workspace but
always clears observations, focused controls, approvals, and other authority tied to external
state. A model/workspace change or explicit New conversation drops the live state, while the
saved conversation, approved SQLite memory, and procedural skills remain cross-session.
Background LLM curation waits for an idle window and is cancelled when a follow-up begins,
preventing simultaneous inference requests to the local model.

During an active desktop-dependent model turn, the core samples a metadata-only
desktop activity snapshot every 250 ms. Two consecutive foreground mismatches
invalidate the observation and interrupt partial model output before it can become
a stale tool call. Windows supplies the foreground identity and time since the
last user input without capturing screen contents. Recovery yields until the user
has been idle for three seconds and pauses after 30 seconds of continuous activity.
The optional decision router can rank only bounded recovery choices: reobserve a
same-application transition, restore the previously authorized non-elevated task
window, or defer to the primary model. The local policy implements the same safe
recovery lifecycle when the router is disabled or unavailable. Every successful
recovery creates a normal fresh capture tool result, so the restarted model does
not need to request the same evidence again.
