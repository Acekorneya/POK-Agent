# POK-Agent

[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://www.buymeacoffee.com/acekorneyab)

POK-Agent is an autonomous general computer and coding harness built primarily
for local models, with cloud bring-your-own-model (BYOM) support. The Rust-first
Windows runtime has a shared CLI and Tauri dashboard, typed tools, risk-based
approval, SQLite FTS5 memory, restricted coding subagents, and a deterministic
model exam. The internal code and package name remains POK-Ai.

See [TODO.md](TODO.md) for the implementation roadmap toward a local Windows
“Jarvis,” including deeper coding workflows, terminal-only model benchmarks,
durable automation, Discord, voice, and safe self-improvement.

## Current MVP boundaries

- Native Windows owns screen capture, OCR/UI Automation queries, and input.
- WSL is supported for editing, core tests, mock-platform tests, and frontend work.
- LM Studio is the default provider; Ollama uses the same OpenAI-compatible adapter.
- Model capabilities are discovered from LM Studio and compatible cloud-provider metadata. Vision
  models receive the numbered image and targets; text-only tool models receive the same locally
  generated OCR/UIA targets without image content. If metadata is missing and a provider rejects an
  image, the request is retried once as text-only and retained that way for the conversation.
- Every session receives the current Windows-local ISO date, weekday, timezone abbreviation, and
  UTC offset. A read-only clock tool supplies exact local/UTC time only when a task needs it.
- Window activation and desktop input run autonomously in interactive sessions. File writes and
  commands still require approval outside exam sandboxes.
- Autonomous policy mode permits file, application, installation, generated-tool, command,
  user-data, network, publishing, messaging, credential, and external work across local drives
  without approval. Commands that can change OS-critical files, boot or disk state, machine-wide
  security, accounts, services, drivers, or system policy still require approval. Native
  secure-desktop and password protections remain.
- POK-Ai does not interact with UAC/secure desktop, password fields, or elevated windows.
- Post-turn curation is tool-free. Facts remain approval-gated drafts, while verified general
  Windows procedures are compiled into sanitized, approved skills for future recall.
- When the built-ins are insufficient, the agent can author a task-specific PowerShell, Python,
  or source-code helper inside the approved workspace, execute it, inspect output, and revise it.
- Voice, Discord gateway integration, and vector memory are intentionally deferred.

## Quick start

From WSL:

```bash
./scripts/setup-wsl.sh
./scripts/check-wsl.sh
cargo run -p pok-ai-cli -- doctor
```

From native Windows PowerShell:

```powershell
.\scripts\setup-windows.ps1
.\scripts\check-windows.ps1
cargo run -p pok-ai-cli -- doctor
npm --prefix apps/desktop run tauri dev
```

For repeatable native testing after code changes, use the PowerShell launcher. Development mode
uses hot reload with an optimized capture path; shipped mode runs the checks, creates the release
executable and installer bundles, then launches the release executable:

```powershell
.\scripts\run-windows.ps1 -Mode Dev
.\scripts\run-windows.ps1 -Mode Shipped
```

Pass `-SkipChecks` only when checks already passed for the same commit, or `-NoLaunch` when a release
artifact should be built without starting it. Use `-Clean` to discard generated Rust build artifacts
before a build; the launcher also does this automatically once when its Rust build profile changes.

### Publish a GitHub release

Once a change is ready for users, bump the matching versions in the workspace `Cargo.toml` and
`apps/desktop/src-tauri/tauri.conf.json`, commit the change, and push it to the branch's upstream.
Install and authenticate [GitHub CLI](https://cli.github.com/) once, then run this from native
Windows PowerShell:

```powershell
gh auth login
.\scripts\run-windows.ps1 -Mode Shipped -Release
```

`-Release` runs the normal native checks and Tauri build, creates a `v<version>` GitHub release with
generated notes, and uploads stable-named x64 assets: the standalone executable, NSIS installer,
portable ZIP, and `SHA256SUMS.txt`. The ZIP contains the executable, example configuration, README,
license, and notice. Publishing stops before the build if the working tree is dirty, the commit is
not pushed, the Cargo and Tauri versions differ, authentication is missing, or that release/tag
already exists. A publishing run never launches the built app. Do not combine `-Release` with
`-SkipChecks` unless the exact pushed commit has already passed the native checks.

These community builds are currently unsigned because generally trusted Windows code-signing
certificates are not free. Windows may therefore show a SmartScreen warning. Users can inspect the
public source, compare the published SHA-256 checksums, build it themselves, or install the NSIS
package. Code signing can be added later without changing this release workflow.

The Windows setup script installs the Visual C++ build tools, WebView2 runtime,
the project-pinned Rust 1.88 MSVC toolchain, and Node.js. The dashboard registers `Ctrl+Alt+Esc` as a
system-wide emergency stop; the button in the dashboard uses the same
cancellation path.

The checked-in development configuration targets LM Studio on the local machine at
`http://127.0.0.1:1234/v1` and model `google/gemma-4-26b-a4b-qat`.
Copy `pok-ai.example.toml` to `%LOCALAPPDATA%\POK-Ai\pok-ai.toml` or pass
`--config` to override it. Start LM Studio's local server and select a
tool-capable model; vision is optional.

POK-Ai controls main-agent sampling with `agent_temperature` in `pok-ai.toml`.
It defaults to `0.0` for reliable tool and coding work. Set a numeric value from
`0` to `2`, or set `agent_temperature = "server_default"` to omit the parameter
and let the selected model/provider choose its required sampling policy.

An image-capable model is not required for desktop testing. When LM Studio or a cloud provider
reports that vision is unavailable, POK-Ai automatically switches that session to text-only
grounding. OCR and UIA still run locally, screenshots remain in diagnostics, and the model operates
through numbered targets and `click_target`.

## Optional JEV or Laya decision router

JEV is TypeSafe's hosted System-1 decision model (state + typed questions in, calibrated
probabilities out — no text generation). POK-Agent speaks the same bounded contract to either the
hosted JEV endpoint or a local Laya/zeiger sidecar, so the architecture is identical; only the
model behind the endpoint differs. JEV is an experimental, optional decision layer for bounded actions, evidence, optional retrieved
context, and inferred-memory review. POK-Agent filters and ranks candidates locally before sending a
small candidate set. For actions, JEV first chooses an operation and then an exact harness-supplied
target. It continues through a bounded structured browser or desktop flow until it selects DONE or
BLOCKED; every action still passes through normal policy, freshness, target, and post-action
verification checks. A low-confidence decision or BLOCKED response with viable reversible actions
starts an adaptive refresh-and-narrow loop instead of immediately waking the primary model. The
operation and exact-target confidences have separate thresholds; repeated uncertain semantic
targets are temporarily withheld for the unchanged state so JEV explores an alternative. Failed
targets are suppressed individually, while the action class remains available for other targets.
After confidence stops improving, at most one tool-free bounded vision request per state may select
only from the current harness candidates before control returns to JEV. Sensitive fields are redacted
individually so one unsafe label does not discard the safe action space. The primary model remains
the reasoning and conversational brain: it generates missing text, reviews consequential commits,
handles unsupported work, and turns fresh evidence into the final user-facing answer. A verified
high-confidence DONE enters persistent tool-free finalization: malformed output may be retried, but
tools remain disabled until the visible answer is produced. Live web-information requests are routed
to the managed browser without unrelated desktop orientation. JEV does not bypass the harness safety
boundary. With JEV disabled, the primary model retains the complete original agent loop and tool use.

The same bounded router contract can run through **Laya**, the local open-source option, with an
optional **local cross-model judge** (zeiger) as a second opinion. All local decision models run as
authenticated loopback-only sidecars that POK-Agent starts on demand; task/UI metadata never leaves
the device on this path. Settings can install Laya 0.3.4 plus the Apache-2.0 typed-decisions
checkpoint (`convaiinnovations/laya-typed-decisions`, pinned to the measured revision) into the
POK-Agent data directory; the sidecar prefers CUDA when at least 2 GiB of VRAM is free and otherwise
uses CPU. Rust retains candidate construction, confidence gates, safety, execution, verification,
and diagnostics.
The Settings panel can select Auto, GPU, or CPU. Selecting Laya preloads the model and waits for its
health check before a task can use it. GPU mode requires an actual CUDA load instead of silently
falling back, and the UI reports the selected device and fallback reason. Installation and loading
progress appear in the panel; helper processes run without opening console windows.
On application exit, POK-Agent explicitly terminates and reaps the Laya sidecar before allowing the
native event loop to close, releasing its CUDA allocation; kill-on-drop remains a crash fallback.

### How the local fast loop reduces primary-model calls

The decision router exists to keep the primary (large) model out of the loop for bounded,
reversible actions. Per observation, the harness fuses UIA + OCR into a bounded numbered target
list, then the local picker makes one ~100 ms two-stage call (operation, then exact target):

```text
observe → fuse UIA+OCR → bounded targets
  → Laya picker (operation + target, one call)
      ├─ gate clears (op ≥ 0.35 and target ≥ 0.90) → execute the tool → re-observe
      ├─ gate rejects → local zeiger judge (4 questions) → unanimous → execute
      └─ judge declines → bounded refinement/refresh → primary model handoff
```

The judge asks three absolute questions (safe/reversible, cheap/low-risk, evidenced by the current
state — now fed the same bounded state the picker saw) plus a **comparative** fourth (picked
candidate vs. the best alternative in the same operation family), and promotes only on unanimous
"yes" with the pick preferred. Verdicts are **memoized** per (task, current step, candidate), so a
window or channel re-proposed after an unchanged re-capture is answered from cache instead of
re-judged. Every attempt logs per-question votes and confidences, replayable offline.

The primary model is consulted only when the fast loop reaches a terminal DONE (fresh evidence is
sufficient to answer), a BLOCKED state, a judge decline, or the bounded action budget — and then it
receives the *curated* working set (window title, relevant targets, OCR text, screenshot), not raw
desktop dumps. Measured on the same real Windows task before/after these changes: **24 turns / 44
tool calls / ~314k prompt tokens → 2-3 turns / 4 tool calls / ~35k prompt tokens**, at parity with
hosted JEV on that task (3 turns / 8 calls / ~65k). JEV remains the higher-accuracy picker on the
20-case harness suite (90% vs. 75% raw picks); closing that gap locally means fine-tuning the
Apache-2.0 picker on harness data, not changing the architecture.

**Missing applications are a decision, not a scavenger hunt.** When the task names a known
application that is not already visible, the harness offers an explicit `open_application` candidate
(bounded to a known-app list, token-boundary matched, launch-gated by policy). The fast loop then
launches it, lists the new window, and activates it — verified live: `open calculator` ran as
launch → list → activate → done with zero primary-model handoffs. Without this the loop could only
click around existing windows looking for an application that was never open.

**The loop is bounded by construction.** Terminal DONE is only offered after the current run has
produced fresh evidence, so the router can never pick a finish the terminal gate must reject; the
refinement budget and withheld-candidate set persist across re-observations, so one uncertain
candidate cannot be re-picked indefinitely (live traces previously showed the same candidate
re-picked eight times after each refresh reset the budget). Evidence gathered for a previous prompt
never authorizes DONE for a new request, so a stale snapshot cannot make the agent answer a fresh
question without gathering anything.

Both local checkpoints are **pinned to validated Hugging Face revisions** (`setup-laya.ps1` /
`setup-zeiger.ps1` `-Revision`): pre-alpha Hub `main` weights regressed silently before (zeiger's
"Round 15" update collapsed judge correlation from 90% to 40% mid-day). See
`docs/decision-router-backends.md` for the full benchmark evidence, cascade tables, and the
cross-model judge design.

### What you'll see in the dashboard

The chat feed tells you who is acting, in plain language:

- **`Laya/JEV selected a candidate`** — the picker chose an action the harness executed. Expand the
  row for the pick, its operation/target probabilities, and the top alternatives (shown with
  friendly names like *Get current time*, *Activate window*, *Click target*).
- **`Laya/JEV finished (evidence sufficient)`** (shown as **Router done**) — the fast loop stopped
  because fresh evidence is enough for the primary model to answer. **`Router blocked`** means no
  safe reversible action remained.
- **`Laya/JEV deferred to the main model`** — only ever for next-action decisions; the primary
  model takes over.
- **`Zeiger approved/held: <action>`** — the local judge double-checked a borderline pick. Approved
  means the harness executed it without the primary model; held means it handed off. The row shows
  the three checks (reversible/cheap/evidenced), the comparative runner-up preference, and whether
  a cached verdict was reused.
- **Context routing is quiet.** The router also decides whether to feed retrieved memory, learned
  skills, or session archive into the primary model's context based on what you're working on.
  Those decisions never show a confusing "deferred" row — they only announce themselves when they
  actually add context: **`Laya/JEV added optional context`**.
- **Install/load progress.** A progress bar appears inside the chat feed, above the composer, and
  in Settings while Laya, the judge model, or the primary local model is installing or loading into
  VRAM — with a one-line status of what is happening.
- **Searchable model picker.** The model control next to the composer is a search box: type to
  filter, Enter picks the top match, Esc closes.

These are user-facing summaries; every decision is recorded with its full payload in the session
`trace.jsonl` (`decision_router_judge_attempt` events include per-question votes) and replayed
offline with `scripts/replay_live_judge_payloads.py`.

Saved conversations reopen from SQLite in chronological pages. The newest page appears immediately;
scrolling upward loads older user messages, tool activity, and assistant responses in their original
sequence without moving prior actions to the bottom of the chat.

Workspace retrieval uses a lazy persistent index under the application data directory. It combines
FTS5 BM25 content/path ranking with symbol-aware ranking and reciprocal-rank fusion, then builds
bounded declaration-oriented source previews. JEV can classify retrieval intent and rerank the
30-item local shortlist. A weak first pass receives at most one bounded recovery request; otherwise
the local ranking is used. The same local index works when JEV is disabled.

The decision router is disabled by default. JEV uses the hosted TypeSafe endpoint, so it requires a TypeSafe API key.
Open **Settings**, enter and save the key in the **Jev decision router** panel, then use the JEV
control beside the chat composer. The choice is stored per conversation; changing it after a
conversation starts creates a new conversation after confirmation.
The enable checkbox remains unavailable until a non-empty key has been stored securely in Windows
Credential Manager. Rust enforces the same requirement, so JEV cannot be enabled by bypassing the
interface. If configuration says JEV is enabled but its key is unavailable at startup, POK-Ai starts
with JEV disabled and explains the requirement in Settings. Clearing or losing the key never causes
the primary task to be sent to JEV.

Laya requires no cloud key and task metadata stays on the device. Off, JEV, and Laya are selected per
conversation; reopening a saved conversation restores that selection. Off preserves the original
all-primary-model behavior.

When enabled, compact task/UI metadata, bounded related-memory or retrieved-context snippets, and
locally shortlisted source-code previews can leave the device. Screenshots, full OCR output,
unrestricted history, ignored files, and detected secrets are excluded from the JEV request. The
first external-routing use retains the
external-data boundary; the first JEV-enabled run asks for explicit confirmation for the selected
workspace. The composer and expandable activity rows show evaluations, handoffs,
selected evidence or actions, latency, and verified outcomes.

All memory operations and optional-context retrieval work without JEV. Exact and conservative local
duplicate checks run before writes, ambiguous inferred facts remain reviewable drafts, and the
Memory library can scan, review, merge, and undo likely duplicates. Counts, arithmetic, dates, token
budgets, and version checks are always computed in Rust; JEV is used only for bounded semantic
choices. Ordinary deletion retains only an exact non-reversible fingerprint. The separate **Forget
& block similar** action clearly discloses that it retains normalized text locally so semantically
similar inferred facts can be suppressed. This split follows TypeSafe's documented
[JEV 1.13 limitations](https://docs.typesafe.ai/model-jaggedness/jev-1.13#counting): precise
counting and numerical work remain deterministic harness responsibilities.

Desktop vision targets one application window by default. For taskbar, Start,
system-tray, minimized, or cross-monitor work, `observe_desktop` returns a compact
numbered topology and up to 40 task-ranked windows. The agent then captures one
selected monitor at readable resolution before input. The responsiveness profile limits model images to a 1280-pixel longest
edge, input coordinates use those model-image pixels, and the harness maps them
back to physical Windows coordinates. Each capture fuses a cached Windows UIA
snapshot with word-level WinRT OCR and adaptive visual slider/track detection into a bounded action registry. Vision models
receive both a clean screenshot and an annotated copy using the same target ids;
tiny OCR fragments remain readable without becoming ordinary click targets.
Monitor captures reserve UIA capacity for Windows Shell controls so large application
trees cannot crowd taskbar or tray buttons out of the model view. For small or unlabeled
controls, `inspect_screen_region` enlarges a target or rectangle from the latest model image and
runs OCR on the derived crop without recapturing the desktop. OCR-only text remains a readable
anchor rather than click authority. For an unnumbered control, the vision model calls
`locate_visual_target` with a labeled bounding box; the harness returns an enlarged confirmation
crop and one-use token, and `click_localized` clicks its validated center on the next turn.
The vision model remains the primary detector; UIA/OCR can geometrically corroborate or refine
its position but are not required for execution.
Diagnostics save both the full source image with the proposed box/center overlay and the enlarged
confirmation crop under the localization id.
`move_pointer`, `drag_target`, and `drag_pointer` still map crop-local pixels back to the
authoritative observation. Bounded-duration drag remains a native input action.
The model normally calls `click_target` with the intended visible label, allowing
one unique fresh label match to correct a mistaken number while ambiguity executes
no input. Model-only bounding boxes remain available for games and canvas controls when UIA/OCR
cannot corroborate them, but direct raw coordinate clicks are rejected. Conversation construction targets 12,000
prompt tokens and retains only the latest visual observation. Interactive runs
allow up to 128 model turns. Older tool cycles are folded into a bounded
continuity ledger while the original task and six recent results stay verbatim.

Interactive turns start a small background curation pass on the same local model.
Review fact proposals in the dashboard, or use `pok-ai memory-drafts` followed by
`pok-ai memory-approve <UUID>`. A desktop procedure is promoted automatically only
when the post-input UIA refresh proves a visible state change. This applies to window navigation,
file management, browsers, settings, forms, creative tools, and communication applications.
External text submissions additionally require exact non-editable UIA/OCR evidence. Learned skills
record the reusable action pattern while replacing recipients, message contents, paths, coordinates,
and target numbers with fresh-task placeholders. They are written as `SKILL.md`, indexed in FTS5,
and can enter later sessions through normal memory recall.
Exam sessions never run the curator or learn skills.

The desktop keeps one live conversation for the selected model and workspace. Follow-up prompts
reuse the complete message/tool history and the same submission ledger; older cycles are compacted
into the continuity ledger as the conversation grows. Protocol-safe checkpoints are also stored in
`conversations.db` under the resolved data directory. Select a conversation in the left sidebar
after restarting the app; POK-Ai restores the saved provider, model, and workspace while
discarding stale desktop authority and cached approvals. The CLI provides `pok-ai sessions list`,
`pok-ai run --continue "follow-up"`, and `pok-ai run --session <UUID> "follow-up"`. Changing
model/workspace starts a clean conversation automatically, and **New conversation** resets it
explicitly. Existing diagnostic traces are offered as reconstructed legacy conversations. Approved
memories and learned skills remain available across new conversations and application restarts.
Background LLM curation is idle-debounced and cancelled by a follow-up so it never competes with
the active local-model request; deterministic verified skills are still saved immediately.

The desktop presents messages and compact live tool steps in one conversation. Expand a step
to inspect command output, file changes, or results; reasoning and the task plan also expand on
demand. The composer supports guidance during a run, pause/resume, and emergency stop. Scrolling
back stops automatic following; **Jump to latest** returns to the live output. The sidebar searches
the latest 50 conversations, grouped by workspace. Restored conversations show messages; historical
tool activity remains in session diagnostics.

Open **Settings → Appearance** to choose **POK dark teal** (default), **Neutral dark**, or **Light**.
The choice is saved on this device. Provider configuration and advanced model controls live in the
settings drawer; memory and generated tools are accessible from the sidebar. **Session details**
contains token usage, context budgeting, compaction controls, and decision-router diagnostics.
Frontend regression checks run with `npm --prefix apps/desktop test`.

When the same checkout is used from WSL and native Windows, npm may retain the
Linux native optional packages and omit the Windows packages required by
Rollup, esbuild, and the Tauri CLI. The Windows setup, check, and launcher
scripts detect and repair those packages using the versions recorded in
`package-lock.json`.

Temporal context follows the prompt-cache-friendly pattern used by the reference harnesses: the
session prefix contains date-only local context, while a compact tail reminder is added if midnight
passes during a long run. Relative phrases such as “today,” “tomorrow,” and “next Monday” therefore
have a deterministic local anchor. “Latest” online claims still require browser/search verification.

## Safety

The emergency stop is available through `Ctrl+Alt+Esc`, the dashboard, and
Ctrl-C in the CLI. The dashboard streams and auto-scrolls model reasoning,
response text, and tool activity so the current action remains visible. Token deltas are
coalesced into 50 ms batches before crossing the Tauri bridge instead of forcing a React render
for every token. Auto-follow pauses when the operator scrolls away from the bottom, and the live
console bounds rendered history while the full JSONL trace remains on disk. Development builds
use optimized dependencies so screenshot resizing and PNG encoding remain responsive under
`tauri dev`. Input
automation still rejects elevated windows, password controls, stale captures,
and coordinates outside the selected window or monitor. Desktop overviews never
authorize input. Exam automation is limited to disposable fixture
repositories and the titled POK-Ai exam window. Every run writes a JSONL trace
and artifacts under `%LOCALAPPDATA%\POK-Ai`.

Single keys and plus-separated shortcuts are supported through the same input tool,
including `Windows+Shift+RightArrow` for moving the active window between monitors.
Self-authored helpers remain project-scoped. Promoted Python, Node, and PowerShell helpers use
tool-local dependency environments, locked environment metadata, and required smoke tests.
Creating or executing code uses the normal file/process policy, while desktop input stays autonomous.

See `docs/LM_STUDIO_API.md` for the current LM Studio endpoint, tool-history,
reasoning-stream, and compatibility decisions.

Developer diagnostics are intentionally not displayed in the application UI.
Development runs store them under `diagnostics/sessions/<session-id>` in this
repository. Each successful capture saves the exact model PNG, a numbered model
PNG, OCR text, fused target JSON, observation JSON without duplicated image
base64, timings, and an OCR/UIA SVG overlay. Embedded image base64 is excluded
from traces and JSON to keep the bundle readable.

Provider entries declare `data_boundary = "local_device"` or
`"external_service"`. Local models receive full-fidelity context. The desktop
requires confirmation before the first inference request to an external
provider in a conversation; confirmed requests receive the same unmasked task
context.

The Generated tools panel can disable or permanently delete an installed
project tool. Deletion removes only its promoted AppData copy and isolated
environment; it does not delete the original helper source from the workspace.

## Support the Project

If you find POK-Ai useful and would like to support its development, you can
buy me a coffee. Your support helps me continue maintaining and improving the
project.

[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://www.buymeacoffee.com/acekorneyab)

Your contributions will help support:

- New features and enhancements
- Maintenance, testing, and bug fixes
- Continued development of POK-Ai

Thank you for your support!

## License and attribution

POK-Ai is free and open-source software licensed under the
[Apache License 2.0](LICENSE). You may use, modify, and redistribute it,
including for commercial purposes, subject to that license.

Redistributions and derivative works must include the Apache 2.0 license and
retain the POK-Ai attribution in [NOTICE](NOTICE). Contributions back to POK-Ai
are welcomed and appreciated, but they are not required by the license.
