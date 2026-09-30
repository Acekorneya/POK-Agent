use std::{
    collections::{BTreeMap, BTreeSet, HashSet, VecDeque},
    fs::OpenOptions,
    io::Write,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use chrono::{DateTime, Local, Utc};
use futures::StreamExt;
use parking_lot::Mutex;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::{
    PokError, Result,
    brain::{
        Brain, BrainEvent, BrainMessage, BrainRequest, CompletedToolCall, MessageContent,
        MessageOrigin,
    },
    builtins::{FastActionsArgs, FastOperation, FastSubgoal},
    context::{automatic_recent_tail_target, automatic_working_set_target},
    conversation_store::{
        ConversationSnapshot, ConversationStore, ConversationSummary, sanitized_persisted_messages,
    },
    decision::{
        DecisionCandidate, DecisionCandidateKind, DecisionPurpose, DecisionRequest, DecisionRouter,
        coding_candidates, desktop_candidates_for_task, harness_capability_candidates,
        indexed_workspace_context_candidates, rank_and_limit_candidates,
        workspace_code_context_candidates,
    },
    memory::{MemoryWriteProvenance, NewProcedure, ProcedureKind, ProcedureStep},
    pause::PauseState,
    policy::PolicyMode,
    retrieval::RetrievalIntent,
    tool::{
        ActiveTaskState, TaskItem, TaskItemStatus, ToolContext, ToolRegistry, VerifiedSubmission,
    },
    types::{
        ArtifactReference, CompletionStatus, CompletionWarning, Observation, RunMetrics,
        RunSummary, SessionEvent, WindowInfo,
    },
};

mod context_window;
mod continuity;
mod curation;
mod decision_router;
mod environment;
mod events;
mod fast_actions;
mod intent;
mod model_output;
mod motor_program;
mod run_loop;
mod skill_replay;
mod tool_results;

use context_window::*;
use continuity::*;
use curation::*;
use decision_router::*;
use environment::*;
use events::*;
use fast_actions::*;
use intent::*;
use model_output::*;
use motor_program::*;
use skill_replay::*;
use tool_results::*;

pub use events::{AgentEvent, SessionObserver};

const CURRENT_VISUAL_PAIR_IMAGES: usize = 2;
const IMAGE_PAYLOAD_BUDGET_BYTES: usize = 20 * 1024 * 1024;

/// How to handle missing information when the user can answer questions.
const INTERACTIVE_QUESTIONS: &str = "When missing information, ambiguity, or a user preference materially changes the next action, call ask_user_question rather than ending the run with a plain-text questionnaire. Group all currently known related questions in one call, ask additional rounds when genuinely needed, and offer concise choices with the recommended option first when practical. Do not ask for information already available from context or tools, and do not repeat answered or dismissed questions.";

/// The same, when no one can answer during the run (the CLI or an
/// unattended run): asking would only end the run with nothing done.
const UNATTENDED_QUESTIONS: &str = "No one can answer questions during this run. When details are ambiguous, act on the most reasonable reading of the request and the visible state, and state what you assumed in your final answer. Do not stop to ask; stop without acting only when every reasonable reading would be unsafe or impossible.";

const SYSTEM_PROMPT: &str = r#"You are POK-Ai, a Windows computer-use and coding agent.
Use only the supplied tools. Observe the desktop before input. Desktop tools use the latest observation;
copy the short observation_id only when needed and never invent or repair an id.
{QUESTIONS}
Capture returns numbered interaction targets fused from UI Automation, OCR, and vision. The clean image
shows the unobscured desktop and the annotated image uses the same ids as action_targets. Prefer the
task-ranked relevant_targets shortlist. For click_target, provide both the listed target_id and the visible
expected_label you intend to activate; the runtime will execute only a matching or uniquely corrected fresh
target. Inspect an ambiguous target instead of guessing. Never use simulate_input for coordinate clicks.
A text target (role "text", often a menu item or label read by OCR) that is marked actionable can be clicked with
click_target and its exact label; use that before locating it visually.
For an unlabeled visible control, call locate_visual_target with its labeled bounding box. Inspect the enlarged
confirmation crop, then call click_localized with the returned one-use localization_id; never guess or directly
click raw coordinates. UIA/OCR may corroborate the model box but are not required for games or canvases.
Use move_pointer for hover-only controls and drag_target or drag_pointer for sliders, splitters, and other
direct-manipulation controls. Derived views are crops of the current authoritative observation, not new desktop
states. Post-action tool results already contain the newest authoritative observation, so reuse them without a
duplicate capture; omit observation_id when necessary rather than copying an older id.
For reading tasks, prefer primary_content when present: it contains compact semantic messages, rows,
or list items and omits unrelated navigation and sidebars. Use ordered_content only when primary_content
is empty or the task requires the wider page context.
You have full Windows keyboard and mouse input: left, middle, and right clicks; pointer movement;
wheel scrolling; individual keys; and plus-separated keyboard shortcuts. Supported named keys include Enter,
Escape, Tab, Backspace, Delete, Insert, arrows, PageUp/PageDown, Home/End, Windows, modifiers, and F1-F12.
Use click_target with button=left, middle, or right for known controls. For page, list, dialog, and POS navigation, use scroll_view with semantic
direction/amount. When you know the label or text being sought, prefer scroll_until_text; both scrolling tools
return a fresh grounded observation, so do not immediately call capture_screen again. Never invent raw wheel distances.
For nested lists or sidebars, supply the matching target_id. If an untargeted scroll cannot infer one unique
task region, use suggested_scroll_targets from its result rather than repeating the same global scroll.
For multi-step input sequences (e.g. click address bar -> type query -> press Enter), use `execute_action_batch`
to execute all steps in sequence against the latest observation. The runtime verifies intermediate state internally
but returns only the final visual observation, so do not request duplicate captures between batch steps.
Its step kinds are `click_target` with a fresh target_id and expected_label, `type_text` with the complete string, and `key`
with a named key or shortcut. Send complete text in one type_text step; the runtime injects and verifies it
as a bulk operation, so never split prose into character or word calls. On Windows, `run_command` executes
PowerShell. Prefer it for deterministic filesystem and directory operations, path checks, process inspection,
application launch, documented application automation, artifact operations, and final verification.
Pass the PowerShell command body directly; never nest another `powershell -Command` invocation.
`run_command` streams output while it runs. If it returns status `running`, the same process is still alive:
use `manage_command` with poll, wait, read, write, submit, close, or kill and the returned task_id instead
of launching the command again. Use background=true only for intentionally long-lived work; use pty=true
when interactive terminal input is required. Before claiming a foreground task is complete, wait for its
terminal status and check its exit code; read additional captured output by cursor when the preview is truncated.
Use desktop tools for browser/GUI-only work and whenever the user explicitly requests interaction in a named
application. In that case the requested application must still be opened and the final result verified there,
but deterministic bulk content or save operations may use documented application APIs instead of slow manual
keystrokes and dialogs. Commands may act only within the current policy's authorized filesystem scope.
Before working around a request, check that the requested feature, file, setting, or state actually exists here.
If it does not, or the request cannot be done in this environment, say so plainly instead of substituting something different.
When the user asks for help with a document, spreadsheet, or other file without naming a different one, they mean the
file they have open, even when its current content looks unrelated: do the work in that file, keep its name and format,
and do not start a new file unless they ask for one. Likewise, a setting or preference they describe without naming an
application belongs to the application they have open in front, not to Windows or another program.
When a document is open in an application, change it in that application, or close it without saving before editing its
file with a command or file tool, then reopen it to verify; otherwise the application's open copy can overwrite the edit.
In spreadsheets and other grids, move with the cell-reference (Name) box and the keyboard instead of clicking cells: type a
cell or range such as B2 or A1:C1 into the Name Box (Ctrl+Shift+F5 in LibreOffice Calc; Ctrl+G or F5 in Excel), then type
values or formulas and use Tab, Enter, and arrows. To check cell contents after a change, select the range, press Ctrl+C,
and call read_clipboard, which returns the copied cells as rows. For bulk content changes, a script or the application's
automation interface is usually faster than cell-by-cell input; verify the result in the application afterwards.
Distinguish requests for information from requests to act. A question such as "is there a way" should be
answered with options and should not trigger probing, scanning, configuration, or other external actions unless
the user also asks you to investigate, set up, or perform them.
After repeated verified no-progress UI actions, change strategy and use command-assisted
setup or verification when this policy permits it.
When the user explicitly asks to remember a fact or detail, or when you learn durable environment context, save it with `remember_fact`.
Never request passwords or UAC/secure-desktop interaction, and never bypass the current policy decision. For code changes,
inspect first, make exact minimal edits, run relevant tests, and repair failures. For large repositories or files,
map and search first, call source_outline, then page through only relevant read_file ranges. Continue with
next_start_line when full traversal is required. Treat summaries and outlines as navigation aids: re-read the
exact current source range and confirm its sha256 before editing or making a final code claim. Use list_windows and
activate_window to select the task application. For taskbar, Start, system tray, minimized, launching,
or cross-monitor work, call observe_desktop, choose a monitor_id, then capture_screen with scope=monitor
before input. When an icon or control is too small or unlabeled, call inspect_screen_region with the latest
observation id and a model-image rectangle to enlarge it and rerun OCR/UIA. Never click the overview. Capture one active/selected window by default and narrow back to
the application after it opens. Coordinates returned by capture and accepted by simulate_input are
pixels in the model image. A successful input result reports the original model point; physical desktop
coordinates remain internal, so never reinterpret a mapped point as an error. Number labels in the annotated image
match the targets array and click_target uses physical coordinates safely. Desktop activation and input execute
autonomously; the user can stop control globally with Ctrl+Alt+Esc. After sending or submitting text,
trust submission.status=verified and stop. If status is uncertain, capture once to verify before retrying;
never repeat a verified or duplicate-suppressed submission. Keyboard chords use kind `keyboard_shortcut`
and a plus-separated key such as `Windows+Shift+RightArrow`. If no built-in tool can complete a task,
you may create a small helper inside the approved workspace with write_file, execute it with run_command,
inspect its output, and revise it with edit_file. After a helper succeeds and would be useful again, you may
explicitly call promote_helper_tool, which requires user approval; never promote untested code or imply that merely running a helper saved it.
Use search_generated_tools and invoke_generated_tool to reuse promoted project tools. Prefer documented OS APIs and never invent commands or
installed modules. Generated helpers receive one JSON value on stdin and must return exactly one JSON value on stdout.
Declare every runtime package during promotion; never make a helper install its own dependencies. For PDFs and
other documents, treat flattened forms and tables as potentially unordered. Do not claim a field/value,
checkbox, or column relationship unless layout coordinates, form metadata, or another structural signal
supports it. State uncertainty or use a layout-aware fallback when structure is ambiguous. Empty extracted
text from an image-only document means OCR is required, not that the document is empty. Do not repeat complete
personal identifiers when they are irrelevant, but preserve exact values when the user's task requires them.
For context menus, use click_target with button `right` directly from a fresh capture;
do not select first or invent mouse-button-down actions. For ordinary web work, prefer the isolated
managed_browser tools: they return structured DOM evidence and use freshness-checked native browser input.
Use browser_navigate with a complete window id only when the user explicitly asks to work in their existing
personal browser; it activates that browser and submits the address atomically. Finish with a concise result."#;

const TEXT_ONLY_GROUNDING_NOTE: &str = "This model does not accept images. Desktop captures are still processed locally and represented as compact numbered OCR/UI Automation targets. Use click_target whenever possible and do not ask for image input.";
const MAX_UNCHANGED_PLAN_UPDATES: u32 = 3;

/// How long a model stream may stay silent (no text, reasoning, tool-call, or
/// usage event) before the request is treated as stalled and retried. Long
/// enough for a local model's prompt processing or a reasoning model's
/// thinking, which stream progress events.
pub(super) const PROVIDER_STREAM_IDLE_LIMIT: Duration = Duration::from_secs(300);

pub struct Session {
    pub id: Uuid,
    brain: Arc<dyn Brain>,
    brain_usage: Arc<crate::brain::BrainUsageCounter>,
    tools: Arc<ToolRegistry>,
    pub context: ToolContext,
    model: String,
    max_turns: u32,
    agent_temperature: Option<f32>,
    agent_seed: Option<u64>,
    reasoning_effort: Option<String>,
    bounded_reasoning_effort: Option<String>,
    full_context: bool,
    messages: Vec<BrainMessage>,
    trace: Mutex<std::fs::File>,
    observer: Option<Arc<dyn SessionObserver>>,
    usage_scale: f64,
    image_input_override: Option<bool>,
    single_system_message_required: bool,
    alternating_roles_required: bool,
    schema_references_unsupported: bool,
    temporal_anchor: TemporalAnchor,
    curation_cancellation: CancellationToken,
    /// Maximum planner turns that act on the computer (a `fast_actions` plan
    /// is one step, however many actions System 1 runs); `None` is unlimited.
    action_step_budget: Option<u32>,
    /// Background learning and curation started after verified runs.
    curation_tasks: Mutex<Vec<tokio::task::JoinHandle<()>>>,
    manual_compaction_requested: bool,
    last_prompt_tokens: Option<u64>,
    average_turn_growth: f64,
    compaction_count: usize,
    consecutive_compaction_failures: u8,
    compaction_cooldown_remaining: u8,
    context_growth_samples: u32,
    pending_clarification: Option<PendingClarification>,
    continuity: ExecutionContinuity,
    last_model_message_hashes: Vec<String>,
    terminal_logged: bool,
    response_max_tokens: u32,
    active_tool_groups: BTreeSet<String>,
    decision_router: Option<Arc<dyn DecisionRouter>>,
    decision_router_config: Option<crate::config::DecisionRouterConfig>,
    /// Opt-in log of router questions and proven answers for training.
    training: Option<Arc<crate::router_training::TrainingRecorder>>,
    decision_router_enabled: bool,
    decision_router_backend: crate::config::DecisionRouterBackend,
    decision_router_failures: u8,
    decision_router_actions: HashSet<String>,
    decision_router_no_progress: BTreeMap<String, u8>,
    decision_router_suppressed_candidates: HashSet<String>,
    decision_router_completed_observations: HashSet<String>,
    decision_router_tool_failures: BTreeMap<String, HashSet<String>>,
    decision_router_cache_key: Option<String>,
    decision_context_cache: BTreeMap<String, Vec<String>>,
    retrieval_intent_cache: Option<(String, RetrievalIntent)>,
    decision_router_fresh_evidence: bool,
    decision_router_last_result: Option<(String, Value)>,
    decision_router_ambiguity_refreshes: u8,
    decision_router_refinement: DecisionRefinementState,
    /// Applications successfully launched during this run, so the launch
    /// candidate is offered at most once per application per run.
    decision_router_launched_applications: HashSet<String>,
    /// Delegated mode withholds raw navigation tools until a `fast_actions`
    /// run hands control back without completing its subgoal.
    raw_navigation_unlocked: u8,
    /// A step System 1 handed back, waiting for the primary model's
    /// resolving action to become a training label (see `record_handback`).
    pending_handback: Option<Value>,
    /// Set while System 1 replays a skill: only exact quoted labels are
    /// clicked, and any other screen hands back instead of a model pick.
    skill_replay_active: bool,
    /// The motor steps (clicks by label, keys, text) this run performed, in
    /// order; a verified run saves them on its skill as a program.
    motor_tape: Vec<Value>,
    /// Images the user attached to the next request (PNG, base64).
    pending_user_images: Vec<String>,
    /// The last observation shown in the dashboard's agent view.
    shown_observation: Option<Uuid>,
    /// A skill whose stored program was replayed this run and stopped before
    /// its end: the program this run records replaces it.
    stale_program: Option<Uuid>,
    /// Whether this run's completion was clean (no result check had to be
    /// retried, no warnings accepted): only such a run leaves a program.
    clean_completion: bool,
    /// Reused judge verdicts: candidate fingerprint + task + current_step ->
    /// the last verdict the judge returned for that exact input. A repeated
    /// candidate across re-observations (the same window re-proposed after a
    /// capture that did not change the state) is re-judged with the identical
    /// payload and answer, so the second and later calls are pure waste.
    /// Keyed on a SHA-256 of the exact judge payload inputs so a different
    /// step or a materially different description always re-judges.
    decision_router_judge_verdicts:
        std::collections::BTreeMap<String, crate::decision::JudgeVerdict>,
    decision_activity: Vec<DecisionActivitySummary>,
    environment_revision: u64,
    conversation_store: Option<Arc<ConversationStore>>,
    conversation_provider: String,
    conversation_title: String,
    conversation_created_at: String,
    legacy_imported: bool,
}

impl Session {
    pub fn cache_approval(&self, key: impl Into<String>) {
        self.context.approval_cache.lock().insert(key.into());
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PendingClarification {
    root_request: String,
    question: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct PersistedSessionState {
    #[serde(default)]
    active_task: ActiveTaskState,
    #[serde(default)]
    input_ledger: crate::tool::InputLedger,
    #[serde(default)]
    artifact_evidence: Vec<crate::tool::ArtifactEvidence>,
    #[serde(default)]
    session_files: BTreeMap<std::path::PathBuf, crate::tool::SessionFileRecord>,
    #[serde(default)]
    pending_clarification: Option<PendingClarification>,
    #[serde(default)]
    continuity: ExecutionContinuity,
    #[serde(default)]
    active_tool_groups: BTreeSet<String>,
    #[serde(default)]
    last_prompt_tokens: Option<u64>,
    #[serde(default)]
    average_turn_growth: f64,
    #[serde(default)]
    compaction_count: usize,
    #[serde(default)]
    decision_router_enabled: bool,
    #[serde(default)]
    decision_router_backend: crate::config::DecisionRouterBackend,
    #[serde(default)]
    decision_activity: Vec<DecisionActivitySummary>,
    #[serde(default)]
    environment_revision: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DecisionActivitySummary {
    purpose: String,
    label: String,
    detail: String,
    elapsed_ms: u64,
    recorded_at: String,
}

impl Session {
    pub fn new(
        brain: Arc<dyn Brain>,
        tools: Arc<ToolRegistry>,
        context: ToolContext,
        model: String,
        max_turns: u32,
    ) -> Result<Self> {
        let counting_brain = crate::brain::CountingBrain::new(brain);
        let brain_usage = counting_brain.counter();
        let brain: Arc<dyn Brain> = Arc::new(counting_brain);
        std::fs::create_dir_all(&context.artifact_dir)?;
        let trace = OpenOptions::new()
            .create(true)
            .append(true)
            .open(context.artifact_dir.join("trace.jsonl"))?;
        let temporal_anchor = current_temporal_anchor();
        let questions = if context.questions.is_some() {
            INTERACTIVE_QUESTIONS
        } else {
            UNATTENDED_QUESTIONS
        };
        let mut system_prompt = format!(
            "{}\n\n{}",
            SYSTEM_PROMPT.replace("{QUESTIONS}", questions),
            temporal_anchor.context
        );
        system_prompt.push_str("\n\n");
        system_prompt.push_str(authorization_context(&context.policy.mode));
        system_prompt.push_str(
            "\n\n<tool_capabilities>Tool schemas are exposed progressively to reduce local-model prompt latency. The complete generic capability families are: desktop (observe and operate Windows applications), system (commands and process management), coding (files and artifacts), memory (durable facts and reusable skills), archive (older session evidence), generated (reusable helpers), and subagent (isolated coding work). A compact catalog on every turn lists all installed tools and whether each is active. If a needed tool is inactive, call discover_tools for its group; this changes schema availability only and never weakens safety policy.</tool_capabilities>",
        );

        // Load global user profile USER.md if it exists, or create a default one
        let user_md_path = context.data_dir.join("USER.md");
        if !user_md_path.exists() {
            let default_profile = "# User Profile\n\n\
                This file contains durable user preferences, environment details, and identity facts for POK-Ai.\n\n\
                - **Preferred Shell:** PowerShell (on Windows), bash (on WSL)\n\
                - **Programming Languages:** Rust, TypeScript, React\n\
                - **Local Servers:** LM Studio on 127.0.0.1:1234\n";
            let _ = std::fs::write(&user_md_path, default_profile);
        }
        let mut user_profile_content = String::new();
        if let Ok(content) = std::fs::read_to_string(&user_md_path) {
            user_profile_content.push_str(content.trim());
        }
        if let Ok(records) = context.memory.list_approved("user", 50) {
            if !records.is_empty() {
                if !user_profile_content.is_empty() {
                    user_profile_content.push_str("\n\n## Learned User Facts:\n");
                }
                for record in records {
                    user_profile_content.push_str(&format!("- {}\n", record.text.trim()));
                }
            }
        }
        if !user_profile_content.is_empty() {
            system_prompt.push_str(&format!(
                "\n\n<user_profile>\n{}\n</user_profile>",
                user_profile_content.trim()
            ));
        }

        // Load workspace-specific PROJECT.md if it exists (equivalent to CLAUDE.md)
        let project_md_path = context.workspace.join("PROJECT.md");
        if let Ok(content) = std::fs::read_to_string(&project_md_path) {
            if !content.trim().is_empty() {
                system_prompt.push_str(&format!(
                    "\n\n<project_profile>\n{}\n</project_profile>",
                    content.trim()
                ));
            }
        }
        let strict_role_layout = model_requires_strict_role_layout(&model);
        let response_max_tokens = initial_response_max_tokens(&model);

        let session = Self {
            id: context.session_id,
            brain,
            brain_usage,
            tools,
            context,
            model,
            max_turns,
            agent_temperature: Some(0.0),
            agent_seed: Some(42),
            reasoning_effort: None,
            bounded_reasoning_effort: None,
            full_context: false,
            messages: vec![BrainMessage::text("system", system_prompt)],
            trace: Mutex::new(trace),
            observer: None,
            usage_scale: 1.0,
            image_input_override: None,
            single_system_message_required: strict_role_layout,
            alternating_roles_required: strict_role_layout,
            schema_references_unsupported: false,
            temporal_anchor,
            curation_cancellation: CancellationToken::new(),
            action_step_budget: None,
            curation_tasks: Mutex::default(),
            manual_compaction_requested: false,
            last_prompt_tokens: None,
            average_turn_growth: 0.0,
            compaction_count: 0,
            consecutive_compaction_failures: 0,
            compaction_cooldown_remaining: 0,
            context_growth_samples: 0,
            pending_clarification: None,
            continuity: ExecutionContinuity::default(),
            last_model_message_hashes: Vec::new(),
            terminal_logged: false,
            response_max_tokens,
            active_tool_groups: BTreeSet::from([
                "control".into(),
                "desktop".into(),
                "browser".into(),
            ]),
            decision_router: None,
            decision_router_config: None,
            training: None,
            decision_router_enabled: false,
            decision_router_backend: crate::config::DecisionRouterBackend::Off,
            decision_router_failures: 0,
            decision_router_actions: HashSet::new(),
            decision_router_no_progress: BTreeMap::new(),
            decision_router_suppressed_candidates: HashSet::new(),
            decision_router_completed_observations: HashSet::new(),
            decision_router_tool_failures: BTreeMap::new(),
            decision_router_cache_key: None,
            decision_context_cache: BTreeMap::new(),
            retrieval_intent_cache: None,
            decision_router_fresh_evidence: false,
            decision_router_last_result: None,
            decision_router_ambiguity_refreshes: 0,
            decision_router_refinement: DecisionRefinementState::default(),
            decision_router_launched_applications: HashSet::new(),
            raw_navigation_unlocked: 0,
            pending_handback: None,
            skill_replay_active: false,
            motor_tape: Vec::new(),
            pending_user_images: Vec::new(),
            shown_observation: None,
            stale_program: None,
            clean_completion: false,
            decision_router_judge_verdicts: std::collections::BTreeMap::new(),
            decision_activity: Vec::new(),
            environment_revision: 0,
            conversation_store: None,
            conversation_provider: String::new(),
            conversation_title: String::new(),
            conversation_created_at: Utc::now().to_rfc3339(),
            legacy_imported: false,
        };
        session.install_user_activity_listener();
        Ok(session)
    }

    pub fn with_conversation_store(
        mut self,
        store: Arc<ConversationStore>,
        provider: impl Into<String>,
    ) -> Result<Self> {
        self.conversation_store = Some(store);
        self.conversation_provider = provider.into();
        Ok(self)
    }

    pub fn restore_conversation(&mut self, snapshot: ConversationSnapshot) -> Result<()> {
        if snapshot.summary.model != self.model
            || snapshot.summary.workspace != self.context.workspace
        {
            return Err(PokError::Other(anyhow::anyhow!(
                "conversation {} belongs to model {} and workspace {}; construct the session with those saved settings before restoring it",
                snapshot.summary.id,
                snapshot.summary.model,
                snapshot.summary.workspace.display()
            )));
        }

        let fresh_system = self.messages.first().cloned();
        self.id = snapshot.summary.id;
        self.context.session_id = snapshot.summary.id;
        self.conversation_provider = snapshot.summary.provider;
        self.conversation_title = snapshot.summary.title;
        self.conversation_created_at = snapshot.summary.created_at;
        self.legacy_imported = snapshot.summary.legacy_imported;
        self.messages = snapshot.messages;
        if !self.messages.iter().any(|message| message.role == "system")
            && let Some(system) = fresh_system
        {
            self.messages.insert(0, system);
        }
        let repair = repair_tool_call_result_pairs(&mut self.messages, "conversation_restore");
        if repair.changed() {
            self.log("tool_pair_history_repaired", serde_json::to_value(&repair)?)?;
        }

        if !snapshot.state.is_null() && snapshot.state != json!({}) {
            let state: PersistedSessionState = serde_json::from_value(snapshot.state)?;
            *self.context.active_task.lock() = state.active_task;
            *self.context.input_ledger.lock() = state.input_ledger;
            *self.context.artifact_evidence.lock() = state.artifact_evidence;
            *self.context.session_files.lock() = state.session_files;
            self.pending_clarification = state.pending_clarification;
            self.continuity = state.continuity;
            if !state.active_tool_groups.is_empty() {
                self.active_tool_groups = state.active_tool_groups;
            }
            self.last_prompt_tokens = state.last_prompt_tokens;
            self.average_turn_growth = state.average_turn_growth;
            self.compaction_count = state.compaction_count;
            self.decision_router_enabled = state.decision_router_enabled;
            self.decision_router_backend = state.decision_router_backend;
            self.decision_activity = state.decision_activity;
            self.environment_revision = state.environment_revision;
        }

        // External UI state and approval decisions cannot safely survive a process restart.
        *self.context.latest_observation.lock() = None;
        *self.context.latest_observation_view.lock() = None;
        *self.context.pending_visual_localization.lock() = None;
        *self.context.focused_control.lock() = None;
        self.context.approval_cache.lock().clear();
        self.messages.push(BrainMessage::text_with_origin(
            "system",
            "This conversation was restored from durable storage. Treat all prior desktop observations as stale, re-observe before any input, and request any approval again when policy requires it.",
            MessageOrigin::SystemReminder,
        ));
        self.persist_conversation("active")
    }

    pub fn persist_conversation(&self, status: &str) -> Result<()> {
        let Some(store) = &self.conversation_store else {
            return Ok(());
        };
        let state = PersistedSessionState {
            active_task: self.context.active_task.lock().clone(),
            input_ledger: self.context.input_ledger.lock().clone(),
            artifact_evidence: self.context.artifact_evidence.lock().clone(),
            session_files: self.context.session_files.lock().clone(),
            pending_clarification: self.pending_clarification.clone(),
            continuity: self.continuity.clone(),
            active_tool_groups: self.active_tool_groups.clone(),
            last_prompt_tokens: self.last_prompt_tokens,
            average_turn_growth: self.average_turn_growth,
            compaction_count: self.compaction_count,
            decision_router_enabled: self.decision_router_enabled,
            decision_router_backend: self.decision_router_backend,
            decision_activity: self.decision_activity.clone(),
            environment_revision: self.environment_revision,
        };
        let now = Utc::now().to_rfc3339();
        store.checkpoint(&ConversationSnapshot {
            summary: ConversationSummary {
                id: self.id,
                title: if self.conversation_title.trim().is_empty() {
                    "New conversation".into()
                } else {
                    self.conversation_title.clone()
                },
                provider: self.conversation_provider.clone(),
                model: self.model.clone(),
                workspace: self.context.workspace.clone(),
                artifact_dir: self.context.artifact_dir.clone(),
                status: status.into(),
                legacy_imported: self.legacy_imported,
                created_at: self.conversation_created_at.clone(),
                updated_at: now,
            },
            messages: sanitized_persisted_messages(&self.messages),
            state: serde_json::to_value(state)?,
        })
    }

    /// Wait up to `limit` for skill learning and memory curation from
    /// finished runs. A short-lived process (the CLI) calls this before it
    /// exits so what a run learned is saved; the desktop app never needs to.
    pub async fn finish_background_work(&self, limit: std::time::Duration) {
        let tasks = std::mem::take(&mut *self.curation_tasks.lock());
        let _ = tokio::time::timeout(limit, futures::future::join_all(tasks)).await;
    }

    pub fn with_observer(mut self, observer: Arc<dyn SessionObserver>) -> Self {
        self.context
            .command_manager
            .set_event_sink(Arc::new(ObserverCommandSink {
                observer: observer.clone(),
            }));
        self.observer = Some(observer);
        self.install_user_activity_listener();
        self
    }

    /// Limit the planner turns that act on the computer. Benchmarks such as
    /// Windows Agent Arena cap agents at a fixed number of steps; the planner
    /// sees the steps it has left and must finish within them.
    pub fn with_action_step_budget(mut self, budget: Option<u32>) -> Self {
        self.action_step_budget = budget.filter(|steps| *steps > 0);
        self
    }

    /// Standing instructions appended to the system prompt (for example a
    /// benchmark's convention for reporting an impossible task).
    pub fn with_standing_instructions(mut self, instructions: Option<&str>) -> Self {
        if let Some(text) = instructions.map(str::trim).filter(|text| !text.is_empty())
            && let Some(first) = self.messages.first_mut()
        {
            first.content.push(MessageContent::Text {
                text: format!("\n\nStanding instructions:\n{text}"),
            });
        }
        self
    }

    pub fn with_temperature(mut self, temperature: Option<f32>) -> Self {
        self.agent_temperature = temperature;
        self
    }

    pub fn configure_model_request(
        &mut self,
        temperature: Option<f32>,
        max_output_tokens: Option<u32>,
        seed: Option<u64>,
        reasoning_effort: Option<String>,
        image_input_override: Option<bool>,
        full_context: bool,
    ) {
        self.agent_temperature = temperature;
        self.response_max_tokens = max_output_tokens
            .map(|tokens| tokens.clamp(256, 65_536))
            .unwrap_or_else(|| initial_response_max_tokens(&self.model));
        self.agent_seed = seed;
        self.reasoning_effort = reasoning_effort.filter(|value| !value.trim().is_empty());
        self.image_input_override = image_input_override;
        self.full_context = full_context;
    }

    pub fn matches_conversation(&self, model: &str, workspace: &std::path::Path) -> bool {
        self.model == model && self.context.workspace == workspace
    }

    pub fn set_policy_mode(&mut self, mode: PolicyMode) {
        if self.context.policy.mode == mode {
            return;
        }
        self.context.policy.mode = mode;
        self.context.approval_cache.lock().clear();
        let replacement = authorization_context(&self.context.policy.mode);
        for message in &mut self.messages {
            for content in &mut message.content {
                if let MessageContent::Text { text } = content
                    && replace_authorization_context(text, replacement)
                {
                    return;
                }
            }
        }
        self.messages.push(BrainMessage::text_with_origin(
            "system",
            replacement,
            MessageOrigin::SystemReminder,
        ));
    }

    pub fn prepare_follow_up(&mut self, cancellation: tokio_util::sync::CancellationToken) {
        self.context.cancellation = cancellation;
        self.context.pause.reset();
        *self.context.latest_observation.lock() = None;
        *self.context.latest_observation_view.lock() = None;
        *self.context.pending_visual_localization.lock() = None;
    }

    pub fn request_compaction(&mut self) {
        self.manual_compaction_requested = true;
    }

    /// Run a request with images the user attached (PNG, base64), for a
    /// vision model to look at. At most [`MAX_USER_IMAGES`] are accepted and
    /// each must be a PNG no larger than [`MAX_USER_IMAGE_BYTES`].
    pub async fn run_with_images(
        &mut self,
        prompt: impl Into<String>,
        images: Vec<String>,
    ) -> Result<RunSummary> {
        self.pending_user_images = validated_user_images(images)?;
        self.run(prompt).await
    }

    /// Run a request with images (for a vision model) and files or folders
    /// the user attached. Attached paths become readable for the agent even
    /// outside the workspace, and the request lists them so the agent opens
    /// them with its file tools or a helper it writes.
    pub async fn run_with_attachments(
        &mut self,
        prompt: impl Into<String>,
        images: Vec<String>,
        files: Vec<String>,
    ) -> Result<RunSummary> {
        let mut prompt = prompt.into();
        let attached = attached_file_paths(&files)?;
        if !attached.is_empty() {
            prompt.push_str(&attached_files_block(&attached));
            self.grant_attached_paths(attached);
        }
        self.pending_user_images = validated_user_images(images)?;
        self.run(prompt).await
    }

    fn grant_attached_paths(&mut self, paths: Vec<std::path::PathBuf>) {
        let mut granted = self.context.attached_paths.lock();
        for path in paths {
            if !granted.contains(&path) {
                granted.push(path);
            }
        }
    }

    pub async fn run(&mut self, prompt: impl Into<String>) -> Result<RunSummary> {
        self.terminal_logged = false;
        let prompt = prompt.into();
        // A file or folder the user names by its full path may be read.
        let named = paths_named_in(&prompt);
        self.grant_attached_paths(named);
        let result = self.run_inner(prompt).await;
        if let Err(error) = &result
            && !self.terminal_logged
        {
            let kind = if matches!(error, PokError::Cancelled) {
                "run_cancelled"
            } else {
                "run_failed"
            };
            let _ = self.log(kind, json!({"error": error.to_string()}));
            self.terminal_logged = true;
            if !matches!(error, PokError::Cancelled) {
                self.emit(AgentEvent::RunFailed {
                    error: error.to_string(),
                });
            }
        }
        if result.is_err() {
            let repair = repair_tool_call_result_pairs(&mut self.messages, "run_exit");
            if repair.changed() {
                let _ = self.log("tool_pair_history_repaired", json!({"report": repair}));
            }
        }
        let status = match &result {
            Ok(_) => "completed",
            Err(PokError::Cancelled) => "interrupted",
            Err(_) => "failed",
        };
        let _ = self.persist_conversation(status);
        result
    }

    fn log(&self, kind: &str, payload: Value) -> Result<()> {
        let event = SessionEvent {
            timestamp: Utc::now(),
            session_id: self.id,
            kind: kind.into(),
            payload,
        };
        let mut trace = self.trace.lock();
        serde_json::to_writer(&mut *trace, &event)?;
        trace.write_all(b"\n")?;
        trace.flush()?;
        Ok(())
    }

    /// Show the newest observation in the dashboard's agent view, once.
    fn show_latest_observation(&mut self) {
        let Some(observation) = self.context.latest_observation.lock().clone() else {
            return;
        };
        if self.shown_observation == Some(observation.version) {
            return;
        }
        if let Some(event) = observation_frame(&observation, &self.context.artifact_dir) {
            self.shown_observation = Some(observation.version);
            self.emit(event);
        }
    }

    fn emit(&self, event: AgentEvent) {
        if let Some(observer) = &self.observer {
            observer.emit(event);
        }
    }

    fn emit_task_state(&self) {
        let state = self.context.active_task.lock().clone();
        self.emit(AgentEvent::TaskStateChanged {
            root_request: state.root_request,
            status: state.status,
            current_step: state.current_step,
            steps: state.steps,
        });
    }

    fn emit_context_status(
        &self,
        prompt_tokens: u64,
        compactions: usize,
        approximate_turns_remaining: Option<u64>,
    ) {
        let budget = self.context.context_budget.lock().clone();
        let schema_chars =
            serde_json::to_string(&self.tools.definitions_for_groups(&self.active_tool_groups))
                .map_or(0, |schema| schema.len());
        let conversation_tokens =
            estimate_context_tokens(&self.messages, schema_chars, self.usage_scale);
        let (archived_entries, archived_tokens) =
            self.context.session_archive.stats().unwrap_or((0, 0));
        self.emit(AgentEvent::ContextStatus {
            remaining_tokens: budget.compact_at_tokens.saturating_sub(conversation_tokens),
            budget,
            prompt_tokens,
            conversation_tokens,
            working_set_target_tokens: self.context.prompt_token_target,
            archived_entries,
            archived_tokens,
            approximate_turns_remaining,
            compactions,
        });
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.curation_cancellation.cancel();
        self.context.command_manager.request_shutdown();
    }
}

/// Every primary-model request in the run, including bounded helpers and
/// compaction; `turns` alone undercounts the model calls the router saves.
fn record_primary_model_usage(metrics: &mut RunMetrics, usage: crate::brain::BrainUsageSnapshot) {
    metrics
        .extras
        .insert("primary_model_requests".into(), json!(usage.requests));
    metrics.extras.insert(
        "primary_model_prompt_tokens".into(),
        json!(usage.prompt_tokens),
    );
    metrics.extras.insert(
        "primary_model_completion_tokens".into(),
        json!(usage.completion_tokens),
    );
}

fn increment_metric(metrics: &mut RunMetrics, name: &str, amount: u64) {
    let current = metrics
        .extras
        .get(name)
        .and_then(Value::as_u64)
        .unwrap_or(0);
    metrics
        .extras
        .insert(name.to_owned(), Value::from(current.saturating_add(amount)));
}

#[cfg(test)]
mod tests;

/// Images one request may carry.
pub const MAX_USER_IMAGES: usize = 8;
/// Largest attached image, decoded.
pub const MAX_USER_IMAGE_BYTES: usize = 10 * 1024 * 1024;

fn validated_user_images(images: Vec<String>) -> Result<Vec<String>> {
    use base64::Engine as _;
    if images.len() > MAX_USER_IMAGES {
        return Err(PokError::Tool(format!(
            "attach at most {MAX_USER_IMAGES} images to one message"
        )));
    }
    images
        .into_iter()
        .map(|image| {
            let encoded = image
                .strip_prefix("data:image/png;base64,")
                .unwrap_or(&image)
                .trim()
                .to_owned();
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(&encoded)
                .map_err(|_| PokError::Tool("an attached image is not valid base64".into()))?;
            if !bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
                return Err(PokError::Tool("attached images must be PNG".into()));
            }
            if bytes.len() > MAX_USER_IMAGE_BYTES {
                return Err(PokError::Tool(
                    "an attached image is larger than 10 MB".into(),
                ));
            }
            Ok(encoded)
        })
        .collect()
}

/// Files and folders one request may attach.
pub const MAX_ATTACHED_FILES: usize = 20;

fn attached_file_paths(files: &[String]) -> Result<Vec<std::path::PathBuf>> {
    if files.len() > MAX_ATTACHED_FILES {
        return Err(PokError::Tool(format!(
            "attach at most {MAX_ATTACHED_FILES} files or folders to one message"
        )));
    }
    files
        .iter()
        .map(|file| {
            dunce::canonicalize(file.trim())
                .map_err(|_| PokError::Tool(format!("the attached file {file:?} no longer exists")))
        })
        .collect()
}

fn human_size(bytes: u64) -> String {
    match bytes {
        0..1_024 => format!("{bytes} B"),
        1_024..1_048_576 => format!("{:.1} KB", bytes as f64 / 1_024.0),
        1_048_576..1_073_741_824 => format!("{:.1} MB", bytes as f64 / 1_048_576.0),
        _ => format!("{:.1} GB", bytes as f64 / 1_073_741_824.0),
    }
}

/// The note appended to a request that lists what the user attached.
pub(crate) fn attached_files_block(paths: &[std::path::PathBuf]) -> String {
    let lines = paths
        .iter()
        .map(|path| {
            let metadata = std::fs::metadata(path).ok();
            if metadata.as_ref().is_some_and(std::fs::Metadata::is_dir) {
                let entries = std::fs::read_dir(path).map(Iterator::count).unwrap_or(0);
                format!("- {} (folder, {entries} entries)", path.display())
            } else {
                let kind = path
                    .extension()
                    .and_then(|extension| extension.to_str())
                    .map_or_else(
                        || "file".to_owned(),
                        |extension| format!(".{} file", extension.to_ascii_lowercase()),
                    );
                let size = metadata.map_or_else(String::new, |metadata| {
                    format!(", {}", human_size(metadata.len()))
                });
                format!("- {} ({kind}{size})", path.display())
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "\n\n<attached_files>\nThe user attached these; they are readable even outside the workspace:\n{lines}\n\
         Open them to answer: read text files directly; for other formats (documents, spreadsheets, archives, media) \
         write a small helper script or run a command that extracts the content, rather than guessing from the name.\n</attached_files>"
    )
}

/// Absolute Windows paths written in a request that exist on this computer.
/// A path with spaces is tried at its full length first, then shortened a
/// word at a time, so trailing prose is not taken for part of the path.
pub(crate) fn paths_named_in(prompt: &str) -> Vec<std::path::PathBuf> {
    // Unix paths (tests, WSL) must start a word, so "x.com/home" is not one.
    let expression = if cfg!(windows) {
        r#"()([A-Za-z]:[\\/][^"<>|*?\r\n]+)"#
    } else {
        r#"(^|[\s"'(])((?:[A-Za-z]:[\\/]|/)[^"<>|*?\r\n]+)"#
    };
    let Ok(pattern) = regex::Regex::new(expression) else {
        return Vec::new();
    };
    let mut found = Vec::new();
    for candidate in pattern
        .captures_iter(prompt)
        .filter_map(|captures| captures.get(2))
    {
        let mut text = candidate
            .as_str()
            .trim_end_matches(['.', ',', ';', ':', ')', '\'', ' '])
            .to_owned();
        loop {
            if let Ok(path) = dunce::canonicalize(&text) {
                if !found.contains(&path) {
                    found.push(path);
                }
                break;
            }
            match text.rfind(' ') {
                Some(space) => {
                    text.truncate(space);
                    text = text
                        .trim_end_matches(['.', ',', ';', ':', ')', '\''])
                        .to_owned();
                }
                None => break,
            }
        }
    }
    found
}
