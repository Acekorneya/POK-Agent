import React, { useEffect, useLayoutEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import "./styles.css";
import { Composer, ConversationSidebar, useDialogFocus } from "./components/workspace";
import { SettingsDrawer } from "./components/settings-drawer";
import "./workspace.css";
import { appendStreamDelta, type ChatMessage } from "./session-display";
import { asRecord, formatElapsed } from "./display-utils";
import { ConversationMessage } from "./components/conversation";

type Approval = { id: string; tool: string; reason: string; arguments: unknown };
type UserQuestion = { id: string; header: string; question: string; options: { label: string; description: string }[]; multi_select: boolean };
type QuestionRequest = { id: string; questions: UserQuestion[] };
type LocalModelPhase = "checking" | "server_offline" | "unloaded" | "unloading" | "loading" | "loaded" | "unsupported" | "error";
type LocalModelRuntimeStatus = {
  provider: string;
  model: string;
  phase: LocalModelPhase;
  detail?: string;
  instance_ids: string[];
  context_length?: number;
};
type Status = {
  provider: string;
  base_url: string;
  model: string;
  platform: string;
  data_dir: string;
  model_runtime?: LocalModelRuntimeStatus;
  requires_api_key?: boolean;
  has_api_key?: boolean;
  data_boundary: "local_device" | "external_service";
};
type DecisionRouterBackend = "off" | "jev" | "laya" | "llm_choice" | "kev";
type LayaDevicePreference = "auto" | "gpu" | "cpu";
type LayaRuntimeStatus = { installed: boolean; running: boolean; phase: string; detail: string; model: string; checkpoint: string; endpoint?: string; device?: string; device_reason?: string; preference: LayaDevicePreference };
type JudgeRuntimeStatus = { installed: boolean; enabled: boolean; running: boolean; phase: string; detail: string; model: string; checkpoint: string; endpoint?: string; device?: string; device_reason?: string };
type DecisionRouterStatus = {
  enabled: boolean;
  backend: DecisionRouterBackend;
  model: string;
  has_api_key: boolean;
  laya: LayaRuntimeStatus;
  judge: JudgeRuntimeStatus;
};
type MemoryRecord = {
  id: string; source: string; text: string; approved: boolean; enabled: boolean; created_at: string;
  updated_at: string; reinforcement_count: number; review_status: string; related_memory_id?: string;
  version: number; provenance: string;
};
type MemoryDuplicateGroup = { id: string; records: MemoryRecord[]; similarity: number };
type ProcedureRecord = {
  id: string;
  kind: "workflow" | "command";
  task_signature: string;
  title: string;
  summary: string;
  applications: string[];
  steps: { tool: string; instruction: string }[];
  command_template?: string;
  evidence: string;
  enabled: boolean;
  success_count: number;
  retrieval_count: number;
  updated_at: string;
};
type MemoryLibrary = { facts: MemoryRecord[]; skills: ProcedureRecord[] };
type TaskStep = { id: string; content: string; status: "pending" | "in_progress" | "completed" };
type TaskProgress = {
  rootRequest: string;
  status: "pending" | "in_progress" | "completed";
  currentStep: string;
  steps: TaskStep[];
};
type ContextBudget = {
  provider: string;
  model: string;
  context_window_tokens: number;
  threshold_percent: number;
  compact_at_tokens: number;
  post_compaction_target_tokens: number;
  output_reserve_tokens: number;
  source: "provider" | "provider_loaded" | "provider_max" | "models_dev" | "manual" | "fallback_unknown";
  known: boolean;
};
type ContextStatus = {
  budget: ContextBudget;
  prompt_tokens: number;
  conversation_tokens: number;
  working_set_target_tokens?: number;
  remaining_tokens: number;
  archived_entries: number;
  archived_tokens: number;
  approximate_turns_remaining?: number;
  compactions: number;
};
type ModelCapabilities = {
  id: string;
  context_length?: number;
  loaded_context_length?: number;
  max_context_length?: number;
  vision?: boolean;
  tool_use?: boolean;
  reasoning?: boolean;
  supported_parameters: string[];
  reasoning_efforts: string[];
  reasoning_default?: string;
  loaded?: boolean;
};
type GeneratedTool = {
  name: string;
  description: string;
  runtime: string;
  capabilities: string[];
  enabled: boolean;
  version: number;
  successes: number;
  failures: number;
  schema_version?: number;
  isolated_environment?: boolean;
  validated_at?: string | null;
  smoke_tests?: number;
  assertions?: number;
};
type GeneratedToolCandidate = { id: string; helper_path: string; task: string; runtime: string; verified_command: string; created_at: string };
type ConversationSummary = {
  id: string;
  title: string;
  provider: string;
  model: string;
  workspace: string;
  status: string;
  updated_at: string;
};
type ResumeConversationPayload = {
    summary: ConversationSummary;
    messages: ConversationHistoryMessage[];
    next_before_sequence: number | null;
    has_more: boolean;
    decision_router_enabled: boolean;
    decision_router_backend: DecisionRouterBackend;
    decision_activity: { purpose: string; label: string; detail: string; elapsed_ms: number; recorded_at: string }[];
};
type ConversationHistoryMessage = {
  sequence: number;
  subsequence?: number;
  sender: "user" | "agent" | "system";
  kind?: "prompt" | "guidance" | "response" | "activity";
  text: string;
  tool?: string;
  arguments?: Record<string, unknown>;
};
type ConversationHistoryPage = {
  messages: ConversationHistoryMessage[];
  next_before_sequence: number | null;
  has_more: boolean;
};
type AgentEvent = {
  type: string;
  waiting?: boolean;
  text?: string;
  turn?: number;
  call_id?: string;
  name?: string;
  arguments?: unknown;
  ok?: boolean;
  detail?: string;
  result?: unknown;
  prompt_tokens?: number;
  completion_tokens?: number;
  cumulative_prompt_tokens?: number;
  cumulative_completion_tokens?: number;
  model?: string;
  session_id?: string;
  error?: string;
  root_request?: string;
  status?: "pending" | "in_progress" | "completed";
  completion_status?: "completed" | "completed_with_warnings";
  current_step?: string;
  steps?: TaskStep[];
  tool?: string;
  reason?: string;
  promoted?: boolean;
  reused?: boolean;
  votes?: Record<string, string>;
  budget?: ContextBudget;
  working_set_target_tokens?: number;
  conversation_tokens?: number;
  remaining_tokens?: number;
  archived_entries?: number;
  archived_tokens?: number;
  approximate_turns_remaining?: number;
  compactions?: number;
  attempt?: number;
  attempts?: number;
  summary_source?: "model" | "model_retry";
  reasoning_chars?: number;
  maximum?: number;
  delay_ms?: number;
  answer?: string;
  interrupted_phase?: string;
  paused_ms?: number;
  note?: string;
  sequence?: number;
  trigger?: "automatic" | "manual";
  transcript_chars?: number;
  elapsed_ms?: number;
  estimated_prompt_tokens?: number;
  tool_count?: number;
  event_kind?: string;
  stage?: string;
  next_max_tokens?: number;
  candidate_count?: number;
  candidate_id?: string;
  description?: string;
  selected_probability?: number;
  confidence?: number;
  operation_probability?: number;
  target_probability?: number;
  operation_confidence?: number;
  target_confidence?: number;
  eligible?: boolean;
  rejection_reason?: string;
  outcome?: string;
  disposition?: string;
  related_memory_id?: string;
  used_jev?: boolean;
  revision?: number;
  previous_window?: string;
  current_window?: string;
  quiet_period_ms?: number;
  action?: string;
  purpose?: string;
  probability_threshold?: number;
  confidence_threshold?: number;
  alternatives?: [string, number][];
  warnings?: Array<{ code?: string; message?: string; artifact_path?: string }>;
  deliverables?: Array<{
    path?: string;
    artifact_type?: string;
    sha256?: string;
    validation_status?: string;
  }>;
  event?: {
    call_id: string;
    task_id: string;
    kind: "output" | "status";
    stream?: "stdout" | "stderr";
    chunk?: string;
    total_bytes: number;
    status: string;
    exit_code?: number;
    termination_reason?: string;
  };
};

type JevMetrics = { evaluations: number; eligible: number; actions: number; evidence: number; cached: number; progress: number; failures: number; latencyMs: number; judgeAttempts: number; judgePromotions: number; last?: string };
const emptyJevMetrics = (): JevMetrics => ({ evaluations: 0, eligible: 0, actions: 0, evidence: 0, cached: 0, progress: 0, failures: 0, latencyMs: 0, judgeAttempts: 0, judgePromotions: 0 });
type JevRuntimeState = "off" | "ready" | "evaluating" | "unavailable";
type ActivityPhase = "idle" | "starting" | "compacting" | "thinking" | "reading" | "navigating" | "acting" | "running" | "waiting" | "paused" | "completed" | "failed";
type ActivityStatus = "running" | "done" | "failed";
type LiveActivity = {
  phase: ActivityPhase;
  label: string;
  detail?: string;
  tool?: string;
  startedAt: number;
};


type SubagentState = {
  id: string;
  title: string;
  targetDir: string;
  status: "running" | "done" | "failed";
  steps: string[];
};

function restoredChatMessage(message: ConversationHistoryMessage, timestamp: string): ChatMessage {
  const type = message.kind ?? (message.sender === "user" ? "prompt" : "response");
  return {
    id: `restored-${message.sequence}-${message.subsequence ?? 0}`,
    sender: message.sender,
    type,
    text: message.text,
    timestamp: new Date(timestamp),
    activityTool: message.tool,
    activityLabel: message.tool ? `Ran ${message.tool}` : undefined,
    activityDetail: message.tool ? "Restored from conversation history" : undefined,
    activityStatus: message.tool ? "done" : undefined,
    activityArgs: message.arguments,
  };
}


/** Adds a note when a click used an accessibility action instead of the mouse. */
function withInputMethod(detail: string | undefined, method: unknown): string | undefined {
  if (typeof method !== "string" || !method.startsWith("ui_automation")) return detail;
  const note = "without moving your mouse";
  return detail ? `${detail} · ${note}` : note;
}

function compactValue(value: unknown, maximum = 90): string | undefined {
  if (typeof value !== "string" && typeof value !== "number") return undefined;
  const normalized = String(value).replace(/\s+/g, " ").trim();
  if (!normalized) return undefined;
  return normalized.length > maximum ? `${normalized.slice(0, maximum - 1)}…` : normalized;
}

function humanizeToolName(name: string): string {
  return name
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function describeTool(name: string, rawArguments: unknown): Pick<LiveActivity, "phase" | "label" | "detail"> {
  const args = asRecord(rawArguments);
  const detail = (...keys: string[]) => {
    for (const key of keys) {
      const value = compactValue(args[key]);
      if (value) return value;
    }
    return undefined;
  };

  const descriptions: Record<string, Pick<LiveActivity, "phase" | "label" | "detail">> = {
    get_current_time: { phase: "reading", label: "Checking current time" },
    list_windows: { phase: "reading", label: "Surveying open windows" },
    observe_desktop: { phase: "reading", label: "Surveying the desktop" },
    capture_screen: { phase: "reading", label: "Reading the screen", detail: detail("window_id", "monitor_id") },
    query_screen_text: { phase: "reading", label: "Reading visible text", detail: detail("query", "text") },
    query_window_tree: { phase: "reading", label: "Reading page structure", detail: detail("window_id") },
    scroll_view: { phase: "reading", label: "Scanning the page", detail: detail("direction", "amount") },
    scroll_until_text: { phase: "reading", label: "Searching the page", detail: detail("text", "query") },
    browser_navigate: { phase: "navigating", label: "Opening a website", detail: detail("query_or_url", "url") },
    managed_browser_open: { phase: "navigating", label: "Opening the managed browser", detail: detail("url") },
    managed_browser_snapshot: { phase: "reading", label: "Reading the managed browser" },
    managed_browser_click: { phase: "acting", label: "Selecting a browser target", detail: detail("target_id") },
    managed_browser_type: { phase: "acting", label: "Typing in the managed browser", detail: detail("target_id") },
    managed_browser_select: { phase: "acting", label: "Choosing a browser option", detail: detail("target_id") },
    managed_browser_hover: { phase: "acting", label: "Inspecting a browser target", detail: detail("target_id") },
    managed_browser_scroll: { phase: "reading", label: "Scanning the managed browser", detail: detail("direction") },
    activate_window: { phase: "navigating", label: "Switching to a window", detail: detail("title", "window_id") },
    open_application: { phase: "navigating", label: "Opening an application", detail: detail("name") },
    fast_actions: { phase: "acting", label: "Carrying out the plan", detail: detail("goal", "target_hint") },
    discover_tools: { phase: "thinking", label: "Loading more tools", detail: detail("groups") },
    locate_visual_target: { phase: "reading", label: "Locating a target on screen", detail: detail("label") },
    click_localized: { phase: "acting", label: "Selecting a located target" },
    inspect_screen_region: { phase: "reading", label: "Inspecting part of the screen" },
    click_target: { phase: "acting", label: "Selecting an interface target", detail: detail("label", "target_id") },
    simulate_input: { phase: "acting", label: "Interacting with the application", detail: detail("text", "keys") },
    execute_action_batch: { phase: "acting", label: "Performing interface actions" },
    read_file: { phase: "reading", label: "Reading a file", detail: detail("path") },
    list_directory: { phase: "reading", label: "Listing workspace files", detail: detail("path") },
    grep_search: { phase: "reading", label: "Searching the workspace", detail: detail("pattern", "query") },
    memory_search: { phase: "reading", label: "Searching memory", detail: detail("query") },
    remember_fact: { phase: "acting", label: "Saving a memory" },
    memory_save: { phase: "acting", label: "Saving to memory" },
    edit_file: { phase: "acting", label: "Editing a file", detail: detail("filepath", "path") },
    write_file: { phase: "acting", label: "Creating a file", detail: detail("filepath", "path") },
    verify_task_outcome: { phase: "reading", label: "Verifying the result" },
    undo_edit: { phase: "acting", label: "Restoring a file", detail: detail("filepath", "path", "undo_record") },
    run_command: { phase: "running", label: "Running a command", detail: detail("command") },
    update_task_plan: { phase: "thinking", label: "Updating the task plan" },
  };

  return descriptions[name] ?? {
    phase: "running",
    label: humanizeToolName(name),
    detail: detail("path", "query", "url", "command"),
  };
}


function useElapsed(startedAt: number, active: boolean): string {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    if (!active) {
      setNow(Date.now());
      return;
    }
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [active, startedAt]);
  return formatElapsed(now - startedAt);
}


function LiveActivityPanel({ activity, busy, turn, tokens }: { activity: LiveActivity; busy: boolean; turn: number; tokens: number }) {
  const working = busy && !["idle", "paused", "completed", "failed"].includes(activity.phase);
  const elapsed = useElapsed(activity.startedAt, working);
  return (
    <div className={`live-activity phase-${activity.phase}`} role="status" aria-live="polite">
      <div className="live-activity-marker" aria-hidden="true"><span /></div>
      <div className="live-activity-copy">
        <span className="live-activity-kicker">{working ? "Working now" : activity.phase === "paused" ? "Control released" : activity.phase === "failed" ? "Needs attention" : "Agent status"}</span>
        <strong>{activity.label}</strong>
        {activity.detail && <span className="live-activity-detail">{activity.detail}</span>}
      </div>
      <div className="live-activity-meta">
        {working && <span>{elapsed}</span>}
        {working && turn > 0 && <span>Turn {turn}</span>}
        {working && tokens > 0 && <span>↑ {tokens.toLocaleString()} tokens</span>}
        {working && <small>Ctrl+Alt+Esc to interrupt</small>}
      </div>
    </div>
  );
}

function TaskProgressPanel({ task }: { task: TaskProgress }) {
  const completed = task.steps.filter((step) => step.status === "completed").length;
  const progress = task.steps.length > 0 ? Math.round((completed / task.steps.length) * 100) : 0;
  return (
    <div className={`task-progress task-${task.status}`}>
      <div className="task-progress-heading">
        <div>
          <span className="task-progress-kicker">{task.status === "completed" ? "Task complete" : "Task progress"}</span>
          <strong>{task.currentStep}</strong>
        </div>
        {task.steps.length > 0 && <span className="task-progress-fraction">{completed}/{task.steps.length}</span>}
      </div>
      <div className="task-progress-goal">{task.rootRequest}</div>
      {task.steps.length > 0 && (
        <>
          <div className="task-progress-track" aria-label={`${progress}% complete`}>
            <div className="task-progress-fill" style={{ width: `${progress}%` }} />
          </div>
          <div className="task-steps">
            {task.steps.map((step) => (
              <div className={`task-step task-step-${step.status}`} key={step.id}>
                <span className="task-step-marker" aria-hidden="true">{step.status === "completed" ? "✓" : step.status === "in_progress" ? "●" : "○"}</span>
                <span>{step.content}</span>
                {step.status === "in_progress" && <small>NOW</small>}
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

function ModelPicker({ models, value, onChange, disabled }: {
  models: string[]; value: string; onChange: (model: string) => void; disabled?: boolean;
}) {
  const [query, setQuery] = useState(value);
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const filtered = query.trim()
    ? models.filter((item) => item.toLowerCase().includes(query.toLowerCase())).slice(0, 100)
    : models.slice(0, 100);
  useEffect(() => { setQuery(value); }, [value]);
  useEffect(() => {
    const close = (event: MouseEvent) => { if (ref.current && !ref.current.contains(event.target as Node)) setOpen(false); };
    const escape = (event: KeyboardEvent) => { if (event.key === "Escape") setOpen(false); };
    window.addEventListener("mousedown", close);
    window.addEventListener("keydown", escape);
    return () => { window.removeEventListener("mousedown", close); window.removeEventListener("keydown", escape); };
  }, []);
  return (
    <div className="model-picker" ref={ref}>
      <input
        role="combobox"
        aria-label="Model"
        aria-expanded={open}
        aria-controls="model-picker-list"
        className="composer-model"
        value={query}
        disabled={disabled}
        placeholder={value || "Search models…"}
        onFocus={() => setOpen(true)}
        onChange={(event) => { setQuery(event.target.value); setOpen(true); }}
        onKeyDown={(event) => {
          if (event.key === "Enter" && open && filtered.length > 0) {
            event.preventDefault();
            onChange(filtered[0]);
            setQuery(filtered[0]);
            setOpen(false);
          }
        }}
      />
      {open && (
        <ul id="model-picker-list" role="listbox" aria-label="Model options" className="model-picker-list">
          {filtered.length === 0 && <li className="model-picker-empty">No matching models</li>}
          {filtered.map((item) => (
            <li key={item} role="option" aria-selected={item === value} className={item === value ? "selected" : ""}
              onMouseDown={(event) => { event.preventDefault(); onChange(item); setQuery(item); setOpen(false); }}
            >{item}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

function InstallProgress({ label, detail }: { label: string; detail?: string }) {
  return (
    <div className="install-progress" role="status" aria-live="polite">
      <div className="install-progress-track"><div className="install-progress-fill" /></div>
      <span className="install-progress-label">{label}{detail ? ` — ${detail}` : ""}</span>
    </div>
  );
}

const PROVIDER_URLS: Record<string, { name: string; url: string }> = {
  openai: { 
    name: "OpenAI Platform API Keys", 
    url: "https://platform.openai.com/api-keys",
  },
  grok: {
    name: "xAI Grok Console API Keys",
    url: "https://console.x.ai/",
  },
  gemini: { name: "Google AI Studio API Keys", url: "https://aistudio.google.com/app/apikey" },
  anthropic: { 
    name: "Anthropic Console API Keys", 
    url: "https://console.anthropic.com/settings/keys",
  },
  openrouter: { name: "OpenRouter API Keys", url: "https://openrouter.ai/keys" },
  groq: { name: "Groq Console Keys", url: "https://console.groq.com/keys" },
  lm_studio: { name: "LM Studio Official Site", url: "https://lmstudio.ai" },
  ollama: { name: "Ollama Official Site", url: "https://ollama.com" },
};

function ModelStatusBanner({ status, model, provider, runtime, requiresApiKey, hasApiKey, onRetry, onConnectKey }: { status: "connecting" | "ready" | "error"; model: string; provider: string; runtime?: LocalModelRuntimeStatus | null; requiresApiKey?: boolean; hasApiKey?: boolean; onRetry: () => void; onConnectKey: () => void }) {
  const providerName = provider.replaceAll("_", " ").toUpperCase();
  if (status === "connecting") {
    return <div className="api-status-banner connecting"><span>⏳ Connecting to {providerName} API...</span></div>;
  }
  if (requiresApiKey && !hasApiKey) {
    return (
      <div className="api-status-banner error" style={{ borderColor: "#ef4444", background: "var(--danger-bg)", color: "var(--danger)" }}>
        <span>🔴 {providerName} Missing API Key</span>
        <button onClick={onConnectKey} style={{ padding: "2px 8px", fontSize: "10px" }}>🔑 Connect Key</button>
      </div>
    );
  }
  if (status === "error") {
    return (
      <div className="api-status-banner error">
        <span>🔴 {providerName} Disconnected / Offline</span>
        <button onClick={onRetry} style={{ padding: "2px 8px", fontSize: "10px" }}>Retry</button>
      </div>
    );
  }
  const isLocal = provider === "lm_studio" || provider === "ollama";
  if (isLocal && runtime) {
    const runtimeProviderName = runtime.provider.replaceAll("_", " ").toUpperCase();
    const displayModel = runtime.model || model || "Model";
    if (runtime.phase === "loading" || runtime.phase === "checking") {
      return <div className="api-status-banner connecting" style={{ borderColor: "#eab308", color: "var(--warning)", background: "var(--surface)" }}><span>⚡ {runtime.phase === "loading" ? `Loading ${displayModel} into GPU VRAM…` : `Checking ${displayModel}…`}</span></div>;
    }
    if (runtime.phase === "unloading") {
      return <div className="api-status-banner connecting" style={{ borderColor: "#eab308", color: "var(--warning)", background: "var(--surface)" }}><span>↧ Unloading {displayModel} from GPU VRAM…</span></div>;
    }
    if (runtime.phase === "loaded") {
      return <div className="api-status-banner ready"><span>🟢 {runtimeProviderName} Ready ({displayModel} · VRAM Loaded)</span><button onClick={onRetry} title="Refresh model status" style={{ padding: "2px 8px", fontSize: "10px" }}>↻</button></div>;
    }
    if (runtime.phase === "unloaded") {
      return <div className="api-status-banner ready" style={{ borderColor: "#d97706", color: "var(--warning)", background: "var(--surface)" }}><span>🟡 {runtimeProviderName} Server Online ({displayModel} · Unloaded)</span><button onClick={onRetry} style={{ padding: "2px 8px", fontSize: "10px" }}>Load</button></div>;
    }
    if (runtime.phase === "unsupported") {
      return <div className="api-status-banner ready" style={{ borderColor: "#64748b", color: "var(--muted)", background: "var(--surface)" }}><span>◌ {runtimeProviderName} Online (load state unavailable)</span></div>;
    }
    return <div className="api-status-banner error"><span>🔴 {runtime.phase === "server_offline" ? `${runtimeProviderName} Server Offline` : `${runtimeProviderName} Model Error`}</span><button onClick={onRetry} style={{ padding: "2px 8px", fontSize: "10px" }}>Retry</button></div>;
  }
  return <div className="api-status-banner ready"><span>🟢 {providerName} Ready ({model || "Connected"}{isLocal ? " · VRAM Loaded" : ""})</span></div>;
}

function ThinkingIndicator({ text, active }: { text: string; active: boolean }) {
  const [expanded, setExpanded] = useState(true);
  if (!text && !active) return null;
  return (
    <div className={`thinking-container ${active ? "active" : ""}`}>
      <div className="thinking-header" onClick={() => setExpanded(!expanded)}>
        <span>🧠 {active ? "Model is reasoning..." : "Reasoning Process"}</span>
        <small>{expanded ? "Collapse ▲" : "Expand ▼"}</small>
      </div>
      {expanded && <div className="thinking-body">{text || "Thinking step-by-step..."}</div>}
    </div>
  );
}

function SubagentCard({ subagent }: { subagent: SubagentState }) {
  const [expanded, setExpanded] = useState(true);
  return (
    <div className={`subagent-card status-${subagent.status}`}>
      <div className="subagent-header" onClick={() => setExpanded(!expanded)}>
        <span className="subagent-badge">🛠️ Subagent</span>
        <span style={{ flex: 1 }}>{subagent.title}</span>
        <span style={{ fontSize: "11px", color: "#66838d" }}>📁 {subagent.targetDir}</span>
        <span className={`status-pill ${subagent.status}`}>{subagent.status.toUpperCase()}</span>
      </div>
      {expanded && (
        <div className="subagent-steps">
          {subagent.steps.map((step, idx) => (
            <div key={idx} className="subagent-step-item">{step}</div>
          ))}
        </div>
      )}
    </div>
  );
}

function TerminalWidget({ command, status, output }: { command: string; status: string; output: string }) {
  const [collapsed, setCollapsed] = useState(false);
  return (
    <div className="terminal-widget">
      <div className="terminal-header">
        <span className="terminal-prompt">$ {command}</span>
        <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
          <span className={`status-pill ${status.toLowerCase()}`}>{status}</span>
          <button onClick={() => setCollapsed(!collapsed)} style={{ padding: "2px 8px", fontSize: "10px" }}>{collapsed ? "Expand" : "Collapse"}</button>
        </div>
      </div>
      {!collapsed && (
        <pre className="terminal-output"><code>{output || "Executing command in workspace..."}</code></pre>
      )}
    </div>
  );
}

export function App() {
  const [status, setStatus] = useState<Status | null>(null);
  const [customEndpoint, setCustomEndpoint] = useState("");
  const [apiStatus, setApiStatus] = useState<"connecting" | "ready" | "error">("connecting");
  const [policyMode, setPolicyMode] = useState<"interactive" | "autonomous">("interactive");
  const [providers, setProviders] = useState<string[]>([]);
  const [selectedProvider, setSelectedProvider] = useState<string>(() => localStorage.getItem("pok_selected_provider") || "lm_studio");
  const [models, setModels] = useState<string[]>([]);
  const [model, setModel] = useState<string>(() => localStorage.getItem("pok_selected_model") || "");
  const [modelSearch, setModelSearch] = useState("");
  const [showCloudModal, setShowCloudModal] = useState(false);
  const [cloudApiKey, setCloudApiKey] = useState("");
  const [cloudKeyError, setCloudKeyError] = useState("");
  const [savingCloudKey, setSavingCloudKey] = useState(false);
  const [decisionRouter, setDecisionRouter] = useState<DecisionRouterStatus | null>(null);
  const [jevEnabled, setJevEnabled] = useState(false);
  const [decisionRouterBackend, setDecisionRouterBackend] = useState<DecisionRouterBackend>("off");
  const [installingLaya, setInstallingLaya] = useState(false);
  const [layaDevice, setLayaDevice] = useState<LayaDevicePreference>("auto");
  const [installingJudge, setInstallingJudge] = useState(false);
  const [togglingJudge, setTogglingJudge] = useState(false);
  const [jevRuntimeState, setJevRuntimeState] = useState<JevRuntimeState>("off");
  const [jevWarning, setJevWarning] = useState<string | null>(null);
  const [decisionRouterKey, setDecisionRouterKey] = useState("");
  const [decisionRouterError, setDecisionRouterError] = useState("");
  const [jevMetrics, setJevMetrics] = useState<JevMetrics>(emptyJevMetrics);
  const decisionRouterBackendRef = useRef<DecisionRouterBackend>("off");
  decisionRouterBackendRef.current = decisionRouterBackend;
  const activeRouterName = () =>
    decisionRouterBackendRef.current === "laya"
      ? "Laya"
      : decisionRouterBackendRef.current === "llm_choice"
        ? "LLM choice"
        : decisionRouterBackendRef.current === "kev"
          ? "kev"
          : "JEV";
  // Single source of truth for "something is installing/loading" so the same
  // progress bar shows in the chat feed, above the composer, and in Settings.
  const installStatus = (): { label: string; detail?: string } | null => {
    if (modelRuntime?.phase === "loading" || modelRuntime?.phase === "checking") {
      return { label: `Loading ${modelRuntime.model || model || "the model"} into GPU VRAM`, detail: modelRuntime?.detail };
    }
    if (modelRuntime?.phase === "unloading") {
      return { label: `Unloading ${modelRuntime.model || model || "the model"} from GPU VRAM`, detail: modelRuntime?.detail };
    }
    if (installingLaya || decisionRouter?.laya.phase === "loading") {
      return { label: "Installing / loading Laya locally", detail: decisionRouter?.laya.detail };
    }
    if (installingJudge || decisionRouter?.judge.phase === "loading") {
      return { label: "Installing the judge model (zeiger)", detail: decisionRouter?.judge.detail };
    }
    return null;
  };
  const [workspace, setWorkspace] = useState("");
  const [prompt, setPrompt] = useState("");
  const [busy, setBusy] = useState(false);
  const [approval, setApproval] = useState<Approval | null>(null);
  const [approvalChoice, setApprovalChoice] = useState<"once" | "session" | "deny" | null>(null);
  const [approvalError, setApprovalError] = useState("");
  const [questionRequest, setQuestionRequest] = useState<QuestionRequest | null>(null);
  const [questionAnswers, setQuestionAnswers] = useState<Record<string, { selected: string[]; custom: string }>>({});
  const [sessionApprovals, setSessionApprovals] = useState<string[]>([]);
  const [memoryLibrary, setMemoryLibrary] = useState<MemoryLibrary>({ facts: [], skills: [] });
  const [memoryDuplicateGroups, setMemoryDuplicateGroups] = useState<MemoryDuplicateGroup[]>([]);
  const [memoryCleanupBusy, setMemoryCleanupBusy] = useState(false);
  const [lastMemoryMerge, setLastMemoryMerge] = useState<string | null>(null);
  const [memoryTab, setMemoryTab] = useState<"facts" | "skills">("facts");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [historyBeforeSequence, setHistoryBeforeSequence] = useState<number | null>(null);
  const [historyHasMore, setHistoryHasMore] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [usage, setUsage] = useState({ currentPrompt: 0, currentCompletion: 0, totalPrompt: 0, totalCompletion: 0 });
  const [taskProgress, setTaskProgress] = useState<TaskProgress | null>(null);
  const [currentTurn, setCurrentTurn] = useState(0);
  const [pauseState, setPauseState] = useState<"running" | "requested" | "paused">("running");
  const [waitingForUser, setWaitingForUser] = useState(false);
  const [liveActivity, setLiveActivity] = useState<LiveActivity>({
    phase: "idle",
    label: "Ready for a task",
    startedAt: Date.now(),
  });
  const [contextStatus, setContextStatus] = useState<ContextStatus | null>(null);
  const [contextWindowOverride, setContextWindowOverride] = useState("");
  const [contextThreshold, setContextThreshold] = useState(80);
  const [contextExpanded, setContextExpanded] = useState(false);
  const [modelCapabilities, setModelCapabilities] = useState<ModelCapabilities | null>(null);
  const [manualCapabilityOverride, setManualCapabilityOverride] = useState(false);
  const [visionMode, setVisionMode] = useState<"auto" | "on" | "off">("auto");
  const [reasoningEffort, setReasoningEffort] = useState("auto");
  const [requestTemperature, setRequestTemperature] = useState("");
  const [requestMaxOutput, setRequestMaxOutput] = useState("");
  const [requestSeed, setRequestSeed] = useState("42");
  const [fullContext, setFullContext] = useState(false);
  const [generatedTools, setGeneratedTools] = useState<GeneratedTool[]>([]);
  const [generatedToolCandidates, setGeneratedToolCandidates] = useState<GeneratedToolCandidate[]>([]);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [modelRuntime, setModelRuntime] = useState<LocalModelRuntimeStatus | null>(null);
  const [refreshingModels, setRefreshingModels] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [navigationOpen, setNavigationOpen] = useState(() => window.innerWidth >= 900);
  const [showJump, setShowJump] = useState(false);
  const jevSelectedTool = useRef<string | null>(null);
  const [activeConversationId, setActiveConversationId] = useState("");
  const activeConversationIdRef = useRef("");
  useDialogFocus(`${settingsOpen}:${showCloudModal}:${approval?.id ?? ""}:${questionRequest?.id ?? ""}`);
  const openSettings = (section = "settings-heading") => {
    setSettingsOpen(true);
    requestAnimationFrame(() => {
      if (section === "settings-heading") document.getElementById("settings-panel")?.scrollTo({ top: 0 });
      else document.getElementById(section)?.scrollIntoView({ block: "start" });
    });
  };

  const consoleRef = useRef<HTMLDivElement>(null);
  const settingsButtonRef = useRef<HTMLButtonElement>(null);
  const activeApproval = useRef<Approval | null>(null);
  const followOutput = useRef(true);
  const initialLoadAttempt = useRef("");
  const emergencyStopRequested = useRef(false);
  const activeLocalSelection = useRef<{ provider: string; model: string } | null>(
    (selectedProvider === "lm_studio" || selectedProvider === "ollama") && model
      ? { provider: selectedProvider, model }
      : null,
  );
  const modelTransitioning = modelRuntime?.phase === "loading"
    || modelRuntime?.phase === "unloading"
    || modelRuntime?.phase === "checking";

  const checkApiStatus = (provider = selectedProvider, targetModel = model) => {
    setApiStatus("connecting");
    invoke<Status>("get_status", { provider, model: targetModel || undefined })
      .then(async (value) => {
        setStatus(value);
        setModelRuntime(value.model_runtime ?? null);
        setCustomEndpoint(value.base_url);
        if (!localStorage.getItem("pok_selected_model")) {
          setModel("");
        }
        if (value.requires_api_key && !value.has_api_key) {
          setModels([]);
          setApiStatus("ready");
          return;
        }
        await fetchModels(provider);
        setApiStatus("ready");
      })
      .catch(() => setApiStatus("error"));
  };

  const handleSaveEndpoint = async () => {
    if (!customEndpoint.trim()) return;
    try {
      setApiStatus("connecting");
      const updated = await invoke<Status>("update_provider_endpoint", {
        provider: selectedProvider,
        endpoint: customEndpoint.trim(),
        model: model || undefined,
      });
      setStatus(updated);
      setModelRuntime(updated.model_runtime ?? null);
      setCustomEndpoint(updated.base_url);
      setApiStatus("ready");
      await fetchModels(selectedProvider);
    } catch (err) {
      setApiStatus("error");
    }
  };

  const fetchModels = async (prov?: string) => {
    const targetProvider = prov || selectedProvider;
    try {
      const fetched = await invoke<string[]>("list_models", { provider: targetProvider });
      setModels(fetched);
      if (!localStorage.getItem("pok_selected_model")) {
        setModel("");
      }
      return fetched;
    } catch (error) {
      setModels([]);
      throw error;
    }
  };

  const refreshLocalModels = async () => {
    if (selectedProvider !== "lm_studio" && selectedProvider !== "ollama") return;
    setRefreshingModels(true);
    try {
      const refreshed = await invoke<string[]>("refresh_models", { provider: selectedProvider });
      setModels(refreshed);
      const selectedStillExists = !model
        || refreshed.some((item) => item.toLowerCase() === model.toLowerCase());
      if (!selectedStillExists) {
        setModel("");
        setModelRuntime(null);
        activeLocalSelection.current = null;
        localStorage.removeItem("pok_selected_model");
        setLiveActivity({
          phase: "idle",
          label: "Local model list refreshed",
          detail: `${model} is no longer available. Select another model.`,
          startedAt: Date.now(),
        });
      } else {
        setLiveActivity({
          phase: "idle",
          label: "Local model list refreshed",
          detail: `${refreshed.length} models available`,
          startedAt: Date.now(),
        });
      }
    } catch (error) {
      setLiveActivity({
        phase: "failed",
        label: "Could not refresh local models",
        detail: compactValue(String(error), 160),
        startedAt: Date.now(),
      });
    } finally {
      setRefreshingModels(false);
    }
  };

  const handleModelChange = async (newModel: string) => {
    setModel(newModel);
    localStorage.setItem("pok_selected_model", newModel);
    setModelSearch("");
    if (!newModel || (selectedProvider !== "lm_studio" && selectedProvider !== "ollama")) return;
    const previous = activeLocalSelection.current;
    setModelRuntime({
      provider: selectedProvider,
      model: newModel,
      phase: "loading",
      detail: `Loading ${newModel}`,
      instance_ids: [],
    });
    setLiveActivity({ phase: "starting", label: `Loading ${newModel}`, detail: "Preparing the local model", startedAt: Date.now() });
    try {
      const loaded = await invoke<LocalModelRuntimeStatus>("switch_local_model", {
        previousProvider: previous?.provider ?? null,
        previousModel: previous?.model ?? null,
        provider: selectedProvider,
        model: newModel,
      });
      setModelRuntime(loaded);
      activeLocalSelection.current = { provider: selectedProvider, model: newModel };
      setLiveActivity({ phase: "idle", label: loaded.phase === "loaded" ? "Ready for a task" : "Model selected", detail: loaded.detail, startedAt: Date.now() });
    } catch (error) {
      setLiveActivity({ phase: "failed", label: "Could not load the model", detail: compactValue(String(error), 160), startedAt: Date.now() });
    }
  };

  const handleProviderChange = async (newProvider: string) => {
    const previous = activeLocalSelection.current;
    setSelectedProvider(newProvider);
    localStorage.setItem("pok_selected_provider", newProvider);
    setModel("");
    setModelRuntime(null);
    localStorage.removeItem("pok_selected_model");
    if (previous && newProvider !== previous.provider && newProvider !== "lm_studio" && newProvider !== "ollama") {
      try {
        await invoke("unload_model", { provider: previous.provider, model: previous.model });
      } catch (error) {
        setLiveActivity({ phase: "failed", label: "Could not unload the previous model", detail: compactValue(String(error), 160), startedAt: Date.now() });
      }
      activeLocalSelection.current = null;
    }
    checkApiStatus(newProvider, "");
  };

  const retryModelConnection = () => {
    if ((selectedProvider === "lm_studio" || selectedProvider === "ollama") && model) {
      setModelRuntime({ provider: selectedProvider, model, phase: "checking", detail: "Checking local model", instance_ids: [] });
      const command = modelRuntime?.phase === "loaded" ? "get_local_model_status" : "load_model";
      void invoke<LocalModelRuntimeStatus>(command, { provider: selectedProvider, model })
        .then((loaded) => {
          setModelRuntime(loaded);
          if (loaded.phase === "loaded") activeLocalSelection.current = { provider: selectedProvider, model };
        })
        .catch((error) => setLiveActivity({ phase: "failed", label: "Could not load the model", detail: compactValue(String(error), 160), startedAt: Date.now() }));
      return;
    }
    checkApiStatus();
  };

  const handleSaveCloudKey = async () => {
    if (!cloudApiKey.trim()) return;
    setSavingCloudKey(true);
    setCloudKeyError("");
    setApiStatus("connecting");
    try {
      await invoke("save_provider_key", { provider: selectedProvider, apiKey: cloudApiKey });
      await fetchModels(selectedProvider);
      const updated = await invoke<Status>("get_status", {
        provider: selectedProvider,
        model: model || undefined,
      });
      setStatus(updated);
      setApiStatus("ready");
      setShowCloudModal(false);
      setCloudApiKey("");
    } catch (error) {
      setApiStatus("error");
      setCloudKeyError(compactValue(String(error), 240) ?? "Unknown provider error");
    } finally {
      setSavingCloudKey(false);
    }
  };

  const handleRouterBackendChange = async (backend: DecisionRouterBackend) => {
    if (busy || backend === decisionRouterBackend) return;
    if (backend === "jev" && !decisionRouter?.has_api_key) {
      setSettingsOpen(true);
      setDecisionRouterError("Save a TypeSafe API key before selecting JEV.");
      return;
    }
    if (backend === "laya" && !decisionRouter?.laya.installed) {
      setSettingsOpen(true);
      setDecisionRouterError("Install Laya in Settings before selecting it.");
      return;
    }
    if ((messages.length > 0 || activeConversationId) && !window.confirm("Start a new conversation with the selected decision router? Your current conversation will remain in history.")) return;
    if (messages.length > 0 || activeConversationId) await newConversation();
    setDecisionRouterError("");
    try {
      if (backend === "laya") setJevRuntimeState("evaluating");
      const next = await invoke<DecisionRouterStatus>("set_decision_router_backend", { backend });
      setDecisionRouter(next);
      setDecisionRouterBackend(backend);
      setJevEnabled(backend !== "off");
      setJevRuntimeState(backend === "off" ? "off" : "ready");
      setJevWarning(null);
    } catch (error) {
      setDecisionRouterError(compactValue(String(error), 300) ?? "Decision router could not be prepared");
      setSettingsOpen(true);
    }
  };

  const handleInstallLaya = async (repair = false) => {
    setInstallingLaya(true);
    setDecisionRouterError("");
    try {
      const next = await invoke<DecisionRouterStatus>("install_laya", { repair });
      setDecisionRouter(next);
      setLayaDevice(next.laya.preference ?? "auto");
    } catch (error) {
      setDecisionRouterError(compactValue(String(error), 300) ?? "Laya installation failed");
    } finally {
      setInstallingLaya(false);
    }
  };

  const handleLayaDeviceChange = async (device: LayaDevicePreference) => {
    if (busy || installingLaya) return;
    setLayaDevice(device);
    setDecisionRouterError("");
    try {
      const next = await invoke<DecisionRouterStatus>("prepare_laya", { device });
      setDecisionRouter(next);
      if (decisionRouterBackend === "laya") setJevRuntimeState("ready");
    } catch (error) {
      setDecisionRouterError(compactValue(String(error), 300) ?? "Laya could not be loaded");
    }
  };

  const handleInstallJudge = async (repair = false) => {
    setInstallingJudge(true);
    setDecisionRouterError("");
    try {
      const next = await invoke<DecisionRouterStatus>("install_judge", { repair });
      setDecisionRouter(next);
    } catch (error) {
      setDecisionRouterError(compactValue(String(error), 300) ?? "Judge-model installation failed");
    } finally {
      setInstallingJudge(false);
    }
  };

  const handleJudgeEnabledChange = async (enabled: boolean) => {
    if (busy || installingJudge || togglingJudge) return;
    setTogglingJudge(true);
    setDecisionRouterError("");
    try {
      const next = await invoke<DecisionRouterStatus>("set_judge_enabled", { enabled });
      setDecisionRouter(next);
    } catch (error) {
      setDecisionRouterError(compactValue(String(error), 300) ?? "Judge model could not be loaded");
    } finally {
      setTogglingJudge(false);
    }
  };

  const handleSaveDecisionRouterKey = async () => {
    if (!decisionRouterKey.trim()) return;
    setDecisionRouterError("");
    try {
      const next = await invoke<DecisionRouterStatus>("save_decision_router_key", { apiKey: decisionRouterKey });
      setDecisionRouter(next);
      setDecisionRouterKey("");
    } catch (error) {
      setDecisionRouterError(compactValue(String(error), 240) ?? "JEV API key could not be saved");
    }
  };

  const refreshModelContext = (targetProvider = selectedProvider, targetModel = model) => {
    if (!targetModel) { setContextStatus(null); return; }
    invoke<{ budget: ContextBudget }>("get_model_context", { provider: targetProvider, model: targetModel })
      .then(({ budget }) => {
        setContextStatus({ budget, prompt_tokens: 0, conversation_tokens: 0, remaining_tokens: budget.compact_at_tokens, archived_entries: 0, archived_tokens: 0, compactions: 0 });
        setContextThreshold(budget.threshold_percent);
        setContextWindowOverride(budget.source === "manual" ? String(budget.context_window_tokens) : "");
      })
      .catch(() => setContextStatus(null));
  };

  const refreshModelCapabilities = (targetProvider = selectedProvider, targetModel = model) => {
    if (!targetModel) { setModelCapabilities(null); return; }
    const saved = localStorage.getItem(`pok_model_settings:${targetProvider}:${targetModel}`);
    let settings: Record<string, unknown> | null = null;
    if (saved) {
      try {
        settings = JSON.parse(saved) as Record<string, unknown>;
        setManualCapabilityOverride(Boolean(settings.manualCapabilityOverride));
        setVisionMode((settings.visionMode as "auto" | "on" | "off") || "auto");
        setRequestTemperature(typeof settings.temperature === "string" ? settings.temperature : "");
        setRequestMaxOutput(typeof settings.maxOutput === "string" ? settings.maxOutput : "");
        setRequestSeed(typeof settings.seed === "string" ? settings.seed : "42");
        setFullContext(Boolean(settings.fullContext));
      } catch { /* Ignore a corrupt local preference and use conservative defaults. */ }
    } else {
      setManualCapabilityOverride(false); setVisionMode("auto"); setReasoningEffort("auto");
      setRequestTemperature(""); setRequestMaxOutput(""); setRequestSeed("42");
      setFullContext(false);
    }
    setModelCapabilities(null);
    invoke<ModelCapabilities>("get_model_capabilities", { provider: targetProvider, model: targetModel })
      .then((capabilities) => {
        setModelCapabilities(capabilities);
        const manualOverride = Boolean(settings?.manualCapabilityOverride);
        const savedEffort = typeof settings?.reasoningEffort === "string" ? settings.reasoningEffort : "auto";
        const settingsVersion = typeof settings?.version === "number" ? settings.version : 0;
        const reportedDefault = capabilities.reasoning_default
          && capabilities.reasoning_efforts.includes(capabilities.reasoning_default)
          ? capabilities.reasoning_default
          : "auto";
        let selectedEffort = settings && (settingsVersion >= 2 || savedEffort !== "auto")
          ? savedEffort
          : reportedDefault;
        if (selectedEffort !== "auto"
          && !manualOverride
          && !capabilities.reasoning_efforts.includes(selectedEffort)) {
          selectedEffort = reportedDefault;
        }
        setReasoningEffort(selectedEffort);
      })
      .catch(() => {
        setModelCapabilities(null);
        setReasoningEffort(Boolean(settings?.manualCapabilityOverride)
          && typeof settings?.reasoningEffort === "string"
          ? settings.reasoningEffort
          : "auto");
      });
  };

  const saveModelRequestSettings = () => {
    if (!model) return;
    localStorage.setItem(`pok_model_settings:${selectedProvider}:${model}`, JSON.stringify({
      version: 2,
      manualCapabilityOverride, visionMode, reasoningEffort,
      temperature: requestTemperature, maxOutput: requestMaxOutput, seed: requestSeed,
      fullContext,
    }));
  };

  const saveContextOverride = async () => {
    if (!model) return;
    await invoke("save_model_context_override", {
      provider: selectedProvider,
      model,
      contextWindowTokens: contextWindowOverride.trim() ? Number(contextWindowOverride) : null,
      thresholdPercent: contextThreshold,
    });
    refreshModelContext();
  };
  const refreshMemoryLibrary = () => invoke<MemoryLibrary>("list_memory_library").then(setMemoryLibrary).catch(() => undefined);
  const refreshGeneratedTools = () => {
    invoke<GeneratedTool[]>("list_generated_tools", { workspace }).then(setGeneratedTools).catch(() => setGeneratedTools([]));
    invoke<GeneratedToolCandidate[]>("list_generated_tool_candidates_command", { workspace }).then(setGeneratedToolCandidates).catch(() => setGeneratedToolCandidates([]));
  };
  const toggleGeneratedTool = async (tool: GeneratedTool) => {
    await invoke("set_generated_tool_enabled", { name: tool.name, enabled: !tool.enabled, workspace });
    refreshGeneratedTools();
  };
  const removeGeneratedTool = async (tool: GeneratedTool) => {
    const confirmed = window.confirm(
      `Delete ${tool.name}?\n\nThis permanently removes the promoted AppData copy and its isolated environment for this workspace. The original helper source file is not deleted.`,
    );
    if (!confirmed) return;
    try {
      const deleted = await invoke<boolean>("delete_generated_tool_command", { name: tool.name, workspace });
      setMessages((items) => [...items, {
        id: `generated-tool-delete-${Date.now()}`,
        sender: "system",
        type: "activity",
        text: deleted
          ? `Deleted ${tool.name} and its isolated generated-tool environment.`
          : `${tool.name} was already absent.`,
        timestamp: new Date(),
      }]);
      refreshGeneratedTools();
    } catch (error) {
      setLiveActivity({
        phase: "failed",
        label: "Could not delete generated tool",
        detail: compactValue(String(error), 160),
        startedAt: Date.now(),
      });
    }
  };

  useEffect(() => {
    refreshModelContext();
    const localTransition = (selectedProvider === "lm_studio" || selectedProvider === "ollama")
      && ["loading", "checking", "unloading"].includes(modelRuntime?.phase ?? "");
    if (!localTransition) refreshModelCapabilities();
  }, [selectedProvider, model, modelRuntime?.phase]);

  useEffect(() => {
    const wideLayout = window.matchMedia("(min-width: 900px)");
    const updateLayout = (event: MediaQueryListEvent) => setNavigationOpen(event.matches);
    wideLayout.addEventListener("change", updateLayout);
    return () => wideLayout.removeEventListener("change", updateLayout);
  }, []);

  useEffect(() => {
    if (!settingsOpen) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key !== "Escape" || approval || questionRequest || showCloudModal) return;
      setSettingsOpen(false);
      settingsButtonRef.current?.focus();
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [settingsOpen, approval, questionRequest, showCloudModal]);

  const closeSettings = () => {
    setSettingsOpen(false);
    settingsButtonRef.current?.focus();
  };

  useEffect(() => {
    if (busy || !model || (selectedProvider !== "lm_studio" && selectedProvider !== "ollama")) return;
    const key = `${selectedProvider}\n${model}`;
    if (modelRuntime?.provider !== selectedProvider || modelRuntime.model !== model || modelRuntime.phase !== "unloaded" || initialLoadAttempt.current === key) return;
    initialLoadAttempt.current = key;
    void invoke<LocalModelRuntimeStatus>("load_model", { provider: selectedProvider, model })
      .then((loaded) => {
        setModelRuntime(loaded);
        if (loaded.phase === "loaded") activeLocalSelection.current = { provider: selectedProvider, model };
      })
      .catch(() => undefined);
  }, [busy, model, modelRuntime, selectedProvider]);

  useLayoutEffect(() => {
    const console = consoleRef.current;
    if (console && followOutput.current) {
      console.scrollTop = console.scrollHeight;
    }
  }, [messages, busy]);

  useEffect(() => {
    activeConversationIdRef.current = activeConversationId;
  }, [activeConversationId]);

  const handleConsoleScroll = () => {
    const console = consoleRef.current;
    if (!console) return;
    if (console.scrollTop <= 80 && historyHasMore && !historyLoading) {
      void loadEarlierHistory();
    }
    const distanceFromBottom = console.scrollHeight - console.scrollTop - console.clientHeight;
    followOutput.current = distanceFromBottom <= 80;
    setShowJump(!followOutput.current);
  };

  // Scrolling up (wheel) or pressing anywhere on the console (including the
  // scrollbar thumb/track) is a manual-scroll intent. It must disable
  // auto-follow synchronously, before the next streamed message's
  // layout-effect snap runs — otherwise the snap wins the race and the view
  // stays pinned to the bottom while the user is trying to scroll up.
  const handleConsoleWheel = (event: React.WheelEvent<HTMLDivElement>) => {
    if (event.deltaY < 0) {
      followOutput.current = false;
      setShowJump(true);
    }
  };

  const handleConsolePointerDown = () => {
    followOutput.current = false;
    setShowJump(true);
  };

  useEffect(() => {
    checkApiStatus();
    invoke<string[]>("list_providers").then(setProviders).catch(() => undefined);
    invoke<DecisionRouterStatus>("get_decision_router_status").then((next) => {
      setDecisionRouter(next);
      setJevEnabled(next.enabled);
      setDecisionRouterBackend(next.enabled ? next.backend : "off");
      setLayaDevice(next.laya.preference ?? "auto");
      setJevRuntimeState(next.backend === "jev" && !next.has_api_key ? "unavailable" : next.enabled ? "ready" : "off");
    }).catch(() => undefined);
    refreshMemoryLibrary();
    refreshGeneratedTools();
    refreshConversations();
    
    const unlisten = listen<Approval>("approval_requested", ({ payload }) => {
      activeApproval.current = payload;
      setApprovalChoice(null);
      setApprovalError("");
      setApproval(payload);
      setLiveActivity({
        phase: "waiting",
        label: "Approval required",
        detail: payload.tool,
        startedAt: Date.now(),
      });
    });
    const unlistenStop = listen("emergency_stopped", () => {
      emergencyStopRequested.current = true;
      activeApproval.current = null;
      setPauseState("running");
      setApproval(null);
      setApprovalChoice(null);
      setLiveActivity({ phase: "failed", label: "Stopped by user", startedAt: Date.now() });
      setMessages((previous) => {
        const stoppedAt = Date.now();
        const messages = previous.map((message) => message.type === "activity"
          && message.activityStatus === "running"
          ? {
              ...message,
              activityStatus: "failed" as const,
              activityDetail: "Stopped by user",
              durationMs: stoppedAt - message.timestamp.getTime(),
            }
          : message);
        return [...messages, {
          id: 'stop-' + Date.now(),
          sender: "system",
          type: "error",
          text: "Emergency stop requested.",
          timestamp: new Date()
        }];
      });
      setBusy(false);
    });
    const unlistenModelRuntime = listen<LocalModelRuntimeStatus>("model_runtime_event", ({ payload }) => {
      setModelRuntime(payload);
      if (payload.phase === "loading") {
        setLiveActivity({ phase: "starting", label: `Loading ${payload.model}`, detail: "Preparing the local model", startedAt: Date.now() });
      } else if (payload.phase === "unloading") {
        setLiveActivity({ phase: "starting", label: `Unloading ${payload.model}`, detail: "Releasing GPU VRAM", startedAt: Date.now() });
      } else if (payload.phase === "loaded") {
        activeLocalSelection.current = { provider: payload.provider, model: payload.model };
        setLiveActivity((current) => current.phase === "starting"
          && (current.label.startsWith("Loading ") || current.label.startsWith("Unloading "))
          ? { phase: "idle", label: "Ready for a task", startedAt: Date.now() }
          : current);
      } else if (payload.phase === "server_offline" || payload.phase === "error") {
        setLiveActivity({ phase: "failed", label: payload.phase === "server_offline" ? "Local model server is offline" : "Local model unavailable", detail: compactValue(payload.detail, 160), startedAt: Date.now() });
      }
    });
    const unlistenLayaRuntime = listen<Partial<LayaRuntimeStatus>>("laya_runtime_event", ({ payload }) => {
      setDecisionRouter((current) => current ? { ...current, laya: { ...current.laya, ...payload } } : current);
      if (payload.preference) setLayaDevice(payload.preference);
      setInstallingLaya(payload.phase === "installing");
      if (decisionRouterBackendRef.current === "laya") {
        setJevRuntimeState(payload.phase === "ready" ? "ready" : payload.phase === "error" ? "unavailable" : "evaluating");
      }
    });

    const unlistenJudgeRuntime = listen<Partial<JudgeRuntimeStatus>>("judge_runtime_event", ({ payload }) => {
      setDecisionRouter((current) => current ? { ...current, judge: { ...current.judge, ...payload } } : current);
      setInstallingJudge(payload.phase === "installing");
    });

    const unlistenAgent = listen<AgentEvent>("agent_event", ({ payload }) => {
      switch (payload.type) {
        case "run_started":
          if (payload.session_id) {
            activeConversationIdRef.current = payload.session_id;
            setActiveConversationId(payload.session_id);
          }
          setJevMetrics(emptyJevMetrics());
          setPauseState("running");
          setUsage({ currentPrompt: 0, currentCompletion: 0, totalPrompt: 0, totalCompletion: 0 });
          setCurrentTurn(0);
          setLiveActivity({ phase: "starting", label: "Preparing the task", startedAt: Date.now() });
          setJevRuntimeState((current) => current === "evaluating" ? "ready" : current);
          setJevWarning(null);
          jevSelectedTool.current = null;
          break;
        case "decision_router_started": {
          const callId = `jev-${payload.purpose ?? "decision"}-${payload.turn ?? 0}`;
          setJevRuntimeState("evaluating");
          setJevWarning(null);
          setLiveActivity({ phase: "thinking", label: `${activeRouterName()} is evaluating options`, detail: `${payload.candidate_count ?? 0} bounded candidates`, startedAt: Date.now() });
          // Only action routing gets a chat row; optional-context/intent
          // routing is plumbing that stays out of the conversation feed.
          if ((payload.purpose ?? "decision") !== "next_action") break;
          setMessages((previous) => [...previous, {
            id: `${callId}-${Date.now()}`,
            sender: "system",
            type: "activity",
            text: `${activeRouterName()} is evaluating options`,
            timestamp: new Date(),
            activityTool: "jev_decision",
            activityCallId: callId,
            activityLabel: `${activeRouterName()} is evaluating options`,
            activityDetail: `${payload.candidate_count ?? 0} bounded candidates`,
            activityStatus: "running",
          }]);
          break;
        }
        case "decision_router_cache_hit":
          setJevMetrics((current) => ({ ...current, cached: current.cached + 1, last: "Unchanged state · no repeated JEV request" }));
          setJevRuntimeState("ready");
          setJevWarning(null);
          break;
        case "decision_router_evaluated":
          setJevRuntimeState("ready");
          setJevWarning(payload.eligible ? null : "Deferred this step");
          jevSelectedTool.current = payload.eligible ? payload.tool ?? null : null;
          setJevMetrics((current) => ({
            ...current,
            evaluations: current.evaluations + 1,
            eligible: current.eligible + (payload.eligible ? 1 : 0),
            latencyMs: current.latencyMs + (payload.elapsed_ms ?? 0),
            last: payload.eligible
              ? `${payload.tool ?? payload.candidate_id ?? "action"} accepted`
              : `Fallback: ${(payload.rejection_reason ?? "not eligible").replaceAll("_", " ")}`,
          }));
          setMessages((previous) => {
            // Optional-context/intent routing has no user-facing "deferral":
            // when the router keeps the local ranking, nothing was declined
            // and no chat row is needed. Only announce when it added context.
            const purpose = payload.purpose ?? "next_action";
            if (purpose !== "next_action" && !payload.eligible) return previous;
            const callId = `jev-${purpose}-${payload.turn ?? 0}`;
            const index = [...previous].reverse().findIndex((message) => message.activityTool === "jev_decision" && message.activityCallId === callId && message.activityStatus === "running");
            const routerName = activeRouterName();
            const isTerminal = payload.candidate_id === "jev_done" || payload.candidate_id === "jev_blocked";
            const label = payload.eligible
              ? isTerminal
                ? payload.candidate_id === "jev_done" ? `${routerName} finished (evidence sufficient)` : `${routerName} blocked — no safe action`
                : payload.tool === "focus_evidence"
                  ? `${routerName} selected relevant evidence`
                  : purpose === "next_action" ? `${routerName} selected a candidate`
                  : purpose === "retrieval_intent" ? `${routerName} classified the request`
                  : purpose === "context_selection" ? `${routerName} added optional context`
                  : `${routerName} selected a candidate`
              : purpose === "next_action"
                ? `${routerName} deferred to the main model`
                : `${routerName} used local ranking`;
            const detail = payload.eligible
              ? purpose === "next_action"
                ? `${payload.description ?? payload.tool ?? payload.candidate_id ?? "Candidate"} · op ${Math.round((payload.operation_probability ?? payload.selected_probability ?? 0) * 100)}% · target ${Math.round((payload.target_probability ?? payload.selected_probability ?? 0) * 100)}%${payload.operation_confidence == null ? "" : ` · ${Math.round(payload.operation_confidence * 100)}% confidence`}`
                : `${payload.description ?? payload.candidate_id ?? "context"}`
              : `${(payload.rejection_reason ?? "not eligible").replaceAll("_", " ")} · ${payload.elapsed_ms ?? 0} ms`;
            const result = {
              selected_candidate: payload.description ?? payload.candidate_id,
              candidate_id: payload.candidate_id,
              tool: payload.tool,
              probability: payload.selected_probability,
              confidence: payload.confidence,
              operation_probability: payload.operation_probability,
              target_probability: payload.target_probability,
              operation_confidence: payload.operation_confidence,
              target_confidence: payload.target_confidence,
              probability_threshold: payload.probability_threshold,
              confidence_threshold: payload.confidence_threshold,
              alternatives: payload.alternatives ?? [],
              rejection_reason: payload.rejection_reason,
            };
            if (index === -1) return [...previous, {
              id: `${callId}-${Date.now()}`, sender: "system", type: "activity", text: label, timestamp: new Date(),
              activityTool: "jev_decision", activityCallId: callId, activityLabel: label, activityDetail: detail,
              activityStatus: "done", activityResult: result,
            }];
            const actualIndex = previous.length - 1 - index;
            return previous.map((message, messageIndex) => messageIndex === actualIndex ? {
              ...message, text: label, activityLabel: label, activityDetail: detail, activityStatus: "done", activityResult: result,
              durationMs: payload.elapsed_ms,
            } : message);
          });
          break;
        case "decision_router_judge_attempted": {
          const toolName = payload.tool ?? "candidate";
          const candidateId = payload.candidate_id ?? "pick";
          const votes = (payload.votes ?? {}) as Record<string, string>;
          const voteSummary = ["reversible", "cheap", "evidenced"]
            .map((question) => `${question.slice(0, 4)}:${votes[question] ?? "?"}`)
            .join(" ");
          const comparativeNote = votes.comparative
            ? ` · ${votes.comparative === "b" ? "runner-up preferred" : "pick preferred"}`
            : "";
          const label = payload.promoted
            ? `Zeiger approved: ${toolName}`
            : `Zeiger held: ${toolName}`;
          const detail = `${voteSummary}${comparativeNote}${payload.reused ? " · cached verdict" : ""}${payload.reason ? ` · ${payload.reason.replaceAll("_", " ")}` : ""}`;
          setJevMetrics((current) => ({
            ...current,
            judgeAttempts: current.judgeAttempts + 1,
            judgePromotions: current.judgePromotions + (payload.promoted ? 1 : 0),
            last: payload.promoted ? `Zeiger approved ${toolName}` : `Zeiger held ${toolName}`,
          }));
          setLiveActivity(payload.promoted
            ? { phase: "acting", label: `Zeiger approved: ${toolName}`, detail, startedAt: Date.now() }
            : { phase: "thinking", label: "Zeiger held the pick", detail, startedAt: Date.now() });
          setMessages((previous) => [...previous, {
            id: `jev-judge-${candidateId}-${Date.now()}`,
            sender: "system",
            type: "activity",
            text: label,
            timestamp: new Date(),
            activityTool: "jev_judge",
            activityCallId: `jev-judge-${payload.turn ?? 0}-${candidateId}`,
            activityLabel: label,
            activityDetail: detail,
            activityStatus: payload.promoted ? "done" : undefined,
            activityResult: { promoted: payload.promoted, votes, reused: payload.reused },
          }]);
          break;
        }
        case "decision_router_action_outcome":
          setJevRuntimeState("ready");
          setJevWarning(payload.ok ? null : "Selected action did not succeed");
          setJevMetrics((current) => ({
            ...current,
            actions: current.actions + (payload.tool === "focus_evidence" ? 0 : 1),
            evidence: current.evidence + (payload.tool === "focus_evidence" ? 1 : 0),
            progress: current.progress + (payload.outcome === "progress" ? 1 : 0),
            failures: current.failures + (payload.ok && payload.outcome !== "failed" ? 0 : 1),
            last: `${payload.tool ?? "action"}: ${(payload.outcome ?? "unknown").replaceAll("_", " ")}`,
          }));
          break;
        case "decision_router_memory_outcome":
          if (payload.session_id && payload.session_id !== activeConversationIdRef.current) break;
          setJevMetrics((current) => ({
            ...current,
            last: `Memory ${(payload.disposition ?? "reviewed").replaceAll("_", " ")}${payload.used_jev ? ` by ${activeRouterName()}` : " locally"}`,
          }));
          setMessages((previous) => [...previous, {
            id: `memory-decision-${Date.now()}`,
            sender: "system",
            type: "activity",
            text: "Memory review completed",
            timestamp: new Date(),
            activityTool: "jev_memory",
            activityLabel: `Memory ${(payload.disposition ?? "reviewed").replaceAll("_", " ")}`,
            activityDetail: payload.used_jev ? `Verified by ${activeRouterName()}` : "Local memory rules",
            activityStatus: "done",
            activityResult: { disposition: payload.disposition, related_memory_id: payload.related_memory_id },
          }]);
          refreshMemoryLibrary();
          break;
        case "compression_started": {
          const sequence = payload.sequence ?? 1;
          const activityKey = `context_compression_${sequence}`;
          setLiveActivity({
            phase: "compacting",
            label: "Condensing conversation history",
            detail: payload.trigger === "manual"
              ? "Requested manually"
              : "Keeping the active task and recent results",
            startedAt: Date.now(),
          });
          setMessages((previous) => {
            const existing = previous.findIndex(message =>
              message.type === "activity" && message.activityTool === activityKey
            );
            const activity: ChatMessage = {
              id: `compression-${sequence}-${Date.now()}`,
              sender: "system",
              type: "activity",
              text: "Condensing conversation history",
              timestamp: new Date(),
              activityTool: activityKey,
              activityLabel: "Condensing conversation history",
              activityDetail: payload.trigger === "manual" ? "Manual compaction" : undefined,
              activityStatus: "running",
              activityAttempts: existing >= 0 ? previous[existing].activityAttempts : undefined,
            };
            if (existing < 0) return [...previous, activity];
            return previous.map((message, index) =>
              index === existing ? { ...activity, id: message.id } : message
            );
          });
          break;
        }
        case "compression_completed": {
          const sequence = payload.sequence ?? 1;
          const activityKey = `context_compression_${sequence}`;
          const duration = formatElapsed(payload.elapsed_ms ?? 0);
          const summaryMethod = payload.summary_source === "model_retry"
            ? `LLM summary succeeded after ${payload.attempts ?? 2} attempts`
            : "LLM summary completed";
          setLiveActivity({
            phase: "thinking",
            label: "Conversation history condensed",
            detail: `${summaryMethod} in ${duration}`,
            startedAt: Date.now(),
          });
          setMessages((previous) => {
            const index = [...previous].reverse().findIndex(
              message => message.type === "activity"
                && message.activityTool === activityKey
                && message.activityStatus === "running"
            );
            if (index === -1) {
              return [...previous, {
                id: `compression-complete-${sequence}-${Date.now()}`,
                sender: "system",
                type: "activity",
                text: "Context condensed",
                timestamp: new Date(),
                activityTool: activityKey,
                activityLabel: "Context condensed",
                activityDetail: `Compaction ${sequence} · ${summaryMethod} · ${duration}`,
                activityStatus: "done",
              }];
            }
            const actualIndex = previous.length - 1 - index;
            return [
              ...previous.slice(0, actualIndex),
              {
                ...previous[actualIndex],
                text: "Context condensed",
                activityLabel: "Context condensed",
                activityDetail: `Compaction ${sequence} · ${summaryMethod} · ${duration}`,
                activityStatus: "done",
              },
              ...previous.slice(actualIndex + 1),
            ];
          });
          break;
        }
        case "compression_failed": {
          const sequence = payload.sequence ?? 1;
          const activityKey = `context_compression_${sequence}`;
          setLiveActivity({
            phase: "thinking",
            label: "LLM context compaction did not complete",
            detail: "Canonical conversation history was preserved",
            startedAt: Date.now(),
          });
          setMessages((previous) => {
            const index = [...previous].reverse().findIndex(
              message => message.type === "activity"
                && message.activityTool === activityKey
            );
            const detail = compactValue(payload.error, 120) ?? "Summary unavailable";
            if (index === -1) {
              return [...previous, {
                id: `compression-failed-${sequence}-${Date.now()}`,
                sender: "system",
                type: "activity",
                text: "LLM context compaction failed; history preserved",
                timestamp: new Date(),
                activityTool: activityKey,
                activityLabel: "LLM context compaction failed",
                activityDetail: detail,
                activityStatus: "done",
                activityAttempts: 1,
              }];
            }
            const actualIndex = previous.length - 1 - index;
            const attempts = (previous[actualIndex].activityAttempts ?? 0) + 1;
            return previous.flatMap((message, messageIndex) => {
              if (message.type === "activity"
                && message.activityTool === activityKey
                && messageIndex !== actualIndex) return [];
              if (messageIndex !== actualIndex) return [message];
              return [{
                ...message,
                text: "LLM context compaction failed; history preserved",
                activityLabel: "LLM context compaction failed",
                activityDetail: `${detail} · ${attempts} attempts`,
                activityStatus: "done",
                activityAttempts: attempts,
              }];
            });
          });
          break;
        }
        case "turn_started":
          setCurrentTurn(payload.turn ?? 0);
          setLiveActivity({
            phase: "thinking",
            label: "Thinking through the next step",
            detail: payload.turn ? `Turn ${payload.turn}` : undefined,
            startedAt: Date.now(),
          });
          break;
        case "model_request_dispatched":
          setLiveActivity({
            phase: "thinking",
            label: "Waiting for the model",
            detail: `${(payload.estimated_prompt_tokens ?? 0).toLocaleString()} prompt tokens · ${payload.tool_count ?? 0} tools`,
            startedAt: Date.now(),
          });
          break;
        case "provider_first_event":
          setLiveActivity({
            phase: "thinking",
            label: payload.event_kind === "reasoning" ? "Model is reasoning" : "Model is responding",
            detail: `First provider event after ${formatElapsed(payload.elapsed_ms ?? 0)}`,
            startedAt: Date.now(),
          });
          break;
        case "reasoning_delta":
          setLiveActivity((current) => current.phase === "thinking"
            ? current
            : { phase: "thinking", label: "Reasoning about the task", startedAt: Date.now() });
          setMessages((prev) => appendStreamDelta(prev, "reasoning", payload.text ?? ""));
          break;
        case "text_delta":
          setLiveActivity((current) => current.phase === "completed"
            ? current
            : { phase: "thinking", label: "Preparing the response", startedAt: Date.now() });
          setMessages((prev) => appendStreamDelta(prev, "response", payload.text ?? ""));
          break;
        case "tool_started": {
          const toolName = payload.name ?? "tool";
          const callId = payload.call_id ?? `legacy-${toolName}-${Date.now()}`;
          const activity = describeTool(toolName, payload.arguments);
          const chosenByJev = jevSelectedTool.current === toolName;
          if (chosenByJev) jevSelectedTool.current = null;
          const toolArgs = asRecord(payload.arguments);
          setLiveActivity({ ...activity, tool: toolName, startedAt: Date.now() });
          setMessages((prev) => [
            ...prev,
            {
              id: 'tool-' + callId,
              sender: "agent",
              type: "activity",
              text: activity.label,
              timestamp: new Date(),
              activityTool: toolName,
              activityCallId: callId,
              activityLabel: activity.label,
              activityDetail: chosenByJev ? `${activity.detail ? `${activity.detail} · ` : ""}Chosen by ${activeRouterName()}` : activity.detail,
              activityStatus: "running",
              activityArgs: toolArgs,
            }
          ]);
          break;
        }
        case "tool_delayed": {
          const stage = payload.stage ?? "tool_execution";
          const label = stage === "desktop_capture_or_accessibility_enrichment"
            ? "Waiting for screenshot, OCR, or UI Automation"
            : stage === "native_window_operation"
              ? "Waiting for the Windows window manager"
              : stage === "desktop_action_and_verification"
                ? "Waiting for desktop action verification"
                : stage === "command_execution"
                  ? "Waiting for the command"
                  : `Waiting for ${(payload.name ?? "tool").replaceAll("_", " ")}`;
          setLiveActivity({
            phase: "acting",
            label,
            detail: `${formatElapsed(payload.elapsed_ms ?? 0)} elapsed`,
            tool: payload.name,
            startedAt: Date.now(),
          });
          break;
        }
        case "tool_finished": {
          const toolName = payload.name ?? "tool";
          const callId = payload.call_id;
          const fallback = describeTool(toolName, {});
          const resultRecord = asRecord(payload.result);
          setLiveActivity(payload.ok
            ? { phase: "thinking", label: "Reviewing the result", detail: fallback.label, startedAt: Date.now() }
            : { phase: "failed", label: `${fallback.label} failed`, detail: compactValue(payload.detail, 140), startedAt: Date.now() });
          setMessages((prev) => {
            const idx = [...prev].reverse().findIndex(
              m => m.type === "activity"
                && m.activityStatus === "running"
                && (callId ? m.activityCallId === callId : m.activityTool === toolName)
            );
            if (idx !== -1) {
              const actualIdx = prev.length - 1 - idx;
              const original = prev[actualIdx];
              const durationMs = Date.now() - original.timestamp.getTime();
              return [
                ...prev.slice(0, actualIdx),
                {
                  ...original,
                  activityStatus: resultRecord.status === "running" ? "running" : payload.ok ? "done" : "failed",
                  activityDetail: payload.ok
                    ? withInputMethod(original.activityDetail, resultRecord.input_method)
                    : compactValue(payload.detail, 140) ?? original.activityDetail,
                  activityResult: resultRecord,
                  durationMs,
                },
                ...prev.slice(actualIdx + 1)
              ];
            }
            return [
              ...prev,
              {
                id: 'tool-fin-' + Date.now(),
                sender: "agent",
                type: "activity",
                text: fallback.label,
                timestamp: new Date(),
                activityTool: toolName,
                activityCallId: callId,
                activityLabel: fallback.label,
                activityStatus: payload.ok ? "done" : "failed",
                activityDetail: payload.ok ? undefined : compactValue(payload.detail, 140),
                activityResult: resultRecord,
              }
            ];
          });
          break;
        }
        case "command_progress": {
          const progress = payload.event;
          if (!progress) break;
          setMessages((previous) => previous.map((message) => {
            if (message.activityTool !== "run_command" || message.activityCallId !== progress.call_id) return message;
            const current = asRecord(message.activityResult);
            const priorOutput = String(current.output ?? "");
            const appended = progress.kind === "output" ? priorOutput + (progress.chunk ?? "") : priorOutput;
            const bounded = appended.length > 64 * 1024 ? appended.slice(-(64 * 1024)) : appended;
            const terminal = ["completed", "failed", "timed_out", "cancelled", "killed"].includes(progress.status);
            return {
              ...message,
              activityStatus: terminal ? (progress.status === "completed" ? "done" : "failed") : "running",
              activityResult: {
                ...current,
                task_id: progress.task_id,
                output: bounded,
                total_bytes: progress.total_bytes,
                status: progress.status,
                exit_code: progress.exit_code,
                termination_reason: progress.termination_reason,
                ui_output_truncated: appended.length > 64 * 1024,
              },
            };
          }));
          break;
        }
        case "usage":
          setUsage({
            currentPrompt: payload.prompt_tokens ?? 0,
            currentCompletion: payload.completion_tokens ?? 0,
            totalPrompt: payload.cumulative_prompt_tokens ?? 0,
            totalCompletion: payload.cumulative_completion_tokens ?? 0,
          });
          break;
        case "context_status":
          if (payload.budget) {
            setContextStatus({
              budget: payload.budget,
              prompt_tokens: payload.prompt_tokens ?? 0,
              conversation_tokens: payload.conversation_tokens ?? payload.prompt_tokens ?? 0,
              working_set_target_tokens: payload.working_set_target_tokens,
              remaining_tokens: payload.remaining_tokens ?? payload.budget.compact_at_tokens,
              archived_entries: payload.archived_entries ?? 0,
              archived_tokens: payload.archived_tokens ?? 0,
              approximate_turns_remaining: payload.approximate_turns_remaining,
              compactions: payload.compactions ?? 0,
            });
          }
          break;
        case "provider_stall_retry":
          setLiveActivity({ phase: "waiting", label: "Giving the model another try", detail: `Attempt ${payload.attempt ?? 1} of ${payload.maximum ?? 3}`, startedAt: Date.now() });
          setMessages((prev) => [...prev, {
            id: 'stall-' + Date.now(), sender: "system", type: "activity",
            text: `Provider returned no action; retrying (${payload.attempt ?? 1}/${payload.maximum ?? 3})…`, timestamp: new Date()
          }]);
          break;
        case "provider_retry":
          setLiveActivity({ phase: "waiting", label: "Waiting for the model provider", detail: `Retrying in ${Math.ceil((payload.delay_ms ?? 0) / 1000)}s`, startedAt: Date.now() });
          setMessages((prev) => [...prev, {
            id: 'provider-retry-' + Date.now(), sender: "system", type: "activity",
            text: `Provider is busy or rate-limited; retrying in ${Math.ceil((payload.delay_ms ?? 0) / 1000)}s (attempt ${payload.attempt ?? 1}${payload.maximum == null ? ", cancel to stop" : `/${payload.maximum}`})…`,
            timestamp: new Date()
          }]);
          break;
        case "guidance_injected":
          setMessages((prev) => {
            const lastUserMsg = [...prev].reverse().find(m => m.sender === "user" && m.type === "guidance");
            if (lastUserMsg && lastUserMsg.text === payload.text) {
              return prev;
            }
            return [
              ...prev,
              {
                id: 'guidance-' + Date.now(),
                sender: "user",
                type: "guidance",
                text: payload.text ?? "",
                timestamp: new Date()
              }
            ];
          });
          break;
        case "task_state_changed":
          setTaskProgress({
            rootRequest: payload.root_request ?? "",
            status: payload.status ?? "in_progress",
            currentStep: payload.current_step ?? "Working on the active request",
            steps: payload.steps ?? [],
          });
          if (payload.status === "completed") {
            setLiveActivity({ phase: "thinking", label: "Wrapping up the result", startedAt: Date.now() });
          } else if (payload.current_step) {
            setLiveActivity((current) => ({
              ...current,
              detail: current.detail ?? payload.current_step,
            }));
          }
          break;
        case "action_blocked":
          setLiveActivity({ phase: "waiting", label: "Action blocked", detail: compactValue(payload.reason, 140), startedAt: Date.now() });
          setMessages((prev) => [
            ...prev,
            {
              id: 'guard-' + Date.now(),
              sender: "system",
              type: "activity",
              text: `⛔ ${payload.tool ?? "guard"}: ${payload.reason ?? "off-task action blocked"}`,
              timestamp: new Date()
            }
          ]);
          break;
        case "model_finished":
          setLiveActivity({ phase: "thinking", label: "Reviewing the model response", startedAt: Date.now() });
          break;
        case "model_output_limit_reached":
          setLiveActivity({
            phase: "thinking",
            label: "Reasoning reached its output limit — continuing concisely",
            detail: `Next response allowance: ${Number(payload.next_max_tokens ?? 0).toLocaleString()} tokens`,
            startedAt: Date.now(),
          });
          break;
        case "pause_requested":
          setPauseState("requested");
          setLiveActivity({ phase: "waiting", label: "Pausing at a safe boundary", startedAt: Date.now() });
          break;
        case "paused":
          setPauseState("paused");
          setLiveActivity({
            phase: "paused",
            label: "Paused — you have control",
            detail: payload.interrupted_phase ? `Stopped during ${payload.interrupted_phase.replaceAll("_", " ")}` : undefined,
            startedAt: Date.now(),
          });
          break;
        case "model_turn_interrupted":
          setMessages((previous) => {
            const retained = [...previous];
            while (retained.at(-1)?.sender === "agent" && ["reasoning", "response"].includes(retained.at(-1)?.type ?? "")) {
              retained.pop();
            }
            return [...retained, {
              id: `pause-interrupt-${Date.now()}`,
              sender: "system",
              type: "activity",
              text: payload.reason ?? "Partial model output discarded because the agent was paused.",
              timestamp: new Date(),
            }];
          });
          break;
        case "environment_changed":
          setLiveActivity({
            phase: "waiting",
            label: "User changed the active window — yielding control",
            detail: payload.current_window ?? "Desktop focus changed",
            startedAt: Date.now(),
          });
          setMessages((previous) => [...previous, {
            id: `environment-change-${payload.revision ?? Date.now()}`,
            sender: "system",
            type: "activity",
            text: "Desktop environment changed",
            timestamp: new Date(),
            activityTool: "environment_recovery",
            activityCallId: `environment-${payload.revision ?? 0}`,
            activityLabel: "User changed the active window — yielding control",
            activityDetail: `${payload.previous_window ?? "Unknown window"} → ${payload.current_window ?? "Unknown window"}`,
            activityStatus: "running",
          }]);
          break;
        case "user_activity_wait":
          setWaitingForUser(Boolean(payload.waiting));
          if (payload.waiting) {
            setLiveActivity({
              phase: "waiting",
              label: "Waiting for you",
              detail: "You're using the mouse or keyboard. The agent continues when you stop.",
              startedAt: Date.now(),
            });
          }
          break;
        case "environment_waiting":
          setLiveActivity({
            phase: "waiting",
            label: "Waiting for you to finish",
            detail: `${Math.round((payload.quiet_period_ms ?? 3000) / 1000)} seconds of inactivity before recovery`,
            startedAt: Date.now(),
          });
          break;
        case "environment_recovered":
          setJevRuntimeState("ready");
          setJevWarning(payload.used_jev ? null : "Desktop recovery used local policy");
          setLiveActivity({
            phase: "starting",
            label: "Desktop context recovered",
            detail: `${(payload.action ?? "recovered").replaceAll("_", " ")}${payload.used_jev ? ` · chosen by ${activeRouterName()}` : " · local policy"}`,
            startedAt: Date.now(),
          });
          setMessages((previous) => {
            const callId = `environment-${payload.revision ?? 0}`;
            const match = [...previous].reverse().findIndex((message) => message.activityCallId === callId && message.activityStatus === "running");
            const actualIndex = match === -1 ? -1 : previous.length - 1 - match;
            const replacement = {
              id: `${callId}-recovered`, sender: "system" as const, type: "activity" as const,
              text: "Desktop context recovered", timestamp: new Date(), activityTool: "environment_recovery",
              activityCallId: callId, activityLabel: "Desktop context recovered",
              activityDetail: `${(payload.action ?? "recovered").replaceAll("_", " ")} · ${payload.used_jev ? `Chosen by ${activeRouterName()}` : "Local recovery"}`,
              activityStatus: "done" as const, durationMs: payload.elapsed_ms,
              activityResult: { revision: payload.revision, action: payload.action, used_jev: payload.used_jev },
            };
            return actualIndex === -1
              ? [...previous, replacement]
              : previous.map((message, index) => index === actualIndex ? { ...message, ...replacement } : message);
          });
          break;
        case "resumed":
          setPauseState("running");
          setLiveActivity({
            phase: "starting",
            label: "Rechecking the desktop",
            detail: payload.note ? `Your note: ${compactValue(payload.note, 100)}` : "Previous screen state was cleared",
            startedAt: Date.now(),
          });
          setMessages((previous) => [...previous, {
            id: `resumed-${Date.now()}`,
            sender: "system",
            type: "activity",
            text: payload.note
              ? `Agent resumed. Desktop state will be rechecked. Your note: ${payload.note}`
              : "Agent resumed. Desktop state will be rechecked before continuing.",
            timestamp: new Date(),
          }]);
          break;
        case "run_completed":
          setWaitingForUser(false);
          // The plan is live working state, not conversation history. Clear it
          // with the run so the completed plan cannot look like active work.
          setTaskProgress(null);
          setCurrentTurn(0);
          setPauseState("running");
          setJevRuntimeState((current) => current === "evaluating" ? "ready" : current);
          setLiveActivity({ phase: "idle", label: "Ready for a task", startedAt: Date.now() });
          if (Array.isArray(payload.deliverables) && payload.deliverables.length > 0) {
            const deliverables = payload.deliverables;
            setMessages((previous) => [...previous, {
              id: `deliverables-${Date.now()}`,
              sender: "system",
              type: "activity",
              text: `Saved outputs:\n${deliverables.slice(0, 8).map((deliverable: unknown) => {
                const item = asRecord(deliverable);
                return `• ${String(item.path ?? "Unknown path")}`;
              }).join("\n")}`,
              timestamp: new Date(),
            }]);
          }
          if (payload.completion_status === "completed_with_warnings" && Array.isArray(payload.warnings) && payload.warnings.length > 0) {
            const completionWarnings = payload.warnings;
            setMessages((previous) => [...previous, {
              id: `completion-warning-${Date.now()}`,
              sender: "system",
              type: "activity",
              text: `Completed with notes:\n${completionWarnings.slice(0, 6).map((warning: unknown) => {
                const item = asRecord(warning);
                return `• ${String(item.message ?? item.code ?? "Review recommended")}`;
              }).join("\n")}`,
              timestamp: new Date(),
            }]);
          }
          window.setTimeout(() => {
            refreshMemoryLibrary();
            refreshGeneratedTools();
          }, 500);
          break;
        case "run_failed":
          setWaitingForUser(false);
          setPauseState("running");
          setJevRuntimeState((current) => current === "evaluating" ? "ready" : current);
          setLiveActivity({ phase: "failed", label: "Task failed", detail: compactValue(payload.error, 160), startedAt: Date.now() });
          break;
      }
    });
    const unlistenQuestion = listen<QuestionRequest>("question_requested", ({ payload }) => {
      setQuestionRequest(payload);
      setQuestionAnswers(Object.fromEntries(payload.questions.map((question) => [question.id, { selected: [], custom: "" }])));
      setLiveActivity({ phase: "waiting", label: "Waiting for your answer", detail: "The agent needs a decision to continue", startedAt: Date.now() });
    });
    
    return () => {
      void unlisten.then((fn) => fn());
      void unlistenStop.then((fn) => fn());
      void unlistenModelRuntime.then((fn) => fn());
      void unlistenLayaRuntime.then((fn) => fn());
      void unlistenJudgeRuntime.then((fn) => fn());
      void unlistenAgent.then((fn) => fn());
      void unlistenQuestion.then((fn) => fn());
    };
  }, []);

  async function run() {
    if (busy) {
      if (pauseState !== "running") return;
      if (!prompt.trim()) return;
      const message = prompt;
      setPrompt("");
      setMessages((prev) => [
        ...prev,
        {
          id: 'user-guidance-' + Date.now(),
          sender: "user",
          type: "guidance",
          text: message,
          timestamp: new Date()
        }
      ]);
      await invoke("send_guidance", { message });
      return;
    }

    if (!prompt.trim() || !model || modelTransitioning) return;
    setBusy(true);
    emergencyStopRequested.current = false;
    const currentPrompt = prompt;
    setCurrentTurn(0);
    setLiveActivity({ phase: "starting", label: "Starting the task", startedAt: Date.now() });
    setTaskProgress({ rootRequest: currentPrompt, status: "in_progress", currentStep: "Starting task", steps: [] });
    setPrompt("");
    setMessages((prev) => [
      ...prev,
      {
        id: 'user-prompt-' + Date.now(),
        sender: "user",
        type: "prompt",
        text: currentPrompt,
        timestamp: new Date()
      }
    ]);
    
    try {
      saveModelRequestSettings();
      await invoke<string>("run_prompt", {
        prompt: currentPrompt, workspace, model, provider: selectedProvider, policyMode,
        temperature: requestTemperature.trim() && (manualCapabilityOverride || modelCapabilities?.supported_parameters.includes("temperature")) ? Number(requestTemperature) : null,
        maxOutputTokens: requestMaxOutput.trim() && (manualCapabilityOverride || modelCapabilities?.supported_parameters.some((item) => ["max_tokens", "max_output_tokens"].includes(item))) ? Number(requestMaxOutput) : null,
        seed: requestSeed.trim() && (manualCapabilityOverride || modelCapabilities?.supported_parameters.includes("seed")) ? Number(requestSeed) : null,
        reasoningEffort: reasoningEffort !== "auto" && (manualCapabilityOverride || modelCapabilities?.reasoning_efforts.includes(reasoningEffort)) ? reasoningEffort : null,
        imageInputOverride: visionMode === "off" ? false : visionMode === "on" && (manualCapabilityOverride || modelCapabilities?.vision === true) ? true : null,
        fullContext,
        decisionRouterEnabled: jevEnabled,
        decisionRouterBackend,
      });
    }
    catch (error) {
      const errorText = String(error);
      if (!emergencyStopRequested.current && !errorText.toLowerCase().includes("cancel")) {
        setLiveActivity({ phase: "failed", label: "Task failed", detail: compactValue(String(error), 160), startedAt: Date.now() });
        setMessages((prev) => [
          ...prev,
          {
            id: 'agent-error-' + Date.now(),
            sender: "system",
            type: "error",
            text: "Error: " + String(error),
            timestamp: new Date()
          }
        ]);
      }
    }
    finally {
      setBusy(false);
      emergencyStopRequested.current = false;
      await refreshConversations();
    }
  }

  async function togglePause() {
    if (!busy || pauseState === "requested") return;
    if (pauseState === "running") {
      setPauseState("requested");
      setLiveActivity({ phase: "waiting", label: "Pausing at a safe boundary", startedAt: Date.now() });
      try {
        await invoke("pause_agent");
      } catch (error) {
        setPauseState("running");
        setLiveActivity({ phase: "failed", label: "Could not pause agent", detail: compactValue(String(error), 140), startedAt: Date.now() });
      }
      return;
    }

    const note = prompt.trim() || undefined;
    try {
      await invoke("resume_agent", { note });
      if (note) {
        setMessages((previous) => [...previous, {
          id: `resume-note-${Date.now()}`,
          sender: "user",
          type: "guidance",
          text: note,
          timestamp: new Date(),
        }]);
      }
      setPrompt("");
      setPauseState("running");
      setLiveActivity({ phase: "starting", label: "Resuming the task", startedAt: Date.now() });
    } catch (error) {
      setLiveActivity({ phase: "failed", label: "Could not resume agent", detail: compactValue(String(error), 140), startedAt: Date.now() });
    }
  }

  async function resolve(allow: boolean, rememberAlways: boolean = false) {
    if (!approval) return;
    const current = approval;
    const choice = allow ? (rememberAlways ? "session" : "once") : "deny";
    setApprovalChoice(choice);
    setApprovalError("");
    try {
      await invoke("resolve_approval", { id: current.id, allow, rememberAlways });
      if (allow && rememberAlways) {
        setSessionApprovals((items) => items.includes(current.tool) ? items : [...items, current.tool]);
        setMessages((items) => [...items, {
          id: `approval-${Date.now()}`,
          sender: "system",
          type: "activity",
          text: `✓ ${current.tool} is allowed for this conversation session.`,
          timestamp: new Date(),
        }]);
      }
      // Resolving one approval can synchronously cause the backend to emit the
      // next approval. Do not let this older handler erase that newer request.
      if (activeApproval.current?.id === current.id) {
        activeApproval.current = null;
        setApproval((pending) => pending?.id === current.id ? null : pending);
        setApprovalChoice(null);
        setLiveActivity({
          phase: allow ? "starting" : "failed",
          label: allow ? "Starting the task" : "Action denied",
          detail: allow ? undefined : "Action was denied by user",
          startedAt: Date.now(),
        });
      }
    } catch (error) {
      if (activeApproval.current?.id === current.id) {
        setApprovalError(String(error));
        setApprovalChoice(null);
      }
    }
  }

  function selectQuestionOption(question: UserQuestion, label: string) {
    setQuestionAnswers((current) => {
      const answer = current[question.id] ?? { selected: [], custom: "" };
      const selected = question.multi_select
        ? (answer.selected.includes(label) ? answer.selected.filter((item) => item !== label) : [...answer.selected, label])
        : [label];
      return { ...current, [question.id]: { ...answer, selected, custom: "" } };
    });
  }

  async function submitQuestions() {
    if (!questionRequest) return;
    const answers = questionRequest.questions.map((question) => ({
      id: question.id,
      selected: questionAnswers[question.id]?.selected ?? [],
      custom: questionAnswers[question.id]?.custom.trim() || null,
    }));
    if (answers.some((answer) => answer.selected.length === 0 && !answer.custom)) return;
    await invoke("resolve_question", { id: questionRequest.id, answers });
    setQuestionRequest(null);
    setLiveActivity({ phase: "thinking", label: "Continuing with your answer", startedAt: Date.now() });
  }

  async function dismissQuestions() {
    if (!questionRequest) return;
    await invoke("dismiss_question", { id: questionRequest.id });
    setQuestionRequest(null);
    setLiveActivity({ phase: "thinking", label: "Continuing with best judgment", startedAt: Date.now() });
  }

  async function approveDraft(id: string) {
    await invoke("approve_memory", { id });
    refreshMemoryLibrary();
  }

  async function rejectDraft(id: string) {
    await invoke("reject_memory", { id });
    refreshMemoryLibrary();
  }

  async function scanMemoryDuplicates() {
    setMemoryCleanupBusy(true);
    try {
      setMemoryDuplicateGroups(await invoke<MemoryDuplicateGroup[]>("scan_memory_duplicates"));
    } finally {
      setMemoryCleanupBusy(false);
    }
  }

  async function mergeMemoryGroup(group: MemoryDuplicateGroup, canonical: MemoryRecord) {
    if (!window.confirm(`Keep “${canonical.text}” and merge the other ${group.records.length - 1} record(s)? You can undo this merge.`)) return;
    const mergeId = await invoke<string>("merge_memories", {
      canonicalId: canonical.id,
      records: group.records.map((record) => ({ id: record.id, version: record.version })),
    });
    setLastMemoryMerge(mergeId);
    setMemoryDuplicateGroups((groups) => groups.filter((item) => item.id !== group.id));
    refreshMemoryLibrary();
  }

  async function undoMemoryMerge() {
    if (!lastMemoryMerge) return;
    await invoke("undo_memory_merge", { mergeId: lastMemoryMerge });
    setLastMemoryMerge(null);
    refreshMemoryLibrary();
    await scanMemoryDuplicates();
  }

  async function toggleMemory(record: MemoryRecord) {
    await invoke("set_memory_enabled", { id: record.id, enabled: !record.enabled });
    refreshMemoryLibrary();
  }

  async function removeMemory(id: string) {
    await invoke("delete_memory", { id });
    refreshMemoryLibrary();
  }

  async function forgetMemory(record: MemoryRecord) {
    if (!window.confirm(`Forget “${record.text}” and prevent similar inferred memories from being learned again? This stores a normalized copy locally for semantic suppression.`)) return;
    await invoke("forget_memory_and_prevent_relearning", { id: record.id });
    refreshMemoryLibrary();
  }

  async function toggleProcedure(record: ProcedureRecord) {
    await invoke("set_procedure_enabled", { id: record.id, enabled: !record.enabled });
    refreshMemoryLibrary();
  }

  async function removeProcedure(id: string) {
    await invoke("delete_procedure", { id });
    refreshMemoryLibrary();
  }

  async function dismissToolCandidate(id: string) {
    await invoke("dismiss_generated_tool_candidate_command", { id, workspace });
    refreshGeneratedTools();
  }

  async function newConversation() {
    await invoke("new_conversation");
    activeConversationIdRef.current = "";
    setActiveConversationId("");
    followOutput.current = true;
    setShowJump(false);
    setMessages([]);
    setHistoryBeforeSequence(null);
    setHistoryHasMore(false);
    setTaskProgress(null);
    setCurrentTurn(0);
    setLiveActivity({ phase: "idle", label: "Ready for a task", startedAt: Date.now() });
    setSessionApprovals([]);
    setUsage({ currentPrompt: 0, currentCompletion: 0, totalPrompt: 0, totalCompletion: 0 });
    const defaultEnabled = decisionRouter?.enabled === true && decisionRouter.has_api_key;
    setJevEnabled(defaultEnabled);
    setDecisionRouterBackend(defaultEnabled ? (decisionRouter?.backend ?? "jev") : "off");
    setJevRuntimeState(decisionRouter?.backend === "jev" && !decisionRouter.has_api_key ? "unavailable" : defaultEnabled ? "ready" : "off");
    setJevWarning(null);
    await refreshConversations();
  }

  async function refreshConversations() {
    try {
      const recent = await invoke<ConversationSummary[]>("list_conversations");
      setConversations(recent);
    } catch {
      setConversations([]);
    }
  }

  async function resumeConversation(sessionId?: string) {
    if (busy) return;
    try {
      const restored = sessionId
        ? await invoke<ResumeConversationPayload>("resume_conversation", { sessionId })
        : await invoke<ResumeConversationPayload>("resume_latest_conversation");
      setSelectedProvider(restored.summary.provider);
      setModel(restored.summary.model);
      setModelRuntime(null);
      setWorkspace(restored.summary.workspace);
      activeConversationIdRef.current = restored.summary.id;
      setActiveConversationId(restored.summary.id);
      setJevEnabled(restored.decision_router_enabled);
      setDecisionRouterBackend(restored.decision_router_backend ?? (restored.decision_router_enabled ? "jev" : "off"));
      setJevRuntimeState(restored.decision_router_backend === "jev" && !decisionRouter?.has_api_key ? "unavailable" : restored.decision_router_enabled ? "ready" : "off");
      setJevWarning(null);
      followOutput.current = true;
      setShowJump(false);
      if (window.innerWidth < 900) setNavigationOpen(false);
      localStorage.setItem("pok_selected_provider", restored.summary.provider);
      localStorage.setItem("pok_selected_model", restored.summary.model);
      checkApiStatus(restored.summary.provider, restored.summary.model);
      const restoredMessages = restored.messages.map((message) =>
        restoredChatMessage(message, restored.summary.updated_at));
      setMessages(restoredMessages);
      setHistoryBeforeSequence(restored.next_before_sequence ?? null);
      setHistoryHasMore(restored.has_more === true);
      setTaskProgress(null);
      setCurrentTurn(0);
      setSessionApprovals([]);
      setLiveActivity({
        phase: "idle",
        label: "Conversation restored",
        detail: `${restored.summary.provider} · ${restored.summary.model}`,
        startedAt: Date.now(),
      });
    } catch (error) {
      setLiveActivity({ phase: "failed", label: "Could not resume conversation", detail: compactValue(String(error), 160), startedAt: Date.now() });
    }
  }

  async function loadEarlierHistory() {
    const sessionId = activeConversationIdRef.current;
    if (!sessionId || !historyHasMore || historyBeforeSequence == null || historyLoading) return;
    const console = consoleRef.current;
    const previousHeight = console?.scrollHeight ?? 0;
    setHistoryLoading(true);
    try {
      const page = await invoke<ConversationHistoryPage>("load_conversation_history", {
        sessionId,
        beforeSequence: historyBeforeSequence,
        limit: 100,
      });
      const older = page.messages.map((message) => restoredChatMessage(message, new Date().toISOString()));
      setMessages((current) => {
        const existing = new Set(current.map((message) => message.id));
        return [...older.filter((message) => !existing.has(message.id)), ...current];
      });
      setHistoryBeforeSequence(page.next_before_sequence ?? null);
      setHistoryHasMore(page.has_more);
      followOutput.current = false;
      requestAnimationFrame(() => {
        if (console) console.scrollTop += console.scrollHeight - previousHeight;
      });
    } finally {
      setHistoryLoading(false);
    }
  }

  function handleKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing && event.keyCode !== 229) {
      event.preventDefault();
      void run();
    }
  }

  const visibleMessages = messages;

  const install = installStatus();

  return <main className={`app-shell ${navigationOpen ? "navigation-open" : "navigation-closed"}`}>
    {navigationOpen && <><button className="navigation-backdrop" aria-label="Close navigation" onClick={() => setNavigationOpen(false)} /><ConversationSidebar conversations={conversations} activeId={activeConversationId} disabled={busy || modelTransitioning}
      onSelect={id => void resumeConversation(id)} onNew={() => { void newConversation(); if (window.innerWidth < 900) setNavigationOpen(false); }} onSettings={openSettings} onClose={() => setNavigationOpen(false)} /></>}
    <header className="app-header">
      <button className="navigation-toggle" aria-label="Toggle navigation" aria-expanded={navigationOpen} onClick={() => setNavigationOpen(open => !open)}>☰</button>
      <div className="brand">
        <h1>{conversations.find(c => c.id === activeConversationId)?.title || messages.find(m => m.type === "prompt")?.text || "New conversation"}</h1>
        <span className="workspace-label" title={workspace}>{workspace || "Default workspace"}</span>
      </div>
      <div className="header-controls">
        <button
          ref={settingsButtonRef}
          type="button"
          className="settings-toggle"
          aria-controls="settings-panel"
          aria-expanded={settingsOpen}
          onClick={() => setSettingsOpen((open) => !open)}
        >
          <span aria-hidden="true">☰</span> Settings
        </button>
        <ModelStatusBanner
          status={apiStatus}
          model={model}
          provider={selectedProvider}
          runtime={modelRuntime}
          requiresApiKey={status?.requires_api_key}
          hasApiKey={status?.has_api_key}
          onRetry={retryModelConnection}
          onConnectKey={() => setShowCloudModal(true)}
        />
        <span className={`status ${status?.requires_api_key && !status?.has_api_key ? "is-error" : apiStatus === "error" || modelRuntime?.phase === "error" || modelRuntime?.phase === "server_offline" ? "is-error" : modelTransitioning ? "is-loading" : busy ? "is-running" : ""}`}>
          {status?.requires_api_key && !status?.has_api_key ? "🔴 MISSING API KEY" : apiStatus === "error" ? "🔴 NOT CONNECTED" : modelRuntime?.phase === "unloading" ? "↧ UNLOADING MODEL..." : modelRuntime?.phase === "loading" || modelRuntime?.phase === "checking" ? "⚡ LOADING MODEL..." : modelRuntime?.phase === "server_offline" ? "🔴 SERVER OFFLINE" : modelRuntime?.phase === "error" ? "🔴 MODEL ERROR" : busy ? `${policyMode.toUpperCase()} · RUNNING` : modelRuntime?.phase === "loaded" ? "READY · MODEL LOADED" : "READY"}
        </span>
      </div>
    </header>
    <section className={`grid ${settingsOpen ? "settings-open" : "settings-closed"}`}>
      <SettingsDrawer open={settingsOpen} onClose={closeSettings}>
        <h2>Model & connection</h2>
        <label>Provider
          <select disabled={busy || modelTransitioning} value={selectedProvider} onChange={(event) => void handleProviderChange(event.target.value)}>
            {providers.map((p) => <option key={p} value={p}>{p.toUpperCase()}</option>)}
          </select>
        </label>
        <div className="provider-boundary" title={status?.data_boundary === "external_service"
          ? "Full task context is sent only after confirmation for this conversation."
          : "Model requests stay on the configured local provider boundary."}>
          {status?.data_boundary === "external_service" ? "☁ External service · confirmation required" : "⌂ Local device · full-fidelity context"}
        </div>
        <button 
          style={{ marginTop: "6px", fontSize: "11px", padding: "4px 8px" }} 
          onClick={() => setShowCloudModal(true)}
        >
          🔑 Connect Provider API Key
        </button>
        <label style={{ marginTop: "10px" }}>
          <span style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: "8px" }}>
            <span>Model ({models.length} Available)</span>
            {(selectedProvider === "lm_studio" || selectedProvider === "ollama") && (
              <button
                type="button"
                title="Refresh models added or removed from the local server"
                disabled={busy || modelTransitioning || refreshingModels}
                onClick={(event) => {
                  event.preventDefault();
                  void refreshLocalModels();
                }}
                style={{ padding: "2px 8px", fontSize: "10px", width: "auto" }}
              >
                {refreshingModels ? "Refreshing…" : "↻ Refresh"}
              </button>
            )}
          </span>
          <input
            type="text"
            placeholder="🔍 Search / filter models..."
            value={modelSearch}
            onChange={(e) => setModelSearch(e.target.value)}
            style={{ width: "100%", padding: "4px 8px", margin: "4px 0 6px 0", fontSize: "11px", background: "var(--surface)", color: "var(--text)", border: "1px solid var(--border)", borderRadius: "4px" }}
          />
          {modelSearch.trim() ? (
            <div className="model-search-results" role="listbox" aria-label="Filtered models">
              {models.filter((item) => item.toLowerCase().includes(modelSearch.toLowerCase())).slice(0, 100).map((item) => (
                <button
                  type="button"
                  role="option"
                  aria-selected={item === model}
                  className={item === model ? "selected" : ""}
                  key={item}
                  disabled={busy || modelTransitioning}
                  onClick={() => { void handleModelChange(item); setModelSearch(""); }}
                >{item}</button>
              ))}
              {models.every((item) => !item.toLowerCase().includes(modelSearch.toLowerCase())) && <div className="model-search-empty">No matching models</div>}
            </div>
          ) : (
            <select disabled={busy || modelTransitioning} value={model} onChange={(event) => void handleModelChange(event.target.value)}>
              <option value="">-- Select a Model --</option>
              {models.map((item) => <option key={item} value={item}>{item}</option>)}
            </select>
          )}
        </label>
        <h2>Agent & permissions</h2>
        <label style={{ marginTop: "10px" }}>Policy Mode
          <select disabled={busy} value={policyMode} onChange={(event) => setPolicyMode(event.target.value as "interactive" | "autonomous")}>
            <option value="interactive">Interactive (Ask for Commands)</option>
            <option value="autonomous">Autonomous — Broad Access with Guardrails</option>
          </select>
        </label>
        <div className="context-settings" id="jev-settings">
          <div className="context-settings-body">
            <strong>Fast decision router</strong>
            <small>
              JEV is the hosted router; Laya is the local open-source router. Both use the same bounded candidates and safety checks.
            </small>
            <label>Default for new conversations
              <select disabled={busy || installingLaya} value={decisionRouterBackend} onChange={(event) => void handleRouterBackendChange(event.target.value as DecisionRouterBackend)}>
                <option value="off">Off — primary LLM handles every decision</option>
                <option value="jev" disabled={!decisionRouter?.has_api_key}>JEV — hosted TypeSafe router</option>
                <option value="laya" disabled={!decisionRouter?.laya.installed}>Laya — local open-source router</option>
              </select>
            </label>
            <small>{decisionRouter?.has_api_key ? "API key connected — JEV can now be enabled" : "Save a TypeSafe API key before JEV can be enabled"}</small>
            <div style={{ display: "flex", gap: "6px" }}>
              <input
                type="password"
                value={decisionRouterKey}
                onChange={(event) => setDecisionRouterKey(event.target.value)}
                placeholder="TypeSafe API key"
              />
              <button disabled={!decisionRouterKey.trim()} onClick={handleSaveDecisionRouterKey}>Save key</button>
            </div>
            <div style={{ display: "flex", gap: "8px", alignItems: "center", marginTop: "8px" }}>
              <button disabled={installingLaya || decisionRouter?.laya.installed} onClick={() => void handleInstallLaya(false)}>
                {installingLaya ? "Installing Laya…" : decisionRouter?.laya.installed ? "Laya installed" : "Install Laya locally"}
              </button>
              {decisionRouter?.laya.installed && <button disabled={installingLaya || busy} onClick={() => void handleInstallLaya(true)}>Repair/update GPU runtime</button>}
              <label>Laya device
                <select disabled={busy || installingLaya || !decisionRouter?.laya.installed} value={layaDevice} onChange={(event) => void handleLayaDeviceChange(event.target.value as LayaDevicePreference)}>
                  <option value="auto">Auto — GPU when suitable</option>
                  <option value="gpu">GPU — request CUDA</option>
                  <option value="cpu">CPU</option>
                </select>
              </label>
            </div>
            <small role="status">
              {decisionRouter?.laya.running ? `✓ Laya ready on ${decisionRouter.laya.device ?? "local device"}. ${decisionRouter.laya.device_reason ?? ""}` : decisionRouter?.laya.detail}
            </small>
            {install?.label.includes("Laya") && <InstallProgress label={install.label} detail={install.detail} />}
            <hr />
            <strong>Cross-model judge (Laya only)</strong>
            <small>
              When Laya's own confidence gate can't resolve a decision, a second local model (zeiger)
              double-checks that specific pick before falling back to the primary LLM. Testing found
              this beats both Laya and JEV judging themselves, with zero dangerous false positives —
              see docs/decision-router-backends.md.
            </small>
            <div style={{ display: "flex", gap: "8px", alignItems: "center", marginTop: "8px" }}>
              <button disabled={installingJudge || decisionRouter?.judge.installed} onClick={() => void handleInstallJudge(false)}>
                {installingJudge ? "Installing judge model…" : decisionRouter?.judge.installed ? "Judge model installed" : "Install judge model locally"}
              </button>
              {decisionRouter?.judge.installed && <button disabled={installingJudge || busy} onClick={() => void handleInstallJudge(true)}>Repair/update</button>}
              <label style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                <input
                  type="checkbox"
                  disabled={busy || installingJudge || togglingJudge || !decisionRouter?.judge.installed || decisionRouterBackend !== "laya"}
                  checked={decisionRouter?.judge.enabled ?? false}
                  onChange={(event) => void handleJudgeEnabledChange(event.target.checked)}
                />
                Enable cross-model judge
              </label>
            </div>
            {decisionRouterBackend !== "laya" && (
              <small>Select Laya as the active router above to use the cross-model judge.</small>
            )}
            <small role="status">
              {decisionRouter?.judge.running ? `✓ Judge ready on ${decisionRouter.judge.device ?? "local device"}. ${decisionRouter.judge.device_reason ?? ""}` : decisionRouter?.judge.detail}
            </small>
            {install?.label.includes("judge") && <InstallProgress label={install.label} detail={install.detail} />}
            {decisionRouterError && <p className="error-text">{decisionRouterError}</p>}
          </div>
        </div>
        {sessionApprovals.length > 0 && (
          <div className="session-approvals" role="status">
            <strong>✓ Allowed this session</strong>
            <span>{sessionApprovals.join(", ")}</span>
          </div>
        )}
        <details className="advanced-settings"><summary>Advanced settings</summary>
        <div className="context-settings">
          <button className="context-settings-toggle" onClick={() => setContextExpanded(!contextExpanded)}>
            Context budget {contextExpanded ? "▲" : "▼"}
          </button>
          {contextExpanded && (
            <div className="context-settings-body">
              <label>Compact at {contextThreshold}%
                <input type="range" min="50" max="95" value={contextThreshold} onChange={(event) => setContextThreshold(Number(event.target.value))} />
              </label>
              <label>Loaded context tokens (optional)
                <input value={contextWindowOverride} onChange={(event) => setContextWindowOverride(event.target.value.replace(/\D/g, ""))} placeholder="Use provider/catalog metadata" />
              </label>
              {(selectedProvider === "lm_studio") && <small>When set, POK-Agent requests this context length when loading the model.</small>}
              <button disabled={!model} onClick={saveContextOverride}>Save for this model</button>
            </div>
          )}
        </div>
        <div className="context-settings">
          <div className="context-settings-body">
            <strong>Model capabilities</strong>
            <small>
              Vision {modelCapabilities?.vision === undefined ? "unknown" : modelCapabilities.vision ? "available" : "unavailable"}
              {" · "}tools {modelCapabilities?.tool_use === undefined ? "unknown" : modelCapabilities.tool_use ? "available" : "unavailable"}
              {" · "}reasoning {modelCapabilities?.reasoning === undefined ? "unknown" : modelCapabilities.reasoning ? "available" : "unavailable"}
            </small>
            <label>Vision input
              <select value={visionMode} onChange={(event) => setVisionMode(event.target.value as "auto" | "on" | "off")}>
                <option value="auto">Auto (provider metadata)</option>
                <option value="off">Off</option>
                <option value="on" disabled={modelCapabilities?.vision !== true && !manualCapabilityOverride}>On</option>
              </select>
            </label>
            <label>Reasoning effort
              <select value={reasoningEffort} onChange={(event) => setReasoningEffort(event.target.value)}>
                <option value="auto">Auto / provider default{modelCapabilities?.reasoning_default ? ` (${modelCapabilities.reasoning_default})` : ""}</option>
                {(modelCapabilities?.reasoning_efforts ?? []).map((effort) => <option key={effort} value={effort}>{effort}</option>)}
                {manualCapabilityOverride && ["none", "minimal", "low", "medium", "high", "xhigh"].filter((effort) => !(modelCapabilities?.reasoning_efforts ?? []).includes(effort)).map((effort) => <option key={effort} value={effort}>{effort} (override)</option>)}
              </select>
            </label>
            <label>Temperature (blank = auto)
              <input disabled={!manualCapabilityOverride && !modelCapabilities?.supported_parameters.includes("temperature")} type="number" min="0" max="2" step="0.05" value={requestTemperature} onChange={(event) => setRequestTemperature(event.target.value)} />
            </label>
            <label>Maximum output tokens (blank = auto)
              <input disabled={!manualCapabilityOverride && !modelCapabilities?.supported_parameters.some((item) => ["max_tokens", "max_output_tokens"].includes(item))} inputMode="numeric" value={requestMaxOutput} onChange={(event) => setRequestMaxOutput(event.target.value.replace(/\D/g, ""))} />
            </label>
            <label>Seed (blank = provider default)
              <input disabled={!manualCapabilityOverride && !modelCapabilities?.supported_parameters.includes("seed")} inputMode="numeric" value={requestSeed} onChange={(event) => setRequestSeed(event.target.value.replace(/\D/g, ""))} />
            </label>
            <label><input type="checkbox" checked={manualCapabilityOverride} onChange={(event) => setManualCapabilityOverride(event.target.checked)} /> Advanced manual override for unknown or misreported capabilities</label>
            <label><input type="checkbox" checked={fullContext} onChange={(event) => setFullContext(event.target.checked)} /> Full Context (send retained history up to the compaction threshold)</label>
            <button disabled={!model} onClick={saveModelRequestSettings}>Save request settings</button>
          </div>
        </div>
        <label style={{ marginTop: "10px" }}>Endpoint URL ({selectedProvider.toUpperCase()})
          <div style={{ display: "flex", gap: "6px", alignItems: "center", marginTop: "4px" }}>
            <input
              type="text"
              value={customEndpoint}
              onChange={(e) => setCustomEndpoint(e.target.value)}
              placeholder="e.g. http://127.0.0.1:1234/v1"
              onKeyDown={(e) => { if (e.key === "Enter") handleSaveEndpoint(); }}
              style={{ width: "100%", padding: "6px 8px", fontSize: "11px", background: "var(--surface)", color: "var(--text)", border: "1px solid var(--border)", borderRadius: "6px" }}
            />
            <button
              type="button"
              style={{ padding: "6px 10px", fontSize: "11px", flex: "none" }}
              onClick={handleSaveEndpoint}
            >
              Save
            </button>
          </div>
        </label>
        <dl style={{ marginTop: "8px" }}><dt>Platform</dt><dd>{status?.platform ?? "…"}</dd></dl>
        </details>
        <h2>Workspace</h2><input disabled={busy} value={workspace} onChange={(event) => setWorkspace(event.target.value)} placeholder="Leave blank for current directory" />
        <div className="memory-library-heading">
          <h2 id="memory-library">Memory library</h2>
          <div className="memory-heading-actions">
            {lastMemoryMerge && <button onClick={() => void undoMemoryMerge()}>Undo merge</button>}
            <button disabled={memoryCleanupBusy} onClick={() => void scanMemoryDuplicates()}>{memoryCleanupBusy ? "Scanning…" : "Scan duplicates"}</button>
            <button onClick={refreshMemoryLibrary}>Refresh</button>
          </div>
        </div>
        <div className="memory-tabs" role="tablist">
          <button className={memoryTab === "facts" ? "active" : ""} onClick={() => setMemoryTab("facts")}>Facts</button>
          <button className={memoryTab === "skills" ? "active" : ""} onClick={() => setMemoryTab("skills")}>Skills</button>
        </div>
        <div className="memory-list-container">
          {memoryTab === "facts" && memoryDuplicateGroups.map((group) => (
            <div className="memory duplicate-group" key={group.id}>
              <strong>Possible duplicates · {Math.round(group.similarity * 100)}% local similarity</strong>
              <small>Select the record to keep. Nothing changes until you confirm.</small>
              {group.records.map((record) => (
                <div className="duplicate-choice" key={record.id}>
                  <span>{record.text}</span>
                  <button onClick={() => void mergeMemoryGroup(group, record)}>Keep this one</button>
                </div>
              ))}
              <button onClick={() => setMemoryDuplicateGroups((groups) => groups.filter((item) => item.id !== group.id))}>Not duplicates</button>
            </div>
          ))}
          {memoryTab === "facts" && (memoryLibrary.facts.length === 0 ? <p className="empty-memory-msg">No saved facts or drafts.</p> : memoryLibrary.facts.map((record) => (
            <div className={`memory ${record.enabled ? "" : "disabled"}`} key={record.id}>
              <small>{record.approved ? record.source : `DRAFT · ${record.source}`} · reinforced {record.reinforcement_count ?? 1}×{record.review_status !== "independent" ? ` · ${record.review_status.replaceAll("_", " ")}` : ""}</small>
              <p>{record.text}</p>
              <div className="memory-actions">
                {!record.approved ? <>
                  <button className="primary" onClick={() => approveDraft(record.id)}>Approve</button>
                  <button className="reject" onClick={() => rejectDraft(record.id)}>Deny</button>
                </> : <>
                  <button onClick={() => toggleMemory(record)}>{record.enabled ? "Disable" : "Enable"}</button>
                  <button className="reject" onClick={() => removeMemory(record.id)}>Delete</button>
                  <button className="reject" title="Stores normalized text locally to block similar inferred memories" onClick={() => void forgetMemory(record)}>Forget & block similar</button>
                </>}
              </div>
            </div>
          )))}
          {memoryTab === "skills" && (memoryLibrary.skills.length === 0 ? <p className="empty-memory-msg">No verified skills learned yet.</p> : memoryLibrary.skills.map((record) => (
            <div className={`memory procedure ${record.enabled ? "" : "disabled"}`} key={record.id}>
              <small>{record.success_count === 0 && record.evidence === "explicit_user_request" ? "USER-CREATED SKILL · unverified" : `SKILL · ${record.success_count} verified successes`} · {record.retrieval_count} activations</small>
              <strong>{record.title}</strong>
              <p>{record.command_template ?? record.summary}</p>
              <span className="memory-evidence">{record.evidence}{record.applications.length ? ` · ${record.applications.join(", ")}` : ""}</span>
              <div className="memory-actions">
                <button onClick={() => toggleProcedure(record)}>{record.enabled ? "Disable" : "Enable"}</button>
                <button className="reject" onClick={() => removeProcedure(record.id)}>Delete</button>
              </div>
            </div>
          )))}
        </div>
        <h2 id="generated-tools">Generated tools</h2>
        <button onClick={refreshGeneratedTools}>Refresh</button>
        <div className="generated-tools-container">
          {generatedToolCandidates.map((candidate) => (
            <div className="generated-tool candidate" key={candidate.id}>
              <div><strong>Promotion candidate</strong><span>{candidate.runtime}</span></div>
              <p>{candidate.task}</p>
              <small>{candidate.helper_path} · tested successfully, not installed</small>
              <button className="reject" onClick={() => dismissToolCandidate(candidate.id)}>Dismiss</button>
            </div>
          ))}
          {generatedTools.length === 0 ? <p className="empty-memory-msg">No project tools promoted yet.</p> : generatedTools.map((tool) => (
            <div className="generated-tool" key={tool.name}>
              <div><strong>{tool.name}</strong><span>v{tool.version}</span></div>
              <p>{tool.description}</p>
              <small>
                {tool.runtime} · {tool.isolated_environment ? `isolated · ${tool.smoke_tests ?? 0} tests / ${tool.assertions ?? 0} checks` : "legacy runtime"} · {tool.capabilities.join(", ")} · {tool.successes} ok / {tool.failures} failed
              </small>
              <div className="generated-tool-actions">
                <button disabled={busy} onClick={() => toggleGeneratedTool(tool)}>{tool.enabled ? "Disable" : "Enable"}</button>
                <button disabled={busy} className="reject" onClick={() => removeGeneratedTool(tool)}>Delete</button>
              </div>
            </div>
          ))}
        </div>
      </SettingsDrawer>
      <section className="panel agent">
        <details className="session-details"><summary>Session details <span>{model || "No model selected"} · {usage.totalCompletion.toLocaleString()} output tokens</span></summary>
        {jevEnabled && <div className="jev-scorecard" role="status" aria-label="Decision router performance diagnostics">
          <div><strong>{decisionRouterBackend.toUpperCase()} PERFORMANCE</strong><span>{jevMetrics.actions} actions · {jevMetrics.evidence} evidence · {jevMetrics.progress} verified progress</span></div>
          <div className="jev-metrics">
            <span>{jevMetrics.evaluations} decisions</span>
            <span>{jevMetrics.eligible} eligible</span>
            <span>{jevMetrics.judgeAttempts} judged</span>
            <span>{jevMetrics.judgePromotions} approved</span>
            <span>{jevMetrics.cached} cached</span>
            <span>{jevMetrics.failures} failed</span>
            <span>{jevMetrics.latencyMs.toLocaleString()} ms overhead</span>
          </div>
          {jevMetrics.last && <small>{jevMetrics.last}</small>}
        </div>}
        <div className={`context-meter ${contextStatus && !contextStatus.budget.known ? "unknown" : ""}`} onClick={() => setContextExpanded(!contextExpanded)}>
          <div className="context-meter-heading">
            <span>Context {contextStatus ? `${contextStatus.conversation_tokens.toLocaleString()} retained · ${contextStatus.prompt_tokens.toLocaleString()} sent` : "—"}</span>
            <span>{contextStatus ? `${contextStatus.remaining_tokens.toLocaleString()} tokens until compact` : "Select a model"}</span>
          </div>
          <div className="context-meter-track"><div className="context-meter-fill" style={{ width: `${contextStatus ? Math.min(100, (contextStatus.prompt_tokens / Math.max(1, contextStatus.budget.compact_at_tokens)) * 100) : 0}%` }} /></div>
          {contextStatus && <div className="context-meter-detail">
            {contextStatus.budget.context_window_tokens.toLocaleString()} token window{contextStatus.working_set_target_tokens ? ` · request working set ${contextStatus.working_set_target_tokens.toLocaleString()}` : ""} · user compact threshold {contextStatus.budget.threshold_percent}% ({contextStatus.budget.compact_at_tokens.toLocaleString()}) · {contextStatus.budget.source.replaceAll("_", " ")}
            {contextStatus.approximate_turns_remaining ? ` · ~${contextStatus.approximate_turns_remaining} turns` : ""} · archive {contextStatus.archived_entries.toLocaleString()} entries / {contextStatus.archived_tokens.toLocaleString()} tokens · {contextStatus.compactions} semantic compactions
            {!contextStatus.budget.known ? " · context size unknown; legacy fallback active" : ""}
          </div>}
          {contextStatus && <button className="compact-now" disabled={busy} onClick={(event) => { event.stopPropagation(); invoke("compact_context").catch(() => undefined); }}>Compact next turn</button>}
        </div>
        <div className="token-line">Current request: {usage.currentPrompt.toLocaleString()} prompt · {usage.currentCompletion.toLocaleString()} completion</div>
        <div className="token-line">Session cumulative: {usage.totalPrompt.toLocaleString()} prompt · {usage.totalCompletion.toLocaleString()} completion</div>
        </details>
        
        <div className="console-container" ref={consoleRef} onScroll={handleConsoleScroll} onWheel={handleConsoleWheel} onPointerDown={handleConsolePointerDown}>
          <div className="console-feed">
            {historyHasMore && <button className="history-loader" disabled={historyLoading} onClick={() => void loadEarlierHistory()}>{historyLoading ? "Loading earlier history…" : "Load earlier history"}</button>}
            {messages.some(m => m.id.startsWith("restored-")) && <p className="session-notice">Conversation restored in chronological order{historyHasMore ? " · Scroll up to load earlier history" : ""}.</p>}
            {messages.length === 0 && !busy ? (
              <div className="console-empty">
                <span className="welcome-mark">P</span>
                <h2>What would you like to do?</h2>
                <p>Build something, explore a project, or work across your desktop.</p>
                <small>You’ll see each step as your agent works.</small>
              </div>
            ) : (
              <>
              {historyLoading && <div className="console-line line-omitted">Loading earlier history…</div>}
              {visibleMessages.map(msg => <ConversationMessage key={msg.id} message={msg} />)}
              </>
            )}

            {busy && waitingForUser && (
              <div className="waiting-for-user" role="status">
                <strong>Waiting for you</strong>
                <span>You're using the mouse or keyboard. The agent will continue on its own once you stop.</span>
              </div>
            )}
            {busy && pauseState !== "paused" && (
              <div className="console-status-line">
                <span className="active-ring" aria-hidden="true"></span>
                <span className="status-text">{liveActivity.label}</span>
              </div>
            )}
            
          </div>
        </div>

        {showJump && <button className="jump-latest" onClick={() => { followOutput.current = true; setShowJump(false); consoleRef.current?.scrollTo({ top: consoleRef.current.scrollHeight }); }}>↓ Jump to latest</button>}
        {install && <InstallProgress label={install.label} detail={install.detail} />}
        {busy && taskProgress && <details className="plan-disclosure"><summary>{taskProgress.currentStep || "Task plan"} · {taskProgress.steps.filter(s => s.status === "completed").length}/{taskProgress.steps.length}</summary><TaskProgressPanel task={taskProgress} /></details>}
        {(busy || liveActivity.phase === "failed") && <LiveActivityPanel activity={liveActivity} busy={busy} turn={currentTurn} tokens={usage.currentCompletion} />}
        <Composer
          prompt={prompt}
          onChange={setPrompt}
          onKeyDown={handleKeyDown}
          placeholder={pauseState === "paused"
            ? "Optional: tell the agent what you changed before resuming..."
            : busy ? "Type correction or guidance and press Enter..." : "Describe a desktop or coding task..."}
        >
        
        <div className="actions">
          <ModelPicker models={models} value={model} onChange={(next) => void handleModelChange(next)} disabled={busy || modelTransitioning} />
          <select aria-label="Permission mode" className="composer-policy" disabled={busy} value={policyMode} onChange={e => setPolicyMode(e.target.value as "interactive" | "autonomous")}><option value="interactive">Ask permission</option><option value="autonomous">Autonomous</option></select>
          <select aria-label="Decision router" className={`jev-toggle jev-${jevRuntimeState} ${jevEnabled ? "active" : ""}`} disabled={busy} value={decisionRouterBackend} onChange={(event) => void handleRouterBackendChange(event.target.value as DecisionRouterBackend)}>
            <option value="off">Router Off</option>
            <option value="jev" disabled={!decisionRouter?.has_api_key}>JEV</option>
            <option value="laya" disabled={!decisionRouter?.laya.installed}>Laya{decisionRouter?.laya.phase === "loading" ? " · Loading…" : decisionRouter?.laya.running ? ` · ${decisionRouter.laya.device ?? "Ready"}` : ""}</option>
          </select>
          {jevEnabled && jevWarning && (
            <span className="jev-warning" role="status" title="No suitable bounded candidate was selected. The main model continued normally and JEV remains enabled.">
              ⚠ {jevWarning}
            </span>
          )}
          <button className="primary" disabled={!prompt.trim() || !model || pauseState !== "running" || (!busy && modelTransitioning)} onClick={run}>
            {busy ? "Send Guidance" : "Run task"}
          </button>
          <button
            className={`pause-toggle pause-${pauseState}`}
            disabled={!busy || pauseState === "requested"}
            onClick={togglePause}
          >
            {pauseState === "paused" ? "Resume agent" : pauseState === "requested" ? "Pausing…" : "Pause agent"}
          </button>
          <button className="stop" onClick={() => invoke("emergency_stop")}>Emergency stop · Ctrl+Alt+Esc</button>
        </div>
        </Composer>
      </section>
    </section>
    {showCloudModal && (
      <div className="modal">
        <div className="dialog">
          <span className="eyebrow">CLOUD PROVIDER SETUP</span>
          <h2>Connect {selectedProvider.toUpperCase()} API Key</h2>
          
          {PROVIDER_URLS[selectedProvider.toLowerCase()] && (
            <div style={{ margin: "12px 0", padding: "12px 14px", background: "var(--raised)", borderRadius: "6px", border: "1px solid #3b82f6" }}>
              <p style={{ margin: "0 0 6px 0", fontSize: "12px", color: "var(--text)" }}>
                Create or copy an API key from the provider console.
              </p>
              <a 
                href={PROVIDER_URLS[selectedProvider.toLowerCase()].url} 
                target="_blank" 
                rel="noreferrer"
                style={{
                  display: "inline-block",
                  background: "var(--surface)",
                  color: "var(--text)",
                  padding: "6px 14px",
                  borderRadius: "4px",
                  textDecoration: "none",
                  fontSize: "12px",
                  fontWeight: 600
                }}
              >
                🌐 Open {PROVIDER_URLS[selectedProvider.toLowerCase()].name} Page ↗
              </a>
            </div>
          )}

          <p style={{ fontSize: "12px", color: "var(--muted)", marginTop: "12px" }}>Paste your API key for {selectedProvider}:</p>
          <input 
            type="password" 
            value={cloudApiKey} 
            onChange={(e) => {
              setCloudApiKey(e.target.value);
              setCloudKeyError("");
            }}
            placeholder={`Enter ${selectedProvider.toUpperCase()} API key...`}
            style={{ width: "100%", padding: "8px", margin: "8px 0 16px 0", background: "var(--surface)", color: "var(--text)", border: "1px solid var(--border)", borderRadius: "4px" }}
          />
          {cloudKeyError && <p className="error-text">Could not connect: {cloudKeyError}</p>}
          <div className="actions">
            <button className="primary" onClick={handleSaveCloudKey} disabled={savingCloudKey}>
              {savingCloudKey ? "Connecting..." : "Save & Connect"}
            </button>
            <button onClick={() => setShowCloudModal(false)} disabled={savingCloudKey}>Cancel</button>
          </div>
        </div>
      </div>
    )}
    {questionRequest && (
      <div className="modal">
        <div className="dialog question-dialog">
          <span className="eyebrow">AGENT QUESTION</span>
          <h2>Choose how to continue</h2>
          {questionRequest.questions.map((question) => {
            const answer = questionAnswers[question.id] ?? { selected: [], custom: "" };
            return <section className="question-card" key={question.id}>
              {question.header && <span className="question-header">{question.header}</span>}
              <p>{question.question}</p>
              <div className="question-options">
                {question.options.map((option) => <button key={option.label}
                  className={`question-option ${answer.selected.includes(option.label) ? "selected" : ""}`}
                  onClick={() => selectQuestionOption(question, option.label)}>
                  <strong>{option.label}</strong>{option.description && <small>{option.description}</small>}
                </button>)}
              </div>
              <input value={answer.custom} placeholder="Other answer…"
                onChange={(event) => setQuestionAnswers((current) => ({ ...current, [question.id]: { selected: [], custom: event.target.value } }))} />
            </section>;
          })}
          <div className="actions">
            <button className="primary" onClick={() => void submitQuestions()}>Continue</button>
            <button onClick={() => void dismissQuestions()}>Let the agent decide</button>
          </div>
        </div>
      </div>
    )}
    {approval && (
      <div className="modal">
        <div className="dialog">
          <span className="eyebrow">APPROVAL REQUIRED</span>
          <h2>{approval.tool}</h2>
          <p>{approval.reason}</p>
          <pre>{JSON.stringify(approval.arguments, null, 2)}</pre>
          <p className="approval-help">Choose whether to permit only this action or every request for this tool until you start a new conversation.</p>
          {approvalError && <p className="approval-error">Could not apply your choice: {approvalError}</p>}
          <div className="actions approval-actions">
            <button className={`approval-choice once ${approvalChoice === "once" ? "selected" : ""}`} disabled={approvalChoice !== null} onClick={() => resolve(true, false)}>{approvalChoice === "once" ? "Allowing once…" : "Allow once"}</button>
            <button className={`approval-choice session ${approvalChoice === "session" ? "selected" : ""}`} disabled={approvalChoice !== null} onClick={() => resolve(true, true)}>{approvalChoice === "session" ? "Activating session permission…" : "Always allow this session"}</button>
            <button className={`approval-choice deny ${approvalChoice === "deny" ? "selected" : ""}`} disabled={approvalChoice !== null} onClick={() => resolve(false, false)}>{approvalChoice === "deny" ? "Denying…" : "Deny"}</button>
          </div>
        </div>
      </div>
    )}
  </main>;
}
