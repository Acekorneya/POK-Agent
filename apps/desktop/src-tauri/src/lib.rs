use std::{
    collections::{BTreeMap, HashMap, HashSet},
    path::{Path, PathBuf},
    process::Stdio,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use async_trait::async_trait;
use parking_lot::Mutex as SyncMutex;
use pok_ai_core::{
    Result,
    brain::{
        LocalModelPhase, LocalModelRuntimeStatus, ModelInfo, from_provider,
        load_local_model_from_status_with_context, load_local_model_with_context,
        local_model_status, unload_local_model, unload_local_model_from_status,
    },
    builtins::{register_desktop_tools, register_memory_tools},
    coding::register_coding_tools,
    config::{Config, DecisionRouterBackend, LayaDevicePreference, ProviderDataBoundary},
    context::{ContextSettings, ModelContextOverride, resolve_context_budget},
    conversation_store::{
        ConversationDisplayMessage, ConversationHistoryPage, ConversationStore, ConversationSummary,
    },
    generated_tools::{
        GeneratedToolCandidate, GeneratedToolSummary, delete_generated_tool,
        dismiss_generated_tool_candidate, list_generated_tool_candidates,
        list_generated_tools as list_project_generated_tools, register_generated_tool_tools,
        set_generated_tool_enabled as set_project_generated_tool_enabled,
    },
    memory::{
        MemoryDuplicateGroup, MemoryRecord, MemoryStore, ProcedureKind, ProcedureRecord,
        atomic_write,
    },
    pause::PauseController,
    policy::{ApprovalDecision, ApprovalHandler, Policy, PolicyMode},
    session::{AgentEvent, Session, SessionObserver},
    subagent::SpawnSubagentTool,
    tool::{
        ToolContext, ToolRegistry, UserQuestion, UserQuestionAnswer, UserQuestionHandler,
        UserQuestionOutcome,
    },
};
use process_wrap::tokio::{ChildWrapper, CommandWrap, KillOnDrop};
#[cfg(windows)]
use process_wrap::tokio::{CreationFlags, JobObject};
use serde::{Deserialize, Serialize};
use tauri::{Emitter, Manager, State};
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutState};
use tokio::{
    process::Command,
    sync::{Mutex, mpsc, oneshot},
};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;
#[cfg(windows)]
use windows::Win32::System::Threading::PROCESS_CREATION_FLAGS;

mod voice;

#[derive(Clone)]
struct AppRuntime {
    config: Arc<SyncMutex<Config>>,
    cancellation: Arc<SyncMutex<CancellationToken>>,
    pause: Arc<PauseController>,
    conversation: Arc<Mutex<Option<Session>>>,
    conversation_provider: Arc<Mutex<Option<String>>>,
    conversation_decision_router_backend: Arc<Mutex<Option<DecisionRouterBackend>>>,
    conversation_judge_enabled: Arc<Mutex<Option<bool>>>,
    laya_process: Arc<Mutex<Option<LayaProcess>>>,
    judge_process: Arc<Mutex<Option<JudgeProcess>>>,
    shutdown_started: Arc<AtomicBool>,
    model_transition: Arc<Mutex<()>>,
    local_api_query: Arc<Mutex<()>>,
    model_runtime_cache: Arc<Mutex<HashMap<(String, String), LocalModelRuntimeStatus>>>,
    model_catalog_cache: Arc<Mutex<HashMap<String, Vec<ModelInfo>>>>,
    user_guidance_queue: Arc<SyncMutex<std::collections::VecDeque<String>>>,
    external_provider_acknowledgments: Arc<Mutex<HashSet<String>>>,
    command_manager: Arc<SyncMutex<Option<pok_ai_core::commands::CommandManager>>>,
    pending_resume: Arc<Mutex<Option<Uuid>>>,
}

struct LayaProcess {
    child: Box<dyn ChildWrapper>,
    endpoint: String,
    device: Option<String>,
    device_reason: Option<String>,
    preference: LayaDevicePreference,
}

impl Drop for LayaProcess {
    fn drop(&mut self) {
        // Same-process safety net for a value dropped outside `stop_laya`
        // (e.g. replaced without an explicit stop). This does NOT protect
        // against the app crashing or being force-killed -- Rust's Drop
        // never runs if the whole process image disappears. That case is
        // covered independently by wrapping the spawn in a Windows Job
        // Object (see `ensure_laya_started`): the OS itself kills every
        // process in the job, including this one and anything it spawned,
        // when the job handle closes -- which Windows does automatically on
        // any process exit, orderly or not. Confirmed necessary in practice:
        // an orphaned Laya process survived 8+ hours after an ungraceful
        // shutdown before this fix.
        let _ = self.child.start_kill();
    }
}

#[derive(Debug, Clone, Serialize)]
struct LayaRuntimeStatus {
    installed: bool,
    running: bool,
    phase: String,
    detail: String,
    model: String,
    checkpoint: String,
    endpoint: Option<String>,
    device: Option<String>,
    device_reason: Option<String>,
    preference: LayaDevicePreference,
}

struct JudgeProcess {
    child: Box<dyn ChildWrapper>,
    endpoint: String,
    device: Option<String>,
    device_reason: Option<String>,
}

impl Drop for JudgeProcess {
    fn drop(&mut self) {
        // Same-process safety net only; see LayaProcess::drop for why the
        // Job Object in `ensure_judge_started` is the real protection
        // against a crash or force-kill.
        let _ = self.child.start_kill();
    }
}

/// A cross-model second opinion for Laya's uncertain picks (see
/// docs/decision-router-backends.md's "Cross-model judge" section).
/// `enabled` is a standalone toggle, not one of `DecisionRouterBackend`'s
/// exclusive choices, since the judge only ever activates alongside the
/// Laya backend, not in place of it.
#[derive(Debug, Clone, Serialize)]
struct JudgeRuntimeStatus {
    installed: bool,
    enabled: bool,
    running: bool,
    phase: String,
    detail: String,
    model: String,
    checkpoint: String,
    endpoint: Option<String>,
    device: Option<String>,
    device_reason: Option<String>,
}

#[cfg(windows)]
fn hide_process_window(command: &mut Command) {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    command.as_std_mut().creation_flags(CREATE_NO_WINDOW);
}

#[cfg(not(windows))]
fn hide_process_window(_command: &mut Command) {}

#[derive(Default)]
struct ApprovalBroker {
    pending: Mutex<HashMap<Uuid, oneshot::Sender<ApprovalDecision>>>,
    app: std::sync::OnceLock<tauri::AppHandle>,
}

#[derive(Default)]
struct QuestionBroker {
    pending: Mutex<HashMap<Uuid, oneshot::Sender<UserQuestionOutcome>>>,
    app: std::sync::OnceLock<tauri::AppHandle>,
}

#[derive(Serialize, Clone)]
struct QuestionRequest {
    id: Uuid,
    questions: Vec<UserQuestion>,
}

#[async_trait]
impl UserQuestionHandler for QuestionBroker {
    async fn ask(&self, questions: Vec<UserQuestion>) -> Result<UserQuestionOutcome> {
        let id = Uuid::new_v4();
        let (sender, receiver) = oneshot::channel();
        self.pending.lock().await.insert(id, sender);
        let app = self
            .app
            .get()
            .ok_or_else(|| pok_ai_core::PokError::Tool("question UI is not ready".into()))?;
        app.emit("question_requested", QuestionRequest { id, questions })
            .map_err(|error| pok_ai_core::PokError::Other(error.into()))?;
        match receiver.await {
            Ok(outcome) => Ok(outcome),
            Err(_) => Err(pok_ai_core::PokError::Tool("question was dismissed".into())),
        }
    }
}

struct TauriSessionObserver {
    sender: mpsc::UnboundedSender<AgentEvent>,
}

#[derive(Default)]
struct EventCoalescer {
    pending: Option<AgentEvent>,
}

impl EventCoalescer {
    fn push(&mut self, event: AgentEvent) -> Vec<AgentEvent> {
        match (&mut self.pending, event) {
            (
                Some(AgentEvent::ReasoningDelta { text: buffered }),
                AgentEvent::ReasoningDelta { text },
            ) => {
                buffered.push_str(&text);
                Vec::new()
            }
            (Some(AgentEvent::TextDelta { text: buffered }), AgentEvent::TextDelta { text }) => {
                buffered.push_str(&text);
                Vec::new()
            }
            (
                Some(AgentEvent::CommandProgress { event: buffered }),
                AgentEvent::CommandProgress { event },
            ) if buffered.call_id == event.call_id
                && buffered.kind == "output"
                && event.kind == "output"
                && buffered.stream == event.stream =>
            {
                if let Some(chunk) = event.chunk {
                    buffered
                        .chunk
                        .get_or_insert_with(String::new)
                        .push_str(&chunk);
                }
                buffered.total_bytes = event.total_bytes;
                buffered.status = event.status;
                Vec::new()
            }
            (
                _,
                event @ (AgentEvent::ReasoningDelta { .. }
                | AgentEvent::TextDelta { .. }
                | AgentEvent::CommandProgress { .. }),
            ) => {
                let flushed = self.pending.replace(event);
                flushed.into_iter().collect()
            }
            (_, event) => {
                let mut output = self.flush();
                output.push(event);
                output
            }
        }
    }

    fn flush(&mut self) -> Vec<AgentEvent> {
        self.pending.take().into_iter().collect()
    }
}

impl TauriSessionObserver {
    fn new(app: tauri::AppHandle) -> Self {
        let (sender, mut receiver) = mpsc::unbounded_channel();
        tauri::async_runtime::spawn(async move {
            let mut coalescer = EventCoalescer::default();
            let mut ticker = tokio::time::interval(Duration::from_millis(50));
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            // Tokio intervals tick immediately once. Consume that tick so the
            // first stream batch gets the same 50 ms window as later batches.
            ticker.tick().await;
            loop {
                tokio::select! {
                    event = receiver.recv() => {
                        let Some(event) = event else {
                            emit_agent_events(&app, coalescer.flush());
                            break;
                        };
                        emit_agent_events(&app, coalescer.push(event));
                    }
                    _ = ticker.tick() => {
                        emit_agent_events(&app, coalescer.flush());
                    }
                }
            }
        });
        Self { sender }
    }
}

impl SessionObserver for TauriSessionObserver {
    fn emit(&self, event: AgentEvent) {
        let _ = self.sender.send(event);
    }
}

fn emit_agent_events(app: &tauri::AppHandle, events: Vec<AgentEvent>) {
    for event in events {
        let _ = app.emit("agent_event", event);
    }
}

#[derive(Serialize, Clone)]
struct ApprovalRequest {
    id: Uuid,
    tool: String,
    reason: String,
    arguments: serde_json::Value,
}

#[async_trait]
impl ApprovalHandler for ApprovalBroker {
    async fn approve(
        &self,
        tool: &str,
        arguments: &serde_json::Value,
        reason: &str,
    ) -> Result<ApprovalDecision> {
        let id = Uuid::new_v4();
        let (sender, receiver) = oneshot::channel();
        self.pending.lock().await.insert(id, sender);
        let app = self
            .app
            .get()
            .ok_or_else(|| pok_ai_core::PokError::Tool("approval UI is not ready".into()))?;
        app.emit(
            "approval_requested",
            ApprovalRequest {
                id,
                tool: tool.into(),
                reason: reason.into(),
                arguments: arguments.clone(),
            },
        )
        .map_err(|error| pok_ai_core::PokError::Other(error.into()))?;
        match tokio::time::timeout(Duration::from_secs(300), receiver).await {
            Ok(Ok(allowed)) => Ok(allowed),
            _ => {
                self.pending.lock().await.remove(&id);
                Ok(ApprovalDecision::Deny)
            }
        }
    }
}

#[derive(Serialize)]
struct Status {
    provider: String,
    base_url: String,
    model: String,
    platform: &'static str,
    data_dir: PathBuf,
    model_runtime: Option<LocalModelRuntimeStatus>,
    requires_api_key: bool,
    has_api_key: bool,
    data_boundary: ProviderDataBoundary,
}

#[derive(Debug, Default, Deserialize, Serialize)]
struct ProviderEndpointOverrides {
    endpoints: BTreeMap<String, String>,
}

fn provider_endpoint_overrides_path(data_dir: &std::path::Path) -> PathBuf {
    data_dir.join("provider-endpoints.json")
}

fn valid_provider_endpoint(endpoint: &str) -> bool {
    endpoint.starts_with("http://") || endpoint.starts_with("https://")
}

fn read_provider_endpoint_overrides(
    data_dir: &std::path::Path,
) -> std::result::Result<ProviderEndpointOverrides, String> {
    let path = provider_endpoint_overrides_path(data_dir);
    match std::fs::read(&path) {
        Ok(bytes) => serde_json::from_slice(&bytes)
            .map_err(|error| format!("could not read {}: {error}", path.display())),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            Ok(ProviderEndpointOverrides::default())
        }
        Err(error) => Err(format!("could not read {}: {error}", path.display())),
    }
}

fn apply_provider_endpoint_overrides(config: &mut Config) -> std::result::Result<(), String> {
    let overrides = read_provider_endpoint_overrides(&config.data_dir)?;
    for (provider, endpoint) in &overrides.endpoints {
        if !valid_provider_endpoint(endpoint) {
            return Err(format!(
                "saved endpoint for provider {provider:?} must start with http:// or https://"
            ));
        }
    }
    for (provider, endpoint) in overrides.endpoints {
        if let Some(provider_config) = config.providers.get_mut(&provider) {
            provider_config.base_url = endpoint;
        }
    }
    Ok(())
}

fn persist_provider_endpoint_override(
    data_dir: &std::path::Path,
    provider: &str,
    endpoint: &str,
) -> std::result::Result<(), String> {
    let mut overrides = read_provider_endpoint_overrides(data_dir)?;
    overrides
        .endpoints
        .insert(provider.to_owned(), endpoint.to_owned());
    let bytes = serde_json::to_vec_pretty(&overrides)
        .map_err(|error| format!("could not encode provider endpoint settings: {error}"))?;
    atomic_write(&provider_endpoint_overrides_path(data_dir), &bytes)
        .map_err(|error| format!("could not save provider endpoint settings: {error}"))
}

fn local_lifecycle_supported(provider: &str) -> bool {
    matches!(
        provider.to_ascii_lowercase().as_str(),
        "lm_studio" | "ollama"
    )
}

fn model_transition_status(
    provider: &str,
    model: &str,
    phase: LocalModelPhase,
    detail: impl Into<String>,
) -> LocalModelRuntimeStatus {
    LocalModelRuntimeStatus {
        provider: provider.into(),
        model: model.into(),
        phase,
        detail: Some(detail.into()),
        instance_ids: Vec::new(),
        context_length: None,
    }
}

fn emit_model_runtime(app: &tauri::AppHandle, status: &LocalModelRuntimeStatus) {
    let _ = app.emit("model_runtime_event", status);
}

fn model_runtime_key(provider: &str, model: &str) -> (String, String) {
    (provider.to_ascii_lowercase(), model.to_ascii_lowercase())
}

async fn cached_model_runtime(
    runtime: &AppRuntime,
    provider: &str,
    model: &str,
) -> Option<LocalModelRuntimeStatus> {
    runtime
        .model_runtime_cache
        .lock()
        .await
        .get(&model_runtime_key(provider, model))
        .cloned()
}

async fn cache_model_runtime(runtime: &AppRuntime, status: &LocalModelRuntimeStatus) {
    runtime.model_runtime_cache.lock().await.insert(
        model_runtime_key(&status.provider, &status.model),
        status.clone(),
    );
    if matches!(
        status.phase,
        LocalModelPhase::Loaded | LocalModelPhase::Unloaded
    ) {
        let provider_prefix = format!("{}\n", status.provider.to_ascii_lowercase());
        for (key, models) in runtime.model_catalog_cache.lock().await.iter_mut() {
            if key.starts_with(&provider_prefix) {
                if let Some(model) = models
                    .iter_mut()
                    .find(|model| model.id.eq_ignore_ascii_case(&status.model))
                {
                    model.loaded = Some(status.phase == LocalModelPhase::Loaded);
                }
            }
        }
    }
}

async fn provider_model_catalog(
    runtime: &AppRuntime,
    provider: &str,
    provider_config: &pok_ai_core::config::ProviderConfig,
) -> std::result::Result<Vec<ModelInfo>, String> {
    let cache_key = format!(
        "{}\n{}",
        provider.to_ascii_lowercase(),
        provider_config.base_url.to_ascii_lowercase()
    );
    if local_lifecycle_supported(provider) {
        if let Some(models) = runtime
            .model_catalog_cache
            .lock()
            .await
            .get(&cache_key)
            .cloned()
        {
            return Ok(models);
        }
        let _query = runtime.local_api_query.lock().await;
        let mut cache = runtime.model_catalog_cache.lock().await;
        if let Some(models) = cache.get(&cache_key) {
            return Ok(models.clone());
        }
        let brain = from_provider(provider_config).map_err(|error| error.to_string())?;
        let models = if provider.eq_ignore_ascii_case("ollama") {
            brain
                .list_models()
                .await
                .map_err(|error| error.to_string())?
                .into_iter()
                .map(|id| ModelInfo {
                    id,
                    context_length: None,
                    loaded_context_length: None,
                    max_context_length: None,
                    vision: None,
                    tool_use: None,
                    reasoning: None,
                    supported_parameters: Vec::new(),
                    reasoning_efforts: Vec::new(),
                    reasoning_default: None,
                    loaded: None,
                })
                .collect()
        } else {
            brain
                .model_info()
                .await
                .map_err(|error| error.to_string())?
        };
        cache.insert(cache_key, models.clone());
        return Ok(models);
    }
    let brain = from_provider(provider_config).map_err(|error| error.to_string())?;
    brain.model_info().await.map_err(|error| error.to_string())
}

async fn economical_model_runtime(
    runtime: &AppRuntime,
    provider: &str,
    provider_config: &pok_ai_core::config::ProviderConfig,
    model: &str,
) -> LocalModelRuntimeStatus {
    if model.trim().is_empty() {
        return model_transition_status(
            provider,
            model,
            LocalModelPhase::Unloaded,
            "Select a model",
        );
    }
    if provider.eq_ignore_ascii_case("lm_studio") {
        return match provider_model_catalog(runtime, provider, provider_config).await {
            Ok(models) => match models
                .into_iter()
                .find(|item| item.id.eq_ignore_ascii_case(model))
            {
                Some(info) => LocalModelRuntimeStatus {
                    provider: provider.into(),
                    model: model.into(),
                    phase: match info.loaded {
                        Some(true) => LocalModelPhase::Loaded,
                        Some(false) => LocalModelPhase::Unloaded,
                        None => LocalModelPhase::Unsupported,
                    },
                    detail: info.loaded.is_none().then(|| {
                        "LM Studio load state is unavailable; using lazy inference".into()
                    }),
                    // Exact instance IDs are only needed when an unload is requested.
                    instance_ids: Vec::new(),
                    context_length: info.loaded_context_length,
                },
                None => model_transition_status(
                    provider,
                    model,
                    LocalModelPhase::Error,
                    format!("Model {model:?} was not found by LM Studio"),
                ),
            },
            Err(error) => {
                model_transition_status(provider, model, LocalModelPhase::ServerOffline, error)
            }
        };
    }
    let _query = runtime.local_api_query.lock().await;
    local_model_status(provider, provider_config, model).await
}

fn provider_config(
    runtime: &AppRuntime,
    provider: &str,
) -> std::result::Result<pok_ai_core::config::ProviderConfig, String> {
    runtime
        .config
        .lock()
        .provider(Some(provider))
        .cloned()
        .map_err(|error| error.to_string())
}

async fn load_model_locked(
    app: &tauri::AppHandle,
    runtime: &AppRuntime,
    provider: &str,
    model: &str,
    provider_config: &pok_ai_core::config::ProviderConfig,
) -> std::result::Result<LocalModelRuntimeStatus, String> {
    let desired_context = {
        let data_dir = runtime.config.lock().data_dir.clone();
        ContextSettings::load(&data_dir)
            .models
            .get(&ContextSettings::key(provider, model))
            .and_then(|value| value.context_window_tokens)
    };
    let cached = cached_model_runtime(runtime, provider, model).await;
    if let Some(current) = cached.as_ref() {
        if current.phase == LocalModelPhase::Unsupported
            || (current.phase == LocalModelPhase::Loaded
                && current.context_length.is_some()
                && desired_context.is_none_or(|desired| current.context_length == Some(desired)))
        {
            emit_model_runtime(app, &current);
            return Ok(current.clone());
        }
    }
    let loading = model_transition_status(
        provider,
        model,
        LocalModelPhase::Loading,
        format!("Loading {model}"),
    );
    emit_model_runtime(app, &loading);
    let result = {
        let _query = runtime.local_api_query.lock().await;
        match cached.filter(|current| current.phase == LocalModelPhase::Unloaded) {
            Some(current) => {
                load_local_model_from_status_with_context(
                    provider,
                    provider_config,
                    model,
                    current,
                    desired_context,
                )
                .await
            }
            None => {
                load_local_model_with_context(provider, provider_config, model, desired_context)
                    .await
            }
        }
    };
    match result {
        Ok(loaded) => {
            cache_model_runtime(runtime, &loaded).await;
            let provider_prefix = format!("{}\n", provider.to_ascii_lowercase());
            runtime
                .model_catalog_cache
                .lock()
                .await
                .retain(|key, _| !key.starts_with(&provider_prefix));
            emit_model_runtime(app, &loaded);
            Ok(loaded)
        }
        Err(error) => {
            let failed =
                model_transition_status(provider, model, LocalModelPhase::Error, error.to_string());
            cache_model_runtime(runtime, &failed).await;
            emit_model_runtime(app, &failed);
            Err(error.to_string())
        }
    }
}

async fn unload_model_locked(
    app: &tauri::AppHandle,
    runtime: &AppRuntime,
    provider: &str,
    model: &str,
    provider_config: &pok_ai_core::config::ProviderConfig,
) -> std::result::Result<LocalModelRuntimeStatus, String> {
    let cached = cached_model_runtime(runtime, provider, model).await;
    if let Some(current) = cached.as_ref() {
        if current.phase == LocalModelPhase::Unloaded {
            emit_model_runtime(app, &current);
            return Ok(current.clone());
        }
    }
    let unloading = model_transition_status(
        provider,
        model,
        LocalModelPhase::Unloading,
        format!("Unloading {model}"),
    );
    emit_model_runtime(app, &unloading);
    let _query = runtime.local_api_query.lock().await;
    let result = match cached.filter(|current| {
        current.phase == LocalModelPhase::Loaded
            && (!provider.eq_ignore_ascii_case("lm_studio") || !current.instance_ids.is_empty())
    }) {
        Some(current) => {
            unload_local_model_from_status(provider, provider_config, model, current).await
        }
        None => unload_local_model(provider, provider_config, model).await,
    };
    match result {
        Ok(unloaded) => {
            cache_model_runtime(runtime, &unloaded).await;
            let provider_prefix = format!("{}\n", provider.to_ascii_lowercase());
            runtime
                .model_catalog_cache
                .lock()
                .await
                .retain(|key, _| !key.starts_with(&provider_prefix));
            emit_model_runtime(app, &unloaded);
            Ok(unloaded)
        }
        Err(error) => {
            let failed =
                model_transition_status(provider, model, LocalModelPhase::Error, error.to_string());
            cache_model_runtime(runtime, &failed).await;
            emit_model_runtime(app, &failed);
            Err(error.to_string())
        }
    }
}

#[derive(Serialize)]
struct ModelContextStatus {
    budget: pok_ai_core::context::ContextBudget,
}

#[tauri::command]
async fn get_model_context(
    provider: String,
    model: String,
    runtime: State<'_, AppRuntime>,
) -> std::result::Result<ModelContextStatus, String> {
    let (provider_config, data_dir) = {
        let config = runtime.config.lock();
        (
            config
                .provider(Some(&provider))
                .cloned()
                .map_err(|error| error.to_string())?,
            config.data_dir.clone(),
        )
    };
    let provider_contexts = provider_model_catalog(runtime.inner(), &provider, &provider_config)
        .await
        .ok()
        .and_then(|models| {
            models
                .into_iter()
                .find(|item| item.id.eq_ignore_ascii_case(&model))
                .map(|item| (item.loaded_context_length, item.max_context_length))
        });
    let budget = resolve_context_budget(
        &data_dir,
        &provider,
        &model,
        provider_contexts.and_then(|value| value.0),
        provider_contexts.and_then(|value| value.1),
        32_768,
    )
    .await;
    Ok(ModelContextStatus { budget })
}

#[tauri::command]
fn save_model_context_override(
    provider: String,
    model: String,
    context_window_tokens: Option<u64>,
    threshold_percent: Option<u8>,
    runtime: State<'_, AppRuntime>,
) -> std::result::Result<(), String> {
    if context_window_tokens.is_some_and(|value| value < 4_000) {
        return Err("context window must be at least 4,000 tokens".into());
    }
    if threshold_percent.is_some_and(|value| !(50..=95).contains(&value)) {
        return Err("compaction threshold must be between 50% and 95%".into());
    }
    let data_dir = runtime.config.lock().data_dir.clone();
    let mut settings = ContextSettings::load(&data_dir);
    settings.models.insert(
        ContextSettings::key(&provider, &model),
        ModelContextOverride {
            context_window_tokens,
            threshold_percent,
        },
    );
    settings.save(&data_dir).map_err(|error| error.to_string())
}

fn resolved_workspace(workspace: &str) -> std::result::Result<PathBuf, String> {
    let path = if workspace.trim().is_empty() {
        std::env::current_dir().map_err(|error| error.to_string())?
    } else {
        PathBuf::from(workspace)
    };
    dunce::canonicalize(path).map_err(|error| error.to_string())
}

#[tauri::command]
fn list_generated_tools(
    workspace: String,
    runtime: State<'_, AppRuntime>,
) -> std::result::Result<Vec<GeneratedToolSummary>, String> {
    let workspace = resolved_workspace(&workspace)?;
    let data_dir = runtime.config.lock().data_dir.clone();
    list_project_generated_tools(&data_dir, &workspace).map_err(|error| error.to_string())
}

#[tauri::command]
fn set_generated_tool_enabled(
    name: String,
    enabled: bool,
    workspace: String,
    runtime: State<'_, AppRuntime>,
) -> std::result::Result<(), String> {
    let workspace = resolved_workspace(&workspace)?;
    let data_dir = runtime.config.lock().data_dir.clone();
    set_project_generated_tool_enabled(&data_dir, &workspace, &name, enabled)
        .map_err(|error| error.to_string())
}

#[tauri::command]
fn delete_generated_tool_command(
    name: String,
    workspace: String,
    runtime: State<'_, AppRuntime>,
) -> std::result::Result<bool, String> {
    let workspace = resolved_workspace(&workspace)?;
    let data_dir = runtime.config.lock().data_dir.clone();
    delete_generated_tool(&data_dir, &workspace, &name).map_err(|error| error.to_string())
}

#[tauri::command]
fn list_generated_tool_candidates_command(
    workspace: String,
    runtime: State<'_, AppRuntime>,
) -> std::result::Result<Vec<GeneratedToolCandidate>, String> {
    let workspace = resolved_workspace(&workspace)?;
    let config = runtime.config.lock();
    list_generated_tool_candidates(&config.data_dir, &workspace).map_err(|error| error.to_string())
}

#[tauri::command]
fn dismiss_generated_tool_candidate_command(
    id: String,
    workspace: String,
    runtime: State<'_, AppRuntime>,
) -> std::result::Result<bool, String> {
    let workspace = resolved_workspace(&workspace)?;
    let config = runtime.config.lock();
    dismiss_generated_tool_candidate(&config.data_dir, &workspace, &id)
        .map_err(|error| error.to_string())
}

#[tauri::command]
async fn get_status(
    provider: Option<String>,
    model: Option<String>,
    runtime: State<'_, AppRuntime>,
) -> std::result::Result<Status, String> {
    let (provider_name, provider_config, default_model, data_dir) = {
        let config = runtime.config.lock();
        let name = provider.unwrap_or_else(|| config.default_provider.clone());
        let provider_config = config
            .provider(Some(&name))
            .cloned()
            .map_err(|error| error.to_string())?;
        (
            name,
            provider_config,
            config.default_model.clone(),
            config.data_dir.clone(),
        )
    };
    let base_url = provider_config.base_url.clone();
    let data_boundary = provider_config.resolved_data_boundary(&provider_name);
    let requires_api_key = data_boundary == ProviderDataBoundary::ExternalService
        && provider_config.api_key_env.is_some();
    let has_api_key = provider_config
        .api_key_env
        .as_deref()
        .and_then(pok_ai_core::brain::get_api_key)
        .is_some_and(|k| !k.trim().is_empty());
    let target_model = model.unwrap_or(default_model);
    let model_runtime = if local_lifecycle_supported(&provider_name) {
        let status = economical_model_runtime(
            runtime.inner(),
            &provider_name,
            &provider_config,
            &target_model,
        )
        .await;
        cache_model_runtime(runtime.inner(), &status).await;
        Some(status)
    } else {
        None
    };
    Ok(Status {
        provider: provider_name,
        base_url,
        model: target_model,
        platform: std::env::consts::OS,
        data_dir,
        model_runtime,
        requires_api_key,
        has_api_key,
        data_boundary,
    })
}

#[tauri::command]
async fn update_provider_endpoint(
    provider: String,
    endpoint: String,
    model: Option<String>,
    runtime: State<'_, AppRuntime>,
) -> std::result::Result<Status, String> {
    let endpoint = endpoint.trim();
    if endpoint.is_empty() {
        return Err("Endpoint URL cannot be empty".into());
    }
    if !valid_provider_endpoint(endpoint) {
        return Err("Endpoint URL must start with http:// or https://".into());
    }
    let (data_dir, default_model, provider_config) = {
        let mut config = runtime.config.lock();
        let data_dir = config.data_dir.clone();
        let default_model = config.default_model.clone();
        let provider_config = config
            .providers
            .get_mut(&provider)
            .ok_or_else(|| format!("Provider {provider:?} is not configured"))?;
        persist_provider_endpoint_override(&data_dir, &provider, endpoint)?;
        provider_config.base_url = endpoint.to_string();
        (data_dir, default_model, provider_config.clone())
    };
    let data_boundary = provider_config.resolved_data_boundary(&provider);
    let requires_api_key = data_boundary == ProviderDataBoundary::ExternalService
        && provider_config.api_key_env.is_some();
    let has_api_key = provider_config
        .api_key_env
        .as_deref()
        .and_then(pok_ai_core::brain::get_api_key)
        .is_some_and(|k| !k.trim().is_empty());
    let target_model = model.unwrap_or(default_model);
    let model_runtime = if local_lifecycle_supported(&provider) {
        runtime
            .model_runtime_cache
            .lock()
            .await
            .retain(|(cached_provider, _), _| !cached_provider.eq_ignore_ascii_case(&provider));
        runtime
            .model_catalog_cache
            .lock()
            .await
            .retain(|key, _| !key.starts_with(&format!("{}\n", provider.to_ascii_lowercase())));
        let status =
            economical_model_runtime(runtime.inner(), &provider, &provider_config, &target_model)
                .await;
        cache_model_runtime(runtime.inner(), &status).await;
        Some(status)
    } else {
        None
    };

    Ok(Status {
        provider: provider.clone(),
        base_url: endpoint.to_string(),
        model: target_model,
        platform: std::env::consts::OS,
        data_dir,
        model_runtime,
        requires_api_key,
        has_api_key,
        data_boundary,
    })
}

#[tauri::command]
async fn get_local_model_status(
    provider: String,
    model: String,
    app: tauri::AppHandle,
    runtime: State<'_, AppRuntime>,
) -> std::result::Result<LocalModelRuntimeStatus, String> {
    let provider_config = provider_config(runtime.inner(), &provider)?;
    let _transition = runtime.model_transition.lock().await;
    let _query = runtime.local_api_query.lock().await;
    let status = local_model_status(&provider, &provider_config, &model).await;
    cache_model_runtime(runtime.inner(), &status).await;
    emit_model_runtime(&app, &status);
    Ok(status)
}

#[tauri::command]
async fn load_model(
    provider: String,
    model: String,
    app: tauri::AppHandle,
    runtime: State<'_, AppRuntime>,
) -> std::result::Result<LocalModelRuntimeStatus, String> {
    let provider_config = provider_config(runtime.inner(), &provider)?;
    let _transition = runtime.model_transition.lock().await;
    load_model_locked(&app, runtime.inner(), &provider, &model, &provider_config).await
}

#[tauri::command]
async fn switch_local_model(
    previous_provider: Option<String>,
    previous_model: Option<String>,
    provider: String,
    model: String,
    app: tauri::AppHandle,
    runtime: State<'_, AppRuntime>,
) -> std::result::Result<LocalModelRuntimeStatus, String> {
    let next_config = provider_config(runtime.inner(), &provider)?;
    let previous = match (previous_provider, previous_model) {
        (Some(previous_provider), Some(previous_model))
            if !previous_model.trim().is_empty()
                && local_lifecycle_supported(&previous_provider)
                && (previous_provider != provider || previous_model != model) =>
        {
            Some((
                previous_provider.clone(),
                previous_model,
                provider_config(runtime.inner(), &previous_provider)?,
            ))
        }
        _ => None,
    };
    let _transition = runtime.model_transition.lock().await;
    if let Some((previous_provider, previous_model, previous_config)) = previous {
        unload_model_locked(
            &app,
            runtime.inner(),
            &previous_provider,
            &previous_model,
            &previous_config,
        )
        .await?;
    }
    load_model_locked(&app, runtime.inner(), &provider, &model, &next_config).await
}

#[tauri::command]
async fn list_models(
    provider: Option<String>,
    runtime: State<'_, AppRuntime>,
) -> std::result::Result<Vec<String>, String> {
    let (provider_name, provider_config) = {
        let config = runtime.config.lock();
        let provider_name = provider.unwrap_or_else(|| config.default_provider.clone());
        let provider_config = config
            .provider(Some(&provider_name))
            .cloned()
            .map_err(|error| error.to_string())?;
        (provider_name, provider_config)
    };
    if local_lifecycle_supported(&provider_name) {
        return provider_model_catalog(runtime.inner(), &provider_name, &provider_config)
            .await
            .map(|models| models.into_iter().map(|model| model.id).collect());
    }
    let brain = from_provider(&provider_config).map_err(|error| error.to_string())?;
    brain.list_models().await.map_err(|error| error.to_string())
}

#[tauri::command]
async fn get_model_capabilities(
    provider: String,
    model: String,
    runtime: State<'_, AppRuntime>,
) -> std::result::Result<ModelInfo, String> {
    let provider_config = provider_config(runtime.inner(), &provider)?;
    let models = provider_model_catalog(runtime.inner(), &provider, &provider_config).await?;
    Ok(models
        .into_iter()
        .find(|item| item.id.eq_ignore_ascii_case(&model))
        .unwrap_or(ModelInfo {
            id: model,
            context_length: None,
            loaded_context_length: None,
            max_context_length: None,
            vision: None,
            tool_use: None,
            reasoning: None,
            supported_parameters: Vec::new(),
            reasoning_efforts: Vec::new(),
            reasoning_default: None,
            loaded: None,
        }))
}

#[tauri::command]
async fn refresh_models(
    provider: String,
    runtime: State<'_, AppRuntime>,
) -> std::result::Result<Vec<String>, String> {
    let provider_config = provider_config(runtime.inner(), &provider)?;
    let provider_prefix = format!("{}\n", provider.to_ascii_lowercase());
    runtime
        .model_catalog_cache
        .lock()
        .await
        .retain(|key, _| !key.starts_with(&provider_prefix));

    if local_lifecycle_supported(&provider) {
        let models = provider_model_catalog(runtime.inner(), &provider, &provider_config).await?;
        let model_ids: Vec<String> = models.into_iter().map(|model| model.id).collect();
        runtime
            .model_runtime_cache
            .lock()
            .await
            .retain(|(cached_provider, cached_model), _| {
                !cached_provider.eq_ignore_ascii_case(&provider)
                    || model_ids
                        .iter()
                        .any(|model| model.eq_ignore_ascii_case(cached_model))
            });
        return Ok(model_ids);
    }
    let brain = from_provider(&provider_config).map_err(|error| error.to_string())?;
    brain.list_models().await.map_err(|error| error.to_string())
}

#[tauri::command]
fn list_memory_drafts(
    runtime: State<'_, AppRuntime>,
) -> std::result::Result<Vec<MemoryRecord>, String> {
    let config = runtime.config.lock();
    MemoryStore::open(config.data_dir.join("memory"))
        .and_then(|memory| memory.list_drafts(20))
        .map_err(|error| error.to_string())
}

#[tauri::command]
fn approve_memory(id: Uuid, runtime: State<'_, AppRuntime>) -> std::result::Result<bool, String> {
    let config = runtime.config.lock();
    MemoryStore::open(config.data_dir.join("memory"))
        .and_then(|memory| memory.approve(id))
        .map_err(|error| error.to_string())
}

#[tauri::command]
fn reject_memory(id: Uuid, runtime: State<'_, AppRuntime>) -> std::result::Result<bool, String> {
    let config = runtime.config.lock();
    MemoryStore::open(config.data_dir.join("memory"))
        .and_then(|memory| memory.reject_draft(id))
        .map_err(|error| error.to_string())
}

#[derive(Serialize)]
struct MemoryLibrary {
    facts: Vec<MemoryRecord>,
    skills: Vec<ProcedureRecord>,
}

#[tauri::command]
fn list_memory_library(
    runtime: State<'_, AppRuntime>,
) -> std::result::Result<MemoryLibrary, String> {
    let config = runtime.config.lock();
    MemoryStore::open(config.data_dir.join("memory"))
        .and_then(|memory| {
            Ok(MemoryLibrary {
                facts: memory.list_memories(200)?,
                skills: memory.list_procedures(Some(ProcedureKind::Workflow), 100)?,
            })
        })
        .map_err(|error| error.to_string())
}

#[tauri::command]
fn set_memory_enabled(
    id: Uuid,
    enabled: bool,
    runtime: State<'_, AppRuntime>,
) -> std::result::Result<bool, String> {
    let config = runtime.config.lock();
    MemoryStore::open(config.data_dir.join("memory"))
        .and_then(|memory| memory.set_memory_enabled(id, enabled))
        .map_err(|error| error.to_string())
}

#[tauri::command]
fn delete_memory(id: Uuid, runtime: State<'_, AppRuntime>) -> std::result::Result<bool, String> {
    let config = runtime.config.lock();
    MemoryStore::open(config.data_dir.join("memory"))
        .and_then(|memory| memory.delete_memory(id))
        .map_err(|error| error.to_string())
}

#[tauri::command]
fn forget_memory_and_prevent_relearning(
    id: Uuid,
    runtime: State<'_, AppRuntime>,
) -> std::result::Result<bool, String> {
    let config = runtime.config.lock();
    MemoryStore::open(config.data_dir.join("memory"))
        .and_then(|memory| memory.forget_memory_and_prevent_relearning(id))
        .map_err(|error| error.to_string())
}

#[tauri::command]
fn scan_memory_duplicates(
    runtime: State<'_, AppRuntime>,
) -> std::result::Result<Vec<MemoryDuplicateGroup>, String> {
    let config = runtime.config.lock();
    MemoryStore::open(config.data_dir.join("memory"))
        .and_then(|memory| memory.find_duplicate_groups(300))
        .map_err(|error| error.to_string())
}

#[derive(Deserialize)]
struct MemoryMergeVersion {
    id: Uuid,
    version: u64,
}

#[tauri::command]
fn merge_memories(
    canonical_id: Uuid,
    records: Vec<MemoryMergeVersion>,
    runtime: State<'_, AppRuntime>,
) -> std::result::Result<Uuid, String> {
    let config = runtime.config.lock();
    let expected = records
        .into_iter()
        .map(|record| (record.id, record.version))
        .collect::<Vec<_>>();
    MemoryStore::open(config.data_dir.join("memory"))
        .and_then(|memory| memory.merge_memories(canonical_id, &expected))
        .map_err(|error| error.to_string())
}

#[tauri::command]
fn undo_memory_merge(
    merge_id: Uuid,
    runtime: State<'_, AppRuntime>,
) -> std::result::Result<bool, String> {
    let config = runtime.config.lock();
    MemoryStore::open(config.data_dir.join("memory"))
        .and_then(|memory| memory.undo_memory_merge(merge_id))
        .map_err(|error| error.to_string())
}

#[tauri::command]
fn set_procedure_enabled(
    id: Uuid,
    enabled: bool,
    runtime: State<'_, AppRuntime>,
) -> std::result::Result<bool, String> {
    let config = runtime.config.lock();
    MemoryStore::open(config.data_dir.join("memory"))
        .and_then(|memory| memory.set_procedure_enabled(id, enabled))
        .map_err(|error| error.to_string())
}

#[tauri::command]
fn delete_procedure(id: Uuid, runtime: State<'_, AppRuntime>) -> std::result::Result<bool, String> {
    let config = runtime.config.lock();
    MemoryStore::open(config.data_dir.join("memory"))
        .and_then(|memory| memory.delete_procedure(id))
        .map_err(|error| error.to_string())
}

#[tauri::command]
async fn resolve_approval(
    id: Uuid,
    allow: bool,
    remember_always: Option<bool>,
    broker: State<'_, Arc<ApprovalBroker>>,
) -> std::result::Result<(), String> {
    let decision = match (allow, remember_always == Some(true)) {
        (false, _) => ApprovalDecision::Deny,
        (true, true) => ApprovalDecision::AllowSession,
        (true, false) => ApprovalDecision::AllowOnce,
    };
    let sender = broker
        .pending
        .lock()
        .await
        .remove(&id)
        .ok_or_else(|| "approval request expired".to_string())?;
    sender
        .send(decision)
        .map_err(|_| "agent is no longer waiting".to_string())
}

#[tauri::command]
async fn resolve_question(
    id: Uuid,
    answers: Vec<UserQuestionAnswer>,
    broker: State<'_, Arc<QuestionBroker>>,
) -> std::result::Result<(), String> {
    let sender = broker
        .pending
        .lock()
        .await
        .remove(&id)
        .ok_or_else(|| "question request expired".to_string())?;
    sender
        .send(UserQuestionOutcome::Answered(answers))
        .map_err(|_| "agent is no longer waiting".to_string())
}

#[tauri::command]
async fn dismiss_question(
    id: Uuid,
    broker: State<'_, Arc<QuestionBroker>>,
) -> std::result::Result<(), String> {
    let sender = broker
        .pending
        .lock()
        .await
        .remove(&id)
        .ok_or_else(|| "question request expired".to_string())?;
    sender
        .send(UserQuestionOutcome::Dismissed)
        .map_err(|_| "agent is no longer waiting".to_string())
}

#[tauri::command]
fn emergency_stop(
    app: tauri::AppHandle,
    runtime: State<'_, AppRuntime>,
    broker: State<'_, Arc<ApprovalBroker>>,
) -> std::result::Result<(), String> {
    runtime.cancellation.lock().cancel();
    if let Some(manager) = runtime.command_manager.lock().clone() {
        tauri::async_runtime::spawn(async move {
            for command in manager.list().await {
                if command.status == pok_ai_core::commands::CommandStatus::Running {
                    let _ = manager.kill(&command.task_id, "emergency_stop").await;
                }
            }
        });
    }
    let _ = app.emit("emergency_stopped", ());
    let broker = broker.inner().clone();
    tauri::async_runtime::spawn(async move {
        broker.pending.lock().await.clear();
    });
    Ok(())
}

#[tauri::command]
fn pause_agent(runtime: State<'_, AppRuntime>) -> std::result::Result<(), String> {
    if runtime.pause.request_pause() {
        Ok(())
    } else {
        Err("the agent is already paused or a pause is pending".into())
    }
}

#[tauri::command]
fn resume_agent(
    note: Option<String>,
    runtime: State<'_, AppRuntime>,
) -> std::result::Result<(), String> {
    if runtime.pause.resume(note) {
        Ok(())
    } else {
        Err("the agent is not paused".into())
    }
}

#[tauri::command]
async fn new_conversation(runtime: State<'_, AppRuntime>) -> std::result::Result<(), String> {
    if let Some(manager) = runtime.command_manager.lock().take() {
        manager.request_shutdown();
    }
    *runtime.conversation.lock().await = None;
    *runtime.conversation_provider.lock().await = None;
    *runtime.conversation_decision_router_backend.lock().await = None;
    *runtime.conversation_judge_enabled.lock().await = None;
    *runtime.pending_resume.lock().await = None;
    runtime
        .external_provider_acknowledgments
        .lock()
        .await
        .clear();
    Ok(())
}

#[derive(Serialize)]
struct ResumeConversationPayload {
    summary: ConversationSummary,
    messages: Vec<ConversationDisplayMessage>,
    next_before_sequence: Option<u64>,
    has_more: bool,
    decision_router_enabled: bool,
    decision_router_backend: DecisionRouterBackend,
    decision_activity: Vec<serde_json::Value>,
}

fn open_conversation_store(
    runtime: &AppRuntime,
) -> std::result::Result<Arc<ConversationStore>, String> {
    let config = runtime.config.lock();
    let store = ConversationStore::open(&config.data_dir).map_err(|error| error.to_string())?;
    store
        .import_legacy_diagnostics(&config.diagnostics_dir)
        .map_err(|error| error.to_string())?;
    Ok(store)
}

#[tauri::command]
fn list_conversations(
    runtime: State<'_, AppRuntime>,
) -> std::result::Result<Vec<ConversationSummary>, String> {
    open_conversation_store(runtime.inner())?
        .list(None, 50)
        .map_err(|error| error.to_string())
}

#[tauri::command]
async fn resume_conversation(
    session_id: Uuid,
    runtime: State<'_, AppRuntime>,
) -> std::result::Result<ResumeConversationPayload, String> {
    let store = open_conversation_store(runtime.inner())?;
    let (summary, state) = store
        .metadata(session_id)
        .map_err(|error| error.to_string())?
        .ok_or_else(|| format!("conversation {session_id} was not found"))?;
    let history = store
        .display_messages_page(session_id, None, 100)
        .map_err(|error| error.to_string())?;
    if let Some(manager) = runtime.command_manager.lock().take() {
        manager.request_shutdown();
    }
    *runtime.conversation.lock().await = None;
    *runtime.conversation_provider.lock().await = None;
    *runtime.conversation_decision_router_backend.lock().await = None;
    *runtime.conversation_judge_enabled.lock().await = None;
    *runtime.pending_resume.lock().await = Some(session_id);
    runtime
        .external_provider_acknowledgments
        .lock()
        .await
        .clear();
    Ok(ResumeConversationPayload {
        decision_activity: state
            .get("decision_activity")
            .and_then(serde_json::Value::as_array)
            .cloned()
            .unwrap_or_default(),
        decision_router_enabled: state
            .get("decision_router_enabled")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        decision_router_backend: state
            .get("decision_router_backend")
            .and_then(|value| serde_json::from_value(value.clone()).ok())
            .unwrap_or_else(|| {
                if state
                    .get("decision_router_enabled")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false)
                {
                    DecisionRouterBackend::Jev
                } else {
                    DecisionRouterBackend::Off
                }
            }),
        summary,
        messages: history.messages,
        next_before_sequence: history.next_before_sequence,
        has_more: history.has_more,
    })
}

/// An image the user dropped on the window, as a data URL, so the dashboard
/// can attach it for a vision model. Only common image types, up to 30 MB.
#[tauri::command]
fn read_dropped_image(path: String) -> std::result::Result<String, String> {
    use base64::Engine as _;
    let path = std::path::PathBuf::from(path);
    let mime = match path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        Some("bmp") => "image/bmp",
        _ => return Err("not an image".into()),
    };
    let size = std::fs::metadata(&path)
        .map_err(|error| error.to_string())?
        .len();
    if size > 30 * 1024 * 1024 {
        return Err("the image is larger than 30 MB".into());
    }
    let bytes = std::fs::read(&path).map_err(|error| error.to_string())?;
    Ok(format!(
        "data:{mime};base64,{}",
        base64::engine::general_purpose::STANDARD.encode(bytes)
    ))
}

/// A saved observation frame for the dashboard's agent view, as a data URL.
/// Only `observation-*.png` files inside the diagnostics folder are served.
#[tauri::command]
fn read_observation_frame(
    path: String,
    runtime: State<'_, AppRuntime>,
) -> std::result::Result<String, String> {
    use base64::Engine as _;
    let diagnostics = runtime.config.lock().diagnostics_dir.clone();
    let root = dunce::canonicalize(&diagnostics).map_err(|error| error.to_string())?;
    let file = dunce::canonicalize(&path).map_err(|_| "frame not found".to_string())?;
    let name = file
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    if !file.starts_with(&root) || !name.starts_with("observation-") || !name.ends_with(".png") {
        return Err("not an observation frame".into());
    }
    let bytes = std::fs::read(&file).map_err(|error| error.to_string())?;
    Ok(format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(bytes)
    ))
}

#[tauri::command]
fn load_conversation_history(
    session_id: Uuid,
    before_sequence: Option<u64>,
    limit: Option<usize>,
    runtime: State<'_, AppRuntime>,
) -> std::result::Result<ConversationHistoryPage, String> {
    open_conversation_store(runtime.inner())?
        .display_messages_page(session_id, before_sequence, limit.unwrap_or(100))
        .map_err(|error| error.to_string())
}

#[tauri::command]
async fn resume_latest_conversation(
    runtime: State<'_, AppRuntime>,
) -> std::result::Result<ResumeConversationPayload, String> {
    let store = open_conversation_store(runtime.inner())?;
    let summary = store
        .list(None, 1)
        .map_err(|error| error.to_string())?
        .into_iter()
        .next()
        .ok_or_else(|| "no saved conversation is available".to_string())?;
    resume_conversation(summary.id, runtime).await
}

#[tauri::command]
async fn stop_command(
    task_id: String,
    runtime: State<'_, AppRuntime>,
) -> std::result::Result<serde_json::Value, String> {
    let manager = runtime
        .command_manager
        .lock()
        .clone()
        .ok_or_else(|| "no active conversation command manager".to_string())?;
    manager
        .kill(&task_id, "user_stop")
        .await
        .map(|snapshot| pok_ai_core::commands::snapshot_json(&snapshot))
        .map_err(|error| error.to_string())
}

#[tauri::command]
async fn read_command_output(
    task_id: String,
    offset: u64,
    max_bytes: usize,
    runtime: State<'_, AppRuntime>,
) -> std::result::Result<serde_json::Value, String> {
    let manager = runtime
        .command_manager
        .lock()
        .clone()
        .ok_or_else(|| "no active conversation command manager".to_string())?;
    let (output, next_offset, truncated) = manager
        .read(&task_id, offset, max_bytes)
        .await
        .map_err(|error| error.to_string())?;
    Ok(serde_json::json!({"output": output, "next_offset": next_offset, "truncated": truncated}))
}

#[tauri::command]
async fn compact_context(runtime: State<'_, AppRuntime>) -> std::result::Result<(), String> {
    let mut conversation = runtime.conversation.lock().await;
    let session = conversation
        .as_mut()
        .ok_or_else(|| "start a conversation before compacting context".to_string())?;
    session.request_compaction();
    Ok(())
}

#[tauri::command]
async fn run_prompt(
    prompt: String,
    mut workspace: String,
    mut model: String,
    mut provider: String,
    policy_mode: Option<String>,
    temperature: Option<f32>,
    max_output_tokens: Option<u32>,
    seed: Option<u64>,
    reasoning_effort: Option<String>,
    image_input_override: Option<bool>,
    full_context: Option<bool>,
    decision_router_enabled: Option<bool>,
    decision_router_backend: Option<DecisionRouterBackend>,
    images: Option<Vec<String>>,
    files: Option<Vec<String>>,
    app: tauri::AppHandle,
    runtime: State<'_, AppRuntime>,
    broker: State<'_, Arc<ApprovalBroker>>,
    question_broker: State<'_, Arc<QuestionBroker>>,
) -> std::result::Result<String, String> {
    let cancellation = CancellationToken::new();
    *runtime.cancellation.lock() = cancellation.clone();
    let conversation_store = open_conversation_store(runtime.inner())?;
    let pending_snapshot = if let Some(id) = *runtime.pending_resume.lock().await {
        Some(
            conversation_store
                .load(id)
                .map_err(|error| error.to_string())?
                .ok_or_else(|| format!("conversation {id} was not found"))?,
        )
    } else {
        None
    };
    if let Some(snapshot) = &pending_snapshot {
        provider = snapshot.summary.provider.clone();
        model = snapshot.summary.model.clone();
        workspace = snapshot.summary.workspace.display().to_string();
    }
    let (
        provider_config,
        data_dir,
        diagnostics_dir,
        command_timeout_seconds,
        input_text_inter_key_pause_ms,
        vision_max_edge,
        visual_history_limit,
        uia_element_limit,
        desktop_enrichment_timeout_ms,
        desktop_deep_enrichment_timeout_ms,
        background_desktop_work,
        model_target_limit,
        fusion_iou_threshold,
        ocr_containment_threshold,
        annotate_targets,
        max_turns,
        agent_temperature,
        mut decision_router_config,
    ) = {
        let config = runtime.config.lock();
        if config.prompt_token_target.is_some() {
            eprintln!(
                "Warning: prompt_token_target is deprecated and ignored; POK-Ai now sizes the model working set automatically."
            );
        }
        (
            config
                .provider(Some(&provider))
                .cloned()
                .map_err(|error| error.to_string())?,
            config.data_dir.clone(),
            config.diagnostics_dir.clone(),
            config.command_timeout_seconds,
            config.input_text_inter_key_pause_ms,
            config.vision_max_edge,
            config.visual_history_limit,
            config.uia_element_limit,
            config.desktop_enrichment_timeout_ms,
            config.desktop_deep_enrichment_timeout_ms,
            config.background_desktop_work,
            config.model_target_limit,
            config.fusion_iou_threshold,
            config.ocr_containment_threshold,
            config.annotate_targets,
            config.max_turns,
            config.agent_temperature,
            config.decision_router.clone(),
        )
    };
    let legacy_requested_decision_router = pending_snapshot
        .as_ref()
        .and_then(|snapshot| snapshot.state.get("decision_router_enabled"))
        .and_then(serde_json::Value::as_bool)
        .or(decision_router_enabled)
        .unwrap_or(decision_router_config.enabled);
    let requested_backend = pending_snapshot
        .as_ref()
        .and_then(|snapshot| snapshot.state.get("decision_router_backend"))
        .and_then(|value| serde_json::from_value(value.clone()).ok())
        .or(decision_router_backend)
        .unwrap_or_else(|| {
            if legacy_requested_decision_router {
                decision_router_config.backend
            } else {
                DecisionRouterBackend::Off
            }
        });
    decision_router_config.backend = requested_backend;
    decision_router_config.enabled = requested_backend != DecisionRouterBackend::Off;
    if requested_backend == DecisionRouterBackend::Jev
        && pok_ai_core::brain::get_api_key(&decision_router_config.api_key_env)
            .is_none_or(|key| key.trim().is_empty())
    {
        return Err("JEV is selected but its TypeSafe API key is unavailable.".into());
    }
    if requested_backend == DecisionRouterBackend::Laya {
        let preference = decision_router_config.laya.device;
        let (endpoint, _) = ensure_laya_started(&app, runtime.inner(), preference).await?;
        decision_router_config.laya.endpoint = endpoint;
        // The judge only ever activates alongside Laya; start it here too
        // whenever the user has previously enabled it, so it's ready before
        // the session's first decision rather than starting cold on first
        // use. A failure here is non-fatal: judge_candidate's default trait
        // implementation is inert (Ok(false)) when unreachable, so Laya's
        // own combined gate still governs eligibility either way.
        if decision_router_config.judge.enabled {
            match ensure_judge_started(&app, runtime.inner()).await {
                Ok((endpoint, _)) => decision_router_config.judge.endpoint = endpoint,
                Err(error) => {
                    let _ = app.emit(
                        "judge_runtime_event",
                        serde_json::json!({
                            "phase": "error", "detail": &error, "installed": true,
                            "running": false, "enabled": true
                        }),
                    );
                }
            }
        }
    } else {
        stop_laya(runtime.inner()).await;
        stop_judge(runtime.inner()).await;
    }
    let effective_decision_router_backend = requested_backend;
    let workspace = if workspace.trim().is_empty() {
        std::env::current_dir()
    } else {
        Ok(PathBuf::from(workspace))
    }
    .map_err(|error| error.to_string())?;
    let workspace = dunce::canonicalize(workspace).map_err(|error| error.to_string())?;
    if provider_config.resolved_data_boundary(&provider) == ProviderDataBoundary::ExternalService {
        let acknowledgment_key = format!(
            "{}\n{}\n{}",
            provider.to_ascii_lowercase(),
            model.to_ascii_lowercase(),
            workspace.display()
        );
        let acknowledged = runtime
            .external_provider_acknowledgments
            .lock()
            .await
            .contains(&acknowledgment_key);
        if !acknowledged {
            let decision = broker
                .approve(
                    "external_provider_data",
                    &serde_json::json!({
                        "provider": provider,
                        "model": model,
                        "endpoint": provider_config.base_url,
                        "data_boundary": "external_service",
                        "context": [
                            "prompts",
                            "workspace paths and tool results",
                            "screenshots and OCR",
                            "document contents"
                        ],
                    }),
                    "This provider is an external service. Full task, desktop, workspace, and document context may leave this device. Confirm before POK-Ai sends an inference request.",
                )
                .await
                .map_err(|error| error.to_string())?;
            if decision == ApprovalDecision::Deny {
                return Err("external inference cancelled before task context was sent".into());
            }
            runtime
                .external_provider_acknowledgments
                .lock()
                .await
                .insert(acknowledgment_key);
        }
    }
    if decision_router_config.enabled && requested_backend == DecisionRouterBackend::Jev {
        let acknowledgment_key = format!(
            "typesafe-decision-router-v2-code-previews\n{}\n{}",
            decision_router_config.model,
            workspace.display()
        );
        let acknowledged = runtime
            .external_provider_acknowledgments
            .lock()
            .await
            .contains(&acknowledgment_key);
        if !acknowledged {
            let decision = broker
                .approve(
                    "external_decision_router_data",
                    &serde_json::json!({
                        "provider": "TypeSafe",
                        "model": decision_router_config.model,
                        "endpoint": decision_router_config.endpoint,
                        "data_boundary": "external_service",
                        "context": ["active task", "current task step", "workspace paths and compact UI labels", "bounded related memory snippets", "bounded locally shortlisted source-code previews with line ranges"],
                    }),
                    "The optional Jev decision router is an external service. Compact task/UI metadata, bounded related memory, and locally shortlisted source-code previews may leave this device. Screenshots, full OCR, unrestricted history, ignored files, and detected secrets remain excluded. Confirm before sending the first request.",
                )
                .await
                .map_err(|error| error.to_string())?;
            if decision == ApprovalDecision::Deny {
                return Err(
                    "external decision routing cancelled before task context was sent".into(),
                );
            }
            runtime
                .external_provider_acknowledgments
                .lock()
                .await
                .insert(acknowledgment_key);
        }
    }
    if local_lifecycle_supported(&provider) {
        let _transition = runtime.model_transition.lock().await;
        tokio::select! {
            () = cancellation.cancelled() => return Err(pok_ai_core::PokError::Cancelled.to_string()),
            result = load_model_locked(
                &app,
                runtime.inner(),
                &provider,
                &model,
                &provider_config,
            ) => result?,
        };
    }
    let brain = from_provider(&provider_config).map_err(|error| error.to_string())?;
    let requested_policy = if policy_mode.as_deref() == Some("autonomous") {
        Policy::autonomous()
    } else {
        Policy::interactive()
    };
    runtime.user_guidance_queue.lock().clear();
    let mut conversation = runtime.conversation.lock().await;
    let mut conversation_provider = runtime.conversation_provider.lock().await;
    let mut conversation_decision_router_backend =
        runtime.conversation_decision_router_backend.lock().await;
    let mut conversation_judge_enabled = runtime.conversation_judge_enabled.lock().await;
    let should_create = conversation
        .as_ref()
        .is_none_or(|session| !session.matches_conversation(&model, &workspace))
        || conversation_provider.as_deref() != Some(provider.as_str())
        || conversation_decision_router_backend.as_ref()
            != Some(&effective_decision_router_backend)
        // Toggling the cross-model judge alone (same provider/model/workspace/
        // backend) would otherwise reuse the already-constructed Session, whose
        // TypeSafeDecisionRouter baked in a snapshot of decision_router_config
        // at creation time -- the judge.enabled flip would silently never reach
        // the live router until something else forced a new session.
        || conversation_judge_enabled.as_ref() != Some(&decision_router_config.judge.enabled);
    if should_create {
        let session_id = pending_snapshot
            .as_ref()
            .map_or_else(Uuid::new_v4, |snapshot| snapshot.summary.id);
        let memory =
            MemoryStore::open(data_dir.join("memory")).map_err(|error| error.to_string())?;
        let mut tools = ToolRegistry::new();
        register_coding_tools(&mut tools);
        register_desktop_tools(&mut tools);
        register_memory_tools(&mut tools);
        register_generated_tool_tools(&mut tools);
        tools.register(SpawnSubagentTool::new(brain.clone(), &model));
        let provider_models = tokio::select! {
            () = cancellation.cancelled() => return Err(pok_ai_core::PokError::Cancelled.to_string()),
            result = provider_model_catalog(runtime.inner(), &provider, &provider_config) => result,
        };
        let provider_contexts = provider_models.ok().and_then(|models| {
            models
                .into_iter()
                .find(|item| item.id.eq_ignore_ascii_case(&model))
                .map(|item| (item.loaded_context_length, item.max_context_length))
        });
        let context_budget = resolve_context_budget(
            &data_dir,
            &provider,
            &model,
            provider_contexts.and_then(|value| value.0),
            provider_contexts.and_then(|value| value.1),
            32_768,
        )
        .await;
        let artifact_dir = pending_snapshot.as_ref().map_or_else(
            || {
                diagnostics_dir
                    .join("sessions")
                    .join(session_id.to_string())
            },
            |snapshot| snapshot.summary.artifact_dir.clone(),
        );
        let command_manager = Arc::new(pok_ai_core::commands::CommandManager::new(
            artifact_dir.clone(),
        ));
        *runtime.command_manager.lock() = Some((*command_manager).clone());
        let context = ToolContext {
            session_id,
            workspace: workspace.clone(),
            data_dir,
            artifact_dir: artifact_dir.clone(),
            policy: requested_policy.clone(),
            approvals: broker.inner().clone(),
            platform: pok_ai_windows::desktop_platform(background_desktop_work),
            memory,
            session_archive: pok_ai_core::session_archive::SessionArchive::open(&artifact_dir)
                .map_err(|error| error.to_string())?,
            cancellation: cancellation.clone(),
            pause: runtime.pause.clone(),
            latest_observation: Default::default(),
            latest_observation_view: Default::default(),
            pending_visual_localization: Default::default(),
            task_hint: Default::default(),
            input_ledger: Default::default(),
            artifact_evidence: Default::default(),
            attached_paths: Default::default(),
            session_files: Default::default(),
            active_task: Default::default(),
            focused_control: Default::default(),
            command_timeout_seconds,
            input_text_inter_key_pause_ms,
            vision_max_edge,
            prompt_token_target: context_budget.compact_at_tokens,
            context_budget: Arc::new(SyncMutex::new(context_budget)),
            visual_history_limit,
            uia_element_limit,
            desktop_enrichment_timeout_ms,
            desktop_deep_enrichment_timeout_ms,
            model_target_limit,
            fusion_iou_threshold,
            ocr_containment_threshold,
            annotate_targets,
            approval_cache: Default::default(),
            user_guidance_queue: runtime.user_guidance_queue.clone(),
            questions: Some(question_broker.inner().clone()),
            command_manager,
            current_tool_call_id: Default::default(),
        };
        let judge_enabled_for_conversation = decision_router_config.judge.enabled;
        let mut session = Session::new(brain, Arc::new(tools), context, model.clone(), max_turns)
            .map_err(|error| error.to_string())?
            .with_conversation_store(conversation_store, provider.clone())
            .map_err(|error| error.to_string())?
            .with_decision_router(
                pok_ai_core::decision::TypeSafeDecisionRouter::from_config(&decision_router_config)
                    .map_err(|error| error.to_string())?,
                decision_router_config,
            );
        if let Some(snapshot) = pending_snapshot {
            session
                .restore_conversation(snapshot)
                .map_err(|error| error.to_string())?;
            *runtime.pending_resume.lock().await = None;
        }
        // One lock, released before building: two `config.lock()` guards in
        // one statement deadlock (the first lives until the statement ends).
        let (action_step_budget, standing_instructions) = {
            let config = runtime.config.lock();
            (
                config.action_step_budget,
                config.standing_instructions.clone(),
            )
        };
        *conversation = Some(
            session
                .with_temperature(agent_temperature.value())
                .with_action_step_budget(action_step_budget)
                .with_standing_instructions(standing_instructions.as_deref())
                .with_observer(Arc::new(TauriSessionObserver::new(app))),
        );
        *conversation_provider = Some(provider);
        *conversation_decision_router_backend = Some(effective_decision_router_backend);
        *conversation_judge_enabled = Some(judge_enabled_for_conversation);
    }
    let session = conversation.as_mut().expect("conversation was created");
    let training_log = runtime.config.lock().decision_router.training_log;
    session.set_training_log(training_log);
    session.configure_model_request(
        temperature.or_else(|| agent_temperature.value()),
        max_output_tokens,
        seed,
        reasoning_effort,
        image_input_override,
        full_context.unwrap_or(false),
    );
    session.set_policy_mode(match requested_policy.mode {
        PolicyMode::Autonomous => PolicyMode::Autonomous,
        PolicyMode::Interactive => PolicyMode::Interactive,
        PolicyMode::Exam { .. } => unreachable!("desktop policy cannot request exam mode"),
    });
    session.prepare_follow_up(cancellation);
    session
        .run_with_attachments(
            prompt,
            images.unwrap_or_default(),
            files.unwrap_or_default(),
        )
        .await
        .map(|summary| summary.answer)
        .map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_endpoint_override_survives_config_reload() {
        let directory = tempfile::tempdir().unwrap();
        let mut config = Config::default();
        config.data_dir = directory.path().to_path_buf();

        persist_provider_endpoint_override(
            &config.data_dir,
            "lm_studio",
            "http://10.0.0.25:1234/v1",
        )
        .unwrap();

        let mut reloaded = Config::default();
        reloaded.data_dir = directory.path().to_path_buf();
        apply_provider_endpoint_overrides(&mut reloaded).unwrap();
        assert_eq!(
            reloaded.providers["lm_studio"].base_url,
            "http://10.0.0.25:1234/v1"
        );
    }

    #[test]
    fn invalid_saved_provider_endpoint_is_rejected() {
        let directory = tempfile::tempdir().unwrap();
        let overrides = ProviderEndpointOverrides {
            endpoints: BTreeMap::from([("lm_studio".into(), "file:///tmp/model".into())]),
        };
        atomic_write(
            &provider_endpoint_overrides_path(directory.path()),
            &serde_json::to_vec(&overrides).unwrap(),
        )
        .unwrap();

        let mut config = Config::default();
        config.data_dir = directory.path().to_path_buf();
        assert!(apply_provider_endpoint_overrides(&mut config).is_err());
        assert_eq!(
            config.providers["lm_studio"].base_url,
            "http://127.0.0.1:1234/v1"
        );
    }

    #[test]
    fn stream_deltas_are_coalesced_without_losing_text() {
        let mut coalescer = EventCoalescer::default();
        for _ in 0..10_000 {
            assert!(
                coalescer
                    .push(AgentEvent::TextDelta { text: "x".into() })
                    .is_empty()
            );
        }
        let output = coalescer.flush();
        assert_eq!(output.len(), 1);
        let AgentEvent::TextDelta { text } = &output[0] else {
            panic!("expected a text delta");
        };
        assert_eq!(text.len(), 10_000);
    }

    #[test]
    fn event_order_is_preserved_and_terminal_events_flush_pending_text() {
        let mut coalescer = EventCoalescer::default();
        assert!(
            coalescer
                .push(AgentEvent::ReasoningDelta { text: "why".into() })
                .is_empty()
        );
        let switched = coalescer.push(AgentEvent::TextDelta {
            text: "answer".into(),
        });
        assert!(matches!(
            switched.as_slice(),
            [AgentEvent::ReasoningDelta { text }] if text == "why"
        ));

        let finished = coalescer.push(AgentEvent::RunCompleted {
            answer: "answer".into(),
            completion_status: pok_ai_core::types::CompletionStatus::Completed,
            warnings: Vec::new(),
            deliverables: Vec::new(),
        });
        assert!(matches!(
            finished.as_slice(),
            [AgentEvent::TextDelta { text }, AgentEvent::RunCompleted { answer, .. }]
                if text == "answer" && answer == "answer"
        ));
    }

    #[test]
    fn compression_events_flush_stream_text_and_preserve_phase_order() {
        let mut coalescer = EventCoalescer::default();
        assert!(
            coalescer
                .push(AgentEvent::TextDelta {
                    text: "partial".into(),
                })
                .is_empty()
        );

        let started = coalescer.push(AgentEvent::CompressionStarted {
            sequence: 2,
            trigger: "automatic".into(),
            transcript_chars: 9_253,
        });
        assert!(matches!(
            started.as_slice(),
            [
                AgentEvent::TextDelta { text },
                AgentEvent::CompressionStarted { sequence: 2, .. }
            ] if text == "partial"
        ));

        let completed = coalescer.push(AgentEvent::CompressionCompleted {
            sequence: 2,
            elapsed_ms: 2_929,
            summary_source: "model_retry".into(),
            attempts: 2,
            prompt_tokens: 4_000,
            completion_tokens: 1_200,
            reasoning_chars: 2_400,
        });
        assert!(matches!(
            completed.as_slice(),
            [AgentEvent::CompressionCompleted {
                sequence: 2,
                elapsed_ms: 2_929,
                ..
            }]
        ));
        let turn = coalescer.push(AgentEvent::TurnStarted { turn: 24 });
        assert!(matches!(
            turn.as_slice(),
            [AgentEvent::TurnStarted { turn: 24 }]
        ));
    }
}

#[tauri::command]
async fn unload_model(
    provider: String,
    model: String,
    app: tauri::AppHandle,
    runtime: State<'_, AppRuntime>,
) -> std::result::Result<LocalModelRuntimeStatus, String> {
    let provider_config = provider_config(runtime.inner(), &provider)?;
    let _transition = runtime.model_transition.lock().await;
    unload_model_locked(&app, runtime.inner(), &provider, &model, &provider_config).await
}

#[tauri::command]
fn send_guidance(
    message: String,
    runtime: State<'_, AppRuntime>,
) -> std::result::Result<(), String> {
    runtime.user_guidance_queue.lock().push_back(message);
    Ok(())
}

#[tauri::command]
fn list_providers(runtime: State<'_, AppRuntime>) -> Vec<String> {
    runtime.config.lock().providers.keys().cloned().collect()
}

#[tauri::command]
#[allow(non_snake_case)]
fn save_provider_key(
    provider: String,
    apiKey: String,
    runtime: State<'_, AppRuntime>,
) -> std::result::Result<(), String> {
    let env_name = {
        let config = runtime.config.lock();
        config
            .provider(Some(&provider))
            .map_err(|error| error.to_string())?
            .api_key_env
            .as_deref()
            .map(|s| s.to_string())
            .ok_or_else(|| format!("provider {provider:?} does not use an API key"))?
    };
    let api_key = apiKey.trim();
    if api_key.is_empty() {
        return Err("API key cannot be empty".into());
    }
    persist_provider_key(&env_name, api_key)?;
    pok_ai_core::brain::set_api_key(&env_name, api_key);
    Ok(())
}

#[derive(Serialize)]
struct DecisionRouterStatus {
    enabled: bool,
    backend: DecisionRouterBackend,
    model: String,
    has_api_key: bool,
    laya: LayaRuntimeStatus,
    judge: JudgeRuntimeStatus,
    /// The opt-in router training log is on, and where its files go.
    training_log: bool,
    training_dir: String,
}

fn judge_root(runtime: &AppRuntime) -> PathBuf {
    runtime.config.lock().data_dir.join("judge")
}

/// Expected revision of the installed judge checkpoint. Must stay in sync
/// with the default `-Revision` of scripts/setup-zeiger.ps1: pre-alpha Hub
/// `main` weights regressed before (zeiger "Round 15" collapsed judge
/// correlation from 90% to 40%), so an existing install whose manifest
/// revision differs must be reinstalled rather than trusted silently.
/// See docs/decision-router-backends.md.
const JUDGE_PINNED_REVISION: &str = "441dcf64aef0";

/// The revision `scripts/setup-laya.ps1` installs by default. Informational
/// for now (the picker weights have not drifted); passed through so the
/// manifest is populated for future enforcement.
const LAYA_PINNED_REVISION: &str = "843893f92cf9";

/// The revision recorded in `<root>/manifest.json` by the setup scripts, if
/// the manifest exists and carries one. Older installs (before the scripts
/// recorded revisions) return `None`.
fn installed_revision(root: &Path) -> Option<String> {
    let manifest = std::fs::read_to_string(root.join("manifest.json")).ok()?;
    let value: serde_json::Value = serde_json::from_str(&manifest).ok()?;
    value
        .get("revision")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
}

/// The installed judge is usable only when its manifest revision matches the
/// pin; a missing or different revision means stale weights and a reinstall.
fn judge_revision_matches(runtime: &AppRuntime) -> bool {
    installed_revision(&judge_root(runtime)).as_deref() == Some(JUDGE_PINNED_REVISION)
}

fn judge_checkpoint(runtime: &AppRuntime) -> PathBuf {
    judge_root(runtime).join("models").join("zeiger-0.6b")
}

fn judge_python(runtime: &AppRuntime) -> PathBuf {
    let runtime_dir = judge_root(runtime).join("runtime");
    if cfg!(windows) {
        runtime_dir.join("Scripts").join("python.exe")
    } else {
        runtime_dir.join("bin").join("python")
    }
}

fn judge_package_dir(runtime: &AppRuntime) -> PathBuf {
    judge_root(runtime).join("zeiger")
}

fn judge_installed(runtime: &AppRuntime) -> bool {
    judge_python(runtime).is_file()
        && judge_checkpoint(runtime)
            .join("model.safetensors")
            .is_file()
        && judge_checkpoint(runtime)
            .join("rl_agent_config.json")
            .is_file()
        && judge_package_dir(runtime)
            .join("zeiger")
            .join("__init__.py")
            .is_file()
}

fn laya_root(runtime: &AppRuntime) -> PathBuf {
    runtime.config.lock().data_dir.join("laya")
}

fn laya_checkpoint(runtime: &AppRuntime) -> PathBuf {
    let models = laya_root(runtime).join("models");
    let typed = models.join("typed-decisions");
    if typed.join("model.safetensors").exists() {
        typed
    } else {
        models.join("english")
    }
}

fn laya_python(runtime: &AppRuntime) -> PathBuf {
    let runtime_dir = laya_root(runtime).join("runtime");
    if cfg!(windows) {
        runtime_dir.join("Scripts").join("python.exe")
    } else {
        runtime_dir.join("bin").join("python")
    }
}

fn bundled_script(app: &tauri::AppHandle, name: &str) -> std::result::Result<PathBuf, String> {
    let resource = app
        .path()
        .resource_dir()
        .map_err(|error| format!("could not locate application resources: {error}"))?
        .join("scripts")
        .join(name);
    if resource.is_file() {
        return Ok(resource);
    }
    let development = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .join("scripts")
        .join(name);
    if development.is_file() {
        Ok(development)
    } else {
        Err(format!("bundled Laya script {name} was not found"))
    }
}

fn laya_installed(runtime: &AppRuntime) -> bool {
    laya_python(runtime).is_file()
        && laya_checkpoint(runtime).join("model.safetensors").is_file()
        && laya_checkpoint(runtime)
            .join("rl_agent_config.json")
            .is_file()
}

async fn laya_runtime_status(runtime: &AppRuntime) -> LayaRuntimeStatus {
    let mut guard = runtime.laya_process.lock().await;
    let running = if let Some(process) = guard.as_mut() {
        process.child.try_wait().ok().flatten().is_none()
    } else {
        false
    };
    if !running {
        *guard = None;
    }
    let process = guard.as_ref();
    LayaRuntimeStatus {
        installed: laya_installed(runtime),
        running,
        phase: if running {
            "ready"
        } else if laya_installed(runtime) {
            "stopped"
        } else {
            "not_installed"
        }
        .into(),
        detail: if running {
            "Laya is ready on this device."
        } else if laya_installed(runtime) {
            "Laya is installed and starts when selected."
        } else {
            "Install the local Laya router to use it."
        }
        .into(),
        model: "laya".into(),
        checkpoint: "convaiinnovations/laya-typed-decisions (Apache-2.0)".into(),
        endpoint: process.map(|value| value.endpoint.clone()),
        device: process.and_then(|value| value.device.clone()),
        device_reason: process.and_then(|value| value.device_reason.clone()),
        preference: runtime.config.lock().decision_router.laya.device,
    }
}

async fn stop_laya(runtime: &AppRuntime) {
    if let Some(mut process) = runtime.laya_process.lock().await.take() {
        let _ = Box::into_pin(process.child.kill()).await;
        let _ = process.child.wait().await;
    }
}

async fn ensure_laya_started(
    app: &tauri::AppHandle,
    runtime: &AppRuntime,
    preference: LayaDevicePreference,
) -> std::result::Result<(String, String), String> {
    {
        let mut guard = runtime.laya_process.lock().await;
        if let Some(process) = guard.as_mut()
            && process
                .child
                .try_wait()
                .map_err(|error| error.to_string())?
                .is_none()
            && process.preference == preference
        {
            let token_env = runtime.config.lock().decision_router.laya.token_env.clone();
            if let Some(token) = pok_ai_core::brain::get_api_key(&token_env) {
                return Ok((process.endpoint.clone(), token));
            }
        }
        *guard = None;
    }
    if !laya_installed(runtime) {
        return Err(
            "Laya is not installed. Open Settings and install the local router first.".into(),
        );
    }
    let listener = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
        .map_err(|error| format!("could not reserve a Laya port: {error}"))?;
    let port = listener
        .local_addr()
        .map_err(|error| error.to_string())?
        .port();
    drop(listener);
    let endpoint = format!("http://127.0.0.1:{port}/v1/systemone");
    let token = format!("{}{}", Uuid::new_v4().simple(), Uuid::new_v4().simple());
    let settings = runtime.config.lock().decision_router.laya.clone();
    let script = bundled_script(app, "laya_sidecar.py")?;
    let checkpoint = laya_checkpoint(runtime);
    let device_arg = match preference {
        LayaDevicePreference::Auto => "auto",
        LayaDevicePreference::Gpu => "gpu",
        LayaDevicePreference::Cpu => "cpu",
    };
    let mut command = CommandWrap::with_new(laya_python(runtime), |command| {
        command
            .arg(&script)
            .arg("--model-dir")
            .arg(&checkpoint)
            .arg("--port")
            .arg(port.to_string())
            .arg("--min-free-vram-mb")
            .arg(settings.min_free_vram_mb.to_string())
            .arg("--device")
            .arg(device_arg)
            .env(&settings.token_env, &token)
            .env("POK_LAYA_TOKEN", &token)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
    });
    command.wrap(KillOnDrop);
    // The job object is the real fix for orphaned GPU-resident processes: it
    // guarantees the OS kills this process (and anything it spawns) when the
    // job handle closes, which Windows does automatically on ANY process
    // exit -- including a crash or a force-kill via Task Manager, where
    // Rust's own Drop impls never get a chance to run at all. Confirmed
    // necessary in practice: an orphaned Laya process survived 8+ hours
    // after an ungraceful shutdown before this fix (see LayaProcess::drop).
    #[cfg(windows)]
    {
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.wrap(CreationFlags(PROCESS_CREATION_FLAGS(CREATE_NO_WINDOW)));
        command.wrap(JobObject);
    }
    let mut child = command
        .spawn()
        .map_err(|error| format!("could not start Laya: {error}"))?;
    let health = format!("http://127.0.0.1:{port}/health");
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(2))
        .build()
        .map_err(|error| error.to_string())?;
    let started = tokio::time::Instant::now();
    let health_payload = loop {
        if let Some(status) = child.try_wait().map_err(|error| error.to_string())? {
            return Err(format!("Laya exited during startup with {status}"));
        }
        if let Ok(response) = client.get(&health).bearer_auth(&token).send().await
            && response.status().is_success()
        {
            break response
                .json::<serde_json::Value>()
                .await
                .unwrap_or_default();
        }
        if started.elapsed() > Duration::from_secs(120) {
            let _ = Box::into_pin(child.kill()).await;
            return Err("Laya did not become ready within 120 seconds.".into());
        }
        tokio::time::sleep(Duration::from_millis(400)).await;
    };
    let device = health_payload
        .get("device")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    let device_reason = health_payload
        .get("device_reason")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    if preference == LayaDevicePreference::Gpu
        && device
            .as_deref()
            .is_none_or(|value| !value.starts_with("cuda"))
    {
        let _ = Box::into_pin(child.kill()).await;
        let reason = device_reason.as_deref().unwrap_or("CUDA was not available");
        return Err(format!(
            "Laya GPU mode could not start: {reason}. Use Repair/update GPU runtime, or select Auto/CPU."
        ));
    }
    pok_ai_core::brain::set_api_key(&settings.token_env, token.clone());
    runtime.laya_process.lock().await.replace(LayaProcess {
        child,
        endpoint: endpoint.clone(),
        device,
        device_reason,
        preference,
    });
    Ok((endpoint, token))
}

async fn judge_runtime_status(runtime: &AppRuntime) -> JudgeRuntimeStatus {
    let mut guard = runtime.judge_process.lock().await;
    let running = if let Some(process) = guard.as_mut() {
        process.child.try_wait().ok().flatten().is_none()
    } else {
        false
    };
    if !running {
        *guard = None;
    }
    let process = guard.as_ref();
    let enabled = runtime.config.lock().decision_router.judge.enabled;
    let installed = judge_installed(runtime);
    let revision_matches = judge_revision_matches(runtime);
    JudgeRuntimeStatus {
        installed,
        enabled,
        running,
        phase: if running {
            "ready"
        } else if installed {
            "stopped"
        } else {
            "not_installed"
        }
        .into(),
        detail: if running {
            "The cross-model judge is ready on this device.".to_string()
        } else if installed && !revision_matches {
            format!(
                "The judge model is installed but its revision does not match the pinned {}; reinstall it to restore validated behavior.",
                JUDGE_PINNED_REVISION
            )
        } else if installed {
            "The judge model is installed and starts when enabled.".to_string()
        } else {
            "Install the local judge model to enable cross-model checking of Laya's uncertain picks.".to_string()
        },
        model: "zeiger".into(),
        checkpoint: "php-ai/zeiger-0.6b".into(),
        endpoint: process.map(|value| value.endpoint.clone()),
        device: process.and_then(|value| value.device.clone()),
        device_reason: process.and_then(|value| value.device_reason.clone()),
    }
}

async fn stop_judge(runtime: &AppRuntime) {
    if let Some(mut process) = runtime.judge_process.lock().await.take() {
        let _ = Box::into_pin(process.child.kill()).await;
        let _ = process.child.wait().await;
    }
}

async fn ensure_judge_started(
    app: &tauri::AppHandle,
    runtime: &AppRuntime,
) -> std::result::Result<(String, String), String> {
    {
        let mut guard = runtime.judge_process.lock().await;
        if let Some(process) = guard.as_mut()
            && process
                .child
                .try_wait()
                .map_err(|error| error.to_string())?
                .is_none()
        {
            let token_env = runtime
                .config
                .lock()
                .decision_router
                .judge
                .token_env
                .clone();
            if let Some(token) = pok_ai_core::brain::get_api_key(&token_env) {
                return Ok((process.endpoint.clone(), token));
            }
        }
        *guard = None;
    }
    if !judge_installed(runtime) {
        return Err("The judge model is not installed. Open Settings and install it first.".into());
    }
    let listener = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
        .map_err(|error| format!("could not reserve a judge-model port: {error}"))?;
    let port = listener
        .local_addr()
        .map_err(|error| error.to_string())?
        .port();
    drop(listener);
    let endpoint = format!("http://127.0.0.1:{port}/v1/systemone");
    let token = format!("{}{}", Uuid::new_v4().simple(), Uuid::new_v4().simple());
    let token_env = runtime
        .config
        .lock()
        .decision_router
        .judge
        .token_env
        .clone();
    let script = bundled_script(app, "zeiger_sidecar.py")?;
    let checkpoint = judge_checkpoint(runtime);
    let package_dir = judge_package_dir(runtime);
    let mut command = CommandWrap::with_new(judge_python(runtime), |command| {
        command
            .arg(&script)
            .arg("--model-dir")
            .arg(&checkpoint)
            .arg("--port")
            .arg(port.to_string())
            .arg("--device")
            .arg("auto")
            .env("PYTHONPATH", &package_dir)
            .env(&token_env, &token)
            .env("POK_JUDGE_TOKEN", &token)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
    });
    command.wrap(KillOnDrop);
    // See ensure_laya_started's identical comment: the Job Object is what
    // actually prevents this GPU-resident process from surviving a crash or
    // force-kill of the app.
    #[cfg(windows)]
    {
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.wrap(CreationFlags(PROCESS_CREATION_FLAGS(CREATE_NO_WINDOW)));
        command.wrap(JobObject);
    }
    let mut child = command
        .spawn()
        .map_err(|error| format!("could not start the judge model: {error}"))?;
    let health = format!("http://127.0.0.1:{port}/health");
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(2))
        .build()
        .map_err(|error| error.to_string())?;
    let started = tokio::time::Instant::now();
    let health_payload = loop {
        if let Some(status) = child.try_wait().map_err(|error| error.to_string())? {
            return Err(format!("Judge model exited during startup with {status}"));
        }
        if let Ok(response) = client.get(&health).bearer_auth(&token).send().await
            && response.status().is_success()
        {
            break response
                .json::<serde_json::Value>()
                .await
                .unwrap_or_default();
        }
        if started.elapsed() > Duration::from_secs(120) {
            let _ = Box::into_pin(child.kill()).await;
            return Err("Judge model did not become ready within 120 seconds.".into());
        }
        tokio::time::sleep(Duration::from_millis(400)).await;
    };
    let device = health_payload
        .get("device")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    let device_reason = health_payload
        .get("device_reason")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    pok_ai_core::brain::set_api_key(&token_env, token.clone());
    runtime.judge_process.lock().await.replace(JudgeProcess {
        child,
        endpoint: endpoint.clone(),
        device,
        device_reason,
    });
    Ok((endpoint, token))
}

async fn decision_router_status(runtime: &AppRuntime) -> DecisionRouterStatus {
    let config = runtime.config.lock().decision_router.clone();
    let has_api_key = pok_ai_core::brain::get_api_key(&config.api_key_env)
        .is_some_and(|key| !key.trim().is_empty());
    DecisionRouterStatus {
        enabled: config.enabled
            && config.backend != DecisionRouterBackend::Off
            && match config.backend {
                DecisionRouterBackend::Jev => has_api_key,
                DecisionRouterBackend::Laya => laya_installed(runtime),
                // The LLM-choice and kev servers are started and managed
                // outside the app.
                DecisionRouterBackend::LlmChoice | DecisionRouterBackend::Kev => true,
                DecisionRouterBackend::Off => false,
            },
        backend: if config.enabled {
            config.backend
        } else {
            DecisionRouterBackend::Off
        },
        model: config.active_model().to_owned(),
        has_api_key,
        laya: laya_runtime_status(runtime).await,
        judge: judge_runtime_status(runtime).await,
        training_log: config.training_log,
        training_dir: runtime
            .config
            .lock()
            .data_dir
            .join("router-training")
            .display()
            .to_string(),
    }
}

#[tauri::command]
async fn get_decision_router_status(
    runtime: State<'_, AppRuntime>,
) -> std::result::Result<DecisionRouterStatus, String> {
    Ok(decision_router_status(runtime.inner()).await)
}

#[tauri::command]
async fn set_decision_router_enabled(
    enabled: bool,
    runtime: State<'_, AppRuntime>,
) -> std::result::Result<DecisionRouterStatus, String> {
    if enabled {
        let api_key_env = runtime.config.lock().decision_router.api_key_env.clone();
        let has_api_key =
            pok_ai_core::brain::get_api_key(&api_key_env).is_some_and(|key| !key.trim().is_empty());
        if !has_api_key {
            return Err(format!(
                "JEV cannot be enabled until a TypeSafe API key is saved ({api_key_env})."
            ));
        }
    }
    {
        let mut config = runtime.config.lock();
        config.decision_router.enabled = enabled;
        config.decision_router.backend = if enabled {
            DecisionRouterBackend::Jev
        } else {
            DecisionRouterBackend::Off
        };
    }
    Ok(decision_router_status(runtime.inner()).await)
}

#[tauri::command]
async fn set_decision_router_backend(
    backend: DecisionRouterBackend,
    app: tauri::AppHandle,
    runtime: State<'_, AppRuntime>,
) -> std::result::Result<DecisionRouterStatus, String> {
    if backend == DecisionRouterBackend::Jev {
        let env_name = runtime.config.lock().decision_router.api_key_env.clone();
        if pok_ai_core::brain::get_api_key(&env_name).is_none_or(|key| key.trim().is_empty()) {
            return Err(format!("JEV requires a TypeSafe API key ({env_name})."));
        }
    }
    if backend == DecisionRouterBackend::Laya && !laya_installed(runtime.inner()) {
        return Err("Laya is not installed. Install it in Settings first.".into());
    }
    if backend == DecisionRouterBackend::Laya {
        let preference = runtime.config.lock().decision_router.laya.device;
        let _ = app.emit(
            "laya_runtime_event",
            serde_json::json!({
                "phase": "loading", "detail": "Loading Laya before it is used…",
                "installed": true, "running": false, "preference": preference
            }),
        );
        let (endpoint, _) = ensure_laya_started(&app, runtime.inner(), preference).await?;
        runtime.config.lock().decision_router.laya.endpoint = endpoint;
        let judge_enabled = runtime.config.lock().decision_router.judge.enabled;
        if judge_enabled
            && let Ok((endpoint, _)) = ensure_judge_started(&app, runtime.inner()).await
        {
            runtime.config.lock().decision_router.judge.endpoint = endpoint;
        }
    } else {
        stop_laya(runtime.inner()).await;
        stop_judge(runtime.inner()).await;
    }
    {
        let mut config = runtime.config.lock();
        config.decision_router.backend = backend;
        config.decision_router.enabled = backend != DecisionRouterBackend::Off;
    }
    Ok(decision_router_status(runtime.inner()).await)
}

/// Turn the router training log on or off; the live conversation picks the
/// change up with its next message.
#[tauri::command]
async fn set_router_training(
    enabled: bool,
    runtime: State<'_, AppRuntime>,
) -> std::result::Result<DecisionRouterStatus, String> {
    runtime.config.lock().decision_router.training_log = enabled;
    Ok(decision_router_status(runtime.inner()).await)
}

#[tauri::command]
async fn install_laya(
    repair: Option<bool>,
    app: tauri::AppHandle,
    runtime: State<'_, AppRuntime>,
) -> std::result::Result<DecisionRouterStatus, String> {
    if laya_installed(runtime.inner()) && !repair.unwrap_or(false) {
        return Ok(decision_router_status(runtime.inner()).await);
    }
    stop_laya(runtime.inner()).await;
    let script = bundled_script(&app, "setup-laya.ps1")?;
    let data_dir = runtime.config.lock().data_dir.clone();
    let program = if cfg!(windows) {
        "powershell.exe"
    } else {
        "pwsh"
    };
    let _ = app.emit("laya_runtime_event", serde_json::json!({
        "phase": "installing", "detail": "Installing Laya and downloading the laya-grounded checkpoint…",
        "installed": false, "running": false
    }));
    let mut command = Command::new(program);
    command
        .arg("-NoProfile")
        .arg("-ExecutionPolicy")
        .arg("Bypass")
        .arg("-File")
        .arg(script)
        .arg("-DataDir")
        .arg(data_dir)
        .arg("-Revision")
        .arg(LAYA_PINNED_REVISION);
    hide_process_window(&mut command);
    let output = command
        .output()
        .await
        .map_err(|error| format!("could not launch Laya installer: {error}"))?;
    if !output.status.success() || !laya_installed(runtime.inner()) {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let details = if !stderr.trim().is_empty() {
            stderr.trim()
        } else if !stdout.trim().is_empty() {
            stdout.trim()
        } else {
            "installation failed without output"
        };
        let message = format!("Laya installation failed: {details}");
        let _ = app.emit(
            "laya_runtime_event",
            serde_json::json!({
                "phase": "error", "detail": &message, "installed": false, "running": false
            }),
        );
        return Err(message);
    }
    let next = decision_router_status(runtime.inner()).await;
    let _ = app.emit("laya_runtime_event", next.laya.clone());
    Ok(next)
}

#[tauri::command]
async fn prepare_laya(
    device: LayaDevicePreference,
    app: tauri::AppHandle,
    runtime: State<'_, AppRuntime>,
) -> std::result::Result<DecisionRouterStatus, String> {
    if !laya_installed(runtime.inner()) {
        return Err("Laya is not installed. Install it in Settings first.".into());
    }
    runtime.config.lock().decision_router.laya.device = device;
    let _ = app.emit("laya_runtime_event", serde_json::json!({
        "phase": "loading", "detail": format!("Loading Laya with {device:?} device preference…"),
        "installed": true, "running": false, "preference": device
    }));
    match ensure_laya_started(&app, runtime.inner(), device).await {
        Ok((endpoint, _)) => {
            runtime.config.lock().decision_router.laya.endpoint = endpoint;
            let next = decision_router_status(runtime.inner()).await;
            let _ = app.emit("laya_runtime_event", next.laya.clone());
            Ok(next)
        }
        Err(error) => {
            let _ = app.emit(
                "laya_runtime_event",
                serde_json::json!({
                    "phase": "error", "detail": &error, "installed": true, "running": false,
                    "preference": device
                }),
            );
            Err(error)
        }
    }
}

#[tauri::command]
async fn install_judge(
    repair: Option<bool>,
    app: tauri::AppHandle,
    runtime: State<'_, AppRuntime>,
) -> std::result::Result<DecisionRouterStatus, String> {
    // A plain file-existence check is not enough: an install made before the
    // revision pin existed (or after a Hub-side regression) can carry stale
    // weights while still "installed". Reinstall whenever the manifest
    // revision does not match the pin, without requiring the Repair flag.
    if judge_installed(runtime.inner())
        && judge_revision_matches(runtime.inner())
        && !repair.unwrap_or(false)
    {
        return Ok(decision_router_status(runtime.inner()).await);
    }
    stop_judge(runtime.inner()).await;
    let script = bundled_script(&app, "setup-zeiger.ps1")?;
    let data_dir = runtime.config.lock().data_dir.clone();
    let program = if cfg!(windows) {
        "powershell.exe"
    } else {
        "pwsh"
    };
    let _ = app.emit(
        "judge_runtime_event",
        serde_json::json!({
            "phase": "installing",
            "detail": "Installing the judge model and downloading the zeiger-0.6b checkpoint…",
            "installed": false, "running": false, "enabled": false
        }),
    );
    let mut command = Command::new(program);
    command
        .arg("-NoProfile")
        .arg("-ExecutionPolicy")
        .arg("Bypass")
        .arg("-File")
        .arg(script)
        .arg("-DataDir")
        .arg(data_dir)
        .arg("-Revision")
        .arg(JUDGE_PINNED_REVISION);
    hide_process_window(&mut command);
    let output = command
        .output()
        .await
        .map_err(|error| format!("could not launch the judge-model installer: {error}"))?;
    if !output.status.success()
        || !judge_installed(runtime.inner())
        || !judge_revision_matches(runtime.inner())
    {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let details = if !stderr.trim().is_empty() {
            stderr.trim()
        } else if !stdout.trim().is_empty() {
            stdout.trim()
        } else {
            "installation failed without output"
        };
        let message = format!("Judge-model installation failed: {details}");
        let _ = app.emit(
            "judge_runtime_event",
            serde_json::json!({
                "phase": "error", "detail": &message, "installed": false, "running": false, "enabled": false
            }),
        );
        return Err(message);
    }
    let next = decision_router_status(runtime.inner()).await;
    let _ = app.emit("judge_runtime_event", next.judge.clone());
    Ok(next)
}

#[tauri::command]
async fn set_judge_enabled(
    enabled: bool,
    app: tauri::AppHandle,
    runtime: State<'_, AppRuntime>,
) -> std::result::Result<DecisionRouterStatus, String> {
    if enabled && !judge_installed(runtime.inner()) {
        return Err("The judge model is not installed. Install it in Settings first.".into());
    }
    if enabled {
        let _ = app.emit(
            "judge_runtime_event",
            serde_json::json!({
                "phase": "loading", "detail": "Loading the judge model…",
                "installed": true, "running": false, "enabled": true
            }),
        );
        match ensure_judge_started(&app, runtime.inner()).await {
            Ok((endpoint, _)) => {
                runtime.config.lock().decision_router.judge.endpoint = endpoint;
            }
            Err(error) => {
                let _ = app.emit(
                    "judge_runtime_event",
                    serde_json::json!({
                        "phase": "error", "detail": &error, "installed": true, "running": false,
                        "enabled": false
                    }),
                );
                return Err(error);
            }
        }
    } else {
        stop_judge(runtime.inner()).await;
    }
    runtime.config.lock().decision_router.judge.enabled = enabled;
    let next = decision_router_status(runtime.inner()).await;
    let _ = app.emit("judge_runtime_event", next.judge.clone());
    Ok(next)
}

#[tauri::command]
#[allow(non_snake_case)]
async fn save_decision_router_key(
    apiKey: String,
    runtime: State<'_, AppRuntime>,
) -> std::result::Result<DecisionRouterStatus, String> {
    let env_name = runtime.config.lock().decision_router.api_key_env.clone();
    let api_key = apiKey.trim();
    if api_key.is_empty() {
        return Err("API key cannot be empty".into());
    }
    persist_provider_key(&env_name, api_key)?;
    pok_ai_core::brain::set_api_key(&env_name, api_key);
    Ok(decision_router_status(runtime.inner()).await)
}

const CREDENTIAL_SERVICE: &str = "POK-Ai";

#[cfg(windows)]
fn persist_provider_key(name: &str, api_key: &str) -> std::result::Result<(), String> {
    use keyring_core::api::CredentialStoreApi;

    windows_native_keyring_store::Store::new()
        .map_err(|error| format!("could not open Windows Credential Manager: {error}"))?
        .build(CREDENTIAL_SERVICE, name, None)
        .map_err(|error| format!("could not create API-key credential: {error}"))?
        .set_password(api_key)
        .map_err(|error| format!("could not save API key in Windows Credential Manager: {error}"))
}

#[cfg(not(windows))]
fn persist_provider_key(_name: &str, _api_key: &str) -> std::result::Result<(), String> {
    Err("secure provider-key storage is available in the native Windows build".into())
}

#[cfg(windows)]
fn load_provider_key(name: &str) -> Option<String> {
    use keyring_core::api::CredentialStoreApi;

    windows_native_keyring_store::Store::new()
        .ok()?
        .build(CREDENTIAL_SERVICE, name, None)
        .ok()?
        .get_password()
        .ok()
        .filter(|key| !key.trim().is_empty())
}

#[cfg(not(windows))]
fn load_provider_key(_name: &str) -> Option<String> {
    None
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let project_config = std::env::current_dir().ok().and_then(|directory| {
        directory
            .ancestors()
            .map(|ancestor| ancestor.join("pok-ai.toml"))
            .find(|candidate| candidate.is_file())
    });
    let mut config = Config::load(project_config.as_deref()).expect("valid POK-Ai configuration");
    if let Err(error) = apply_provider_endpoint_overrides(&mut config) {
        eprintln!("Ignoring saved provider endpoint overrides: {error}");
    }
    for env_name in config
        .providers
        .values()
        .filter_map(|provider| provider.api_key_env.as_deref())
    {
        if let Some(api_key) = load_provider_key(env_name) {
            pok_ai_core::brain::set_api_key(env_name, api_key);
        }
    }
    if let Some(api_key) = load_provider_key(&config.decision_router.api_key_env) {
        pok_ai_core::brain::set_api_key(&config.decision_router.api_key_env, api_key);
    }
    if config.decision_router.enabled
        && config.decision_router.backend == DecisionRouterBackend::Jev
        && pok_ai_core::brain::get_api_key(&config.decision_router.api_key_env)
            .is_none_or(|key| key.trim().is_empty())
    {
        eprintln!(
            "JEV was configured as enabled but {} is unavailable; starting with JEV disabled",
            config.decision_router.api_key_env
        );
        config.decision_router.enabled = false;
    }
    let runtime = AppRuntime {
        config: Arc::new(SyncMutex::new(config)),
        cancellation: Arc::new(SyncMutex::new(CancellationToken::new())),
        pause: Arc::new(PauseController::default()),
        conversation: Arc::new(Mutex::new(None)),
        conversation_provider: Arc::new(Mutex::new(None)),
        conversation_decision_router_backend: Arc::new(Mutex::new(None)),
        conversation_judge_enabled: Arc::new(Mutex::new(None)),
        laya_process: Arc::new(Mutex::new(None)),
        judge_process: Arc::new(Mutex::new(None)),
        shutdown_started: Arc::new(AtomicBool::new(false)),
        model_transition: Arc::new(Mutex::new(())),
        local_api_query: Arc::new(Mutex::new(())),
        model_runtime_cache: Arc::new(Mutex::new(HashMap::new())),
        model_catalog_cache: Arc::new(Mutex::new(HashMap::new())),
        user_guidance_queue: Arc::new(SyncMutex::new(std::collections::VecDeque::new())),
        external_provider_acknowledgments: Arc::new(Mutex::new(HashSet::new())),
        command_manager: Arc::new(SyncMutex::new(None)),
        pending_resume: Arc::new(Mutex::new(None)),
    };
    let broker = Arc::new(ApprovalBroker::default());
    let broker_setup = broker.clone();
    let question_broker = Arc::new(QuestionBroker::default());
    let question_broker_setup = question_broker.clone();
    let emergency_shortcut = Shortcut::new(Some(Modifiers::CONTROL | Modifiers::ALT), Code::Escape);
    let app = tauri::Builder::default()
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(move |app, shortcut, event| {
                    if shortcut != &emergency_shortcut
                        && voice::hotkey_event(
                            app,
                            shortcut,
                            event.state() == ShortcutState::Pressed,
                        )
                    {
                        return;
                    }
                    if shortcut == &emergency_shortcut && event.state() == ShortcutState::Pressed {
                        app.state::<AppRuntime>().cancellation.lock().cancel();
                        let _ = app.emit("emergency_stopped", ());
                        let broker = app.state::<Arc<ApprovalBroker>>().inner().clone();
                        tauri::async_runtime::spawn(async move {
                            broker.pending.lock().await.clear();
                        });
                    }
                })
                .build(),
        )
        .manage(voice::VoiceState::default())
        .manage(runtime)
        .manage(broker)
        .manage(question_broker)
        .setup(move |app| {
            let _ = broker_setup.app.set(app.handle().clone());
            let _ = question_broker_setup.app.set(app.handle().clone());
            app.global_shortcut().register(emergency_shortcut)?;
            voice::register_saved_hotkey(app.handle());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_status,
            read_observation_frame,
            read_dropped_image,
            voice::voice_status,
            voice::set_voice_settings,
            voice::install_voice_models,
            voice::start_voice,
            voice::stop_voice,
            voice::set_voice_hotkey,
            voice::set_voice_hotkey_mode,
            update_provider_endpoint,
            get_local_model_status,
            load_model,
            switch_local_model,
            unload_model,
            list_providers,
            save_provider_key,
            get_decision_router_status,
            set_decision_router_enabled,
            set_decision_router_backend,
            install_laya,
            prepare_laya,
            install_judge,
            set_judge_enabled,
            set_router_training,
            save_decision_router_key,
            list_models,
            get_model_capabilities,
            refresh_models,
            list_memory_drafts,
            list_memory_library,
            approve_memory,
            reject_memory,
            set_memory_enabled,
            delete_memory,
            forget_memory_and_prevent_relearning,
            scan_memory_duplicates,
            merge_memories,
            undo_memory_merge,
            set_procedure_enabled,
            delete_procedure,
            resolve_approval,
            resolve_question,
            dismiss_question,
            emergency_stop,
            stop_command,
            read_command_output,
            pause_agent,
            resume_agent,
            new_conversation,
            list_conversations,
            resume_conversation,
            resume_latest_conversation,
            load_conversation_history,
            compact_context,
            run_prompt,
            get_model_context,
            save_model_context_override,
            list_generated_tools,
            list_generated_tool_candidates_command,
            dismiss_generated_tool_candidate_command,
            set_generated_tool_enabled,
            delete_generated_tool_command,
            send_guidance
        ])
        .build(tauri::generate_context!())
        .expect("error while building POK-Ai");
    app.run(|app_handle, event| {
        if let tauri::RunEvent::ExitRequested { api, code, .. } = event {
            let runtime = app_handle.state::<AppRuntime>();
            if !runtime.shutdown_started.swap(true, Ordering::SeqCst) {
                api.prevent_exit();
                runtime.cancellation.lock().cancel();
                if let Some(manager) = runtime.command_manager.lock().take() {
                    manager.request_shutdown();
                }
                let app_handle = app_handle.clone();
                let runtime = runtime.inner().clone();
                tauri::async_runtime::spawn(async move {
                    stop_laya(&runtime).await;
                    stop_judge(&runtime).await;
                    app_handle.exit(code.unwrap_or(0));
                });
            }
        }
    });
}
