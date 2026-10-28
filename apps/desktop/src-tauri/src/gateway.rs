//! The always-on Telegram gateway inside the desktop app: settings commands,
//! the host that turns messages into dashboard runs, and the system tray.
//!
//! A message becomes a `gateway_task` event. The dashboard runs it exactly
//! like a prompt typed in the composer (so approvals, activity cards and the
//! agent view all work) and reports the answer through `gateway_task_done`,
//! which sends it back to the chat.

use std::sync::{
    Arc, OnceLock,
    atomic::{AtomicBool, Ordering},
};

use async_trait::async_trait;
use parking_lot::Mutex as SyncMutex;
use pok_ai_core::gateway::{
    Gateway, GatewayHost, GatewayTask, TaskOrigin,
    access::{AccessControl, AllowedUser, PendingPairing},
    schedule::{JobStore, parse_schedule},
    telegram::TelegramClient,
};
use serde::{Deserialize, Serialize};
use tauri::{Emitter, Manager, State};
use tokio_util::sync::CancellationToken;

use super::{AppRuntime, ApprovalBroker, ApprovalRequest};

/// The saved bot token lives under this name (environment variable or
/// Windows Credential Manager entry), like a provider key.
const TOKEN_ENV: &str = "POK_TELEGRAM_BOT_TOKEN";
const POLL_SECONDS: u64 = 25;
/// A task that never reports back (the window crashed) frees the gateway after this long.
const TASK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60 * 60 * 3);

struct InFlight {
    id: String,
    chat_id: i64,
    prompt: String,
    scheduled: Option<String>,
    started: std::time::Instant,
}

#[derive(Default)]
struct Shared {
    gateway: SyncMutex<Option<Arc<Gateway>>>,
    in_flight: SyncMutex<Option<InFlight>>,
    pending_approval: SyncMutex<Option<uuid::Uuid>>,
    bot: SyncMutex<Option<String>>,
    error: SyncMutex<Option<String>>,
    running: AtomicBool,
}

#[derive(Default)]
pub struct GatewayState {
    shared: Arc<Shared>,
    cancel: SyncMutex<Option<CancellationToken>>,
    stores: OnceLock<(Arc<AccessControl>, Arc<JobStore>)>,
    /// Serializes start and stop.
    lifecycle: tokio::sync::Mutex<()>,
}

impl GatewayState {
    pub fn is_running(&self) -> bool {
        self.shared.running.load(Ordering::SeqCst)
    }

    fn stores(&self, runtime: &AppRuntime) -> (Arc<AccessControl>, Arc<JobStore>) {
        self.stores
            .get_or_init(|| {
                let directory = gateway_dir(runtime);
                (
                    Arc::new(AccessControl::open(&directory.join("access.json"))),
                    Arc::new(JobStore::open(&directory.join("jobs.json"))),
                )
            })
            .clone()
    }
}

fn gateway_dir(runtime: &AppRuntime) -> std::path::PathBuf {
    runtime.config.lock().data_dir.join("gateway")
}

#[derive(Serialize, Deserialize, Default)]
struct GatewaySettings {
    #[serde(default)]
    enabled: bool,
}

fn load_settings(runtime: &AppRuntime) -> GatewaySettings {
    std::fs::read_to_string(gateway_dir(runtime).join("settings.json"))
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default()
}

fn save_settings(runtime: &AppRuntime, settings: &GatewaySettings) -> Result<(), String> {
    let directory = gateway_dir(runtime);
    std::fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
    std::fs::write(
        directory.join("settings.json"),
        serde_json::to_vec_pretty(settings).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())
}

fn preview(text: &str, limit: usize) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= limit {
        flat
    } else {
        format!("{}…", flat.chars().take(limit).collect::<String>())
    }
}

/// Runs messages through the dashboard.
struct AppHost {
    app: tauri::AppHandle,
    shared: Arc<Shared>,
}

impl AppHost {
    fn flight_summary(&self) -> Option<String> {
        self.shared.in_flight.lock().as_ref().map(|flight| {
            let minutes = flight.started.elapsed().as_secs() / 60;
            format!("{} ({minutes} min)", preview(&flight.prompt, 120))
        })
    }
}

#[async_trait]
impl GatewayHost for AppHost {
    async fn submit(&self, task: GatewayTask) -> Result<(), String> {
        {
            let mut flight = self.shared.in_flight.lock();
            if let Some(current) = flight.as_ref()
                && current.started.elapsed() < TASK_TIMEOUT
            {
                return Err(
                    "I am busy with another task. Send /status to see it, or /stop to cancel it."
                        .into(),
                );
            }
            *flight = Some(InFlight {
                id: task.id.clone(),
                chat_id: task.chat_id,
                prompt: task.prompt.clone(),
                scheduled: match &task.origin {
                    TaskOrigin::Schedule { job } => Some(job.clone()),
                    TaskOrigin::Message { .. } => None,
                },
                started: std::time::Instant::now(),
            });
        }
        let (source, who) = match &task.origin {
            TaskOrigin::Message { user } => ("telegram", user.clone()),
            TaskOrigin::Schedule { job } => ("schedule", job.clone()),
        };
        self.app
            .emit(
                "gateway_task",
                serde_json::json!({
                    "id": task.id, "prompt": task.prompt, "source": source, "who": who
                }),
            )
            .map_err(|error| {
                *self.shared.in_flight.lock() = None;
                format!("The app could not take that task: {error}")
            })
    }

    async fn stop(&self) -> String {
        if self.shared.in_flight.lock().take().is_none() {
            return "Nothing is running.".into();
        }
        *self.shared.pending_approval.lock() = None;
        let runtime = self.app.state::<AppRuntime>();
        let broker = self.app.state::<Arc<ApprovalBroker>>();
        let _ = super::emergency_stop(self.app.clone(), runtime, broker);
        let _ = self.app.emit("emergency_stopped", ());
        "Stopped.".into()
    }

    async fn new_conversation(&self) -> String {
        if self.shared.in_flight.lock().is_some() {
            return "A task is running. Send /stop first.".into();
        }
        match self.app.emit("gateway_new_conversation", ()) {
            Ok(()) => "Started a fresh conversation.".into(),
            Err(error) => format!("Could not start one: {error}"),
        }
    }

    async fn status(&self) -> String {
        let mut text = match self.flight_summary() {
            Some(summary) => format!("Working on: {summary}"),
            None => "Idle. Send a message to give me a task.".into(),
        };
        if self.shared.pending_approval.lock().is_some() {
            text.push_str("\nWaiting for your permission: /approve or /deny.");
        }
        text
    }

    async fn approve(&self, allow: bool) -> String {
        let Some(id) = self.shared.pending_approval.lock().take() else {
            return "Nothing is waiting for permission.".into();
        };
        let broker = self.app.state::<Arc<ApprovalBroker>>();
        let Some(sender) = broker.pending.lock().await.remove(&id) else {
            return "That request already expired.".into();
        };
        let decision = if allow {
            pok_ai_core::policy::ApprovalDecision::AllowOnce
        } else {
            pok_ai_core::policy::ApprovalDecision::Deny
        };
        let _ = sender.send(decision);
        if allow { "Approved." } else { "Denied." }.into()
    }

    fn changed(&self) {
        let _ = self.app.emit("gateway_changed", ());
    }
}

/// The agent is waiting for permission: tell the chat that sent the task.
pub fn announce_approval(app: &tauri::AppHandle, request: &ApprovalRequest) {
    let state = app.state::<GatewayState>();
    let Some(chat_id) = state
        .shared
        .in_flight
        .lock()
        .as_ref()
        .map(|flight| flight.chat_id)
    else {
        return;
    };
    let Some(gateway) = state.shared.gateway.lock().clone() else {
        return;
    };
    *state.shared.pending_approval.lock() = Some(request.id);
    let message = format!(
        "Permission needed: {} - {}\nReply /approve or /deny, or answer on the PC.",
        request.tool,
        preview(&request.reason, 240)
    );
    tauri::async_runtime::spawn(async move { gateway.send(chat_id, &message).await });
}

#[derive(Serialize)]
pub struct JobView {
    id: String,
    name: String,
    prompt: String,
    schedule: String,
    enabled: bool,
    last_run: Option<String>,
}

#[derive(Serialize)]
pub struct GatewayStatus {
    running: bool,
    enabled: bool,
    has_token: bool,
    bot: Option<String>,
    error: Option<String>,
    busy: bool,
    task: Option<String>,
    allowed: Vec<AllowedUser>,
    pending: Vec<PendingPairing>,
    home_chat: Option<i64>,
    jobs: Vec<JobView>,
}

fn status(runtime: &AppRuntime, state: &GatewayState) -> GatewayStatus {
    let (access, jobs) = state.stores(runtime);
    let shared = &state.shared;
    GatewayStatus {
        running: state.is_running(),
        enabled: load_settings(runtime).enabled,
        has_token: pok_ai_core::brain::get_api_key(TOKEN_ENV)
            .is_some_and(|token| !token.trim().is_empty()),
        bot: shared.bot.lock().clone(),
        error: shared.error.lock().clone(),
        busy: shared.in_flight.lock().is_some(),
        task: shared
            .in_flight
            .lock()
            .as_ref()
            .map(|flight| preview(&flight.prompt, 120)),
        allowed: access.allowed(),
        pending: access.pending(),
        home_chat: access.home_chat(),
        jobs: jobs
            .list()
            .into_iter()
            .map(|job| JobView {
                schedule: job.schedule.describe(),
                last_run: job.last_run.map(|at| at.to_rfc3339()),
                id: job.id,
                name: job.name,
                prompt: job.prompt,
                enabled: job.enabled,
            })
            .collect(),
    }
}

async fn start(app: &tauri::AppHandle) -> Result<(), String> {
    let state = app.state::<GatewayState>();
    let runtime = app.state::<AppRuntime>();
    let _lifecycle = state.lifecycle.lock().await;
    if state.is_running() {
        return Ok(());
    }
    let token = pok_ai_core::brain::get_api_key(TOKEN_ENV)
        .filter(|token| !token.trim().is_empty())
        .ok_or_else(|| "Save a Telegram bot token first.".to_string())?;
    let client = TelegramClient::new(&token).map_err(|error| error.to_string())?;
    let bot = client
        .get_me()
        .await
        .map_err(|error| error.to_string())?
        .username;
    let (access, jobs) = state.stores(runtime.inner());
    let host = Arc::new(AppHost {
        app: app.clone(),
        shared: state.shared.clone(),
    });
    let gateway = Arc::new(Gateway::new(client, access, jobs, host, POLL_SECONDS));
    let cancel = CancellationToken::new();
    *state.cancel.lock() = Some(cancel.clone());
    *state.shared.gateway.lock() = Some(gateway.clone());
    *state.shared.bot.lock() = Some(bot);
    *state.shared.error.lock() = None;
    state.shared.running.store(true, Ordering::SeqCst);
    tauri::async_runtime::spawn(gateway.run(cancel));
    Ok(())
}

async fn stop(app: &tauri::AppHandle) {
    let state = app.state::<GatewayState>();
    let _lifecycle = state.lifecycle.lock().await;
    if let Some(cancel) = state.cancel.lock().take() {
        cancel.cancel();
    }
    // Dropping the gateway also drops its host, which holds `shared`.
    *state.shared.gateway.lock() = None;
    *state.shared.in_flight.lock() = None;
    *state.shared.pending_approval.lock() = None;
    state.shared.running.store(false, Ordering::SeqCst);
}

/// Start the gateway at launch when it was left on.
pub fn autostart(app: &tauri::AppHandle) {
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        let enabled = load_settings(&app.state::<AppRuntime>()).enabled;
        if enabled && let Err(error) = start(&app).await {
            *app.state::<GatewayState>().shared.error.lock() = Some(error);
            let _ = app.emit("gateway_changed", ());
        }
    });
}

#[tauri::command]
pub fn gateway_status(
    runtime: State<'_, AppRuntime>,
    state: State<'_, GatewayState>,
) -> GatewayStatus {
    status(runtime.inner(), state.inner())
}

#[tauri::command]
pub async fn gateway_save_token(
    token: String,
    app: tauri::AppHandle,
    runtime: State<'_, AppRuntime>,
    state: State<'_, GatewayState>,
) -> Result<GatewayStatus, String> {
    let token = token.trim().to_owned();
    // Ask Telegram first: a wrong token should not replace a working one.
    let bot = pok_ai_core::gateway::verify_token(&token)
        .await
        .map_err(|error| error.to_string())?;
    super::persist_provider_key(TOKEN_ENV, &token)?;
    pok_ai_core::brain::set_api_key(TOKEN_ENV, token);
    *state.shared.bot.lock() = Some(bot);
    *state.shared.error.lock() = None;
    // A new token needs a restart of a running gateway.
    if state.is_running() {
        stop(&app).await;
        start(&app).await?;
    }
    Ok(status(runtime.inner(), state.inner()))
}

#[tauri::command]
pub async fn gateway_set_enabled(
    enabled: bool,
    app: tauri::AppHandle,
    runtime: State<'_, AppRuntime>,
    state: State<'_, GatewayState>,
) -> Result<GatewayStatus, String> {
    if enabled {
        if let Err(error) = start(&app).await {
            *state.shared.error.lock() = Some(error.clone());
            return Err(error);
        }
    } else {
        stop(&app).await;
        *state.shared.bot.lock() = None;
    }
    save_settings(runtime.inner(), &GatewaySettings { enabled })?;
    Ok(status(runtime.inner(), state.inner()))
}

#[tauri::command]
pub fn gateway_approve_pairing(
    code: String,
    runtime: State<'_, AppRuntime>,
    state: State<'_, GatewayState>,
) -> Result<GatewayStatus, String> {
    let (access, _) = state.stores(runtime.inner());
    let user = access
        .approve(&code)
        .map_err(|error| error.to_string())?
        .ok_or_else(|| {
            "That code is not open any more. Ask them to message the bot again.".to_string()
        })?;
    if let Some(gateway) = state.shared.gateway.lock().clone() {
        let chat = access.home_chat().unwrap_or(user.id);
        tauri::async_runtime::spawn(async move {
            gateway
                .send(
                    chat,
                    "You are approved. Send me a task, or /help for the commands.",
                )
                .await;
        });
    }
    Ok(status(runtime.inner(), state.inner()))
}

#[tauri::command]
pub fn gateway_deny_pairing(
    code: String,
    runtime: State<'_, AppRuntime>,
    state: State<'_, GatewayState>,
) -> GatewayStatus {
    state.stores(runtime.inner()).0.deny(&code);
    status(runtime.inner(), state.inner())
}

#[tauri::command]
pub fn gateway_remove_user(
    user_id: i64,
    runtime: State<'_, AppRuntime>,
    state: State<'_, GatewayState>,
) -> Result<GatewayStatus, String> {
    state
        .stores(runtime.inner())
        .0
        .remove(user_id)
        .map_err(|error| error.to_string())?;
    Ok(status(runtime.inner(), state.inner()))
}

#[tauri::command]
pub fn gateway_add_job(
    name: String,
    schedule: String,
    prompt: String,
    runtime: State<'_, AppRuntime>,
    state: State<'_, GatewayState>,
) -> Result<GatewayStatus, String> {
    // The schedule is typed the same way as in chat: "daily 09:00" or "every 30m".
    let (parsed, _) = parse_schedule(&format!("{} x", schedule.trim())).ok_or_else(|| {
        "Use \"daily 09:00\" or \"every 30m\" (at least every 5 minutes).".to_string()
    })?;
    state
        .stores(runtime.inner())
        .1
        .add(&name, &prompt, parsed)
        .map_err(|error| error.to_string())?;
    Ok(status(runtime.inner(), state.inner()))
}

#[tauri::command]
pub fn gateway_remove_job(
    id: String,
    runtime: State<'_, AppRuntime>,
    state: State<'_, GatewayState>,
) -> Result<GatewayStatus, String> {
    state
        .stores(runtime.inner())
        .1
        .remove(&id)
        .map_err(|error| error.to_string())?;
    Ok(status(runtime.inner(), state.inner()))
}

#[tauri::command]
pub fn gateway_set_job_enabled(
    id: String,
    enabled: bool,
    runtime: State<'_, AppRuntime>,
    state: State<'_, GatewayState>,
) -> Result<GatewayStatus, String> {
    state
        .stores(runtime.inner())
        .1
        .set_enabled(&id, enabled)
        .map_err(|error| error.to_string())?;
    Ok(status(runtime.inner(), state.inner()))
}

/// The dashboard finished (or could not start) a gateway task: send the
/// answer back to the chat that asked.
#[tauri::command]
pub async fn gateway_task_done(
    id: String,
    ok: bool,
    text: String,
    state: State<'_, GatewayState>,
) -> Result<(), String> {
    let flight = {
        let mut current = state.shared.in_flight.lock();
        match current.as_ref() {
            Some(flight) if flight.id == id => current.take(),
            _ => None,
        }
    };
    let Some(flight) = flight else {
        // Stopped or replaced meanwhile: the chat already heard about it.
        return Ok(());
    };
    *state.shared.pending_approval.lock() = None;
    let Some(gateway) = state.shared.gateway.lock().clone() else {
        return Ok(());
    };
    let body = if ok {
        text
    } else {
        format!("That did not work: {}", preview(&text, 600))
    };
    let message = match &flight.scheduled {
        Some(job) => format!("Scheduled task \"{job}\":\n{body}"),
        None => body,
    };
    gateway.send(flight.chat_id, &message).await;
    Ok(())
}

/// Closing the window hides it while the gateway runs; the tray keeps the app alive.
pub fn keep_running_on_close(window: &tauri::Window, event: &tauri::WindowEvent) {
    if let tauri::WindowEvent::CloseRequested { api, .. } = event
        && window.app_handle().state::<GatewayState>().is_running()
    {
        api.prevent_close();
        let _ = window.hide();
    }
}

fn show_main_window(app: &tauri::AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.unminimize();
        let _ = window.set_focus();
    }
}

/// A tray icon to bring the window back or quit while the gateway runs hidden.
pub fn install_tray(app: &tauri::App) -> tauri::Result<()> {
    use tauri::{
        menu::{Menu, MenuItem},
        tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    };
    let open = MenuItem::with_id(app, "open", "Open POK-Agent", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
    let menu = Menu::with_items(app, &[&open, &quit])?;
    let mut builder = TrayIconBuilder::new()
        .tooltip("POK-Agent")
        .menu(&menu)
        .show_menu_on_left_click(false)
        .on_menu_event(|app, event| match event.id.as_ref() {
            "open" => show_main_window(app),
            "quit" => app.exit(0),
            _ => {}
        })
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                show_main_window(tray.app_handle());
            }
        });
    if let Some(icon) = app.default_window_icon() {
        builder = builder.icon(icon.clone());
    }
    builder.build(app)?;
    Ok(())
}
