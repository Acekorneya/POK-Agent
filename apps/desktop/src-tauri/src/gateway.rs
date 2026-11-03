//! The always-on gateway inside the desktop app (Telegram and WhatsApp):
//! settings commands, the host that turns messages into dashboard runs, and
//! the system tray.
//!
//! A message becomes a `gateway_task` event, with any pictures (PNG data
//! URLs) and saved file paths it came with. The dashboard runs it exactly
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
    Channel, Gateway, GatewayHost, GatewayTask, TaskOrigin,
    access::{AccessControl, AllowedUser, PendingPairing},
    schedule::{JobStore, parse_schedule},
    sidecar,
    telegram::{TelegramChannel, TelegramClient},
    whatsapp::{LinkState, WhatsAppChannel, WhatsAppConfig, WhatsAppMode},
};
use serde::{Deserialize, Serialize};
use tauri::{Emitter, Manager, State};
use tokio_util::sync::CancellationToken;

use super::{AppRuntime, ApprovalBroker, ApprovalRequest};

/// The saved bot token lives under this name (environment variable or
/// Windows Credential Manager entry), like a provider key.
const TOKEN_ENV: &str = "POK_TELEGRAM_BOT_TOKEN";
const POLL_SECONDS: u64 = 25;
const TRAY_ID: &str = "main";
/// The value under `HKCU\Software\Microsoft\Windows\CurrentVersion\Run`.
const AUTOSTART_NAME: &str = "POK-Agent";
#[cfg(windows)]
const RUN_KEY: &str = r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run";
/// A task that never reports back (the window crashed) frees the gateway after this long.
const TASK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60 * 60 * 3);

struct InFlight {
    id: String,
    /// Where the answer goes, as `<channel>:<id>`.
    chat: String,
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
    whatsapp: SyncMutex<Option<Arc<WhatsAppChannel>>>,
    error: SyncMutex<Option<String>>,
    running: AtomicBool,
}

impl Shared {
    /// A short description of the task in flight, if any.
    fn current_task(&self) -> Option<String> {
        self.in_flight
            .lock()
            .as_ref()
            .map(|flight| preview(&flight.prompt, 120))
    }
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

/// Holds the WhatsApp session (its keys) and the component program.
fn whatsapp_dir(runtime: &AppRuntime) -> std::path::PathBuf {
    runtime.config.lock().data_dir.join("whatsapp")
}

/// Folders besides the installed copy where the WhatsApp component may be:
/// next to the app, and a development build in the source tree.
fn whatsapp_folders() -> Vec<std::path::PathBuf> {
    let mut folders = Vec::new();
    if let Ok(exe) = std::env::current_exe()
        && let Some(parent) = exe.parent()
    {
        folders.push(parent.to_owned());
    }
    folders.push(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../sidecars/whatsapp/target/release"),
    );
    folders
}

fn whatsapp_program(runtime: &AppRuntime) -> Option<std::path::PathBuf> {
    sidecar::find(&runtime.config.lock().data_dir, &whatsapp_folders())
}

#[derive(Serialize, Deserialize, Default, Clone)]
struct GatewaySettings {
    #[serde(default)]
    enabled: bool,
    /// Listen on WhatsApp (a linked device) as well.
    #[serde(default)]
    whatsapp: bool,
    #[serde(default)]
    whatsapp_mode: WhatsAppMode,
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
                chat: task.chat.clone(),
                prompt: task.prompt.clone(),
                scheduled: match &task.origin {
                    TaskOrigin::Schedule { job } => Some(job.clone()),
                    TaskOrigin::Message { .. } => None,
                },
                started: std::time::Instant::now(),
            });
        }
        let (source, who) = match &task.origin {
            TaskOrigin::Message { channel, user } => (channel.clone(), user.clone()),
            TaskOrigin::Schedule { job } => ("schedule".to_owned(), job.clone()),
        };
        // Pictures go as PNG data URLs, the way the composer attaches them.
        let images: Vec<String> = task
            .images
            .iter()
            .map(|png| format!("data:image/png;base64,{png}"))
            .collect();
        let files: Vec<String> = task
            .files
            .iter()
            .map(|path| path.display().to_string())
            .collect();
        self.app
            .emit(
                "gateway_task",
                serde_json::json!({
                    "id": task.id, "prompt": task.prompt, "source": source, "who": who,
                    "images": images, "files": files,
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
        // The dashboard may still show this request: close it there too.
        let _ = self.app.emit("approval_resolved", id.to_string());
        if allow { "Approved." } else { "Denied." }.into()
    }

    fn changed(&self) {
        let _ = self.app.emit("gateway_changed", ());
    }
}

/// The agent is waiting for permission: tell the chat that sent the task.
pub fn announce_approval(app: &tauri::AppHandle, request: &ApprovalRequest) {
    let state = app.state::<GatewayState>();
    let Some(chat) = state
        .shared
        .in_flight
        .lock()
        .as_ref()
        .map(|flight| flight.chat.clone())
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
    tauri::async_runtime::spawn(async move { gateway.send(&chat, &message).await });
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
pub struct WhatsAppView {
    /// Listening on WhatsApp is switched on.
    enabled: bool,
    mode: WhatsAppMode,
    /// The component program is on this computer.
    installed: bool,
    /// This computer has a saved WhatsApp link.
    linked: bool,
    state: LinkState,
    /// The QR code to scan, as an SVG image.
    qr_svg: Option<String>,
    account: Option<String>,
    error: Option<String>,
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
    home_chat: Option<String>,
    whatsapp: WhatsAppView,
    jobs: Vec<JobView>,
}

fn status(runtime: &AppRuntime, state: &GatewayState) -> GatewayStatus {
    let (access, jobs) = state.stores(runtime);
    let shared = &state.shared;
    // One lock, read once: the mutex is not reentrant, and this runs on the UI
    // thread, so locking it twice in one expression would freeze the window.
    let task = shared.current_task();
    let settings = load_settings(runtime);
    // One lock, read once, for the same reason.
    let channel = shared.whatsapp.lock().clone();
    let link = channel.as_ref().map(|channel| channel.status());
    let linked = pok_ai_core::gateway::whatsapp::session_exists(&whatsapp_dir(runtime));
    GatewayStatus {
        running: state.is_running(),
        enabled: settings.enabled,
        whatsapp: WhatsAppView {
            enabled: settings.whatsapp,
            mode: settings.whatsapp_mode,
            installed: whatsapp_program(runtime).is_some(),
            linked,
            state: link.as_ref().map_or(LinkState::Off, |link| link.state),
            qr_svg: link.as_ref().and_then(|link| link.qr_svg.clone()),
            account: link.as_ref().and_then(|link| link.account.clone()),
            error: link.and_then(|link| link.error),
        },
        has_token: pok_ai_core::brain::get_api_key(TOKEN_ENV)
            .is_some_and(|token| !token.trim().is_empty()),
        bot: shared.bot.lock().clone(),
        error: shared.error.lock().clone(),
        busy: task.is_some(),
        task,
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
    let settings = load_settings(runtime.inner());
    let mut channels: Vec<Arc<dyn Channel>> = Vec::new();
    let mut problems = Vec::new();
    let mut bot = None;

    let token = pok_ai_core::brain::get_api_key(TOKEN_ENV).filter(|token| !token.trim().is_empty());
    if let Some(token) = token {
        let telegram = async {
            let client = TelegramClient::new(&token).map_err(|error| error.to_string())?;
            let username = client
                .get_me()
                .await
                .map_err(|error| error.to_string())?
                .username;
            Ok::<_, String>((client, username))
        };
        match telegram.await {
            Ok((client, username)) => {
                bot = Some(username);
                channels.push(Arc::new(TelegramChannel::new(client, POLL_SECONDS)));
            }
            Err(error) => problems.push(format!("Telegram: {error}")),
        }
    }

    let whatsapp = settings.whatsapp.then(|| {
        WhatsAppChannel::new(WhatsAppConfig {
            // A missing program is reported in the WhatsApp status.
            sidecar: whatsapp_program(runtime.inner())
                .unwrap_or_else(|| sidecar::installed_path(&runtime.config.lock().data_dir)),
            data_dir: whatsapp_dir(runtime.inner()),
            mode: settings.whatsapp_mode,
        })
    });
    if let Some(whatsapp) = &whatsapp {
        channels.push(whatsapp.clone());
    }
    if channels.is_empty() {
        return Err(if problems.is_empty() {
            "Save a Telegram bot token or turn on WhatsApp first.".to_owned()
        } else {
            problems.join(" ")
        });
    }

    let (access, jobs) = state.stores(runtime.inner());
    let host = Arc::new(AppHost {
        app: app.clone(),
        shared: state.shared.clone(),
    });
    let gateway = Arc::new(Gateway::new(
        channels,
        access,
        jobs,
        host,
        gateway_dir(runtime.inner()).join("inbox"),
    ));
    let cancel = CancellationToken::new();
    *state.cancel.lock() = Some(cancel.clone());
    *state.shared.gateway.lock() = Some(gateway.clone());
    *state.shared.bot.lock() = bot;
    *state.shared.whatsapp.lock() = whatsapp;
    // One channel failing to start does not take the other down; say why.
    *state.shared.error.lock() = (!problems.is_empty()).then(|| problems.join(" "));
    state.shared.running.store(true, Ordering::SeqCst);
    tauri::async_runtime::spawn(gateway.run(cancel));
    refresh_tray(app);
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
    *state.shared.whatsapp.lock() = None;
    *state.shared.in_flight.lock() = None;
    *state.shared.pending_approval.lock() = None;
    state.shared.running.store(false, Ordering::SeqCst);
    refresh_tray(app);
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
        refresh_tray(&app);
    }
    let mut settings = load_settings(runtime.inner());
    settings.enabled = enabled;
    save_settings(runtime.inner(), &settings)?;
    Ok(status(runtime.inner(), state.inner()))
}

/// Whether new settings change what a running gateway listens on. Picking a
/// mode while WhatsApp is off changes nothing that is running, and restarting
/// would only bounce Telegram's polling (dropping what arrives meanwhile).
fn listening_changed(before: &GatewaySettings, after: &GatewaySettings) -> bool {
    before.whatsapp != after.whatsapp
        || (after.whatsapp && before.whatsapp_mode != after.whatsapp_mode)
}

/// Turn listening on WhatsApp on or off (and choose whose messages count).
/// A running gateway restarts so its channels match.
#[tauri::command]
pub async fn gateway_set_whatsapp(
    enabled: bool,
    mode: Option<WhatsAppMode>,
    app: tauri::AppHandle,
    runtime: State<'_, AppRuntime>,
    state: State<'_, GatewayState>,
) -> Result<GatewayStatus, String> {
    let before = load_settings(runtime.inner());
    let mut settings = before.clone();
    settings.whatsapp = enabled;
    if let Some(mode) = mode {
        settings.whatsapp_mode = mode;
    }
    if enabled {
        settings.enabled = true;
    }
    save_settings(runtime.inner(), &settings)?;
    if !listening_changed(&before, &settings) {
        return Ok(status(runtime.inner(), state.inner()));
    }
    let was_running = state.is_running();
    if was_running {
        stop(&app).await;
    }
    if enabled || was_running {
        match start(&app).await {
            Ok(()) => {}
            Err(error) if enabled => {
                *state.shared.error.lock() = Some(error.clone());
                return Err(error);
            }
            // WhatsApp was the only channel: the gateway simply goes quiet.
            Err(_) => {
                settings.enabled = false;
                save_settings(runtime.inner(), &settings)?;
            }
        }
    }
    Ok(status(runtime.inner(), state.inner()))
}

/// Unlink this computer from WhatsApp and forget the saved session.
#[tauri::command]
pub async fn gateway_whatsapp_logout(
    runtime: State<'_, AppRuntime>,
    state: State<'_, GatewayState>,
) -> Result<GatewayStatus, String> {
    let channel = state.shared.whatsapp.lock().clone();
    match channel {
        Some(channel) => channel.logout().await.map_err(|error| error.to_string())?,
        None => pok_ai_core::gateway::whatsapp::forget_session(&whatsapp_dir(runtime.inner())),
    }
    Ok(status(runtime.inner(), state.inner()))
}

/// Download the WhatsApp component for this version of the app, verified
/// against the release checksums.
#[tauri::command]
pub async fn gateway_install_whatsapp(
    app: tauri::AppHandle,
    runtime: State<'_, AppRuntime>,
    state: State<'_, GatewayState>,
) -> Result<GatewayStatus, String> {
    if !cfg!(windows) {
        return Err("The WhatsApp component is available in the Windows build.".into());
    }
    let destination = sidecar::installed_path(&runtime.config.lock().data_dir);
    let release = format!(
        "https://github.com/Acekorneya/POK-Agent/releases/download/v{}",
        env!("CARGO_PKG_VERSION")
    );
    sidecar::install(&release, &destination)
        .await
        .map_err(|error| error.to_string())?;
    // A gateway already waiting for the missing program starts it now.
    if state.is_running() && load_settings(runtime.inner()).whatsapp {
        stop(&app).await;
        start(&app).await?;
    }
    Ok(status(runtime.inner(), state.inner()))
}

#[tauri::command]
pub fn gateway_approve_pairing(
    code: String,
    runtime: State<'_, AppRuntime>,
    state: State<'_, GatewayState>,
) -> Result<GatewayStatus, String> {
    let (access, _) = state.stores(runtime.inner());
    // The chat to tell is known only until the code is used up.
    let asked_from = access.pending_chat(&code);
    let user = access
        .approve(&code)
        .map_err(|error| error.to_string())?
        .ok_or_else(|| {
            "That code is not open any more. Ask them to message the bot again.".to_string()
        })?;
    if let Some(gateway) = state.shared.gateway.lock().clone() {
        let chat = asked_from.or_else(|| access.home_chat()).unwrap_or(user.id);
        tauri::async_runtime::spawn(async move {
            gateway
                .send(
                    &chat,
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
    user_id: String,
    runtime: State<'_, AppRuntime>,
    state: State<'_, GatewayState>,
) -> Result<GatewayStatus, String> {
    state
        .stores(runtime.inner())
        .0
        .remove(&user_id)
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

#[cfg(windows)]
fn reg(args: &[&str]) -> std::io::Result<std::process::Output> {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    std::process::Command::new("reg")
        .args(args)
        .creation_flags(CREATE_NO_WINDOW)
        .output()
}

/// Whether POK-Agent starts, hidden in the tray, when the user signs in.
#[tauri::command]
pub fn get_autostart() -> bool {
    #[cfg(windows)]
    {
        reg(&["query", RUN_KEY, "/v", AUTOSTART_NAME]).is_ok_and(|output| output.status.success())
    }
    #[cfg(not(windows))]
    {
        false
    }
}

/// Start with Windows (per user, no admin rights): a Run entry that launches
/// this executable with `--background`.
#[tauri::command]
pub fn set_autostart(enabled: bool) -> Result<bool, String> {
    #[cfg(windows)]
    {
        if enabled {
            let exe = std::env::current_exe().map_err(|error| error.to_string())?;
            let command = format!("\"{}\" --background", exe.display());
            let output = reg(&[
                "add",
                RUN_KEY,
                "/v",
                AUTOSTART_NAME,
                "/t",
                "REG_SZ",
                "/d",
                &command,
                "/f",
            ])
            .map_err(|error| error.to_string())?;
            if !output.status.success() {
                return Err("Windows would not save the start-up entry.".into());
            }
        } else {
            // Removing an entry that is not there is already the wanted state.
            reg(&["delete", RUN_KEY, "/v", AUTOSTART_NAME, "/f"])
                .map_err(|error| error.to_string())?;
        }
        Ok(get_autostart())
    }
    #[cfg(not(windows))]
    {
        let _ = enabled;
        Err("Start with Windows is available in the Windows build.".into())
    }
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
    gateway.send(&flight.chat, &message).await;
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

/// Say in the tray tooltip whether the agent is listening on Telegram.
fn refresh_tray(app: &tauri::AppHandle) {
    let state = app.state::<GatewayState>();
    // Each lock is taken once, in its own statement.
    let bot = state.shared.bot.lock().clone();
    let whatsapp = state.shared.whatsapp.lock().is_some();
    let mut listening = Vec::new();
    if let Some(bot) = bot {
        listening.push(format!("Telegram (@{bot})"));
    }
    if whatsapp {
        listening.push("WhatsApp".to_owned());
    }
    let text = if state.is_running() && !listening.is_empty() {
        format!("POK-Agent · listening on {}", listening.join(" and "))
    } else {
        "POK-Agent".to_owned()
    };
    if let Some(tray) = app.tray_by_id(TRAY_ID) {
        let _ = tray.set_tooltip(Some(text));
    }
}

/// The window starts hidden. Show it, unless the app was started with
/// `--background` (at sign-in) and the gateway is about to come back up, in
/// which case it stays in the tray and just listens.
pub fn place_window_at_launch(app: &tauri::App, background: bool) {
    let runtime = app.state::<AppRuntime>();
    let settings = load_settings(runtime.inner());
    let will_listen = settings.enabled
        && (settings.whatsapp
            || pok_ai_core::brain::get_api_key(TOKEN_ENV)
                .is_some_and(|token| !token.trim().is_empty()));
    if !(background && will_listen) {
        show_main_window(app.handle());
    }
}

pub fn show_main_window(app: &tauri::AppHandle) {
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
    let mut builder = TrayIconBuilder::with_id(TRAY_ID)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_task_in_flight_is_read_with_a_single_lock() {
        let shared = Shared::default();
        assert_eq!(shared.current_task(), None);
        *shared.in_flight.lock() = Some(InFlight {
            id: "a".into(),
            chat: "telegram:1".into(),
            prompt: "open   the\ncalendar".into(),
            scheduled: None,
            started: std::time::Instant::now(),
        });
        // Locking twice in one expression deadlocked the UI thread once.
        let task = shared.current_task();
        assert_eq!(task.as_deref(), Some("open the calendar"));
        assert!(
            shared.in_flight.try_lock().is_some(),
            "the guard is released"
        );
    }

    #[test]
    fn only_a_real_change_of_channels_restarts_the_gateway() {
        let off = GatewaySettings::default();
        let on = GatewaySettings {
            whatsapp: true,
            ..GatewaySettings::default()
        };
        let on_as_bot = GatewaySettings {
            whatsapp_mode: WhatsAppMode::Bot,
            ..on.clone()
        };
        assert!(listening_changed(&off, &on), "turning it on");
        assert!(listening_changed(&on, &off), "turning it off");
        assert!(listening_changed(&on, &on_as_bot), "another mode while on");
        assert!(!listening_changed(&on, &on), "nothing changed");
        let off_but_bot = GatewaySettings {
            whatsapp_mode: WhatsAppMode::Bot,
            ..off.clone()
        };
        assert!(
            !listening_changed(&off, &off_but_bot),
            "a mode picked while WhatsApp is off leaves Telegram alone"
        );
    }

    #[test]
    fn old_settings_files_still_load() {
        // Written before WhatsApp existed: only `enabled`.
        let settings: GatewaySettings = serde_json::from_str(r#"{"enabled": true}"#).unwrap();
        assert!(settings.enabled && !settings.whatsapp);
        assert_eq!(settings.whatsapp_mode, WhatsAppMode::SelfChat);
    }

    #[test]
    fn long_prompts_are_shortened_for_status_lines() {
        let long = "word ".repeat(100);
        let short = preview(&long, 20);
        assert!(short.chars().count() <= 21);
        assert!(short.ends_with('…'));
    }
}
