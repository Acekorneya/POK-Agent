//! Voice input for the composer: model install, settings, and the microphone
//! session. Transcription runs in `pok-ai-voice`; this module downloads its
//! models and forwards its events to the dashboard as `voice_event`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::Mutex as SyncMutex;
use pok_ai_voice::{VoiceEngine, VoiceEvent, VoiceMode, VoiceModels, VoiceSession};
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager, State};
use tauri_plugin_global_shortcut::{GlobalShortcutExt, Shortcut};

use super::AppRuntime;

/// Loaded models and the open microphone session, if any.
#[derive(Default)]
pub(crate) struct VoiceState {
    engine: SyncMutex<Option<Arc<VoiceEngine>>>,
    session: SyncMutex<Option<VoiceSession>>,
    installing: SyncMutex<bool>,
    /// The system-wide shortcut that toggles listening, once registered.
    hotkey: SyncMutex<Option<Shortcut>>,
}

/// Toggles listening from any application.
pub(crate) const DEFAULT_VOICE_HOTKEY: &str = "Ctrl+Alt+M";

/// How the voice shortcut behaves.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum HotkeyMode {
    /// Press once to start listening, again to stop.
    #[default]
    Toggle,
    /// Listen while the shortcut is held; releasing it stops.
    PushToTalk,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct VoiceSettings {
    #[serde(default)]
    pub mode: VoiceMode,
    /// Microphone name; `None` follows the system default.
    #[serde(default)]
    pub device: Option<String>,
    #[serde(default = "default_hotkey")]
    pub hotkey: String,
    #[serde(default)]
    pub hotkey_mode: HotkeyMode,
}

fn default_hotkey() -> String {
    DEFAULT_VOICE_HOTKEY.into()
}

impl Default for VoiceSettings {
    fn default() -> Self {
        Self {
            mode: VoiceMode::default(),
            device: None,
            hotkey: default_hotkey(),
            hotkey_mode: HotkeyMode::default(),
        }
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct VoiceStatus {
    supported: bool,
    mode: VoiceMode,
    device: Option<String>,
    devices: Vec<String>,
    english_ready: bool,
    multilingual_ready: bool,
    listening: bool,
    installing: bool,
    hotkey: String,
    hotkey_mode: HotkeyMode,
    /// Whether the shortcut is registered with the system right now.
    hotkey_active: bool,
    /// Download size still needed per mode, in MB.
    english_download_mb: u32,
    multilingual_download_mb: u32,
}

fn data_dir(runtime: &AppRuntime) -> PathBuf {
    runtime.config.lock().data_dir.clone()
}

fn settings_path(data_dir: &Path) -> PathBuf {
    data_dir.join("voice").join("settings.json")
}

pub(crate) fn load_settings(data_dir: &Path) -> VoiceSettings {
    std::fs::read_to_string(settings_path(data_dir))
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default()
}

fn save_settings(data_dir: &Path, settings: &VoiceSettings) -> Result<(), String> {
    let path = settings_path(data_dir);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let text = serde_json::to_string_pretty(settings).map_err(|error| error.to_string())?;
    std::fs::write(path, text).map_err(|error| error.to_string())
}

fn missing_mb(models: &VoiceModels, mode: VoiceMode) -> u32 {
    pok_ai_voice::downloads(mode)
        .iter()
        .filter(|download| !models.is_installed(download))
        .map(|download| download.approximate_mb)
        .sum()
}

#[tauri::command]
pub(crate) fn voice_status(
    runtime: State<'_, AppRuntime>,
    voice: State<'_, VoiceState>,
) -> VoiceStatus {
    let data_dir = data_dir(runtime.inner());
    let models = VoiceModels::new(&data_dir);
    let settings = load_settings(&data_dir);
    VoiceStatus {
        supported: cfg!(windows),
        mode: settings.mode,
        device: settings.device,
        devices: pok_ai_voice::input_devices(),
        english_ready: models.ready(VoiceMode::English),
        multilingual_ready: models.ready(VoiceMode::Multilingual),
        listening: voice.session.lock().is_some(),
        installing: *voice.installing.lock(),
        hotkey: settings.hotkey.clone(),
        hotkey_mode: settings.hotkey_mode,
        hotkey_active: voice.hotkey.lock().is_some(),
        english_download_mb: missing_mb(&models, VoiceMode::English),
        multilingual_download_mb: missing_mb(&models, VoiceMode::Multilingual),
    }
}

#[tauri::command]
pub(crate) fn set_voice_settings(
    mode: VoiceMode,
    device: Option<String>,
    runtime: State<'_, AppRuntime>,
    voice: State<'_, VoiceState>,
) -> Result<(), String> {
    let data_dir = data_dir(runtime.inner());
    let previous = load_settings(&data_dir);
    let settings = VoiceSettings {
        mode,
        device: device.filter(|name| !name.trim().is_empty()),
        hotkey: previous.hotkey.clone(),
        hotkey_mode: previous.hotkey_mode,
    };
    save_settings(&data_dir, &settings)?;
    if previous.mode != settings.mode {
        // The next start loads the models for the new mode.
        *voice.engine.lock() = None;
    }
    Ok(())
}

#[derive(Clone, Serialize)]
struct InstallProgress {
    label: String,
    detail: String,
    done: bool,
    error: Option<String>,
}

/// Download and unpack the models `mode` needs, reporting `voice_install`
/// progress. Already installed models are skipped.
#[tauri::command]
pub(crate) async fn install_voice_models(
    mode: VoiceMode,
    app: tauri::AppHandle,
    runtime: State<'_, AppRuntime>,
    voice: State<'_, VoiceState>,
) -> Result<(), String> {
    {
        let mut installing = voice.installing.lock();
        if *installing {
            return Err("voice models are already being installed".into());
        }
        *installing = true;
    }
    let models = VoiceModels::new(&data_dir(runtime.inner()));
    let result = install(&app, &models, mode).await;
    *voice.installing.lock() = false;
    let _ = app.emit(
        "voice_install",
        InstallProgress {
            label: if result.is_ok() {
                "Voice input is ready".into()
            } else {
                "Voice install failed".into()
            },
            detail: String::new(),
            done: true,
            error: result.as_ref().err().cloned(),
        },
    );
    result
}

async fn install(
    app: &tauri::AppHandle,
    models: &VoiceModels,
    mode: VoiceMode,
) -> Result<(), String> {
    std::fs::create_dir_all(models.root()).map_err(|error| error.to_string())?;
    let client = reqwest::Client::builder()
        .user_agent("POK-Agent voice installer")
        .build()
        .map_err(|error| error.to_string())?;
    for download in pok_ai_voice::downloads(mode) {
        if models.is_installed(download) {
            continue;
        }
        let file_name = download.url.rsplit('/').next().unwrap_or("model");
        let target = models.root().join(format!("{file_name}.part"));
        let mut response = client
            .get(download.url)
            .send()
            .await
            .and_then(reqwest::Response::error_for_status)
            .map_err(|error| format!("{}: {error}", download.name))?;
        let total = response.content_length().unwrap_or(0);
        let mut file = tokio::fs::File::create(&target)
            .await
            .map_err(|error| error.to_string())?;
        let mut received = 0_u64;
        let mut reported = 0_u64;
        while let Some(chunk) = response.chunk().await.map_err(|error| error.to_string())? {
            tokio::io::AsyncWriteExt::write_all(&mut file, &chunk)
                .await
                .map_err(|error| error.to_string())?;
            received += chunk.len() as u64;
            if received - reported > 4 * 1024 * 1024 || received == total {
                reported = received;
                let _ = app.emit(
                    "voice_install",
                    InstallProgress {
                        label: format!("Downloading {}", download.name),
                        detail: if total > 0 {
                            format!("{} of {} MB", received / 1_048_576, total / 1_048_576)
                        } else {
                            format!("{} MB", received / 1_048_576)
                        },
                        done: false,
                        error: None,
                    },
                );
            }
        }
        tokio::io::AsyncWriteExt::flush(&mut file)
            .await
            .map_err(|error| error.to_string())?;
        drop(file);
        if file_name.ends_with(".tar.bz2") {
            let _ = app.emit(
                "voice_install",
                InstallProgress {
                    label: format!("Unpacking {}", download.name),
                    detail: String::new(),
                    done: false,
                    error: None,
                },
            );
            unpack(&target, models.root()).await?;
            let _ = std::fs::remove_file(&target);
            models.prune(download).map_err(|error| error.to_string())?;
        } else {
            std::fs::rename(&target, models.root().join(download.installed_as))
                .map_err(|error| error.to_string())?;
        }
        if !models.is_installed(download) {
            return Err(format!("{} did not install completely", download.name));
        }
    }
    Ok(())
}

/// Unpack a `.tar.bz2` with the `tar` that ships with Windows 10 and later.
async fn unpack(archive: &Path, into: &Path) -> Result<(), String> {
    let mut command = tokio::process::Command::new("tar");
    command.arg("-xjf").arg(archive).arg("-C").arg(into);
    #[cfg(windows)]
    command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    let output = command
        .output()
        .await
        .map_err(|error| format!("tar: {error}"))?;
    if output.status.success() {
        Ok(())
    } else {
        Err(format!(
            "unpacking failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ))
    }
}

/// Start listening. Loads the models on first use (a few seconds).
#[tauri::command]
pub(crate) async fn start_voice(app: AppHandle) -> Result<(), String> {
    start(&app).await
}

/// Stop listening; the phrase in progress still arrives as a final event.
#[tauri::command]
pub(crate) async fn stop_voice(app: AppHandle) -> Result<(), String> {
    stop(&app).await
}

async fn start(app: &AppHandle) -> Result<(), String> {
    let voice = app.state::<VoiceState>();
    if voice.session.lock().is_some() {
        return Ok(());
    }
    let data_dir = data_dir(app.state::<AppRuntime>().inner());
    let settings = load_settings(&data_dir);
    let engine = voice.engine.lock().clone();
    let engine = match engine {
        Some(engine) if engine.mode() == settings.mode => engine,
        _ => {
            let _ = app.emit("voice_event", serde_json::json!({"kind": "loading"}));
            let models = VoiceModels::new(&data_dir);
            let mode = settings.mode;
            let engine =
                tauri::async_runtime::spawn_blocking(move || VoiceEngine::load(&models, mode))
                    .await
                    .map_err(|error| error.to_string())?
                    .map_err(|error| error.to_string())?;
            let engine = Arc::new(engine);
            *voice.engine.lock() = Some(engine.clone());
            engine
        }
    };
    let sink_app = app.clone();
    let sink: pok_ai_voice::EventSink = Arc::new(move |event: VoiceEvent| {
        let _ = sink_app.emit("voice_event", &event);
    });
    let device = settings.device.clone();
    let session =
        tauri::async_runtime::spawn_blocking(move || engine.listen(device.as_deref(), sink))
            .await
            .map_err(|error| error.to_string())?
            .map_err(|error| error.to_string())?;
    *voice.session.lock() = Some(session);
    let _ = app.emit("voice_event", serde_json::json!({"kind": "listening"}));
    Ok(())
}

async fn stop(app: &AppHandle) -> Result<(), String> {
    let session = app.state::<VoiceState>().session.lock().take();
    if let Some(session) = session {
        tauri::async_runtime::spawn_blocking(move || session.stop())
            .await
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}

/// What a press or release of the voice shortcut does: `Some(true)` starts
/// listening, `Some(false)` stops, `None` does nothing.
pub(crate) fn hotkey_action(mode: HotkeyMode, pressed: bool, listening: bool) -> Option<bool> {
    match (mode, pressed) {
        (HotkeyMode::Toggle, true) => Some(!listening),
        (HotkeyMode::Toggle, false) => None,
        // Holding the key repeats the press; only the first one starts.
        (HotkeyMode::PushToTalk, true) => (!listening).then_some(true),
        (HotkeyMode::PushToTalk, false) => Some(false),
    }
}

/// The voice shortcut was pressed or released, in any application. Returns
/// whether the shortcut was the voice shortcut.
pub(crate) fn hotkey_event(app: &AppHandle, shortcut: &Shortcut, pressed: bool) -> bool {
    let voice = app.state::<VoiceState>();
    if voice.hotkey.lock().as_ref() != Some(shortcut) {
        return false;
    }
    let listening = voice.session.lock().is_some();
    let mode = load_settings(&data_dir(app.state::<AppRuntime>().inner())).hotkey_mode;
    let Some(start_listening) = hotkey_action(mode, pressed, listening) else {
        return true;
    };
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        let result = if start_listening {
            start(&app).await
        } else {
            stop(&app).await
        };
        if let Err(error) = result {
            let _ = app.emit(
                "voice_event",
                serde_json::json!({"kind": "stopped", "error": error}),
            );
        }
    });
    true
}

/// Parse a shortcut such as "Ctrl+Alt+M". It needs a modifier unless it is a
/// function key, so ordinary typing is never captured, and it may not take
/// the emergency stop.
pub(crate) fn parse_hotkey(text: &str) -> Result<Shortcut, String> {
    let text = text.trim();
    let parts: Vec<String> = text
        .split('+')
        .map(|part| part.trim().to_ascii_lowercase())
        .collect();
    let key = parts.last().cloned().unwrap_or_default();
    let modifiers = parts.len().saturating_sub(1);
    let function_key = key.len() > 1
        && key.starts_with('f')
        && key[1..]
            .parse::<u8>()
            .is_ok_and(|number| (1..=24).contains(&number));
    if modifiers == 0 && !function_key {
        return Err(
            "Use a modifier (Ctrl, Alt, Shift or Win) with the key, or a function key such as F9."
                .into(),
        );
    }
    if matches!(key.as_str(), "escape" | "esc")
        && parts.iter().any(|part| part == "ctrl" || part == "control")
        && parts.iter().any(|part| part == "alt")
    {
        return Err("Ctrl+Alt+Esc is the emergency stop.".into());
    }
    text.parse::<Shortcut>()
        .map_err(|error| format!("{text:?} is not a shortcut this system understands: {error}"))
}

/// Register the saved shortcut at startup. Another application may already
/// own it; the Voice settings page then shows it as inactive.
pub(crate) fn register_saved_hotkey(app: &AppHandle) {
    let settings = load_settings(&data_dir(app.state::<AppRuntime>().inner()));
    if let Err(error) = replace_hotkey(app, &settings.hotkey) {
        eprintln!(
            "voice shortcut {} was not registered: {error}",
            settings.hotkey
        );
    }
}

fn replace_hotkey(app: &AppHandle, text: &str) -> Result<Shortcut, String> {
    let shortcut = parse_hotkey(text)?;
    let voice = app.state::<VoiceState>();
    let mut current = voice.hotkey.lock();
    if current.as_ref() == Some(&shortcut) {
        return Ok(shortcut);
    }
    if let Some(previous) = current.take() {
        let _ = app.global_shortcut().unregister(previous);
    }
    app.global_shortcut().register(shortcut).map_err(|error| {
        format!("{text} is already used by another application ({error}). Choose another shortcut.")
    })?;
    *current = Some(shortcut);
    Ok(shortcut)
}

/// Choose between toggling and push-to-talk.
#[tauri::command]
pub(crate) fn set_voice_hotkey_mode(
    mode: HotkeyMode,
    runtime: State<'_, AppRuntime>,
) -> Result<(), String> {
    let data_dir = data_dir(runtime.inner());
    let mut settings = load_settings(&data_dir);
    settings.hotkey_mode = mode;
    save_settings(&data_dir, &settings)
}

/// Change the listening shortcut; it works from any application.
#[tauri::command]
pub(crate) fn set_voice_hotkey(
    hotkey: String,
    app: AppHandle,
    runtime: State<'_, AppRuntime>,
) -> Result<String, String> {
    replace_hotkey(&app, &hotkey)?;
    let data_dir = data_dir(runtime.inner());
    let mut settings = load_settings(&data_dir);
    settings.hotkey = hotkey.trim().to_owned();
    save_settings(&data_dir, &settings)?;
    Ok(settings.hotkey)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voice_settings_round_trip_and_default() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(load_settings(dir.path()), VoiceSettings::default());
        let settings = VoiceSettings {
            mode: VoiceMode::Multilingual,
            device: Some("USB Microphone".into()),
            hotkey: "Ctrl+Shift+F9".into(),
            hotkey_mode: HotkeyMode::PushToTalk,
        };
        save_settings(dir.path(), &settings).unwrap();
        assert_eq!(load_settings(dir.path()), settings);
    }

    #[test]
    fn older_settings_files_get_the_default_shortcut() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("voice")).unwrap();
        std::fs::write(
            settings_path(dir.path()),
            r#"{"mode":"english","device":null}"#,
        )
        .unwrap();
        assert_eq!(load_settings(dir.path()).hotkey, DEFAULT_VOICE_HOTKEY);
    }

    #[test]
    fn shortcuts_need_a_modifier_and_never_take_the_emergency_stop() {
        assert!(parse_hotkey("Ctrl+Alt+M").is_ok());
        assert!(parse_hotkey("F9").is_ok());
        assert!(parse_hotkey("Ctrl+Shift+Space").is_ok());
        assert!(
            parse_hotkey("M").is_err(),
            "a bare letter would capture typing"
        );
        assert!(parse_hotkey("Ctrl+Alt+Escape").is_err());
    }

    #[test]
    fn push_to_talk_listens_while_held_and_toggle_on_each_press() {
        use HotkeyMode::*;
        assert_eq!(hotkey_action(Toggle, true, false), Some(true));
        assert_eq!(hotkey_action(Toggle, true, true), Some(false));
        assert_eq!(
            hotkey_action(Toggle, false, true),
            None,
            "releasing does nothing"
        );
        assert_eq!(hotkey_action(PushToTalk, true, false), Some(true));
        assert_eq!(
            hotkey_action(PushToTalk, true, true),
            None,
            "key repeat while held"
        );
        assert_eq!(
            hotkey_action(PushToTalk, false, true),
            Some(false),
            "release stops"
        );
    }
}
