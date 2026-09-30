//! Voice input for the composer: model install, settings, and the microphone
//! session. Transcription runs in `pok-ai-voice`; this module downloads its
//! models and forwards its events to the dashboard as `voice_event`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::Mutex as SyncMutex;
use pok_ai_voice::{VoiceEngine, VoiceEvent, VoiceMode, VoiceModels, VoiceSession};
use serde::{Deserialize, Serialize};
use tauri::{Emitter, State};

use super::AppRuntime;

/// Loaded models and the open microphone session, if any.
#[derive(Default)]
pub(crate) struct VoiceState {
    engine: SyncMutex<Option<Arc<VoiceEngine>>>,
    session: SyncMutex<Option<VoiceSession>>,
    installing: SyncMutex<bool>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub(crate) struct VoiceSettings {
    #[serde(default)]
    pub mode: VoiceMode,
    /// Microphone name; `None` follows the system default.
    #[serde(default)]
    pub device: Option<String>,
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
pub(crate) async fn start_voice(
    app: tauri::AppHandle,
    runtime: State<'_, AppRuntime>,
    voice: State<'_, VoiceState>,
) -> Result<(), String> {
    if voice.session.lock().is_some() {
        return Ok(());
    }
    let data_dir = data_dir(runtime.inner());
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

/// Stop listening; the phrase in progress still arrives as a final event.
#[tauri::command]
pub(crate) async fn stop_voice(voice: State<'_, VoiceState>) -> Result<(), String> {
    let session = voice.session.lock().take();
    if let Some(session) = session {
        tauri::async_runtime::spawn_blocking(move || session.stop())
            .await
            .map_err(|error| error.to_string())?;
    }
    Ok(())
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
        };
        save_settings(dir.path(), &settings).unwrap();
        assert_eq!(load_settings(dir.path()), settings);
    }
}
