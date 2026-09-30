#[cfg(not(windows))]
use async_trait::async_trait;
#[cfg(not(windows))]
use pok_ai_core::{
    PokError, Result,
    platform::DesktopPlatform,
    types::{InputAction, OcrBlock, Screenshot, UiElement, WindowInfo},
};

#[derive(Debug)]
pub struct WindowsDesktop {
    #[cfg(windows)]
    uia_gate: std::sync::Arc<tokio::sync::Semaphore>,
}

// Not derivable: the Windows field must be a fresh counting semaphore
// (Arc::default() would be an empty Arc), so the manual impl is required.
#[allow(clippy::derivable_impls)]
impl Default for WindowsDesktop {
    fn default() -> Self {
        Self {
            #[cfg(windows)]
            uia_gate: std::sync::Arc::new(tokio::sync::Semaphore::new(1)),
        }
    }
}

impl WindowsDesktop {
    pub fn new() -> Self {
        Self::default()
    }
}

/// The one-at-a-time UI Automation permit. A walk that outlasted an earlier
/// capture's time limit still holds it; waiting briefly for it to finish
/// keeps the next capture's structure instead of dropping to OCR at once.
#[cfg(windows)]
async fn uia_permit(
    gate: &std::sync::Arc<tokio::sync::Semaphore>,
) -> pok_ai_core::Result<tokio::sync::OwnedSemaphorePermit> {
    match tokio::time::timeout(
        std::time::Duration::from_millis(1_000),
        gate.clone().acquire_owned(),
    )
    .await
    {
        Ok(Ok(permit)) => Ok(permit),
        _ => Err(pok_ai_core::PokError::Tool(
            "UI Automation enrichment is still recovering from a prior slow provider; using screenshot/OCR fallback"
                .into(),
        )),
    }
}

/// The desktop platform for an agent session. With `background_work`, the
/// agent works in its own window while the user keeps using another one
/// (see [`pok_ai_core::agent_window`]).
pub fn desktop_platform(
    background_work: bool,
) -> std::sync::Arc<dyn pok_ai_core::platform::DesktopPlatform> {
    if background_work {
        // Capturing and acting on windows that are not in front is only safe
        // behind AgentWindowDesktop, which makes the agent's window the one
        // every input check authorizes.
        #[cfg(windows)]
        native::allow_background_windows();
        pok_ai_core::agent_window::AgentWindowDesktop::new(std::sync::Arc::new(
            WindowsDesktop::new(),
        ))
    } else {
        std::sync::Arc::new(WindowsDesktop::new())
    }
}

#[cfg(not(windows))]
#[async_trait]
impl DesktopPlatform for WindowsDesktop {
    async fn capture_screens(&self) -> Result<Vec<Screenshot>> {
        Err(unsupported())
    }
    async fn query_ocr(&self) -> Result<Vec<OcrBlock>> {
        Err(unsupported())
    }
    async fn query_ui_tree(&self) -> Result<Vec<UiElement>> {
        Err(unsupported())
    }
    async fn foreground_window(&self) -> Result<Option<WindowInfo>> {
        Err(unsupported())
    }
    async fn cursor_position(&self) -> Result<Option<(i32, i32)>> {
        Err(unsupported())
    }
    async fn simulate_input(&self, _action: &InputAction) -> Result<()> {
        Err(unsupported())
    }
}

#[cfg(not(windows))]
fn unsupported() -> PokError {
    PokError::Unsupported("native desktop control requires the Windows build".into())
}

/// The Start-menu app ID for an application name, from `Get-StartApps`
/// entries of (display name, app ID). Store and per-user apps such as
/// Settings or Discord are not on PATH, so `cmd /C start <name>` shows a
/// "Windows cannot find" dialog while the spawn itself succeeds; launching
/// the resolved ID through `shell:AppsFolder` works like the Start menu.
/// Exact names win; otherwise one unique prefix or containing match is used.
pub fn resolve_start_app<'a>(name: &str, apps: &'a [(String, String)]) -> Option<&'a str> {
    let wanted = name.trim().to_lowercase();
    if wanted.is_empty() {
        return None;
    }
    if let Some((_, id)) = apps.iter().find(|(app, _)| app.to_lowercase() == wanted) {
        return Some(id);
    }
    let unique =
        |matches: Vec<&'a (String, String)>| (matches.len() == 1).then(|| matches[0].1.as_str());
    unique(
        apps.iter()
            .filter(|(app, _)| {
                let app = app.to_lowercase();
                app.starts_with(&wanted)
                    && app[wanted.len()..]
                        .chars()
                        .next()
                        .is_none_or(|next| !next.is_alphanumeric())
            })
            .collect(),
    )
    .or_else(|| {
        unique(
            apps.iter()
                .filter(|(app, _)| app.to_lowercase().contains(&wanted))
                .collect(),
        )
    })
}

/// Lines of text to type. Line breaks are entered separately (as Shift+Enter)
/// so a chat box, where Enter sends, is never submitted mid-text.
#[cfg_attr(not(windows), allow(dead_code))]
fn typed_lines(text: &str) -> Vec<&str> {
    text.split("\r\n")
        .flat_map(|chunk| chunk.split(['\n', '\r']))
        .collect()
}

#[cfg(test)]
mod typed_lines_tests {
    use super::typed_lines;

    #[test]
    fn line_breaks_split_the_text_without_losing_empty_lines() {
        assert_eq!(typed_lines("hello"), vec!["hello"]);
        assert_eq!(typed_lines("a\nb"), vec!["a", "b"]);
        assert_eq!(typed_lines("a\r\nb\rc"), vec!["a", "b", "c"]);
        assert_eq!(typed_lines("a\n\nb"), vec!["a", "", "b"]);
        assert_eq!(typed_lines("a\n"), vec!["a", ""]);
    }
}

/// What the current keyboard layout lets us type as real keys.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(not(windows), allow(dead_code))]
struct KeyLayout {
    /// The top-row digit keys produce 0-9 without Shift (not true on AZERTY).
    digits_unshifted: bool,
    /// The A-Z virtual keys produce the Latin letters A-Z.
    latin_letters: bool,
    caps_lock: bool,
}

/// One step of typing: a real key press, or characters sent as Unicode.
#[derive(Debug, PartialEq, Eq)]
#[cfg_attr(not(windows), allow(dead_code))]
enum Keystroke {
    Key { vk: u16, shift: bool },
    Text(String),
}

/// Digits, letters and spaces are typed as real key presses, like a person
/// typing, because many business applications act on key events (a zip code
/// looked up as its last digit is keyed); text sent as Unicode arrives like a
/// paste that such handlers never see. Other characters, which may need
/// layout-specific keys, go as Unicode.
#[cfg_attr(not(windows), allow(dead_code))]
fn keystrokes(text: &str, layout: KeyLayout) -> Vec<Keystroke> {
    let mut strokes = Vec::new();
    for character in text.chars() {
        let key = match character {
            ' ' => Some((0x20, false)),
            '0'..='9' if layout.digits_unshifted => Some((character as u16, false)),
            'a'..='z' | 'A'..='Z' if layout.latin_letters => Some((
                character.to_ascii_uppercase() as u16,
                character.is_ascii_uppercase() != layout.caps_lock,
            )),
            _ => None,
        };
        match (key, strokes.last_mut()) {
            (Some((vk, shift)), _) => strokes.push(Keystroke::Key { vk, shift }),
            (None, Some(Keystroke::Text(run))) => run.push(character),
            (None, _) => strokes.push(Keystroke::Text(character.to_string())),
        }
    }
    strokes
}

#[cfg(test)]
mod keystroke_tests {
    use super::{KeyLayout, Keystroke, keystrokes};

    const US: KeyLayout = KeyLayout {
        digits_unshifted: true,
        latin_letters: true,
        caps_lock: false,
    };

    fn key(vk: u8, shift: bool) -> Keystroke {
        Keystroke::Key {
            vk: u16::from(vk),
            shift,
        }
    }

    #[test]
    fn digits_letters_and_spaces_are_real_keys() {
        assert_eq!(
            keystrokes("10001", US),
            vec![
                key(b'1', false),
                key(b'0', false),
                key(b'0', false),
                key(b'0', false),
                key(b'1', false)
            ]
        );
        assert_eq!(
            keystrokes("Ab 1", US),
            vec![
                key(b'A', true),
                key(b'B', false),
                key(b' ', false),
                key(b'1', false)
            ]
        );
    }

    #[test]
    fn other_characters_are_sent_as_unicode_runs() {
        assert_eq!(
            keystrokes("a-b@é", US),
            vec![
                key(b'A', false),
                Keystroke::Text("-".into()),
                key(b'B', false),
                Keystroke::Text("@é".into()),
            ]
        );
    }

    #[test]
    fn caps_lock_and_layout_are_respected() {
        let caps = KeyLayout {
            caps_lock: true,
            ..US
        };
        // With Caps Lock on, Shift makes a lowercase letter.
        assert_eq!(
            keystrokes("aB", caps),
            vec![key(b'A', true), key(b'B', false)]
        );
        // AZERTY-like layouts need Shift for digits: send those as Unicode.
        let azerty = KeyLayout {
            digits_unshifted: false,
            ..US
        };
        assert_eq!(keystrokes("12", azerty), vec![Keystroke::Text("12".into())]);
    }
}

#[cfg(test)]
mod start_app_tests {
    use super::resolve_start_app;

    #[test]
    fn resolves_store_and_per_user_apps_by_display_name() {
        let apps = vec![
            ("Settings".to_owned(), "windows.immersivecontrolpanel_cw5n1h2txyewy!microsoft.windows.immersivecontrolpanel".to_owned()),
            ("Discord".to_owned(), "com.squirrel.Discord.Discord".to_owned()),
            ("Discord PTB".to_owned(), "com.squirrel.DiscordPTB.DiscordPTB".to_owned()),
            ("Visual Studio Code".to_owned(), "Microsoft.VisualStudioCode".to_owned()),
            ("Visual Studio 2022".to_owned(), "VisualStudio.2022".to_owned()),
        ];
        assert_eq!(
            resolve_start_app("settings", &apps),
            Some(
                "windows.immersivecontrolpanel_cw5n1h2txyewy!microsoft.windows.immersivecontrolpanel"
            )
        );
        // An exact name beats a longer name that starts with it.
        assert_eq!(
            resolve_start_app("Discord", &apps),
            Some("com.squirrel.Discord.Discord")
        );
        // A unique partial match resolves; an ambiguous one does not.
        assert_eq!(
            resolve_start_app("code", &apps),
            Some("Microsoft.VisualStudioCode")
        );
        assert_eq!(resolve_start_app("Visual Studio", &apps), None);
        assert_eq!(resolve_start_app("notepad", &apps), None);
    }
}

#[cfg(windows)]
mod native {
    use std::{
        collections::{BTreeMap, HashSet},
        sync::{OnceLock, mpsc},
        time::Instant,
    };

    use base64::Engine;
    use chrono::Utc;
    use enigo::{Axis, Button, Coordinate, Direction, Enigo, Key, Keyboard, Mouse, Settings};
    use pok_ai_core::{
        PokError, Result,
        platform::{
            DesktopActivitySnapshot, DesktopPlatform, PatternAction, PatternActionOutcome,
            PatternActionRequest, PatternScrollRequest, PatternTextOutcome, PatternTextRequest,
        },
        types::{
            CaptureRequest, CaptureScope, CaptureTarget, DesktopCapture, InputAction, MonitorInfo,
            MouseButton, OcrBlock, Rect, Screenshot, UiElement, WindowInfo,
        },
    };
    use screenshots::{
        Screen,
        image::{
            ColorType, GenericImageView, ImageEncoder, RgbaImage,
            codecs::png::{CompressionType, FilterType as PngFilterType, PngEncoder},
            imageops::FilterType,
        },
    };
    use sysinfo::{Pid, ProcessesToUpdate, System};
    use uiautomation::{
        UIAutomation, UIElement,
        patterns::{
            UIInvokePattern, UILegacyIAccessiblePattern, UISelectionItemPattern, UITextPattern,
            UITogglePattern, UIValuePattern,
        },
        types::{ControlType, TreeScope, UIProperty},
    };
    use windows::{Globalization::Language, Media::Ocr::OcrEngine};
    use winsafe as w;

    use super::WindowsDesktop;

    /// Installed Start-menu apps as (display name, app ID), read once per
    /// process with `Get-StartApps`; empty if PowerShell is unavailable.
    fn start_apps() -> &'static [(String, String)] {
        static APPS: OnceLock<Vec<(String, String)>> = OnceLock::new();
        APPS.get_or_init(|| {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            let output = std::process::Command::new("powershell.exe")
                .args([
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    // PowerShell writes the legacy codepage by default; one
                    // accented app name then made the JSON unparseable and
                    // every launch fell back to `cmd start`.
                    "[Console]::OutputEncoding = [Text.Encoding]::UTF8; \
                     Get-StartApps | Select-Object Name, AppID | ConvertTo-Json -Compress",
                ])
                .creation_flags(CREATE_NO_WINDOW)
                .output();
            let Ok(output) = output else {
                return Vec::new();
            };
            let text = String::from_utf8_lossy(&output.stdout);
            let Ok(value) =
                serde_json::from_str::<serde_json::Value>(text.trim_start_matches('\u{feff}'))
            else {
                return Vec::new();
            };
            let entries = match value {
                serde_json::Value::Array(items) => items,
                single @ serde_json::Value::Object(_) => vec![single],
                _ => Vec::new(),
            };
            entries
                .iter()
                .filter_map(|entry| {
                    Some((
                        entry.get("Name")?.as_str()?.to_owned(),
                        entry.get("AppID")?.as_str()?.to_owned(),
                    ))
                })
                .collect()
        })
    }

    #[async_trait::async_trait]
    impl DesktopPlatform for WindowsDesktop {
        async fn capture_screens(&self) -> Result<Vec<Screenshot>> {
            tokio::task::spawn_blocking(|| {
                capture_target(CaptureRequest {
                    scope: CaptureScope::All,
                    window_id: None,
                    monitor_id: None,
                    region: None,
                    max_edge: 1_600,
                })
                .map(|capture| capture.screenshots)
            })
            .await
            .map_err(|error| PokError::Other(error.into()))?
        }

        async fn launch_application(&self, name: &str) -> Result<()> {
            let name = name.trim();
            if name.is_empty()
                || name.len() > 64
                || name.contains(['/', '\\', ':', '"', '\'', '&', '|', '<', '>', '^', '%', ';'])
                || name.split_whitespace().count() > 3
            {
                return Err(PokError::Tool(
                    "application name must be a bare name such as 'notepad'".into(),
                ));
            }
            let name = name.to_owned();
            tokio::task::spawn_blocking(move || {
                let mut command = match super::resolve_start_app(&name, start_apps()) {
                    Some(app_id) => {
                        let mut command = std::process::Command::new("explorer.exe");
                        command.arg(format!("shell:AppsFolder\\{app_id}"));
                        command
                    }
                    None => {
                        let mut command = std::process::Command::new("cmd");
                        command.args(["/C", "start", "", &name]);
                        command
                    }
                };
                #[cfg(windows)]
                {
                    use std::os::windows::process::CommandExt;
                    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
                    command.creation_flags(CREATE_NO_WINDOW);
                }
                command
                    .spawn()
                    .map(|_| ())
                    .map_err(|error| PokError::Tool(format!("could not launch {name}: {error}")))
            })
            .await
            .map_err(|error| PokError::Other(error.into()))?
        }

        async fn capture_target(&self, request: &CaptureRequest) -> Result<DesktopCapture> {
            let request = request.clone();
            tokio::task::spawn_blocking(move || capture_target(request))
                .await
                .map_err(|error| PokError::Other(error.into()))?
        }

        async fn query_ocr(&self) -> Result<Vec<OcrBlock>> {
            tokio::task::spawn_blocking(|| {
                ocr_target(CaptureRequest {
                    scope: CaptureScope::All,
                    window_id: None,
                    monitor_id: None,
                    region: None,
                    max_edge: 1_600,
                })
            })
            .await
            .map_err(|error| PokError::Other(error.into()))?
        }

        async fn query_ocr_target(&self, request: &CaptureRequest) -> Result<Vec<OcrBlock>> {
            let request = request.clone();
            tokio::task::spawn_blocking(move || ocr_target(request))
                .await
                .map_err(|error| PokError::Other(error.into()))?
        }

        async fn query_ocr_capture(
            &self,
            capture: &DesktopCapture,
            _request: &CaptureRequest,
        ) -> Result<Vec<OcrBlock>> {
            let capture = capture.clone();
            tokio::task::spawn_blocking(move || ocr_capture(&capture))
                .await
                .map_err(|error| PokError::Other(error.into()))?
        }

        async fn query_ui_tree(&self) -> Result<Vec<UiElement>> {
            let permit = super::uia_permit(&self.uia_gate).await?;
            tokio::task::spawn_blocking(move || {
                let _permit = permit;
                ui_tree()
            })
            .await
            .map_err(|error| PokError::Other(error.into()))?
        }

        async fn query_ui_tree_target(&self, request: &CaptureRequest) -> Result<Vec<UiElement>> {
            let request = request.clone();
            let permit = super::uia_permit(&self.uia_gate).await?;
            tokio::task::spawn_blocking(move || {
                let _permit = permit;
                ui_tree_target(&request, 1_000)
            })
            .await
            .map_err(|error| PokError::Other(error.into()))?
        }

        async fn query_ui_tree_target_limited(
            &self,
            request: &CaptureRequest,
            limit: usize,
        ) -> Result<Vec<UiElement>> {
            let request = request.clone();
            let permit = super::uia_permit(&self.uia_gate).await?;
            tokio::task::spawn_blocking(move || {
                let _permit = permit;
                ui_tree_target(&request, limit)
            })
            .await
            .map_err(|error| PokError::Other(error.into()))?
        }

        async fn list_monitors(&self) -> Result<Vec<MonitorInfo>> {
            tokio::task::spawn_blocking(monitors)
                .await
                .map_err(|error| PokError::Other(error.into()))?
        }

        async fn foreground_window(&self) -> Result<Option<WindowInfo>> {
            tokio::task::spawn_blocking(foreground)
                .await
                .map_err(|error| PokError::Other(error.into()))?
        }

        async fn desktop_activity_snapshot(&self) -> Result<DesktopActivitySnapshot> {
            tokio::task::spawn_blocking(activity_snapshot)
                .await
                .map_err(|error| PokError::Other(error.into()))?
        }

        async fn perform_pattern_action(
            &self,
            request: &PatternActionRequest,
        ) -> Result<Option<PatternActionOutcome>> {
            // A busy UI Automation provider falls back to physical input
            // rather than queueing behind it.
            let Ok(permit) = self.uia_gate.clone().try_acquire_owned() else {
                return Ok(None);
            };
            let request = request.clone();
            tokio::task::spawn_blocking(move || {
                let _permit = permit;
                pattern_action(&request)
            })
            .await
            .map_err(|error| PokError::Other(error.into()))?
        }

        async fn perform_pattern_text(
            &self,
            request: &PatternTextRequest,
        ) -> Result<Option<PatternTextOutcome>> {
            let Ok(permit) = self.uia_gate.clone().try_acquire_owned() else {
                return Ok(None);
            };
            let request = request.clone();
            tokio::task::spawn_blocking(move || {
                let _permit = permit;
                pattern_text(&request)
            })
            .await
            .map_err(|error| PokError::Other(error.into()))?
        }

        async fn perform_pattern_scroll(
            &self,
            request: &PatternScrollRequest,
        ) -> Result<Option<PatternActionOutcome>> {
            let Ok(permit) = self.uia_gate.clone().try_acquire_owned() else {
                return Ok(None);
            };
            let request = request.clone();
            tokio::task::spawn_blocking(move || {
                let _permit = permit;
                pattern_scroll(&request)
            })
            .await
            .map_err(|error| PokError::Other(error.into()))?
        }

        async fn list_windows(&self) -> Result<Vec<WindowInfo>> {
            tokio::task::spawn_blocking(windows)
                .await
                .map_err(|error| PokError::Other(error.into()))?
        }

        async fn activate_window(&self, window_id: &str) -> Result<WindowInfo> {
            let window_id = window_id.to_owned();
            tokio::task::spawn_blocking(move || activate(&window_id))
                .await
                .map_err(|error| PokError::Other(error.into()))?
        }

        async fn cursor_position(&self) -> Result<Option<(i32, i32)>> {
            tokio::task::spawn_blocking(|| {
                let enigo = Enigo::new(&Settings::default()).map_err(tool_error)?;
                enigo.location().map(Some).map_err(tool_error)
            })
            .await
            .map_err(|error| PokError::Other(error.into()))?
        }

        async fn simulate_input(&self, action: &InputAction) -> Result<()> {
            let action = action.clone();
            let result = tokio::task::spawn_blocking(move || input(action))
                .await
                .map_err(|error| PokError::Other(error.into()))?;
            physical_input::note_injected();
            result
        }

        async fn simulate_text(&self, text: &str, inter_key_pause_ms: u64) -> Result<()> {
            let text = text.to_owned();
            let result = tokio::task::spawn_blocking(move || input_text(&text, inter_key_pause_ms))
                .await
                .map_err(|error| PokError::Other(error.into()))?;
            physical_input::note_injected();
            result
        }

        async fn focused_text(&self, max_chars: usize) -> Result<Option<String>> {
            tokio::task::spawn_blocking(move || focused_text(max_chars))
                .await
                .map_err(|error| PokError::Other(error.into()))?
        }

        async fn replace_focused_text(&self, text: &str) -> Result<bool> {
            let text = text.to_owned();
            tokio::task::spawn_blocking(move || replace_focused_text(&text))
                .await
                .map_err(|error| PokError::Other(error.into()))?
        }
    }

    fn capture_target(request: CaptureRequest) -> Result<DesktopCapture> {
        static CAPTURE_MUTEX: OnceLock<std::sync::Mutex<()>> = OnceLock::new();
        let _capture_guard = CAPTURE_MUTEX
            .get_or_init(|| std::sync::Mutex::new(()))
            .lock()
            .map_err(|_| PokError::Tool("desktop capture mutex is poisoned".into()))?;
        let total_started = Instant::now();
        let target = resolve_target(&request)?;
        let native_started = Instant::now();
        let in_front = foreground()?.is_some_and(|window| window.id == target.id);
        let source = if matches!(target.scope, CaptureScope::Window) && !in_front {
            // A covered window's pixels on screen belong to whatever is on
            // top of it; render the window itself instead.
            capture_covered_window(&target)?
        } else {
            capture_rect(&target.bounds)?
        };
        let native_ms = elapsed_ms(native_started);
        let resize_started = Instant::now();
        let model = resize_for_model(&source, request.max_edge.max(1));
        let resize_ms = elapsed_ms(resize_started);
        let source_png_started = Instant::now();
        let source_png = encode_png(&source, PngFilterType::NoFilter)?;
        let source_png_ms = elapsed_ms(source_png_started);
        let model_width = model.width();
        let model_height = model.height();
        let model_png_started = Instant::now();
        let model_png = encode_png(&model, PngFilterType::Sub)?;
        let model_png_ms = elapsed_ms(model_png_started);
        Ok(DesktopCapture {
            screenshots: vec![Screenshot {
                monitor: MonitorInfo {
                    id: target.id.clone(),
                    bounds: target.bounds.clone(),
                    scale_factor: 1.0,
                    primary: false,
                },
                png_base64: model_png,
                source_png_base64: Some(source_png),
                model_width,
                model_height,
                captured_at: Utc::now(),
            }],
            target,
            timings_ms: BTreeMap::from([
                ("native".into(), native_ms),
                ("resize".into(), resize_ms),
                ("source_png".into(), source_png_ms),
                ("model_png".into(), model_png_ms),
                ("total".into(), elapsed_ms(total_started)),
            ]),
        })
    }

    fn monitors() -> Result<Vec<MonitorInfo>> {
        Ok(Screen::all()
            .map_err(tool_error)?
            .into_iter()
            .map(|screen| {
                let info = screen.display_info;
                MonitorInfo {
                    id: info.id.to_string(),
                    bounds: Rect {
                        x: info.x,
                        y: info.y,
                        width: info.width,
                        height: info.height,
                    },
                    scale_factor: f64::from(info.scale_factor),
                    primary: info.is_primary,
                }
            })
            .collect())
    }

    fn ocr_target(request: CaptureRequest) -> Result<Vec<OcrBlock>> {
        ocr_capture(&capture_target(request)?)
    }

    fn ocr_capture(capture: &DesktopCapture) -> Result<Vec<OcrBlock>> {
        let screenshot = capture
            .screenshots
            .first()
            .ok_or_else(|| PokError::Tool("capture contains no image".into()))?;
        let encoded = screenshot
            .source_png_base64
            .as_ref()
            .unwrap_or(&screenshot.png_base64);
        let png = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .map_err(tool_error)?;
        let directory = tempfile::tempdir().map_err(PokError::Io)?;
        let path = directory.path().join("capture.png");
        std::fs::write(&path, &png).map_err(PokError::Io)?;
        let mut output = ocr_worker().recognize(path, capture.target.bounds.clone(), 1.0)?;
        let image = screenshots::image::load_from_memory(&png)
            .map_err(tool_error)?
            .into_rgba8();
        for (index, region) in highlighted_text_regions(&image).into_iter().enumerate() {
            let crop = image
                .view(
                    region.x as u32,
                    region.y as u32,
                    region.width,
                    region.height,
                )
                .to_image();
            let enlarged = screenshots::image::imageops::resize(
                &crop,
                region.width.saturating_mul(3),
                region.height.saturating_mul(3),
                FilterType::Lanczos3,
            );
            let mut enhanced = screenshots::image::DynamicImage::ImageRgba8(enlarged).into_luma8();
            screenshots::image::imageops::invert(&mut enhanced);
            let enhanced_path = directory.path().join(format!("highlight-{index}.png"));
            enhanced.save(&enhanced_path).map_err(tool_error)?;
            let origin = Rect {
                x: capture.target.bounds.x + region.x,
                y: capture.target.bounds.y + region.y,
                width: region.width,
                height: region.height,
            };
            for mut block in ocr_worker().recognize(enhanced_path, origin, 3.0)? {
                block.selected = Some(true);
                block.variant = Some("highlight_inverted_3x".into());
                if !output.iter().any(|existing| {
                    existing.text.eq_ignore_ascii_case(&block.text)
                        && rects_overlap(&existing.bounds, &block.bounds)
                }) {
                    output.push(block);
                }
            }
        }
        Ok(output)
    }

    fn highlighted_text_regions(image: &RgbaImage) -> Vec<Rect> {
        let mut row_runs = Vec::new();
        let mut start = None;
        for y in 0..image.height() {
            let highlighted = (0..image.width())
                .filter(|x| {
                    let pixel = image.get_pixel(*x, y).0;
                    pixel[2] > pixel[0].saturating_add(45)
                        && pixel[2] > pixel[1].saturating_add(20)
                        && pixel[2] > 100
                })
                .count()
                >= 20;
            match (start, highlighted) {
                (None, true) => start = Some(y),
                (Some(first), false) => {
                    row_runs.push((first, y.saturating_sub(1)));
                    start = None;
                }
                _ => {}
            }
        }
        if let Some(first) = start {
            row_runs.push((first, image.height().saturating_sub(1)));
        }
        row_runs
            .into_iter()
            .filter(|(top, bottom)| {
                let height = bottom.saturating_sub(*top).saturating_add(1);
                (6..=48).contains(&height)
            })
            .filter_map(|(top, bottom)| {
                let mut left = image.width();
                let mut right = 0;
                for y in top..=bottom {
                    for x in 0..image.width() {
                        let pixel = image.get_pixel(x, y).0;
                        if pixel[2] > pixel[0].saturating_add(45)
                            && pixel[2] > pixel[1].saturating_add(20)
                            && pixel[2] > 100
                        {
                            left = left.min(x);
                            right = right.max(x);
                        }
                    }
                }
                (left < right).then(|| {
                    let padding = 4;
                    let x = left.saturating_sub(padding);
                    let y = top.saturating_sub(padding);
                    let right = right
                        .saturating_add(padding)
                        .min(image.width().saturating_sub(1));
                    let bottom = bottom
                        .saturating_add(padding)
                        .min(image.height().saturating_sub(1));
                    Rect {
                        x: i32::try_from(x).unwrap_or(i32::MAX),
                        y: i32::try_from(y).unwrap_or(i32::MAX),
                        width: right.saturating_sub(x).saturating_add(1),
                        height: bottom.saturating_sub(y).saturating_add(1),
                    }
                })
            })
            .take(8)
            .collect()
    }

    fn rects_overlap(left: &Rect, right: &Rect) -> bool {
        i64::from(left.x) < i64::from(right.x) + i64::from(right.width)
            && i64::from(left.x) + i64::from(left.width) > i64::from(right.x)
            && i64::from(left.y) < i64::from(right.y) + i64::from(right.height)
            && i64::from(left.y) + i64::from(left.height) > i64::from(right.y)
    }

    struct OcrJob {
        path: std::path::PathBuf,
        origin: Rect,
        scale: f64,
        reply: mpsc::Sender<Result<Vec<OcrBlock>>>,
    }

    struct OcrWorker {
        sender: mpsc::Sender<OcrJob>,
    }

    impl OcrWorker {
        fn recognize(
            &self,
            path: std::path::PathBuf,
            origin: Rect,
            scale: f64,
        ) -> Result<Vec<OcrBlock>> {
            let (reply, result) = mpsc::channel();
            self.sender
                .send(OcrJob {
                    path,
                    origin,
                    scale,
                    reply,
                })
                .map_err(|_| PokError::Tool("Windows OCR worker stopped".into()))?;
            result
                .recv()
                .map_err(|_| PokError::Tool("Windows OCR worker did not reply".into()))?
        }
    }

    fn ocr_worker() -> &'static OcrWorker {
        static WORKER: OnceLock<OcrWorker> = OnceLock::new();
        WORKER.get_or_init(|| {
            let (sender, jobs) = mpsc::channel::<OcrJob>();
            std::thread::Builder::new()
                .name("pok-ai-winrt-ocr".into())
                .spawn(move || {
                    let engine = create_ocr_engine();
                    for job in jobs {
                        let output = engine
                            .as_ref()
                            .map_err(|error| PokError::Tool(error.clone()))
                            .and_then(|engine| {
                                recognize_words(engine, &job.path, &job.origin, job.scale)
                            });
                        let _ = job.reply.send(output);
                    }
                })
                .expect("spawn Windows OCR worker");
            OcrWorker { sender }
        })
    }

    fn create_ocr_engine() -> std::result::Result<OcrEngine, String> {
        let languages = OcrEngine::AvailableRecognizerLanguages().map_err(|e| e.to_string())?;
        let tag = languages
            .First()
            .and_then(|iterator| iterator.Current())
            .and_then(|language| language.LanguageTag())
            .map_err(|e| e.to_string())?;
        let language = Language::CreateLanguage(&tag).map_err(|e| e.to_string())?;
        OcrEngine::TryCreateFromLanguage(&language).map_err(|e| e.to_string())
    }

    fn recognize_words(
        engine: &OcrEngine,
        path: &std::path::Path,
        origin: &Rect,
        scale: f64,
    ) -> Result<Vec<OcrBlock>> {
        let bitmap = win_ocr_bitmap(path)?;
        let result = engine
            .RecognizeAsync(&bitmap)
            .and_then(|operation| operation.get())
            .map_err(tool_error)?;
        let mut output = Vec::new();
        let lines = result.Lines().map_err(tool_error)?;
        for line_index in 0..lines.Size().map_err(tool_error)? {
            let line = lines.GetAt(line_index).map_err(tool_error)?;
            let words = line.Words().map_err(tool_error)?;
            for word_index in 0..words.Size().map_err(tool_error)? {
                let word = words.GetAt(word_index).map_err(tool_error)?;
                let bounds = word.BoundingRect().map_err(tool_error)?;
                let text = word.Text().map_err(tool_error)?.to_string_lossy();
                if !text.trim().is_empty() {
                    output.push(OcrBlock {
                        text: text.trim().into(),
                        bounds: Rect {
                            x: origin.x + (f64::from(bounds.X) / scale).round() as i32,
                            y: origin.y + (f64::from(bounds.Y) / scale).round() as i32,
                            width: (f64::from(bounds.Width) / scale).round().max(0.0) as u32,
                            height: (f64::from(bounds.Height) / scale).round().max(0.0) as u32,
                        },
                        confidence: None,
                        selected: None,
                        variant: Some("raw".into()),
                    });
                }
            }
        }
        Ok(output)
    }

    fn win_ocr_bitmap(
        path: &std::path::Path,
    ) -> Result<windows::Graphics::Imaging::SoftwareBitmap> {
        use windows::{
            Graphics::Imaging::BitmapDecoder,
            Storage::{FileAccessMode, StorageFile},
            core::HSTRING,
        };
        let path = std::fs::canonicalize(path)
            .map_err(PokError::Io)?
            .to_string_lossy()
            .replace("\\\\?\\", "");
        let file = StorageFile::GetFileFromPathAsync(&HSTRING::from(path))
            .and_then(|operation| operation.get())
            .map_err(tool_error)?;
        let stream = file
            .OpenAsync(FileAccessMode::Read)
            .and_then(|operation| operation.get())
            .map_err(tool_error)?;
        let decoder = BitmapDecoder::CreateWithIdAsync(
            BitmapDecoder::PngDecoderId().map_err(tool_error)?,
            &stream,
        )
        .and_then(|operation| operation.get())
        .map_err(tool_error)?;
        decoder
            .GetSoftwareBitmapAsync()
            .and_then(|operation| operation.get())
            .map_err(tool_error)
    }

    fn resolve_target(request: &CaptureRequest) -> Result<CaptureTarget> {
        match request.scope {
            CaptureScope::ActiveWindow => {
                let window =
                    foreground()?.ok_or_else(|| PokError::Tool("no foreground window".into()))?;
                Ok(window_target(window, CaptureScope::ActiveWindow))
            }
            CaptureScope::Window => {
                let requested = request
                    .window_id
                    .as_deref()
                    .ok_or_else(|| PokError::Tool("window_id is required".into()))?;
                // With background work there may be no foreground window at
                // all (a disconnected remote session); the requested window
                // is still reachable directly.
                if let Some(window) = foreground()?.filter(|window| window.id == requested) {
                    return Ok(window_target(window, CaptureScope::Window));
                }
                if background_windows_allowed() {
                    let window = windows()?.into_iter().find(|window| window.id == requested);
                    return match window {
                        Some(window) if window.visible && !window.minimized => {
                            Ok(window_target(window, CaptureScope::Window))
                        }
                        Some(_) => Err(PokError::Tool(format!(
                            "window {requested:?} is minimized or hidden; call activate_window to restore it"
                        ))),
                        None => Err(PokError::Tool(format!(
                            "window {requested:?} no longer exists; call list_windows for the current windows"
                        ))),
                    };
                }
                Err(PokError::Tool(format!(
                    "window {requested:?} is not foreground; call activate_window first"
                )))
            }
            CaptureScope::Monitor => {
                let monitor_id = request
                    .monitor_id
                    .as_deref()
                    .ok_or_else(|| PokError::Tool("monitor_id is required".into()))?;
                let screen = Screen::all()
                    .map_err(tool_error)?
                    .into_iter()
                    .find(|screen| screen.display_info.id.to_string() == monitor_id)
                    .ok_or_else(|| PokError::Tool(format!("unknown monitor {monitor_id:?}")))?;
                let info = screen.display_info;
                Ok(CaptureTarget {
                    scope: CaptureScope::Monitor,
                    id: monitor_id.into(),
                    title: format!("Monitor {monitor_id}"),
                    process_name: String::new(),
                    bounds: Rect {
                        x: info.x,
                        y: info.y,
                        width: info.width,
                        height: info.height,
                    },
                })
            }
            CaptureScope::Region => {
                let bounds = request
                    .region
                    .clone()
                    .ok_or_else(|| PokError::Tool("region bounds are required".into()))?;
                if bounds.width == 0 || bounds.height == 0 {
                    return Err(PokError::Tool("region bounds must be non-empty".into()));
                }
                let desktop = virtual_bounds(&Screen::all().map_err(tool_error)?)?;
                if !rect_contains(&desktop, &bounds) {
                    return Err(PokError::Tool(
                        "region bounds must remain inside the virtual desktop".into(),
                    ));
                }
                Ok(CaptureTarget {
                    scope: CaptureScope::Region,
                    id: format!(
                        "region:{}:{}:{}:{}",
                        bounds.x, bounds.y, bounds.width, bounds.height
                    ),
                    title: "Screen region".into(),
                    process_name: String::new(),
                    bounds,
                })
            }
            CaptureScope::All => {
                let screens = Screen::all().map_err(tool_error)?;
                let bounds = virtual_bounds(&screens)?;
                Ok(CaptureTarget {
                    scope: CaptureScope::All,
                    id: "all-monitors".into(),
                    title: format!("All monitors ({} displays)", screens.len()),
                    process_name: String::new(),
                    bounds,
                })
            }
        }
    }

    fn window_target(window: WindowInfo, scope: CaptureScope) -> CaptureTarget {
        CaptureTarget {
            scope,
            id: window.id,
            title: window.title,
            process_name: window.process_name,
            bounds: window.bounds,
        }
    }

    fn virtual_bounds(screens: &[Screen]) -> Result<Rect> {
        let left = screens
            .iter()
            .map(|screen| screen.display_info.x)
            .min()
            .ok_or_else(|| PokError::Tool("no monitors found".into()))?;
        let top = screens
            .iter()
            .map(|screen| screen.display_info.y)
            .min()
            .ok_or_else(|| PokError::Tool("no monitors found".into()))?;
        let right = screens
            .iter()
            .map(|screen| i64::from(screen.display_info.x) + i64::from(screen.display_info.width))
            .max()
            .unwrap_or(i64::from(left));
        let bottom = screens
            .iter()
            .map(|screen| i64::from(screen.display_info.y) + i64::from(screen.display_info.height))
            .max()
            .unwrap_or(i64::from(top));
        Ok(Rect {
            x: left,
            y: top,
            width: u32::try_from(right - i64::from(left)).map_err(tool_error)?,
            height: u32::try_from(bottom - i64::from(top)).map_err(tool_error)?,
        })
    }

    fn capture_rect(bounds: &Rect) -> Result<RgbaImage> {
        let mut canvas = RgbaImage::new(bounds.width, bounds.height);
        let target_right = i64::from(bounds.x) + i64::from(bounds.width);
        let target_bottom = i64::from(bounds.y) + i64::from(bounds.height);
        for screen in Screen::all().map_err(tool_error)? {
            let info = screen.display_info;
            let left = bounds.x.max(info.x);
            let top = bounds.y.max(info.y);
            let right = target_right.min(i64::from(info.x) + i64::from(info.width));
            let bottom = target_bottom.min(i64::from(info.y) + i64::from(info.height));
            if i64::from(left) >= right || i64::from(top) >= bottom {
                continue;
            }
            let width = u32::try_from(right - i64::from(left)).map_err(tool_error)?;
            let height = u32::try_from(bottom - i64::from(top)).map_err(tool_error)?;
            let part = screen
                .capture_area(left - info.x, top - info.y, width, height)
                .map_err(|error| {
                    PokError::Tool(format!(
                        "target capture failed on monitor {}: {error}",
                        info.id
                    ))
                })?;
            screenshots::image::imageops::overlay(
                &mut canvas,
                &part,
                i64::from(left - bounds.x),
                i64::from(top - bounds.y),
            );
        }
        Ok(canvas)
    }

    fn resize_for_model(source: &RgbaImage, max_edge: u32) -> RgbaImage {
        let edge = source.width().max(source.height());
        if edge <= max_edge {
            return source.clone();
        }
        let scale = f64::from(max_edge) / f64::from(edge);
        let width = (f64::from(source.width()) * scale).round().max(1.0) as u32;
        let height = (f64::from(source.height()) * scale).round().max(1.0) as u32;
        screenshots::image::imageops::resize(source, width, height, FilterType::Triangle)
    }

    fn encode_png(image: &RgbaImage, filter: PngFilterType) -> Result<String> {
        let mut png = Vec::new();
        PngEncoder::new_with_quality(&mut png, CompressionType::Fast, filter)
            .write_image(
                image.as_raw(),
                image.width(),
                image.height(),
                ColorType::Rgba8,
            )
            .map_err(tool_error)?;
        Ok(base64::engine::general_purpose::STANDARD.encode(png))
    }

    fn elapsed_ms(started: Instant) -> u64 {
        u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
    }

    fn foreground() -> Result<Option<WindowInfo>> {
        Ok(w::HWND::GetForegroundWindow().and_then(|hwnd| native_window_info(&hwnd)))
    }

    fn activity_snapshot() -> Result<DesktopActivitySnapshot> {
        let last_input = w::GetLastInputInfo().map_err(tool_error)?;
        // LASTINPUTINFO uses the wrapping 32-bit tick counter. Wrapping
        // subtraction remains correct for the short quiet periods used here.
        let now = w::GetTickCount64();
        let since_input = u64::from((now as u32).wrapping_sub(last_input.dwTime));
        Ok(DesktopActivitySnapshot {
            foreground_window: foreground()?,
            last_user_input_ms: Some(since_input),
            last_physical_input_ms: physical_input::idle_ms(now, now.saturating_sub(since_input)),
            captured_at: Utc::now(),
        })
    }

    /// Lowercase, single-spaced label used to match a control by name.
    fn normalized_label(value: &str) -> String {
        value
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase()
    }

    /// Whether two rectangles overlap by at least half of the smaller one.
    fn mostly_overlaps(element: &uiautomation::types::Rect, target: &Rect) -> bool {
        let (left, top) = (element.get_left(), element.get_top());
        let (right, bottom) = (element.get_right(), element.get_bottom());
        let target_right = target.x + i32::try_from(target.width).unwrap_or(i32::MAX);
        let target_bottom = target.y + i32::try_from(target.height).unwrap_or(i32::MAX);
        let width = (right.min(target_right) - left.max(target.x)).max(0);
        let height = (bottom.min(target_bottom) - top.max(target.y)).max(0);
        let overlap = i64::from(width) * i64::from(height);
        let element_area = i64::from((right - left).max(0)) * i64::from((bottom - top).max(0));
        let target_area = i64::from(target.width) * i64::from(target.height);
        let smaller = element_area.min(target_area);
        smaller > 0 && overlap * 2 >= smaller
    }

    static BACKGROUND_WINDOWS: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);

    /// Let captures and pattern actions reach windows that are not in front.
    pub fn allow_background_windows() {
        BACKGROUND_WINDOWS.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    fn background_windows_allowed() -> bool {
        BACKGROUND_WINDOWS.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The HWND value encoded in a native window id (`pid:HANDLE(0x...)`).
    fn window_handle_value(window_id: &str) -> Option<u64> {
        let hex = window_id.split("HANDLE(0x").nth(1)?.split(')').next()?;
        u64::from_str_radix(hex, 16).ok()
    }

    /// Render a window that is covered by other windows (PrintWindow with
    /// full-content rendering), sized to the window's desktop bounds.
    fn capture_covered_window(target: &CaptureTarget) -> Result<RgbaImage> {
        let handle = window_handle_value(&target.id)
            .ok_or_else(|| PokError::Tool(format!("unrecognized window id {:?}", target.id)))?;
        let window = xcap::Window::all()
            .map_err(tool_error)?
            .into_iter()
            .find(|window| window.id().is_ok_and(|id| u64::from(id) == handle))
            .ok_or_else(|| PokError::Tool(format!("window {:?} is gone", target.id)))?;
        let image = window.capture_image().map_err(tool_error)?;
        let (width, height) = (image.width(), image.height());
        let image = RgbaImage::from_raw(width, height, image.into_raw())
            .ok_or_else(|| PokError::Tool("window capture returned a malformed image".into()))?;
        if (width, height) == (target.bounds.width, target.bounds.height) {
            return Ok(image);
        }
        // Keep one physical pixel per image pixel so target coordinates map
        // exactly as they do for on-screen captures.
        Ok(screenshots::image::imageops::resize(
            &image,
            target.bounds.width.max(1),
            target.bounds.height.max(1),
            FilterType::Triangle,
        ))
    }

    /// Whether an element is the control itself rather than something that
    /// shares its name: the Text label inside it, or a named container
    /// around it. Only an element exposing an action pattern, or a text
    /// field or document, qualifies.
    fn is_operable(element: &UIElement) -> bool {
        element.get_pattern::<UIInvokePattern>().is_ok()
            || element.get_pattern::<UISelectionItemPattern>().is_ok()
            || element.get_pattern::<UITogglePattern>().is_ok()
            || element
                .get_pattern::<uiautomation::patterns::UIExpandCollapsePattern>()
                .is_ok()
            // Text fields and documents: focused for typing.
            || (matches!(
                element.get_control_type(),
                Ok(ControlType::Edit | ControlType::Document)
            ) && (element.get_pattern::<UIValuePattern>().is_ok()
                || element.get_pattern::<UITextPattern>().is_ok()))
    }

    /// The control under `point` (or its nearest named ancestor) when its
    /// name and bounds match the observation.
    fn element_at_point(
        automation: &UIAutomation,
        point: w::POINT,
        wanted: &str,
        bounds: &Rect,
    ) -> Result<Option<UIElement>> {
        let walker = automation.get_control_view_walker().map_err(tool_error)?;
        let mut element = automation
            .element_from_point(uiautomation::types::Point::new(point.x, point.y))
            .map_err(tool_error)?;
        // The element under the point can be a child of the control the
        // observation named (the text inside a button); walk up to it.
        for _ in 0..5 {
            let name_matches = normalized_label(&element.get_name().unwrap_or_default()) == wanted;
            let bounds_match = element
                .get_bounding_rectangle()
                .is_ok_and(|element_bounds| mostly_overlaps(&element_bounds, bounds));
            if name_matches && bounds_match && is_operable(&element) {
                return Ok(Some(element));
            }
            match walker.get_parent(&element) {
                Ok(parent) => element = parent,
                Err(_) => break,
            }
        }
        Ok(None)
    }

    /// The control named `request.name` inside the requested top-level
    /// window whose bounds match the observation; `None` unless exactly one
    /// such control exists.
    fn element_in_window(
        automation: &UIAutomation,
        request: &PatternActionRequest,
        wanted: &str,
    ) -> Result<Option<UIElement>> {
        let Some(root) = window_root(automation, &request.window_id)? else {
            return Ok(None);
        };
        let condition = automation
            .create_property_condition(UIProperty::Name, request.name.as_str().into(), None)
            .map_err(tool_error)?;
        let mut matches = root
            .find_all(TreeScope::Descendants, &condition)
            .unwrap_or_default()
            .into_iter()
            .filter(|element| {
                is_operable(element)
                    && normalized_label(&element.get_name().unwrap_or_default()) == wanted
                    && element
                        .get_bounding_rectangle()
                        .is_ok_and(|bounds| mostly_overlaps(&bounds, &request.bounds))
            });
        let first = matches.next();
        Ok(if matches.next().is_none() {
            first
        } else {
            None
        })
    }

    /// The top-level UI Automation element of a native window id.
    fn window_root(automation: &UIAutomation, window_id: &str) -> Result<Option<UIElement>> {
        let walker = automation.get_control_view_walker().map_err(tool_error)?;
        let desktop = automation.get_root_element().map_err(tool_error)?;
        Ok(walker
            .get_children(&desktop)
            .unwrap_or_default()
            .into_iter()
            .find(|element| element_id(element).as_deref() == Some(window_id)))
    }

    /// Whether an element is a native text field whose Value pattern can take
    /// text directly. Web content (Chromium, Electron, Firefox) is excluded:
    /// setting its value from outside can leave the page's own state behind,
    /// so a chat message could be sent empty.
    fn settable_text_field(element: &UIElement) -> bool {
        matches!(
            element.get_control_type(),
            Ok(ControlType::Edit | ControlType::Document)
        ) && element.is_enabled().unwrap_or(false)
            && !element.is_password().unwrap_or(true)
            && !matches!(
                element.get_framework_id().unwrap_or_default().as_str(),
                "Chrome" | "Gecko"
            )
            && element
                .get_pattern::<UIValuePattern>()
                .and_then(|pattern| pattern.is_readonly())
                .is_ok_and(|read_only| !read_only)
    }

    /// Set a native text field's contents through its Value pattern and read
    /// them back. `None` when the field does not accept it.
    fn pattern_text(request: &PatternTextRequest) -> Result<Option<PatternTextOutcome>> {
        let point = w::POINT {
            x: request.point.0,
            y: request.point.1,
        };
        let Some(hwnd) = w::HWND::WindowFromPoint(point) else {
            return Ok(None);
        };
        let root = match hwnd.GetAncestor(w::co::GA::ROOT) {
            Some(root) => root,
            None => hwnd,
        };
        let (_, pid) = root.GetWindowThreadProcessId();
        let point_in_window = native_window_id(&root, pid) == request.window_id;
        if !point_in_window && !background_windows_allowed() {
            return Ok(None);
        }
        let automation = UIAutomation::new().map_err(tool_error)?;
        let field = if point_in_window {
            // The field under the point, or the nearest field around it.
            let walker = automation.get_control_view_walker().map_err(tool_error)?;
            let mut element = automation
                .element_from_point(uiautomation::types::Point::new(point.x, point.y))
                .map_err(tool_error)?;
            let mut found = None;
            for _ in 0..5 {
                let overlaps = element
                    .get_bounding_rectangle()
                    .is_ok_and(|bounds| mostly_overlaps(&bounds, &request.bounds));
                if overlaps && settable_text_field(&element) {
                    found = Some(element);
                    break;
                }
                match walker.get_parent(&element) {
                    Ok(parent) => element = parent,
                    Err(_) => break,
                }
            }
            found
        } else if request.name.trim().is_empty() {
            // An unnamed field in a covered window cannot be identified.
            None
        } else {
            let Some(root) = window_root(&automation, &request.window_id)? else {
                return Ok(None);
            };
            let condition = automation
                .create_property_condition(UIProperty::Name, request.name.as_str().into(), None)
                .map_err(tool_error)?;
            let mut matches = root
                .find_all(TreeScope::Descendants, &condition)
                .unwrap_or_default()
                .into_iter()
                .filter(|element| {
                    settable_text_field(element)
                        && element
                            .get_bounding_rectangle()
                            .is_ok_and(|bounds| mostly_overlaps(&bounds, &request.bounds))
                });
            let first = matches.next();
            if matches.next().is_none() {
                first
            } else {
                None
            }
        };
        let Some(field) = field else {
            return Ok(None);
        };
        let Ok(pattern) = field.get_pattern::<UIValuePattern>() else {
            return Ok(None);
        };
        let before = pattern.get_value().unwrap_or_default();
        let value = if request.append {
            format!("{before}{}", request.text)
        } else {
            request.text.clone()
        };
        if pattern.set_value(&value).is_err() {
            return Ok(None);
        }
        let after = pattern.get_value().unwrap_or_default();
        Ok(Some(PatternTextOutcome { before, after }))
    }

    /// Scroll the area at `request.point` through the UI Automation Scroll
    /// pattern, so the cursor never moves. `None` when nothing there scrolls
    /// in the requested direction.
    fn pattern_scroll(request: &PatternScrollRequest) -> Result<Option<PatternActionOutcome>> {
        use uiautomation::{patterns::UIScrollPattern, types::ScrollAmount};

        let point = w::POINT {
            x: request.point.0,
            y: request.point.1,
        };
        let Some(hwnd) = w::HWND::WindowFromPoint(point) else {
            return Ok(None);
        };
        let root = match hwnd.GetAncestor(w::co::GA::ROOT) {
            Some(root) => root,
            None => hwnd,
        };
        let (_, pid) = root.GetWindowThreadProcessId();
        let point_in_window = native_window_id(&root, pid) == request.window_id;
        if !point_in_window && !background_windows_allowed() {
            return Ok(None);
        }
        let scrolls = |pattern: &UIScrollPattern| {
            (request.vertical == 0 || pattern.is_vertically_scrollable().unwrap_or(false))
                && (request.horizontal == 0
                    || pattern.is_horizontally_scrollable().unwrap_or(false))
        };
        let automation = UIAutomation::new().map_err(tool_error)?;
        let pattern = if point_in_window {
            // The nearest scrollable ancestor of the element at the point.
            let walker = automation.get_control_view_walker().map_err(tool_error)?;
            let mut element = automation
                .element_from_point(uiautomation::types::Point::new(point.x, point.y))
                .map_err(tool_error)?;
            let mut found = None;
            for _ in 0..25 {
                if let Ok(pattern) = element.get_pattern::<UIScrollPattern>()
                    && scrolls(&pattern)
                {
                    found = Some(pattern);
                    break;
                }
                match walker.get_parent(&element) {
                    Ok(parent) => element = parent,
                    Err(_) => break,
                }
            }
            found
        } else {
            // The agent's window is covered: the smallest scrollable area in
            // its own tree that contains the point.
            let Some(root) = window_root(&automation, &request.window_id)? else {
                return Ok(None);
            };
            let condition = automation
                .create_property_condition(UIProperty::IsScrollPatternAvailable, true.into(), None)
                .map_err(tool_error)?;
            root.find_all(TreeScope::Descendants, &condition)
                .unwrap_or_default()
                .into_iter()
                .filter_map(|element| {
                    let bounds = element.get_bounding_rectangle().ok()?;
                    let contains = bounds.get_left() <= point.x
                        && point.x < bounds.get_right()
                        && bounds.get_top() <= point.y
                        && point.y < bounds.get_bottom();
                    let pattern = element.get_pattern::<UIScrollPattern>().ok()?;
                    (contains && scrolls(&pattern)).then(|| {
                        let area = i64::from(bounds.get_right() - bounds.get_left())
                            * i64::from(bounds.get_bottom() - bounds.get_top());
                        (area, pattern)
                    })
                })
                .min_by_key(|(area, _)| *area)
                .map(|(_, pattern)| pattern)
        };
        let Some(pattern) = pattern else {
            return Ok(None);
        };
        let step = |direction: i8| match (direction.signum(), request.page) {
            (0, _) => ScrollAmount::NoAmount,
            (-1, true) => ScrollAmount::LargeDecrement,
            (-1, false) => ScrollAmount::SmallDecrement,
            (_, true) => ScrollAmount::LargeIncrement,
            (_, false) => ScrollAmount::SmallIncrement,
        };
        pattern
            .scroll(step(request.horizontal), step(request.vertical))
            .map_err(tool_error)?;
        Ok(Some(PatternActionOutcome {
            pattern: "scroll".into(),
        }))
    }

    /// Operate the observed control through a UI Automation pattern without
    /// moving the cursor. Only patterns whose meaning matches the requested
    /// mouse action unambiguously are used; everything else returns `None`
    /// so the caller clicks physically.
    fn pattern_action(request: &PatternActionRequest) -> Result<Option<PatternActionOutcome>> {
        let point = w::POINT {
            x: request.point.0,
            y: request.point.1,
        };
        // The point must still belong to the window that authorized input,
        // not to a window that has since moved on top of it.
        let hwnd = w::HWND::WindowFromPoint(point)
            .ok_or_else(|| PokError::Tool("no window at the target point".into()))?;
        let root = match hwnd.GetAncestor(w::co::GA::ROOT) {
            Some(root) => root,
            None => hwnd,
        };
        let (_, pid) = root.GetWindowThreadProcessId();
        let point_in_window = native_window_id(&root, pid) == request.window_id;
        if !point_in_window && !background_windows_allowed() {
            return Err(PokError::Tool(
                "the target point is covered by another window".into(),
            ));
        }
        let wanted = normalized_label(&request.name);
        if wanted.is_empty() {
            return Ok(None);
        }
        let automation = UIAutomation::new().map_err(tool_error)?;
        let matched = if point_in_window {
            match element_at_point(&automation, point, &wanted, &request.bounds)? {
                Some(element) => Some(element),
                // Nothing actionable under the point, such as a named row
                // whose action is a same-named button inside it: use the one
                // same-named control within the target's bounds, if unique.
                None => element_in_window(&automation, request, &wanted)?,
            }
        } else {
            // The agent's window is behind another one: search its own
            // accessibility tree instead of hit-testing the screen.
            element_in_window(&automation, request, &wanted)?
        };
        let Some(element) = matched else {
            return Ok(None);
        };
        if element.is_password().unwrap_or(false) {
            return Err(PokError::Tool(
                "pattern input is not allowed on password fields".into(),
            ));
        }
        if !element.is_enabled().unwrap_or(false) {
            return Ok(None);
        }
        let Ok(control) = element.get_control_type() else {
            return Ok(None);
        };
        let invoke = || {
            element
                .get_pattern::<UIInvokePattern>()
                .and_then(|pattern| pattern.invoke())
                .ok()
                .map(|()| "invoke")
        };
        let select = || {
            element
                .get_pattern::<UISelectionItemPattern>()
                .and_then(|pattern| pattern.select())
                .ok()
                .map(|()| "select")
        };
        let toggle = || {
            element
                .get_pattern::<UITogglePattern>()
                .and_then(|pattern| pattern.toggle())
                .ok()
                .map(|()| "toggle")
        };
        let expand_collapse = || {
            let pattern = element
                .get_pattern::<uiautomation::patterns::UIExpandCollapsePattern>()
                .ok()?;
            match pattern.get_state() {
                Ok(uiautomation::types::ExpandCollapseState::Expanded) => {
                    pattern.collapse().ok().map(|()| "collapse")
                }
                _ => pattern.expand().ok().map(|()| "expand"),
            }
        };
        let pattern = match request.action {
            PatternAction::Activate => match control {
                // A menu or drop-down button without Invoke opens (or closes)
                // its menu; a toggle button flips.
                ControlType::Button
                | ControlType::Hyperlink
                | ControlType::MenuItem
                | ControlType::SplitButton => invoke().or_else(expand_collapse).or_else(toggle),
                ControlType::CheckBox => toggle().or_else(invoke),
                ControlType::RadioButton | ControlType::TabItem => select().or_else(invoke),
                // Clicking a combo box opens or closes its list.
                ControlType::ComboBox => expand_collapse(),
                // A single click on an item selects it when the item is
                // selectable (File Explorer files and folders, navigation
                // panes, where selecting navigates); an item that is only
                // invokable (a clickable row) runs its click action. Invoke
                // on a selectable item is its double-click (open), so it is
                // never used for a single click there.
                ControlType::ListItem | ControlType::TreeItem | ControlType::DataItem => {
                    select().or_else(invoke).or_else(expand_collapse)
                }
                // Clicking a text field focuses it. Focus changes the active
                // window, so this is done only while the agent's window is
                // in front; otherwise the real click waits for the user.
                ControlType::Edit | ControlType::Document if point_in_window => {
                    element.set_focus().ok().map(|()| "focus")
                }
                _ => None,
            },
            PatternAction::Invoke => invoke(),
            PatternAction::Open => match control {
                // Invoke opens a file, folder or row. An item without it (a
                // navigation-tree entry) is opened by selecting it; its legacy
                // default action there only expands or collapses.
                ControlType::ListItem | ControlType::TreeItem | ControlType::DataItem => {
                    invoke().or_else(select).or_else(|| {
                        element
                            .get_pattern::<UILegacyIAccessiblePattern>()
                            .and_then(|pattern| pattern.do_default_action())
                            .ok()
                            .map(|()| "default_action")
                    })
                }
                _ => None,
            },
        };
        Ok(pattern.map(|pattern| PatternActionOutcome {
            pattern: pattern.into(),
        }))
    }

    /// Tells the user's own mouse and keyboard input apart from input this
    /// process injects. GetLastInputInfo counts both, so input the system saw
    /// after the agent's last injection finished can only be the user's. The
    /// gate checks before each action, so input during an injection burst
    /// does not need to be attributed.
    mod physical_input {
        use std::sync::atomic::{AtomicU64, Ordering};

        /// Tick (GetTickCount64) at which the agent's last injected input
        /// finished; 0 before the first injection.
        static LAST_INJECTED_END: AtomicU64 = AtomicU64::new(0);
        /// Slack for the system to register injected input after SendInput.
        const INJECTION_SLACK_MS: u64 = 150;

        pub fn note_injected() {
            LAST_INJECTED_END.store(winsafe::GetTickCount64(), Ordering::Relaxed);
        }

        /// Milliseconds since the user's last input, or `None` when all
        /// recent input is the agent's own. `system_input_tick` is
        /// GetLastInputInfo converted to the GetTickCount64 clock.
        pub fn idle_ms(now: u64, system_input_tick: u64) -> Option<u64> {
            let injected_end = LAST_INJECTED_END.load(Ordering::Relaxed);
            (system_input_tick > injected_end + INJECTION_SLACK_MS)
                .then(|| now.saturating_sub(system_input_tick))
        }
    }

    fn windows() -> Result<Vec<WindowInfo>> {
        let mut output = Vec::new();
        w::EnumWindows(|hwnd| {
            if let Some(window) = native_window_info(&hwnd).filter(|window| {
                window.visible
                    && (window.minimized || (window.bounds.width > 0 && window.bounds.height > 0))
                    && !window.title.trim().is_empty()
            }) {
                output.push(window);
            }
            true
        })
        .map_err(tool_error)?;
        output.sort_by(|left, right| {
            left.process_name
                .cmp(&right.process_name)
                .then_with(|| left.title.cmp(&right.title))
        });
        output.dedup_by(|left, right| left.id == right.id);
        Ok(output)
    }

    fn activate(window_id: &str) -> Result<WindowInfo> {
        let mut target = None;
        w::EnumWindows(|hwnd| {
            let (_, pid) = hwnd.GetWindowThreadProcessId();
            if target.is_none() && native_window_id(&hwnd, pid) == window_id {
                target = Some(hwnd);
            }
            true
        })
        .map_err(tool_error)?;
        let hwnd = target
            .ok_or_else(|| PokError::Tool(format!("window {window_id:?} no longer exists")))?;
        let activated = native_window_info(&hwnd)
            .ok_or_else(|| PokError::Tool(format!("window {window_id:?} is unavailable")))?;
        for attempt in 0..3 {
            if hwnd.IsIconic() {
                let _ = hwnd.ShowWindowAsync(w::co::SW::RESTORE);
            }
            // Windows only lets the process that produced the latest input
            // take the foreground; otherwise the window just flashes on the
            // taskbar. Escalate without clicking or typing anything.
            match attempt {
                0 => {
                    let _ = hwnd.SetForegroundWindow();
                }
                1 => {
                    // A zero-distance mouse movement: the cursor stays put,
                    // but this process now owns the latest input.
                    if let Ok(mut enigo) = enigo_mutex().lock() {
                        let _ = enigo.move_mouse(0, 0, Coordinate::Rel);
                    }
                    physical_input::note_injected();
                    let _ = hwnd.SetForegroundWindow();
                }
                _ => {
                    // Share the foreground thread's input state for the
                    // switch, then detach again.
                    let ours = w::GetCurrentThreadId();
                    let theirs = w::HWND::GetForegroundWindow()
                        .map(|front| front.GetWindowThreadProcessId().0)
                        .filter(|theirs| *theirs != ours);
                    if let Some(theirs) = theirs {
                        let _ = w::AttachThreadInput(theirs, ours, true);
                        let _ = hwnd.BringWindowToTop();
                        let _ = hwnd.SetForegroundWindow();
                        let _ = w::AttachThreadInput(theirs, ours, false);
                    } else {
                        let _ = hwnd.SetForegroundWindow();
                    }
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(120));
            if let Some(current) = foreground()? {
                if current.id == window_id {
                    return Ok(current);
                }
            }
        }
        let current = foreground()?;
        Err(PokError::Tool(format!(
            "activation did not make {:?} the foreground window after bounded native attempts; current foreground is {}",
            activated.title,
            current.as_ref().map_or_else(
                || "unknown".into(),
                |window| format!("{:?} ({})", window.title, window.process_name)
            )
        )))
    }

    fn native_window_id(hwnd: &w::HWND, pid: u32) -> String {
        // Keep this identical to the UI Automation element ID format so a
        // capture or action can safely resolve a window discovered natively.
        format!("{pid}:HANDLE(0x{hwnd:x})")
    }

    fn native_window_info(hwnd: &w::HWND) -> Option<WindowInfo> {
        let bounds = hwnd.GetWindowRect().ok()?;
        let (_, pid) = hwnd.GetWindowThreadProcessId();
        Some(WindowInfo {
            id: native_window_id(hwnd, pid),
            title: hwnd.GetWindowText().unwrap_or_default(),
            process_name: cached_process_name(pid),
            bounds: Rect {
                x: bounds.left,
                y: bounds.top,
                width: u32::try_from(bounds.right.saturating_sub(bounds.left)).unwrap_or(0),
                height: u32::try_from(bounds.bottom.saturating_sub(bounds.top)).unwrap_or(0),
            },
            elevated: false,
            // A cloaked window is not on screen: a modern app's content window
            // drawn inside its frame (Settings' SystemSettings.exe inside
            // ApplicationFrameHost), a suspended app, or another virtual
            // desktop. It cannot be captured or used on its own; the frame is.
            visible: hwnd.IsWindowVisible() && (hwnd.IsIconic() || !is_cloaked(hwnd)),
            minimized: hwnd.IsIconic(),
        })
    }

    fn is_cloaked(hwnd: &w::HWND) -> bool {
        matches!(
            hwnd.DwmGetWindowAttribute(w::co::DWMWA::CLOAKED),
            Ok(w::DwmAttr::Cloaked(flags)) if flags.raw() != 0
        )
    }

    fn element_id(element: &UIElement) -> Option<String> {
        let pid = element.get_process_id().ok()?;
        let handle = element.get_native_window_handle().ok()?;
        (!handle.is_invalid()).then(|| format!("{pid}:{handle}"))
    }

    fn cached_process_name(pid: u32) -> String {
        static SYSTEM_CACHE: OnceLock<
            std::sync::Mutex<(System, std::collections::HashMap<u32, String>)>,
        > = OnceLock::new();
        let state = SYSTEM_CACHE.get_or_init(|| {
            std::sync::Mutex::new((System::new(), std::collections::HashMap::new()))
        });
        let Ok(mut guard) = state.lock() else {
            return String::new();
        };
        if let Some(name) = guard.1.get(&pid) {
            return name.clone();
        }
        let (system, cache) = &mut *guard;
        system.refresh_processes(ProcessesToUpdate::Some(&[Pid::from_u32(pid)]), true);
        let name = system
            .process(Pid::from_u32(pid))
            .map(|process| process.name().to_string_lossy().into_owned())
            .unwrap_or_default();
        if !name.is_empty() {
            cache.insert(pid, name.clone());
        }
        name
    }

    fn ui_tree() -> Result<Vec<UiElement>> {
        let automation = UIAutomation::new().map_err(tool_error)?;
        let walker = automation.get_control_view_walker().map_err(tool_error)?;
        let focused = automation.get_focused_element().map_err(tool_error)?;
        let mut root = focused;
        while let Ok(parent) = walker.get_parent(&root) {
            if parent.get_process_id().ok() != root.get_process_id().ok() {
                break;
            }
            root = parent;
        }
        cached_descendants(&automation, &root, None, 1_000)
    }

    fn ui_tree_target(request: &CaptureRequest, limit: usize) -> Result<Vec<UiElement>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let target = resolve_target(request)?;
        let automation = UIAutomation::new().map_err(tool_error)?;
        let mut root = if matches!(
            request.scope,
            CaptureScope::ActiveWindow | CaptureScope::Window
        ) {
            let walker = automation.get_control_view_walker().map_err(tool_error)?;
            let desktop = automation.get_root_element().map_err(tool_error)?;
            let top_level = walker.get_children(&desktop).unwrap_or_default();
            let matches =
                |element: &UIElement| element_id(element).as_deref() == Some(target.id.as_str());
            let target_pid = target
                .id
                .split(':')
                .next()
                .and_then(|pid| pid.parse::<u32>().ok());
            top_level
                .iter()
                .find(|element| matches(element))
                .cloned()
                // A dialog owned by an application window (LibreOffice's, for
                // one) sits under that window in the UI Automation tree, not
                // at the top level.
                .or_else(|| {
                    top_level
                        .iter()
                        .filter(|window| window.get_process_id().ok() == target_pid)
                        .flat_map(|window| walker.get_children(window).unwrap_or_default())
                        .find(|element| matches(element))
                })
                .ok_or_else(|| {
                    PokError::Tool(format!(
                        "capture target {:?} no longer has a top-level UI Automation element",
                        target.id
                    ))
                })?
        } else {
            automation.get_root_element().map_err(tool_error)?
        };
        // Scope the accessibility tree to the focused nested dialog when one
        // exists. Legacy business applications often leave the controls behind
        // a modal visible in UIA; flattening both trees gives small models
        // plausible but incorrect click targets.
        if matches!(
            request.scope,
            CaptureScope::ActiveWindow | CaptureScope::Window
        ) {
            let walker = automation.get_control_view_walker().map_err(tool_error)?;
            if let Ok(mut focused) = automation.get_focused_element() {
                let root_pid = root.get_process_id().ok();
                let target_area =
                    u64::from(target.bounds.width).saturating_mul(u64::from(target.bounds.height));
                let mut modal = None;
                loop {
                    if focused.get_process_id().ok() != root_pid {
                        break;
                    }
                    let kind = focused
                        .get_localized_control_type()
                        .unwrap_or_default()
                        .to_ascii_lowercase();
                    if kind == "window" || kind == "dialog" {
                        if let Ok(bounds) = focused.get_bounding_rectangle() {
                            let bounds = rect(bounds);
                            let area =
                                u64::from(bounds.width).saturating_mul(u64::from(bounds.height));
                            if area > 0
                                && area.saturating_mul(10) < target_area.saturating_mul(9)
                                && !focused.get_name().unwrap_or_default().trim().is_empty()
                            {
                                modal = Some(focused.clone());
                            }
                        }
                    }
                    let Ok(parent) = walker.get_parent(&focused) else {
                        break;
                    };
                    focused = parent;
                }
                if let Some(modal) = modal {
                    root = modal;
                }
            }
        }
        if matches!(
            request.scope,
            CaptureScope::Monitor | CaptureScope::Region | CaptureScope::All
        ) {
            shell_first_descendants(&automation, &root, &target.bounds, limit)
        } else {
            cached_descendants(&automation, &root, Some(&target.bounds), limit)
        }
    }

    fn shell_first_descendants(
        automation: &UIAutomation,
        desktop: &UIElement,
        target_bounds: &Rect,
        limit: usize,
    ) -> Result<Vec<UiElement>> {
        let walker = automation.get_control_view_walker().map_err(tool_error)?;
        let shell_roots = walker
            .get_children(desktop)
            .unwrap_or_default()
            .into_iter()
            .filter(|element| {
                matches!(
                    element.get_classname().unwrap_or_default().as_str(),
                    "Shell_TrayWnd" | "Shell_SecondaryTrayWnd"
                )
            })
            .collect::<Vec<_>>();
        let reserve = limit.div_ceil(4).clamp(32.min(limit), 128.min(limit));
        let mut output = Vec::new();
        for shell in shell_roots {
            let mut shell_elements = cached_descendants(
                automation,
                &shell,
                Some(target_bounds),
                reserve.saturating_sub(output.len()),
            )?;
            for element in &mut shell_elements {
                element.desktop_shell = true;
            }
            output.extend(shell_elements);
            if output.len() >= reserve {
                break;
            }
        }
        let remaining = limit.saturating_sub(output.len());
        if remaining > 0 {
            output.extend(cached_descendants(
                automation,
                desktop,
                Some(target_bounds),
                remaining,
            )?);
        }
        let mut seen = HashSet::new();
        output.retain(|element| {
            seen.insert((
                element.name.clone(),
                element.control_type.clone(),
                element.bounds.x,
                element.bounds.y,
                element.bounds.width,
                element.bounds.height,
            ))
        });
        output.truncate(limit);
        Ok(output)
    }

    fn cached_descendants(
        automation: &UIAutomation,
        root: &UIElement,
        target_bounds: Option<&Rect>,
        limit: usize,
    ) -> Result<Vec<UiElement>> {
        let cache = automation.create_cache_request().map_err(tool_error)?;
        for property in [
            UIProperty::Name,
            UIProperty::LocalizedControlType,
            UIProperty::AutomationId,
            UIProperty::BoundingRectangle,
            UIProperty::IsEnabled,
            UIProperty::IsPassword,
            UIProperty::IsOffscreen,
            UIProperty::IsKeyboardFocusable,
            UIProperty::HasKeyboardFocus,
            UIProperty::SelectionItemIsSelected,
            UIProperty::LegacyIAccessibleState,
            UIProperty::ValueValue,
        ] {
            cache.add_property(property).map_err(tool_error)?;
        }
        let condition = automation
            .create_property_condition(UIProperty::IsOffscreen, false.into(), None)
            .map_err(tool_error)?;
        // Walk one level at a time instead of asking for every descendant at
        // once, and never enter a table or data grid: a spreadsheet exposes
        // its whole sheet as one table, and enumerating it hangs the
        // application (cells are reached through its cell-reference box).
        let mut elements = Vec::new();
        let mut pending = std::collections::VecDeque::from([root.clone()]);
        let mut visited = 0_usize;
        while let Some(parent) = pending.pop_front() {
            if elements.len() >= limit || visited >= limit.saturating_mul(4).max(64) {
                break;
            }
            visited += 1;
            let children = parent
                .find_all_build_cache(TreeScope::Children, &condition, &cache)
                .unwrap_or_default();
            for child in children {
                let grid = child
                    .get_cached_localized_control_type()
                    .map(|kind| kind.to_ascii_lowercase())
                    .is_ok_and(|kind| matches!(kind.as_str(), "table" | "data grid" | "datagrid"));
                if let Some(element) = cached_element(&child)
                    && target_bounds.is_none_or(|bounds| overlaps_rect(&element.bounds, bounds))
                {
                    elements.push(element);
                }
                if !grid {
                    pending.push_back(child);
                }
            }
        }
        elements.truncate(limit);
        Ok(elements)
    }

    fn overlaps_rect(left: &Rect, right: &Rect) -> bool {
        i64::from(left.x) < i64::from(right.x) + i64::from(right.width)
            && i64::from(left.x) + i64::from(left.width) > i64::from(right.x)
            && i64::from(left.y) < i64::from(right.y) + i64::from(right.height)
            && i64::from(left.y) + i64::from(left.height) > i64::from(right.y)
    }

    fn rect_contains(outer: &Rect, inner: &Rect) -> bool {
        inner.x >= outer.x
            && inner.y >= outer.y
            && i64::from(inner.x) + i64::from(inner.width)
                <= i64::from(outer.x) + i64::from(outer.width)
            && i64::from(inner.y) + i64::from(inner.height)
                <= i64::from(outer.y) + i64::from(outer.height)
    }

    fn cached_element(element: &UIElement) -> Option<UiElement> {
        let rectangle = rect(element.get_cached_bounding_rectangle().ok()?);
        if rectangle.width == 0 || rectangle.height == 0 {
            return None;
        }
        let password = element.is_cached_password().unwrap_or(false);
        let value = (!password)
            .then(|| {
                element
                    .get_cached_property_value(UIProperty::ValueValue)
                    .ok()
            })
            .flatten()
            .and_then(|value| TryInto::<String>::try_into(value).ok())
            .filter(|value| !value.is_empty());
        let selected_by_pattern = element
            .get_cached_property_value(UIProperty::SelectionItemIsSelected)
            .ok()
            .and_then(|value| TryInto::<bool>::try_into(value).ok());
        let legacy_state = element
            .get_cached_property_value(UIProperty::LegacyIAccessibleState)
            .ok()
            .and_then(|value| TryInto::<i32>::try_into(value).ok());
        // MSAA STATE_SYSTEM_SELECTED = 0x2 and STATE_SYSTEM_CHECKED = 0x10.
        let selected =
            selected_by_pattern.or_else(|| legacy_state.map(|state| state & (0x2 | 0x10) != 0));
        let focused = element
            .get_cached_property_value(UIProperty::HasKeyboardFocus)
            .ok()
            .and_then(|value| TryInto::<bool>::try_into(value).ok())
            .unwrap_or_else(|| legacy_state.is_some_and(|state| state & 0x4 != 0));
        Some(UiElement {
            name: element.get_cached_name().unwrap_or_default(),
            control_type: element
                .get_cached_localized_control_type()
                .unwrap_or_else(|_| "unknown".into()),
            automation_id: element
                .get_cached_automation_id()
                .ok()
                .filter(|id| !id.is_empty()),
            value,
            bounds: rectangle,
            enabled: element.is_cached_enabled().unwrap_or(false),
            password,
            offscreen: element.is_cached_offscreen().unwrap_or(false),
            keyboard_focusable: element.is_cached_keyboard_focusable().unwrap_or(false),
            clickable_point: None,
            selected,
            focused,
            desktop_shell: false,
        })
    }

    fn rect(rect: uiautomation::types::Rect) -> Rect {
        Rect {
            x: rect.get_left(),
            y: rect.get_top(),
            width: u32::try_from(rect.get_width().max(0)).unwrap_or(0),
            height: u32::try_from(rect.get_height().max(0)).unwrap_or(0),
        }
    }

    fn enigo_mutex() -> &'static std::sync::Mutex<Enigo> {
        static ENIGO_MUTEX: OnceLock<std::sync::Mutex<Enigo>> = OnceLock::new();
        ENIGO_MUTEX.get_or_init(|| {
            std::sync::Mutex::new(Enigo::new(&Settings::default()).expect("enigo init"))
        })
    }

    fn input(action: InputAction) -> Result<()> {
        let Ok(mut enigo) = enigo_mutex().lock() else {
            return Err(PokError::Tool("Enigo mutex lock error".into()));
        };
        match action {
            InputAction::Move { x, y } => enigo
                .move_mouse(x, y, Coordinate::Abs)
                .map_err(tool_error)?,
            InputAction::Click { x, y, button } => {
                enigo
                    .move_mouse(x, y, Coordinate::Abs)
                    .map_err(tool_error)?;
                enigo
                    .button(
                        match button {
                            MouseButton::Left => Button::Left,
                            MouseButton::Right => Button::Right,
                            MouseButton::Middle => Button::Middle,
                        },
                        Direction::Click,
                    )
                    .map_err(tool_error)?;
            }
            InputAction::DoubleClick { x, y, button } => {
                enigo
                    .move_mouse(x, y, Coordinate::Abs)
                    .map_err(tool_error)?;
                let button = match button {
                    MouseButton::Left => Button::Left,
                    MouseButton::Right => Button::Right,
                    MouseButton::Middle => Button::Middle,
                };
                enigo.button(button, Direction::Click).map_err(tool_error)?;
                std::thread::sleep(std::time::Duration::from_millis(80));
                enigo.button(button, Direction::Click).map_err(tool_error)?;
            }
            InputAction::Drag {
                start_x,
                start_y,
                end_x,
                end_y,
                button,
                duration_ms,
            } => {
                let button = match button {
                    MouseButton::Left => Button::Left,
                    MouseButton::Right => Button::Right,
                    MouseButton::Middle => Button::Middle,
                };
                enigo
                    .move_mouse(start_x, start_y, Coordinate::Abs)
                    .map_err(tool_error)?;
                enigo.button(button, Direction::Press).map_err(tool_error)?;
                let steps = (duration_ms / 16).clamp(4, 60);
                let step_delay = std::time::Duration::from_millis((duration_ms / steps).max(1));
                let movement = (|| -> Result<()> {
                    for step in 1..=steps {
                        let x = i64::from(start_x)
                            + (i64::from(end_x) - i64::from(start_x))
                                * i64::try_from(step).unwrap_or(i64::MAX)
                                / i64::try_from(steps).unwrap_or(1);
                        let y = i64::from(start_y)
                            + (i64::from(end_y) - i64::from(start_y))
                                * i64::try_from(step).unwrap_or(i64::MAX)
                                / i64::try_from(steps).unwrap_or(1);
                        enigo
                            .move_mouse(
                                i32::try_from(x).unwrap_or(if x < 0 { i32::MIN } else { i32::MAX }),
                                i32::try_from(y).unwrap_or(if y < 0 { i32::MIN } else { i32::MAX }),
                                Coordinate::Abs,
                            )
                            .map_err(tool_error)?;
                        std::thread::sleep(step_delay);
                    }
                    Ok(())
                })();
                let release = enigo.button(button, Direction::Release).map_err(tool_error);
                movement?;
                release?;
            }
            InputAction::TypeText { text, .. } => enigo.text(&text).map_err(tool_error)?,
            InputAction::Key { key } => input_key_or_shortcut(&mut enigo, &key)?,
            InputAction::Scroll { delta_x, delta_y } => {
                if delta_x != 0 {
                    enigo
                        .scroll(delta_x, Axis::Horizontal)
                        .map_err(tool_error)?;
                }
                if delta_y != 0 {
                    enigo.scroll(delta_y, Axis::Vertical).map_err(tool_error)?;
                }
            }
        }
        Ok(())
    }

    const MAX_TEXT_INPUT_CHARS_PER_BATCH: usize = 8_192;

    fn text_batches(text: &str, max_chars: usize) -> Vec<&str> {
        if text.is_empty() {
            return Vec::new();
        }
        let mut batches = Vec::new();
        let mut start = 0;
        let mut count = 0;
        for (index, _) in text.char_indices() {
            if count == max_chars {
                batches.push(&text[start..index]);
                start = index;
                count = 0;
            }
            count += 1;
        }
        batches.push(&text[start..]);
        batches
    }

    /// What the current keyboard layout types without Shift, and Caps Lock.
    fn current_key_layout() -> super::KeyLayout {
        let char_of = |vk: u32| w::MapVirtualKey(vk, w::co::MAPVK::VK_TO_CHAR) & 0xFFFF;
        super::KeyLayout {
            digits_unshifted: (u32::from(b'0')..=u32::from(b'9')).all(|vk| char_of(vk) == vk),
            latin_letters: (u32::from(b'A')..=u32::from(b'Z')).all(|vk| char_of(vk) == vk),
            caps_lock: w::GetKeyState(w::co::VK::CAPITAL).1,
        }
    }

    /// Type one batch: real key presses for digits, letters and spaces (so
    /// the application's key handlers run), Unicode for everything else.
    fn type_keys(
        enigo: &mut Enigo,
        text: &str,
        layout: super::KeyLayout,
        pause: std::time::Duration,
    ) -> Result<()> {
        // A short gap between real keys lets older applications handle each
        // key message before the next, as with a fast typist.
        let key_gap = pause.min(std::time::Duration::from_millis(8));
        for stroke in super::keystrokes(text, layout) {
            match stroke {
                super::Keystroke::Text(run) => enigo.text(&run).map_err(tool_error)?,
                super::Keystroke::Key { vk, shift } => {
                    if shift {
                        enigo
                            .key(Key::Shift, Direction::Press)
                            .map_err(tool_error)?;
                    }
                    let typed = enigo
                        .key(Key::Other(u32::from(vk)), Direction::Click)
                        .map_err(tool_error);
                    if shift {
                        enigo
                            .key(Key::Shift, Direction::Release)
                            .map_err(tool_error)?;
                    }
                    typed?;
                    if !key_gap.is_zero() {
                        std::thread::sleep(key_gap);
                    }
                }
            }
        }
        Ok(())
    }

    fn input_text(text: &str, inter_key_pause_ms: u64) -> Result<()> {
        if text.contains('\0') {
            return Err(PokError::Tool(
                "text input cannot contain a null character".into(),
            ));
        }
        let pause = std::time::Duration::from_millis(inter_key_pause_ms);
        let Ok(mut enigo) = enigo_mutex().lock() else {
            return Err(PokError::Tool("Enigo mutex lock error".into()));
        };
        let lines = super::typed_lines(text);
        let line_count = lines.len();
        let layout = current_key_layout();
        // Keystrokes go to whatever window is in front. If the user switches
        // windows mid-text, stop rather than type into their window.
        let front = || w::HWND::GetForegroundWindow().map(|hwnd| hwnd.ptr() as usize);
        let typing_into = front();
        let mut typed_chars = 0_usize;
        for (line_index, line) in lines.into_iter().enumerate() {
            let batches = text_batches(line, MAX_TEXT_INPUT_CHARS_PER_BATCH);
            let batch_count = batches.len();
            for (index, batch) in batches.into_iter().enumerate() {
                if front() != typing_into {
                    return Err(PokError::Tool(format!(
                        "typing stopped after {typed_chars} characters because the window in front changed (the user switched windows); check the field and retry"
                    )));
                }
                type_keys(&mut enigo, batch, layout, pause)?;
                typed_chars += batch.chars().count();
                if index + 1 < batch_count && !pause.is_zero() {
                    std::thread::sleep(pause);
                }
            }
            if line_index + 1 < line_count {
                if front() != typing_into {
                    return Err(PokError::Tool(format!(
                        "typing stopped after {typed_chars} characters because the window in front changed (the user switched windows); check the field and retry"
                    )));
                }
                // A new line, not Enter: in chat boxes Enter sends the
                // message, Shift+Enter adds a line (and does so in ordinary
                // text editors too). Sending stays a separate Enter action.
                enigo
                    .key(Key::Shift, Direction::Press)
                    .map_err(tool_error)?;
                let typed = enigo.key(Key::Return, Direction::Click).map_err(tool_error);
                enigo
                    .key(Key::Shift, Direction::Release)
                    .map_err(tool_error)?;
                typed?;
                if !pause.is_zero() {
                    std::thread::sleep(pause);
                }
            }
        }
        Ok(())
    }

    fn focused_text(max_chars: usize) -> Result<Option<String>> {
        let automation = UIAutomation::new().map_err(tool_error)?;
        let element = automation.get_focused_element().map_err(tool_error)?;
        if element.is_password().unwrap_or(false) {
            return Ok(None);
        }
        // A spreadsheet's focused sheet is a table: its "text" is every cell,
        // and building it hangs the application. Only a field is read.
        if element
            .get_localized_control_type()
            .map(|kind| kind.to_ascii_lowercase())
            .is_ok_and(|kind| matches!(kind.as_str(), "table" | "data grid" | "datagrid"))
        {
            return Ok(None);
        }
        if let Ok(value) = element
            .get_pattern::<UIValuePattern>()
            .and_then(|pattern| pattern.get_value())
        {
            return Ok(Some(value.chars().take(max_chars).collect()));
        }
        let Ok(range) = element
            .get_pattern::<UITextPattern>()
            .and_then(|pattern| pattern.get_document_range())
        else {
            return Ok(None);
        };
        let limit = i32::try_from(max_chars).unwrap_or(i32::MAX);
        Ok(range.get_text(limit).ok())
    }

    fn replace_focused_text(text: &str) -> Result<bool> {
        let automation = UIAutomation::new().map_err(tool_error)?;
        let element = automation.get_focused_element().map_err(tool_error)?;
        if element.is_password().unwrap_or(false) || !element.is_enabled().unwrap_or(false) {
            return Ok(false);
        }
        let Ok(pattern) = element.get_pattern::<UIValuePattern>() else {
            return Ok(false);
        };
        match pattern.is_readonly() {
            Ok(false) => Ok(pattern.set_value(text).is_ok()),
            Ok(true) | Err(_) => Ok(false),
        }
    }

    fn parse_key(key: &str) -> Result<Key> {
        let normalized = key.trim().to_ascii_lowercase().replace(['_', '-', ' '], "");
        match normalized.as_str() {
            "enter" | "return" => Ok(Key::Return),
            "escape" | "esc" => Ok(Key::Escape),
            "tab" => Ok(Key::Tab),
            "backspace" => Ok(Key::Backspace),
            "delete" => Ok(Key::Delete),
            "insert" | "ins" => Ok(Key::Insert),
            "space" => Ok(Key::Space),
            "up" | "uparrow" => Ok(Key::UpArrow),
            "down" | "downarrow" => Ok(Key::DownArrow),
            "left" | "leftarrow" => Ok(Key::LeftArrow),
            "right" | "rightarrow" => Ok(Key::RightArrow),
            "pageup" | "pgup" => Ok(Key::PageUp),
            "pagedown" | "pgdn" => Ok(Key::PageDown),
            "home" => Ok(Key::Home),
            "end" => Ok(Key::End),
            "capslock" => Ok(Key::CapsLock),
            "f1" => Ok(Key::F1),
            "f2" => Ok(Key::F2),
            "f3" => Ok(Key::F3),
            "f4" => Ok(Key::F4),
            "f5" => Ok(Key::F5),
            "f6" => Ok(Key::F6),
            "f7" => Ok(Key::F7),
            "f8" => Ok(Key::F8),
            "f9" => Ok(Key::F9),
            "f10" => Ok(Key::F10),
            "f11" => Ok(Key::F11),
            "f12" => Ok(Key::F12),
            "windows" | "win" | "lwin" | "meta" | "super" => Ok(Key::LWin),
            "shift" => Ok(Key::Shift),
            "control" | "ctrl" => Ok(Key::Control),
            "alt" => Ok(Key::Alt),
            other if other.chars().count() == 1 => {
                Ok(Key::Unicode(other.chars().next().expect("one character")))
            }
            _ => Err(PokError::Tool(format!("unsupported key {key:?}"))),
        }
    }

    fn input_key_or_shortcut(enigo: &mut Enigo, value: &str) -> Result<()> {
        let keys = value
            .split('+')
            .map(str::trim)
            .filter(|key| !key.is_empty())
            .map(parse_key)
            .collect::<Result<Vec<_>>>()?;
        if keys.is_empty() {
            return Err(PokError::Tool("key cannot be empty".into()));
        }
        if keys.len() == 1 {
            return enigo.key(keys[0], Direction::Click).map_err(tool_error);
        }
        for key in &keys {
            enigo.key(*key, Direction::Press).map_err(tool_error)?;
        }
        for key in keys.iter().rev() {
            enigo.key(*key, Direction::Release).map_err(tool_error)?;
        }
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn fast_png_encoding_preserves_dimensions_and_pixels() {
            let image = RgbaImage::from_pixel(64, 32, screenshots::image::Rgba([4, 8, 15, 255]));
            let encoded = encode_png(&image, PngFilterType::Sub).expect("PNG encodes");
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(encoded)
                .expect("base64 decodes");
            let decoded = screenshots::image::load_from_memory(&bytes)
                .expect("PNG decodes")
                .into_rgba8();
            assert_eq!(decoded.dimensions(), image.dimensions());
            assert_eq!(decoded.get_pixel(0, 0), image.get_pixel(0, 0));
        }

        #[test]
        fn navigation_keys_accept_common_local_model_spellings() {
            assert_eq!(parse_key("Page_Down").unwrap(), Key::PageDown);
            assert_eq!(parse_key("page-up").unwrap(), Key::PageUp);
            assert_eq!(parse_key("Home").unwrap(), Key::Home);
            assert_eq!(parse_key("End").unwrap(), Key::End);
            assert_eq!(parse_key("F12").unwrap(), Key::F12);
        }

        #[test]
        fn text_batches_preserve_first_character_unicode_and_controls() {
            let text = "Dear I\tcafé 世界 😀\r\nDone";
            let batches = text_batches(text, 7);
            assert_eq!(batches.concat(), text);
            assert_eq!(
                batches.first().and_then(|batch| batch.chars().next()),
                Some('D')
            );
            assert!(batches.iter().all(|batch| batch.chars().count() <= 7));
        }

        #[test]
        fn ordinary_text_is_one_batch() {
            assert_eq!(text_batches("complete ordinary text", 8_192).len(), 1);
        }
    }

    fn tool_error(error: impl std::fmt::Display) -> PokError {
        PokError::Tool(error.to_string())
    }
}
