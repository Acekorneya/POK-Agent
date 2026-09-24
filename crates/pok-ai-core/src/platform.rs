use std::{sync::Arc, time::Instant};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    PokError, Result,
    types::{
        CaptureRequest, CaptureTarget, DesktopCapture, InputAction, MonitorInfo, Observation,
        OcrBlock, Screenshot, UiElement, WindowInfo,
    },
};

/// A deliberately small, read-only view of the interactive desktop. It is
/// safe to sample while a run is active and contains no screen or OCR data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesktopActivitySnapshot {
    pub foreground_window: Option<WindowInfo>,
    /// Milliseconds since the operating system last observed user input. A
    /// platform may return `None` when it cannot distinguish activity.
    pub last_user_input_ms: Option<u64>,
    /// Milliseconds since the last input from a physical mouse or keyboard,
    /// excluding input the agent itself injected. `None` when the platform
    /// cannot tell the two apart.
    #[serde(default)]
    pub last_physical_input_ms: Option<u64>,
    pub captured_at: DateTime<Utc>,
}

/// A cursor-free control action performed through the platform's
/// accessibility patterns instead of synthesized mouse input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PatternAction {
    /// What a left click does: invoke a button, toggle a check box, select
    /// a list or tab item, or expand/collapse a tree node or menu.
    Activate,
    /// What a double click does on an item: open it (the item's default
    /// action), for example a folder or file in File Explorer.
    Open,
    /// Run the control's Invoke action only; used when selecting an item
    /// changed nothing (a gallery tile opens on click, it is not selected).
    Invoke,
}

/// The control a pattern action applies to, as it appeared in the observation
/// that authorized the action. Coordinates are physical desktop pixels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternActionRequest {
    pub window_id: String,
    pub name: String,
    pub control_type: String,
    pub bounds: crate::types::Rect,
    pub point: (i32, i32),
    pub action: PatternAction,
}

/// A cursor-free scroll of the scrollable area at `point` inside a window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternScrollRequest {
    pub window_id: String,
    pub point: (i32, i32),
    /// -1 scrolls up, 1 down, 0 leaves the vertical axis alone.
    pub vertical: i8,
    /// -1 scrolls left, 1 right, 0 leaves the horizontal axis alone.
    pub horizontal: i8,
    /// A page (large increment) instead of a small step.
    pub page: bool,
}

/// Text entered into a field through its Value pattern, without the
/// keyboard. `name` may be empty: text fields are often unnamed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternTextRequest {
    pub window_id: String,
    pub name: String,
    pub bounds: crate::types::Rect,
    pub point: (i32, i32),
    pub text: String,
    /// Add to the existing text instead of replacing it.
    pub append: bool,
}

/// The field's text before and after a pattern text entry, read back from
/// the control itself.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternTextOutcome {
    pub before: String,
    pub after: String,
}

/// Which accessibility pattern carried out a pattern action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternActionOutcome {
    pub pattern: String,
}

#[async_trait]
// Public trait contract: a few platform methods legitimately take many
// parameters; restructuring the public API to satisfy the lint would churn
// every implementor for no benefit.
#[allow(clippy::too_many_arguments)]
pub trait DesktopPlatform: Send + Sync {
    async fn capture_screens(&self) -> Result<Vec<Screenshot>>;
    /// Launch an installed application by bare name (e.g. `notepad`). The
    /// platform rejects names containing path separators or shell
    /// metacharacters; launching is a process-execution action and is
    /// approval-gated by policy outside autonomous mode.
    async fn launch_application(&self, _name: &str) -> Result<()> {
        Err(PokError::Tool(
            "launching applications is unsupported on this platform".into(),
        ))
    }
    async fn capture_target(&self, request: &CaptureRequest) -> Result<DesktopCapture> {
        let screenshots = self.capture_screens().await?;
        let foreground = self.foreground_window().await?;
        let bounds = request
            .region
            .clone()
            .or_else(|| foreground.as_ref().map(|window| window.bounds.clone()))
            .or_else(|| screenshots.first().map(|shot| shot.monitor.bounds.clone()))
            .unwrap_or(crate::types::Rect {
                x: 0,
                y: 0,
                width: 0,
                height: 0,
            });
        Ok(DesktopCapture {
            target: CaptureTarget {
                scope: request.scope.clone(),
                id: if matches!(request.scope, crate::types::CaptureScope::Region) {
                    "screen-region".into()
                } else {
                    foreground
                        .as_ref()
                        .map_or_else(String::new, |window| window.id.clone())
                },
                title: if matches!(request.scope, crate::types::CaptureScope::Region) {
                    "Screen region".into()
                } else {
                    foreground
                        .as_ref()
                        .map_or_else(String::new, |window| window.title.clone())
                },
                process_name: foreground
                    .as_ref()
                    .map_or_else(String::new, |window| window.process_name.clone()),
                bounds,
            },
            screenshots,
            timings_ms: Default::default(),
        })
    }
    async fn query_ocr(&self) -> Result<Vec<OcrBlock>>;
    async fn query_ocr_target(&self, _request: &CaptureRequest) -> Result<Vec<OcrBlock>> {
        self.query_ocr().await
    }
    async fn query_ocr_capture(
        &self,
        capture: &DesktopCapture,
        request: &CaptureRequest,
    ) -> Result<Vec<OcrBlock>> {
        let _ = capture;
        self.query_ocr_target(request).await
    }
    async fn query_ui_tree(&self) -> Result<Vec<UiElement>>;
    async fn query_ui_tree_target(&self, _request: &CaptureRequest) -> Result<Vec<UiElement>> {
        self.query_ui_tree().await
    }
    async fn query_ui_tree_target_limited(
        &self,
        request: &CaptureRequest,
        limit: usize,
    ) -> Result<Vec<UiElement>> {
        let mut elements = self.query_ui_tree_target(request).await?;
        elements.truncate(limit);
        Ok(elements)
    }
    async fn list_monitors(&self) -> Result<Vec<MonitorInfo>> {
        Ok(self
            .capture_screens()
            .await?
            .into_iter()
            .map(|screenshot| screenshot.monitor)
            .collect())
    }
    async fn list_windows(&self) -> Result<Vec<WindowInfo>> {
        Ok(self.foreground_window().await?.into_iter().collect())
    }
    async fn activate_window(&self, _window_id: &str) -> Result<WindowInfo> {
        self.foreground_window().await?.ok_or_else(|| {
            crate::PokError::Tool("window activation is unavailable on this platform".into())
        })
    }
    async fn foreground_window(&self) -> Result<Option<WindowInfo>>;
    /// Really bring a window to the front (focus, no mouse or keyboard),
    /// even where `activate_window` only chooses the window to work in.
    async fn bring_to_front(&self, window_id: &str) -> Result<WindowInfo> {
        self.activate_window(window_id).await
    }
    async fn desktop_activity_snapshot(&self) -> Result<DesktopActivitySnapshot> {
        Ok(DesktopActivitySnapshot {
            foreground_window: self.foreground_window().await?,
            last_user_input_ms: None,
            last_physical_input_ms: None,
            captured_at: Utc::now(),
        })
    }
    /// Carry out `request` on the matching control without moving the
    /// cursor. `Ok(None)` means the platform or control supports no suitable
    /// pattern and the caller should fall back to physical input; an error
    /// means the control could not be matched safely and nothing was done.
    async fn perform_pattern_action(
        &self,
        _request: &PatternActionRequest,
    ) -> Result<Option<PatternActionOutcome>> {
        Ok(None)
    }
    /// Set a native text field's contents without the keyboard. `Ok(None)`
    /// means the field does not accept it (rich editors, web content,
    /// password fields) and the caller should type instead.
    async fn perform_pattern_text(
        &self,
        _request: &PatternTextRequest,
    ) -> Result<Option<PatternTextOutcome>> {
        Ok(None)
    }
    /// Scroll the area at `request.point` without the mouse wheel. `Ok(None)`
    /// means nothing scrollable was found there and the caller should use
    /// physical scrolling.
    async fn perform_pattern_scroll(
        &self,
        _request: &PatternScrollRequest,
    ) -> Result<Option<PatternActionOutcome>> {
        Ok(None)
    }
    async fn cursor_position(&self) -> Result<Option<(i32, i32)>>;
    async fn simulate_input(&self, action: &InputAction) -> Result<()>;
    async fn simulate_text(&self, text: &str, _inter_key_pause_ms: u64) -> Result<()> {
        self.simulate_input(&InputAction::TypeText {
            text: text.to_owned(),
            replace_existing: false,
        })
        .await
    }
    async fn focused_text(&self, _max_chars: usize) -> Result<Option<String>> {
        Ok(None)
    }
    async fn replace_focused_text(&self, _text: &str) -> Result<bool> {
        Ok(false)
    }

    async fn observe(
        &self,
        request: &CaptureRequest,
        include_ocr: bool,
        include_ui: bool,
        ui_element_limit: usize,
        enrichment_timeout: std::time::Duration,
    ) -> Result<Observation> {
        let total_started = Instant::now();
        let capture_started = Instant::now();
        let capture = self.capture_target(request).await?;
        let capture_ms = elapsed_ms(capture_started);
        let mut observation = self
            .observe_capture(
                capture,
                request,
                include_ocr,
                include_ui,
                ui_element_limit,
                capture_ms,
                enrichment_timeout,
            )
            .await?;
        observation
            .timings_ms
            .insert("total".into(), elapsed_ms(total_started));
        Ok(observation)
    }

    async fn observe_capture(
        &self,
        capture: DesktopCapture,
        request: &CaptureRequest,
        include_ocr: bool,
        include_ui: bool,
        ui_element_limit: usize,
        capture_ms: u64,
        enrichment_timeout: std::time::Duration,
    ) -> Result<Observation> {
        // Native capture and WinRT OCR both acquire display resources. Capture
        // has already completed before these enrichment queries begin.
        let enrichment_started = Instant::now();
        let (foreground_window, cursor, ocr_result, ui_result) = tokio::join!(
            self.foreground_window(),
            self.cursor_position(),
            async {
                let started = Instant::now();
                if include_ocr {
                    match tokio::time::timeout(
                        enrichment_timeout,
                        self.query_ocr_capture(&capture, request),
                    )
                    .await
                    {
                        Ok(result) => (result, elapsed_ms(started), false),
                        Err(_) => (
                            Err(crate::PokError::Tool(format!(
                                "OCR enrichment exceeded {} ms",
                                enrichment_timeout.as_millis()
                            ))),
                            elapsed_ms(started),
                            true,
                        ),
                    }
                } else {
                    (Ok(Vec::new()), elapsed_ms(started), false)
                }
            },
            async {
                let started = Instant::now();
                if include_ui {
                    match tokio::time::timeout(
                        enrichment_timeout,
                        self.query_ui_tree_target_limited(request, ui_element_limit),
                    )
                    .await
                    {
                        Ok(result) => (result, elapsed_ms(started), false),
                        Err(_) => (
                            Err(crate::PokError::Tool(format!(
                                "UI Automation enrichment exceeded {} ms",
                                enrichment_timeout.as_millis()
                            ))),
                            elapsed_ms(started),
                            true,
                        ),
                    }
                } else {
                    (Ok(Vec::new()), elapsed_ms(started), false)
                }
            },
        );
        let mut warnings = Vec::new();
        let ocr = ocr_result.0.unwrap_or_else(|error| {
            warnings.push(format!("OCR unavailable: {error}"));
            Vec::new()
        });
        let ui_elements = ui_result.0.unwrap_or_else(|error| {
            warnings.push(format!("UI Automation unavailable: {error}"));
            Vec::new()
        });
        let mut timings_ms = std::collections::BTreeMap::from([
            ("capture".into(), capture_ms),
            ("ocr".into(), ocr_result.1),
            ("uia".into(), ui_result.1),
            ("ocr_timed_out".into(), u64::from(ocr_result.2)),
            ("uia_timed_out".into(), u64::from(ui_result.2)),
            (
                "enrichment_timeout".into(),
                u64::try_from(enrichment_timeout.as_millis()).unwrap_or(u64::MAX),
            ),
        ]);
        timings_ms.extend(
            capture
                .timings_ms
                .iter()
                .map(|(name, elapsed)| (format!("capture_{name}"), *elapsed)),
        );
        timings_ms.insert(
            "total".into(),
            capture_ms.saturating_add(elapsed_ms(enrichment_started)),
        );
        Ok(Observation {
            version: Uuid::new_v4(),
            captured_at: Utc::now(),
            foreground_window: foreground_window?,
            target: Some(capture.target),
            cursor: cursor?,
            screenshots: capture.screenshots,
            ocr,
            ui_elements,
            targets: Vec::new(),
            timings_ms,
            warnings,
        })
    }
}

fn elapsed_ms(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

#[derive(Default)]
pub struct MockDesktop {
    observation: Mutex<Option<Observation>>,
    actions: Mutex<Vec<InputAction>>,
    /// When set, controls accept pattern actions and they are recorded here
    /// instead of producing synthesized input.
    pattern_support: Mutex<Option<Vec<PatternActionRequest>>>,
    /// Scripted `last_physical_input_ms` readings, consumed one per activity
    /// snapshot; the last value repeats.
    physical_input: Mutex<Vec<u64>>,
    pattern_scrolls: Mutex<Vec<PatternScrollRequest>>,
    /// Text of the one field that accepts pattern text entry, when enabled.
    pattern_text_field: Mutex<Option<String>>,
    /// Top-level windows beyond the observed foreground one.
    extra_windows: Mutex<Vec<WindowInfo>>,
    /// The real foreground window once changed by `activate_window`.
    active_window: Mutex<Option<WindowInfo>>,
    activations: Mutex<Vec<String>>,
}

impl MockDesktop {
    pub fn new(observation: Observation) -> Arc<Self> {
        Arc::new(Self {
            observation: Mutex::new(Some(observation)),
            ..Self::default()
        })
    }

    pub fn actions(&self) -> Vec<InputAction> {
        self.actions.lock().clone()
    }

    /// Let controls accept pattern actions (recorded, not simulated).
    pub fn enable_pattern_actions(&self) {
        *self.pattern_support.lock() = Some(Vec::new());
    }

    pub fn pattern_actions(&self) -> Vec<PatternActionRequest> {
        self.pattern_support.lock().clone().unwrap_or_default()
    }

    pub fn pattern_scrolls(&self) -> Vec<PatternScrollRequest> {
        self.pattern_scrolls.lock().clone()
    }

    /// Give targets a field that accepts pattern text entry, starting with
    /// `text`.
    pub fn enable_pattern_text(&self, text: &str) {
        *self.pattern_text_field.lock() = Some(text.to_owned());
    }

    pub fn pattern_text(&self) -> Option<String> {
        self.pattern_text_field.lock().clone()
    }

    /// Script the physical-input idle readings of successive snapshots.
    pub fn script_physical_input(&self, readings: Vec<u64>) {
        *self.physical_input.lock() = readings;
    }

    /// Add top-level windows that exist beside the observed foreground one.
    pub fn add_windows(&self, windows: Vec<WindowInfo>) {
        self.extra_windows.lock().extend(windows);
    }

    /// Windows brought to the real foreground, in order.
    pub fn activations(&self) -> Vec<String> {
        self.activations.lock().clone()
    }

    /// Make `window_id` the real foreground, as a user switching windows would.
    pub fn switch_foreground(&self, window_id: &str) {
        let window = self
            .all_windows()
            .into_iter()
            .find(|window| window.id == window_id);
        *self.active_window.lock() = window;
    }

    fn all_windows(&self) -> Vec<WindowInfo> {
        let mut windows = self
            .observation
            .lock()
            .as_ref()
            .and_then(|observation| observation.foreground_window.clone())
            .into_iter()
            .collect::<Vec<_>>();
        for window in self.extra_windows.lock().iter() {
            if !windows.iter().any(|known| known.id == window.id) {
                windows.push(window.clone());
            }
        }
        windows
    }
}

#[async_trait]
impl DesktopPlatform for MockDesktop {
    async fn capture_screens(&self) -> Result<Vec<Screenshot>> {
        Ok(self
            .observation
            .lock()
            .as_ref()
            .map(|o| o.screenshots.clone())
            .unwrap_or_default())
    }
    async fn query_ocr(&self) -> Result<Vec<OcrBlock>> {
        Ok(self
            .observation
            .lock()
            .as_ref()
            .map(|o| o.ocr.clone())
            .unwrap_or_default())
    }
    async fn query_ui_tree(&self) -> Result<Vec<UiElement>> {
        Ok(self
            .observation
            .lock()
            .as_ref()
            .map(|o| o.ui_elements.clone())
            .unwrap_or_default())
    }
    async fn foreground_window(&self) -> Result<Option<WindowInfo>> {
        if let Some(active) = self.active_window.lock().clone() {
            return Ok(Some(active));
        }
        Ok(self
            .observation
            .lock()
            .as_ref()
            .and_then(|o| o.foreground_window.clone()))
    }
    async fn list_windows(&self) -> Result<Vec<WindowInfo>> {
        Ok(self.all_windows())
    }
    async fn activate_window(&self, window_id: &str) -> Result<WindowInfo> {
        let window = self
            .all_windows()
            .into_iter()
            .find(|window| window.id == window_id)
            .ok_or_else(|| PokError::Tool(format!("unknown window {window_id:?}")))?;
        self.activations.lock().push(window_id.to_owned());
        *self.active_window.lock() = Some(window.clone());
        Ok(window)
    }
    async fn cursor_position(&self) -> Result<Option<(i32, i32)>> {
        Ok(self.observation.lock().as_ref().and_then(|o| o.cursor))
    }
    async fn simulate_input(&self, action: &InputAction) -> Result<()> {
        self.actions.lock().push(action.clone());
        Ok(())
    }
    async fn desktop_activity_snapshot(&self) -> Result<DesktopActivitySnapshot> {
        let last_physical_input_ms = {
            let mut readings = self.physical_input.lock();
            if readings.len() > 1 {
                Some(readings.remove(0))
            } else {
                readings.first().copied()
            }
        };
        Ok(DesktopActivitySnapshot {
            foreground_window: self.foreground_window().await?,
            last_user_input_ms: last_physical_input_ms,
            last_physical_input_ms,
            captured_at: Utc::now(),
        })
    }
    async fn perform_pattern_action(
        &self,
        request: &PatternActionRequest,
    ) -> Result<Option<PatternActionOutcome>> {
        let mut support = self.pattern_support.lock();
        let Some(recorded) = support.as_mut() else {
            return Ok(None);
        };
        recorded.push(request.clone());
        Ok(Some(PatternActionOutcome {
            pattern: match request.action {
                PatternAction::Activate | PatternAction::Invoke => "invoke".into(),
                PatternAction::Open => "default_action".into(),
            },
        }))
    }
    async fn perform_pattern_text(
        &self,
        request: &PatternTextRequest,
    ) -> Result<Option<PatternTextOutcome>> {
        let mut field = self.pattern_text_field.lock();
        let Some(before) = field.clone() else {
            return Ok(None);
        };
        let after = if request.append {
            format!("{before}{}", request.text)
        } else {
            request.text.clone()
        };
        *field = Some(after.clone());
        Ok(Some(PatternTextOutcome { before, after }))
    }
    async fn perform_pattern_scroll(
        &self,
        request: &PatternScrollRequest,
    ) -> Result<Option<PatternActionOutcome>> {
        if self.pattern_support.lock().is_none() {
            return Ok(None);
        }
        self.pattern_scrolls.lock().push(request.clone());
        Ok(Some(PatternActionOutcome {
            pattern: "scroll".into(),
        }))
    }
}
