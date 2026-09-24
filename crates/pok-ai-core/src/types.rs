use std::{collections::BTreeMap, path::PathBuf};

use chrono::{DateTime, Utc};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

impl Rect {
    pub fn contains(&self, x: i32, y: i32) -> bool {
        x >= self.x
            && y >= self.y
            && i64::from(x) < i64::from(self.x) + i64::from(self.width)
            && i64::from(y) < i64::from(self.y) + i64::from(self.height)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MonitorInfo {
    pub id: String,
    pub bounds: Rect,
    pub scale_factor: f64,
    pub primary: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Screenshot {
    pub monitor: MonitorInfo,
    pub png_base64: String,
    #[serde(default, skip_serializing, skip_deserializing)]
    pub source_png_base64: Option<String>,
    pub model_width: u32,
    pub model_height: u32,
    pub captured_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CaptureScope {
    #[default]
    ActiveWindow,
    Window,
    Monitor,
    /// A bounded physical screen rectangle derived from a fresh observation.
    Region,
    All,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CaptureRequest {
    #[serde(default)]
    pub scope: CaptureScope,
    #[serde(default)]
    pub window_id: Option<String>,
    #[serde(default)]
    pub monitor_id: Option<String>,
    #[serde(default)]
    pub region: Option<Rect>,
    pub max_edge: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CaptureTarget {
    pub scope: CaptureScope,
    pub id: String,
    pub title: String,
    pub process_name: String,
    pub bounds: Rect,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DesktopCapture {
    pub target: CaptureTarget,
    pub screenshots: Vec<Screenshot>,
    #[serde(default)]
    pub timings_ms: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct OcrBlock {
    pub text: String,
    pub bounds: Rect,
    pub confidence: Option<f32>,
    #[serde(default)]
    pub selected: Option<bool>,
    #[serde(default)]
    pub variant: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct UiElement {
    pub name: String,
    pub control_type: String,
    pub automation_id: Option<String>,
    pub value: Option<String>,
    pub bounds: Rect,
    pub enabled: bool,
    pub password: bool,
    #[serde(default)]
    pub offscreen: bool,
    #[serde(default)]
    pub keyboard_focusable: bool,
    #[serde(default)]
    pub clickable_point: Option<(i32, i32)>,
    #[serde(default)]
    pub selected: Option<bool>,
    #[serde(default)]
    pub focused: bool,
    /// True when the native adapter resolved this element from a desktop Shell root.
    #[serde(default)]
    pub desktop_shell: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TargetSource {
    Uia,
    Ocr,
    UiaOcr,
    Visual,
    VisualOcr,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct InteractionTarget {
    pub id: String,
    pub name: String,
    pub control_type: String,
    pub bounds: Rect,
    pub source: TargetSource,
    pub confidence: Option<f32>,
    pub enabled: bool,
    pub actionable: bool,
    pub click_point: Option<(i32, i32)>,
    #[serde(default)]
    pub selected: Option<bool>,
    #[serde(default)]
    pub focused: bool,
    #[serde(default)]
    pub desktop_shell: bool,
    #[serde(default)]
    pub grounding_variant: Option<String>,
    #[serde(default)]
    pub rank_score: u16,
    #[serde(default)]
    pub rank_reasons: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct WindowInfo {
    #[serde(default)]
    pub id: String,
    pub title: String,
    pub process_name: String,
    pub bounds: Rect,
    pub elevated: bool,
    #[serde(default = "default_true")]
    pub visible: bool,
    #[serde(default)]
    pub minimized: bool,
}

const fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Observation {
    pub version: Uuid,
    pub captured_at: DateTime<Utc>,
    pub foreground_window: Option<WindowInfo>,
    pub target: Option<CaptureTarget>,
    pub cursor: Option<(i32, i32)>,
    pub screenshots: Vec<Screenshot>,
    pub ocr: Vec<OcrBlock>,
    pub ui_elements: Vec<UiElement>,
    #[serde(default)]
    pub targets: Vec<InteractionTarget>,
    #[serde(default)]
    pub timings_ms: BTreeMap<String, u64>,
    #[serde(default)]
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InputAction {
    Move {
        x: i32,
        y: i32,
    },
    Click {
        x: i32,
        y: i32,
        button: MouseButton,
    },
    DoubleClick {
        x: i32,
        y: i32,
        button: MouseButton,
    },
    Drag {
        start_x: i32,
        start_y: i32,
        end_x: i32,
        end_y: i32,
        button: MouseButton,
        duration_ms: u64,
    },
    TypeText {
        text: String,
        #[serde(default)]
        replace_existing: bool,
    },
    Key {
        key: String,
    },
    Scroll {
        delta_x: i32,
        delta_y: i32,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MouseButton {
    Left,
    Right,
    Middle,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionEvent {
    pub timestamp: DateTime<Utc>,
    pub session_id: Uuid,
    pub kind: String,
    pub payload: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RunMetrics {
    pub turns: u32,
    pub tool_calls: u32,
    pub malformed_tool_calls: u32,
    pub approvals: u32,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub elapsed_ms: u64,
    pub extras: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSummary {
    pub session_id: Uuid,
    pub answer: String,
    pub metrics: RunMetrics,
    pub artifact_dir: PathBuf,
    #[serde(default)]
    pub completion_status: CompletionStatus,
    #[serde(default)]
    pub warnings: Vec<CompletionWarning>,
    #[serde(default)]
    pub deliverables: Vec<ArtifactReference>,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CompletionStatus {
    #[default]
    Completed,
    CompletedWithWarnings,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CompletionWarning {
    pub code: String,
    pub message: String,
    pub artifact_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArtifactReference {
    pub path: PathBuf,
    pub artifact_type: String,
    pub sha256: String,
    pub validation_status: String,
}
