use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    sync::Arc,
};

use async_trait::async_trait;
use chrono::Utc;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::{
    PokError, Result,
    brain::BrainTool,
    context::ContextBudget,
    memory::MemoryStore,
    platform::DesktopPlatform,
    policy::{ApprovalDecision, ApprovalHandler, Policy, PolicyDecision, RiskClass},
    types::{Observation, Rect},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingSubmission {
    pub text: String,
    pub destination: String,
    pub destination_label: String,
    pub submitted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiedSubmission {
    pub normalized_text: String,
    pub destination: String,
    pub destination_label: String,
    pub evidence: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InputLedger {
    pub pending: Option<PendingSubmission>,
    pub verified: Vec<VerifiedSubmission>,
}

#[derive(Debug, Clone, Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct ArtifactEvidence {
    #[serde(default = "Uuid::new_v4")]
    #[schemars(with = "String")]
    pub evidence_id: Uuid,
    pub path: PathBuf,
    pub operation: String,
    pub artifact_type: String,
    pub integrity: String,
    #[serde(default = "default_artifact_validation_status")]
    pub validation_status: String,
    pub size_bytes: u64,
    pub sha256: String,
    pub modified_at: String,
    #[serde(default)]
    pub structure: Value,
    #[serde(default)]
    pub warnings: Vec<ArtifactWarning>,
    pub observed_at: String,
}

fn default_artifact_validation_status() -> String {
    "needs_review".into()
}

#[derive(Debug, Clone, Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct ArtifactWarning {
    pub severity: String,
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionFileRecord {
    pub sha256: String,
    pub created_by_session: bool,
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Serialize, serde::Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum TaskItemStatus {
    Pending,
    InProgress,
    Completed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct TaskItem {
    #[serde(deserialize_with = "deserialize_task_item_id")]
    #[schemars(with = "String")]
    pub id: String,
    #[serde(alias = "description")]
    pub content: String,
    pub status: TaskItemStatus,
}

fn deserialize_task_item_id<'de, D>(deserializer: D) -> std::result::Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = <Value as serde::Deserialize>::deserialize(deserializer)?;
    match value {
        Value::String(value) => Ok(value),
        Value::Number(value) => Ok(value.to_string()),
        _ => Err(serde::de::Error::custom(
            "task step id must be a string or number",
        )),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveTaskState {
    pub root_request: String,
    pub latest_guidance: Vec<String>,
    pub status: TaskItemStatus,
    pub current_step: String,
    pub steps: Vec<TaskItem>,
    pub plan_updated: bool,
    /// When the request started: files written since then may be its result.
    #[serde(skip, default = "std::time::SystemTime::now")]
    pub started_at: std::time::SystemTime,
}

impl Default for ActiveTaskState {
    fn default() -> Self {
        Self {
            root_request: String::new(),
            latest_guidance: Vec::new(),
            status: TaskItemStatus::InProgress,
            current_step: "Understand and complete the active request".into(),
            steps: Vec::new(),
            plan_updated: false,
            started_at: std::time::SystemTime::now(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FocusedControl {
    pub app: String,
    pub label: String,
    pub control_type: String,
}

#[derive(Debug, Clone)]
pub struct DerivedObservationView {
    pub id: String,
    pub source_observation_id: String,
    pub observation: Observation,
}

#[derive(Debug, Clone)]
pub struct PendingVisualLocalization {
    pub id: String,
    pub source_observation_id: String,
    pub source_observation: Observation,
    pub label: String,
    pub proposed_model_bounds: Rect,
    pub resolved_physical_bounds: Rect,
    pub resolved_physical_point: (i32, i32),
    pub corroboration: String,
}

#[derive(Clone)]
pub struct ToolContext {
    pub session_id: Uuid,
    pub workspace: PathBuf,
    pub data_dir: PathBuf,
    pub artifact_dir: PathBuf,
    pub policy: Policy,
    pub approvals: Arc<dyn ApprovalHandler>,
    pub platform: Arc<dyn DesktopPlatform>,
    pub memory: Arc<MemoryStore>,
    pub session_archive: Arc<crate::session_archive::SessionArchive>,
    pub cancellation: CancellationToken,
    pub pause: Arc<crate::pause::PauseController>,
    pub latest_observation: Arc<Mutex<Option<Observation>>>,
    pub latest_observation_view: Arc<Mutex<Option<DerivedObservationView>>>,
    pub pending_visual_localization: Arc<Mutex<Option<PendingVisualLocalization>>>,
    pub task_hint: Arc<Mutex<String>>,
    pub input_ledger: Arc<Mutex<InputLedger>>,
    pub artifact_evidence: Arc<Mutex<Vec<ArtifactEvidence>>>,
    /// Files and folders the user attached or named in the request: readable
    /// (never writable) even outside the workspace.
    pub attached_paths: Arc<Mutex<Vec<PathBuf>>>,
    pub session_files: Arc<Mutex<BTreeMap<PathBuf, SessionFileRecord>>>,
    pub active_task: Arc<Mutex<ActiveTaskState>>,
    pub focused_control: Arc<Mutex<Option<FocusedControl>>>,
    pub command_timeout_seconds: u64,
    pub input_text_inter_key_pause_ms: u64,
    pub vision_max_edge: u32,
    pub prompt_token_target: u64,
    pub context_budget: Arc<Mutex<ContextBudget>>,
    pub visual_history_limit: usize,
    pub uia_element_limit: usize,
    pub desktop_enrichment_timeout_ms: u64,
    pub desktop_deep_enrichment_timeout_ms: u64,
    pub model_target_limit: usize,
    pub fusion_iou_threshold: f32,
    pub ocr_containment_threshold: f32,
    pub annotate_targets: bool,
    pub approval_cache: Arc<Mutex<BTreeSet<String>>>,
    pub user_guidance_queue: Arc<Mutex<std::collections::VecDeque<String>>>,
    pub questions: Option<Arc<dyn UserQuestionHandler>>,
    pub command_manager: Arc<crate::commands::CommandManager>,
    pub current_tool_call_id: Arc<Mutex<Option<String>>>,
}

#[derive(Debug, Clone, Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct UserQuestionOption {
    pub label: String,
    #[serde(default)]
    pub description: String,
}

#[derive(Debug, Clone, Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct UserQuestion {
    pub id: String,
    #[serde(default)]
    pub header: String,
    pub question: String,
    #[serde(default)]
    pub options: Vec<UserQuestionOption>,
    #[serde(default)]
    pub multi_select: bool,
}

#[derive(Debug, Clone, Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct UserQuestionAnswer {
    pub id: String,
    #[serde(default)]
    pub selected: Vec<String>,
    #[serde(default)]
    pub custom: Option<String>,
}

#[derive(Debug, Clone)]
pub enum UserQuestionOutcome {
    Answered(Vec<UserQuestionAnswer>),
    Dismissed,
}

#[async_trait]
pub trait UserQuestionHandler: Send + Sync {
    async fn ask(&self, questions: Vec<UserQuestion>) -> Result<UserQuestionOutcome>;
}

impl ToolContext {
    pub fn resolve_workspace_path(&self, input: impl AsRef<Path>) -> Result<PathBuf> {
        let input = input.as_ref();
        let joined = if input.is_absolute() {
            input.to_path_buf()
        } else {
            self.workspace.join(input)
        };
        let resolved = if joined.exists() {
            dunce::canonicalize(&joined)?
        } else {
            let parent = joined
                .parent()
                .ok_or_else(|| PokError::OutsideWorkspace(joined.clone()))?;
            dunce::canonicalize(parent)?.join(
                joined
                    .file_name()
                    .ok_or_else(|| PokError::OutsideWorkspace(joined.clone()))?,
            )
        };
        let workspace = dunce::canonicalize(&self.workspace)?;
        if !path_allowed(&self.policy.mode, &resolved, &workspace) {
            return Err(PokError::OutsideWorkspace(resolved));
        }
        Ok(resolved)
    }

    /// A workspace path, or a result of the current request outside it, so
    /// the result can be read back: a file this session already produced or
    /// observed, or one the user's request names by file name that was
    /// written since the request started (a document an application saved
    /// to Documents). Other outside paths stay refused, and exams never read
    /// outside their sandbox.
    pub fn resolve_readable_artifact(&self, input: impl AsRef<Path>) -> Result<PathBuf> {
        let input = input.as_ref();
        self.resolve_workspace_path(input).or_else(|error| {
            if matches!(self.policy.mode, crate::policy::PolicyMode::Exam { .. }) {
                return Err(error);
            }
            dunce::canonicalize(input)
                .ok()
                .filter(|canonical| {
                    self.artifact_evidence
                        .lock()
                        .iter()
                        .any(|evidence| &evidence.path == canonical)
                        || self.requested_and_written_this_task(canonical)
                        || self
                            .attached_paths
                            .lock()
                            .iter()
                            .any(|attached| canonical.starts_with(attached))
                })
                .ok_or(error)
        })
    }

    fn requested_and_written_this_task(&self, path: &Path) -> bool {
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            return false;
        };
        let task = self.active_task.lock();
        let named = task
            .root_request
            .to_lowercase()
            .contains(&name.to_lowercase());
        named
            && std::fs::metadata(path)
                .and_then(|metadata| metadata.modified())
                .is_ok_and(|modified| modified >= task.started_at)
    }

    pub fn write_artifact<T: Serialize>(&self, name: &str, value: &T) -> Result<PathBuf> {
        std::fs::create_dir_all(&self.artifact_dir)?;
        let path = self.artifact_dir.join(name);
        crate::memory::atomic_write(&path, &serde_json::to_vec_pretty(value)?)?;
        Ok(path)
    }
}

fn path_allowed(mode: &crate::policy::PolicyMode, resolved: &Path, workspace: &Path) -> bool {
    matches!(mode, crate::policy::PolicyMode::Autonomous) || resolved.starts_with(workspace)
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn input_schema(&self) -> Value;
    fn risk(&self) -> RiskClass;
    fn risk_for_arguments(&self, _arguments: &Value) -> RiskClass {
        self.risk()
    }
    fn target_path(&self, _arguments: &Value) -> Option<PathBuf> {
        None
    }
    fn approval_key(&self, _arguments: &Value) -> Option<String> {
        None
    }
    async fn execute(&self, arguments: Value, context: &ToolContext) -> Result<Value>;
}

#[derive(Default)]
pub struct ToolRegistry {
    tools: BTreeMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<T: Tool + 'static>(&mut self, tool: T) {
        self.tools.insert(tool.name().into(), Arc::new(tool));
    }

    pub fn definitions(&self) -> Vec<BrainTool> {
        self.tools
            .values()
            .map(|tool| BrainTool {
                name: tool.name().into(),
                description: tool.description().into(),
                input_schema: tool.input_schema(),
            })
            .collect()
    }

    pub fn definitions_for_groups(&self, groups: &BTreeSet<String>) -> Vec<BrainTool> {
        self.tools
            .values()
            .filter(|tool| groups.contains(tool_group(tool.name())))
            .map(|tool| BrainTool {
                name: tool.name().into(),
                description: tool.description().into(),
                input_schema: tool.input_schema(),
            })
            .collect()
    }

    /// A bounded inventory that lets the model know every installed capability
    /// without paying the token cost of every JSON schema on every turn.
    pub fn compact_catalog(&self, active_groups: &BTreeSet<String>) -> String {
        self.tools
            .values()
            .map(|tool| {
                let group = tool_group(tool.name());
                let status = if active_groups.contains(group) {
                    "active"
                } else {
                    "inactive"
                };
                let description = tool
                    .description()
                    .split(['.', '\n'])
                    .next()
                    .unwrap_or(tool.description())
                    .trim();
                let summary = if description.chars().count() > 120 {
                    format!("{}…", description.chars().take(119).collect::<String>())
                } else {
                    description.to_owned()
                };
                format!("- {} [{}; {}]: {}", tool.name(), group, status, summary)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn names(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }

    pub fn input_schema(&self, name: &str) -> Option<Value> {
        self.tools.get(name).map(|tool| tool.input_schema())
    }

    /// Arguments coerced against a registered tool's schema the same way
    /// `call` coerces them (for example "true" to true). Used for tools the
    /// session runs itself, such as `fast_actions`, so a small formatting
    /// slip by the model is not a hard failure.
    pub fn normalized_arguments(&self, name: &str, arguments: Value) -> Value {
        let Some(tool) = self.tools.get(name) else {
            return arguments;
        };
        let schema = crate::brain::reference_free_schema(&tool.input_schema());
        normalize_tool_arguments(arguments, &schema, "$", &mut Vec::new())
    }

    pub async fn call(&self, name: &str, arguments: Value, context: &ToolContext) -> Result<Value> {
        if context.cancellation.is_cancelled() {
            return Err(PokError::Cancelled);
        }
        let tool = self
            .tools
            .get(name)
            .ok_or_else(|| PokError::Tool(format!("unknown tool {name:?}")))?;
        let raw_arguments = arguments;
        let schema = crate::brain::reference_free_schema(&tool.input_schema());
        let mut argument_normalizations = Vec::new();
        let arguments = normalize_tool_arguments(
            raw_arguments.clone(),
            &schema,
            "$",
            &mut argument_normalizations,
        );
        let read_only = tool.risk_for_arguments(&arguments) == RiskClass::ReadOnly;
        let target = tool
            .target_path(&arguments)
            .map(|path| {
                if read_only {
                    context.resolve_readable_artifact(path)
                } else {
                    context.resolve_workspace_path(path)
                }
            })
            .transpose()?;
        match context
            .policy
            .decide(name, tool.risk_for_arguments(&arguments), target.as_deref())
        {
            PolicyDecision::Allow => {}
            PolicyDecision::Deny { reason } => {
                return Err(PokError::PolicyDenied {
                    tool: name.into(),
                    reason,
                });
            }
            PolicyDecision::RequireApproval { reason } => {
                let approval_key = tool
                    .approval_key(&arguments)
                    .or_else(|| Some(name.to_owned()));
                let already_approved = approval_key
                    .as_ref()
                    .is_some_and(|key| context.approval_cache.lock().contains(key));
                if !already_approved {
                    let decision = context.approvals.approve(name, &arguments, &reason).await?;
                    if !apply_approval_decision(
                        decision,
                        approval_key.as_deref(),
                        &context.approval_cache,
                    ) {
                        return Err(PokError::PolicyDenied {
                            tool: name.into(),
                            reason: "user declined approval".into(),
                        });
                    }
                }
            }
        }
        let started = Utc::now();
        let result = tool.execute(arguments.clone(), context).await;
        let record = serde_json::json!({
            "tool": name, "arguments": arguments,
            "raw_arguments": (!argument_normalizations.is_empty()).then_some(raw_arguments),
            "argument_normalizations": argument_normalizations,
            "started_at": started,
            "finished_at": Utc::now(), "ok": result.is_ok(),
            "result": result.as_ref().ok().map(diagnostic_value), "error": result.as_ref().err().map(ToString::to_string),
        });
        let safe_name = format!("tool-{}-{}.json", started.timestamp_millis(), name);
        let _ = context.write_artifact(&safe_name, &record);
        result
    }
}

fn normalize_tool_arguments(
    value: Value,
    schema: &Value,
    path: &str,
    repairs: &mut Vec<String>,
) -> Value {
    let schema = preferred_schema_branch(schema, &value);
    match schema_type(schema) {
        Some("array") => {
            let values = match value {
                Value::Array(values) => values,
                Value::Object(mut object) if object.len() == 1 && object.contains_key("item") => {
                    let item = object.remove("item").expect("item key was checked");
                    repairs.push(format!("{path}: unwrapped XML-style item array"));
                    match item {
                        Value::Array(values) => values,
                        item => vec![item],
                    }
                }
                value => {
                    repairs.push(format!("{path}: wrapped singleton as array"));
                    vec![value]
                }
            };
            let item_schema = schema.get("items").unwrap_or(&Value::Null);
            Value::Array(
                values
                    .into_iter()
                    .enumerate()
                    .map(|(index, value)| {
                        normalize_tool_arguments(
                            value,
                            item_schema,
                            &format!("{path}[{index}]"),
                            repairs,
                        )
                    })
                    .collect(),
            )
        }
        Some("object") => {
            let Value::Object(mut object) = value else {
                return value;
            };
            if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
                let supplied_keys = object.keys().cloned().collect::<Vec<_>>();
                for supplied in supplied_keys {
                    if properties.contains_key(&supplied) {
                        continue;
                    }
                    let supplied_key = normalized_property_name(&supplied);
                    let mut candidates = properties
                        .keys()
                        .filter(|candidate| normalized_property_name(candidate) == supplied_key);
                    let Some(canonical) = candidates.next().cloned() else {
                        continue;
                    };
                    if candidates.next().is_some() || object.contains_key(&canonical) {
                        continue;
                    }
                    if let Some(value) = object.remove(&supplied) {
                        repairs.push(format!("{path}.{supplied}: renamed to {canonical}"));
                        object.insert(canonical, value);
                    }
                }
                for (name, property_schema) in properties {
                    if let Some(value) = object.remove(name) {
                        object.insert(
                            name.clone(),
                            normalize_tool_arguments(
                                value,
                                property_schema,
                                &format!("{path}.{name}"),
                                repairs,
                            ),
                        );
                    }
                }
            }
            Value::Object(object)
        }
        Some("boolean") => match value {
            Value::String(text) if text.eq_ignore_ascii_case("true") => {
                repairs.push(format!("{path}: converted string to boolean"));
                Value::Bool(true)
            }
            Value::String(text) if text.eq_ignore_ascii_case("false") => {
                repairs.push(format!("{path}: converted string to boolean"));
                Value::Bool(false)
            }
            value => value,
        },
        _ => value,
    }
}

fn preferred_schema_branch<'a>(schema: &'a Value, value: &Value) -> &'a Value {
    for key in ["anyOf", "oneOf"] {
        if let Some(branches) = schema.get(key).and_then(Value::as_array) {
            if let Some(branch) = branches.iter().find(|branch| {
                schema_type(branch)
                    .is_some_and(|expected| value_matches_schema_type(value, expected))
            }) {
                return branch;
            }
            if let Some(branch) = branches
                .iter()
                .find(|branch| schema_type(branch).is_some_and(|kind| kind != "null"))
            {
                return branch;
            }
        }
    }
    schema
}

fn schema_type(schema: &Value) -> Option<&str> {
    match schema.get("type")? {
        Value::String(value) => Some(value),
        Value::Array(values) => values
            .iter()
            .filter_map(Value::as_str)
            .find(|value| *value != "null"),
        _ => None,
    }
}

fn value_matches_schema_type(value: &Value, expected: &str) -> bool {
    match expected {
        "array" => {
            value.is_array()
                || value
                    .as_object()
                    .is_some_and(|object| object.len() == 1 && object.contains_key("item"))
        }
        "object" => value.is_object(),
        "boolean" => {
            value.is_boolean()
                || value.as_str().is_some_and(|text| {
                    text.eq_ignore_ascii_case("true") || text.eq_ignore_ascii_case("false")
                })
        }
        "string" => value.is_string(),
        "number" | "integer" => value.is_number(),
        "null" => value.is_null(),
        _ => false,
    }
}

fn normalized_property_name(value: &str) -> String {
    value
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

/// Stable, provider-neutral capability families used to progressively expose
/// schemas. Every installed tool remains discoverable; this only reduces the
/// schema payload sent on turns that do not need it.
pub fn tool_group(name: &str) -> &'static str {
    match name {
        "get_current_time" | "update_task_plan" | "discover_tools" | "fast_actions" => "control",
        "observe_desktop"
        | "capture_screen"
        | "inspect_screen_region"
        | "list_windows"
        | "open_application"
        | "activate_window"
        | "browser_navigate"
        | "query_screen_text"
        | "read_clipboard"
        | "query_window_tree"
        | "click_target"
        | "locate_visual_target"
        | "click_localized"
        | "scroll_view"
        | "scroll_until_text"
        | "type_text"
        | "simulate_input"
        | "execute_action_batch" => "desktop",
        "managed_browser_open"
        | "managed_browser_snapshot"
        | "managed_browser_click"
        | "managed_browser_type"
        | "managed_browser_select"
        | "managed_browser_hover"
        | "managed_browser_scroll" => "browser",
        "run_command" | "manage_command" => "system",
        "list_directory" | "read_file" | "grep_search" | "source_outline" | "write_file"
        | "edit_file" | "undo_edit" | "inspect_artifact" => "coding",
        "memory_search" | "memory_save" | "remember_fact" | "skill_search" | "skill_load"
        | "skill_create" => "memory",
        "context_search" | "context_read" => "archive",
        "search_generated_tools"
        | "invoke_generated_tool"
        | "read_generated_tool_output"
        | "promote_helper_tool"
        | "set_generated_tool_enabled" => "generated",
        "spawn_subagent" => "subagent",
        _ => "other",
    }
}

fn apply_approval_decision(
    decision: ApprovalDecision,
    approval_key: Option<&str>,
    approval_cache: &Mutex<BTreeSet<String>>,
) -> bool {
    match decision {
        ApprovalDecision::Deny => false,
        ApprovalDecision::AllowOnce => true,
        ApprovalDecision::AllowSession => {
            if let Some(key) = approval_key {
                approval_cache.lock().insert(key.to_owned());
            }
            true
        }
    }
}

pub(crate) fn diagnostic_value(value: &Value) -> Value {
    match value {
        Value::Object(object) => Value::Object(
            object
                .iter()
                .map(|(key, value)| {
                    if key == "png_base64" {
                        (
                            key.clone(),
                            Value::String(format!(
                                "<redacted: {} base64 characters; see PNG artifact>",
                                value.as_str().map_or(0, str::len)
                            )),
                        )
                    } else {
                        (key.clone(), diagnostic_value(value))
                    }
                })
                .collect(),
        ),
        Value::Array(values) => Value::Array(values.iter().map(diagnostic_value).collect()),
        _ => value.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn approval_scope_distinguishes_once_from_session() {
        let cache = Mutex::new(BTreeSet::new());
        assert!(apply_approval_decision(
            ApprovalDecision::AllowOnce,
            Some("run_command"),
            &cache,
        ));
        assert!(cache.lock().is_empty());

        assert!(apply_approval_decision(
            ApprovalDecision::AllowSession,
            Some("run_command"),
            &cache,
        ));
        assert!(cache.lock().contains("run_command"));
        assert!(!apply_approval_decision(
            ApprovalDecision::Deny,
            Some("write_file"),
            &cache,
        ));
        assert!(!cache.lock().contains("write_file"));
    }

    #[test]
    fn autonomous_paths_are_broad_while_interactive_paths_remain_scoped() {
        let workspace = Path::new("/workspace");
        let external = Path::new("/outside/document.txt");
        assert!(path_allowed(
            &crate::policy::PolicyMode::Autonomous,
            external,
            workspace
        ));
        assert!(!path_allowed(
            &crate::policy::PolicyMode::Interactive,
            external,
            workspace
        ));
        assert!(path_allowed(
            &crate::policy::PolicyMode::Interactive,
            Path::new("/workspace/file.txt"),
            workspace
        ));
    }

    fn question_schema() -> Value {
        json!({
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                            "options": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "label": {"type": "string"},
                                        "description": {"type": "string"}
                                    }
                                }
                            },
                            "multi_select": {"type": "boolean"}
                        }
                    }
                }
            }
        })
    }

    #[test]
    fn tool_argument_normalization_repairs_provider_item_wrappers_recursively() {
        let raw = json!({
            "questions": {
                "item": [
                    {
                        "question": "Which email?",
                        "multiSelect": "false",
                        "options": {
                            "item": {
                                "label": "Use saved email",
                                "description": "Use the email already in context"
                            }
                        }
                    },
                    {
                        "question": "Include a cover letter?",
                        "options": {
                            "item": [
                                {"label": "Yes (Recommended)", "description": "Include it"},
                                {"label": "No", "description": "Skip it"}
                            ]
                        }
                    }
                ]
            }
        });
        let mut repairs = Vec::new();
        let normalized = normalize_tool_arguments(raw, &question_schema(), "$", &mut repairs);

        assert!(normalized["questions"].is_array());
        assert!(normalized["questions"][0]["options"].is_array());
        assert_eq!(normalized["questions"][0]["multi_select"], false);
        assert!(repairs.iter().any(|repair| repair.contains("item array")));
        assert!(repairs.iter().any(|repair| repair.contains("multiSelect")));
        assert!(repairs.iter().any(|repair| repair.contains("boolean")));
    }

    #[test]
    fn canonical_tool_arguments_are_left_unchanged() {
        let raw = json!({
            "questions": [{
                "question": "Continue?",
                "options": [{"label": "Yes", "description": "Continue"}],
                "multi_select": false
            }]
        });
        let mut repairs = Vec::new();
        let normalized =
            normalize_tool_arguments(raw.clone(), &question_schema(), "$", &mut repairs);
        assert_eq!(normalized, raw);
        assert!(repairs.is_empty());
    }

    #[test]
    fn tool_argument_normalization_does_not_guess_ambiguous_values() {
        let raw = json!({
            "questions": [{
                "question": "Continue?",
                "options": [{"label": "Yes"}],
                "multi_select": "sometimes",
                "unexpectedField": "preserve me"
            }]
        });
        let mut repairs = Vec::new();
        let normalized = normalize_tool_arguments(raw, &question_schema(), "$", &mut repairs);
        assert_eq!(normalized["questions"][0]["multi_select"], "sometimes");
        assert_eq!(normalized["questions"][0]["unexpectedField"], "preserve me");
    }

    #[test]
    fn captured_question_payload_variants_all_reach_canonical_shape() {
        let question = || {
            json!({
                "question": "What should I use?",
                "options": {"item": {"label": "Default", "description": "Use the default"}},
                "multiSelect": "false"
            })
        };
        let payloads = [
            json!({"questions": {"item": [question()]}}),
            json!({"questions": {"item": question()}}),
            json!({"questions": [question()]}),
            json!({"questions": [question(), question(), question(), question(), question()]}),
        ];

        for (index, payload) in payloads.into_iter().enumerate() {
            let mut repairs = Vec::new();
            let normalized =
                normalize_tool_arguments(payload, &question_schema(), "$", &mut repairs);
            let questions = normalized["questions"].as_array().unwrap();
            assert_eq!(questions.len(), if index == 3 { 5 } else { 1 });
            for question in questions {
                assert!(question["options"].is_array());
                assert_eq!(question["multi_select"], false);
            }
        }
    }
}
