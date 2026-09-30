use std::{
    path::{Path, PathBuf},
    process::Stdio,
    time::Duration,
};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::{io::AsyncWriteExt, process::Command};
use uuid::Uuid;

use crate::{
    PokError, Result,
    memory::atomic_write,
    policy::{PolicyMode, RiskClass},
    process_window,
    tool::{Tool, ToolContext, ToolRegistry},
};

const MAX_OUTPUT: usize = 1024 * 1024;
const DIRECT_OUTPUT_LIMIT: usize = 64 * 1024;
const MAX_DIAGNOSTIC_STREAM: usize = 8 * 1024;
const MAX_ARTIFACT_PAGE: usize = 64 * 1024;

pub fn register_generated_tool_tools(registry: &mut ToolRegistry) {
    registry.register(PromoteHelperTool);
    registry.register(SearchGeneratedToolsTool);
    registry.register(InvokeGeneratedToolTool);
    registry.register(ReadGeneratedToolOutputTool);
    registry.register(SetGeneratedToolEnabledTool);
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ScriptRuntime {
    Powershell,
    Python,
    Node,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedToolManifest {
    pub schema_version: u32,
    pub name: String,
    pub description: String,
    pub runtime: ScriptRuntime,
    pub entrypoint: String,
    pub input_schema: Value,
    pub capabilities: Vec<String>,
    pub timeout_seconds: u64,
    pub max_output_bytes: usize,
    pub source_sha256: String,
    pub enabled: bool,
    pub version: u32,
    pub successes: u64,
    pub failures: u64,
    pub consecutive_protocol_failures: u32,
    #[serde(default)]
    pub dependencies: Vec<String>,
    #[serde(default)]
    pub environment_sha256: String,
    #[serde(default)]
    pub validation: GeneratedToolValidation,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GeneratedToolValidation {
    pub validated_at: Option<DateTime<Utc>>,
    pub smoke_tests: u32,
    pub assertions: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct GeneratedToolSmokeTest {
    pub input: Value,
    #[serde(default)]
    pub assertions: Vec<OutputAssertion>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OutputAssertion {
    Exists { pointer: String },
    NonEmpty { pointer: String },
    Type { pointer: String, expected: String },
    MaxMissingRatio { pointer: String, maximum: f64 },
}

#[derive(Debug, Clone, Serialize)]
pub struct GeneratedToolSummary {
    pub name: String,
    pub description: String,
    pub runtime: ScriptRuntime,
    pub capabilities: Vec<String>,
    pub enabled: bool,
    pub version: u32,
    pub successes: u64,
    pub failures: u64,
    pub schema_version: u32,
    pub isolated_environment: bool,
    pub validated_at: Option<DateTime<Utc>>,
    pub smoke_tests: u32,
    pub assertions: u32,
    pub available: bool,
    pub unavailable_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedToolCandidate {
    pub id: String,
    pub helper_path: PathBuf,
    pub task: String,
    pub runtime: ScriptRuntime,
    pub verified_command: String,
    pub source_sha256: String,
    pub created_at: DateTime<Utc>,
}

pub fn list_generated_tools(
    data_dir: &Path,
    workspace: &Path,
) -> Result<Vec<GeneratedToolSummary>> {
    let root = project_root_for(data_dir, workspace)?;
    Ok(std::fs::read_dir(root)?
        .filter_map(std::result::Result::ok)
        .filter_map(|entry| {
            load_manifest(&entry.path())
                .ok()
                .map(|manifest| (entry.path(), manifest))
        })
        .map(|(dir, manifest)| {
            let health = generated_tool_health(&dir, &manifest);
            GeneratedToolSummary {
                name: format!("generated__{}", manifest.name),
                description: manifest.description,
                runtime: manifest.runtime,
                capabilities: manifest.capabilities,
                enabled: manifest.enabled,
                version: manifest.version,
                successes: manifest.successes,
                failures: manifest.failures,
                schema_version: manifest.schema_version,
                isolated_environment: manifest.schema_version >= 2
                    && !manifest.environment_sha256.is_empty(),
                validated_at: manifest.validation.validated_at,
                smoke_tests: manifest.validation.smoke_tests,
                assertions: manifest.validation.assertions,
                available: manifest.enabled && health.is_ok(),
                unavailable_reason: health.err(),
            }
        })
        .collect())
}

pub fn set_generated_tool_enabled(
    data_dir: &Path,
    workspace: &Path,
    name: &str,
    enabled: bool,
) -> Result<()> {
    let name = safe_tool_name(name.trim_start_matches("generated__"))?;
    let dir = project_root_for(data_dir, workspace)?.join(name);
    let mut manifest = load_manifest(&dir)?;
    manifest.enabled = enabled;
    if enabled {
        manifest.consecutive_protocol_failures = 0;
    }
    save_manifest(&dir, &manifest)
}

pub fn delete_generated_tool(data_dir: &Path, workspace: &Path, name: &str) -> Result<bool> {
    let name = safe_tool_name(name.trim_start_matches("generated__"))?;
    let root = project_root_for(data_dir, workspace)?;
    let dir = root.join(&name);
    if !dir.exists() {
        return Ok(false);
    }
    // Refuse to recursively remove an arbitrary directory that merely happens
    // to share a valid tool name. A readable manifest proves this is an
    // installed generated-tool directory under the project-scoped root.
    let manifest = load_manifest(&dir)?;
    if manifest.name != name {
        return Err(PokError::Tool(
            "generated tool manifest name does not match its installation directory".into(),
        ));
    }
    std::fs::remove_dir_all(&dir)?;
    Ok(true)
}

pub fn record_generated_tool_candidate(
    data_dir: &Path,
    workspace: &Path,
    helper_path: &Path,
    task: &str,
    runtime: ScriptRuntime,
    verified_command: &str,
) -> Result<Option<GeneratedToolCandidate>> {
    let canonical_workspace = dunce::canonicalize(workspace)?;
    let source_result = if helper_path.is_absolute() {
        dunce::canonicalize(helper_path)
    } else {
        dunce::canonicalize(canonical_workspace.join(helper_path))
    };
    let source = match source_result {
        Ok(source) => source,
        Err(_) => return Ok(None),
    };
    if !source.starts_with(&canonical_workspace) || !source.is_file() {
        return Ok(None);
    }
    let bytes = std::fs::read(&source)?;
    if bytes.is_empty() || bytes.len() > 2 * 1024 * 1024 {
        return Ok(None);
    }
    let relative = source
        .strip_prefix(&canonical_workspace)
        .map_err(|error| PokError::Tool(error.to_string()))?
        .to_path_buf();
    let source_sha256 = hex_hash(&bytes);
    let project_root = project_root_for(data_dir, workspace)?;
    let already_promoted = std::fs::read_dir(&project_root)?
        .filter_map(std::result::Result::ok)
        .filter_map(|entry| load_manifest(&entry.path()).ok())
        .any(|manifest| manifest.source_sha256 == source_sha256);
    if already_promoted {
        return Ok(None);
    }
    let id = hex_hash(
        format!(
            "{}|{}|{}",
            canonical_workspace.display(),
            relative.display(),
            source_sha256
        )
        .as_bytes(),
    )[..16]
        .to_string();
    let candidate = GeneratedToolCandidate {
        id: id.clone(),
        helper_path: relative,
        task: task.split_whitespace().collect::<Vec<_>>().join(" "),
        runtime,
        verified_command: verified_command.into(),
        source_sha256,
        created_at: Utc::now(),
    };
    let dir = project_root.join("candidates");
    std::fs::create_dir_all(&dir)?;
    atomic_write(
        &dir.join(format!("{id}.json")),
        &serde_json::to_vec_pretty(&candidate)?,
    )?;
    Ok(Some(candidate))
}

pub fn list_generated_tool_candidates(
    data_dir: &Path,
    workspace: &Path,
) -> Result<Vec<GeneratedToolCandidate>> {
    let dir = project_root_for(data_dir, workspace)?.join("candidates");
    std::fs::create_dir_all(&dir)?;
    let mut candidates = std::fs::read_dir(dir)?
        .filter_map(std::result::Result::ok)
        .filter_map(|entry| {
            serde_json::from_slice::<GeneratedToolCandidate>(&std::fs::read(entry.path()).ok()?)
                .ok()
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| right.created_at.cmp(&left.created_at));
    Ok(candidates)
}

pub fn dismiss_generated_tool_candidate(
    data_dir: &Path,
    workspace: &Path,
    id: &str,
) -> Result<bool> {
    if id.len() != 16 || !id.chars().all(|character| character.is_ascii_hexdigit()) {
        return Err(PokError::Tool("invalid generated-tool candidate id".into()));
    }
    let path = project_root_for(data_dir, workspace)?
        .join("candidates")
        .join(format!("{id}.json"));
    match std::fs::remove_file(path) {
        Ok(()) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error.into()),
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct PromoteArgs {
    helper_path: PathBuf,
    name: String,
    description: String,
    runtime: ScriptRuntime,
    input_schema: Value,
    #[serde(default)]
    capabilities: Vec<String>,
    #[serde(default)]
    #[schemars(skip)]
    smoke_test_input: Option<Value>,
    #[serde(default)]
    dependencies: Vec<String>,
    #[serde(default)]
    smoke_tests: Vec<GeneratedToolSmokeTest>,
}

struct PromoteHelperTool;
#[async_trait]
impl Tool for PromoteHelperTool {
    fn name(&self) -> &'static str {
        "promote_helper_tool"
    }
    fn description(&self) -> &'static str {
        "Promote a tested helper into a reusable project-scoped tool. The helper receives exactly one JSON value on stdin and must write exactly one JSON value to stdout with no extra stdout text. Declare all runtime packages in dependencies; installing packages while the helper runs is rejected. Provide at least one smoke_tests entry with semantic assertions. Assertion pointers use RFC 6901 syntax: /content selects the content property and /items/0 selects the first item. Supported assertions are exists, non_empty, type, and max_missing_ratio."
    }
    fn input_schema(&self) -> Value {
        schema::<PromoteArgs>()
    }
    fn risk(&self) -> RiskClass {
        RiskClass::ProcessExecution
    }
    fn target_path(&self, args: &Value) -> Option<PathBuf> {
        serde_json::from_value::<PromoteArgs>(args.clone())
            .ok()
            .map(|args| args.helper_path)
    }
    async fn execute(&self, args: Value, context: &ToolContext) -> Result<Value> {
        let args: PromoteArgs = serde_json::from_value(args)?;
        let safe_name = safe_tool_name(&args.name)?;
        if args.description.trim().is_empty() {
            return Err(PokError::Tool("description cannot be empty".into()));
        }
        validate_schema(&args.input_schema)?;
        validate_capabilities(&args.capabilities)?;
        validate_dependencies(&args.dependencies)?;
        let tests = normalized_smoke_tests(args.smoke_tests, args.smoke_test_input)?;
        validate_smoke_test_contract(&tests)?;
        runtime_available(&args.runtime)?;
        let source = context.resolve_workspace_path(&args.helper_path)?;
        if !source.is_file() {
            return Err(PokError::Tool("helper_path must be a file".into()));
        }
        let bytes = std::fs::read(&source)?;
        if bytes.len() > 2 * 1024 * 1024 {
            return Err(PokError::Tool("helper exceeds 2 MiB".into()));
        }
        let root = project_root(context)?;
        let tool_dir = root.join(&safe_name);
        let staging_dir = root.join(format!(".staging-{safe_name}-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&staging_dir)?;
        let extension = source
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or(match args.runtime {
                ScriptRuntime::Powershell => "ps1",
                ScriptRuntime::Python => "py",
                ScriptRuntime::Node => "js",
            });
        let entrypoint = format!("tool.{extension}");
        atomic_write(&staging_dir.join(&entrypoint), &bytes)?;
        let prior = load_manifest(&tool_dir).ok();
        let mut manifest = GeneratedToolManifest {
            schema_version: 2,
            name: safe_name.clone(),
            description: args.description.trim().into(),
            runtime: args.runtime,
            entrypoint,
            input_schema: args.input_schema,
            capabilities: normalized_capabilities(args.capabilities),
            timeout_seconds: 60,
            max_output_bytes: MAX_OUTPUT,
            source_sha256: hex_hash(&bytes),
            enabled: matches!(context.policy.mode, PolicyMode::Autonomous),
            version: prior
                .as_ref()
                .map_or(1, |old| old.version.saturating_add(1)),
            successes: prior.as_ref().map_or(0, |old| old.successes),
            failures: prior.as_ref().map_or(0, |old| old.failures),
            consecutive_protocol_failures: 0,
            dependencies: args.dependencies,
            environment_sha256: String::new(),
            validation: GeneratedToolValidation::default(),
        };
        if let Err(error) = prepare_environment(&staging_dir, &manifest, context).await {
            let _ = std::fs::remove_dir_all(&staging_dir);
            return Err(error);
        }
        let lock = match environment_lock(&staging_dir, &manifest).await {
            Ok(lock) => lock,
            Err(error) => {
                let _ = std::fs::remove_dir_all(&staging_dir);
                return Err(promotion_failure(
                    "dependency_setup",
                    error.to_string(),
                    None,
                ));
            }
        };
        atomic_write(&staging_dir.join("dependencies.lock"), &lock)?;
        manifest.environment_sha256 = hex_hash(&lock);
        let mut smoke_results = Vec::new();
        let mut assertion_count = 0_u32;
        for test in &tests {
            let value = execute_manifest(&staging_dir, &manifest, test.input.clone(), context)
                .await
                .inspect_err(|_| {
                    let _ = std::fs::remove_dir_all(&staging_dir);
                })?;
            if let Err(error) = validate_output_assertions(&value, &test.assertions) {
                let _ = std::fs::remove_dir_all(&staging_dir);
                return Err(promotion_failure("assertion", error.to_string(), None));
            }
            assertion_count = assertion_count
                .saturating_add(u32::try_from(test.assertions.len()).unwrap_or(u32::MAX));
            smoke_results.push(value);
        }
        let post_smoke_lock = match environment_lock(&staging_dir, &manifest).await {
            Ok(lock) => lock,
            Err(error) => {
                let _ = std::fs::remove_dir_all(&staging_dir);
                return Err(promotion_failure(
                    "dependency_setup",
                    error.to_string(),
                    None,
                ));
            }
        };
        if post_smoke_lock != lock {
            let added = added_lock_entries(&lock, &post_smoke_lock);
            let _ = std::fs::remove_dir_all(&staging_dir);
            return Err(promotion_failure(
                "dependency_setup",
                "the helper changed its runtime environment during smoke tests; declare every package in dependencies and remove runtime installation",
                Some(json!({"added_packages": added})),
            ));
        }
        manifest.successes = manifest
            .successes
            .saturating_add(u64::try_from(tests.len()).unwrap_or(u64::MAX));
        manifest.validation = GeneratedToolValidation {
            validated_at: Some(Utc::now()),
            smoke_tests: u32::try_from(tests.len()).unwrap_or(u32::MAX),
            assertions: assertion_count,
        };
        save_manifest(&staging_dir, &manifest)?;
        let backup_dir = root.join(format!(".backup-{safe_name}-{}", Uuid::new_v4()));
        if tool_dir.exists() {
            std::fs::rename(&tool_dir, &backup_dir)?;
        }
        if let Err(error) = std::fs::rename(&staging_dir, &tool_dir) {
            if backup_dir.exists() {
                let _ = std::fs::rename(&backup_dir, &tool_dir);
            }
            return Err(error.into());
        }
        if backup_dir.exists() {
            let _ = std::fs::remove_dir_all(backup_dir);
        }
        if let Ok(relative_source) = source.strip_prefix(dunce::canonicalize(&context.workspace)?) {
            for candidate in list_generated_tool_candidates(&context.data_dir, &context.workspace)?
            {
                if candidate.helper_path == relative_source
                    && candidate.source_sha256 == manifest.source_sha256
                {
                    let _ = dismiss_generated_tool_candidate(
                        &context.data_dir,
                        &context.workspace,
                        &candidate.id,
                    );
                }
            }
        }
        Ok(
            json!({"promoted": true, "tool_name": format!("generated__{safe_name}"), "enabled": manifest.enabled, "version": manifest.version, "manifest": tool_dir.join("tool.toml"), "smoke_test_results": smoke_results, "validation": manifest.validation}),
        )
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SearchArgs {
    #[serde(default)]
    query: String,
}
struct SearchGeneratedToolsTool;
#[async_trait]
impl Tool for SearchGeneratedToolsTool {
    fn name(&self) -> &'static str {
        "search_generated_tools"
    }
    fn description(&self) -> &'static str {
        "List or search reusable generated tools available to the current project, including enabled state and permissions."
    }
    fn input_schema(&self) -> Value {
        schema::<SearchArgs>()
    }
    fn risk(&self) -> RiskClass {
        RiskClass::ReadOnly
    }
    async fn execute(&self, args: Value, context: &ToolContext) -> Result<Value> {
        let args: SearchArgs = serde_json::from_value(args)?;
        let query = args.query.to_ascii_lowercase();
        let tools = manifests(context)?.into_iter().filter(|(_, manifest)| query.is_empty() || manifest.name.to_ascii_lowercase().contains(&query) || manifest.description.to_ascii_lowercase().contains(&query)).map(|(dir, manifest)| {
            let health = generated_tool_health(&dir, &manifest);
            json!({
            "name": format!("generated__{}", manifest.name), "description": manifest.description, "enabled": manifest.enabled,
            "available": manifest.enabled && health.is_ok(), "unavailable_reason": health.err(),
            "runtime": manifest.runtime, "capabilities": manifest.capabilities, "version": manifest.version,
            "successes": manifest.successes, "failures": manifest.failures, "input_schema": manifest.input_schema,
        })}).collect::<Vec<_>>();
        let candidates = list_generated_tool_candidates(&context.data_dir, &context.workspace)?
            .into_iter()
            .filter(|candidate| {
                query.is_empty()
                    || candidate.task.to_ascii_lowercase().contains(&query)
                    || candidate
                        .helper_path
                        .to_string_lossy()
                        .to_ascii_lowercase()
                        .contains(&query)
            })
            .collect::<Vec<_>>();
        Ok(json!({
            "tools": tools,
            "promotion_candidates": candidates,
            "candidate_note": "Candidates are verified helper executions, not installed tools. Inspect the source and explicitly call promote_helper_tool with a JSON input schema to install one."
        }))
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct InvokeArgs {
    name: String,
    #[serde(default)]
    input: Value,
}
struct InvokeGeneratedToolTool;
#[async_trait]
impl Tool for InvokeGeneratedToolTool {
    fn name(&self) -> &'static str {
        "invoke_generated_tool"
    }
    fn description(&self) -> &'static str {
        "Invoke an enabled project-scoped generated tool by name. Pass input as an actual JSON value matching the returned manifest schema, not as JSON text inside a string. Input is sent as JSON on stdin; output must be one JSON value on stdout."
    }
    fn input_schema(&self) -> Value {
        schema::<InvokeArgs>()
    }
    fn risk(&self) -> RiskClass {
        RiskClass::ProcessExecution
    }
    async fn execute(&self, args: Value, context: &ToolContext) -> Result<Value> {
        let args: InvokeArgs = serde_json::from_value(args)?;
        let name = safe_tool_name(args.name.trim_start_matches("generated__"))?;
        let dir = project_root(context)?.join(&name);
        let mut manifest = load_manifest(&dir)?;
        if !manifest.enabled {
            return Err(PokError::Tool(format!(
                "generated tool {name:?} is disabled"
            )));
        }
        if let Err(reason) = generated_tool_health(&dir, &manifest) {
            manifest.failures += 1;
            manifest.enabled = false;
            save_manifest(&dir, &manifest)?;
            return Err(generated_tool_installation_failure(&name, &reason));
        }
        let (input, input_normalized) =
            normalize_generated_tool_input(args.input, &manifest.input_schema)?;
        match execute_manifest(&dir, &manifest, input, context).await {
            Ok(value) => {
                manifest.successes += 1;
                manifest.consecutive_protocol_failures = 0;
                save_manifest(&dir, &manifest)?;
                artifact_backed_result(
                    enrich_generated_tool_result(value, input_normalized),
                    context,
                )
            }
            Err(error) => {
                manifest.failures += 1;
                manifest.consecutive_protocol_failures += 1;
                if manifest.consecutive_protocol_failures >= 3 {
                    manifest.enabled = false;
                }
                save_manifest(&dir, &manifest)?;
                Err(error)
            }
        }
    }
}

fn generated_tool_installation_failure(name: &str, detail: &str) -> PokError {
    PokError::Tool(
        json!({
            "failure_attribution": "environment_failure",
            "generated_tool": format!("generated__{name}"),
            "installation_error": detail,
            "instruction": "Use another available tool or recreate and promote the helper before retrying this generated tool.",
        })
        .to_string(),
    )
}

fn normalize_generated_tool_input(input: Value, schema: &Value) -> Result<(Value, bool)> {
    let expects_object = schema.get("type").and_then(Value::as_str) == Some("object");
    let (input, normalized) = if expects_object {
        match input {
            Value::String(text) => match serde_json::from_str::<Value>(&text) {
                Ok(parsed) if parsed.is_object() => (parsed, true),
                _ => {
                    return Err(generated_tool_input_error(
                        "input must be a JSON object, not a string",
                        schema,
                    ));
                }
            },
            input => (input, false),
        }
    } else {
        (input, false)
    };
    validate_generated_tool_input(&input, schema)?;
    Ok((input, normalized))
}

fn validate_generated_tool_input(input: &Value, schema: &Value) -> Result<()> {
    let expected_type = schema.get("type").and_then(Value::as_str);
    if expected_type.is_some_and(|expected| !json_value_matches_type(input, expected)) {
        return Err(generated_tool_input_error(
            &format!(
                "input root type is {}; expected {}",
                json_value_type(input),
                expected_type.unwrap_or("declared type")
            ),
            schema,
        ));
    }
    if let Some(required) = schema.get("required").and_then(Value::as_array) {
        let object = input
            .as_object()
            .ok_or_else(|| generated_tool_input_error("required fields need an object", schema))?;
        for name in required.iter().filter_map(Value::as_str) {
            if !object.contains_key(name) {
                return Err(generated_tool_input_error(
                    &format!("missing required input field {name:?}"),
                    schema,
                ));
            }
        }
    }
    if let (Some(object), Some(properties)) = (
        input.as_object(),
        schema.get("properties").and_then(Value::as_object),
    ) {
        for (name, property_schema) in properties {
            let Some(value) = object.get(name) else {
                continue;
            };
            if let Some(expected) = property_schema.get("type").and_then(Value::as_str)
                && !json_value_matches_type(value, expected)
            {
                return Err(generated_tool_input_error(
                    &format!(
                        "input field {name:?} is {}; expected {expected}",
                        json_value_type(value)
                    ),
                    schema,
                ));
            }
        }
    }
    Ok(())
}

fn json_value_matches_type(value: &Value, expected: &str) -> bool {
    match expected {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "number" => value.is_number(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        _ => true,
    }
}

fn json_value_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn generated_tool_input_error(detail: &str, schema: &Value) -> PokError {
    PokError::Tool(
        json!({
            "failure_attribution": "caller_input",
            "input_error": detail,
            "expected_schema": schema,
            "instruction": "Correct the input value and retry once. Do not modify or disable the generated tool.",
        })
        .to_string(),
    )
}

fn enrich_generated_tool_result(value: Value, input_normalized: bool) -> Value {
    match value {
        Value::Object(mut object) => {
            object
                .entry("input_normalized")
                .or_insert(Value::Bool(input_normalized));
            Value::Object(object)
        }
        value => json!({"result": value, "input_normalized": input_normalized}),
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ReadGeneratedToolOutputArgs {
    /// Opaque artifact identifier returned by invoke_generated_tool.
    artifact_id: String,
    /// UTF-8 byte offset at which to continue reading.
    #[serde(default)]
    offset: usize,
    /// Maximum number of bytes to return, capped at 65536.
    #[serde(default = "default_artifact_page")]
    max_bytes: usize,
}

const fn default_artifact_page() -> usize {
    16 * 1024
}

struct ReadGeneratedToolOutputTool;
#[async_trait]
impl Tool for ReadGeneratedToolOutputTool {
    fn name(&self) -> &'static str {
        "read_generated_tool_output"
    }
    fn description(&self) -> &'static str {
        "Read a bounded UTF-8 chunk from an oversized generated-tool result saved in this session. Use only the opaque artifact_id returned by invoke_generated_tool, then continue with next_offset until truncated is false."
    }
    fn input_schema(&self) -> Value {
        schema::<ReadGeneratedToolOutputArgs>()
    }
    fn risk(&self) -> RiskClass {
        RiskClass::ReadOnly
    }
    async fn execute(&self, args: Value, context: &ToolContext) -> Result<Value> {
        let args: ReadGeneratedToolOutputArgs = serde_json::from_value(args)?;
        read_artifact_page(context, &args.artifact_id, args.offset, args.max_bytes)
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct EnableArgs {
    name: String,
    enabled: bool,
}
struct SetGeneratedToolEnabledTool;
#[async_trait]
impl Tool for SetGeneratedToolEnabledTool {
    fn name(&self) -> &'static str {
        "set_generated_tool_enabled"
    }
    fn description(&self) -> &'static str {
        "Enable or disable a project-scoped generated tool. Enabling resets its protocol-failure counter."
    }
    fn input_schema(&self) -> Value {
        schema::<EnableArgs>()
    }
    fn risk(&self) -> RiskClass {
        RiskClass::WorkspaceWrite
    }
    async fn execute(&self, args: Value, context: &ToolContext) -> Result<Value> {
        let args: EnableArgs = serde_json::from_value(args)?;
        let name = safe_tool_name(args.name.trim_start_matches("generated__"))?;
        let dir = project_root(context)?.join(&name);
        let mut manifest = load_manifest(&dir)?;
        manifest.enabled = args.enabled;
        if args.enabled {
            manifest.consecutive_protocol_failures = 0;
        }
        save_manifest(&dir, &manifest)?;
        Ok(json!({"name": format!("generated__{name}"), "enabled": manifest.enabled}))
    }
}

fn schema<T: JsonSchema>() -> Value {
    serde_json::to_value(schema_for!(T)).expect("schema")
}
fn safe_tool_name(name: &str) -> Result<String> {
    let name = name.trim().to_ascii_lowercase().replace('-', "_");
    if name.len() < 2
        || name.len() > 64
        || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        || name.starts_with(|c: char| c.is_ascii_digit())
    {
        return Err(PokError::Tool(
            "tool name must be 2-64 ASCII letters, numbers, or underscores and start with a letter"
                .into(),
        ));
    }
    Ok(name)
}
fn validate_schema(schema: &Value) -> Result<()> {
    if schema.get("type").and_then(Value::as_str) != Some("object") {
        return Err(PokError::Tool(
            "input_schema must describe a JSON object".into(),
        ));
    }
    Ok(())
}
fn validate_capabilities(capabilities: &[String]) -> Result<()> {
    const ALLOWED: &[&str] = &[
        "workspace_read",
        "workspace_write",
        "network",
        "process",
        "desktop_input",
        "secrets",
    ];
    if let Some(value) = capabilities
        .iter()
        .find(|value| !ALLOWED.contains(&value.as_str()))
    {
        return Err(PokError::Tool(format!("unsupported capability {value:?}")));
    }
    Ok(())
}
fn validate_dependencies(dependencies: &[String]) -> Result<()> {
    if dependencies.len() > 64 {
        return Err(PokError::Tool(
            "generated tools may declare at most 64 dependencies".into(),
        ));
    }
    if let Some(value) = dependencies.iter().find(|value| {
        value.is_empty()
            || value.len() > 200
            || value
                .chars()
                .any(|character| character.is_control() || matches!(character, ';' | '|' | '&'))
    }) {
        return Err(PokError::Tool(format!(
            "invalid generated-tool dependency {value:?}"
        )));
    }
    Ok(())
}

fn normalized_smoke_tests(
    smoke_tests: Vec<GeneratedToolSmokeTest>,
    legacy_input: Option<Value>,
) -> Result<Vec<GeneratedToolSmokeTest>> {
    if !smoke_tests.is_empty() {
        return Ok(smoke_tests);
    }
    Ok(legacy_input
        .map(|input| {
            vec![GeneratedToolSmokeTest {
                input,
                assertions: vec![OutputAssertion::NonEmpty {
                    pointer: String::new(),
                }],
            }]
        })
        .unwrap_or_default())
}

fn validate_smoke_test_contract(tests: &[GeneratedToolSmokeTest]) -> Result<()> {
    if tests.is_empty() {
        return Err(PokError::Tool(
            "at least one smoke test is required before promotion".into(),
        ));
    }
    for test in tests {
        if test.assertions.is_empty() {
            return Err(PokError::Tool(
                "every generated-tool smoke test requires at least one semantic output assertion"
                    .into(),
            ));
        }
        for assertion in &test.assertions {
            let pointer = match assertion {
                OutputAssertion::Exists { pointer }
                | OutputAssertion::NonEmpty { pointer }
                | OutputAssertion::Type { pointer, .. }
                | OutputAssertion::MaxMissingRatio { pointer, .. } => pointer,
            };
            validate_json_pointer(pointer)?;
            match assertion {
                OutputAssertion::Type { expected, .. }
                    if !matches!(
                        expected.as_str(),
                        "null" | "boolean" | "number" | "string" | "array" | "object"
                    ) =>
                {
                    return Err(PokError::Tool(format!(
                        "unsupported assertion type {expected:?}"
                    )));
                }
                OutputAssertion::MaxMissingRatio { maximum, .. }
                    if !(0.0..=1.0).contains(maximum) =>
                {
                    return Err(PokError::Tool(
                        "max_missing_ratio maximum must be between 0 and 1".into(),
                    ));
                }
                _ => {}
            }
        }
    }
    Ok(())
}

fn validate_json_pointer(pointer: &str) -> Result<()> {
    if !pointer.is_empty() && !pointer.starts_with('/') {
        return Err(PokError::Tool(format!(
            "assertion pointer must be an RFC 6901 JSON pointer: {pointer:?}"
        )));
    }
    let bytes = pointer.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'~' {
            if bytes
                .get(index + 1)
                .is_none_or(|next| !matches!(next, b'0' | b'1'))
            {
                return Err(PokError::Tool(format!(
                    "assertion pointer contains an invalid RFC 6901 escape: {pointer:?}"
                )));
            }
            index += 2;
        } else {
            index += 1;
        }
    }
    Ok(())
}

fn validate_output_assertions(value: &Value, assertions: &[OutputAssertion]) -> Result<()> {
    for assertion in assertions {
        let (pointer, passed, expectation) = match assertion {
            OutputAssertion::Exists { pointer } => (
                pointer,
                value.pointer(pointer).is_some(),
                "to exist".to_string(),
            ),
            OutputAssertion::NonEmpty { pointer } => {
                let actual = value.pointer(pointer);
                let non_empty = actual.is_some_and(|item| match item {
                    Value::Null => false,
                    Value::String(text) => !text.trim().is_empty() && text != "N/A",
                    Value::Array(items) => !items.is_empty(),
                    Value::Object(items) => !items.is_empty(),
                    _ => true,
                });
                (pointer, non_empty, "to be non-empty".to_string())
            }
            OutputAssertion::Type { pointer, expected } => {
                let actual = value.pointer(pointer);
                let matches = match expected.as_str() {
                    "null" => actual.is_some_and(Value::is_null),
                    "boolean" => actual.is_some_and(Value::is_boolean),
                    "number" => actual.is_some_and(Value::is_number),
                    "string" => actual.is_some_and(Value::is_string),
                    "array" => actual.is_some_and(Value::is_array),
                    "object" => actual.is_some_and(Value::is_object),
                    _ => {
                        return Err(PokError::Tool(format!(
                            "unsupported assertion type {expected:?}"
                        )));
                    }
                };
                (pointer, matches, format!("to have type {expected}"))
            }
            OutputAssertion::MaxMissingRatio { pointer, maximum } => {
                if !(0.0..=1.0).contains(maximum) {
                    return Err(PokError::Tool(
                        "max_missing_ratio maximum must be between 0 and 1".into(),
                    ));
                }
                let actual = value.pointer(pointer);
                let (missing, total) = actual.map_or((1_u64, 1_u64), missing_leaf_counts);
                let ratio = missing as f64 / total.max(1) as f64;
                (
                    pointer,
                    ratio <= *maximum,
                    format!("to have missing-value ratio at most {maximum} (actual {ratio:.3})"),
                )
            }
        };
        validate_json_pointer(pointer)?;
        if !passed {
            return Err(PokError::Tool(format!(
                "smoke-test assertion failed: expected {pointer:?} {expectation}"
            )));
        }
    }
    Ok(())
}

fn promotion_failure(phase: &str, detail: impl Into<String>, extra: Option<Value>) -> PokError {
    let mut payload = json!({
        "phase": phase,
        "detail": detail.into(),
    });
    if let (Some(object), Some(extra)) = (payload.as_object_mut(), extra) {
        if let Some(fields) = extra.as_object() {
            object.extend(fields.clone());
        }
    }
    PokError::Tool(payload.to_string())
}

fn added_lock_entries(before: &[u8], after: &[u8]) -> Vec<String> {
    let before = String::from_utf8_lossy(before)
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_owned)
        .collect::<std::collections::BTreeSet<_>>();
    String::from_utf8_lossy(after)
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !before.contains(*line))
        .take(64)
        .map(|line| line.chars().take(240).collect())
        .collect()
}

fn missing_leaf_counts(value: &Value) -> (u64, u64) {
    match value {
        Value::Array(items) if items.is_empty() => (1, 1),
        Value::Array(items) => items.iter().fold((0_u64, 0_u64), |(missing, total), item| {
            let (item_missing, item_total) = missing_leaf_counts(item);
            (
                missing.saturating_add(item_missing),
                total.saturating_add(item_total),
            )
        }),
        Value::Object(items) if items.is_empty() => (1, 1),
        Value::Object(items) => items
            .values()
            .fold((0_u64, 0_u64), |(missing, total), item| {
                let (item_missing, item_total) = missing_leaf_counts(item);
                (
                    missing.saturating_add(item_missing),
                    total.saturating_add(item_total),
                )
            }),
        Value::Null => (1, 1),
        Value::String(text) if text.trim().is_empty() || text.eq_ignore_ascii_case("n/a") => (1, 1),
        _ => (0, 1),
    }
}

async fn checked_setup_command(
    command: &mut Command,
    context: &ToolContext,
    detail: impl Into<String>,
) -> Result<()> {
    let detail = detail.into();
    process_window::hide_tokio(command);
    let output = command.output().await.map_err(|error| {
        execution_failure(
            context,
            "dependency_setup",
            format!("{detail}: {error}"),
            None,
            &[],
            &[],
        )
    })?;
    if !output.status.success() {
        return Err(execution_failure(
            context,
            "dependency_setup",
            detail,
            output.status.code(),
            &output.stdout,
            &output.stderr,
        ));
    }
    Ok(())
}

async fn prepare_environment(
    dir: &Path,
    manifest: &GeneratedToolManifest,
    context: &ToolContext,
) -> Result<()> {
    if manifest.dependencies.is_empty() && !matches!(manifest.runtime, ScriptRuntime::Python) {
        return Ok(());
    }
    match manifest.runtime {
        ScriptRuntime::Python => {
            let python = if cfg!(windows) {
                "python.exe"
            } else {
                "python3"
            };
            checked_setup_command(
                Command::new(python)
                    .args(["-m", "venv"])
                    .arg(dir.join(".venv")),
                context,
                "could not create isolated Python environment",
            )
            .await?;
            if !manifest.dependencies.is_empty() {
                let pip = private_python(dir);
                checked_setup_command(
                    Command::new(pip)
                        .args(["-m", "pip", "install", "--disable-pip-version-check"])
                        .args(&manifest.dependencies),
                    context,
                    "could not install generated-tool Python dependencies",
                )
                .await?;
            }
        }
        ScriptRuntime::Node => {
            if !manifest.dependencies.is_empty() {
                checked_setup_command(
                    Command::new(if cfg!(windows) { "npm.cmd" } else { "npm" })
                        .args(["install", "--ignore-scripts", "--no-audit", "--no-fund"])
                        .args(&manifest.dependencies)
                        .current_dir(dir),
                    context,
                    "could not install generated-tool Node dependencies",
                )
                .await?;
            }
        }
        ScriptRuntime::Powershell => {
            if !manifest.dependencies.is_empty() {
                let modules = dir.join("modules");
                std::fs::create_dir_all(&modules).map_err(|error| {
                    promotion_failure("dependency_setup", error.to_string(), None)
                })?;
                for dependency in &manifest.dependencies {
                    checked_setup_command(
                        Command::new("powershell.exe")
                            .args([
                                "-NoProfile",
                                "-NonInteractive",
                                "-Command",
                                "param($Name,$Path) Save-Module -Name $Name -Path $Path -Force",
                            ])
                            .arg(dependency)
                            .arg(&modules),
                        context,
                        format!("could not save PowerShell module {dependency:?}"),
                    )
                    .await?;
                }
            }
        }
    }
    Ok(())
}

async fn environment_lock(dir: &Path, manifest: &GeneratedToolManifest) -> Result<Vec<u8>> {
    match manifest.runtime {
        ScriptRuntime::Python => {
            let mut command = Command::new(private_python(dir));
            command.args(["-m", "pip", "freeze", "--all"]);
            process_window::hide_tokio(&mut command);
            let output = command.output().await?;
            if !output.status.success() {
                return Err(PokError::Tool(
                    "could not lock generated-tool Python environment".into(),
                ));
            }
            Ok(output.stdout)
        }
        ScriptRuntime::Node => {
            let path = dir.join("package-lock.json");
            if path.is_file() {
                Ok(std::fs::read(path)?)
            } else {
                Ok(manifest.dependencies.join("\n").into_bytes())
            }
        }
        ScriptRuntime::Powershell => Ok(manifest.dependencies.join("\n").into_bytes()),
    }
}

fn private_python(dir: &Path) -> PathBuf {
    if cfg!(windows) {
        dir.join(".venv").join("Scripts").join("python.exe")
    } else {
        dir.join(".venv").join("bin").join("python")
    }
}
fn normalized_capabilities(mut capabilities: Vec<String>) -> Vec<String> {
    if !capabilities.iter().any(|value| value == "process") {
        capabilities.push("process".into());
    }
    capabilities.sort();
    capabilities.dedup();
    capabilities
}

fn generated_tool_health(
    dir: &Path,
    manifest: &GeneratedToolManifest,
) -> std::result::Result<(), String> {
    let entry = dir.join(&manifest.entrypoint);
    if !entry.is_file() {
        return Err(format!("entrypoint is missing: {}", manifest.entrypoint));
    }
    let source =
        std::fs::read(&entry).map_err(|error| format!("entrypoint cannot be read: {error}"))?;
    if hex_hash(&source) != manifest.source_sha256 {
        return Err("entrypoint hash does not match the manifest".into());
    }
    if manifest.schema_version >= 2 && !manifest.environment_sha256.is_empty() {
        match manifest.runtime {
            ScriptRuntime::Python if !private_python(dir).is_file() => {
                return Err("isolated Python environment is missing".into());
            }
            ScriptRuntime::Node
                if !manifest.dependencies.is_empty() && !dir.join("node_modules").is_dir() =>
            {
                return Err("isolated Node environment is missing".into());
            }
            ScriptRuntime::Powershell
                if !manifest.dependencies.is_empty() && !dir.join("modules").is_dir() =>
            {
                return Err("isolated PowerShell module directory is missing".into());
            }
            _ => {}
        }
    }
    if matches!(manifest.runtime, ScriptRuntime::Python) && private_python(dir).is_file() {
        let mut command = std::process::Command::new(private_python(dir));
        command
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        process_window::hide_std(&mut command);
        let status = command
            .status()
            .map_err(|error| format!("isolated Python runtime is unavailable: {error}"))?;
        if !status.success() {
            return Err("isolated Python runtime failed its availability check".into());
        }
    } else {
        runtime_available(&manifest.runtime).map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn runtime_available(runtime: &ScriptRuntime) -> Result<()> {
    let command = match runtime {
        ScriptRuntime::Powershell => "powershell.exe",
        ScriptRuntime::Python => {
            if cfg!(windows) {
                "python.exe"
            } else {
                "python3"
            }
        }
        ScriptRuntime::Node => {
            if cfg!(windows) {
                "node.exe"
            } else {
                "node"
            }
        }
    };
    let mut check = std::process::Command::new(command);
    match runtime {
        ScriptRuntime::Powershell => {
            check.args([
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "$PSVersionTable.PSVersion.ToString()",
            ]);
        }
        _ => {
            check.arg("--version");
        }
    }
    process_window::hide_std(&mut check);
    let status = check
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|_| PokError::Tool(format!("runtime {command} is not installed")))?;
    if !status.success() {
        return Err(PokError::Tool(format!(
            "runtime {command} failed its availability check"
        )));
    }
    Ok(())
}
fn project_root(context: &ToolContext) -> Result<PathBuf> {
    project_root_for(&context.data_dir, &context.workspace)
}
fn project_root_for(data_dir: &Path, workspace: &Path) -> Result<PathBuf> {
    let canonical = dunce::canonicalize(workspace)?;
    let hash = hex_hash(canonical.to_string_lossy().as_bytes());
    let root = data_dir.join("generated-tools").join(&hash[..16]);
    std::fs::create_dir_all(&root)?;
    Ok(root)
}
fn manifests(context: &ToolContext) -> Result<Vec<(PathBuf, GeneratedToolManifest)>> {
    let root = project_root(context)?;
    Ok(std::fs::read_dir(root)?
        .filter_map(std::result::Result::ok)
        .filter(|entry| entry.path().is_dir())
        .filter_map(|entry| {
            load_manifest(&entry.path())
                .ok()
                .map(|manifest| (entry.path(), manifest))
        })
        .collect())
}
fn load_manifest(dir: &Path) -> Result<GeneratedToolManifest> {
    toml::from_str(&std::fs::read_to_string(dir.join("tool.toml"))?)
        .map_err(|error| PokError::Tool(format!("invalid generated tool manifest: {error}")))
}
fn save_manifest(dir: &Path, manifest: &GeneratedToolManifest) -> Result<()> {
    atomic_write(
        &dir.join("tool.toml"),
        toml::to_string_pretty(manifest)
            .map_err(|error| PokError::Tool(error.to_string()))?
            .as_bytes(),
    )
}
fn hex_hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn bounded_text(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes);
    if text.len() <= MAX_DIAGNOSTIC_STREAM {
        return text.into_owned();
    }
    let mut end = MAX_DIAGNOSTIC_STREAM;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &text[..end])
}

fn write_execution_diagnostic(
    context: &ToolContext,
    phase: &str,
    status: Option<i32>,
    stdout: &[u8],
    stderr: &[u8],
) -> Option<String> {
    let name = format!("generated-tool-failure-{}.json", Uuid::new_v4());
    context
        .write_artifact(
            &name,
            &json!({
                "phase": phase,
                "exit_code": status,
                "stdout": bounded_text(stdout),
                "stderr": bounded_text(stderr),
                "stdout_bytes": stdout.len(),
                "stderr_bytes": stderr.len(),
            }),
        )
        .ok()
        .map(|_| name)
}

fn execution_failure(
    context: &ToolContext,
    phase: &str,
    detail: impl Into<String>,
    status: Option<i32>,
    stdout: &[u8],
    stderr: &[u8],
) -> PokError {
    let artifact = write_execution_diagnostic(context, phase, status, stdout, stderr);
    promotion_failure(
        phase,
        detail,
        Some(json!({
            "exit_code": status,
            "stdout": bounded_text(stdout),
            "stderr": bounded_text(stderr),
            "diagnostic_artifact": artifact,
        })),
    )
}

fn artifact_backed_result(value: Value, context: &ToolContext) -> Result<Value> {
    let bytes = serde_json::to_vec(&value)?;
    if bytes.len() <= DIRECT_OUTPUT_LIMIT {
        return Ok(value);
    }
    std::fs::create_dir_all(&context.artifact_dir)?;
    let artifact_id = Uuid::new_v4();
    let name = format!("generated-tool-output-{artifact_id}.json");
    atomic_write(&context.artifact_dir.join(&name), &bytes)?;
    let preview_end = utf8_page_end(&bytes, 0, DIRECT_OUTPUT_LIMIT)?;
    Ok(json!({
        "artifact_backed": true,
        "artifact_id": artifact_id,
        "byte_count": bytes.len(),
        "preview": std::str::from_utf8(&bytes[..preview_end])
            .map_err(|error| PokError::Tool(format!("generated tool returned invalid UTF-8 JSON: {error}")))?,
        "next_offset": preview_end,
        "truncated": preview_end < bytes.len(),
    }))
}

fn utf8_page_end(bytes: &[u8], start: usize, max_bytes: usize) -> Result<usize> {
    if start > bytes.len()
        || !std::str::from_utf8(bytes).is_ok_and(|text| text.is_char_boundary(start))
    {
        return Err(PokError::Tool(
            "offset must be a valid UTF-8 boundary within the artifact".into(),
        ));
    }
    let mut end = start.saturating_add(max_bytes).min(bytes.len());
    while end > start && std::str::from_utf8(bytes).is_ok_and(|text| !text.is_char_boundary(end)) {
        end -= 1;
    }
    if end == start && start < bytes.len() {
        return Err(PokError::Tool(
            "max_bytes is too small to include the next UTF-8 character".into(),
        ));
    }
    Ok(end)
}

fn read_artifact_page(
    context: &ToolContext,
    artifact_id: &str,
    offset: usize,
    max_bytes: usize,
) -> Result<Value> {
    let artifact_id = Uuid::parse_str(artifact_id)
        .map_err(|_| PokError::Tool("invalid generated-tool artifact id".into()))?;
    if max_bytes == 0 || max_bytes > MAX_ARTIFACT_PAGE {
        return Err(PokError::Tool(format!(
            "max_bytes must be between 1 and {MAX_ARTIFACT_PAGE}"
        )));
    }
    let path = context
        .artifact_dir
        .join(format!("generated-tool-output-{artifact_id}.json"));
    let bytes = std::fs::read(&path).map_err(|_| {
        PokError::Tool("generated-tool artifact is unavailable in this session".into())
    })?;
    let end = utf8_page_end(&bytes, offset, max_bytes)?;
    let chunk = std::str::from_utf8(&bytes[offset..end])
        .map_err(|error| PokError::Tool(format!("artifact is not valid UTF-8: {error}")))?;
    Ok(json!({
        "artifact_id": artifact_id,
        "offset": offset,
        "next_offset": end,
        "byte_count": bytes.len(),
        "chunk": chunk,
        "truncated": end < bytes.len(),
    }))
}

async fn execute_manifest(
    dir: &Path,
    manifest: &GeneratedToolManifest,
    input: Value,
    context: &ToolContext,
) -> Result<Value> {
    let entry = dir.join(&manifest.entrypoint);
    if hex_hash(&std::fs::read(&entry)?) != manifest.source_sha256 {
        return Err(PokError::Tool(
            "generated tool source hash does not match its manifest".into(),
        ));
    }
    let mut command = match manifest.runtime {
        ScriptRuntime::Powershell => {
            let mut c = Command::new("powershell.exe");
            c.args([
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
            ]);
            c.arg(&entry);
            c
        }
        ScriptRuntime::Python => {
            let mut c = Command::new(if dir.join(".venv").exists() {
                private_python(dir)
            } else if cfg!(windows) {
                PathBuf::from("python.exe")
            } else {
                PathBuf::from("python3")
            });
            c.arg(&entry);
            c
        }
        ScriptRuntime::Node => {
            let mut c = Command::new(if cfg!(windows) { "node.exe" } else { "node" });
            c.arg(&entry);
            c
        }
    };
    command
        .current_dir(&context.workspace)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    process_window::hide_tokio(&mut command);
    command.env_clear();
    for key in ["PATH", "SystemRoot", "WINDIR", "TEMP", "TMP"] {
        if let Some(value) = std::env::var_os(key) {
            command.env(key, value);
        }
    }
    command.env("POK_WORKSPACE", &context.workspace);
    if matches!(manifest.runtime, ScriptRuntime::Node) {
        command.env("NODE_PATH", dir.join("node_modules"));
    }
    if matches!(manifest.runtime, ScriptRuntime::Powershell) && dir.join("modules").is_dir() {
        command.env("PSModulePath", dir.join("modules"));
    }
    let mut child = command.spawn().map_err(|error| {
        execution_failure(
            context,
            "runtime",
            format!("cannot start generated tool: {error}"),
            None,
            &[],
            &[],
        )
    })?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(&serde_json::to_vec(&input)?).await?;
    }
    let output = tokio::time::timeout(
        Duration::from_secs(manifest.timeout_seconds.clamp(1, 300)),
        child.wait_with_output(),
    )
    .await
    .map_err(|_| {
        execution_failure(
            context,
            "runtime",
            "generated tool timed out",
            None,
            &[],
            &[],
        )
    })??;
    if output.stdout.len() > manifest.max_output_bytes.min(MAX_OUTPUT)
        || output.stderr.len() > MAX_OUTPUT
    {
        return Err(execution_failure(
            context,
            "protocol",
            "generated tool output exceeded 1 MiB",
            output.status.code(),
            &output.stdout,
            &output.stderr,
        ));
    }
    if !output.status.success() {
        return Err(execution_failure(
            context,
            "runtime",
            format!("generated tool exited with {}", output.status),
            output.status.code(),
            &output.stdout,
            &output.stderr,
        ));
    }
    if !output.stderr.is_empty() {
        std::fs::create_dir_all(&context.artifact_dir)?;
        atomic_write(
            &context
                .artifact_dir
                .join(format!("generated-tool-stderr-{}.txt", Uuid::new_v4())),
            &output.stderr,
        )?;
    }
    serde_json::from_slice(&output.stdout).map_err(|error| {
        execution_failure(
            context,
            "json_parsing",
            format!("generated tool must write exactly one JSON value to stdout: {error}"),
            output.status.code(),
            &output.stdout,
            &output.stderr,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn generated_names_are_collision_scoped_and_sanitized() {
        assert_eq!(safe_tool_name("Weather_Check").unwrap(), "weather_check");
        assert!(safe_tool_name("run command").is_err());
        assert!(safe_tool_name("1bad").is_err());
    }

    #[test]
    fn semantic_smoke_assertions_reject_missing_or_empty_data() {
        let value = json!({"price": {"close": 12.5}, "cashflow": {}});
        assert!(
            validate_output_assertions(
                &value,
                &[
                    OutputAssertion::Exists {
                        pointer: "/price/close".into()
                    },
                    OutputAssertion::Type {
                        pointer: "/price/close".into(),
                        expected: "number".into()
                    }
                ]
            )
            .is_ok()
        );
        assert!(
            validate_output_assertions(
                &value,
                &[OutputAssertion::NonEmpty {
                    pointer: "/cashflow".into()
                }]
            )
            .is_err()
        );
    }

    #[test]
    fn object_input_accepts_one_json_encoded_layer() {
        let schema = json!({
            "type": "object",
            "required": ["filepath"],
            "properties": {"filepath": {"type": "string"}}
        });
        let (input, normalized) = normalize_generated_tool_input(
            Value::String(r#"{"filepath":"C:\\report.pdf"}"#.into()),
            &schema,
        )
        .unwrap();
        assert!(normalized);
        assert_eq!(input["filepath"], r"C:\report.pdf");
        assert!(
            normalize_generated_tool_input(json!({"filepath": 7}), &schema)
                .unwrap_err()
                .to_string()
                .contains("caller_input")
        );
    }

    #[test]
    fn promotion_schema_hides_legacy_smoke_input_and_documents_current_contract() {
        let schema = PromoteHelperTool.input_schema();
        assert!(
            schema
                .pointer("/$defs/PromoteArgs/properties/smoke_test_input")
                .or_else(|| schema.pointer("/properties/smoke_test_input"))
                .is_none()
        );
        assert!(PromoteHelperTool.description().contains("/content"));
        assert!(
            PromoteHelperTool
                .description()
                .contains("exactly one JSON value")
        );
    }

    #[test]
    fn smoke_contract_rejects_invalid_pointers_types_and_ranges_early() {
        for assertion in [
            OutputAssertion::Exists {
                pointer: "content".into(),
            },
            OutputAssertion::Exists {
                pointer: "/bad~2escape".into(),
            },
            OutputAssertion::Type {
                pointer: "/content".into(),
                expected: "integer".into(),
            },
            OutputAssertion::MaxMissingRatio {
                pointer: "/content".into(),
                maximum: 1.1,
            },
        ] {
            assert!(
                validate_smoke_test_contract(&[GeneratedToolSmokeTest {
                    input: json!({}),
                    assertions: vec![assertion],
                }])
                .is_err()
            );
        }
    }

    #[test]
    fn current_smoke_tests_take_precedence_over_legacy_input() {
        let tests = vec![GeneratedToolSmokeTest {
            input: json!({"current": true}),
            assertions: vec![OutputAssertion::Exists {
                pointer: "/ok".into(),
            }],
        }];
        let normalized = normalized_smoke_tests(tests, Some(json!({"legacy": true}))).unwrap();
        assert_eq!(normalized.len(), 1);
        assert_eq!(normalized[0].input, json!({"current": true}));
    }

    fn test_context(root: &Path, session_id: Uuid) -> ToolContext {
        ToolContext {
            session_id,
            workspace: root.to_path_buf(),
            data_dir: root.join("data"),
            artifact_dir: root.join("artifacts").join(session_id.to_string()),
            policy: crate::policy::Policy::interactive(),
            approvals: Arc::new(crate::policy::AllowApprovals),
            platform: Arc::new(crate::platform::MockDesktop::default()),
            memory: crate::memory::MemoryStore::open(root.join("memory")).unwrap(),
            session_archive: crate::session_archive::SessionArchive::open(
                &root.join("artifacts").join(session_id.to_string()),
            )
            .unwrap(),
            cancellation: tokio_util::sync::CancellationToken::new(),
            pause: Arc::new(crate::pause::PauseController::default()),
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
            command_timeout_seconds: 10,
            input_text_inter_key_pause_ms: 0,
            vision_max_edge: 640,
            prompt_token_target: 4_000,
            context_budget: Arc::new(parking_lot::Mutex::new(crate::context::ContextBudget::new(
                "test".into(),
                "test".into(),
                8_000,
                80,
                crate::context::ContextSource::FallbackUnknown,
            ))),
            visual_history_limit: 1,
            uia_element_limit: 10,
            desktop_enrichment_timeout_ms: 2_000,
            desktop_deep_enrichment_timeout_ms: 10_000,
            model_target_limit: 10,
            fusion_iou_threshold: 0.1,
            ocr_containment_threshold: 0.6,
            annotate_targets: false,
            approval_cache: Default::default(),
            user_guidance_queue: Default::default(),
            questions: None,
            command_manager: Arc::new(crate::commands::CommandManager::new(
                root.join("artifacts").join(session_id.to_string()),
            )),
            current_tool_call_id: Default::default(),
        }
    }

    #[test]
    fn oversized_results_are_paged_and_session_scoped() {
        let root = tempfile::tempdir().unwrap();
        let first = test_context(root.path(), Uuid::new_v4());
        let second = test_context(root.path(), Uuid::new_v4());
        let value = json!({"content": "x".repeat(DIRECT_OUTPUT_LIMIT + 100)});
        let metadata = artifact_backed_result(value, &first).unwrap();
        let artifact_id = metadata["artifact_id"].as_str().unwrap();
        assert_eq!(metadata["artifact_backed"], true);
        let page = read_artifact_page(&first, artifact_id, 0, 1024).unwrap();
        assert_eq!(page["truncated"], true);
        assert!(read_artifact_page(&second, artifact_id, 0, 1024).is_err());
        assert!(read_artifact_page(&first, "../../escape", 0, 1024).is_err());
    }

    #[test]
    fn manifest_round_trips_with_json_schema() {
        let manifest = GeneratedToolManifest {
            schema_version: 1,
            name: "sample".into(),
            description: "Sample".into(),
            runtime: ScriptRuntime::Powershell,
            entrypoint: "tool.ps1".into(),
            input_schema: json!({"type":"object","properties":{"value":{"type":"string"}}}),
            capabilities: vec!["process".into()],
            timeout_seconds: 60,
            max_output_bytes: MAX_OUTPUT,
            source_sha256: "abc".into(),
            enabled: false,
            version: 1,
            successes: 0,
            failures: 0,
            consecutive_protocol_failures: 0,
            dependencies: Vec::new(),
            environment_sha256: String::new(),
            validation: GeneratedToolValidation::default(),
        };
        let encoded = toml::to_string(&manifest).unwrap();
        let decoded: GeneratedToolManifest = toml::from_str(&encoded).unwrap();
        assert_eq!(decoded.input_schema["type"], "object");
    }

    #[test]
    fn generated_tool_health_rejects_missing_entrypoint_before_runtime_use() {
        let directory = tempfile::tempdir().unwrap();
        let manifest = GeneratedToolManifest {
            schema_version: 1,
            name: "missing".into(),
            description: "Missing helper".into(),
            runtime: ScriptRuntime::Python,
            entrypoint: "tool.py".into(),
            input_schema: json!({"type": "object"}),
            capabilities: vec!["process".into()],
            timeout_seconds: 60,
            max_output_bytes: MAX_OUTPUT,
            source_sha256: "abc".into(),
            enabled: true,
            version: 1,
            successes: 0,
            failures: 0,
            consecutive_protocol_failures: 0,
            dependencies: Vec::new(),
            environment_sha256: String::new(),
            validation: GeneratedToolValidation::default(),
        };
        let reason = generated_tool_health(directory.path(), &manifest).unwrap_err();
        assert!(reason.contains("entrypoint is missing"));
    }

    #[test]
    fn successful_helper_candidate_is_project_scoped_and_deduplicated() {
        let data = tempfile::tempdir().unwrap();
        let workspace = tempfile::tempdir().unwrap();
        let helper = workspace.path().join("report.py");
        std::fs::write(&helper, "print('{}')").unwrap();
        let first = record_generated_tool_candidate(
            data.path(),
            workspace.path(),
            Path::new("report.py"),
            "Generate report",
            ScriptRuntime::Python,
            "python report.py",
        )
        .unwrap()
        .unwrap();
        let second = record_generated_tool_candidate(
            data.path(),
            workspace.path(),
            Path::new("report.py"),
            "Generate report",
            ScriptRuntime::Python,
            "python report.py",
        )
        .unwrap()
        .unwrap();
        assert_eq!(first.id, second.id);
        assert_eq!(
            list_generated_tool_candidates(data.path(), workspace.path())
                .unwrap()
                .len(),
            1
        );
        assert!(
            dismiss_generated_tool_candidate(data.path(), workspace.path(), &first.id).unwrap()
        );
    }

    #[test]
    fn deleting_a_generated_tool_removes_only_its_installed_directory() {
        let data = tempfile::tempdir().unwrap();
        let workspace = tempfile::tempdir().unwrap();
        let source = workspace.path().join("sample.py");
        std::fs::write(&source, "print('{}')").unwrap();
        let dir = project_root_for(data.path(), workspace.path())
            .unwrap()
            .join("sample");
        std::fs::create_dir_all(dir.join(".venv")).unwrap();
        std::fs::write(dir.join(".venv").join("owned-by-tool"), "x").unwrap();
        save_manifest(
            &dir,
            &GeneratedToolManifest {
                schema_version: 2,
                name: "sample".into(),
                description: "Sample".into(),
                runtime: ScriptRuntime::Python,
                entrypoint: "tool.py".into(),
                input_schema: json!({"type": "object"}),
                capabilities: vec!["process".into()],
                timeout_seconds: 60,
                max_output_bytes: MAX_OUTPUT,
                source_sha256: "abc".into(),
                enabled: false,
                version: 1,
                successes: 0,
                failures: 0,
                consecutive_protocol_failures: 0,
                dependencies: vec![],
                environment_sha256: "lock".into(),
                validation: GeneratedToolValidation::default(),
            },
        )
        .unwrap();

        assert!(delete_generated_tool(data.path(), workspace.path(), "generated__sample").unwrap());
        assert!(!dir.exists());
        assert!(source.exists(), "helper source must not be removed");
        assert!(!delete_generated_tool(data.path(), workspace.path(), "sample").unwrap());
        assert!(delete_generated_tool(data.path(), workspace.path(), "../escape").is_err());
    }
}
