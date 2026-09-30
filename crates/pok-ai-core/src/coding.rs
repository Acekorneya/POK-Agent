use std::{
    collections::{BTreeMap, BTreeSet},
    io::Read,
    path::{Path, PathBuf},
    time::{Duration, SystemTime},
};

use async_trait::async_trait;
use ignore::{WalkBuilder, overrides::OverrideBuilder};
use regex::Regex;
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::{
    PokError, Result,
    memory::atomic_write,
    policy::RiskClass,
    tool::{ArtifactEvidence, ArtifactWarning, SessionFileRecord, Tool, ToolContext},
};

const MAX_FILE_BYTES: u64 = 2 * 1024 * 1024;
pub fn register_coding_tools(registry: &mut crate::tool::ToolRegistry) {
    registry.register(ReadFileTool);
    registry.register(SourceOutlineTool);
    registry.register(ListDirectoryTool);
    registry.register(GrepSearchTool);
    registry.register(WriteFileTool);
    registry.register(EditFileTool);
    registry.register(UndoEditTool);
    registry.register(InspectArtifactTool);
    registry.register(RunCommandTool);
    registry.register(ManageCommandTool);
}

#[derive(Debug, Deserialize, JsonSchema)]
struct WriteFileArgs {
    filepath: PathBuf,
    content: String,
}

struct WriteFileTool;
#[async_trait]
impl Tool for WriteFileTool {
    fn name(&self) -> &'static str {
        "write_file"
    }
    fn description(&self) -> &'static str {
        "Create a UTF-8 file inside the current authorized filesystem scope. It may safely rewrite a file created by this same session only while its hash still matches the session ledger; pre-existing or externally changed files are refused. Rewrites record undo state. This can author a task-specific helper before run_command executes it."
    }
    fn input_schema(&self) -> Value {
        schema::<WriteFileArgs>()
    }
    fn risk(&self) -> RiskClass {
        RiskClass::WorkspaceWrite
    }
    fn target_path(&self, args: &Value) -> Option<PathBuf> {
        serde_json::from_value::<WriteFileArgs>(args.clone())
            .ok()
            .map(|args| args.filepath)
    }
    async fn execute(&self, args: Value, context: &ToolContext) -> Result<Value> {
        let args: WriteFileArgs = serde_json::from_value(args)?;
        if args.content.len() as u64 > MAX_FILE_BYTES {
            return Err(PokError::Tool(format!(
                "file exceeds {MAX_FILE_BYTES} byte limit"
            )));
        }
        let path = context.resolve_workspace_path(args.filepath)?;
        let mut operation = "created";
        let mut undo_path = None;
        if path.exists() {
            let before = std::fs::read_to_string(&path)?;
            let current_hash = hash_text(&before);
            let owned = context
                .session_files
                .lock()
                .get(&path)
                .cloned()
                .filter(|record| safe_session_rewrite(record, &current_hash));
            if owned.is_none() {
                return Err(PokError::Tool(
                    "file already exists and is not an unchanged file created by this session; use edit_file for a precise change".into(),
                ));
            }
            let record = UndoRecord {
                path: path.clone(),
                before,
                after_hash: hash_text(&args.content),
            };
            let record_path = write_undo_record(context, &record)?;
            undo_path = Some(record_path);
            operation = "rewritten";
        }
        atomic_write(&path, args.content.as_bytes())?;
        let sha256 = hash_text(&args.content);
        context.session_files.lock().insert(
            path.clone(),
            SessionFileRecord {
                sha256: sha256.clone(),
                created_by_session: true,
            },
        );
        context
            .artifact_evidence
            .lock()
            .push(inspect_artifact_path(&path, operation)?);
        Ok(
            json!({"path": path, "sha256": sha256, "operation": operation, "undo_record": undo_path}),
        )
    }
}

fn safe_session_rewrite(record: &SessionFileRecord, current_hash: &str) -> bool {
    record.created_by_session && record.sha256 == current_hash
}

fn schema<T: JsonSchema>() -> Value {
    serde_json::to_value(schema_for!(T)).expect("schema serializes")
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ReadFileArgs {
    path: PathBuf,
    /// One-based line at which to start reading.
    #[serde(default = "first_line")]
    start_line: usize,
    /// Maximum number of lines to return.
    #[serde(default = "default_read_lines")]
    line_count: usize,
}
const fn first_line() -> usize {
    1
}
const fn default_read_lines() -> usize {
    400
}
const MAX_READ_LINES: usize = 2_000;
const MAX_READ_CHARS: usize = 32_000;

struct ReadFileTool;
#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &'static str {
        "read_file"
    }
    fn description(&self) -> &'static str {
        "Read a bounded, line-addressable UTF-8 source/file chunk inside the authorized filesystem scope. Small files are returned completely. For large files, continue from next_start_line or search/outline first instead of requesting the whole file."
    }
    fn input_schema(&self) -> Value {
        schema::<ReadFileArgs>()
    }
    fn risk(&self) -> RiskClass {
        RiskClass::ReadOnly
    }
    fn target_path(&self, args: &Value) -> Option<PathBuf> {
        serde_json::from_value::<ReadFileArgs>(args.clone())
            .ok()
            .map(|a| a.path)
    }
    async fn execute(&self, args: Value, context: &ToolContext) -> Result<Value> {
        let args: ReadFileArgs = serde_json::from_value(args)?;
        if args.start_line == 0 {
            return Err(PokError::Tool("start_line must be one-based".into()));
        }
        if args.line_count == 0 || args.line_count > MAX_READ_LINES {
            return Err(PokError::Tool(format!(
                "line_count must be between 1 and {MAX_READ_LINES}"
            )));
        }
        let path = context.resolve_workspace_path(args.path)?;
        let metadata = std::fs::metadata(&path)?;
        if metadata.len() > MAX_FILE_BYTES {
            return Err(PokError::Tool(format!(
                "file exceeds {MAX_FILE_BYTES} byte limit"
            )));
        }
        let content = std::fs::read_to_string(&path)?;
        Ok(read_file_page(
            &path,
            &content,
            args.start_line,
            args.line_count,
        ))
    }
}

fn read_file_page(path: &Path, content: &str, start_line: usize, line_count: usize) -> Value {
    let lines = content.lines().collect::<Vec<_>>();
    let total_lines = lines.len();
    let start_index = start_line.saturating_sub(1).min(total_lines);
    let requested_end = start_index.saturating_add(line_count).min(total_lines);
    let mut end_index = start_index;
    let mut page_chars: usize = 0;
    while end_index < requested_end {
        let additional = lines[end_index].chars().count().saturating_add(1);
        if end_index > start_index && page_chars.saturating_add(additional) > MAX_READ_CHARS {
            break;
        }
        page_chars = page_chars.saturating_add(additional);
        end_index += 1;
    }
    let mut page = lines[start_index..end_index].join("\n");
    let line_content_truncated = page.chars().count() > MAX_READ_CHARS;
    if line_content_truncated {
        page = page.chars().take(MAX_READ_CHARS).collect();
    }
    let truncated = end_index < total_lines;
    json!({
        "path": path,
        "sha256": hash_text(content),
        "total_lines": total_lines,
        "start_line": if total_lines == 0 { 0 } else { start_index + 1 },
        "end_line": end_index,
        "content": page,
        "truncated": truncated,
        "next_start_line": truncated.then_some(end_index + 1),
        "line_content_truncated": line_content_truncated,
        "note": line_content_truncated.then_some("An unusually long/minified line was truncated; search it instead of loading it into model context."),
    })
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SourceOutlineArgs {
    path: PathBuf,
    #[serde(default = "default_outline_entries")]
    max_entries: usize,
}
const fn default_outline_entries() -> usize {
    300
}
const MAX_OUTLINE_ENTRIES: usize = 1_000;

#[derive(Debug, Serialize)]
struct OutlineEntry {
    line: usize,
    kind: &'static str,
    declaration: String,
}

struct SourceOutlineTool;
#[async_trait]
impl Tool for SourceOutlineTool {
    fn name(&self) -> &'static str {
        "source_outline"
    }
    fn description(&self) -> &'static str {
        "Return a bounded language-neutral outline of likely modules, classes, types, and functions in a UTF-8 source file. Use it before paging through a large file; verify exact ranges with read_file before editing."
    }
    fn input_schema(&self) -> Value {
        schema::<SourceOutlineArgs>()
    }
    fn risk(&self) -> RiskClass {
        RiskClass::ReadOnly
    }
    fn target_path(&self, args: &Value) -> Option<PathBuf> {
        serde_json::from_value::<SourceOutlineArgs>(args.clone())
            .ok()
            .map(|args| args.path)
    }
    async fn execute(&self, args: Value, context: &ToolContext) -> Result<Value> {
        let args: SourceOutlineArgs = serde_json::from_value(args)?;
        if args.max_entries == 0 || args.max_entries > MAX_OUTLINE_ENTRIES {
            return Err(PokError::Tool(format!(
                "max_entries must be between 1 and {MAX_OUTLINE_ENTRIES}"
            )));
        }
        let path = context.resolve_workspace_path(args.path)?;
        let metadata = std::fs::metadata(&path)?;
        if metadata.len() > MAX_FILE_BYTES {
            return Err(PokError::Tool(format!(
                "file exceeds {MAX_FILE_BYTES} byte limit"
            )));
        }
        let content = std::fs::read_to_string(&path)?;
        let (entries, total_entries) = source_outline(&content, args.max_entries);
        Ok(json!({
            "path": path,
            "sha256": hash_text(&content),
            "total_lines": content.lines().count(),
            "entries": entries,
            "total_entries": total_entries,
            "truncated": total_entries > args.max_entries,
            "note": "Heuristic outline; read exact source ranges before editing."
        }))
    }
}

fn source_outline(content: &str, max_entries: usize) -> (Vec<OutlineEntry>, usize) {
    let declarations = Regex::new(
        r"(?x)^\s*(?:
            (?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(fn)\s+\w+ |
            (class)\s+\w+ |
            (?:export\s+)?(?:default\s+)?(?:async\s+)?(function)\s+\w+ |
            (?:export\s+)?(?:abstract\s+)?(?:class|interface|type|enum)\s+\w+ |
            (?:pub\s+)?(?:struct|enum|trait|mod)\s+\w+ |
            (def)\s+\w+ |
            (?:async\s+)?(func)\s+\w+
        )",
    )
    .expect("outline regex");
    let mut total = 0;
    let mut entries = Vec::new();
    for (index, line) in content.lines().enumerate() {
        let Some(captures) = declarations.captures(line) else {
            continue;
        };
        total += 1;
        if entries.len() >= max_entries {
            continue;
        }
        let kind = if captures.get(1).is_some()
            || captures.get(3).is_some()
            || captures.get(5).is_some()
            || captures.get(6).is_some()
        {
            "function"
        } else if captures.get(2).is_some() {
            "class"
        } else {
            "type_or_module"
        };
        entries.push(OutlineEntry {
            line: index + 1,
            kind,
            declaration: line.trim().chars().take(300).collect(),
        });
    }
    (entries, total)
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ListArgs {
    #[serde(default = "dot")]
    path: PathBuf,
    #[serde(default = "default_depth")]
    max_depth: usize,
}
fn dot() -> PathBuf {
    PathBuf::from(".")
}
const fn default_depth() -> usize {
    4
}

struct ListDirectoryTool;
#[async_trait]
impl Tool for ListDirectoryTool {
    fn name(&self) -> &'static str {
        "list_directory"
    }
    fn description(&self) -> &'static str {
        "List files recursively inside the current authorized filesystem scope while respecting ignore files."
    }
    fn input_schema(&self) -> Value {
        schema::<ListArgs>()
    }
    fn risk(&self) -> RiskClass {
        RiskClass::ReadOnly
    }
    async fn execute(&self, args: Value, context: &ToolContext) -> Result<Value> {
        let args: ListArgs = serde_json::from_value(args)?;
        let root = context.resolve_workspace_path(args.path)?;
        let paths: Vec<String> = WalkBuilder::new(&root)
            .max_depth(Some(args.max_depth))
            .hidden(false)
            .build()
            .filter_map(std::result::Result::ok)
            .filter(|entry| entry.file_type().is_some_and(|kind| kind.is_file()))
            .take(5_000)
            .map(|entry| {
                entry
                    .path()
                    .strip_prefix(&context.workspace)
                    .unwrap_or(entry.path())
                    .display()
                    .to_string()
            })
            .collect();
        Ok(json!({"root": root, "files": paths}))
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct GrepArgs {
    pattern: String,
    #[serde(default = "dot")]
    path: PathBuf,
    #[serde(default = "default_matches")]
    max_matches: usize,
    #[serde(default)]
    offset: usize,
    #[serde(default = "default_context_lines")]
    context_lines: usize,
    #[serde(default)]
    include_globs: Vec<String>,
    #[serde(default)]
    exclude_globs: Vec<String>,
}
const fn default_matches() -> usize {
    50
}
const fn default_context_lines() -> usize {
    2
}
const MAX_SEARCH_MATCHES: usize = 500;
const MAX_SEARCH_CONTEXT_LINES: usize = 10;

#[derive(Debug, Serialize)]
struct SearchMatch {
    path: String,
    line: usize,
    text: String,
    context_start_line: usize,
    context: String,
}

struct GrepSearchTool;
#[async_trait]
impl Tool for GrepSearchTool {
    fn name(&self) -> &'static str {
        "grep_search"
    }
    fn description(&self) -> &'static str {
        "Search text across files using a Rust regular expression, bounded results, optional include/exclude globs, surrounding lines, and offset pagination. Prefer this before reading large files."
    }
    fn input_schema(&self) -> Value {
        schema::<GrepArgs>()
    }
    fn risk(&self) -> RiskClass {
        RiskClass::ReadOnly
    }
    async fn execute(&self, args: Value, context: &ToolContext) -> Result<Value> {
        let args: GrepArgs = serde_json::from_value(args)?;
        if args.max_matches == 0 || args.max_matches > MAX_SEARCH_MATCHES {
            return Err(PokError::Tool(format!(
                "max_matches must be between 1 and {MAX_SEARCH_MATCHES}"
            )));
        }
        if args.context_lines > MAX_SEARCH_CONTEXT_LINES {
            return Err(PokError::Tool(format!(
                "context_lines must not exceed {MAX_SEARCH_CONTEXT_LINES}"
            )));
        }
        let regex = Regex::new(&args.pattern).map_err(|error| PokError::Tool(error.to_string()))?;
        let root = context.resolve_workspace_path(args.path)?;
        let mut override_builder = OverrideBuilder::new(&root);
        for pattern in &args.include_globs {
            override_builder
                .add(pattern)
                .map_err(|error| PokError::Tool(error.to_string()))?;
        }
        for pattern in &args.exclude_globs {
            override_builder
                .add(&format!("!{pattern}"))
                .map_err(|error| PokError::Tool(error.to_string()))?;
        }
        let overrides = override_builder
            .build()
            .map_err(|error| PokError::Tool(error.to_string()))?;
        let mut walker = WalkBuilder::new(root);
        walker.hidden(false).overrides(overrides);
        let mut matches = Vec::new();
        let mut seen_matches = 0;
        let mut has_more = false;
        'files: for entry in walker.build().filter_map(std::result::Result::ok) {
            if matches.len() > args.max_matches {
                break;
            }
            if !entry.file_type().is_some_and(|kind| kind.is_file()) {
                continue;
            }
            let Ok(metadata) = entry.metadata() else {
                continue;
            };
            if metadata.len() > MAX_FILE_BYTES {
                continue;
            }
            let Ok(content) = std::fs::read_to_string(entry.path()) else {
                continue;
            };
            let lines = content.lines().collect::<Vec<_>>();
            for (index, line) in lines.iter().enumerate() {
                if regex.is_match(line) {
                    if seen_matches < args.offset {
                        seen_matches += 1;
                        continue;
                    }
                    if matches.len() == args.max_matches {
                        has_more = true;
                        break 'files;
                    }
                    let context_start = index.saturating_sub(args.context_lines);
                    let context_end = index
                        .saturating_add(args.context_lines)
                        .saturating_add(1)
                        .min(lines.len());
                    matches.push(SearchMatch {
                        path: entry
                            .path()
                            .strip_prefix(&context.workspace)
                            .unwrap_or(entry.path())
                            .display()
                            .to_string(),
                        line: index + 1,
                        text: line.chars().take(1_000).collect(),
                        context_start_line: context_start + 1,
                        context: lines[context_start..context_end]
                            .join("\n")
                            .chars()
                            .take(4_000)
                            .collect(),
                    });
                    seen_matches += 1;
                }
            }
        }
        let returned = matches.len();
        Ok(json!({
            "matches": matches,
            "offset": args.offset,
            "returned": returned,
            "truncated": has_more,
            "next_offset": has_more.then_some(args.offset.saturating_add(returned)),
        }))
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct EditArgs {
    filepath: PathBuf,
    target_content: String,
    replacement_content: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct UndoRecord {
    path: PathBuf,
    before: String,
    after_hash: String,
}

struct EditFileTool;
#[async_trait]
impl Tool for EditFileTool {
    fn name(&self) -> &'static str {
        "edit_file"
    }
    fn description(&self) -> &'static str {
        "Atomically replace one exact, uniquely matching text block and record undo state."
    }
    fn input_schema(&self) -> Value {
        schema::<EditArgs>()
    }
    fn risk(&self) -> RiskClass {
        RiskClass::WorkspaceWrite
    }
    fn target_path(&self, args: &Value) -> Option<PathBuf> {
        serde_json::from_value::<EditArgs>(args.clone())
            .ok()
            .map(|a| a.filepath)
    }
    async fn execute(&self, args: Value, context: &ToolContext) -> Result<Value> {
        let args: EditArgs = serde_json::from_value(args)?;
        if args.target_content.is_empty() {
            return Err(PokError::Tool("target_content cannot be empty".into()));
        }
        let path = context.resolve_workspace_path(args.filepath)?;
        let before = std::fs::read_to_string(&path)?;
        let count = before.match_indices(&args.target_content).count();
        if count != 1 {
            return Err(PokError::Tool(format!(
                "target_content must occur exactly once; found {count}"
            )));
        }
        let after = before.replacen(&args.target_content, &args.replacement_content, 1);
        atomic_write(&path, after.as_bytes())?;
        let record = UndoRecord {
            path: path.clone(),
            before,
            after_hash: hash_text(&after),
        };
        let undo_path = write_undo_record(context, &record)?;
        if let Some(owned) = context.session_files.lock().get_mut(&path) {
            owned.sha256 = record.after_hash.clone();
        }
        context
            .artifact_evidence
            .lock()
            .push(inspect_artifact_path(&path, "edited")?);
        Ok(json!({"path": path, "undo_record": undo_path, "new_sha256": record.after_hash}))
    }
}

fn write_undo_record(context: &ToolContext, record: &UndoRecord) -> Result<PathBuf> {
    std::fs::create_dir_all(context.data_dir.join("undo"))?;
    let undo_path = context.data_dir.join("undo").join(format!(
        "{}-{}-{}.json",
        context.session_id,
        chrono::Utc::now().timestamp_millis(),
        uuid::Uuid::new_v4()
    ));
    atomic_write(&undo_path, &serde_json::to_vec(record)?)?;
    Ok(undo_path)
}

#[derive(Debug, Deserialize, JsonSchema)]
struct InspectArtifactArgs {
    path: PathBuf,
}

struct InspectArtifactTool;

#[async_trait]
impl Tool for InspectArtifactTool {
    fn name(&self) -> &'static str {
        "inspect_artifact"
    }

    fn description(&self) -> &'static str {
        "Inspect a local artifact using lightweight format-neutral evidence: existence, size, hash, text/image metadata, or package structure. Use this before claiming that a created file meets the user's requested content or visual result."
    }

    fn input_schema(&self) -> Value {
        schema::<InspectArtifactArgs>()
    }

    fn risk(&self) -> RiskClass {
        RiskClass::ReadOnly
    }

    fn target_path(&self, args: &Value) -> Option<PathBuf> {
        serde_json::from_value::<InspectArtifactArgs>(args.clone())
            .ok()
            .map(|args| args.path)
    }

    async fn execute(&self, args: Value, context: &ToolContext) -> Result<Value> {
        let args: InspectArtifactArgs = serde_json::from_value(args)?;
        let path = context.resolve_readable_artifact(&args.path)?;
        let evidence = inspect_artifact_path(&path, "inspected")?;
        context.artifact_evidence.lock().push(evidence.clone());
        Ok(serde_json::to_value(evidence)?)
    }
}

fn inspect_artifact_path(path: &Path, operation: &str) -> Result<ArtifactEvidence> {
    let metadata = std::fs::metadata(path)?;
    if !metadata.is_file() {
        return Err(PokError::Tool(format!(
            "artifact path is not a regular file: {}",
            path.display()
        )));
    }
    let bytes = std::fs::read(path)?;
    let sha256 = hash_bytes(&bytes);
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let (artifact_type, integrity, structure) = if bytes.starts_with(b"PK\x03\x04") {
        inspect_package(path)?
    } else if let Ok(text) = std::str::from_utf8(&bytes) {
        let lines = text.lines().count();
        let preview = text.chars().take(4_096).collect::<String>();
        (
            if is_source_extension(&extension) {
                "text_source"
            } else {
                "text"
            }
            .to_owned(),
            "readable".to_owned(),
            json!({
                "line_count": lines,
                "character_count": text.chars().count(),
                "preview": preview,
                "preview_truncated": text.chars().count() > 4_096,
            }),
        )
    } else if let Ok((width, height)) = image::image_dimensions(path) {
        (
            "image".to_owned(),
            "decodable".to_owned(),
            json!({"width": width, "height": height, "extension": extension}),
        )
    } else {
        (
            "binary".to_owned(),
            "metadata_only".to_owned(),
            json!({"extension": extension}),
        )
    };
    let modified_at = metadata
        .modified()
        .ok()
        .map(chrono::DateTime::<chrono::Utc>::from)
        .unwrap_or_else(chrono::Utc::now)
        .to_rfc3339();
    let warnings = if artifact_type.ends_with("_package") {
        package_warnings(&structure)
    } else {
        Vec::new()
    };
    let validation_status = if warnings.iter().any(|warning| warning.severity == "error") {
        "invalid"
    } else if !warnings.is_empty() || integrity == "metadata_only" {
        "needs_review"
    } else {
        "valid"
    };
    Ok(ArtifactEvidence {
        evidence_id: uuid::Uuid::new_v4(),
        path: path.to_path_buf(),
        operation: operation.to_owned(),
        artifact_type,
        integrity,
        validation_status: validation_status.into(),
        size_bytes: metadata.len(),
        sha256,
        modified_at,
        structure,
        warnings,
        observed_at: chrono::Utc::now().to_rfc3339(),
    })
}

fn inspect_package(path: &Path) -> Result<(String, String, Value)> {
    let file = std::fs::File::open(path)?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|error| PokError::Tool(format!("invalid ZIP/package artifact: {error}")))?;
    let mut names = Vec::new();
    let mut all_names = BTreeSet::new();
    let mut document_xml = 0_u32;
    let mut worksheet_xml = 0_u32;
    let mut slide_xml = 0_u32;
    let mut chart_xml = 0_u32;
    let mut media_files = 0_u32;
    let mut package_text = BTreeMap::new();
    for index in 0..archive.len() {
        let mut entry = archive
            .by_index(index)
            .map_err(|error| PokError::Tool(format!("invalid package entry: {error}")))?;
        let name = entry.name().replace('\\', "/");
        all_names.insert(name.clone());
        document_xml += u32::from(name == "word/document.xml");
        worksheet_xml +=
            u32::from(name.starts_with("xl/worksheets/sheet") && name.ends_with(".xml"));
        slide_xml += u32::from(name.starts_with("ppt/slides/slide") && name.ends_with(".xml"));
        chart_xml += u32::from(name.contains("/charts/chart") && name.ends_with(".xml"));
        media_files += u32::from(name.contains("/media/") && !name.ends_with('/'));
        if names.len() < 200 {
            names.push(name.clone());
        }
        if is_semantic_package_part(&name) && entry.size() <= 2 * 1024 * 1024 {
            let mut text = String::new();
            if entry.read_to_string(&mut text).is_ok() {
                package_text.insert(name, text);
            }
        }
    }
    let missing_relationship_targets =
        missing_package_relationship_targets(&package_text, &all_names);
    let family = if document_xml > 0 {
        "document_package"
    } else if worksheet_xml > 0 {
        "workbook_package"
    } else if slide_xml > 0 {
        "presentation_package"
    } else {
        "zip_package"
    };
    let semantics = semantic_package_summary(family, &package_text);
    Ok((
        family.into(),
        "package_readable".into(),
        json!({
            "entry_count": archive.len(),
            "entries": names,
            "entries_truncated": archive.len() > 200,
            "document_parts": document_xml,
            "worksheet_parts": worksheet_xml,
            "slide_parts": slide_xml,
            "chart_parts": chart_xml,
            "media_files": media_files,
            "missing_relationship_targets": missing_relationship_targets,
            "semantic": semantics,
        }),
    ))
}

fn is_semantic_package_part(name: &str) -> bool {
    name == "xl/workbook.xml"
        || name == "xl/sharedStrings.xml"
        || name == "word/document.xml"
        || (name.starts_with("xl/worksheets/") && name.ends_with(".xml"))
        || (name.starts_with("xl/charts/") && name.ends_with(".xml"))
        || (name.starts_with("ppt/slides/") && name.ends_with(".xml"))
        || name.ends_with(".rels")
        || name == "[Content_Types].xml"
}

fn missing_package_relationship_targets(
    parts: &BTreeMap<String, String>,
    names: &BTreeSet<String>,
) -> Vec<String> {
    let mut missing = Vec::new();
    for (relationship_part, xml) in parts.iter().filter(|(name, _)| name.ends_with(".rels")) {
        let Ok(document) = roxmltree::Document::parse(xml) else {
            continue;
        };
        let base = relationship_source_directory(relationship_part);
        for relationship in document
            .descendants()
            .filter(|node| node.is_element() && node.tag_name().name() == "Relationship")
        {
            if relationship.attribute("TargetMode") == Some("External") {
                continue;
            }
            let Some(target) = relationship.attribute("Target") else {
                continue;
            };
            let resolved = normalize_package_path(&base, target);
            if !names.contains(&resolved) {
                missing.push(format!("{relationship_part} -> {resolved}"));
            }
        }
    }
    missing
}

fn relationship_source_directory(relationship_part: &str) -> String {
    if relationship_part == "_rels/.rels" {
        return String::new();
    }
    let Some((prefix, file)) = relationship_part.rsplit_once("/_rels/") else {
        return String::new();
    };
    let source_name = file.strip_suffix(".rels").unwrap_or(file);
    let source = format!("{prefix}/{source_name}");
    source
        .rsplit_once('/')
        .map_or(String::new(), |(dir, _)| dir.into())
}

fn normalize_package_path(base: &str, target: &str) -> String {
    let mut components = Vec::new();
    let combined = if target.starts_with('/') || base.is_empty() {
        target.trim_start_matches('/').to_owned()
    } else {
        format!("{base}/{target}")
    };
    for component in combined.split('/') {
        match component {
            "" | "." => {}
            ".." => {
                components.pop();
            }
            value => components.push(value),
        }
    }
    components.join("/")
}

fn semantic_package_summary(family: &str, parts: &BTreeMap<String, String>) -> Value {
    let xml_errors = parts
        .iter()
        .filter_map(|(name, xml)| {
            roxmltree::Document::parse(xml)
                .err()
                .map(|error| format!("{name}: {error}"))
        })
        .collect::<Vec<_>>();
    let text_values = |xml: &str| {
        Regex::new(r"(?s)<(?:\w+:)?t[^>]*>(.*?)</(?:\w+:)?t>")
            .expect("valid text element regex")
            .captures_iter(xml)
            .filter_map(|capture| capture.get(1))
            .map(|value| decode_xml_text(value.as_str()))
            .filter(|value| !value.trim().is_empty())
            .take(500)
            .collect::<Vec<_>>()
    };
    if family == "workbook_package" {
        let workbook_relationships = workbook_relationship_targets(parts);
        let workbook_sheets = workbook_sheet_parts(parts, &workbook_relationships);
        let sheet_names = workbook_sheets
            .iter()
            .map(|(name, _)| name.clone())
            .collect::<Vec<_>>();
        let worksheets = workbook_sheets
            .iter()
            .filter_map(|(sheet_name, part)| {
                let xml = parts.get(part)?;
                let dimensions = Regex::new(r#"<dimension\b[^>]*\bref="([^"]+)""#)
                    .expect("valid dimension regex")
                    .captures(xml)
                    .and_then(|capture| capture.get(1))
                    .map(|value| value.as_str().to_owned());
                let (row_count, column_count) = dimensions
                    .as_deref()
                    .and_then(worksheet_dimension_counts)
                    .unwrap_or((0, 0));
                json!({
                    "name": sheet_name,
                    "part": part,
                    "dimensions": dimensions,
                    "row_count": row_count,
                    "column_count": column_count,
                    "cell_count": xml.matches("<c ").count(),
                })
                .into()
            })
            .collect::<Vec<_>>();
        let charts = parts
            .iter()
            .filter(|(name, _)| name.starts_with("xl/charts/"))
            .map(|(name, xml)| inspect_chart_xml(name, xml))
            .collect::<Vec<_>>();
        json!({
            "sheet_names": sheet_names,
            "worksheets": worksheets,
            "charts": charts,
            "shared_text": parts.get("xl/sharedStrings.xml").map(|xml| text_values(xml)).unwrap_or_default(),
            "xml_errors": xml_errors,
        })
    } else if family == "presentation_package" {
        let slides = parts
            .iter()
            .filter(|(name, _)| name.starts_with("ppt/slides/"))
            .map(|(name, xml)| json!({"part": name, "text": text_values(xml)}))
            .collect::<Vec<_>>();
        json!({"slides": slides, "xml_errors": xml_errors})
    } else if family == "document_package" {
        json!({"text": parts.get("word/document.xml").map(|xml| text_values(xml)).unwrap_or_default(), "xml_errors": xml_errors})
    } else {
        json!({"xml_errors": xml_errors})
    }
}

fn workbook_relationship_targets(parts: &BTreeMap<String, String>) -> BTreeMap<String, String> {
    let Some(xml) = parts.get("xl/_rels/workbook.xml.rels") else {
        return BTreeMap::new();
    };
    let Ok(document) = roxmltree::Document::parse(xml) else {
        return BTreeMap::new();
    };
    document
        .descendants()
        .filter(|node| node.is_element() && node.tag_name().name() == "Relationship")
        .filter_map(|node| {
            let id = node.attribute("Id")?;
            let target = node.attribute("Target")?;
            Some((id.to_owned(), normalize_package_path("xl", target)))
        })
        .collect()
}

fn workbook_sheet_parts(
    parts: &BTreeMap<String, String>,
    relationships: &BTreeMap<String, String>,
) -> Vec<(String, String)> {
    let Some(xml) = parts.get("xl/workbook.xml") else {
        return Vec::new();
    };
    let Ok(document) = roxmltree::Document::parse(xml) else {
        return Vec::new();
    };
    document
        .descendants()
        .filter(|node| node.is_element() && node.tag_name().name() == "sheet")
        .filter_map(|node| {
            let name = node.attribute("name")?;
            let relationship_id = node
                .attributes()
                .find(|attribute| attribute.name() == "id")?
                .value();
            let part = relationships.get(relationship_id)?;
            Some((decode_xml_text(name), part.clone()))
        })
        .collect()
}

fn worksheet_dimension_counts(reference: &str) -> Option<(u64, u64)> {
    let end = reference.rsplit_once(':').map_or(reference, |(_, end)| end);
    let letters = end
        .chars()
        .take_while(|character| character.is_ascii_alphabetic())
        .collect::<String>();
    let row = end
        .chars()
        .skip_while(|character| character.is_ascii_alphabetic())
        .collect::<String>()
        .parse::<u64>()
        .ok()?;
    let column = letters.chars().try_fold(0_u64, |value, character| {
        let digit = u64::from(character.to_ascii_uppercase() as u8 - b'A' + 1);
        value.checked_mul(26)?.checked_add(digit)
    })?;
    Some((row, column))
}

fn inspect_chart_xml(name: &str, xml: &str) -> Value {
    let chart_type = [
        "barChart",
        "lineChart",
        "pieChart",
        "scatterChart",
        "areaChart",
        "doughnutChart",
        "radarChart",
    ]
    .iter()
    .find(|kind| xml.contains(&format!("<c:{kind}")) || xml.contains(&format!("<{kind}")))
    .copied()
    .unwrap_or("unknown");
    let mut series = Vec::new();
    if let Ok(document) = roxmltree::Document::parse(xml) {
        for block in document
            .descendants()
            .filter(|node| node.is_element() && node.tag_name().name() == "ser")
        {
            let formulas = block
                .descendants()
                .filter(|node| node.is_element() && node.tag_name().name() == "f")
                .filter_map(|node| node.text())
                .map(str::to_owned)
                .collect::<Vec<_>>();
            let cached_values = block
                .descendants()
                .filter(|node| node.is_element() && node.tag_name().name() == "v")
                .filter_map(|node| node.text())
                .map(str::to_owned)
                .collect::<Vec<_>>();
            let point_count = block
                .descendants()
                .filter(|node| node.is_element() && node.tag_name().name() == "pt")
                .count();
            series.push(json!({
                "formulas": formulas,
                "cached_values": cached_values,
                    "point_count": point_count,
            }));
        }
    }
    json!({
        "part": name,
        "chart_type": chart_type,
        "series_count": series.len(),
        "series": series,
    })
}

fn decode_xml_text(value: &str) -> String {
    value
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

fn package_warnings(structure: &Value) -> Vec<ArtifactWarning> {
    let mut warnings = Vec::new();
    let sheet_names = structure
        .pointer("/semantic/sheet_names")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
    let worksheet_dimensions = structure
        .pointer("/semantic/worksheets")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|worksheet| {
            Some((
                worksheet.get("name")?.as_str()?.to_owned(),
                (
                    worksheet.get("row_count")?.as_u64()?,
                    worksheet.get("column_count")?.as_u64()?,
                ),
            ))
        })
        .collect::<BTreeMap<_, _>>();
    if let Some(targets) = structure
        .get("missing_relationship_targets")
        .and_then(Value::as_array)
    {
        warnings.extend(
            targets
                .iter()
                .filter_map(Value::as_str)
                .map(|message| ArtifactWarning {
                    severity: "error".into(),
                    code: "missing_package_relationship_target".into(),
                    message: message.into(),
                }),
        );
    }
    if let Some(errors) = structure
        .pointer("/semantic/xml_errors")
        .and_then(Value::as_array)
    {
        warnings.extend(
            errors
                .iter()
                .filter_map(Value::as_str)
                .map(|message| ArtifactWarning {
                    severity: "error".into(),
                    code: "malformed_package_xml".into(),
                    message: message.into(),
                }),
        );
    }
    let Some(charts) = structure
        .pointer("/semantic/charts")
        .and_then(Value::as_array)
    else {
        return warnings;
    };
    for chart in charts {
        let series = chart
            .get("series")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        if series.is_empty() {
            warnings.push(ArtifactWarning {
                severity: "warning".into(),
                code: "chart_without_series".into(),
                message: "chart part contains no readable data series".into(),
            });
        }
        if chart.get("chart_type").and_then(Value::as_str) == Some("unknown") {
            warnings.push(ArtifactWarning {
                severity: "warning".into(),
                code: "unknown_chart_type".into(),
                message: "chart type could not be identified".into(),
            });
        }
        let one_point = series
            .iter()
            .filter(|item| item.get("point_count").and_then(Value::as_u64).unwrap_or(0) <= 1)
            .count();
        if series.len() >= 4 && one_point == series.len() {
            warnings.push(ArtifactWarning {
                severity: "warning".into(),
                code: "many_one_point_series".into(),
                message: format!(
                    "chart has {} separate one-point series; categories may have been encoded as series",
                    series.len()
                ),
            });
        }
        let mut series_signatures = BTreeSet::new();
        let mut value_formulas = BTreeSet::new();
        let mut duplicate_series = false;
        for item in &series {
            let formulas = item
                .get("formulas")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>();
            if formulas.len() < 2 {
                warnings.push(ArtifactWarning {
                    severity: "warning".into(),
                    code: "incomplete_chart_series".into(),
                    message: "chart series is missing a category or value reference".into(),
                });
            }
            for formula in &formulas {
                if let Some((sheet, row, column)) = chart_formula_reference(formula) {
                    match worksheet_dimensions.get(&sheet) {
                        None if !sheet_names.contains(&sheet) => warnings.push(ArtifactWarning {
                            severity: "warning".into(),
                            code: "chart_reference_missing_sheet".into(),
                            message: format!("chart formula references missing sheet: {formula}"),
                        }),
                        Some((max_row, max_column)) if row > *max_row || column > *max_column => {
                            warnings.push(ArtifactWarning {
                                severity: "warning".into(),
                                code: "chart_reference_outside_worksheet".into(),
                                message: format!(
                                    "chart formula exceeds worksheet dimensions: {formula}"
                                ),
                            });
                        }
                        None | Some(_) => {}
                    }
                }
                if formula_references_header(formula) {
                    warnings.push(ArtifactWarning {
                        severity: "warning".into(),
                        code: "possible_header_reference".into(),
                        message: format!("chart formula may include a header cell: {formula}"),
                    });
                }
            }
            let signature = formulas.join("\u{1f}");
            if !signature.is_empty() && !series_signatures.insert(signature) {
                duplicate_series = true;
            }
            if let Some(value_formula) = formulas.last()
                && !value_formulas.insert((*value_formula).to_owned())
            {
                duplicate_series = true;
            }
        }
        if duplicate_series {
            warnings.push(ArtifactWarning {
                severity: "warning".into(),
                code: "duplicate_chart_references".into(),
                message: "chart series repeat the same value range or complete series references"
                    .into(),
            });
        }
    }
    warnings
}

fn chart_formula_reference(formula: &str) -> Option<(String, u64, u64)> {
    let (sheet, range) = formula.rsplit_once('!')?;
    let sheet = sheet.trim().trim_matches('\'').replace("''", "'");
    let end = range.rsplit_once(':').map_or(range, |(_, end)| end);
    let cell = end.replace('$', "");
    let (row, column) = worksheet_dimension_counts(&cell)?;
    Some((sheet, row, column))
}

fn formula_references_header(formula: &str) -> bool {
    Regex::new(r"\$[A-Za-z]+\$1(?:\D|$)")
        .expect("valid header reference regex")
        .is_match(formula)
}

fn is_source_extension(extension: &str) -> bool {
    matches!(
        extension,
        "c" | "cc"
            | "cpp"
            | "cs"
            | "css"
            | "go"
            | "html"
            | "java"
            | "js"
            | "json"
            | "jsx"
            | "md"
            | "ps1"
            | "py"
            | "rs"
            | "sh"
            | "toml"
            | "ts"
            | "tsx"
            | "yaml"
            | "yml"
    )
}

#[derive(Debug, Deserialize, JsonSchema)]
struct UndoArgs {
    undo_record: PathBuf,
}

struct UndoEditTool;
#[async_trait]
impl Tool for UndoEditTool {
    fn name(&self) -> &'static str {
        "undo_edit"
    }
    fn description(&self) -> &'static str {
        "Restore an edit when the file still matches the recorded post-edit hash."
    }
    fn input_schema(&self) -> Value {
        schema::<UndoArgs>()
    }
    fn risk(&self) -> RiskClass {
        RiskClass::WorkspaceWrite
    }
    async fn execute(&self, args: Value, context: &ToolContext) -> Result<Value> {
        let args: UndoArgs = serde_json::from_value(args)?;
        let undo_root = dunce::canonicalize(context.data_dir.join("undo"))?;
        let record_path = dunce::canonicalize(args.undo_record)?;
        if !record_path.starts_with(&undo_root) {
            return Err(PokError::Tool(
                "undo record is outside POK-Agent undo storage".into(),
            ));
        }
        let record: UndoRecord = serde_json::from_slice(&std::fs::read(record_path)?)?;
        let path = context.resolve_workspace_path(record.path)?;
        let current = std::fs::read_to_string(&path)?;
        if hash_text(&current) != record.after_hash {
            return Err(PokError::Tool(
                "file changed after the edit; refusing destructive undo".into(),
            ));
        }
        atomic_write(&path, record.before.as_bytes())?;
        if let Some(owned) = context.session_files.lock().get_mut(&path) {
            owned.sha256 = hash_text(&record.before);
        }
        context
            .artifact_evidence
            .lock()
            .push(inspect_artifact_path(&path, "restored")?);
        Ok(json!({"restored": path}))
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct CommandArgs {
    command: String,
    #[serde(default = "dot")]
    cwd: PathBuf,
    #[serde(default)]
    timeout_seconds: Option<u64>,
    #[serde(default)]
    background: bool,
    #[serde(default)]
    pty: bool,
}

struct RunCommandTool;
#[async_trait]
impl Tool for RunCommandTool {
    fn name(&self) -> &'static str {
        "run_command"
    }
    fn description(&self) -> &'static str {
        "Run a complete shell command with live output and safe process-tree ownership. Fast commands finish inline; after 30 seconds a still-running foreground command is returned as a background task without restarting it. Set background=true to return immediately. Use manage_command to poll, read, wait, kill, or send stdin. timeout_seconds is an optional hard execution limit; 0 means unlimited. On Windows submit the PowerShell body directly (Windows PowerShell 5.1): its Set-Content/Out-File -Encoding UTF8 write a byte-order mark that breaks JSON, settings, and CSV readers, so write text files with [IO.File]::WriteAllText($path, $text, [Text.UTF8Encoding]::new($false)). PTY requests must run in background."
    }
    fn input_schema(&self) -> Value {
        schema::<CommandArgs>()
    }
    fn risk(&self) -> RiskClass {
        RiskClass::ProcessExecution
    }
    fn risk_for_arguments(&self, arguments: &Value) -> RiskClass {
        let Ok(args) = serde_json::from_value::<CommandArgs>(arguments.clone()) else {
            return RiskClass::ProcessExecution;
        };
        if command_is_os_critical(&args.command) {
            RiskClass::HighImpact
        } else {
            RiskClass::ProcessExecution
        }
    }
    fn approval_key(&self, _args: &Value) -> Option<String> {
        Some("run_command".into())
    }
    fn target_path(&self, args: &Value) -> Option<PathBuf> {
        serde_json::from_value::<CommandArgs>(args.clone())
            .ok()
            .map(|a| a.cwd)
    }
    async fn execute(&self, args: Value, context: &ToolContext) -> Result<Value> {
        let args: CommandArgs = serde_json::from_value(args)?;
        if args.pty && !args.background {
            return Err(PokError::Tool(
                "pty=true requires background=true so the model can send input with manage_command"
                    .into(),
            ));
        }
        let cwd = context.resolve_workspace_path(args.cwd)?;
        let mut before_files = workspace_file_snapshot(&cwd);
        for candidate in command_artifact_candidates(&args.command, "", "") {
            let candidate = if candidate.is_absolute() {
                candidate
            } else {
                cwd.join(candidate)
            };
            let Ok(path) = context.resolve_workspace_path(candidate) else {
                continue;
            };
            let canonical = dunce::canonicalize(&path).unwrap_or(path);
            if let Ok(metadata) = std::fs::metadata(&canonical) {
                before_files.insert(canonical, (metadata.len(), metadata.modified().ok()));
            }
        }
        let call_id = context
            .current_tool_call_id
            .lock()
            .clone()
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        let hard_timeout = args
            .timeout_seconds
            .filter(|seconds| *seconds > 0)
            .map(Duration::from_secs);
        let started = context
            .command_manager
            .spawn(
                call_id,
                args.command.clone(),
                cwd.clone(),
                hard_timeout,
                args.pty,
            )
            .await?;
        let snapshot = if args.background {
            started
        } else {
            tokio::select! {
                () = context.cancellation.cancelled() => {
                    let _ = context.command_manager.kill(&started.task_id, "cancelled").await;
                    return Err(PokError::Cancelled);
                }
                result = context.command_manager.wait(
                    &started.task_id,
                    Duration::from_secs(context.command_timeout_seconds.max(1)),
                ) => result?,
            }
        };
        let terminal = snapshot.status != crate::commands::CommandStatus::Running;
        let is_success = snapshot.status == crate::commands::CommandStatus::Completed
            && snapshot.exit_code == Some(0);
        let stdout_text = snapshot.stdout.clone();
        let stderr_text = snapshot.stderr.clone();
        let failure_kind = (!is_success).then(|| command_failure_kind(&stderr_text));
        let failure_summary = (!is_success).then(|| command_failure_summary(&stderr_text));
        let mut artifacts = Vec::new();
        if is_success {
            let mut candidates =
                command_artifact_candidates(&args.command, &stdout_text, &stderr_text)
                    .into_iter()
                    .map(|candidate| {
                        if candidate.is_absolute() {
                            candidate
                        } else {
                            cwd.join(candidate)
                        }
                    })
                    .collect::<Vec<_>>();
            let recent_changes = recent_command_changes(&cwd, &before_files);
            let changed_paths = recent_changes
                .iter()
                .map(|path| dunce::canonicalize(path).unwrap_or_else(|_| path.clone()))
                .collect::<BTreeSet<_>>();
            candidates.extend(recent_changes);
            let mut seen = BTreeSet::new();
            for candidate in candidates {
                let Ok(path) = context.resolve_workspace_path(candidate) else {
                    continue;
                };
                let canonical = dunce::canonicalize(&path).unwrap_or(path);
                if !seen.insert(canonical.clone()) {
                    continue;
                }
                let was_present = before_files.contains_key(&canonical);
                if let Ok(mut evidence) = inspect_artifact_path(&canonical, "command_observed") {
                    let current_state = std::fs::metadata(&canonical)
                        .ok()
                        .map(|metadata| (metadata.len(), metadata.modified().ok()));
                    let metadata_changed = before_files.get(&canonical) != current_state.as_ref();
                    let known_hash_changed = context
                        .session_files
                        .lock()
                        .get(&canonical)
                        .is_some_and(|record| record.sha256 != evidence.sha256);
                    evidence.operation = command_artifact_operation_from_evidence(
                        was_present,
                        changed_paths.contains(&canonical),
                        metadata_changed,
                        known_hash_changed,
                    )
                    .into();
                    context.session_files.lock().insert(
                        canonical,
                        SessionFileRecord {
                            sha256: evidence.sha256.clone(),
                            created_by_session: !was_present,
                        },
                    );
                    context.artifact_evidence.lock().push(evidence.clone());
                    artifacts.push(evidence);
                }
            }
        }
        let mut value = crate::commands::snapshot_json(&snapshot);
        value["backgrounded"] = json!(!terminal);
        value["failure_kind"] = json!(failure_kind);
        value["failure_summary"] = json!(failure_summary);
        value["artifacts"] = json!(artifacts);
        let timed_out = value.get("status").and_then(Value::as_str) == Some("timed_out");
        if (!terminal || timed_out) && automates_application(&args.command) {
            value["hint"] = json!(
                "This command drives an application through COM automation. COM calls wait while that application shows a dialog or is busy, so the command is likely blocked, not slow. Look at the application's window (a dialog may need an answer) or do the step through its UI instead of waiting on this command again."
            );
        }
        Ok(value)
    }
}

/// Whether a command scripts another application through COM automation,
/// which blocks while that application shows a dialog.
fn automates_application(command: &str) -> bool {
    let command = command.to_ascii_lowercase();
    command.contains("-comobject")
        || command.contains("getactiveobject")
        || command.contains("createobject(")
}

fn command_artifact_operation(
    was_present: bool,
    metadata_changed: bool,
    known_hash_changed: bool,
) -> &'static str {
    if !was_present {
        "command_created"
    } else if metadata_changed || known_hash_changed {
        "command_modified"
    } else {
        "command_observed"
    }
}

fn command_artifact_operation_from_evidence(
    was_present: bool,
    changed_during_command: bool,
    metadata_changed: bool,
    known_hash_changed: bool,
) -> &'static str {
    if !was_present && !changed_during_command {
        // A path first mentioned by command output is not proof the command
        // created it. Creation requires pre/post evidence from this execution.
        "command_observed"
    } else {
        command_artifact_operation(was_present, metadata_changed, known_hash_changed)
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum CommandAction {
    List,
    Poll,
    Read,
    Wait,
    Kill,
    Write,
    Submit,
    Close,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ManageCommandArgs {
    action: CommandAction,
    #[serde(default)]
    task_id: Option<String>,
    #[serde(default)]
    offset: Option<u64>,
    #[serde(default)]
    max_bytes: Option<usize>,
    #[serde(default)]
    timeout_seconds: Option<u64>,
    #[serde(default)]
    data: Option<String>,
}

struct ManageCommandTool;
#[async_trait]
impl Tool for ManageCommandTool {
    fn name(&self) -> &'static str {
        "manage_command"
    }
    fn description(&self) -> &'static str {
        "Manage commands started by run_command in this conversation. Actions: list; poll status and preview; read paginated full output; wait without killing on wait timeout; kill the full process tree; write raw stdin; submit stdin plus Enter; close stdin/send EOF."
    }
    fn input_schema(&self) -> Value {
        schema::<ManageCommandArgs>()
    }
    fn risk(&self) -> RiskClass {
        RiskClass::ReadOnly
    }
    async fn execute(&self, args: Value, context: &ToolContext) -> Result<Value> {
        let args: ManageCommandArgs = serde_json::from_value(args)?;
        if matches!(args.action, CommandAction::List) {
            return Ok(
                json!({"commands": context.command_manager.list().await.iter().map(crate::commands::snapshot_json).collect::<Vec<_>>() }),
            );
        }
        let task_id = args
            .task_id
            .ok_or_else(|| PokError::Tool("task_id is required for this action".into()))?;
        match args.action {
            CommandAction::List => unreachable!(),
            CommandAction::Poll => context
                .command_manager
                .get(&task_id)
                .await
                .map(|s| crate::commands::snapshot_json(&s))
                .ok_or_else(|| PokError::Tool(format!("unknown command task id {task_id}"))),
            CommandAction::Wait => {
                let snapshot = tokio::select! {
                    () = context.cancellation.cancelled() => return Err(PokError::Cancelled),
                    result = context.command_manager.wait(&task_id, Duration::from_secs(args.timeout_seconds.unwrap_or(30).clamp(1, 600))) => result?,
                };
                let mut value = crate::commands::snapshot_json(&snapshot);
                if snapshot.status == crate::commands::CommandStatus::Completed {
                    value["artifacts"] = json!(observe_completed_command_artifacts(
                        context,
                        &snapshot.command,
                        &snapshot.cwd,
                        &snapshot.stdout,
                        &snapshot.stderr,
                    ));
                }
                Ok(value)
            }
            CommandAction::Kill => Ok(crate::commands::snapshot_json(
                &context.command_manager.kill(&task_id, "model_kill").await?,
            )),
            CommandAction::Read => {
                let (output, next_offset, truncated) = context
                    .command_manager
                    .read(
                        &task_id,
                        args.offset.unwrap_or(0),
                        args.max_bytes.unwrap_or(16 * 1024),
                    )
                    .await?;
                Ok(
                    json!({"task_id": task_id, "output": output, "next_offset": next_offset, "truncated": truncated}),
                )
            }
            CommandAction::Write | CommandAction::Submit => {
                let submit = matches!(args.action, CommandAction::Submit);
                context
                    .command_manager
                    .write(&task_id, args.data.unwrap_or_default().as_bytes(), submit)
                    .await?;
                Ok(json!({"task_id": task_id, "status": "input_sent", "submitted": submit}))
            }
            CommandAction::Close => {
                context.command_manager.close_stdin(&task_id).await?;
                Ok(json!({"task_id": task_id, "status": "stdin_closed"}))
            }
        }
    }
}

fn command_failure_kind(stderr: &str) -> &'static str {
    let detail = stderr.to_ascii_lowercase();
    if detail.contains("parsererror")
        || detail.contains("syntaxerror")
        || detail.contains("unexpected token")
    {
        "syntax"
    } else if detail.contains("modulenotfound")
        || detail.contains("cannot find module")
        || detail.contains("is not recognized")
        || detail.contains("not installed")
    {
        "dependency"
    } else if detail.contains("cannot find the path")
        || detail.contains("no such file")
        || detail.contains("not found")
    {
        "environment"
    } else {
        "command"
    }
}

fn command_failure_summary(stderr: &str) -> String {
    let meaningful = stderr
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("command failed without diagnostic output");
    meaningful.chars().take(500).collect()
}

fn command_artifact_candidates(command: &str, stdout: &str, stderr: &str) -> Vec<PathBuf> {
    let combined = format!("{command}\n{stdout}\n{stderr}");
    let windows = Regex::new(r#"(?i)[a-z]:\\[^\r\n"'|<>*?]+?\.[a-z0-9]{1,10}"#)
        .expect("valid Windows artifact path regex");
    let unix = Regex::new(r#"(?:^|\s)(/[^\s"'|<>*?]+\.[a-zA-Z0-9]{1,10})"#)
        .expect("valid Unix artifact path regex");
    let relative = Regex::new(
        r#"(?:^|[\s"'=])((?:\.[/\\])?[a-zA-Z0-9_. -]+?\.[a-zA-Z0-9]{1,10})(?:$|[\s"'])"#,
    )
    .expect("valid relative artifact path regex");
    let mut output = windows
        .find_iter(&combined)
        .map(|matched| PathBuf::from(matched.as_str().trim()))
        .collect::<Vec<_>>();
    output.extend(
        unix.captures_iter(&combined)
            .filter_map(|captures| captures.get(1))
            .map(|matched| PathBuf::from(matched.as_str())),
    );
    output.extend(
        relative
            .captures_iter(&combined)
            .filter_map(|captures| captures.get(1))
            .map(|matched| PathBuf::from(matched.as_str().trim())),
    );
    output.sort();
    output.dedup();
    output
}

fn observe_completed_command_artifacts(
    context: &ToolContext,
    command: &str,
    cwd: &Path,
    stdout: &str,
    stderr: &str,
) -> Vec<ArtifactEvidence> {
    let mut artifacts = Vec::new();
    let mut seen = BTreeSet::new();
    for candidate in command_artifact_candidates(command, stdout, stderr) {
        let candidate = if candidate.is_absolute() {
            candidate
        } else {
            cwd.join(candidate)
        };
        let Ok(path) = context.resolve_workspace_path(candidate) else {
            continue;
        };
        let canonical = dunce::canonicalize(&path).unwrap_or(path);
        if !seen.insert(canonical.clone()) {
            continue;
        }
        let Ok(mut evidence) = inspect_artifact_path(&canonical, "command_observed") else {
            continue;
        };
        let operation =
            context
                .session_files
                .lock()
                .get(&canonical)
                .map_or("command_observed", |prior| {
                    if prior.sha256 == evidence.sha256 {
                        "command_observed"
                    } else {
                        "command_modified"
                    }
                });
        evidence.operation = operation.into();
        context.session_files.lock().insert(
            canonical,
            SessionFileRecord {
                sha256: evidence.sha256.clone(),
                created_by_session: operation == "command_created",
            },
        );
        context.artifact_evidence.lock().push(evidence.clone());
        artifacts.push(evidence);
    }
    artifacts
}

fn workspace_file_snapshot(root: &Path) -> BTreeMap<PathBuf, (u64, Option<SystemTime>)> {
    WalkBuilder::new(root)
        .max_depth(Some(6))
        .hidden(false)
        .filter_entry(|entry| !excluded_artifact_scan_entry(entry.path()))
        .build()
        .filter_map(std::result::Result::ok)
        .filter(|entry| entry.file_type().is_some_and(|kind| kind.is_file()))
        .take(10_000)
        .filter_map(|entry| {
            let path = dunce::canonicalize(entry.path()).ok()?;
            let metadata = entry.metadata().ok()?;
            Some((path, (metadata.len(), metadata.modified().ok())))
        })
        .collect()
}

fn recent_command_changes(
    root: &Path,
    before: &BTreeMap<PathBuf, (u64, Option<SystemTime>)>,
) -> Vec<PathBuf> {
    workspace_file_snapshot(root)
        .into_iter()
        .filter_map(|(path, state)| (before.get(&path) != Some(&state)).then_some(path))
        .take(200)
        .collect()
}

fn excluded_artifact_scan_entry(path: &Path) -> bool {
    path.components().any(|component| {
        matches!(
            component
                .as_os_str()
                .to_string_lossy()
                .to_ascii_lowercase()
                .as_str(),
            ".git" | "node_modules" | "target" | "dist" | "diagnostics" | "__pycache__"
        )
    })
}

fn command_is_os_critical(command: &str) -> bool {
    let normalized = command
        .to_ascii_lowercase()
        .replace('/', "\\")
        .replace("hkey_local_machine", "hklm")
        .replace("hkey_classes_root", "hkcr");
    const OS_CRITICAL_MARKERS: &[&str] = &[
        "format-volume",
        "format.com",
        "diskpart",
        "clear-disk",
        "initialize-disk",
        "remove-partition",
        "bcdedit",
        "bootrec",
        "manage-bde",
        "set-netfirewall",
        "new-netfirewall",
        "remove-netfirewall",
        "netsh advfirewall",
        "set-mppreference",
        "add-mppreference",
        "remove-mppreference",
        "set-netipsec",
        "new-localuser",
        "remove-localuser",
        "set-localuser",
        "add-localgroupmember",
        "remove-localgroupmember",
        "net user",
        "net localgroup",
        "set-service",
        "new-service",
        "remove-service",
        "sc.exe create",
        "sc.exe config",
        "sc.exe delete",
        "pnputil /add-driver",
        "pnputil /delete-driver",
        "dism.exe /online",
        "dism /online",
        "set-executionpolicy",
        "-verb runas",
        "schtasks /create",
        "register-scheduledtask",
        "set-scheduledtask",
        "unregister-scheduledtask",
        "secedit",
        "auditpol /set",
        "reg import",
    ];
    OS_CRITICAL_MARKERS
        .iter()
        .any(|marker| normalized.contains(marker))
        || mutates_machine_registry(&normalized)
        || mutates_protected_windows_path(&normalized)
}

fn mutates_machine_registry(command: &str) -> bool {
    let targets_machine_hive = command.contains("hklm") || command.contains("hkcr");
    let registry_cli_mutation = ["reg add", "reg.exe add", "reg delete", "reg.exe delete"]
        .iter()
        .any(|marker| command.contains(marker));
    let powershell_registry_mutation = [
        "set-itemproperty",
        "new-itemproperty",
        "remove-itemproperty",
        "new-item",
        "remove-item",
        "rename-item",
        "copy-item",
        "move-item",
        "clear-item",
    ]
    .iter()
    .any(|marker| command.contains(marker));
    targets_machine_hive && (registry_cli_mutation || powershell_registry_mutation)
}

fn mutates_protected_windows_path(command: &str) -> bool {
    const MUTATION_MARKERS: &[&str] = &[
        "remove-item",
        "del ",
        "erase ",
        "rmdir ",
        "rd ",
        "move-item",
        "copy-item",
        "set-content",
        "add-content",
        "out-file",
        "new-item",
        "rename-item",
        "takeown",
        "icacls",
        "attrib",
        "rm ",
        "mv ",
        "cp ",
    ];
    const PROTECTED_PATH_MARKERS: &[&str] = &[
        "\\windows\\",
        "%windir%\\",
        "%systemroot%\\",
        "$env:windir\\",
        "$env:systemroot\\",
        "\\boot\\",
        "\\efi\\",
        "\\bootmgr",
        "\\pagefile.sys",
        "\\hiberfil.sys",
    ];
    MUTATION_MARKERS
        .iter()
        .any(|marker| command.contains(marker))
        && PROTECTED_PATH_MARKERS
            .iter()
            .any(|marker| command.contains(marker))
}

#[cfg(test)]
fn powershell_utf8_script(command: &str) -> String {
    format!(
        "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); \
         $OutputEncoding = [Console]::OutputEncoding; \
         $ErrorActionPreference = 'Stop'; \
         $global:LASTEXITCODE = 0; \
         try {{ & {{ {command} }}; \
         if ($global:LASTEXITCODE -ne 0) {{ exit $global:LASTEXITCODE }} \
         }} catch {{ [Console]::Error.WriteLine(($_ | Out-String)); exit 1 }}"
    )
}

fn hash_text(text: &str) -> String {
    hash_bytes(text.as_bytes())
}

fn hash_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    #[test]
    fn com_automation_commands_are_recognized() {
        assert!(super::automates_application(
            "$w = New-Object -ComObject Word.Application"
        ));
        assert!(super::automates_application(
            "[Runtime.InteropServices.Marshal]::GetActiveObject('Excel.Application')"
        ));
        assert!(!super::automates_application("Get-ChildItem C:\\"));
    }

    use super::*;

    #[tokio::test]
    async fn a_result_saved_outside_the_workspace_can_be_read_back() {
        let workspace = tempfile::tempdir().unwrap();
        let documents = tempfile::tempdir().unwrap();
        let saved = documents.path().join("report.txt");
        std::fs::write(&saved, "line one").unwrap();
        let context = crate::builtins::tests::mock_input_context(
            std::sync::Arc::new(crate::platform::MockDesktop::default()),
            &workspace,
        );
        // Through the registry, whose path check runs before the tool.
        let mut registry = crate::tool::ToolRegistry::new();
        register_coding_tools(&mut registry);
        let inspect =
            |path: &Path| registry.call("inspect_artifact", json!({"path": path}), &context);
        // Not produced or observed by this session: outside paths are refused.
        assert!(inspect(&saved).await.is_err());
        let observed =
            inspect_artifact_path(&dunce::canonicalize(&saved).unwrap(), "command_observed")
                .unwrap();
        context.artifact_evidence.lock().push(observed);
        let inspected = inspect(&saved).await.unwrap();
        assert_eq!(inspected["operation"], "inspected");
        let other = documents.path().join("private.txt");
        std::fs::write(&other, "secret").unwrap();
        assert!(inspect(&other).await.is_err());
        // A file the request names, saved by an application during the task;
        // the same name written before the request started is not its result.
        let requested = documents.path().join("Word-Test.docx");
        std::fs::write(&requested, "saved by the app").unwrap();
        let mut task = crate::tool::ActiveTaskState {
            root_request: "save it as word-test.docx in Documents".into(),
            ..crate::tool::ActiveTaskState::default()
        };
        task.started_at = std::time::SystemTime::now() + std::time::Duration::from_secs(60);
        *context.active_task.lock() = task;
        assert!(inspect(&requested).await.is_err());
        context.active_task.lock().started_at =
            std::time::SystemTime::now() - std::time::Duration::from_secs(60);
        assert!(inspect(&requested).await.is_ok());
        // Writing outside the workspace stays refused for a known result.
        assert!(
            registry
                .call(
                    "write_file",
                    json!({"path": saved, "content": "changed"}),
                    &context
                )
                .await
                .is_err()
        );
        assert_eq!(std::fs::read_to_string(&saved).unwrap(), "line one");
    }

    #[test]
    fn large_source_files_are_read_in_bounded_pages() {
        let content = (1..=9_000)
            .map(|line| format!("line_{line}"))
            .collect::<Vec<_>>()
            .join("\n");
        let page = read_file_page(Path::new("large.rs"), &content, 8_801, 400);
        assert_eq!(page["total_lines"], 9_000);
        assert_eq!(page["start_line"], 8_801);
        assert_eq!(page["end_line"], 9_000);
        assert_eq!(page["truncated"], false);
        assert!(
            page["content"]
                .as_str()
                .is_some_and(|text| text.starts_with("line_8801") && text.ends_with("line_9000"))
        );
    }

    #[test]
    fn source_outline_finds_definitions_without_returning_file_body() {
        let content =
            "mod alpha;\n\npub struct Thing;\n\nimpl Thing {\n    pub async fn run(&self) {}\n}\n";
        let (entries, total) = source_outline(content, 10);
        assert_eq!(total, 3);
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].line, 1);
        assert_eq!(entries[2].kind, "function");
    }

    #[test]
    fn artifact_inspection_is_format_neutral_for_text() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("notes.txt");
        std::fs::write(&path, "alpha\nbeta\n").unwrap();
        let evidence = inspect_artifact_path(&path, "inspected").unwrap();
        assert_eq!(evidence.artifact_type, "text");
        assert_eq!(evidence.integrity, "readable");
        assert_eq!(evidence.structure["line_count"], 2);
        assert_eq!(evidence.operation, "inspected");
    }

    #[test]
    fn package_inspection_reports_generic_document_components() {
        use std::io::Write;

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("deck.bin");
        let file = std::fs::File::create(&path).unwrap();
        let mut writer = zip::ZipWriter::new(file);
        let options = zip::write::SimpleFileOptions::default();
        writer.start_file("[Content_Types].xml", options).unwrap();
        writer.write_all(b"<Types />").unwrap();
        writer.start_file("ppt/slides/slide1.xml", options).unwrap();
        writer.write_all(b"<slide />").unwrap();
        writer.start_file("ppt/media/image1.png", options).unwrap();
        writer.write_all(b"not-a-real-image").unwrap();
        writer.finish().unwrap();

        let evidence = inspect_artifact_path(&path, "inspected").unwrap();
        assert_eq!(evidence.artifact_type, "presentation_package");
        assert_eq!(evidence.structure["slide_parts"], 1);
        assert_eq!(evidence.structure["media_files"], 1);
    }

    #[test]
    fn command_artifact_discovery_accepts_windows_and_unix_paths() {
        let paths = command_artifact_candidates(
            r#"Start-Process "C:\Users\person\Downloads\report.pptx""#,
            "Saved to /tmp/output/report.txt",
            "",
        );
        assert!(
            paths
                .iter()
                .any(|path| path.to_string_lossy().ends_with("report.pptx"))
        );
        assert!(
            paths
                .iter()
                .any(|path| path == Path::new("/tmp/output/report.txt"))
        );
    }

    #[test]
    fn command_artifact_discovery_accepts_relative_output_paths() {
        let paths = command_artifact_candidates(
            "python create_chart.py",
            "Saved Resultados_Analisis.xlsx",
            "",
        );
        assert!(
            paths
                .iter()
                .any(|path| path == Path::new("Resultados_Analisis.xlsx"))
        );
    }

    #[test]
    fn powershell_wrapper_turns_error_records_into_failed_processes() {
        let script = powershell_utf8_script("Write-Error 'broken'");
        assert!(script.contains("$ErrorActionPreference = 'Stop'"));
        assert!(script.contains("catch"));
        assert!(script.contains("exit 1"));
        assert_eq!(
            command_failure_kind("ParserError: Unexpected token"),
            "syntax"
        );
        assert_eq!(
            command_failure_kind("ModuleNotFoundError: demo"),
            "dependency"
        );
    }

    #[test]
    fn safe_rewrite_requires_session_ownership_and_unchanged_hash() {
        let owned = SessionFileRecord {
            sha256: "abc".into(),
            created_by_session: true,
        };
        assert!(safe_session_rewrite(&owned, "abc"));
        assert!(!safe_session_rewrite(&owned, "changed"));
        assert!(!safe_session_rewrite(
            &SessionFileRecord {
                sha256: "abc".into(),
                created_by_session: false,
            },
            "abc"
        ));
    }

    #[test]
    fn preexisting_command_paths_are_observations_not_created_outputs() {
        assert_eq!(
            command_artifact_operation(true, false, false),
            "command_observed"
        );
        assert_eq!(
            command_artifact_operation(true, true, false),
            "command_modified"
        );
        assert_eq!(
            command_artifact_operation(false, false, false),
            "command_created"
        );
        assert_eq!(
            command_artifact_operation_from_evidence(false, false, false, false),
            "command_observed"
        );
        assert_eq!(
            command_artifact_operation_from_evidence(false, true, false, false),
            "command_created"
        );
    }

    #[test]
    fn workbook_semantics_detect_many_one_point_series() {
        use std::io::Write;

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("chart.xlsx");
        let file = std::fs::File::create(&path).unwrap();
        let mut writer = zip::ZipWriter::new(file);
        let options = zip::write::SimpleFileOptions::default();
        writer.start_file("xl/workbook.xml", options).unwrap();
        writer.write_all(br#"<workbook xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Results" r:id="rId1"/></sheets></workbook>"#).unwrap();
        writer
            .start_file("xl/_rels/workbook.xml.rels", options)
            .unwrap();
        writer.write_all(br#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>"#).unwrap();
        writer
            .start_file("xl/worksheets/sheet1.xml", options)
            .unwrap();
        writer
            .write_all(br#"<worksheet><dimension ref="A1:B10"/></worksheet>"#)
            .unwrap();
        writer.start_file("xl/charts/chart1.xml", options).unwrap();
        let series = (0..6)
            .map(|index| {
                format!(
                    r#"<c:ser><c:tx><c:strRef><c:f>Sheet1!$B${}</c:f></c:strRef></c:tx><c:val><c:numRef><c:f>Sheet1!$B${}</c:f><c:numCache><c:pt idx="0"><c:v>{index}</c:v></c:pt></c:numCache></c:numRef></c:val></c:ser>"#,
                    index + 1,
                    index + 1
                )
            })
            .collect::<String>();
        writer
            .write_all(
                format!(r#"<c:chart xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart"><c:barChart>{series}</c:barChart></c:chart>"#).as_bytes(),
            )
            .unwrap();
        writer.finish().unwrap();

        let evidence = inspect_artifact_path(&path, "inspected").unwrap();
        assert_eq!(evidence.structure["semantic"]["sheet_names"][0], "Results");
        assert_eq!(
            evidence.structure["semantic"]["worksheets"],
            json!([{
                "name": "Results",
                "part": "xl/worksheets/sheet1.xml",
                "dimensions": "A1:B10",
                "row_count": 10,
                "column_count": 2,
                "cell_count": 0
            }])
        );
        assert_eq!(
            evidence.structure["semantic"]["charts"][0]["series_count"],
            6
        );
        assert!(evidence.warnings.iter().any(
            |warning| warning.code == "many_one_point_series" && warning.severity == "warning"
        ));
    }

    #[test]
    fn workbook_chart_parser_accepts_default_xml_namespace() {
        use std::io::Write;

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("chart.xlsx");
        let file = std::fs::File::create(&path).unwrap();
        let mut writer = zip::ZipWriter::new(file);
        let options = zip::write::SimpleFileOptions::default();
        writer.start_file("xl/workbook.xml", options).unwrap();
        writer.write_all(br#"<workbook xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Results" r:id="rId1"/></sheets></workbook>"#).unwrap();
        writer
            .start_file("xl/_rels/workbook.xml.rels", options)
            .unwrap();
        writer.write_all(br#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>"#).unwrap();
        writer
            .start_file("xl/worksheets/sheet1.xml", options)
            .unwrap();
        writer
            .write_all(br#"<worksheet><dimension ref="A1:D10"/></worksheet>"#)
            .unwrap();
        writer.start_file("xl/charts/chart1.xml", options).unwrap();
        writer.write_all(br#"<chartSpace xmlns="http://schemas.openxmlformats.org/drawingml/2006/chart"><chart><plotArea><barChart><ser><cat><numRef><f>Results!$A$2:$A$10</f></numRef></cat><val><numRef><f>Results!$B$2:$B$10</f></numRef></val></ser><ser><cat><numRef><f>Results!$A$2:$A$10</f></numRef></cat><val><numRef><f>Results!$C$2:$C$10</f></numRef></val></ser><ser><cat><numRef><f>Results!$A$2:$A$10</f></numRef></cat><val><numRef><f>Results!$D$2:$D$10</f></numRef></val></ser></barChart></plotArea></chart></chartSpace>"#).unwrap();
        writer.finish().unwrap();

        let evidence = inspect_artifact_path(&path, "inspected").unwrap();
        assert_eq!(
            evidence.structure["semantic"]["charts"][0]["series_count"],
            3
        );
        assert_eq!(evidence.validation_status, "valid");
    }

    #[test]
    fn workbook_chart_without_series_is_advisory() {
        let structure = json!({
            "semantic": {
                "xml_errors": [],
                "charts": [{"chart_type": "barChart", "series": []}]
            },
            "missing_relationship_targets": []
        });
        let warnings = package_warnings(&structure);
        assert!(warnings.iter().any(|warning| {
            warning.code == "chart_without_series" && warning.severity == "warning"
        }));
    }

    #[test]
    fn workbook_chart_references_must_resolve_to_sheet_bounds() {
        let structure = json!({
            "semantic": {
                "xml_errors": [],
                "sheet_names": ["Results"],
                "worksheets": [{"name": "Results", "row_count": 10, "column_count": 4}],
                "charts": [{
                    "chart_type": "barChart",
                    "series": [{
                        "formulas": ["Missing!$A$2:$A$10", "Results!$B$2:$B$20"],
                        "point_count": 9
                    }]
                }]
            },
            "missing_relationship_targets": []
        });
        let warnings = package_warnings(&structure);
        assert!(
            warnings
                .iter()
                .any(|warning| warning.code == "chart_reference_missing_sheet")
        );
        assert!(
            warnings
                .iter()
                .any(|warning| warning.code == "chart_reference_outside_worksheet")
        );
    }

    #[test]
    fn package_relationship_resolution_handles_nested_rels() {
        assert_eq!(
            relationship_source_directory("xl/worksheets/_rels/sheet1.xml.rels"),
            "xl/worksheets"
        );
        assert_eq!(
            normalize_package_path("xl/worksheets", "../drawings/drawing1.xml"),
            "xl/drawings/drawing1.xml"
        );
        assert_eq!(worksheet_dimension_counts("A1:F31"), Some((31, 6)));
        assert_eq!(worksheet_dimension_counts("A1:AA9000"), Some((9000, 27)));
    }

    #[test]
    fn shared_chart_categories_are_not_duplicate_value_references() {
        let structure = json!({
            "semantic": {
                "charts": [{
                    "series": [
                        {"formulas": ["Data!$A$2:$A$30", "Data!$B$2:$B$30"], "point_count": 29},
                        {"formulas": ["Data!$A$2:$A$30", "Data!$C$2:$C$30"], "point_count": 29},
                        {"formulas": ["Data!$A$2:$A$30", "Data!$D$2:$D$30"], "point_count": 29}
                    ]
                }]
            }
        });
        assert!(
            package_warnings(&structure)
                .iter()
                .all(|warning| warning.code != "duplicate_chart_references")
        );
    }

    #[test]
    fn exact_edit_requires_one_match() {
        let text = "one\ntwo\ntwo\n";
        assert_eq!(text.match_indices("two").count(), 2);
    }

    #[test]
    fn coding_registry_exposes_self_authored_helper_path() {
        let mut registry = crate::tool::ToolRegistry::new();
        register_coding_tools(&mut registry);
        let names = registry.names();
        assert!(names.iter().any(|name| name == "write_file"));
        assert!(names.iter().any(|name| name == "source_outline"));
        assert!(names.iter().any(|name| name == "edit_file"));
        assert!(names.iter().any(|name| name == "inspect_artifact"));
        assert!(names.iter().any(|name| name == "run_command"));
        assert!(!names.iter().any(|name| name == "verify_task_outcome"));

        let active = BTreeSet::from(["coding".to_owned()]);
        let catalog = registry.compact_catalog(&active);
        assert!(catalog.contains("run_command [system; inactive]"));
        assert!(catalog.contains("read_file [coding; active]"));
    }

    #[test]
    fn powershell_commands_enable_utf8_without_nesting_another_shell() {
        let script = powershell_utf8_script("Write-Output 'Pokémon °C'");
        assert!(script.contains("[Console]::OutputEncoding"));
        assert!(script.contains("$OutputEncoding"));
        assert!(script.contains("Write-Output 'Pokémon °C'"));
        assert!(script.ends_with("exit 1 }"));
        assert!(!script.contains("powershell -Command"));
    }

    #[test]
    fn command_risk_only_flags_os_critical_actions() {
        assert!(command_is_os_critical(
            "Remove-Item -Recurse -Force C:\\Windows\\System32\\drivers\\bad.sys"
        ));
        assert!(command_is_os_critical(
            "Set-ItemProperty -Path 'HKLM:\\Software\\Microsoft\\Windows' -Name Test -Value 1"
        ));
        assert!(command_is_os_critical(
            "reg.exe delete \"HKLM\\Software\\Example\" /f"
        ));
        assert!(command_is_os_critical("Start-Process cmd -Verb RunAs"));
        assert!(!command_is_os_critical(
            "Remove-Item -Recurse -Force C:\\Users\\person\\Documents"
        ));
        assert!(!command_is_os_critical("git push origin main"));
        assert!(!command_is_os_critical(
            "Get-ChildItem C:\\Users\\person\\Documents"
        ));
        assert!(!command_is_os_critical(
            "Start-Process 'H:\\NZB_Complete\\Disclosure.Day.2026.mkv'"
        ));
        assert!(!command_is_os_critical("python -m pip install requests"));
        assert!(!command_is_os_critical("shutdown.exe /s /t 0"));
    }
}
