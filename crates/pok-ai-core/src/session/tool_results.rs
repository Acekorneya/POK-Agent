//! Tool results as the model and dashboard see them: bounded projections, tool-call/result pairing, fresh-evidence and artifact checks.

use super::*;

pub(super) fn tool_result_messages(
    call: &CompletedToolCall,
    result: Result<Value>,
    send_images: bool,
) -> Vec<BrainMessage> {
    let mut images = Vec::new();
    let tool_text = match result {
        Ok(mut value) => {
            extract_images(&mut value, &mut images);
            if images.len() > CURRENT_VISUAL_PAIR_IMAGES {
                images.drain(..images.len() - CURRENT_VISUAL_PAIR_IMAGES);
            }
            project_tool_result_for_model(call.name.as_str(), value).to_string()
        }
        Err(error) => json!({"error": error.to_string()}).to_string(),
    };
    let mut messages = vec![BrainMessage {
        role: "tool".into(),
        content: vec![MessageContent::Text { text: tool_text }],
        origin: MessageOrigin::ToolResult,
        tool_call_id: Some(call.id.clone()),
        tool_calls: Vec::new(),
    }];
    if send_images && !images.is_empty() {
        let mut content = vec![MessageContent::Text {
            text: "Current desktop screenshots returned by capture_screen.".into(),
        }];
        content.extend(images);
        messages.push(BrainMessage {
            role: "user".into(),
            content,
            origin: MessageOrigin::ToolImage,
            tool_call_id: None,
            tool_calls: Vec::new(),
        });
    }
    messages
}

pub(super) const LIVE_COMMAND_STREAM_CHARS: usize = 32 * 1024;
pub(super) const COMMAND_MODEL_OUTPUT_CHARS: usize = 12 * 1024;
pub(super) const MODEL_TOOL_RESULT_CHARS: usize = 32 * 1024;

/// Model view of an oversized managed-browser snapshot. The generic preview
/// cut the JSON at a fixed length, and `snapshot_id` sits after the full
/// element list, so the model never saw it and every click it attempted was
/// rejected as stale. This keeps the snapshot id and page facts first, a
/// bounded text excerpt, and only actionable elements as id/role/name.
pub(super) fn project_browser_snapshot_for_model(value: &Value) -> Value {
    const ELEMENTS: usize = 200;
    let elements = value
        .get("elements")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    let actionable = elements
        .iter()
        .filter(|element| element.get("actionable").and_then(Value::as_bool) == Some(true))
        .collect::<Vec<_>>();
    let compact = actionable
        .iter()
        .take(ELEMENTS)
        .map(|element| {
            let mut item = serde_json::Map::new();
            for key in ["id", "role"] {
                if let Some(field) = element.get(key) {
                    item.insert(key.into(), field.clone());
                }
            }
            if let Some(name) = element.get("name").and_then(Value::as_str) {
                item.insert("name".into(), Value::String(truncate_chars(name, 160)));
            }
            let sensitive = element.get("sensitive").and_then(Value::as_bool) == Some(true);
            if !sensitive
                && let Some(text) = element.get("value").and_then(Value::as_str)
                && !text.is_empty()
            {
                item.insert("value".into(), Value::String(truncate_chars(text, 100)));
            }
            for key in ["checked", "selected", "input_type"] {
                if let Some(field) = element.get(key).filter(|field| !field.is_null()) {
                    item.insert(key.into(), field.clone());
                }
            }
            if element.get("disabled").and_then(Value::as_bool) == Some(true) {
                item.insert("disabled".into(), Value::Bool(true));
            }
            Value::Object(item)
        })
        .collect::<Vec<_>>();
    let mut projected = serde_json::Map::new();
    for key in [
        "snapshot_id",
        "url",
        "title",
        "loading",
        "fingerprint",
        "_pok_continuity",
    ] {
        if let Some(item) = value.get(key) {
            projected.insert(key.into(), item.clone());
        }
    }
    if let Some(text) = value.get("text").and_then(Value::as_str) {
        projected.insert("text".into(), Value::String(truncate_chars(text, 6_000)));
    }
    projected.insert("elements".into(), Value::Array(compact));
    projected.insert(
        "model_projection".into(),
        json!({
            "truncated": true,
            "elements_total": elements.len(),
            "actionable_total": actionable.len(),
            "actionable_shown": actionable.len().min(ELEMENTS),
            "reason": "Only actionable elements are listed; use snapshot_id and an element id to act."
        }),
    );
    Value::Object(projected)
}

pub(super) fn project_tool_result_for_model(name: &str, mut value: Value) -> Value {
    if name == "run_command" {
        for key in ["stdout", "stderr"] {
            if let Some(text) = value
                .get_mut(key)
                .and_then(|item| item.as_str())
                .map(str::to_owned)
            {
                let original_chars = text.chars().count();
                if original_chars > COMMAND_MODEL_OUTPUT_CHARS {
                    value[key] =
                        Value::String(truncate_head_tail(&text, COMMAND_MODEL_OUTPUT_CHARS));
                    value[format!("{key}_model_truncated")] = Value::Bool(true);
                    value[format!("{key}_original_chars")] = json!(original_chars);
                }
            }
        }
    }
    if name == "grep_search"
        && let Some(matches) = value.get_mut("matches").and_then(Value::as_array_mut)
    {
        matches.truncate(20);
        for item in matches {
            if let Some(context) = item.get_mut("context")
                && let Some(text) = context.as_str()
            {
                *context = Value::String(truncate_chars(text, 1_500));
            }
        }
    }
    if name == "list_directory"
        && let Some(files) = value.get_mut("files").and_then(Value::as_array_mut)
    {
        let original = files.len();
        files.truncate(500);
        if files.len() < original {
            value["model_files_truncated"] = json!(true);
            value["total_files_before_model_projection"] = json!(original);
        }
    }
    if value.to_string().chars().count() <= MODEL_TOOL_RESULT_CHARS {
        return value;
    }
    if value.get("snapshot_id").is_some() && value.get("elements").is_some() {
        return project_browser_snapshot_for_model(&value);
    }
    if value.get("observation_id").is_some() {
        let mut projected = serde_json::Map::new();
        for key in [
            "observation_id",
            "captured_at",
            "executed",
            "success",
            "action",
            "focus",
            "target",
            "state",
            "state_change",
            "submission",
            "observed_url",
            "page_status",
            "verification_state",
            "verification_issues",
            "cache",
            "_pok_continuity",
        ] {
            if let Some(item) = value.get(key) {
                projected.insert(key.into(), item.clone());
            }
        }
        for (key, limit) in [
            ("relevant_targets", 40),
            ("action_targets", 60),
            ("suggested_scroll_targets", 20),
            ("targets", 60),
            ("primary_content", 80),
            ("ordered_content", 80),
        ] {
            if let Some(items) = value.get(key).and_then(Value::as_array) {
                projected.insert(
                    key.into(),
                    Value::Array(items.iter().take(limit).cloned().collect()),
                );
            } else if let Some(text) = value.get(key).and_then(Value::as_str) {
                projected.insert(key.into(), Value::String(truncate_chars(text, 12_000)));
            }
        }
        projected.insert("model_projection".into(), json!({
            "truncated": true,
            "reason": "Full observation retained in diagnostics; model view bounded for latency."
        }));
        let mut projected = Value::Object(projected);
        bound_model_json(&mut projected, 2_000, 100);
        if projected.to_string().chars().count() > MODEL_TOOL_RESULT_CHARS {
            if let Some(object) = projected.as_object_mut() {
                for key in [
                    "relevant_targets",
                    "action_targets",
                    "suggested_scroll_targets",
                    "targets",
                    "primary_content",
                    "ordered_content",
                ] {
                    if let Some(items) = object.get_mut(key).and_then(Value::as_array_mut) {
                        items.truncate(20);
                    }
                }
            }
            bound_model_json(&mut projected, 500, 20);
        }
        return projected;
    }
    let serialized = value.to_string();
    json!({
        "model_projection": {
            "truncated": true,
            "original_chars": serialized.chars().count(),
            "reason": "Full tool result retained in diagnostics."
        },
        "preview": truncate_chars(&serialized, MODEL_TOOL_RESULT_CHARS),
    })
}

pub(super) fn truncate_head_tail(value: &str, limit: usize) -> String {
    if value.chars().count() <= limit {
        return value.to_owned();
    }
    const MARKER: &str = "\n... output omitted; use manage_command read ...\n";
    let available = limit.saturating_sub(MARKER.chars().count());
    let head = available / 3;
    let tail = available.saturating_sub(head);
    let beginning = value.chars().take(head).collect::<String>();
    let ending = value
        .chars()
        .rev()
        .take(tail)
        .collect::<String>()
        .chars()
        .rev()
        .collect::<String>();
    format!("{beginning}{MARKER}{ending}")
}

pub(super) fn bound_model_json(value: &mut Value, max_string_chars: usize, max_array_items: usize) {
    match value {
        Value::String(text) => {
            if text.chars().count() > max_string_chars {
                *text = text.chars().take(max_string_chars).collect();
            }
        }
        Value::Array(items) => {
            items.truncate(max_array_items);
            for item in items {
                bound_model_json(item, max_string_chars, max_array_items);
            }
        }
        Value::Object(object) => {
            for item in object.values_mut() {
                bound_model_json(item, max_string_chars, max_array_items);
            }
        }
        _ => {}
    }
}

pub(super) fn tool_finished_ok(name: &str, result: &Result<Value>) -> bool {
    let Ok(value) = result else {
        return false;
    };
    if value.get("recovery_required").and_then(Value::as_bool) == Some(true) {
        return false;
    }
    if name == "execute_action_batch" {
        return value.get("success").and_then(Value::as_bool) == Some(true)
            && value.get("verified_progress").and_then(Value::as_bool) == Some(true);
    }
    if !matches!(name, "run_command" | "manage_command") {
        return true;
    }
    if name == "manage_command" && value.get("status").and_then(Value::as_str).is_none() {
        return true;
    }
    value.get("success").and_then(Value::as_bool) != Some(false)
        && value
            .get("exit_code")
            .and_then(Value::as_i64)
            .is_none_or(|code| code == 0)
}

pub(super) fn tool_finished_detail(name: &str, result: &Result<Value>) -> String {
    match result {
        Err(error) => truncate_chars(&error.to_string(), 240),
        Ok(value) if name == "execute_action_batch" && !tool_finished_ok(name, result) => value
            .get("stop_reason")
            .or_else(|| value.get("error"))
            .and_then(Value::as_str)
            .map_or_else(
                || "batch completed without verified progress".into(),
                str::to_owned,
            ),
        Ok(value) if name == "run_command" && !tool_finished_ok(name, result) => {
            let exit_code = value
                .get("exit_code")
                .and_then(Value::as_i64)
                .map_or_else(|| "unknown".into(), |code| code.to_string());
            let stderr = value
                .get("stderr")
                .and_then(Value::as_str)
                .filter(|text| !text.trim().is_empty())
                .map(|text| truncate_chars(text.trim(), 180));
            stderr.map_or_else(
                || format!("command exited with code {exit_code}"),
                |stderr| format!("exit code {exit_code}: {stderr}"),
            )
        }
        Ok(value)
            if matches!(
                name,
                "capture_screen"
                    | "query_screen_text"
                    | "query_window_tree"
                    | "scroll_view"
                    | "scroll_until_text"
            ) && value
                .get("warnings")
                .and_then(Value::as_array)
                .is_some_and(|warnings| !warnings.is_empty()) =>
        {
            let warning = value
                .get("warnings")
                .and_then(Value::as_array)
                .and_then(|warnings| warnings.first())
                .and_then(Value::as_str)
                .unwrap_or("optional desktop enrichment was unavailable");
            format!(
                "completed with screenshot fallback: {}",
                truncate_chars(warning, 180)
            )
        }
        Ok(_) => "completed".into(),
    }
}

pub(super) fn tool_finished_event_result(name: &str, result: &Result<Value>) -> Option<Value> {
    if matches!(name, "write_file" | "edit_file" | "undo_edit") {
        return match result {
            Ok(value) => Some(json!({
                "operation": match name {
                    "write_file" => value.get("operation").and_then(Value::as_str).unwrap_or("created"),
                    "edit_file" => "edited",
                    _ => "restored",
                },
                "path": value.get("path").or_else(|| value.get("restored")),
            })),
            Err(error) => Some(json!({
                "success": false,
                "error": truncate_chars(&error.to_string(), 240),
            })),
        };
    }
    if name == "inspect_artifact" {
        return result.as_ref().ok().cloned();
    }
    // Desktop input reports whether it used the real mouse and keyboard or
    // an accessibility action that left the user's cursor alone.
    if let Ok(value) = result
        && let Some(method) = value.get("input_method").and_then(Value::as_str)
    {
        return Some(json!({"input_method": method}));
    }
    if name != "run_command" {
        return None;
    }
    let value = match result {
        Ok(value) => value,
        Err(error) => {
            return Some(json!({
                "success": false,
                "stderr": truncate_chars(&error.to_string(), LIVE_COMMAND_STREAM_CHARS),
                "ui_output_truncated": error.to_string().chars().count() > LIVE_COMMAND_STREAM_CHARS,
            }));
        }
    };
    let stdout = value.get("stdout").and_then(Value::as_str).unwrap_or("");
    let stderr = value.get("stderr").and_then(Value::as_str).unwrap_or("");
    Some(json!({
        "task_id": value.get("task_id"),
        "status": value.get("status"),
        "backgrounded": value.get("backgrounded"),
        "cwd": value.get("cwd"),
        "exit_code": value.get("exit_code"),
        "success": value.get("success"),
        "failure_kind": value.get("failure_kind"),
        "failure_summary": value.get("failure_summary"),
        "stdout": truncate_chars(stdout, LIVE_COMMAND_STREAM_CHARS),
        "stderr": truncate_chars(stderr, LIVE_COMMAND_STREAM_CHARS),
        "artifacts": value.get("artifacts"),
        "total_bytes": value.get("total_bytes"),
        "log_bytes": value.get("log_bytes"),
        "output_file": value.get("output_file"),
        "output_truncated": value.get("output_truncated"),
        "ui_output_truncated": stdout.chars().count() > LIVE_COMMAND_STREAM_CHARS
            || stderr.chars().count() > LIVE_COMMAND_STREAM_CHARS,
    }))
}

pub(super) fn extract_images(value: &mut Value, images: &mut Vec<MessageContent>) {
    match value {
        Value::Object(object) => {
            if let Some(base64) = object
                .remove("png_base64")
                .and_then(|value| value.as_str().map(str::to_owned))
            {
                images.push(MessageContent::ImagePng { base64 });
            }
            for value in object.values_mut() {
                extract_images(value, images);
            }
        }
        Value::Array(values) => {
            for value in values {
                extract_images(value, images);
            }
        }
        _ => {}
    }
}

pub(super) fn tool_call_result_pairs_valid(messages: &[BrainMessage]) -> bool {
    let mut pending = VecDeque::<String>::new();
    for message in messages {
        if message.role == "assistant" && !message.tool_calls.is_empty() {
            if !pending.is_empty() {
                return false;
            }
            pending.extend(message.tool_calls.iter().map(|call| call.id.clone()));
            continue;
        }
        if message.role == "tool" {
            let Some(expected) = pending.pop_front() else {
                return false;
            };
            if message.tool_call_id.as_deref() != Some(expected.as_str()) {
                return false;
            }
        } else if !pending.is_empty() {
            return false;
        }
    }
    pending.is_empty()
}

#[derive(Debug, Clone, Default, Serialize)]
pub(super) struct ToolPairRepairReport {
    pub(super) displaced_results_moved: usize,
    pub(super) duplicate_results_removed: usize,
    pub(super) orphan_results_removed: usize,
    pub(super) synthetic_results_inserted: usize,
}

impl ToolPairRepairReport {
    pub(super) fn changed(&self) -> bool {
        self.displaced_results_moved > 0
            || self.duplicate_results_removed > 0
            || self.orphan_results_removed > 0
            || self.synthetic_results_inserted > 0
    }
}

/// Normalize provider-facing tool history without replaying any action. Tool
/// results are pulled into the contiguous run after their owning assistant
/// message, duplicates/orphans are removed, and genuinely unanswered calls get
/// an explicit synthetic failure result. This is safe at provider and persisted
/// history boundaries, never while a tool batch is still executing.
pub(super) fn repair_tool_call_result_pairs(
    messages: &mut Vec<BrainMessage>,
    reason: &str,
) -> ToolPairRepairReport {
    let original = std::mem::take(messages);
    let mut repaired = Vec::with_capacity(original.len());
    let mut report = ToolPairRepairReport::default();
    let mut index = 0;

    while index < original.len() {
        let message = original[index].clone();
        if message.role != "assistant" || message.tool_calls.is_empty() {
            if message.role == "tool" {
                report.orphan_results_removed += 1;
            } else {
                repaired.push(message);
            }
            index += 1;
            continue;
        }

        let call_ids = message
            .tool_calls
            .iter()
            .map(|call| call.id.clone())
            .collect::<Vec<_>>();
        let declared = call_ids.iter().cloned().collect::<BTreeSet<_>>();
        repaired.push(message);

        let mut segment_end = index + 1;
        while segment_end < original.len() && original[segment_end].role != "assistant" {
            segment_end += 1;
        }

        let mut results = BTreeMap::<String, (BrainMessage, usize)>::new();
        let mut deferred = Vec::new();
        for (offset, candidate) in original[index + 1..segment_end].iter().cloned().enumerate() {
            if candidate.role != "tool" {
                deferred.push(candidate);
                continue;
            }
            let Some(id) = candidate.tool_call_id.clone() else {
                report.orphan_results_removed += 1;
                continue;
            };
            if !declared.contains(&id) {
                report.orphan_results_removed += 1;
                continue;
            }
            if results.insert(id, (candidate, offset)).is_some() {
                report.duplicate_results_removed += 1;
            }
        }

        for (expected_offset, call) in call_ids.iter().enumerate() {
            if let Some((result, actual_offset)) = results.remove(call) {
                if actual_offset != expected_offset {
                    report.displaced_results_moved += 1;
                }
                repaired.push(result);
            } else {
                report.synthetic_results_inserted += 1;
                repaired.push(BrainMessage {
                    role: "tool".into(),
                    content: vec![MessageContent::Text {
                        text: json!({
                            "error": "tool execution did not produce a durable result",
                            "recovery": reason,
                            "instruction": "Re-check current state before deciding whether a retry is safe."
                        })
                        .to_string(),
                    }],
                    origin: MessageOrigin::ToolResult,
                    tool_call_id: Some(call.clone()),
                    tool_calls: Vec::new(),
                });
            }
        }
        repaired.extend(deferred);
        index = segment_end;
    }

    *messages = repaired;
    debug_assert!(tool_call_result_pairs_valid(messages));
    report
}

pub(super) fn truncate_chars(value: &str, limit: usize) -> String {
    if value.chars().count() <= limit {
        return value.to_owned();
    }
    let mut truncated = value.chars().take(limit).collect::<String>();
    truncated.push('…');
    truncated
}

pub(super) fn is_grounding_evidence_tool(name: &str) -> bool {
    matches!(
        name,
        "browser_navigate"
            | "managed_browser_open"
            | "managed_browser_snapshot"
            | "managed_browser_click"
            | "managed_browser_type"
            | "managed_browser_select"
            | "managed_browser_hover"
            | "managed_browser_scroll"
            | "capture_screen"
            | "inspect_screen_region"
            | "scroll_view"
            | "scroll_until_text"
            | "query_screen_text"
            | "query_window_tree"
            | "read_file"
            | "grep_search"
            | "memory_search"
            | "get_current_time"
            | "list_directory"
            | "run_command"
            | "manage_command"
    )
}

pub(super) fn qualifies_as_fresh_evidence(name: &str, value: &Value, ok: bool) -> bool {
    is_grounding_evidence_tool(name)
        && ok
        && navigation_result_is_ready(value)
        && value.get("status").and_then(Value::as_str) != Some("running")
        && value.get("executed").and_then(Value::as_bool) != Some(false)
}

pub(super) fn tool_changed_artifact(
    name: &str,
    value: &Value,
    observed_hashes: &mut BTreeMap<String, String>,
) -> bool {
    let mut candidates = Vec::new();
    if matches!(name, "write_file" | "edit_file" | "undo_edit") {
        let path = value
            .get("path")
            .or_else(|| value.get("restored"))
            .and_then(Value::as_str);
        let hash = value
            .get("sha256")
            .or_else(|| value.get("new_sha256"))
            .and_then(Value::as_str);
        if let (Some(path), Some(hash)) = (path, hash) {
            candidates.push((path.to_owned(), hash.to_owned()));
        }
    }
    if name == "run_command" {
        candidates.extend(
            value
                .get("artifacts")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|artifact| {
                    if artifact.get("operation").and_then(Value::as_str) == Some("command_observed")
                    {
                        return None;
                    }
                    Some((
                        artifact.get("path")?.as_str()?.to_owned(),
                        artifact.get("sha256")?.as_str()?.to_owned(),
                    ))
                }),
        );
    }
    let mut changed = false;
    for (path, hash) in candidates {
        if observed_hashes.insert(path, hash.clone()).as_deref() != Some(hash.as_str()) {
            changed = true;
        }
    }
    changed
}

pub(super) fn latest_deliverable_change(
    evidence: &[crate::tool::ArtifactEvidence],
) -> Option<(usize, &crate::tool::ArtifactEvidence)> {
    evidence
        .iter()
        .enumerate()
        .rev()
        .find(|(_, item)| {
            is_artifact_change_operation(&item.operation) && item.artifact_type != "text_source"
        })
        .or_else(|| {
            evidence
                .iter()
                .enumerate()
                .rev()
                .find(|(_, item)| is_artifact_change_operation(&item.operation))
        })
}

pub(super) fn authoritative_deliverables(
    evidence: &[crate::tool::ArtifactEvidence],
) -> Vec<ArtifactReference> {
    let mut latest = BTreeMap::new();
    for item in evidence
        .iter()
        .filter(|item| is_artifact_change_operation(&item.operation))
    {
        latest.insert(item.path.clone(), item);
    }
    latest
        .into_values()
        .map(|item| ArtifactReference {
            path: item.path.clone(),
            artifact_type: item.artifact_type.clone(),
            sha256: item.sha256.clone(),
            validation_status: item.validation_status.clone(),
        })
        .collect()
}

pub(super) fn is_artifact_change_operation(operation: &str) -> bool {
    !matches!(operation, "inspected" | "command_observed")
}

pub(super) fn has_blocking_artifact_warning(evidence: &crate::tool::ArtifactEvidence) -> bool {
    evidence
        .warnings
        .iter()
        .any(|warning| warning.severity == "error")
}

pub(super) fn completion_warnings_for_deliverable(
    deliverable: Option<&crate::tool::ArtifactEvidence>,
    evidence: &[crate::tool::ArtifactEvidence],
    inspected: bool,
    visual_verification_missing: bool,
) -> Vec<CompletionWarning> {
    let mut warnings = Vec::new();
    if let Some(deliverable) = deliverable {
        for warning in evidence
            .iter()
            .filter(|item| item.path == deliverable.path && item.sha256 == deliverable.sha256)
            .flat_map(|item| &item.warnings)
            .filter(|warning| warning.severity != "error")
        {
            if !warnings.iter().any(|existing: &CompletionWarning| {
                existing.code == warning.code && existing.message == warning.message
            }) {
                warnings.push(CompletionWarning {
                    code: warning.code.clone(),
                    message: warning.message.clone(),
                    artifact_path: Some(deliverable.path.clone()),
                });
            }
        }
        if !inspected {
            let message = if is_artifact_change_operation(&deliverable.operation) {
                "The output was created or changed successfully but was not structurally inspected at its final hash."
            } else {
                "The existing artifact was used without structural inspection at its observed hash."
            };
            warnings.push(CompletionWarning {
                code: "artifact_not_structurally_inspected".into(),
                message: message.into(),
                artifact_path: Some(deliverable.path.clone()),
            });
        }
        if visual_verification_missing {
            warnings.push(CompletionWarning {
                code: "application_view_not_verified".into(),
                message: "The artifact was not freshly verified in its application after the final change.".into(),
                artifact_path: Some(deliverable.path.clone()),
            });
        }
    }
    warnings.truncate(20);
    warnings
}

pub(super) fn artifact_title_matches(path: &std::path::Path, title: &str) -> bool {
    let normalize = |value: &str| {
        value
            .chars()
            .filter(|character| character.is_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect::<String>()
    };
    let title = normalize(title);
    let raw_path = path.to_string_lossy();
    let file_name = raw_path
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(raw_path.as_ref());
    let stem = file_name
        .rsplit_once('.')
        .map_or(file_name, |(stem, _)| stem);
    let stem = normalize(stem);
    !stem.is_empty() && title.contains(&stem)
}

pub(super) fn is_focused_visual_artifact_evidence(name: &str, value: &Value) -> bool {
    name == "capture_screen"
        && value.get("verification_state").and_then(Value::as_str) == Some("clean")
        && matches!(
            value.pointer("/target/scope").and_then(Value::as_str),
            Some("window" | "active_window")
        )
        && value
            .pointer("/target/app")
            .and_then(Value::as_str)
            .is_some_and(|app| !app.trim().is_empty())
        && value.get("executed").and_then(Value::as_bool) != Some(false)
}
