//! Reading model output: recovering malformed, XML-style, or text-embedded tool calls and classifying each turn's payload.

use super::*;

pub(super) fn extract_json_object(text: &str) -> Option<&str> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    (end >= start).then_some(&text[start..=end])
}

pub(super) fn bounded_json_value(text: &str, reasoning: &str) -> Option<Value> {
    [text, reasoning].into_iter().find_map(|source| {
        let json = extract_json_object(source)?;
        serde_json::from_str(json).ok()
    })
}

pub(super) fn model_id_matches(discovered: &str, selected: &str) -> bool {
    discovered == selected
        || discovered.split_once('@').map(|(id, _)| id) == Some(selected)
        || selected.split_once('@').map(|(id, _)| id) == Some(discovered)
}

pub(super) fn recover_tool_calls(text: &str, allowed_names: &[String]) -> Vec<CompletedToolCall> {
    let mut candidates = Vec::new();
    let mut remaining = text;
    while let Some(start) = remaining.find("[TOOL_REQUEST]") {
        let after = &remaining[start + "[TOOL_REQUEST]".len()..];
        let Some(end) = after.find("[END_TOOL_REQUEST]") else {
            break;
        };
        candidates.push(&after[..end]);
        remaining = &after[end + "[END_TOOL_REQUEST]".len()..];
    }
    if candidates.is_empty() && text.trim_start().starts_with('{') {
        candidates.push(text.trim());
    }
    candidates
        .into_iter()
        .filter_map(|candidate| {
            let value: Value = serde_json::from_str(candidate).ok()?;
            let name = value.get("name").and_then(Value::as_str)?.to_owned();
            if !allowed_names.contains(&name) {
                return None;
            }
            let arguments = value.get("arguments").cloned().unwrap_or_else(|| json!({}));
            Some(CompletedToolCall {
                id: Uuid::new_v4().to_string(),
                name,
                arguments,
            })
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ModelTurnPayloadKind {
    ValidToolCalls,
    MalformedToolCalls,
    VisibleText,
    ReasoningOnly,
    TrulyEmpty,
}

impl ModelTurnPayloadKind {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::ValidToolCalls => "valid_tool_calls",
            Self::MalformedToolCalls => "malformed_tool_calls",
            Self::VisibleText => "visible_text",
            Self::ReasoningOnly => "reasoning_only",
            Self::TrulyEmpty => "truly_empty",
        }
    }
}

pub(super) fn model_turn_payload_kind(
    calls: &[CompletedToolCall],
    text: &str,
    reasoning: &str,
    malformed: &[(String, String, String)],
) -> ModelTurnPayloadKind {
    let only_malformed_bookkeeping = !malformed.is_empty()
        && malformed
            .iter()
            .all(|(name, _, _)| name == "update_task_plan");
    if calls.is_empty() && !text.trim().is_empty() && only_malformed_bookkeeping {
        ModelTurnPayloadKind::VisibleText
    } else if !malformed.is_empty() {
        ModelTurnPayloadKind::MalformedToolCalls
    } else if !calls.is_empty() {
        ModelTurnPayloadKind::ValidToolCalls
    } else if !text.trim().is_empty() {
        ModelTurnPayloadKind::VisibleText
    } else if !reasoning.trim().is_empty() {
        ModelTurnPayloadKind::ReasoningOnly
    } else {
        ModelTurnPayloadKind::TrulyEmpty
    }
}

pub(super) fn recover_malformed_tool_call(
    malformed: &(String, String, String),
    tools: &ToolRegistry,
) -> Option<CompletedToolCall> {
    let (name, raw_arguments, _) = malformed;
    let schema = crate::brain::reference_free_schema(&tools.input_schema(name)?);
    let mut enum_fields = BTreeMap::<String, BTreeSet<String>>::new();
    collect_schema_enum_fields(&schema, &mut enum_fields);
    let bare_value =
        Regex::new(r#"("([^"\\]+)"\s*:\s*)([A-Za-z_][A-Za-z0-9_-]*)(\s*[,}\]])"#).ok()?;
    let mut changed = false;
    let repaired = bare_value.replace_all(raw_arguments, |captures: &regex::Captures<'_>| {
        let field = captures.get(2).map_or("", |value| value.as_str());
        let candidate = captures.get(3).map_or("", |value| value.as_str());
        if enum_fields
            .get(field)
            .is_some_and(|values| values.contains(candidate))
        {
            changed = true;
            format!("{}\"{}\"{}", &captures[1], candidate, &captures[4])
        } else {
            captures[0].to_owned()
        }
    });
    if !changed {
        return None;
    }
    let arguments: Value = serde_json::from_str(&repaired).ok()?;
    arguments.as_object()?;
    Some(CompletedToolCall {
        id: Uuid::new_v4().to_string(),
        name: name.clone(),
        arguments,
    })
}

pub(super) fn collect_schema_enum_fields(
    schema: &Value,
    fields: &mut BTreeMap<String, BTreeSet<String>>,
) {
    if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
        for (name, property) in properties {
            let values = fields.entry(name.clone()).or_default();
            collect_schema_enum_values(property, values);
        }
    }
    match schema {
        Value::Object(object) => {
            for value in object.values() {
                collect_schema_enum_fields(value, fields);
            }
        }
        Value::Array(values) => {
            for value in values {
                collect_schema_enum_fields(value, fields);
            }
        }
        _ => {}
    }
}

pub(super) fn collect_schema_enum_values(schema: &Value, values: &mut BTreeSet<String>) {
    if let Some(items) = schema.get("enum").and_then(Value::as_array) {
        values.extend(items.iter().filter_map(Value::as_str).map(str::to_owned));
    }
    match schema {
        Value::Object(object) => {
            for value in object.values() {
                collect_schema_enum_values(value, values);
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_schema_enum_values(item, values);
            }
        }
        _ => {}
    }
}

pub(super) fn recover_xml_tool_call(
    content: &str,
    allowed_names: &[String],
) -> Option<CompletedToolCall> {
    let tool_start = content.find("<tool_call>")?;
    let tool_body = &content[tool_start + "<tool_call>".len()..];
    let tool_end = tool_body.find("</tool_call>")?;
    let tool_body = &tool_body[..tool_end];
    let function_start = tool_body.find("<function=")?;
    let function_body = &tool_body[function_start + "<function=".len()..];
    let name_end = function_body.find('>')?;
    let name = function_body[..name_end]
        .trim()
        .trim_matches(['"', '\''])
        .to_owned();
    if !allowed_names.contains(&name) {
        return None;
    }
    let arguments_body = &function_body[name_end + 1..];
    let function_end = arguments_body.find("</function>")?;
    let raw_arguments = arguments_body[..function_end].trim();
    let arguments = if raw_arguments.is_empty() {
        json!({})
    } else if raw_arguments.starts_with("<parameter=") {
        parse_xml_tool_parameters(raw_arguments)?
    } else {
        serde_json::from_str::<Value>(raw_arguments)
            .ok()
            .filter(Value::is_object)?
    };
    Some(CompletedToolCall {
        id: Uuid::new_v4().to_string(),
        name,
        arguments,
    })
}

pub(super) fn parse_xml_tool_parameters(mut content: &str) -> Option<Value> {
    let mut arguments = serde_json::Map::new();
    while let Some(start) = content.find("<parameter=") {
        content = &content[start + "<parameter=".len()..];
        let name_end = content.find('>')?;
        let name = content[..name_end].trim().trim_matches(['"', '\'']);
        if name.is_empty() {
            return None;
        }
        content = &content[name_end + 1..];
        let end = content.find("</parameter>")?;
        let raw_value = content[..end].trim_matches(['\r', '\n']);
        let value = match raw_value.trim() {
            "true" => json!(true),
            "false" => json!(false),
            "null" => Value::Null,
            scalar if scalar.parse::<i64>().is_ok() => json!(scalar.parse::<i64>().ok()?),
            scalar if scalar.parse::<f64>().is_ok() => json!(scalar.parse::<f64>().ok()?),
            _ => json!(raw_value),
        };
        arguments.insert(name.to_owned(), value);
        content = &content[end + "</parameter>".len()..];
    }
    (!arguments.is_empty()).then_some(Value::Object(arguments))
}

pub(super) fn contains_tool_protocol_markup(text: &str) -> bool {
    [
        "<tool_call>",
        "<function=",
        "<parameter=",
        "<parameter name=",
        "</parameter>",
        "<invoke name=",
        "</invoke>",
        "[TOOL_REQUEST]",
    ]
    .iter()
    .any(|marker| text.contains(marker))
}

pub(super) fn brain_event_kind(event: &BrainEvent) -> &'static str {
    match event {
        BrainEvent::TextDelta { .. } => "text",
        BrainEvent::ReasoningDelta { .. } => "reasoning",
        BrainEvent::ToolCall { .. } => "tool_call",
        BrainEvent::MalformedToolCall { .. } => "malformed_tool_call",
        BrainEvent::Usage { .. } => "usage",
        BrainEvent::Finished { .. } => "finished",
    }
}

pub(super) fn tool_delay_stage(name: &str) -> &'static str {
    match name {
        "capture_screen"
        | "inspect_screen_region"
        | "locate_visual_target"
        | "query_screen_text"
        | "query_window_tree" => "desktop_capture_or_accessibility_enrichment",
        "observe_desktop" | "list_windows" | "activate_window" => "native_window_operation",
        "click_target"
        | "click_localized"
        | "scroll_view"
        | "scroll_until_text"
        | "type_text"
        | "simulate_input"
        | "execute_action_batch" => "desktop_action_and_verification",
        "run_command" | "manage_command" => "command_execution",
        _ => "tool_execution",
    }
}
