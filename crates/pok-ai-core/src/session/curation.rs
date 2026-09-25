//! Post-turn curation: durable memory proposals and verification, and learning reusable procedures and helper tools from verified workflows.

use super::*;

#[derive(serde::Deserialize)]
pub(super) struct CurationEnvelope {
    #[serde(default)]
    pub(super) proposals: Vec<CurationProposal>,
}

#[derive(serde::Deserialize)]
pub(super) struct CurationProposal {
    pub(super) source: String,
    pub(super) text: String,
}

pub(super) fn workflow_step(name: &str, arguments: &Value, result: &Value) -> Option<Value> {
    const PROCEDURE_TOOLS: &[&str] = &[
        "observe_desktop",
        "list_windows",
        "activate_window",
        "browser_navigate",
        "capture_screen",
        "query_screen_text",
        "query_window_tree",
        "click_target",
        "locate_visual_target",
        "click_localized",
        "move_pointer",
        "drag_pointer",
        "drag_target",
        "type_text",
        "scroll_view",
        "scroll_until_text",
        "simulate_input",
        "execute_action_batch",
        "run_command",
    ];
    if !PROCEDURE_TOOLS.contains(&name) {
        return None;
    }
    let action = result.get("action").map(|action| {
        let mut action = action.clone();
        if action.get("kind").and_then(Value::as_str) == Some("type_text") {
            action["text"] = Value::String("<USER_TEXT>".into());
        }
        if action.get("model_point").is_some() {
            action["model_point"] = Value::String("<FRESH_UI_TARGET>".into());
        }
        if action.get("target_id").is_some() {
            action["target_id"] = Value::String("<FRESH_TARGET_ID>".into());
        }
        if action.get("query").is_some() {
            action["query"] = Value::String("<TARGET_TEXT>".into());
        }
        if action.get("label").is_some() {
            action["label"] = Value::String("<CURRENT_UI_LABEL>".into());
        }
        if action.get("target_label").is_some() {
            action["target_label"] = Value::String("<CURRENT_UI_LABEL>".into());
        }
        action
    });
    let target = result.get("target").map(|target| {
        json!({
            "scope": target.get("scope"),
            "app": target.get("app"),
        })
    });
    let focus = result.get("focus").map_or_else(
        || {
            result.get("process_name").map(|app| {
                json!({
                    "app": app,
                })
            })
        },
        |focus| Some(json!({"app": focus.get("app")})),
    );
    Some(json!({
        "tool": name,
        "executed": result.get("executed").cloned().unwrap_or(Value::Bool(true)),
        "success": result.get("success").cloned().unwrap_or(Value::Bool(true)),
        "arguments": sanitize_workflow_arguments(name, arguments),
        "action": action,
        "target": target,
        "focus": focus,
        "state_change_count": result.pointer("/state_change/added_text").and_then(Value::as_array).map_or(0, Vec::len)
            + result.pointer("/state_change/removed_control_count").and_then(Value::as_u64).unwrap_or(0) as usize
            + usize::from(result.pointer("/state_change/focus_changed").and_then(Value::as_bool).unwrap_or(false))
            + result.pointer("/state_change/selected_changed").and_then(Value::as_array).map_or(0, Vec::len)
            + usize::from(result.pointer("/state_change/focused_control_changed").and_then(Value::as_bool).unwrap_or(false)),
        "submission_status": result.pointer("/submission/status"),
        "outcome": result.pointer("/_pok_continuity/outcome"),
        "error": result.get("error"),
    }))
}

pub(super) fn sanitize_workflow_arguments(name: &str, arguments: &Value) -> Value {
    let mut sanitized = arguments.clone();
    let Some(object) = sanitized.as_object_mut() else {
        return Value::Null;
    };
    for key in [
        "text",
        "message",
        "content",
        "recipient",
        "target_id",
        "observation_id",
        "model_point",
        "window_id",
        "monitor_id",
    ] {
        if object.contains_key(key) {
            object.insert(
                key.into(),
                Value::String(
                    match key {
                        "text" | "message" | "content" => "<USER_TEXT>",
                        "recipient" => "<CURRENT_RECIPIENT>",
                        "target_id" => "<FRESH_TARGET_ID>",
                        "observation_id" => "<FRESH_OBSERVATION>",
                        "model_point" => "<FRESH_UI_TARGET>",
                        "window_id" => "<CURRENT_WINDOW>",
                        "monitor_id" => "<CURRENT_MONITOR>",
                        _ => "<CURRENT_VALUE>",
                    }
                    .into(),
                ),
            );
        }
    }
    if name == "browser_navigate" {
        for key in ["query_or_url", "url"] {
            if object.contains_key(key) {
                object.insert(key.into(), Value::String("<CURRENT_DESTINATION>".into()));
            }
        }
    }
    if matches!(name, "run_command" | "manage_command") {
        if let Some(command) = object.get("command").and_then(Value::as_str) {
            object.insert(
                "command".into(),
                Value::String(normalize_command_template(command)),
            );
        }
        if object.contains_key("cwd") {
            object.insert("cwd".into(), Value::String("<CURRENT_WORKSPACE>".into()));
        }
    }
    if name == "execute_action_batch" {
        object.remove("steps");
        object.insert(
            "steps".into(),
            Value::String("<SANITIZED_CURRENT_ACTION_SEQUENCE>".into()),
        );
    }
    sanitized
}

pub(super) fn normalize_command_template(command: &str) -> String {
    let mut normalized = command.trim().replace('\n', " ");
    let whitespace = Regex::new(r"\s+").expect("static regex");
    normalized = whitespace.replace_all(&normalized, " ").into_owned();
    let secrets = Regex::new(
        r#"(?i)(api[_-]?key|token|password|secret)\s*[:=]\s*("[^"]*"|'[^']*'|[^\s;|]+)"#,
    )
    .expect("static regex");
    normalized = secrets
        .replace_all(&normalized, "$1=<REDACTED>")
        .into_owned();
    let task_values = Regex::new(
        r#"(?i)(--?(?:body|message|recipient|text|query|location|latitude|longitude))(?:\s+|=)("[^"]*"|'[^']*'|[^\s;|]+)"#,
    )
    .expect("static regex");
    normalized = task_values
        .replace_all(&normalized, "$1=<CURRENT_VALUE>")
        .into_owned();
    let url_queries = Regex::new(r#"(https?://[^\s"'?]+)\?[^\s"']+"#).expect("static regex");
    normalized = url_queries
        .replace_all(&normalized, "$1?<CURRENT_QUERY_PARAMETERS>")
        .into_owned();
    let windows_paths = Regex::new(r#"(?i)([A-Z]:\\Users\\)[^\\\s"']+"#).expect("static regex");
    normalized = windows_paths
        .replace_all(&normalized, "${1}<USER>")
        .into_owned();
    let unix_home = Regex::new(r#"(/home/)[^/\s"']+"#).expect("static regex");
    normalized = unix_home
        .replace_all(&normalized, "${1}<USER>")
        .into_owned();
    if normalized.len() > 2_000 {
        normalized.truncate(2_000);
        normalized.push('…');
    }
    normalized
}

#[derive(Debug)]
pub(super) struct WorkflowQualification {
    pub(super) eligible: bool,
    pub(super) applications: Vec<String>,
    pub(super) evidence: &'static str,
}

pub(super) fn qualify_workflow(
    workflow: &[Value],
    verified: &[VerifiedSubmission],
) -> WorkflowQualification {
    let application_for = |step: &Value| {
        step.pointer("/focus/app")
            .or_else(|| step.pointer("/target/app"))
            .and_then(Value::as_str)
            .filter(|app| !app.trim().is_empty() && !is_system_shell_application(app))
            .map(str::to_owned)
    };
    let mut applications = workflow
        .iter()
        .filter(|step| {
            successful_workflow_step(step)
                && matches!(
                    step.get("tool").and_then(Value::as_str),
                    Some(
                        "activate_window"
                            | "browser_navigate"
                            | "click_target"
                            | "type_text"
                            | "scroll_view"
                            | "scroll_until_text"
                            | "simulate_input"
                            | "execute_action_batch"
                    )
                )
        })
        .filter_map(application_for)
        .collect::<Vec<_>>();
    if applications.is_empty() {
        applications = workflow.iter().filter_map(application_for).collect();
    }
    applications.sort();
    applications.dedup();
    let input_indices = workflow
        .iter()
        .enumerate()
        .filter(|(_, step)| {
            successful_workflow_step(step)
                && matches!(
                    step.get("tool").and_then(Value::as_str),
                    Some(
                        "click_target"
                            | "type_text"
                            | "scroll_view"
                            | "scroll_until_text"
                            | "simulate_input"
                            | "activate_window"
                            | "browser_navigate"
                            | "execute_action_batch"
                    )
                )
        })
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let changed = workflow.iter().any(|step| {
        successful_workflow_step(step)
            && step
                .get("state_change_count")
                .and_then(Value::as_u64)
                .is_some_and(|count| count > 0)
    });
    let (eligible, evidence) = if !verified.is_empty() {
        (true, "verified_external_outcome")
    } else if !input_indices.is_empty() && changed {
        // Desktop input tools synchronously refresh UIA after execution. Their
        // state-change evidence is therefore already a post-input observation.
        (true, "post_input_uia_state_change")
    } else {
        (false, "insufficient_verification")
    };
    WorkflowQualification {
        eligible,
        applications,
        evidence,
    }
}

pub(super) fn workflow_quality_rejection(
    workflow: &[Value],
    metrics: &RunMetrics,
) -> Option<&'static str> {
    if workflow.iter().any(|step| {
        matches!(
            step.get("outcome").and_then(Value::as_str),
            Some("blocked" | "blocked_repeat")
        )
    }) {
        return Some("workflow contained blocked repeated actions");
    }
    let tool_errors = metrics
        .extras
        .get("tool_errors")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    if tool_errors > 2 {
        return Some("workflow exceeded the tool-error quality limit");
    }
    let recent_verified_change = workflow.iter().rev().take(8).any(|step| {
        successful_workflow_step(step)
            && step
                .get("state_change_count")
                .and_then(Value::as_u64)
                .is_some_and(|count| count > 0)
    });
    if !recent_verified_change {
        return Some("workflow lacked fresh terminal state-change evidence");
    }
    None
}

pub(super) fn workflow_learning_rejection(
    prompt: &str,
    qualification: &WorkflowQualification,
    workflow: &[Value],
    metrics: &RunMetrics,
) -> Option<&'static str> {
    if requests_live_information(prompt) && qualification.evidence == "post_input_uia_state_change"
    {
        Some("read-only live information was not outcome-verified")
    } else {
        workflow_quality_rejection(workflow, metrics)
    }
}

pub(super) fn successful_workflow_step(step: &Value) -> bool {
    step.get("executed").and_then(Value::as_bool) != Some(false)
        && step.get("success").and_then(Value::as_bool) != Some(false)
        && step.get("error").is_none_or(Value::is_null)
        && !matches!(
            step.get("outcome").and_then(Value::as_str),
            Some("no_progress" | "blocked" | "blocked_repeat" | "error")
        )
}

pub(super) fn is_system_shell_application(app: &str) -> bool {
    matches!(
        app.trim()
            .trim_matches(['[', ']'])
            .to_ascii_lowercase()
            .as_str(),
        "system process"
            | "applicationframehost.exe"
            | "searchhost.exe"
            | "shellexperiencehost.exe"
            | "startmenuexperiencehost.exe"
    )
}

pub(super) fn learn_verified_procedures(
    memory: &crate::memory::MemoryStore,
    prompt: &str,
    workflow: &[Value],
    qualification: &WorkflowQualification,
) -> Result<Vec<(crate::memory::ProcedureRecord, bool)>> {
    let signature = task_signature(prompt);
    if signature.is_empty() || is_trivial_memory_task(prompt) {
        return Ok(Vec::new());
    }
    let title = concise_task_title(prompt);
    let mut learned = Vec::new();
    if qualification.eligible {
        let steps = procedure_steps(workflow);
        if steps.iter().any(|step| {
            matches!(
                step.tool.as_str(),
                "activate_window"
                    | "browser_navigate"
                    | "click_target"
                    | "type_text"
                    | "scroll_view"
                    | "scroll_until_text"
                    | "simulate_input"
                    | "execute_action_batch"
            )
        }) {
            let fingerprint = procedure_fingerprint(
                ProcedureKind::Workflow,
                &signature,
                &qualification.applications,
                &steps,
                None,
            );
            learned.push(memory.save_or_reinforce_procedure(NewProcedure {
                kind: ProcedureKind::Workflow,
                task_signature: signature.clone(),
                title: title.clone(),
                summary: format!(
                    "Previously successful approach for this task using {}. Re-observe every target and value.",
                    if qualification.applications.is_empty() {
                        "the Windows desktop".into()
                    } else {
                        qualification.applications.join(", ")
                    }
                ),
                applications: qualification.applications.clone(),
                steps,
                command_template: None,
                evidence: qualification.evidence.into(),
                fingerprint,
                verified_successes: 1,
            })?);
        }
    }
    Ok(learned)
}

pub(super) fn record_helper_candidates(
    context: &ToolContext,
    prompt: &str,
    workflow: &[Value],
) -> Vec<Result<Option<crate::generated_tools::GeneratedToolCandidate>>> {
    let mut helpers = std::collections::BTreeMap::new();
    for (path, runtime, command) in workflow.iter().filter_map(|step| {
        if step.get("tool").and_then(Value::as_str) != Some("run_command")
            || step.get("success").and_then(Value::as_bool) != Some(true)
        {
            return None;
        }
        let command = step.pointer("/arguments/command")?.as_str()?;
        if command.contains("<REDACTED>") {
            return None;
        }
        helper_from_command(command).map(|(path, runtime)| (path, runtime, command))
    }) {
        helpers.insert(path.clone(), (path, runtime, command));
    }
    helpers
        .into_values()
        .map(|(path, runtime, command)| {
            crate::generated_tools::record_generated_tool_candidate(
                &context.data_dir,
                &context.workspace,
                &path,
                &concise_task_title(prompt),
                runtime,
                command,
            )
        })
        .collect()
}

pub(super) fn helper_from_command(
    command: &str,
) -> Option<(std::path::PathBuf, crate::generated_tools::ScriptRuntime)> {
    let script = Regex::new(
        r#"(?i)(?:\"([^\"]+\.(?:ps1|py|m?js))\"|'([^']+\.(?:ps1|py|m?js))'|([^\s;|&()]+\.(?:ps1|py|m?js)))"#,
    )
    .expect("valid helper command path regex");
    script.captures_iter(command).find_map(|captures| {
        let token = captures
            .get(1)
            .or_else(|| captures.get(2))
            .or_else(|| captures.get(3))?
            .as_str();
        let lower = token.to_ascii_lowercase();
        let runtime = if lower.ends_with(".ps1") {
            crate::generated_tools::ScriptRuntime::Powershell
        } else if lower.ends_with(".py") {
            crate::generated_tools::ScriptRuntime::Python
        } else if lower.ends_with(".js") || lower.ends_with(".mjs") {
            crate::generated_tools::ScriptRuntime::Node
        } else {
            return None;
        };
        Some((std::path::PathBuf::from(token), runtime))
    })
}

pub(super) fn procedure_steps(workflow: &[Value]) -> Vec<ProcedureStep> {
    let mut seen = std::collections::BTreeSet::new();
    let mut steps = workflow
        .iter()
        .filter(|step| successful_workflow_step(step))
        .filter_map(|step| {
            let tool = step.get("tool")?.as_str()?;
            let instruction = match tool {
                "observe_desktop" | "list_windows" => {
                    "Discover the current desktop and task applications."
                }
                "activate_window" => "Activate the current task application.",
                "browser_navigate" => {
                    "Navigate to `<CURRENT_DESTINATION>` in the current browser window."
                }
                "capture_screen"
                | "inspect_screen_region"
                | "query_screen_text"
                | "query_window_tree" => {
                    "Read the current application and resolve fresh UIA/OCR targets."
                }
                "click_target" => "Click the freshly resolved `<CURRENT_UI_LABEL>` target.",
                "type_text" => {
                    "Type the complete current task text into a freshly verified control."
                }
                "scroll_view" => "Scroll the freshly observed task region.",
                "scroll_until_text" => {
                    "Scroll until the current task's `<TARGET_TEXT>` is visible."
                }
                "simulate_input" => "Apply current task input only to a freshly verified control.",
                "execute_action_batch" => {
                    "Execute a freshly grounded action sequence using current task values."
                }
                "run_command" => "Run the verified command template and require a successful exit.",
                _ => return None,
            };
            let procedure_step = ProcedureStep {
                tool: tool.into(),
                instruction: instruction.into(),
            };
            seen.insert((
                procedure_step.tool.clone(),
                procedure_step.instruction.clone(),
            ))
            .then_some(procedure_step)
        })
        .collect::<Vec<_>>();
    steps.truncate(12);
    steps
}

pub(super) fn task_signature(prompt: &str) -> String {
    const STOP_WORDS: &[&str] = &[
        "a", "an", "and", "are", "can", "could", "for", "from", "i", "in", "is", "it", "me", "my",
        "of", "on", "please", "the", "this", "to", "using", "want", "with", "you",
    ];
    let mut words = prompt
        .split_whitespace()
        .map(|word| {
            word.chars()
                .filter(|character| character.is_alphanumeric() || *character == '_')
                .collect::<String>()
                .to_ascii_lowercase()
        })
        .map(|word| match word.as_str() {
            "create" | "generate" | "make" => "create".into(),
            "find" | "locate" | "search" => "find".into(),
            "get" | "fetch" | "retrieve" => "get".into(),
            "launch" | "open" | "start" => "open".into(),
            "post" | "send" => "send".into(),
            _ => word,
        })
        .filter(|word| word.len() > 1 && !STOP_WORDS.contains(&word.as_str()))
        .collect::<Vec<_>>();
    words.sort();
    words.dedup();
    words.truncate(24);
    words.join(" ")
}

pub(super) fn inferred_tool_groups(prompt: &str) -> BTreeSet<String> {
    let normalized = prompt.to_ascii_lowercase();
    let contains_any = |terms: &[&str]| terms.iter().any(|term| normalized.contains(term));
    let mut groups = BTreeSet::from(["control".into(), "desktop".into()]);
    if contains_any(&[
        "browser",
        "website",
        "web site",
        "webpage",
        "web page",
        "url",
        "http://",
        "https://",
        "navigate",
        "online",
        "search the web",
    ]) {
        groups.insert("browser".into());
    }
    if contains_any(&[
        "code",
        "repo",
        "repository",
        "file",
        "folder",
        "directory",
        "command",
        "terminal",
        "powershell",
        "script",
        "build",
        "test",
        "compile",
        "document",
        "pdf",
    ]) {
        groups.insert("coding".into());
        groups.insert("system".into());
    }
    if contains_any(&[
        "command",
        "terminal",
        "powershell",
        "shell",
        "process",
        "service",
        "network",
        "subnet",
        "ip address",
        "port",
        "system information",
        "device",
        "weather",
        "forecast",
        "news",
        "stock price",
        "exchange rate",
        "current online",
        "live external",
    ]) {
        groups.insert("system".into());
    }
    if contains_any(&["remember", "memory", "skill", "procedure", "preference"]) {
        groups.insert("memory".into());
    }
    if contains_any(&[
        "earlier session",
        "previous session",
        "conversation history",
        "archive",
    ]) {
        groups.insert("archive".into());
    }
    if contains_any(&[
        "generated tool",
        "helper tool",
        "reusable helper",
        "promote helper",
    ]) {
        groups.insert("generated".into());
    }
    if contains_any(&["subagent", "sub-agent", "parallel coding"]) {
        groups.insert("subagent".into());
    }
    groups
}

pub(super) fn task_plan_fingerprint(state: &ActiveTaskState) -> String {
    let bytes = serde_json::to_vec(&json!({
        "status": state.status,
        "current_step": state.current_step,
        "steps": state.steps,
    }))
    .unwrap_or_default();
    format!("{:x}", Sha256::digest(bytes))
}

pub(super) fn generated_tool_guidance(
    context: &ToolContext,
    prompt: &str,
) -> Option<(String, Value)> {
    let query_terms = capability_terms(prompt);
    let score = |text: &str| capability_match_score(&query_terms, text);
    let all_tools =
        crate::generated_tools::list_generated_tools(&context.data_dir, &context.workspace).ok()?;
    let mut tools = all_tools
        .iter()
        .filter_map(|tool| {
            score(&format!(
                "{} {} {}",
                tool.name,
                tool.description,
                tool.capabilities.join(" ")
            ))
            .map(|relevance| (relevance, tool.clone()))
        })
        .collect::<Vec<_>>();
    tools.sort_by(|left, right| right.0.cmp(&left.0));
    let mut candidates = crate::generated_tools::list_generated_tool_candidates(
        &context.data_dir,
        &context.workspace,
    )
    .ok()?
    .into_iter()
    .filter_map(|candidate| {
        let relevance = score(&format!(
            "{} {}",
            candidate.task,
            candidate.helper_path.display()
        ))?;
        Some((relevance, candidate))
    })
    .collect::<Vec<_>>();
    candidates.sort_by(|left, right| right.0.cmp(&left.0));
    tools.truncate(3);
    candidates.truncate(2);
    let skills = context
        .memory
        .list_procedures(Some(ProcedureKind::Workflow), 100)
        .ok()
        .unwrap_or_default()
        .into_iter()
        .filter(|skill| skill.enabled)
        .collect::<Vec<_>>();
    if all_tools.is_empty() && skills.is_empty() && candidates.is_empty() {
        return None;
    }
    let mut lines = vec!["<capability_catalog>".to_string()];
    lines.push(
        "Reusable learned skills and generated tools follow. If their schemas are not in the current compact tool set, enable the memory or generated family with `discover_tools`; then use `skill_load` or `search_generated_tools` for full details, and invoke only a strong task match.".into(),
    );
    lines.extend(
        skills
            .iter()
            .map(|skill| format!("- Skill `{}`: {}", skill.id, skill.summary)),
    );
    lines.extend(all_tools.iter().map(|tool| {
        format!(
            "- Generated tool `{}` (enabled={}): {}",
            tool.name, tool.enabled, tool.description
        )
    }));
    lines.push("</capability_catalog>".into());
    if !tools.is_empty() || !candidates.is_empty() {
        lines.push(
            "Strong capability matches for the current request are listed below. Prefer them over recreating equivalent scripts.".into(),
        );
    }
    lines.extend(tools.iter().map(|(_, tool)| {
        format!(
            "- Tool `{}` (enabled={}): {}",
            tool.name, tool.enabled, tool.description
        )
    }));
    lines.extend(candidates.iter().map(|(_, candidate)| {
        format!(
            "- Promotion candidate `{}` at `{}`: {}. It is not installed; inspect it and use `promote_helper_tool` only if reuse is warranted and approval is granted.",
            candidate.id,
            candidate.helper_path.display(),
            candidate.task
        )
    }));
    Some((
        {
            let mut block = lines.join("\n");
            if block.len() > 4_096 {
                block.truncate(4_095);
                block.push('…');
            }
            block
        },
        json!({
            "tools": tools.iter().map(|(score, tool)| json!({
                "name": tool.name,
                "enabled": tool.enabled,
                "score": score,
            })).collect::<Vec<_>>(),
            "candidates": candidates.iter().map(|(score, candidate)| json!({
                "id": candidate.id,
                "helper_path": candidate.helper_path,
                "score": score,
            })).collect::<Vec<_>>(),
        }),
    ))
}

pub(super) fn capability_match_score(query_terms: &[String], text: &str) -> Option<usize> {
    let candidate = capability_terms(text);
    let matches = query_terms
        .iter()
        .filter(|term| candidate.contains(term))
        .count();
    let high_signal = query_terms.iter().any(|term| {
        candidate.contains(term)
            && (term.len() >= 6 || matches!(term.as_str(), "pdf" | "mkv" | "docx" | "xlsx"))
    });
    (matches >= 2 || (matches == 1 && high_signal)).then_some(matches)
}

pub(super) fn capability_terms(text: &str) -> Vec<String> {
    const GENERIC: &[&str] = &[
        "and", "are", "can", "could", "data", "file", "for", "from", "get", "have", "into", "open",
        "please", "read", "that", "the", "this", "tool", "use", "using", "want", "with", "you",
    ];
    let mut terms = text
        .split(|character: char| !character.is_alphanumeric() && character != '_')
        .map(str::to_ascii_lowercase)
        .map(|term| match term.as_str() {
            "documents" => "document".into(),
            "movies" => "movie".into(),
            "stocks" => "stock".into(),
            _ => term,
        })
        .filter(|term| term.len() >= 3 && !GENERIC.contains(&term.as_str()))
        .collect::<Vec<_>>();
    terms.sort();
    terms.dedup();
    terms
}

pub(super) fn concise_task_title(prompt: &str) -> String {
    let normalized = prompt.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut title = normalized.chars().take(100).collect::<String>();
    if normalized.chars().count() > 100 {
        title.push('…');
    }
    title
}

pub(super) fn is_trivial_memory_task(prompt: &str) -> bool {
    let lower = prompt.to_ascii_lowercase();
    ["what time", "current time", "what date", "current date"]
        .iter()
        .any(|phrase| lower.contains(phrase))
}

pub(super) fn procedure_fingerprint(
    kind: ProcedureKind,
    signature: &str,
    applications: &[String],
    steps: &[ProcedureStep],
    command: Option<&str>,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(format!("{kind:?}|{signature}|").as_bytes());
    hasher.update(applications.join("|").to_ascii_lowercase().as_bytes());
    for step in steps {
        hasher.update(step.tool.as_bytes());
        hasher.update(step.instruction.as_bytes());
    }
    if let Some(command) = command {
        hasher.update(command.to_ascii_lowercase().as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn curate_turn(
    brain: Arc<dyn Brain>,
    model: &str,
    prompt: &str,
    answer: &str,
    workflow: &[Value],
    qualification: &WorkflowQualification,
    related_memories: &[Value],
    cancellation: &CancellationToken,
) -> Result<CurationEnvelope> {
    let request = BrainRequest {
        model: model.to_owned(),
        messages: vec![
            BrainMessage::text(
                "system",
                "You are a restricted durable-fact curator. You have no tools. Propose only stable user preferences or durable environment facts directly supported by the completed interaction and absent from RELATED MEMORIES. Never preserve recipients, message bodies, paths, coordinates, target numbers, transient values, secrets, or guesses. The harness learns verified procedures separately. Return JSON only: {\"proposals\":[{\"source\":\"user|environment\",\"text\":\"...\"}]}.",
            ),
            BrainMessage::text(
                "user",
                format!(
                    "USER REQUEST:\n{prompt}\n\nAGENT RESULT:\n{answer}\n\nRELATED MEMORIES:\n{}\n\nSANITIZED WORKFLOW:\n{}\n\nWORKFLOW ELIGIBLE: {}\nVERIFICATION: {}\nAPPLICATIONS: {}",
                    serde_json::to_string(related_memories)?,
                    serde_json::to_string(workflow)?,
                    qualification.eligible,
                    qualification.evidence,
                    qualification.applications.join(", "),
                ),
            ),
        ],
        tools: Vec::new(),
        temperature: Some(0.0),
        max_tokens: Some(800),
        seed: Some(42),
        reasoning_effort: None,
    };
    let mut stream = brain.stream(request);
    let mut text = String::new();
    loop {
        tokio::select! {
            () = cancellation.cancelled() => return Err(PokError::Cancelled),
            event = stream.next() => match event {
                Some(event) => {
                    if let BrainEvent::TextDelta { text: delta } = event? {
                        text.push_str(&delta);
                    }
                }
                None => break,
            }
        }
    }
    let json = extract_json_object(&text)
        .ok_or_else(|| PokError::Provider("curator returned no JSON object".into()))?;
    Ok(serde_json::from_str::<CurationEnvelope>(json)?)
}

pub(super) async fn verify_and_save_curated_memory(
    memory: &Arc<crate::memory::MemoryStore>,
    router: Option<Arc<dyn DecisionRouter>>,
    config: Option<&crate::config::DecisionRouterConfig>,
    source: &str,
    text: &str,
    cancellation: &CancellationToken,
) -> Result<(
    Option<crate::memory::MemoryRecord>,
    String,
    Option<Uuid>,
    bool,
)> {
    let local_save = || {
        let outcome = memory.save_with_provenance(
            source,
            text,
            false,
            MemoryWriteProvenance::InferredCuration,
        )?;
        Ok((
            Some(outcome.record),
            outcome.disposition,
            outcome.related_memory_id,
            false,
        ))
    };
    let (Some(router), Some(config)) = (router, config) else {
        return local_save();
    };
    if !config.enabled || memory_text_may_be_sensitive(text) {
        return local_save();
    }
    let related = memory.related_memories(text, 4)?;
    if related.is_empty() {
        return local_save();
    }
    let mut candidates = vec![
        DecisionCandidate {
            id: "create".into(),
            tool: "memory_create".into(),
            arguments: json!({}),
            description: "Create a new durable memory because the proposal adds distinct information".into(),
            kind: DecisionCandidateKind::Memory,
            local_score: 0.0,
        },
        DecisionCandidate {
            id: "reject".into(),
            tool: "memory_reject".into(),
            arguments: json!({}),
            description: "Reject the proposal because it is transient, unsupported, sensitive, or not useful as durable memory".into(),
            kind: DecisionCandidateKind::Memory,
            local_score: 0.0,
        },
    ];
    for (index, record) in related.iter().enumerate() {
        let existing = truncate_chars(&record.text, 240);
        for (disposition, description) in [
            ("reinforce", "The proposal has the same durable meaning as"),
            ("propose_update", "The proposal may update or supersede"),
            ("conflict", "The proposal conflicts with"),
        ] {
            candidates.push(DecisionCandidate {
                id: format!("{disposition}_{index}"),
                tool: format!("memory_{disposition}"),
                arguments: json!({"memory_id": record.id}),
                description: format!("{description} existing memory: {existing}"),
                kind: DecisionCandidateKind::Memory,
                local_score: record.reinforcement_count as f64 / 100.0,
            });
        }
    }
    let request = DecisionRequest {
        purpose: DecisionPurpose::MemoryVerification,
        task: "Verify a proposed durable memory".into(),
        current_step: "Classify the proposal against related existing memories".into(),
        candidates: candidates.clone(),
        state: json!({
            "proposed_memory": truncate_chars(text, 280),
            "source": source,
        }),
    };
    let decision = tokio::select! {
        () = cancellation.cancelled() => return Err(PokError::Cancelled),
        result = router.decide(request) => match result {
            Ok(result) => result,
            Err(_) => return local_save(),
        },
    };
    let confidence = decision.confidence;
    let valid = decision.model == config.active_model()
        && decision.selected_probability.is_finite()
        && confidence.is_some_and(f64::is_finite)
        && (0.0..=1.0).contains(&decision.selected_probability)
        && confidence.is_some_and(|value| (0.0..=1.0).contains(&value))
        && decision
            .probabilities
            .values()
            .all(|score| score.is_finite() && (0.0..=1.0).contains(score))
        && decision.selected_probability >= config.min_selected_probability
        && confidence.is_some_and(|value| value >= config.min_confidence);
    if !valid {
        return local_save();
    }
    let Some(selected) = decision
        .candidate_id
        .as_deref()
        .and_then(|id| candidates.iter().find(|candidate| candidate.id == id))
    else {
        return local_save();
    };
    match selected.id.as_str() {
        "create" => {
            let outcome = memory.save_with_provenance(
                source,
                text,
                false,
                MemoryWriteProvenance::InferredCuration,
            )?;
            Ok((
                Some(outcome.record),
                "created".into(),
                outcome.related_memory_id,
                true,
            ))
        }
        "reject" => Ok((None, "rejected".into(), None, true)),
        id => {
            let related_id = selected
                .arguments
                .get("memory_id")
                .and_then(Value::as_str)
                .and_then(|id| Uuid::parse_str(id).ok());
            let Some(related_id) = related_id else {
                return local_save();
            };
            if id.starts_with("reinforce_") {
                let record = memory.reinforce_memory(
                    related_id,
                    false,
                    MemoryWriteProvenance::InferredCuration,
                )?;
                Ok((
                    Some(record),
                    "reinforced_jev".into(),
                    Some(related_id),
                    true,
                ))
            } else {
                let outcome = memory.save_with_provenance(
                    source,
                    text,
                    false,
                    MemoryWriteProvenance::InferredCuration,
                )?;
                let status = if id.starts_with("conflict_") {
                    "conflict"
                } else {
                    "proposed_update"
                };
                let record =
                    memory.mark_memory_review(outcome.record.id, status, Some(related_id))?;
                Ok((Some(record), status.into(), Some(related_id), true))
            }
        }
    }
}

pub(super) fn memory_text_may_be_sensitive(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    [
        "password",
        "api key",
        "secret",
        "access token",
        "private key",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
        || text.split_whitespace().any(|part| {
            part.len() >= 40
                && part
                    .chars()
                    .all(|ch| ch.is_ascii_alphanumeric() || "_-".contains(ch))
        })
}

impl Session {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn schedule_curation(
        &self,
        prompt: String,
        answer: String,
        workflow: Vec<Value>,
        verified: Vec<VerifiedSubmission>,
        verified_completion: bool,
        metrics: RunMetrics,
        cancellation: CancellationToken,
    ) {
        if !verified_completion {
            let _ = self.log(
                "curation_skipped",
                json!({
                    "reason": "task completion was not grounded in fresh tool evidence",
                    "prompt": prompt,
                }),
            );
            return;
        }
        let mut qualification = qualify_workflow(&workflow, &verified);
        let verified_artifact = self.context.artifact_evidence.lock().iter().any(|item| {
            item.operation == "inspected"
                && item.validation_status != "invalid"
                && !has_blocking_artifact_warning(item)
        });
        if requests_artifact_outcome(&prompt) && verified_artifact {
            qualification.eligible = true;
            qualification.evidence = "verified_artifact_outcome";
        }
        let rejection = workflow_learning_rejection(&prompt, &qualification, &workflow, &metrics);
        if let Some(reason) = rejection {
            qualification.eligible = false;
            qualification.evidence = "quality_gate_rejected";
            let _ = self.log(
                "procedure_rejected",
                json!({"reason": reason, "prompt": prompt}),
            );
        }
        match learn_verified_procedures(&self.context.memory, &prompt, &workflow, &qualification) {
            Ok(records) => {
                for (record, created) in records {
                    let _ = self.log(
                        if created {
                            "procedure_learned"
                        } else {
                            "procedure_reinforced"
                        },
                        json!({
                            "id": record.id,
                            "kind": record.kind,
                            "title": record.title,
                            "fingerprint": record.fingerprint,
                            "success_count": record.success_count,
                            "evidence": record.evidence,
                        }),
                    );
                }
            }
            Err(error) => {
                let _ = self.log(
                    "procedure_rejected",
                    json!({"reason": error.to_string(), "prompt": prompt}),
                );
            }
        }
        for candidate in record_helper_candidates(&self.context, &prompt, &workflow) {
            match candidate {
                Ok(Some(candidate)) => {
                    let _ = self.log(
                        "generated_tool_candidate_recorded",
                        json!({
                            "id": candidate.id,
                            "helper_path": candidate.helper_path,
                            "runtime": candidate.runtime,
                            "source_sha256": candidate.source_sha256,
                        }),
                    );
                }
                Ok(None) => {}
                Err(error) => {
                    let _ = self.log(
                        "generated_tool_candidate_rejected",
                        json!({"reason": error.to_string()}),
                    );
                }
            }
        }
        let model_curation_enabled = matches!(
            self.context.policy.mode,
            PolicyMode::Interactive | PolicyMode::Autonomous
        );
        let brain = self.brain.clone();
        let memory = self.context.memory.clone();
        let model = self.model.clone();
        let artifact_dir = self.context.artifact_dir.clone();
        let decision_router = self.decision_router.clone();
        let decision_router_config = self.decision_router_config.clone();
        let observer = self.observer.clone();
        let curation_session_id = self.id;
        tokio::spawn(async move {
            if tokio::time::timeout(std::time::Duration::from_secs(3), cancellation.cancelled())
                .await
                .is_ok()
            {
                return;
            }
            if !model_curation_enabled {
                let artifact = json!({
                    "status": "complete",
                    "mode": "verified_workflow_only",
                    "drafts": [],
                });
                let _ = std::fs::create_dir_all(&artifact_dir);
                let _ = crate::memory::atomic_write(
                    &artifact_dir.join(format!("curation-{}.json", Uuid::new_v4())),
                    &serde_json::to_vec_pretty(&artifact).unwrap_or_default(),
                );
                return;
            }
            let related_memories = memory
                .related_memories(&format!("{prompt} {answer}"), 8)
                .unwrap_or_default()
                .into_iter()
                .map(|record| {
                    json!({
                        "id": record.id,
                        "source": record.source,
                        "text": record.text.chars().take(280).collect::<String>(),
                        "approved": record.approved,
                        "enabled": record.enabled,
                    })
                })
                .collect::<Vec<_>>();
            let result = curate_turn(
                brain,
                &model,
                &prompt,
                &answer,
                &workflow,
                &qualification,
                &related_memories,
                &cancellation,
            )
            .await;
            let artifact = match result {
                Ok(envelope) => {
                    let mut saved = Vec::new();
                    for proposal in envelope.proposals.into_iter().take(4) {
                        if proposal.text.trim().is_empty() {
                            continue;
                        }
                        let outcome = verify_and_save_curated_memory(
                            &memory,
                            decision_router.clone(),
                            decision_router_config.as_ref(),
                            &proposal.source,
                            &proposal.text,
                            &cancellation,
                        )
                        .await;
                        if let Ok((record, disposition, related_memory_id, used_jev)) = outcome {
                            if let Some(observer) = &observer {
                                observer.emit(AgentEvent::DecisionRouterMemoryOutcome {
                                    session_id: curation_session_id,
                                    disposition: disposition.clone(),
                                    related_memory_id,
                                    used_jev,
                                });
                            }
                            if let Some(record) = record {
                                saved.push(record);
                            }
                        }
                    }
                    json!({"status": "complete", "drafts": saved})
                }
                Err(error) => json!({
                    "status": "partial",
                    "error": error.to_string(),
                }),
            };
            let _ = std::fs::create_dir_all(&artifact_dir);
            let _ = crate::memory::atomic_write(
                &artifact_dir.join(format!("curation-{}.json", Uuid::new_v4())),
                &serde_json::to_vec_pretty(&artifact).unwrap_or_default(),
            );
        });
    }
}
