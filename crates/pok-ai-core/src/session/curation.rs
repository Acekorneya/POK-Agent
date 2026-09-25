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
        "open_application",
        "managed_browser_open",
        "managed_browser_snapshot",
        "managed_browser_click",
        "managed_browser_type",
        "managed_browser_select",
        "managed_browser_scroll",
    ];
    if name == "fast_actions" {
        return Some(fast_actions_workflow_step(arguments, result));
    }
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
        "page_fingerprint": result.get("fingerprint").filter(|_| name.starts_with("managed_browser_")),
        "submission_status": result.pointer("/submission/status"),
        "outcome": result.pointer("/_pok_continuity/outcome"),
        "error": result.get("error"),
    }))
}

/// Add a workflow step. A managed-browser action that leaves a different page
/// than the previous browser step saw counts as a state change.
pub(super) fn push_workflow_step(workflow: &mut Vec<Value>, mut step: Value) {
    if let Some(fingerprint) = step.get("page_fingerprint").and_then(Value::as_str) {
        let previous = workflow
            .iter()
            .rev()
            .find_map(|earlier| earlier.get("page_fingerprint").and_then(Value::as_str));
        let acted = step
            .get("tool")
            .and_then(Value::as_str)
            .is_some_and(|tool| tool != "managed_browser_snapshot");
        if acted && previous.is_some_and(|previous| previous != fingerprint) {
            step["state_change_count"] = json!(1);
        }
    }
    workflow.push(step);
}

/// A delegated plan as a workflow step: the plan's subgoals that reached
/// their goal (also when the plan handed back later), with the label each one acted on, so a later run can issue the
/// same plan in one call. Every action that made progress counts as a state
/// change.
fn fast_actions_workflow_step(arguments: &Value, result: &Value) -> Value {
    let subgoals = result
        .get("subgoals")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_else(|| vec![result.clone()]);
    let hints = std::iter::once(arguments)
        .chain(
            arguments
                .get("then")
                .and_then(Value::as_array)
                .into_iter()
                .flatten(),
        )
        .map(|node| node.get("target_hint").and_then(Value::as_str))
        .collect::<Vec<_>>();
    let mut changes = 0;
    let reached = subgoals
        .iter()
        .enumerate()
        .filter(|(_, subgoal)| {
            matches!(
                subgoal.get("status").and_then(Value::as_str),
                Some("done" | "unverified")
            )
        })
        .map(|(index, subgoal)| {
            let acted = subgoal
                .get("steps")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter(|step| step.get("ok").and_then(Value::as_bool) == Some(true))
                .collect::<Vec<_>>();
            let progressed = acted
                .iter()
                .filter(|step| step.get("outcome").and_then(Value::as_str) == Some("progress"))
                .count();
            // A subgoal whose done_when was confirmed after acting changed the
            // screen even when its steps only report the new page as observed
            // (managed-browser clicks).
            let confirmed =
                subgoal.get("status").and_then(Value::as_str) == Some("done") && !acted.is_empty();
            changes += progressed.max(usize::from(confirmed));
            let progressed = acted
                .into_iter()
                .filter(|step| {
                    matches!(
                        step.get("outcome").and_then(Value::as_str),
                        Some("progress" | "observed")
                    )
                })
                .collect::<Vec<_>>();
            json!({
                "goal": subgoal.get("goal"),
                "target_hint": hints.get(index).copied().flatten(),
                "done_when": subgoal.get("done_when"),
                "status": subgoal.get("status"),
                "actions": progressed.iter().map(|step| json!({
                    "tool": step.get("tool"),
                    "target": step.get("target"),
                })).collect::<Vec<_>>(),
            })
        })
        .collect::<Vec<_>>();
    let status = result.get("status").and_then(Value::as_str);
    json!({
        "tool": "fast_actions",
        "executed": true,
        // A plan that handed back part-way still reached these subgoals.
        "success": !reached.is_empty(),
        "arguments": sanitize_workflow_arguments("fast_actions", arguments),
        "subgoals": reached,
        "target": result.pointer("/observation/target").map(|target| json!({
            "scope": target.get("scope"),
            "app": target.get("app"),
        })),
        "state_change_count": changes,
        "outcome": status,
        "error": result.get("error"),
    })
}

/// One reusable instruction per subgoal of a delegated plan that reached its
/// goal: what to act on and how to know it worked, in the plan's own quoted
/// labels.
fn fast_actions_instructions(step: &Value) -> Vec<String> {
    step.get("subgoals")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|subgoal| {
            let target = subgoal
                .get("target_hint")
                .and_then(Value::as_str)
                .map(str::to_owned)
                .or_else(|| {
                    let label = subgoal.pointer("/actions/0/target")?.as_str()?;
                    Some(label.split(" (").next().unwrap_or(label).to_owned())
                })?;
            let done_when = subgoal.get("done_when")?.as_str()?;
            Some(truncate_chars(
                &format!("target_hint {target}; done_when {done_when}"),
                300,
            ))
        })
        .filter(|instruction| !looks_private(instruction))
        .collect()
}

/// Personal or run-specific details a skill must not keep: long numbers,
/// email addresses, and file paths.
pub(super) fn looks_private(value: &str) -> bool {
    let mut digits = 0;
    let long_number = value.chars().any(|character| {
        digits = if character.is_ascii_digit() {
            digits + 1
        } else {
            0
        };
        digits >= 6
    });
    long_number
        || value.contains('@')
        || value.contains(":\\")
        || value.contains("/home/")
        || value.contains("/Users/")
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

#[derive(Debug, Clone)]
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
                            | "fast_actions"
                            | "managed_browser_open"
                            | "managed_browser_click"
                            | "managed_browser_type"
                            | "managed_browser_select"
                            | "managed_browser_scroll"
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
                            | "fast_actions"
                            | "managed_browser_open"
                            | "managed_browser_click"
                            | "managed_browser_type"
                            | "managed_browser_select"
                            | "managed_browser_scroll"
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

/// The procedure a verified workflow teaches, if it is worth keeping: a
/// non-trivial task that acted on the desktop or a page.
pub(super) fn verified_procedure(
    prompt: &str,
    workflow: &[Value],
    qualification: &WorkflowQualification,
) -> Option<NewProcedure> {
    let signature = task_signature(prompt);
    if signature.is_empty() || is_trivial_memory_task(prompt) || !qualification.eligible {
        return None;
    }
    let steps = procedure_steps(workflow);
    if !steps.iter().any(|step| {
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
                | "fast_actions"
                | "managed_browser_open"
                | "managed_browser_click"
                | "managed_browser_type"
                | "managed_browser_select"
        )
    }) {
        return None;
    }
    let fingerprint = procedure_fingerprint(
        ProcedureKind::Workflow,
        &signature,
        &qualification.applications,
        &steps,
        None,
    );
    Some(NewProcedure {
        kind: ProcedureKind::Workflow,
        task_signature: signature,
        title: concise_task_title(prompt),
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
    })
}

pub(super) fn learn_verified_procedures(
    memory: &crate::memory::MemoryStore,
    prompt: &str,
    workflow: &[Value],
    qualification: &WorkflowQualification,
) -> Result<Vec<(crate::memory::ProcedureRecord, bool)>> {
    let Some(procedure) = verified_procedure(prompt, workflow, qualification) else {
        return Ok(Vec::new());
    };
    Ok(vec![memory.save_or_reinforce_procedure(procedure)?])
}

/// Append one event to a session's `trace.jsonl` from background work that
/// does not hold the session (post-turn curation).
fn append_trace(artifact_dir: &std::path::Path, session_id: Uuid, kind: &str, payload: Value) {
    let event = SessionEvent {
        timestamp: Utc::now(),
        session_id,
        kind: kind.into(),
        payload,
    };
    if let Ok(line) = serde_json::to_string(&event)
        && let Ok(mut file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(artifact_dir.join("trace.jsonl"))
    {
        let _ = file.write_all(format!("{line}\n").as_bytes());
    }
}

/// How a verified workflow was folded into the skill library.
#[derive(Debug, Clone, Serialize)]
pub(super) struct LearnedSkill {
    pub(super) id: Uuid,
    pub(super) title: String,
    /// `created`, `reinforced` (same task), or `merged` (the router judged it
    /// the same task as an existing skill).
    pub(super) outcome: &'static str,
    pub(super) rewritten: bool,
    /// Why the LLM's rewrite was not used, when one was attempted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) rewrite_rejected: Option<String>,
    pub(super) success_count: u64,
}

/// Learn from a verified workflow without growing duplicate skills:
/// 1. the same task (kind, signature, apps) reinforces its skill (memory);
/// 2. a similar skill for a differently worded task is offered to the fast
///    decision model as "same task as skill A, B, C, or new?" (layer 2);
/// 3. a created or changed skill is rewritten by the LLM into reusable
///    guidance, validated, with the previous text kept for undo (layer 3).
#[allow(clippy::too_many_arguments)]
pub(super) async fn learn_skill_with_review(
    memory: &crate::memory::MemoryStore,
    router: Option<Arc<dyn DecisionRouter>>,
    config: Option<&crate::config::DecisionRouterConfig>,
    brain: Arc<dyn Brain>,
    model: &str,
    reasoning: &HelperReasoning,
    prompt: &str,
    workflow: &[Value],
    qualification: &WorkflowQualification,
    cancellation: &CancellationToken,
) -> Result<Option<LearnedSkill>> {
    let Some(procedure) = verified_procedure(prompt, workflow, qualification) else {
        return Ok(None);
    };
    let same_task_exists = memory
        .search_skills(&procedure.task_signature, 20)?
        .iter()
        .any(|skill| {
            skill.kind == procedure.kind
                && skill.task_signature == procedure.task_signature
                && skill.applications == procedure.applications
        });
    let mut merged_into = None;
    if !same_task_exists && let (Some(router), Some(config)) = (router.as_ref(), config) {
        let candidates = memory.similar_procedures(&procedure, 3)?;
        if !candidates.is_empty() {
            merged_into = same_skill_choice(
                router.as_ref(),
                config,
                &procedure,
                &candidates,
                cancellation,
            )
            .await
            .map(|index| candidates[index].id);
        }
    }
    let (record, outcome, steps_changed) = if let Some(id) = merged_into {
        let record = memory.merge_into_procedure(&id.to_string(), &procedure)?;
        (record, "merged", true)
    } else {
        let (record, created) = memory.save_or_reinforce_procedure(procedure.clone())?;
        let adopted = !created && record.steps == procedure.steps;
        (
            record,
            if created { "created" } else { "reinforced" },
            created || adopted,
        )
    };
    let mut rewritten = false;
    let mut rewrite_rejected = None;
    if steps_changed {
        let allowed = workflow
            .iter()
            .filter_map(|step| step.get("tool").and_then(Value::as_str))
            .map(str::to_owned)
            .collect::<BTreeSet<_>>();
        match write_skill_text(
            brain,
            model,
            reasoning,
            prompt,
            workflow,
            &record,
            outcome != "created",
            &allowed,
            cancellation,
        )
        .await
        {
            Ok(Ok(text)) => {
                memory.rewrite_procedure(record.id, &text.title, &text.summary, &text.steps)?;
                rewritten = true;
            }
            Ok(Err(reason)) => rewrite_rejected = Some(reason),
            Err(error) => rewrite_rejected = Some(error.to_string()),
        }
    }
    Ok(Some(LearnedSkill {
        id: record.id,
        title: record.title,
        outcome,
        rewritten,
        rewrite_rejected,
        success_count: record.success_count,
    }))
}

/// Layer 2: ask the fast decision model whether a new workflow is the same
/// task as one of a few similar saved skills. `None` means a new skill; a
/// match must clear the router's completion bar (0.80 by default).
async fn same_skill_choice(
    router: &dyn DecisionRouter,
    config: &crate::config::DecisionRouterConfig,
    procedure: &NewProcedure,
    candidates: &[crate::memory::ProcedureRecord],
    cancellation: &CancellationToken,
) -> Option<usize> {
    let mut options = candidates
        .iter()
        .enumerate()
        .map(|(index, skill)| {
            (
                format!("s{index}"),
                format!(
                    "the new task is the same task as saved skill \"{}\" ({})",
                    truncate_chars(&skill.title, 120),
                    skill.applications.join(", ")
                ),
            )
        })
        .collect::<Vec<_>>();
    options.push((
        "new".into(),
        "the new task is a different task that needs its own skill".into(),
    ));
    let state = json!({
        "new_task": {
            "title": procedure.title,
            "applications": procedure.applications,
            "steps": procedure.steps.iter().map(|step| &step.tool).collect::<Vec<_>>(),
        },
        "saved_skills": candidates.iter().enumerate().map(|(index, skill)| json!({
            "id": format!("s{index}"),
            "title": skill.title,
            "summary": skill.summary,
            "applications": skill.applications,
            "steps": skill.steps.iter().map(|step| &step.tool).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
    });
    let choice = tokio::select! {
        () = cancellation.cancelled() => return None,
        choice = router.choose_condition("decide whether a verified workflow updates an existing skill", &options, &state) => choice.ok()?,
    };
    (choice.accepted && choice.probability >= config.min_completion_probability)
        .then(|| choice.option_id.strip_prefix('s')?.parse::<usize>().ok())
        .flatten()
        .filter(|index| *index < candidates.len())
}

/// A skill's reusable text as written by the LLM (layer 3).
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub(super) struct SkillText {
    pub(super) title: String,
    pub(super) summary: String,
    pub(super) steps: Vec<ProcedureStep>,
}

#[allow(clippy::too_many_arguments)]
async fn write_skill_text(
    brain: Arc<dyn Brain>,
    model: &str,
    reasoning: &HelperReasoning,
    prompt: &str,
    workflow: &[Value],
    existing: &crate::memory::ProcedureRecord,
    updating: bool,
    allowed_tools: &BTreeSet<String>,
    cancellation: &CancellationToken,
) -> Result<std::result::Result<SkillText, String>> {
    let existing_text = json!({
        "title": existing.title,
        "summary": existing.summary,
        "steps": existing.steps,
    });
    let request = BrainRequest {
        model: model.to_owned(),
        messages: vec![
            BrainMessage::text(
                "system",
                "You write reusable skills for a Windows computer-use agent. A skill is short, generic guidance for doing a kind of task again: which application, which screens, and which visible controls to use, in order. Write steps as instructions that quote visible labels (for example: open \"System\", then \"Display\"). Never include coordinates, target ids, file paths, recipients, message text, names of people, numbers from the screen, or anything personal. Use only the tools that appear in the workflow. For fast_actions steps keep the form `target_hint <quoted label>; done_when <condition quoting visible text>`, one step per subgoal in order, so the whole plan can be sent again as one fast_actions call. When updating an existing skill, keep what still works and improve it with what this run shows. Return JSON only: {\"title\":\"imperative title, at most 80 characters\",\"summary\":\"one sentence on when to use this skill\",\"steps\":[{\"tool\":\"tool name\",\"instruction\":\"what to do\"}]} with at most 12 steps.",
            ),
            BrainMessage::text(
                "user",
                format!(
                    "TASK THAT WAS COMPLETED:\n{}\n\n{} SKILL:\n{}\n\nVERIFIED WORKFLOW (sanitized):\n{}",
                    truncate_chars(prompt, 600),
                    if updating { "EXISTING" } else { "DRAFT" },
                    existing_text,
                    truncate_chars(&serde_json::to_string(workflow)?, 6_000),
                ),
            ),
        ],
        tools: Vec::new(),
        temperature: Some(0.0),
        max_tokens: None,
        seed: Some(42),
        reasoning_effort: None,
    };
    helper_reply(brain, request, reasoning, cancellation, |text| {
        parse_skill_text(text, allowed_tools)
    })
    .await
}

/// Reasoning for post-turn helper calls (skill writing, memory curation):
/// the user's chosen level first; the session's light level only when that
/// reply had no usable answer.
#[derive(Debug, Clone, Default)]
pub(super) struct HelperReasoning {
    pub(super) chosen: Option<String>,
    pub(super) fallback: Option<String>,
}

/// Output budget for a helper call. Its answer is short JSON; the rest is
/// room for a reasoning model's thinking at whatever level the user chose.
const HELPER_MAX_TOKENS: u32 = 16_000;

/// Run a helper request and validate its text with `accept`, retrying once
/// at the fallback reasoning level when the reply is unusable (for example a
/// reasoning model that ran out of room before answering).
async fn helper_reply<T>(
    brain: Arc<dyn Brain>,
    mut request: BrainRequest,
    reasoning: &HelperReasoning,
    cancellation: &CancellationToken,
    accept: impl Fn(&str) -> std::result::Result<T, String>,
) -> Result<std::result::Result<T, String>> {
    let mut attempts = vec![reasoning.chosen.clone()];
    if reasoning.fallback != reasoning.chosen {
        attempts.push(reasoning.fallback.clone());
    }
    let mut rejection = String::new();
    for effort in attempts {
        request.reasoning_effort = effort;
        request.max_tokens = Some(HELPER_MAX_TOKENS);
        let mut stream = brain.stream(request.clone());
        let mut text = String::new();
        let mut finish = None;
        loop {
            tokio::select! {
                () = cancellation.cancelled() => return Err(PokError::Cancelled),
                event = stream.next() => match event {
                    Some(event) => match event? {
                        BrainEvent::TextDelta { text: delta } => text.push_str(&delta),
                        BrainEvent::Finished { reason } => finish = reason,
                        _ => {}
                    },
                    None => break,
                }
            }
        }
        match accept(&text) {
            Ok(value) => return Ok(Ok(value)),
            Err(reason) => {
                rejection = format!(
                    "{reason} (reasoning {}, finish {}); reply began: {}",
                    request.reasoning_effort.as_deref().unwrap_or("default"),
                    finish.as_deref().unwrap_or("none"),
                    truncate_chars(text.trim(), 160)
                );
            }
        }
    }
    Ok(Err(rejection))
}

/// Accept an LLM-written skill only when it is well formed, uses tools from
/// the workflow, and carries nothing personal or run-specific.
pub(super) fn parse_skill_text(
    text: &str,
    allowed_tools: &BTreeSet<String>,
) -> std::result::Result<SkillText, String> {
    let object = extract_json_object(text).ok_or("no JSON object")?;
    let skill: SkillText =
        serde_json::from_str(object).map_err(|error| format!("invalid skill JSON: {error}"))?;
    let title = skill.title.trim();
    let summary = skill.summary.trim();
    if title.is_empty() || title.chars().count() > 80 {
        return Err("title empty or over 80 characters".into());
    }
    if summary.is_empty() || summary.chars().count() > 300 {
        return Err("summary empty or over 300 characters".into());
    }
    if skill.steps.is_empty() || skill.steps.len() > 12 {
        return Err("needs 1 to 12 steps".into());
    }
    if looks_private(title) || looks_private(summary) {
        return Err("title or summary has personal details".into());
    }
    for step in &skill.steps {
        if !allowed_tools.contains(&step.tool) {
            return Err(format!(
                "step uses tool {:?} not in the workflow",
                step.tool
            ));
        }
        if step.instruction.trim().is_empty() || step.instruction.chars().count() > 300 {
            return Err("step instruction empty or over 300 characters".into());
        }
        if looks_private(&step.instruction) {
            return Err("step has personal details".into());
        }
    }
    Ok(SkillText {
        title: title.to_owned(),
        summary: summary.to_owned(),
        steps: skill.steps,
    })
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
        .flat_map(|step| {
            if step.get("tool").and_then(Value::as_str) == Some("fast_actions") {
                return fast_actions_instructions(step)
                    .into_iter()
                    .map(|instruction| ProcedureStep {
                        tool: "fast_actions".into(),
                        instruction,
                    })
                    .collect::<Vec<_>>();
            }
            procedure_step(step).into_iter().collect()
        })
        .filter(|step| seen.insert((step.tool.clone(), step.instruction.clone())))
        .collect::<Vec<_>>();
    steps.truncate(12);
    steps
}

fn procedure_step(step: &Value) -> Option<ProcedureStep> {
    let tool = step.get("tool")?.as_str()?;
    let instruction = match tool {
        "observe_desktop" | "list_windows" => "Discover the current desktop and task applications.",
        "activate_window" => "Activate the current task application.",
        "open_application" => "Open the task application by name.",
        "browser_navigate" => "Navigate to `<CURRENT_DESTINATION>` in the current browser window.",
        "capture_screen" | "inspect_screen_region" | "query_screen_text" | "query_window_tree" => {
            "Read the current application and resolve fresh UIA/OCR targets."
        }
        "click_target" => "Click the freshly resolved `<CURRENT_UI_LABEL>` target.",
        "type_text" => "Type the complete current task text into a freshly verified control.",
        "scroll_view" => "Scroll the freshly observed task region.",
        "scroll_until_text" => "Scroll until the current task's `<TARGET_TEXT>` is visible.",
        "simulate_input" => "Apply current task input only to a freshly verified control.",
        "execute_action_batch" => {
            "Execute a freshly grounded action sequence using current task values."
        }
        "run_command" => "Run the verified command template and require a successful exit.",
        "managed_browser_open" => "Open `<CURRENT_DESTINATION>` in the managed browser.",
        "managed_browser_snapshot" => "Read the current managed-browser page.",
        "managed_browser_click" => {
            "Click the freshly resolved `<CURRENT_UI_LABEL>` element in the managed browser."
        }
        "managed_browser_type" => {
            "Type the current task text into a freshly resolved managed-browser field."
        }
        "managed_browser_select" => "Choose the current task value in a managed-browser list.",
        "managed_browser_scroll" => "Scroll the managed-browser page.",
        _ => return None,
    };
    Some(ProcedureStep {
        tool: tool.into(),
        instruction: instruction.into(),
    })
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
    reasoning: &HelperReasoning,
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
        max_tokens: None,
        seed: Some(42),
        reasoning_effort: None,
    };
    helper_reply(brain, request, reasoning, cancellation, |text| {
        let json = extract_json_object(text).ok_or("curator returned no JSON object")?;
        serde_json::from_str::<CurationEnvelope>(json)
            .map_err(|error| format!("invalid curation JSON: {error}"))
    })
    .await?
    .map_err(PokError::Provider)
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
        let model_curation_enabled = matches!(
            self.context.policy.mode,
            PolicyMode::Interactive | PolicyMode::Autonomous
        );
        // With model curation, skills are learned in the background with the
        // router and LLM review below; otherwise deterministically here.
        match if model_curation_enabled {
            Ok(Vec::new())
        } else {
            learn_verified_procedures(&self.context.memory, &prompt, &workflow, &qualification)
        } {
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
        let brain = self.brain.clone();
        let memory = self.context.memory.clone();
        let model = self.model.clone();
        let reasoning = HelperReasoning {
            chosen: self.reasoning_effort.clone(),
            fallback: self.bounded_reasoning_effort.clone(),
        };
        let artifact_dir = self.context.artifact_dir.clone();
        let decision_router = self.decision_router.clone();
        let decision_router_config = self.decision_router_config.clone();
        let observer = self.observer.clone();
        let curation_session_id = self.id;
        let task = tokio::spawn(async move {
            if tokio::time::timeout(std::time::Duration::from_secs(3), cancellation.cancelled())
                .await
                .is_ok()
            {
                // A follow-up arrived: skip the model calls, but keep what the
                // run proved as a plain skill (no router or LLM review).
                if model_curation_enabled
                    && let Ok(records) =
                        learn_verified_procedures(&memory, &prompt, &workflow, &qualification)
                {
                    for (record, created) in records {
                        append_trace(
                            &artifact_dir,
                            curation_session_id,
                            if created {
                                "procedure_learned"
                            } else {
                                "procedure_reinforced"
                            },
                            json!({"id": record.id, "title": record.title,
                                "success_count": record.success_count, "review": "skipped"}),
                        );
                    }
                }
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
            let skill = learn_skill_with_review(
                &memory,
                decision_router.clone(),
                decision_router_config.as_ref(),
                brain.clone(),
                &model,
                &reasoning,
                &prompt,
                &workflow,
                &qualification,
                &cancellation,
            )
            .await;
            append_trace(
                &artifact_dir,
                curation_session_id,
                match &skill {
                    Ok(Some(learned)) if learned.outcome == "created" => "procedure_learned",
                    Ok(Some(_)) => "procedure_reinforced",
                    Ok(None) => "procedure_not_learned",
                    Err(_) => "procedure_rejected",
                },
                match &skill {
                    Ok(Some(learned)) => serde_json::to_value(learned).unwrap_or_default(),
                    Ok(None) => json!({"reason": "no eligible verified workflow"}),
                    Err(error) => json!({"reason": error.to_string()}),
                },
            );
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
                &reasoning,
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
        let mut tasks = self.curation_tasks.lock();
        tasks.retain(|task| !task.is_finished());
        tasks.push(task);
    }
}
