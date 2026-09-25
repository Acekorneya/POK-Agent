//! Execution continuity: verified state after each action, the action ledger, and goal/commit tracking that keeps follow-ups on the newest verified state.

use super::*;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(super) struct VerifiedState {
    pub(super) observation_id: Option<String>,
    pub(super) app: Option<String>,
    pub(super) title: Option<String>,
    pub(super) observed_url: Option<String>,
    pub(super) url_provenance: Option<String>,
    pub(super) requested_destination: Option<String>,
    pub(super) page_status: String,
    pub(super) evidence: Vec<String>,
    pub(super) last_action: String,
    pub(super) outcome: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(super) struct ActionRecord {
    pub(super) attempts: u32,
    pub(super) failures: u32,
    pub(super) no_progress: u32,
    pub(super) last_outcome: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(super) struct ExecutionContinuity {
    pub(super) current: Option<VerifiedState>,
    pub(super) actions: BTreeMap<String, ActionRecord>,
    pub(super) actions_since_strategy_checkpoint: u32,
    pub(super) state_revision: u64,
    #[serde(default)]
    pub(super) unchanged_click_no_progress: u32,
    #[serde(default)]
    pub(super) grounding_required: bool,
}

#[derive(Debug)]
pub(super) enum PreCallDecision {
    Execute {
        signature: String,
        prior_attempts: u32,
    },
    Suppress {
        signature: String,
        outcome: &'static str,
        reason: String,
        repeat_count: u32,
    },
}

#[derive(Debug)]
pub(super) struct ContinuityFeedback {
    pub(super) outcome: String,
    pub(super) warning: Option<String>,
    pub(super) current_state: Option<VerifiedState>,
}

impl ExecutionContinuity {
    pub(super) fn reset(&mut self) {
        *self = Self::default();
    }

    pub(super) fn before_call(
        &self,
        name: &str,
        arguments: &Value,
        observation: Option<&crate::types::Observation>,
    ) -> PreCallDecision {
        let signature = action_signature(name, arguments, observation, self.state_revision);
        if !is_guarded_action(name) {
            return PreCallDecision::Execute {
                signature,
                prior_attempts: 0,
            };
        }

        if self.grounding_required
            && matches!(
                name,
                "click_target"
                    | "click_localized"
                    | "move_pointer"
                    | "drag_pointer"
                    | "drag_target"
                    | "simulate_input"
                    | "execute_action_batch"
            )
        {
            return PreCallDecision::Suppress {
                signature,
                outcome: "blocked_repeat",
                reason: "Further input is paused after repeated clicks without verified progress. Call locate_visual_target, inspect_screen_region, query_screen_text, or query_window_tree to re-ground from the current observation before clicking again.".into(),
                repeat_count: self.unchanged_click_no_progress,
            };
        }

        if name == "browser_navigate" {
            let requested = requested_destination(arguments);
            if requested.as_deref().is_some_and(|destination| {
                self.current
                    .as_ref()
                    .filter(|state| state.page_status == "loaded")
                    .and_then(|state| state.observed_url.as_deref())
                    .is_some_and(|current| destinations_match(current, destination))
            }) {
                return PreCallDecision::Suppress {
                    signature,
                    outcome: "already_current",
                    reason: "The browser is already at the requested destination. Continue from the current page instead of navigating again.".into(),
                    repeat_count: self
                        .actions
                        .get(&action_signature(
                            name,
                            arguments,
                            observation,
                            self.state_revision,
                        ))
                        .map_or(1, |record| record.attempts),
                };
            }
        }

        let Some(record) = self.actions.get(&signature) else {
            return PreCallDecision::Execute {
                signature,
                prior_attempts: 0,
            };
        };
        let repeated_navigation = name == "browser_navigate" && record.attempts >= 2;
        // Window activation already performs its own bounded native fallbacks.
        // Repeating the same exhausted activation against the same foreground
        // state only burns a model turn; allow it again after the state changes.
        let repeated_failure = record.failures >= if name == "activate_window" { 1 } else { 2 };
        let repeated_no_progress = record.no_progress >= 2;
        if repeated_navigation || repeated_failure || repeated_no_progress {
            let reason = format!(
                "Blocked {name}: this action has already been attempted {} times and its latest outcome was {}. Trust the current verified state and choose a different next action.",
                record.attempts, record.last_outcome
            );
            return PreCallDecision::Suppress {
                signature,
                outcome: "blocked_repeat",
                reason,
                repeat_count: record.attempts,
            };
        }
        PreCallDecision::Execute {
            signature,
            prior_attempts: record.attempts,
        }
    }

    pub(super) fn after_call(
        &mut self,
        name: &str,
        arguments: &Value,
        signature: &str,
        prior_attempts: u32,
        result: &Result<Value>,
    ) -> ContinuityFeedback {
        let previous = self.current.clone();
        let mut state = result
            .as_ref()
            .ok()
            .and_then(|value| verified_state(name, arguments, value, previous.as_ref()));
        let state_changed = state.as_ref().is_some_and(|current| {
            previous
                .as_ref()
                .is_none_or(|previous| state_identity(previous) != state_identity(current))
        });
        let explicit_change = result.as_ref().ok().is_some_and(result_reports_change);
        let page_error = state
            .as_ref()
            .is_some_and(|current| current.page_status == "error_page");
        let recovery_required = result.as_ref().ok().is_some_and(|value| {
            value.get("recovery_required").and_then(Value::as_bool) == Some(true)
                || value.get("status").and_then(Value::as_str) == Some("recovery_required")
        });
        let outcome = if result.is_err() || page_error || recovery_required {
            "failed"
        } else if !is_guarded_action(name) {
            "observed"
        } else if explicit_change || state_changed {
            "progress"
        } else {
            "no_progress"
        };

        if let Some(current) = &mut state {
            current.last_action = name.to_owned();
            current.outcome = outcome.to_owned();
            self.current = Some(current.clone());
        } else if let Some(current) = &mut self.current {
            current.last_action = name.to_owned();
            current.outcome = outcome.to_owned();
        }

        let mut warning = if is_guarded_action(name) {
            let record = self.actions.entry(signature.to_owned()).or_default();
            record.attempts = record.attempts.saturating_add(1);
            record.last_outcome = outcome.to_owned();
            if outcome == "failed" {
                record.failures = record.failures.saturating_add(1);
            } else if outcome == "no_progress" {
                record.failures = 0;
                record.no_progress = record.no_progress.saturating_add(1);
            } else {
                record.failures = 0;
                record.no_progress = 0;
            }
            (prior_attempts >= 1 && (outcome != "progress" || name == "browser_navigate")).then(|| {
                format!(
                    "Repeated-action warning: {name} has now been attempted {} times. Its latest outcome is {outcome}. Do not repeat it unchanged again; use the current verified state or change strategy.",
                    record.attempts
                )
            })
        } else {
            None
        };
        if matches!(
            name,
            "capture_screen"
                | "inspect_screen_region"
                | "locate_visual_target"
                | "query_screen_text"
                | "query_window_tree"
        ) && result.is_ok()
            || name == "activate_window"
                && result
                    .as_ref()
                    .is_ok_and(|value| value.get("capture").is_some())
        {
            self.unchanged_click_no_progress = 0;
            self.grounding_required = false;
        } else if matches!(
            name,
            "click_target"
                | "click_localized"
                | "move_pointer"
                | "drag_pointer"
                | "drag_target"
                | "simulate_input"
                | "execute_action_batch"
        ) {
            if matches!(outcome, "no_progress" | "failed") {
                self.unchanged_click_no_progress =
                    self.unchanged_click_no_progress.saturating_add(1);
                if self.unchanged_click_no_progress >= 2 {
                    self.grounding_required = true;
                    warning.get_or_insert_with(|| {
                        "Grounding required: repeated input on the unchanged UI made no verified progress. Inspect or query the current screen before any further click.".into()
                    });
                }
            } else if outcome == "progress" {
                self.unchanged_click_no_progress = 0;
                self.grounding_required = false;
            }
        }
        if is_guarded_action(name) {
            self.actions_since_strategy_checkpoint =
                self.actions_since_strategy_checkpoint.saturating_add(1);
            if self.actions_since_strategy_checkpoint >= 8 {
                self.actions_since_strategy_checkpoint = 0;
                warning.get_or_insert_with(|| {
                    "Strategy checkpoint: several desktop actions have been used on this task step. Summarize only newly verified evidence before acting again. If the same page, facts, or outcome are repeating, change method: query accessible text, use a safe read-only command/API, choose another source, or answer from the evidence already gathered.".into()
                });
            }
        }
        if outcome == "progress" {
            self.state_revision = self.state_revision.saturating_add(1);
        }

        ContinuityFeedback {
            outcome: outcome.to_owned(),
            warning,
            current_state: self.current.clone(),
        }
    }

    pub(super) fn suppressed_result(
        &self,
        name: &str,
        outcome: &str,
        reason: &str,
        signature: &str,
        repeat_count: u32,
        observation: Option<&crate::types::Observation>,
    ) -> Value {
        let current_targets = observation
            .map(|observation| {
                observation
                    .targets
                    .iter()
                    .filter(|target| target.selected == Some(true) || target.focused)
                    .take(8)
                    .map(|target| {
                        json!({
                            "target_id": target.id,
                            "name": target.name,
                            "role": target.control_type,
                            "selected": target.selected,
                            "focused": target.focused,
                            "bounds": target.bounds,
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        json!({
            "executed": false,
            "_pok_continuity": {
                "outcome": outcome,
                "action": name,
                "signature": signature,
                "repeat_count": repeat_count,
                "current_state": self.current,
                "current_targets": current_targets,
                "instruction": format!(
                    "{reason} Recovery is now required: capture or query the current UI, inspect selected/focused exact targets, then choose a different target or interaction method. The blocked signature must not be retried unchanged."
                ),
                "recovery_tools": ["locate_visual_target", "inspect_screen_region", "query_screen_text", "query_window_tree"],
            }
        })
    }
}

pub(super) fn enrich_continuity_result(mut value: Value, feedback: &ContinuityFeedback) -> Value {
    let continuity = json!({
        "outcome": feedback.outcome,
        "current_state": feedback.current_state,
        "warning": feedback.warning,
        "instruction": if feedback.warning.is_some() {
            "The verified state is newer than the model-authored plan. Advance from it and do not repeat the same action unchanged."
        } else {
            "Treat this current state as authoritative for the next action."
        },
    });
    if let Some(object) = value.as_object_mut() {
        object.insert("_pok_continuity".into(), continuity);
        value
    } else {
        json!({"result": value, "_pok_continuity": continuity})
    }
}

pub(super) fn is_continuity_tool(name: &str) -> bool {
    matches!(
        name,
        "activate_window"
            | "browser_navigate"
            | "capture_screen"
            | "inspect_screen_region"
            | "locate_visual_target"
            | "click_target"
            | "click_localized"
            | "move_pointer"
            | "drag_pointer"
            | "drag_target"
            | "execute_action_batch"
            | "observe_desktop"
            | "query_screen_text"
            | "query_window_tree"
            | "scroll_until_text"
            | "scroll_view"
            | "simulate_input"
            | "managed_browser_open"
            | "managed_browser_snapshot"
            | "managed_browser_click"
            | "managed_browser_type"
            | "managed_browser_select"
            | "managed_browser_hover"
            | "managed_browser_scroll"
    )
}

pub(super) fn is_guarded_action(name: &str) -> bool {
    matches!(
        name,
        "activate_window"
            | "browser_navigate"
            | "click_localized"
            | "click_target"
            | "drag_pointer"
            | "drag_target"
            | "execute_action_batch"
            | "move_pointer"
            | "scroll_until_text"
            | "scroll_view"
            | "simulate_input"
    )
}

pub(super) fn action_signature(
    name: &str,
    arguments: &Value,
    observation: Option<&crate::types::Observation>,
    state_revision: u64,
) -> String {
    let mut normalized = arguments.clone();
    if let Some(object) = normalized.as_object_mut() {
        object.remove("observation_id");
        object.remove("view_id");
        if matches!(
            name,
            "activate_window"
                | "click_localized"
                | "move_pointer"
                | "drag_pointer"
                | "drag_target"
                | "simulate_input"
                | "execute_action_batch"
        ) {
            if let Some(foreground) =
                observation.and_then(|observation| observation.foreground_window.as_ref())
            {
                object.insert(
                    "_foreground_state".into(),
                    json!({
                        "id": foreground.id,
                        "app": foreground.process_name,
                        "title": foreground.title,
                    }),
                );
            }
        }
        if name == "click_target" {
            let target_id = object.get("target_id").and_then(|value| {
                value
                    .as_str()
                    .map(str::to_owned)
                    .or_else(|| value.as_u64().map(|value| value.to_string()))
            });
            if let Some(label) = target_id.and_then(|target_id| {
                observation.and_then(|observation| {
                    observation
                        .targets
                        .iter()
                        .find(|target| target.id == target_id)
                        .map(|target| target.name.trim().to_ascii_lowercase())
                })
            }) {
                object.remove("target_id");
                object.insert("target_label".into(), Value::String(label));
            }
        }
    }
    let canonical = canonical_json(&normalized);
    let digest = Sha256::digest(format!("{state_revision}:{name}:{canonical}").as_bytes());
    let short = digest[..12]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("{name}:{short}")
}

pub(super) fn goal_action_key(
    root_request: &str,
    tool_name: &str,
    arguments: &Value,
    observation: Option<&crate::types::Observation>,
) -> Option<String> {
    let observation = observation?;
    let target = match tool_name {
        "click_target" => {
            let target_id = arguments.get("target_id").and_then(|value| {
                value
                    .as_str()
                    .map(str::to_owned)
                    .or_else(|| value.as_u64().map(|value| value.to_string()))
            })?;
            observation
                .targets
                .iter()
                .find(|target| target.id == target_id)
        }
        "drag_target" => {
            let target_id = arguments.get("source_target_id").and_then(Value::as_str)?;
            observation
                .targets
                .iter()
                .find(|target| target.id == target_id)
        }
        "click_localized" => None,
        "simulate_input" => {
            let kind = arguments.get("kind").and_then(Value::as_str)?;
            if !matches!(kind, "click" | "left_click") {
                return None;
            }
            let (x, y) = (
                arguments.get("x").and_then(Value::as_i64)?,
                arguments.get("y").and_then(Value::as_i64)?,
            );
            let (x, y) = (i32::try_from(x).ok()?, i32::try_from(y).ok()?);
            observation
                .targets
                .iter()
                .filter(|target| target.actionable && target.bounds.contains(x, y))
                .min_by_key(|target| {
                    u64::from(target.bounds.width) * u64::from(target.bounds.height)
                })
        }
        "execute_action_batch" => arguments
            .get("steps")
            .and_then(Value::as_array)?
            .iter()
            .rev()
            .filter(|step| {
                matches!(
                    step.get("kind").and_then(Value::as_str),
                    Some("click_target" | "click" | "left_click")
                )
            })
            .filter_map(|step| {
                let target_id = step.get("target_id").and_then(|value| {
                    value
                        .as_str()
                        .map(str::to_owned)
                        .or_else(|| value.as_u64().map(|value| value.to_string()))
                })?;
                observation
                    .targets
                    .iter()
                    .find(|target| target.id == target_id)
            })
            .find(|target| {
                requested_commit_label(root_request, &target.name.to_ascii_lowercase()).is_some()
            }),
        _ => return None,
    }?;
    let action = requested_commit_label(root_request, &target.name.to_ascii_lowercase())?;
    // Native dialog handles change every time a workflow is reopened. Scope a
    // terminal action to its stable task request and parent workflow title, not
    // to a one-off dialog handle, so a second Print dialog cannot bypass the
    // duplicate-action guard.
    let workflow = observation
        .foreground_window
        .as_ref()
        .map(|window| format!("{}|{}", window.process_name, window.title))
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    let request_digest = Sha256::digest(root_request.trim().as_bytes());
    Some(format!(
        "{action}|{workflow}|{:02x}{:02x}{:02x}{:02x}",
        request_digest[0], request_digest[1], request_digest[2], request_digest[3]
    ))
}

pub(super) fn goal_action_context_key(
    action: &str,
    window_id: &str,
    label: &str,
    control_type: &str,
    target: Option<(&crate::types::Rect, &str)>,
) -> String {
    let target = target.map_or_else(String::new, |(bounds, id)| {
        format!(
            "{id}:{}:{}:{}:{}",
            bounds.x, bounds.y, bounds.width, bounds.height
        )
    });
    format!(
        "{}|{}|{}|{}|{}",
        action,
        window_id.trim().to_ascii_lowercase(),
        label.trim().to_ascii_lowercase(),
        control_type.trim().to_ascii_lowercase(),
        target,
    )
}

pub(super) fn verified_terminal_goal_action(
    root_request: &str,
    tool_name: &str,
    result: &Value,
) -> Option<(String, Value)> {
    if tool_name == "execute_action_batch" {
        return result
            .get("step_results")
            .and_then(Value::as_array)?
            .iter()
            .rev()
            .filter(|step| step.get("status").and_then(Value::as_str) == Some("ok"))
            .find_map(|step| {
                verified_terminal_goal_action(root_request, "click_target", step.get("result")?)
            });
    }
    if !matches!(tool_name, "click_target" | "simulate_input")
        || result.get("executed").and_then(Value::as_bool) == Some(false)
    {
        return None;
    }
    let label = result
        .pointer("/action/label")
        .and_then(Value::as_str)
        .or_else(|| {
            result
                .pointer("/action_context/label")
                .and_then(Value::as_str)
        })?
        .trim()
        .to_ascii_lowercase();
    let key = requested_commit_label(root_request, &label)?;
    let removed = result
        .pointer("/state_change/removed_control_count")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let focus_changed = result
        .pointer("/state_change/focus_changed")
        .and_then(Value::as_bool)
        == Some(true);
    let success_text = result
        .pointer("/state_change/added_text")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .find(|text| {
            let lower = text.to_ascii_lowercase();
            [
                "complete",
                "completed",
                "printed",
                "printing",
                "recorded",
                "saved",
                "sent",
                "submitted",
                "success",
                "uploaded",
            ]
            .iter()
            .any(|marker| lower.contains(marker))
        })
        .map(str::to_owned);
    let still_has_commit_control =
        result
            .get("targets")
            .and_then(Value::as_array)
            .is_some_and(|targets| {
                targets.iter().any(|target| {
                    target
                        .get("label")
                        .and_then(Value::as_str)
                        .and_then(|label| {
                            requested_commit_label(root_request, &label.to_ascii_lowercase())
                        })
                        .as_deref()
                        == Some(key.as_str())
                })
            });
    // A first-stage Print control commonly opens a second Print dialog. It is
    // only a terminal submission when that post-action state has no further
    // same-purpose commit control. Explicit success text upgrades it to a
    // confirmed result; otherwise a closed/disappeared control is submitted.
    let status = if success_text.is_some() {
        "confirmed"
    } else if (removed > 0 || focus_changed) && !still_has_commit_control {
        "submitted"
    } else {
        return None;
    };
    let context = result.get("action_context")?;
    let window_id = context.get("window_id").and_then(Value::as_str)?;
    let control_type = context
        .get("control_type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let target_id = context
        .get("target_id")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let bounds = context.get("bounds").and_then(|bounds| {
        Some(crate::types::Rect {
            x: i32::try_from(bounds.get("x")?.as_i64()?).ok()?,
            y: i32::try_from(bounds.get("y")?.as_i64()?).ok()?,
            width: u32::try_from(bounds.get("width")?.as_u64()?).ok()?,
            height: u32::try_from(bounds.get("height")?.as_u64()?).ok()?,
        })
    });
    let key = goal_action_context_key(
        &key,
        window_id,
        &label,
        control_type,
        bounds.as_ref().map(|bounds| (bounds, target_id)),
    );
    Some((
        key,
        json!({
            "status": status,
            "removed_control_count": removed,
            "focus_changed": focus_changed,
            "success_text": success_text,
        }),
    ))
}

pub(super) fn canonical_json(value: &Value) -> String {
    match value {
        Value::Object(object) => {
            let mut entries = object.iter().collect::<Vec<_>>();
            entries.sort_by(|left, right| left.0.cmp(right.0));
            let body = entries
                .into_iter()
                .map(|(key, value)| {
                    format!(
                        "{}:{}",
                        serde_json::to_string(key).unwrap_or_default(),
                        canonical_json(value)
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            format!("{{{body}}}")
        }
        Value::Array(values) => {
            let body = values
                .iter()
                .map(canonical_json)
                .collect::<Vec<_>>()
                .join(",");
            format!("[{body}]")
        }
        _ => value.to_string(),
    }
}

pub(super) fn requested_destination(arguments: &Value) -> Option<String> {
    ["query_or_url", "url", "query", "destination"]
        .into_iter()
        .find_map(|key| arguments.get(key).and_then(Value::as_str))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

pub(super) fn destinations_match(current: &str, requested: &str) -> bool {
    fn normalize(value: &str) -> String {
        value
            .trim()
            .trim_end_matches('/')
            .trim_start_matches("https://")
            .trim_start_matches("http://")
            .trim_start_matches("www.")
            .to_ascii_lowercase()
    }
    normalize(current) == normalize(requested)
}

pub(super) fn verified_state(
    name: &str,
    arguments: &Value,
    result: &Value,
    previous: Option<&VerifiedState>,
) -> Option<VerifiedState> {
    let source = state_source(result)?;
    let mut state = previous.cloned().unwrap_or_default();
    state.observation_id = source
        .get("observation_id")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or(state.observation_id);
    state.app = source
        .pointer("/focus/app")
        .or_else(|| source.pointer("/target/app"))
        .or_else(|| source.get("process_name"))
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or(state.app);
    state.title = source
        .pointer("/focus/title")
        .or_else(|| source.pointer("/target/title"))
        .or_else(|| source.get("title"))
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or(state.title);
    state.requested_destination = requested_destination(arguments).or(state.requested_destination);
    let address_bar_url = source
        .get("observed_url")
        .and_then(Value::as_str)
        .filter(|text| looks_like_url(text))
        .map(str::to_owned)
        .or_else(|| {
            source
                .get("targets")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter(|target| {
                    target
                        .get("role")
                        .and_then(Value::as_str)
                        .is_some_and(|role| role.eq_ignore_ascii_case("edit"))
                })
                .filter_map(|target| {
                    target
                        .get("label")
                        .or_else(|| target.get("name"))
                        .and_then(Value::as_str)
                })
                .map(str::trim)
                .find(|text| looks_like_url(text))
                .map(str::to_owned)
        });

    let added_text = source
        .pointer("/state_change/added_text")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .take(8)
        .map(str::to_owned)
        .collect::<Vec<_>>();
    if !added_text.is_empty() {
        state.evidence = added_text;
    } else {
        let fresh_evidence = state
            .title
            .iter()
            .chain(address_bar_url.iter())
            .cloned()
            .collect::<Vec<_>>();
        if !fresh_evidence.is_empty() {
            state.evidence = fresh_evidence;
        }
    }
    state.evidence.truncate(8);

    let status_text = state
        .title
        .iter()
        .chain(
            state
                .title
                .is_none()
                .then_some(state.evidence.iter())
                .into_iter()
                .flatten(),
        )
        .map(|text| text.to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(" ");
    let reported_page_status = source
        .get("page_status")
        .and_then(Value::as_str)
        .filter(|status| matches!(*status, "loaded" | "loading" | "error_page"));
    state.page_status = if let Some(status) = reported_page_status {
        if address_bar_url.is_some() {
            state.observed_url = address_bar_url.clone();
            state.url_provenance = address_bar_url.as_ref().map(|_| "uia_value".into());
        }
        status.into()
    } else if [
        "404",
        "not found",
        "page unavailable",
        "site can't be reached",
        "site cannot be reached",
    ]
    .iter()
    .any(|pattern| status_text.contains(pattern))
    {
        "error_page".into()
    } else {
        let refreshes_browser_location = matches!(
            name,
            "browser_navigate" | "capture_screen" | "query_ui_tree" | "query_text"
        );
        if address_bar_url.is_some() || refreshes_browser_location {
            state.observed_url = address_bar_url.clone();
            state.url_provenance = address_bar_url.as_ref().map(|_| "uia_value".into());
        }
        if state.title.is_some() || state.observed_url.is_some() {
            "loaded".into()
        } else {
            "uncertain".into()
        }
    };
    if state.page_status == "error_page" {
        // A failed navigation is not evidence that the requested destination
        // became the browser's current URL.
        state.observed_url = address_bar_url;
        state.url_provenance = state.observed_url.as_ref().map(|_| "uia_value".into());
    }
    state.last_action = name.to_owned();
    Some(state)
}

pub(super) fn state_source(value: &Value) -> Option<&Value> {
    if value.get("observation_id").is_some()
        || value.get("focus").is_some()
        || value.get("target").is_some()
        || value.get("state_change").is_some()
        || value.get("process_name").is_some()
    {
        return Some(value);
    }
    if let Some(steps) = value.get("step_results").and_then(Value::as_array) {
        return steps
            .iter()
            .rev()
            .find_map(|step| step.get("result").and_then(state_source));
    }
    None
}

pub(super) fn looks_like_url(text: &str) -> bool {
    let lower = text.trim().to_ascii_lowercase();
    lower.starts_with("http://")
        || lower.starts_with("https://")
        || lower.starts_with("www.")
        || (lower.contains('.') && !lower.contains(' ') && lower.len() <= 2_048)
}

pub(super) fn state_identity(state: &VerifiedState) -> String {
    canonical_json(&json!({
        "app": state.app,
        "title": state.title,
        "observed_url": state.observed_url,
        "page_status": state.page_status,
    }))
}

pub(super) fn result_reports_change(value: &Value) -> bool {
    let Some(source) = state_source(value) else {
        return false;
    };
    if let Some(effect) = source
        .pointer("/verification/effect")
        .and_then(Value::as_bool)
    {
        return effect;
    }
    source
        .pointer("/state_change/added_text")
        .and_then(Value::as_array)
        .is_some_and(|items| !items.is_empty())
        || source
            .pointer("/state_change/removed_control_count")
            .and_then(Value::as_u64)
            .is_some_and(|count| count > 0)
        || source
            .pointer("/state_change/focus_changed")
            .and_then(Value::as_bool)
            == Some(true)
        || source
            .pointer("/state_change/selected_changed")
            .and_then(Value::as_array)
            .is_some_and(|items| !items.is_empty())
        || source
            .pointer("/state_change/focused_control_changed")
            .and_then(Value::as_bool)
            == Some(true)
        || source
            .pointer("/attempts")
            .and_then(Value::as_array)
            .is_some_and(|attempts| {
                attempts.iter().any(|attempt| {
                    attempt.get("viewport_changed").and_then(Value::as_bool) == Some(true)
                })
            })
}
