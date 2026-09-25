//! The shared input pipeline: fresh-observation authority, grounding reconciliation, duplicate-submission ledger, effect verification, waiting for the user to be idle, and cursor-free UI Automation patterns.

use super::*;

pub(super) fn current_observation(
    context: &ToolContext,
    observation_id: Option<&str>,
    tool_name: &str,
) -> Result<Observation> {
    let observation = context.latest_observation.lock().clone().ok_or_else(|| {
        let hint = if tool_name == "click_target" {
            " Window ids from list_windows are not target ids; use activate_window, or browser_navigate for a URL/search."
        } else {
            ""
        };
        PokError::Tool(format!(
            "capture_screen must be called before {tool_name}.{hint}"
        ))
    })?;
    if let Some(supplied) = observation_id {
        let current_short = observation_id_for(&observation);
        let current_uuid = observation.version.to_string();
        if supplied != current_short && supplied != current_uuid {
            return Err(PokError::Tool(format!(
                "stale observation {supplied:?}; use latest {current_short:?} or omit observation_id"
            )));
        }
    }
    Ok(observation)
}

pub(super) fn current_observation_or_view(
    context: &ToolContext,
    observation_id: Option<&str>,
    view_id: Option<&str>,
    tool_name: &str,
) -> Result<Observation> {
    let source = current_observation(context, observation_id, tool_name)?;
    // A view_id that is really the source observation's own handle names no
    // derived view; the observation itself is what the caller means.
    let Some(view_id) = view_id.filter(|view_id| *view_id != observation_id_for(&source)) else {
        return Ok(source);
    };
    let view = context
        .latest_observation_view
        .lock()
        .clone()
        .ok_or_else(|| {
            PokError::Tool("no current derived view; inspect the source again".into())
        })?;
    if view.id != view_id || view.source_observation_id != observation_id_for(&source) {
        return Err(PokError::Tool(format!(
            "stale derived view {view_id:?}; use latest {:?} or inspect the source again",
            view.id
        )));
    }
    Ok(view.observation)
}

pub(super) async fn observe_foreground_window(
    context: &ToolContext,
    window: &WindowInfo,
) -> Result<Observation> {
    let request = CaptureRequest {
        scope: CaptureScope::Window,
        window_id: Some(window.id.clone()),
        monitor_id: None,
        region: None,
        max_edge: context.vision_max_edge,
    };
    let mut observation = context
        .platform
        .observe(
            &request,
            true,
            true,
            context.uia_element_limit,
            Duration::from_millis(context.desktop_enrichment_timeout_ms),
        )
        .await?;
    observation.targets = build_targets(
        &observation,
        context.model_target_limit,
        context.fusion_iou_threshold,
        context.ocr_containment_threshold,
        &context.task_hint.lock(),
    );
    Ok(observation)
}

pub(super) fn action_target<'a>(
    observation: &'a Observation,
    action: &InputAction,
    model_action: &Value,
) -> Option<&'a InteractionTarget> {
    let target_id = model_action.get("target_id").and_then(Value::as_str);
    target_id
        .and_then(|id| observation.targets.iter().find(|target| target.id == id))
        .or_else(|| match action {
            InputAction::Click { x, y, .. } => observation
                .targets
                .iter()
                .filter(|target| target.actionable && target.bounds.contains(*x, *y))
                .min_by_key(|target| {
                    u64::from(target.bounds.width) * u64::from(target.bounds.height)
                }),
            _ => None,
        })
}

pub(super) fn unique_rebased_target<'a>(
    previous: &InteractionTarget,
    refreshed: &'a Observation,
) -> Option<&'a InteractionTarget> {
    let label = normalized_text(&previous.name);
    let control_type = normalized_text(&previous.control_type);
    let matches = refreshed
        .targets
        .iter()
        .filter(|target| {
            target.actionable
                && normalized_text(&target.name) == label
                && normalized_text(&target.control_type) == control_type
        })
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [only] => Some(*only),
        _ => None,
    }
}

pub(super) enum InputGrounding {
    Reused,
    Rebased {
        observation: Observation,
        action: InputAction,
        previous_target_id: String,
        target_id: String,
    },
    NewTargetRequired {
        observation: Observation,
        reason: String,
    },
}

pub(super) async fn reconcile_input_grounding(
    context: &ToolContext,
    observation: &Observation,
    action: &InputAction,
    model_action: &Value,
) -> Result<InputGrounding> {
    let expected = observation
        .foreground_window
        .as_ref()
        .ok_or_else(|| PokError::Tool("input requires an observed foreground window".into()))?;
    let current = context.platform.foreground_window().await?;
    if foreground_matches(&expected.id, current.as_ref()) {
        return Ok(InputGrounding::Reused);
    }
    let current = current.ok_or_else(|| {
        PokError::Tool("desktop foreground became unavailable before input; observe again".into())
    })?;
    let refreshed = observe_foreground_window(context, &current).await?;
    let related_process = expected
        .process_name
        .eq_ignore_ascii_case(&current.process_name);
    if related_process
        && let Some(previous) = action_target(observation, action, model_action)
        && let Some(rebased) = unique_rebased_target(previous, &refreshed)
    {
        let (x, y) = rebased.click_point.unwrap_or((
            rebased.bounds.x + i32::try_from(rebased.bounds.width / 2).unwrap_or(i32::MAX),
            rebased.bounds.y + i32::try_from(rebased.bounds.height / 2).unwrap_or(i32::MAX),
        ));
        if rebased.bounds.contains(x, y) {
            let target_id = rebased.id.clone();
            return Ok(InputGrounding::Rebased {
                observation: refreshed,
                action: InputAction::Click {
                    x,
                    y,
                    button: match action {
                        InputAction::Click { button, .. } => *button,
                        _ => MouseButton::Left,
                    },
                },
                previous_target_id: previous.id.clone(),
                target_id,
            });
        }
    }
    Ok(InputGrounding::NewTargetRequired {
        observation: refreshed,
        reason: if related_process {
            "the application opened a new dialog whose next target is not uniquely determined"
        } else {
            "foreground changed to a different application"
        }
        .into(),
    })
}

pub(super) fn ensure_targeted_capture(observation: &Observation) -> Result<()> {
    let target = observation
        .target
        .as_ref()
        .ok_or_else(|| PokError::Tool("observation has no capture target".into()))?;
    if matches!(target.scope, CaptureScope::All) {
        return Err(PokError::Tool(
            "input is blocked for an all-screen overview; capture one monitor or window first"
                .into(),
        ));
    }
    Ok(())
}

pub(super) fn observation_id(version: Uuid) -> String {
    format!("obs_{}", &version.to_string()[..8])
}

/// The observation id input tools accept for `observation`; decision
/// candidates must use this exact form or every click fails as stale.
pub(crate) fn observation_id_for(observation: &Observation) -> String {
    observation_id(observation.version)
}

pub(super) fn destination(observation: &Observation) -> (String, String) {
    observation.foreground_window.as_ref().map_or_else(
        || ("unknown".into(), "Unknown destination".into()),
        |window| {
            let label = format!("{} — {}", window.title, window.process_name);
            (
                format!(
                    "{}|{}",
                    window.process_name.to_ascii_lowercase(),
                    window.title.to_ascii_lowercase()
                ),
                label,
            )
        },
    )
}

/// The destination of typed text: the window plus, when known, the field.
/// Different fields of one form are different destinations, while a chat's
/// message box stays the same destination between attempts.
pub(super) fn field_destination(
    observation: &Observation,
    field: Option<&InteractionTarget>,
) -> (String, String) {
    let (window, label) = destination(observation);
    let Some(field) = field.or_else(|| observation.targets.iter().find(|target| target.focused))
    else {
        return (window, label);
    };
    let key = if field.name.trim().is_empty() {
        format!(
            "{}@{},{}",
            field.control_type, field.bounds.x, field.bounds.y
        )
    } else {
        format!("{}:{}", field.control_type, field.name.trim())
    };
    (format!("{window}|{}", key.to_ascii_lowercase()), label)
}

pub(super) fn normalized_text(value: &str) -> String {
    value
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
}

pub(super) fn submission_evidence(observation: &Observation, text: &str) -> Option<String> {
    let expected = normalized_text(text);
    if expected.len() < 3 {
        return None;
    }
    if let Some(candidate) = observation
        .ui_elements
        .iter()
        .filter(|element| !element.control_type.eq_ignore_ascii_case("edit"))
        .map(|element| element.name.trim())
        .find(|candidate| normalized_text(candidate).contains(&expected))
    {
        return Some(candidate.chars().take(300).collect());
    }
    let editors = observation
        .ui_elements
        .iter()
        .filter(|element| element.control_type.eq_ignore_ascii_case("edit"))
        .map(|element| &element.bounds)
        .collect::<Vec<_>>();
    observation
        .ocr
        .iter()
        .filter(|block| !editors.iter().any(|editor| overlaps(&block.bounds, editor)))
        .map(|block| block.text.trim())
        .find(|candidate| normalized_text(candidate).contains(&expected))
        .map(|candidate| candidate.chars().take(300).collect())
}

pub(super) fn verified_submission_value(record: &VerifiedSubmission, status: &str) -> Value {
    json!({
        "status": status,
        "destination": record.destination_label,
        "evidence": record.evidence,
        "instruction": "The submission is already verified. Do not send it again.",
    })
}

pub(super) fn duplicate_submission_value(
    ledger: &InputLedger,
    destination: &str,
    text: &str,
    replace: bool,
) -> Option<Value> {
    let normalized = normalized_text(text);
    if let Some(record) = ledger
        .verified
        .iter()
        .find(|record| record.destination == destination && record.normalized_text == normalized)
    {
        return Some(verified_submission_value(record, "suppressed_duplicate"));
    }
    ledger
        .pending
        .as_ref()
        .filter(|pending| {
            pending.destination == destination
                && normalized_text(&pending.text) == normalized
                // Replacing an unsent draft with the same text cannot
                // duplicate it; retrying that is how a field the application
                // did not register gets filled again.
                && (pending.submitted || !replace)
        })
        .map(|pending| {
            json!({
                "status": if pending.submitted { "verification_required" } else { "draft_already_typed" },
                "destination": pending.destination_label,
                "instruction": if pending.submitted {
                    "Capture the window to verify the prior submission before retrying."
                } else {
                    "The same draft was already typed; submit or inspect it instead of typing it again."
                },
            })
        })
}

pub(super) fn reconcile_pending_submission(
    observation: &Observation,
    context: &ToolContext,
) -> Option<Value> {
    let pending = context.input_ledger.lock().pending.clone()?;
    if !pending.submitted {
        return None;
    }
    if let Some(evidence) = submission_evidence(observation, &pending.text) {
        let record = VerifiedSubmission {
            normalized_text: normalized_text(&pending.text),
            destination: pending.destination,
            destination_label: pending.destination_label,
            evidence,
        };
        let value = verified_submission_value(&record, "verified");
        let mut ledger = context.input_ledger.lock();
        if !ledger.verified.iter().any(|existing| {
            existing.destination == record.destination
                && existing.normalized_text == record.normalized_text
        }) {
            ledger.verified.push(record);
        }
        ledger.pending = None;
        Some(value)
    } else {
        context.input_ledger.lock().pending = None;
        Some(json!({
            "status": "not_found_after_capture",
            "destination": pending.destination_label,
            "instruction": "The submitted text was not found after a fresh observation; refocus the editor before retrying.",
        }))
    }
}

pub(super) fn state_change_value(
    before: &Observation,
    after: &Observation,
    task_hint: &str,
) -> Value {
    let previous = before
        .ui_elements
        .iter()
        .map(|element| normalized_text(&element.name))
        .filter(|name| !name.is_empty())
        .collect::<HashSet<_>>();
    let current = after
        .ui_elements
        .iter()
        .map(|element| normalized_text(&element.name))
        .filter(|name| !name.is_empty())
        .collect::<HashSet<_>>();
    let terms = ranking_terms(task_hint);
    let mut added = after
        .ui_elements
        .iter()
        .filter_map(|element| {
            let normalized = normalized_text(&element.name);
            (!normalized.is_empty() && !previous.contains(&normalized)).then(|| {
                let matches = terms
                    .iter()
                    .filter(|term| normalized.contains(term.as_str()))
                    .count();
                (
                    matches,
                    element.name.trim().chars().take(220).collect::<String>(),
                )
            })
        })
        .collect::<Vec<_>>();
    added.sort_by_key(|(matches, text)| (std::cmp::Reverse(*matches), text.len()));
    added.dedup_by(|left, right| normalized_text(&left.1) == normalized_text(&right.1));
    added.truncate(8);
    let removed_count = previous.difference(&current).count();
    let focus_changed = before
        .foreground_window
        .as_ref()
        .map(|window| (&window.id, &window.title, &window.process_name))
        != after
            .foreground_window
            .as_ref()
            .map(|window| (&window.id, &window.title, &window.process_name));
    let selected_key = |target: &crate::types::InteractionTarget| {
        format!(
            "{}|{}|{}:{}:{}:{}",
            normalized_text(&target.name),
            normalized_text(&target.control_type),
            target.bounds.x,
            target.bounds.y,
            target.bounds.width,
            target.bounds.height
        )
    };
    let previous_selected = before
        .targets
        .iter()
        .filter(|target| {
            target.selected == Some(true)
                && matches!(target.source, TargetSource::Uia | TargetSource::UiaOcr)
        })
        .map(&selected_key)
        .collect::<HashSet<_>>();
    let mut selected_changed = after
        .targets
        .iter()
        .filter(|target| {
            target.selected == Some(true)
                && matches!(target.source, TargetSource::Uia | TargetSource::UiaOcr)
        })
        .filter(|target| !previous_selected.contains(&selected_key(target)))
        .map(|target| target.name.trim())
        .filter(|name| !name.is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    selected_changed.sort();
    selected_changed.dedup();
    selected_changed.truncate(8);
    let focused_key = |observation: &Observation| {
        observation
            .targets
            .iter()
            .find(|target| target.focused)
            .map(selected_key)
    };
    let focused_control_changed = focused_key(before) != focused_key(after);
    let visual_change_ratio = visual_change_ratio(before, after).unwrap_or(0.0);
    json!({
        "added_text": added.into_iter().map(|(_, text)| text).collect::<Vec<_>>(),
        "removed_control_count": removed_count,
        "focus_changed": focus_changed,
        "selected_changed": selected_changed,
        "focused_control_changed": focused_control_changed,
        "visual_change_ratio": visual_change_ratio,
    })
}

pub(super) fn visual_change_ratio(before: &Observation, after: &Observation) -> Option<f64> {
    let before = before.screenshots.first()?;
    let after = after.screenshots.first()?;
    let decode = |encoded: &str| {
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .ok()?;
        let image = image::load_from_memory(&bytes).ok()?.into_luma8();
        Some(image::imageops::resize(
            &image,
            64,
            64,
            image::imageops::FilterType::Triangle,
        ))
    };
    let before = decode(&before.png_base64)?;
    let after = decode(&after.png_base64)?;
    let changed = before
        .pixels()
        .zip(after.pixels())
        .filter(|(left, right)| left.0[0].abs_diff(right.0[0]) >= 24)
        .count();
    Some(changed as f64 / (64.0 * 64.0))
}

pub(super) fn action_verified_effect(
    action: &InputAction,
    state_change: &Value,
    typing_verification: &TextVerification,
) -> (bool, &'static str) {
    let visual = state_change
        .get("visual_change_ratio")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let structural = state_change
        .get("added_text")
        .and_then(Value::as_array)
        .is_some_and(|items| !items.is_empty())
        || state_change
            .get("removed_control_count")
            .and_then(Value::as_u64)
            .is_some_and(|count| count > 0)
        || state_change.get("focus_changed").and_then(Value::as_bool) == Some(true)
        || state_change
            .get("selected_changed")
            .and_then(Value::as_array)
            .is_some_and(|items| !items.is_empty())
        || state_change
            .get("focused_control_changed")
            .and_then(Value::as_bool)
            == Some(true);
    match action {
        InputAction::Move { .. } => (false, "cursor_movement_only"),
        InputAction::Scroll { .. } => {
            let effect = structural || visual >= 0.02;
            (
                effect,
                if effect {
                    "viewport_changed"
                } else {
                    "no_stable_change"
                },
            )
        }
        InputAction::Drag { .. } => {
            let effect = structural || visual >= 0.005;
            (
                effect,
                if effect {
                    "drag_changed_ui"
                } else {
                    "no_stable_change"
                },
            )
        }
        InputAction::TypeText { .. } => {
            let effect = matches!(typing_verification, TextVerification::Verified { .. });
            (
                effect,
                if effect {
                    "typed_text_verified"
                } else {
                    "typing_unverified"
                },
            )
        }
        InputAction::Click { .. } | InputAction::DoubleClick { .. } | InputAction::Key { .. } => {
            let effect = structural || visual >= 0.01;
            (
                effect,
                if effect {
                    "stable_ui_change"
                } else {
                    "no_stable_change"
                },
            )
        }
    }
}

pub(super) fn model_action_value(args: &InputArgs) -> Value {
    match args.kind.as_str() {
        "move" | "mouse_move" | "click" | "left_click" | "right_click" | "middle_click" => json!({
            "kind": if matches!(args.kind.as_str(), "move" | "mouse_move") { "move" } else { "click" },
            "model_point": [args.x, args.y],
            "button": match args.kind.as_str() {
                "right_click" => Some(MouseButton::Right),
                "middle_click" => Some(MouseButton::Middle),
                _ => args.button,
            },
            "coordinate_space": "model_image_pixels",
            "mapping_succeeded": true,
        }),
        "text" | "type" | "type_text" => json!({
            "kind": "type_text",
            "text": args.text,
            "replace_existing": args.replace_existing,
        }),
        "keyboard" | "key" | "key_press" | "shortcut" | "keyboard_shortcut" => {
            json!({"kind": "key", "key": args.key})
        }
        "scroll" | "mouse_scroll_down" | "scroll_down" | "mouse_scroll_up" | "scroll_up" => {
            let (delta_x, delta_y) = legacy_scroll_deltas(args);
            json!({
                "kind": "scroll",
                "delta_x": delta_x,
                "delta_y": delta_y,
            })
        }
        other => json!({"kind": other}),
    }
}

pub(super) async fn execute_input(
    action: InputAction,
    observation: Observation,
    context: &ToolContext,
    model_action: Value,
) -> Result<Value> {
    *context.pending_visual_localization.lock() = None;
    let mut action = action;
    let mut observation = observation;
    let mut model_action = model_action;
    let mut grounding = json!({"status": "reused", "automatic": false});
    match reconcile_input_grounding(context, &observation, &action, &model_action).await? {
        InputGrounding::Reused => {}
        InputGrounding::Rebased {
            observation: refreshed,
            action: rebound_action,
            previous_target_id,
            target_id,
        } => {
            observation = refreshed;
            action = rebound_action;
            if let Some(action) = model_action.as_object_mut() {
                action.insert("target_id".into(), json!(target_id));
            }
            grounding = json!({
                "status": "rebound_after_foreground_transition",
                "automatic": true,
                "previous_target_id": previous_target_id,
                "target_id": target_id,
                "instruction": "The application opened a related dialog. The equivalent uniquely matched target was safely rebound."
            });
        }
        InputGrounding::NewTargetRequired {
            observation: refreshed,
            reason,
        } => {
            *context.latest_observation.lock() = Some(refreshed.clone());
            *context.latest_observation_view.lock() = None;
            *context.pending_visual_localization.lock() = None;
            let mut result = model_observation_value(&refreshed, context.annotate_targets)?;
            result["executed"] = json!(false);
            result["action"] = model_action;
            result["grounding"] = json!({
                "status": "new_target_required",
                "automatic": true,
                "reason": reason,
                "instruction": "This fresh foreground state is authoritative. Choose a listed target; do not retry the prior action unchanged."
            });
            result["note"] = json!(
                "A foreground dialog appeared before input. No physical action was sent to the old screen."
            );
            return Ok(result);
        }
    }
    // Never join or start a call the user did not ask for: a voice channel
    // or call button connects their microphone or camera.
    if matches!(
        action,
        InputAction::Click { .. } | InputAction::DoubleClick { .. }
    ) {
        let clicked = input_action_context(&observation, &action, &model_action);
        if let Some(label) = clicked.get("label").and_then(Value::as_str)
            && crate::policy::is_call_control(label)
            && !crate::policy::request_allows_calls(&context.active_task.lock().root_request)
        {
            return Err(PokError::PolicyDenied {
                tool: "simulate_input".into(),
                reason: format!(
                    "{label:?} joins or starts a voice or video call, which the user did not ask for; read who is in it from the visible channel list instead"
                ),
            });
        }
    }
    let before = observation.clone();
    let action_context = input_action_context(&observation, &action, &model_action);
    let typed_field = model_action
        .get("target_id")
        .and_then(Value::as_str)
        .and_then(|id| observation.targets.iter().find(|target| target.id == id));
    let (destination, destination_label) = field_destination(&observation, typed_field);
    if let InputAction::TypeText {
        text,
        replace_existing,
    } = &action
    {
        validate_browser_navigation(
            text,
            &context.task_hint.lock(),
            context.focused_control.lock().as_ref(),
        )?;
        let ledger = context.input_ledger.lock();
        if let Some(submission) =
            duplicate_submission_value(&ledger, &destination, text, *replace_existing)
        {
            return Ok(json!({
                "executed": false,
                "action": model_action,
                "observation_id": observation_id_for(&observation),
                "submission": submission,
            }));
        }
    }
    if matches!(
        action,
        InputAction::Click { .. }
            | InputAction::DoubleClick { .. }
            | InputAction::Drag { .. }
            | InputAction::Move { .. }
    ) {
        *context.focused_control.lock() = None;
    }
    context.policy.validate_input(&action, &observation)?;
    // Text already entered through the field's Value pattern: its before
    // and after contents were read back from the control itself.
    let pattern_entered = model_action.get("pattern_text").and_then(|entered| {
        serde_json::from_value::<crate::platform::PatternTextOutcome>(entered.clone()).ok()
    });
    let before_text = if let Some(entered) = &pattern_entered {
        Some(entered.before.clone())
    } else if matches!(action, InputAction::TypeText { .. }) {
        // Reading the focused field may bring the agent's window forward;
        // wait until the user is not typing or moving the mouse first.
        wait_for_user_idle(context).await?;
        context.platform.focused_text(65_536).await?
    } else {
        None
    };
    // Cursor-free first: a control that supports an accessibility pattern is
    // operated without moving the user's mouse; anything else, or anything
    // the platform cannot match safely, uses physical input as before.
    let mut input_method = if pattern_entered.is_some() {
        "ui_automation:value".to_owned()
    } else {
        match pattern_input(context, &action, &observation, &model_action).await? {
            Some(pattern) => format!("ui_automation:{pattern}"),
            None => {
                simulate_input_guarded(context, &action, &observation).await?;
                "physical".to_owned()
            }
        }
    };
    let mut after_text = if let Some(entered) = &pattern_entered {
        Some(entered.after.clone())
    } else if matches!(action, InputAction::TypeText { .. }) {
        context.platform.focused_text(65_536).await?
    } else {
        None
    };
    let mut retry_count = 0_u8;
    let mut typing_verification = match &action {
        InputAction::TypeText {
            text,
            replace_existing,
        } => verify_typed_text(
            before_text.as_deref(),
            after_text.as_deref(),
            text,
            *replace_existing,
        ),
        _ => TextVerification::NotApplicable,
    };
    // Some applications (word processors in particular) apply keystrokes a
    // moment after they are sent. Re-read before calling it a mismatch.
    if let InputAction::TypeText {
        text,
        replace_existing,
    } = &action
        && input_method == "physical"
    {
        for _ in 0..8 {
            if !matches!(typing_verification, TextVerification::Mismatch { .. }) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(250)).await;
            after_text = context.platform.focused_text(65_536).await?;
            typing_verification = verify_typed_text(
                before_text.as_deref(),
                after_text.as_deref(),
                text,
                *replace_existing,
            );
        }
    }
    if let InputAction::TypeText {
        text,
        replace_existing,
    } = &action
        && matches!(typing_verification, TextVerification::Mismatch { .. })
        && input_method == "physical"
        && (*replace_existing || before_text.as_deref().is_some_and(str::is_empty))
    {
        let expected_window_id = observation
            .foreground_window
            .as_ref()
            .map(|window| window.id.as_str())
            .ok_or_else(|| PokError::Tool("input requires an observed foreground window".into()))?;
        simulate_input_for_window(
            context,
            &InputAction::Key {
                key: "Ctrl+A".into(),
            },
            expected_window_id,
        )
        .await?;
        simulate_input_for_window(
            context,
            &InputAction::Key {
                key: "Delete".into(),
            },
            expected_window_id,
        )
        .await?;
        simulate_input_for_window(
            context,
            &InputAction::TypeText {
                text: text.clone(),
                replace_existing: false,
            },
            expected_window_id,
        )
        .await?;
        retry_count = 1;
        after_text = context.platform.focused_text(65_536).await?;
        typing_verification = verify_typed_text(Some(""), after_text.as_deref(), text, true);
    }
    let is_submit = matches!(&action, InputAction::Key { key } if key.eq_ignore_ascii_case("enter"))
        && context.input_ledger.lock().pending.is_some();
    if is_submit {
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
    }
    if may_open_related_window(&action) {
        tokio::time::sleep(std::time::Duration::from_millis(150)).await;
    }
    let foreground = context.platform.foreground_window().await?;
    let mut refresh_request =
        refresh_request_after_input(&observation, foreground.as_ref(), context.vision_max_edge)?;
    let mut after = context
        .platform
        .observe(
            &refresh_request,
            true,
            true,
            context.uia_element_limit,
            Duration::from_millis(context.desktop_enrichment_timeout_ms),
        )
        .await?;
    let fusion_started = Instant::now();
    after.targets = build_targets(
        &after,
        context.model_target_limit,
        context.fusion_iou_threshold,
        context.ocr_containment_threshold,
        &context.task_hint.lock(),
    );
    after.timings_ms.insert(
        "fusion".into(),
        u64::try_from(fusion_started.elapsed().as_millis()).unwrap_or(u64::MAX),
    );
    let mut state_change = state_change_value(&before, &after, &context.task_hint.lock());
    let may_navigate = matches!(
        &action,
        InputAction::Click { .. } | InputAction::DoubleClick { .. }
    ) || matches!(&action, InputAction::Key { key } if key.eq_ignore_ascii_case("enter"));
    let may_settle = matches!(
        &action,
        InputAction::Click { .. }
            | InputAction::DoubleClick { .. }
            | InputAction::Drag { .. }
            | InputAction::Key { .. }
    );
    for delay_ms in [200_u64, 600_u64] {
        if state_change_has_effect(&state_change) || !may_settle {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
        let foreground = context.platform.foreground_window().await?;
        refresh_request = refresh_request_after_input(
            &observation,
            foreground.as_ref(),
            context.vision_max_edge,
        )?;
        after = context
            .platform
            .observe(
                &refresh_request,
                true,
                true,
                context.uia_element_limit,
                Duration::from_millis(context.desktop_enrichment_timeout_ms),
            )
            .await?;
        let fusion_started = Instant::now();
        after.targets = build_targets(
            &after,
            context.model_target_limit,
            context.fusion_iou_threshold,
            context.ocr_containment_threshold,
            &context.task_hint.lock(),
        );
        after.timings_ms.insert(
            "fusion".into(),
            u64::try_from(fusion_started.elapsed().as_millis()).unwrap_or(u64::MAX),
        );
        state_change = state_change_value(&before, &after, &context.task_hint.lock());
    }
    // Selecting an item changed nothing: in galleries and similar lists a
    // click runs the item's action instead, which is its Invoke pattern.
    if input_method == "ui_automation:select"
        && !state_change_has_effect(&state_change)
        && let InputAction::Click { x, y, .. } = &action
        && let Some(target) = model_action
            .get("target_id")
            .and_then(Value::as_str)
            .and_then(|id| observation.targets.iter().find(|target| target.id == id))
        && let Some(window) = observation.foreground_window.as_ref()
    {
        let request = crate::platform::PatternActionRequest {
            window_id: window.id.clone(),
            name: target.name.clone(),
            control_type: target.control_type.clone(),
            bounds: target.bounds.clone(),
            point: (*x, *y),
            action: crate::platform::PatternAction::Invoke,
        };
        if let Ok(Some(_)) = context.platform.perform_pattern_action(&request).await {
            input_method = "ui_automation:select+invoke".into();
            tokio::time::sleep(std::time::Duration::from_millis(400)).await;
            let foreground = context.platform.foreground_window().await?;
            refresh_request = refresh_request_after_input(
                &observation,
                foreground.as_ref(),
                context.vision_max_edge,
            )?;
            after = context
                .platform
                .observe(
                    &refresh_request,
                    true,
                    true,
                    context.uia_element_limit,
                    Duration::from_millis(context.desktop_enrichment_timeout_ms),
                )
                .await?;
            after.targets = build_targets(
                &after,
                context.model_target_limit,
                context.fusion_iou_threshold,
                context.ocr_containment_threshold,
                &context.task_hint.lock(),
            );
            state_change = state_change_value(&before, &after, &context.task_hint.lock());
        }
    }
    let mut navigation_readiness = None;
    if may_navigate && browser_location_changed(&before, &after) {
        let before_identity = browser_page_identity(&before);
        let (settled, readiness) = settle_browser_navigation(
            context,
            &refresh_request,
            &before_identity,
            "browser navigation",
            NAVIGATION_SETTLE_TIMEOUT,
        )
        .await?;
        after = settled;
        let fusion_started = Instant::now();
        after.targets = build_targets(
            &after,
            context.model_target_limit,
            context.fusion_iou_threshold,
            context.ocr_containment_threshold,
            &context.task_hint.lock(),
        );
        after.timings_ms.insert(
            "fusion".into(),
            u64::try_from(fusion_started.elapsed().as_millis()).unwrap_or(u64::MAX),
        );
        state_change = state_change_value(&before, &after, &context.task_hint.lock());
        navigation_readiness = Some(readiness);
    }

    let submission = match &action {
        InputAction::TypeText { text, .. }
            if matches!(typing_verification, TextVerification::Verified { .. }) =>
        {
            context.input_ledger.lock().pending = Some(PendingSubmission {
                text: text.clone(),
                destination: destination.clone(),
                destination_label: destination_label.clone(),
                submitted: false,
            });
            Some(json!({
                "status": "draft_verified",
                "destination": destination_label,
            }))
        }
        InputAction::TypeText { .. } => {
            context.input_ledger.lock().pending = None;
            None
        }
        InputAction::Key { key } if key.eq_ignore_ascii_case("enter") => {
            let pending = context.input_ledger.lock().pending.clone();
            pending.map(|mut pending| {
                pending.submitted = true;
                if let Some(evidence) = submission_evidence(&after, &pending.text) {
                    let record = VerifiedSubmission {
                        normalized_text: normalized_text(&pending.text),
                        destination: pending.destination,
                        destination_label: pending.destination_label,
                        evidence,
                    };
                    let value = verified_submission_value(&record, "verified");
                    let mut ledger = context.input_ledger.lock();
                    ledger.verified.push(record);
                    ledger.pending = None;
                    value
                } else {
                    context.input_ledger.lock().pending = Some(pending.clone());
                    json!({
                        "status": "uncertain",
                        "destination": pending.destination_label,
                        "instruction": "Capture the window and verify the submitted text before any retry.",
                    })
                }
            })
        }
        _ => None,
    };
    *context.latest_observation.lock() = Some(after.clone());
    *context.latest_observation_view.lock() = None;
    *context.pending_visual_localization.lock() = None;
    let diagnostic = crate::tool::diagnostic_value(&serde_json::to_value(&after)?);
    context.write_artifact(&format!("observation-{}.json", after.version), &diagnostic)?;
    save_observation_visuals(context, &after)?;
    context.write_artifact(
        &format!("observation-{}-targets.json", after.version),
        &after.targets,
    )?;
    context.write_artifact(
        &format!("input-{}.json", after.version),
        &json!({
            "executed_physical_action": action,
            "physical_cursor": after.cursor,
            "observation_uuid": after.version,
        }),
    )?;
    let text_input = matches!(action, InputAction::TypeText { .. });
    let mut result = if text_input {
        compact_post_input_value(&after)?
    } else {
        model_observation_value(&after, context.annotate_targets)?
    };
    result["executed"] = json!(true);
    result["action"] = model_action;
    result["action_context"] = action_context;
    result["focus"] = json!(after.foreground_window.as_ref().map(|window| json!({
        "title": window.title,
        "app": window.process_name,
    })));
    result["grounding"] = grounding;
    result["input_method"] = json!(input_method);
    let (verified_effect, verification_evidence) =
        action_verified_effect(&action, &state_change, &typing_verification);
    result["verification"] = json!({
        "effect": verified_effect,
        "evidence": verification_evidence,
    });
    result["state_change"] = state_change;
    if let (Some(before_window), Some(after_window)) = (
        before.foreground_window.as_ref(),
        after.foreground_window.as_ref(),
    ) && before_window.id != after_window.id
        && before_window
            .process_name
            .eq_ignore_ascii_case(&after_window.process_name)
    {
        result["foreground_transition"] = json!({
            "kind": "related_transient",
            "parent_window_id": before_window.id,
            "window_id": after_window.id,
            "app": after_window.process_name,
            "title": after_window.title,
            "instruction": "Continue in this current related popup/dialog. Do not reactivate the parent window unless this transient closes. If it is an editor, type or use keyboard input now."
        });
    }
    if let Some(readiness) = navigation_readiness {
        result["navigation_readiness"] = serde_json::to_value(&readiness)?;
        result["page_status"] = json!(match readiness.status {
            NavigationReadinessStatus::Ready => "loaded",
            NavigationReadinessStatus::Loading => "loading",
            NavigationReadinessStatus::ErrorPage => "error_page",
        });
    }
    result["refresh"] = json!({
        "scope": refresh_request.scope,
        "narrowed_to_foreground_window": matches!(refresh_request.scope, CaptureScope::Window)
            && observation.target.as_ref().is_some_and(|target| !matches!(target.scope, CaptureScope::Window)),
        "complete": true,
    });
    if !text_input {
        result["note"] = json!(
            "This is the complete post-action state. Reuse it for the next action; capture again only after state may have changed."
        );
    }
    if let InputAction::TypeText {
        text,
        replace_existing,
    } = &action
    {
        result["typing"] = typing_verification_value(
            &typing_verification,
            text.chars().count(),
            after_text.as_deref(),
            *replace_existing,
            retry_count,
        );
        match typing_verification {
            TextVerification::Verified { .. } => {
                result["note"] = json!(
                    "Text entry was verified from focused-control read-back. Capture again only when visual targets are needed."
                );
            }
            TextVerification::Unavailable => {
                result["note"] = json!(
                    "Bulk text input completed, but focused-control read-back is unavailable. Inspect the destination before saving or submitting."
                );
            }
            TextVerification::Mismatch { .. } => {
                result["executed"] = json!(false);
                result["recovery_required"] = json!(true);
                result["note"] = json!(
                    "Text input was physically attempted but read-back did not match after recovery. Do not submit, save, or continue from this draft; replace the corrupted text before proceeding."
                );
            }
            TextVerification::NotApplicable => {}
        }
    }
    if let Some(submission) = submission {
        result["submission"] = submission;
    }
    Ok(result)
}

pub(super) fn may_open_related_window(action: &InputAction) -> bool {
    matches!(
        action,
        InputAction::Click { .. } | InputAction::DoubleClick { .. }
    ) || matches!(action, InputAction::Key { key } if key.eq_ignore_ascii_case("enter"))
}

pub(super) fn input_action_context(
    observation: &Observation,
    action: &InputAction,
    model_action: &Value,
) -> Value {
    let target = model_action
        .get("target_id")
        .and_then(Value::as_str)
        .and_then(|id| observation.targets.iter().find(|target| target.id == id))
        .or_else(|| match action {
            InputAction::Click { x, y, .. } | InputAction::DoubleClick { x, y, .. } => observation
                .targets
                .iter()
                .filter(|target| target.actionable && target.bounds.contains(*x, *y))
                .min_by_key(|target| {
                    u64::from(target.bounds.width) * u64::from(target.bounds.height)
                }),
            InputAction::Drag {
                start_x, start_y, ..
            } => observation
                .targets
                .iter()
                .filter(|target| target.actionable && target.bounds.contains(*start_x, *start_y))
                .min_by_key(|target| {
                    u64::from(target.bounds.width) * u64::from(target.bounds.height)
                }),
            _ => None,
        });
    json!({
        "window_id": observation.foreground_window.as_ref().map(|window| window.id.as_str()),
        "window_title": observation.foreground_window.as_ref().map(|window| window.title.as_str()),
        "target_id": target.map(|target| target.id.as_str()),
        "label": target.map(|target| target.name.as_str()),
        "control_type": target.map(|target| target.control_type.as_str()),
        "bounds": target.map(|target| &target.bounds),
    })
}

pub(super) fn browser_location_changed(before: &Observation, after: &Observation) -> bool {
    let Some(before_target) = before.target.as_ref() else {
        return false;
    };
    let Some(after_target) = after.target.as_ref() else {
        return false;
    };
    if !["chrome", "msedge", "firefox", "brave"]
        .iter()
        .any(|browser| {
            after_target
                .process_name
                .to_ascii_lowercase()
                .contains(browser)
        })
    {
        return false;
    }
    browser_address_url(before, before_target)
        .zip(browser_address_url(after, after_target))
        .is_some_and(|(before, after)| {
            normalized_browser_location(&before) != normalized_browser_location(&after)
        })
}

pub(super) async fn simulate_input_guarded(
    context: &ToolContext,
    action: &InputAction,
    observation: &Observation,
) -> Result<()> {
    let expected_window_id = observation
        .foreground_window
        .as_ref()
        .map(|window| window.id.as_str())
        .ok_or_else(|| PokError::Tool("input requires an observed foreground window".into()))?;
    simulate_input_for_window(context, action, expected_window_id).await
}

/// Physical input within this long means the user is using the machine.
pub(super) const USER_ACTIVE_MS: u64 = 1_500;
/// Once the agent has yielded, the user must be idle this long before
/// physical input resumes, so a brief pause mid-task is not interrupted.
pub(super) const USER_IDLE_RESUME_MS: u64 = 2_500;

/// Hold physical input while the user is operating the mouse or keyboard,
/// then resume automatically once they have been idle. Only physical input
/// is gated; the agent's own injected input never counts as user activity.
pub(super) async fn wait_for_user_idle(context: &ToolContext) -> Result<u64> {
    let started = Instant::now();
    let mut threshold = USER_ACTIVE_MS;
    let mut waiting = false;
    let result = loop {
        if let Err(error) = context.pause.ensure_action_allowed() {
            break Err(error);
        }
        let idle = match context.platform.desktop_activity_snapshot().await {
            Ok(snapshot) => snapshot.last_physical_input_ms,
            Err(error) => break Err(error),
        };
        if idle.is_none_or(|idle| idle >= threshold) {
            break Ok(u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX));
        }
        if !waiting {
            waiting = true;
            context.pause.notify_waiting_for_user(true);
        }
        threshold = USER_IDLE_RESUME_MS;
        tokio::select! {
            () = context.cancellation.cancelled() => break Err(PokError::Cancelled),
            () = tokio::time::sleep(Duration::from_millis(250)) => {}
        }
    };
    if waiting {
        context.pause.notify_waiting_for_user(false);
    }
    result
}

/// Try the cursor-free equivalent of a left click or double click on the
/// observed target. Returns the pattern used, or `None` to fall back to
/// physical input (unsupported action, OCR-only target, unsupported control,
/// or a control the platform could not match safely).
pub(super) async fn pattern_input(
    context: &ToolContext,
    action: &InputAction,
    observation: &Observation,
    model_action: &Value,
) -> Result<Option<String>> {
    let (point, pattern_action) = match action {
        InputAction::Click {
            x,
            y,
            button: MouseButton::Left,
        } => ((*x, *y), crate::platform::PatternAction::Activate),
        InputAction::DoubleClick {
            x,
            y,
            button: MouseButton::Left,
        } => ((*x, *y), crate::platform::PatternAction::Open),
        _ => return Ok(None),
    };
    let Some(target) = model_action
        .get("target_id")
        .and_then(Value::as_str)
        .and_then(|id| observation.targets.iter().find(|target| target.id == id))
    else {
        return Ok(None);
    };
    // Only targets backed by an accessibility element can be acted on
    // through a pattern; OCR and visual targets exist only as pixels.
    if !matches!(target.source, TargetSource::Uia | TargetSource::UiaOcr) {
        return Ok(None);
    }
    let Some(window) = observation.foreground_window.as_ref() else {
        return Ok(None);
    };
    context.pause.ensure_action_allowed()?;
    // Verification captures the foreground window, so pattern input follows
    // the same foreground rule as physical input; on a mismatch the physical
    // path reports the stale observation.
    let current = context.platform.foreground_window().await?;
    if !foreground_matches(&window.id, current.as_ref()) {
        return Ok(None);
    }
    let request = crate::platform::PatternActionRequest {
        window_id: window.id.clone(),
        name: target.name.clone(),
        control_type: target.control_type.clone(),
        bounds: target.bounds.clone(),
        point,
        action: pattern_action,
    };
    Ok(context
        .platform
        .perform_pattern_action(&request)
        .await
        .ok()
        .flatten()
        .map(|outcome| outcome.pattern))
}

pub(super) async fn simulate_input_for_window(
    context: &ToolContext,
    action: &InputAction,
    expected_window_id: &str,
) -> Result<()> {
    context.pause.ensure_action_allowed()?;
    wait_for_user_idle(context).await?;
    let current = context.platform.foreground_window().await?;
    if !foreground_matches(expected_window_id, current.as_ref()) {
        *context.latest_observation.lock() = None;
        *context.latest_observation_view.lock() = None;
        *context.pending_visual_localization.lock() = None;
        *context.focused_control.lock() = None;
        return Err(PokError::Tool(format!(
            "desktop control changed before input: expected foreground {expected_window_id:?}, current foreground is {}; stale observation invalidated—observe and activate the intended window again",
            current.as_ref().map_or_else(
                || "unknown".into(),
                |window| format!("{:?} ({})", window.title, window.process_name)
            )
        )));
    }
    context.pause.ensure_action_allowed()?;
    match action {
        InputAction::TypeText {
            text,
            replace_existing,
        } => {
            if *replace_existing && context.platform.replace_focused_text(text).await? {
                return Ok(());
            }
            if *replace_existing {
                context
                    .platform
                    .simulate_input(&InputAction::Key {
                        key: "Ctrl+A".into(),
                    })
                    .await?;
                context
                    .platform
                    .simulate_input(&InputAction::Key {
                        key: "Delete".into(),
                    })
                    .await?;
            }
            context
                .platform
                .simulate_text(text, context.input_text_inter_key_pause_ms)
                .await
        }
        _ => context.platform.simulate_input(action).await,
    }
}

#[derive(Debug)]
pub(super) enum TextVerification {
    NotApplicable,
    Verified { observed_chars: usize },
    Unavailable,
    Mismatch { observed_chars: usize },
}

pub(super) fn normalize_control_text(value: &str) -> String {
    value.replace("\r\n", "\n").replace('\r', "\n")
}

pub(super) fn verify_typed_text(
    before: Option<&str>,
    after: Option<&str>,
    requested: &str,
    replace_existing: bool,
) -> TextVerification {
    let Some(after) = after else {
        return TextVerification::Unavailable;
    };
    let after = normalize_control_text(after);
    let requested = normalize_control_text(requested);
    let observed_chars = after.chars().count();
    let verified = if replace_existing || before.is_some_and(str::is_empty) {
        after == requested
    } else {
        let before = before.map(normalize_control_text);
        after.contains(&requested) && before.as_deref() != Some(after.as_str())
    };
    if verified {
        TextVerification::Verified { observed_chars }
    } else {
        TextVerification::Mismatch { observed_chars }
    }
}

pub(super) fn typing_verification_value(
    verification: &TextVerification,
    requested_chars: usize,
    observed: Option<&str>,
    replace_existing: bool,
    retry_count: u8,
) -> Value {
    let (status, observed_chars) = match verification {
        TextVerification::NotApplicable => ("not_applicable", None),
        TextVerification::Verified { observed_chars } => ("verified", Some(*observed_chars)),
        TextVerification::Unavailable => ("unavailable", None),
        TextVerification::Mismatch { observed_chars } => ("mismatch", Some(*observed_chars)),
    };
    let excerpt = matches!(verification, TextVerification::Mismatch { .. })
        .then(|| {
            observed.map(|text| {
                let mut excerpt = text.chars().take(160).collect::<String>();
                if text.chars().count() > 160 {
                    excerpt.push('…');
                }
                excerpt
            })
        })
        .flatten();
    json!({
        "status": status,
        "strategy": if replace_existing { "uia_value_or_batched_unicode_replace" } else { "batched_unicode" },
        "requested_chars": requested_chars,
        "observed_chars": observed_chars,
        "observed_excerpt": excerpt,
        "retry_count": retry_count,
    })
}

pub(super) fn foreground_matches(expected_window_id: &str, current: Option<&WindowInfo>) -> bool {
    current.map(|window| window.id.as_str()) == Some(expected_window_id)
}

pub(super) fn refresh_request_after_input(
    observation: &Observation,
    foreground: Option<&WindowInfo>,
    max_edge: u32,
) -> Result<CaptureRequest> {
    let target = observation
        .target
        .as_ref()
        .ok_or_else(|| PokError::Tool("observation has no capture target".into()))?;
    let foreground_in_authorized_view = foreground.is_some_and(|window| {
        !window.id.is_empty()
            && window.visible
            && !window.elevated
            && overlaps(&window.bounds, &target.bounds)
    });
    if foreground_in_authorized_view {
        let window = foreground.expect("checked above");
        return Ok(CaptureRequest {
            scope: CaptureScope::Window,
            window_id: Some(window.id.clone()),
            monitor_id: None,
            region: None,
            max_edge,
        });
    }
    Ok(CaptureRequest {
        scope: target.scope.clone(),
        window_id: matches!(target.scope, CaptureScope::Window).then(|| target.id.clone()),
        monitor_id: matches!(target.scope, CaptureScope::Monitor).then(|| target.id.clone()),
        region: matches!(target.scope, CaptureScope::Region).then(|| target.bounds.clone()),
        max_edge,
    })
}

pub(super) fn state_change_has_effect(state_change: &Value) -> bool {
    state_change
        .get("added_text")
        .and_then(Value::as_array)
        .is_some_and(|items| !items.is_empty())
        || state_change
            .get("removed_control_count")
            .and_then(Value::as_u64)
            .is_some_and(|count| count > 0)
        || state_change.get("focus_changed").and_then(Value::as_bool) == Some(true)
        || state_change
            .get("selected_changed")
            .and_then(Value::as_array)
            .is_some_and(|items| !items.is_empty())
        || state_change
            .get("focused_control_changed")
            .and_then(Value::as_bool)
            == Some(true)
}

pub(super) fn validate_browser_navigation(
    text: &str,
    active_request: &str,
    focused_control: Option<&FocusedControl>,
) -> Result<()> {
    let Some(focus) = focused_control else {
        return Ok(());
    };
    let app = focus.app.to_ascii_lowercase();
    let label = focus.label.to_ascii_lowercase();
    let control_type = focus.control_type.to_ascii_lowercase();
    let browser = app.contains("chrome") || app.contains("msedge") || app.contains("firefox");
    let address_control = control_type.contains("edit")
        && (label.contains("address") || label.contains("search bar") || label.contains("omnibox"));
    if !browser || !address_control {
        return Ok(());
    }

    let requested_hosts = extract_hosts(active_request);
    let Some(typed_host) = parse_host(text) else {
        return Ok(());
    };
    if requested_hosts.is_empty()
        || requested_hosts
            .iter()
            .any(|host| host == &typed_host || typed_host.ends_with(&format!(".{host}")))
    {
        return Ok(());
    }
    Err(PokError::Tool(format!(
        "off-task browser navigation blocked: the active request names {}, but the address bar input targets {typed_host}. Re-read the active request before typing.",
        requested_hosts.join(", ")
    )))
}

pub(super) fn extract_hosts(text: &str) -> Vec<String> {
    let mut hosts = text
        .split_whitespace()
        .filter_map(|token| {
            let candidate = token.trim_matches(|character: char| {
                matches!(
                    character,
                    '(' | ')' | '[' | ']' | '{' | '}' | '<' | '>' | ',' | ';' | '\'' | '"'
                )
            });
            if candidate.contains('@') && !candidate.contains("://") {
                return None;
            }
            parse_host(candidate)
        })
        .collect::<Vec<_>>();
    hosts.sort();
    hosts.dedup();
    hosts
}

pub(super) fn parse_host(value: &str) -> Option<String> {
    let value = value.trim().trim_end_matches(['.', '?', '!', ':']);
    if value.chars().any(char::is_whitespace) || !value.contains('.') {
        return None;
    }
    let candidate = if value.contains("://") {
        value.to_owned()
    } else {
        format!("https://{value}")
    };
    let host = reqwest::Url::parse(&candidate)
        .ok()?
        .host_str()?
        .to_ascii_lowercase();
    Some(host.strip_prefix("www.").unwrap_or(&host).to_owned())
}

pub(super) fn input_action(
    args: &InputArgs,
    target: &Rect,
    screenshot: &Screenshot,
) -> Result<InputAction> {
    let physical_point = || -> Result<(i32, i32)> {
        let x = args
            .x
            .ok_or_else(|| PokError::Tool("x is required".into()))?;
        let y = args
            .y
            .ok_or_else(|| PokError::Tool("y is required".into()))?;
        if x < 0
            || y < 0
            || u32::try_from(x).unwrap_or(u32::MAX) >= screenshot.model_width
            || u32::try_from(y).unwrap_or(u32::MAX) >= screenshot.model_height
        {
            return Err(PokError::Tool(format!(
                "point ({x}, {y}) is outside model image {}x{}",
                screenshot.model_width, screenshot.model_height
            )));
        }
        Ok((
            target.x + scale_offset(x, screenshot.model_width, target.width),
            target.y + scale_offset(y, screenshot.model_height, target.height),
        ))
    };
    match args.kind.as_str() {
        "move" | "mouse_move" => {
            let (x, y) = physical_point()?;
            Ok(InputAction::Move { x, y })
        }
        "click" | "left_click" | "right_click" | "middle_click" => {
            let (x, y) = physical_point()?;
            let button = match args.kind.as_str() {
                "right_click" => MouseButton::Right,
                "middle_click" => MouseButton::Middle,
                _ => args.button.unwrap_or(MouseButton::Left),
            };
            Ok(InputAction::Click { x, y, button })
        }
        // Some smaller local models shorten this enum value despite seeing the schema.
        // Accepting the obvious alias avoids wasting a complete inference turn.
        "text" | "type_text" | "type" => Ok(InputAction::TypeText {
            text: args
                .text
                .clone()
                .ok_or_else(|| PokError::Tool("text is required".into()))?,
            replace_existing: args.replace_existing,
        }),
        "keyboard" | "key" | "key_press" | "shortcut" | "keyboard_shortcut" => {
            Ok(InputAction::Key {
                key: args
                    .key
                    .clone()
                    .ok_or_else(|| PokError::Tool("key is required".into()))?,
            })
        }
        "scroll" => {
            let (delta_x, delta_y) = legacy_scroll_deltas(args);
            if delta_x == 0 && delta_y == 0 {
                return Err(PokError::Tool(
                    "scroll requires non-zero delta_x or delta_y; prefer scroll_view with direction and amount"
                        .into(),
                ));
            }
            Ok(InputAction::Scroll { delta_x, delta_y })
        }
        "mouse_scroll_down" | "scroll_down" => Ok(InputAction::Scroll {
            delta_x: 0,
            delta_y: bounded_wheel_delta(args.delta_y.unwrap_or(3)).abs(),
        }),
        "mouse_scroll_up" | "scroll_up" => Ok(InputAction::Scroll {
            delta_x: 0,
            delta_y: -bounded_wheel_delta(args.delta_y.unwrap_or(3)).abs(),
        }),
        other => Err(PokError::Tool(format!(
            "unsupported input kind {other:?}; use move, click/left_click/right_click/middle_click, type_text/text, key/keyboard/key_press/keyboard_shortcut, or prefer scroll_view"
        ))),
    }
}

pub(super) fn bounded_wheel_delta(value: i32) -> i32 {
    value.clamp(-20, 20)
}

pub(super) fn legacy_scroll_deltas(args: &InputArgs) -> (i32, i32) {
    match args.kind.as_str() {
        "mouse_scroll_down" | "scroll_down" => {
            (0, bounded_wheel_delta(args.delta_y.unwrap_or(3)).abs())
        }
        "mouse_scroll_up" | "scroll_up" => {
            (0, -bounded_wheel_delta(args.delta_y.unwrap_or(3)).abs())
        }
        _ => (
            bounded_wheel_delta(args.delta_x.unwrap_or(0)),
            bounded_wheel_delta(args.delta_y.unwrap_or(0)),
        ),
    }
}
