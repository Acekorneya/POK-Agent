//! `execute_action_batch`: several grounded steps in one call, validated up front and verified step by step.

use super::*;

#[derive(Debug, Deserialize, JsonSchema)]
pub(super) struct ActionStep {
    /// Action kind. Accepted values: click_target (aliases: click, left_click),
    /// type_text (aliases: type, text), or key (aliases: keyboard,
    /// keyboard_shortcut, shortcut, key_press).
    pub(super) kind: String,
    /// Fresh numbered target: required by click_target/click/left_click, and
    /// optional for type_text, which then focuses that field before typing.
    #[serde(default)]
    pub(super) target_id: Option<TargetId>,
    /// Visible label intended by this click step.
    #[serde(default)]
    pub(super) expected_label: Option<String>,
    /// Complete text for a type_text step, or compatibility key value.
    #[serde(default)]
    pub(super) text: Option<String>,
    #[serde(default, alias = "clear_current_text")]
    pub(super) replace_existing: bool,
    #[serde(default)]
    pub(super) key: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub(super) struct ExecuteActionBatchArgs {
    #[serde(default, alias = "observation_version")]
    pub(super) observation_id: Option<String>,
    pub(super) steps: Vec<ActionStep>,
}

pub(super) fn batch_step_key(step: &ActionStep) -> Option<&str> {
    step.key.as_deref().or(step.text.as_deref())
}

pub(crate) fn normalized_batch_kind(kind: &str) -> &str {
    match kind {
        "click" | "left_click" => "click_target",
        "text" | "type" => "type_text",
        "keyboard" | "keyboard_shortcut" | "shortcut" | "key_press" => "key",
        other => other,
    }
}

pub(super) fn batch_steps_repeat(left: &ActionStep, right: &ActionStep) -> bool {
    normalized_batch_kind(&left.kind) == normalized_batch_kind(&right.kind)
        && left.target_id.as_ref().map(TargetId::normalized)
            == right.target_id.as_ref().map(TargetId::normalized)
        && left.expected_label == right.expected_label
        && left.text == right.text
        && left.replace_existing == right.replace_existing
        && batch_step_key(left) == batch_step_key(right)
}

pub(super) fn strip_inline_images(value: &mut Value) {
    match value {
        Value::Object(object) => {
            object.remove("png_base64");
            for value in object.values_mut() {
                strip_inline_images(value);
            }
        }
        Value::Array(values) => {
            for value in values {
                strip_inline_images(value);
            }
        }
        _ => {}
    }
}

pub(super) fn compact_batch_step_result(value: &mut Value) {
    strip_inline_images(value);
    let mut compact = serde_json::Map::new();
    for key in [
        "observation_id",
        "executed",
        "action",
        "focus",
        "state_change",
        "verification",
        "typing",
        "submission",
        "grounding",
    ] {
        if let Some(item) = value.get(key) {
            compact.insert(key.into(), item.clone());
        }
    }
    *value = Value::Object(compact);
}

pub(super) fn is_batch_target_click(kind: &str) -> bool {
    matches!(kind, "click_target" | "click" | "left_click")
}

pub(super) struct ExecuteActionBatchTool;
#[async_trait]
impl Tool for ExecuteActionBatchTool {
    fn name(&self) -> &'static str {
        "execute_action_batch"
    }
    fn description(&self) -> &'static str {
        "Execute grounded input steps sequentially against the latest observation. The runtime verifies intermediate state but returns only the final visual observation. Repeated identical steps stop when the first settled action makes no progress. Click steps require target_id and expected_label; the full batch is semantically validated before input. Other steps are type_text and key; a type_text step with target_id focuses that field first, so each value lands in its own field. Desktop forms often look up or validate a field (a ZIP code filling in the city) only on the next key: follow each form value with Tab. Example: [{\"kind\":\"type_text\",\"target_id\":\"12\",\"text\":\"10001\",\"replace_existing\":true},{\"kind\":\"key\",\"key\":\"Tab\"},{\"kind\":\"click_target\",\"target_id\":\"7\",\"expected_label\":\"Search\"},{\"kind\":\"type_text\",\"text\":\"complete query\",\"replace_existing\":true},{\"kind\":\"key\",\"key\":\"Enter\"}]."
    }
    fn input_schema(&self) -> Value {
        schema::<ExecuteActionBatchArgs>()
    }
    fn risk(&self) -> RiskClass {
        RiskClass::DesktopInput
    }
    async fn execute(&self, args: Value, context: &ToolContext) -> Result<Value> {
        let args: ExecuteActionBatchArgs = serde_json::from_value(args)?;
        if args.steps.is_empty() {
            return Err(PokError::Tool("action batch cannot be empty".into()));
        }
        let observation = current_observation(
            context,
            args.observation_id.as_deref(),
            "execute_action_batch",
        )?;
        ensure_targeted_capture(&observation)?;
        // Validate the complete batch before the first physical input. This prevents
        // a malformed later step from leaving an earlier click or typed draft behind.
        let mut planned_focus = context.focused_control.lock().clone();
        for (index, step) in args.steps.iter().enumerate() {
            match step.kind.as_str() {
                kind if is_batch_target_click(kind) => {
                    let target_elem = resolve_batch_target(&observation, step, index)?;
                    planned_focus = Some(FocusedControl {
                        app: observation
                            .foreground_window
                            .as_ref()
                            .map_or_else(String::new, |window| window.process_name.clone()),
                        label: target_elem.name.clone(),
                        control_type: target_elem.control_type.clone(),
                    });
                }
                "text" | "type" | "type_text" => {
                    let text = step.text.as_deref().ok_or_else(|| {
                        PokError::Tool(format!("text is required at batch step {index}"))
                    })?;
                    validate_browser_navigation(
                        text,
                        &context.task_hint.lock(),
                        planned_focus.as_ref(),
                    )?;
                }
                "keyboard" | "keyboard_shortcut" | "shortcut" | "key" | "key_press" => {
                    let key = batch_step_key(step);
                    if key.is_none_or(str::is_empty) {
                        return Err(PokError::Tool(format!(
                            "key is required at batch step {index} (text is also accepted as a compatibility fallback)"
                        )));
                    }
                }
                other => {
                    return Err(PokError::Tool(format!(
                        "unsupported batch step kind: {other}"
                    )));
                }
            }
        }

        let mut step_results = Vec::new();
        let mut verified_progress = false;
        for (index, step) in args.steps.iter().enumerate() {
            let step_observation = context
                .latest_observation
                .lock()
                .clone()
                .unwrap_or_else(|| observation.clone());
            ensure_targeted_capture(&step_observation)?;
            let target = step_observation
                .target
                .as_ref()
                .expect("validated capture target");
            let screenshot = step_observation
                .screenshots
                .first()
                .ok_or_else(|| PokError::Tool("observation has no model image metadata".into()))?;
            let res = match step.kind.as_str() {
                kind if is_batch_target_click(kind) => {
                    let target_elem = resolve_batch_target(&step_observation, step, index)?;
                    let target_id = target_elem.id.clone();
                    let (x, y) = target_elem.click_point.unwrap_or((
                        target_elem.bounds.x
                            + i32::try_from(target_elem.bounds.width / 2).unwrap_or(i32::MAX),
                        target_elem.bounds.y
                            + i32::try_from(target_elem.bounds.height / 2).unwrap_or(i32::MAX),
                    ));
                    let focused_control = FocusedControl {
                        app: observation
                            .foreground_window
                            .as_ref()
                            .map_or_else(String::new, |window| window.process_name.clone()),
                        label: target_elem.name.clone(),
                        control_type: target_elem.control_type.clone(),
                    };
                    let result = execute_input(
                        InputAction::Click {
                            x,
                            y,
                            button: MouseButton::Left,
                        },
                        step_observation.clone(),
                        context,
                        json!({
                            "kind": "click_target",
                            "target_id": target_id,
                            "label": target_elem.name,
                            "expected_label": step.expected_label,
                        }),
                    )
                    .await;
                    if result.is_ok() {
                        *context.focused_control.lock() = Some(focused_control);
                    }
                    result
                }
                "text" | "type" | "type_text" => {
                    // A type step naming its field goes into that field, not
                    // whichever one kept focus from the previous step: focus it
                    // first (without the mouse when the control allows it).
                    let mut typing_observation = step_observation.clone();
                    let mut focused = Ok(Value::Null);
                    if step.target_id.is_some() {
                        let field = batch_type_field(&observation, &step_observation, step, index)?;
                        if !accepts_typed_text(&field.control_type) {
                            return Err(PokError::Tool(format!(
                                "batch step {index} types into target {:?}, a {} ({:?}), not a text field; use the edit or combo box target for the field",
                                field.id, field.control_type, field.name
                            )));
                        }
                        let (x, y) = field.click_point.unwrap_or((
                            field.bounds.x
                                + i32::try_from(field.bounds.width / 2).unwrap_or(i32::MAX),
                            field.bounds.y
                                + i32::try_from(field.bounds.height / 2).unwrap_or(i32::MAX),
                        ));
                        let focused_control = FocusedControl {
                            app: step_observation
                                .foreground_window
                                .as_ref()
                                .map_or_else(String::new, |window| window.process_name.clone()),
                            label: field.name.clone(),
                            control_type: field.control_type.clone(),
                        };
                        focused = execute_input(
                            InputAction::Click {
                                x,
                                y,
                                button: MouseButton::Left,
                            },
                            step_observation.clone(),
                            context,
                            json!({
                                "kind": "click_target",
                                "target_id": field.id,
                                "label": field.name,
                            }),
                        )
                        .await;
                        if focused.is_ok() {
                            *context.focused_control.lock() = Some(focused_control);
                            typing_observation = context
                                .latest_observation
                                .lock()
                                .clone()
                                .unwrap_or(typing_observation);
                        }
                    }
                    match focused {
                        Err(error) => Err(error),
                        Ok(value)
                            if value.get("executed").and_then(Value::as_bool) == Some(false) =>
                        {
                            Ok(value)
                        }
                        Ok(_) => {
                            let input_args = InputArgs {
                                observation_id: args.observation_id.clone(),
                                kind: "type".into(),
                                x: None,
                                y: None,
                                button: None,
                                text: step.text.clone(),
                                replace_existing: step.replace_existing,
                                key: None,
                                delta_x: None,
                                delta_y: None,
                            };
                            let action = input_action(&input_args, &target.bounds, screenshot)?;
                            let mut model_action = model_action_value(&input_args);
                            if let Some(target_id) = &step.target_id {
                                model_action["target_id"] = json!(target_id.normalized());
                            }
                            execute_input(action, typing_observation, context, model_action).await
                        }
                    }
                }
                "keyboard" | "keyboard_shortcut" | "shortcut" | "key" | "key_press" => {
                    let input_args = InputArgs {
                        observation_id: args.observation_id.clone(),
                        kind: "keyboard_shortcut".into(),
                        x: None,
                        y: None,
                        button: None,
                        text: None,
                        replace_existing: false,
                        key: batch_step_key(step).map(str::to_owned),
                        delta_x: None,
                        delta_y: None,
                    };
                    let action = input_action(&input_args, &target.bounds, screenshot)?;
                    let model_action = model_action_value(&input_args);
                    execute_input(action, step_observation.clone(), context, model_action).await
                }
                other => Err(PokError::Tool(format!(
                    "unsupported batch step kind: {other}"
                ))),
            };

            match res {
                Ok(mut val) if val.get("executed").and_then(Value::as_bool) != Some(false) => {
                    let effect = val
                        .pointer("/verification/effect")
                        .and_then(Value::as_bool)
                        .unwrap_or(false);
                    verified_progress |= effect;
                    let repeated_no_progress = !effect
                        && args
                            .steps
                            .get(index + 1)
                            .is_some_and(|next| batch_steps_repeat(step, next));
                    if repeated_no_progress {
                        let final_observation_id = val.get("observation_id").cloned();
                        step_results
                            .push(json!({"step": index, "status": "no_progress", "result": val}));
                        return Ok(json!({
                            "success": false,
                            "verified_progress": verified_progress,
                            "failed_at_step": index,
                            "stop_reason": "repeated_no_progress",
                            "final_observation_id": final_observation_id,
                            "step_results": step_results,
                        }));
                    }
                    if index + 1 < args.steps.len() {
                        compact_batch_step_result(&mut val);
                    }
                    step_results.push(json!({"step": index, "status": "ok", "result": val}));
                }
                Ok(val) => {
                    let final_observation_id = val.get("observation_id").cloned();
                    step_results.push(json!({
                        "step": index,
                        "status": "error",
                        "error": "input action did not verify",
                        "result": val,
                    }));
                    return Ok(json!({
                        "success": false,
                        "verified_progress": verified_progress,
                        "failed_at_step": index,
                        "error": "input action did not verify",
                        "final_observation_id": final_observation_id,
                        "step_results": step_results,
                    }));
                }
                Err(e) => {
                    step_results
                        .push(json!({"step": index, "status": "error", "error": e.to_string()}));
                    let final_observation_id = context
                        .latest_observation
                        .lock()
                        .as_ref()
                        .map(observation_id_for);
                    return Ok(json!({
                        "success": false,
                        "verified_progress": verified_progress,
                        "failed_at_step": index,
                        "error": e.to_string(),
                        "final_observation_id": final_observation_id,
                        "step_results": step_results,
                    }));
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(150)).await;
        }

        let final_observation_id = context
            .latest_observation
            .lock()
            .as_ref()
            .map(observation_id_for);
        Ok(json!({
            "success": verified_progress,
            "verified_progress": verified_progress,
            "final_observation_id": final_observation_id,
            "total_steps": args.steps.len(),
            "step_results": step_results,
        }))
    }
}

/// The field a batch type step names. Target ids come from the screen the
/// planner saw; earlier steps may have refreshed it and renumbered targets, so
/// the same field is found again by label and type, or by where it was.
pub(super) fn batch_type_field<'a>(
    planned: &'a Observation,
    current: &'a Observation,
    step: &ActionStep,
    index: usize,
) -> Result<&'a InteractionTarget> {
    let target_id = step
        .target_id
        .as_ref()
        .map(TargetId::normalized)
        .ok_or_else(|| PokError::Tool(format!("target_id is required at batch step {index}")))?;
    let named = planned
        .targets
        .iter()
        .chain(current.targets.iter())
        .find(|target| target.id == target_id)
        .ok_or_else(|| {
            PokError::Tool(format!(
                "unknown target {target_id:?} at batch step {index}; capture again and use a listed target id"
            ))
        })?;
    let same = |target: &&InteractionTarget| {
        target.id == named.id
            && target.name == named.name
            && target.control_type == named.control_type
    };
    Ok(current
        .targets
        .iter()
        .find(same)
        .or_else(|| unique_rebased_target(named, current))
        .unwrap_or(named))
}

pub(super) fn resolve_batch_target<'a>(
    observation: &'a Observation,
    step: &ActionStep,
    index: usize,
) -> Result<&'a InteractionTarget> {
    let target_id = step
        .target_id
        .as_ref()
        .map(TargetId::normalized)
        .ok_or_else(|| PokError::Tool(format!("target_id is required at batch step {index}")))?;
    let expected = step
        .expected_label
        .as_deref()
        .map(str::trim)
        .filter(|label| !label.is_empty());
    let requested = observation
        .targets
        .iter()
        .find(|target| target.id == target_id);
    let Some(requested) = requested else {
        if let Some(expected) = expected {
            let candidates = semantic_target_candidates(observation, expected);
            return match candidates.as_slice() {
                [candidate] => Ok(*candidate),
                [] => Err(PokError::Tool(format!(
                    "target {target_id:?} is no longer present and expected_label {expected:?} has no unique current match at batch step {index}"
                ))),
                _ => Err(PokError::Tool(format!(
                    "target {target_id:?} is no longer present and expected_label {expected:?} is ambiguous at batch step {index}"
                ))),
            };
        }
        return Err(PokError::Tool(format!(
            "unknown target {target_id:?} at batch step {index}"
        )));
    };
    if let Some(expected) = expected {
        if crate::grounding::grounding_quality(requested) != "low"
            && label_match_score(expected, &requested.name) >= 2
        {
            return Ok(requested);
        }
        let candidates = semantic_target_candidates(observation, expected);
        return match candidates.as_slice() {
            [candidate] => Ok(*candidate),
            [] => Err(PokError::Tool(format!(
                "target {target_id:?} does not match expected_label {expected:?} at batch step {index}; no input was executed"
            ))),
            _ => Err(PokError::Tool(format!(
                "expected_label {expected:?} is ambiguous at batch step {index}; no input was executed"
            ))),
        };
    }
    if crate::grounding::grounding_quality(requested) == "high" {
        Ok(requested)
    } else {
        Err(PokError::Tool(format!(
            "expected_label is required for uncertain target {target_id:?} at batch step {index}; no input was executed"
        )))
    }
}
