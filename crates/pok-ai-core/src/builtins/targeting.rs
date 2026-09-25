//! Acting on a grounded target: click targets, visual localization clicks, pointer moves and drags, raw input, and text entry.

use super::*;

#[derive(Debug, Deserialize, JsonSchema)]
pub(super) struct InputArgs {
    #[serde(default, alias = "observation_version")]
    pub(super) observation_id: Option<String>,
    pub(super) kind: String,
    #[serde(default)]
    pub(super) x: Option<i32>,
    #[serde(default)]
    pub(super) y: Option<i32>,
    #[serde(default)]
    pub(super) button: Option<MouseButton>,
    #[serde(default)]
    pub(super) text: Option<String>,
    #[serde(default, alias = "clear_current_text")]
    pub(super) replace_existing: bool,
    #[serde(default)]
    pub(super) key: Option<String>,
    #[serde(default)]
    pub(super) delta_x: Option<i32>,
    #[serde(default)]
    pub(super) delta_y: Option<i32>,
}

pub(super) struct SimulateInputTool;

pub(super) fn is_raw_click_kind(kind: &str) -> bool {
    matches!(
        kind.trim().to_ascii_lowercase().as_str(),
        "click" | "left_click" | "right_click" | "middle_click" | "double_click"
    )
}

#[derive(Debug, Deserialize, JsonSchema)]
pub(super) struct LocateVisualTargetArgs {
    /// Latest observation containing the control. Omit to use the current authoritative state.
    #[serde(default, alias = "observation_version")]
    pub(super) observation_id: Option<String>,
    /// Optional enlarged view returned by inspect_screen_region.
    #[serde(default)]
    pub(super) view_id: Option<String>,
    /// Visible name or concise description of the intended control.
    pub(super) label: String,
    /// Bounding box in pixels of the supplied model image.
    pub(super) x: i32,
    pub(super) y: i32,
    pub(super) width: u32,
    pub(super) height: u32,
}

pub(super) struct LocateVisualTargetTool;

#[derive(Debug, Deserialize, JsonSchema)]
pub(super) struct ClickLocalizedArgs {
    /// One-use token returned by locate_visual_target after visual confirmation.
    pub(super) localization_id: String,
    #[serde(default)]
    pub(super) button: Option<MouseButton>,
}

pub(super) struct ClickLocalizedTool;

#[derive(Debug, Deserialize, JsonSchema)]
pub(super) struct MovePointerArgs {
    #[serde(default, alias = "observation_version")]
    pub(super) observation_id: Option<String>,
    #[serde(default)]
    pub(super) view_id: Option<String>,
    /// X coordinate in pixels of the supplied model image.
    pub(super) x: i32,
    /// Y coordinate in pixels of the supplied model image.
    pub(super) y: i32,
}

pub(super) struct MovePointerTool;

#[derive(Debug, Deserialize, JsonSchema)]
pub(super) struct DragPointerArgs {
    #[serde(default, alias = "observation_version")]
    pub(super) observation_id: Option<String>,
    #[serde(default)]
    pub(super) view_id: Option<String>,
    /// Starting X coordinate in pixels of the supplied model image.
    pub(super) start_x: i32,
    /// Starting Y coordinate in pixels of the supplied model image.
    pub(super) start_y: i32,
    /// Ending X coordinate in pixels of the supplied model image.
    pub(super) end_x: i32,
    /// Ending Y coordinate in pixels of the supplied model image.
    pub(super) end_y: i32,
    #[serde(default)]
    pub(super) button: Option<MouseButton>,
    /// Gesture duration in milliseconds, clamped to 100..=2000.
    #[serde(default = "default_drag_duration_ms")]
    pub(super) duration_ms: u64,
}

pub(super) const fn default_drag_duration_ms() -> u64 {
    350
}

pub(super) struct DragPointerTool;

#[derive(Debug, Deserialize, JsonSchema)]
pub(super) struct DragTargetArgs {
    #[serde(default, alias = "observation_version")]
    pub(super) observation_id: Option<String>,
    #[serde(default)]
    pub(super) view_id: Option<String>,
    pub(super) source_target_id: TargetId,
    /// Visible or accessibility label for the source target.
    pub(super) expected_source_label: String,
    /// Horizontal displacement in pixels of the supplied model image.
    #[serde(default)]
    pub(super) delta_x: i32,
    /// Vertical displacement in pixels of the supplied model image.
    #[serde(default)]
    pub(super) delta_y: i32,
    #[serde(default)]
    pub(super) button: Option<MouseButton>,
    #[serde(default = "default_drag_duration_ms")]
    pub(super) duration_ms: u64,
}

pub(super) struct DragTargetTool;

#[derive(Debug, Clone, Copy, Default, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(super) enum TextEntryMode {
    #[default]
    Append,
    Replace,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub(super) struct TypeTextArgs {
    #[serde(default, alias = "observation_version")]
    pub(super) observation_id: Option<String>,
    pub(super) text: String,
    #[serde(default)]
    pub(super) target_id: Option<TargetId>,
    #[serde(default)]
    pub(super) mode: TextEntryMode,
}

pub(super) struct TypeTextTool;

#[derive(Debug, Deserialize, JsonSchema)]
pub(super) struct ClickTargetArgs {
    #[serde(default, alias = "observation_version")]
    pub(super) observation_id: Option<String>,
    /// Optional derived view returned by inspect_screen_region.
    #[serde(default)]
    pub(super) view_id: Option<String>,
    pub(super) target_id: TargetId,
    /// Visible control text or accessible name the model intends to activate.
    #[serde(default)]
    pub(super) expected_label: Option<String>,
    #[serde(default)]
    pub(super) button: Option<MouseButton>,
    /// Double-click instead of a single click, to open items such as folders
    /// and files in File Explorer where a single click only selects them.
    #[serde(default)]
    pub(super) double_click: bool,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(untagged)]
pub(super) enum TargetId {
    Text(String),
    Number(u32),
}

impl TargetId {
    pub(super) fn normalized(&self) -> String {
        match self {
            Self::Text(value) => value.trim().to_owned(),
            Self::Number(value) => value.to_string(),
        }
    }
}

pub(super) struct ClickTargetTool;
#[async_trait]
impl Tool for ClickTargetTool {
    fn name(&self) -> &'static str {
        "click_target"
    }
    fn description(&self) -> &'static str {
        "Click a numbered interaction target from the latest annotated capture. Supply expected_label as the visible text or accessible name you intend to activate. If the id and label disagree, the harness corrects only one unique fresh semantic match; ambiguity executes no input. button may be left (default), middle, or right. Set double_click to open items that a single click only selects, such as folders and files in File Explorer. Prefer this over guessing raw coordinates."
    }
    fn input_schema(&self) -> Value {
        let mut value = schema::<ClickTargetArgs>();
        if let Some(required) = value.get_mut("required").and_then(Value::as_array_mut)
            && !required.iter().any(|item| item == "expected_label")
        {
            required.push(json!("expected_label"));
        }
        value
    }
    fn risk(&self) -> RiskClass {
        RiskClass::DesktopInput
    }
    async fn execute(&self, args: Value, context: &ToolContext) -> Result<Value> {
        let args: ClickTargetArgs = serde_json::from_value(args)?;
        let observation = current_observation_or_view(
            context,
            args.observation_id.as_deref(),
            args.view_id.as_deref(),
            "click_target",
        )?;
        ensure_targeted_capture(&observation)?;
        let requested_target_id = args.target_id.normalized();
        let requested_target = observation
            .targets
            .iter()
            .find(|target| target.id == requested_target_id)
            .ok_or_else(|| {
                PokError::Tool(format!(
                    "unknown target {requested_target_id:?}; capture the window again and use a listed target id"
                ))
            })?;
        let expected_label = args
            .expected_label
            .as_deref()
            .map(str::trim)
            .filter(|label| !label.is_empty());
        let (target, resolution) = if let Some(expected_label) = expected_label {
            if crate::grounding::grounding_quality(requested_target) != "low"
                && label_match_score(expected_label, &requested_target.name) >= 2
            {
                (requested_target, "id_and_label_match")
            } else {
                let candidates = semantic_target_candidates(&observation, expected_label);
                if candidates.len() == 1 {
                    (candidates[0], "corrected_unique_label")
                } else {
                    return target_resolution_recovery(
                        &observation,
                        requested_target,
                        expected_label,
                        &candidates,
                        context,
                    );
                }
            }
        } else if crate::grounding::grounding_quality(requested_target) == "high" {
            (requested_target, "legacy_high_quality_id")
        } else {
            return target_resolution_recovery(&observation, requested_target, "", &[], context);
        };
        let target_id = target.id.clone();
        let (x, y) = target.click_point.unwrap_or((
            target.bounds.x + i32::try_from(target.bounds.width / 2).unwrap_or(i32::MAX),
            target.bounds.y + i32::try_from(target.bounds.height / 2).unwrap_or(i32::MAX),
        ));
        if !target.bounds.contains(x, y) {
            return Err(PokError::Tool(
                "target click point is outside its bounds".into(),
            ));
        }
        let target_label = target.name.clone();
        let selection_control = matches!(
            target.control_type.trim().to_ascii_lowercase().as_str(),
            "check box" | "checkbox" | "radio button" | "toggle" | "toggle button"
        );
        if target.selected == Some(true) && selection_control {
            let mut value = model_observation_value(&observation, context.annotate_targets)?;
            value["executed"] = json!(false);
            value["already_selected"] = json!(true);
            value["action"] = json!({
                "kind": "click_target",
                "target_id": target_id,
                "label": target_label,
                "expected_label": expected_label,
                "resolution": resolution,
            });
            value["state_change"] =
                state_change_value(&observation, &observation, &context.task_hint.lock());
            value["instruction"] = json!(
                "This selection control is already selected. Treat this step as satisfied and continue without clicking it again."
            );
            return Ok(value);
        }
        let focused_control = FocusedControl {
            app: observation
                .foreground_window
                .as_ref()
                .map_or_else(String::new, |window| window.process_name.clone()),
            label: target.name.clone(),
            control_type: target.control_type.clone(),
        };
        let model_action = json!({
            "kind": "click_target",
            "target_id": target_id,
            "label": target_label,
            "expected_label": expected_label,
            "resolution": resolution,
            "requested_target": {
                "id": requested_target_id,
                "label": requested_target.name,
            },
            "resolved_target": {
                "id": target.id,
                "label": target.name,
            },
            "mapping_succeeded": true,
        });
        let button = args.button.unwrap_or(MouseButton::Left);
        let action = if args.double_click {
            InputAction::DoubleClick { x, y, button }
        } else {
            InputAction::Click { x, y, button }
        };
        let result = execute_input(action, observation, context, model_action).await?;
        *context.focused_control.lock() = Some(focused_control);
        Ok(result)
    }
}

pub(super) fn label_tokens(value: &str) -> Vec<String> {
    value
        .to_ascii_lowercase()
        .split(|character: char| !character.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(str::to_owned)
        .collect()
}

pub(super) fn label_match_score(expected: &str, actual: &str) -> u8 {
    let expected = label_tokens(expected);
    let actual = label_tokens(actual);
    if expected.is_empty() {
        0
    } else if expected == actual {
        3
    } else if actual.starts_with(&expected) {
        2
    } else if actual
        .windows(expected.len())
        .any(|window| window == expected.as_slice())
    {
        1
    } else {
        0
    }
}

pub(super) fn semantic_target_candidates<'a>(
    observation: &'a Observation,
    expected_label: &str,
) -> Vec<&'a InteractionTarget> {
    let ranked = observation
        .targets
        .iter()
        .filter_map(|target| {
            let score = label_match_score(expected_label, &target.name);
            (target.actionable && crate::grounding::grounding_quality(target) != "low" && score > 0)
                .then_some((score, target))
        })
        .collect::<Vec<_>>();
    let best = ranked.iter().map(|(score, _)| *score).max().unwrap_or(0);
    ranked
        .into_iter()
        .filter_map(|(score, target)| (score == best).then_some(target))
        .take(9)
        .collect()
}

pub(super) fn target_resolution_recovery(
    observation: &Observation,
    requested: &InteractionTarget,
    expected_label: &str,
    candidates: &[&InteractionTarget],
    context: &ToolContext,
) -> Result<Value> {
    let target = observation
        .target
        .as_ref()
        .ok_or_else(|| PokError::Tool("observation contains no capture target".into()))?;
    let screenshot = observation
        .screenshots
        .first()
        .ok_or_else(|| PokError::Tool("observation contains no model image".into()))?;
    let candidate_values = candidates
        .iter()
        .take(8)
        .map(|candidate| {
            let bounds = local_rect(&candidate.bounds, &target.bounds, screenshot);
            json!({
                "id": candidate.id,
                "label": candidate.name,
                "role": candidate.control_type,
                "box": [bounds.x, bounds.y, bounds.width, bounds.height],
                "grounding_quality": crate::grounding::grounding_quality(candidate),
            })
        })
        .collect::<Vec<_>>();
    let mut value = model_observation_value(observation, context.annotate_targets)?;
    value["executed"] = json!(false);
    value["status"] = json!("recovery_required");
    value["recovery_required"] = json!(true);
    value["resolution"] = json!(if expected_label.is_empty() {
        "semantic_confirmation_required"
    } else if candidates.is_empty() {
        "no_semantic_match"
    } else {
        "ambiguous_semantic_match"
    });
    value["requested_target"] = json!({
        "id": requested.id,
        "label": requested.name,
        "grounding_quality": crate::grounding::grounding_quality(requested),
    });
    value["expected_label"] = json!(expected_label);
    value["candidate_targets"] = json!(candidate_values);
    value["instruction"] = json!(if expected_label.is_empty() {
        "No input was executed because this uncertain target requires expected_label. Use the visible text or accessible target name and retry."
    } else if candidates.is_empty() {
        "No input was executed because the selected target did not match the intended label. Query the current targets or inspect the relevant region before retrying."
    } else {
        "No input was executed because more than one fresh target matches the intended label. Inspect a candidate target or choose the exact listed id before retrying."
    });
    Ok(value)
}

pub(super) fn model_point_to_physical(
    observation: &Observation,
    x: i32,
    y: i32,
) -> Result<(i32, i32)> {
    ensure_targeted_capture(observation)?;
    let target = observation
        .target
        .as_ref()
        .ok_or_else(|| PokError::Tool("observation contains no capture target".into()))?;
    let screenshot = observation
        .screenshots
        .first()
        .ok_or_else(|| PokError::Tool("observation contains no model image".into()))?;
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
        target.bounds.x + scale_offset(x, screenshot.model_width, target.bounds.width),
        target.bounds.y + scale_offset(y, screenshot.model_height, target.bounds.height),
    ))
}

pub(super) fn checked_drag_duration(duration_ms: u64) -> Result<u64> {
    if !(100..=2_000).contains(&duration_ms) {
        return Err(PokError::Tool(
            "duration_ms must be between 100 and 2000".into(),
        ));
    }
    Ok(duration_ms)
}

pub(super) fn visual_localization_matches(
    proposed: &Rect,
    candidate: &Rect,
    iou_threshold: f32,
    containment_threshold: f32,
) -> bool {
    let intersection = intersection_area(proposed, candidate);
    if intersection == 0 {
        return false;
    }
    let proposed_area = u64::from(proposed.width) * u64::from(proposed.height);
    let candidate_area = u64::from(candidate.width) * u64::from(candidate.height);
    let union = proposed_area
        .saturating_add(candidate_area)
        .saturating_sub(intersection)
        .max(1);
    let iou = intersection as f32 / union as f32;
    let smaller_coverage = intersection as f32 / proposed_area.min(candidate_area).max(1) as f32;
    iou >= iou_threshold || smaller_coverage >= containment_threshold
}

pub(super) fn localization_candidate<'a>(
    observation: &'a Observation,
    proposed: &Rect,
    label: &str,
    iou_threshold: f32,
    containment_threshold: f32,
) -> Option<&'a InteractionTarget> {
    let proposed_center = (
        proposed.x + i32::try_from(proposed.width / 2).unwrap_or(i32::MAX),
        proposed.y + i32::try_from(proposed.height / 2).unwrap_or(i32::MAX),
    );
    let distance_squared = |target: &InteractionTarget| {
        let center_x = target.bounds.x + i32::try_from(target.bounds.width / 2).unwrap_or(i32::MAX);
        let center_y =
            target.bounds.y + i32::try_from(target.bounds.height / 2).unwrap_or(i32::MAX);
        let dx = i64::from(center_x) - i64::from(proposed_center.0);
        let dy = i64::from(center_y) - i64::from(proposed_center.1);
        dx.saturating_mul(dx).saturating_add(dy.saturating_mul(dy))
    };
    let proximity_limit = i64::from(proposed.width.max(proposed.height)) * 35 / 100;
    let proximity_limit_squared = proximity_limit.saturating_mul(proximity_limit);
    let mut candidates = observation
        .targets
        .iter()
        .filter(|target| {
            target.actionable
                && visual_localization_matches(
                    proposed,
                    &target.bounds,
                    iou_threshold,
                    containment_threshold,
                )
                && distance_squared(target) <= proximity_limit_squared
        })
        .collect::<Vec<_>>();
    candidates.sort_by_key(|target| {
        (
            distance_squared(target),
            std::cmp::Reverse(label_match_score(label, &target.name)),
        )
    });
    let best = *candidates.first()?;
    let Some(second) = candidates.get(1).copied() else {
        return Some(best);
    };
    let best_label = label_match_score(label, &best.name);
    let second_label = label_match_score(label, &second.name);
    if best_label > second_label {
        return Some(best);
    }
    let ambiguity_margin = i64::from(proposed.width.min(proposed.height) / 10).max(8);
    let separated = distance_squared(second).saturating_sub(distance_squared(best))
        >= ambiguity_margin.saturating_mul(ambiguity_margin);
    separated.then_some(best)
}

#[async_trait]
impl Tool for LocateVisualTargetTool {
    fn name(&self) -> &'static str {
        "locate_visual_target"
    }

    fn description(&self) -> &'static str {
        "Propose a labeled bounding box for a visible control before any coordinate click. Returns an enlarged confirmation image and a one-use localization_id. No input is executed. Use click_target instead when a matching numbered target exists."
    }

    fn input_schema(&self) -> Value {
        schema::<LocateVisualTargetArgs>()
    }

    fn risk(&self) -> RiskClass {
        RiskClass::ReadOnly
    }

    async fn execute(&self, args: Value, context: &ToolContext) -> Result<Value> {
        let args: LocateVisualTargetArgs = serde_json::from_value(args)?;
        let label = args.label.trim();
        if label.is_empty() {
            return Err(PokError::Tool("label must not be empty".into()));
        }
        let source = current_observation(
            context,
            args.observation_id.as_deref(),
            "locate_visual_target",
        )?;
        let observation = current_observation_or_view(
            context,
            args.observation_id.as_deref(),
            args.view_id.as_deref(),
            "locate_visual_target",
        )?;
        ensure_targeted_capture(&observation)?;
        let target = observation
            .target
            .as_ref()
            .ok_or_else(|| PokError::Tool("observation contains no capture target".into()))?;
        let screenshot = observation
            .screenshots
            .first()
            .ok_or_else(|| PokError::Tool("observation contains no model image".into()))?;
        let proposed_model_bounds = clamp_model_rect(
            Rect {
                x: args.x,
                y: args.y,
                width: args.width,
                height: args.height,
            },
            screenshot.model_width,
            screenshot.model_height,
        )?;
        let proposed_physical_bounds = model_rect_to_physical(
            &proposed_model_bounds,
            screenshot.model_width,
            screenshot.model_height,
            &target.bounds,
        )?;
        let matched = localization_candidate(
            &observation,
            &proposed_physical_bounds,
            label,
            context.fusion_iou_threshold,
            context.ocr_containment_threshold,
        );
        let (resolved_physical_bounds, resolved_physical_point, corroboration, matched_target) =
            if let Some(candidate) = matched {
                let point = if candidate.actionable {
                    candidate.click_point.unwrap_or((
                        candidate.bounds.x
                            + i32::try_from(candidate.bounds.width / 2).unwrap_or(i32::MAX),
                        candidate.bounds.y
                            + i32::try_from(candidate.bounds.height / 2).unwrap_or(i32::MAX),
                    ))
                } else {
                    (
                        proposed_physical_bounds.x
                            + i32::try_from(proposed_physical_bounds.width / 2).unwrap_or(i32::MAX),
                        proposed_physical_bounds.y
                            + i32::try_from(proposed_physical_bounds.height / 2)
                                .unwrap_or(i32::MAX),
                    )
                };
                (
                    if candidate.actionable {
                        candidate.bounds.clone()
                    } else {
                        proposed_physical_bounds.clone()
                    },
                    point,
                    if candidate.actionable {
                        "actionable_target_geometry"
                    } else {
                        "ocr_label"
                    },
                    Some(json!({
                        "target_id": candidate.id,
                        "label": candidate.name,
                        "source": candidate.source,
                        "actionable": candidate.actionable,
                    })),
                )
            } else {
                (
                    proposed_physical_bounds.clone(),
                    (
                        proposed_physical_bounds.x
                            + i32::try_from(proposed_physical_bounds.width / 2).unwrap_or(i32::MAX),
                        proposed_physical_bounds.y
                            + i32::try_from(proposed_physical_bounds.height / 2)
                                .unwrap_or(i32::MAX),
                    ),
                    "model_only",
                    None,
                )
            };
        if !resolved_physical_bounds.contains(resolved_physical_point.0, resolved_physical_point.1)
        {
            return Err(PokError::Tool(
                "resolved click point is outside the localized bounds".into(),
            ));
        }

        let padding = 32_u32;
        let left = proposed_model_bounds
            .x
            .saturating_sub(padding as i32)
            .max(0);
        let top = proposed_model_bounds
            .y
            .saturating_sub(padding as i32)
            .max(0);
        let right = (i64::from(proposed_model_bounds.x)
            + i64::from(proposed_model_bounds.width)
            + i64::from(padding))
        .min(i64::from(screenshot.model_width));
        let bottom = (i64::from(proposed_model_bounds.y)
            + i64::from(proposed_model_bounds.height)
            + i64::from(padding))
        .min(i64::from(screenshot.model_height));
        let confirmation_model_bounds = Rect {
            x: left,
            y: top,
            width: u32::try_from((right - i64::from(left)).max(1)).unwrap_or(u32::MAX),
            height: u32::try_from((bottom - i64::from(top)).max(1)).unwrap_or(u32::MAX),
        };
        let confirmation_physical_bounds = model_rect_to_physical(
            &confirmation_model_bounds,
            screenshot.model_width,
            screenshot.model_height,
            &target.bounds,
        )?;
        let localization_id = format!("loc_{}", &Uuid::new_v4().simple().to_string()[..8]);
        let source_png_name = format!("localization-{localization_id}-source.png");
        let confirmation_png_name = format!("localization-{localization_id}-confirmation.png");
        let mut annotated_source = DesktopCapture {
            target: target.clone(),
            screenshots: observation.screenshots.clone(),
            timings_ms: Default::default(),
        };
        annotate_localization_capture(
            &mut annotated_source,
            &proposed_physical_bounds,
            &target.bounds,
            resolved_physical_point,
        )?;
        save_localization_png(
            context,
            &source_png_name,
            &annotated_source
                .screenshots
                .first()
                .ok_or_else(|| PokError::Tool("annotated source has no image".into()))?
                .png_base64,
        )?;
        let mut capture = derived_capture(
            &observation,
            &confirmation_physical_bounds,
            &localization_id,
            context.vision_max_edge,
        )?;
        annotate_localization_capture(
            &mut capture,
            &proposed_physical_bounds,
            &confirmation_physical_bounds,
            resolved_physical_point,
        )?;
        save_localization_png(
            context,
            &confirmation_png_name,
            &capture
                .screenshots
                .first()
                .ok_or_else(|| PokError::Tool("confirmation crop has no image".into()))?
                .png_base64,
        )?;
        let request = CaptureRequest {
            scope: CaptureScope::Region,
            window_id: None,
            monitor_id: None,
            region: Some(confirmation_physical_bounds),
            max_edge: context.vision_max_edge,
        };
        let mut confirmation = context
            .platform
            .observe_capture(
                capture,
                &request,
                true,
                false,
                0,
                0,
                Duration::from_millis(context.desktop_deep_enrichment_timeout_ms),
            )
            .await?;
        confirmation.version = source.version;
        confirmation.targets = build_targets(
            &confirmation,
            context.model_target_limit,
            context.fusion_iou_threshold,
            context.ocr_containment_threshold,
            &context.task_hint.lock(),
        );
        *context.latest_observation_view.lock() = Some(DerivedObservationView {
            id: localization_id.clone(),
            source_observation_id: observation_id_for(&source),
            observation: confirmation.clone(),
        });
        *context.pending_visual_localization.lock() = Some(PendingVisualLocalization {
            id: localization_id.clone(),
            source_observation_id: observation_id_for(&source),
            source_observation: source,
            label: label.to_owned(),
            proposed_model_bounds: proposed_model_bounds.clone(),
            resolved_physical_bounds: resolved_physical_bounds.clone(),
            resolved_physical_point,
            corroboration: corroboration.into(),
        });
        context.write_artifact(
            &format!("localization-{localization_id}.json"),
            &json!({
                "localization_id": localization_id,
                "label": label,
                "proposed_model_bounds": proposed_model_bounds,
                "resolved_physical_bounds": resolved_physical_bounds,
                "resolved_physical_point": resolved_physical_point,
                "corroboration": corroboration,
                "matched_target": matched_target,
                "diagnostic_artifacts": {
                    "annotated_source_png": source_png_name,
                    "confirmation_crop_png": confirmation_png_name,
                },
                "source_observation_id": observation_id_for(
                    context.latest_observation.lock().as_ref().expect("source exists")
                ),
            }),
        )?;
        let mut value = model_observation_value(&confirmation, context.annotate_targets)?;
        value["localization_id"] = json!(localization_id);
        value["label"] = json!(label);
        value["proposed_model_bounds"] = json!(proposed_model_bounds);
        value["resolved_physical_bounds"] = json!("<internal>");
        value["corroboration"] = json!(corroboration);
        value["matched_target"] = matched_target.unwrap_or(Value::Null);
        value["diagnostic_artifacts"] = json!({
            "annotated_source_png": source_png_name,
            "confirmation_crop_png": confirmation_png_name,
        });
        value["confirmation_required"] = json!(true);
        value["instruction"] = json!(
            "Inspect this enlarged crop. If it shows the intended control, call click_localized with localization_id. Otherwise call locate_visual_target again with a corrected box. Do not call capture_screen."
        );
        Ok(value)
    }
}

#[async_trait]
impl Tool for ClickLocalizedTool {
    fn name(&self) -> &'static str {
        "click_localized"
    }

    fn description(&self) -> &'static str {
        "Confirm and click the center of a fresh one-use localization returned by locate_visual_target. Takes no coordinates."
    }

    fn input_schema(&self) -> Value {
        schema::<ClickLocalizedArgs>()
    }

    fn risk(&self) -> RiskClass {
        RiskClass::DesktopInput
    }

    async fn execute(&self, args: Value, context: &ToolContext) -> Result<Value> {
        let args: ClickLocalizedArgs = serde_json::from_value(args)?;
        let current = current_observation(context, None, "click_localized")?;
        let pending = context
            .pending_visual_localization
            .lock()
            .clone()
            .ok_or_else(|| {
                PokError::Tool(
                    "no pending visual localization; call locate_visual_target first".into(),
                )
            })?;
        if pending.id != args.localization_id {
            return Err(PokError::Tool(format!(
                "stale localization {:?}; confirm the latest localization or locate again",
                args.localization_id
            )));
        }
        if pending.source_observation_id != observation_id_for(&current) {
            *context.pending_visual_localization.lock() = None;
            return Err(PokError::Tool(
                "the desktop observation changed after localization; locate the control again"
                    .into(),
            ));
        }
        let foreground = context.platform.foreground_window().await?;
        let expected = pending.source_observation.foreground_window.as_ref();
        if expected.is_some_and(|expected| !foreground_matches(&expected.id, foreground.as_ref())) {
            *context.pending_visual_localization.lock() = None;
            return Err(PokError::Tool(
                "desktop focus changed after localization; capture or activate the intended window and locate again"
                    .into(),
            ));
        }
        *context.pending_visual_localization.lock() = None;
        execute_input(
            InputAction::Click {
                x: pending.resolved_physical_point.0,
                y: pending.resolved_physical_point.1,
                button: args.button.unwrap_or(MouseButton::Left),
            },
            pending.source_observation,
            context,
            json!({
                "kind": "click_localized",
                "localization_id": pending.id,
                "label": pending.label,
                "proposed_model_bounds": pending.proposed_model_bounds,
                "resolved_bounds": "<internal>",
                "corroboration": pending.corroboration,
                "mapping_succeeded": true,
            }),
        )
        .await
    }
}

#[async_trait]
impl Tool for MovePointerTool {
    fn name(&self) -> &'static str {
        "move_pointer"
    }

    fn description(&self) -> &'static str {
        "Move the mouse to x/y pixels of the latest supplied model image without clicking. Cursor movement is not task progress by itself."
    }

    fn input_schema(&self) -> Value {
        schema::<MovePointerArgs>()
    }

    fn risk(&self) -> RiskClass {
        RiskClass::DesktopInput
    }

    async fn execute(&self, args: Value, context: &ToolContext) -> Result<Value> {
        let args: MovePointerArgs = serde_json::from_value(args)?;
        let observation = current_observation_or_view(
            context,
            args.observation_id.as_deref(),
            args.view_id.as_deref(),
            "move_pointer",
        )?;
        let (x, y) = model_point_to_physical(&observation, args.x, args.y)?;
        execute_input(
            InputAction::Move { x, y },
            observation,
            context,
            json!({
                "kind": "move",
                "model_point": [args.x, args.y],
                "mapping_succeeded": true,
            }),
        )
        .await
    }
}

#[async_trait]
impl Tool for DragPointerTool {
    fn name(&self) -> &'static str {
        "drag_pointer"
    }

    fn description(&self) -> &'static str {
        "Atomically drag between two points expressed in pixels of the latest supplied model image. Prefer drag_target when the source has a numbered target."
    }

    fn input_schema(&self) -> Value {
        schema::<DragPointerArgs>()
    }

    fn risk(&self) -> RiskClass {
        RiskClass::DesktopInput
    }

    async fn execute(&self, args: Value, context: &ToolContext) -> Result<Value> {
        let args: DragPointerArgs = serde_json::from_value(args)?;
        let duration_ms = checked_drag_duration(args.duration_ms)?;
        let observation = current_observation_or_view(
            context,
            args.observation_id.as_deref(),
            args.view_id.as_deref(),
            "drag_pointer",
        )?;
        let (start_x, start_y) = model_point_to_physical(&observation, args.start_x, args.start_y)?;
        let (end_x, end_y) = model_point_to_physical(&observation, args.end_x, args.end_y)?;
        let button = args.button.unwrap_or(MouseButton::Left);
        execute_input(
            InputAction::Drag {
                start_x,
                start_y,
                end_x,
                end_y,
                button,
                duration_ms,
            },
            observation,
            context,
            json!({
                "kind": "drag",
                "model_start": [args.start_x, args.start_y],
                "model_end": [args.end_x, args.end_y],
                "button": button,
                "duration_ms": duration_ms,
                "mapping_succeeded": true,
            }),
        )
        .await
    }
}

#[async_trait]
impl Tool for DragTargetTool {
    fn name(&self) -> &'static str {
        "drag_target"
    }

    fn description(&self) -> &'static str {
        "Drag a numbered target by a model-image pixel displacement. Supply the expected source label; the harness executes only a matching or uniquely corrected target. Useful for sliders and other custom controls."
    }

    fn input_schema(&self) -> Value {
        schema::<DragTargetArgs>()
    }

    fn risk(&self) -> RiskClass {
        RiskClass::DesktopInput
    }

    async fn execute(&self, args: Value, context: &ToolContext) -> Result<Value> {
        let args: DragTargetArgs = serde_json::from_value(args)?;
        if args.delta_x == 0 && args.delta_y == 0 {
            return Err(PokError::Tool(
                "drag_target requires non-zero delta_x or delta_y".into(),
            ));
        }
        let duration_ms = checked_drag_duration(args.duration_ms)?;
        let observation = current_observation_or_view(
            context,
            args.observation_id.as_deref(),
            args.view_id.as_deref(),
            "drag_target",
        )?;
        ensure_targeted_capture(&observation)?;
        let requested_id = args.source_target_id.normalized();
        let requested = observation
            .targets
            .iter()
            .find(|target| target.id == requested_id)
            .ok_or_else(|| PokError::Tool(format!("unknown source target {requested_id:?}")))?;
        let expected = args.expected_source_label.trim();
        if expected.is_empty() {
            return Err(PokError::Tool(
                "expected_source_label cannot be empty".into(),
            ));
        }
        let (target, resolution) = if crate::grounding::grounding_quality(requested) != "low"
            && label_match_score(expected, &requested.name) >= 2
        {
            (requested, "id_and_label_match")
        } else {
            let candidates = semantic_target_candidates(&observation, expected);
            if candidates.len() != 1 {
                return target_resolution_recovery(
                    &observation,
                    requested,
                    expected,
                    &candidates,
                    context,
                );
            }
            (candidates[0], "corrected_unique_label")
        };
        let screenshot = observation
            .screenshots
            .first()
            .ok_or_else(|| PokError::Tool("observation contains no model image".into()))?;
        let capture = observation
            .target
            .as_ref()
            .ok_or_else(|| PokError::Tool("observation contains no capture target".into()))?;
        let (start_x, start_y) = target.click_point.unwrap_or((
            target.bounds.x + i32::try_from(target.bounds.width / 2).unwrap_or(i32::MAX),
            target.bounds.y + i32::try_from(target.bounds.height / 2).unwrap_or(i32::MAX),
        ));
        let end_x = start_x.saturating_add(scale_offset(
            args.delta_x,
            screenshot.model_width,
            capture.bounds.width,
        ));
        let end_y = start_y.saturating_add(scale_offset(
            args.delta_y,
            screenshot.model_height,
            capture.bounds.height,
        ));
        let button = args.button.unwrap_or(MouseButton::Left);
        let target_id = target.id.clone();
        let target_label = target.name.clone();
        let expected = expected.to_owned();
        execute_input(
            InputAction::Drag {
                start_x,
                start_y,
                end_x,
                end_y,
                button,
                duration_ms,
            },
            observation,
            context,
            json!({
                "kind": "drag_target",
                "target_id": target_id,
                "label": target_label,
                "expected_label": expected,
                "resolution": resolution,
                "model_delta": [args.delta_x, args.delta_y],
                "button": button,
                "duration_ms": duration_ms,
                "mapping_succeeded": true,
            }),
        )
        .await
    }
}

#[async_trait]
impl Tool for SimulateInputTool {
    fn name(&self) -> &'static str {
        "simulate_input"
    }
    fn description(&self) -> &'static str {
        "Perform one validated keyboard or non-click mouse action against the latest observed foreground window. Send an entire string with kind=type_text, or use kind=key for a named key/chord. Coordinate clicks are rejected: use click_target or locate_visual_target followed by click_localized. Mouse move and legacy raw scroll remain available."
    }
    fn input_schema(&self) -> Value {
        schema::<InputArgs>()
    }
    fn risk(&self) -> RiskClass {
        RiskClass::DesktopInput
    }
    async fn execute(&self, args: Value, context: &ToolContext) -> Result<Value> {
        let args: InputArgs = serde_json::from_value(args)?;
        if is_raw_click_kind(&args.kind) {
            return Err(PokError::Tool(
                "raw coordinate clicks require locate_visual_target followed by click_localized; use click_target for a numbered control"
                    .into(),
            ));
        }
        let observation =
            current_observation(context, args.observation_id.as_deref(), "simulate_input")?;
        ensure_targeted_capture(&observation)?;
        let target = observation
            .target
            .as_ref()
            .expect("validated capture target");
        let screenshot = observation
            .screenshots
            .first()
            .ok_or_else(|| PokError::Tool("observation has no model image metadata".into()))?;
        let action = input_action(&args, &target.bounds, screenshot)?;
        let model_action = model_action_value(&args);
        execute_input(action, observation, context, model_action).await
    }
}

#[async_trait]
impl Tool for TypeTextTool {
    fn name(&self) -> &'static str {
        "type_text"
    }

    fn description(&self) -> &'static str {
        "Type the complete desired string into the focused control in one call and verify it by UI Automation. Provide target_id of the listed text field whenever you can: native fields (forms, search boxes, file name boxes in Save dialogs) are then filled directly without using the user's keyboard or mouse, even in a background window; other fields are focused and typed into. mode=append preserves existing text; mode=replace clears/replaces it. Line breaks in the text are entered as Shift+Enter, a new line that does not send a chat message. This tool never presses Enter: to send a message or submit a form, press Enter as a separate key action after the text is verified. Desktop forms often look up or validate a field (a ZIP code filling in the city) only on the key that leaves it: press Tab after entering such a value."
    }

    fn input_schema(&self) -> Value {
        schema::<TypeTextArgs>()
    }

    fn risk(&self) -> RiskClass {
        RiskClass::DesktopInput
    }

    async fn execute(&self, args: Value, context: &ToolContext) -> Result<Value> {
        let args: TypeTextArgs = serde_json::from_value(args)?;
        if args.text.is_empty() {
            return Err(PokError::Tool("text must not be empty".into()));
        }
        let observation =
            current_observation(context, args.observation_id.as_deref(), "type_text")?;
        ensure_targeted_capture(&observation)?;
        let append = matches!(args.mode, TextEntryMode::Append);
        let mut model_action = json!({
            "kind": "type_text",
            "text": args.text,
            "mode": if append { "append" } else { "replace" },
        });
        if let Some(target_id) = args.target_id.as_ref().map(TargetId::normalized) {
            let target = observation
                .targets
                .iter()
                .find(|target| target.id == target_id)
                .ok_or_else(|| {
                    PokError::Tool(format!(
                        "unknown target {target_id:?}; capture again and use a listed target id"
                    ))
                })?;
            let (x, y) = target.click_point.unwrap_or((
                target.bounds.x + i32::try_from(target.bounds.width / 2).unwrap_or(i32::MAX),
                target.bounds.y + i32::try_from(target.bounds.height / 2).unwrap_or(i32::MAX),
            ));
            if !accepts_typed_text(&target.control_type) {
                return Err(PokError::Tool(format!(
                    "target {:?} is a {} ({:?}), not a text field; typing there would \
                     select or rename it instead of entering text. Use the listed edit or \
                     combo box target for the field, or omit target_id to type into the \
                     control that already has keyboard focus",
                    target.id, target.control_type, target.name
                )));
            }
            model_action["target_id"] = json!(target.id);
            // Without the keyboard first: a native text field takes the text
            // through its Value pattern, even in a window behind the user's.
            if let Some(entered) =
                pattern_text(context, &observation, target, (x, y), &args.text, append).await?
            {
                model_action["pattern_text"] = serde_json::to_value(entered)?;
                return execute_input(
                    InputAction::TypeText {
                        text: args.text.clone(),
                        replace_existing: !append,
                    },
                    observation,
                    context,
                    model_action,
                )
                .await;
            }
            // Otherwise focus the field (without the mouse where the control
            // allows it) and type.
            let click = InputAction::Click {
                x,
                y,
                button: MouseButton::Left,
            };
            if pattern_input(
                context,
                &click,
                &observation,
                &json!({"target_id": target.id}),
            )
            .await?
            .is_none()
            {
                simulate_input_guarded(context, &click, &observation).await?;
            }
            *context.focused_control.lock() = Some(FocusedControl {
                app: observation
                    .foreground_window
                    .as_ref()
                    .map_or_else(String::new, |window| window.process_name.clone()),
                label: target.name.clone(),
                control_type: target.control_type.clone(),
            });
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        execute_input(
            InputAction::TypeText {
                text: args.text.clone(),
                replace_existing: !append,
            },
            observation,
            context,
            model_action,
        )
        .await
    }
}

/// Controls whose activation selects, opens or toggles something rather than
/// placing a caret: typing "into" them selects items by type-ahead or starts
/// an in-place rename.
pub(super) fn accepts_typed_text(control_type: &str) -> bool {
    !matches!(
        control_type.to_ascii_lowercase().as_str(),
        "list item"
            | "listitem"
            | "tree item"
            | "treeitem"
            | "data item"
            | "button"
            | "split button"
            | "menu item"
            | "tab item"
            | "hyperlink"
            | "link"
            | "check box"
            | "checkbox"
            | "radio button"
    )
}

/// Enter text into a native field through its Value pattern, without the
/// keyboard. `None` when the field does not accept it.
pub(super) async fn pattern_text(
    context: &ToolContext,
    observation: &Observation,
    target: &InteractionTarget,
    point: (i32, i32),
    text: &str,
    append: bool,
) -> Result<Option<crate::platform::PatternTextOutcome>> {
    if !matches!(target.source, TargetSource::Uia | TargetSource::UiaOcr) {
        return Ok(None);
    }
    let Some(window) = observation.foreground_window.as_ref() else {
        return Ok(None);
    };
    context.pause.ensure_action_allowed()?;
    let current = context.platform.foreground_window().await?;
    if !foreground_matches(&window.id, current.as_ref()) {
        return Ok(None);
    }
    // The same checks as typed input: no password fields, no elevated
    // windows, and no typed browser navigation or duplicate submission.
    let typed = InputAction::TypeText {
        text: text.to_owned(),
        replace_existing: !append,
    };
    context.policy.validate_input(&typed, observation)?;
    validate_browser_navigation(
        text,
        &context.task_hint.lock(),
        context.focused_control.lock().as_ref(),
    )?;
    let (destination, _) = field_destination(observation, Some(target));
    if duplicate_submission_value(&context.input_ledger.lock(), &destination, text, !append)
        .is_some()
    {
        return Ok(None);
    }
    Ok(context
        .platform
        .perform_pattern_text(&crate::platform::PatternTextRequest {
            window_id: window.id.clone(),
            name: target.name.clone(),
            bounds: target.bounds.clone(),
            point,
            text: text.to_owned(),
            append,
        })
        .await
        .ok()
        .flatten())
}
