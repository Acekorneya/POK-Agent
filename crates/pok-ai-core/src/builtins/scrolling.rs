//! Semantic scrolling: scroll tools, verified scrolling with cursor-free patterns, wheel, keyboard, and scrollbar fallbacks.

use super::*;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(super) enum ScrollDirection {
    Up,
    Down,
    Left,
    Right,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(super) enum ScrollAmount {
    Small,
    #[default]
    Page,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub(super) struct ScrollViewArgs {
    #[serde(default, alias = "observation_version")]
    pub(super) observation_id: Option<String>,
    pub(super) direction: ScrollDirection,
    #[serde(default)]
    pub(super) amount: ScrollAmount,
    #[serde(default = "one_scroll", alias = "count")]
    pub(super) repeat: u8,
    #[serde(default)]
    pub(super) target_id: Option<TargetId>,
}

pub(super) const fn one_scroll() -> u8 {
    1
}

#[derive(Debug, Deserialize, JsonSchema)]
pub(super) struct ScrollUntilTextArgs {
    #[serde(default, alias = "observation_version")]
    pub(super) observation_id: Option<String>,
    pub(super) query: String,
    #[serde(default = "scroll_down")]
    pub(super) direction: ScrollDirection,
    #[serde(default = "default_scroll_pages")]
    pub(super) max_pages: u8,
    #[serde(default)]
    pub(super) target_id: Option<TargetId>,
}

pub(super) const fn scroll_down() -> ScrollDirection {
    ScrollDirection::Down
}

pub(super) const fn default_scroll_pages() -> u8 {
    8
}

pub(super) struct ScrollViewTool;

pub(super) struct ScrollAttempt {
    pub(super) observation: Observation,
    pub(super) attempts: Vec<Value>,
    pub(super) method: Option<&'static str>,
    pub(super) viewport_changed: bool,
    pub(super) edge_reached: bool,
}

#[async_trait]
impl Tool for ScrollViewTool {
    fn name(&self) -> &'static str {
        "scroll_view"
    }

    fn description(&self) -> &'static str {
        "Scroll the latest observed Windows page, list, dialog, or POS panel using semantic direction and distance. Use amount=page for normal navigation or small for fine movement. Optionally target a fresh numbered control. The tool verifies content movement and falls back from the wheel to keyboard and then a positively detected scrollbar. It automatically recaptures and returns the new grounded view, scroll_method, attempts, and stop_reason; do not call capture_screen immediately afterward."
    }

    fn input_schema(&self) -> Value {
        schema::<ScrollViewArgs>()
    }

    fn risk(&self) -> RiskClass {
        RiskClass::DesktopInput
    }

    async fn execute(&self, args: Value, context: &ToolContext) -> Result<Value> {
        let args: ScrollViewArgs = serde_json::from_value(args)?;
        if !(1..=5).contains(&args.repeat) {
            return Err(PokError::Tool("repeat must be between 1 and 5".into()));
        }
        let before = current_observation(context, args.observation_id.as_deref(), "scroll_view")?;
        ensure_targeted_capture(&before)?;
        let inferred_target = args
            .target_id
            .is_none()
            .then(|| infer_scroll_target(&before))
            .flatten();
        let effective_target = args
            .target_id
            .as_ref()
            .or(inferred_target.as_ref().map(|target| &target.0));
        let (point, target_label) = scroll_point(&before, effective_target)?;
        let notches = scroll_notches(args.direction, args.amount);
        let initial_observation = before.clone();
        let mut observation = before;
        let mut attempts = Vec::new();
        let mut scroll_method = None;
        let mut edge_reached = false;
        for _ in 0..args.repeat {
            let result =
                verified_scroll(context, observation, point, args.direction, args.amount).await?;
            observation = result.observation;
            attempts.extend(result.attempts);
            if result.viewport_changed {
                scroll_method = result.method;
            } else {
                edge_reached = result.edge_reached;
                break;
            }
        }
        let viewport_changed =
            observation_view_changed(&initial_observation, &observation, args.direction);
        let stop_reason = if viewport_changed {
            "scrolled"
        } else if edge_reached {
            "edge_reached"
        } else {
            "no_scroll_effect"
        };
        let mut value = model_observation_value(&observation, context.annotate_targets)?;
        value["executed"] = json!(true);
        value["action"] = json!({
            "kind": "scroll",
            "direction": args.direction,
            "amount": args.amount,
            "repeat": args.repeat,
            "wheel_notches": {"x": notches.0, "y": notches.1},
            "target_id": args.target_id.as_ref().map(TargetId::normalized),
            "target_label": target_label,
            "inferred_target_id": inferred_target.as_ref().map(|target| target.0.normalized()),
            "inferred_target_basis": inferred_target.as_ref().map(|target| target.1.clone()),
        });
        value["attempts"] = Value::Array(attempts);
        value["scroll_method"] =
            scroll_method.map_or(Value::Null, |method| Value::String(method.to_owned()));
        value["stop_reason"] = json!(stop_reason);
        value["viewport_changed"] = json!(viewport_changed);
        value["edge_reached"] = json!(edge_reached);
        if !viewport_changed {
            value["suggested_scroll_targets"] = suggested_scroll_targets(&initial_observation);
            value["instruction"] = json!(
                "The untargeted scroll did not move the intended content. Retry with target_id from suggested_scroll_targets or use scroll_until_text with the known label."
            );
        }
        value["state_change"] = state_change_value(
            &initial_observation,
            &observation,
            &context.task_hint.lock(),
        );
        Ok(value)
    }
}

pub(super) fn scroll_target_candidate(target: &InteractionTarget) -> bool {
    matches!(
        target.control_type.to_ascii_lowercase().as_str(),
        "list" | "list item" | "tree" | "tree item" | "navigation" | "pane" | "document" | "group"
    ) || target.actionable
}

/// The main document area (the web page in a browser), when the window has
/// one covering a good part of it.
pub(super) fn main_document_region(observation: &Observation) -> Option<Rect> {
    let window = observation.target.as_ref()?.bounds.clone();
    let window_area = f64::from(window.width) * f64::from(window.height);
    observation
        .ui_elements
        .iter()
        .filter(|element| {
            !element.offscreen && element.control_type.eq_ignore_ascii_case("document")
        })
        .map(|element| element.bounds.clone())
        .filter(|bounds| {
            window_area > 0.0
                && f64::from(bounds.width) * f64::from(bounds.height) >= window_area * 0.3
        })
        .max_by_key(|bounds| u64::from(bounds.width) * u64::from(bounds.height))
}

fn centre_inside(inner: &Rect, outer: &Rect) -> bool {
    let x = i64::from(inner.x) + i64::from(inner.width) / 2;
    let y = i64::from(inner.y) + i64::from(inner.height) / 2;
    x >= i64::from(outer.x)
        && y >= i64::from(outer.y)
        && x < i64::from(outer.x) + i64::from(outer.width)
        && y < i64::from(outer.y) + i64::from(outer.height)
}

pub(super) fn infer_scroll_target(observation: &Observation) -> Option<(TargetId, String)> {
    // Never scroll over browser chrome (tabs, toolbar, bookmarks bar): a
    // bookmark whose name shares a word with the task once received every
    // wheel notch meant for the page.
    let document = main_document_region(observation);
    let mut candidates = observation
        .targets
        .iter()
        .filter(|target| target.rank_score >= 10 && scroll_target_candidate(target))
        .filter(|target| {
            document
                .as_ref()
                .is_none_or(|region| centre_inside(&target.bounds, region))
        })
        .collect::<Vec<_>>();
    candidates.sort_by_key(|target| std::cmp::Reverse(target.rank_score));
    let best = candidates.first()?;
    if candidates
        .get(1)
        .is_some_and(|next| next.rank_score == best.rank_score)
    {
        return None;
    }
    Some((
        TargetId::Text(best.id.clone()),
        format!(
            "unique task-matching {} target {:?}",
            best.control_type, best.name
        ),
    ))
}

pub(super) fn suggested_scroll_targets(observation: &Observation) -> Value {
    let mut candidates = observation
        .targets
        .iter()
        .filter(|target| scroll_target_candidate(target))
        .collect::<Vec<_>>();
    candidates.sort_by_key(|target| std::cmp::Reverse(target.rank_score));
    Value::Array(
        candidates
            .into_iter()
            .take(8)
            .map(|target| {
                json!({
                    "id": target.id,
                    "label": target.name,
                    "role": target.control_type,
                    "rank_score": target.rank_score,
                })
            })
            .collect(),
    )
}

pub(super) struct ScrollUntilTextTool;

#[async_trait]
impl Tool for ScrollUntilTextTool {
    fn name(&self) -> &'static str {
        "scroll_until_text"
    }

    fn description(&self) -> &'static str {
        "Find visible text in a Windows page, list, dialog, or POS screen by scrolling a bounded number of verified pages. Use | for literal alternatives, for example temperature|Temp|°F|Current. It searches OCR, UI Automation, and numbered targets after every page, uses safe keyboard and detected-scrollbar fallbacks when wheel input has no effect, and returns the final grounded view and stop_reason. Prefer this when you know the label or text you need."
    }

    fn input_schema(&self) -> Value {
        schema::<ScrollUntilTextArgs>()
    }

    fn risk(&self) -> RiskClass {
        RiskClass::DesktopInput
    }

    async fn execute(&self, args: Value, context: &ToolContext) -> Result<Value> {
        let args: ScrollUntilTextArgs = serde_json::from_value(args)?;
        let queries = scroll_query_alternatives(&args.query)?;
        if !(1..=20).contains(&args.max_pages) {
            return Err(PokError::Tool("max_pages must be between 1 and 20".into()));
        }
        let mut observation =
            current_observation(context, args.observation_id.as_deref(), "scroll_until_text")?;
        ensure_targeted_capture(&observation)?;
        let initial_observation = observation.clone();
        let (point, target_label) = scroll_point(&observation, args.target_id.as_ref())?;
        let scan_started = Instant::now();
        let mut page_timings = Vec::new();
        let mut matched = visible_text_match_any(&observation, &queries);
        let mut pages_scrolled = 0_u8;
        let mut attempts = Vec::new();
        let mut scroll_method = None;
        let mut stop_reason = if matched.is_some() {
            "found"
        } else {
            "max_pages"
        };

        while matched.is_none() && pages_scrolled < args.max_pages {
            if context.cancellation.is_cancelled() {
                return Err(PokError::Cancelled);
            }
            let result = verified_scroll(
                context,
                observation,
                point,
                args.direction,
                ScrollAmount::Page,
            )
            .await?;
            observation = result.observation;
            page_timings.push(
                observation
                    .timings_ms
                    .get("total")
                    .copied()
                    .unwrap_or_default(),
            );
            attempts.extend(result.attempts);
            if !result.viewport_changed {
                stop_reason = if result.edge_reached {
                    "edge_reached"
                } else {
                    "no_scroll_effect"
                };
                break;
            }
            scroll_method = result.method;
            pages_scrolled = pages_scrolled.saturating_add(1);
            matched = visible_text_match_any(&observation, &queries);
            if matched.is_some() {
                stop_reason = "found";
                break;
            }
        }

        let mut value = model_observation_value(&observation, context.annotate_targets)?;
        value["executed"] = json!(true);
        value["action"] = json!({
            "kind": "scroll_until_text",
            "query": args.query,
            "direction": args.direction,
            "max_pages": args.max_pages,
            "target_id": args.target_id.as_ref().map(TargetId::normalized),
            "target_label": target_label,
        });
        value["found"] = json!(matched.is_some());
        value["matched_query"] = matched
            .as_ref()
            .map_or(Value::Null, |(query, _)| Value::String(query.clone()));
        value["matched_text"] = matched.map_or(Value::Null, |(_, text)| Value::String(text));
        value["pages_scrolled"] = json!(pages_scrolled);
        value["scan_timings_ms"] = json!({
            "total": u64::try_from(scan_started.elapsed().as_millis()).unwrap_or(u64::MAX),
            "page_observations": page_timings,
        });
        value["attempts"] = Value::Array(attempts);
        value["scroll_method"] =
            scroll_method.map_or(Value::Null, |method| Value::String(method.to_owned()));
        value["stop_reason"] = json!(stop_reason);
        value["edge_reached"] = json!(stop_reason == "edge_reached");
        value["viewport_changed"] = json!(observation_view_changed(
            &initial_observation,
            &observation,
            args.direction,
        ));
        value["state_change"] = state_change_value(
            &initial_observation,
            &observation,
            &context.task_hint.lock(),
        );
        Ok(value)
    }
}

pub(super) fn scroll_notches(direction: ScrollDirection, amount: ScrollAmount) -> (i32, i32) {
    let distance = match amount {
        ScrollAmount::Small => 3,
        ScrollAmount::Page => 8,
    };
    match direction {
        // Enigo 0.6 uses positive lengths for down/right and negative for up/left.
        ScrollDirection::Up => (0, -distance),
        ScrollDirection::Down => (0, distance),
        ScrollDirection::Left => (-distance, 0),
        ScrollDirection::Right => (distance, 0),
    }
}

pub(super) fn scroll_point(
    observation: &Observation,
    target_id: Option<&TargetId>,
) -> Result<((i32, i32), Option<String>)> {
    let capture = observation
        .target
        .as_ref()
        .ok_or_else(|| PokError::Tool("observation has no capture target".into()))?;
    if let Some(target_id) = target_id {
        let target_id = target_id.normalized();
        let target = observation
            .targets
            .iter()
            .find(|target| target.id == target_id)
            .ok_or_else(|| {
                PokError::Tool(format!(
                    "unknown scroll target {target_id:?}; use a target from the latest capture"
                ))
            })?;
        let point = target.click_point.unwrap_or((
            target.bounds.x + i32::try_from(target.bounds.width / 2).unwrap_or(i32::MAX),
            target.bounds.y + i32::try_from(target.bounds.height / 2).unwrap_or(i32::MAX),
        ));
        if !capture.bounds.contains(point.0, point.1) {
            return Err(PokError::Tool(
                "scroll target is outside the captured input region".into(),
            ));
        }
        return Ok((point, Some(target.name.clone())));
    }
    // Untargeted: the middle of the page when there is one, else the window.
    let area = main_document_region(observation)
        .filter(|region| {
            capture.bounds.contains(
                region.x + i32::try_from(region.width / 2).unwrap_or(i32::MAX),
                region.y + i32::try_from(region.height / 2).unwrap_or(i32::MAX),
            )
        })
        .unwrap_or_else(|| capture.bounds.clone());
    Ok((
        (
            area.x + i32::try_from(area.width / 2).unwrap_or(i32::MAX),
            area.y + i32::try_from(area.height / 2).unwrap_or(i32::MAX),
        ),
        None,
    ))
}

pub(super) async fn perform_scroll(
    context: &ToolContext,
    observation: &Observation,
    point: (i32, i32),
    notches: (i32, i32),
) -> Result<()> {
    let move_action = InputAction::Move {
        x: point.0,
        y: point.1,
    };
    context.policy.validate_input(&move_action, observation)?;
    simulate_input_guarded(context, &move_action, observation).await?;
    let scroll_action = InputAction::Scroll {
        delta_x: notches.0,
        delta_y: notches.1,
    };
    context.policy.validate_input(&scroll_action, observation)?;
    simulate_input_guarded(context, &scroll_action, observation).await
}

pub(super) async fn verified_scroll(
    context: &ToolContext,
    before: Observation,
    point: (i32, i32),
    direction: ScrollDirection,
    amount: ScrollAmount,
) -> Result<ScrollAttempt> {
    ensure_scroll_window_foreground(context, &before).await?;
    let mut attempts = Vec::new();
    // Cursor-free first: scroll the area through its accessibility pattern
    // so the user's mouse stays where it is. Keep it only if the view moved.
    if let Some(window) = before.foreground_window.as_ref() {
        context.pause.ensure_action_allowed()?;
        let (vertical, horizontal) = match direction {
            ScrollDirection::Up => (-1, 0),
            ScrollDirection::Down => (1, 0),
            ScrollDirection::Left => (0, -1),
            ScrollDirection::Right => (0, 1),
        };
        let request = crate::platform::PatternScrollRequest {
            window_id: window.id.clone(),
            point,
            vertical,
            horizontal,
            page: matches!(amount, ScrollAmount::Page),
        };
        if let Ok(Some(_)) = context.platform.perform_pattern_scroll(&request).await {
            tokio::time::sleep(std::time::Duration::from_millis(180)).await;
            let after_pattern = observe_after_scroll(context, &before).await?;
            let pattern_changed = observation_view_changed(&before, &after_pattern, direction);
            attempts.push(json!({
                "method": "ui_automation",
                "viewport_changed": pattern_changed,
            }));
            if pattern_changed {
                return Ok(ScrollAttempt {
                    observation: after_pattern,
                    attempts,
                    method: Some("ui_automation"),
                    viewport_changed: true,
                    edge_reached: false,
                });
            }
        }
    }
    let notches = scroll_notches(direction, amount);
    perform_scroll(context, &before, point, notches).await?;
    tokio::time::sleep(std::time::Duration::from_millis(180)).await;
    let after_wheel = observe_after_scroll(context, &before).await?;
    let wheel_changed = observation_view_changed(&before, &after_wheel, direction);
    attempts.push(json!({
        "method": "wheel",
        "viewport_changed": wheel_changed,
    }));
    if wheel_changed {
        return Ok(ScrollAttempt {
            observation: after_wheel,
            attempts,
            method: Some("wheel"),
            viewport_changed: true,
            edge_reached: false,
        });
    }

    let selection_control_active = after_wheel.targets.iter().any(|target| {
        target.selected == Some(true)
            || (target.focused
                && matches!(
                    target.control_type.to_ascii_lowercase().as_str(),
                    "list"
                        | "list item"
                        | "listitem"
                        | "combo box"
                        | "combobox"
                        | "tree item"
                        | "treeitem"
                ))
    });
    if selection_control_active {
        attempts.push(json!({
            "method": "keyboard",
            "attempted": false,
            "reason": "selection_control_active",
            "instruction": "Do not use PageUp/PageDown blindly on a selection control. Use the selected/list-item targets from the refreshed observation.",
        }));
        return Ok(ScrollAttempt {
            observation: after_wheel,
            attempts,
            method: None,
            viewport_changed: false,
            edge_reached: false,
        });
    }

    let key = scroll_fallback_key(direction, amount);
    let key_action = InputAction::Key {
        key: key.to_owned(),
    };
    context.policy.validate_input(&key_action, &after_wheel)?;
    simulate_input_guarded(context, &key_action, &after_wheel).await?;
    tokio::time::sleep(std::time::Duration::from_millis(180)).await;
    let after_keyboard = observe_after_scroll(context, &after_wheel).await?;
    let keyboard_changed = observation_view_changed(&after_wheel, &after_keyboard, direction);
    attempts.push(json!({
        "method": "keyboard",
        "key": key,
        "viewport_changed": keyboard_changed,
    }));
    if keyboard_changed {
        return Ok(ScrollAttempt {
            observation: after_keyboard,
            attempts,
            method: Some("keyboard"),
            viewport_changed: true,
            edge_reached: false,
        });
    }

    let scrollbar = scrollbar_fallback(&after_keyboard, direction);
    if scrollbar.edge_reached {
        attempts.push(json!({
            "method": "scrollbar",
            "attempted": false,
            "edge_reached": true,
        }));
        return Ok(ScrollAttempt {
            observation: after_keyboard,
            attempts,
            method: None,
            viewport_changed: false,
            edge_reached: true,
        });
    }
    let Some(click_point) = scrollbar.click_point else {
        attempts.push(json!({
            "method": "scrollbar",
            "attempted": false,
            "reason": "not_detected",
        }));
        return Ok(ScrollAttempt {
            observation: after_keyboard,
            attempts,
            method: None,
            viewport_changed: false,
            edge_reached: false,
        });
    };

    let click = InputAction::Click {
        x: click_point.0,
        y: click_point.1,
        button: MouseButton::Left,
    };
    context.policy.validate_input(&click, &after_keyboard)?;
    simulate_input_guarded(context, &click, &after_keyboard).await?;
    tokio::time::sleep(std::time::Duration::from_millis(180)).await;
    let after_scrollbar = observe_after_scroll(context, &after_keyboard).await?;
    let scrollbar_changed = observation_view_changed(&after_keyboard, &after_scrollbar, direction);
    attempts.push(json!({
        "method": "scrollbar",
        "attempted": true,
        "viewport_changed": scrollbar_changed,
    }));
    Ok(ScrollAttempt {
        observation: after_scrollbar,
        attempts,
        method: scrollbar_changed.then_some("scrollbar"),
        viewport_changed: scrollbar_changed,
        edge_reached: false,
    })
}

pub(super) async fn ensure_scroll_window_foreground(
    context: &ToolContext,
    observation: &Observation,
) -> Result<()> {
    let Some(target) = observation.target.as_ref() else {
        return Err(PokError::Tool("observation has no capture target".into()));
    };
    if !matches!(
        target.scope,
        CaptureScope::Window | CaptureScope::ActiveWindow
    ) {
        return Ok(());
    }
    let foreground = context.platform.foreground_window().await?;
    if foreground
        .as_ref()
        .is_some_and(|window| window.id == target.id)
    {
        return Ok(());
    }
    context.platform.activate_window(&target.id).await?;
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    let foreground = context.platform.foreground_window().await?;
    if foreground
        .as_ref()
        .is_some_and(|window| window.id == target.id)
    {
        Ok(())
    } else {
        Err(PokError::Tool(
            "scroll target could not be activated safely".into(),
        ))
    }
}

pub(super) const fn scroll_fallback_key(
    direction: ScrollDirection,
    amount: ScrollAmount,
) -> &'static str {
    match (direction, amount) {
        (ScrollDirection::Up, ScrollAmount::Page) => "PageUp",
        (ScrollDirection::Down, ScrollAmount::Page) => "PageDown",
        (ScrollDirection::Up, ScrollAmount::Small) => "UpArrow",
        (ScrollDirection::Down, ScrollAmount::Small) => "DownArrow",
        (ScrollDirection::Left, _) => "LeftArrow",
        (ScrollDirection::Right, _) => "RightArrow",
    }
}

#[derive(Default)]
pub(super) struct ScrollbarFallback {
    pub(super) click_point: Option<(i32, i32)>,
    pub(super) edge_reached: bool,
}

pub(super) fn scrollbar_fallback(
    observation: &Observation,
    direction: ScrollDirection,
) -> ScrollbarFallback {
    let capture = match observation.target.as_ref() {
        Some(capture) => capture,
        None => return ScrollbarFallback::default(),
    };
    let vertical = matches!(direction, ScrollDirection::Up | ScrollDirection::Down);
    if let Some(element) = observation.ui_elements.iter().find(|element| {
        let control_type = normalized_text(&element.control_type).replace(' ', "");
        let correct_axis = if vertical {
            element.bounds.height > element.bounds.width.saturating_mul(3)
        } else {
            element.bounds.width > element.bounds.height.saturating_mul(3)
        };
        !element.offscreen
            && element.enabled
            && control_type.contains("scrollbar")
            && correct_axis
            && overlaps(&element.bounds, &capture.bounds)
    }) {
        let point = scrollbar_track_point(&element.bounds, direction);
        return ScrollbarFallback {
            click_point: capture.bounds.contains(point.0, point.1).then_some(point),
            edge_reached: false,
        };
    }
    visual_scrollbar_fallback(observation, direction)
}

pub(super) fn scrollbar_track_point(bounds: &Rect, direction: ScrollDirection) -> (i32, i32) {
    let quarter_x = i32::try_from(bounds.width / 4).unwrap_or(i32::MAX);
    let quarter_y = i32::try_from(bounds.height / 4).unwrap_or(i32::MAX);
    match direction {
        ScrollDirection::Up => (
            bounds.x + i32::try_from(bounds.width / 2).unwrap_or(i32::MAX),
            bounds.y + quarter_y,
        ),
        ScrollDirection::Down => (
            bounds.x + i32::try_from(bounds.width / 2).unwrap_or(i32::MAX),
            bounds.y + quarter_y.saturating_mul(3),
        ),
        ScrollDirection::Left => (
            bounds.x + quarter_x,
            bounds.y + i32::try_from(bounds.height / 2).unwrap_or(i32::MAX),
        ),
        ScrollDirection::Right => (
            bounds.x + quarter_x.saturating_mul(3),
            bounds.y + i32::try_from(bounds.height / 2).unwrap_or(i32::MAX),
        ),
    }
}

pub(super) fn visual_scrollbar_fallback(
    observation: &Observation,
    direction: ScrollDirection,
) -> ScrollbarFallback {
    let Some(capture) = observation.target.as_ref() else {
        return ScrollbarFallback::default();
    };
    let Some(screenshot) = observation.screenshots.first() else {
        return ScrollbarFallback::default();
    };
    let encoded = screenshot
        .source_png_base64
        .as_deref()
        .unwrap_or(&screenshot.png_base64);
    let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(encoded) else {
        return ScrollbarFallback::default();
    };
    let Ok(image) = image::load_from_memory(&bytes).map(|image| image.into_luma8()) else {
        return ScrollbarFallback::default();
    };
    let vertical = matches!(direction, ScrollDirection::Up | ScrollDirection::Down);
    let candidate = if vertical {
        detect_edge_thumb(&image, true)
    } else {
        detect_edge_thumb(&image, false)
    };
    let Some((cross, start, end, extent)) = candidate else {
        return ScrollbarFallback::default();
    };
    let toward_start = matches!(direction, ScrollDirection::Up | ScrollDirection::Left);
    let margin = 6_u32;
    let at_edge = if toward_start {
        start <= margin
    } else {
        end.saturating_add(margin) >= extent
    };
    if at_edge {
        return ScrollbarFallback {
            click_point: None,
            edge_reached: true,
        };
    }
    let primary = if toward_start {
        start.saturating_sub(margin)
    } else {
        end.saturating_add(margin).min(extent.saturating_sub(1))
    };
    let (image_x, image_y) = if vertical {
        (cross, primary)
    } else {
        (primary, cross)
    };
    let physical_x = capture.bounds.x
        + scale_offset(
            i32::try_from(image_x).unwrap_or(i32::MAX),
            image.width(),
            capture.bounds.width,
        );
    let physical_y = capture.bounds.y
        + scale_offset(
            i32::try_from(image_y).unwrap_or(i32::MAX),
            image.height(),
            capture.bounds.height,
        );
    ScrollbarFallback {
        click_point: capture
            .bounds
            .contains(physical_x, physical_y)
            .then_some((physical_x, physical_y)),
        edge_reached: false,
    }
}

pub(super) fn detect_edge_thumb(
    image: &image::GrayImage,
    vertical: bool,
) -> Option<(u32, u32, u32, u32)> {
    let (width, height) = image.dimensions();
    let (cross_extent, primary_extent) = if vertical {
        (width, height)
    } else {
        (height, width)
    };
    if cross_extent < 8 || primary_extent < 80 {
        return None;
    }
    let gutter = 20_u32.min(cross_extent / 8).max(4);
    let mut candidates = Vec::new();
    for cross in cross_extent.saturating_sub(gutter)..cross_extent.saturating_sub(2) {
        let values = (0..primary_extent)
            .map(|primary| {
                let (x, y) = if vertical {
                    (cross, primary)
                } else {
                    (primary, cross)
                };
                image.get_pixel(x, y).0[0]
            })
            .collect::<Vec<_>>();
        let mut sorted = values.clone();
        sorted.sort_unstable();
        let median = sorted[sorted.len() / 2];
        let mut run_start = None;
        for (index, value) in values
            .iter()
            .copied()
            .chain(std::iter::once(median))
            .enumerate()
        {
            let contrasting = value.abs_diff(median) >= 36;
            match (run_start, contrasting) {
                (None, true) => run_start = Some(index),
                (Some(start), false) => {
                    let length = index.saturating_sub(start);
                    if length >= 10 && length <= usize::try_from(primary_extent / 2).unwrap_or(0) {
                        candidates.push((
                            cross,
                            u32::try_from(start).unwrap_or(0),
                            u32::try_from(index.saturating_sub(1)).unwrap_or(u32::MAX),
                        ));
                    }
                    run_start = None;
                }
                _ => {}
            }
        }
    }
    candidates.iter().copied().find_map(|candidate| {
        let adjacent = (candidate.0.saturating_sub(2)..=candidate.0.saturating_add(2))
            .filter(|cross| *cross != candidate.0)
            .filter(|cross| {
                candidates
                    .iter()
                    .copied()
                    .any(|other| other.0 == *cross && candidates_similar(candidate, other))
            })
            .count();
        (adjacent >= 2).then_some((candidate.0, candidate.1, candidate.2, primary_extent))
    })
}

pub(super) fn candidates_similar(left: (u32, u32, u32), right: (u32, u32, u32)) -> bool {
    left.1.abs_diff(right.1) <= 4 && left.2.abs_diff(right.2) <= 4
}

pub(super) async fn observe_after_scroll(
    context: &ToolContext,
    before: &Observation,
) -> Result<Observation> {
    let target = before
        .target
        .as_ref()
        .ok_or_else(|| PokError::Tool("observation has no capture target".into()))?;
    let request = CaptureRequest {
        scope: target.scope.clone(),
        window_id: matches!(target.scope, CaptureScope::Window).then(|| target.id.clone()),
        monitor_id: matches!(target.scope, CaptureScope::Monitor).then(|| target.id.clone()),
        region: matches!(target.scope, CaptureScope::Region).then(|| target.bounds.clone()),
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
    let fusion_started = Instant::now();
    observation.targets = build_targets(
        &observation,
        context.model_target_limit,
        context.fusion_iou_threshold,
        context.ocr_containment_threshold,
        &context.task_hint.lock(),
    );
    observation.timings_ms.insert(
        "fusion".into(),
        u64::try_from(fusion_started.elapsed().as_millis()).unwrap_or(u64::MAX),
    );
    *context.latest_observation.lock() = Some(observation.clone());
    *context.latest_observation_view.lock() = None;
    *context.pending_visual_localization.lock() = None;
    let diagnostic = crate::tool::diagnostic_value(&serde_json::to_value(&observation)?);
    context.write_artifact(
        &format!("observation-{}.json", observation.version),
        &diagnostic,
    )?;
    save_observation_visuals(context, &observation)?;
    context.write_artifact(
        &format!("observation-{}-targets.json", observation.version),
        &observation.targets,
    )?;
    Ok(observation)
}

pub(super) fn observation_view_changed(
    before: &Observation,
    after: &Observation,
    direction: ScrollDirection,
) -> bool {
    if coherent_grounded_movement(before, after, direction) {
        return true;
    }
    let before_text = visible_text_set(before);
    let after_text = visible_text_set(after);
    let union = before_text.union(&after_text).count();
    let changed = before_text.symmetric_difference(&after_text).count();
    union >= 5 && changed >= 3 && changed.saturating_mul(5) >= union
}

pub(super) fn coherent_grounded_movement(
    before: &Observation,
    after: &Observation,
    direction: ScrollDirection,
) -> bool {
    let before_positions = grounded_text_positions(before);
    let after_positions = grounded_text_positions(after);
    let mut matching_deltas = 0_usize;
    for (text, before_points) in before_positions {
        let Some(after_points) = after_positions.get(&text) else {
            continue;
        };
        if before_points.len() != 1 || after_points.len() != 1 {
            continue;
        }
        let delta_x = after_points[0].0 - before_points[0].0;
        let delta_y = after_points[0].1 - before_points[0].1;
        let matches = match direction {
            ScrollDirection::Up => delta_y >= 8,
            ScrollDirection::Down => delta_y <= -8,
            ScrollDirection::Left => delta_x >= 8,
            ScrollDirection::Right => delta_x <= -8,
        };
        if matches {
            matching_deltas = matching_deltas.saturating_add(1);
        }
    }
    matching_deltas >= 3
}

pub(super) fn grounded_text_positions(
    observation: &Observation,
) -> HashMap<String, Vec<(i32, i32)>> {
    let mut positions: HashMap<String, Vec<(i32, i32)>> = HashMap::new();
    for (text, bounds) in observation
        .ui_elements
        .iter()
        .filter(|element| !element.offscreen)
        .map(|element| (element.name.as_str(), &element.bounds))
        .chain(
            observation
                .ocr
                .iter()
                .map(|block| (block.text.as_str(), &block.bounds)),
        )
    {
        let text = normalized_text(text);
        if text.is_empty() {
            continue;
        }
        positions.entry(text).or_default().push((
            bounds.x + i32::try_from(bounds.width / 2).unwrap_or(i32::MAX),
            bounds.y + i32::try_from(bounds.height / 2).unwrap_or(i32::MAX),
        ));
    }
    positions
}

pub(super) fn visible_text_set(observation: &Observation) -> HashSet<String> {
    observation
        .ui_elements
        .iter()
        .map(|element| element.name.as_str())
        .chain(observation.ocr.iter().map(|block| block.text.as_str()))
        .chain(
            observation
                .targets
                .iter()
                .map(|target| target.name.as_str()),
        )
        .map(normalized_text)
        .filter(|text| !text.is_empty())
        .collect()
}

pub(super) fn visible_text_match(observation: &Observation, query: &str) -> Option<String> {
    observation
        .targets
        .iter()
        .map(|target| target.name.as_str())
        .chain(
            observation
                .ui_elements
                .iter()
                .map(|element| element.name.as_str()),
        )
        .chain(observation.ocr.iter().map(|block| block.text.as_str()))
        .find(|text| normalized_text(text).contains(query))
        .map(|text| text.trim().chars().take(300).collect())
}

pub(super) fn visible_text_match_any(
    observation: &Observation,
    queries: &[String],
) -> Option<(String, String)> {
    queries
        .iter()
        .find_map(|query| visible_text_match(observation, query).map(|text| (query.clone(), text)))
}

pub(super) fn scroll_query_alternatives(query: &str) -> Result<Vec<String>> {
    const MAX_ALTERNATIVES: usize = 8;
    const MAX_ALTERNATIVE_CHARS: usize = 200;
    let mut seen = HashSet::new();
    let mut alternatives = Vec::new();
    for candidate in query.split('|') {
        let candidate = normalized_text(candidate);
        if candidate.is_empty() || !seen.insert(candidate.clone()) {
            continue;
        }
        if candidate.chars().count() > MAX_ALTERNATIVE_CHARS {
            return Err(PokError::Tool(format!(
                "scroll search alternatives must be at most {MAX_ALTERNATIVE_CHARS} characters"
            )));
        }
        alternatives.push(candidate);
        if alternatives.len() > MAX_ALTERNATIVES {
            return Err(PokError::Tool(format!(
                "scroll search supports at most {MAX_ALTERNATIVES} alternatives"
            )));
        }
    }
    if alternatives.is_empty() {
        return Err(PokError::Tool("query cannot be empty".into()));
    }
    Ok(alternatives)
}
