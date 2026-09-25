//! Windows and applications: listing, launching, and activating windows, and recovering when activation lands on a shell surface.

use super::*;

pub(super) struct ListWindowsTool;
#[async_trait]
impl Tool for ListWindowsTool {
    fn name(&self) -> &'static str {
        "list_windows"
    }
    fn description(&self) -> &'static str {
        "List visible top-level application windows so one can be selected for activation and capture."
    }
    fn input_schema(&self) -> Value {
        json!({"type": "object", "properties": {}})
    }
    fn risk(&self) -> RiskClass {
        RiskClass::ReadOnly
    }
    async fn execute(&self, _args: Value, context: &ToolContext) -> Result<Value> {
        let (windows, monitors) = tokio::join!(
            context.platform.list_windows(),
            context.platform.list_monitors(),
        );
        let windows = windows?;
        let total = windows.len();
        let summaries = compact_windows(windows, &monitors?, &context.task_hint.lock(), 100);
        Ok(json!({
            "windows": summaries,
            "window_count": total,
            "truncated": total > 100,
            "instruction": "Window ids are not click targets. To search or open a URL, call browser_navigate with the complete browser window id. To inspect a window, activate_window and then capture_screen.",
        }))
    }
}

pub(super) struct OpenApplicationTool;

#[async_trait]
impl Tool for OpenApplicationTool {
    fn name(&self) -> &'static str {
        "open_application"
    }
    fn description(&self) -> &'static str {
        "Launch an installed application by its Start-menu name or executable (for example Settings, Discord, notepad) when the task needs an application that is not already open. It waits for the new window and returns its id, or reports that no window appeared. Then activate_window that window before interacting with it."
    }
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Bare application name or executable, for example 'notepad'"}
            },
            "required": ["name"]
        })
    }
    fn risk(&self) -> RiskClass {
        RiskClass::ProcessExecution
    }
    async fn execute(&self, args: Value, context: &ToolContext) -> Result<Value> {
        let name = args
            .get("name")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .ok_or_else(|| PokError::Tool("open_application requires a name".into()))?;
        // Idempotent per run: if an application window whose process or title
        // matches the name is already present, do not launch another instance.
        // Live traces showed notepad launched dozens of times because both the
        // router and the primary model kept calling open_application.
        let normalized = name.to_ascii_lowercase();
        let before = context.platform.list_windows().await.unwrap_or_default();
        let already_open = before.iter().any(|window| {
            window
                .process_name
                .to_ascii_lowercase()
                .contains(&normalized)
                || window.title.to_ascii_lowercase().contains(&normalized)
        });
        if already_open {
            return Ok(json!({
                "launched": false,
                "already_open": true,
                "instruction": "The application is already open. Use list_windows to find its window id and activate_window to focus it; do not call open_application again.",
            }));
        }
        context.platform.launch_application(name).await?;
        // A spawn can succeed while Windows only shows a "cannot find"
        // dialog, so success is reported only once a new window appears.
        let known = before
            .iter()
            .map(|window| window.id.clone())
            .collect::<HashSet<_>>();
        // Shell hosts show windows titled with the requested name without
        // being the application: explorer.exe for "Windows cannot find"
        // dialogs, cmd.exe/conhost.exe for the `start` fallback. They only
        // count when that host itself was requested.
        let launched_window = |window: &crate::types::WindowInfo| {
            let label = format!("{} {}", window.title, window.process_name).to_ascii_lowercase();
            let process = window.process_name.to_ascii_lowercase();
            let host = process.trim_end_matches(".exe");
            let shell_host = matches!(host, "explorer" | "cmd" | "conhost" | "powershell" | "pwsh");
            label.contains(&normalized) && (!shell_host || normalized.contains(host))
        };
        let started = Instant::now();
        let mut new_window = None;
        while started.elapsed() < Duration::from_secs(8) {
            if context.cancellation.is_cancelled() {
                return Err(PokError::Cancelled);
            }
            let windows = context.platform.list_windows().await.unwrap_or_default();
            new_window = windows
                .into_iter()
                .filter(|window| !known.contains(&window.id))
                .max_by_key(&launched_window);
            if new_window.as_ref().is_some_and(launched_window) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(400)).await;
        }
        let matched = new_window.as_ref().is_some_and(launched_window);
        Ok(match new_window {
            Some(window) if matched => {
                // A freshly launched application often cannot take the
                // foreground by itself (it only flashes on the taskbar) and
                // may not finish starting until it has focus. Bring it to the
                // front without mouse or keyboard, once the user is idle.
                wait_for_user_idle(context).await?;
                let brought_to_front = context.platform.bring_to_front(&window.id).await.is_ok();
                json!({
                    "launched": name,
                    "window": {"id": window.id, "title": window.title, "app": window.process_name},
                    "brought_to_front": brought_to_front,
                    "instruction": "The application window appeared and was brought to the front. Capture it (capture_screen with this window id) before any input."
                })
            }
            Some(window) => json!({
                "launched": false,
                "unexpected_window": {"id": window.id, "title": window.title, "app": window.process_name},
                "instruction": "A different window appeared instead of the application, possibly an error dialog saying it could not be found. Capture it, dismiss it, and open the application another way (for example from the Start menu)."
            }),
            None => json!({
                "launched": false,
                "instruction": "No new window appeared within 8 seconds. Check list_windows and the screen; the application may be starting slowly, may already be open elsewhere, or may not be installed under this name."
            }),
        })
    }
}

pub(super) fn compact_monitor(monitor: &MonitorInfo, number: usize) -> Value {
    json!({
        "id": monitor.id,
        "label": format!("M{number}"),
        "primary": monitor.primary,
        "position": [monitor.bounds.x, monitor.bounds.y],
        "size": [monitor.bounds.width, monitor.bounds.height],
        "scale_factor": monitor.scale_factor,
    })
}

pub(super) fn compact_windows(
    windows: Vec<WindowInfo>,
    monitors: &[MonitorInfo],
    task: &str,
    limit: usize,
) -> Vec<Value> {
    let terms = ranking_terms(task);
    let mut windows = windows
        .into_iter()
        .filter(|window| !window.title.trim().is_empty() && window.title != "Program Manager")
        .map(|window| {
            let haystack = format!(
                "{} {}",
                window.title.to_ascii_lowercase(),
                window.process_name.to_ascii_lowercase()
            );
            let matches = terms
                .iter()
                .filter(|term| haystack.contains(term.as_str()))
                .count()
                .min(4) as u16;
            let score = matches * 20 + u16::from(!window.minimized) * 2 + u16::from(window.visible);
            (score, window)
        })
        .collect::<Vec<_>>();
    windows.sort_by(|(left_score, left), (right_score, right)| {
        right_score
            .cmp(left_score)
            .then_with(|| left.title.cmp(&right.title))
    });
    windows
        .into_iter()
        .take(limit)
        .enumerate()
        .map(|(index, (_, window))| {
            json!({
                "id": window.id,
                "alias": format!("window_{}", index + 1),
                "title": window.title,
                "app": window.process_name,
                "monitor_id": best_monitor(&window.bounds, monitors).map(|monitor| monitor.id.clone()),
                "state": if window.minimized { "minimized" } else { "visible" },
            })
        })
        .collect()
}

pub(super) fn ranking_terms(task: &str) -> Vec<String> {
    const STOP: &[&str] = &[
        "the", "and", "for", "with", "this", "that", "can", "you", "please",
    ];
    task.to_ascii_lowercase()
        .split(|character: char| !character.is_ascii_alphanumeric())
        .filter(|word| word.len() > 2 && !STOP.contains(word))
        .map(str::to_owned)
        .collect()
}

pub(super) fn best_monitor<'a>(
    bounds: &Rect,
    monitors: &'a [MonitorInfo],
) -> Option<&'a MonitorInfo> {
    let best = monitors
        .iter()
        .max_by_key(|monitor| intersection_area(bounds, &monitor.bounds));
    best.filter(|monitor| intersection_area(bounds, &monitor.bounds) > 0)
        .or_else(|| monitors.iter().find(|monitor| monitor.primary))
}

pub(super) fn intersection_area(left: &Rect, right: &Rect) -> u64 {
    let x1 = i64::from(left.x).max(i64::from(right.x));
    let y1 = i64::from(left.y).max(i64::from(right.y));
    let x2 = (i64::from(left.x) + i64::from(left.width))
        .min(i64::from(right.x) + i64::from(right.width));
    let y2 = (i64::from(left.y) + i64::from(left.height))
        .min(i64::from(right.y) + i64::from(right.height));
    u64::try_from((x2 - x1).max(0)).unwrap_or(0) * u64::try_from((y2 - y1).max(0)).unwrap_or(0)
}

#[derive(Debug, Deserialize, JsonSchema)]
pub(super) struct ActivateWindowArgs {
    pub(super) window_id: String,
}

pub(super) struct ActivateWindowTool;
#[async_trait]
impl Tool for ActivateWindowTool {
    fn name(&self) -> &'static str {
        "activate_window"
    }
    fn description(&self) -> &'static str {
        "Switch to a window returned by list_windows and return a fresh capture of it (the same result as capture_screen with scope window). With background work the window can stay behind the user's windows; keep working from this window capture, not a monitor capture, which shows whatever is on top. Interactive desktop sessions execute this autonomously."
    }
    fn input_schema(&self) -> Value {
        schema::<ActivateWindowArgs>()
    }
    fn risk(&self) -> RiskClass {
        RiskClass::DesktopInput
    }
    fn approval_key(&self, arguments: &Value) -> Option<String> {
        arguments
            .get("window_id")
            .and_then(Value::as_str)
            .and_then(|id| id.split_once(':'))
            .map(|(process_id, _)| format!("activate_process:{process_id}"))
    }
    async fn execute(&self, args: Value, context: &ToolContext) -> Result<Value> {
        let mut args: ActivateWindowArgs = serde_json::from_value(args)?;
        args.window_id = resolve_window_id(context, &args.window_id).await?;
        let target = context
            .platform
            .list_windows()
            .await?
            .into_iter()
            .find(|window| window.id == args.window_id)
            .ok_or_else(|| {
                PokError::Tool(format!("window {:?} no longer exists", args.window_id))
            })?;
        let transient_overlay_dismissed = context
            .platform
            .foreground_window()
            .await
            .ok()
            .flatten()
            .filter(is_transient_shell_overlay)
            .map(|window| format!("{} ({})", window.title, window.process_name));
        if transient_overlay_dismissed.is_some() {
            context
                .platform
                .simulate_input(&InputAction::Key {
                    key: "Escape".into(),
                })
                .await?;
            tokio::time::sleep(Duration::from_millis(120)).await;
        }
        let native = context.platform.activate_window(&args.window_id).await;
        *context.focused_control.lock() = None;
        if let Ok(window) = native {
            let window_id = window.id.clone();
            let mut value = activation_success_value(
                window,
                "native",
                vec![json!({"strategy": "native", "verified": true})],
                transient_overlay_dismissed,
            )?;
            // Return the chosen window as it is now. With background work it
            // may stay behind the user's windows, where a monitor capture
            // would show the user's screen instead of it.
            match CaptureScreenTool
                .execute(json!({"scope": "window", "window_id": window_id}), context)
                .await
            {
                Ok(capture) => value["capture"] = capture,
                Err(error) => value["capture_error"] = json!(error.to_string()),
            }
            return Ok(value);
        }
        let native_error = native
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default();
        recover_window_activation(context, &target, native_error, transient_overlay_dismissed).await
    }
}

pub(super) const MAX_ALT_TAB_RECOVERY_STEPS: usize = 8;

pub(super) fn activation_success_value(
    window: WindowInfo,
    strategy: &str,
    attempts: Vec<Value>,
    transient_overlay_dismissed: Option<String>,
) -> Result<Value> {
    let mut value = serde_json::to_value(window)?;
    value["status"] = json!("activated");
    value["verified"] = json!(true);
    value["activation_strategy"] = json!(strategy);
    value["attempts"] = json!(attempts);
    if let Some(dismissed) = transient_overlay_dismissed {
        value["transient_overlay_dismissed"] = json!(dismissed);
    }
    Ok(value)
}

pub(super) async fn recover_window_activation(
    context: &ToolContext,
    target: &WindowInfo,
    native_error: String,
    transient_overlay_dismissed: Option<String>,
) -> Result<Value> {
    let mut attempts = vec![json!({
        "strategy": "native",
        "verified": false,
        "error": native_error,
    })];
    if target.elevated {
        return activation_recovery_required(
            context,
            target,
            attempts,
            "target window is elevated",
            transient_overlay_dismissed,
            None,
        )
        .await;
    }
    let mut monitors = context.platform.list_monitors().await?;
    let preferred_id = best_monitor(&target.bounds, &monitors)
        .map(|monitor| monitor.id.clone())
        .ok_or_else(|| PokError::Tool("window activation recovery found no monitor".into()))?;
    monitors.sort_by_key(|monitor| monitor.id != preferred_id);
    let monitor = monitors[0].clone();
    let mut first_observation = None;
    let mut matches = Vec::new();
    for candidate_monitor in &monitors {
        let observation = observe_recovery_scope(
            context,
            CaptureScope::Monitor,
            Some(candidate_monitor.id.clone()),
            None,
        )
        .await?;
        for candidate in matching_shell_targets(&observation, target) {
            matches.push((
                observation.clone(),
                candidate.clone(),
                candidate_monitor.clone(),
            ));
        }
        if candidate_monitor.id == preferred_id {
            first_observation = Some(observation);
        }
    }
    let mut observation = first_observation
        .ok_or_else(|| PokError::Tool("preferred activation monitor disappeared".into()))?;
    let mut recovery_monitor = monitor.clone();
    if matches.len() == 1 {
        let (matched_observation, candidate, matched_monitor) = matches.remove(0);
        observation = matched_observation;
        recovery_monitor = matched_monitor;
        let (x, y) = candidate.click_point.unwrap_or((
            candidate.bounds.x + i32::try_from(candidate.bounds.width / 2).unwrap_or(i32::MAX),
            candidate.bounds.y + i32::try_from(candidate.bounds.height / 2).unwrap_or(i32::MAX),
        ));
        let click = InputAction::Click {
            x,
            y,
            button: MouseButton::Left,
        };
        context.policy.validate_input(&click, &observation)?;
        simulate_input_guarded(context, &click, &observation).await?;
        tokio::time::sleep(Duration::from_millis(220)).await;
        let foreground = context.platform.foreground_window().await?;
        let verified = foreground
            .as_ref()
            .is_some_and(|window| window.id == target.id);
        attempts.push(json!({
            "strategy": "taskbar_unique_match",
            "target_id": candidate.id,
            "label": candidate.name,
            "verified": verified,
        }));
        if let Some(window) = foreground.filter(|window| window.id == target.id) {
            *context.latest_observation.lock() = None;
            *context.latest_observation_view.lock() = None;
            *context.pending_visual_localization.lock() = None;
            *context.focused_control.lock() = None;
            return activation_success_value(
                window,
                "taskbar_unique_match",
                attempts,
                transient_overlay_dismissed,
            );
        }
    } else {
        attempts.push(json!({
            "strategy": "taskbar_unique_match",
            "attempted": false,
            "candidate_count": matches.len(),
            "reason": if matches.is_empty() { "no high-confidence match" } else { "ambiguous match" },
        }));
    }

    let visible_count = context
        .platform
        .list_windows()
        .await?
        .into_iter()
        .filter(|window| window.visible)
        .count();
    for step in 1..=visible_count.min(MAX_ALT_TAB_RECOVERY_STEPS) {
        recovery_monitor = monitor.clone();
        observation = observe_recovery_scope(
            context,
            CaptureScope::Monitor,
            Some(monitor.id.clone()),
            None,
        )
        .await?;
        if observation
            .foreground_window
            .as_ref()
            .is_some_and(|window| window.elevated)
        {
            attempts.push(json!({
                "strategy": "alt_tab",
                "step": step,
                "attempted": false,
                "reason": "current foreground is elevated",
            }));
            break;
        }
        let key = InputAction::Key {
            key: "Alt+Tab".into(),
        };
        context.policy.validate_input(&key, &observation)?;
        simulate_input_guarded(context, &key, &observation).await?;
        tokio::time::sleep(Duration::from_millis(180)).await;
        let foreground = context.platform.foreground_window().await?;
        let verified = foreground
            .as_ref()
            .is_some_and(|window| window.id == target.id);
        attempts.push(json!({
            "strategy": "alt_tab",
            "step": step,
            "verified": verified,
            "foreground": foreground.as_ref().map(|window| format!("{} ({})", window.title, window.process_name)),
        }));
        if let Some(window) = foreground.filter(|window| window.id == target.id) {
            *context.latest_observation.lock() = None;
            *context.latest_observation_view.lock() = None;
            *context.pending_visual_localization.lock() = None;
            *context.focused_control.lock() = None;
            return activation_success_value(
                window,
                "alt_tab",
                attempts,
                transient_overlay_dismissed,
            );
        }
    }
    activation_recovery_required(
        context,
        target,
        attempts,
        "bounded exact activation recovery was exhausted",
        transient_overlay_dismissed,
        Some((&observation, &recovery_monitor)),
    )
    .await
}

pub(super) fn matching_shell_targets<'a>(
    observation: &'a Observation,
    window: &WindowInfo,
) -> Vec<&'a InteractionTarget> {
    let title = normalized_text(&window.title);
    let process = window
        .process_name
        .trim_end_matches(".exe")
        .to_ascii_lowercase();
    let compact = |value: &str| {
        value
            .chars()
            .filter(|character| character.is_ascii_alphanumeric())
            .collect::<String>()
            .to_ascii_lowercase()
    };
    let title_compact = compact(&title);
    let process_compact = compact(&process);
    let bounds = observation.target.as_ref().map(|target| &target.bounds);
    let mut scored = observation
        .targets
        .iter()
        .filter(|candidate| {
            candidate.actionable
                && candidate.desktop_shell
                && matches!(candidate.source, TargetSource::Uia | TargetSource::UiaOcr)
                && bounds.is_some_and(|bounds| near_capture_edge(&candidate.bounds, bounds))
        })
        .filter_map(|candidate| {
            let label = compact(&candidate.name);
            let score = u8::from(title_compact.len() >= 4 && label.starts_with(&title_compact)) * 6
                + u8::from(process_compact.len() >= 4 && label.contains(&process_compact)) * 4;
            (score >= 4).then_some((score, candidate))
        })
        .collect::<Vec<_>>();
    scored.sort_by_key(|(score, _)| std::cmp::Reverse(*score));
    let Some(best) = scored.first().map(|(score, _)| *score) else {
        return Vec::new();
    };
    scored
        .into_iter()
        .filter(|(score, _)| *score == best)
        .map(|(_, candidate)| candidate)
        .collect()
}

pub(super) fn near_capture_edge(candidate: &Rect, capture: &Rect) -> bool {
    let edge_x = (capture.width / 8).max(48);
    let edge_y = (capture.height / 8).max(48);
    let right = i64::from(candidate.x) + i64::from(candidate.width);
    let bottom = i64::from(candidate.y) + i64::from(candidate.height);
    candidate.x
        < capture
            .x
            .saturating_add(i32::try_from(edge_x).unwrap_or(i32::MAX))
        || candidate.y
            < capture
                .y
                .saturating_add(i32::try_from(edge_y).unwrap_or(i32::MAX))
        || right > i64::from(capture.x) + i64::from(capture.width.saturating_sub(edge_x))
        || bottom > i64::from(capture.y) + i64::from(capture.height.saturating_sub(edge_y))
}

pub(super) async fn observe_recovery_scope(
    context: &ToolContext,
    scope: CaptureScope,
    monitor_id: Option<String>,
    region: Option<Rect>,
) -> Result<Observation> {
    let request = CaptureRequest {
        scope,
        window_id: None,
        monitor_id,
        region,
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
    *context.latest_observation.lock() = Some(observation.clone());
    *context.latest_observation_view.lock() = None;
    *context.pending_visual_localization.lock() = None;
    context.write_artifact(
        &format!("observation-{}.json", observation.version),
        &crate::tool::diagnostic_value(&serde_json::to_value(&observation)?),
    )?;
    save_observation_visuals(context, &observation)?;
    context.write_artifact(
        &format!("observation-{}-targets.json", observation.version),
        &observation.targets,
    )?;
    Ok(observation)
}

pub(super) async fn activation_recovery_required(
    context: &ToolContext,
    target: &WindowInfo,
    attempts: Vec<Value>,
    reason: &str,
    transient_overlay_dismissed: Option<String>,
    recovery_source: Option<(&Observation, &MonitorInfo)>,
) -> Result<Value> {
    let mut observation = if let Some((source, monitor)) = recovery_source {
        let region = shell_recovery_region(source, monitor);
        observe_recovery_scope(context, CaptureScope::Region, None, Some(region)).await?
    } else {
        let monitors = context.platform.list_monitors().await?;
        let monitor = best_monitor(&target.bounds, &monitors)
            .ok_or_else(|| PokError::Tool("activation recovery found no monitor".into()))?;
        let monitor_observation = observe_recovery_scope(
            context,
            CaptureScope::Monitor,
            Some(monitor.id.clone()),
            None,
        )
        .await?;
        let region = shell_recovery_region(&monitor_observation, monitor);
        observe_recovery_scope(context, CaptureScope::Region, None, Some(region)).await?
    };
    observation.warnings.push(reason.into());
    *context.latest_observation.lock() = Some(observation.clone());
    *context.latest_observation_view.lock() = None;
    *context.pending_visual_localization.lock() = None;
    context.write_artifact(
        &format!("activation-recovery-{}.json", observation.version),
        &json!({
            "requested_window_id": target.id,
            "requested_title": target.title,
            "requested_app": target.process_name,
            "reason": reason,
            "attempts": &attempts,
            "recovery_observation_id": observation.version,
        }),
    )?;
    let mut value = model_observation_value(&observation, context.annotate_targets)?;
    value["status"] = json!("recovery_required");
    value["recovery_required"] = json!(true);
    value["verified"] = json!(false);
    value["executed"] = json!(false);
    value["requested_window"] = json!({
        "id": target.id,
        "title": target.title,
        "app": target.process_name,
        "state": if target.minimized { "minimized" } else { "visible" },
    });
    value["attempts"] = json!(attempts);
    value["reason"] = json!(reason);
    value["instruction"] = json!(
        "Use this enlarged current GUI state and its targets. If needed, inspect_screen_region again, use keyboard navigation, or call locate_visual_target with a box local to this image. Do not repeat activate_window until desktop state changes."
    );
    if let Some(dismissed) = transient_overlay_dismissed {
        value["transient_overlay_dismissed"] = json!(dismissed);
    }
    if context.annotate_targets && !observation.targets.is_empty() {
        save_annotated_model_image(context, &observation, &value)?;
    }
    Ok(value)
}

pub(super) fn shell_recovery_region(observation: &Observation, monitor: &MonitorInfo) -> Rect {
    let shell_bounds = observation
        .targets
        .iter()
        .filter(|target| {
            target.actionable
                && target.desktop_shell
                && matches!(target.source, TargetSource::Uia | TargetSource::UiaOcr)
                && near_capture_edge(&target.bounds, &monitor.bounds)
        })
        .map(|target| &target.bounds)
        .collect::<Vec<_>>();
    if shell_bounds.is_empty() {
        let height = (monitor.bounds.height / 6)
            .max(96)
            .min(monitor.bounds.height);
        return Rect {
            x: monitor.bounds.x,
            y: monitor.bounds.y.saturating_add(
                i32::try_from(monitor.bounds.height.saturating_sub(height)).unwrap_or(i32::MAX),
            ),
            width: monitor.bounds.width,
            height,
        };
    }
    let left = shell_bounds
        .iter()
        .map(|bounds| bounds.x)
        .min()
        .unwrap_or(monitor.bounds.x);
    let top = shell_bounds
        .iter()
        .map(|bounds| bounds.y)
        .min()
        .unwrap_or(monitor.bounds.y);
    let right = shell_bounds
        .iter()
        .map(|bounds| i64::from(bounds.x) + i64::from(bounds.width))
        .max()
        .unwrap_or(i64::from(left));
    let bottom = shell_bounds
        .iter()
        .map(|bounds| i64::from(bounds.y) + i64::from(bounds.height))
        .max()
        .unwrap_or(i64::from(top));
    let padding = 24_i64;
    let monitor_right = i64::from(monitor.bounds.x) + i64::from(monitor.bounds.width);
    let monitor_bottom = i64::from(monitor.bounds.y) + i64::from(monitor.bounds.height);
    let x = i64::from(left)
        .saturating_sub(padding)
        .max(i64::from(monitor.bounds.x));
    let y = i64::from(top)
        .saturating_sub(padding)
        .max(i64::from(monitor.bounds.y));
    let right = right.saturating_add(padding).min(monitor_right);
    let bottom = bottom.saturating_add(padding).min(monitor_bottom);
    Rect {
        x: i32::try_from(x).unwrap_or(monitor.bounds.x),
        y: i32::try_from(y).unwrap_or(monitor.bounds.y),
        width: u32::try_from((right - x).max(1)).unwrap_or(monitor.bounds.width),
        height: u32::try_from((bottom - y).max(1)).unwrap_or(monitor.bounds.height),
    }
}

pub(super) fn is_transient_shell_overlay(window: &WindowInfo) -> bool {
    let process = window.process_name.to_ascii_lowercase();
    let title = window.title.trim().to_ascii_lowercase();
    process == "shellhost.exe"
        && matches!(
            title.as_str(),
            "quick settings" | "notification center" | "calendar"
        )
}
