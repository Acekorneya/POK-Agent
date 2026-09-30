//! Seeing the screen: desktop overview, window/monitor capture, region inspection, visual localization, and the observation the model receives (ordered content, OCR, annotations, saved visuals).

use super::*;

pub(super) struct ObserveDesktopTool;
#[async_trait]
impl Tool for ObserveDesktopTool {
    fn name(&self) -> &'static str {
        "observe_desktop"
    }
    fn description(&self) -> &'static str {
        "Return a compact multi-monitor overview and task-ranked windows. Use this for taskbar, Start, tray, minimized, launching, or cross-monitor tasks; then capture one monitor before input."
    }
    fn input_schema(&self) -> Value {
        json!({"type": "object", "properties": {}})
    }
    fn risk(&self) -> RiskClass {
        RiskClass::ReadOnly
    }
    async fn execute(&self, _args: Value, context: &ToolContext) -> Result<Value> {
        let started = Instant::now();
        *context.latest_observation.lock() = None;
        *context.latest_observation_view.lock() = None;
        *context.pending_visual_localization.lock() = None;
        *context.focused_control.lock() = None;
        let request = CaptureRequest {
            scope: CaptureScope::All,
            window_id: None,
            monitor_id: None,
            region: None,
            max_edge: context.vision_max_edge,
        };
        let (capture, monitors, windows) = tokio::join!(
            context.platform.capture_target(&request),
            context.platform.list_monitors(),
            context.platform.list_windows(),
        );
        let capture = capture?;
        let mut monitors = monitors?;
        monitors.sort_by_key(|monitor| (!monitor.primary, monitor.bounds.y, monitor.bounds.x));
        let windows = windows?;
        let total_windows = windows.len();
        let summaries = compact_windows(windows, &monitors, &context.task_hint.lock(), 40);
        let screenshot = capture
            .screenshots
            .first()
            .ok_or_else(|| PokError::Tool("desktop overview returned no image".into()))?;
        let overview_png =
            annotate_monitor_overview(&capture.target.bounds, screenshot, &monitors)?;
        let version = Uuid::new_v4();
        let value = json!({
            "overview_version": version,
            "input_authorized": false,
            "instruction": "Choose a monitor_id and call capture_screen with scope=monitor before clicking.",
            "desktop_bounds": capture.target.bounds,
            "monitors": monitors.iter().enumerate().map(|(index, monitor)| compact_monitor(monitor, index + 1)).collect::<Vec<_>>(),
            "windows": summaries,
            "window_count": total_windows,
            "windows_truncated": total_windows > 40,
            "timings_ms": {
                "total": u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
            },
            "screenshots": [{
                "png_base64": overview_png,
                "width": screenshot.model_width,
                "height": screenshot.model_height,
                "annotated_monitors": true,
            }],
        });
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(
                value
                    .pointer("/screenshots/0/png_base64")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
            )
            .map_err(|error| PokError::Tool(format!("invalid overview PNG: {error}")))?;
        std::fs::create_dir_all(&context.artifact_dir)?;
        crate::memory::atomic_write(
            &context
                .artifact_dir
                .join(format!("desktop-overview-{version}.png")),
            &bytes,
        )?;
        context.write_artifact(
            &format!("desktop-overview-{version}.json"),
            &crate::tool::diagnostic_value(&value),
        )?;
        Ok(value)
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
pub(super) struct CaptureArgs {
    #[serde(default)]
    pub(super) scope: CaptureScope,
    #[serde(default)]
    pub(super) window_id: Option<String>,
    #[serde(default)]
    pub(super) monitor_id: Option<String>,
    #[serde(default = "default_true")]
    pub(super) include_ocr: bool,
    #[serde(default = "default_true")]
    pub(super) include_ui_tree: bool,
    /// `fast` bounds optional OCR/UIA work for responsive computer use.
    /// `deep` waits longer when accessibility structure is essential.
    #[serde(default)]
    pub(super) enrichment: CaptureEnrichment,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(super) enum CaptureEnrichment {
    #[default]
    Fast,
    Deep,
    None,
}

pub(super) struct CaptureScreenTool;
#[async_trait]
impl Tool for CaptureScreenTool {
    fn name(&self) -> &'static str {
        "capture_screen"
    }
    fn description(&self) -> &'static str {
        "Capture a target window by default and return a numbered image plus compact UIA/OCR interaction targets."
    }
    fn input_schema(&self) -> Value {
        schema::<CaptureArgs>()
    }
    fn risk(&self) -> RiskClass {
        RiskClass::ReadOnly
    }
    async fn execute(&self, args: Value, context: &ToolContext) -> Result<Value> {
        let mut args: CaptureArgs = serde_json::from_value(args)?;
        if let Some(window_id) = args.window_id.as_deref() {
            args.window_id = Some(resolve_window_id(context, window_id).await?);
        }
        let scope = normalized_capture_scope(
            args.scope,
            args.window_id.as_deref(),
            args.monitor_id.as_deref(),
        )?;
        if let Some(window_id) = args.window_id.as_deref() {
            validate_full_window_id(window_id)?;
        }
        if matches!(scope, CaptureScope::Window) && args.window_id.is_none() {
            return Err(PokError::Tool(
                "window_id is required when scope is window".into(),
            ));
        }
        if matches!(scope, CaptureScope::Monitor) && args.monitor_id.is_none() {
            return Err(PokError::Tool(
                "monitor_id is required when scope is monitor".into(),
            ));
        }
        let request = CaptureRequest {
            scope,
            window_id: args.window_id,
            monitor_id: args.monitor_id,
            region: None,
            max_edge: context.vision_max_edge,
        };
        let (include_ocr, include_ui_tree, enrichment_timeout_ms) = match args.enrichment {
            CaptureEnrichment::Fast => (
                args.include_ocr,
                args.include_ui_tree,
                context.desktop_enrichment_timeout_ms,
            ),
            CaptureEnrichment::Deep => (
                args.include_ocr,
                args.include_ui_tree,
                context.desktop_deep_enrichment_timeout_ms,
            ),
            CaptureEnrichment::None => (false, false, context.desktop_enrichment_timeout_ms),
        };
        let total_started = Instant::now();
        let capture_started = Instant::now();
        let capture = context.platform.capture_target(&request).await?;
        let capture_ms = u64::try_from(capture_started.elapsed().as_millis()).unwrap_or(u64::MAX);
        let current_foreground = context.platform.foreground_window().await?;
        let cached = context.latest_observation.lock().clone();
        let cache_source = cached.as_ref().filter(|prior| {
            reusable_observation(
                prior,
                &capture,
                current_foreground.as_ref(),
                include_ocr,
                include_ui_tree,
            )
        });
        let cache_hit = cache_source.is_some();
        let cache_source_id = cache_source.map(|prior| observation_id(prior.version));
        let mut observation = if let Some(prior) = cache_source {
            let mut reused = prior.clone();
            reused.version = Uuid::new_v4();
            reused.captured_at = chrono::Utc::now();
            reused.foreground_window = current_foreground;
            reused.cursor = context.platform.cursor_position().await?;
            reused.target = Some(capture.target.clone());
            reused.screenshots = capture.screenshots.clone();
            reused.timings_ms = capture
                .timings_ms
                .iter()
                .map(|(name, elapsed)| (format!("capture_{name}"), *elapsed))
                .collect();
            reused.timings_ms.insert("capture".into(), capture_ms);
            reused.timings_ms.insert("ocr".into(), 0);
            reused.timings_ms.insert("uia".into(), 0);
            reused.timings_ms.insert("cache_hit".into(), 1);
            reused.timings_ms.insert(
                "total".into(),
                u64::try_from(total_started.elapsed().as_millis()).unwrap_or(u64::MAX),
            );
            reused
        } else {
            context
                .platform
                .observe_capture(
                    capture,
                    &request,
                    include_ocr,
                    include_ui_tree,
                    context.uia_element_limit,
                    capture_ms,
                    std::time::Duration::from_millis(enrichment_timeout_ms),
                )
                .await?
        };
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
        let submission = reconcile_pending_submission(&observation, context);
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
        let mut value = model_observation_value(&observation, context.annotate_targets)?;
        let verification_issues = observation_verification_issues(&observation);
        value["verification_state"] = json!(if verification_issues.is_empty() {
            "clean"
        } else {
            "issue_detected"
        });
        value["verification_issues"] = json!(verification_issues);
        value["cache"] = json!({
            "hit": cache_hit,
            "basis": cache_hit.then_some("exact_target_image_and_foreground"),
            "source_observation_id": cache_source_id,
        });
        if cache_hit {
            value["state"] = json!("unchanged");
            value["ordered_content"] = json!([]);
            value["content_order"] = json!(
                "unchanged from source_observation_id; reuse its text and current target list"
            );
        }
        if let Some(submission) = submission {
            value["submission"] = submission;
        }
        if context.annotate_targets && !observation.targets.is_empty() {
            save_annotated_model_image(context, &observation, &value)?;
        }
        Ok(value)
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
pub(super) struct InspectScreenRegionArgs {
    /// The exact id returned by the most recent capture_screen or region inspection.
    pub(super) observation_id: String,
    /// Optional A1-H8 cell from the grid described by the source observation.
    #[serde(default)]
    pub(super) grid_cell: Option<String>,
    /// Fresh numbered target to enlarge instead of calculating a rectangle.
    #[serde(default)]
    pub(super) target_id: Option<TargetId>,
    /// Region coordinates in pixels of the supplied model image.
    #[serde(default)]
    pub(super) x: Option<i32>,
    #[serde(default)]
    pub(super) y: Option<i32>,
    #[serde(default)]
    pub(super) width: Option<u32>,
    #[serde(default)]
    pub(super) height: Option<u32>,
    /// Padding in model-image pixels when target_id is used.
    #[serde(default)]
    pub(super) padding: Option<u32>,
    /// Optional model-image edge. Values above the configured vision limit are clamped.
    #[serde(default)]
    pub(super) max_edge: Option<u32>,
}

pub(super) struct InspectScreenRegionTool;

#[async_trait]
impl Tool for InspectScreenRegionTool {
    fn name(&self) -> &'static str {
        "inspect_screen_region"
    }

    fn description(&self) -> &'static str {
        "Create a derived enlarged view from a fresh target_id, A1-H8 grid_cell, or rectangle without staling the source observation. Rectangle coordinates are local to the source model image and are safely clamped. Use the returned view_id with pointer tools when acting in the crop."
    }

    fn input_schema(&self) -> Value {
        schema::<InspectScreenRegionArgs>()
    }

    fn risk(&self) -> RiskClass {
        RiskClass::ReadOnly
    }

    async fn execute(&self, args: Value, context: &ToolContext) -> Result<Value> {
        let args: InspectScreenRegionArgs = serde_json::from_value(args)?;
        let parent =
            context.latest_observation.lock().clone().ok_or_else(|| {
                PokError::Tool("no current observation; capture_screen first".into())
            })?;
        let parent_id = observation_id_for(&parent);
        if args.observation_id != parent_id {
            return Err(PokError::Tool(format!(
                "observation {:?} is stale; inspect only the latest observation {:?}",
                args.observation_id, parent_id
            )));
        }
        let screenshot = parent
            .screenshots
            .first()
            .ok_or_else(|| PokError::Tool("observation contains no model image".into()))?;
        let target = parent
            .target
            .as_ref()
            .ok_or_else(|| PokError::Tool("observation contains no capture target".into()))?;
        let supplied_rectangle =
            args.x.is_some() || args.y.is_some() || args.width.is_some() || args.height.is_some();
        let selector_count = usize::from(args.target_id.is_some())
            + usize::from(args.grid_cell.is_some())
            + usize::from(supplied_rectangle);
        if selector_count != 1 {
            return Err(PokError::Tool(
                "use exactly one of target_id, grid_cell, or x/y/width/height".into(),
            ));
        }
        let model_region = if let Some(target_id) = args.target_id.as_ref() {
            let target_id = target_id.normalized();
            let candidate = parent
                .targets
                .iter()
                .find(|candidate| candidate.id == target_id)
                .ok_or_else(|| PokError::Tool(format!("unknown target {target_id:?}")))?;
            let candidate = local_rect(&candidate.bounds, &target.bounds, screenshot);
            let padding = i32::try_from(args.padding.unwrap_or(24).min(128)).unwrap_or(128);
            let left = candidate.x.saturating_sub(padding).max(0);
            let top = candidate.y.saturating_sub(padding).max(0);
            let right = (i64::from(candidate.x) + i64::from(candidate.width) + i64::from(padding))
                .min(i64::from(screenshot.model_width));
            let bottom =
                (i64::from(candidate.y) + i64::from(candidate.height) + i64::from(padding))
                    .min(i64::from(screenshot.model_height));
            Rect {
                x: left,
                y: top,
                width: u32::try_from((right - i64::from(left)).max(1)).unwrap_or(u32::MAX),
                height: u32::try_from((bottom - i64::from(top)).max(1)).unwrap_or(u32::MAX),
            }
        } else if let Some(grid_cell) = args.grid_cell.as_deref() {
            grid_cell_rect(grid_cell, screenshot.model_width, screenshot.model_height)?
        } else {
            let (x, y, width, height) = match (args.x, args.y, args.width, args.height) {
                (Some(x), Some(y), Some(width), Some(height)) => (x, y, width, height),
                _ => {
                    return Err(PokError::Tool(
                        "provide target_id or all of x, y, width, and height".into(),
                    ));
                }
            };
            if x < 0 || y < 0 || width == 0 || height == 0 {
                return Err(PokError::Tool(
                    "region must have non-negative x/y and non-zero width/height".into(),
                ));
            }
            clamp_model_rect(
                Rect {
                    x,
                    y,
                    width,
                    height,
                },
                screenshot.model_width,
                screenshot.model_height,
            )?
        };
        let physical = model_rect_to_physical(
            &model_region,
            screenshot.model_width,
            screenshot.model_height,
            &target.bounds,
        )?;
        let request = CaptureRequest {
            scope: CaptureScope::Region,
            window_id: None,
            monitor_id: None,
            region: Some(physical.clone()),
            max_edge: args
                .max_edge
                .unwrap_or(context.vision_max_edge.max(1))
                .clamp(1, context.vision_max_edge.max(1)),
        };
        let view_id = format!("view_{}", &Uuid::new_v4().simple().to_string()[..8]);
        let capture = derived_capture(&parent, &physical, &view_id, request.max_edge)?;
        let mut observation = context
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
        let expected_foreground = parent
            .foreground_window
            .as_ref()
            .map(|window| window.id.as_str());
        if expected_foreground.is_some_and(|expected| {
            !foreground_matches(expected, observation.foreground_window.as_ref())
        }) {
            return Err(PokError::Tool(
                "desktop focus changed while deriving the view; capture the intended window again"
                    .into(),
            ));
        }
        observation.version = parent.version;
        observation.ui_elements = parent
            .ui_elements
            .iter()
            .filter(|element| overlaps(&element.bounds, &physical))
            .cloned()
            .collect();
        observation.targets = build_targets(
            &observation,
            context.model_target_limit,
            context.fusion_iou_threshold,
            context.ocr_containment_threshold,
            &context.task_hint.lock(),
        );
        context.write_artifact(
            &format!("derived-{view_id}.json"),
            &crate::tool::diagnostic_value(&serde_json::to_value(&observation)?),
        )?;
        *context.latest_observation_view.lock() = Some(DerivedObservationView {
            id: view_id.clone(),
            source_observation_id: parent_id.clone(),
            observation: observation.clone(),
        });
        let mut value = model_observation_value(&observation, context.annotate_targets)?;
        value["view_id"] = json!(view_id);
        value["source_observation_id"] = json!(parent_id);
        value["source_model_region"] = json!({
            "x": model_region.x,
            "y": model_region.y,
            "width": model_region.width,
            "height": model_region.height,
        });
        value["instruction"] = json!(
            "Reason from this enlarged derived view. Use its view_id with click_target, locate_visual_target, drag_target, drag_pointer, or move_pointer. Coordinates are local to this view; the source observation remains authoritative."
        );
        Ok(value)
    }
}

pub(super) fn grid_cell_rect(cell: &str, width: u32, height: u32) -> Result<Rect> {
    let normalized = cell.trim().to_ascii_uppercase();
    let mut chars = normalized.chars();
    let column = chars
        .next()
        .filter(|value| ('A'..='H').contains(value))
        .ok_or_else(|| PokError::Tool("grid_cell must be A1 through H8".into()))?;
    let row = chars
        .next()
        .and_then(|value| value.to_digit(10))
        .filter(|value| (1..=8).contains(value))
        .ok_or_else(|| PokError::Tool("grid_cell must be A1 through H8".into()))?;
    if chars.next().is_some() {
        return Err(PokError::Tool("grid_cell must be A1 through H8".into()));
    }
    let column = u32::from(column) - u32::from('A');
    let row = row - 1;
    let left = width.saturating_mul(column) / 8;
    let top = height.saturating_mul(row) / 8;
    let right = width.saturating_mul(column + 1) / 8;
    let bottom = height.saturating_mul(row + 1) / 8;
    Ok(Rect {
        x: i32::try_from(left).unwrap_or(i32::MAX),
        y: i32::try_from(top).unwrap_or(i32::MAX),
        width: right.saturating_sub(left).max(1),
        height: bottom.saturating_sub(top).max(1),
    })
}

pub(super) fn clamp_model_rect(rect: Rect, width: u32, height: u32) -> Result<Rect> {
    if rect.x < 0 || rect.y < 0 || rect.width == 0 || rect.height == 0 {
        return Err(PokError::Tool(
            "region must have non-negative x/y and non-zero width/height".into(),
        ));
    }
    let x = u32::try_from(rect.x).unwrap_or(u32::MAX);
    let y = u32::try_from(rect.y).unwrap_or(u32::MAX);
    if x >= width || y >= height {
        return Err(PokError::Tool(format!(
            "region starts outside model image {width}x{height}"
        )));
    }
    Ok(Rect {
        x: rect.x,
        y: rect.y,
        width: rect.width.min(width - x).max(1),
        height: rect.height.min(height - y).max(1),
    })
}

pub(super) fn derived_capture(
    parent: &Observation,
    physical: &Rect,
    view_id: &str,
    max_edge: u32,
) -> Result<DesktopCapture> {
    let parent_target = parent
        .target
        .as_ref()
        .ok_or_else(|| PokError::Tool("observation contains no capture target".into()))?;
    let screenshot = parent
        .screenshots
        .first()
        .ok_or_else(|| PokError::Tool("observation contains no source image".into()))?;
    let encoded = screenshot
        .source_png_base64
        .as_deref()
        .unwrap_or(&screenshot.png_base64);
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .map_err(|error| PokError::Tool(format!("invalid source image: {error}")))?;
    let source = image::load_from_memory(&bytes)
        .map_err(|error| PokError::Tool(format!("invalid source image: {error}")))?
        .into_rgba8();
    let local_x = scale_offset(
        physical.x - parent_target.bounds.x,
        parent_target.bounds.width,
        source.width(),
    )
    .max(0) as u32;
    let local_y = scale_offset(
        physical.y - parent_target.bounds.y,
        parent_target.bounds.height,
        source.height(),
    )
    .max(0) as u32;
    let crop_width = scale_length(physical.width, parent_target.bounds.width, source.width())
        .max(1)
        .min(source.width().saturating_sub(local_x));
    let crop_height = scale_length(
        physical.height,
        parent_target.bounds.height,
        source.height(),
    )
    .max(1)
    .min(source.height().saturating_sub(local_y));
    let crop =
        image::imageops::crop_imm(&source, local_x, local_y, crop_width, crop_height).to_image();
    let longest = crop.width().max(crop.height()).max(1);
    let model = if longest > max_edge.max(1) {
        let width =
            (u64::from(crop.width()) * u64::from(max_edge.max(1)) / u64::from(longest)) as u32;
        let height =
            (u64::from(crop.height()) * u64::from(max_edge.max(1)) / u64::from(longest)) as u32;
        image::imageops::resize(
            &crop,
            width.max(1),
            height.max(1),
            image::imageops::FilterType::Triangle,
        )
    } else {
        crop.clone()
    };
    let encode = |image: &image::RgbaImage| -> Result<String> {
        let mut cursor = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(image.clone())
            .write_to(&mut cursor, image::ImageFormat::Png)
            .map_err(|error| PokError::Tool(format!("failed to encode derived view: {error}")))?;
        Ok(base64::engine::general_purpose::STANDARD.encode(cursor.into_inner()))
    };
    Ok(DesktopCapture {
        target: CaptureTarget {
            scope: CaptureScope::Region,
            id: view_id.to_owned(),
            title: "Derived screen view".into(),
            process_name: parent_target.process_name.clone(),
            bounds: physical.clone(),
        },
        screenshots: vec![Screenshot {
            monitor: MonitorInfo {
                id: view_id.to_owned(),
                bounds: physical.clone(),
                scale_factor: screenshot.monitor.scale_factor,
                primary: screenshot.monitor.primary,
            },
            png_base64: encode(&model)?,
            source_png_base64: Some(encode(&crop)?),
            model_width: model.width(),
            model_height: model.height(),
            captured_at: screenshot.captured_at,
        }],
        timings_ms: Default::default(),
    })
}

pub(super) fn annotate_localization_capture(
    capture: &mut DesktopCapture,
    proposed: &Rect,
    confirmation: &Rect,
    resolved_point: (i32, i32),
) -> Result<()> {
    let screenshot = capture
        .screenshots
        .first_mut()
        .ok_or_else(|| PokError::Tool("localization confirmation has no image".into()))?;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(&screenshot.png_base64)
        .map_err(|error| PokError::Tool(format!("invalid confirmation image: {error}")))?;
    let mut image = image::load_from_memory(&bytes)
        .map_err(|error| PokError::Tool(format!("invalid confirmation image: {error}")))?
        .into_rgba8();
    let map_x = |value: i32| {
        scale_offset(
            value.saturating_sub(confirmation.x),
            confirmation.width,
            image.width(),
        )
        .clamp(
            0,
            i32::try_from(image.width().saturating_sub(1)).unwrap_or(i32::MAX),
        ) as u32
    };
    let map_y = |value: i32| {
        scale_offset(
            value.saturating_sub(confirmation.y),
            confirmation.height,
            image.height(),
        )
        .clamp(
            0,
            i32::try_from(image.height().saturating_sub(1)).unwrap_or(i32::MAX),
        ) as u32
    };
    let left = map_x(proposed.x);
    let top = map_y(proposed.y);
    let right = map_x(
        proposed
            .x
            .saturating_add(i32::try_from(proposed.width).unwrap_or(i32::MAX)),
    );
    let bottom = map_y(
        proposed
            .y
            .saturating_add(i32::try_from(proposed.height).unwrap_or(i32::MAX)),
    );
    let center_x = map_x(resolved_point.0);
    let center_y = map_y(resolved_point.1);
    let color = image::Rgba([255, 40, 80, 255]);
    for thickness in 0..3_u32 {
        let x1 = left.saturating_add(thickness).min(right);
        let x2 = right.saturating_sub(thickness).max(left);
        let y1 = top.saturating_add(thickness).min(bottom);
        let y2 = bottom.saturating_sub(thickness).max(top);
        for x in x1..=x2 {
            image.put_pixel(x, y1, color);
            image.put_pixel(x, y2, color);
        }
        for y in y1..=y2 {
            image.put_pixel(x1, y, color);
            image.put_pixel(x2, y, color);
        }
    }
    for offset in -7_i32..=7 {
        let x = center_x as i32 + offset;
        let y = center_y as i32 + offset;
        if x >= 0 && (x as u32) < image.width() {
            image.put_pixel(x as u32, center_y, color);
        }
        if y >= 0 && (y as u32) < image.height() {
            image.put_pixel(center_x, y as u32, color);
        }
    }
    let mut cursor = Cursor::new(Vec::new());
    image::DynamicImage::ImageRgba8(image)
        .write_to(&mut cursor, image::ImageFormat::Png)
        .map_err(|error| PokError::Tool(format!("failed to encode confirmation image: {error}")))?;
    screenshot.png_base64 = base64::engine::general_purpose::STANDARD.encode(cursor.into_inner());
    Ok(())
}

pub(super) fn save_localization_png(
    context: &ToolContext,
    name: &str,
    encoded: &str,
) -> Result<()> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .map_err(|error| PokError::Tool(format!("invalid localization PNG: {error}")))?;
    std::fs::create_dir_all(&context.artifact_dir)?;
    crate::memory::atomic_write(&context.artifact_dir.join(name), &bytes)
}

pub(super) fn model_rect_to_physical(
    model: &Rect,
    model_width: u32,
    model_height: u32,
    physical: &Rect,
) -> Result<Rect> {
    if model_width == 0 || model_height == 0 || physical.width == 0 || physical.height == 0 {
        return Err(PokError::Tool("cannot map an empty screen image".into()));
    }
    let scale_floor = |offset: i64, model_extent: u32, physical_extent: u32| {
        offset.saturating_mul(i64::from(physical_extent)) / i64::from(model_extent)
    };
    let scale_ceil = |offset: i64, model_extent: u32, physical_extent: u32| {
        (offset
            .saturating_mul(i64::from(physical_extent))
            .saturating_add(i64::from(model_extent) - 1))
            / i64::from(model_extent)
    };
    let left = scale_floor(i64::from(model.x), model_width, physical.width);
    let top = scale_floor(i64::from(model.y), model_height, physical.height);
    let right = scale_ceil(
        i64::from(model.x) + i64::from(model.width),
        model_width,
        physical.width,
    );
    let bottom = scale_ceil(
        i64::from(model.y) + i64::from(model.height),
        model_height,
        physical.height,
    );
    Ok(Rect {
        x: physical
            .x
            .saturating_add(i32::try_from(left).unwrap_or(i32::MAX)),
        y: physical
            .y
            .saturating_add(i32::try_from(top).unwrap_or(i32::MAX)),
        width: u32::try_from((right - left).max(1)).unwrap_or(u32::MAX),
        height: u32::try_from((bottom - top).max(1)).unwrap_or(u32::MAX),
    })
}

pub(super) fn observation_verification_issues(observation: &Observation) -> Vec<String> {
    let mut candidates = observation
        .target
        .iter()
        .filter_map(|target| verification_issue_title(&target.title))
        .chain(observation.ui_elements.iter().filter_map(|element| {
            let role = element.control_type.to_ascii_lowercase();
            if role.contains("dialog") || role.contains("window") {
                verification_issue_title(&element.name)
                    .or_else(|| verification_issue_text(&element.name))
            } else if role.contains("text") {
                verification_issue_text(&element.name)
            } else {
                None
            }
        }))
        .chain(
            observation
                .ocr
                .iter()
                .filter_map(|block| verification_issue_text(&block.text)),
        )
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>();
    candidates.sort();
    candidates.dedup();
    candidates.truncate(8);
    candidates
}

pub(super) fn verification_issue_text(text: &str) -> Option<String> {
    const ISSUE_MARKERS: &[&str] = &[
        "problem with some content",
        "cannot be opened",
        "could not be opened",
        "corrupt",
        "unreadable",
        "try to repair",
        "needs to be repaired",
        "was repaired",
        "recover as much",
        "failed to",
    ];
    let normalized = text.trim().to_ascii_lowercase();
    ISSUE_MARKERS
        .iter()
        .any(|marker| normalized.contains(marker))
        .then(|| text.trim().chars().take(300).collect::<String>())
}

pub(super) fn verification_issue_title(text: &str) -> Option<String> {
    let normalized = text.trim().to_ascii_lowercase();
    [
        " - repaired",
        " - recovered",
        " - error",
        "error - ",
        "corrupt",
        "unreadable",
    ]
    .iter()
    .any(|marker| normalized.contains(marker))
    .then(|| text.trim().chars().take(300).collect::<String>())
}

pub(super) fn reusable_observation(
    prior: &Observation,
    capture: &DesktopCapture,
    current_foreground: Option<&WindowInfo>,
    include_ocr: bool,
    include_ui: bool,
) -> bool {
    let Some(prior_target) = prior.target.as_ref() else {
        return false;
    };
    let same_rect = |left: &Rect, right: &Rect| {
        left.x == right.x
            && left.y == right.y
            && left.width == right.width
            && left.height == right.height
    };
    let same_target = prior_target.scope == capture.target.scope
        && prior_target.id == capture.target.id
        && same_rect(&prior_target.bounds, &capture.target.bounds);
    let same_image = prior.screenshots.len() == capture.screenshots.len()
        && prior
            .screenshots
            .iter()
            .zip(&capture.screenshots)
            .all(|(left, right)| {
                left.png_base64 == right.png_base64
                    && left.model_width == right.model_width
                    && left.model_height == right.model_height
            });
    let same_foreground = match (prior.foreground_window.as_ref(), current_foreground) {
        (None, None) => true,
        (Some(left), Some(right)) => left.id == right.id,
        _ => false,
    };
    let has_requested_enrichment =
        (!include_ocr || !prior.ocr.is_empty()) && (!include_ui || !prior.ui_elements.is_empty());
    same_target && same_image && same_foreground && has_requested_enrichment
}

pub(super) fn normalized_capture_scope(
    scope: CaptureScope,
    window_id: Option<&str>,
    monitor_id: Option<&str>,
) -> Result<CaptureScope> {
    if window_id.is_some() && monitor_id.is_some() {
        return Err(PokError::Tool(
            "window_id and monitor_id cannot be used together".into(),
        ));
    }
    match scope {
        CaptureScope::ActiveWindow if window_id.is_some() => Ok(CaptureScope::Window),
        CaptureScope::ActiveWindow if monitor_id.is_some() => Ok(CaptureScope::Monitor),
        CaptureScope::Window if monitor_id.is_some() => {
            Err(PokError::Tool("scope=window cannot use monitor_id".into()))
        }
        CaptureScope::Monitor if window_id.is_some() => {
            Err(PokError::Tool("scope=monitor cannot use window_id".into()))
        }
        CaptureScope::Region => Err(PokError::Tool(
            "use inspect_screen_region with the latest observation id and model-image rectangle"
                .into(),
        )),
        other => Ok(other),
    }
}

pub(super) fn validate_full_window_id(window_id: &str) -> Result<()> {
    let valid = window_id
        .split_once(":HANDLE(0x")
        .is_some_and(|(pid, handle)| {
            !pid.is_empty()
                && pid.chars().all(|c| c.is_ascii_digit())
                && handle.ends_with(')')
                && handle[..handle.len().saturating_sub(1)]
                    .chars()
                    .all(|c| c.is_ascii_hexdigit())
        });
    if !valid {
        return Err(PokError::Tool(format!(
            "invalid window_id {window_id:?}; call list_windows and copy the complete PID:HANDLE(0x...) id"
        )));
    }
    Ok(())
}

pub(super) async fn resolve_window_id(context: &ToolContext, supplied: &str) -> Result<String> {
    if validate_full_window_id(supplied).is_ok() {
        return Ok(supplied.to_owned());
    }
    let windows = context.platform.list_windows().await?;
    let ranked = compact_windows(
        windows.clone(),
        &context.platform.list_monitors().await?,
        &context.task_hint.lock(),
        100,
    );
    if let Some(index) = supplied
        .strip_prefix("window_")
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|index| *index > 0)
    {
        if let Some(id) = ranked
            .get(index - 1)
            .and_then(|window| window.get("id"))
            .and_then(Value::as_str)
        {
            return Ok(id.to_owned());
        }
    }
    let normalized = supplied
        .trim()
        .trim_start_matches("HANDLE(")
        .trim_end_matches(')')
        .to_ascii_lowercase();
    let matches = windows
        .iter()
        .filter(|window| {
            let id = window.id.to_ascii_lowercase();
            id == normalized
                || id.starts_with(&format!("{normalized}:"))
                || id.starts_with(&format!("{normalized}(0x"))
                || id.ends_with(&format!("handle({normalized})"))
                || id.contains(&format!("handle({normalized}"))
        })
        .collect::<Vec<_>>();
    if matches.len() == 1 {
        return Ok(matches[0].id.clone());
    }
    Err(PokError::Tool(format!(
        "invalid or ambiguous window_id {supplied:?}; call list_windows and use a window_N alias or complete PID:HANDLE(0x...) id"
    )))
}

pub(crate) fn model_observation_value(observation: &Observation, annotate: bool) -> Result<Value> {
    let target = observation
        .target
        .as_ref()
        .ok_or_else(|| PokError::Tool("capture returned no target metadata".into()))?;
    let screenshot = observation
        .screenshots
        .first()
        .ok_or_else(|| PokError::Tool("capture returned no model image".into()))?;
    let localize = |bounds: &Rect| local_rect(bounds, &target.bounds, screenshot);
    let target_value = |element: &InteractionTarget| {
        let bounds = localize(&element.bounds);
        let interaction_point = element
            .click_point
            .map(|(x, y)| local_point(x, y, &target.bounds, screenshot));
        // Text only OCR read, shown once: click_target accepts it by its
        // exact label, so present it as clickable.
        let clickable_text = unique_exact_ocr_text(observation, element, &element.name);
        let quality = crate::grounding::grounding_quality(element);
        json!({
            "id": element.id,
            "label": element.name,
            "role": element.control_type,
            "box": [bounds.x, bounds.y, bounds.width, bounds.height],
            "source": element.source,
            "actionable": element.actionable || clickable_text,
            "grounding_quality": if clickable_text && quality == "low" { "medium" } else { quality },
            "selected": element.selected,
            "focused": element.focused,
            "desktop_shell": element.desktop_shell,
            "grounding_variant": element.grounding_variant,
            "confidence": element.confidence,
            "interaction_point": interaction_point.map(|(x, y)| vec![x, y]),
        })
    };
    let relevant = crate::grounding::relevant_targets(&observation.targets, 16);
    let annotated_target_ids = sparse_annotation_target_ids(observation, &relevant, 48);
    let targets = observation
        .targets
        .iter()
        .filter(|element| {
            overlaps(&element.bounds, &target.bounds) && annotated_target_ids.contains(&element.id)
        })
        .map(target_value)
        .collect::<Vec<_>>();
    let action_target_count = targets.len();
    let low_quality_target_count = observation
        .targets
        .iter()
        .filter(|target| crate::grounding::grounding_quality(target) == "low")
        .count();
    let relevant_targets = relevant
        .iter()
        .filter(|element| {
            overlaps(&element.bounds, &target.bounds) && annotated_target_ids.contains(&element.id)
        })
        .map(|element| target_value(element))
        .collect::<Vec<_>>();
    let image = if annotate && !annotated_target_ids.is_empty() {
        annotate_screenshot(observation, screenshot, &annotated_target_ids)?
    } else {
        screenshot.png_base64.clone()
    };
    let screenshots = if annotate && !annotated_target_ids.is_empty() {
        vec![
            json!({
                "view": "clean",
                "png_base64": screenshot.png_base64,
                "width": screenshot.model_width,
                "height": screenshot.model_height,
                "annotated_targets": false,
            }),
            json!({
                "view": "annotated",
                "png_base64": image,
                "width": screenshot.model_width,
                "height": screenshot.model_height,
                "annotated_targets": true,
            }),
        ]
    } else {
        vec![json!({
            "view": "clean",
            "png_base64": screenshot.png_base64,
            "width": screenshot.model_width,
            "height": screenshot.model_height,
            "annotated_targets": false,
        })]
    };
    let ordered_content = ordered_content_value(observation, target, screenshot);
    let (primary_content_kind, primary_content) =
        primary_content_value(observation, target, screenshot);
    let observed_url = browser_address_url(observation, target);
    Ok(json!({
        "observation_id": observation_id(observation.version),
        "captured_at": observation.captured_at,
        "target": {
            "scope": target.scope,
            "id": target.id,
            "title": target.title,
            "app": target.process_name,
            "size": [target.bounds.width, target.bounds.height],
        },
        "coordinate_space": {
            "kind": "model_image_pixels",
            "width": screenshot.model_width,
            "height": screenshot.model_height,
            "origin": "top_left"
        },
        "cursor": observation.cursor.map(|(x, y)| local_point(x, y, &target.bounds, screenshot)),
        "screenshots": screenshots,
        "annotation_mode": if annotate { "action_registry" } else { "none" },
        "annotated_target_ids": annotated_target_ids,
        "relevant_targets": relevant_targets,
        "action_targets": targets.clone(),
        "targets": targets,
        "observed_url": observed_url,
        "primary_content_kind": primary_content_kind,
        "primary_content": primary_content,
        "ordered_content": ordered_content,
        "content_order": "top_to_bottom_then_left_to_right; separator rows divide content above from content below",
        "counts": {
            "targets": observation.targets.len(),
            "action_targets": action_target_count,
            "low_quality_targets_suppressed": low_quality_target_count,
            "ocr_blocks": observation.ocr.len(),
            "uia_elements": observation.ui_elements.len(),
        },
        "timings_ms": observation.timings_ms,
        "warnings": observation.warnings,
    }))
}

pub(super) fn compact_post_input_value(observation: &Observation) -> Result<Value> {
    let target = observation
        .target
        .as_ref()
        .ok_or_else(|| PokError::Tool("capture returned no target metadata".into()))?;
    Ok(json!({
        "observation_id": observation_id(observation.version),
        "captured_at": observation.captured_at,
        "target": {
            "scope": target.scope,
            "id": target.id,
            "title": target.title,
            "app": target.process_name,
            "size": [target.bounds.width, target.bounds.height],
        },
        "observed_url": browser_address_url(observation, target),
        "warnings": observation.warnings,
        "timings_ms": observation.timings_ms,
    }))
}

pub(super) fn primary_content_value(
    observation: &Observation,
    target: &CaptureTarget,
    screenshot: &Screenshot,
) -> (&'static str, Vec<Value>) {
    let role_kind = |role: &str| {
        let role = role.trim().to_ascii_lowercase();
        if role.contains("message") {
            Some(("messages", 0_u8))
        } else if role.contains("data item") || role.contains("row") {
            Some(("table_rows", 1))
        } else if role.contains("list item") {
            Some(("list_items", 2))
        } else if role.contains("document") {
            Some(("document", 3))
        } else {
            None
        }
    };
    let best_priority = observation
        .ui_elements
        .iter()
        .filter(|element| overlaps(&element.bounds, &target.bounds))
        .filter(|element| !element.name.trim().is_empty())
        .filter_map(|element| role_kind(&element.control_type).map(|(_, priority)| priority))
        .min();
    let Some(best_priority) = best_priority else {
        return ("none", Vec::new());
    };
    let kind = observation
        .ui_elements
        .iter()
        .find_map(|element| {
            role_kind(&element.control_type)
                .filter(|(_, priority)| *priority == best_priority)
                .map(|(kind, _)| kind)
        })
        .unwrap_or("none");
    let mut candidates = observation
        .ui_elements
        .iter()
        .filter(|element| overlaps(&element.bounds, &target.bounds))
        .filter_map(|element| {
            let text = element.name.trim();
            let (item_kind, priority) = role_kind(&element.control_type)?;
            (priority == best_priority && item_kind == kind && !text.is_empty()).then_some((
                element.bounds.y,
                element.bounds.x,
                &element.bounds,
                element.control_type.trim(),
                text,
            ))
        })
        .collect::<Vec<_>>();
    candidates.sort_by_key(|(y, x, _, _, _)| (*y, *x));

    let mut seen = std::collections::HashSet::new();
    let mut character_count = 0_usize;
    let items = candidates
        .into_iter()
        .filter_map(|(_, _, bounds, role, text)| {
            let key = normalized_text(text);
            if !seen.insert(key) || character_count.saturating_add(text.len()) > 8_000 {
                return None;
            }
            character_count = character_count.saturating_add(text.len());
            let local = local_rect(bounds, &target.bounds, screenshot);
            Some(json!({
                "role": role,
                "text": text,
                "box": [local.x, local.y, local.width, local.height],
                "source": "uia",
            }))
        })
        .take(16)
        .collect();
    (kind, items)
}

pub(super) fn sparse_annotation_target_ids(
    observation: &Observation,
    relevant: &[&InteractionTarget],
    limit: usize,
) -> Vec<String> {
    let mut ids = relevant
        .iter()
        .map(|target| target.id.clone())
        .collect::<Vec<_>>();
    let mut candidates = observation.targets.iter().collect::<Vec<_>>();
    candidates.sort_by_key(|target| {
        std::cmp::Reverse((
            target.focused,
            target.selected == Some(true),
            crate::grounding::grounding_quality(target) == "high",
            target.rank_score,
            target.actionable,
        ))
    });
    for target in candidates {
        if ids.len() >= limit {
            break;
        }
        if crate::grounding::grounding_quality(target) != "low"
            && (target.focused || target.selected == Some(true) || target.actionable)
            && !ids.contains(&target.id)
        {
            ids.push(target.id.clone());
        }
    }
    // Then text that only OCR read and that appears once (menus and labels
    // in applications with little UI Automation), most task-relevant first:
    // click_target accepts it by its exact label, so the planner needs its id.
    let mut texts = observation
        .targets
        .iter()
        .filter(|target| {
            crate::grounding::grounding_quality(target) == "low"
                && !target.id.is_empty()
                && !ids.contains(&target.id)
                && unique_exact_ocr_text(observation, target, &target.name)
        })
        .collect::<Vec<_>>();
    texts.sort_by_key(|target| std::cmp::Reverse(target.rank_score));
    for target in texts {
        if ids.len() >= limit {
            break;
        }
        ids.push(target.id.clone());
    }
    ids
}

pub(super) fn browser_address_url(
    observation: &Observation,
    target: &CaptureTarget,
) -> Option<String> {
    let app = target.process_name.to_ascii_lowercase();
    if !["chrome", "msedge", "firefox", "brave"]
        .iter()
        .any(|browser| app.contains(browser))
    {
        return None;
    }
    observation.ui_elements.iter().find_map(|element| {
        if element.password || !element.control_type.eq_ignore_ascii_case("edit") {
            return None;
        }
        let identity = format!(
            "{} {}",
            element.name,
            element.automation_id.as_deref().unwrap_or_default()
        )
        .to_ascii_lowercase();
        if !["address", "omnibox", "search bar", "urlbar"]
            .iter()
            .any(|marker| identity.contains(marker))
        {
            return None;
        }
        element
            .value
            .as_deref()
            .map(str::trim)
            .filter(|value| looks_like_browser_url(value))
            .map(str::to_owned)
    })
}

pub(super) fn looks_like_browser_url(value: &str) -> bool {
    let value = value.to_ascii_lowercase();
    value.starts_with("http://")
        || value.starts_with("https://")
        || value.starts_with("www.")
        || (value.contains('.') && !value.contains(char::is_whitespace))
}

pub(super) fn ordered_content_value(
    observation: &Observation,
    target: &CaptureTarget,
    screenshot: &Screenshot,
) -> Vec<Value> {
    let mut candidates = observation
        .ui_elements
        .iter()
        .filter(|element| overlaps(&element.bounds, &target.bounds))
        .filter_map(|element| {
            let text = element.name.trim();
            let role = element.control_type.trim();
            let semantic_role = role.to_ascii_lowercase();
            (!text.is_empty()
                && [
                    "document",
                    "heading",
                    "list item",
                    "message",
                    "separator",
                    "text",
                ]
                .iter()
                .any(|candidate| semantic_role.contains(candidate)))
            .then(|| {
                (
                    element.bounds.y,
                    element.bounds.x,
                    element.bounds.clone(),
                    role.to_string(),
                    text.to_string(),
                    "uia",
                )
            })
        })
        .collect::<Vec<_>>();
    candidates.extend(
        crate::grounding::merge_ocr_lines(&observation.ocr)
            .into_iter()
            .filter(|block| overlaps(&block.bounds, &target.bounds))
            .filter_map(|block| {
                let text = block.text.trim().to_string();
                (!text.is_empty()).then(|| {
                    (
                        block.bounds.y,
                        block.bounds.x,
                        block.bounds,
                        "text".into(),
                        text,
                        "ocr",
                    )
                })
            }),
    );
    candidates.sort_by_key(|(y, x, _, _, _, source)| (*y, *x, *source != "uia"));

    let mut seen = std::collections::HashSet::new();
    let mut character_count = 0_usize;
    candidates
        .into_iter()
        .filter_map(|(_, _, bounds, role, text, source)| {
            let key = (
                normalized_text(&text),
                bounds.x / 4,
                bounds.y / 4,
                bounds.width / 4,
                bounds.height / 4,
            );
            if !seen.insert(key) || character_count.saturating_add(text.len()) > 12_000 {
                return None;
            }
            character_count = character_count.saturating_add(text.len());
            let local = local_rect(&bounds, &target.bounds, screenshot);
            Some(json!({
                "role": role,
                "text": text,
                "box": [local.x, local.y, local.width, local.height],
                "source": source,
            }))
        })
        .take(120)
        .collect()
}

pub(super) fn local_rect(bounds: &Rect, target: &Rect, screenshot: &Screenshot) -> Rect {
    let left = bounds.x.max(target.x);
    let top = bounds.y.max(target.y);
    let right = (i64::from(bounds.x) + i64::from(bounds.width))
        .min(i64::from(target.x) + i64::from(target.width));
    let bottom = (i64::from(bounds.y) + i64::from(bounds.height))
        .min(i64::from(target.y) + i64::from(target.height));
    let (x, y) = local_point(left, top, target, screenshot);
    Rect {
        x,
        y,
        width: scale_length(
            (right - i64::from(left)).max(0) as u32,
            target.width,
            screenshot.model_width,
        ),
        height: scale_length(
            (bottom - i64::from(top)).max(0) as u32,
            target.height,
            screenshot.model_height,
        ),
    }
}

pub(super) fn ocr_value(observation: &Observation, task_hint: &str, limit: usize) -> Result<Value> {
    let target = observation
        .target
        .as_ref()
        .ok_or_else(|| PokError::Tool("capture returned no target metadata".into()))?;
    let screenshot = observation
        .screenshots
        .first()
        .ok_or_else(|| PokError::Tool("capture returned no model image".into()))?;
    let mut blocks = crate::grounding::merge_ocr_lines(&observation.ocr)
        .into_iter()
        .filter(|block| overlaps(&block.bounds, &target.bounds))
        .collect::<Vec<_>>();
    let terms = ranking_terms(task_hint);
    blocks.sort_by_key(|block| {
        let text = block.text.to_ascii_lowercase();
        let matches = terms
            .iter()
            .filter(|term| text.contains(term.as_str()))
            .count();
        (std::cmp::Reverse(matches), block.bounds.y, block.bounds.x)
    });
    let total = blocks.len();
    blocks.truncate(limit);
    let lines = blocks
        .iter()
        .map(|block| {
            let bounds = local_rect(&block.bounds, &target.bounds, screenshot);
            json!({
                "text": block.text,
                "box": [bounds.x, bounds.y, bounds.width, bounds.height],
            })
        })
        .collect::<Vec<_>>();
    Ok(json!({
        "observation_id": observation_id(observation.version),
        "lines": lines,
        "line_count": total,
        "truncated": total > limit,
    }))
}

pub(super) fn local_point(x: i32, y: i32, target: &Rect, screenshot: &Screenshot) -> (i32, i32) {
    (
        scale_offset(x - target.x, target.width, screenshot.model_width),
        scale_offset(y - target.y, target.height, screenshot.model_height),
    )
}

pub(super) fn scale_offset(value: i32, source: u32, model: u32) -> i32 {
    if source == 0 {
        return 0;
    }
    ((f64::from(value) * f64::from(model) / f64::from(source)).round()) as i32
}

pub(super) fn scale_length(value: u32, source: u32, model: u32) -> u32 {
    if source == 0 {
        return 0;
    }
    (f64::from(value) * f64::from(model) / f64::from(source)).round() as u32
}

pub(super) fn annotate_screenshot(
    observation: &Observation,
    screenshot: &Screenshot,
    annotated_target_ids: &[String],
) -> Result<String> {
    let target = observation
        .target
        .as_ref()
        .ok_or_else(|| PokError::Tool("capture returned no target metadata".into()))?;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(&screenshot.png_base64)
        .map_err(|error| PokError::Tool(format!("invalid captured PNG: {error}")))?;
    let mut image = image::load_from_memory_with_format(&bytes, image::ImageFormat::Png)
        .map_err(|error| PokError::Tool(format!("cannot decode captured PNG: {error}")))?
        .to_rgba8();
    for item in observation
        .targets
        .iter()
        .filter(|item| annotated_target_ids.contains(&item.id))
    {
        let rect = local_rect(&item.bounds, &target.bounds, screenshot);
        let color = match item.source {
            crate::types::TargetSource::Uia => [57, 198, 149, 255],
            crate::types::TargetSource::Ocr => [255, 159, 67, 255],
            crate::types::TargetSource::UiaOcr => [50, 210, 230, 255],
            crate::types::TargetSource::Visual => [168, 85, 247, 255],
            crate::types::TargetSource::VisualOcr => [217, 70, 239, 255],
        };
        draw_rect(&mut image, &rect, color);
        let badge_y = if rect.y >= 10 {
            rect.y.saturating_sub(10)
        } else {
            rect.y
        };
        draw_badge(
            &mut image,
            rect.x.max(0) as u32,
            badge_y.max(0) as u32,
            &item.id,
        );
    }
    let mut encoded = Cursor::new(Vec::new());
    image::DynamicImage::ImageRgba8(image)
        .write_to(&mut encoded, image::ImageFormat::Png)
        .map_err(|error| PokError::Tool(format!("cannot encode annotated PNG: {error}")))?;
    Ok(base64::engine::general_purpose::STANDARD.encode(encoded.into_inner()))
}

pub(super) fn annotate_monitor_overview(
    desktop_bounds: &Rect,
    screenshot: &Screenshot,
    monitors: &[MonitorInfo],
) -> Result<String> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(&screenshot.png_base64)
        .map_err(|error| PokError::Tool(format!("invalid desktop PNG: {error}")))?;
    let mut image = image::load_from_memory_with_format(&bytes, image::ImageFormat::Png)
        .map_err(|error| PokError::Tool(format!("cannot decode desktop PNG: {error}")))?
        .to_rgba8();
    for (index, monitor) in monitors.iter().enumerate() {
        let rect = local_rect(&monitor.bounds, desktop_bounds, screenshot);
        draw_rect(&mut image, &rect, [250, 204, 21, 255]);
        draw_badge(
            &mut image,
            rect.x.max(0) as u32,
            rect.y.max(0) as u32,
            &(index + 1).to_string(),
        );
    }
    let mut encoded = Cursor::new(Vec::new());
    image::DynamicImage::ImageRgba8(image)
        .write_to(&mut encoded, image::ImageFormat::Png)
        .map_err(|error| PokError::Tool(format!("cannot encode desktop overview: {error}")))?;
    Ok(base64::engine::general_purpose::STANDARD.encode(encoded.into_inner()))
}

pub(super) fn draw_rect(image: &mut image::RgbaImage, rect: &Rect, color: [u8; 4]) {
    let width = image.width();
    let height = image.height();
    if width == 0 || height == 0 || rect.width == 0 || rect.height == 0 {
        return;
    }
    let left = rect.x.max(0) as u32;
    let top = rect.y.max(0) as u32;
    let right =
        (i64::from(rect.x) + i64::from(rect.width) - 1).clamp(0, i64::from(width - 1)) as u32;
    let bottom =
        (i64::from(rect.y) + i64::from(rect.height) - 1).clamp(0, i64::from(height - 1)) as u32;
    if left > right || top > bottom {
        return;
    }
    for offset in 0..2 {
        let x1 = (left + offset).min(right);
        let x2 = right.saturating_sub(offset).max(left);
        let y1 = (top + offset).min(bottom);
        let y2 = bottom.saturating_sub(offset).max(top);
        for x in x1..=x2 {
            image.put_pixel(x, y1, image::Rgba(color));
            image.put_pixel(x, y2, image::Rgba(color));
        }
        for y in y1..=y2 {
            image.put_pixel(x1, y, image::Rgba(color));
            image.put_pixel(x2, y, image::Rgba(color));
        }
    }
}

pub(super) const DIGITS: [[u8; 5]; 10] = [
    [0b111, 0b101, 0b101, 0b101, 0b111],
    [0b010, 0b110, 0b010, 0b010, 0b111],
    [0b111, 0b001, 0b111, 0b100, 0b111],
    [0b111, 0b001, 0b111, 0b001, 0b111],
    [0b101, 0b101, 0b111, 0b001, 0b001],
    [0b111, 0b100, 0b111, 0b001, 0b111],
    [0b111, 0b100, 0b111, 0b101, 0b111],
    [0b111, 0b001, 0b010, 0b010, 0b010],
    [0b111, 0b101, 0b111, 0b101, 0b111],
    [0b111, 0b101, 0b111, 0b001, 0b111],
];

pub(super) fn draw_badge(image: &mut image::RgbaImage, x: u32, y: u32, label: &str) {
    let digits = label
        .bytes()
        .filter(|byte| byte.is_ascii_digit())
        .collect::<Vec<_>>();
    let badge_width = 4 + digits.len() as u32 * 4;
    for py in y..(y + 9).min(image.height()) {
        for px in x..(x + badge_width).min(image.width()) {
            image.put_pixel(px, py, image::Rgba([15, 23, 42, 235]));
        }
    }
    for (index, byte) in digits.into_iter().enumerate() {
        let glyph = DIGITS[(byte - b'0') as usize];
        for (row, bits) in glyph.into_iter().enumerate() {
            for column in 0..3 {
                if bits & (1 << (2 - column)) != 0 {
                    let px = x + 2 + index as u32 * 4 + column;
                    let py = y + 2 + row as u32;
                    if px < image.width() && py < image.height() {
                        image.put_pixel(px, py, image::Rgba([255, 255, 255, 255]));
                    }
                }
            }
        }
    }
}

pub(super) fn save_annotated_model_image(
    context: &ToolContext,
    observation: &Observation,
    value: &Value,
) -> Result<()> {
    let Some(encoded) =
        value
            .get("screenshots")
            .and_then(Value::as_array)
            .and_then(|screenshots| {
                screenshots.iter().find_map(|screenshot| {
                    (screenshot.get("view").and_then(Value::as_str) == Some("annotated"))
                        .then(|| screenshot.get("png_base64").and_then(Value::as_str))
                        .flatten()
                })
            })
    else {
        return Ok(());
    };
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .map_err(|error| PokError::Tool(format!("invalid annotated PNG: {error}")))?;
    crate::memory::atomic_write(
        &context
            .artifact_dir
            .join(format!("observation-{}-annotated.png", observation.version)),
        &bytes,
    )
}

pub(super) fn save_observation_visuals(
    context: &ToolContext,
    observation: &Observation,
) -> Result<()> {
    for screenshot in &observation.screenshots {
        let monitor_id = safe_filename(&screenshot.monitor.id);
        let prefix = format!("observation-{}-monitor-{monitor_id}", observation.version);
        let png_name = format!("{prefix}.png");
        let png = base64::engine::general_purpose::STANDARD
            .decode(&screenshot.png_base64)
            .map_err(|error| PokError::Tool(format!("invalid captured PNG: {error}")))?;
        crate::memory::atomic_write(&context.artifact_dir.join(&png_name), &png)?;
        if let Some(source) = &screenshot.source_png_base64 {
            let source = base64::engine::general_purpose::STANDARD
                .decode(source)
                .map_err(|error| PokError::Tool(format!("invalid source PNG: {error}")))?;
            crate::memory::atomic_write(
                &context.artifact_dir.join(format!("{prefix}-source.png")),
                &source,
            )?;
        }

        let monitor = &screenshot.monitor.bounds;
        let mut overlays = String::new();
        for element in observation
            .ui_elements
            .iter()
            .filter(|element| overlaps(&element.bounds, monitor))
        {
            let x = element.bounds.x - monitor.x;
            let y = element.bounds.y - monitor.y;
            overlays.push_str(&format!(
                r#"<rect x="{x}" y="{y}" width="{}" height="{}" class="uia"><title>UIA: {} ({})</title></rect>"#,
                element.bounds.width,
                element.bounds.height,
                xml_escape(&element.name),
                xml_escape(&element.control_type),
            ));
        }
        for block in observation
            .ocr
            .iter()
            .filter(|block| overlaps(&block.bounds, monitor))
        {
            let x = block.bounds.x - monitor.x;
            let y = block.bounds.y - monitor.y;
            overlays.push_str(&format!(
                r#"<rect x="{x}" y="{y}" width="{}" height="{}" class="ocr"><title>OCR: {}</title></rect>"#,
                block.bounds.width,
                block.bounds.height,
                xml_escape(&block.text),
            ));
        }
        let svg = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">
<style>.uia{{fill:none;stroke:#39c695;stroke-width:2}}.ocr{{fill:none;stroke:#ff9f43;stroke-width:3;stroke-dasharray:8 5}}</style>
<image href="{png_name}" width="{}" height="{}"/>{overlays}</svg>"#,
            monitor.width,
            monitor.height,
            monitor.width,
            monitor.height,
            monitor.width,
            monitor.height,
        );
        crate::memory::atomic_write(
            &context.artifact_dir.join(format!("{prefix}-overlay.svg")),
            svg.as_bytes(),
        )?;
    }
    let ocr_text = observation
        .ocr
        .iter()
        .map(|block| {
            format!(
                "[{},{} {}x{}] {}",
                block.bounds.x, block.bounds.y, block.bounds.width, block.bounds.height, block.text
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    crate::memory::atomic_write(
        &context
            .artifact_dir
            .join(format!("observation-{}-ocr.txt", observation.version)),
        ocr_text.as_bytes(),
    )?;
    Ok(())
}

pub(super) fn overlaps(left: &Rect, right: &Rect) -> bool {
    i64::from(left.x) < i64::from(right.x) + i64::from(right.width)
        && i64::from(left.x) + i64::from(left.width) > i64::from(right.x)
        && i64::from(left.y) < i64::from(right.y) + i64::from(right.height)
        && i64::from(left.y) + i64::from(left.height) > i64::from(right.y)
}

pub(crate) fn safe_filename(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .collect()
}

pub(super) fn xml_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

pub(super) struct QueryScreenTextTool;
#[async_trait]
impl Tool for QueryScreenTextTool {
    fn name(&self) -> &'static str {
        "query_screen_text"
    }
    fn description(&self) -> &'static str {
        "Return OCR text from the latest targeted capture using model-image pixel coordinates."
    }
    fn input_schema(&self) -> Value {
        json!({"type": "object", "properties": {}})
    }
    fn risk(&self) -> RiskClass {
        RiskClass::ReadOnly
    }
    async fn execute(&self, _args: Value, context: &ToolContext) -> Result<Value> {
        let observation = context.latest_observation.lock().clone().ok_or_else(|| {
            PokError::Tool("capture_screen with include_ocr=true must be called first".into())
        })?;
        ocr_value(&observation, &context.task_hint.lock(), 100)
    }
}

pub(super) struct QueryWindowTreeTool;
#[async_trait]
impl Tool for QueryWindowTreeTool {
    fn name(&self) -> &'static str {
        "query_window_tree"
    }
    fn description(&self) -> &'static str {
        "Return the latest compact task-ranked UIA/OCR targets without repeating raw Windows tree data."
    }
    fn input_schema(&self) -> Value {
        json!({"type": "object", "properties": {}})
    }
    fn risk(&self) -> RiskClass {
        RiskClass::ReadOnly
    }
    async fn execute(&self, _args: Value, context: &ToolContext) -> Result<Value> {
        let observation = context.latest_observation.lock().clone().ok_or_else(|| {
            PokError::Tool("capture_screen must be called before query_window_tree".into())
        })?;
        let value = model_observation_value(&observation, false)?;
        Ok(json!({
            "observation_id": observation_id(observation.version),
            "target": value.get("target").cloned().unwrap_or(Value::Null),
            "targets": value.get("targets").cloned().unwrap_or_else(|| json!([])),
            "count": observation.targets.len(),
        }))
    }
}

/// Read the clipboard as text. Copying a spreadsheet or grid range puts it
/// there as tab-separated rows, so select-copy-read verifies cell contents
/// that screenshots and UI Automation do not expose reliably.
pub(super) struct ReadClipboardTool;

/// Rows of a tab-separated copy (spreadsheets, data grids), bounded.
pub(super) fn clipboard_rows(text: &str) -> Option<Vec<Vec<String>>> {
    if !text.contains('\t') {
        return None;
    }
    Some(
        text.lines()
            .take(100)
            .map(|line| {
                line.split('\t')
                    .take(30)
                    .map(|cell| crate::decision::bounded_text(cell, 200))
                    .collect()
            })
            .collect(),
    )
}

#[async_trait]
impl Tool for ReadClipboardTool {
    fn name(&self) -> &'static str {
        "read_clipboard"
    }
    fn description(&self) -> &'static str {
        "Return the current clipboard text. To check spreadsheet or grid contents after a change, select the range, press Ctrl+C, then call this: copied cells come back as rows of columns. Read-only; it does not change the clipboard."
    }
    fn input_schema(&self) -> Value {
        json!({"type": "object", "properties": {}})
    }
    fn risk(&self) -> RiskClass {
        RiskClass::ReadOnly
    }
    async fn execute(&self, _args: Value, _context: &ToolContext) -> Result<Value> {
        if !cfg!(windows) {
            return Err(PokError::Unsupported("read_clipboard needs Windows".into()));
        }
        let output = tokio::time::timeout(
            Duration::from_secs(10),
            tokio::process::Command::new("powershell")
                .args([
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    "[Console]::OutputEncoding = [Text.Encoding]::UTF8; Get-Clipboard -Raw",
                ])
                .output(),
        )
        .await
        .map_err(|_| PokError::Tool("reading the clipboard timed out".into()))??;
        let text = String::from_utf8_lossy(&output.stdout)
            .trim_end_matches(['\r', '\n'])
            .replace("\r\n", "\n");
        let rows = clipboard_rows(&text);
        Ok(json!({
            "text": crate::decision::bounded_text(&text, 20_000),
            "chars": text.chars().count(),
            "row_count": rows.as_ref().map(Vec::len),
            "column_count": rows.as_ref().and_then(|rows| rows.iter().map(Vec::len).max()),
            "rows": rows,
        }))
    }
}
