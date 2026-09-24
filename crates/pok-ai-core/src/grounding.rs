use std::{
    cmp::Ordering,
    collections::{HashSet, VecDeque},
};

use base64::Engine;

use crate::types::{InteractionTarget, Observation, OcrBlock, Rect, TargetSource, UiElement};

const INTERACTIVE_TYPES: &[&str] = &[
    "button",
    "check box",
    "checkbox",
    "combo box",
    "combobox",
    "edit",
    "hyperlink",
    "link",
    "list item",
    "menu item",
    "radio button",
    "slider",
    "spinner",
    "split button",
    "tab item",
    "tree item",
];

pub fn build_targets(
    observation: &Observation,
    limit: usize,
    iou_threshold: f32,
    containment_threshold: f32,
    task_hint: &str,
) -> Vec<InteractionTarget> {
    let Some(window) = observation.target.as_ref() else {
        return Vec::new();
    };
    let window_area = area(&window.bounds).max(1);
    let mut targets = observation
        .ui_elements
        .iter()
        .filter(|element| valid_uia(element, &window.bounds, window_area))
        .map(|element| InteractionTarget {
            id: String::new(),
            name: normalize_text(&element.name),
            control_type: normalize_text(&element.control_type),
            bounds: element.bounds.clone(),
            source: TargetSource::Uia,
            confidence: None,
            enabled: element.enabled,
            actionable: is_actionable(element),
            click_point: is_actionable(element)
                .then(|| {
                    element
                        .clickable_point
                        .filter(|(x, y)| element.bounds.contains(*x, *y))
                })
                .flatten(),
            selected: element.selected,
            focused: element.focused,
            desktop_shell: element.desktop_shell,
            grounding_variant: None,
            rank_score: 0,
            rank_reasons: Vec::new(),
        })
        .collect::<Vec<_>>();
    let terms = task_terms(task_hint);
    let useful_uia = targets.iter().filter(|target| target.actionable).count();
    let task_has_uia_match = targets.iter().any(|target| {
        let label = target.name.to_ascii_lowercase();
        terms.iter().any(|term| label.contains(term))
    });
    if useful_uia < 3 || !task_has_uia_match {
        for candidate in visual_affordance_candidates(observation, 32) {
            if !targets.iter().any(|target| {
                intersection_area(&target.bounds, &candidate.bounds).saturating_mul(2)
                    >= area(&candidate.bounds).max(1)
            }) {
                targets.push(candidate);
            }
        }
    }
    let mut fill_name_from_ocr = targets
        .iter()
        .map(|target| target.name.is_empty())
        .collect::<Vec<_>>();

    let mut ocr_candidates = observation.ocr.clone();
    for line in merge_ocr_lines(&observation.ocr) {
        if !ocr_candidates.iter().any(|block| {
            block.text == line.text
                && block.bounds.x == line.bounds.x
                && block.bounds.y == line.bounds.y
                && block.bounds.width == line.bounds.width
                && block.bounds.height == line.bounds.height
        }) {
            ocr_candidates.push(line);
        }
    }
    for block in ocr_candidates
        .iter()
        .filter(|block| valid_ocr(block, &window.bounds))
    {
        let best = targets
            .iter()
            .enumerate()
            .filter(|(_, target)| {
                overlap_score(
                    &target.bounds,
                    &block.bounds,
                    iou_threshold,
                    containment_threshold,
                ) && ocr_matches_target(target, block)
            })
            .min_by_key(|(_, target)| area(&target.bounds))
            .map(|(index, _)| index);
        if let Some(index) = best {
            targets[index].source = match targets[index].source {
                TargetSource::Visual | TargetSource::VisualOcr => TargetSource::VisualOcr,
                _ => TargetSource::UiaOcr,
            };
            targets[index].confidence = block.confidence;
            targets[index].selected = targets[index].selected.or(block.selected);
            targets[index].grounding_variant = block.variant.clone();
            if fill_name_from_ocr[index] {
                let text = normalize_text(&block.text);
                if !text.is_empty()
                    && !targets[index]
                        .name
                        .split_whitespace()
                        .any(|existing| existing.eq_ignore_ascii_case(&text))
                {
                    if !targets[index].name.is_empty() {
                        targets[index].name.push(' ');
                    }
                    targets[index].name.push_str(&text);
                }
            }
        } else {
            targets.push(InteractionTarget {
                id: String::new(),
                name: normalize_text(&block.text),
                control_type: "text".into(),
                bounds: block.bounds.clone(),
                source: TargetSource::Ocr,
                confidence: block.confidence,
                enabled: true,
                actionable: false,
                click_point: None,
                selected: block.selected,
                focused: false,
                desktop_shell: false,
                grounding_variant: block.variant.clone(),
                rank_score: 0,
                rank_reasons: Vec::new(),
            });
            fill_name_from_ocr.push(false);
        }
    }

    for target in &mut targets {
        if target.name.is_empty() && matches!(target.source, TargetSource::Visual) {
            target.name = match target.control_type.as_str() {
                "slider" => "Visual slider".into(),
                "scroll bar" => "Visual scroll bar".into(),
                _ => "Visual control".into(),
            };
        }
        let (score, reasons) = target_rank(target, &terms);
        target.rank_score = score;
        target.rank_reasons = reasons;
    }
    targets = select_targets(targets, limit, &window.bounds);
    targets.sort_by(spatial_order);
    for (index, target) in targets.iter_mut().enumerate() {
        target.id = (index + 1).to_string();
    }
    targets
}

fn visual_affordance_candidates(observation: &Observation, limit: usize) -> Vec<InteractionTarget> {
    let Some(capture) = observation.target.as_ref() else {
        return Vec::new();
    };
    let Some(screenshot) = observation.screenshots.first() else {
        return Vec::new();
    };
    let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(&screenshot.png_base64) else {
        return Vec::new();
    };
    let Ok(image) = image::load_from_memory(&bytes).map(|image| image.into_rgb8()) else {
        return Vec::new();
    };
    let (width, height) = image.dimensions();
    if width < 32 || height < 32 {
        return Vec::new();
    }
    let pixel_count = usize::try_from(u64::from(width) * u64::from(height)).unwrap_or(usize::MAX);
    if pixel_count > 4_000_000 {
        return Vec::new();
    }
    let mut mask = vec![false; pixel_count];
    for y in 0..height {
        for x in 0..width {
            let pixel = image.get_pixel(x, y).0;
            let maximum = *pixel.iter().max().unwrap_or(&0);
            let minimum = *pixel.iter().min().unwrap_or(&0);
            let saturated = maximum.abs_diff(minimum) >= 48 && maximum >= 64;
            let index = usize::try_from(u64::from(y) * u64::from(width) + u64::from(x))
                .unwrap_or(usize::MAX);
            if index < mask.len() {
                mask[index] = saturated;
            }
        }
    }
    let mut visited = vec![false; mask.len()];
    let mut candidates = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let index = usize::try_from(u64::from(y) * u64::from(width) + u64::from(x))
                .unwrap_or(usize::MAX);
            if index >= mask.len() || !mask[index] || visited[index] {
                continue;
            }
            visited[index] = true;
            let mut queue = VecDeque::from([(x, y)]);
            let (mut left, mut top, mut right, mut bottom) = (x, y, x, y);
            let mut area_pixels = 0_u32;
            let mut row_counts = std::collections::HashMap::<u32, u32>::new();
            let mut column_counts = std::collections::HashMap::<u32, u32>::new();
            while let Some((current_x, current_y)) = queue.pop_front() {
                area_pixels = area_pixels.saturating_add(1);
                *row_counts.entry(current_y).or_default() += 1;
                *column_counts.entry(current_x).or_default() += 1;
                left = left.min(current_x);
                top = top.min(current_y);
                right = right.max(current_x);
                bottom = bottom.max(current_y);
                for (next_x, next_y) in [
                    (current_x.saturating_sub(1), current_y),
                    (current_x.saturating_add(1), current_y),
                    (current_x, current_y.saturating_sub(1)),
                    (current_x, current_y.saturating_add(1)),
                ] {
                    if next_x >= width || next_y >= height {
                        continue;
                    }
                    let next =
                        usize::try_from(u64::from(next_y) * u64::from(width) + u64::from(next_x))
                            .unwrap_or(usize::MAX);
                    if next < mask.len() && mask[next] && !visited[next] {
                        visited[next] = true;
                        queue.push_back((next_x, next_y));
                    }
                }
            }
            let component_width = right.saturating_sub(left).saturating_add(1);
            let component_height = bottom.saturating_sub(top).saturating_add(1);
            let vertical = component_height >= 60
                && component_height >= component_width.saturating_mul(4)
                && (4..=80).contains(&component_width);
            let horizontal = component_width >= 60
                && component_width >= component_height.saturating_mul(4)
                && (4..=80).contains(&component_height);
            let component_area = component_width.saturating_mul(component_height).max(1);
            if area_pixels < 30
                || area_pixels.saturating_mul(100) < component_area.saturating_mul(8)
                || (!vertical && !horizontal)
            {
                continue;
            }
            let (model_x, model_y) = if vertical {
                let thumb_y = row_counts
                    .iter()
                    .max_by_key(|(_, count)| **count)
                    .map_or(top, |(row, _)| *row);
                (left + component_width / 2, thumb_y)
            } else {
                let thumb_x = column_counts
                    .iter()
                    .max_by_key(|(_, count)| **count)
                    .map_or(left, |(column, _)| *column);
                (thumb_x, top + component_height / 2)
            };
            let physical = Rect {
                x: capture.bounds.x + scale_coordinate(left, width, capture.bounds.width),
                y: capture.bounds.y + scale_coordinate(top, height, capture.bounds.height),
                width: scale_extent(component_width, width, capture.bounds.width).max(1),
                height: scale_extent(component_height, height, capture.bounds.height).max(1),
            };
            let click_point = (
                capture.bounds.x + scale_coordinate(model_x, width, capture.bounds.width),
                capture.bounds.y + scale_coordinate(model_y, height, capture.bounds.height),
            );
            candidates.push(InteractionTarget {
                id: String::new(),
                name: String::new(),
                control_type: "slider".into(),
                bounds: physical,
                source: TargetSource::Visual,
                confidence: Some(0.76),
                enabled: true,
                actionable: true,
                click_point: Some(click_point),
                selected: None,
                focused: false,
                desktop_shell: false,
                grounding_variant: Some(
                    if vertical {
                        "deterministic_vertical_track"
                    } else {
                        "deterministic_horizontal_track"
                    }
                    .into(),
                ),
                rank_score: 0,
                rank_reasons: Vec::new(),
            });
        }
    }
    candidates.sort_by_key(|candidate| std::cmp::Reverse(area(&candidate.bounds)));
    let mut deduplicated: Vec<InteractionTarget> = Vec::new();
    for candidate in candidates {
        if deduplicated.iter().all(|existing| {
            intersection_area(&existing.bounds, &candidate.bounds).saturating_mul(2)
                < area(&candidate.bounds).min(area(&existing.bounds)).max(1)
        }) {
            deduplicated.push(candidate);
            if deduplicated.len() >= limit {
                break;
            }
        }
    }
    deduplicated
}

fn scale_coordinate(value: u32, source: u32, destination: u32) -> i32 {
    if source == 0 {
        return 0;
    }
    i32::try_from(
        (u64::from(value) * u64::from(destination) + u64::from(source / 2)) / u64::from(source),
    )
    .unwrap_or(i32::MAX)
}

fn scale_extent(value: u32, source: u32, destination: u32) -> u32 {
    if source == 0 {
        return 0;
    }
    u32::try_from(
        (u64::from(value) * u64::from(destination) + u64::from(source / 2)) / u64::from(source),
    )
    .unwrap_or(u32::MAX)
}

fn ocr_matches_target(target: &InteractionTarget, block: &OcrBlock) -> bool {
    if !target.actionable
        && matches!(
            target.control_type.trim().to_ascii_lowercase().as_str(),
            "document" | "group" | "pane" | "window"
        )
    {
        return false;
    }
    let target_name = normalize_text(&target.name).to_ascii_lowercase();
    if target_name.is_empty() {
        return true;
    }
    let ocr_text = normalize_text(&block.text).to_ascii_lowercase();
    if ocr_text.is_empty() {
        return false;
    }
    if target_name.contains(&ocr_text) || ocr_text.contains(&target_name) {
        return true;
    }
    let target_terms = target_name
        .split(|character: char| !character.is_ascii_alphanumeric())
        .filter(|term| !term.is_empty())
        .collect::<HashSet<_>>();
    ocr_text
        .split(|character: char| !character.is_ascii_alphanumeric())
        .filter(|term| !term.is_empty())
        .any(|term| target_terms.contains(term))
}

pub fn relevant_targets(targets: &[InteractionTarget], limit: usize) -> Vec<&InteractionTarget> {
    let mut ranked = targets.iter().collect::<Vec<_>>();
    ranked.sort_by_key(|target| std::cmp::Reverse(target_priority(target)));
    ranked
        .into_iter()
        .filter(|target| {
            grounding_quality(target) != "low"
                && (target.rank_score > 0 || target.focused || target.selected == Some(true))
        })
        .take(limit)
        .collect()
}

fn select_targets(
    mut targets: Vec<InteractionTarget>,
    limit: usize,
    bounds: &Rect,
) -> Vec<InteractionTarget> {
    targets.sort_by_key(|target| std::cmp::Reverse(target_priority(target)));
    let mut selected = Vec::with_capacity(limit.min(targets.len()));
    let mut used = vec![false; targets.len()];
    let mut take_matching = |predicate: &dyn Fn(&InteractionTarget) -> bool, cap: usize| {
        let mut taken = 0;
        for (index, target) in targets.iter().enumerate() {
            if selected.len() >= limit || taken >= cap {
                break;
            }
            if !used[index] && predicate(target) {
                selected.push(target.clone());
                used[index] = true;
                taken += 1;
            }
        }
    };
    take_matching(&|target| target.desktop_shell, 32);
    take_matching(&|target| target.rank_score >= 10, 32);
    take_matching(
        &|target| {
            matches!(
                target.control_type.to_ascii_lowercase().as_str(),
                "edit" | "text box" | "document"
            )
        },
        16,
    );
    take_matching(
        &|target| target.actionable && grounding_quality(target) != "low",
        48,
    );

    let mut occupied = HashSet::new();
    for (index, target) in targets.iter().enumerate() {
        if selected.len() >= limit {
            break;
        }
        let center_x = i64::from(target.bounds.x) + i64::from(target.bounds.width / 2);
        let center_y = i64::from(target.bounds.y) + i64::from(target.bounds.height / 2);
        let column =
            ((center_x - i64::from(bounds.x)).max(0) * 4 / i64::from(bounds.width.max(1))).min(3);
        let row =
            ((center_y - i64::from(bounds.y)).max(0) * 4 / i64::from(bounds.height.max(1))).min(3);
        if !used[index] && occupied.insert((row, column)) {
            selected.push(target.clone());
            used[index] = true;
        }
    }
    for (index, target) in targets.into_iter().enumerate() {
        if selected.len() >= limit {
            break;
        }
        if !used[index] {
            selected.push(target);
        }
    }
    selected
}

pub fn merge_ocr_lines(blocks: &[OcrBlock]) -> Vec<OcrBlock> {
    let mut words = blocks
        .iter()
        .filter(|block| !block.text.trim().is_empty())
        .cloned()
        .collect::<Vec<_>>();
    words.sort_by_key(|block| (block.bounds.y, block.bounds.x));
    let mut lines: Vec<OcrBlock> = Vec::new();
    for word in words {
        let word_center = i64::from(word.bounds.y) + i64::from(word.bounds.height / 2);
        let matching = lines.iter_mut().rev().find(|line| {
            let line_center = i64::from(line.bounds.y) + i64::from(line.bounds.height / 2);
            let vertical_tolerance = i64::from(line.bounds.height.max(word.bounds.height).max(8));
            let line_right = i64::from(line.bounds.x) + i64::from(line.bounds.width);
            (word_center - line_center).abs() <= vertical_tolerance / 2
                && i64::from(word.bounds.x) >= line_right - 4
                && i64::from(word.bounds.x) - line_right <= 64
        });
        if let Some(line) = matching {
            if !line.text.is_empty() {
                line.text.push(' ');
            }
            line.text.push_str(word.text.trim());
            let right = (i64::from(line.bounds.x) + i64::from(line.bounds.width))
                .max(i64::from(word.bounds.x) + i64::from(word.bounds.width));
            let bottom = (i64::from(line.bounds.y) + i64::from(line.bounds.height))
                .max(i64::from(word.bounds.y) + i64::from(word.bounds.height));
            line.bounds.x = line.bounds.x.min(word.bounds.x);
            line.bounds.y = line.bounds.y.min(word.bounds.y);
            line.bounds.width = u32::try_from(right - i64::from(line.bounds.x)).unwrap_or(u32::MAX);
            line.bounds.height =
                u32::try_from(bottom - i64::from(line.bounds.y)).unwrap_or(u32::MAX);
            line.confidence = match (line.confidence, word.confidence) {
                (Some(left), Some(right)) => Some(left.min(right)),
                (left, right) => left.or(right),
            };
            if word.selected == Some(true) {
                line.selected = Some(true);
            }
            if line.variant.is_none() {
                line.variant = word.variant;
            }
        } else {
            lines.push(word);
        }
    }
    lines
}

fn valid_uia(element: &UiElement, window: &Rect, window_area: u64) -> bool {
    if element.password
        || element.offscreen
        || !element.enabled
        || element.bounds.width == 0
        || element.bounds.height == 0
        || !overlaps(&element.bounds, window)
    {
        return false;
    }
    let named = !element.name.trim().is_empty();
    let actionable = is_actionable(element);
    let too_large = area(&element.bounds).saturating_mul(100) > window_area.saturating_mul(40);
    actionable || named && !too_large
}

fn valid_ocr(block: &OcrBlock, window: &Rect) -> bool {
    !block.text.trim().is_empty()
        && block.bounds.width > 0
        && block.bounds.height > 0
        && overlaps(&block.bounds, window)
}

fn is_actionable(element: &UiElement) -> bool {
    let kind = element.control_type.to_ascii_lowercase();
    INTERACTIVE_TYPES.iter().any(|value| kind == *value)
}

pub fn grounding_quality(target: &InteractionTarget) -> &'static str {
    if !target.enabled || !target.actionable || target.name.trim().is_empty() {
        return "low";
    }
    let role = target.control_type.trim().to_ascii_lowercase();
    let structural = INTERACTIVE_TYPES.iter().any(|value| role == *value);
    if structural && matches!(target.source, TargetSource::Uia | TargetSource::UiaOcr) {
        return "high";
    }
    if structural
        && matches!(
            target.source,
            TargetSource::Visual | TargetSource::VisualOcr
        )
        && target
            .confidence
            .is_some_and(|confidence| confidence >= 0.72)
    {
        return "medium";
    }
    "low"
}

fn quality_priority(target: &InteractionTarget) -> u8 {
    match grounding_quality(target) {
        "high" => 2,
        "medium" => 1,
        _ => 0,
    }
}

fn target_priority(target: &InteractionTarget) -> (u16, u8, bool, bool, usize, u64) {
    (
        target.rank_score,
        quality_priority(target),
        matches!(
            target.source,
            TargetSource::UiaOcr | TargetSource::VisualOcr
        ),
        matches!(target.source, TargetSource::Uia | TargetSource::UiaOcr),
        usize::from(!target.name.is_empty()),
        area(&target.bounds),
    )
}

fn task_terms(task: &str) -> Vec<String> {
    const STOP: &[&str] = &[
        "the", "and", "for", "with", "this", "that", "can", "you", "please",
    ];
    task.to_ascii_lowercase()
        .split(|character: char| !character.is_ascii_alphanumeric())
        .filter(|word| word.len() > 2 && !STOP.contains(word))
        .map(str::to_owned)
        .collect()
}

fn target_rank(target: &InteractionTarget, terms: &[String]) -> (u16, Vec<String>) {
    let label = target.name.to_ascii_lowercase();
    let role = target.control_type.to_ascii_lowercase();
    let matches = terms
        .iter()
        .filter(|term| label.contains(term.as_str()) || role.contains(term.as_str()))
        .count()
        .min(3) as u16;
    let system = [
        "start",
        "search",
        "taskbar",
        "notification",
        "system tray",
        "overflow",
    ]
    .iter()
    .any(|term| label.contains(term) || role.contains(term));
    let mut score = matches * 10;
    let mut reasons = Vec::new();
    if matches > 0 {
        reasons.push(format!("task_match:{matches}"));
    }
    if system {
        score += 8;
        reasons.push("system_control".into());
    }
    if target.desktop_shell {
        score += 8;
        reasons.push("desktop_shell".into());
    }
    let quality = grounding_quality(target);
    if target.actionable && quality != "low" {
        score += 4;
        reasons.push("actionable".into());
    }
    if matches!(target.source, TargetSource::UiaOcr) {
        score += 3;
        reasons.push("uia_ocr".into());
    } else if matches!(target.source, TargetSource::Uia) {
        score += 2;
        reasons.push("uia".into());
    } else if matches!(
        target.source,
        TargetSource::Visual | TargetSource::VisualOcr
    ) {
        score += 2;
        reasons.push("visual_affordance".into());
    }
    match quality {
        "high" => {
            score += 8;
            reasons.push("grounding:high".into());
        }
        "medium" => {
            score += 3;
            reasons.push("grounding:medium".into());
        }
        _ => reasons.push("grounding:low".into()),
    }
    (score, reasons)
}

fn spatial_order(left: &InteractionTarget, right: &InteractionTarget) -> Ordering {
    left.bounds
        .y
        .cmp(&right.bounds.y)
        .then_with(|| left.bounds.x.cmp(&right.bounds.x))
        .then_with(|| left.control_type.cmp(&right.control_type))
        .then_with(|| left.name.cmp(&right.name))
}

fn overlap_score(left: &Rect, right: &Rect, iou_threshold: f32, containment: f32) -> bool {
    let intersection = intersection_area(left, right);
    if intersection == 0 {
        return false;
    }
    let union = area(left) + area(right) - intersection;
    let iou = intersection as f32 / union.max(1) as f32;
    let right_coverage = intersection as f32 / area(right).max(1) as f32;
    iou >= iou_threshold || right_coverage >= containment
}

fn intersection_area(left: &Rect, right: &Rect) -> u64 {
    let x1 = i64::from(left.x).max(i64::from(right.x));
    let y1 = i64::from(left.y).max(i64::from(right.y));
    let x2 = (i64::from(left.x) + i64::from(left.width))
        .min(i64::from(right.x) + i64::from(right.width));
    let y2 = (i64::from(left.y) + i64::from(left.height))
        .min(i64::from(right.y) + i64::from(right.height));
    u64::try_from((x2 - x1).max(0)).unwrap_or(0) * u64::try_from((y2 - y1).max(0)).unwrap_or(0)
}

fn overlaps(left: &Rect, right: &Rect) -> bool {
    intersection_area(left, right) > 0
}

fn area(rect: &Rect) -> u64 {
    u64::from(rect.width) * u64::from(rect.height)
}

fn normalize_text(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use uuid::Uuid;

    use crate::types::{CaptureScope, CaptureTarget, MonitorInfo, Observation, Screenshot};

    fn observation() -> Observation {
        Observation {
            version: Uuid::new_v4(),
            captured_at: Utc::now(),
            foreground_window: None,
            target: Some(CaptureTarget {
                scope: CaptureScope::ActiveWindow,
                id: "window".into(),
                title: "Fixture".into(),
                process_name: "fixture.exe".into(),
                bounds: Rect {
                    x: 0,
                    y: 0,
                    width: 800,
                    height: 600,
                },
            }),
            cursor: None,
            screenshots: Vec::new(),
            ocr: vec![OcrBlock {
                text: "Search".into(),
                bounds: Rect {
                    x: 110,
                    y: 110,
                    width: 80,
                    height: 20,
                },
                confidence: Some(0.9),
                selected: None,
                variant: None,
            }],
            ui_elements: vec![UiElement {
                name: String::new(),
                control_type: "edit".into(),
                automation_id: Some("address".into()),
                value: None,
                bounds: Rect {
                    x: 100,
                    y: 100,
                    width: 400,
                    height: 40,
                },
                enabled: true,
                password: false,
                offscreen: false,
                keyboard_focusable: true,
                clickable_point: Some((300, 120)),
                selected: None,
                focused: false,
                desktop_shell: false,
            }],
            targets: Vec::new(),
            timings_ms: Default::default(),
            warnings: Vec::new(),
        }
    }

    #[test]
    fn ocr_enriches_containing_uia_target() {
        let targets = build_targets(&observation(), 120, 0.1, 0.6, "search");
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].name, "Search");
        assert_eq!(targets[0].source, TargetSource::UiaOcr);
        assert_eq!(targets[0].click_point, Some((300, 120)));
    }

    #[test]
    fn unmatched_ocr_becomes_target() {
        let mut observation = observation();
        observation.ui_elements.clear();
        let targets = build_targets(&observation, 120, 0.1, 0.6, "search");
        assert_eq!(targets[0].source, TargetSource::Ocr);
        assert_eq!(targets[0].id, "1");
    }

    #[test]
    fn tiny_ocr_fragments_remain_readable_but_are_not_action_targets() {
        let mut observation = observation();
        observation.ui_elements.clear();
        observation.ocr = vec![OcrBlock {
            text: "Wst".into(),
            bounds: Rect {
                x: 100,
                y: 100,
                width: 20,
                height: 9,
            },
            confidence: None,
            selected: None,
            variant: None,
        }];
        let targets = build_targets(&observation, 120, 0.1, 0.6, "open result");
        assert_eq!(targets.len(), 1);
        assert!(!targets[0].actionable);
        assert_eq!(grounding_quality(&targets[0]), "low");
        assert!(relevant_targets(&targets, 16).is_empty());
    }

    #[test]
    fn complete_uia_link_outranks_a_nearby_ocr_fragment() {
        let mut observation = observation();
        observation.ui_elements = vec![UiElement {
            name: "Project Atlas Sep 11, 2025 $6.99".into(),
            control_type: "link".into(),
            automation_id: None,
            value: None,
            bounds: Rect {
                x: 100,
                y: 140,
                width: 500,
                height: 50,
            },
            enabled: true,
            password: false,
            offscreen: false,
            keyboard_focusable: true,
            clickable_point: Some((350, 165)),
            selected: None,
            focused: false,
            desktop_shell: false,
        }];
        observation.ocr = vec![OcrBlock {
            text: "Wst".into(),
            bounds: Rect {
                x: 700,
                y: 100,
                width: 20,
                height: 9,
            },
            confidence: None,
            selected: None,
            variant: None,
        }];
        let targets = build_targets(&observation, 120, 0.1, 0.6, "open Project Atlas");
        let relevant = relevant_targets(&targets, 16);
        assert_eq!(relevant[0].name, "Project Atlas Sep 11, 2025 $6.99");
        assert_eq!(grounding_quality(relevant[0]), "high");
    }

    #[test]
    fn unrelated_ocr_inside_named_container_remains_a_readable_anchor() {
        let mut observation = observation();
        observation.ui_elements[0].name = "Tab One".into();
        observation.ui_elements[0].control_type = "pane".into();
        observation.ocr[0].text = "H/scopy- H".into();
        let targets = build_targets(&observation, 120, 0.1, 0.6, "ring up h/s copy");
        assert_eq!(targets.len(), 2);
        let copy = targets
            .iter()
            .find(|target| target.name == "H/scopy- H")
            .expect("OCR control should not be swallowed by its container");
        assert!(!copy.actionable);
        assert_eq!(copy.source, TargetSource::Ocr);
        assert!(copy.rank_score >= 10);
    }

    #[test]
    fn conflicting_ocr_does_not_relabel_named_uia_control() {
        let mut observation = observation();
        observation.ui_elements[0].name = "Yes".into();
        observation.ui_elements[0].control_type = "button".into();
        observation.ocr[0].text = "S/scopy- C".into();
        let targets = build_targets(&observation, 120, 0.1, 0.6, "select copy");
        assert!(targets.iter().any(|target| target.name == "Yes"));
        assert!(targets.iter().any(|target| target.name == "S/scopy- C"));
    }

    #[test]
    fn selected_adaptive_ocr_state_reaches_the_model_target() {
        let mut observation = observation();
        observation.ui_elements.clear();
        observation.ocr[0].selected = Some(true);
        observation.ocr[0].variant = Some("highlight_inverted_3x".into());
        let targets = build_targets(&observation, 120, 0.1, 0.6, "search");
        assert_eq!(targets[0].selected, Some(true));
        assert_eq!(
            targets[0].grounding_variant.as_deref(),
            Some("highlight_inverted_3x")
        );
    }

    #[test]
    fn focusable_structural_pane_is_not_a_click_target() {
        let mut observation = observation();
        observation.ocr.clear();
        observation.ui_elements[0].name = "Carrier choices".into();
        observation.ui_elements[0].control_type = "pane".into();
        observation.ui_elements[0].keyboard_focusable = true;
        let targets = build_targets(&observation, 120, 0.1, 0.6, "choose UPS");
        assert_eq!(targets.len(), 1);
        assert!(!targets[0].actionable);
        assert_eq!(targets[0].click_point, None);
    }

    #[test]
    fn target_payload_is_bounded_and_numbered() {
        let mut observation = observation();
        observation.ui_elements.clear();
        observation.ocr = (0..200)
            .map(|index| OcrBlock {
                text: format!("item {index}"),
                bounds: Rect {
                    x: (index % 20) * 30,
                    y: (index / 20) * 30,
                    width: 20,
                    height: 20,
                },
                confidence: None,
                selected: None,
                variant: None,
            })
            .collect();
        let targets = build_targets(&observation, 120, 0.1, 0.6, "item");
        assert_eq!(targets.len(), 120);
        assert_eq!(targets.first().unwrap().id, "1");
        assert_eq!(targets.last().unwrap().id, "120");
    }

    #[test]
    fn task_match_wins_a_small_target_budget() {
        let mut observation = observation();
        observation.ui_elements.clear();
        observation.ocr = vec![
            OcrBlock {
                text: "Unrelated".into(),
                bounds: Rect {
                    x: 10,
                    y: 10,
                    width: 80,
                    height: 20,
                },
                confidence: None,
                selected: None,
                variant: None,
            },
            OcrBlock {
                text: "Discord".into(),
                bounds: Rect {
                    x: 100,
                    y: 10,
                    width: 80,
                    height: 20,
                },
                confidence: None,
                selected: None,
                variant: None,
            },
        ];
        let targets = build_targets(&observation, 1, 0.1, 0.6, "open Discord");
        assert_eq!(targets[0].name, "Discord");
        assert!(targets[0].rank_score >= 10);
        assert!(
            targets[0]
                .rank_reasons
                .iter()
                .any(|reason| reason.starts_with("task_match"))
        );
    }

    #[test]
    fn relevant_targets_keep_task_match_ahead_of_spatial_order() {
        let mut observation = observation();
        observation.ui_elements = vec![
            UiElement {
                name: "Unrelated".into(),
                bounds: Rect {
                    x: 10,
                    y: 10,
                    width: 80,
                    height: 20,
                },
                clickable_point: Some((50, 20)),
                ..observation.ui_elements[0].clone()
            },
            UiElement {
                name: "KNY Servers".into(),
                bounds: Rect {
                    x: 10,
                    y: 400,
                    width: 80,
                    height: 20,
                },
                clickable_point: Some((50, 410)),
                ..observation.ui_elements[0].clone()
            },
        ];
        observation.ocr = vec![
            OcrBlock {
                text: "Unrelated".into(),
                bounds: Rect {
                    x: 10,
                    y: 10,
                    width: 80,
                    height: 20,
                },
                confidence: None,
                selected: None,
                variant: None,
            },
            OcrBlock {
                text: "KNY Servers".into(),
                bounds: Rect {
                    x: 10,
                    y: 400,
                    width: 80,
                    height: 20,
                },
                confidence: None,
                selected: None,
                variant: None,
            },
        ];
        let targets = build_targets(&observation, 20, 0.1, 0.6, "open KNY server");
        let relevant = relevant_targets(&targets, 16);
        assert_eq!(relevant[0].name, "KNY Servers");
    }

    #[test]
    fn saturated_vertical_track_becomes_a_visual_slider_target() {
        let mut observation = observation();
        observation.ui_elements.clear();
        observation.ocr.clear();
        let mut image = image::RgbImage::from_pixel(200, 200, image::Rgb([30, 30, 30]));
        for y in 30..170 {
            for x in 96..104 {
                image.put_pixel(x, y, image::Rgb([20, 170, 240]));
            }
        }
        let mut bytes = Vec::new();
        image::DynamicImage::ImageRgb8(image)
            .write_to(
                &mut std::io::Cursor::new(&mut bytes),
                image::ImageFormat::Png,
            )
            .expect("encode fixture");
        observation.screenshots = vec![Screenshot {
            monitor: MonitorInfo {
                id: "fixture".into(),
                bounds: Rect {
                    x: 0,
                    y: 0,
                    width: 800,
                    height: 600,
                },
                scale_factor: 1.0,
                primary: true,
            },
            png_base64: base64::engine::general_purpose::STANDARD.encode(bytes),
            source_png_base64: None,
            model_width: 200,
            model_height: 200,
            captured_at: Utc::now(),
        }];
        let targets = build_targets(&observation, 20, 0.1, 0.6, "adjust the slider");
        let slider = targets
            .iter()
            .find(|target| target.source == TargetSource::Visual)
            .expect("visual slider target");
        assert_eq!(slider.control_type, "slider");
        assert!(slider.actionable);
        assert!(slider.click_point.is_some());
    }

    #[test]
    fn editor_after_large_uia_tree_survives_model_target_cap() {
        let mut observation = observation();
        observation.ocr.clear();
        observation.ui_elements = (0..620)
            .map(|index| UiElement {
                name: format!("Sidebar item {index}"),
                control_type: "button".into(),
                automation_id: None,
                value: None,
                bounds: Rect {
                    x: 10,
                    y: 10 + index % 550,
                    width: 120,
                    height: 20,
                },
                enabled: true,
                password: false,
                offscreen: false,
                keyboard_focusable: false,
                clickable_point: None,
                selected: None,
                focused: false,
                desktop_shell: false,
            })
            .collect();
        observation.ui_elements.push(UiElement {
            name: "Message @recipient".into(),
            control_type: "edit".into(),
            automation_id: Some("message-editor".into()),
            value: None,
            bounds: Rect {
                x: 200,
                y: 540,
                width: 500,
                height: 40,
            },
            enabled: true,
            password: false,
            offscreen: false,
            keyboard_focusable: true,
            clickable_point: None,
            selected: None,
            focused: false,
            desktop_shell: false,
        });
        let targets = build_targets(&observation, 120, 0.1, 0.6, "send recipient a message");
        assert_eq!(targets.len(), 120);
        assert!(targets.iter().any(|target| {
            target.name == "Message @recipient" && target.control_type == "edit"
        }));
    }

    #[test]
    fn desktop_shell_control_survives_a_small_model_target_cap() {
        let mut observation = observation();
        observation.ocr.clear();
        let fixture = observation.ui_elements[0].clone();
        observation.ui_elements = (0..200)
            .map(|index| UiElement {
                name: format!("Application control {index}"),
                bounds: Rect {
                    x: 10 + index % 700,
                    y: 10 + index % 500,
                    width: 60,
                    height: 20,
                },
                control_type: "button".into(),
                automation_id: None,
                clickable_point: None,
                keyboard_focusable: false,
                desktop_shell: false,
                ..fixture.clone()
            })
            .collect();
        observation.ui_elements.push(UiElement {
            name: "Fixture Player - 1 running window".into(),
            bounds: Rect {
                x: 380,
                y: 555,
                width: 48,
                height: 40,
            },
            control_type: "button".into(),
            automation_id: Some("taskbar-button".into()),
            clickable_point: Some((404, 575)),
            keyboard_focusable: true,
            desktop_shell: true,
            ..fixture
        });
        let targets = build_targets(&observation, 8, 0.1, 0.6, "open another application");
        assert!(targets.iter().any(|target| {
            target.desktop_shell && target.name == "Fixture Player - 1 running window"
        }));
    }

    #[test]
    fn adjacent_ocr_words_are_also_available_as_a_line() {
        let blocks = vec![
            OcrBlock {
                text: "Message".into(),
                bounds: Rect {
                    x: 10,
                    y: 20,
                    width: 60,
                    height: 20,
                },
                confidence: Some(0.9),
                selected: None,
                variant: None,
            },
            OcrBlock {
                text: "@recipient".into(),
                bounds: Rect {
                    x: 76,
                    y: 20,
                    width: 80,
                    height: 20,
                },
                confidence: Some(0.8),
                selected: None,
                variant: None,
            },
        ];
        let lines = merge_ocr_lines(&blocks);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].text, "Message @recipient");
        assert_eq!(lines[0].bounds.width, 146);
    }
}
