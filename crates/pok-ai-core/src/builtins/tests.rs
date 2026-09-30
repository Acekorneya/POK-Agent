use std::io::Cursor;

use base64::Engine;
use chrono::Utc;

use super::*;

#[test]
fn text_goes_only_into_controls_that_take_typing() {
    // A selected file in a Save dialog list is not its file name box:
    // typing there renames the file or jumps between items.
    for control in ["list item", "Tree Item", "button", "menu item", "check box"] {
        assert!(!accepts_typed_text(control), "{control}");
    }
    for control in ["edit", "combo box", "document", "pane", "custom"] {
        assert!(accepts_typed_text(control), "{control}");
    }
}

/// A desktop whose launch opens either the app window or the shell's
/// "cannot find" dialog, to test launch verification.
struct LaunchDesktop {
    windows: parking_lot::Mutex<Vec<crate::types::WindowInfo>>,
    opens: crate::types::WindowInfo,
}

#[async_trait::async_trait]
impl crate::platform::DesktopPlatform for LaunchDesktop {
    async fn capture_screens(&self) -> Result<Vec<crate::types::Screenshot>> {
        Ok(Vec::new())
    }
    async fn query_ocr(&self) -> Result<Vec<crate::types::OcrBlock>> {
        Ok(Vec::new())
    }
    async fn query_ui_tree(&self) -> Result<Vec<crate::types::UiElement>> {
        Ok(Vec::new())
    }
    async fn foreground_window(&self) -> Result<Option<crate::types::WindowInfo>> {
        Ok(self.windows.lock().last().cloned())
    }
    async fn list_windows(&self) -> Result<Vec<crate::types::WindowInfo>> {
        Ok(self.windows.lock().clone())
    }
    async fn launch_application(&self, _name: &str) -> Result<()> {
        self.windows.lock().push(self.opens.clone());
        Ok(())
    }
    async fn cursor_position(&self) -> Result<Option<(i32, i32)>> {
        Ok(None)
    }
    async fn simulate_input(&self, _action: &crate::types::InputAction) -> Result<()> {
        Ok(())
    }
}

fn launch_window(id: &str, title: &str, process: &str) -> crate::types::WindowInfo {
    crate::types::WindowInfo {
        id: id.into(),
        title: title.into(),
        process_name: process.into(),
        bounds: crate::types::Rect {
            x: 0,
            y: 0,
            width: 400,
            height: 300,
        },
        elevated: false,
        visible: true,
        minimized: false,
    }
}

#[tokio::test]
async fn open_application_reports_success_only_when_its_window_appears() {
    for (opens, launched) in [
        (
            launch_window("app", "Settings", "ApplicationFrameHost.exe"),
            true,
        ),
        // The shell's "Windows cannot find 'Settings'" dialog.
        (launch_window("dialog", "Settings", "explorer.exe"), false),
        // The console window left by the `cmd /C start` fallback.
        (launch_window("console", "Settings", "cmd.exe"), false),
    ] {
        let temp = tempfile::tempdir().unwrap();
        let platform = std::sync::Arc::new(LaunchDesktop {
            windows: parking_lot::Mutex::new(vec![launch_window("ide", "Editor", "code.exe")]),
            opens,
        });
        let context = crate::tool::ToolContext {
            session_id: Uuid::new_v4(),
            workspace: temp.path().to_path_buf(),
            data_dir: temp.path().to_path_buf(),
            artifact_dir: temp.path().join("artifacts"),
            policy: crate::policy::Policy::interactive(),
            approvals: std::sync::Arc::new(crate::policy::AllowApprovals),
            platform,
            memory: crate::memory::MemoryStore::open(temp.path().join("memory")).unwrap(),
            session_archive: crate::session_archive::SessionArchive::open(
                &temp.path().join("artifacts"),
            )
            .unwrap(),
            cancellation: tokio_util::sync::CancellationToken::new(),
            pause: std::sync::Arc::new(crate::pause::PauseController::default()),
            latest_observation: Default::default(),
            latest_observation_view: Default::default(),
            pending_visual_localization: Default::default(),
            task_hint: Default::default(),
            input_ledger: Default::default(),
            artifact_evidence: Default::default(),
            attached_paths: Default::default(),
            session_files: Default::default(),
            active_task: Default::default(),
            focused_control: Default::default(),
            command_timeout_seconds: 10,
            input_text_inter_key_pause_ms: 50,
            vision_max_edge: 640,
            prompt_token_target: 4_000,
            context_budget: std::sync::Arc::new(parking_lot::Mutex::new(
                crate::context::ContextBudget::new(
                    "test".into(),
                    "test".into(),
                    8_000,
                    80,
                    crate::context::ContextSource::FallbackUnknown,
                ),
            )),
            visual_history_limit: 1,
            uia_element_limit: 512,
            desktop_enrichment_timeout_ms: 2_000,
            desktop_deep_enrichment_timeout_ms: 10_000,
            model_target_limit: 16,
            fusion_iou_threshold: 0.1,
            ocr_containment_threshold: 0.6,
            annotate_targets: true,
            approval_cache: Default::default(),
            user_guidance_queue: Default::default(),
            questions: None,
            command_manager: std::sync::Arc::new(crate::commands::CommandManager::new(
                temp.path().join("artifacts"),
            )),
            current_tool_call_id: Default::default(),
        };
        let result = OpenApplicationTool
            .execute(json!({"name": "Settings"}), &context)
            .await
            .unwrap();
        assert_eq!(result["launched"] != json!(false), launched, "{result}");
        if !launched {
            assert!(result["unexpected_window"]["app"].is_string(), "{result}");
        }
    }
}
use crate::types::{
    CaptureScope, CaptureTarget, InteractionTarget, MonitorInfo, TargetSource, UiElement,
};

#[test]
fn question_schema_hides_internal_ids_and_has_no_small_batch_cap() {
    let schema = crate::brain::reference_free_schema(&AskUserQuestionTool.input_schema());
    let questions = &schema["properties"]["questions"];
    let question = &questions["items"];
    assert_eq!(questions["type"], "array");
    assert!(questions.get("maxItems").is_none());
    assert!(question["properties"].get("id").is_none());
    assert!(
        question["required"]
            .as_array()
            .is_some_and(|required| required.iter().any(|field| field == "question"))
    );
}

#[test]
fn question_preparation_accepts_large_batches_and_generates_unique_ids() {
    let items = (0..5)
        .map(|index| AskUserQuestionItem {
            id: (index == 1).then(|| "question_1".into()),
            header: String::new(),
            question: format!("Question {}?", index + 1),
            options: Vec::new(),
            multi_select: false,
        })
        .collect();
    let questions = prepare_user_questions(items).unwrap();
    assert_eq!(questions.len(), 5);
    assert_eq!(questions[0].id, "question_1");
    assert_eq!(questions[1].id, "question_1_2");
    assert_eq!(
        questions
            .iter()
            .map(|question| question.id.as_str())
            .collect::<HashSet<_>>()
            .len(),
        5
    );
}

#[test]
fn explicit_skill_requests_are_distinct_from_fact_memory() {
    assert!(explicit_skill_request(
        "Create a skill for playing music in whichever desktop app I name"
    ));
    assert!(!explicit_skill_request("Remember that I live in Seattle"));
}

fn screenshot() -> Screenshot {
    Screenshot {
        monitor: MonitorInfo {
            id: "window".into(),
            bounds: Rect {
                x: 2_500,
                y: 200,
                width: 2_000,
                height: 1_000,
            },
            scale_factor: 1.0,
            primary: false,
        },
        png_base64: String::new(),
        source_png_base64: None,
        model_width: 1_600,
        model_height: 800,
        captured_at: Utc::now(),
    }
}

fn scroll_observation(lines: &[(&str, i32)], png_marker: &str) -> Observation {
    let screenshot = Screenshot {
        png_base64: png_marker.into(),
        ..screenshot()
    };
    let bounds = screenshot.monitor.bounds.clone();
    Observation {
        version: Uuid::new_v4(),
        captured_at: Utc::now(),
        foreground_window: Some(WindowInfo {
            id: "window".into(),
            title: "Scrollable fixture".into(),
            process_name: "fixture.exe".into(),
            bounds: bounds.clone(),
            elevated: false,
            visible: true,
            minimized: false,
        }),
        target: Some(CaptureTarget {
            scope: CaptureScope::Window,
            id: "window".into(),
            title: "Scrollable fixture".into(),
            process_name: "fixture.exe".into(),
            bounds,
        }),
        cursor: Some((3_500, 700)),
        screenshots: vec![screenshot],
        ocr: lines
            .iter()
            .enumerate()
            .map(|(index, (text, y))| crate::types::OcrBlock {
                text: (*text).into(),
                bounds: Rect {
                    x: 2_700 + i32::try_from(index).unwrap_or(0) * 10,
                    y: *y,
                    width: 200,
                    height: 20,
                },
                confidence: Some(0.95),
                selected: None,
                variant: None,
            })
            .collect(),
        ui_elements: Vec::new(),
        targets: Vec::new(),
        timings_ms: Default::default(),
        warnings: Vec::new(),
    }
}

fn selection_target(name: &str, selected: bool, focused: bool) -> InteractionTarget {
    InteractionTarget {
        id: name.into(),
        name: name.into(),
        control_type: "radio button".into(),
        bounds: Rect {
            x: 20,
            y: 20,
            width: 100,
            height: 24,
        },
        source: TargetSource::Uia,
        confidence: None,
        enabled: true,
        actionable: true,
        click_point: Some((70, 32)),
        selected: Some(selected),
        focused,
        desktop_shell: false,
        grounding_variant: None,
        rank_score: 1,
        rank_reasons: vec!["fixture".into()],
    }
}

#[test]
fn a_target_that_left_the_screen_is_not_rebased() {
    let previous = selection_target("Save", false, false);
    let mut refreshed = scroll_observation(&[], "after");
    assert!(unique_rebased_target(&previous, &refreshed).is_none());
    refreshed.targets = vec![selection_target("Save", false, false)];
    assert!(unique_rebased_target(&previous, &refreshed).is_some());
    refreshed
        .targets
        .push(selection_target("Save", true, false));
    assert!(unique_rebased_target(&previous, &refreshed).is_none());
}

#[test]
fn selection_and_control_focus_are_verified_state_changes() {
    let mut before = scroll_observation(&[], "before");
    before.targets = vec![selection_target("FedEx Ground", false, false)];
    let mut after = before.clone();
    after.targets = vec![selection_target("FedEx Ground", true, true)];

    let change = state_change_value(&before, &after, "select FedEx Ground");
    assert_eq!(change["selected_changed"], json!(["FedEx Ground"]));
    assert_eq!(change["focused_control_changed"], true);
    assert!(state_change_has_effect(&change));
}

#[test]
fn desktop_registry_exposes_exact_clock_tool() {
    let mut registry = ToolRegistry::new();
    register_desktop_tools(&mut registry);
    assert!(
        registry
            .names()
            .iter()
            .any(|name| name == "get_current_time")
    );
    assert!(
        registry
            .names()
            .iter()
            .any(|name| name == "inspect_screen_region")
    );
}

#[test]
fn screen_region_maps_model_pixels_to_bounded_physical_coordinates() {
    let mapped = model_rect_to_physical(
        &Rect {
            x: 400,
            y: 200,
            width: 400,
            height: 200,
        },
        1_600,
        800,
        &Rect {
            x: 2_500,
            y: 200,
            width: 2_000,
            height: 1_000,
        },
    )
    .unwrap();
    assert_eq!(mapped.x, 3_000);
    assert_eq!(mapped.y, 450);
    assert_eq!(mapped.width, 500);
    assert_eq!(mapped.height, 250);
}

fn shell_target(id: &str, name: &str, x: i32) -> InteractionTarget {
    InteractionTarget {
        id: id.into(),
        name: name.into(),
        control_type: "button".into(),
        bounds: Rect {
            x,
            y: 1_020,
            width: 48,
            height: 48,
        },
        source: TargetSource::Uia,
        confidence: None,
        enabled: true,
        actionable: true,
        click_point: Some((x + 24, 1_044)),
        selected: None,
        focused: false,
        desktop_shell: true,
        grounding_variant: None,
        rank_score: 10,
        rank_reasons: vec!["fixture".into()],
    }
}

#[test]
fn taskbar_recovery_requires_one_exact_generic_identity_match() {
    let mut observation = scroll_observation(&[], "desktop");
    observation.target = Some(CaptureTarget {
        scope: CaptureScope::Monitor,
        id: "monitor-1".into(),
        title: "Monitor 1".into(),
        process_name: String::new(),
        bounds: Rect {
            x: 0,
            y: 0,
            width: 1_920,
            height: 1_080,
        },
    });
    observation.targets = vec![
        shell_target("target_1", "Fixture Player - 1 running window", 800),
        shell_target("target_2", "Unrelated Editor", 860),
    ];
    let window = WindowInfo {
        id: "42:HANDLE(0x2A)".into(),
        title: "Fixture Player".into(),
        process_name: "fixtureplayer.exe".into(),
        bounds: Rect {
            x: 100,
            y: 100,
            width: 800,
            height: 600,
        },
        elevated: false,
        visible: true,
        minimized: true,
    };
    assert_eq!(matching_shell_targets(&observation, &window).len(), 1);
    observation.targets.push(shell_target(
        "target_3",
        "Fixture Player - second desktop",
        920,
    ));
    assert_eq!(matching_shell_targets(&observation, &window).len(), 2);
}

#[test]
fn identical_task_plan_is_not_a_material_update() {
    let steps = vec![TaskItem {
        id: "network".into(),
        content: "Find the desktop network subnet".into(),
        status: TaskItemStatus::InProgress,
    }];
    let state = ActiveTaskState {
        status: TaskItemStatus::InProgress,
        current_step: "Find the desktop network subnet".into(),
        steps: steps.clone(),
        plan_updated: true,
        ..ActiveTaskState::default()
    };
    assert!(!task_plan_changed(
        &state,
        TaskItemStatus::InProgress,
        "Find the desktop network subnet",
        &steps,
    ));
    assert!(task_plan_changed(
        &state,
        TaskItemStatus::InProgress,
        "Scan for the device",
        &steps,
    ));
}

#[test]
fn desktop_registry_exposes_semantic_scroll_tools() {
    let mut registry = ToolRegistry::new();
    register_desktop_tools(&mut registry);
    let names = registry.names();
    assert!(names.iter().any(|name| name == "scroll_view"));
    assert!(names.iter().any(|name| name == "scroll_until_text"));
}

#[test]
fn only_allowlisted_transient_shell_overlays_are_auto_dismissed() {
    let window = |title: &str, process_name: &str| WindowInfo {
        id: "window".into(),
        title: title.into(),
        process_name: process_name.into(),
        bounds: Rect {
            x: 0,
            y: 0,
            width: 800,
            height: 600,
        },
        elevated: false,
        visible: true,
        minimized: false,
    };
    assert!(is_transient_shell_overlay(&window(
        "Quick settings",
        "ShellHost.exe"
    )));
    assert!(is_transient_shell_overlay(&window(
        "Notification Center",
        "ShellHost.exe"
    )));
    assert!(!is_transient_shell_overlay(&window(
        "User document",
        "notepad.exe"
    )));
}

#[test]
fn observation_cache_requires_exact_image_target_and_foreground() {
    let prior = scroll_observation(&[("message", 300)], "same-image");
    let capture = DesktopCapture {
        target: prior.target.clone().unwrap(),
        screenshots: prior.screenshots.clone(),
        timings_ms: Default::default(),
    };
    assert!(reusable_observation(
        &prior,
        &capture,
        prior.foreground_window.as_ref(),
        true,
        false,
    ));

    let mut changed = capture;
    changed.screenshots[0].png_base64 = "changed-image".into();
    assert!(!reusable_observation(
        &prior,
        &changed,
        prior.foreground_window.as_ref(),
        true,
        false,
    ));
}

#[test]
fn ordered_content_preserves_separator_position() {
    let mut observation = scroll_observation(
        &[("older message", 300), ("NEW", 400), ("newer message", 500)],
        "image",
    );
    observation.targets = build_targets(&observation, 20, 0.1, 0.6, "");
    let value = model_observation_value(&observation, false).unwrap();
    let texts = value["ordered_content"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|item| item["text"].as_str())
        .collect::<Vec<_>>();
    assert_eq!(texts, ["older message", "NEW", "newer message"]);
}

#[test]
fn primary_content_prefers_messages_over_unrelated_sidebar_items() {
    let mut observation = scroll_observation(&[], "image");
    let target = observation.target.as_ref().unwrap().bounds.clone();
    let element = |name: &str, control_type: &str, x: i32, y: i32| UiElement {
        name: name.into(),
        control_type: control_type.into(),
        automation_id: None,
        value: None,
        bounds: Rect {
            x: target.x + x,
            y: target.y + y,
            width: 500,
            height: 40,
        },
        enabled: true,
        password: false,
        offscreen: false,
        keyboard_focusable: false,
        clickable_point: None,
        selected: None,
        focused: false,
        desktop_shell: false,
    };
    observation.ui_elements = vec![
        element("Mr. Wick, first message, 9:08 PM", "Message", 250, 200),
        element("Mr. Wick, second message, 9:06 PM", "Message", 250, 260),
        element("DarkSPIDER", "List Item", 1_000, 200),
    ];
    let value = model_observation_value(&observation, false).unwrap();
    assert_eq!(value["primary_content_kind"], "messages");
    let content = value["primary_content"].as_array().unwrap();
    assert_eq!(content.len(), 2);
    assert!(content.iter().all(|item| {
        item["text"]
            .as_str()
            .is_some_and(|text| text.starts_with("Mr. Wick"))
    }));
}

#[test]
fn post_action_monitor_refresh_narrows_to_foreground_window() {
    let mut observation = scroll_observation(&[("message", 300)], "image");
    observation.target.as_mut().unwrap().scope = CaptureScope::Monitor;
    observation.target.as_mut().unwrap().id = "monitor-1".into();
    let request =
        refresh_request_after_input(&observation, observation.foreground_window.as_ref(), 1_600)
            .unwrap();
    assert_eq!(request.scope, CaptureScope::Window);
    assert_eq!(request.window_id.as_deref(), Some("window"));
}

#[test]
fn live_foreground_guard_detects_user_focus_change() {
    let current = WindowInfo {
        id: "user-window".into(),
        title: "User application".into(),
        process_name: "user.exe".into(),
        bounds: Rect {
            x: 0,
            y: 0,
            width: 800,
            height: 600,
        },
        elevated: false,
        visible: true,
        minimized: false,
    };
    assert!(foreground_matches("user-window", Some(&current)));
    assert!(!foreground_matches("agent-window", Some(&current)));
    assert!(!foreground_matches("agent-window", None));
}

#[test]
fn semantic_scroll_uses_bounded_wheel_notches() {
    assert_eq!(
        scroll_notches(ScrollDirection::Down, ScrollAmount::Page),
        (0, 8)
    );
    assert_eq!(
        scroll_notches(ScrollDirection::Up, ScrollAmount::Small),
        (0, -3)
    );
    assert_eq!(
        scroll_notches(ScrollDirection::Right, ScrollAmount::Page),
        (8, 0)
    );
    assert_eq!(
        scroll_notches(ScrollDirection::Left, ScrollAmount::Small),
        (-3, 0)
    );
}

#[test]
fn local_model_mouse_aliases_preserve_button_and_scroll_direction() {
    let make_args = |kind: &str| InputArgs {
        observation_id: None,
        kind: kind.into(),
        x: Some(800),
        y: Some(400),
        button: None,
        text: None,
        replace_existing: false,
        key: None,
        delta_x: None,
        delta_y: None,
    };
    assert!(matches!(
        input_action(
            &make_args("right_click"),
            &screenshot().monitor.bounds,
            &screenshot()
        )
        .unwrap(),
        InputAction::Click {
            button: MouseButton::Right,
            ..
        }
    ));
    assert!(matches!(
        input_action(
            &make_args("middle_click"),
            &screenshot().monitor.bounds,
            &screenshot()
        )
        .unwrap(),
        InputAction::Click {
            button: MouseButton::Middle,
            ..
        }
    ));
    assert!(matches!(
        input_action(
            &make_args("mouse_scroll_down"),
            &screenshot().monitor.bounds,
            &screenshot()
        )
        .unwrap(),
        InputAction::Scroll {
            delta_x: 0,
            delta_y: 3
        }
    ));
    assert!(matches!(
        input_action(
            &make_args("mouse_scroll_up"),
            &screenshot().monitor.bounds,
            &screenshot()
        )
        .unwrap(),
        InputAction::Scroll {
            delta_x: 0,
            delta_y: -3
        }
    ));
}

#[test]
fn screenshot_noise_does_not_claim_scroll_progress() {
    let lines = [
        ("Alpha", 300),
        ("Bravo", 350),
        ("Charlie", 400),
        ("Delta", 450),
        ("Echo", 500),
    ];
    let before = scroll_observation(&lines, "clock-at-9-27");
    let mut after = scroll_observation(&lines, "clock-at-9-28");
    after.cursor = Some((3_520, 720));
    assert!(!observation_view_changed(
        &before,
        &after,
        ScrollDirection::Down
    ));
}

#[test]
fn coherent_grounded_displacement_verifies_requested_scroll() {
    let before = scroll_observation(
        &[
            ("Alpha", 300),
            ("Bravo", 350),
            ("Charlie", 400),
            ("Delta", 450),
        ],
        "before",
    );
    let after = scroll_observation(
        &[
            ("Alpha", 200),
            ("Bravo", 250),
            ("Charlie", 300),
            ("Delta", 350),
        ],
        "after",
    );
    assert!(observation_view_changed(
        &before,
        &after,
        ScrollDirection::Down
    ));
    assert!(!observation_view_changed(
        &before,
        &after,
        ScrollDirection::Up
    ));
}

#[test]
fn substantial_text_replacement_verifies_virtualized_scroll() {
    let before = scroll_observation(
        &[
            ("Alpha", 300),
            ("Bravo", 350),
            ("Charlie", 400),
            ("Delta", 450),
            ("Echo", 500),
        ],
        "before",
    );
    let after = scroll_observation(
        &[
            ("Foxtrot", 300),
            ("Golf", 350),
            ("Hotel", 400),
            ("India", 450),
            ("Juliet", 500),
        ],
        "after",
    );
    assert!(observation_view_changed(
        &before,
        &after,
        ScrollDirection::Down
    ));
}

#[test]
fn keyboard_fallback_matches_direction_and_amount() {
    assert_eq!(
        scroll_fallback_key(ScrollDirection::Down, ScrollAmount::Page),
        "PageDown"
    );
    assert_eq!(
        scroll_fallback_key(ScrollDirection::Up, ScrollAmount::Small),
        "UpArrow"
    );
    assert_eq!(
        scroll_fallback_key(ScrollDirection::Right, ScrollAmount::Page),
        "RightArrow"
    );
}

#[tokio::test]
async fn open_application_is_idempotent_when_already_open() {
    // Regression: live traces showed notepad launched dozens of times
    // because open_application was re-invoked for an application that was
    // already running. The tool must refuse to launch a second instance
    // when a matching window is already present.
    let mut observation = scroll_observation(&[], "");
    observation.foreground_window = Some(crate::types::WindowInfo {
        id: "n1".into(),
        title: "Untitled - Notepad".into(),
        process_name: "notepad.exe".into(),
        bounds: crate::types::Rect {
            x: 0,
            y: 0,
            width: 800,
            height: 600,
        },
        elevated: false,
        visible: true,
        minimized: false,
    });
    let platform = crate::platform::MockDesktop::new(observation);
    let temp = tempfile::tempdir().unwrap();
    let context = crate::tool::ToolContext {
        session_id: Uuid::new_v4(),
        workspace: temp.path().to_path_buf(),
        data_dir: temp.path().to_path_buf(),
        artifact_dir: temp.path().join("artifacts"),
        policy: crate::policy::Policy::interactive(),
        approvals: std::sync::Arc::new(crate::policy::AllowApprovals),
        platform: platform.clone(),
        memory: crate::memory::MemoryStore::open(temp.path().join("memory")).unwrap(),
        session_archive: crate::session_archive::SessionArchive::open(
            &temp.path().join("artifacts"),
        )
        .unwrap(),
        cancellation: tokio_util::sync::CancellationToken::new(),
        pause: std::sync::Arc::new(crate::pause::PauseController::default()),
        latest_observation: Default::default(),
        latest_observation_view: Default::default(),
        pending_visual_localization: Default::default(),
        task_hint: Default::default(),
        input_ledger: Default::default(),
        artifact_evidence: Default::default(),
        attached_paths: Default::default(),
        session_files: Default::default(),
        active_task: Default::default(),
        focused_control: Default::default(),
        command_timeout_seconds: 10,
        input_text_inter_key_pause_ms: 50,
        vision_max_edge: 640,
        prompt_token_target: 4_000,
        context_budget: std::sync::Arc::new(parking_lot::Mutex::new(
            crate::context::ContextBudget::new(
                "test".into(),
                "test".into(),
                8_000,
                80,
                crate::context::ContextSource::FallbackUnknown,
            ),
        )),
        visual_history_limit: 1,
        uia_element_limit: 512,
        desktop_enrichment_timeout_ms: 2_000,
        desktop_deep_enrichment_timeout_ms: 10_000,
        model_target_limit: 16,
        fusion_iou_threshold: 0.1,
        ocr_containment_threshold: 0.6,
        annotate_targets: true,
        approval_cache: Default::default(),
        user_guidance_queue: Default::default(),
        questions: None,
        command_manager: std::sync::Arc::new(crate::commands::CommandManager::new(
            temp.path().join("artifacts"),
        )),
        current_tool_call_id: Default::default(),
    };
    let result = OpenApplicationTool
        .execute(json!({"name": "notepad"}), &context)
        .await
        .unwrap();
    assert_eq!(
        result.get("already_open").and_then(Value::as_bool),
        Some(true)
    );
    assert_eq!(result.get("launched").and_then(Value::as_bool), Some(false));
}

pub(crate) fn mock_input_context(
    platform: std::sync::Arc<crate::platform::MockDesktop>,
    temp: &tempfile::TempDir,
) -> crate::tool::ToolContext {
    crate::tool::ToolContext {
        session_id: Uuid::new_v4(),
        workspace: temp.path().to_path_buf(),
        data_dir: temp.path().to_path_buf(),
        artifact_dir: temp.path().join("artifacts"),
        policy: crate::policy::Policy::interactive(),
        approvals: std::sync::Arc::new(crate::policy::AllowApprovals),
        platform,
        memory: crate::memory::MemoryStore::open(temp.path().join("memory")).unwrap(),
        session_archive: crate::session_archive::SessionArchive::open(
            &temp.path().join("artifacts"),
        )
        .unwrap(),
        cancellation: tokio_util::sync::CancellationToken::new(),
        pause: std::sync::Arc::new(crate::pause::PauseController::default()),
        latest_observation: Default::default(),
        latest_observation_view: Default::default(),
        pending_visual_localization: Default::default(),
        task_hint: Default::default(),
        input_ledger: Default::default(),
        artifact_evidence: Default::default(),
        attached_paths: Default::default(),
        session_files: Default::default(),
        active_task: Default::default(),
        focused_control: Default::default(),
        command_timeout_seconds: 10,
        input_text_inter_key_pause_ms: 50,
        vision_max_edge: 640,
        prompt_token_target: 4_000,
        context_budget: std::sync::Arc::new(parking_lot::Mutex::new(
            crate::context::ContextBudget::new(
                "test".into(),
                "test".into(),
                8_000,
                80,
                crate::context::ContextSource::FallbackUnknown,
            ),
        )),
        visual_history_limit: 1,
        uia_element_limit: 100,
        desktop_enrichment_timeout_ms: 2_000,
        desktop_deep_enrichment_timeout_ms: 10_000,
        model_target_limit: 20,
        fusion_iou_threshold: 0.1,
        ocr_containment_threshold: 0.6,
        annotate_targets: false,
        approval_cache: Default::default(),
        user_guidance_queue: Default::default(),
        questions: None,
        command_manager: std::sync::Arc::new(crate::commands::CommandManager::new(
            temp.path().join("artifacts"),
        )),
        current_tool_call_id: Default::default(),
    }
}

fn pattern_fixture(source: TargetSource) -> Observation {
    let mut observation = scroll_observation(&[], "");
    let mut target = selection_target("Open", false, false);
    target.control_type = "button".into();
    target.source = source;
    observation.targets = vec![target];
    observation
}

#[tokio::test]
async fn clicking_a_voice_channel_needs_a_request_to_join() {
    let mut observation = scroll_observation(&[], "discord");
    let window = observation.target.as_ref().unwrap().bounds.clone();
    let mut channel = selection_target("Lounge (voice channel), 3 users", false, false);
    channel.control_type = "tree item".into();
    channel.bounds.x = window.x + 40;
    channel.bounds.y = window.y + 40;
    channel.click_point = Some((window.x + 90, window.y + 52));
    observation.targets = vec![channel];
    let platform = crate::platform::MockDesktop::new(observation.clone());
    let temp = tempfile::tempdir().unwrap();
    let context = mock_input_context(platform.clone(), &temp);
    context.active_task.lock().root_request = "who is in the voice channels".into();
    let click = InputAction::Click {
        x: window.x + 90,
        y: window.y + 52,
        button: MouseButton::Left,
    };
    let denied = execute_input(
        click.clone(),
        observation.clone(),
        &context,
        json!({"kind": "click_target", "target_id": "Lounge (voice channel), 3 users"}),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(denied.contains("voice or video call"), "{denied}");
    assert!(platform.actions().is_empty());
    // Asked to join: the click goes ahead.
    context.active_task.lock().root_request = "join the Lounge voice channel".into();
    let _ = execute_input(
        click,
        observation,
        &context,
        json!({"kind": "click_target", "target_id": "Lounge (voice channel), 3 users"}),
    )
    .await;
    assert!(!platform.actions().is_empty());
}

#[tokio::test]
async fn batch_type_steps_go_into_the_fields_they_name() {
    let mut observation = scroll_observation(&[], "form");
    let window = observation.target.as_ref().unwrap().bounds.clone();
    let (left, top) = (window.x + 40, window.y + 40);
    let mut zip = selection_target("Zip", false, true);
    zip.control_type = "edit".into();
    zip.id = "40".into();
    zip.bounds.x = left;
    zip.bounds.y = top;
    zip.click_point = Some((left + 50, top + 12));
    let mut length = zip.clone();
    length.id = "59".into();
    length.name = "Length".into();
    length.focused = false;
    length.bounds.y = top + 60;
    length.click_point = Some((left + 50, top + 72));
    observation.targets = vec![zip, length];
    let platform = crate::platform::MockDesktop::new(observation.clone());
    let temp = tempfile::tempdir().unwrap();
    let context = mock_input_context(platform.clone(), &temp);
    *context.latest_observation.lock() = Some(observation.clone());
    let outcome = ExecuteActionBatchTool
        .execute(
            json!({"observation_id": observation.version.to_string(), "steps": [
                {"kind": "type_text", "target_id": "40", "text": "10001", "replace_existing": true},
                {"kind": "key", "key": "Tab"},
                {"kind": "type_text", "target_id": "59", "text": "9", "replace_existing": true},
            ]}),
            &context,
        )
        .await;
    let actions = platform.actions();
    let length_click = actions
        .iter()
        .position(|action| matches!(action, InputAction::Click { y, .. } if *y == top + 72));
    let length_typed = actions
        .iter()
        .position(|action| matches!(action, InputAction::TypeText { text, .. } if text == "9"));
    // The second value is typed only after its own field was focused.
    assert!(
        matches!((length_click, length_typed), (Some(click), Some(typed)) if click < typed),
        "{outcome:?} {actions:?}"
    );
}

#[tokio::test]
async fn left_clicks_on_accessible_controls_use_patterns_not_the_mouse() {
    let observation = pattern_fixture(TargetSource::Uia);
    let platform = crate::platform::MockDesktop::new(observation.clone());
    platform.enable_pattern_actions();
    let temp = tempfile::tempdir().unwrap();
    let context = mock_input_context(platform.clone(), &temp);
    let click = InputAction::Click {
        x: 70,
        y: 32,
        button: MouseButton::Left,
    };
    let used = pattern_input(
        &context,
        &click,
        &observation,
        &json!({"target_id": "Open"}),
    )
    .await
    .unwrap();
    assert_eq!(used.as_deref(), Some("invoke"));
    let open = InputAction::DoubleClick {
        x: 70,
        y: 32,
        button: MouseButton::Left,
    };
    let used = pattern_input(&context, &open, &observation, &json!({"target_id": "Open"}))
        .await
        .unwrap();
    assert_eq!(used.as_deref(), Some("default_action"));
    let requests = platform.pattern_actions();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].window_id, "window");
    assert_eq!(requests[1].action, crate::platform::PatternAction::Open);
    assert!(platform.actions().is_empty(), "no mouse input");
}

#[tokio::test]
async fn pixel_targets_right_clicks_and_unsupported_platforms_fall_back_to_the_mouse() {
    let temp = tempfile::tempdir().unwrap();
    let click = InputAction::Click {
        x: 70,
        y: 32,
        button: MouseButton::Left,
    };
    let action = json!({"target_id": "Open"});
    // An OCR-only target has no accessibility element to act on.
    let ocr = pattern_fixture(TargetSource::Ocr);
    let platform = crate::platform::MockDesktop::new(ocr.clone());
    platform.enable_pattern_actions();
    let context = mock_input_context(platform.clone(), &temp);
    assert!(
        pattern_input(&context, &click, &ocr, &action)
            .await
            .unwrap()
            .is_none()
    );
    // A right click opens a context menu, which needs the real mouse.
    let uia = pattern_fixture(TargetSource::Uia);
    let right = InputAction::Click {
        x: 70,
        y: 32,
        button: MouseButton::Right,
    };
    assert!(
        pattern_input(&context, &right, &uia, &action)
            .await
            .unwrap()
            .is_none()
    );
    assert!(platform.pattern_actions().is_empty());
    // A platform without pattern support declines.
    let plain = crate::platform::MockDesktop::new(uia.clone());
    let context = mock_input_context(plain, &temp);
    assert!(
        pattern_input(&context, &click, &uia, &action)
            .await
            .unwrap()
            .is_none()
    );
}

#[test]
fn an_observation_handle_passed_as_view_id_selects_the_observation() {
    let observation = pattern_fixture(TargetSource::Uia);
    let platform = crate::platform::MockDesktop::new(observation.clone());
    let temp = tempfile::tempdir().unwrap();
    let context = mock_input_context(platform, &temp);
    *context.latest_observation.lock() = Some(observation.clone());
    let handle = observation_id_for(&observation);
    let resolved =
        current_observation_or_view(&context, Some(&handle), Some(&handle), "click_target")
            .unwrap();
    assert_eq!(resolved.version, observation.version);
    assert!(
        current_observation_or_view(&context, Some(&handle), Some("view_x"), "click_target")
            .is_err()
    );
}

#[tokio::test]
async fn physical_input_waits_while_the_user_is_active_then_resumes() {
    let observation = pattern_fixture(TargetSource::Ocr);
    let platform = crate::platform::MockDesktop::new(observation);
    // Active 200 ms ago, then 1 s ago (still under the resume threshold),
    // then idle long enough.
    platform.script_physical_input(vec![200, 1_000, 3_000]);
    let temp = tempfile::tempdir().unwrap();
    let context = mock_input_context(platform.clone(), &temp);
    let waited = wait_for_user_idle(&context).await.unwrap();
    assert!(waited >= 400, "waited {waited} ms");
    // An idle user is not waited on.
    platform.script_physical_input(vec![60_000]);
    assert!(wait_for_user_idle(&context).await.unwrap() < 100);
}

#[tokio::test]
async fn waiting_for_the_user_is_reported_then_cleared() {
    let observation = pattern_fixture(TargetSource::Ocr);
    let platform = crate::platform::MockDesktop::new(observation);
    platform.script_physical_input(vec![100, 3_000]);
    let temp = tempfile::tempdir().unwrap();
    let context = mock_input_context(platform, &temp);
    let reports = std::sync::Arc::new(parking_lot::Mutex::new(Vec::new()));
    let sink = reports.clone();
    context
        .pause
        .set_user_activity_listener(std::sync::Arc::new(move |waiting| {
            sink.lock().push(waiting);
        }));
    wait_for_user_idle(&context).await.unwrap();
    assert_eq!(*reports.lock(), vec![true, false]);
}

#[tokio::test]
async fn scrolling_tries_the_cursor_free_pattern_before_the_wheel() {
    let lines = [("Alpha", 300), ("Bravo", 350), ("Charlie", 400)];
    let mut observation = scroll_observation(&lines, "");
    let mut encoded = Cursor::new(Vec::new());
    image::DynamicImage::new_rgba8(16, 16)
        .write_to(&mut encoded, image::ImageFormat::Png)
        .unwrap();
    observation.screenshots[0].png_base64 =
        base64::engine::general_purpose::STANDARD.encode(encoded.into_inner());
    let platform = crate::platform::MockDesktop::new(observation.clone());
    platform.enable_pattern_actions();
    let temp = tempfile::tempdir().unwrap();
    let context = mock_input_context(platform.clone(), &temp);
    let result = verified_scroll(
        &context,
        observation,
        (3_500, 700),
        ScrollDirection::Down,
        ScrollAmount::Page,
    )
    .await
    .unwrap();
    let scrolls = platform.pattern_scrolls();
    assert_eq!(scrolls.len(), 1);
    assert_eq!((scrolls[0].vertical, scrolls[0].horizontal), (1, 0));
    assert!(scrolls[0].page);
    // The mock view never moves, so the wheel still gets its turn.
    assert_eq!(result.attempts[0]["method"], "ui_automation");
    assert_eq!(result.attempts[1]["method"], "wheel");
}

#[tokio::test]
async fn ineffective_wheel_uses_keyboard_before_declaring_no_effect() {
    let lines = [
        ("Alpha", 300),
        ("Bravo", 350),
        ("Charlie", 400),
        ("Delta", 450),
        ("Echo", 500),
    ];
    let mut observation = scroll_observation(&lines, "");
    let mut encoded = Cursor::new(Vec::new());
    image::DynamicImage::new_rgba8(16, 16)
        .write_to(&mut encoded, image::ImageFormat::Png)
        .unwrap();
    observation.screenshots[0].png_base64 =
        base64::engine::general_purpose::STANDARD.encode(encoded.into_inner());
    let platform = crate::platform::MockDesktop::new(observation.clone());
    let temp = tempfile::tempdir().unwrap();
    let context = crate::tool::ToolContext {
        session_id: Uuid::new_v4(),
        workspace: temp.path().to_path_buf(),
        data_dir: temp.path().to_path_buf(),
        artifact_dir: temp.path().join("artifacts"),
        policy: crate::policy::Policy::interactive(),
        approvals: std::sync::Arc::new(crate::policy::AllowApprovals),
        platform: platform.clone(),
        memory: crate::memory::MemoryStore::open(temp.path().join("memory")).unwrap(),
        session_archive: crate::session_archive::SessionArchive::open(
            &temp.path().join("artifacts"),
        )
        .unwrap(),
        cancellation: tokio_util::sync::CancellationToken::new(),
        pause: std::sync::Arc::new(crate::pause::PauseController::default()),
        latest_observation: Default::default(),
        latest_observation_view: Default::default(),
        pending_visual_localization: Default::default(),
        task_hint: Default::default(),
        input_ledger: Default::default(),
        artifact_evidence: Default::default(),
        attached_paths: Default::default(),
        session_files: Default::default(),
        active_task: Default::default(),
        focused_control: Default::default(),
        command_timeout_seconds: 10,
        input_text_inter_key_pause_ms: 50,
        vision_max_edge: 640,
        prompt_token_target: 4_000,
        context_budget: std::sync::Arc::new(parking_lot::Mutex::new(
            crate::context::ContextBudget::new(
                "test".into(),
                "test".into(),
                8_000,
                80,
                crate::context::ContextSource::FallbackUnknown,
            ),
        )),
        visual_history_limit: 1,
        uia_element_limit: 100,
        desktop_enrichment_timeout_ms: 2_000,
        desktop_deep_enrichment_timeout_ms: 10_000,
        model_target_limit: 20,
        fusion_iou_threshold: 0.1,
        ocr_containment_threshold: 0.6,
        annotate_targets: false,
        approval_cache: Default::default(),
        user_guidance_queue: Default::default(),
        questions: None,
        command_manager: std::sync::Arc::new(crate::commands::CommandManager::new(
            temp.path().join("artifacts"),
        )),
        current_tool_call_id: Default::default(),
    };

    let result = verified_scroll(
        &context,
        observation,
        (3_500, 700),
        ScrollDirection::Down,
        ScrollAmount::Page,
    )
    .await
    .unwrap();
    assert!(!result.viewport_changed);
    assert!(!result.edge_reached);
    assert_eq!(result.attempts[0]["method"], "wheel");
    assert_eq!(result.attempts[1]["method"], "keyboard");
    assert_eq!(result.attempts[2]["reason"], "not_detected");
    assert!(matches!(
        platform.actions().as_slice(),
        [
            InputAction::Move { .. },
            InputAction::Scroll {
                delta_x: 0,
                delta_y: 8
            },
            InputAction::Key { key }
        ] if key == "PageDown"
    ));
}

#[test]
fn visual_scrollbar_is_detected_without_blind_edge_clicking() {
    let mut image = image::GrayImage::from_pixel(200, 200, image::Luma([20]));
    for x in 190..=194 {
        for y in 0..=40 {
            image.put_pixel(x, y, image::Luma([180]));
        }
    }
    let mut encoded = Cursor::new(Vec::new());
    image::DynamicImage::ImageLuma8(image)
        .write_to(&mut encoded, image::ImageFormat::Png)
        .unwrap();
    let mut observation = scroll_observation(&[], "");
    observation.target.as_mut().unwrap().bounds = Rect {
        x: -1_000,
        y: 200,
        width: 200,
        height: 200,
    };
    observation.screenshots[0].source_png_base64 =
        Some(base64::engine::general_purpose::STANDARD.encode(encoded.into_inner()));

    let down = visual_scrollbar_fallback(&observation, ScrollDirection::Down);
    assert_eq!(down.click_point, Some((-810, 246)));
    assert!(!down.edge_reached);
    let up = visual_scrollbar_fallback(&observation, ScrollDirection::Up);
    assert!(up.click_point.is_none());
    assert!(up.edge_reached);

    observation.screenshots[0].source_png_base64 = None;
    assert!(
        visual_scrollbar_fallback(&observation, ScrollDirection::Down)
            .click_point
            .is_none()
    );
}

#[test]
fn legacy_raw_scroll_is_clamped_to_safe_wheel_notches() {
    let args = InputArgs {
        observation_id: None,
        kind: "scroll".into(),
        x: None,
        y: None,
        button: None,
        text: None,
        replace_existing: false,
        key: None,
        delta_x: Some(5_000),
        delta_y: Some(-5_000),
    };
    assert!(matches!(
        input_action(&args, &screenshot().monitor.bounds, &screenshot()).unwrap(),
        InputAction::Scroll {
            delta_x: 20,
            delta_y: -20
        }
    ));
}

#[test]
fn visible_text_search_uses_targets_uia_and_ocr_case_insensitively() {
    let mut observation = Observation {
        version: Uuid::new_v4(),
        captured_at: Utc::now(),
        foreground_window: None,
        target: None,
        cursor: None,
        screenshots: Vec::new(),
        ocr: vec![crate::types::OcrBlock {
            text: "Checkout Total".into(),
            bounds: Rect {
                x: 0,
                y: 0,
                width: 100,
                height: 20,
            },
            confidence: Some(0.9),
            selected: None,
            variant: None,
        }],
        ui_elements: Vec::new(),
        targets: Vec::new(),
        timings_ms: Default::default(),
        warnings: Vec::new(),
    };
    assert_eq!(
        visible_text_match(&observation, "checkout total").as_deref(),
        Some("Checkout Total")
    );
    let alternatives = scroll_query_alternatives("temperature| Temp |°F|Current|temp").unwrap();
    assert_eq!(alternatives, vec!["temperature", "temp", "°f", "current"]);
    let (matched_query, matched_text) =
        visible_text_match_any(&observation, &["total".into(), "missing".into()])
            .expect("alternative should match");
    assert_eq!(matched_query, "total");
    assert_eq!(matched_text, "Checkout Total");
    observation.ocr.clear();
    assert!(visible_text_match(&observation, "checkout").is_none());
}

#[test]
fn scroll_query_alternatives_are_bounded() {
    assert!(scroll_query_alternatives("|||").is_err());
    assert!(
        scroll_query_alternatives("a|b|c|d|e|f|g|h|i")
            .unwrap_err()
            .to_string()
            .contains("at most 8")
    );
}

#[test]
fn bare_email_does_not_become_navigation_host() {
    assert!(extract_hosts("Open Outlook and email frontdesk@example-shop.com").is_empty());
    assert_eq!(
        extract_hosts("Use frontdesk@example-shop.com and open outlook.live.com"),
        vec!["outlook.live.com"]
    );
}

#[test]
fn blocks_weather_url_when_active_browser_task_names_youtube() {
    let focus = FocusedControl {
        app: "chrome.exe".into(),
        label: "Address and search bar".into(),
        control_type: "Edit".into(),
    };
    let error = validate_browser_navigation(
        "wttr.in/Springfield+IL?format=j1",
        "Check youtube.com for the latest Gamers Nexus video",
        Some(&focus),
    )
    .expect_err("an unrelated hostname must be rejected");
    assert!(
        error
            .to_string()
            .contains("off-task browser navigation blocked")
    );
    assert!(error.to_string().contains("youtube.com"));
}

#[test]
fn permits_requested_browser_host_and_normal_search_text() {
    let focus = FocusedControl {
        app: "Google Chrome".into(),
        label: "Address and search bar".into(),
        control_type: "Edit control".into(),
    };
    let request = "Check youtube.com for the latest Gamers Nexus video";
    validate_browser_navigation(
        "https://www.youtube.com/@GamersNexus/videos",
        request,
        Some(&focus),
    )
    .expect("the requested hostname should be allowed");
    validate_browser_navigation("Gamers Nexus latest video", request, Some(&focus))
        .expect("ordinary search text should be allowed");
}

#[test]
fn maps_model_pixels_to_physical_window_coordinates() {
    let args = InputArgs {
        observation_id: Some(observation_id(Uuid::new_v4())),
        kind: "click".into(),
        x: Some(800),
        y: Some(400),
        button: Some(MouseButton::Left),
        text: None,
        replace_existing: false,
        key: None,
        delta_x: None,
        delta_y: None,
    };
    let target = Rect {
        x: 2_500,
        y: 200,
        width: 2_000,
        height: 1_000,
    };
    let action = input_action(&args, &target, &screenshot()).expect("valid click");
    assert!(matches!(
        action,
        InputAction::Click {
            x: 3_500,
            y: 700,
            ..
        }
    ));
}

#[test]
fn rejects_model_coordinates_outside_image() {
    let args = InputArgs {
        observation_id: Some(observation_id(Uuid::new_v4())),
        kind: "click".into(),
        x: Some(1_600),
        y: Some(20),
        button: None,
        text: None,
        replace_existing: false,
        key: None,
        delta_x: None,
        delta_y: None,
    };
    assert!(input_action(&args, &screenshot().monitor.bounds, &screenshot()).is_err());
}

#[test]
fn accepts_left_click_alias_and_reports_model_coordinates() {
    let args = InputArgs {
        observation_id: None,
        kind: "left_click".into(),
        x: Some(520),
        y: Some(715),
        button: None,
        text: None,
        replace_existing: false,
        key: None,
        delta_x: None,
        delta_y: None,
    };
    assert!(matches!(
        input_action(&args, &screenshot().monitor.bounds, &screenshot()).unwrap(),
        InputAction::Click { .. }
    ));
    let value = model_action_value(&args);
    assert_eq!(value["model_point"], json!([520, 715]));
    assert_eq!(value["mapping_succeeded"], true);
    assert!(value.get("physical_point").is_none());
}

#[test]
fn accepts_local_model_keyboard_shortcut_alias() {
    let args = InputArgs {
        observation_id: None,
        kind: "keyboard_shortcut".into(),
        x: None,
        y: None,
        button: None,
        text: None,
        replace_existing: false,
        key: Some("Windows+Shift+RightArrow".into()),
        delta_x: None,
        delta_y: None,
    };
    assert!(matches!(
        input_action(&args, &screenshot().monitor.bounds, &screenshot()).unwrap(),
        InputAction::Key { key } if key == "Windows+Shift+RightArrow"
    ));
    assert_eq!(model_action_value(&args)["kind"], "key");
}

#[test]
fn accepts_small_model_text_and_keyboard_aliases() {
    let text_args = InputArgs {
        observation_id: None,
        kind: "text".into(),
        x: None,
        y: None,
        button: None,
        text: Some("whole sentence".into()),
        replace_existing: true,
        key: None,
        delta_x: None,
        delta_y: None,
    };
    assert!(matches!(
        input_action(
            &text_args,
            &screenshot().monitor.bounds,
            &screenshot()
        )
        .unwrap(),
        InputAction::TypeText {
            text,
            replace_existing: true,
        } if text == "whole sentence"
    ));

    let keyboard_args = InputArgs {
        observation_id: None,
        kind: "keyboard".into(),
        x: None,
        y: None,
        button: None,
        text: None,
        replace_existing: false,
        key: Some("Tab".into()),
        delta_x: None,
        delta_y: None,
    };
    assert!(matches!(
        input_action(
            &keyboard_args,
            &screenshot().monitor.bounds,
            &screenshot()
        )
        .unwrap(),
        InputAction::Key { key } if key == "Tab"
    ));
}

#[test]
fn batch_keyboard_step_accepts_text_as_a_small_model_fallback() {
    assert!(is_batch_target_click("click_target"));
    assert!(is_batch_target_click("click"));
    assert!(is_batch_target_click("left_click"));
    assert!(!is_batch_target_click("right_click"));

    let canonical = ActionStep {
        kind: "key".into(),
        target_id: None,
        expected_label: None,
        text: Some("ignored fallback".into()),
        replace_existing: false,
        key: Some("Ctrl+L".into()),
    };
    assert_eq!(batch_step_key(&canonical), Some("Ctrl+L"));

    let fallback = ActionStep {
        kind: "key".into(),
        target_id: None,
        expected_label: None,
        text: Some("Enter".into()),
        replace_existing: false,
        key: None,
    };
    assert_eq!(batch_step_key(&fallback), Some("Enter"));

    let missing = ActionStep {
        kind: "key".into(),
        target_id: None,
        expected_label: None,
        text: None,
        replace_existing: false,
        key: None,
    };
    assert_eq!(batch_step_key(&missing), None);
}

#[test]
fn batch_detects_repeated_aliases_and_compacts_intermediate_visuals() {
    let first = ActionStep {
        kind: "keyboard".into(),
        target_id: None,
        expected_label: None,
        text: None,
        replace_existing: false,
        key: Some("Right".into()),
    };
    let second = ActionStep {
        kind: "key".into(),
        target_id: None,
        expected_label: None,
        text: None,
        replace_existing: false,
        key: Some("Right".into()),
    };
    assert!(batch_steps_repeat(&first, &second));

    let mut result = json!({
        "observation_id": "obs-latest",
        "executed": true,
        "verification": {"effect": false, "evidence": "no_stable_change"},
        "state_change": {"added_text": []},
        "screenshots": [
            {"view": "clean", "png_base64": "clean"},
            {"view": "annotated", "png_base64": "annotated"}
        ],
        "ordered_content": [{"text": "private screen contents"}],
    });
    compact_batch_step_result(&mut result);
    let encoded = result.to_string();
    assert!(encoded.contains("obs-latest"));
    assert!(encoded.contains("no_stable_change"));
    assert!(!encoded.contains("png_base64"));
    assert!(!encoded.contains("private screen contents"));
}

#[test]
fn click_schema_requires_the_models_intended_label() {
    let schema = crate::brain::reference_free_schema(&ClickTargetTool.input_schema());
    assert!(
        schema["required"]
            .as_array()
            .is_some_and(|required| required.iter().any(|field| field == "expected_label"))
    );
}

#[test]
fn desktop_registry_requires_localization_instead_of_raw_click_point() {
    let mut registry = ToolRegistry::new();
    register_desktop_tools(&mut registry);
    let names = registry.names();
    assert!(names.iter().any(|name| name == "locate_visual_target"));
    assert!(names.iter().any(|name| name == "click_localized"));
    assert!(!names.iter().any(|name| name == "click_point"));

    let locate = crate::brain::reference_free_schema(
        &registry.input_schema("locate_visual_target").unwrap(),
    );
    let required = locate["required"].as_array().unwrap();
    for field in ["label", "x", "y", "width", "height"] {
        assert!(required.iter().any(|item| item == field));
    }
    assert!(is_raw_click_kind("left_click"));
    assert!(is_raw_click_kind("RIGHT_CLICK"));
    assert!(!is_raw_click_kind("key"));
    assert!(!is_raw_click_kind("mouse_move"));
}

#[test]
fn visual_localization_uses_overlap_and_prefers_actionable_semantic_target() {
    let proposed = Rect {
        x: 400,
        y: 340,
        width: 90,
        height: 40,
    };
    let ocr = InteractionTarget {
        id: "1".into(),
        name: "Play".into(),
        control_type: "text".into(),
        bounds: Rect {
            x: 430,
            y: 350,
            width: 35,
            height: 15,
        },
        source: TargetSource::Ocr,
        confidence: Some(0.9),
        enabled: true,
        actionable: false,
        click_point: None,
        selected: None,
        focused: false,
        desktop_shell: false,
        grounding_variant: None,
        rank_score: 0,
        rank_reasons: Vec::new(),
    };
    let mut actionable = ocr.clone();
    actionable.id = "2".into();
    actionable.control_type = "button".into();
    actionable.bounds = proposed.clone();
    actionable.source = TargetSource::Uia;
    actionable.actionable = true;
    actionable.click_point = Some((445, 360));
    let mut observation = scroll_observation(&[], "image");
    observation.targets = vec![ocr, actionable];

    let matched =
        localization_candidate(&observation, &proposed, "Play", 0.1, 0.6).expect("matching target");
    assert_eq!(matched.id, "2");
    assert_eq!(matched.click_point, Some((445, 360)));
    assert!(visual_localization_matches(
        &proposed,
        &matched.bounds,
        0.1,
        0.6
    ));
}

#[test]
fn visual_localization_correlates_by_geometry_when_labels_differ() {
    let proposed = Rect {
        x: 1_712,
        y: 968,
        width: 635,
        height: 327,
    };
    let target =
        |id: &str, name: &str, x: i32, y: i32, width: u32, height: u32| InteractionTarget {
            id: id.into(),
            name: name.into(),
            control_type: "link".into(),
            bounds: Rect {
                x,
                y,
                width,
                height,
            },
            source: TargetSource::Uia,
            confidence: None,
            enabled: true,
            actionable: true,
            click_point: None,
            selected: None,
            focused: false,
            desktop_shell: false,
            grounding_variant: None,
            rank_score: 0,
            rank_reasons: Vec::new(),
        };
    let mut observation = scroll_observation(&[], "image");
    observation.targets = vec![
        target("116", "More actions", 2_269, 1_201, 42, 42),
        target("121", "Go to channel Queble", 1_713, 1_207, 38, 38),
        target(
            "122",
            "here's a pretty cool Godot plugin! 4 minutes, 17 seconds",
            1_761,
            1_207,
            260,
            24,
        ),
        target("125", "Queble", 1_761, 1_232, 46, 19),
    ];

    let matched = localization_candidate(
        &observation,
        &proposed,
        "here's a pretty cool Godot plugin video thumbnail",
        0.1,
        0.6,
    )
    .expect("closest deterministic control");
    assert_eq!(matched.id, "122");
}

#[test]
fn visual_localization_does_not_snap_between_equally_close_controls() {
    let proposed = Rect {
        x: 100,
        y: 100,
        width: 200,
        height: 100,
    };
    let target = |id: &str, x: i32| InteractionTarget {
        id: id.into(),
        name: format!("control {id}"),
        control_type: "button".into(),
        bounds: Rect {
            x,
            y: 135,
            width: 40,
            height: 30,
        },
        source: TargetSource::Uia,
        confidence: None,
        enabled: true,
        actionable: true,
        click_point: None,
        selected: None,
        focused: false,
        desktop_shell: false,
        grounding_variant: None,
        rank_score: 0,
        rank_reasons: Vec::new(),
    };
    let mut observation = scroll_observation(&[], "image");
    observation.targets = vec![target("left", 150), target("right", 210)];
    assert!(localization_candidate(&observation, &proposed, "unknown", 0.1, 0.6).is_none());
}

#[test]
fn visual_localization_allows_model_only_canvas_boxes() {
    let proposed = Rect {
        x: 50,
        y: 60,
        width: 80,
        height: 30,
    };
    let observation = scroll_observation(&[], "image");
    assert!(localization_candidate(&observation, &proposed, "game control", 0.1, 0.6).is_none());
    assert_eq!(
        (
            proposed.x + i32::try_from(proposed.width / 2).unwrap(),
            proposed.y + i32::try_from(proposed.height / 2).unwrap()
        ),
        (90, 75)
    );
}

#[test]
fn localization_diagnostic_overlay_draws_box_and_center_marker() {
    let mut encoded = Cursor::new(Vec::new());
    image::DynamicImage::new_rgba8(100, 80)
        .write_to(&mut encoded, image::ImageFormat::Png)
        .unwrap();
    let bounds = Rect {
        x: 200,
        y: 300,
        width: 100,
        height: 80,
    };
    let mut capture = DesktopCapture {
        target: CaptureTarget {
            scope: CaptureScope::Window,
            id: "window".into(),
            title: "window".into(),
            process_name: "test.exe".into(),
            bounds: bounds.clone(),
        },
        screenshots: vec![Screenshot {
            monitor: MonitorInfo {
                id: "monitor".into(),
                bounds: bounds.clone(),
                scale_factor: 1.0,
                primary: true,
            },
            png_base64: base64::engine::general_purpose::STANDARD.encode(encoded.into_inner()),
            source_png_base64: None,
            model_width: 100,
            model_height: 80,
            captured_at: Utc::now(),
        }],
        timings_ms: Default::default(),
    };
    let proposed = Rect {
        x: 220,
        y: 320,
        width: 40,
        height: 20,
    };
    annotate_localization_capture(&mut capture, &proposed, &bounds, (240, 330)).unwrap();
    let png = base64::engine::general_purpose::STANDARD
        .decode(&capture.screenshots[0].png_base64)
        .unwrap();
    let image = image::load_from_memory(&png).unwrap().into_rgba8();
    let marker = image::Rgba([255, 40, 80, 255]);
    assert_eq!(*image.get_pixel(20, 20), marker);
    assert_eq!(*image.get_pixel(40, 30), marker);
}

#[test]
fn semantic_click_matching_uses_complete_tokens_and_ignores_low_quality_fragments() {
    assert!(label_match_score("Keep Digging", "Keep Digging Sep 11, 2025 $6.99") > 0);
    assert!(
        label_match_score("Keep Digging", "Keep Digging Sep 11")
            > label_match_score("Keep Digging", "Just Keep Digging")
    );

    let mut observation = scroll_observation(&[], "image");
    let mut fragment = selection_target("Wst", false, false);
    fragment.id = "75".into();
    fragment.control_type = "text".into();
    fragment.source = TargetSource::UiaOcr;
    fragment.bounds.width = 20;
    fragment.bounds.height = 9;
    let mut result = selection_target("Keep Digging Sep 11, 2025 $6.99", false, false);
    result.id = "78".into();
    result.control_type = "link".into();
    result.source = TargetSource::UiaOcr;
    result.bounds.width = 879;
    result.bounds.height = 73;
    observation.targets = vec![fragment, result];

    let candidates = semantic_target_candidates(&observation, "Keep Digging");
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].id, "78");
    assert!(semantic_target_candidates(&observation, "Wst").is_empty());
    assert_eq!(
        crate::grounding::grounding_quality(&observation.targets[0]),
        "low"
    );
}

#[test]
fn typed_text_verification_detects_exact_truncated_and_unavailable_results() {
    assert!(matches!(
        verify_typed_text(Some(""), Some("complete text"), "complete text", true),
        TextVerification::Verified { observed_chars: 13 }
    ));
    assert!(matches!(
        verify_typed_text(
            Some(""),
            Some("this s"),
            "this was a complete sentence",
            true
        ),
        TextVerification::Mismatch { observed_chars: 6 }
    ));
    assert!(matches!(
        verify_typed_text(None, None, "not observable", false),
        TextVerification::Unavailable
    ));
}

#[test]
fn typed_text_verification_normalizes_windows_line_endings() {
    assert!(matches!(
        verify_typed_text(Some(""), Some("first\r\nsecond"), "first\nsecond", true),
        TextVerification::Verified { .. }
    ));
}

#[test]
fn typed_text_verification_detects_missing_uppercase_initials() {
    let requested =
        "Dear [Boss's Name],\n\nI am away Monday.\n\nThank you.\n\nSincerely,\n[Your Name]";
    let corrupted = "ear [oss's ame],\n\n am away onday.\n\nhank you.\n\nincerely,\n[our ame]";
    assert!(matches!(
        verify_typed_text(Some(""), Some(corrupted), requested, true),
        TextVerification::Mismatch { .. }
    ));
    let value = typing_verification_value(
        &verify_typed_text(Some(""), Some(corrupted), requested, true),
        requested.chars().count(),
        Some(corrupted),
        true,
        1,
    );
    assert_eq!(value["strategy"], "uia_value_or_batched_unicode_replace");
    assert_eq!(value["retry_count"], 1);
    assert_eq!(value["status"], "mismatch");
}

#[test]
fn compact_post_input_omits_screenshots_and_target_lists() {
    let observation = scroll_observation(&[("typed text", 300)], "image");
    let value = compact_post_input_value(&observation).unwrap();
    assert!(value.get("observation_id").is_some());
    assert!(value.get("target").is_some());
    assert!(value.get("screenshots").is_none());
    assert!(value.get("targets").is_none());
    assert!(value.get("ordered_content").is_none());
}

#[test]
fn verified_submission_suppresses_only_same_destination() {
    let ledger = InputLedger {
        pending: None,
        verified: vec![VerifiedSubmission {
            normalized_text: "hello there".into(),
            destination: "discord.exe|@recipient - discord".into(),
            destination_label: "@recipient - Discord".into(),
            evidence: "Sender, hello there, 4:39 PM".into(),
        }],
    };
    let duplicate = duplicate_submission_value(
        &ledger,
        "discord.exe|@recipient - discord",
        "hello   there",
        true,
    )
    .unwrap();
    assert_eq!(duplicate["status"], "suppressed_duplicate");
    assert!(
        duplicate_submission_value(
            &ledger,
            "discord.exe|@someone-else - discord",
            "hello there",
            true,
        )
        .is_none()
    );
}

#[test]
fn an_unsent_draft_can_be_replaced_but_not_typed_twice() {
    let mut ledger = InputLedger {
        pending: Some(PendingSubmission {
            text: "10001".into(),
            destination: "example_pos.exe|examplepos|edit@120,340".into(),
            destination_label: "Example POS".into(),
            submitted: false,
        }),
        verified: Vec::new(),
    };
    let field = "example_pos.exe|examplepos|edit@120,340";
    // Retyping the same value over it (replace) is safe to retry.
    assert!(duplicate_submission_value(&ledger, field, "10001", true).is_none());
    // Appending it again would duplicate the text.
    assert_eq!(
        duplicate_submission_value(&ledger, field, "10001", false).unwrap()["status"],
        "draft_already_typed"
    );
    // Another field of the same form is another destination.
    assert!(
        duplicate_submission_value(
            &ledger,
            "example_pos.exe|examplepos|edit@120,380",
            "10001",
            false
        )
        .is_none()
    );
    // Once sent, even a replace waits for verification.
    ledger.pending.as_mut().unwrap().submitted = true;
    assert_eq!(
        duplicate_submission_value(&ledger, field, "10001", true).unwrap()["status"],
        "verification_required"
    );
}

#[test]
fn typed_text_destinations_tell_form_fields_apart() {
    let mut observation = scroll_observation(&[], "form");
    let mut zip = selection_target("", false, true);
    zip.control_type = "edit".into();
    let mut city = zip.clone();
    city.focused = false;
    city.bounds.y += 40;
    observation.targets = vec![zip.clone(), city.clone()];
    let (zip_field, _) = field_destination(&observation, Some(&zip));
    let (city_field, _) = field_destination(&observation, Some(&city));
    assert_ne!(zip_field, city_field);
    // Without a target id, the focused field is the destination.
    assert_eq!(field_destination(&observation, None).0, zip_field);
    // A labeled field keeps its identity when it grows or moves.
    let mut message = zip.clone();
    message.name = "Message #general".into();
    let (before, _) = field_destination(&observation, Some(&message));
    message.bounds.y -= 60;
    assert_eq!(field_destination(&observation, Some(&message)).0, before);
}

#[test]
fn submission_verification_requires_non_editable_evidence() {
    let mut observation = Observation {
        version: Uuid::new_v4(),
        captured_at: Utc::now(),
        foreground_window: None,
        target: None,
        cursor: None,
        screenshots: Vec::new(),
        ocr: Vec::new(),
        ui_elements: vec![crate::types::UiElement {
            name: "hello there".into(),
            control_type: "edit".into(),
            automation_id: None,
            value: None,
            bounds: Rect {
                x: 0,
                y: 0,
                width: 100,
                height: 20,
            },
            enabled: true,
            password: false,
            offscreen: false,
            keyboard_focusable: true,
            clickable_point: None,
            selected: None,
            focused: false,
            desktop_shell: false,
        }],
        targets: Vec::new(),
        timings_ms: Default::default(),
        warnings: Vec::new(),
    };
    observation.ocr.push(crate::types::OcrBlock {
        text: "hello there".into(),
        bounds: Rect {
            x: 5,
            y: 2,
            width: 80,
            height: 16,
        },
        confidence: Some(0.9),
        selected: None,
        variant: None,
    });
    assert!(submission_evidence(&observation, "hello there").is_none());
    observation.ui_elements.push(crate::types::UiElement {
        name: "Sender, hello there, 4:39 PM".into(),
        control_type: "message".into(),
        automation_id: None,
        value: None,
        bounds: Rect {
            x: 0,
            y: 30,
            width: 200,
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
    });
    assert!(submission_evidence(&observation, "hello there").is_some());
}

#[test]
fn numbered_overlay_is_a_valid_png() {
    let mut source = Cursor::new(Vec::new());
    image::DynamicImage::new_rgba8(100, 50)
        .write_to(&mut source, image::ImageFormat::Png)
        .unwrap();
    let mut screenshot = screenshot();
    screenshot.monitor.bounds = Rect {
        x: 0,
        y: 0,
        width: 100,
        height: 50,
    };
    screenshot.model_width = 100;
    screenshot.model_height = 50;
    screenshot.png_base64 = base64::engine::general_purpose::STANDARD.encode(source.into_inner());
    let observation = Observation {
        version: Uuid::new_v4(),
        captured_at: Utc::now(),
        foreground_window: None,
        target: Some(CaptureTarget {
            scope: CaptureScope::ActiveWindow,
            id: "fixture".into(),
            title: "Fixture".into(),
            process_name: "fixture.exe".into(),
            bounds: screenshot.monitor.bounds.clone(),
        }),
        cursor: None,
        screenshots: vec![screenshot.clone()],
        ocr: Vec::new(),
        ui_elements: Vec::new(),
        targets: vec![InteractionTarget {
            id: "12".into(),
            name: "Open".into(),
            control_type: "button".into(),
            bounds: Rect {
                x: 10,
                y: 10,
                width: 40,
                height: 20,
            },
            source: TargetSource::UiaOcr,
            confidence: Some(0.9),
            enabled: true,
            actionable: true,
            click_point: Some((30, 20)),
            selected: None,
            focused: false,
            desktop_shell: false,
            grounding_variant: None,
            rank_score: 10,
            rank_reasons: vec!["fixture".into()],
        }],
        timings_ms: Default::default(),
        warnings: Vec::new(),
    };
    let encoded = annotate_screenshot(&observation, &screenshot, &["12".into()]).unwrap();
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .unwrap();
    assert_eq!(image::load_from_memory(&bytes).unwrap().width(), 100);
    let value = model_observation_value(&observation, true).unwrap();
    assert_eq!(value["annotation_mode"], "action_registry");
    assert_eq!(value["annotated_target_ids"], json!(["12"]));
    assert_eq!(value["relevant_targets"][0]["label"], "Open");
    assert_eq!(value["action_targets"][0]["label"], "Open");
    assert_eq!(value["screenshots"][0]["view"], "clean");
    assert_eq!(value["screenshots"][1]["view"], "annotated");
}

#[test]
fn unique_task_matching_target_is_inferred_for_nested_scroll() {
    let mut observation = scroll_observation(&[], "image");
    observation.targets = vec![
        InteractionTarget {
            id: "56".into(),
            name: "KNY Servers".into(),
            control_type: "tree item".into(),
            bounds: Rect {
                x: 10,
                y: 200,
                width: 30,
                height: 30,
            },
            source: TargetSource::Uia,
            confidence: None,
            enabled: true,
            actionable: true,
            click_point: Some((25, 215)),
            selected: Some(false),
            focused: false,
            desktop_shell: false,
            grounding_variant: None,
            rank_score: 14,
            rank_reasons: vec!["task_match:1".into()],
        },
        InteractionTarget {
            id: "80".into(),
            name: "Messages".into(),
            control_type: "list".into(),
            bounds: Rect {
                x: 200,
                y: 100,
                width: 500,
                height: 500,
            },
            source: TargetSource::Uia,
            confidence: None,
            enabled: true,
            actionable: false,
            click_point: None,
            selected: None,
            focused: false,
            desktop_shell: false,
            grounding_variant: None,
            rank_score: 2,
            rank_reasons: vec!["uia".into()],
        },
    ];
    let inferred = infer_scroll_target(&observation).expect("unique relevant target");
    assert_eq!(inferred.0.normalized(), "56");
    assert!(inferred.1.contains("KNY Servers"));
}

#[test]
fn desktop_windows_are_task_ranked_compact_and_monitor_mapped() {
    let monitors = vec![
        MonitorInfo {
            id: "left".into(),
            bounds: Rect {
                x: -1920,
                y: 0,
                width: 1920,
                height: 1080,
            },
            scale_factor: 1.0,
            primary: false,
        },
        MonitorInfo {
            id: "primary".into(),
            bounds: Rect {
                x: 0,
                y: 0,
                width: 1920,
                height: 1080,
            },
            scale_factor: 1.0,
            primary: true,
        },
    ];
    let mut windows = (0..50)
        .map(|index| WindowInfo {
            id: format!("window-{index}"),
            title: format!("Unrelated {index}"),
            process_name: "other.exe".into(),
            bounds: Rect {
                x: 100,
                y: 100,
                width: 800,
                height: 600,
            },
            elevated: false,
            visible: true,
            minimized: false,
        })
        .collect::<Vec<_>>();
    windows.push(WindowInfo {
        id: "discord".into(),
        title: "Discord".into(),
        process_name: "Discord.exe".into(),
        bounds: Rect {
            x: -1800,
            y: 20,
            width: 1000,
            height: 900,
        },
        elevated: false,
        visible: true,
        minimized: false,
    });
    let summaries = compact_windows(windows, &monitors, "open Discord", 40);
    assert_eq!(summaries.len(), 40);
    assert_eq!(summaries[0]["id"], "discord");
    assert_eq!(summaries[0]["monitor_id"], "left");
    assert!(serde_json::to_string(&summaries).unwrap().len() < 8_000);
}

// Tests that lived at the top level of builtins.rs.
#[test]
fn supplied_window_id_implies_window_capture_and_must_be_exact() {
    assert_eq!(
        normalized_capture_scope(
            CaptureScope::ActiveWindow,
            Some("88300:HANDLE(0x40ea0)"),
            None
        )
        .unwrap(),
        CaptureScope::Window
    );
    assert!(validate_full_window_id("88300:HANDLE(0x40ea0)").is_ok());
    assert!(
        validate_full_window_id("88300")
            .unwrap_err()
            .to_string()
            .contains("complete PID:HANDLE")
    );
}

#[test]
fn plan_shorthand_becomes_a_structured_in_progress_plan() {
    let steps = parse_plan_shorthand("- Open Chrome\n- Search for the winner\n- Report the source");
    assert_eq!(steps.len(), 3);
    assert_eq!(steps[0].status, TaskItemStatus::InProgress);
    assert_eq!(steps[1].status, TaskItemStatus::Pending);
    assert_eq!(steps[0].content, "Open Chrome");
}

#[test]
fn json_encoded_plan_string_preserves_structured_steps_and_current_step() {
    let mut update: UpdateTaskPlanArgs = serde_json::from_value(json!({
        "current_step": "Read the current state",
        "plan": "[{\"id\":\"1\",\"content\":\"Read the current state\",\"status\":\"in_progress\"},{\"id\":\"2\",\"content\":\"Answer the user\",\"status\":\"pending\"}]"
    }))
    .unwrap();

    hydrate_plan_update(&mut update, &ActiveTaskState::default());

    assert_eq!(update.steps.len(), 2);
    assert_eq!(update.steps[0].content, "Read the current state");
    assert_eq!(
        update.current_step.as_deref(),
        Some("Read the current state")
    );
}

#[test]
fn task_steps_accept_small_model_description_and_numeric_id_aliases() {
    let step: TaskItem = serde_json::from_value(json!({
        "id": 1,
        "description": "Open the browser",
        "status": "completed"
    }))
    .unwrap();
    assert_eq!(step.id, "1");
    assert_eq!(step.content, "Open the browser");
    assert_eq!(step.status, TaskItemStatus::Completed);
}

#[test]
fn completed_status_alone_finishes_an_existing_plan() {
    let existing = ActiveTaskState {
        steps: parse_plan_shorthand("- Open the site\n- Read the result\n- Answer the user"),
        ..ActiveTaskState::default()
    };
    let mut update: UpdateTaskPlanArgs =
        serde_json::from_value(json!({"status": "completed"})).unwrap();

    hydrate_plan_update(&mut update, &existing);

    assert_eq!(update.steps.len(), 3);
    assert!(
        update
            .steps
            .iter()
            .all(|step| step.status == TaskItemStatus::Completed)
    );
    assert_eq!(update.current_step.as_deref(), Some("Answer the user"));
}

#[test]
fn structured_plan_infers_missing_current_step() {
    let steps = vec![
        TaskItem {
            id: "1".into(),
            content: "Open browser".into(),
            status: TaskItemStatus::Completed,
        },
        TaskItem {
            id: "2".into(),
            content: "Read forecast".into(),
            status: TaskItemStatus::InProgress,
        },
    ];
    assert_eq!(
        resolved_current_step(TaskItemStatus::InProgress, None, &steps),
        "Read forecast"
    );
    assert_eq!(
        resolved_plan_status(None, &steps),
        TaskItemStatus::InProgress
    );
    let completed = steps
        .iter()
        .cloned()
        .map(|mut step| {
            step.status = TaskItemStatus::Completed;
            step
        })
        .collect::<Vec<_>>();
    assert_eq!(
        resolved_current_step(TaskItemStatus::Completed, None, &completed),
        "Read forecast"
    );
    assert_eq!(
        resolved_plan_status(None, &completed),
        TaskItemStatus::Completed
    );
}

#[test]
fn browser_navigation_accepts_common_url_alias() {
    let args: BrowserNavigateArgs = serde_json::from_value(json!({
        "window_id": "88300:HANDLE(0x40ea0)",
        "url": "https://weather.gov"
    }))
    .unwrap();
    assert_eq!(args.query_or_url, "https://weather.gov");
}

#[test]
fn navigation_readiness_rejects_new_url_with_stale_page_identity() {
    let before = BrowserPageIdentity {
        title: "previous ticker".into(),
        url: Some("finance.example/previous".into()),
        content: "previous table".into(),
        ..BrowserPageIdentity::default()
    };
    let stale = BrowserPageIdentity {
        title: "previous ticker".into(),
        url: Some("https://finance.example/current".into()),
        content: "previous table".into(),
        ..BrowserPageIdentity::default()
    };
    let (ready, error, url_match, title_changed, content_changed) = navigation_sample_readiness(
        &before,
        &stale,
        "https://finance.example/current",
        true,
        None,
        1,
    );
    assert!(url_match);
    assert!(!ready);
    assert!(!error);
    assert!(!title_changed);
    assert!(!content_changed);

    let settled = BrowserPageIdentity {
        title: "current ticker".into(),
        url: stale.url.clone(),
        content: "current table".into(),
        ..BrowserPageIdentity::default()
    };
    assert!(
        navigation_sample_readiness(
            &before,
            &settled,
            "https://finance.example/current",
            true,
            None,
            2,
        )
        .0
    );
}

#[test]
fn navigation_readiness_reports_loading_and_error_pages() {
    let before = BrowserPageIdentity::default();
    let loading = BrowserPageIdentity {
        title: "loading".into(),
        url: Some("example.test/report".into()),
        content: "please wait".into(),
        ..BrowserPageIdentity::default()
    };
    assert!(
        !navigation_sample_readiness(
            &before,
            &loading,
            "https://example.test/report",
            true,
            None,
            2,
        )
        .0
    );
    let error = BrowserPageIdentity {
        title: "site can't be reached".into(),
        ..loading
    };
    let state = navigation_sample_readiness(
        &before,
        &error,
        "https://example.test/report",
        true,
        None,
        2,
    );
    assert!(state.1);
}

#[test]
fn repair_and_recovery_states_are_not_clean_verification() {
    assert!(
        verification_issue_text("We found a problem with some content and can try to repair it.")
            .is_some()
    );
    assert!(verification_issue_title("Report.xlsx - Repaired - Excel").is_some());
    assert!(verification_issue_text("Report.xlsx - Excel").is_none());
    assert!(verification_issue_text("Auto repair shop inventory").is_none());
}

#[test]
fn a_copied_grid_range_reads_back_as_rows() {
    let rows =
        super::capture::clipboard_rows("Date\tFirst Name\tSales\n2024-01-02\tAda\t120\n").unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0], ["Date", "First Name", "Sales"]);
    assert_eq!(rows[1][2], "120");
    assert!(super::capture::clipboard_rows("plain text, no cells").is_none());
}

#[tokio::test]
async fn a_command_ending_in_an_ellipsis_waits_for_its_dialog() {
    let mut observation = scroll_observation(&[], "calc");
    let window = observation.target.as_ref().unwrap().bounds.clone();
    let mut command = selection_target("Paragraph...", false, false);
    command.control_type = "menu item".into();
    command.click_point = Some((window.x + 60, window.y + 40));
    observation.targets = vec![command];
    let main = observation
        .foreground_window
        .clone()
        .expect("foreground window");
    let platform = crate::platform::MockDesktop::new(observation.clone());
    let mut dialog = main.clone();
    dialog.id = "dialog".into();
    dialog.title = "Paragraph".into();
    dialog.bounds.width = 300;
    dialog.bounds.height = 200;
    platform.add_windows(vec![dialog]);
    let temp = tempfile::tempdir().unwrap();
    let context = mock_input_context(platform.clone(), &temp);
    // The application opens the dialog a moment after the click, later than
    // the ordinary settle checks look.
    let opener = platform.clone();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(1_100)).await;
        opener.switch_foreground("dialog");
    });
    execute_input(
        InputAction::Click {
            x: window.x + 60,
            y: window.y + 40,
            button: MouseButton::Left,
        },
        observation,
        &context,
        json!({"kind": "click_target", "label": "Paragraph...", "expected_label": "Paragraph..."}),
    )
    .await
    .unwrap();
    let latest = context.latest_observation.lock().clone().unwrap();
    assert_eq!(latest.foreground_window.unwrap().id, "dialog");
}

#[tokio::test]
async fn replacing_text_never_selects_everything_in_a_control_without_readable_text() {
    // A spreadsheet grid does not expose its text: Ctrl+A there selects the
    // whole sheet, and Delete would erase every cell.
    let observation = scroll_observation(&[], "calc");
    let platform = crate::platform::MockDesktop::new(observation.clone());
    let temp = tempfile::tempdir().unwrap();
    let context = mock_input_context(platform.clone(), &temp);
    let _ = execute_input(
        InputAction::TypeText {
            text: "=TEXT(C1;\"0.00\")".into(),
            replace_existing: true,
        },
        observation,
        &context,
        json!({"kind": "type_text"}),
    )
    .await;
    let actions = platform.actions();
    assert!(
        !actions
            .iter()
            .any(|action| matches!(action, InputAction::Key { key } if key.eq_ignore_ascii_case("ctrl+a"))),
        "{actions:?}"
    );
    assert!(
        actions
            .iter()
            .any(|action| matches!(action, InputAction::TypeText { .. }))
    );
}

fn ocr_label(id: &str, name: &str, x: i32, y: i32) -> InteractionTarget {
    let mut target = selection_target(name, false, false);
    target.id = id.into();
    target.control_type = "text".into();
    target.source = TargetSource::Ocr;
    target.actionable = false;
    target.bounds = Rect {
        x,
        y,
        width: 40,
        height: 12,
    };
    target.click_point = None;
    target
}

#[tokio::test]
async fn click_target_clicks_ocr_text_named_exactly_and_shown_once() {
    let mut observation = scroll_observation(&[], "calc");
    let window = observation.target.as_ref().unwrap().bounds.clone();
    observation.targets = vec![
        ocr_label("15", "Format", window.x + 160, window.y + 30),
        ocr_label("17", "Sheet", window.x + 260, window.y + 30),
        ocr_label("42", "Sheet", window.x + 60, window.y + 500),
    ];
    let platform = crate::platform::MockDesktop::new(observation.clone());
    let temp = tempfile::tempdir().unwrap();
    let context = mock_input_context(platform.clone(), &temp);
    *context.latest_observation.lock() = Some(observation.clone());
    let id = observation_id_for(&observation);
    ClickTargetTool
        .execute(
            json!({"observation_id": id, "target_id": "15", "expected_label": "Format"}),
            &context,
        )
        .await
        .unwrap();
    assert!(
        platform
            .actions()
            .iter()
            .any(|action| matches!(action, InputAction::Click { .. })),
        "{:?}",
        platform.actions()
    );

    // Text shown twice could be anything on the screen: no click.
    let platform = crate::platform::MockDesktop::new(observation.clone());
    let context = mock_input_context(platform.clone(), &temp);
    *context.latest_observation.lock() = Some(observation.clone());
    let _ = ClickTargetTool
        .execute(
            json!({"observation_id": id, "target_id": "17", "expected_label": "Sheet"}),
            &context,
        )
        .await;
    assert!(
        !platform
            .actions()
            .iter()
            .any(|action| matches!(action, InputAction::Click { .. }))
    );
}

#[test]
fn the_planner_sees_unique_ocr_labels_as_clickable() {
    let mut observation = scroll_observation(&[], "calc");
    let window = observation.target.as_ref().unwrap().bounds.clone();
    observation.targets = vec![
        ocr_label("15", "Format", window.x + 160, window.y + 30),
        ocr_label("17", "Sheet", window.x + 260, window.y + 30),
        ocr_label("42", "Sheet", window.x + 60, window.y + 500),
    ];
    let value = model_observation_value(&observation, false).unwrap();
    let shown = value["action_targets"].as_array().unwrap();
    let format = shown
        .iter()
        .find(|target| target["id"] == "15")
        .unwrap_or_else(|| panic!("Format missing: {value}"));
    assert_eq!(format["actionable"], true);
    assert_eq!(format["grounding_quality"], "medium");
    // Text shown twice cannot be clicked by its label, so it is not offered.
    assert!(!shown.iter().any(|target| target["label"] == "Sheet"));
}

fn page_element(name: &str, control_type: &str, bounds: Rect) -> UiElement {
    UiElement {
        name: name.into(),
        control_type: control_type.into(),
        automation_id: None,
        value: None,
        bounds,
        enabled: true,
        password: false,
        offscreen: false,
        keyboard_focusable: false,
        clickable_point: None,
        selected: None,
        focused: false,
        desktop_shell: false,
    }
}

/// A browser window: toolbar and bookmarks bar across the top, the web page
/// (a large Document) below.
fn browser_observation() -> Observation {
    let mut observation = scroll_observation(&[], "image");
    let window = observation.target.as_ref().unwrap().bounds.clone();
    let page = Rect {
        x: window.x,
        y: window.y + 120,
        width: window.width,
        height: window.height - 120,
    };
    observation.ui_elements = vec![page_element("Example feed", "Document", page)];
    observation.targets = vec![InteractionTarget {
        id: "31".into(),
        name: "News Archives | Example".into(),
        control_type: "button".into(),
        bounds: Rect {
            x: window.x + 300,
            y: window.y + 80,
            width: 180,
            height: 28,
        },
        source: TargetSource::Uia,
        confidence: None,
        enabled: true,
        actionable: true,
        click_point: None,
        selected: None,
        focused: false,
        desktop_shell: false,
        grounding_variant: None,
        rank_score: 14,
        rank_reasons: vec!["task_match:1".into()],
    }];
    observation
}

#[test]
fn an_untargeted_scroll_never_lands_on_the_bookmarks_bar() {
    // The task mentions "news"; a bookmark named "News Archives" matches it
    // but sits above the page, where the wheel does nothing.
    let observation = browser_observation();
    assert!(infer_scroll_target(&observation).is_none());
    let document = main_document_region(&observation).unwrap();
    let (point, label) = scroll_point(&observation, None).unwrap();
    assert!(label.is_none());
    assert!(
        document.contains(point.0, point.1),
        "scrolls the middle of the page"
    );
    assert!(point.1 > observation.target.as_ref().unwrap().bounds.y + 120);
}

#[test]
fn a_page_with_a_spinner_is_not_ready_even_after_its_title_changes() {
    let mut observation = browser_observation();
    let window = observation.target.as_ref().unwrap().bounds.clone();
    observation.ui_elements.push(page_element(
        "",
        "ProgressBar",
        Rect {
            x: window.x + 900,
            y: window.y + 400,
            width: 40,
            height: 40,
        },
    ));
    let identity = browser_page_identity(&observation);
    assert!(identity.busy);
    let before = BrowserPageIdentity {
        title: "previous page".into(),
        ..BrowserPageIdentity::default()
    };
    let loading = BrowserPageIdentity {
        title: "home / example".into(),
        url: Some("example.test/home".into()),
        ..identity
    };
    let (ready, ..) = navigation_sample_readiness(
        &before,
        &loading,
        "https://example.test/home",
        true,
        None,
        3,
    );
    assert!(
        !ready,
        "the tab title changed but the feed is still loading"
    );
    let loaded = BrowserPageIdentity {
        busy: false,
        ..loading
    };
    assert!(
        navigation_sample_readiness(&before, &loaded, "https://example.test/home", true, None, 3).0
    );
}

#[test]
fn a_settled_page_ignores_ticking_timers_but_not_new_content() {
    let names = |extra: &[&str]| {
        let mut landmarks: Vec<String> =
            ["home", "for you", "following", "what is happening", "post"]
                .iter()
                .chain(extra)
                .map(|name| name.to_string())
                .filter(|name| {
                    name.chars()
                        .filter(|character| character.is_alphabetic())
                        .count()
                        >= 3
                })
                .collect();
        landmarks.sort();
        landmarks
    };
    // A video timer ("0:10" -> "0:11") is not a landmark, so it cannot keep
    // the page from settling.
    assert!(landmarks_settled(&names(&["0:10"]), &names(&["0:11"])));
    // Posts still arriving change the landmarks.
    assert!(!landmarks_settled(
        &names(&[]),
        &names(&[
            "first post author",
            "second post author",
            "third post",
            "fourth post"
        ])
    ));
    // An empty page never counts as settled.
    assert!(!landmarks_settled(&[], &[]));
}

#[test]
fn loading_indicators_are_recognised_by_role_or_name() {
    let rect = Rect {
        x: 0,
        y: 0,
        width: 10,
        height: 10,
    };
    assert!(is_loading_indicator(&page_element(
        "",
        "ProgressBar",
        rect.clone()
    )));
    assert!(is_loading_indicator(&page_element(
        "Loading…",
        "Text",
        rect.clone()
    )));
    assert!(!is_loading_indicator(&page_element(
        "Loading dock photos",
        "Link",
        rect
    )));
}
