use super::*;

fn launch_observation(foreground: Option<(&str, &str)>) -> crate::types::Observation {
    crate::types::Observation {
        version: Uuid::new_v4(),
        captured_at: chrono::Utc::now(),
        foreground_window: foreground.map(|(title, process_name)| crate::types::WindowInfo {
            id: "window".into(),
            title: title.into(),
            process_name: process_name.into(),
            bounds: crate::types::Rect {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
            elevated: false,
            visible: true,
            minimized: false,
        }),
        target: None,
        cursor: None,
        screenshots: Vec::new(),
        ocr: Vec::new(),
        ui_elements: Vec::new(),
        targets: Vec::new(),
        timings_ms: BTreeMap::new(),
        warnings: Vec::new(),
    }
}

#[test]
fn explicit_urls_navigate_directly_but_files_do_not() {
    assert_eq!(
        requested_url("go to example.com in the browser", "").as_deref(),
        Some("https://example.com")
    );
    assert_eq!(
        requested_url("visit github.com/anthropics", "").as_deref(),
        Some("https://github.com/anthropics")
    );
    assert_eq!(
        requested_url("open https://docs.rs/serde and summarize it", "").as_deref(),
        Some("https://docs.rs/serde")
    );
    // File-like tokens never become navigations.
    assert_eq!(requested_url("open pok-ai.example.toml", ""), None);
    assert_eq!(requested_url("read the report.txt file", ""), None);
    assert_eq!(requested_url("what is the weather", ""), None);
}

#[test]
fn launch_candidate_only_for_known_apps_that_are_not_visible() {
    // A launch intent naming a known app that is not on screen.
    assert_eq!(
        requested_application("open notepad", "", None, None, &HashSet::new()).as_deref(),
        Some("notepad")
    );
    assert_eq!(
        requested_application(
            "type hello into the open notepad window",
            "",
            None,
            None,
            &HashSet::new()
        )
        .as_deref(),
        Some("notepad")
    );
    assert_eq!(
        requested_application("open calculator", "", None, None, &HashSet::new()).as_deref(),
        Some("calc")
    );
    // The app is already the foreground window.
    let visible = launch_observation(Some(("Untitled - Notepad", "notepad.exe")));
    assert_eq!(
        requested_application("open notepad", "", Some(&visible), None, &HashSet::new()),
        None
    );
    // The app was already listed by the last list_windows action.
    let listed = (
        "list_windows".to_string(),
        json!({"windows": [{"title": "Untitled - Notepad", "process_name": "notepad.exe"}]}),
    );
    assert_eq!(
        requested_application("open notepad", "", None, Some(&listed), &HashSet::new()),
        None
    );
    // An application already launched this run is not offered again.
    let launched = HashSet::from(["notepad".to_string()]);
    assert_eq!(
        requested_application("open notepad", "", None, None, &launched),
        None
    );
    // No launch intent, unknown apps, and token-boundary safety.
    assert_eq!(
        requested_application("read the file", "", None, None, &HashSet::new()),
        None
    );
    assert_eq!(
        requested_application("reset the password", "", None, None, &HashSet::new()),
        None
    );
    assert_eq!(
        requested_application("open the report", "", None, None, &HashSet::new()),
        None
    );
}

#[test]
fn terminal_goal_action_requires_post_commit_evidence() {
    let preview = json!({
        "executed": true,
        "action": {"label": "Print"},
        "state_change": {
            "removed_control_count": 0,
            "focus_changed": true,
            "added_text": []
        }
    });
    assert!(
        verified_terminal_goal_action("Print the completed document", "click_target", &preview)
            .is_none()
    );

    let committed = json!({
        "executed": true,
        "action": {"label": "Print"},
        "action_context": {
            "window_id": "42:HANDLE(0xabc)",
            "target_id": "9",
            "label": "Print",
            "control_type": "button",
            "bounds": {"x": 10, "y": 20, "width": 80, "height": 24}
        },
        "state_change": {
            "removed_control_count": 28,
            "focus_changed": true,
            "added_text": ["Printing page 1 of 1"]
        }
    });
    let (key, evidence) =
        verified_terminal_goal_action("Print the completed document", "click_target", &committed)
            .expect("affirmative post-commit text is terminal evidence");
    assert!(key.starts_with("print|42:handle(0xabc)|print|button|9:"));
    assert_eq!(evidence["removed_control_count"], 28);
    assert_eq!(evidence["status"], "confirmed");

    let submitted = json!({
        "executed": true,
        "action": {"label": "Print"},
        "action_context": {
            "window_id": "42:HANDLE(0xabc)",
            "target_id": "9",
            "label": "Print",
            "control_type": "button",
            "bounds": {"x": 10, "y": 20, "width": 80, "height": 24}
        },
        "targets": [],
        "state_change": {
            "removed_control_count": 28,
            "focus_changed": false,
            "added_text": []
        }
    });
    let (_, submitted_evidence) =
        verified_terminal_goal_action("Print the completed document", "click_target", &submitted)
            .expect("a closed final print dialog is a one-time submission");
    assert_eq!(submitted_evidence["status"], "submitted");

    let mut opens_followup_dialog = submitted.clone();
    opens_followup_dialog["targets"] = json!([{"label": "Print"}]);
    assert!(
        verified_terminal_goal_action(
            "Print the completed document",
            "click_target",
            &opens_followup_dialog,
        )
        .is_none()
    );

    let mut raw_click = committed.clone();
    raw_click["action"] = json!({"kind": "click"});
    assert!(
        verified_terminal_goal_action(
            "Print the completed document",
            "simulate_input",
            &raw_click,
        )
        .is_some()
    );
}

#[test]
fn goal_action_identity_distinguishes_same_label_in_different_windows() {
    let first = goal_action_context_key("print", "parent", "Print", "button", None);
    let second = goal_action_context_key("print", "dialog", "Print", "button", None);
    assert_ne!(first, second);
}

#[test]
fn active_window_capture_is_focused_visual_evidence() {
    let value = json!({
        "verification_state": "clean",
        "executed": true,
        "target": {"scope": "active_window", "app": "viewer.exe"}
    });
    assert!(is_focused_visual_artifact_evidence(
        "capture_screen",
        &value
    ));
}

#[test]
fn user_guidance_requires_an_explicit_positive_repeat_request() {
    assert!(guidance_requests_repeat("Please print again"));
    assert!(guidance_requests_repeat("Please make another copy"));
    assert!(!guidance_requests_repeat("It worked; do not print again"));
    assert!(!guidance_requests_repeat(
        "It worked, but the frame option was left selected"
    ));
    assert!(guidance_confirms_completion(
        "It worked and you accomplished the task"
    ));
}

#[test]
fn workflow_quality_rejects_repeats_and_accepts_recent_progress() {
    let metrics = RunMetrics::default();
    assert_eq!(
        workflow_quality_rejection(
            &[json!({
                "executed": false,
                "outcome": "blocked_repeat",
                "state_change_count": 0
            })],
            &metrics
        ),
        Some("workflow contained blocked repeated actions")
    );
    let qualification = WorkflowQualification {
        eligible: true,
        applications: vec!["chrome.exe".into()],
        evidence: "post_input_uia_state_change",
    };
    assert_eq!(
        workflow_learning_rejection(
            "Check the current prices this week",
            &qualification,
            &[json!({
                "executed": true,
                "success": true,
                "outcome": "progress",
                "state_change_count": 1
            })],
            &metrics,
        ),
        Some("read-only live information was not outcome-verified")
    );
    assert_eq!(
        workflow_quality_rejection(
            &[json!({
                "executed": true,
                "success": true,
                "outcome": "progress",
                "state_change_count": 1
            })],
            &metrics
        ),
        None
    );
}

#[test]
fn helper_command_parser_preserves_quoted_paths_with_spaces() {
    let (path, runtime) =
        helper_from_command(r#"python "F:\Coding Adventures\helpers\fetch prices.py" --today"#)
            .expect("quoted Python helper should be recognized");
    assert_eq!(
        path,
        std::path::PathBuf::from(r"F:\Coding Adventures\helpers\fetch prices.py")
    );
    assert!(matches!(
        runtime,
        crate::generated_tools::ScriptRuntime::Python
    ));
}

#[derive(Default)]
struct RecordingObserver {
    events: Mutex<Vec<String>>,
}

impl SessionObserver for RecordingObserver {
    fn emit(&self, event: AgentEvent) {
        let label = match event {
            AgentEvent::CompressionStarted { sequence, .. } => {
                format!("compression_started:{sequence}")
            }
            AgentEvent::CompressionCompleted { sequence, .. } => {
                format!("compression_completed:{sequence}")
            }
            AgentEvent::CompressionFailed { sequence, .. } => {
                format!("compression_failed:{sequence}")
            }
            AgentEvent::TurnStarted { turn } => format!("turn_started:{turn}"),
            AgentEvent::ToolStarted { name, .. } => format!("tool_started:{name}"),
            AgentEvent::DecisionRouterStarted { purpose, .. } => {
                format!("router_started:{purpose}")
            }
            AgentEvent::DecisionRouterEvaluated {
                purpose,
                rejection_reason,
                ..
            } => format!(
                "router_evaluated:{purpose}:{}",
                rejection_reason.as_deref().unwrap_or("picked")
            ),
            _ => return,
        };
        self.events.lock().push(label);
    }
}

fn channel_state() -> VerifiedState {
    VerifiedState {
        observation_id: Some("obs_channel".into()),
        app: Some("msedge.exe".into()),
        title: Some("WorldofAI - YouTube".into()),
        observed_url: Some("https://www.youtube.com/@intheworldofai".into()),
        url_provenance: Some("uia_value".into()),
        requested_destination: None,
        page_status: "loaded".into(),
        evidence: vec!["Latest from WorldofAI".into()],
        last_action: "click_target".into(),
        outcome: "progress".into(),
    }
}

#[test]
fn stale_model_plan_is_overridden_by_verified_state() {
    let state = ActiveTaskState {
        root_request: "Find WorldofAI's latest video".into(),
        current_step: "Search for WorldofAI channel".into(),
        ..ActiveTaskState::default()
    };
    let reminder = active_task_reminder(
        &state,
        Some(&channel_state()),
        true,
        "- run_command [system; inactive]: Run a complete shell command",
    );
    let MessageContent::Text { text } = &reminder.content[0] else {
        panic!("expected text reminder");
    };
    assert!(text.contains("MODEL PLAN STEP (may be stale)"));
    assert!(text.contains("CURRENT VERIFIED STATE"));
    assert!(text.contains("WorldofAI - YouTube"));
    assert!(text.contains("https://www.youtube.com/@intheworldofai"));
    assert!(text.contains("advance the plan instead of repeating"));
}

#[test]
fn repeated_failed_navigation_warns_then_blocks() {
    let arguments = json!({
        "window_id": "73960:HANDLE(0xc1598)",
        "query_or_url": "https://www.youtube.com/@WorldofAI"
    });
    let failed_page = Ok(json!({
        "observation_id": "obs_404",
        "target": {
            "app": "msedge.exe",
            "title": "404 Not Found - Microsoft Edge"
        },
        "targets": [
            {"id": "1", "label": "https://www.youtube.com/@WorldofAI"}
        ]
    }));
    let mut continuity = ExecutionContinuity::default();

    let PreCallDecision::Execute {
        signature,
        prior_attempts,
    } = continuity.before_call("browser_navigate", &arguments, None)
    else {
        panic!("first navigation should execute");
    };
    let feedback = continuity.after_call(
        "browser_navigate",
        &arguments,
        &signature,
        prior_attempts,
        &failed_page,
    );
    assert_eq!(feedback.outcome, "failed");
    assert!(feedback.warning.is_none());

    let PreCallDecision::Execute { prior_attempts, .. } =
        continuity.before_call("browser_navigate", &arguments, None)
    else {
        panic!("one corrective repeat should execute");
    };
    let feedback = continuity.after_call(
        "browser_navigate",
        &arguments,
        &signature,
        prior_attempts,
        &failed_page,
    );
    assert!(feedback.warning.is_some());

    let PreCallDecision::Suppress {
        outcome, reason, ..
    } = continuity.before_call("browser_navigate", &arguments, None)
    else {
        panic!("third identical failed navigation should be blocked");
    };
    assert_eq!(outcome, "blocked_repeat");
    assert!(reason.contains("choose a different next action"));
}

#[test]
fn loading_navigation_is_not_fresh_grounding_evidence() {
    assert!(!navigation_result_is_ready(&json!({
        "navigation_readiness": {"status": "loading"}
    })));
    assert!(!navigation_result_is_ready(&json!({
        "navigation_readiness": {"status": "error_page"}
    })));
    assert!(navigation_result_is_ready(&json!({
        "navigation_readiness": {"status": "ready"}
    })));
    assert!(navigation_result_is_ready(&json!({})));
    assert!(qualifies_as_fresh_evidence(
        "get_current_time",
        &json!({"local_iso8601": "2026-09-19T19:25:00-07:00"}),
        true,
    ));
    assert!(!qualifies_as_fresh_evidence(
        "get_current_time",
        &json!({"status": "running"}),
        true,
    ));
    assert!(!qualifies_as_fresh_evidence(
        "get_current_time",
        &json!({"executed": false}),
        true,
    ));
}

#[test]
fn navigation_to_current_observed_url_is_suppressed_as_satisfied() {
    let continuity = ExecutionContinuity {
        current: Some(channel_state()),
        ..ExecutionContinuity::default()
    };
    let decision = continuity.before_call(
        "browser_navigate",
        &json!({
            "window_id": "73960:HANDLE(0xc1598)",
            "query_or_url": "youtube.com/@intheworldofai/"
        }),
        None,
    );
    let PreCallDecision::Suppress { outcome, .. } = decision else {
        panic!("already-current navigation should not execute");
    };
    assert_eq!(outcome, "already_current");
}

#[test]
fn repeated_no_progress_action_blocks_but_observation_does_not() {
    let mut continuity = ExecutionContinuity {
        current: Some(channel_state()),
        ..ExecutionContinuity::default()
    };
    let arguments = json!({"amount": "page", "direction": "down"});
    let unchanged = Ok(json!({
        "observation_id": "obs_same",
        "focus": {"app": "msedge.exe", "title": "WorldofAI - YouTube"},
        "attempts": [{"method": "wheel", "viewport_changed": false}],
        "state_change": {
            "added_text": [],
            "removed_control_count": 0,
            "focus_changed": false
        }
    }));
    for attempt in 0..2 {
        let PreCallDecision::Execute {
            signature,
            prior_attempts,
        } = continuity.before_call("scroll_view", &arguments, None)
        else {
            panic!("attempt {attempt} should execute");
        };
        continuity.after_call(
            "scroll_view",
            &arguments,
            &signature,
            prior_attempts,
            &unchanged,
        );
    }
    assert!(matches!(
        continuity.before_call("scroll_view", &arguments, None),
        PreCallDecision::Suppress {
            outcome: "blocked_repeat",
            ..
        }
    ));
    assert!(matches!(
        continuity.before_call("capture_screen", &json!({}), None),
        PreCallDecision::Execute { .. }
    ));
}

#[test]
fn different_click_ids_cannot_bypass_unchanged_page_grounding_recovery() {
    let mut continuity = ExecutionContinuity {
        current: Some(channel_state()),
        ..ExecutionContinuity::default()
    };
    let unchanged = Ok(json!({
        "observation_id": "obs_same",
        "focus": {"app": "msedge.exe", "title": "WorldofAI - YouTube"},
        "state_change": {
            "added_text": [],
            "removed_control_count": 0,
            "focus_changed": false
        }
    }));
    for target_id in ["71", "75"] {
        let arguments = json!({"target_id": target_id, "expected_label": "Result"});
        let PreCallDecision::Execute {
            signature,
            prior_attempts,
        } = continuity.before_call("click_target", &arguments, None)
        else {
            panic!("the first two distinct clicks should reach verification");
        };
        continuity.after_call(
            "click_target",
            &arguments,
            &signature,
            prior_attempts,
            &unchanged,
        );
    }
    assert!(matches!(
        continuity.before_call(
            "click_target",
            &json!({"target_id": "78", "expected_label": "Result"}),
            None,
        ),
        PreCallDecision::Suppress { .. }
    ));

    continuity.after_call(
        "query_screen_text",
        &json!({}),
        "query_screen_text:test",
        0,
        &Ok(json!({"observation_id": "obs_same", "lines": []})),
    );
    assert!(matches!(
        continuity.before_call(
            "click_target",
            &json!({"target_id": "78", "expected_label": "Result"}),
            None,
        ),
        PreCallDecision::Execute { .. }
    ));
}

#[test]
fn exhausted_activation_recovery_is_failed_and_blocks_an_identical_retry() {
    let mut continuity = ExecutionContinuity::default();
    let arguments = json!({"window_id": "42:HANDLE(0x2A)"});
    let PreCallDecision::Execute {
        signature,
        prior_attempts,
    } = continuity.before_call("activate_window", &arguments, None)
    else {
        panic!("first activation should execute");
    };
    let result = Ok(json!({
        "status": "recovery_required",
        "recovery_required": true,
        "executed": false,
        "observation_id": "obs_shell",
        "target": {"scope": "region", "title": "Screen region", "app": ""},
        "attempts": [{"strategy": "native", "verified": false}]
    }));
    let feedback = continuity.after_call(
        "activate_window",
        &arguments,
        &signature,
        prior_attempts,
        &result,
    );
    assert_eq!(feedback.outcome, "failed");
    assert!(matches!(
        continuity.before_call("activate_window", &arguments, None),
        PreCallDecision::Suppress {
            outcome: "blocked_repeat",
            ..
        }
    ));
}

#[test]
fn repeated_scroll_with_verified_progress_remains_allowed() {
    let mut continuity = ExecutionContinuity::default();
    let arguments = json!({"amount": "page", "direction": "down"});
    for (index, label) in ["Bluetooth", "Printers & scanners", "Mouse"]
        .iter()
        .enumerate()
    {
        let PreCallDecision::Execute {
            signature,
            prior_attempts,
        } = continuity.before_call("scroll_view", &arguments, None)
        else {
            panic!("progressing scroll {index} should execute");
        };
        let result = Ok(json!({
            "observation_id": format!("settings-{index}"),
            "focus": {"app": "SystemSettings.exe", "title": "Bluetooth & devices"},
            "attempts": [{"method": "wheel", "viewport_changed": true}],
            "state_change": {
                "added_text": [label],
                "removed_control_count": 1,
                "focus_changed": false
            }
        }));
        let feedback = continuity.after_call(
            "scroll_view",
            &arguments,
            &signature,
            prior_attempts,
            &result,
        );
        assert_eq!(feedback.outcome, "progress");
        assert!(
            feedback.warning.is_none(),
            "successful repeated scrolling should not produce a loop warning"
        );
    }
    assert!(matches!(
        continuity.before_call("scroll_view", &arguments, None),
        PreCallDecision::Execute { .. }
    ));
    assert_eq!(
        continuity
            .current
            .as_ref()
            .and_then(|state| state.app.as_deref()),
        Some("SystemSettings.exe")
    );
}

#[test]
fn continuity_reset_starts_a_new_computer_use_request_cleanly() {
    let mut continuity = ExecutionContinuity {
        current: Some(VerifiedState {
            app: Some("notepad.exe".into()),
            title: Some("notes.txt - Notepad".into()),
            page_status: "loaded".into(),
            ..VerifiedState::default()
        }),
        ..ExecutionContinuity::default()
    };
    continuity.actions.insert(
        "click_target:old".into(),
        ActionRecord {
            attempts: 2,
            failures: 0,
            no_progress: 2,
            last_outcome: "no_progress".into(),
        },
    );

    continuity.reset();

    assert!(continuity.current.is_none());
    assert!(continuity.actions.is_empty());
}

#[test]
fn continuity_metadata_preserves_original_tool_result() {
    let feedback = ContinuityFeedback {
        outcome: "progress".into(),
        warning: None,
        current_state: Some(VerifiedState {
            app: Some("explorer.exe".into()),
            title: Some("Downloads".into()),
            page_status: "loaded".into(),
            ..VerifiedState::default()
        }),
    };
    let enriched = enrich_continuity_result(
        json!({"executed": true, "selected_item": "report.pdf"}),
        &feedback,
    );

    assert_eq!(enriched["executed"], true);
    assert_eq!(enriched["selected_item"], "report.pdf");
    assert_eq!(enriched["_pok_continuity"]["outcome"], "progress");
    assert_eq!(
        enriched["_pok_continuity"]["current_state"]["app"],
        "explorer.exe"
    );
}

#[test]
fn capacity_errors_retry_until_cancelled() {
    let rate_limit = PokError::ProviderTransient {
        message: "HTTP 429 Too Many Requests".into(),
        retry_after_ms: None,
        retry_until_cancelled: true,
    };
    assert!(provider_retry_permitted(&rate_limit, 10_000, 3));
    assert!(!provider_retry_permitted(&rate_limit, 0, 0));

    let connection = PokError::ProviderTransient {
        message: "connection reset".into(),
        retry_after_ms: None,
        retry_until_cancelled: false,
    };
    assert!(provider_retry_permitted(&connection, 2, 3));
    assert!(!provider_retry_permitted(&connection, 3, 3));
}

#[test]
fn provider_retry_backoff_increases_and_honors_server_delay() {
    assert_eq!(provider_retry_delay_ms(1, Some(1_234)), 1_234);

    let first = provider_retry_delay_ms(1, None);
    let second = provider_retry_delay_ms(2, None);
    let capped = provider_retry_delay_ms(20, None);
    assert!((2_000..=3_000).contains(&first));
    assert!((4_000..=6_000).contains(&second));
    assert!((30_000..=45_000).contains(&capped));
}

#[test]
fn image_rejection_retries_as_text_only_without_losing_grounding_text() {
    let request = BrainRequest {
        model: "text-model".into(),
        messages: vec![BrainMessage {
            role: "user".into(),
            content: vec![
                MessageContent::Text {
                    text: "Current desktop screenshot.".into(),
                },
                MessageContent::ImagePng {
                    base64: "AA==".into(),
                },
            ],
            origin: MessageOrigin::ToolImage,
            tool_call_id: None,
            tool_calls: Vec::new(),
        }],
        tools: Vec::new(),
        temperature: Some(0.0),
        max_tokens: Some(100),
        seed: None,
        reasoning_effort: None,
    };
    assert!(request_contains_images(&request));
    let request = request_without_images(request);
    assert!(!request_contains_images(&request));
    let rendered = serde_json::to_string(&request.messages).unwrap();
    assert!(rendered.contains("OCR/UI Automation targets"));
    assert!(is_image_input_rejection(&PokError::Provider(
        "messages contain images, but this model does not support image inputs".into()
    )));
    assert!(is_image_input_rejection(&PokError::Provider(
        "provider error: HTTP 400 Bad Request: {\"error\":{\"message\":\"Provider returned error\",\"code\":400,\"metadata\":{\"raw\":\"{\\\"error\\\":{\\\"message\\\":\\\"The request is invalid.\\\",\\\"type\\\":\\\"invalid_request_error\\\",\\\"param\\\":\\\"\\\",\\\"code\\\":\\\"invalid_request\\\"}}\",\"provider_name\":\"Nex AGI\",\"is_byok\":false,\"provider_error_code\":\"invalid_request\"}},\"user_id\":\"user_2uoT9YTQvJjJ92k6QrYw7eTh2ag\"}".into()
    )));
}

#[test]
fn strict_templates_receive_one_leading_system_message() {
    let request = BrainRequest {
        model: "strict-template-model".into(),
        messages: vec![
            BrainMessage::text("system", "base instructions"),
            BrainMessage::text("system", "runtime context"),
            BrainMessage::text("user", "do the task"),
            BrainMessage::text("assistant", "working"),
            BrainMessage::text("system", "late reminder"),
            BrainMessage::text("user", "continue"),
        ],
        tools: Vec::new(),
        temperature: Some(0.0),
        max_tokens: Some(100),
        seed: None,
        reasoning_effort: None,
    };

    let request = request_with_single_leading_system_message(request);
    let roles = request
        .messages
        .iter()
        .map(|message| message.role.as_str())
        .collect::<Vec<_>>();
    assert_eq!(roles, ["system", "user", "assistant", "user"]);
    let MessageContent::Text { text: first } = &request.messages[0].content[0] else {
        panic!("expected text system message");
    };
    let MessageContent::Text { text: second } = &request.messages[0].content[2] else {
        panic!("expected merged runtime context");
    };
    let MessageContent::Text { text: third } = &request.messages[0].content[4] else {
        panic!("expected merged late reminder");
    };
    assert_eq!(first, "base instructions");
    assert_eq!(second, "runtime context");
    assert_eq!(third, "late reminder");
}

#[test]
fn mistral_templates_receive_alternating_plain_conversation_roles() {
    let request = BrainRequest {
        model: "mistral-strict-template".into(),
        messages: vec![
            BrainMessage::text("system", "base instructions"),
            BrainMessage::text("system", "runtime context"),
            BrainMessage::text("user", "what time is it"),
            BrainMessage::text_with_origin(
                "user",
                "Active task reminder",
                MessageOrigin::SystemReminder,
            ),
            BrainMessage {
                role: "assistant".into(),
                content: vec![MessageContent::Text {
                    text: "I will check".into(),
                }],
                origin: MessageOrigin::Assistant,
                tool_call_id: None,
                tool_calls: vec![CompletedToolCall {
                    id: "clock-call".into(),
                    name: "get_current_time".into(),
                    arguments: json!({}),
                }],
            },
            BrainMessage {
                role: "tool".into(),
                content: vec![MessageContent::Text {
                    text: "11:15 PM".into(),
                }],
                origin: MessageOrigin::ToolResult,
                tool_call_id: Some("clock-call".into()),
                tool_calls: Vec::new(),
            },
            BrainMessage {
                role: "user".into(),
                content: vec![
                    MessageContent::Text {
                        text: "Current desktop screenshot".into(),
                    },
                    MessageContent::ImagePng {
                        base64: "duplicate-tool-image".into(),
                    },
                ],
                origin: MessageOrigin::ToolImage,
                tool_call_id: None,
                tool_calls: Vec::new(),
            },
            BrainMessage::text_with_origin(
                "user",
                "Post-tool active task reminder",
                MessageOrigin::SystemReminder,
            ),
        ],
        tools: Vec::new(),
        temperature: None,
        max_tokens: Some(100),
        seed: None,
        reasoning_effort: None,
    };

    let request = request_with_alternating_conversation_roles(request);
    let roles = request
        .messages
        .iter()
        .map(|message| message.role.as_str())
        .collect::<Vec<_>>();
    assert_eq!(roles, ["system", "user", "assistant", "tool"]);
    let rendered = serde_json::to_string(&request.messages).unwrap();
    assert!(rendered.contains("what time is it"));
    assert!(rendered.contains("Active task reminder"));
    assert!(rendered.contains("I will check"));
    assert!(rendered.contains("clock-call"));
    assert!(rendered.contains("Post-tool active task reminder"));
    assert!(!rendered.contains("duplicate-tool-image"));
    assert!(strict_conversation_roles_are_valid(&request.messages));
}

#[test]
fn strict_mistral_layout_supports_repeated_tool_cycles_and_followups() {
    let assistant_call = |id: &str, name: &str| BrainMessage {
        role: "assistant".into(),
        content: Vec::new(),
        origin: MessageOrigin::Assistant,
        tool_call_id: None,
        tool_calls: vec![CompletedToolCall {
            id: id.into(),
            name: name.into(),
            arguments: json!({}),
        }],
    };
    let tool_result = |id: &str| BrainMessage {
        role: "tool".into(),
        content: vec![MessageContent::Text { text: "{}".into() }],
        origin: MessageOrigin::ToolResult,
        tool_call_id: Some(id.into()),
        tool_calls: Vec::new(),
    };

    let request = BrainRequest {
        model: "mistralai/ministral-3-14b-reasoning".into(),
        messages: vec![
            BrainMessage::text("system", "base"),
            BrainMessage::text("user", "what time is it"),
            assistant_call("clock", "get_current_time"),
            tool_result("clock"),
            BrainMessage::text("assistant", "It is 11:27 PM."),
            BrainMessage::text("system", "recalled context"),
            BrainMessage::text("user", "check the weather"),
            assistant_call("windows", "list_windows"),
            tool_result("windows"),
            assistant_call("navigate", "browser_navigate"),
            tool_result("navigate"),
            BrainMessage {
                role: "user".into(),
                content: vec![MessageContent::ImagePng {
                    base64: "screen".into(),
                }],
                origin: MessageOrigin::ToolImage,
                tool_call_id: None,
                tool_calls: Vec::new(),
            },
            BrainMessage::text_with_origin(
                "user",
                "Stay focused on current weather",
                MessageOrigin::SystemReminder,
            ),
        ],
        tools: Vec::new(),
        temperature: None,
        max_tokens: Some(100),
        seed: None,
        reasoning_effort: None,
    };

    let request = request_with_alternating_conversation_roles(request);
    let roles = request
        .messages
        .iter()
        .map(|message| message.role.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        roles,
        [
            "system",
            "user",
            "assistant",
            "tool",
            "assistant",
            "user",
            "assistant",
            "tool",
            "assistant",
            "tool"
        ]
    );
    assert!(strict_conversation_roles_are_valid(&request.messages));
}

#[test]
fn mistral_models_enable_strict_layout_without_changing_other_models() {
    assert!(model_requires_strict_role_layout(
        "mistralai/ministral-3-14b-reasoning"
    ));
    assert!(model_requires_strict_role_layout("mistral-small-instruct"));
    assert!(!model_requires_strict_role_layout("qwen3.5-35b-a3b"));
    assert!(!model_requires_strict_role_layout("gemma-3-27b"));
}

#[test]
fn detects_strict_system_message_template_rejection() {
    assert!(is_system_message_order_rejection(&PokError::Provider(
        "Jinja Exception: System message must be at the beginning.".into(),
    )));
    assert!(!is_system_message_order_rejection(&PokError::Provider(
        "HTTP 400 invalid request".into(),
    )));
    assert!(is_role_alternation_rejection(&PokError::Provider(
        "Jinja Exception: After the optional system message, conversation roles must alternate user and assistant roles except for tool calls and results.".into(),
    )));
    assert!(!is_role_alternation_rejection(&PokError::Provider(
        "HTTP 500 model worker unavailable".into(),
    )));
}

#[test]
fn detects_only_unsupported_temperature_errors() {
    for message in [
        "Unsupported parameter: temperature",
        "This model does not support temperature",
        "temperature: unknown parameter",
        "Unrecognized request argument supplied: temperature",
    ] {
        assert!(is_unsupported_temperature_rejection(&PokError::Provider(
            message.into()
        )));
    }
    assert!(!is_unsupported_temperature_rejection(&PokError::Provider(
        "temperature must be between 0 and 2".into()
    )));
    assert!(!is_unsupported_temperature_rejection(&PokError::Provider(
        "HTTP 400 invalid request".into()
    )));
}

#[test]
fn recovers_lm_studio_default_tool_format() {
    let calls = recover_tool_calls(
        r#"[TOOL_REQUEST]{"name":"read_file","arguments":{"path":"x"}}[END_TOOL_REQUEST]"#,
        &["read_file".into()],
    );
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "read_file");
}

#[test]
fn recovers_first_qwen_reasoning_tool_tag_without_executing_followups() {
    let call = recover_xml_tool_call(
        "<tool_call><function=capture_screen>{}</function></tool_call>\
         <tool_call><function=observe_desktop></function></tool_call>",
        &["capture_screen".into(), "observe_desktop".into()],
    )
    .expect("first strict tool tag should recover");
    assert_eq!(call.name, "capture_screen");
    assert_eq!(call.arguments, json!({}));
}

#[test]
fn xml_tool_recovery_rejects_unknown_or_unclosed_calls() {
    assert!(
        recover_xml_tool_call(
            "<tool_call><function=unknown>{}</function></tool_call>",
            &["capture_screen".into()],
        )
        .is_none()
    );
    assert!(
        recover_xml_tool_call(
            "<tool_call><function=capture_screen><parameter=x>1</function></tool_call>",
            &["capture_screen".into()],
        )
        .is_none()
    );
}

#[test]
fn recovers_xml_parameter_tool_format_without_leaking_it_as_an_answer() {
    let call = recover_xml_tool_call(
        "Before acting. <tool_call><function=run_command><parameter=command>Write-Output 'ready'\nWrite-Output 'done'</parameter><parameter=timeout_seconds>90</parameter></function></tool_call>",
        &["run_command".into()],
    )
    .expect("XML parameter tool call should recover");
    assert_eq!(call.name, "run_command");
    assert_eq!(call.arguments["timeout_seconds"], 90);
    assert!(call.arguments["command"].as_str().unwrap().contains("done"));
    assert!(contains_tool_protocol_markup(
        "<tool_call><function=run_command></function></tool_call>"
    ));
}

#[test]
fn malformed_and_reasoning_only_turns_are_not_provider_empty() {
    let malformed = vec![(
        "update_task_plan".into(),
        r#"{"status":completed}"#.into(),
        "invalid json".into(),
    )];
    assert_eq!(
        model_turn_payload_kind(&[], "", "", &malformed),
        ModelTurnPayloadKind::MalformedToolCalls
    );
    assert_eq!(
        model_turn_payload_kind(&[], "The answer is day 49.", "", &malformed),
        ModelTurnPayloadKind::VisibleText
    );
    let malformed_action = vec![(
        "simulate_input".into(),
        r#"{"action":click}"#.into(),
        "invalid json".into(),
    )];
    assert_eq!(
        model_turn_payload_kind(&[], "I clicked it.", "", &malformed_action),
        ModelTurnPayloadKind::MalformedToolCalls
    );
    assert_eq!(
        model_turn_payload_kind(&[], "", "answer exists in reasoning", &[]),
        ModelTurnPayloadKind::ReasoningOnly
    );
    assert_eq!(
        model_turn_payload_kind(&[], "", "", &[]),
        ModelTurnPayloadKind::TrulyEmpty
    );
}

#[test]
fn repairs_only_schema_declared_bare_enum_values() {
    let mut registry = ToolRegistry::new();
    crate::builtins::register_desktop_tools(&mut registry);
    let repaired = recover_malformed_tool_call(
        &(
            "click_target".into(),
            r#"{"button":right,"target_id":"63"}"#.into(),
            "expected value".into(),
        ),
        &registry,
    )
    .expect("mouse button enum should be repaired");
    assert_eq!(
        repaired.arguments,
        json!({"button": "right", "target_id": "63"})
    );
    assert!(
        recover_malformed_tool_call(
            &(
                "run_command".into(),
                r#"{"command":dangerous}"#.into(),
                "expected value".into(),
            ),
            &registry,
        )
        .is_none()
    );
}

#[test]
fn reasoning_length_is_not_an_empty_provider_response() {
    assert_eq!(initial_response_max_tokens("deepseek-v4-flash"), 16_384);
    assert_eq!(
        initial_response_max_tokens("thinkingcap-qwen3.8-27b"),
        16_384
    );
    assert_eq!(initial_response_max_tokens("ordinary-model"), 4_096);
    assert!(reasoning_output_limit_reached(
        Some("length"),
        &[],
        "",
        "substantial private reasoning",
    ));
    assert!(!reasoning_output_limit_reached(
        Some("stop"),
        &[],
        "",
        "substantial private reasoning",
    ));
    assert!(!reasoning_output_limit_reached(
        Some("length"),
        &[],
        "visible answer",
        "reasoning",
    ));
}

#[test]
fn advisory_artifact_findings_are_completion_notes_not_blockers() {
    let evidence = crate::tool::ArtifactEvidence {
        evidence_id: Uuid::new_v4(),
        path: "result.xlsx".into(),
        operation: "created".into(),
        artifact_type: "workbook_package".into(),
        integrity: "package_readable".into(),
        validation_status: "needs_review".into(),
        size_bytes: 10,
        sha256: "abc".into(),
        modified_at: "now".into(),
        structure: json!({}),
        warnings: vec![crate::tool::ArtifactWarning {
            severity: "warning".into(),
            code: "chart_layout_review".into(),
            message: "Chart layout may be improved".into(),
        }],
        observed_at: "now".into(),
    };
    assert!(!has_blocking_artifact_warning(&evidence));
    let warnings = completion_warnings_for_deliverable(
        Some(&evidence),
        std::slice::from_ref(&evidence),
        true,
        false,
    );
    assert_eq!(warnings.len(), 1);
    assert_eq!(warnings[0].code, "chart_layout_review");

    let mut blocking = evidence;
    blocking.warnings[0].severity = "error".into();
    assert!(has_blocking_artifact_warning(&blocking));
}

#[test]
fn authoritative_deliverables_exclude_preexisting_observations() {
    let observed = crate::tool::ArtifactEvidence {
        evidence_id: Uuid::new_v4(),
        path: "input.pdf".into(),
        operation: "command_observed".into(),
        artifact_type: "binary".into(),
        integrity: "metadata_only".into(),
        validation_status: "needs_review".into(),
        size_bytes: 10,
        sha256: "input".into(),
        modified_at: "now".into(),
        structure: json!({}),
        warnings: Vec::new(),
        observed_at: "now".into(),
    };
    let mut created = observed.clone();
    created.path = "output.xlsx".into();
    created.operation = "command_created".into();
    created.artifact_type = "workbook_package".into();
    created.sha256 = "output".into();
    let deliverables = authoritative_deliverables(&[observed, created]);
    assert_eq!(deliverables.len(), 1);
    assert_eq!(
        deliverables[0].path,
        std::path::PathBuf::from("output.xlsx")
    );
}

#[test]
fn screenshot_tool_result_separates_tool_text_from_user_images() {
    let call = CompletedToolCall {
        id: "call-1".into(),
        name: "capture_screen".into(),
        arguments: json!({}),
    };
    let messages = tool_result_messages(
        &call,
        Ok(json!({
            "screenshots": [
                {"view": "clean", "png_base64": "AA=="},
                {"view": "annotated", "png_base64": "AQ=="}
            ]
        })),
        true,
    );
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].role, "tool");
    assert_eq!(messages[0].tool_call_id.as_deref(), Some("call-1"));
    assert!(!matches!(
        messages[0].content[0],
        MessageContent::ImagePng { .. }
    ));
    assert_eq!(messages[1].role, "user");
    assert!(matches!(
        messages[1].content[1],
        MessageContent::ImagePng { .. }
    ));
    assert!(matches!(
        messages[1].content[2],
        MessageContent::ImagePng { .. }
    ));
}

#[test]
fn nested_batch_tool_result_emits_only_the_final_visual_pair() {
    let call = CompletedToolCall {
        id: "call-batch".into(),
        name: "execute_action_batch".into(),
        arguments: json!({}),
    };
    let step_results = (0..9)
        .map(|index| {
            json!({
                "step": index,
                "result": {
                    "screenshots": [
                        {"view": "clean", "png_base64": format!("clean-{index}")},
                        {"view": "annotated", "png_base64": format!("annotated-{index}")}
                    ]
                }
            })
        })
        .collect::<Vec<_>>();
    let messages = tool_result_messages(
        &call,
        Ok(json!({"success": true, "step_results": step_results})),
        true,
    );

    assert_eq!(count_images(&messages), CURRENT_VISUAL_PAIR_IMAGES);
    let rendered = serde_json::to_string(&messages).unwrap();
    assert!(rendered.contains("clean-8"));
    assert!(rendered.contains("annotated-8"));
    assert!(!rendered.contains("annotated-7"));
}

#[test]
fn recursively_removes_images_from_any_tool_result() {
    let call = CompletedToolCall {
        id: "call-2".into(),
        name: "simulate_input".into(),
        arguments: json!({}),
    };
    let messages = tool_result_messages(
        &call,
        Ok(json!({"observation": {"screenshots": [{"png_base64": "AA=="}]}})),
        true,
    );
    assert_eq!(messages.len(), 2);
    let MessageContent::Text { text } = &messages[0].content[0] else {
        panic!("tool result must be text");
    };
    assert!(!text.contains("png_base64"));
    assert!(!text.contains("AA=="));
    assert!(matches!(
        messages[1].content[1],
        MessageContent::ImagePng { .. }
    ));
}

#[test]
fn text_only_model_gets_targets_without_image_message() {
    let call = CompletedToolCall {
        id: "call-text".into(),
        name: "capture_screen".into(),
        arguments: json!({}),
    };
    let messages = tool_result_messages(
        &call,
        Ok(json!({
            "targets": [{"id": "7", "name": "Address bar", "control_type": "edit"}],
            "screenshots": [{"png_base64": "AA==", "width": 1600, "height": 900}]
        })),
        false,
    );
    assert_eq!(messages.len(), 1);
    let MessageContent::Text { text } = &messages[0].content[0] else {
        panic!("tool result must be text");
    };
    assert!(text.contains("Address bar"));
    assert!(!text.contains("png_base64"));
    assert!(!text.contains("AA=="));
}

#[test]
fn command_completion_event_carries_bounded_output_and_logical_status() {
    let successful = Ok(json!({
        "cwd": "C:\\workspace",
        "exit_code": 0,
        "success": true,
        "stdout": "Pokémon °C\r\n",
        "stderr": "",
        "output_truncated": false,
    }));
    assert!(tool_finished_ok("run_command", &successful));
    let payload = tool_finished_event_result("run_command", &successful).unwrap();
    assert_eq!(payload["stdout"], "Pokémon °C\r\n");
    assert_eq!(payload["exit_code"], 0);
    assert_eq!(payload["ui_output_truncated"], false);

    let failed = Ok(json!({
        "exit_code": 7,
        "success": false,
        "stdout": "",
        "stderr": "command failed",
        "output_truncated": false,
    }));
    assert!(!tool_finished_ok("run_command", &failed));
    assert!(tool_finished_detail("run_command", &failed).contains("exit code 7"));

    assert!(!tool_finished_ok(
        "execute_action_batch",
        &Ok(json!({"success": false, "verified_progress": false}))
    ));
    assert_eq!(
        tool_finished_detail(
            "execute_action_batch",
            &Ok(json!({
                "success": false,
                "verified_progress": false,
                "stop_reason": "repeated_no_progress"
            }))
        ),
        "repeated_no_progress"
    );
    assert!(tool_finished_ok(
        "execute_action_batch",
        &Ok(json!({"success": true, "verified_progress": true}))
    ));

    let long = Ok(json!({
        "exit_code": 0,
        "success": true,
        "stdout": "x".repeat(LIVE_COMMAND_STREAM_CHARS + 1),
        "stderr": "",
        "output_truncated": false,
    }));
    let bounded = tool_finished_event_result("run_command", &long).unwrap();
    assert_eq!(
        bounded["stdout"].as_str().unwrap().chars().count(),
        LIVE_COMMAND_STREAM_CHARS + 1
    );
    assert!(bounded["stdout"].as_str().unwrap().ends_with('…'));
    assert_eq!(bounded["ui_output_truncated"], true);

    let created = Ok(json!({"path": "C:\\workspace\\new.rs", "sha256": "abc"}));
    let created_payload = tool_finished_event_result("write_file", &created).unwrap();
    assert_eq!(created_payload["operation"], "created");
    assert_eq!(created_payload["path"], "C:\\workspace\\new.rs");

    assert!(tool_finished_event_result("capture_screen", &successful).is_none());
}

#[test]
fn unchanged_command_artifact_does_not_reset_verification() {
    let artifact = json!({"path": "C:\\work\\result.xlsx", "sha256": "abc"});
    let result = json!({"success": true, "artifacts": [artifact]});
    let mut hashes = BTreeMap::new();
    assert!(tool_changed_artifact("run_command", &result, &mut hashes));
    assert!(!tool_changed_artifact("run_command", &result, &mut hashes));
    let changed = json!({
        "success": true,
        "artifacts": [{"path": "C:\\work\\result.xlsx", "sha256": "def"}]
    });
    assert!(tool_changed_artifact("run_command", &changed, &mut hashes));
}

#[test]
fn model_tool_projection_bounds_large_generic_and_command_results() {
    let generic = project_tool_result_for_model(
        "future_tool",
        json!({"payload": "x".repeat(MODEL_TOOL_RESULT_CHARS + 10_000)}),
    );
    assert_eq!(
        generic.pointer("/model_projection/truncated"),
        Some(&Value::Bool(true))
    );
    assert!(
        generic
            .to_string()
            .chars()
            .count()
            .lt(&(MODEL_TOOL_RESULT_CHARS + 2_000))
    );

    let command = project_tool_result_for_model(
        "run_command",
        json!({"stdout": "x".repeat(LIVE_COMMAND_STREAM_CHARS + 100), "stderr": ""}),
    );
    assert_eq!(command["stdout_model_truncated"], true);
    assert!(
        command["stdout"]
            .as_str()
            .is_some_and(|text| text.chars().count() <= COMMAND_MODEL_OUTPUT_CHARS)
    );
}

#[test]
fn provider_context_overflow_patterns_are_recoverable() {
    let error = PokError::Provider("context length exceeded for this model".into());
    assert!(is_context_overflow_rejection(&error));
    let unrelated = PokError::Provider("authentication failed".into());
    assert!(!is_context_overflow_rejection(&unrelated));
}

#[test]
fn serialized_tool_finished_event_includes_command_result() {
    let event = AgentEvent::ToolFinished {
        call_id: "call-1".into(),
        name: "run_command".into(),
        ok: true,
        detail: "completed".into(),
        result: Some(json!({"stdout": "True\r\n", "exit_code": 0})),
    };
    let encoded = serde_json::to_value(event).unwrap();
    assert_eq!(encoded["type"], "tool_finished");
    assert_eq!(encoded["call_id"], "call-1");
    assert_eq!(encoded["result"]["stdout"], "True\r\n");
}

#[test]
fn capability_matching_accepts_lm_studio_variants() {
    assert!(model_id_matches(
        "google/gemma-4-26b-a4b",
        "google/gemma-4-26b-a4b@q4_k_m"
    ));
    assert!(!model_id_matches("model-a", "model-b"));
}

#[test]
fn capability_recall_requires_a_specific_match() {
    let movie = capability_terms("open this movie file");
    assert_eq!(
        capability_match_score(&movie, "Read and extract text from PDF documents"),
        None
    );
    let pdf = capability_terms("extract the tables from this PDF");
    assert!(capability_match_score(&pdf, "PDF table extraction utility").is_some());
    let weather = capability_terms("show today's weather data");
    assert_eq!(
        capability_match_score(&weather, "Retrieve current stock market data"),
        None
    );
}

#[test]
fn verified_workflow_procedure_uses_placeholders_and_sanitized_trace() {
    let step = workflow_step(
        "simulate_input",
        &json!({"text": "private payload"}),
        &json!({
            "action": {"kind": "type_text", "text": "private payload"},
            "submission": {"status": "verified"},
        }),
    )
    .unwrap();
    assert_eq!(step.pointer("/action/text"), Some(&json!("<USER_TEXT>")));
    let record = VerifiedSubmission {
        normalized_text: "private payload".into(),
        destination: "discord.exe|@private-recipient - discord".into(),
        destination_label: "@private-recipient - Discord".into(),
        evidence: "Sender, private payload".into(),
    };
    let workflow = vec![
        workflow_step(
            "capture_screen",
            &json!({}),
            &json!({"target": {"scope": "window", "app": "Discord.exe"}}),
        )
        .unwrap(),
        step,
    ];
    let qualification = qualify_workflow(&workflow, std::slice::from_ref(&record));
    let dir = tempfile::tempdir().unwrap();
    let memory = crate::memory::MemoryStore::open(dir.path()).unwrap();
    let learned = learn_verified_procedures(
        &memory,
        "send a message in the current Discord channel",
        &workflow,
        &qualification,
    )
    .unwrap();
    assert_eq!(learned.len(), 1);
    let encoded = serde_json::to_string(&learned[0].0).unwrap();
    assert!(!encoded.contains("private payload"));
    assert!(!encoded.contains("private-recipient"));
    assert_eq!(learned[0].0.applications, vec!["Discord.exe"]);
}

#[test]
fn general_windows_flow_qualifies_from_changed_state_and_recapture() {
    let workflow = vec![
        workflow_step(
            "capture_screen",
            &json!({}),
            &json!({"target": {"scope": "window", "app": "explorer.exe"}}),
        )
        .unwrap(),
        workflow_step(
            "click_target",
            &json!({"target_id": "42"}),
            &json!({
                "executed": true,
                "action": {"kind": "click_target", "target_id": "42", "label": "Open"},
                "focus": {"app": "explorer.exe", "title": "Private Folder"},
                "state_change": {"added_text": ["Folder opened"]},
            }),
        )
        .unwrap(),
        workflow_step(
            "capture_screen",
            &json!({}),
            &json!({"target": {"scope": "window", "app": "explorer.exe"}}),
        )
        .unwrap(),
    ];
    let qualification = qualify_workflow(&workflow, &[]);
    assert!(qualification.eligible);
    assert_eq!(qualification.evidence, "post_input_uia_state_change");
    let dir = tempfile::tempdir().unwrap();
    let memory = crate::memory::MemoryStore::open(dir.path()).unwrap();
    let learned = learn_verified_procedures(
        &memory,
        "open the requested folder",
        &workflow,
        &qualification,
    )
    .unwrap();
    assert_eq!(learned.len(), 1);
    let encoded = serde_json::to_string(&learned[0].0).unwrap();
    assert!(encoded.contains("explorer.exe"));
    assert!(!encoded.contains("Private Folder"));
    assert!(!encoded.contains("target_id"));
}

#[test]
fn unchanged_desktop_input_does_not_become_a_procedure() {
    let workflow = vec![workflow_step(
        "click_target",
        &json!({"target_id": "4"}),
        &json!({
            "executed": true,
            "action": {"kind": "click_target", "target_id": "4", "label": "Button"},
            "focus": {"app": "settings.exe"},
            "state_change": {"added_text": [], "removed_control_count": 0, "focus_changed": false},
        }),
    )
    .unwrap()];
    assert!(!qualify_workflow(&workflow, &[]).eligible);
}

#[test]
fn learned_hybrid_workflow_filters_failures_and_does_not_learn_incidental_command() {
    let dir = tempfile::tempdir().unwrap();
    let memory = crate::memory::MemoryStore::open(dir.path()).unwrap();
    let workflow = vec![
        workflow_step(
            "run_command",
            &json!({"command": "Start-Process notepad", "cwd": "."}),
            &json!({"success": true, "exit_code": 0}),
        )
        .unwrap(),
        workflow_step(
            "click_target",
            &json!({"target_id": "8"}),
            &json!({
                "executed": false,
                "action": {"kind": "click_target", "target_id": "8"},
                "_pok_continuity": {"outcome": "no_progress"},
                "focus": {"app": "Notepad.exe"}
            }),
        )
        .unwrap(),
        workflow_step(
            "type_text",
            &json!({"target_id": "27", "text": "private task text"}),
            &json!({
                "executed": true,
                "action": {"kind": "type_text", "text": "private task text"},
                "focus": {"app": "Notepad.exe"},
                "state_change": {"added_text": ["document updated"]}
            }),
        )
        .unwrap(),
    ];
    let qualification = qualify_workflow(&workflow, &[]);
    assert!(qualification.eligible);
    let learned = learn_verified_procedures(
        &memory,
        "open Notepad and type the requested text",
        &workflow,
        &qualification,
    )
    .unwrap();
    assert_eq!(learned.len(), 1, "only the hybrid workflow is learned");
    assert_eq!(learned[0].0.kind, ProcedureKind::Workflow);
    assert!(
        learned[0]
            .0
            .steps
            .iter()
            .any(|step| step.tool == "type_text")
    );
    assert!(
        learned[0]
            .0
            .steps
            .iter()
            .all(|step| step.tool != "click_target")
    );
    let encoded = serde_json::to_string(&learned[0].0).unwrap();
    assert!(!encoded.contains("private task text"));
}

#[test]
fn completed_command_is_sanitized_deduplicated_and_retrievable_after_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let workflow = vec![
        workflow_step(
            "run_command",
            &json!({
                "command": "python helper.py --api-key=private-value",
                "cwd": "C:\\Users\\alice\\project"
            }),
            &json!({"success": true, "exit_code": 0}),
        )
        .unwrap(),
    ];
    let command = workflow[0]
        .pointer("/arguments/command")
        .and_then(Value::as_str)
        .unwrap();
    assert!(command.contains("<REDACTED>"));
    assert!(!command.contains("private-value"));
    assert_eq!(
        workflow[0].pointer("/arguments/cwd"),
        Some(&json!("<CURRENT_WORKSPACE>"))
    );

    let memory = crate::memory::MemoryStore::open(dir.path()).unwrap();
    let qualification = qualify_workflow(&workflow, &[]);
    let learned = learn_verified_procedures(
        &memory,
        "run the reusable helper report",
        &workflow,
        &qualification,
    )
    .unwrap();
    assert!(learned.is_empty(), "redacted commands must not be learned");
    drop(memory);
    let reopened = crate::memory::MemoryStore::open(dir.path()).unwrap();
    assert!(
        reopened
            .retrieve_context("reusable helper report")
            .unwrap()
            .commands
            .is_empty()
    );
}

#[test]
fn successful_one_off_command_is_not_learned() {
    let dir = tempfile::tempdir().unwrap();
    let memory = crate::memory::MemoryStore::open(dir.path()).unwrap();
    let workflow = vec![
        workflow_step(
            "run_command",
            &json!({"command": "python helper.py --format json", "cwd": "."}),
            &json!({"success": true, "exit_code": 0}),
        )
        .unwrap(),
    ];
    assert!(memory.list_procedures(None, 10).unwrap().is_empty());
    let qualification = qualify_workflow(&workflow, &[]);
    let learned = learn_verified_procedures(
        &memory,
        "generate the workspace inventory report",
        &workflow,
        &qualification,
    )
    .unwrap();
    assert!(learned.is_empty());
    assert!(memory.list_procedures(None, 10).unwrap().is_empty());
}

#[test]
fn bounded_context_keeps_one_image_and_compacts_old_tools() {
    let mut messages = vec![
        BrainMessage::text("system", "system"),
        BrainMessage::text("user", "perform a desktop task"),
    ];
    for index in 0..20 {
        messages.push(BrainMessage {
            role: "assistant".into(),
            content: Vec::new(),
            origin: MessageOrigin::Assistant,
            tool_call_id: None,
            tool_calls: vec![CompletedToolCall {
                id: format!("call-{index}"),
                name: "capture_screen".into(),
                arguments: json!({}),
            }],
        });
        messages.push(BrainMessage {
            role: "tool".into(),
            content: vec![MessageContent::Text {
                text: "x".repeat(20_000),
            }],
            origin: MessageOrigin::ToolResult,
            tool_call_id: Some(format!("call-{index}")),
            tool_calls: Vec::new(),
        });
        messages.push(BrainMessage {
            role: "user".into(),
            content: vec![
                MessageContent::Text {
                    text: "screenshot".into(),
                },
                MessageContent::ImagePng {
                    base64: "AA==".into(),
                },
            ],
            origin: MessageOrigin::ToolImage,
            tool_call_id: None,
            tool_calls: Vec::new(),
        });
    }
    let (bounded, estimate, actions) = bounded_context(&messages, 1, 16_000, 4_000, 1.0);
    assert_eq!(count_images(&bounded), 1);
    assert!(estimate <= 16_000, "estimate was {estimate}");
    assert!(!actions.is_empty());
}

#[test]
fn bounded_context_caps_one_nested_visual_message_to_current_pair() {
    let mut content = vec![MessageContent::Text {
        text: "batch screenshots".into(),
    }];
    content.extend((0..18).map(|index| MessageContent::ImagePng {
        base64: format!("image-{index}"),
    }));
    let messages = vec![
        BrainMessage::text("system", "system"),
        BrainMessage::text("user", "perform a desktop task"),
        BrainMessage {
            role: "user".into(),
            content,
            origin: MessageOrigin::ToolImage,
            tool_call_id: None,
            tool_calls: Vec::new(),
        },
    ];

    let (bounded, _, actions) = bounded_context(&messages, 1, 16_000, 1_000, 1.0);
    let rendered = serde_json::to_string(&bounded).unwrap();
    assert_eq!(count_images(&bounded), CURRENT_VISUAL_PAIR_IMAGES);
    assert!(rendered.contains("image-16"));
    assert!(rendered.contains("image-17"));
    assert!(!rendered.contains("image-15"));
    assert!(actions.iter().any(|action| action == "removed_images:16"));
}

#[test]
fn image_byte_budget_prunes_oldest_images_first() {
    let mut messages = vec![BrainMessage {
        role: "user".into(),
        content: vec![
            MessageContent::ImagePng {
                base64: "QUFB".into(),
            },
            MessageContent::ImagePng {
                base64: "QkJC".into(),
            },
            MessageContent::ImagePng {
                base64: "Q0ND".into(),
            },
        ],
        origin: MessageOrigin::ToolImage,
        tool_call_id: None,
        tool_calls: Vec::new(),
    }];

    assert_eq!(image_payload_stats(&messages).decoded_bytes, 9);
    assert_eq!(prune_images_to_decoded_budget(&mut messages, 3), 2);
    assert_eq!(image_payload_stats(&messages).decoded_bytes, 3);
    assert_eq!(count_images(&messages), 1);
    assert!(serde_json::to_string(&messages).unwrap().contains("Q0ND"));
}

#[test]
fn image_payload_limit_errors_are_recoverable_without_disabling_vision() {
    assert!(is_image_payload_too_large(&PokError::Provider(
        "HTTP 413 Payload Too Large: Downloaded image content cannot exceed 30MB".into()
    )));
    assert!(!is_image_payload_too_large(&PokError::Provider(
        "HTTP 429 rate limited".into()
    )));
}

#[test]
fn bounded_context_keeps_only_the_newest_full_grounded_observation() {
    let observation = |id: &str, label: &str| {
        json!({
            "observation_id": id,
            "executed": true,
            "targets": [{"id": "7", "label": label}],
            "ordered_content": [{"text": label}],
            "action": {"kind": "click_target", "target_id": "7"},
            "state_change": {"added_text": [label]}
        })
        .to_string()
    };
    let mut messages = vec![
        BrainMessage::text("system", "system"),
        BrainMessage::text("user", "perform a desktop task"),
    ];
    for (index, label) in ["stale-private-label", "current-label"].iter().enumerate() {
        messages.push(BrainMessage {
            role: "assistant".into(),
            content: Vec::new(),
            origin: MessageOrigin::Assistant,
            tool_call_id: None,
            tool_calls: vec![CompletedToolCall {
                id: format!("call-{index}"),
                name: "click_target".into(),
                arguments: json!({"target_id": "7"}),
            }],
        });
        messages.push(BrainMessage {
            role: "tool".into(),
            content: vec![MessageContent::Text {
                text: observation(&format!("obs-{index}"), label),
            }],
            origin: MessageOrigin::ToolResult,
            tool_call_id: Some(format!("call-{index}")),
            tool_calls: Vec::new(),
        });
    }

    let (bounded, _, actions) = bounded_context(&messages, 1, 16_000, 1_000, 1.0);
    let rendered = serde_json::to_string(&bounded).unwrap();
    assert!(rendered.contains("current-label"));
    assert!(rendered.contains("Superseded observation compacted"));
    assert!(!rendered.contains("\"targets\":[{\"id\":\"7\",\"label\":\"stale-private-label\"}]"));
    assert!(
        actions
            .iter()
            .any(|action| action == "compacted_superseded_observations:1")
    );
}

#[test]
fn long_runs_keep_task_recent_cycles_and_continuity_ledger() {
    let mut messages = vec![
        BrainMessage::text("system", "system"),
        BrainMessage::text("user", "open the requested website and finish the task"),
    ];
    for index in 0..128 {
        messages.push(BrainMessage {
            role: "assistant".into(),
            content: Vec::new(),
            origin: MessageOrigin::Assistant,
            tool_call_id: None,
            tool_calls: vec![CompletedToolCall {
                id: format!("call-{index}"),
                name: "simulate_input".into(),
                arguments: json!({"kind": "click", "x": index, "y": 100}),
            }],
        });
        messages.push(BrainMessage {
            role: "tool".into(),
            content: vec![MessageContent::Text {
                text: json!({"executed": true, "step": index}).to_string(),
            }],
            origin: MessageOrigin::ToolResult,
            tool_call_id: Some(format!("call-{index}")),
            tool_calls: Vec::new(),
        });
    }

    // A tight target forces the over-budget reduction path.
    let (bounded, estimate, actions) = bounded_context(&messages, 1, 3_000, 4_000, 1.0);
    let rendered = serde_json::to_string(&bounded).unwrap();
    assert!(rendered.contains("open the requested website"));
    assert!(rendered.contains("AUTOMATIC CONTEXT COMPACTION"));
    assert!(rendered.contains("call-127"));
    assert!(!rendered.contains("\"id\":\"call-0\""));
    assert!(estimate <= 16_000, "estimate was {estimate}");
    assert!(
        actions
            .iter()
            .any(|action| action.starts_with("continuity_ledger:"))
    );
}

fn read_file_cycle(index: usize, content: &str) -> [BrainMessage; 2] {
    let path = format!("src/file-{index}.rs");
    [
        BrainMessage {
            role: "assistant".into(),
            content: Vec::new(),
            origin: MessageOrigin::Assistant,
            tool_call_id: None,
            tool_calls: vec![CompletedToolCall {
                id: format!("call-{index}"),
                name: "read_file".into(),
                arguments: json!({"path": path}),
            }],
        },
        BrainMessage {
            role: "tool".into(),
            content: vec![MessageContent::Text {
                text: json!({
                    "path": path,
                    "sha256": format!("hash-{index}"),
                    "total_lines": 1,
                    "start_line": 1,
                    "end_line": 1,
                    "content": content,
                    "truncated": false,
                })
                .to_string(),
            }],
            origin: MessageOrigin::ToolResult,
            tool_call_id: Some(format!("call-{index}")),
            tool_calls: Vec::new(),
        },
    ]
}

#[test]
fn bounded_context_keeps_many_tool_results_while_they_fit_the_budget() {
    let mut messages = vec![
        BrainMessage::text("system", "system"),
        BrainMessage::text("user", "review the module files and report the bug"),
    ];
    for index in 0..30 {
        messages.extend(read_file_cycle(
            index,
            &format!("UNIQUE_CONTENT_{index}_{}", "x".repeat(120)),
        ));
    }

    let (bounded, _, actions) = bounded_context(&messages, 1, 16_000, 4_000, 1.0);
    let rendered = serde_json::to_string(&bounded).unwrap();
    // All 30 recent results fit, so none is thrown away; the old fixed cap of
    // six made the model re-read the same files every turn.
    for index in 0..30 {
        assert!(
            rendered.contains(&format!("UNIQUE_CONTENT_{index}_")),
            "tool result {index} was compacted while under budget"
        );
    }
    assert!(
        !actions
            .iter()
            .any(|action| action.starts_with("compacted_tool_results")
                || action.starts_with("continuity_ledger:"))
    );
}

#[test]
fn bounded_context_compacts_oldest_tool_results_only_over_budget() {
    let mut messages = vec![
        BrainMessage::text("system", "system"),
        BrainMessage::text("user", "review the module files and report the bug"),
    ];
    for index in 0..30 {
        messages.extend(read_file_cycle(
            index,
            &format!("UNIQUE_CONTENT_{index}_{}", "x".repeat(1_000)),
        ));
    }

    let (bounded, _, actions) = bounded_context(&messages, 1, 5_000, 4_000, 1.0);
    let rendered = serde_json::to_string(&bounded).unwrap();
    assert!(
        actions
            .iter()
            .any(|action| action.starts_with("compacted_tool_results"))
    );
    // Over budget the oldest are reduced first; the survivors are the newest
    // contiguous run, and the floor keeps a real working set instead of the
    // old fixed six results.
    let survivors = (0..30)
        .filter(|index| rendered.contains(&format!("UNIQUE_CONTENT_{index}_")))
        .collect::<Vec<_>>();
    // The budget wins when even the floor does not fit, but the old fixed cap
    // of six is gone: a real working set survives.
    assert!(
        survivors.len() > 6,
        "only {} results survived",
        survivors.len()
    );
    assert!(survivors.len() < 30);
    assert_eq!(survivors.first(), Some(&(30 - survivors.len())));
    assert_eq!(survivors.last(), Some(&29));
}

#[test]
fn duplicate_read_stub_reuses_a_retained_unchanged_page() {
    let path = "src/lib.rs";
    let content = "pub fn example() {}";
    let call = CompletedToolCall {
        id: "call-new".into(),
        name: "read_file".into(),
        arguments: json!({"path": path}),
    };
    let result = || {
        Ok(json!({
            "path": path,
            "sha256": "abc123",
            "total_lines": 1,
            "start_line": 1,
            "end_line": 1,
            "content": content,
            "truncated": false,
        }))
    };
    let mut messages = vec![BrainMessage::text("user", "read the file")];
    messages.extend(read_file_cycle(0, content));
    // The helper matches on the exact path, hash, and page.
    messages[2].content = vec![MessageContent::Text {
        text: json!({
            "path": path,
            "sha256": "abc123",
            "total_lines": 1,
            "start_line": 1,
            "end_line": 1,
            "content": content,
            "truncated": false,
        })
        .to_string(),
    }];

    let retained = retained_tool_result_ids(&messages);
    let stub = duplicate_read_stub(&call, &result(), &messages, &retained)
        .expect("retained read is reused");
    assert_eq!(stub.get("duplicate_read"), Some(&Value::Bool(true)));
    assert_eq!(stub.get("content_omitted"), Some(&Value::Bool(true)));
    assert!(stub.get("content").is_none());

    // A changed file is a new read, not a duplicate.
    let changed = Ok(json!({
        "path": path,
        "sha256": "def456",
        "total_lines": 1,
        "start_line": 1,
        "end_line": 1,
        "content": "pub fn example() { changed(); }",
        "truncated": false,
    }));
    assert!(duplicate_read_stub(&call, &changed, &messages, &retained).is_none());

    // A result the projection did not carry verbatim is read again.
    assert!(
        duplicate_read_stub(&call, &result(), &messages, &BTreeSet::new()).is_none(),
        "only results the model can still see may be reused"
    );
}
#[test]
fn system_temporal_anchor_contains_local_iso_date_and_zone() {
    let anchor = current_temporal_anchor();
    assert_eq!(anchor.local_date.len(), 10);
    assert!(anchor.context.contains(&anchor.local_date));
    assert!(anchor.context.contains("Today's local date"));
    assert!(anchor.context.contains("UTC"));
    assert!(anchor.context.contains("verify freshness"));
}

#[test]
fn managed_web_search_recognizes_news_without_an_explicit_url() {
    assert!(requests_managed_web_search(
        "is there any recent headline news that I should know about"
    ));
    assert!(requests_managed_web_search(
        "look up the latest Rust release"
    ));
    assert!(!requests_managed_web_search("what time is it right now"));
    assert!(!requests_managed_web_search(
        "check the current general chat channel in the active application"
    ));
}

#[test]
fn authorization_context_changes_without_losing_other_system_context() {
    let mut prompt = format!(
        "prefix\n{}\nprofile remains",
        authorization_context(&PolicyMode::Interactive)
    );
    assert!(replace_authorization_context(
        &mut prompt,
        authorization_context(&PolicyMode::Autonomous)
    ));
    assert!(prompt.contains("Autonomous mode is active"));
    assert!(!prompt.contains("Interactive mode is active"));
    assert!(prompt.contains("prefix"));
    assert!(prompt.contains("profile remains"));
}

#[test]
fn context_compaction_preserves_midnight_temporal_update() {
    let mut messages = vec![
        BrainMessage::text("system", "cached date is 2026-07-18"),
        BrainMessage::text("user", "continue overnight"),
        BrainMessage::text(
            "system",
            "<temporal_context>Today's local date is 2026-07-19.</temporal_context>",
        ),
    ];
    for index in 0..25 {
        messages.push(BrainMessage {
            role: "assistant".into(),
            content: Vec::new(),
            origin: MessageOrigin::Assistant,
            tool_call_id: None,
            tool_calls: vec![CompletedToolCall {
                id: format!("clock-call-{index}"),
                name: "get_current_time".into(),
                arguments: json!({}),
            }],
        });
        messages.push(BrainMessage {
            role: "tool".into(),
            content: vec![MessageContent::Text {
                text: json!({"step": index}).to_string(),
            }],
            origin: MessageOrigin::ToolResult,
            tool_call_id: Some(format!("clock-call-{index}")),
            tool_calls: Vec::new(),
        });
    }
    let mut compactions = Vec::new();
    let compacted = compact_old_tool_cycles(&messages, &mut compactions);
    let rendered = serde_json::to_string(&compacted).unwrap();
    assert!(rendered.contains("Today's local date is 2026-07-19"));
    assert!(
        compactions
            .iter()
            .any(|item| item.starts_with("continuity_ledger:"))
    );
}

use crate::memory::MemoryStore;
use async_trait::async_trait;

struct MockApproval;
#[async_trait]
impl crate::policy::ApprovalHandler for MockApproval {
    async fn approve(
        &self,
        _tool: &str,
        _arguments: &serde_json::Value,
        _reason: &str,
    ) -> Result<crate::policy::ApprovalDecision> {
        Ok(crate::policy::ApprovalDecision::AllowOnce)
    }
}

struct MockBrain {
    summary_response: String,
    calls: AtomicU64,
}

#[async_trait]
impl Brain for MockBrain {
    async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec!["mock".into()])
    }
    fn stream(&self, request: BrainRequest) -> crate::brain::BrainStream {
        assert!(request.max_tokens.is_none());
        if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
            return Box::pin(futures::stream::iter(vec![
                Ok(BrainEvent::ReasoningDelta {
                    text: "Thinking through the complete history.".into(),
                }),
                Ok(BrainEvent::Finished {
                    reason: Some("stop".into()),
                }),
            ]));
        }
        let text = self.summary_response.clone();
        let stream = futures::stream::iter(vec![
            Ok(BrainEvent::TextDelta { text }),
            Ok(BrainEvent::Finished {
                reason: Some("stop".into()),
            }),
        ]);
        Box::pin(stream)
    }
}

struct InterruptibleBrain {
    calls: AtomicU64,
}

struct ReasoningSummaryBrain;

#[async_trait]
impl Brain for ReasoningSummaryBrain {
    async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec!["reasoning-summary".into()])
    }

    fn stream(&self, _request: BrainRequest) -> crate::brain::BrainStream {
        Box::pin(futures::stream::iter(vec![
            Ok(BrainEvent::ReasoningDelta {
                text: "Summary emitted through reasoning.".into(),
            }),
            Ok(BrainEvent::Finished {
                reason: Some("stop".into()),
            }),
        ]))
    }
}

#[tokio::test]
async fn summary_does_not_store_reasoning_when_visible_text_is_empty() {
    let attempt = generate_summary(
        &ReasoningSummaryBrain,
        "reasoning-summary",
        "summarize this",
        None,
        None,
        &CancellationToken::new(),
    )
    .await
    .unwrap();
    assert!(attempt.text.is_empty());
    assert!(attempt.reasoning_chars > 0);
    assert!(!valid_history_summary(&attempt.text));
}

#[async_trait]
impl Brain for InterruptibleBrain {
    async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec!["mock-model".into()])
    }

    fn stream(&self, _request: BrainRequest) -> crate::brain::BrainStream {
        let attempt = self.calls.fetch_add(1, Ordering::SeqCst);
        if attempt == 0 {
            Box::pin(async_stream::stream! {
                tokio::time::sleep(Duration::from_secs(30)).await;
                yield Ok(BrainEvent::TextDelta { text: "stale output".into() });
            })
        } else {
            Box::pin(futures::stream::iter(vec![
                Ok(BrainEvent::TextDelta {
                    text: "Task continued after resume.".into(),
                }),
                Ok(BrainEvent::Finished {
                    reason: Some("stop".into()),
                }),
            ]))
        }
    }
}

#[tokio::test]
async fn pause_interrupts_model_stream_and_resume_starts_fresh_turn() {
    let temp_dir = tempfile::tempdir().unwrap();
    let pause = Arc::new(crate::pause::PauseController::default());
    let context = ToolContext {
        session_id: Uuid::new_v4(),
        workspace: temp_dir.path().to_path_buf(),
        data_dir: temp_dir.path().to_path_buf(),
        artifact_dir: temp_dir.path().join("artifacts"),
        policy: crate::policy::Policy::interactive(),
        approvals: Arc::new(MockApproval),
        platform: Arc::new(crate::platform::MockDesktop::default()),
        memory: MemoryStore::open(temp_dir.path().join("memory")).unwrap(),
        session_archive: crate::session_archive::SessionArchive::open(
            &temp_dir.path().join("artifacts"),
        )
        .unwrap(),
        cancellation: CancellationToken::new(),
        pause: pause.clone(),
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
        context_budget: Arc::new(Mutex::new(crate::context::ContextBudget::new(
            "test".into(),
            "test".into(),
            8_000,
            80,
            crate::context::ContextSource::FallbackUnknown,
        ))),
        visual_history_limit: 1,
        uia_element_limit: 100,
        desktop_enrichment_timeout_ms: 2_000,
        desktop_deep_enrichment_timeout_ms: 10_000,
        model_target_limit: 5,
        fusion_iou_threshold: 0.1,
        ocr_containment_threshold: 0.6,
        annotate_targets: false,
        approval_cache: Default::default(),
        user_guidance_queue: Default::default(),
        questions: None,
        command_manager: Arc::new(crate::commands::CommandManager::new(
            temp_dir.path().join("artifacts"),
        )),
        current_tool_call_id: Default::default(),
    };
    let brain = Arc::new(InterruptibleBrain {
        calls: AtomicU64::new(0),
    });
    let session = Session::new(
        brain.clone(),
        Arc::new(ToolRegistry::default()),
        context,
        "mock-model".into(),
        4,
    )
    .unwrap();

    let run = tokio::spawn(async move {
        let mut session = session;
        session.run("continue the desktop task").await
    });
    tokio::time::timeout(Duration::from_secs(2), async {
        while brain.calls.load(Ordering::SeqCst) == 0 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    assert!(pause.request_pause());
    tokio::time::timeout(Duration::from_secs(2), async {
        while pause.state() != PauseState::Paused {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    assert!(pause.resume(Some("opened another application".into())));
    let summary = tokio::time::timeout(Duration::from_secs(2), run)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(summary.answer, "Task continued after resume.");
    assert_eq!(brain.calls.load(Ordering::SeqCst), 2);
    let trace = std::fs::read_to_string(summary.artifact_dir.join("trace.jsonl")).unwrap();
    assert!(trace.contains("\"kind\":\"model_turn_interrupted\""));
    assert!(trace.contains("\"kind\":\"agent_resumed\""));
    assert!(trace.contains("opened another application"));
}

#[tokio::test]
async fn test_compress_history_if_needed() {
    let brain = Arc::new(MockBrain {
        summary_response: "Mocked history summary".into(),
        calls: AtomicU64::new(0),
    });
    let tools = Arc::new(ToolRegistry::default());
    let temp_dir = tempfile::tempdir().unwrap();
    let memory = MemoryStore::open(temp_dir.path().join("memory")).unwrap();
    let context = ToolContext {
        session_id: Uuid::new_v4(),
        workspace: temp_dir.path().to_path_buf(),
        data_dir: temp_dir.path().to_path_buf(),
        artifact_dir: temp_dir.path().join("artifacts"),
        policy: crate::policy::Policy::interactive(),
        approvals: Arc::new(MockApproval),
        platform: Arc::new(crate::platform::MockDesktop::default()),
        memory,
        session_archive: crate::session_archive::SessionArchive::open(
            &temp_dir.path().join("artifacts"),
        )
        .unwrap(),
        cancellation: CancellationToken::new(),
        pause: Arc::new(crate::pause::PauseController::default()),
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
        prompt_token_target: 4000,
        context_budget: Arc::new(Mutex::new(crate::context::ContextBudget::new(
            "test".into(),
            "test".into(),
            8_000,
            80,
            crate::context::ContextSource::FallbackUnknown,
        ))),
        visual_history_limit: 1,
        uia_element_limit: 100,
        desktop_enrichment_timeout_ms: 2_000,
        desktop_deep_enrichment_timeout_ms: 10_000,
        model_target_limit: 5,
        fusion_iou_threshold: 0.1,
        ocr_containment_threshold: 0.6,
        annotate_targets: false,
        approval_cache: Default::default(),
        user_guidance_queue: Default::default(),
        questions: None,
        command_manager: Arc::new(crate::commands::CommandManager::new(
            temp_dir.path().join("artifacts"),
        )),
        current_tool_call_id: Default::default(),
    };

    let observer = Arc::new(RecordingObserver::default());
    let mut session = Session::new(brain.clone(), tools, context, "mock-model".into(), 10)
        .unwrap()
        .with_observer(observer.clone());

    // Populate canonical history beyond the configured hard threshold. On
    // this deliberately tiny test window, output and safety reserves lower
    // the effective boundary below the raw 80% value.
    session
        .messages
        .push(BrainMessage::text("user", "Initial task")); // index 1
    session
        .messages
        .push(BrainMessage::text("assistant", "Sure, starting.")); // index 2
    session
        .messages
        .push(BrainMessage::text("user", "A".repeat(20_000))); // index 3
    for index in 0..10 {
        session.messages.push(BrainMessage::text(
            "assistant",
            format!("Working on step {index}"),
        ));
        session
            .messages
            .push(BrainMessage::text("tool", format!("Tool output {index}")));
    }
    session
        .messages
        .push(BrainMessage::text("user", "Last prompt")); // index 24

    // Pre-compaction check
    assert_eq!(session.messages.len(), 25); // system prompt + initial task + 23 messages
    let hard_threshold = session.context.context_budget.lock().compact_at_tokens;
    assert!(
        !session
            .compress_history_if_needed("Last prompt", hard_threshold.saturating_sub(1), false,)
            .await
            .unwrap()
    );

    assert!(
        session
            .compress_history_if_needed("Last prompt", hard_threshold.saturating_add(1), false,)
            .await
            .unwrap()
    );

    // Post-compaction check
    // The messages at index 3 and 4 should have been compressed into 1 system summary message.
    // We kept the system prompts, initial task (index 2), and the last 6 initial tail starts.
    // Let's verify that the total message count is reduced and the system summary is present.
    let rendered = serde_json::to_string(&session.messages).unwrap();
    assert!(rendered.contains("Mocked history summary"));
    assert!(!rendered.contains("Initial task"));
    assert!(rendered.contains("Last prompt"));
    assert_eq!(brain.calls.load(Ordering::SeqCst), 2);
    let trace = std::fs::read_to_string(session.context.artifact_dir.join("trace.jsonl")).unwrap();
    assert!(trace.contains("\"summary_source\":\"model_retry\""));
    assert!(!trace.contains("deterministic"));
    let last_user = session
        .messages
        .iter()
        .rfind(|message| message.origin == MessageOrigin::UserInput)
        .unwrap();
    assert!(format_transcript_for_summary(std::slice::from_ref(last_user)).contains("Last prompt"));
    assert_eq!(
        observer.events.lock().as_slice(),
        ["compression_started:1", "compression_completed:1"]
    );
}

#[test]
fn active_task_reminder_keeps_latest_request_authoritative() {
    let state = ActiveTaskState {
        root_request: "Check youtube.com for the latest Gamers Nexus video".into(),
        latest_guidance: vec!["Use Chrome".into()],
        current_step: "Open the channel videos page".into(),
        ..ActiveTaskState::default()
    };

    let reminder = active_task_reminder(
        &state,
        None,
        true,
        "- capture_screen [desktop; active]: Capture the selected window",
    );
    assert_eq!(reminder.origin, MessageOrigin::SystemReminder);
    let text = match reminder.content.first().expect("text reminder") {
        MessageContent::Text { text } => text,
        MessageContent::ImagePng { .. } => panic!("expected a text reminder"),
    };
    assert!(text.contains("ACTIVE USER REQUEST (authoritative)"));
    assert!(text.contains("youtube.com"));
    assert!(text.contains("screenshots"));
    assert!(!text.contains("Springfield"));
}

#[test]
fn dynamic_completion_policy_keeps_only_narrow_guards() {
    assert!(is_grounding_evidence_tool("capture_screen"));
    assert!(substantive_candidate_answer(
        "The captured weather results show a mostly clear evening with an overnight low near 85 degrees and an active excessive heat warning."
    ));
    assert_eq!(
        request_for_classification("  Also,   please tell me the time "),
        "tell me the time"
    );
    assert!(requests_live_information(
        "and how is the weather outside right now"
    ));
    assert!(is_clarification_request(
        "To continue, I need to know your location. Could you please provide your city, state, or ZIP code?"
    ));
    assert!(!is_clarification_request(
        "Tonight will be clear and hot with a low near 87°F. Would you like tomorrow's forecast?"
    ));
    assert!(is_clarification_request(
        "Which customer record should I select in Example POS?"
    ));
    assert_eq!(
        classify_attempt_failure(
            "write_file",
            &PokError::OutsideWorkspace(std::path::PathBuf::from("/outside/report.xlsx"))
        ),
        "policy_constraint"
    );
    assert!(is_unresolved_task_failure(
        "UPS was not showing up, so I could not complete the task."
    ));
    assert!(!is_unresolved_task_failure(
        "UPS is selected and the shipment is complete."
    ));
    assert!(requests_live_information("What is the current weather?"));
    assert!(!requests_live_information(
        "Explain why weather systems form."
    ));
    assert!(requests_current_screen_observation(
        "What do you see on this screen?"
    ));
    assert!(requests_current_screen_observation(
        "Which item is currently selected?"
    ));
    assert!(!requests_current_screen_observation(
        "Explain how list selection works."
    ));
    assert!(requests_artifact_outcome(
        "Create a presentation and save the file."
    ));
    assert!(requests_artifact_outcome(
        "Open Word, write a letter, and save it to my desktop."
    ));
    assert!(requests_artifact_outcome(
        "Create this and save it to Downloads."
    ));
    assert!(!requests_artifact_outcome(
        "Save time by explaining the keyboard shortcut."
    ));
    assert!(requests_visual_outcome(
        "Open the document so I can see the layout."
    ));
    assert!(!is_focused_visual_artifact_evidence(
        "observe_desktop",
        &json!({"screenshots": []})
    ));
    assert!(!is_focused_visual_artifact_evidence(
        "capture_screen",
        &json!({"target": {"scope": "monitor", "app": ""}})
    ));
    assert!(is_focused_visual_artifact_evidence(
        "capture_screen",
        &json!({"verification_state": "clean", "target": {"scope": "window", "app": "EXCEL.EXE"}})
    ));
    assert!(!is_focused_visual_artifact_evidence(
        "capture_screen",
        &json!({"verification_state": "issue_detected", "target": {"scope": "window", "app": "EXCEL.EXE"}})
    ));
    assert!(artifact_title_matches(
        std::path::Path::new("C:\\work\\Blood_Test_Results.xlsx"),
        "Blood Test Results.xlsx - Excel"
    ));
    assert_eq!(
        classify_attempt_failure(
            "capture_screen",
            &PokError::Tool("control unavailable".into())
        ),
        "grounding_gap"
    );
    assert!(is_safety_refusal(
        "I cannot enter a password on the secure desktop."
    ));
    assert!(!tool_finished_ok(
        "type_text",
        &Ok(json!({"executed": false, "recovery_required": true}))
    ));
}

#[test]
fn navigation_submission_is_not_a_confirmed_observed_url() {
    let state = verified_state(
        "browser_navigate",
        &json!({"destination": "https://outlook.live.com"}),
        &json!({
            "observation_id": "obs_1",
            "target": {"app": "msedge.exe", "title": "Outlook"}
        }),
        Some(&channel_state()),
    )
    .expect("verified state");
    assert_eq!(
        state.requested_destination.as_deref(),
        Some("https://outlook.live.com")
    );
    assert_eq!(state.observed_url, None);
    assert_eq!(state.url_provenance, None);
}

#[test]
fn confirmed_address_value_replaces_stale_url() {
    let state = verified_state(
        "capture_screen",
        &json!({}),
        &json!({
            "observation_id": "obs_2",
            "target": {"app": "msedge.exe", "title": "Mail - Outlook"},
            "observed_url": "https://outlook.live.com/mail/0/"
        }),
        Some(&channel_state()),
    )
    .expect("verified state");
    assert_eq!(
        state.observed_url.as_deref(),
        Some("https://outlook.live.com/mail/0/")
    );
    assert_eq!(state.url_provenance.as_deref(), Some("uia_value"));
}

#[test]
fn tool_groups_preserve_desktop_capability_and_expand_from_intent() {
    let desktop = inferred_tool_groups("Play some music in the active application");
    assert!(desktop.contains("desktop"));
    assert!(desktop.contains("control"));
    assert!(!desktop.contains("coding"));

    let browser = inferred_tool_groups(
        "Open the requested website in the isolated managed browser and inspect the page",
    );
    assert!(browser.contains("browser"));

    let coding = inferred_tool_groups("Fix the Rust file and run the tests");
    assert!(coding.contains("desktop"));
    assert!(coding.contains("coding"));
    assert!(coding.contains("system"));

    let network = inferred_tool_groups("Find the receiver on my network subnet");
    assert!(network.contains("system"));
    assert!(!network.contains("coding"));

    let skill = inferred_tool_groups("Create a reusable skill and remember it");
    assert!(skill.contains("memory"));
}

#[test]
fn repair_makes_parallel_tool_results_contiguous_and_ordered() {
    let call = |id: &str| CompletedToolCall {
        id: id.into(),
        name: "test_tool".into(),
        arguments: json!({}),
    };
    let result = |id: &str| BrainMessage {
        role: "tool".into(),
        content: vec![MessageContent::Text { text: id.into() }],
        origin: MessageOrigin::ToolResult,
        tool_call_id: Some(id.into()),
        tool_calls: Vec::new(),
    };
    let mut messages = vec![
        BrainMessage {
            role: "assistant".into(),
            content: Vec::new(),
            origin: MessageOrigin::Assistant,
            tool_call_id: None,
            tool_calls: vec![call("first"), call("second")],
        },
        result("first"),
        BrainMessage::text_with_origin(
            "user",
            "deferred harness reminder",
            MessageOrigin::SystemReminder,
        ),
        result("second"),
    ];

    assert!(!tool_call_result_pairs_valid(&messages));
    let report = repair_tool_call_result_pairs(&mut messages, "test");
    assert!(report.changed());
    assert!(tool_call_result_pairs_valid(&messages));
    assert_eq!(messages[1].tool_call_id.as_deref(), Some("first"));
    assert_eq!(messages[2].tool_call_id.as_deref(), Some("second"));
    assert_eq!(messages[3].origin, MessageOrigin::SystemReminder);
}

#[test]
fn repair_synthesizes_interrupted_results_and_removes_orphans() {
    let mut messages = vec![
        BrainMessage {
            role: "assistant".into(),
            content: Vec::new(),
            origin: MessageOrigin::Assistant,
            tool_call_id: None,
            tool_calls: vec![CompletedToolCall {
                id: "missing".into(),
                name: "test_tool".into(),
                arguments: json!({}),
            }],
        },
        BrainMessage {
            role: "tool".into(),
            content: vec![MessageContent::Text {
                text: "orphan".into(),
            }],
            origin: MessageOrigin::ToolResult,
            tool_call_id: Some("different".into()),
            tool_calls: Vec::new(),
        },
    ];

    let report = repair_tool_call_result_pairs(&mut messages, "cancelled");
    assert_eq!(report.synthetic_results_inserted, 1);
    assert_eq!(report.orphan_results_removed, 1);
    assert!(tool_call_result_pairs_valid(&messages));
    assert_eq!(messages[1].tool_call_id.as_deref(), Some("missing"));
}

#[test]
fn successful_read_only_jev_outcomes_are_observations_not_failures() {
    assert_eq!(
        decision_router_outcome_metric("observed", true),
        "decision_router_observed"
    );
    assert_eq!(
        decision_router_outcome_metric("failed", true),
        "decision_router_failed"
    );
    assert_eq!(
        decision_router_outcome_metric("observed", false),
        "decision_router_failed"
    );
}

#[test]
fn blocked_with_viable_candidates_is_refined_instead_of_handed_off() {
    assert!(refinable_decision_rejection(Some(
        "blocked_with_viable_candidates"
    )));
    assert!(refinable_decision_rejection(Some(
        "operation_below_confidence_threshold"
    )));
    assert!(!refinable_decision_rejection(Some("invalid_scores")));
}

#[test]
fn semantic_candidate_fingerprint_ignores_fresh_observation_ids() {
    let candidate = |observation_id: &str, target_id: &str| DecisionCandidate {
        id: format!("click_{target_id}"),
        tool: "click_target".into(),
        arguments: json!({
            "observation_id": observation_id,
            "target_id": target_id,
            "expected_label": "General channel",
            "button": "left"
        }),
        description: "Click the enabled tree item whose label is General channel".into(),
        kind: DecisionCandidateKind::Action,
        local_score: 1.0,
    };
    assert_eq!(
        decision_candidate_fingerprint(&candidate("obs:1", "42")),
        decision_candidate_fingerprint(&candidate("obs:2", "99"))
    );
    assert_ne!(
        decision_candidate_fingerprint(&candidate("obs:1", "42")),
        decision_candidate_fingerprint(&DecisionCandidate {
            arguments: json!({
                "observation_id": "obs:2",
                "target_id": "100",
                "expected_label": "Support channel",
                "button": "left"
            }),
            description: "Click the enabled tree item whose label is Support channel".into(),
            ..candidate("obs:2", "100")
        })
    );
}

#[test]
fn live_web_requests_do_not_require_desktop_orientation() {
    for request in [
        "How is the weather outside right now?",
        "What was the stock price after market close?",
        "Find today's exchange rate",
        "Show me the latest headlines",
    ] {
        assert!(requests_managed_web_search(request), "{request}");
        assert!(!requests_desktop_interaction(request), "{request}");
    }
    assert!(requests_desktop_interaction(
        "What is the person streaming in the current chat channel?"
    ));
    assert!(requests_desktop_action("Open the calculator"));
    assert!(!requests_managed_web_search("What time is it right now?"));
}

#[test]
fn bounded_json_uses_visible_text_then_reasoning_fallback() {
    assert_eq!(
        bounded_json_value("```json\n{\"candidate_id\":\"click_2\"}\n```", "")
            .and_then(|value| value["candidate_id"].as_str().map(str::to_owned)),
        Some("click_2".into())
    );
    assert_eq!(
        bounded_json_value("", "analysis then {\"refresh\":true}")
            .and_then(|value| value["refresh"].as_bool()),
        Some(true)
    );
    assert!(bounded_json_value("not json", "also not json").is_none());
}

#[test]
fn operation_and_target_confidence_are_independent() {
    let config = crate::config::DecisionRouterConfig::default();
    assert_eq!(config.min_operation_confidence, 0.60);
    assert_eq!(config.min_confidence, 0.75);
    let mut decision = crate::decision::DecisionResult {
        candidate_id: Some("click_2".into()),
        selected_probability: 0.87,
        confidence: Some(0.61),
        operation: Some("CLICK".into()),
        operation_probability: Some(0.74),
        target_probability: Some(0.87),
        operation_confidence: Some(0.61),
        target_confidence: Some(0.86),
        redacted_state_fields: 0,
        dropped_candidates: 0,
        model: config.model.clone(),
        probabilities: BTreeMap::from([("click_2".into(), 0.87)]),
        progress_probability: None,
        backend_metadata: None,
    };
    assert!(ordinary_decision_scores_eligible(&decision, &config));
    decision.target_confidence = Some(0.74);
    assert!(!ordinary_decision_scores_eligible(&decision, &config));
    decision.target_confidence = Some(0.86);
    decision.operation_confidence = Some(0.59);
    assert!(!ordinary_decision_scores_eligible(&decision, &config));
}

#[test]
fn laya_combined_gate_lets_a_confident_target_offset_a_clustering_band_operation() {
    let mut config = crate::config::DecisionRouterConfig::default();
    config.backend = crate::config::DecisionRouterBackend::Laya;
    assert_eq!(config.laya.min_operation_probability, 0.35);
    assert_eq!(config.laya.min_target_probability, 0.90);
    // Representative of the real clustering-band traffic that motivated
    // this change: operation_probability in the 0.44-0.57 band that used
    // to defer to the LLM 100% of the time, target already certain.
    let clustering_band_correct = crate::decision::DecisionResult {
        candidate_id: Some("list_visible_windows".into()),
        selected_probability: 1.0,
        confidence: Some(0.5),
        operation: Some("LIST_WINDOWS".into()),
        operation_probability: Some(0.4291),
        target_probability: Some(1.0),
        operation_confidence: Some(0.5),
        target_confidence: Some(1.0),
        redacted_state_fields: 0,
        dropped_candidates: 0,
        model: config.active_model().to_string(),
        probabilities: BTreeMap::from([("list_visible_windows".into(), 1.0)]),
        progress_probability: None,
        backend_metadata: None,
    };
    assert!(ordinary_decision_scores_eligible(
        &clustering_band_correct,
        &config
    ));

    // A wrong-candidate decision with a merely-moderate target (not
    // near-certain, as happens on genuine multi-way ties) must still be
    // rejected rather than promoted by the lowered operation floor alone.
    let mut ambiguous_wrong = clustering_band_correct.clone();
    ambiguous_wrong.target_probability = Some(0.60);
    assert!(!ordinary_decision_scores_eligible(
        &ambiguous_wrong,
        &config
    ));
}

#[test]
fn structural_bypass_fires_only_for_a_single_offered_action() {
    let action = DecisionCandidate {
        id: "list_visible_windows".into(),
        tool: "list_windows".into(),
        arguments: json!({}),
        description: "the only offered action".into(),
        kind: DecisionCandidateKind::Action,
        local_score: 0.7,
    };
    let done = DecisionCandidate {
        id: "jev_done".into(),
        tool: "__done__".into(),
        arguments: json!({}),
        description: "terminal".into(),
        kind: DecisionCandidateKind::Action,
        local_score: 0.0,
    };
    let blocked = DecisionCandidate {
        id: "jev_blocked".into(),
        tool: "__blocked__".into(),
        arguments: json!({}),
        description: "terminal".into(),
        kind: DecisionCandidateKind::Action,
        local_score: 0.0,
    };

    // One actionable candidate alongside terminal candidates: bypass.
    let candidates = vec![action.clone(), done.clone(), blocked.clone()];
    assert_eq!(
        structural_bypass_candidate(&candidates, 1),
        Some(action.clone())
    );

    // Two actionable candidates: no bypass, a real choice exists.
    let mut second_action = action.clone();
    second_action.id = "list_visible_windows_2".into();
    let candidates = vec![action, second_action, done, blocked];
    assert_eq!(structural_bypass_candidate(&candidates, 2), None);
}

#[test]
fn laya_uses_native_choice_probabilities_instead_of_jev_confidence_scale() {
    let mut config = crate::config::DecisionRouterConfig {
        enabled: true,
        backend: crate::config::DecisionRouterBackend::Laya,
        ..crate::config::DecisionRouterConfig::default()
    };
    config.laya.model = "laya".into();
    let mut decision = crate::decision::DecisionResult {
        candidate_id: Some("get_current_time".into()),
        selected_probability: 1.0,
        confidence: Some(0.0883),
        operation: Some("GET_CURRENT_TIME".into()),
        operation_probability: Some(0.5466),
        target_probability: Some(1.0),
        operation_confidence: Some(0.0883),
        target_confidence: Some(1.0),
        redacted_state_fields: 0,
        dropped_candidates: 0,
        model: "laya".into(),
        probabilities: BTreeMap::from([("get_current_time".into(), 1.0)]),
        progress_probability: Some(0.6142),
        backend_metadata: Some(json!({"backend": "laya", "device": "cuda"})),
    };
    assert!(ordinary_decision_scores_eligible(&decision, &config));

    decision.operation_probability = Some(config.min_selected_probability - 0.01);
    assert!(!ordinary_decision_scores_eligible(&decision, &config));
}

#[test]
fn repeated_uncertain_candidate_is_withheld_for_current_state() {
    let mut state = DecisionRefinementState {
        state_revision: 7,
        ..DecisionRefinementState::default()
    };
    assert!(!record_uncertain_candidate(&mut state, "same".into()));
    assert!(record_uncertain_candidate(&mut state, "same".into()));
    assert!(state.temporarily_withheld.contains("same"));
    assert!(!record_uncertain_candidate(&mut state, "alternate".into()));
    assert_eq!(state.state_revision, 7);
}

#[test]
fn completed_observation_is_not_offered_again_until_progress_resets_it() {
    let candidate = DecisionCandidate {
        id: "read_current_state".into(),
        tool: "read_state".into(),
        arguments: json!({}),
        description: "Read the structured state requested by the user".into(),
        kind: DecisionCandidateKind::Action,
        local_score: 1.0,
    };
    let fingerprint = decision_candidate_fingerprint(&candidate);
    let mut completed = HashSet::from([fingerprint]);
    assert!(!decision_candidate_available(
        &candidate,
        &HashSet::new(),
        &completed,
        &HashSet::new(),
    ));

    completed.clear();
    assert!(decision_candidate_available(
        &candidate,
        &HashSet::new(),
        &completed,
        &HashSet::new(),
    ));
}

struct FastStubRouter {
    satisfied_after: usize,
    target_probability: f64,
    checks: AtomicU64,
    decided_tools: parking_lot::Mutex<Vec<Vec<String>>>,
    pick_label: Option<&'static str>,
    condition_pick: Option<&'static str>,
    condition_probability: f64,
    condition_questions: AtomicU64,
    batched: AtomicU64,
}

#[async_trait]
impl DecisionRouter for FastStubRouter {
    async fn decide(&self, request: DecisionRequest) -> Result<crate::decision::DecisionResult> {
        // Production routers reject duplicate candidate IDs outright.
        let mut ids = HashSet::new();
        assert!(
            request
                .candidates
                .iter()
                .all(|candidate| ids.insert(candidate.id.clone())),
            "duplicate candidate IDs sent to the router"
        );
        self.decided_tools.lock().push(
            request
                .candidates
                .iter()
                .map(|candidate| candidate.tool.clone())
                .collect(),
        );
        let first = self
            .pick_label
            .and_then(|label| {
                request
                    .candidates
                    .iter()
                    .find(|candidate| candidate.description.contains(label))
            })
            .unwrap_or(&request.candidates[0])
            .id
            .clone();
        Ok(crate::decision::DecisionResult {
            candidate_id: Some(first.clone()),
            selected_probability: self.target_probability,
            confidence: Some(self.target_probability),
            operation: None,
            operation_probability: Some(1.0),
            target_probability: Some(self.target_probability),
            operation_confidence: Some(1.0),
            target_confidence: Some(self.target_probability),
            redacted_state_fields: 0,
            dropped_candidates: 0,
            model: "stub-router".into(),
            probabilities: BTreeMap::from([(first, self.target_probability)]),
            progress_probability: None,
            backend_metadata: None,
        })
    }

    async fn check_condition(
        &self,
        _goal: &str,
        _condition: &str,
        _state: &Value,
    ) -> Result<crate::decision::ConditionVerdict> {
        let index = self.checks.fetch_add(1, Ordering::SeqCst) as usize;
        let satisfied = index >= self.satisfied_after;
        Ok(crate::decision::ConditionVerdict {
            satisfied,
            answer: if satisfied { "yes" } else { "no" }.into(),
            probability: 0.9,
            model: "stub-router".into(),
        })
    }

    async fn decide_with_completion(
        &self,
        request: DecisionRequest,
        goal: &str,
        condition: &str,
        state: &Value,
    ) -> Result<(
        crate::decision::DecisionResult,
        crate::decision::ConditionVerdict,
    )> {
        self.batched.fetch_add(1, Ordering::SeqCst);
        let decision = self.decide(request).await?;
        // Counted as a batched call, not a separate completion check.
        let index = self.checks.load(Ordering::SeqCst) as usize;
        let _ = (goal, condition, state);
        let satisfied = index >= self.satisfied_after;
        Ok((
            decision,
            crate::decision::ConditionVerdict {
                satisfied,
                answer: if satisfied { "yes" } else { "no" }.into(),
                probability: 0.9,
                model: "stub-router".into(),
            },
        ))
    }

    async fn choose_condition(
        &self,
        _goal: &str,
        options: &[(String, String)],
        _state: &Value,
    ) -> Result<crate::decision::ConditionChoice> {
        self.condition_questions.fetch_add(1, Ordering::SeqCst);
        Ok(self.condition_answer(options))
    }

    async fn decide_fanout(
        &self,
        request: DecisionRequest,
        _goal: &str,
        completion: Option<(&str, &Value)>,
        condition: Option<(&[(String, String)], &Value)>,
    ) -> Result<crate::decision::FanoutAnswers> {
        self.batched.fetch_add(1, Ordering::SeqCst);
        let decision = self.decide(request).await?;
        // Counted as one fan-out request, not as separate questions.
        let satisfied = self.checks.load(Ordering::SeqCst) as usize >= self.satisfied_after;
        Ok(crate::decision::FanoutAnswers {
            decision,
            completion: completion.map(|_| crate::decision::ConditionVerdict {
                satisfied,
                answer: if satisfied { "yes" } else { "no" }.into(),
                probability: 0.9,
                model: "stub-router".into(),
            }),
            condition: condition.map(|(options, _)| self.condition_answer(options)),
        })
    }

    fn training_body(&self, request: DecisionRequest) -> Option<Value> {
        let criteria = request
            .candidates
            .iter()
            .map(|candidate| (candidate.id.clone(), json!(candidate.description)))
            .collect::<serde_json::Map<_, _>>();
        Some(
            json!({"model": "stub-router", "state": {"task": request.task},
            "questions": {"operation": {"type": "choice"}, "target_CLICK": {"type": "choice", "criteria": criteria}}}),
        )
    }

    fn completion_training_body(
        &self,
        goal: &str,
        condition: &str,
        _state: &Value,
    ) -> Option<Value> {
        Some(
            json!({"model": "stub-router", "state": {"goal": goal, "done_when": condition},
            "questions": {"satisfied": {"type": "choice"}}}),
        )
    }
}

impl FastStubRouter {
    fn condition_answer(&self, options: &[(String, String)]) -> crate::decision::ConditionChoice {
        let picked = self
            .condition_pick
            .and_then(|text| options.iter().find(|(_, option)| option.contains(text)))
            .map_or_else(|| "none".to_owned(), |(id, _)| id.clone());
        crate::decision::ConditionChoice {
            option_id: picked,
            probability: self.condition_probability,
            accepted: self.condition_probability >= 0.6,
            model: "stub-router".into(),
        }
    }
}

struct RecordingInputTool {
    name: &'static str,
    calls: Arc<parking_lot::Mutex<Vec<Value>>>,
    /// When set, input leaves the mock screen unchanged (a click that
    /// does nothing); otherwise each input adds visible text.
    static_screen: Arc<std::sync::atomic::AtomicBool>,
}

#[async_trait]
impl crate::tool::Tool for RecordingInputTool {
    fn name(&self) -> &'static str {
        self.name
    }
    fn description(&self) -> &'static str {
        "test input"
    }
    fn input_schema(&self) -> Value {
        json!({"type": "object"})
    }
    fn risk(&self) -> crate::policy::RiskClass {
        crate::policy::RiskClass::ReadOnly
    }
    async fn execute(&self, arguments: Value, context: &ToolContext) -> Result<Value> {
        let count = {
            let mut calls = self.calls.lock();
            calls.push(arguments);
            calls.len()
        };
        if !self.static_screen.load(Ordering::SeqCst)
            && let Some(observation) = context.latest_observation.lock().as_mut()
        {
            observation.ocr.push(crate::types::OcrBlock {
                text: format!("{} result {count}", self.name),
                bounds: crate::types::Rect {
                    x: 0,
                    y: 0,
                    width: 10,
                    height: 10,
                },
                confidence: None,
                selected: None,
                variant: None,
            });
        }
        Ok(json!({
            "state_change": {"added_text": ["changed"]},
            "verification": {"effect": true},
        }))
    }
}

fn fast_link(id: &str, name: &str) -> crate::types::InteractionTarget {
    crate::types::InteractionTarget {
        id: id.into(),
        name: name.into(),
        control_type: "Link".into(),
        bounds: crate::types::Rect {
            x: 0,
            y: 0,
            width: 40,
            height: 20,
        },
        source: crate::types::TargetSource::Uia,
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
    }
}

struct FastHarness {
    session: Session,
    router: Arc<FastStubRouter>,
    clicks: Arc<parking_lot::Mutex<Vec<Value>>>,
    scrolls: Arc<parking_lot::Mutex<Vec<Value>>>,
    batches: Arc<parking_lot::Mutex<Vec<Value>>>,
    static_screen: Arc<std::sync::atomic::AtomicBool>,
    _temp_dir: tempfile::TempDir,
}

fn fast_harness(
    router: FastStubRouter,
    targets: Vec<crate::types::InteractionTarget>,
    mode: crate::config::DecisionRouterMode,
) -> FastHarness {
    let temp_dir = tempfile::tempdir().unwrap();
    let clicks = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let scrolls = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let static_screen = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mut tools = ToolRegistry::default();
    tools.register(RecordingInputTool {
        name: "click_target",
        calls: clicks.clone(),
        static_screen: static_screen.clone(),
    });
    tools.register(RecordingInputTool {
        name: "scroll_view",
        calls: scrolls.clone(),
        static_screen: static_screen.clone(),
    });
    let batches = Arc::new(parking_lot::Mutex::new(Vec::new()));
    tools.register(RecordingInputTool {
        name: "execute_action_batch",
        calls: batches.clone(),
        static_screen: static_screen.clone(),
    });
    let context = ToolContext {
        session_id: Uuid::new_v4(),
        workspace: temp_dir.path().to_path_buf(),
        data_dir: temp_dir.path().to_path_buf(),
        artifact_dir: temp_dir.path().join("artifacts"),
        policy: crate::policy::Policy::interactive(),
        approvals: Arc::new(MockApproval),
        platform: Arc::new(crate::platform::MockDesktop::default()),
        memory: MemoryStore::open(temp_dir.path().join("memory")).unwrap(),
        session_archive: crate::session_archive::SessionArchive::open(
            &temp_dir.path().join("artifacts"),
        )
        .unwrap(),
        cancellation: CancellationToken::new(),
        pause: Arc::new(crate::pause::PauseController::default()),
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
        context_budget: Arc::new(Mutex::new(crate::context::ContextBudget::new(
            "test".into(),
            "test".into(),
            8_000,
            80,
            crate::context::ContextSource::FallbackUnknown,
        ))),
        visual_history_limit: 1,
        uia_element_limit: 100,
        desktop_enrichment_timeout_ms: 2_000,
        desktop_deep_enrichment_timeout_ms: 10_000,
        model_target_limit: 5,
        fusion_iou_threshold: 0.1,
        ocr_containment_threshold: 0.6,
        annotate_targets: false,
        approval_cache: Default::default(),
        user_guidance_queue: Default::default(),
        questions: None,
        command_manager: Arc::new(crate::commands::CommandManager::new(
            temp_dir.path().join("artifacts"),
        )),
        current_tool_call_id: Default::default(),
    };
    *context.latest_observation.lock() = Some(crate::types::Observation {
        version: Uuid::new_v4(),
        captured_at: chrono::Utc::now(),
        foreground_window: Some(crate::types::WindowInfo {
            id: "window-1".into(),
            title: "Chat".into(),
            process_name: "chat.exe".into(),
            bounds: crate::types::Rect {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
            elevated: false,
            visible: true,
            minimized: false,
        }),
        target: None,
        cursor: None,
        screenshots: Vec::new(),
        ocr: Vec::new(),
        ui_elements: Vec::new(),
        targets,
        timings_ms: BTreeMap::new(),
        warnings: Vec::new(),
    });
    context.active_task.lock().root_request = "Open the general channel".into();
    let router = Arc::new(router);
    let config = crate::config::DecisionRouterConfig {
        enabled: true,
        backend: crate::config::DecisionRouterBackend::Jev,
        model: "stub-router".into(),
        mode,
        ..Default::default()
    };
    let brain = Arc::new(MockBrain {
        summary_response: String::new(),
        calls: AtomicU64::new(0),
    });
    let session = Session::new(brain, Arc::new(tools), context, "mock-model".into(), 4)
        .unwrap()
        .with_decision_router(Some(router.clone() as Arc<dyn DecisionRouter>), config);
    FastHarness {
        session,
        router,
        clicks,
        scrolls,
        batches,
        static_screen,
        _temp_dir: temp_dir,
    }
}

fn fast_router(satisfied_after: usize, target_probability: f64) -> FastStubRouter {
    FastStubRouter {
        satisfied_after,
        target_probability,
        checks: AtomicU64::new(0),
        decided_tools: Default::default(),
        pick_label: None,
        condition_pick: None,
        condition_probability: 0.9,
        condition_questions: AtomicU64::new(0),
        batched: AtomicU64::new(0),
    }
}

#[test]
fn decision_activity_names_the_backend_that_answered() {
    let mut harness = fast_harness(
        fast_router(0, 0.99),
        Vec::new(),
        crate::config::DecisionRouterMode::Delegated,
    );
    if let Some(config) = harness.session.decision_router_config.as_mut() {
        config.backend = crate::config::DecisionRouterBackend::Laya;
    }
    harness.session.record_decision_activity(
        "context_selection",
        "JEV selected optional context",
        "selected",
        40,
    );
    harness
        .session
        .record_decision_activity("fast_actions", "Fast actions done", "", 1);
    let labels = harness
        .session
        .decision_activity
        .iter()
        .map(|activity| activity.label.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        labels,
        ["Laya selected optional context", "Fast actions done"]
    );
}

#[tokio::test]
async fn fast_actions_returns_done_without_acting_when_condition_already_holds() {
    let mut harness = fast_harness(
        fast_router(0, 0.99),
        vec![fast_link("t1", "general")],
        crate::config::DecisionRouterMode::Delegated,
    );
    let mut metrics = RunMetrics::default();
    let result = harness
        .session
        .run_fast_actions(
            1,
            &json!({"goal": "open general", "done_when": "general is open", "allowed_operations": ["click"]}),
            &mut metrics,
        )
        .await
        .unwrap();
    assert_eq!(result["status"], "done");
    assert!(harness.clicks.lock().is_empty());
    assert_eq!(result["steps"].as_array().unwrap().len(), 0);
    assert_eq!(harness.session.raw_navigation_unlocked, 0);
}

#[tokio::test]
async fn fast_actions_does_not_accept_the_target_label_as_completion_before_acting() {
    // done_when quotes the very link the node is about to click: seeing
    // that label on the current page is not evidence the goal was reached.
    // The router would answer "done" to every check; it must not be asked
    // before the target is clicked.
    let mut harness = fast_harness(
        fast_router(0, 0.99),
        vec![fast_link("t1", "Understanding Ownership")],
        crate::config::DecisionRouterMode::Delegated,
    );
    let mut metrics = RunMetrics::default();
    let result = harness
        .session
        .run_fast_actions(
            1,
            &json!({"goal": "open the chapter", "target_hint": "\"Understanding Ownership\"", "done_when": "heading \"Understanding Ownership\" is visible", "allowed_operations": ["click"]}),
            &mut metrics,
        )
        .await
        .unwrap();
    assert_eq!(result["status"], "done");
    assert_eq!(harness.clicks.lock().len(), 1);
}

#[tokio::test]
async fn fast_actions_executes_single_candidate_then_stops_on_completion() {
    let mut harness = fast_harness(
        fast_router(1, 0.99),
        vec![fast_link("t1", "general")],
        crate::config::DecisionRouterMode::Delegated,
    );
    let mut metrics = RunMetrics::default();
    let result = harness
        .session
        .run_fast_actions(
            1,
            &json!({"goal": "open general", "target_hint": "general", "done_when": "general is open", "allowed_operations": ["click"]}),
            &mut metrics,
        )
        .await
        .unwrap();
    assert_eq!(result["status"], "done");
    assert_eq!(harness.clicks.lock().len(), 1);
    assert_eq!(harness.clicks.lock()[0]["target_id"], "t1");
    // A single candidate is forced; the router is never asked to pick it.
    assert!(harness.router.decided_tools.lock().is_empty());
    assert_eq!(metrics.extras["fast_actions_steps"], json!(1));
}

#[tokio::test]
async fn fast_actions_runs_a_chained_plan_in_one_primary_model_call() {
    let mut harness = fast_harness(
        fast_router(1, 0.99),
        vec![fast_link("t1", "general")],
        crate::config::DecisionRouterMode::Delegated,
    );
    let mut metrics = RunMetrics::default();
    let result = harness
        .session
        .run_fast_actions(
            1,
            &json!({
                "goal": "open general",
                "done_when": "general is open",
                "allowed_operations": ["click"],
                "then": [{"goal": "show members", "done_when": "member list is visible"}],
            }),
            &mut metrics,
        )
        .await
        .unwrap();
    assert_eq!(result["status"], "done");
    assert_eq!(result["completed_subgoals"], 2);
    assert_eq!(result["subgoals"].as_array().unwrap().len(), 2);
    assert_eq!(harness.clicks.lock().len(), 1);
    assert_eq!(metrics.extras["fast_actions_runs"], json!(1));
}

#[tokio::test]
async fn training_log_keeps_questions_grounding_answered_with_their_answers() {
    let mut harness = fast_harness(
        fast_router(1, 0.2),
        vec![
            fast_link("t1", "System"),
            fast_link("t2", "Windows spotlight, dynamic images"),
            fast_link("t3", "Home"),
        ],
        crate::config::DecisionRouterMode::Delegated,
    );
    let dir = tempfile::tempdir().unwrap();
    let recorder = Arc::new(crate::router_training::TrainingRecorder::new(
        dir.path(),
        harness.session.id,
        &[],
    ));
    harness.session.training = Some(recorder.clone());
    let mut metrics = RunMetrics::default();
    harness
        .session
        .run_fast_actions(
            1,
            &json!({"goal": "open System", "target_hint": "\"System\"", "done_when": "\"System\" is visible"}),
            &mut metrics,
        )
        .await
        .unwrap();
    let records = std::fs::read_to_string(recorder.path())
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    let labels = records
        .iter()
        .filter(|record| record["kind"] == "label")
        .map(|record| {
            (
                record["question"].as_str().unwrap().to_owned(),
                record["label"].as_str().unwrap().to_owned(),
                record["source"].as_str().unwrap().to_owned(),
            )
        })
        .collect::<Vec<_>>();
    // The quoted-label click is the answer to the target question: the
    // option (by its request id) describing "System".
    let target = labels
        .iter()
        .find(|(question, _, source)| question == "target_CLICK" && source == "quoted_label_match")
        .unwrap_or_else(|| panic!("{labels:?}"));
    let question = records
        .iter()
        .find(|record| {
            record["kind"] == "question"
                && record["source"] == "grounding"
                && record["questions"].get("target_CLICK").is_some()
        })
        .unwrap();
    let answer = question["questions"]["target_CLICK"]["criteria"][&target.1]
        .as_str()
        .unwrap();
    assert!(
        answer.contains("System") && !answer.contains("spotlight"),
        "{answer}"
    );
    // ...and the grounded completion checks are yes/no answers.
    assert!(
        labels
            .iter()
            .any(|(question, _, source)| question == "satisfied" && source == "grounding"),
        "{labels:?}"
    );
    // Every label points at a recorded question.
    let ids = records
        .iter()
        .filter(|record| record["kind"] == "question")
        .map(|record| record["id"].clone())
        .collect::<Vec<_>>();
    assert!(
        records
            .iter()
            .filter(|record| record["kind"] == "label")
            .all(|label| ids.contains(&label["id"]))
    );
}

#[tokio::test]
async fn fast_actions_executes_an_unambiguous_label_match_without_the_router() {
    let mut harness = fast_harness(
        fast_router(1, 0.2),
        vec![
            fast_link("t1", "System"),
            fast_link("t2", "Windows spotlight, dynamic images"),
            fast_link("t3", "Home"),
        ],
        crate::config::DecisionRouterMode::Delegated,
    );
    let mut metrics = RunMetrics::default();
    let result = harness
        .session
        .run_fast_actions(
            1,
            &json!({"goal": "open System", "target_hint": "System", "done_when": "System page is shown"}),
            &mut metrics,
        )
        .await
        .unwrap();
    assert_eq!(result["status"], "done");
    assert_eq!(harness.clicks.lock()[0]["target_id"], "t1");
    assert!(harness.router.decided_tools.lock().is_empty());
}

#[tokio::test]
async fn fast_actions_hands_back_a_confident_pick_that_contradicts_the_hint() {
    // The stub router always picks the first-ranked candidate with high
    // confidence; with a hint naming a lower-ranked label, the local
    // cross-check must still refuse to execute the contradicting pick.
    let mut router = fast_router(usize::MAX, 0.99);
    router.pick_label = Some("Rust Programming Language");
    let mut harness = fast_harness(
        router,
        vec![
            fast_link("t1", "Understanding Ownership"),
            fast_link("t2", "Understanding Ownership chapter"),
            fast_link("t3", "Rust Programming Language"),
        ],
        crate::config::DecisionRouterMode::Delegated,
    );
    let observer = Arc::new(RecordingObserver {
        events: Mutex::new(Vec::new()),
    });
    harness.session.observer = Some(observer.clone());
    let mut metrics = RunMetrics::default();
    let result = harness
        .session
        .run_fast_actions(
            1,
            &json!({"goal": "open the Rust Programming Language ownership chapter", "target_hint": "the Understanding Ownership chapter link", "done_when": "ownership chapter is open", "max_steps": 1}),
            &mut metrics,
        )
        .await
        .unwrap();
    // The dashboard sees the router working and why it handed back.
    let events = observer.events.lock().clone();
    assert!(
        events.contains(&"router_started:fast_actions".to_owned()),
        "{events:?}"
    );
    assert!(
        events.contains(&"router_evaluated:fast_actions:contradicted_by_local_evidence".to_owned()),
        "{events:?}"
    );
    let clicked = harness
        .clicks
        .lock()
        .iter()
        .map(|call| call["target_id"].as_str().unwrap_or_default().to_owned())
        .collect::<Vec<_>>();
    assert!(!clicked.contains(&"t3".to_owned()), "{result}");
    assert_eq!(result["status"], "uncertain");
    assert!(!harness.router.decided_tools.lock().is_empty());
}

#[tokio::test]
async fn fast_actions_stops_after_the_named_target_instead_of_wandering() {
    let mut harness = fast_harness(
        fast_router(usize::MAX, 0.99),
        vec![
            fast_link("t1", "4. Understanding Ownership"),
            fast_link("t2", "The Rust Programming Language"),
            fast_link("t3", "Understanding Ownership"),
        ],
        crate::config::DecisionRouterMode::Delegated,
    );
    let mut metrics = RunMetrics::default();
    let result = harness
        .session
        .run_fast_actions(
            1,
            &json!({"goal": "open chapter 4", "target_hint": "link \"4. Understanding Ownership\"", "done_when": "chapter 4 is open"}),
            &mut metrics,
        )
        .await
        .unwrap();
    assert_eq!(result["status"], "unverified");
    assert_eq!(harness.clicks.lock().len(), 1);
    assert_eq!(harness.clicks.lock()[0]["target_id"], "t1");
    assert_eq!(harness.session.raw_navigation_unlocked, 0);
}

fn test_window(id: &str, title: &str, process: &str) -> crate::types::WindowInfo {
    crate::types::WindowInfo {
        id: id.into(),
        title: title.into(),
        process_name: process.into(),
        bounds: crate::types::Rect {
            x: 0,
            y: 0,
            width: 800,
            height: 600,
        },
        elevated: false,
        visible: true,
        minimized: false,
    }
}

/// A shell surface keeps the foreground and refuses to yield it.
struct StuckForegroundDesktop;

#[async_trait]
impl crate::platform::DesktopPlatform for StuckForegroundDesktop {
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
        Ok(Some(test_window("search", "Search", "SearchHost.exe")))
    }
    async fn list_windows(&self) -> Result<Vec<crate::types::WindowInfo>> {
        Ok(vec![
            test_window("task", "Settings", "SystemSettings.exe"),
            test_window("search", "Search", "SearchHost.exe"),
        ])
    }
    async fn activate_window(&self, _window_id: &str) -> Result<crate::types::WindowInfo> {
        Err(PokError::Tool(
            "activation did not make \"Settings\" the foreground window".into(),
        ))
    }
    async fn cursor_position(&self) -> Result<Option<(i32, i32)>> {
        Ok(None)
    }
    async fn simulate_input(&self, _action: &crate::types::InputAction) -> Result<()> {
        Ok(())
    }
}

#[tokio::test]
async fn failed_focus_restore_hands_the_change_to_the_primary_model() {
    let mut harness = fast_harness(
        fast_router(0, 0.99),
        Vec::new(),
        crate::config::DecisionRouterMode::Delegated,
    );
    harness.session.decision_router = None;
    harness.session.context.platform = Arc::new(StuckForegroundDesktop);
    let mut metrics = RunMetrics::default();
    harness
        .session
        .recover_environment_change(
            1,
            test_window("task", "Settings", "SystemSettings.exe"),
            Some(test_window("search", "Search", "SearchHost.exe")),
            &mut metrics,
        )
        .await
        .expect("a failed reactivation must not end the run");
    let last = harness.session.messages.last().expect("recovery message");
    let text = match last.content.first() {
        Some(MessageContent::Text { text }) => text.clone(),
        _ => String::new(),
    };
    assert!(text.contains("desktop-environment-event"), "{text}");
}

#[test]
fn oversized_browser_snapshots_keep_their_id_for_the_model() {
    let elements = (0..600)
        .map(|index| {
            json!({
                "id": format!("b{index}"),
                "role": if index % 4 == 0 { "link" } else { "generic" },
                "name": format!("Element {index} {}", "x".repeat(200)),
                "actionable": index % 4 == 0,
                "bounds": {"x": 0, "y": index, "width": 10, "height": 10},
                "guard": "g".repeat(300),
                "sensitive": index == 8,
                "value": if index == 8 { "hunter2" } else { "" },
            })
        })
        .collect::<Vec<_>>();
    let snapshot = json!({
        "elements": elements,
        "text": "t".repeat(20_000),
        "title": "The Rust Programming Language",
        "url": "https://doc.rust-lang.org/book/",
        "snapshot_id": "884354e3-092c-476c-b819-22dd25887059",
    });
    let projected = project_tool_result_for_model("managed_browser_snapshot", snapshot);
    assert_eq!(
        projected["snapshot_id"],
        "884354e3-092c-476c-b819-22dd25887059"
    );
    let shown = projected["elements"].as_array().unwrap();
    assert_eq!(shown.len(), 150);
    assert!(shown.iter().all(|element| element.get("bounds").is_none()));
    assert!(!projected.to_string().contains("hunter2"));
    assert!(projected.to_string().chars().count() <= MODEL_TOOL_RESULT_CHARS + 20_000);
}

#[test]
fn navigation_only_batches_are_distinguished_from_input_batches() {
    assert!(pure_navigation_batch(&json!({"steps": [
        {"kind": "click_target", "target_id": "3"},
        {"kind": "click", "target_id": "4"},
    ]})));
    assert!(!pure_navigation_batch(&json!({"steps": [
        {"kind": "click_target", "target_id": "3"},
        {"kind": "type_text", "text": "Settings"},
    ]})));
    assert!(!pure_navigation_batch(&json!({"steps": []})));
}

#[test]
fn leaked_invoke_style_tool_markup_is_not_a_final_answer() {
    // Live DeepSeek output: a tool call emitted as visible text with
    // finish_reason stop, which previously completed the run.
    let leaked = "On the System page now. Let me click Display.\n\nexpected_label\" string=\"true\">Display\n</parameter>\n<parameter name=\"observation_id\">obs_fbb5bcb6</parameter>\n</invoke>";
    assert!(contains_tool_protocol_markup(leaked));
    assert!(!contains_tool_protocol_markup(
        "The refresh rate is 165 Hz, shown under Advanced display."
    ));
}

fn fast_link_at(id: &str, name: &str, x: i32, y: i32) -> crate::types::InteractionTarget {
    let mut target = fast_link(id, name);
    target.bounds = crate::types::Rect {
        x,
        y,
        width: 90,
        height: 20,
    };
    target
}

#[tokio::test]
async fn quoted_done_when_is_decided_by_grounding_without_the_router() {
    let mut harness = fast_harness(
        fast_router(usize::MAX, 0.99),
        vec![fast_link("t1", "System"), fast_link("t2", "Display")],
        crate::config::DecisionRouterMode::Delegated,
    );
    let mut metrics = RunMetrics::default();
    let done = harness
        .session
        .run_fast_actions(
            1,
            &json!({"goal": "open System", "done_when": "\"Display\" is visible"}),
            &mut metrics,
        )
        .await
        .unwrap();
    assert_eq!(done["status"], "done");
    assert!(harness.clicks.lock().is_empty());

    let pending = harness
        .session
        .run_fast_actions(
            1,
            &json!({"goal": "open System", "target_hint": "\"System\"", "done_when": "\"Advanced display\" is visible"}),
            &mut metrics,
        )
        .await
        .unwrap();
    // The named target was clicked; the quoted condition stays unmet in
    // this static fixture, so the node hands back as unverified.
    assert_eq!(pending["status"], "unverified");
    assert_eq!(harness.clicks.lock().len(), 1);
    assert_eq!(harness.router.checks.load(Ordering::SeqCst), 0);
    assert_eq!(metrics.extras["fast_actions_grounded_checks"], json!(3));
}

#[tokio::test]
async fn fast_actions_reads_values_and_says_when_nothing_needed_doing() {
    let mut harness = fast_harness(
        fast_router(usize::MAX, 0.99),
        vec![
            fast_link_at("t1", "Refresh rate", 20, 100),
            fast_link_at("t2", "165 Hz", 400, 100),
        ],
        crate::config::DecisionRouterMode::Delegated,
    );
    let mut metrics = RunMetrics::default();
    let result = harness
        .session
        .run_fast_actions(
            1,
            &json!({
                "goal": "show the refresh rate",
                "done_when": "\"Refresh rate\" is visible",
                "read": [{"name": "rate", "label": "\"Refresh rate\""}],
            }),
            &mut metrics,
        )
        .await
        .unwrap();
    assert_eq!(result["status"], "done");
    assert_eq!(result["reads"]["rate"]["text"], "165 Hz");
    // The condition already held: the planner is told nothing was needed,
    // rather than seeing an empty step list it takes for a failure.
    assert!(result["note"].as_str().unwrap().contains("already held"));
    // This fixture has no screenshot, so only the observation id is returned.
    assert!(result["observation_id"].is_string());
}

async fn run_plan(harness: &mut FastHarness, plan: Value) -> (Value, RunMetrics) {
    let mut metrics = RunMetrics::default();
    let result = harness
        .session
        .run_fast_actions(1, &plan, &mut metrics)
        .await
        .unwrap();
    (result, metrics)
}

fn clicked_ids(harness: &FastHarness) -> Vec<String> {
    harness
        .clicks
        .lock()
        .iter()
        .map(|call| call["target_id"].as_str().unwrap_or_default().to_owned())
        .collect()
}

#[tokio::test]
async fn fast_actions_coerces_string_booleans_like_other_tools() {
    let mut harness = fast_harness(
        fast_router(usize::MAX, 0.99),
        vec![fast_link("t1", "Sign in"), fast_link("t2", "general")],
        crate::config::DecisionRouterMode::Delegated,
    );
    let mut tools = ToolRegistry::default();
    crate::builtins::register_desktop_tools(&mut tools);
    harness.session.tools = Arc::new(tools);
    let (result, _) = run_plan(
        &mut harness,
        json!({
            "goal": "open general",
            "target_hint": "\"general\"",
            "done_when": "\"general chat\" is visible",
            "on_interrupt": [{"when": "\"Sign in\" is visible", "stop": "true"}],
        }),
    )
    .await;
    assert_eq!(result["status"], "interrupted", "{result}");
}

#[tokio::test]
async fn double_click_operation_opens_the_named_target() {
    let mut harness = fast_harness(
        fast_router(usize::MAX, 0.99),
        vec![fast_link("t1", "Program Files"), fast_link("t2", "Windows")],
        crate::config::DecisionRouterMode::Delegated,
    );
    run_plan(
        &mut harness,
        json!({
            "goal": "open Program Files",
            "target_hint": "\"Program Files\"",
            "done_when": "\"Adobe\" is visible",
            "allowed_operations": ["double_click"],
        }),
    )
    .await;
    let clicks = harness.clicks.lock();
    assert_eq!(clicks.len(), 1);
    assert_eq!(clicks[0]["target_id"], "t1");
    assert_eq!(clicks[0]["double_click"], true);
}

#[tokio::test]
async fn avoid_removes_forbidden_targets_even_when_named() {
    let mut harness = fast_harness(
        fast_router(usize::MAX, 0.99),
        vec![
            fast_link("t1", "Lounge (voice channel)"),
            fast_link("t2", "general"),
        ],
        crate::config::DecisionRouterMode::Delegated,
    );
    let (result, _) = run_plan(
        &mut harness,
        json!({
            "goal": "open the lounge",
            "target_hint": "\"Lounge (voice channel)\"",
            "done_when": "\"Lounge chat\" is visible",
            "avoid": ["voice channel"],
            "max_steps": 1,
        }),
    )
    .await;
    assert!(
        !clicked_ids(&harness).contains(&"t1".to_owned()),
        "{result}"
    );
}

#[tokio::test]
async fn quoted_branch_is_taken_locally_without_the_router() {
    let mut harness = fast_harness(
        fast_router(usize::MAX, 0.99),
        vec![fast_link("t1", "Channels"), fast_link("t2", "general")],
        crate::config::DecisionRouterMode::Delegated,
    );
    let (result, _) = run_plan(
        &mut harness,
        json!({
            "goal": "check the server",
            "done_when": "\"Channels\" is visible",
            "branches": [
                {"when": "\"Sign in\" is visible", "then": [{"goal": "sign in", "done_when": "\"Welcome\" is visible"}]},
                {"when": "\"general\" is visible", "then": [{"goal": "open general", "target_hint": "\"general\"", "done_when": "\"general\" is visible"}]},
            ],
        }),
    )
    .await;
    assert_eq!(result["status"], "done", "{result}");
    assert_eq!(
        result["subgoals"][1]["branch_taken"],
        "\"general\" is visible"
    );
    assert_eq!(harness.router.condition_questions.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn unquoted_branch_uses_the_router_and_otherwise_is_the_default() {
    let mut router = fast_router(usize::MAX, 0.99);
    router.condition_pick = Some("members list is collapsed");
    let mut harness = fast_harness(
        router,
        vec![fast_link("t1", "Channels"), fast_link("t2", "Show members")],
        crate::config::DecisionRouterMode::Delegated,
    );
    let (result, _) = run_plan(
        &mut harness,
        json!({
            "goal": "check the server",
            "done_when": "\"Channels\" is visible",
            "branches": [
                {"when": "the members list is collapsed", "then": [{"goal": "expand members", "target_hint": "\"Show members\"", "done_when": "\"Member list\" is visible"}]},
                {"when": "otherwise", "then": []},
            ],
        }),
    )
    .await;
    assert_eq!(
        result["subgoals"][1]["branch_taken"],
        "the members list is collapsed"
    );
    assert_eq!(clicked_ids(&harness), vec!["t2".to_owned()]);
    assert_eq!(harness.router.condition_questions.load(Ordering::SeqCst), 1);

    // With no condition true and no otherwise branch, the plan hands back.
    let mut stuck = fast_harness(
        fast_router(usize::MAX, 0.99),
        vec![fast_link("t1", "Channels")],
        crate::config::DecisionRouterMode::Delegated,
    );
    let (result, _) = run_plan(
        &mut stuck,
        json!({
            "goal": "check the server",
            "done_when": "\"Channels\" is visible",
            "branches": [{"when": "\"Sign in\" is visible", "then": []}],
        }),
    )
    .await;
    assert_eq!(result["status"], "uncertain_branch", "{result}");
    assert_eq!(
        stuck.session.raw_navigation_unlocked,
        RAW_NAVIGATION_UNLOCK_TURNS
    );
}

#[tokio::test]
async fn interrupt_rules_click_a_quoted_button_or_stop() {
    let mut button = fast_link("b1", "Accept");
    button.control_type = "button".into();
    let mut harness = fast_harness(
        fast_router(usize::MAX, 0.99),
        vec![button, fast_link("t1", "general")],
        crate::config::DecisionRouterMode::Delegated,
    );
    let (_, metrics) = run_plan(
        &mut harness,
        json!({
            "goal": "open general",
            "target_hint": "\"general\"",
            "done_when": "\"general chat\" is visible",
            "on_interrupt": [{"when": "\"Accept\" is visible", "target_hint": "\"Accept\""}],
            "max_steps": 3,
        }),
    )
    .await;
    let clicked = clicked_ids(&harness);
    assert_eq!(
        clicked.first().map(String::as_str),
        Some("b1"),
        "{clicked:?}"
    );
    assert_eq!(metrics.extras["fast_actions_interrupts"], json!(1));

    let mut stopping = fast_harness(
        fast_router(usize::MAX, 0.99),
        vec![fast_link("t1", "Sign in"), fast_link("t2", "general")],
        crate::config::DecisionRouterMode::Delegated,
    );
    let (result, _) = run_plan(
        &mut stopping,
        json!({
            "goal": "open general",
            "target_hint": "\"general\"",
            "done_when": "\"general chat\" is visible",
            "on_interrupt": [{"when": "\"Sign in\" is visible", "stop": true}],
        }),
    )
    .await;
    assert_eq!(result["status"], "interrupted", "{result}");
    assert!(stopping.clicks.lock().is_empty());
}

#[tokio::test]
async fn unquoted_interrupt_questions_ride_along_with_the_target_pick() {
    let mut harness = fast_harness(
        fast_router(usize::MAX, 0.99),
        vec![fast_link("t1", "general"), fast_link("t2", "random")],
        crate::config::DecisionRouterMode::Delegated,
    );
    let (_, metrics) = run_plan(
        &mut harness,
        json!({
            "goal": "open a chat channel",
            "target_hint": "a chat channel",
            "done_when": "a chat channel is open",
            "on_interrupt": [{"when": "a sign-in dialog is visible", "stop": true}],
            "max_steps": 3,
        }),
    )
    .await;
    assert!(!harness.clicks.lock().is_empty());
    // The first step had a router pick, so its interrupt question rode
    // along; later steps without a pick settle the question on their own.
    assert_eq!(metrics.extras["fast_actions_fanout_conditions"], json!(1));
    assert_eq!(harness.router.batched.load(Ordering::SeqCst), 1);
}

#[test]
fn nested_plan_quirks_from_real_models_are_accepted() {
    // A plan a model sent verbatim: item-wrapped lists inside nested
    // steps, and a last step that only reads a value.
    let mut arguments = json!({
        "allowed_operations": ["click", "scroll"],
        "done_when": "heading \"Advanced display\" is visible",
        "goal": "open the System > Display page and scroll to the refresh rate setting",
        "target_hint": "\"System\"",
        "then": [
            {
                "allowed_operations": {"item": ["click", "scroll"]},
                "done_when": "the refresh rate dropdown value is visible",
                "goal": "click the \"Advanced display\" link",
                "target_hint": "\"Advanced display\""
            },
            {"read": {"item": {"label": "\"Refresh rate\"", "name": "refresh_rate"}}}
        ]
    });
    repair_fast_plan(&mut arguments, true);
    let args: FastActionsArgs = serde_json::from_value(arguments.clone()).unwrap();
    assert!(FastPlan::from_args(&args).is_ok());
    let read_step = &arguments["then"][1];
    assert_eq!(read_step["goal"], "read the requested values");
    assert_eq!(read_step["done_when"], "\"Refresh rate\" is visible");
    assert_eq!(read_step["read"][0]["name"], "refresh_rate");
    assert_eq!(
        arguments["then"][0]["allowed_operations"],
        json!(["click", "scroll"])
    );
}

#[tokio::test]
async fn a_weak_interrupt_match_does_not_stop_the_run() {
    // 0.64 clears the completion bar but not the interrupt bar: a false
    // "a permission dialog appeared" must not throw away the whole run.
    let mut router = fast_router(usize::MAX, 0.99);
    router.condition_pick = Some("administrator");
    router.condition_probability = 0.64;
    let mut harness = fast_harness(
        router,
        vec![fast_link("t1", "Display"), fast_link("t2", "Sound")],
        crate::config::DecisionRouterMode::Delegated,
    );
    let (result, _) = run_plan(
        &mut harness,
        json!({
            "goal": "open Display settings",
            "target_hint": "\"Display\"",
            "done_when": "\"Advanced display\" is visible",
            "on_interrupt": [{"when": "a permission or administrator dialog appears", "stop": true}],
            "max_steps": 2,
        }),
    )
    .await;
    assert_ne!(result["status"], "interrupted", "{result}");
    assert!(!harness.clicks.lock().is_empty());
}

#[tokio::test]
async fn a_fanned_out_interrupt_match_acts_before_the_target() {
    let mut router = fast_router(usize::MAX, 0.99);
    router.condition_pick = Some("sign-in");
    let mut harness = fast_harness(
        router,
        vec![fast_link("t1", "general"), fast_link("t2", "random")],
        crate::config::DecisionRouterMode::Delegated,
    );
    let (result, _) = run_plan(
        &mut harness,
        json!({
            "goal": "open a chat channel",
            "target_hint": "a chat channel",
            "done_when": "a chat channel is open",
            "on_interrupt": [{"when": "a sign-in dialog is visible", "stop": true}],
            "max_steps": 3,
        }),
    )
    .await;
    assert_eq!(result["status"], "interrupted", "{result}");
    assert!(harness.clicks.lock().is_empty());
    assert_eq!(harness.router.condition_questions.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn oversized_plans_are_rejected_before_acting() {
    let mut harness = fast_harness(
        fast_router(0, 0.99),
        vec![fast_link("t1", "general")],
        crate::config::DecisionRouterMode::Delegated,
    );
    let steps = (0..64)
        .map(
            |index| json!({"goal": format!("step {index}"), "done_when": "\"general\" is visible"}),
        )
        .collect::<Vec<_>>();
    let mut metrics = RunMetrics::default();
    let error = harness
        .session
        .run_fast_actions(
            1,
            &json!({"goal": "start", "done_when": "\"general\" is visible", "then": steps}),
            &mut metrics,
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("at most 64"), "{error}");
    let rules = (0..5)
        .map(|_| json!({"when": "\"x\" is visible", "stop": true}))
        .collect::<Vec<_>>();
    assert!(
        harness
            .session
            .run_fast_actions(
                1,
                &json!({"goal": "start", "done_when": "\"general\" is visible", "on_interrupt": rules}),
                &mut metrics,
            )
            .await
            .is_err()
    );
}

#[tokio::test]
async fn untrusted_backends_never_pick_targets_grounding_cannot_decide() {
    let mut harness = fast_harness(
        fast_router(usize::MAX, 0.99),
        vec![
            fast_link("t1", "general chat"),
            fast_link("t2", "general news"),
        ],
        crate::config::DecisionRouterMode::Delegated,
    );
    if let Some(config) = harness.session.decision_router_config.as_mut() {
        // Not yet measured in the live matrix, so untrusted by default.
        config.backend = crate::config::DecisionRouterBackend::LlmChoice;
        config.llm_choice.model = "stub-router".into();
    }
    let (result, _) = run_plan(
        &mut harness,
        json!({"goal": "open general", "done_when": "general is open"}),
    )
    .await;
    assert_eq!(result["status"], "uncertain", "{result}");
    assert!(harness.router.decided_tools.lock().is_empty());
    assert!(harness.clicks.lock().is_empty());
    // Unquoted completion is not judged by an untrusted backend either.
    assert_eq!(harness.router.checks.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn an_enabled_judge_vetoes_even_confident_router_picks() {
    let mut harness = fast_harness(
        fast_router(usize::MAX, 0.99),
        vec![
            fast_link("t1", "general chat"),
            fast_link("t2", "general news"),
        ],
        crate::config::DecisionRouterMode::Delegated,
    );
    if let Some(config) = harness.session.decision_router_config.as_mut() {
        config.judge.enabled = true;
    }
    // The stub router uses the trait's default judge, which withholds.
    let (result, metrics) = run_plan(
        &mut harness,
        json!({"goal": "open general", "done_when": "general is open"}),
    )
    .await;
    assert_eq!(result["status"], "uncertain", "{result}");
    assert!(harness.clicks.lock().is_empty());
    assert_eq!(metrics.extras["fast_actions_judge_calls"], json!(1));
}

#[tokio::test]
async fn completion_and_target_pick_share_one_request_when_batched() {
    let plan = json!({
        "goal": "open general",
        "done_when": "the general channel is open",
        "allowed_operations": ["click"],
        "max_steps": 1,
    });
    let mut batched = fast_harness(
        fast_router(usize::MAX, 0.99),
        vec![
            fast_link("t1", "general chat"),
            fast_link("t2", "general news"),
        ],
        crate::config::DecisionRouterMode::Delegated,
    );
    run_plan(&mut batched, plan.clone()).await;
    assert_eq!(batched.router.batched.load(Ordering::SeqCst), 1);
    // The step's completion rode in the batch; only the final budget
    // check after the click is asked on its own.
    assert_eq!(batched.router.checks.load(Ordering::SeqCst), 1);
    assert_eq!(batched.clicks.lock().len(), 1);

    let mut separate = fast_harness(
        fast_router(usize::MAX, 0.99),
        vec![
            fast_link("t1", "general chat"),
            fast_link("t2", "general news"),
        ],
        crate::config::DecisionRouterMode::Delegated,
    );
    if let Some(config) = separate.session.decision_router_config.as_mut() {
        config.batch_router_questions = false;
    }
    run_plan(&mut separate, plan).await;
    assert_eq!(separate.router.batched.load(Ordering::SeqCst), 0);
    assert_eq!(separate.router.checks.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn deferred_completion_is_settled_before_an_exact_label_click() {
    // Completion already holds: the deferred check must run and stop the
    // node before the exact-label candidate is clicked.
    let mut harness = fast_harness(
        fast_router(0, 0.99),
        vec![fast_link("t1", "general"), fast_link("t2", "random")],
        crate::config::DecisionRouterMode::Delegated,
    );
    let (result, _) = run_plan(
        &mut harness,
        json!({"goal": "open general", "target_hint": "\"general\"", "done_when": "the general channel is open"}),
    )
    .await;
    assert_eq!(result["status"], "done", "{result}");
    assert!(harness.clicks.lock().is_empty());
    assert_eq!(harness.router.batched.load(Ordering::SeqCst), 0);
    assert_eq!(harness.router.checks.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn fast_actions_returns_uncertain_alternatives_instead_of_guessing() {
    let mut harness = fast_harness(
        fast_router(usize::MAX, 0.2),
        vec![
            fast_link("t1", "general chat"),
            fast_link("t2", "general news"),
        ],
        crate::config::DecisionRouterMode::Delegated,
    );
    let mut metrics = RunMetrics::default();
    let result = harness
        .session
        .run_fast_actions(
            1,
            &json!({"goal": "open general", "done_when": "general is open", "allowed_operations": ["click"]}),
            &mut metrics,
        )
        .await
        .unwrap();
    assert_eq!(result["status"], "uncertain");
    assert!(harness.clicks.lock().is_empty());
    // Handing a subgoal back unlocks raw navigation for the primary model.
    assert_eq!(
        harness.session.raw_navigation_unlocked,
        RAW_NAVIGATION_UNLOCK_TURNS
    );
    assert!(result["next"].is_string());
    assert!(!result["alternatives"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn fast_actions_offers_only_allowed_operations() {
    let mut harness = fast_harness(
        fast_router(usize::MAX, 0.99),
        vec![fast_link("t1", "general")],
        crate::config::DecisionRouterMode::Delegated,
    );
    let mut metrics = RunMetrics::default();
    let result = harness
        .session
        .run_fast_actions(
            1,
            &json!({"goal": "find older messages", "done_when": "older messages are visible", "allowed_operations": ["scroll"], "max_steps": 2}),
            &mut metrics,
        )
        .await
        .unwrap();
    assert!(harness.clicks.lock().is_empty());
    assert!(!harness.scrolls.lock().is_empty());
    assert!(
        harness
            .router
            .decided_tools
            .lock()
            .iter()
            .flatten()
            .all(|tool| tool == "scroll_view")
    );
    assert!(result["steps"].as_array().unwrap().len() <= 2);
}

#[test]
fn delegated_catalog_marks_withheld_navigation_tools_locked() {
    let catalog = "- managed_browser_click [browser; active]: Click\n- managed_browser_select [browser; active]: Select".to_string();
    let locked = delegated_catalog(catalog.clone(), true);
    assert!(locked.contains("managed_browser_click [browser; locked: use fast_actions]"));
    assert!(locked.contains("managed_browser_select [browser; active]"));
    assert_eq!(delegated_catalog(catalog.clone(), false), catalog);
}

#[tokio::test]
async fn fast_actions_scrolls_when_the_quoted_target_is_not_visible() {
    // 0.6 is below the click gate but above the read-only scroll tier.
    let mut harness = fast_harness(
        fast_router(usize::MAX, 0.6),
        vec![
            fast_link("t1", "General"),
            fast_link("t2", "Personalization"),
        ],
        crate::config::DecisionRouterMode::Delegated,
    );
    let mut metrics = RunMetrics::default();
    harness
        .session
        .run_fast_actions(
            1,
            &json!({"goal": "open advanced display", "target_hint": "\"Advanced display\"", "done_when": "\"Advanced display\" is open", "allowed_operations": ["click", "scroll"], "max_steps": 1}),
            &mut metrics,
        )
        .await
        .unwrap();
    assert!(harness.clicks.lock().is_empty());
    assert_eq!(harness.scrolls.lock().len(), 1);
    assert!(
        harness
            .router
            .decided_tools
            .lock()
            .iter()
            .flatten()
            .all(|tool| tool == "scroll_view")
    );
}

#[tokio::test]
async fn fast_actions_scroll_search_can_keep_scrolling_the_same_way() {
    let mut router = fast_router(usize::MAX, 0.9);
    router.pick_label = Some("down by one page");
    let mut harness = fast_harness(
        router,
        vec![fast_link("t1", "General")],
        crate::config::DecisionRouterMode::Delegated,
    );
    let mut metrics = RunMetrics::default();
    harness
        .session
        .run_fast_actions(
            1,
            &json!({"goal": "open advanced display", "target_hint": "\"Advanced display\"", "done_when": "\"Advanced display\" is open", "allowed_operations": ["scroll"], "max_steps": 3}),
            &mut metrics,
        )
        .await
        .unwrap();
    let scrolls = harness.scrolls.lock();
    assert_eq!(scrolls.len(), 3, "{scrolls:?}");
    assert!(
        scrolls
            .iter()
            .all(|scroll| scroll["direction"] == "down" && scroll["amount"] == "page")
    );
}

#[tokio::test]
async fn fast_actions_scroll_search_defaults_down_when_the_model_is_unsure() {
    let mut harness = fast_harness(
        fast_router(usize::MAX, 0.2),
        vec![fast_link("t1", "General")],
        crate::config::DecisionRouterMode::Delegated,
    );
    let mut metrics = RunMetrics::default();
    let result = harness
        .session
        .run_fast_actions(
            1,
            &json!({"goal": "open advanced display", "target_hint": "\"Advanced display\"", "done_when": "\"Advanced display\" is open", "allowed_operations": ["click", "scroll"], "max_steps": 1}),
            &mut metrics,
        )
        .await
        .unwrap();
    assert_ne!(result["status"], "uncertain");
    let scrolls = harness.scrolls.lock();
    assert_eq!(scrolls.len(), 1);
    assert_eq!(scrolls[0]["direction"], "down");
    assert_eq!(scrolls[0]["amount"], "page");
}

#[tokio::test]
async fn fast_actions_keeps_the_click_gate_for_an_unsure_click() {
    // The quoted target is visible, but a second similar label makes it
    // ambiguous; a 0.6 click pick stays below the click gate.
    let mut harness = fast_harness(
        fast_router(usize::MAX, 0.6),
        vec![
            fast_link("t1", "Display"),
            fast_link("t2", "Display settings"),
        ],
        crate::config::DecisionRouterMode::Delegated,
    );
    let mut metrics = RunMetrics::default();
    let result = harness
        .session
        .run_fast_actions(
            1,
            &json!({"goal": "open the display page", "target_hint": "display page", "done_when": "the display page is open", "allowed_operations": ["click", "scroll"], "max_steps": 1}),
            &mut metrics,
        )
        .await
        .unwrap();
    assert!(harness.clicks.lock().is_empty());
    assert_eq!(result["status"], "uncertain");
}

#[tokio::test]
async fn a_click_that_changes_nothing_is_not_reported_as_done() {
    // The link label is also the quoted condition: after a click that
    // leaves the page unchanged, the label proves nothing.
    let mut harness = fast_harness(
        fast_router(0, 0.99),
        vec![fast_link("t1", "Understanding Ownership")],
        crate::config::DecisionRouterMode::Delegated,
    );
    harness.static_screen.store(true, Ordering::SeqCst);
    let mut metrics = RunMetrics::default();
    let result = harness
        .session
        .run_fast_actions(
            1,
            &json!({"goal": "open the chapter", "target_hint": "\"Understanding Ownership\"", "done_when": "heading \"Understanding Ownership\" is visible", "allowed_operations": ["click"]}),
            &mut metrics,
        )
        .await
        .unwrap();
    assert_eq!(result["status"], "stalled", "{result}");
    assert_eq!(harness.clicks.lock().len(), 1);
}

#[tokio::test]
async fn the_requested_commit_is_handed_back_ready_to_confirm() {
    let mut harness = fast_harness(
        fast_router(usize::MAX, 0.99),
        vec![fast_link("t1", "Save"), fast_link("t2", "Cancel")],
        crate::config::DecisionRouterMode::Delegated,
    );
    harness.session.context.active_task.lock().root_request =
        "Type the note and save it as notes.txt".into();
    let mut metrics = RunMetrics::default();
    let result = harness
        .session
        .run_fast_actions(
            1,
            &json!({"goal": "save the file", "target_hint": "\"Save\"", "done_when": "the dialog is closed", "allowed_operations": ["click"]}),
            &mut metrics,
        )
        .await
        .unwrap();
    assert_eq!(result["status"], "commit_required", "{result}");
    assert!(
        harness.clicks.lock().is_empty(),
        "the commit is not pressed"
    );
    let ready = &result["alternatives"][0];
    assert_eq!(ready["tool"], "click_target");
    assert_eq!(ready["arguments"]["expected_label"], "Save");
}

#[tokio::test]
async fn fast_actions_accepts_a_quoted_window_title_that_is_already_in_front() {
    // The planner asks for a window that is already in the foreground and
    // quotes its title in both the hint and done_when.
    let mut harness = fast_harness(
        fast_router(usize::MAX, 0.99),
        vec![fast_link("t1", "general")],
        crate::config::DecisionRouterMode::Delegated,
    );
    let mut metrics = RunMetrics::default();
    let result = harness
        .session
        .run_fast_actions(
            1,
            &json!({"goal": "bring the chat window forward", "target_hint": "\"Chat\"", "done_when": "the window titled \"Chat\" is in the foreground", "allowed_operations": ["activate_window", "click"]}),
            &mut metrics,
        )
        .await
        .unwrap();
    assert_eq!(result["status"], "done", "{result}");
    assert!(harness.clicks.lock().is_empty());
}

#[tokio::test]
async fn fast_actions_scroll_goal_sends_unique_candidate_ids() {
    // A goal that mentions scrolling makes the task candidates include
    // scroll options too; they must not reach the router twice.
    let mut harness = fast_harness(
        fast_router(usize::MAX, 0.99),
        vec![fast_link("t1", "general")],
        crate::config::DecisionRouterMode::Delegated,
    );
    let mut metrics = RunMetrics::default();
    let result = harness
        .session
        .run_fast_actions(
            1,
            &json!({"goal": "Scroll down the page until older messages show", "done_when": "older messages are visible", "allowed_operations": ["click", "scroll"], "max_steps": 2}),
            &mut metrics,
        )
        .await
        .unwrap();
    assert!(!harness.router.decided_tools.lock().is_empty());
    assert_ne!(result["status"], "unavailable");
}

#[tokio::test]
async fn fast_actions_budget_bounds_the_run() {
    let mut harness = fast_harness(
        fast_router(usize::MAX, 0.99),
        vec![
            fast_link("t1", "general"),
            fast_link("t2", "general news"),
            fast_link("t3", "general chat"),
        ],
        crate::config::DecisionRouterMode::Delegated,
    );
    let mut metrics = RunMetrics::default();
    let result = harness
        .session
        .run_fast_actions(
            1,
            &json!({"goal": "open general", "done_when": "general is open", "allowed_operations": ["click"], "max_steps": 2}),
            &mut metrics,
        )
        .await
        .unwrap();
    assert_eq!(result["status"], "budget_exhausted");
    assert_eq!(harness.clicks.lock().len(), 2);
    // The same target is never re-attempted within one run.
    let clicked = harness
        .clicks
        .lock()
        .iter()
        .map(|call| call["target_id"].clone())
        .collect::<HashSet<_>>();
    assert_eq!(clicked.len(), 2);
}

#[tokio::test]
async fn fast_actions_reports_unavailable_without_router() {
    let mut harness = fast_harness(
        fast_router(0, 0.99),
        Vec::new(),
        crate::config::DecisionRouterMode::Delegated,
    );
    harness.session.decision_router = None;
    let mut metrics = RunMetrics::default();
    let result = harness
        .session
        .run_fast_actions(
            1,
            &json!({"goal": "open general", "done_when": "general is open"}),
            &mut metrics,
        )
        .await
        .unwrap();
    assert_eq!(result["status"], "unavailable");
}

#[test]
fn delegated_mode_is_default_and_advertised_in_the_system_prompt() {
    assert_eq!(
        crate::config::DecisionRouterConfig::default().mode,
        crate::config::DecisionRouterMode::Delegated
    );
    let delegated = fast_harness(
        fast_router(0, 0.99),
        Vec::new(),
        crate::config::DecisionRouterMode::Delegated,
    );
    assert!(delegated.session.delegated_decision_router());
    let system_text = |session: &Session| match session.messages[0].content.first() {
        Some(MessageContent::Text { text }) => text.clone(),
        _ => String::new(),
    };
    assert!(system_text(&delegated.session).contains("fast_actions"));
    let legacy = fast_harness(
        fast_router(0, 0.99),
        Vec::new(),
        crate::config::DecisionRouterMode::RouterFirst,
    );
    assert!(!legacy.session.delegated_decision_router());
    assert!(!system_text(&legacy.session).contains("fast_actions"));
}

#[tokio::test]
async fn counting_brain_counts_every_request_and_usage() {
    struct UsageBrain;
    #[async_trait]
    impl Brain for UsageBrain {
        async fn list_models(&self) -> Result<Vec<String>> {
            Ok(Vec::new())
        }
        fn stream(&self, _request: BrainRequest) -> crate::brain::BrainStream {
            Box::pin(futures::stream::iter(vec![Ok(BrainEvent::Usage {
                prompt_tokens: 10,
                completion_tokens: 3,
            })]))
        }
    }
    let brain = crate::brain::CountingBrain::new(Arc::new(UsageBrain));
    let counter = brain.counter();
    let start = counter.snapshot();
    for _ in 0..2 {
        let request = BrainRequest {
            model: "m".into(),
            messages: Vec::new(),
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            seed: None,
            reasoning_effort: None,
        };
        let _ = brain.stream(request).collect::<Vec<_>>().await;
    }
    let usage = counter.snapshot().since(start);
    assert_eq!(usage.requests, 2);
    assert_eq!(usage.prompt_tokens, 20);
    assert_eq!(usage.completion_tokens, 6);
}

struct SkillWriterBrain {
    response: String,
    calls: AtomicU64,
}

#[async_trait]
impl Brain for SkillWriterBrain {
    async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec!["mock".into()])
    }
    fn stream(&self, _request: BrainRequest) -> crate::brain::BrainStream {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(futures::stream::iter(vec![
            Ok(BrainEvent::TextDelta {
                text: self.response.clone(),
            }),
            Ok(BrainEvent::Finished {
                reason: Some("stop".into()),
            }),
        ]))
    }
}

fn explorer_workflow() -> Vec<Value> {
    vec![
        workflow_step(
            "capture_screen",
            &json!({}),
            &json!({"target": {"scope": "window", "app": "explorer.exe"}}),
        )
        .unwrap(),
        workflow_step(
            "click_target",
            &json!({"target_id": "42"}),
            &json!({
                "executed": true,
                "action": {"kind": "click_target", "target_id": "42", "label": "Open"},
                "focus": {"app": "explorer.exe", "title": "Private Folder"},
                "state_change": {"added_text": ["Folder opened"]},
            }),
        )
        .unwrap(),
        workflow_step(
            "capture_screen",
            &json!({}),
            &json!({"target": {"scope": "window", "app": "explorer.exe"}}),
        )
        .unwrap(),
    ]
}

#[test]
fn written_skills_use_the_workflows_tools_and_stay_impersonal() {
    let allowed = BTreeSet::from(["capture_screen".to_owned(), "click_target".to_owned()]);
    let skill = |title: &str, tool: &str, instruction: &str| {
        json!({"title": title, "summary": "Use to open a folder in File Explorer.",
            "steps": [{"tool": tool, "instruction": instruction}]})
        .to_string()
    };
    let good = parse_skill_text(
        &format!(
            "Here it is: {}",
            skill(
                "Open a folder in File Explorer",
                "click_target",
                "Click the folder named in the request, e.g. \"Documents\"."
            )
        ),
        &allowed,
    )
    .unwrap();
    assert_eq!(good.title, "Open a folder in File Explorer");
    // A tool the workflow never used.
    assert!(parse_skill_text(&skill("Open a folder", "run_command", "dir"), &allowed).is_err());
    // Personal or run-specific details.
    assert!(
        parse_skill_text(
            &skill(
                "Open a folder",
                "click_target",
                "Open C:\\Users\\person\\Documents"
            ),
            &allowed
        )
        .is_err()
    );
    assert!(
        parse_skill_text(
            &skill("Open a folder", "click_target", "Search order 1234567890"),
            &allowed
        )
        .is_err()
    );
    assert!(
        parse_skill_text(
            &skill("Mail someone@example.com", "click_target", "Open"),
            &allowed
        )
        .is_err()
    );
    assert!(parse_skill_text("not json", &allowed).is_err());
}

#[tokio::test]
async fn verified_workflows_grow_one_improved_skill_per_task() {
    let dir = tempfile::tempdir().unwrap();
    let memory = crate::memory::MemoryStore::open(dir.path()).unwrap();
    let workflow = explorer_workflow();
    let qualification = qualify_workflow(&workflow, &[]);
    let brain = Arc::new(SkillWriterBrain {
        response: json!({
            "title": "Open a folder in File Explorer",
            "summary": "Use to open a named folder in File Explorer.",
            "steps": [
                {"tool": "capture_screen", "instruction": "Capture the File Explorer window."},
                {"tool": "click_target", "instruction": "Open the folder named in the request by its visible label."}
            ]
        })
        .to_string(),
        calls: AtomicU64::new(0),
    });
    let cancellation = CancellationToken::new();
    let config = crate::config::DecisionRouterConfig::default();
    let learn = |prompt: &'static str, router: Option<Arc<dyn DecisionRouter>>| {
        let memory = memory.clone();
        let brain = brain.clone();
        let workflow = workflow.clone();
        let qualification = qualification.clone();
        let cancellation = cancellation.clone();
        let config = config.clone();
        async move {
            learn_skill_with_review(
                &memory,
                router,
                Some(&config),
                brain,
                "mock",
                &HelperReasoning::default(),
                prompt,
                &workflow,
                &qualification,
                &cancellation,
            )
            .await
            .unwrap()
            .unwrap()
        }
    };
    // A new task: created, then rewritten into reusable guidance.
    let first = learn("open the requested folder", None).await;
    assert_eq!((first.outcome, first.rewritten), ("created", true));
    let skills = memory.list_procedures(None, 10).unwrap();
    assert_eq!(skills[0].title, "Open a folder in File Explorer");
    assert_eq!(skills[0].steps.len(), 2);
    // The same task again only reinforces it.
    let again = learn("open the requested folder", None).await;
    assert_eq!((again.outcome, again.success_count), ("reinforced", 2));
    // The same task in other words: the router judges it the same skill.
    let mut router = fast_router(usize::MAX, 0.99);
    router.condition_pick = Some("Open a folder in File Explorer");
    router.condition_probability = 0.9;
    let merged = learn(
        "show the folder I asked for in explorer",
        Some(Arc::new(router)),
    )
    .await;
    assert_eq!(merged.outcome, "merged");
    assert_eq!(memory.list_procedures(None, 10).unwrap().len(), 1);
    // Below the router's bar it stays a separate skill.
    let mut unsure = fast_router(usize::MAX, 0.99);
    unsure.condition_pick = Some("Open a folder in File Explorer");
    unsure.condition_probability = 0.7;
    let separate = learn(
        "list what is inside the folder in explorer",
        Some(Arc::new(unsure)),
    )
    .await;
    assert_eq!(separate.outcome, "created");
    assert_eq!(memory.list_procedures(None, 10).unwrap().len(), 2);
}

#[test]
fn a_rewritten_skill_can_be_restored() {
    let dir = tempfile::tempdir().unwrap();
    let memory = crate::memory::MemoryStore::open(dir.path()).unwrap();
    let workflow = explorer_workflow();
    let qualification = qualify_workflow(&workflow, &[]);
    let (skill, _) = learn_verified_procedures(
        &memory,
        "open the requested folder",
        &workflow,
        &qualification,
    )
    .unwrap()
    .remove(0);
    let steps = vec![crate::memory::ProcedureStep {
        tool: "click_target".into(),
        instruction: "Open the folder by its label.".into(),
    }];
    memory
        .rewrite_procedure(
            skill.id,
            "Open a folder in File Explorer",
            "Use to open a folder.",
            &steps,
        )
        .unwrap();
    assert!(memory.revert_procedure(skill.id).unwrap());
    let restored = memory.list_procedures(None, 10).unwrap().remove(0);
    assert_eq!(restored.title, skill.title);
    assert_eq!(restored.steps, skill.steps);
    assert!(!memory.revert_procedure(skill.id).unwrap());
}

#[tokio::test]
async fn a_relevant_learned_skill_is_given_to_the_model_once() {
    let learned = tempfile::tempdir().unwrap();
    let memory = MemoryStore::open(learned.path()).unwrap();
    let workflow = explorer_workflow();
    let qualification = qualify_workflow(&workflow, &[]);
    learn_verified_procedures(
        &memory,
        "open the requested folder in explorer",
        &workflow,
        &qualification,
    )
    .unwrap();
    // Counted from the trace: the loaded skill message itself may later be
    // compacted out of a small context window.
    let skill_loads = |session: &Session| {
        std::fs::read_to_string(session.context.artifact_dir.join("trace.jsonl"))
            .unwrap_or_default()
            .lines()
            .filter(|line| line.contains("\"kind\":\"skill_auto_loaded\""))
            .count()
    };
    for (prompt, expected) in [
        ("open the requested folder in explorer", 1),
        ("what time is it in Tokyo", 0),
    ] {
        let mut harness = fast_harness(
            fast_router(usize::MAX, 0.99),
            Vec::new(),
            crate::config::DecisionRouterMode::Delegated,
        );
        harness.session.context.memory = memory.clone();
        harness.session.brain = Arc::new(SkillWriterBrain {
            response: "Done.".into(),
            calls: AtomicU64::new(0),
        });
        harness.session.run(prompt).await.unwrap();
        assert_eq!(skill_loads(&harness.session), expected, "{prompt}");
    }
    // The run answered without any verified on-screen result, so the skill
    // it used is charged one failure (and ranks a little lower).
    let skill = memory.list_procedures(None, 10).unwrap().remove(0);
    let ranked = memory
        .search_skills_scored("open the requested folder in explorer", 1)
        .unwrap();
    assert_eq!(ranked[0].1.id, skill.id);
    assert!(ranked[0].0 < 1.0);
}

#[test]
fn a_delegated_plan_becomes_a_replayable_skill() {
    let arguments = json!({
        "goal": "open the System page",
        "target_hint": "list item \"System\"",
        "done_when": "heading \"System\" is visible",
        "allowed_operations": ["click"],
        "then": [{
            "goal": "open the Display page",
            "target_hint": "list item \"Display\"",
            "done_when": "heading \"Display\" is visible",
        }, {
            "goal": "open the Sound page",
            "target_hint": "list item \"Sound\"",
            "done_when": "heading \"Sound\" is visible",
        }],
    });
    let clicked = |label: &str| {
        json!([{"tool": "click_target", "ok": true, "outcome": "progress",
            "target": format!("\"{label}\" (list item, enabled)")}])
    };
    let result = json!({
        "status": "stuck",
        "observation": {"target": {"scope": "window", "app": "SystemSettings.exe"}},
        "subgoals": [
            {"goal": "open the System page", "done_when": "heading \"System\" is visible",
             "status": "done", "steps": clicked("System")},
            {"goal": "open the Display page", "done_when": "heading \"Display\" is visible",
             "status": "unverified", "steps": clicked("Display")},
            {"goal": "open the Sound page", "done_when": "heading \"Sound\" is visible",
             "status": "stuck", "steps": []},
        ],
    });
    let step = workflow_step("fast_actions", &arguments, &result).unwrap();
    assert_eq!(step["state_change_count"], 2);
    // Handed back at the third subgoal, but the first two were reached.
    assert_eq!(step["success"], true);
    assert_eq!(step["subgoals"].as_array().unwrap().len(), 2);
    let done = json!({"status": "done", "subgoals": result["subgoals"].as_array().unwrap()[..2]});
    let workflow = vec![workflow_step("fast_actions", &arguments, &done).unwrap()];
    let qualification = qualify_workflow(&workflow, &[]);
    assert!(qualification.eligible);
    let procedure =
        verified_procedure("open the display settings page", &workflow, &qualification).unwrap();
    let instructions = procedure
        .steps
        .iter()
        .map(|step| (step.tool.as_str(), step.instruction.as_str()))
        .collect::<Vec<_>>();
    assert_eq!(
        instructions,
        [
            (
                "fast_actions",
                "target_hint list item \"System\"; done_when heading \"System\" is visible"
            ),
            (
                "fast_actions",
                "target_hint list item \"Display\"; done_when heading \"Display\" is visible"
            ),
        ]
    );
}

#[test]
fn browser_clicks_that_change_the_page_make_a_learnable_workflow() {
    let page = |fingerprint: &str| json!({"fingerprint": fingerprint, "url": "https://example.test/", "title": "Example"});
    let mut workflow = Vec::new();
    for (tool, fingerprint) in [
        ("managed_browser_open", "page-a"),
        ("managed_browser_click", "page-a"),
        ("managed_browser_snapshot", "page-b"),
    ] {
        push_workflow_step(
            &mut workflow,
            workflow_step(tool, &json!({}), &page(fingerprint)).unwrap(),
        );
    }
    // Nothing changed the page yet: an unchanged click and a snapshot.
    assert!(workflow.iter().all(|step| step["state_change_count"] == 0));
    assert!(!qualify_workflow(&workflow, &[]).eligible);
    push_workflow_step(
        &mut workflow,
        workflow_step("managed_browser_click", &json!({}), &page("page-c")).unwrap(),
    );
    assert_eq!(workflow[3]["state_change_count"], 1);
    let qualification = qualify_workflow(&workflow, &[]);
    assert!(qualification.eligible);
    let procedure = verified_procedure(
        "open the example chapter in the browser",
        &workflow,
        &qualification,
    )
    .unwrap();
    assert!(
        procedure
            .steps
            .iter()
            .any(|step| step.tool == "managed_browser_click")
    );
}

struct StallingBrain {
    calls: AtomicU64,
}

#[async_trait]
impl Brain for StallingBrain {
    async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec!["mock".into()])
    }
    fn max_retries(&self) -> u32 {
        2
    }
    fn stream(&self, _request: BrainRequest) -> crate::brain::BrainStream {
        if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
            // The first request never answers.
            Box::pin(futures::stream::pending())
        } else {
            Box::pin(futures::stream::iter(vec![
                Ok(BrainEvent::TextDelta {
                    text: "Answered after a retry.".into(),
                }),
                Ok(BrainEvent::Finished {
                    reason: Some("stop".into()),
                }),
            ]))
        }
    }
}

#[tokio::test(start_paused = true)]
async fn a_silent_provider_stream_is_retried_instead_of_hanging() {
    let mut harness = fast_harness(
        fast_router(usize::MAX, 0.99),
        Vec::new(),
        crate::config::DecisionRouterMode::Delegated,
    );
    let brain = Arc::new(StallingBrain {
        calls: AtomicU64::new(0),
    });
    harness.session.brain = brain.clone();
    let result = harness.session.run("say hello").await.unwrap();
    assert_eq!(result.answer, "Answered after a retry.");
    assert_eq!(brain.calls.load(Ordering::SeqCst), 2);
}

#[test]
fn a_confirmed_browser_plan_step_counts_as_a_state_change() {
    let result = json!({
        "status": "done",
        "goal": "open the chapter",
        "done_when": "page title shows \"Chapter Four\"",
        "steps": [{"tool": "managed_browser_click", "ok": true, "outcome": "observed",
            "target": "\"4. Chapter Four\" (browser link, visible)"}],
    });
    let arguments = json!({"goal": "open the chapter", "target_hint": "link \"4. Chapter Four\"",
        "done_when": "page title shows \"Chapter Four\""});
    let workflow = vec![workflow_step("fast_actions", &arguments, &result).unwrap()];
    assert_eq!(workflow[0]["state_change_count"], 1);
    let qualification = qualify_workflow(&workflow, &[]);
    assert!(qualification.eligible);
    let procedure = verified_procedure(
        "open chapter four of the example book",
        &workflow,
        &qualification,
    )
    .unwrap();
    assert_eq!(
        procedure.steps[0].instruction,
        "target_hint link \"4. Chapter Four\"; done_when page title shows \"Chapter Four\""
    );
}

#[tokio::test]
async fn an_unusable_helper_reply_is_retried_at_the_light_reasoning_level() {
    struct EffortBrain {
        efforts: parking_lot::Mutex<Vec<Option<String>>>,
    }
    #[async_trait]
    impl Brain for EffortBrain {
        async fn list_models(&self) -> Result<Vec<String>> {
            Ok(vec!["mock".into()])
        }
        fn stream(&self, request: BrainRequest) -> crate::brain::BrainStream {
            let effort = request.reasoning_effort.clone();
            self.efforts.lock().push(effort.clone());
            // At the user's maximum level this model runs out of room.
            let events = if effort.as_deref() == Some("max") {
                vec![Ok(BrainEvent::Finished {
                    reason: Some("length".into()),
                })]
            } else {
                vec![
                    Ok(BrainEvent::TextDelta {
                        text: "{\"proposals\":[]}".into(),
                    }),
                    Ok(BrainEvent::Finished {
                        reason: Some("stop".into()),
                    }),
                ]
            };
            Box::pin(futures::stream::iter(events))
        }
    }
    let brain = Arc::new(EffortBrain {
        efforts: Default::default(),
    });
    let reasoning = HelperReasoning {
        chosen: Some("max".into()),
        fallback: Some("low".into()),
    };
    let envelope = curate_turn(
        brain.clone(),
        "mock",
        &reasoning,
        "open the folder",
        "Opened.",
        &[],
        &qualify_workflow(&[], &[]),
        &[],
        &CancellationToken::new(),
    )
    .await
    .unwrap();
    assert!(envelope.proposals.is_empty());
    assert_eq!(
        *brain.efforts.lock(),
        [Some("max".to_owned()), Some("low".to_owned())]
    );
}

#[tokio::test]
async fn a_follow_up_that_cancels_review_still_keeps_the_learned_skill() {
    let harness = fast_harness(
        fast_router(usize::MAX, 0.99),
        Vec::new(),
        crate::config::DecisionRouterMode::Delegated,
    );
    let cancellation = CancellationToken::new();
    harness.session.schedule_curation(
        "open the requested folder in explorer".into(),
        "Opened the folder.".into(),
        explorer_workflow(),
        Vec::new(),
        true,
        RunMetrics::default(),
        cancellation.clone(),
    );
    // The user sends a follow-up right away.
    cancellation.cancel();
    harness
        .session
        .finish_background_work(Duration::from_secs(10))
        .await;
    let skills = harness
        .session
        .context
        .memory
        .list_procedures(None, 10)
        .unwrap();
    assert_eq!(skills.len(), 1);
    assert_eq!(skills[0].success_count, 1);
}

#[test]
fn a_command_that_edits_an_open_document_is_flagged() {
    let window = |title: &str, app: &str| crate::types::WindowInfo {
        id: "w".into(),
        title: title.into(),
        process_name: app.into(),
        bounds: crate::types::Rect {
            x: 0,
            y: 0,
            width: 800,
            height: 600,
        },
        elevated: false,
        visible: true,
        minimized: false,
    };
    let windows = vec![
        window("Report_Q3.xlsx - LibreOffice Calc", "soffice.bin"),
        window("Documents - File Explorer", "explorer.exe"),
    ];
    let command = json!({"command": "python -c \"import openpyxl; wb=openpyxl.load_workbook(r'C:\\Users\\Example\\Downloads\\Report_Q3.xlsx')\""});
    let documents = referenced_document_names("run_command", &command);
    assert_eq!(documents, ["Report_Q3.xlsx"]);
    let warning = open_document_warning(&documents, &windows).unwrap();
    assert_eq!(warning["open_in"][0]["app"], "soffice.bin");
    // A file nobody has open is fine.
    let other = referenced_document_names("write_file", &json!({"path": "C:\\data\\notes.txt"}));
    assert!(open_document_warning(&other, &windows).is_none());
    // The warning survives trimming of a large result.
    let big = json!({"stdout": "x".repeat(200_000), "open_document_warning": warning});
    let projected = project_tool_result_for_model("run_command", big);
    assert!(projected.get("open_document_warning").is_some());
}

#[test]
fn only_turns_that_act_count_as_steps() {
    for tool in [
        "fast_actions",
        "click_target",
        "run_command",
        "managed_browser_click",
        "type_text",
    ] {
        assert!(super::run_loop::is_acting_tool(tool), "{tool}");
    }
    for tool in [
        "capture_screen",
        "update_task_plan",
        "managed_browser_snapshot",
        "remember_fact",
        "read_file",
    ] {
        assert!(!super::run_loop::is_acting_tool(tool), "{tool}");
    }
}

/// A planner that clicks whenever it may, and otherwise answers.
struct KeepsClickingBrain {
    requests: AtomicU64,
    tool_requests: AtomicU64,
}

#[async_trait]
impl Brain for KeepsClickingBrain {
    async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec!["mock".into()])
    }
    fn stream(&self, request: BrainRequest) -> crate::brain::BrainStream {
        let index = self.requests.fetch_add(1, Ordering::SeqCst);
        let events = if request.tools.iter().any(|tool| tool.name == "click_target") {
            self.tool_requests.fetch_add(1, Ordering::SeqCst);
            vec![
                Ok(BrainEvent::ToolCall {
                    call: CompletedToolCall {
                        id: format!("call-{index}"),
                        name: "click_target".into(),
                        arguments: json!({"target_id": "1", "expected_label": "Next"}),
                    },
                }),
                Ok(BrainEvent::Finished {
                    reason: Some("tool_calls".into()),
                }),
            ]
        } else {
            vec![
                Ok(BrainEvent::TextDelta {
                    text: "Stopped at the step budget; clicked Next twice.".into(),
                }),
                Ok(BrainEvent::Finished {
                    reason: Some("stop".into()),
                }),
            ]
        };
        Box::pin(futures::stream::iter(events))
    }
}

#[tokio::test]
async fn the_run_stops_acting_when_the_step_budget_is_used() {
    let mut harness = fast_harness(
        fast_router(usize::MAX, 0.99),
        vec![fast_link("1", "Next")],
        crate::config::DecisionRouterMode::Delegated,
    );
    // Without a router the planner acts directly.
    harness.session.decision_router = None;
    harness.session.decision_router_config = None;
    let brain = Arc::new(KeepsClickingBrain {
        requests: AtomicU64::new(0),
        tool_requests: AtomicU64::new(0),
    });
    harness.session.brain = brain.clone();
    harness.session.action_step_budget = Some(2);
    let result = harness
        .session
        .run("keep going to the next page")
        .await
        .unwrap();
    assert!(result.answer.contains("step budget"), "{}", result.answer);
    assert_eq!(brain.tool_requests.load(Ordering::SeqCst), 2);
    assert_eq!(harness.clicks.lock().len(), 2);
}

#[test]
fn a_task_finished_by_a_verified_command_can_be_learned() {
    let command = workflow_step(
        "run_command",
        &json!({"command": "Set-Content -Path settings.json -Value '{}'"}),
        &json!({"exit_code": 0, "goal_action": {"status": "verified"}}),
    )
    .unwrap();
    let observe = workflow_step(
        "capture_screen",
        &json!({}),
        &json!({"target": {"scope": "window", "app": "Code.exe"}}),
    )
    .unwrap();
    let workflow = vec![observe, command];
    let qualification = qualify_workflow(&workflow, &[]);
    assert!(qualification.eligible);
    assert_eq!(qualification.evidence, "verified_command_outcome");
    // No UI state change, but the verified outcome passes the learning gate.
    assert_eq!(
        workflow_learning_rejection(
            "change the editor setting",
            &qualification,
            &workflow,
            &RunMetrics::default()
        ),
        None
    );
    // An unverified command alone is still not enough.
    let unverified = vec![
        workflow_step(
            "run_command",
            &json!({"command": "dir"}),
            &json!({"exit_code": 0}),
        )
        .unwrap(),
    ];
    assert!(!qualify_workflow(&unverified, &[]).eligible);
}

#[tokio::test]
async fn system_1_types_the_planners_input_into_a_grounded_field() {
    let mut harness = fast_harness(
        fast_router(usize::MAX, 0.99),
        vec![fast_link("5", "Name Box"), fast_link("6", "Print")],
        crate::config::DecisionRouterMode::Delegated,
    );
    let (result, metrics) = run_plan(
        &mut harness,
        json!({
            "goal": "go to cell B2",
            "target_hint": "\"Name Box\"",
            "done_when": "\"execute_action_batch result 1\" is visible",
            "input": [{"text": "B2", "replace_existing": true}, {"key": "Enter"}],
        }),
    )
    .await;
    assert_eq!(result["status"], "done", "{result}");
    let batches = harness.batches.lock().clone();
    assert_eq!(batches.len(), 1);
    let steps = batches[0]["steps"].as_array().unwrap();
    // Focus and type in one guarded step, then the planner's key.
    assert_eq!(steps[0]["kind"], "type_text");
    assert_eq!(steps[0]["target_id"], "5");
    assert_eq!(steps[0]["text"], "B2");
    assert_eq!(steps[1], json!({"kind": "key", "key": "Enter"}));
    assert_eq!(metrics.extras["fast_actions_input_nodes"], 1);
    // No clicks were spent choosing a target.
    assert!(harness.clicks.lock().is_empty());
}

#[tokio::test]
async fn system_1_does_not_type_into_a_field_it_cannot_ground() {
    let mut harness = fast_harness(
        fast_router(usize::MAX, 0.99),
        vec![fast_link("6", "Print"), fast_link("7", "Save")],
        crate::config::DecisionRouterMode::Delegated,
    );
    let (result, _) = run_plan(
        &mut harness,
        json!({
            "goal": "type the total",
            "target_hint": "\"Total amount\"",
            "done_when": "\"1200\" is visible",
            "input": [{"text": "1200"}],
        }),
    )
    .await;
    assert_eq!(result["status"], "uncertain", "{result}");
    assert!(harness.batches.lock().is_empty());
}

#[tokio::test]
async fn keys_without_a_target_go_to_the_focused_control() {
    let mut harness = fast_harness(
        fast_router(usize::MAX, 0.99),
        vec![fast_link("6", "Print")],
        crate::config::DecisionRouterMode::Delegated,
    );
    let (result, _) = run_plan(
        &mut harness,
        json!({
            "goal": "select columns A to C",
            "done_when": "\"execute_action_batch result 1\" is visible",
            "input": [{"key": "Ctrl+Shift+F5"}, {"text": "A1:C1", "replace_existing": true}, {"key": "Enter"}],
        }),
    )
    .await;
    assert_eq!(result["status"], "done", "{result}");
    let batches = harness.batches.lock().clone();
    let steps = batches[0]["steps"].as_array().unwrap();
    assert_eq!(steps.len(), 3);
    assert_eq!(steps[0]["key"], "Ctrl+Shift+F5");
    assert!(steps[1].get("target_id").is_none());
}

#[test]
fn an_input_entry_needs_exactly_one_of_key_or_text() {
    let plan = |input: Value| {
        let args: FastActionsArgs = serde_json::from_value(json!({
            "goal": "enter a value", "done_when": "\"done\" is visible", "input": input,
        }))
        .unwrap();
        FastPlan::from_args(&args).map(|_| ())
    };
    assert!(plan(json!([{"key": "Enter"}])).is_ok());
    assert!(plan(json!([{"key": "Enter", "text": "x"}])).is_err());
    assert!(plan(json!([{}])).is_err());
    assert!(plan(json!(vec![json!({"key": "Tab"}); 41])).is_err());
}

#[test]
fn a_session_no_one_can_answer_does_not_invite_questions() {
    assert_eq!(SYSTEM_PROMPT.matches("{QUESTIONS}").count(), 1);
    let harness = fast_harness(
        fast_router(usize::MAX, 0.99),
        vec![fast_link("6", "Print")],
        crate::config::DecisionRouterMode::Delegated,
    );
    assert!(harness.session.context.questions.is_none());
    let system = harness
        .session
        .messages
        .iter()
        .filter(|message| message.role == "system")
        .flat_map(|message| &message.content)
        .filter_map(|part| match part {
            MessageContent::Text { text } => Some(text.as_str()),
            MessageContent::ImagePng { .. } => None,
        })
        .collect::<String>();
    assert!(system.contains(UNATTENDED_QUESTIONS));
    assert!(!system.contains("call ask_user_question"));
    assert!(!system.contains("{QUESTIONS}"));
}

fn popup_ui(
    name: &str,
    kind: &str,
    x: i32,
    y: i32,
    width: u32,
    height: u32,
) -> crate::types::UiElement {
    crate::types::UiElement {
        name: name.into(),
        control_type: kind.into(),
        automation_id: None,
        value: None,
        bounds: crate::types::Rect {
            x,
            y,
            width,
            height,
        },
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

fn popup_window(
    id: &str,
    title: &str,
    process: &str,
    width: u32,
    height: u32,
) -> crate::types::WindowInfo {
    crate::types::WindowInfo {
        id: id.into(),
        title: title.into(),
        process_name: process.into(),
        bounds: crate::types::Rect {
            x: 0,
            y: 0,
            width,
            height,
        },
        elevated: false,
        visible: true,
        minimized: false,
    }
}

fn popup_observation(
    window: crate::types::WindowInfo,
    ui_elements: Vec<crate::types::UiElement>,
) -> crate::types::Observation {
    crate::types::Observation {
        version: Uuid::new_v4(),
        captured_at: chrono::Utc::now(),
        foreground_window: Some(window),
        target: None,
        cursor: None,
        screenshots: Vec::new(),
        ocr: Vec::new(),
        ui_elements,
        targets: Vec::new(),
        timings_ms: BTreeMap::new(),
        warnings: Vec::new(),
    }
}

fn error_dialog() -> Vec<crate::types::UiElement> {
    vec![
        popup_ui("Save", "Button", 10, 10, 40, 20),
        popup_ui("Invalid Entry", "Window", 200, 200, 300, 150),
        popup_ui("Invalid reference in formula.", "Text", 220, 230, 200, 20),
        popup_ui("OK", "Button", 300, 310, 60, 24),
        popup_ui("Help", "Button", 370, 310, 60, 24),
    ]
}

#[test]
fn a_dialog_that_opened_during_a_step_is_read_for_the_planner() {
    let sheet = popup_window("w1", "Budget - Example Calc", "soffice.bin", 800, 600);
    let baseline = PopupBaseline::of(Some(&popup_observation(sheet.clone(), Vec::new())));
    let now = popup_observation(sheet, error_dialog());
    let popup = unexpected_popup(
        &now,
        &baseline,
        "go to cell B2 \"Name Box\" \"B2\" is selected",
    )
    .unwrap()
    .value();
    assert_eq!(popup["title"], "Invalid Entry");
    assert_eq!(popup["text"], "Invalid reference in formula.");
    // Only the dialog's own buttons, not the window's toolbar behind it.
    assert_eq!(popup["buttons"], json!(["OK", "Help"]));
}

#[test]
fn dialogs_the_step_started_with_or_the_plan_names_are_expected() {
    let sheet = popup_window("w1", "Budget - Example Calc", "soffice.bin", 800, 600);
    let now = popup_observation(sheet.clone(), error_dialog());
    // Already open when the step started: part of the task.
    assert!(unexpected_popup(&now, &PopupBaseline::of(Some(&now)), "press OK").is_none());
    let baseline = PopupBaseline::of(Some(&popup_observation(sheet, Vec::new())));
    // The plan quotes text the dialog shows, or names the dialog itself.
    assert!(unexpected_popup(&now, &baseline, "confirm \"Invalid reference\" is shown").is_none());
    assert!(unexpected_popup(&now, &baseline, "close the invalid entry dialog").is_none());
}

#[test]
fn a_dialog_window_of_the_same_application_counts_but_menus_and_other_apps_do_not() {
    let sheet = popup_window("w1", "Budget - Example Calc", "soffice.bin", 800, 600);
    let baseline = PopupBaseline::of(Some(&popup_observation(sheet, Vec::new())));
    let dialog = popup_window("w2", "Keep current format?", "soffice.bin", 400, 200);
    let popup =
        unexpected_popup(&popup_observation(dialog, Vec::new()), &baseline, "save").unwrap();
    assert_eq!(popup.title, "Keep current format?");
    assert_eq!(popup.window_id.as_deref(), Some("w2"));
    assert!(popup.needs_closer_look());
    let menu = popup_window("w3", "", "soffice.bin", 200, 300);
    assert!(
        unexpected_popup(&popup_observation(menu, Vec::new()), &baseline, "open menu").is_none()
    );
    let other = popup_window("w4", "Example Notes", "notepad.exe", 400, 200);
    assert!(unexpected_popup(&popup_observation(other, Vec::new()), &baseline, "save").is_none());
}

/// Typing that raises a dialog in the mock window.
struct DialogRaisingBatch;

#[async_trait]
impl crate::tool::Tool for DialogRaisingBatch {
    fn name(&self) -> &'static str {
        "execute_action_batch"
    }
    fn description(&self) -> &'static str {
        "test batch"
    }
    fn input_schema(&self) -> Value {
        json!({"type": "object"})
    }
    fn risk(&self) -> crate::policy::RiskClass {
        crate::policy::RiskClass::ReadOnly
    }
    async fn execute(&self, _arguments: Value, context: &ToolContext) -> Result<Value> {
        if let Some(observation) = context.latest_observation.lock().as_mut() {
            observation.ui_elements.extend(error_dialog());
        }
        Ok(json!({"ok": true}))
    }
}

#[tokio::test]
async fn system_1_hands_back_a_dialog_its_typing_raised_with_what_it_says() {
    let mut harness = fast_harness(
        fast_router(usize::MAX, 0.99),
        vec![fast_link("6", "Print")],
        crate::config::DecisionRouterMode::Delegated,
    );
    let mut tools = ToolRegistry::default();
    tools.register(DialogRaisingBatch);
    harness.session.tools = Arc::new(tools);
    let (result, metrics) = run_plan(
        &mut harness,
        json!({
            "goal": "enter the formula",
            "done_when": "\"1200\" is visible",
            "input": [{"text": "=SUM(A1:A)"}, {"key": "Enter"}],
        }),
    )
    .await;
    assert_eq!(result["status"], "popup", "{result}");
    assert_eq!(result["popup"]["text"], "Invalid reference in formula.");
    assert_eq!(result["popup"]["buttons"], json!(["OK", "Help"]));
    assert_eq!(metrics.extras["fast_actions_popups"], 1);
}

fn fast_ocr_text(id: &str, name: &str, x: i32) -> crate::types::InteractionTarget {
    let mut target = fast_link_at(id, name, x, 0);
    target.control_type = "text".into();
    target.source = crate::types::TargetSource::Ocr;
    target.actionable = false;
    target
}

#[tokio::test]
async fn system_1_clicks_a_quoted_menu_label_that_only_ocr_can_read() {
    // Applications such as LibreOffice expose their menu bar only as text.
    let mut harness = fast_harness(
        fast_router(usize::MAX, 0.99),
        vec![
            fast_ocr_text("15", "Format", 0),
            fast_ocr_text("19", "Tools", 100),
        ],
        crate::config::DecisionRouterMode::Delegated,
    );
    let (result, _) = run_plan(
        &mut harness,
        json!({
            "goal": "open the Format menu",
            "target_hint": "menu \"Format\"",
            "done_when": "\"click_target result 1\" is visible",
        }),
    )
    .await;
    assert_eq!(result["status"], "done", "{result}");
    let clicks = harness.clicks.lock().clone();
    assert_eq!(clicks.len(), 1);
    assert_eq!(clicks[0]["target_id"], "15");
}

#[tokio::test]
async fn system_1_does_not_guess_between_repeated_ocr_text() {
    let mut harness = fast_harness(
        fast_router(usize::MAX, 0.99),
        vec![
            fast_ocr_text("17", "Sheet", 0),
            fast_ocr_text("42", "Sheet", 300),
        ],
        crate::config::DecisionRouterMode::Delegated,
    );
    let (result, _) = run_plan(
        &mut harness,
        json!({
            "goal": "open the Sheet menu",
            "target_hint": "\"Sheet\"",
            "done_when": "\"Insert Sheet\" is visible",
        }),
    )
    .await;
    assert_ne!(result["status"], "done", "{result}");
    assert!(harness.clicks.lock().is_empty());
}

/// The task's application opened a dialog in front of its main window.
struct AppDialogDesktop {
    activations: Arc<parking_lot::Mutex<Vec<String>>>,
}

#[async_trait]
impl crate::platform::DesktopPlatform for AppDialogDesktop {
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
        Ok(Some(test_window("dialog", "Format Cells", "soffice.bin")))
    }
    async fn list_windows(&self) -> Result<Vec<crate::types::WindowInfo>> {
        Ok(vec![
            test_window("sheet", "Budget.xlsx - Example Calc", "soffice.bin"),
            test_window("dialog", "Format Cells", "soffice.bin"),
        ])
    }
    async fn activate_window(&self, window_id: &str) -> Result<crate::types::WindowInfo> {
        self.activations.lock().push(window_id.into());
        Ok(test_window(window_id, "window", "soffice.bin"))
    }
    async fn cursor_position(&self) -> Result<Option<(i32, i32)>> {
        Ok(None)
    }
    async fn simulate_input(&self, _action: &crate::types::InputAction) -> Result<()> {
        Ok(())
    }
}

#[tokio::test]
async fn a_dialog_the_task_app_opens_is_kept_not_switched_away_from() {
    let mut harness = fast_harness(
        fast_router(0, 0.99),
        Vec::new(),
        crate::config::DecisionRouterMode::Delegated,
    );
    let activations = Arc::new(parking_lot::Mutex::new(Vec::new()));
    harness.session.context.platform = Arc::new(AppDialogDesktop {
        activations: activations.clone(),
    });
    let mut metrics = RunMetrics::default();
    harness
        .session
        .recover_environment_change(
            1,
            test_window("sheet", "Budget.xlsx - Example Calc", "soffice.bin"),
            Some(test_window("dialog", "Format Cells", "soffice.bin")),
            &mut metrics,
        )
        .await
        .unwrap();
    // Switching back to the sheet would close over the agent's own dialog.
    assert!(!activations.lock().contains(&"sheet".to_owned()));
    let trace =
        std::fs::read_to_string(harness.session.context.artifact_dir.join("trace.jsonl")).unwrap();
    let recovered = trace
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .find(|event| event["kind"] == "environment_recovered")
        .expect("recovery logged");
    assert_eq!(recovered["payload"]["action"], "observe_current");
    assert_eq!(recovered["payload"]["used_jev"], false);
}

#[test]
fn typed_text_counts_as_seen_only_when_it_newly_shows_on_screen() {
    use crate::decision::typed_text_visible;
    // In the formula bar after typing, with a misread "0" as "O".
    assert_eq!(
        typed_text_visible("Name Box A1", "Name Box A1 =SUM(A1:A1O)", "=SUM(A1:A10)"),
        Some(true)
    );
    assert_eq!(
        typed_text_visible("Name Box A1", "Name Box A2", "=SUM(A1:A10)"),
        Some(false)
    );
    // Already on screen before typing proves nothing; too short to tell.
    assert_eq!(typed_text_visible("Total 1200", "Total 1200", "1200"), None);
    assert_eq!(typed_text_visible("", "B2", "B2"), None);
}

/// A batch that reports a failed step inside an OK result.
struct FailingStepBatch;

#[async_trait]
impl crate::tool::Tool for FailingStepBatch {
    fn name(&self) -> &'static str {
        "execute_action_batch"
    }
    fn description(&self) -> &'static str {
        "test batch"
    }
    fn input_schema(&self) -> Value {
        json!({"type": "object"})
    }
    fn risk(&self) -> crate::policy::RiskClass {
        crate::policy::RiskClass::ReadOnly
    }
    async fn execute(&self, _arguments: Value, _context: &ToolContext) -> Result<Value> {
        Ok(json!({
            "success": false,
            "failed_at_step": 1,
            "error": "input action did not verify",
            "step_results": [],
        }))
    }
}

#[tokio::test]
async fn a_batch_step_that_failed_is_not_reported_as_typed() {
    let mut harness = fast_harness(
        fast_router(usize::MAX, 0.99),
        vec![fast_link("6", "Print")],
        crate::config::DecisionRouterMode::Delegated,
    );
    let mut tools = ToolRegistry::default();
    tools.register(FailingStepBatch);
    harness.session.tools = Arc::new(tools);
    let (result, _) = run_plan(
        &mut harness,
        json!({
            "goal": "enter the total",
            "done_when": "\"1200\" is visible",
            "input": [{"text": "1200"}, {"key": "Enter"}],
        }),
    )
    .await;
    assert_eq!(result["status"], "input_failed", "{result}");
    assert!(
        result["reason"]
            .as_str()
            .unwrap()
            .contains("did not verify")
    );
}

#[tokio::test]
async fn system_1_right_clicks_the_quoted_target() {
    let mut harness = fast_harness(
        fast_router(usize::MAX, 0.99),
        vec![fast_ocr_text("30", "Sheet1", 0)],
        crate::config::DecisionRouterMode::Delegated,
    );
    let (result, _) = run_plan(
        &mut harness,
        json!({
            "goal": "open the sheet tab's context menu",
            "target_hint": "sheet tab \"Sheet1\"",
            "done_when": "\"click_target result 1\" is visible",
            "allowed_operations": ["right_click"],
        }),
    )
    .await;
    assert_eq!(result["status"], "done", "{result}");
    let clicks = harness.clicks.lock().clone();
    assert_eq!(clicks[0]["target_id"], "30");
    assert_eq!(clicks[0]["button"], "right");
}

#[tokio::test]
async fn system_1_hovers_and_drags_between_quoted_labels() {
    let mut harness = fast_harness(
        fast_router(usize::MAX, 0.99),
        vec![
            fast_ocr_text("15", "Format", 0),
            fast_ocr_text("31", "Sheet2", 200),
            fast_ocr_text("30", "Sheet1", 100),
        ],
        crate::config::DecisionRouterMode::Delegated,
    );
    let hovers = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let drags = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let mut tools = ToolRegistry::default();
    for (name, calls) in [("hover_target", &hovers), ("drag_target", &drags)] {
        tools.register(RecordingInputTool {
            name,
            calls: calls.clone(),
            static_screen: harness.static_screen.clone(),
        });
    }
    harness.session.tools = Arc::new(tools);
    let (hovered, _) = run_plan(
        &mut harness,
        json!({
            "goal": "show the Format menu's tooltip",
            "target_hint": "\"Format\"",
            "done_when": "\"hover_target result 1\" is visible",
            "allowed_operations": ["hover"],
        }),
    )
    .await;
    assert_eq!(hovered["status"], "done", "{hovered}");
    assert_eq!(hovers.lock()[0]["target_id"], "15");
    assert!(hovers.lock()[0].get("button").is_none());

    let (dragged, metrics) = run_plan(
        &mut harness,
        json!({
            "goal": "move Sheet2 before Sheet1",
            "target_hint": "drag \"Sheet2\" onto \"Sheet1\"",
            "done_when": "\"drag_target result 1\" is visible",
            "allowed_operations": ["drag"],
        }),
    )
    .await;
    assert_eq!(dragged["status"], "done", "{dragged}");
    let drag = drags.lock()[0].clone();
    assert_eq!(drag["source_target_id"], "31");
    assert_eq!(drag["destination_target_id"], "30");
    assert_eq!(metrics.extras["fast_actions_drag_nodes"], 1);
}

#[tokio::test]
async fn a_drag_without_both_labels_found_once_does_nothing() {
    let mut harness = fast_harness(
        fast_router(usize::MAX, 0.99),
        vec![fast_ocr_text("31", "Sheet2", 200)],
        crate::config::DecisionRouterMode::Delegated,
    );
    let drags = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let mut tools = ToolRegistry::default();
    tools.register(RecordingInputTool {
        name: "drag_target",
        calls: drags.clone(),
        static_screen: harness.static_screen.clone(),
    });
    harness.session.tools = Arc::new(tools);
    let (result, _) = run_plan(
        &mut harness,
        json!({
            "goal": "move Sheet2 before Sheet1",
            "target_hint": "drag \"Sheet2\" onto \"Sheet1\"",
            "done_when": "\"Sheet2\" is left of \"Sheet1\"",
            "allowed_operations": ["drag"],
        }),
    )
    .await;
    assert_eq!(result["status"], "uncertain", "{result}");
    assert!(drags.lock().is_empty());
}

#[test]
fn a_short_plain_hint_names_its_label_exactly_when_it_is_on_screen() {
    let sheet = popup_window("w1", "Budget - Example Calc", "soffice.bin", 800, 600);
    let mut observation = popup_observation(sheet, Vec::new());
    observation.targets = vec![fast_link("7", "OK"), fast_link("8", "Discard All")];
    let screen = Some(&observation);
    assert_eq!(plain_label_as_quoted("OK", screen), "\"OK\"");
    assert_eq!(
        plain_label_as_quoted(" Discard All ", screen),
        "\"Discard All\""
    );
    // A short description that names no visible label is left alone.
    assert_eq!(
        plain_label_as_quoted("display page", screen),
        "display page"
    );
    // Already quoted, or a longer description: unchanged.
    assert_eq!(
        plain_label_as_quoted("menu \"File\"", screen),
        "menu \"File\""
    );
    let described = "the OK button of the dialog below";
    assert_eq!(plain_label_as_quoted(described, screen), described);
    assert_eq!(plain_label_as_quoted("OK", None), "OK");
}

#[tokio::test]
async fn system_1_takes_a_fresh_look_when_the_input_guard_asks_for_one() {
    let mut harness = fast_harness(
        fast_router(usize::MAX, 0.99),
        vec![fast_ocr_text("8", "File", 0)],
        crate::config::DecisionRouterMode::Delegated,
    );
    // Earlier input made no verified progress: more clicks need a new look.
    harness.session.continuity.grounding_required = true;
    let captures = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let mut tools = ToolRegistry::default();
    tools.register(RecordingInputTool {
        name: "click_target",
        calls: harness.clicks.clone(),
        static_screen: harness.static_screen.clone(),
    });
    tools.register(RecordingInputTool {
        name: "capture_screen",
        calls: captures.clone(),
        static_screen: Arc::new(std::sync::atomic::AtomicBool::new(true)),
    });
    harness.session.tools = Arc::new(tools);
    let (result, metrics) = run_plan(
        &mut harness,
        json!({
            "goal": "open the File menu",
            "target_hint": "\"File\"",
            "done_when": "\"click_target result 1\" is visible",
        }),
    )
    .await;
    assert_eq!(result["status"], "done", "{result}");
    assert_eq!(captures.lock().len(), 1);
    assert_eq!(harness.clicks.lock()[0]["target_id"], "8");
    assert_eq!(metrics.extras["fast_actions_regrounds"], 1);
}

#[tokio::test]
async fn without_a_target_hint_the_goals_quoted_label_names_the_target() {
    let mut harness = fast_harness(
        fast_router(usize::MAX, 0.99),
        vec![
            fast_ocr_text("12", "Table", 0),
            fast_ocr_text("13", "Tools", 100),
        ],
        crate::config::DecisionRouterMode::Delegated,
    );
    let (result, _) = run_plan(
        &mut harness,
        json!({
            "goal": "open the \"Table\" menu",
            "done_when": "\"click_target result 1\" is visible",
        }),
    )
    .await;
    assert_eq!(result["status"], "done", "{result}");
    assert_eq!(harness.clicks.lock()[0]["target_id"], "12");
}

#[test]
fn plans_with_nested_item_wrappers_and_single_entries_are_repaired() {
    // Shapes a planner produced: doubly wrapped input, a branch's then given
    // as one wrapped object, a then given as one object, and a wrapped
    // operations list.
    let mut plan = json!({
        "goal": "format the essay",
        "done_when": "\"12 pt\" is shown",
        "allowed_operations": {"item": ["click", "hover"]},
        "branches": [{
            "when": "otherwise",
            "then": {"item": {
                "goal": "type 12 into the font size box",
                "done_when": "the font size box shows 12 pt",
                "input": {"item": {"item": [{"text": "12", "replace_existing": "true"}, {"key": "Enter"}]}},
                "target_hint": "\"Font Size\"",
            }},
        }],
        "then": {"goal": "save", "done_when": "saved", "input": {"item": [{"key": "Ctrl+S"}]}},
    });
    repair_fast_plan(&mut plan, true);
    let args: FastActionsArgs = serde_json::from_value(plan).expect("repaired plan parses");
    assert_eq!(args.allowed_operations.len(), 2);
    assert_eq!(args.then.len(), 1);
    assert_eq!(args.then[0].input.len(), 1);
    assert_eq!(args.branches[0].then[0].input.len(), 2);
    assert_eq!(
        args.branches[0].then[0].input[0].text.as_deref(),
        Some("12")
    );
}

#[tokio::test]
async fn clicking_the_only_named_target_continues_the_plan_instead_of_handing_back() {
    // A menu label that only OCR reads is the only candidate. After the click
    // the menu is open (not quoted, so not provable locally): the next step
    // must still run rather than the plan stopping with no candidates.
    let mut harness = fast_harness(
        fast_router(usize::MAX, 0.2),
        vec![fast_ocr_text("12", "Table", 0)],
        crate::config::DecisionRouterMode::Delegated,
    );
    let (result, _) = run_plan(
        &mut harness,
        json!({
            "goal": "open the Table menu",
            "target_hint": "\"Table\"",
            "done_when": "the Table menu is open",
            "then": [{"goal": "convert", "target_hint": "\"click_target result 1\"", "done_when": "\"click_target result 2\" is visible"}],
        }),
    )
    .await;
    let first = &result["subgoals"][0];
    assert_eq!(first["status"], "unverified", "{result}");
    // The plan went on to its next step.
    assert_eq!(result["subgoals"][1]["goal"], "convert", "{result}");
}

#[tokio::test]
async fn the_primary_models_fix_after_a_hand_back_becomes_a_training_record() {
    let mut harness = fast_harness(
        fast_router(usize::MAX, 0.99),
        vec![fast_link("6", "Print")],
        crate::config::DecisionRouterMode::Delegated,
    );
    let dir = tempfile::tempdir().unwrap();
    let recorder = Arc::new(crate::router_training::TrainingRecorder::new(
        dir.path(),
        harness.session.id,
        &[],
    ));
    harness.session.training = Some(recorder.clone());
    // System 1 cannot find the named target and hands back.
    let (result, _) = run_plan(
        &mut harness,
        json!({"goal": "open the chart wizard", "target_hint": "\"Chart Wizard\"", "done_when": "\"Chart Type\" is visible"}),
    )
    .await;
    assert_ne!(result["status"], "done", "{result}");
    assert!(harness.session.pending_handback.is_some());
    // A failed attempt by the primary model is not the answer...
    harness.session.record_handback(
        "click_localized",
        &json!({"localization_id": "loc_1"}),
        "no_progress",
        &Ok(json!({})),
    );
    // ...the action that makes progress is.
    harness.session.record_handback(
        "click_target",
        &json!({"observation_id": "obs_1", "target_id": "9", "expected_label": "Insert Chart"}),
        "progress",
        &Ok(json!({})),
    );
    assert!(harness.session.pending_handback.is_none());
    let records = std::fs::read_to_string(recorder.path())
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .filter(|record| record["kind"] == "handback")
        .collect::<Vec<_>>();
    assert_eq!(records.len(), 1, "{records:?}");
    let record = &records[0];
    assert_eq!(record["status"], result["status"]);
    assert_eq!(record["target_hint"], "\"Chart Wizard\"");
    assert_eq!(record["resolution"]["tool"], "click_target");
    assert_eq!(record["resolution"]["label"], "Insert Chart");
    assert!(
        !record["resolution"]["arguments"]
            .as_str()
            .unwrap()
            .contains("obs_1")
    );
}

#[tokio::test]
async fn a_hand_back_the_primary_model_never_resolves_leaves_no_record() {
    let mut harness = fast_harness(
        fast_router(usize::MAX, 0.99),
        vec![fast_link("6", "Print")],
        crate::config::DecisionRouterMode::Delegated,
    );
    let dir = tempfile::tempdir().unwrap();
    let recorder = Arc::new(crate::router_training::TrainingRecorder::new(
        dir.path(),
        harness.session.id,
        &[],
    ));
    harness.session.training = Some(recorder.clone());
    run_plan(
        &mut harness,
        json!({"goal": "open the chart wizard", "target_hint": "\"Chart Wizard\"", "done_when": "\"Chart Type\" is visible"}),
    )
    .await;
    for _ in 0..3 {
        harness.session.record_handback(
            "click_target",
            &json!({"target_id": "9", "expected_label": "Insert Chart"}),
            "no_progress",
            &Ok(json!({})),
        );
    }
    assert!(harness.session.pending_handback.is_none());
    let text = std::fs::read_to_string(recorder.path()).unwrap_or_default();
    assert!(!text.contains("\"handback\""));
}

fn replay_step(tool: &str, instruction: &str) -> ProcedureStep {
    ProcedureStep {
        tool: tool.into(),
        instruction: instruction.into(),
    }
}

fn replay_skill_record(
    success_count: u64,
    applications: &[&str],
) -> crate::memory::ProcedureRecord {
    crate::memory::ProcedureRecord {
        id: Uuid::new_v4(),
        kind: ProcedureKind::Workflow,
        task_signature: "open system settings".into(),
        title: "Open System settings".into(),
        summary: "Open the System page.".into(),
        applications: applications.iter().map(|app| (*app).to_owned()).collect(),
        steps: vec![
            replay_step(
                "observe_desktop",
                "Discover the current desktop and task applications.",
            ),
            replay_step(
                "fast_actions",
                "target_hint \"System\"; done_when \"Advanced display\" is visible",
            ),
            replay_step(
                "execute_action_batch",
                "Execute a freshly grounded action sequence using current task values.",
            ),
        ],
        command_template: None,
        evidence: String::new(),
        enabled: true,
        success_count,
        failure_count: 0,
        // What a verified run of this skill recorded: open "System", then
        // "Display".
        program: program_from_tape(
            &[json!({"click": "System"}), json!({"click": "Display"})],
            "open system settings",
        ),
        retrieval_count: 0,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        fingerprint: "replay-test".into(),
    }
}

#[test]
fn skill_trust_follows_successes_and_failures() {
    let mut skill = replay_skill_record(0, &["chat.exe"]);
    assert!((skill_trust(&skill) - 0.5).abs() < 1e-9);
    skill.success_count = 2;
    assert!(skill_trust(&skill) >= 0.6);
    skill.failure_count = 3;
    assert!(skill_trust(&skill) < 0.6);
}

/// A fast harness whose tool registry also answers the activation and
/// capture the replay performs, on a desktop showing the observed window.
fn replay_harness() -> FastHarness {
    let mut harness = fast_harness(
        fast_router(usize::MAX, 0.99),
        vec![fast_link("t1", "System"), fast_link("t2", "Bluetooth")],
        crate::config::DecisionRouterMode::Delegated,
    );
    let observation = harness
        .session
        .context
        .latest_observation
        .lock()
        .clone()
        .unwrap();
    harness.session.context.platform = crate::platform::MockDesktop::new(observation);
    let mut tools = ToolRegistry::default();
    tools.register(RecordingInputTool {
        name: "click_target",
        calls: harness.clicks.clone(),
        static_screen: harness.static_screen.clone(),
    });
    // Launches are recorded in the harness's (otherwise unused) batch list.
    tools.register(RecordingInputTool {
        name: "open_application",
        calls: harness.batches.clone(),
        static_screen: harness.static_screen.clone(),
    });
    // Typing into a clicked field runs as a batch: recorded with the clicks.
    tools.register(RecordingInputTool {
        name: "execute_action_batch",
        calls: harness.clicks.clone(),
        static_screen: harness.static_screen.clone(),
    });
    for name in ["activate_window", "capture_screen"] {
        tools.register(RecordingInputTool {
            name,
            calls: Default::default(),
            static_screen: harness.static_screen.clone(),
        });
    }
    harness.session.tools = Arc::new(tools);
    harness
}

#[tokio::test]
async fn a_trusted_skill_is_replayed_by_system_one_before_the_planner() {
    let mut harness = replay_harness();
    let mut metrics = RunMetrics::default();
    let mut workflow = Vec::new();
    let used = harness
        .session
        .replay_skill(
            &replay_skill_record(2, &["chat.exe"]),
            0.8,
            "open system settings",
            &mut metrics,
            &mut workflow,
        )
        .await
        .unwrap();
    assert_eq!(used, 1);
    assert_eq!(harness.clicks.lock().len(), 1);
    assert_eq!(metrics.extras["skill_replays"], json!(1));
    assert_eq!(metrics.extras["skill_replay_steps_completed"], json!(1));
    assert_eq!(workflow.len(), 1);
    let note = match harness.session.messages.last().unwrap().content.first() {
        Some(MessageContent::Text { text }) => text.clone(),
        _ => String::new(),
    };
    assert!(note.contains("System 1 replayed"), "{note}");
    assert!(note.contains("target_hint \"System\""), "{note}");
}

#[tokio::test]
async fn a_skill_is_not_replayed_when_unproven_or_its_application_is_closed() {
    for (skill, reason) in [
        (replay_skill_record(0, &["chat.exe"]), "unverified"),
        (
            replay_skill_record(2, &["notes.exe"]),
            "application_not_open",
        ),
    ] {
        let mut harness = replay_harness();
        let messages = harness.session.messages.len();
        let used = harness
            .session
            .replay_skill(
                &skill,
                0.8,
                "open system settings",
                &mut RunMetrics::default(),
                &mut Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(used, 0, "{reason}");
        assert!(harness.clicks.lock().is_empty(), "{reason}");
        assert_eq!(harness.session.messages.len(), messages, "{reason}");
        let trace =
            std::fs::read_to_string(harness.session.context.artifact_dir.join("trace.jsonl"))
                .unwrap_or_default();
        assert!(trace.contains(reason), "{reason}");
    }
}

#[test]
fn failure_lessons_stay_general_and_private_details_are_dropped() {
    let lesson = accepted_lesson(
        r#"{"lesson": "When changing a cell format in a spreadsheet, use the Format Cells dialog  instead of typing the format into the cell."}"#,
    )
    .unwrap()
    .unwrap();
    assert!(lesson.starts_with("Lesson from an earlier failed attempt: When changing"));
    assert!(!lesson.contains("  "));
    assert_eq!(accepted_lesson(r#"{"lesson": null}"#).unwrap(), None);
    // Task values and personal details never become a lesson.
    for private in [
        r#"{"lesson": "When saving, write to C:\\Users\\example\\report.xlsx."}"#,
        r#"{"lesson": "When emailing, send to someone@example.com."}"#,
        r#"{"lesson": "When filling the form, use order 5550123456."}"#,
    ] {
        assert_eq!(accepted_lesson(private).unwrap(), None, "{private}");
    }
    assert!(accepted_lesson("no json here").is_err());
}

#[tokio::test]
async fn system_one_finds_or_opens_the_application_a_skill_names() {
    let trace = |harness: &FastHarness| {
        std::fs::read_to_string(harness.session.context.artifact_dir.join("trace.jsonl"))
            .unwrap_or_default()
    };
    // A Store app records no process; the skill's "Open" step names it and
    // its window is titled with that name.
    let mut store_app = replay_skill_record(2, &[]);
    store_app
        .steps
        .insert(1, replay_step("open_application", "Open \"Chat\"."));
    // An older skill that names no application: its title does.
    let mut described = replay_skill_record(2, &[]);
    described.title = "Open System settings from the Chat app".into();
    for skill in [store_app, described] {
        let mut harness = replay_harness();
        let used = harness
            .session
            .replay_skill(
                &skill,
                0.8,
                "open system settings",
                &mut RunMetrics::default(),
                &mut Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(used, 1, "{}", skill.title);
        assert_eq!(harness.clicks.lock().len(), 1, "{}", skill.title);
        assert!(
            harness.batches.lock().is_empty(),
            "{}: nothing to open",
            skill.title
        );
    }
    // Not running: opening it is the skill's first motor step.
    let mut closed = replay_skill_record(2, &[]);
    closed
        .steps
        .insert(1, replay_step("open_application", "Open \"Notes\"."));
    let mut harness = replay_harness();
    let used = harness
        .session
        .replay_skill(
            &closed,
            0.8,
            "open system settings",
            &mut RunMetrics::default(),
            &mut Vec::new(),
        )
        .await
        .unwrap();
    assert_eq!(
        harness.batches.lock().as_slice(),
        [json!({"name": "Notes"})]
    );
    // The stub launches nothing, so no window appears and nothing is clicked.
    assert_eq!(used, 0);
    assert!(harness.clicks.lock().is_empty());
    assert!(trace(&harness).contains("application_did_not_open"));
}

#[test]
fn a_store_app_skill_records_the_application_by_window_name() {
    let step = |app: &str, title: &str| {
        workflow_step(
            "click_target",
            &json!({"target_id": "3"}),
            &json!({
                "focus": {"app": app, "title": title},
                "state_change": {"added_text": ["changed"]},
            }),
        )
        .unwrap()
    };
    let store = step("ApplicationFrameHost.exe", "Settings");
    assert_eq!(store.pointer("/focus/title"), Some(&json!("Settings")));
    // Other windows' titles can name private documents and are not kept.
    let document = step("Notepad.exe", "private-notes.txt - Notepad");
    assert!(document.pointer("/focus/title").is_none());
    let qualification = qualify_workflow(&[store, document], &[]);
    assert_eq!(qualification.applications, ["Notepad.exe", "Settings"]);
}

#[tokio::test]
async fn a_replay_never_clicks_anything_but_the_saved_label() {
    // The saved label is not on this screen; a confident fast model would
    // otherwise pick another target (a live trace clicked "Search").
    let mut skill = replay_skill_record(2, &["chat.exe"]);
    skill.program = program_from_tape(
        &[
            json!({"click": "Display"}),
            json!({"click": "Advanced display"}),
        ],
        "open system settings",
    );
    for targets in [
        vec![fast_link("t1", "System"), fast_link("t2", "Bluetooth")],
        // A single candidate is not clicked just for being the only one.
        vec![fast_link("t1", "System")],
    ] {
        let mut harness = replay_harness();
        if let Some(observation) = harness.session.context.latest_observation.lock().as_mut() {
            observation.targets = targets;
        }
        let used = harness
            .session
            .replay_skill(
                &skill,
                0.8,
                "open system settings",
                &mut RunMetrics::default(),
                &mut Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(used, 1);
        assert!(harness.clicks.lock().is_empty());
        assert!(!harness.session.skill_replay_active);
        let note = match harness.session.messages.last().unwrap().content.first() {
            Some(MessageContent::Text { text }) => text.clone(),
            _ => String::new(),
        };
        assert!(note.contains("Nothing was changed by the replay"), "{note}");
    }
}

#[tokio::test]
async fn a_replay_uses_the_application_window_not_the_desktop_or_a_dialog() {
    let window =
        |id: &str, title: &str, process: &str, width: u32, height: u32| crate::types::WindowInfo {
            id: id.into(),
            title: title.into(),
            process_name: process.into(),
            bounds: crate::types::Rect {
                x: 0,
                y: 0,
                width,
                height,
            },
            elevated: false,
            visible: true,
            minimized: false,
        };
    let names = ["explorer.exe".to_owned()];
    // The desktop shares File Explorer's process but is not its window.
    assert!(!skill_application_window(
        &window("1", "Program Manager", "explorer.exe", 1920, 1080),
        &names,
        ""
    ));
    assert!(skill_application_window(
        &window("2", "Documents - File Explorer", "explorer.exe", 800, 600),
        &names,
        ""
    ));
}

#[test]
fn plan_shapes_that_failed_in_the_arena_are_repaired() {
    let parse = |mut plan: Value| {
        repair_fast_plan(&mut plan, true);
        let args: FastActionsArgs =
            serde_json::from_value(plan.clone()).unwrap_or_else(|error| panic!("{error}: {plan}"));
        FastPlan::from_args(&args).unwrap_or_else(|error| panic!("{error}: {plan}"));
        args
    };
    // A wrapped list inside a list, and doubly wrapped operations in a
    // branch's wrapped step.
    let args = parse(json!({
        "goal": "open the folder",
        "done_when": "\"Example\" is visible",
        "input": [{"item": [{"key": "Ctrl+L"}, {"text": "C:\\Example", "replace_existing": "true"}, {"key": "Enter"}]}],
        "branches": [{
            "when": "\"Fonts\" is not shown",
            "then": {"item": {
                "allowed_operations": {"item": {"item": ["click", "scroll"]}},
                "goal": "open fonts",
                "done_when": "\"Fonts\" is shown",
                "target_hint": "\"Fonts\"",
            }},
        }],
        "then": [{
            "allowed_operations": {"item": {"item": ["click", "scroll"]}},
            "goal": "set the font",
            "done_when": "\"Example Serif\" is shown",
            "input": {"item": {"replace_existing": "true", "text": "Example Serif"}},
            "target_hint": "\"Default\"",
        }],
    }));
    assert_eq!(args.input.len(), 3);
    assert_eq!(args.then[0].allowed_operations.len(), 2);
    assert_eq!(args.branches[0].then[0].allowed_operations.len(), 2);
    // A missing goal or done_when is taken from the other; empty entries are
    // dropped; a plain string in input is a key or text.
    let args = parse(json!({
        "done_when": "\"Saved\" is visible",
        "allowed_operations": ["click", ""],
        "input": ["", "Ctrl+S", {"text": "notes"}],
        "then": [{"goal": "close the dialog", "target_hint": "\"OK\""}],
    }));
    assert_eq!(args.goal, "\"Saved\" is visible");
    assert_eq!(args.allowed_operations.len(), 1);
    assert_eq!(args.input.len(), 2);
    assert_eq!(args.input[0].key.as_deref(), Some("Ctrl+S"));
    assert_eq!(args.then[0].done_when, "close the dialog");
    // A long run of keys and text (a column of formulas) is one step.
    let mut input = Vec::new();
    for row in 2..=21 {
        input.push(json!({"text": format!("=B{row}-C{row}")}));
        input.push(json!({"key": "Enter"}));
    }
    let args = parse(
        json!({"goal": "fill the column", "done_when": "\"=B21-C21\" is entered", "input": input}),
    );
    assert_eq!(args.input.len(), 40);
}

#[test]
fn a_long_horizon_plan_fits_in_one_call() {
    let step = |index: usize| {
        json!({
            "goal": format!("open item {index}"),
            "target_hint": format!("\"Item {index}\""),
            "done_when": format!("\"Item {index}\" is selected"),
        })
    };
    let plan = |steps: usize| {
        let mut plan = step(0);
        plan["then"] = Value::Array((1..steps).map(step).collect());
        repair_fast_plan(&mut plan, true);
        let args: FastActionsArgs = serde_json::from_value(plan).unwrap();
        FastPlan::from_args(&args)
    };
    assert_eq!(plan(50).unwrap().total_nodes, 50);
    assert_eq!(plan(64).unwrap().total_nodes, 64);
    assert!(plan(65).is_err());
}

#[test]
fn a_run_records_its_motor_steps_from_every_acting_tool() {
    let click = motor_steps(
        "click_target",
        &json!({"target_id": "4", "expected_label": "Format"}),
        &json!({"action": {"label": "Format"}, "executed": true}),
    );
    assert_eq!(click, [json!({"click": "Format"})]);
    let key = motor_steps(
        "simulate_input",
        &json!({"kind": "key", "key": "Ctrl+D"}),
        &json!({"action": {"kind": "key", "key": "Ctrl+D"}}),
    );
    assert_eq!(key, [json!({"key": "Ctrl+D"})]);
    let batch = motor_steps(
        "execute_action_batch",
        &json!({"steps": [
            {"kind": "click", "expected_label": "Name Box"},
            {"kind": "type", "text": "B2:B11", "replace_existing": true},
            {"kind": "key", "key": "Enter"},
        ]}),
        &json!({"success": true}),
    );
    assert_eq!(
        batch,
        [
            json!({"click": "Name Box"}),
            json!({"text": "B2:B11", "replace": true}),
            json!({"key": "Enter"}),
        ]
    );
    // Looking leaves nothing; a vision click or a command cannot be repeated
    // exactly, so it ends a program there.
    assert!(motor_steps("capture_screen", &json!({}), &json!({})).is_empty());
    assert_eq!(
        motor_steps("click_localized", &json!({}), &json!({"executed": true})),
        [json!({"stop": "click_localized"})]
    );
    // An action that did not run leaves nothing.
    assert!(
        motor_steps(
            "click_target",
            &json!({"expected_label": "OK"}),
            &json!({"executed": false})
        )
        .is_empty()
    );
}

#[test]
fn a_program_keeps_request_text_only_as_a_fingerprint() {
    let tape = vec![
        json!({"click": "Name Box"}),
        json!({"text": "D2", "replace": true}),
        json!({"key": "Enter"}),
        json!({"text": "Example Report", "replace": true}),
        json!({"text": "=B2-C2", "replace": false}),
        json!({"text": "someone@example.com", "replace": false}),
    ];
    let program = program_from_tape(
        &tape,
        "Name the sheet Example Report and add a profit column",
    )
    .unwrap();
    let steps = program["steps"].as_array().unwrap();
    // Text the request supplied is not stored, only found again later.
    assert!(steps[3].get("text").is_none());
    assert_eq!(steps[3]["request_text"]["chars"], 14);
    // Text the agent wrote itself (a cell address, a formula) is kept.
    assert_eq!(steps[1]["text"], "D2");
    assert_eq!(steps[4]["text"], "=B2-C2");
    // Private-looking text ends the program.
    assert_eq!(steps[5], json!({"stop": "private_text"}));
    assert!(!program.to_string().contains("Example Report"));
    // Fewer than two repeatable steps is not a program.
    assert!(program_from_tape(&[json!({"stop": "run_command"})], "anything").is_none());
}

#[test]
fn a_program_becomes_one_long_plan_for_the_same_task() {
    let request = "Name the sheet Example Report and add a profit column";
    let program = program_from_tape(
        &[
            json!({"click": "Sheet"}),
            json!({"click": "Rename Sheet..."}),
            json!({"text": "Example Report", "replace": true}),
            json!({"key": "Enter"}),
            json!({"click": "Name Box"}),
            json!({"text": "D1", "replace": true}),
            json!({"key": "Enter"}),
            json!({"stop": "click_localized"}),
            json!({"click": "Never reached"}),
        ],
        request,
    )
    .unwrap();
    let nodes = program_nodes(
        &program,
        "Rename a sheet",
        &task_signature(request),
        request,
    );
    assert_eq!(nodes.len(), 5);
    assert_eq!(nodes[0]["target_hint"], "\"Sheet\"");
    assert_eq!(nodes[0]["done_when"], "\"Rename Sheet...\" is visible");
    // Typing is its own step, into the control focused after the click, and
    // the request's own text is found again for it.
    assert!(nodes[2].get("target_hint").is_none());
    assert_eq!(nodes[2]["input"][0]["text"], "Example Report");
    assert_eq!(nodes[2]["input"][1]["key"], "Enter");
    assert_eq!(nodes[3]["target_hint"], "\"Name Box\"");
    assert_eq!(nodes[4]["input"].as_array().unwrap().len(), 2);
    // A request asking for another name gets that name typed in the same
    // slot (between "sheet" and "and add"), never the old one.
    let other = "Name the sheet Quarterly Totals and add a profit column";
    let nodes = program_nodes(&program, "Rename a sheet", &task_signature(request), other);
    assert_eq!(nodes.len(), 5);
    assert_eq!(nodes[2]["input"][0]["text"], "Quarterly Totals");
    assert!(
        !serde_json::to_string(&nodes)
            .unwrap()
            .contains("Example Report")
    );
    // Without that slot in the request, the replay stops before typing.
    let unrelated = "Add a profit column to the sheet";
    let nodes = program_nodes(
        &program,
        "Rename a sheet",
        &task_signature(request),
        unrelated,
    );
    assert_eq!(nodes.len(), 2);
    assert!(nodes.iter().all(|node| node.get("input").is_none()));
}

#[test]
fn a_similar_task_fills_its_own_value_into_the_recorded_slot() {
    let request = "Add Kyoto to the world clock";
    let program = program_from_tape(
        &[
            json!({"click": "World clock"}),
            json!({"click": "Add a new city"}),
            json!({"text": "Kyoto", "replace": true}),
            json!({"key": "Enter"}),
        ],
        request,
    )
    .unwrap();
    let other = "Add \"Lisbon\" to the world clock";
    assert!(same_task(&task_signature(request), other));
    let nodes = program_nodes(&program, "Add a city", &task_signature(request), other);
    assert_eq!(nodes[2]["input"][0]["text"], "Lisbon");
}

#[tokio::test]
async fn system_one_replays_a_stored_program_end_to_end() {
    let request = "open system settings";
    let mut skill = replay_skill_record(2, &["chat.exe"]);
    skill.task_signature = task_signature(request);
    skill.program = program_from_tape(
        &[
            json!({"click": "System"}),
            json!({"click": "Bluetooth"}),
            json!({"key": "Enter"}),
        ],
        request,
    );
    let mut harness = replay_harness();
    harness.session.motor_tape.clear();
    let used = harness
        .session
        .replay_skill(
            &skill,
            0.9,
            request,
            &mut RunMetrics::default(),
            &mut Vec::new(),
        )
        .await
        .unwrap();
    assert_eq!(used, 1);
    // Both recorded clicks ran (the second with its key), in order, and the
    // replayed steps are on this run's tape again.
    let clicks = harness.clicks.lock().clone();
    assert_eq!(clicks.len(), 3, "{clicks:?}");
    assert_eq!(clicks[0]["expected_label"], "System");
    assert_eq!(clicks[1]["expected_label"], "Bluetooth");
    // The key goes to the control focused after the click, as recorded.
    let typed = clicks[2]["steps"].to_string();
    assert!(typed.contains("Enter"), "{typed}");
    assert!(
        harness
            .session
            .motor_tape
            .iter()
            .any(|step| step["click"] == "Bluetooth")
    );
    assert!(harness.session.stale_program.is_none());
    let trace =
        std::fs::read_to_string(harness.session.context.artifact_dir.join("trace.jsonl")).unwrap();
    assert!(trace.contains("\"mode\":\"program\""), "{trace}");
}

#[tokio::test]
async fn a_program_that_no_longer_fits_is_marked_for_replacement() {
    let request = "open system settings";
    let mut skill = replay_skill_record(2, &["chat.exe"]);
    skill.task_signature = task_signature(request);
    skill.program = program_from_tape(
        &[json!({"click": "System"}), json!({"click": "Display"})],
        request,
    );
    let mut harness = replay_harness();
    harness
        .session
        .replay_skill(
            &skill,
            0.9,
            request,
            &mut RunMetrics::default(),
            &mut Vec::new(),
        )
        .await
        .unwrap();
    // "Display" is not on this screen: the replay stops, and the program this
    // run finishes with will replace the stored one.
    assert_eq!(harness.session.stale_program, Some(skill.id));
}

#[test]
fn the_leaner_program_wins_unless_the_stored_one_went_stale() {
    let dir = tempfile::tempdir().unwrap();
    let memory = crate::memory::MemoryStore::open(dir.path()).unwrap();
    let workflow = explorer_workflow();
    let qualification = qualify_workflow(&workflow, &[]);
    let (skill, _) = learn_verified_procedures(
        &memory,
        "open the requested folder",
        &workflow,
        &qualification,
    )
    .unwrap()
    .remove(0);
    let program =
        |steps: usize| json!({"version": 1, "steps": vec![json!({"key": "Enter"}); steps]});
    let stored = |memory: &crate::memory::MemoryStore| {
        memory.load_skill(skill.id).unwrap().0.program.unwrap()["steps"]
            .as_array()
            .unwrap()
            .len()
    };
    assert!(attach_program(&memory, skill.id, Some(&program(5)), None));
    // A longer path does not replace a working one...
    assert!(!attach_program(&memory, skill.id, Some(&program(8)), None));
    assert!(attach_program(&memory, skill.id, Some(&program(3)), None));
    assert_eq!(stored(&memory), 3);
    // ...unless the stored one stopped before its end when replayed.
    assert!(attach_program(
        &memory,
        skill.id,
        Some(&program(7)),
        Some(skill.id)
    ));
    assert_eq!(stored(&memory), 7);
}

#[test]
fn a_replay_leaves_commit_shortcuts_to_the_planner() {
    let request = "write a note and save it";
    let program = program_from_tape(
        &[
            json!({"click": "Document"}),
            json!({"text": "Draft", "replace": false}),
            json!({"key": "Ctrl+S"}),
            json!({"click": "Save"}),
        ],
        request,
    )
    .unwrap();
    let nodes = program_nodes(&program, "Save a note", &task_signature(request), request);
    // The click and the typing replay; the save shortcut (and all after it)
    // is the planner's to confirm.
    assert_eq!(nodes.len(), 2);
    assert_eq!(nodes[1]["input"].as_array().unwrap().len(), 1);
    assert!(!nodes[0].to_string().contains("Ctrl+S"));
}

#[tokio::test]
async fn the_first_turn_starts_with_the_application_on_screen() {
    // An application in front: captured once and handed to the planner as
    // the result of its first look.
    let mut harness = replay_harness();
    let before = harness.session.messages.len();
    harness.session.initial_look(true).await.unwrap();
    let added = &harness.session.messages[before..];
    assert_eq!(added[0].role, "assistant");
    assert_eq!(added[0].tool_calls[0].name, "capture_screen");
    assert_eq!(added[1].role, "tool");
    assert_eq!(
        added[1].tool_call_id.as_deref(),
        Some(added[0].tool_calls[0].id.as_str())
    );
    // It counts as the run's fresh evidence, as the planner's own look would:
    // a run that acts from it and succeeds is verified, and its skill learns.
    assert!(harness.session.decision_router_fresh_evidence);
    // A terminal in front (a question typed at the command line): nothing.
    let mut harness = replay_harness();
    if let Some(observation) = harness.session.context.latest_observation.lock().as_mut() {
        observation.foreground_window.as_mut().unwrap().process_name = "WindowsTerminal.exe".into();
    }
    let observation = harness
        .session
        .context
        .latest_observation
        .lock()
        .clone()
        .unwrap();
    harness.session.context.platform = crate::platform::MockDesktop::new(observation);
    let before = harness.session.messages.len();
    harness.session.initial_look(true).await.unwrap();
    assert_eq!(harness.session.messages.len(), before);
}

#[test]
fn a_program_replays_for_the_same_or_a_reworded_request_only() {
    let original = task_signature("Please turn off system notifications in Windows Settings");
    assert!(same_task(
        &original,
        "Please turn off system notifications in Windows Settings"
    ));
    // Reworded, same task.
    assert!(same_task(
        &original,
        "turn off the system notifications in Windows Settings please"
    ));
    // A different task in the same app.
    assert!(!same_task(
        &original,
        "Turn on Do not disturb in Windows Settings"
    ));
    assert!(!same_task("", "anything"));
}

#[test]
fn the_agent_view_gets_the_saved_frame_and_its_targets_as_fractions() {
    let dir = tempfile::tempdir().unwrap();
    let mut observation = launch_observation(Some(("Settings", "SystemSettings.exe")));
    observation.screenshots = vec![crate::types::Screenshot {
        monitor: crate::types::MonitorInfo {
            id: "window:0x1".into(),
            bounds: crate::types::Rect {
                x: 100,
                y: 50,
                width: 800,
                height: 400,
            },
            scale_factor: 1.0,
            primary: true,
        },
        png_base64: String::new(),
        source_png_base64: None,
        model_width: 800,
        model_height: 400,
        captured_at: chrono::Utc::now(),
    }];
    let target = |id: &str, x: i32| -> crate::types::InteractionTarget {
        serde_json::from_value(json!({
            "id": id, "name": format!("Button {id}"), "control_type": "Button",
            "bounds": {"x": x, "y": 150, "width": 80, "height": 40},
            "source": "uia_ocr", "confidence": null, "enabled": true, "actionable": true, "click_point": null
        }))
        .unwrap()
    };
    // One target on the frame, one entirely to its left (another monitor).
    observation.targets = vec![target("1", 500), target("2", -300)];

    // No saved frame yet: nothing to show.
    assert!(observation_frame(&observation, dir.path()).is_none());
    std::fs::write(
        dir.path().join(format!(
            "observation-{}-monitor-window_0x1.png",
            observation.version
        )),
        b"png",
    )
    .unwrap();
    let Some(AgentEvent::ObservationCaptured {
        window_title,
        targets,
        image_path,
        ..
    }) = observation_frame(&observation, dir.path())
    else {
        panic!("expected a frame event");
    };
    assert_eq!(window_title.as_deref(), Some("Settings"));
    assert!(image_path.ends_with("-monitor-window_0x1.png"));
    assert_eq!(targets.len(), 1);
    assert_eq!(targets[0].id, "1");
    assert_eq!(targets[0].source, "uia_ocr");
    assert!((targets[0].x - 0.5).abs() < 1e-6 && (targets[0].y - 0.25).abs() < 1e-6);
    assert!((targets[0].width - 0.1).abs() < 1e-6 && (targets[0].height - 0.1).abs() < 1e-6);
}

struct ImageCheckingBrain {
    user_images: parking_lot::Mutex<Vec<usize>>,
}

#[async_trait]
impl Brain for ImageCheckingBrain {
    async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec!["mock".into()])
    }
    fn stream(&self, request: BrainRequest) -> crate::brain::BrainStream {
        let images = request
            .messages
            .iter()
            .filter(|message| message.origin == MessageOrigin::UserInput)
            .flat_map(|message| &message.content)
            .filter(|part| matches!(part, MessageContent::ImagePng { .. }))
            .count();
        self.user_images.lock().push(images);
        Box::pin(futures::stream::iter(vec![
            Ok(BrainEvent::TextDelta {
                text: "A red square.".into(),
            }),
            Ok(BrainEvent::Finished {
                reason: Some("stop".into()),
            }),
        ]))
    }
}

/// A 1x1 PNG.
const TINY_PNG: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==";

#[tokio::test]
async fn images_the_user_attaches_reach_the_model_in_their_message() {
    let mut harness = fast_harness(
        fast_router(usize::MAX, 0.99),
        Vec::new(),
        crate::config::DecisionRouterMode::Delegated,
    );
    let brain = Arc::new(ImageCheckingBrain {
        user_images: Default::default(),
    });
    harness.session.brain = brain.clone();
    let result = harness
        .session
        .run_with_images(
            "what is in this picture?",
            vec![format!("data:image/png;base64,{TINY_PNG}")],
        )
        .await
        .unwrap();
    assert_eq!(result.answer, "A red square.");
    assert_eq!(brain.user_images.lock().first(), Some(&1));
    // The next request carries none unless the user attaches again.
    assert!(harness.session.pending_user_images.is_empty());
}

#[test]
fn attached_images_must_be_a_few_real_pngs() {
    assert_eq!(
        validated_user_images(vec![TINY_PNG.into()]).unwrap(),
        [TINY_PNG]
    );
    assert!(validated_user_images(vec!["not base64!".into()]).is_err());
    // A JPEG header is refused; the dashboard converts pictures to PNG.
    use base64::Engine as _;
    let jpeg = base64::engine::general_purpose::STANDARD.encode([0xFF, 0xD8, 0xFF, 0xE0]);
    assert!(validated_user_images(vec![jpeg]).is_err());
    assert!(validated_user_images(vec![TINY_PNG.into(); MAX_USER_IMAGES + 1]).is_err());
}

fn observation_cycle(index: usize, page: &str, lines: &[&str]) -> [BrainMessage; 2] {
    let ordered: Vec<Value> = lines.iter().map(|text| json!({"text": text})).collect();
    [
        BrainMessage {
            role: "assistant".into(),
            content: Vec::new(),
            origin: MessageOrigin::Assistant,
            tool_call_id: None,
            tool_calls: vec![CompletedToolCall {
                id: format!("call-{index}"),
                name: "click_target".into(),
                arguments: json!({"target_id": "7"}),
            }],
        },
        BrainMessage {
            role: "tool".into(),
            content: vec![MessageContent::Text {
                text: json!({
                    "observation_id": format!("obs_{index}"),
                    "observed_url": page,
                    "ordered_content": ordered,
                    "targets": [],
                })
                .to_string(),
            }],
            origin: MessageOrigin::ToolResult,
            tool_call_id: Some(format!("call-{index}")),
            tool_calls: Vec::new(),
        },
    ]
}

#[test]
fn pages_already_read_survive_compaction_so_the_planner_does_not_reread_them() {
    // A long run alternating between two pages of an example social site,
    // as when the planner forgot what the notifications page said.
    let mut messages = vec![
        BrainMessage::text("system", "system"),
        BrainMessage::text("user", "check my example account for anything new today"),
    ];
    for index in 0..40 {
        let cycle = if index % 2 == 0 {
            observation_cycle(
                index,
                "social.example/notifications",
                &[
                    "Notifications",
                    "Recent post from Example Author · 5h",
                    "▸",
                    "Deals Alerts turned on · 21h",
                ],
            )
        } else {
            observation_cycle(
                index,
                "social.example/home",
                &["Home", "For you", "Example headline one · 2h"],
            )
        };
        messages.extend(cycle);
    }
    let (bounded, _, actions) = bounded_context(&messages, 1, 3_000, 4_000, 1.0);
    assert!(
        actions
            .iter()
            .any(|action| action.starts_with("continuity_ledger:"))
    );
    let rendered = serde_json::to_string(&bounded).unwrap();
    assert!(rendered.contains("Pages already read"));
    assert!(rendered.contains("social.example/notifications (read "));
    assert!(rendered.contains("Recent post from Example Author · 5h"));
    assert!(rendered.contains("Example headline one · 2h"));
    // Icons and one-character marks are not notes.
    assert!(!rendered.contains("| ▸ |"));

    // The next compaction carries the pages forward and keeps counting.
    let mut longer = bounded.clone();
    for index in 40..70 {
        longer.extend(observation_cycle(
            index,
            "social.example/home",
            &["Home", "Example headline two · 1h"],
        ));
    }
    let (again, _, _) = bounded_context(&longer, 1, 3_000, 4_000, 1.0);
    let rendered = serde_json::to_string(&again).unwrap();
    assert!(
        rendered.contains("Recent post from Example Author · 5h"),
        "notifications text is still known"
    );
    assert_eq!(
        rendered.matches(PAGES_READ_HEADER).count(),
        1,
        "one pages block, not repeated"
    );
}

#[test]
fn a_page_note_keeps_readable_text_in_order_without_repeats() {
    let value = json!({
        "observed_url": "social.example/notifications",
        "ordered_content": [
            {"text": "social.example/notifications"}, {"text": "@"}, {"text": "Notifications"},
            {"text": "Notifications"}, {"text": "Recent post from Example Author"}
        ]
    });
    let (page, note) = page_note(&value).unwrap();
    assert_eq!(page, "social.example/notifications");
    assert_eq!(note, "Notifications | Recent post from Example Author");
}

#[tokio::test]
async fn a_word_already_on_screen_does_not_skip_opening_the_named_page() {
    // The home page's sidebar already shows "Trending"; the plan opens the
    // Explore page and quotes "Trending" in its condition. The step must
    // still open Explore instead of reporting done without acting.
    let mut harness = fast_harness(
        fast_router(1, 0.99),
        vec![fast_link("t1", "Explore"), fast_link("t2", "Trending")],
        crate::config::DecisionRouterMode::Delegated,
    );
    let mut metrics = RunMetrics::default();
    let result = harness
        .session
        .run_fast_actions(
            1,
            &json!({"goal": "Open the Explore page", "target_hint": "\"Explore\"",
                "done_when": "the Explore page is open showing \"Trending\" and news stories",
                "allowed_operations": ["click"], "max_steps": 3}),
            &mut metrics,
        )
        .await
        .unwrap();
    let clicks = harness.clicks.lock();
    assert_eq!(clicks.len(), 1, "{result}");
    assert_eq!(clicks[0]["target_id"], "t1", "Explore is opened first");
    assert_eq!(result["status"], "done");
}

#[test]
fn files_named_or_attached_become_readable_even_outside_the_workspace() {
    let outside = tempfile::tempdir().unwrap();
    let report = outside.path().join("quarterly report.xlsx");
    std::fs::write(&report, b"PK\x03\x04 example").unwrap();
    let folder = outside.path().join("photos");
    std::fs::create_dir(&folder).unwrap();
    std::fs::write(folder.join("a.png"), b"x").unwrap();

    // Named in the request, with prose after the path.
    let prompt = format!("summarize {} for me please", report.display());
    let named = paths_named_in(&prompt);
    assert_eq!(named, [dunce::canonicalize(&report).unwrap()]);
    // A web address is not a path.
    assert!(paths_named_in("open x.com/home and read it").is_empty());

    let block = attached_files_block(&[
        dunce::canonicalize(&report).unwrap(),
        dunce::canonicalize(&folder).unwrap(),
    ]);
    assert!(
        block.contains("quarterly report.xlsx (.xlsx file, 12 B)"),
        "{block}"
    );
    assert!(block.contains("photos (folder, 1 entries)"), "{block}");
    assert!(block.contains("write a small helper script"));
    assert!(
        attached_file_paths(&[outside.path().join("missing.txt").display().to_string()]).is_err()
    );
}

#[tokio::test]
async fn an_attached_file_outside_the_workspace_can_be_read_by_the_file_tools() {
    let outside = tempfile::tempdir().unwrap();
    let notes = outside.path().join("notes.txt");
    std::fs::write(&notes, b"hello").unwrap();
    let mut harness = fast_harness(
        fast_router(usize::MAX, 0.99),
        Vec::new(),
        crate::config::DecisionRouterMode::Delegated,
    );
    assert!(
        harness
            .session
            .context
            .resolve_readable_artifact(&notes)
            .is_err()
    );
    harness.session.brain = Arc::new(ImageCheckingBrain {
        user_images: Default::default(),
    });
    harness
        .session
        .run_with_attachments(
            "what does this say?",
            Vec::new(),
            vec![notes.display().to_string()],
        )
        .await
        .unwrap();
    assert!(
        harness
            .session
            .context
            .resolve_readable_artifact(&notes)
            .is_ok()
    );
    let request = harness
        .session
        .messages
        .iter()
        .rev()
        .find(|message| message.origin == MessageOrigin::UserInput)
        .unwrap();
    let MessageContent::Text { text } = &request.content[0] else {
        panic!()
    };
    assert!(text.contains("<attached_files>") && text.contains("notes.txt"));
}

fn tip_popup(buttons: &[&str], text: &str) -> Popup {
    Popup {
        title: "Tip of the day".into(),
        text: text.into(),
        buttons: buttons.iter().map(|button| (*button).to_owned()).collect(),
        window_id: None,
    }
}

#[test]
fn system_1_puts_away_an_ambient_notice_but_not_a_question_about_the_users_work() {
    let tip = tip_popup(
        &["Show me", "Not now"],
        "Pin your favourite folders for quick access.",
    );
    assert_eq!(ambient_dismissal(&tip).as_deref(), Some("Not now"));
    // The preferred dismissal wins over a later one in the list.
    let reminder = tip_popup(&["Close", "Remind me later"], "A new version is ready.");
    assert_eq!(
        ambient_dismissal(&reminder).as_deref(),
        Some("Remind me later")
    );
    // Anything about data, access, or a failure is the planner's call.
    for text in [
        "Do you want to save changes to Example.txt?",
        "The file could not be opened because an error occurred.",
        "Allow this app to use your location?",
        "Are you sure you want to leave?",
        "Restart now to finish updating.",
    ] {
        assert_eq!(
            ambient_dismissal(&tip_popup(&["Not now"], text)),
            None,
            "{text}"
        );
    }
    // OK and Cancel may confirm or abandon something: never a reflex.
    assert_eq!(
        ambient_dismissal(&tip_popup(&["OK", "Cancel"], "Pin your folders.")),
        None
    );
    assert_eq!(
        ambient_dismissal(&tip_popup(&[], "Pin your folders.")),
        None
    );
}

#[test]
fn what_changed_says_in_one_line_what_an_action_did() {
    let opened = json!({
        "focus": {"title": "Settings", "app": "example.exe"},
        "state_change": {
            "added_text": ["Display", "Night light", "", "Scale", "Resolution", "Orientation"],
            "removed_control_count": 3,
            "focus_changed": true,
            "selected_changed": ["System"],
            "visual_change_ratio": 0.4,
        },
    });
    assert_eq!(
        what_changed(&opened).unwrap(),
        "now in \"Settings\"; selected \"System\"; new on screen: \"Display\", \"Night light\", \"Scale\", \"Resolution\"; 3 items closed or left the view"
    );
    let dialog = json!({
        "foreground_transition": {"title": "Print"},
        "page_status": "loading",
        "state_change": {"focus_changed": true},
    });
    assert_eq!(
        what_changed(&dialog).unwrap(),
        "dialog \"Print\" opened; page still loading"
    );
    let nothing = json!({"state_change": {"visual_change_ratio": 0.0}});
    assert_eq!(what_changed(&nothing).unwrap(), "nothing visible changed");
    let redraw = json!({"state_change": {"visual_change_ratio": 0.3}});
    assert_eq!(
        what_changed(&redraw).unwrap(),
        "the view changed but no new text appeared"
    );
    assert_eq!(what_changed(&json!({"ok": true})), None);
}

/// A click that opens a "Tip of the day" notice in the mock window; clicking
/// "Not now" closes it unless `sticky` keeps it on screen.
struct TipRaisingClick {
    calls: Arc<parking_lot::Mutex<Vec<Value>>>,
    sticky: bool,
}

fn tip_elements() -> Vec<crate::types::UiElement> {
    vec![
        popup_ui("Tip of the day", "Window", 200, 200, 300, 150),
        popup_ui(
            "Pin your favourite folders for quick access.",
            "Text",
            220,
            230,
            200,
            20,
        ),
        popup_ui("Show me", "Button", 230, 310, 60, 24),
        popup_ui("Not now", "Button", 300, 310, 60, 24),
    ]
}

#[async_trait]
impl crate::tool::Tool for TipRaisingClick {
    fn name(&self) -> &'static str {
        "click_target"
    }
    fn description(&self) -> &'static str {
        "test click"
    }
    fn input_schema(&self) -> Value {
        json!({"type": "object"})
    }
    fn risk(&self) -> crate::policy::RiskClass {
        crate::policy::RiskClass::ReadOnly
    }
    async fn execute(&self, arguments: Value, context: &ToolContext) -> Result<Value> {
        let dismissing = arguments["expected_label"] == "Not now";
        self.calls.lock().push(arguments);
        let mut guard = context.latest_observation.lock();
        let observation = guard.as_mut().unwrap();
        if dismissing {
            if !self.sticky {
                observation
                    .ui_elements
                    .retain(|element| !tip_elements().iter().any(|tip| tip.name == element.name));
                observation
                    .targets
                    .retain(|target| target.name != "Not now");
            }
            return Ok(json!({
                "state_change": {"removed_control_count": 4, "visual_change_ratio": 0.2},
                "verification": {"effect": true},
            }));
        }
        observation.ui_elements.extend(tip_elements());
        let mut not_now = fast_link("90", "Not now");
        not_now.control_type = "Button".into();
        observation.targets.push(not_now);
        Ok(json!({
            "focus": {"title": "Files", "app": "example.exe"},
            "state_change": {"added_text": ["Tip of the day"], "focus_changed": false},
            "verification": {"effect": true},
        }))
    }
}

fn tip_harness(sticky: bool) -> (FastHarness, Arc<parking_lot::Mutex<Vec<Value>>>) {
    let mut harness = fast_harness(
        fast_router(usize::MAX, 0.99),
        vec![fast_link("6", "Downloads")],
        crate::config::DecisionRouterMode::Delegated,
    );
    let calls = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let mut tools = ToolRegistry::default();
    tools.register(TipRaisingClick {
        calls: calls.clone(),
        sticky,
    });
    harness.session.tools = Arc::new(tools);
    (harness, calls)
}

#[tokio::test]
async fn system_1_dismisses_an_ambient_tip_itself_and_keeps_it_out_of_muscle_memory() {
    let (mut harness, calls) = tip_harness(false);
    let (result, metrics) = run_plan(
        &mut harness,
        json!({
            "goal": "open Downloads",
            "target_hint": "\"Downloads\"",
            "done_when": "\"Example report\" is visible",
            "max_steps": 4,
        }),
    )
    .await;
    assert_ne!(result["status"], "popup", "{result}");
    let labels = calls
        .lock()
        .iter()
        .map(|call| {
            call["expected_label"]
                .as_str()
                .unwrap_or_default()
                .to_owned()
        })
        .collect::<Vec<_>>();
    assert_eq!(labels[..2], ["Downloads", "Not now"], "{labels:?}");
    assert_eq!(
        result["dismissed_popups"][0]["title"], "Tip of the day",
        "{result}"
    );
    assert_eq!(result["dismissed_popups"][0]["button"], "Not now");
    assert_eq!(metrics.extras["fast_actions_popups_dismissed"], 1);
    let steps = result["steps"].as_array().unwrap();
    assert_eq!(steps[0]["changed"], "new on screen: \"Tip of the day\"");
    assert_eq!(steps[1]["dismissed_popup"], true);
    assert!(result["what_changed"].is_string(), "{result}");
    // The tip is not part of the task: replaying "Not now" without the tip
    // on screen would break the replay.
    let recorded = serde_json::to_string(&harness.session.motor_tape).unwrap();
    assert!(recorded.contains("Downloads"), "{recorded}");
    assert!(!recorded.contains("Not now"), "{recorded}");
}

#[tokio::test]
async fn a_notice_that_comes_back_after_its_dismissal_goes_to_the_planner() {
    let (mut harness, calls) = tip_harness(true);
    let (result, metrics) = run_plan(
        &mut harness,
        json!({
            "goal": "open Downloads",
            "target_hint": "\"Downloads\"",
            "done_when": "\"Example report\" is visible",
            "max_steps": 4,
        }),
    )
    .await;
    assert_eq!(result["status"], "popup", "{result}");
    assert_eq!(result["popup"]["title"], "Tip of the day");
    let dismissals = calls
        .lock()
        .iter()
        .filter(|call| call["expected_label"] == "Not now")
        .count();
    assert_eq!(dismissals, 1);
    assert_eq!(metrics.extras["fast_actions_popups_dismissed"], 1);
}

#[test]
fn compactable_history_at_minimum_matches_the_compaction_slice() {
    let system = BrainMessage::text("system", "system instructions");
    let active = BrainMessage::text("user", "the active request");

    // Nothing precedes the active request: compaction has no candidate.
    assert!(compressible_history_at_minimum(&[
        system.clone(),
        active.clone()
    ]));

    // A real earlier turn remains compactable.
    let conversation = vec![
        system.clone(),
        BrainMessage::text("user", "an earlier request"),
        BrainMessage::text("assistant", "an earlier answer"),
        active.clone(),
    ];
    assert!(!compressible_history_at_minimum(&conversation));

    // Once earlier turns were replaced by one summary, the meter must report
    // "at minimum" even though fixed tool schemas keep the total above the
    // compaction threshold.
    let summarized = vec![
        system,
        BrainMessage::text_with_origin(
            "system",
            "[Historical conversation summary: earlier work]",
            MessageOrigin::HistorySummary,
        ),
        active,
    ];
    assert!(compressible_history_at_minimum(&summarized));
}
