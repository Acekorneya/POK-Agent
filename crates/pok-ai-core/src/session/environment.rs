//! The desktop environment around a run: recovery when the foreground changes, pause checkpoints, waiting for the user, and time/authorization context.

use super::*;

#[derive(Debug, Clone)]
pub(super) struct TemporalAnchor {
    pub(super) local_date: String,
    pub(super) context: String,
}

pub(super) fn temporal_anchor_at(now: DateTime<Local>) -> TemporalAnchor {
    let local_date = now.format("%Y-%m-%d").to_string();
    let context = format!(
        "<temporal_context>Today's local date is {} ({}). The Windows local time zone is {} (UTC{}). Interpret today, tomorrow, yesterday, weekdays, and undated times relative to this date and zone. For requests about latest or current online information, verify freshness using available browser or search tools; the clock alone is not evidence that external information is current.</temporal_context>",
        local_date,
        now.format("%A"),
        now.format("%Z"),
        now.format("%:z"),
    );
    TemporalAnchor {
        local_date,
        context,
    }
}

pub(super) fn current_temporal_anchor() -> TemporalAnchor {
    temporal_anchor_at(Local::now())
}

pub(super) fn authorization_context(mode: &PolicyMode) -> &'static str {
    match mode {
        PolicyMode::Autonomous => {
            "<authorization>Autonomous mode is active. Ordinary filesystem, application, installation, generated-tool, command, user-data, network, publishing, messaging, credential, and external actions may run without approval and may use absolute paths outside the workspace. Commands that can modify OS-critical files, boot or disk state, machine-wide security, accounts, services, drivers, or system policy still require explicit approval. Secure desktop, password controls, elevated-window input, and stale or untargeted desktop input remain blocked.</authorization>"
        }
        PolicyMode::Interactive => {
            "<authorization>Interactive mode is active. File tools remain workspace-scoped and state-changing commands require approval.</authorization>"
        }
        PolicyMode::Exam { .. } => {
            "<authorization>Exam mode is active. All filesystem, process, and desktop actions must remain inside the configured exam sandbox and allowlist.</authorization>"
        }
    }
}

pub(super) fn replace_authorization_context(text: &mut String, replacement: &str) -> bool {
    let Some(start) = text.find("<authorization>") else {
        return false;
    };
    let Some(relative_end) = text[start..].find("</authorization>") else {
        return false;
    };
    let end = start + relative_end + "</authorization>".len();
    text.replace_range(start..end, replacement);
    true
}

pub(super) fn navigation_result_is_ready(value: &Value) -> bool {
    value
        .pointer("/navigation_readiness/status")
        .and_then(Value::as_str)
        .is_none_or(|status| status == "ready")
}

impl Session {
    /// Record every hold for the user (and its length) in the trace, and
    /// tell the dashboard when one is attached.
    pub(super) fn install_user_activity_listener(&self) {
        let trace = self
            .trace
            .lock()
            .try_clone()
            .ok()
            .map(|file| Arc::new(Mutex::new(file)));
        let observer = self.observer.clone();
        let session_id = self.id;
        let started = Arc::new(Mutex::new(None::<Instant>));
        self.context
            .pause
            .set_user_activity_listener(Arc::new(move |waiting| {
                let waited_ms = {
                    let mut started = started.lock();
                    if waiting {
                        *started = Some(Instant::now());
                        None
                    } else {
                        started.take().map(|since| {
                            u64::try_from(since.elapsed().as_millis()).unwrap_or(u64::MAX)
                        })
                    }
                };
                if let Some(trace) = &trace {
                    let event = SessionEvent {
                        timestamp: Utc::now(),
                        session_id,
                        kind: "user_activity_wait".into(),
                        payload: json!({"waiting": waiting, "waited_ms": waited_ms}),
                    };
                    let mut file = trace.lock();
                    if serde_json::to_writer(&mut *file, &event).is_ok() {
                        let _ = file.write_all(b"\n");
                        let _ = file.flush();
                    }
                }
                if let Some(observer) = &observer {
                    observer.emit(AgentEvent::UserActivityWait { waiting });
                }
            }));
    }

    pub(super) async fn recover_environment_change(
        &mut self,
        turn: u32,
        previous: WindowInfo,
        mut current: Option<WindowInfo>,
        metrics: &mut RunMetrics,
    ) -> Result<()> {
        const QUIET_MS: u64 = 3_000;
        const PAUSE_AFTER_MS: u64 = 30_000;

        let started = Instant::now();
        self.environment_revision = self.environment_revision.saturating_add(1);
        let revision = self.environment_revision;
        let previous_label = format!("{} ({})", previous.title, previous.process_name);
        let current_label = current
            .as_ref()
            .map(|window| format!("{} ({})", window.title, window.process_name));
        *self.context.latest_observation.lock() = None;
        *self.context.latest_observation_view.lock() = None;
        *self.context.pending_visual_localization.lock() = None;
        *self.context.focused_control.lock() = None;
        self.decision_router_cache_key = None;
        self.decision_router_last_result = None;
        self.decision_router_ambiguity_refreshes = 0;
        self.decision_router_refinement = DecisionRefinementState::default();
        // The environment changed: evidence captured before the change must
        // not authorize a terminal DONE after it.
        self.decision_router_fresh_evidence = false;
        self.emit(AgentEvent::EnvironmentChanged {
            revision,
            previous_window: Some(previous_label.clone()),
            current_window: current_label.clone(),
        });
        self.emit(AgentEvent::EnvironmentWaiting {
            revision,
            quiet_period_ms: QUIET_MS,
        });
        self.log(
            "environment_changed",
            json!({
                "revision": revision,
                "previous_window": previous_label,
                "current_window": current_label,
                "stale_observation_invalidated": true,
            }),
        )?;

        loop {
            let snapshot = self.context.platform.desktop_activity_snapshot().await?;
            current = snapshot.foreground_window;
            // Physical input when the platform can tell it apart, so the
            // agent's own injected input never reads as the user's.
            if snapshot
                .last_physical_input_ms
                .or(snapshot.last_user_input_ms)
                .is_none_or(|elapsed| elapsed >= QUIET_MS)
            {
                break;
            }
            if started.elapsed() >= Duration::from_millis(PAUSE_AFTER_MS) {
                self.context.pause.request_pause();
                self.pause_checkpoint("continuous_user_activity").await?;
                self.emit(AgentEvent::EnvironmentRecovered {
                    revision,
                    action: "resumed_after_user_pause".into(),
                    used_jev: false,
                    elapsed_ms: started.elapsed().as_millis() as u64,
                });
                return Ok(());
            }
            tokio::select! {
                () = self.context.cancellation.cancelled() => return Err(PokError::Cancelled),
                () = tokio::time::sleep(Duration::from_millis(250)) => {}
            }
        }

        let available_windows = self
            .context
            .platform
            .list_windows()
            .await
            .unwrap_or_default();
        let original_available = available_windows
            .iter()
            .any(|window| window.id == previous.id && !window.elevated);
        let same_process = current.as_ref().is_some_and(|window| {
            !window.elevated
                && window
                    .process_name
                    .eq_ignore_ascii_case(&previous.process_name)
        });
        let mut candidates = Vec::new();
        if same_process && let Some(window) = &current {
            candidates.push(DecisionCandidate {
                id: "observe_current".into(),
                tool: "capture_screen".into(),
                arguments: json!({"scope": "window", "window_id": window.id}),
                description: "Observe the new foreground state in the same application without changing focus".into(),
                kind: DecisionCandidateKind::Action,
                local_score: 0.85,
            });
        }
        if original_available {
            candidates.push(DecisionCandidate {
                id: "reactivate_original".into(),
                tool: "capture_screen".into(),
                arguments: json!({"scope": "window", "window_id": previous.id}),
                description:
                    "Reactivate the previously authorized task window and capture fresh evidence"
                        .into(),
                kind: DecisionCandidateKind::Action,
                local_score: 0.8,
            });
        }
        candidates.push(DecisionCandidate {
            id: "defer_to_llm".into(),
            tool: "defer_to_llm".into(),
            arguments: json!({}),
            description: "Do not change focus; tell the primary model that the environment changed"
                .into(),
            kind: DecisionCandidateKind::Action,
            local_score: 0.4,
        });

        let local_choice = if same_process {
            "observe_current"
        } else if original_available {
            "reactivate_original"
        } else {
            "defer_to_llm"
        };
        let mut selected_id = local_choice.to_owned();
        let mut used_jev = false;
        if self.decision_router_failures < 3
            && let (Some(router), Some(config)) = (
                self.decision_router.clone(),
                self.decision_router_config.clone(),
            )
        {
            self.emit(AgentEvent::DecisionRouterStarted {
                turn,
                purpose: "environment_recovery".into(),
                candidate_count: candidates.len(),
            });
            let decision_started = Instant::now();
            let task = self.context.active_task.lock().clone();
            let result = tokio::select! {
                () = self.context.cancellation.cancelled() => return Err(PokError::Cancelled),
                result = router.decide(DecisionRequest {
                    purpose: DecisionPurpose::EnvironmentRecovery,
                    task: task.root_request,
                    current_step: task.current_step,
                    candidates: candidates.clone(),
                    state: json!({
                        "previous_window": {"application": previous.process_name, "title": previous.title},
                        "current_window": current.as_ref().map(|window| json!({"application": window.process_name, "title": window.title})),
                        "environment_revision": revision,
                        "user_quiet_ms": QUIET_MS,
                    }),
                }) => result,
            };
            let elapsed_ms = decision_started.elapsed().as_millis() as u64;
            match result {
                Ok(decision) => {
                    self.decision_router_failures = 0;
                    let selected = decision
                        .candidate_id
                        .as_deref()
                        .and_then(|id| candidates.iter().find(|candidate| candidate.id == id));
                    // Environment recovery uses JEV's independent noul scores,
                    // which do not carry the confidence field used by its choice
                    // question shape.
                    let scores_valid = decision.selected_probability.is_finite()
                        && decision
                            .probabilities
                            .values()
                            .all(|value| value.is_finite());
                    let eligible = scores_valid
                        && decision.model == config.active_model()
                        && decision.selected_probability >= config.min_selected_probability
                        && selected.is_some();
                    if eligible {
                        selected_id = selected.expect("selected candidate checked").id.clone();
                        used_jev = true;
                    }
                    self.emit(AgentEvent::DecisionRouterEvaluated {
                        turn,
                        purpose: "environment_recovery".into(),
                        candidate_count: candidates.len(),
                        candidate_id: decision.candidate_id,
                        tool: selected.map(|candidate| candidate.tool.clone()),
                        description: selected.map(|candidate| candidate.description.clone()),
                        selected_probability: decision.selected_probability,
                        confidence: decision.confidence,
                        operation_probability: decision.operation_probability,
                        target_probability: decision.target_probability,
                        operation_confidence: decision.operation_confidence,
                        target_confidence: decision.target_confidence,
                        eligible,
                        rejection_reason: (!eligible).then(|| "local_recovery_fallback".into()),
                        probability_threshold: config.min_selected_probability,
                        confidence_threshold: None,
                        alternatives: decision.probabilities.into_iter().take(3).collect(),
                        elapsed_ms,
                    });
                }
                Err(error) => {
                    self.decision_router_failures = self.decision_router_failures.saturating_add(1);
                    self.log(
                        "decision_router_fallback",
                        json!({
                            "purpose": "environment_recovery",
                            "reason": error.to_string(),
                            "revision": revision,
                        }),
                    )?;
                }
            }
        }

        let mut action = selected_id.clone();
        if selected_id == "reactivate_original" {
            // A person may resume input while JEV is evaluating. Yield again
            // rather than stealing focus from a newly active interaction.
            loop {
                let activity = self.context.platform.desktop_activity_snapshot().await?;
                if activity
                    .last_physical_input_ms
                    .or(activity.last_user_input_ms)
                    .is_none_or(|elapsed| elapsed >= QUIET_MS)
                {
                    break;
                }
                if started.elapsed() >= Duration::from_millis(PAUSE_AFTER_MS) {
                    self.context.pause.request_pause();
                    self.pause_checkpoint("continuous_user_activity").await?;
                    self.emit(AgentEvent::EnvironmentRecovered {
                        revision,
                        action: "resumed_after_user_pause".into(),
                        used_jev: false,
                        elapsed_ms: started.elapsed().as_millis() as u64,
                    });
                    return Ok(());
                }
                tokio::select! {
                    () = self.context.cancellation.cancelled() => return Err(PokError::Cancelled),
                    () = tokio::time::sleep(Duration::from_millis(250)) => {}
                }
            }
            // Recheck immediately before changing focus. The platform and normal
            // capture tool retain the final safety authority.
            if !self
                .context
                .platform
                .list_windows()
                .await?
                .iter()
                .any(|window| window.id == previous.id && !window.elevated)
            {
                action = "defer_to_llm".into();
            } else if let Err(error) = self.context.platform.activate_window(&previous.id).await {
                // A shell surface such as Start or Search can keep the
                // foreground. Failing to restore focus is recoverable: hand the
                // changed environment to the primary model instead of ending
                // the run.
                self.log(
                    "environment_reactivation_failed",
                    json!({
                        "revision": revision,
                        "window": previous_label,
                        "error": error.to_string(),
                    }),
                )?;
                action = "defer_to_llm".into();
            }
        }

        if action == "observe_current" {
            let selected_window_id = current.as_ref().map(|window| window.id.as_str());
            let foreground = self.context.platform.foreground_window().await?;
            if foreground.as_ref().map(|window| window.id.as_str()) != selected_window_id
                || foreground.as_ref().is_some_and(|window| window.elevated)
            {
                action = "defer_to_llm".into();
            }
        }

        if matches!(action.as_str(), "reactivate_original" | "observe_current") {
            let window_id = if action == "reactivate_original" {
                previous.id.clone()
            } else {
                current
                    .as_ref()
                    .map(|window| window.id.clone())
                    .unwrap_or_default()
            };
            let call = CompletedToolCall {
                id: format!("environment-recovery-{}", Uuid::new_v4()),
                name: "capture_screen".into(),
                arguments: json!({"scope": "window", "window_id": window_id}),
            };
            self.messages.push(BrainMessage {
                role: "assistant".into(),
                content: Vec::new(),
                origin: MessageOrigin::Assistant,
                tool_call_id: None,
                tool_calls: vec![call.clone()],
            });
            self.emit(AgentEvent::ToolStarted {
                call_id: call.id.clone(),
                name: call.name.clone(),
                arguments: call.arguments.clone(),
            });
            let result = self
                .tools
                .call(&call.name, call.arguments.clone(), &self.context)
                .await;
            let ok = tool_finished_ok(&call.name, &result);
            self.emit(AgentEvent::ToolFinished {
                call_id: call.id.clone(),
                name: call.name.clone(),
                ok,
                detail: tool_finished_detail(&call.name, &result),
                result: tool_finished_event_result(&call.name, &result),
            });
            if let Ok(value) = &result
                && qualifies_as_fresh_evidence(&call.name, value, ok)
            {
                self.decision_router_fresh_evidence = true;
                increment_metric(metrics, "environment_fresh_evidence", 1);
            }
            self.messages
                .extend(tool_result_messages(&call, result, true));
            metrics.tool_calls = metrics.tool_calls.saturating_add(1);
        } else {
            self.messages.push(BrainMessage::text_with_origin(
                "system",
                format!(
                    "<desktop-environment-event revision=\"{revision}\">The foreground window changed from {previous_label} to {}. Earlier observations and targets were invalidated. Re-establish the intended window and obtain fresh evidence before desktop input.</desktop-environment-event>",
                    current_label.as_deref().unwrap_or("unknown")
                ),
                MessageOrigin::SystemReminder,
            ));
        }
        increment_metric(metrics, "environment_recoveries", 1);
        self.emit(AgentEvent::EnvironmentRecovered {
            revision,
            action: action.clone(),
            used_jev,
            elapsed_ms: started.elapsed().as_millis() as u64,
        });
        self.record_decision_activity(
            "environment_recovery",
            if used_jev {
                "JEV adapted to a desktop change"
            } else {
                "Local desktop recovery completed"
            },
            action.clone(),
            started.elapsed().as_millis() as u64,
        );
        self.log(
            "environment_recovered",
            json!({
                "revision": revision,
                "action": action,
                "used_jev": used_jev,
                "elapsed_ms": started.elapsed().as_millis(),
            }),
        )?;
        Ok(())
    }

    pub(super) async fn pause_checkpoint(&mut self, phase: &str) -> Result<bool> {
        if self.context.pause.state() == PauseState::Running {
            return Ok(false);
        }
        self.emit(AgentEvent::PauseRequested);
        let previous_foreground = self
            .context
            .latest_observation
            .lock()
            .as_ref()
            .and_then(|observation| observation.foreground_window.as_ref())
            .map(|window| format!("{} ({})", window.title, window.process_name));
        *self.context.latest_observation.lock() = None;
        *self.context.latest_observation_view.lock() = None;
        *self.context.pending_visual_localization.lock() = None;
        *self.context.focused_control.lock() = None;
        self.context.pause.mark_paused();
        self.log(
            "agent_paused",
            json!({
                "phase": phase,
                "previous_foreground": previous_foreground,
            }),
        )?;
        self.emit(AgentEvent::Paused {
            interrupted_phase: phase.into(),
        });
        self.context
            .pause
            .wait_until_resumed(&self.context.cancellation)
            .await?;

        let current_foreground = self
            .context
            .platform
            .foreground_window()
            .await?
            .map(|window| format!("{} ({})", window.title, window.process_name));
        let (requested_at, note) = self.context.pause.take_resume_metadata();
        let paused_ms = requested_at.map_or(0, |started| {
            u64::try_from((Utc::now() - started).num_milliseconds().max(0)).unwrap_or(u64::MAX)
        });
        let note_text = note.as_deref().map_or_else(
            || "No change note was supplied.".into(),
            |value| format!("The user supplied this change note: {value}"),
        );
        self.messages.push(BrainMessage::text_with_origin(
            "user",
            format!(
                "<desktop-control-event>The user paused the agent and took control of the shared desktop for {paused_ms} ms. \
                 Previous foreground: {}. Current foreground: {}. {note_text} \
                 All earlier observations, target ids, coordinates, and focus assumptions are invalid. \
                 Continue the same active task, but first call observe_desktop or list_windows and then capture the intended target before any desktop input.</desktop-control-event>",
                previous_foreground.as_deref().unwrap_or("unknown"),
                current_foreground.as_deref().unwrap_or("unknown"),
            ),
            MessageOrigin::UserGuidance,
        ));
        self.log(
            "agent_resumed",
            json!({
                "paused_ms": paused_ms,
                "note": note,
                "previous_foreground": previous_foreground,
                "current_foreground": current_foreground,
                "stale_observation_invalidated": true,
            }),
        )?;
        self.emit(AgentEvent::Resumed {
            paused_ms,
            note,
            previous_foreground,
            current_foreground,
        });
        Ok(true)
    }
}
