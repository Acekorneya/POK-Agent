//! Router-first decisions (System 1): intent and optional-context routing, next-action picks with refinement and judge review, and the router training-log hookup.

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DecisionRouterStep {
    Executed,
    Refine,
    Done,
    Handoff,
}

#[derive(Debug, Clone, Default)]
pub(super) struct DecisionRefinementState {
    pub(super) attempts: usize,
    pub(super) stagnant: usize,
    pub(super) best_confidence: f64,
    pub(super) state_revision: u64,
    pub(super) last_candidate_fingerprint: Option<String>,
    pub(super) repeated_candidate: usize,
    pub(super) temporarily_withheld: HashSet<String>,
    pub(super) visual_rescue_used: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum DecisionAssist {
    Candidate(String),
    Refresh,
    Unsupported,
}

pub(super) fn decision_candidate_fingerprint(candidate: &DecisionCandidate) -> String {
    let mut arguments = candidate.arguments.clone();
    if let Some(object) = arguments.as_object_mut() {
        for transient in ["observation_id", "snapshot_id", "view_id"] {
            object.remove(transient);
        }
        if let Some(label) = object
            .get("expected_label")
            .and_then(Value::as_str)
            .map(|value| value.trim().to_ascii_lowercase())
        {
            object.remove("target_id");
            object.insert("semantic_label".into(), Value::String(label));
        }
    }
    let semantic = candidate
        .description
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase();
    let digest = Sha256::digest(
        format!(
            "{}:{}:{}",
            candidate.tool,
            canonical_json(&arguments),
            semantic
        )
        .as_bytes(),
    );
    digest[..12]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

pub(super) fn decision_candidate_available(
    candidate: &DecisionCandidate,
    suppressed: &HashSet<String>,
    completed_observations: &HashSet<String>,
    temporarily_withheld: &HashSet<String>,
) -> bool {
    let fingerprint = decision_candidate_fingerprint(candidate);
    !suppressed.contains(&fingerprint)
        && !completed_observations.contains(&fingerprint)
        && !temporarily_withheld.contains(&fingerprint)
}

/// The candidate to bypass the decision router for, when exactly one
/// non-terminal candidate was offered: no judgment call exists, since
/// whatever generated the candidate set already decided it is the only
/// reversible option. Returns `None` whenever a real choice exists
/// (`actionable_len != 1`), even if `candidates` happens to contain a
/// single non-terminal entry for some other reason.
pub(super) fn structural_bypass_candidate(
    candidates: &[DecisionCandidate],
    actionable_len: usize,
) -> Option<DecisionCandidate> {
    if actionable_len != 1 {
        return None;
    }
    candidates
        .iter()
        .find(|candidate| !candidate.tool.starts_with("__"))
        .cloned()
}

pub(super) fn decision_confidence(result: &crate::decision::DecisionResult) -> f64 {
    result
        .operation_confidence
        .zip(result.target_confidence)
        .map_or(0.0, |(operation, target)| operation.min(target))
}

pub(super) fn record_uncertain_candidate(
    state: &mut DecisionRefinementState,
    fingerprint: String,
) -> bool {
    if state.last_candidate_fingerprint.as_deref() == Some(fingerprint.as_str()) {
        state.repeated_candidate = state.repeated_candidate.saturating_add(1);
    } else {
        state.last_candidate_fingerprint = Some(fingerprint.clone());
        state.repeated_candidate = 1;
    }
    if state.repeated_candidate >= 2 {
        state.temporarily_withheld.insert(fingerprint)
    } else {
        false
    }
}

pub(super) fn ordinary_decision_scores_eligible(
    decision: &crate::decision::DecisionResult,
    config: &crate::config::DecisionRouterConfig,
) -> bool {
    if config.native_choice_probabilities() {
        // Laya's `confidence` is normalized inverse entropy, not the calibrated
        // confidence emitted by JEV. Its native choice probabilities are the
        // comparable decision signal; bounded execution and verification stay
        // identical after selection.
        decision
            .operation_probability
            .is_some_and(|score| score >= config.laya.min_operation_probability)
            && decision
                .target_probability
                .is_some_and(|score| score >= config.laya.min_target_probability)
    } else {
        decision.selected_probability >= config.min_selected_probability
            && decision
                .operation_probability
                .is_some_and(|score| score >= config.min_selected_probability)
            && decision
                .operation_confidence
                .is_some_and(|score| score >= config.min_operation_confidence)
            && decision
                .target_confidence
                .is_some_and(|score| score >= config.min_confidence)
    }
}

pub(super) fn refinable_decision_rejection(reason: Option<&str>) -> bool {
    matches!(
        reason,
        Some(
            "below_probability_threshold"
                | "operation_below_probability_threshold"
                | "target_below_probability_threshold"
                | "operation_below_confidence_threshold"
                | "target_below_confidence_threshold"
                | "blocked_with_viable_candidates"
                | "done_below_operation_probability"
                | "done_below_operation_confidence"
        )
    )
}

pub(super) fn local_retrieval_intent(task: &str, current_step: &str) -> RetrievalIntent {
    let query = format!("{task} {current_step}").to_ascii_lowercase();
    if [
        "implement",
        "edit",
        "fix",
        "refactor",
        "source",
        "function",
        "code",
    ]
    .iter()
    .any(|term| query.contains(term))
    {
        RetrievalIntent::Implementation
    } else if ["explain", "understand", "how", "why", "documentation"]
        .iter()
        .any(|term| query.contains(term))
    {
        RetrievalIntent::Explanation
    } else {
        RetrievalIntent::General
    }
}

pub(super) fn decision_router_outcome_metric(outcome: &str, ok: bool) -> &'static str {
    match outcome {
        "progress" => "decision_router_progress",
        "no_progress" => "decision_router_no_progress",
        "failed" | "blocked" | "blocked_repeat" | "error" => "decision_router_failed",
        _ if ok => "decision_router_observed",
        _ => "decision_router_failed",
    }
}

impl Session {
    pub fn with_decision_router(
        mut self,
        router: Option<Arc<dyn DecisionRouter>>,
        config: crate::config::DecisionRouterConfig,
    ) -> Self {
        self.decision_router_enabled = config.enabled && router.is_some();
        self.decision_router_backend = if self.decision_router_enabled {
            config.backend
        } else {
            crate::config::DecisionRouterBackend::Off
        };
        let delegated = self.decision_router_enabled
            && config.mode == crate::config::DecisionRouterMode::Delegated;
        let training_log = config.training_log;
        self.decision_router = router;
        self.decision_router_config = Some(config);
        self.set_training_log(training_log);
        if delegated
            && let Some(MessageContent::Text { text }) = self
                .messages
                .first_mut()
                .filter(|message| message.role == "system")
                .and_then(|message| message.content.first_mut())
        {
            text.push_str("\n\nFast actions: a fast decision model is enabled. After a fresh capture, do navigation (clicks, scrolls, window switches, managed-browser clicks) through fast_actions, planning several steps in one call instead of clicking yourself. Plan the whole path you already know as one plan with then, up to 64 steps (menus, dialog fields, typed values, the confirming key; a step's input takes up to 40 keys and texts): each plan costs you a turn, and a long plan is safe because System 1 stops and hands back at the first step whose screen does not match, commits stay with you, and the whole plan counts as one action step. The result includes the new screen, so do not capture again to check it.\n- Quote exact visible text: target_hint \"list item \\\"System\\\"\", done_when \"heading \\\"Advanced display\\\" is visible\". Quoted text is checked locally; unquoted conditions are judged by the fast model.\n- Every target_hint quotes one target's name copied from the newest observation (a shortcut suffix such as \\\"(Ctrl+Shift+X)\\\" may be left off). Never quote a description (\\\"document body\\\", \\\"the first cell\\\"): for a place without a label, use input keys (Ctrl+Home, the Name Box) or a nearby labeled control. An unquoted or empty target_hint leaves the fast model to guess, so it often hands back.\n- allowed_operations: click (default), double_click (to open folders or files that a single click only selects), right_click (context menu), hover (menus that open on hover, tooltips), drag (target_hint quotes what to drag, then where to drop it: \"drag \\\"Sheet2\\\" onto \\\"Sheet1\\\"\"), scroll, activate_window, browser_click, browser_scroll.\n- then chains steps; branches picks the next steps by a when condition (use \"otherwise\" as the default); on_interrupt handles popups (click a quoted target or stop); avoid lists phrases never to click; read returns values beside quoted labels so you can answer without another capture.\n- Raw navigation tools stay withheld until a step hands back as uncertain, uncertain_branch, stalled, no_candidates, or commit_required; then they return for two turns.
- commit_required: System 1 stopped before the step that completes the request (save, send, submit, ...) because that decision is yours, not System 1's; you do not need to ask the user. The result gives the exact tool and arguments: when the request calls for that step, perform it with that one call.\n- input: keys and text for System 1 to enter in a step, in order, e.g. [{\"key\": \"Ctrl+Shift+F5\"}, {\"text\": \"B2\", \"replace_existing\": true}, {\"key\": \"Enter\"}]. With a quoted target_hint that field is focused first; without one, input goes to the focused control. You write every key and character; put a known sequence of shortcuts, menu keys, and field values in steps' input so one fast_actions call carries out the whole chain instead of one call per key. Quote in done_when what the screen should show afterward (the value in the Name Box, a cell's result, a field's text) so it is checked locally.\n- popup: a dialog opened that your plan did not expect (an error, warning, or question); the result gives its title, text, and buttons, so act on what it says. Add an on_interrupt rule for a dialog you expect.\n- System 1 never clicks a consequential commit (save, send, submit, delete) on its own; it hands it back as commit_required.");
        }
        self
    }

    /// Turn the opt-in router training log on or off for this conversation,
    /// from the next question on. Files go to
    /// `<data_dir>/router-training/<session>.jsonl`.
    pub fn set_training_log(&mut self, enabled: bool) {
        if let Some(config) = self.decision_router_config.as_mut() {
            config.training_log = enabled;
        }
        let Some(router) = self.decision_router.clone() else {
            self.training = None;
            return;
        };
        if !enabled {
            router.set_training_recorder(None);
            if self.training.take().is_some() {
                let _ = self.log("router_training_log", json!({"enabled": false}));
            }
        } else if self.training.is_none() {
            let mut excluded = self
                .decision_router_config
                .as_ref()
                .map(|config| config.training_exclude.clone())
                .unwrap_or_default();
            excluded.extend(crate::router_training::local_exclusions(
                &self.context.data_dir,
            ));
            let recorder = Arc::new(crate::router_training::TrainingRecorder::new(
                &self.context.data_dir.join("router-training"),
                self.id,
                &excluded,
            ));
            router.set_training_recorder(Some(recorder.clone()));
            let _ = self.log(
                "router_training_log",
                json!({"enabled": true, "file": recorder.path()}),
            );
            self.training = Some(recorder);
        }
    }

    /// True when the primary model plans and delegates bounded action runs
    /// to the router through `fast_actions` instead of router-first turns.
    pub(super) fn delegated_decision_router(&self) -> bool {
        self.decision_router.is_some()
            && self
                .decision_router_config
                .as_ref()
                .is_some_and(|config| config.mode == crate::config::DecisionRouterMode::Delegated)
    }

    pub(super) fn record_decision_activity(
        &mut self,
        purpose: &str,
        label: impl Into<String>,
        detail: impl Into<String>,
        elapsed_ms: u64,
    ) {
        // Activity labels are written as "JEV …"; name the backend that
        // actually answered (Laya, kev, …).
        let mut label = label.into();
        if let Some(rest) = label.strip_prefix("JEV ")
            && let Some(config) = self.decision_router_config.as_ref()
        {
            label = format!("{} {rest}", config.backend.display_name());
        }
        self.decision_activity.push(DecisionActivitySummary {
            purpose: purpose.into(),
            label,
            detail: detail.into(),
            elapsed_ms,
            recorded_at: Utc::now().to_rfc3339(),
        });
        if self.decision_activity.len() > 32 {
            self.decision_activity
                .drain(..self.decision_activity.len().saturating_sub(32));
        }
    }

    pub(super) async fn retrieval_intent(&mut self, task: &str) -> Result<RetrievalIntent> {
        let current_step = self.context.active_task.lock().current_step.clone();
        let cache_key = format!("{task}\n{current_step}");
        if let Some((cached_key, intent)) = &self.retrieval_intent_cache
            && cached_key == &cache_key
        {
            return Ok(*intent);
        }
        let local = local_retrieval_intent(task, &current_step);
        let (Some(router), Some(config)) = (
            self.decision_router.clone(),
            self.decision_router_config.clone(),
        ) else {
            self.retrieval_intent_cache = Some((cache_key, local));
            return Ok(local);
        };
        if self.decision_router_failures >= 3 {
            return Ok(local);
        }
        let candidates = [
            ("implementation", "Locate or change the source code that owns the requested behavior"),
            ("explanation", "Understand or explain behavior using code, documentation, configuration, or examples"),
            ("general", "Retrieve broadly relevant optional information without a code-implementation preference"),
        ]
        .into_iter()
        .map(|(id, description)| DecisionCandidate {
            id: id.into(),
            tool: "select_retrieval_intent".into(),
            arguments: json!({"intent": id}),
            description: description.into(),
            kind: DecisionCandidateKind::Context,
            local_score: 0.0,
        })
        .collect::<Vec<_>>();
        self.emit(AgentEvent::DecisionRouterStarted {
            turn: 0,
            purpose: "retrieval_intent".into(),
            candidate_count: candidates.len(),
        });
        let started = Instant::now();
        let result = tokio::select! {
            () = self.context.cancellation.cancelled() => return Err(PokError::Cancelled),
            result = router.decide(DecisionRequest {
                purpose: DecisionPurpose::RetrievalIntent,
                task: task.to_owned(),
                current_step,
                candidates: candidates.clone(),
                state: json!({"classification_only": true}),
            }) => result,
        };
        let (intent, jev_selected, selected_probability, confidence) = match result {
            Ok(decision)
                if decision.model == config.active_model()
                    && decision.selected_probability.is_finite()
                    && decision.selected_probability >= 0.5
                    && decision
                        .confidence
                        .is_some_and(|value| value.is_finite() && value >= 0.6) =>
            {
                self.decision_router_failures = 0;
                let intent = match decision.candidate_id.as_deref() {
                    Some("implementation") => RetrievalIntent::Implementation,
                    Some("explanation") => RetrievalIntent::Explanation,
                    Some("general") => RetrievalIntent::General,
                    _ => local,
                };
                (
                    intent,
                    true,
                    decision.selected_probability,
                    decision.confidence,
                )
            }
            Ok(decision) => (
                local,
                false,
                decision.selected_probability,
                decision.confidence,
            ),
            Err(error) => {
                self.decision_router_failures = self.decision_router_failures.saturating_add(1);
                self.log(
                    "decision_router_fallback",
                    json!({
                        "purpose": "retrieval_intent",
                        "reason": error.to_string(),
                        "consecutive_failures": self.decision_router_failures,
                    }),
                )?;
                (local, false, 0.0, None)
            }
        };
        let elapsed_ms = started.elapsed().as_millis() as u64;
        self.record_decision_activity(
            "retrieval_intent",
            "JEV classified retrieval intent",
            format!("selected {intent:?}"),
            elapsed_ms,
        );
        let selected_id = match intent {
            RetrievalIntent::Implementation => "implementation",
            RetrievalIntent::Explanation => "explanation",
            RetrievalIntent::General => "general",
        };
        let selected = candidates
            .iter()
            .find(|candidate| candidate.id == selected_id);
        self.emit(AgentEvent::DecisionRouterEvaluated {
            turn: 0,
            purpose: "retrieval_intent".into(),
            candidate_count: candidates.len(),
            candidate_id: Some(selected_id.into()),
            tool: selected.map(|candidate| candidate.tool.clone()),
            description: selected.map(|candidate| candidate.description.clone()),
            selected_probability,
            confidence,
            operation_probability: None,
            target_probability: None,
            operation_confidence: None,
            target_confidence: None,
            eligible: jev_selected,
            rejection_reason: (!jev_selected).then(|| "local_intent_fallback".into()),
            probability_threshold: 0.5,
            confidence_threshold: Some(0.6),
            alternatives: Vec::new(),
            elapsed_ms,
        });
        self.log(
            "retrieval_intent_selected",
            json!({
                "intent": intent,
                "local_fallback": !jev_selected,
                "selected_probability": selected_probability,
                "confidence": confidence,
                "elapsed_ms": elapsed_ms,
            }),
        )?;
        self.retrieval_intent_cache = Some((cache_key, intent));
        Ok(intent)
    }

    pub(super) async fn route_optional_context(
        &mut self,
        task: &str,
        candidates: Vec<DecisionCandidate>,
    ) -> Result<Vec<String>> {
        let (Some(router), Some(config)) = (
            self.decision_router.clone(),
            self.decision_router_config.clone(),
        ) else {
            return Ok(Vec::new());
        };
        if candidates.is_empty() || self.decision_router_failures >= 3 {
            return Ok(Vec::new());
        }
        let cache_key = {
            let mut hasher = Sha256::new();
            hasher.update(task.as_bytes());
            hasher.update(config.active_model().as_bytes());
            for candidate in &candidates {
                hasher.update(candidate.id.as_bytes());
                hasher.update(candidate.description.as_bytes());
            }
            format!("{:x}", hasher.finalize())
        };
        if let Some(selected) = self.decision_context_cache.get(&cache_key).cloned() {
            self.record_decision_activity(
                "context_selection",
                "JEV reused an unchanged context ranking",
                format!("reused {} selected candidates", selected.len()),
                0,
            );
            return Ok(selected);
        }
        self.emit(AgentEvent::DecisionRouterStarted {
            turn: 0,
            purpose: "context_selection".into(),
            candidate_count: candidates.len(),
        });
        let started = Instant::now();
        let decision = tokio::select! {
            () = self.context.cancellation.cancelled() => return Err(PokError::Cancelled),
            result = router.decide(DecisionRequest {
                purpose: DecisionPurpose::ContextSelection,
                task: task.to_owned(),
                current_step: "Select optional retrieved context for the primary model".into(),
                candidates: candidates.clone(),
                state: json!({"optional_context_only": true}),
            }) => match result {
                Ok(decision) => {
                    self.decision_router_failures = 0;
                    decision
                }
                Err(error) => {
                    self.decision_router_failures = self.decision_router_failures.saturating_add(1);
                    self.record_decision_activity(
                        "context_selection",
                        "Context selection used local ranking",
                        error.to_string(),
                        started.elapsed().as_millis() as u64,
                    );
                    self.log("decision_router_fallback", json!({
                        "purpose": "context_selection",
                        "reason": error.to_string(),
                        "consecutive_failures": self.decision_router_failures,
                    }))?;
                    return Ok(Vec::new());
                }
            },
        };
        let selected = decision
            .candidate_id
            .as_deref()
            .and_then(|id| candidates.iter().find(|candidate| candidate.id == id));
        let scores_are_valid = decision.selected_probability.is_finite()
            && (0.0..=1.0).contains(&decision.selected_probability)
            && decision
                .probabilities
                .values()
                .all(|score| score.is_finite() && (0.0..=1.0).contains(score));
        let eligible = scores_are_valid
            && decision.model == config.active_model()
            && decision.selected_probability >= 0.5
            && selected.is_some();
        let rejection_reason = if !scores_are_valid {
            Some("invalid_scores")
        } else if decision.candidate_id.is_none() {
            Some("model_fallback")
        } else if selected.is_none() {
            Some("unknown_candidate")
        } else if decision.model != config.active_model() {
            Some("model_mismatch")
        } else if decision.selected_probability < 0.5 {
            Some("below_probability_threshold")
        } else {
            None
        };
        let mut alternatives = decision
            .probabilities
            .iter()
            .map(|(id, score)| (id.clone(), *score))
            .collect::<Vec<_>>();
        alternatives.sort_by(|left, right| right.1.total_cmp(&left.1));
        alternatives.truncate(3);
        let elapsed_ms = started.elapsed().as_millis() as u64;
        self.record_decision_activity(
            "context_selection",
            if eligible {
                "JEV selected optional context"
            } else {
                "Context selection used local ranking"
            },
            rejection_reason.unwrap_or("selected"),
            elapsed_ms,
        );
        self.emit(AgentEvent::DecisionRouterEvaluated {
            turn: 0,
            purpose: "context_selection".into(),
            candidate_count: candidates.len(),
            candidate_id: decision.candidate_id.clone(),
            tool: selected.map(|candidate| candidate.tool.clone()),
            description: selected.map(|candidate| candidate.description.clone()),
            selected_probability: decision.selected_probability,
            confidence: decision.confidence,
            operation_probability: decision.operation_probability,
            target_probability: decision.target_probability,
            operation_confidence: decision.operation_confidence,
            target_confidence: decision.target_confidence,
            eligible,
            rejection_reason: rejection_reason.map(str::to_owned),
            probability_threshold: 0.5,
            confidence_threshold: None,
            alternatives,
            elapsed_ms,
        });
        self.log(
            "decision_router_context",
            json!({
                "eligible": eligible,
                "candidate_count": candidates.len(),
                "selected": selected.map(|candidate| &candidate.id),
                "rejection_reason": rejection_reason,
            }),
        )?;
        let mut selected_ids = if scores_are_valid && decision.model == config.active_model() {
            decision
                .probabilities
                .iter()
                .filter(|(id, score)| {
                    **score >= 0.5 && candidates.iter().any(|candidate| candidate.id == **id)
                })
                .map(|(id, score)| (id.clone(), *score))
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        selected_ids.sort_by(|left, right| right.1.total_cmp(&left.1));
        selected_ids.truncate(3);
        if selected_ids.is_empty() {
            let recovery_candidates = candidates
                .iter()
                .take(16)
                .cloned()
                .map(|mut candidate| {
                    if let Some(text) = candidate.arguments.get("text").and_then(Value::as_str) {
                        candidate.description = format!(
                            "{}\nFuller bounded preview:\n{}",
                            candidate.description,
                            truncate_chars(text, 1_200)
                        );
                    }
                    candidate
                })
                .collect::<Vec<_>>();
            let changed_evidence = recovery_candidates
                .iter()
                .zip(candidates.iter())
                .any(|(recovery, original)| recovery.description != original.description);
            if !recovery_candidates.is_empty() && changed_evidence {
                let recovery_started = Instant::now();
                let recovery = tokio::select! {
                    () = self.context.cancellation.cancelled() => return Err(PokError::Cancelled),
                    result = router.decide(DecisionRequest {
                        purpose: DecisionPurpose::ContextSelection,
                        task: task.to_owned(),
                        current_step: "Recovery: select directly useful optional context from fuller bounded previews".into(),
                        candidates: recovery_candidates.clone(),
                        state: json!({"optional_context_only": true, "recovery": true}),
                    }) => result,
                };
                if let Ok(recovery) = recovery
                    && recovery.model == config.active_model()
                    && recovery
                        .probabilities
                        .values()
                        .all(|score| score.is_finite() && (0.0..=1.0).contains(score))
                {
                    let mut recovered = recovery
                        .probabilities
                        .into_iter()
                        .filter(|(id, score)| {
                            *score >= 0.5
                                && recovery_candidates
                                    .iter()
                                    .any(|candidate| candidate.id == *id)
                        })
                        .collect::<Vec<_>>();
                    recovered.sort_by(|left, right| right.1.total_cmp(&left.1));
                    recovered.truncate(3);
                    selected_ids = recovered;
                }
                self.log(
                    "decision_router_context_recovery",
                    json!({
                        "candidate_count": recovery_candidates.len(),
                        "selected_count": selected_ids.len(),
                        "elapsed_ms": recovery_started.elapsed().as_millis(),
                    }),
                )?;
            }
        }
        let selected_ids = selected_ids
            .into_iter()
            .map(|(id, _)| id)
            .collect::<Vec<_>>();
        if self.decision_context_cache.len() >= 32
            && let Some(oldest) = self.decision_context_cache.keys().next().cloned()
        {
            self.decision_context_cache.remove(&oldest);
        }
        self.decision_context_cache
            .insert(cache_key, selected_ids.clone());
        Ok(selected_ids)
    }

    pub(super) async fn bounded_json_response(
        &self,
        mut request: BrainRequest,
        helper: &str,
    ) -> Result<Value> {
        request.reasoning_effort = self.bounded_reasoning_effort.clone();
        if self.single_system_message_required {
            request = request_with_single_leading_system_message(request);
        }
        if self.alternating_roles_required {
            request = request_with_alternating_conversation_roles(request);
        }
        self.log(
            "primary_model_helper_request",
            json!({"role": helper, "tools": 0, "reasoning_effort": &request.reasoning_effort}),
        )?;
        let mut stream = self.brain.stream(request);
        let mut text = String::new();
        let mut reasoning = String::new();
        loop {
            let event = tokio::select! {
                () = self.context.cancellation.cancelled() => return Err(PokError::Cancelled),
                event = stream.next() => event,
            };
            let Some(event) = event else { break };
            match event? {
                BrainEvent::TextDelta { text: delta } => text.push_str(&delta),
                BrainEvent::ReasoningDelta { text: delta } => reasoning.push_str(&delta),
                _ => {}
            }
        }
        let value = bounded_json_value(&text, &reasoning);
        self.log(
            "primary_model_helper_response",
            json!({
                "role": helper,
                "valid_json": value.is_some(),
                "visible_chars": text.chars().count(),
                "reasoning_chars": reasoning.chars().count(),
            }),
        )?;
        value.ok_or_else(|| PokError::Provider(format!("{helper} returned no valid JSON object")))
    }

    pub(super) async fn generate_decision_action_text(
        &self,
        task: &str,
        target_description: &str,
    ) -> Result<String> {
        let request = BrainRequest {
            model: self.model.clone(),
            messages: vec![
                BrainMessage::text(
                    "system",
                    "You supply one missing text value for a structured computer-use action. You have no tools. Follow the user request, do not add commentary, and never provide a password or authentication secret. Return JSON only: {\"text\":\"...\"}.",
                ),
                BrainMessage::text(
                    "user",
                    format!(
                        "ACTIVE USER REQUEST:\n{}\n\nTARGET FIELD:\n{}",
                        task,
                        target_description.chars().take(600).collect::<String>()
                    ),
                ),
            ],
            tools: Vec::new(),
            temperature: Some(0.0),
            max_tokens: Some(512),
            seed: self.agent_seed,
            reasoning_effort: self.bounded_reasoning_effort.clone(),
        };
        let value = self.bounded_json_response(request, "text_helper").await?;
        let text = value
            .get("text")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty() && value.len() <= 8_000)
            .ok_or_else(|| PokError::Provider("text helper returned no bounded text".into()))?;
        Ok(text.to_owned())
    }

    pub(super) async fn review_decision_commit(
        &self,
        task: &str,
        action: &str,
        observation: Option<&Observation>,
    ) -> Result<bool> {
        let pending = self.context.input_ledger.lock().pending.clone();
        let mut content = vec![MessageContent::Text {
            text: format!(
                "ACTIVE USER REQUEST:\n{}\n\nPROPOSED COMMIT ACTION:\n{}\n\nPENDING DRAFT:\n{}\n\nDESTINATION:\n{}\n\nReturn JSON only.",
                task,
                action.chars().take(800).collect::<String>(),
                pending
                    .as_ref()
                    .map(|pending| pending.text.chars().take(4_000).collect::<String>())
                    .unwrap_or_default(),
                pending
                    .as_ref()
                    .map(|pending| pending.destination_label.as_str())
                    .unwrap_or("current verified destination"),
            ),
        }];
        if self.image_input_override != Some(false)
            && let Some(base64) = observation
                .and_then(|observation| observation.screenshots.first())
                .map(|screenshot| screenshot.png_base64.clone())
        {
            content.push(MessageContent::ImagePng { base64 });
        }
        let request = BrainRequest {
            model: self.model.clone(),
            messages: vec![
                BrainMessage::text(
                    "system",
                    "Review one consequential computer-use commit immediately before execution. Approve only when it matches the active user request, destination, draft, and fresh evidence. You have no tools. Return JSON only: {\"approve\":true|false,\"reason\":\"...\"}.",
                ),
                BrainMessage {
                    role: "user".into(),
                    content,
                    origin: MessageOrigin::UserInput,
                    tool_call_id: None,
                    tool_calls: Vec::new(),
                },
            ],
            tools: Vec::new(),
            temperature: Some(0.0),
            max_tokens: Some(512),
            seed: self.agent_seed,
            reasoning_effort: self.bounded_reasoning_effort.clone(),
        };
        Ok(self
            .bounded_json_response(request, "safety_review")
            .await?
            .get("approve")
            .and_then(Value::as_bool)
            == Some(true))
    }

    pub(super) async fn assist_decision_target(
        &self,
        task: &str,
        current_step: &str,
        candidates: &[DecisionCandidate],
        state: &Value,
        observation: Option<&Observation>,
    ) -> Result<DecisionAssist> {
        let bounded_candidates = candidates
            .iter()
            .filter(|candidate| !candidate.tool.starts_with("__"))
            .take(12)
            .map(|candidate| {
                json!({
                    "id": candidate.id,
                    "tool": candidate.tool,
                    "description": candidate.description.chars().take(500).collect::<String>(),
                })
            })
            .collect::<Vec<_>>();
        let mut content = vec![MessageContent::Text {
            text: format!(
                "ACTIVE USER REQUEST:\n{}\n\nCURRENT STEP:\n{}\n\nCURRENT STRUCTURED STATE:\n{}\n\nALLOWED CANDIDATES:\n{}\n\nReturn JSON only.",
                task.chars().take(1_500).collect::<String>(),
                current_step.chars().take(750).collect::<String>(),
                serde_json::to_string(state)?
                    .chars()
                    .take(6_000)
                    .collect::<String>(),
                serde_json::to_string(&bounded_candidates)?,
            ),
        }];
        if self.image_input_override != Some(false)
            && let Some(base64) = observation
                .and_then(|observation| observation.screenshots.first())
                .map(|screenshot| screenshot.png_base64.clone())
        {
            content.push(MessageContent::ImagePng { base64 });
        }
        let request = BrainRequest {
            model: self.model.clone(),
            messages: vec![
                BrainMessage::text(
                    "system",
                    "You are a bounded visual assistant for JEV. You have no tools and may not invent an action. Select exactly one supplied candidate only when current evidence supports it. Otherwise request one refresh or report unsupported. Return JSON only as {\"candidate_id\":\"id\"}, {\"refresh\":true}, or {\"unsupported\":true}.",
                ),
                BrainMessage {
                    role: "user".into(),
                    content,
                    origin: MessageOrigin::UserInput,
                    tool_call_id: None,
                    tool_calls: Vec::new(),
                },
            ],
            tools: Vec::new(),
            temperature: Some(0.0),
            max_tokens: Some(512),
            seed: self.agent_seed,
            reasoning_effort: self.bounded_reasoning_effort.clone(),
        };
        let value = match self
            .bounded_json_response(request.clone(), "visual_rescue")
            .await
        {
            Ok(value) => value,
            Err(error) if request_contains_images(&request) && is_image_input_rejection(&error) => {
                let text_only_request = request_without_images(request);
                self.bounded_json_response(text_only_request, "visual_rescue")
                    .await?
            }
            Err(error) => return Err(error),
        };
        if let Some(id) = value.get("candidate_id").and_then(Value::as_str)
            && bounded_candidates
                .iter()
                .any(|candidate| candidate.get("id").and_then(Value::as_str) == Some(id))
        {
            return Ok(DecisionAssist::Candidate(id.to_owned()));
        }
        if value.get("refresh").and_then(Value::as_bool) == Some(true) {
            return Ok(DecisionAssist::Refresh);
        }
        Ok(DecisionAssist::Unsupported)
    }

    pub(super) async fn try_decision_router_action(
        &mut self,
        turn: u32,
        metrics: &mut RunMetrics,
    ) -> Result<DecisionRouterStep> {
        let (Some(router), Some(config)) = (
            self.decision_router.clone(),
            self.decision_router_config.clone(),
        ) else {
            return Ok(DecisionRouterStep::Handoff);
        };
        if self.decision_router_failures >= 3 {
            return Ok(DecisionRouterStep::Handoff);
        }
        let observation = self.context.latest_observation.lock().clone();
        let elevated = observation.as_ref().is_some_and(|observation| {
            observation
                .foreground_window
                .as_ref()
                .is_some_and(|window| window.elevated)
        });
        let task = self.context.active_task.lock().clone();
        let web_task = requests_managed_web_search(&task.root_request)
            || requests_managed_web_search(&task.current_step);
        let desktop_task = requests_desktop_interaction(&task.root_request)
            || requests_desktop_interaction(&task.current_step)
            || (!web_task
                && (requests_desktop_action(&task.root_request)
                    || requests_desktop_action(&task.current_step)));
        // A re-observation/refresh bumps the state revision, but it must not
        // reset the refinement budget or the withheld-candidate set: doing so
        // re-offered the same uncertain candidate after every refresh and let the
        // loop run indefinitely (live traces showed one candidate re-picked eight
        // times with identical scores). The budget and withhold now persist for
        // the run; the primary-model handoff remains the safety valve.
        self.decision_router_refinement.state_revision = self.continuity.state_revision;
        let mut candidates = coding_candidates(
            &task.root_request,
            &self.context.workspace,
            config.max_candidates.min(250),
        );
        candidates.extend(harness_capability_candidates(
            &task.root_request,
            &task.current_step,
            &self.tools.names(),
            &self.active_tool_groups,
            config.max_candidates.min(250),
        ));
        // If the task asks for an application that is not visible, offer a
        // launch candidate. Without this the fast loop could only activate or
        // click existing windows and would otherwise click around looking for
        // an application that was never open.
        if let Some(application) = requested_application(
            &task.root_request,
            &task.current_step,
            observation.as_ref(),
            self.decision_router_last_result.as_ref(),
            &self.decision_router_launched_applications,
        ) {
            candidates.push(DecisionCandidate {
                id: "open_application".into(),
                tool: "open_application".into(),
                arguments: json!({ "name": application }),
                description: format!(
                    "Launch the {application} application because the task needs it and no matching window is visible; then list and activate its window before interacting"
                ),
                kind: DecisionCandidateKind::Action,
                local_score: 0.9,
            });
        }
        if desktop_task
            && !elevated
            && let Some(observation) = &observation
        {
            candidates.extend(desktop_candidates_for_task(
                observation,
                &task.root_request,
                &task.current_step,
                config.max_candidates.min(250),
            ));
        }
        if web_task {
            candidates.extend(
                crate::browser::decision_candidates(
                    &self.context,
                    &task.root_request,
                    &task.current_step,
                    config.max_candidates.min(250),
                )
                .await,
            );
        }
        if desktop_task && observation.is_none() {
            match self.decision_router_last_result.as_ref() {
                Some((tool, value)) if tool == "list_windows" => {
                    for (index, window) in value
                        .get("windows")
                        .and_then(Value::as_array)
                        .into_iter()
                        .flatten()
                        .take(config.max_candidates.min(100))
                        .enumerate()
                    {
                        let Some(window_id) = window.get("id").and_then(Value::as_str) else {
                            continue;
                        };
                        let title = window.get("title").and_then(Value::as_str).unwrap_or("");
                        let app = window.get("app").and_then(Value::as_str).unwrap_or("");
                        candidates.push(DecisionCandidate {
                            id: format!("activate_window_{index}"),
                            tool: "activate_window".into(),
                            arguments: json!({"window_id": window_id}),
                            description: format!("{title:?} ({app} window)"),
                            kind: DecisionCandidateKind::Action,
                            local_score: 0.8,
                        });
                    }
                }
                Some((tool, value)) if tool == "activate_window" => {
                    if let Some(window_id) = value.get("id").and_then(Value::as_str) {
                        candidates.push(DecisionCandidate {
                            id: "capture_activated_window".into(),
                            tool: "capture_screen".into(),
                            arguments: json!({
                                "scope": "window",
                                "window_id": window_id,
                                "include_ocr": true,
                                "include_ui_tree": true,
                                "enrichment": "fast"
                            }),
                            description: "Capture fresh structured state for the activated window"
                                .into(),
                            kind: DecisionCandidateKind::Action,
                            local_score: 1.0,
                        });
                    }
                }
                _ => candidates.push(DecisionCandidate {
                    id: "list_visible_windows".into(),
                    tool: "list_windows".into(),
                    arguments: json!({}),
                    description: "List visible desktop windows to identify the application relevant to the task"
                        .into(),
                    kind: DecisionCandidateKind::Action,
                    local_score: 0.7,
                }),
            }
        }
        // An explicit URL or domain navigates directly; searching for the literal
        // request text would land on a search page instead of the requested site.
        if let Some(url) = requested_url(&task.root_request, &task.current_step)
            && !candidates
                .iter()
                .any(|candidate| candidate.tool.starts_with("managed_browser_"))
        {
            candidates.push(DecisionCandidate {
                id: "open_explicit_url".into(),
                tool: "managed_browser_open".into(),
                arguments: json!({"url": url}),
                description: format!(
                    "Open {url} directly in the isolated managed browser and read its content"
                ),
                kind: DecisionCandidateKind::Action,
                local_score: 1.0,
            });
        }
        if web_task
            && !candidates
                .iter()
                .any(|candidate| candidate.tool.starts_with("managed_browser_"))
            && let Ok(mut url) = reqwest::Url::parse(&config.search_url)
        {
            url.query_pairs_mut()
                .append_pair("q", task.root_request.trim());
            candidates.push(DecisionCandidate {
                id: "search_web_for_request".into(),
                tool: "managed_browser_open".into(),
                arguments: json!({"url": url.as_str()}),
                description:
                    "Search the web in the isolated managed browser using the complete user request"
                        .into(),
                kind: DecisionCandidateKind::Action,
                local_score: 0.9,
            });
        }
        if desktop_task
            && candidates.is_empty()
            && observation
                .as_ref()
                .is_some_and(|observation| !observation.screenshots.is_empty())
        {
            increment_metric(metrics, "decision_router_vision_handoffs", 1);
            self.record_decision_activity(
                "next_action",
                "JEV handed ambiguous visual state to the primary model",
                "no reliable structured candidate",
                0,
            );
            self.log(
                "decision_router_vision_handoff",
                json!({"turn": turn, "reason": "no_reliable_structured_candidate"}),
            )?;
            return Ok(DecisionRouterStep::Handoff);
        }
        candidates.retain(|candidate| {
            let signature = action_signature(
                &candidate.tool,
                &candidate.arguments,
                observation.as_ref(),
                self.continuity.state_revision,
            );
            !self.decision_router_actions.contains(&signature)
                && decision_candidate_available(
                    candidate,
                    &self.decision_router_suppressed_candidates,
                    &self.decision_router_completed_observations,
                    &self.decision_router_refinement.temporarily_withheld,
                )
        });
        let done_description = if self.decision_router_fresh_evidence {
            if let Some((tool, _)) = self.decision_router_last_result.as_ref() {
                format!(
                    "Current verified evidence is sufficient (observed from {tool}); stop the fast action loop and let the primary model explain the result to the user"
                )
            } else {
                "Current verified evidence is sufficient; stop the fast action loop and let the primary model explain the result to the user"
                    .into()
            }
        } else {
            "Current verified evidence is sufficient; stop the fast action loop and let the primary model explain the result to the user"
                .into()
        };
        let blocked_description = if let Some(obs) = observation.as_ref() {
            let window_title = obs
                .foreground_window
                .as_ref()
                .map(|w| w.title.as_str())
                .unwrap_or("the current window");
            format!(
                "No offered reversible structured action in {window_title:?} can safely advance the active task; hand control to the primary model"
            )
        } else {
            "No offered reversible structured action can safely advance the task; hand control to the primary model"
                .into()
        };
        candidates.push(DecisionCandidate {
            id: "jev_blocked".into(),
            tool: "__blocked__".into(),
            arguments: json!({}),
            description: blocked_description,
            kind: DecisionCandidateKind::Action,
            local_score: 0.0,
        });
        // DONE is only offered once this run has produced fresh evidence.
        // Offering it earlier invited picks the terminal gate must always
        // reject (done_without_fresh_evidence), which then cost a judge
        // round trip and a handoff per turn instead of letting the picker
        // choose the observation/action that actually gathers evidence.
        if self.decision_router_fresh_evidence {
            candidates.push(DecisionCandidate {
                id: "jev_done".into(),
                tool: "__done__".into(),
                arguments: json!({}),
                description: done_description,
                kind: DecisionCandidateKind::Action,
                local_score: 0.0,
            });
        }
        let (mut terminal, actionable): (Vec<_>, Vec<_>) = candidates
            .into_iter()
            .partition(|candidate| candidate.tool.starts_with("__"));
        // Exactly one non-terminal candidate means no judgment call exists:
        // whatever generated the candidate set already decided this is the
        // only reversible option, so asking the router to score it wastes a
        // round trip and re-litigates a decision that was already made by
        // the candidate-generation code. Bounded offline+live testing on
        // this exact case shape showed the router's own probability/
        // committee judgment is unreliable here (worse than chance on
        // forced single-candidate scenes), while skipping it costs nothing
        // and changes nothing when the single candidate is genuinely safe.
        let actionable_len = actionable.len();
        let candidate_limit = if self.decision_router_refinement.attempts > 0 {
            config.max_candidates.min(8)
        } else {
            config.max_candidates
        };
        let mut candidates = rank_and_limit_candidates(
            actionable,
            candidate_limit.saturating_sub(terminal.len()).max(1),
        );
        candidates.append(&mut terminal);
        if candidates.is_empty() {
            return Ok(DecisionRouterStep::Handoff);
        }
        let browser_state = if web_task {
            crate::browser::decision_state(&self.context).await
        } else {
            json!({})
        };
        let has_browser_state = browser_state
            .as_object()
            .is_some_and(|state| !state.is_empty());
        let last_action = self
            .decision_router_last_result
            .as_ref()
            .map(|(tool, result)| {
                if serde_json::to_vec(result).is_ok_and(|encoded| encoded.len() <= 4_096) {
                    json!({"tool": tool, "result": result})
                } else {
                    json!({"tool": tool, "result_available": true, "result_omitted": "oversized"})
                }
            });
        // Only evidence gathered during the current run authorizes a terminal
        // DONE. A stale observation or browser snapshot from a previous prompt is
        // not fresh evidence for the new request.
        let has_fresh_structured_evidence = self.decision_router_fresh_evidence;
        let state = json!({
            "window": observation.as_ref().and_then(|observation| observation.foreground_window.as_ref()).map(|window| json!({
                "application": window.process_name,
                "title": window.title,
            })),
            "browser": browser_state,
            "last_action": last_action.clone(),
            "previous_outcome": self.continuity.current.as_ref().map(|state| &state.outcome),
            "state_revision": self.continuity.state_revision,
            "refinement": {
                "attempt": self.decision_router_refinement.attempts,
                "stagnant": self.decision_router_refinement.stagnant,
                "best_confidence": self.decision_router_refinement.best_confidence,
                "suppressed_candidates": self.decision_router_suppressed_candidates.len(),
                "temporarily_withheld": self.decision_router_refinement.temporarily_withheld.len(),
                "visual_rescue_used": self.decision_router_refinement.visual_rescue_used,
            },
        });
        let cache_key = {
            let mut hasher = Sha256::new();
            hasher.update(task.root_request.as_bytes());
            hasher.update(task.current_step.as_bytes());
            hasher.update(self.continuity.state_revision.to_le_bytes());
            hasher.update(self.decision_router_refinement.attempts.to_le_bytes());
            if let Some(last_action) = &last_action {
                hasher.update(serde_json::to_vec(last_action)?);
            }
            for candidate in &candidates {
                hasher.update(candidate.id.as_bytes());
                hasher.update(candidate.description.as_bytes());
            }
            format!("{:x}", hasher.finalize())
        };
        if self.decision_router_cache_key.as_deref() == Some(cache_key.as_str()) {
            increment_metric(metrics, "decision_router_cache_hits", 1);
            self.record_decision_activity(
                "next_action",
                "JEV skipped an unchanged decision",
                "unchanged state avoided a repeated JEV request",
                0,
            );
            self.emit(AgentEvent::DecisionRouterCacheHit {
                turn,
                purpose: "next_action".into(),
                candidate_count: candidates.len(),
            });
            self.log(
                "decision_router_cache_hit",
                json!({"turn": turn, "candidate_count": candidates.len(), "purpose": "next_action"}),
            )?;
            return Ok(DecisionRouterStep::Handoff);
        }
        self.decision_router_cache_key = Some(cache_key);
        let started = Instant::now();
        let decision = if let Some(bypass_candidate) =
            structural_bypass_candidate(&candidates, actionable_len)
        {
            self.log(
                "decision_router_structural_bypass",
                json!({
                    "turn": turn,
                    "tool": bypass_candidate.tool,
                    "candidate_id": bypass_candidate.id,
                }),
            )?;
            crate::decision::DecisionResult {
                candidate_id: Some(bypass_candidate.id.clone()),
                selected_probability: 1.0,
                confidence: Some(1.0),
                operation: None,
                operation_probability: Some(1.0),
                target_probability: Some(1.0),
                operation_confidence: Some(1.0),
                target_confidence: Some(1.0),
                redacted_state_fields: 0,
                dropped_candidates: 0,
                model: config.active_model().to_string(),
                probabilities: BTreeMap::from([(bypass_candidate.id, 1.0)]),
                progress_probability: None,
                backend_metadata: Some(json!({"bypass": "single_candidate"})),
            }
        } else {
            self.emit(AgentEvent::DecisionRouterStarted {
                turn,
                purpose: "next_action".into(),
                candidate_count: candidates.len(),
            });
            let decision_call = router.decide(DecisionRequest {
                purpose: DecisionPurpose::NextAction,
                task: task.root_request.clone(),
                current_step: task.current_step.clone(),
                candidates: candidates.clone(),
                state: state.clone(),
            });
            let result = tokio::select! {
                () = self.context.cancellation.cancelled() => return Err(PokError::Cancelled),
                result = decision_call => result,
            };
            match result {
                Ok(decision) => {
                    self.decision_router_failures = 0;
                    decision
                }
                Err(error) => {
                    let locally_withheld = error.to_string().contains("sensitive-data filter");
                    if !locally_withheld {
                        self.decision_router_failures =
                            self.decision_router_failures.saturating_add(1);
                    }
                    self.record_decision_activity(
                        "next_action",
                        "JEV used local fallback",
                        error.to_string(),
                        started.elapsed().as_millis() as u64,
                    );
                    self.log(
                        "decision_router_fallback",
                        json!({
                            "turn": turn,
                            "reason": error.to_string(),
                            "consecutive_failures": self.decision_router_failures,
                            "local_data_boundary_handoff": locally_withheld,
                            "elapsed_ms": started.elapsed().as_millis(),
                        }),
                    )?;
                    return Ok(DecisionRouterStep::Handoff);
                }
            }
        };
        let selected = decision
            .candidate_id
            .as_deref()
            .and_then(|id| candidates.iter().find(|candidate| candidate.id == id));
        let selected_tool = selected.map(|candidate| candidate.tool.as_str());
        let probability_threshold = config.min_selected_probability;
        let operation_probability_threshold = if config.native_choice_probabilities() {
            config.laya.min_operation_probability
        } else {
            probability_threshold
        };
        let target_probability_threshold = if config.native_choice_probabilities() {
            config.laya.min_target_probability
        } else {
            probability_threshold
        };
        let terminal_probability_threshold = if config.native_choice_probabilities() {
            config.laya.min_terminal_probability
        } else {
            config.min_terminal_probability
        };
        let viable_action_count = candidates
            .iter()
            .filter(|candidate| !candidate.tool.starts_with("__"))
            .count();
        let scores_are_valid = decision.selected_probability.is_finite()
            && (0.0..=1.0).contains(&decision.selected_probability)
            && decision
                .probabilities
                .values()
                .all(|probability| probability.is_finite() && (0.0..=1.0).contains(probability));
        let ordinary_eligible = scores_are_valid
            && decision.model == config.active_model()
            && selected.is_some()
            && ordinary_decision_scores_eligible(&decision, &config);
        let terminal_done = selected_tool == Some("__done__");
        let terminal_blocked = selected_tool == Some("__blocked__");
        let terminal_confidence_eligible = config.native_choice_probabilities()
            || decision
                .operation_confidence
                .is_some_and(|score| score >= config.min_terminal_confidence);
        let terminal_eligible = scores_are_valid
            && decision.model == config.active_model()
            && selected.is_some()
            && decision
                .operation_probability
                .is_some_and(|score| score >= terminal_probability_threshold)
            && terminal_confidence_eligible
            && has_fresh_structured_evidence;
        let blocked_eligible = scores_are_valid
            && decision.model == config.active_model()
            && selected.is_some()
            && viable_action_count == 0
            && decision
                .operation_probability
                .is_some_and(|score| score >= terminal_probability_threshold)
            && terminal_confidence_eligible;
        let eligible = if terminal_done {
            terminal_eligible
        } else if terminal_blocked {
            blocked_eligible
        } else {
            ordinary_eligible
        };
        let rejection_reason = if !scores_are_valid {
            Some("invalid_scores")
        } else if decision.candidate_id.is_none() {
            Some("model_fallback")
        } else if selected.is_none() {
            Some("unknown_candidate")
        } else if decision.model != config.active_model() {
            Some("model_mismatch")
        } else if terminal_blocked && viable_action_count > 0 {
            Some("blocked_with_viable_candidates")
        } else if terminal_done && !has_fresh_structured_evidence {
            Some("done_without_fresh_evidence")
        } else if terminal_done
            && decision
                .operation_probability
                .is_none_or(|score| score < terminal_probability_threshold)
        {
            Some("done_below_operation_probability")
        } else if terminal_done
            && !config.native_choice_probabilities()
            && decision
                .operation_confidence
                .is_none_or(|score| score < config.min_terminal_confidence)
        {
            Some("done_below_operation_confidence")
        } else if !terminal_done
            && !terminal_blocked
            && !config.native_choice_probabilities()
            && decision
                .operation_probability
                .is_none_or(|score| score < operation_probability_threshold)
        {
            Some("operation_below_probability_threshold")
        } else if !terminal_done
            && !terminal_blocked
            && config.native_choice_probabilities()
            && decision
                .operation_probability
                .is_none_or(|score| score < config.laya.min_operation_probability)
        {
            // ordinary_decision_scores_eligible requires both operation_probability
            // and target_probability to clear their thresholds for Laya; this arm
            // was missing, so a decision rejected only on operation_probability
            // (target_probability fine) fell through to `None` below, which is not
            // in refinable_decision_rejection's list — it skipped the local Refine
            // retry and went straight to Handoff instead of getting a second,
            // narrower attempt.
            Some("operation_below_probability_threshold")
        } else if !terminal_done
            && !terminal_blocked
            && config.native_choice_probabilities()
            && decision
                .target_probability
                .is_none_or(|score| score < target_probability_threshold)
        {
            Some("target_below_probability_threshold")
        } else if !terminal_done
            && !terminal_blocked
            && !config.native_choice_probabilities()
            && decision
                .operation_confidence
                .is_none_or(|score| score < config.min_operation_confidence)
        {
            Some("operation_below_confidence_threshold")
        } else if !terminal_done
            && !terminal_blocked
            && decision
                .target_confidence
                .is_none_or(|score| score < config.min_confidence)
        {
            Some("target_below_confidence_threshold")
        } else if decision.selected_probability < probability_threshold {
            Some("below_probability_threshold")
        } else {
            None
        };
        let uncertain_decision = refinable_decision_rejection(rejection_reason);
        let elapsed_ms = started.elapsed().as_millis() as u64;
        increment_metric(metrics, "decision_router_evaluations", 1);
        increment_metric(metrics, "decision_router_latency_ms", elapsed_ms);
        if decision.redacted_state_fields > 0 || decision.dropped_candidates > 0 {
            increment_metric(
                metrics,
                "decision_router_redacted_fields",
                decision.redacted_state_fields as u64,
            );
            increment_metric(
                metrics,
                "decision_router_dropped_candidates",
                decision.dropped_candidates as u64,
            );
            self.log(
                "decision_router_local_redaction",
                json!({
                    "turn": turn,
                    "redacted_state_fields": decision.redacted_state_fields,
                    "dropped_candidates": decision.dropped_candidates,
                }),
            )?;
        }
        if eligible {
            increment_metric(metrics, "decision_router_eligible", 1);
        } else {
            increment_metric(metrics, "decision_router_fallbacks", 1);
        }
        self.record_decision_activity(
            "next_action",
            if eligible {
                "JEV selected a bounded candidate"
            } else if uncertain_decision {
                "JEV is refining an uncertain decision"
            } else {
                "JEV deferred to the main model"
            },
            rejection_reason.unwrap_or("selected"),
            elapsed_ms,
        );
        self.log(
            "decision_router_result",
            json!({
                "candidate_count": candidates.len(),
                "turn": turn,
                "model": decision.model,
                "backend_metadata": decision.backend_metadata,
                "candidate_id": decision.candidate_id,
                "candidate_tool": selected.map(|candidate| &candidate.tool),
                "candidate_description": selected.map(|candidate| &candidate.description),
                "selected_probability": decision.selected_probability,
                "operation": decision.operation,
                "operation_probability": decision.operation_probability,
                "target_probability": decision.target_probability,
                "operation_confidence": decision.operation_confidence,
                "target_confidence": decision.target_confidence,
                "confidence": decision.confidence,
                "advisory_progress_probability": decision.progress_probability,
                "eligible": eligible,
                "rejection_reason": rejection_reason,
                "scoring_contract": if config.native_choice_probabilities() { "native_choice_probabilities" } else { "jev_calibrated_confidence" },
                "thresholds": {
                    "selected_probability": probability_threshold,
                    "operation_probability": operation_probability_threshold,
                    "target_probability": target_probability_threshold,
                    "terminal_probability": terminal_probability_threshold,
                    "terminal_confidence": config.min_terminal_confidence,
                    "operation_confidence": config.min_operation_confidence,
                    "target_confidence": config.min_confidence,
                },
                "elapsed_ms": elapsed_ms,
            }),
        )?;
        self.emit(AgentEvent::DecisionRouterEvaluated {
            turn,
            purpose: "next_action".into(),
            candidate_count: candidates.len(),
            candidate_id: decision.candidate_id.clone(),
            tool: selected.map(|candidate| candidate.tool.clone()),
            description: selected.map(|candidate| candidate.description.clone()),
            selected_probability: decision.selected_probability,
            confidence: decision.confidence,
            operation_probability: decision.operation_probability,
            target_probability: decision.target_probability,
            operation_confidence: decision.operation_confidence,
            target_confidence: decision.target_confidence,
            eligible,
            rejection_reason: rejection_reason.map(str::to_owned),
            probability_threshold,
            confidence_threshold: None,
            alternatives: {
                let mut values = decision
                    .probabilities
                    .iter()
                    .map(|(id, score)| (id.clone(), *score))
                    .collect::<Vec<_>>();
                values.sort_by(|left, right| right.1.total_cmp(&left.1));
                values.truncate(3);
                values
            },
            elapsed_ms,
        });
        if uncertain_decision {
            if let Some(candidate) = selected.filter(|candidate| !candidate.tool.starts_with("__"))
            {
                let fingerprint = decision_candidate_fingerprint(candidate);
                if record_uncertain_candidate(
                    &mut self.decision_router_refinement,
                    fingerprint.clone(),
                ) {
                    self.log(
                        "decision_router_refinement_candidate_withheld",
                        json!({
                            "turn": turn,
                            "fingerprint": fingerprint,
                            "repetitions": self.decision_router_refinement.repeated_candidate,
                            "state_revision": self.continuity.state_revision,
                        }),
                    )?;
                }
            }
            let score = decision_confidence(&decision);
            let improved = score
                >= self.decision_router_refinement.best_confidence
                    + config.min_refinement_confidence_gain;
            if improved {
                self.decision_router_refinement.best_confidence = score;
                self.decision_router_refinement.stagnant = 0;
            } else {
                self.decision_router_refinement.stagnant =
                    self.decision_router_refinement.stagnant.saturating_add(1);
            }
            self.decision_router_refinement.attempts =
                self.decision_router_refinement.attempts.saturating_add(1);
        }
        let refresh_candidate = (uncertain_decision
            && self.decision_router_ambiguity_refreshes == 0)
            .then(|| {
                if has_browser_state {
                    Some(DecisionCandidate {
                        id: "refresh_ambiguous_browser".into(),
                        tool: "managed_browser_snapshot".into(),
                        arguments: json!({}),
                        description: "Refresh structured browser state once after an ambiguous JEV decision"
                            .into(),
                        kind: DecisionCandidateKind::Action,
                        local_score: 1.0,
                    })
                } else {
                    observation
                        .as_ref()
                        .and_then(|observation| observation.foreground_window.as_ref())
                        .map(|window| DecisionCandidate {
                            id: "refresh_ambiguous_desktop".into(),
                            tool: "capture_screen".into(),
                            arguments: json!({
                                "scope": "window",
                                "window_id": window.id,
                                "include_ocr": true,
                                "include_ui_tree": true,
                                "enrichment": "fast"
                            }),
                            description: "Refresh structured desktop state once after an ambiguous JEV decision"
                                .into(),
                            kind: DecisionCandidateKind::Action,
                            local_score: 1.0,
                        })
                }
            })
            .flatten();
        let too_many_tool_failures = self
            .decision_router_tool_failures
            .values()
            .any(|failures| failures.len() >= 3);
        let refinement_available = uncertain_decision
            && !too_many_tool_failures
            && self.decision_router_refinement.attempts < config.max_refinement_attempts
            && self.decision_router_refinement.stagnant < config.max_stagnant_refinements;
        // A plain (non-refresh) refine only changes the input by narrowing to
        // config.max_candidates.min(8) candidates (see candidate_limit above). When
        // the current attempt already has that few or fewer, narrowing again is a
        // no-op: Laya is deterministic, so retrying produces the identical
        // rejection and just burns a wasted round trip before still deferring.
        let narrowing_would_help = candidates.len() > config.max_candidates.min(8);
        // Logged unconditionally (not only on promotion) so a live trace can
        // distinguish "the judge was never invoked" (e.g. disabled, or this
        // rejection reason isn't judge-eligible) from "the judge was invoked
        // and declined or errored" -- both looked identical (silence) before
        // this log existed, which made a real enable/wiring bug (the toggle
        // never reaching an already-open conversation's Session) impossible
        // to diagnose from trace.jsonl alone.
        let judge_promoted = if !eligible
            && !terminal_done
            && !terminal_blocked
            && uncertain_decision
            && config.native_choice_probabilities()
            && let Some(judge_candidate) = selected
        {
            let judge_key = {
                let mut hasher = Sha256::new();
                hasher.update(task.root_request.as_bytes());
                hasher.update(task.current_step.as_bytes());
                hasher.update(decision_candidate_fingerprint(judge_candidate).as_bytes());
                format!("{:x}", hasher.finalize())
            };
            let (outcome, reused_verdict) =
                if let Some(verdict) = self.decision_router_judge_verdicts.get(&judge_key) {
                    // Same candidate, same step, same state class: the judge
                    // already answered this exact question; re-asking returns the
                    // identical verdict and only burns a round trip.
                    (Ok(verdict.clone()), true)
                } else {
                    let outcome = router
                        .judge_candidate(
                            &task.root_request,
                            &task.current_step,
                            judge_candidate,
                            &candidates,
                            &state,
                        )
                        .await;
                    if let Ok(verdict) = &outcome {
                        self.decision_router_judge_verdicts
                            .insert(judge_key, verdict.clone());
                    }
                    (outcome, false)
                };
            let verdict = outcome.as_ref().ok();
            self.emit(AgentEvent::DecisionRouterJudgeAttempted {
                turn,
                tool: judge_candidate.tool.clone(),
                candidate_id: judge_candidate.id.clone(),
                reason: rejection_reason.map(str::to_owned),
                promoted: verdict.map(|v| v.promoted).unwrap_or(false),
                reused: reused_verdict,
                votes: verdict.map(|v| v.votes.clone()).unwrap_or_default(),
            });
            self.log(
                "decision_router_judge_attempt",
                json!({
                    "turn": turn,
                    "tool": judge_candidate.tool,
                    "candidate_id": judge_candidate.id,
                    "candidate_description": crate::decision::bounded_text(&judge_candidate.description, 300),
                    "task": crate::decision::bounded_text(&task.root_request, 1_000),
                    "current_step": crate::decision::bounded_text(&task.current_step, 500),
                    "reason": rejection_reason,
                    "promoted": verdict.map(|v| v.promoted).unwrap_or(false),
                    "reused": reused_verdict,
                    "votes": verdict.map(|v| v.votes.clone()).unwrap_or_default(),
                    "confidences": verdict.map(|v| v.confidences.clone()).unwrap_or_default(),
                    "probabilities": verdict.map(|v| v.probabilities.clone()).unwrap_or_default(),
                    "error": outcome.as_ref().err().map(ToString::to_string),
                }),
            )?;
            verdict
                .filter(|v| v.promoted)
                .map(|_| judge_candidate.clone())
        } else {
            None
        };
        let candidate = if eligible {
            selected
                .expect("eligible JEV decision has a selected candidate")
                .clone()
        } else if let Some(judge_candidate) = judge_promoted {
            self.record_decision_activity(
                "next_action",
                "A second local model judged the uncertain pick as safe",
                rejection_reason.unwrap_or("uncertain"),
                elapsed_ms,
            );
            self.log(
                "decision_router_judge_promoted",
                json!({
                    "turn": turn,
                    "tool": judge_candidate.tool,
                    "candidate_id": judge_candidate.id,
                    "reason": rejection_reason,
                }),
            )?;
            judge_candidate
        } else if refinement_available && let Some(candidate) = refresh_candidate {
            self.decision_router_ambiguity_refreshes = 1;
            self.log(
                "decision_router_ambiguity_refresh",
                json!({"turn": turn, "tool": candidate.tool, "reason": rejection_reason}),
            )?;
            candidate
        } else if refinement_available && narrowing_would_help {
            self.decision_router_cache_key = None;
            self.record_decision_activity(
                "next_action",
                "JEV refined an uncertain decision",
                rejection_reason.unwrap_or("uncertain"),
                elapsed_ms,
            );
            self.log(
                "decision_router_refinement",
                json!({
                    "turn": turn,
                    "attempt": self.decision_router_refinement.attempts,
                    "stagnant": self.decision_router_refinement.stagnant,
                    "confidence": decision_confidence(&decision),
                    "best_confidence": self.decision_router_refinement.best_confidence,
                    "candidate_count": candidates.len(),
                    "reason": rejection_reason,
                }),
            )?;
            return Ok(DecisionRouterStep::Refine);
        } else if uncertain_decision {
            if self.decision_router_refinement.visual_rescue_used {
                self.log(
                    "decision_router_bounded_assist",
                    json!({"turn": turn, "outcome": "skipped", "reason": "already_used_for_state"}),
                )?;
                return Ok(DecisionRouterStep::Handoff);
            }
            self.decision_router_refinement.visual_rescue_used = true;
            increment_metric(metrics, "decision_router_bounded_assists", 1);
            let assist = self
                .assist_decision_target(
                    &task.root_request,
                    &task.current_step,
                    &candidates,
                    &state,
                    observation.as_ref(),
                )
                .await;
            let assist = match assist {
                Ok(assist) => assist,
                Err(PokError::Cancelled) => return Err(PokError::Cancelled),
                Err(error) => {
                    if is_image_input_rejection(&error) {
                        self.image_input_override = Some(false);
                        prune_stale_images(&mut self.messages, 0);
                    }
                    self.log(
                        "decision_router_bounded_assist",
                        json!({"turn": turn, "outcome": "error", "error": error.to_string()}),
                    )?;
                    return Ok(DecisionRouterStep::Handoff);
                }
            };
            match assist {
                DecisionAssist::Candidate(id) => {
                    let Some(candidate) = candidates.iter().find(|candidate| candidate.id == id)
                    else {
                        return Ok(DecisionRouterStep::Handoff);
                    };
                    self.log(
                        "decision_router_bounded_assist",
                        json!({"turn": turn, "outcome": "candidate", "candidate_id": id}),
                    )?;
                    candidate.clone()
                }
                DecisionAssist::Refresh => {
                    let Some(candidate) = refresh_candidate else {
                        return Ok(DecisionRouterStep::Handoff);
                    };
                    self.log(
                        "decision_router_bounded_assist",
                        json!({"turn": turn, "outcome": "refresh"}),
                    )?;
                    candidate
                }
                DecisionAssist::Unsupported => {
                    self.log(
                        "decision_router_bounded_assist",
                        json!({"turn": turn, "outcome": "unsupported"}),
                    )?;
                    return Ok(DecisionRouterStep::Handoff);
                }
            }
        } else {
            return Ok(DecisionRouterStep::Handoff);
        };
        if candidate.tool == "__done__" || candidate.tool == "__blocked__" {
            let outcome = if candidate.tool == "__done__" {
                "terminal_done"
            } else {
                "terminal_blocked"
            };
            self.record_decision_activity(
                "next_action",
                if candidate.tool == "__done__" {
                    "JEV completed its structured action loop"
                } else {
                    "JEV handed an unsupported step to the primary model"
                },
                outcome,
                elapsed_ms,
            );
            self.log(
                "decision_router_terminal",
                json!({"turn": turn, "outcome": outcome, "probability": decision.selected_probability}),
            )?;
            return Ok(if candidate.tool == "__done__" {
                DecisionRouterStep::Done
            } else {
                DecisionRouterStep::Handoff
            });
        }
        if candidate.kind == DecisionCandidateKind::Evidence {
            let label = candidate
                .arguments
                .get("label")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let role = candidate
                .arguments
                .get("role")
                .and_then(Value::as_str)
                .unwrap_or("UI element");
            self.messages.push(BrainMessage::text_with_origin(
                "system",
                format!(
                    "<system-reminder>JEV selected this existing structured UI evidence as relevant to the active step: {role}: {label}. Treat it as focused evidence only; reason normally and do not claim completion without the usual evidence.</system-reminder>"
                ),
                MessageOrigin::RetrievedContext,
            ));
            increment_metric(metrics, "decision_router_evidence_focused", 1);
            self.emit(AgentEvent::DecisionRouterActionOutcome {
                turn,
                tool: "focus_evidence".into(),
                outcome: "evidence_focused".into(),
                ok: true,
            });
            self.log(
                "decision_router_evidence",
                json!({"turn": turn, "candidate_id": candidate.id, "label": label, "role": role}),
            )?;
            return Ok(DecisionRouterStep::Handoff);
        }
        let signature = action_signature(
            &candidate.tool,
            &candidate.arguments,
            observation.as_ref(),
            self.continuity.state_revision,
        );
        if self.decision_router_actions.contains(&signature) {
            let fingerprint = decision_candidate_fingerprint(&candidate);
            self.decision_router_suppressed_candidates
                .insert(fingerprint.clone());
            self.decision_router_cache_key = None;
            self.log(
                "decision_router_candidate_suppressed",
                json!({"turn": turn, "tool": candidate.tool, "fingerprint": fingerprint, "reason": "semantic_repeat"}),
            )?;
            return Ok(DecisionRouterStep::Refine);
        }
        let mut arguments = candidate.arguments.clone();
        if arguments
            .get("generate_value_with_primary_model")
            .and_then(Value::as_bool)
            == Some(true)
        {
            let value = match self
                .generate_decision_action_text(&task.root_request, &candidate.description)
                .await
            {
                Ok(value) => value,
                Err(PokError::Cancelled) => return Err(PokError::Cancelled),
                Err(error) => {
                    self.record_decision_activity(
                        "text_helper",
                        "JEV handed text generation to the primary model",
                        error.to_string(),
                        0,
                    );
                    self.log(
                        "decision_router_text_helper_handoff",
                        json!({"turn": turn, "reason": error.to_string()}),
                    )?;
                    return Ok(DecisionRouterStep::Handoff);
                }
            };
            if let Some(object) = arguments.as_object_mut() {
                object.remove("generate_value_with_primary_model");
                object.insert("value".into(), Value::String(value));
            }
        }
        if requested_commit_label(&task.root_request, &candidate.description).is_some() {
            let approved = match self
                .review_decision_commit(
                    &task.root_request,
                    &candidate.description,
                    observation.as_ref(),
                )
                .await
            {
                Ok(approved) => approved,
                Err(PokError::Cancelled) => return Err(PokError::Cancelled),
                Err(error) => {
                    self.log(
                        "decision_router_commit_review",
                        json!({"turn": turn, "approved": false, "error": error.to_string()}),
                    )?;
                    return Ok(DecisionRouterStep::Handoff);
                }
            };
            increment_metric(metrics, "decision_router_commit_reviews", 1);
            self.log(
                "decision_router_commit_review",
                json!({
                    "turn": turn,
                    "approved": approved,
                    "candidate_id": candidate.id,
                    "state_revision": self.continuity.state_revision,
                }),
            )?;
            if !approved {
                return Ok(DecisionRouterStep::Handoff);
            }
        }
        let call = CompletedToolCall {
            id: format!("decision-router-{}", Uuid::new_v4()),
            name: candidate.tool.clone(),
            arguments,
        };
        let pre_call =
            self.continuity
                .before_call(&call.name, &call.arguments, observation.as_ref());
        let (signature, prior_attempts) = match pre_call {
            PreCallDecision::Execute {
                signature,
                prior_attempts,
            } => (signature, prior_attempts),
            PreCallDecision::Suppress { reason, .. } => {
                let fingerprint = decision_candidate_fingerprint(&candidate);
                self.decision_router_suppressed_candidates
                    .insert(fingerprint.clone());
                self.decision_router_cache_key = None;
                self.log(
                    "decision_router_action_suppressed",
                    json!({"turn": turn, "tool": call.name, "reason": reason, "fingerprint": fingerprint}),
                )?;
                return Ok(DecisionRouterStep::Refine);
            }
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
        if let Ok(value) = &result {
            self.decision_router_last_result = Some((call.name.clone(), value.clone()));
        }
        // A successful launch is recorded for the run so the launch candidate
        // is not offered again: a repeated launch re-opened the application
        // and looped (live trace: notepad launched 28 times).
        if call.name == "open_application"
            && result.is_ok()
            && let Some(name) = call.arguments.get("name").and_then(Value::as_str)
        {
            self.decision_router_launched_applications
                .insert(name.trim().to_ascii_lowercase());
        }
        if call.name == "discover_tools"
            && result.is_ok()
            && let Some(groups) = call.arguments.get("groups").and_then(Value::as_array)
        {
            for group in groups.iter().filter_map(Value::as_str) {
                if matches!(
                    group,
                    "browser"
                        | "desktop"
                        | "system"
                        | "coding"
                        | "memory"
                        | "archive"
                        | "generated"
                        | "subagent"
                        | "mcp"
                        | "other"
                ) {
                    self.active_tool_groups.insert(group.into());
                }
            }
        }
        let feedback = self.continuity.after_call(
            &call.name,
            &call.arguments,
            &signature,
            prior_attempts,
            &result,
        );
        let ok = tool_finished_ok(&call.name, &result);
        let produced_fresh_evidence = result
            .as_ref()
            .is_ok_and(|value| qualifies_as_fresh_evidence(&call.name, value, ok));
        if let Ok(value) = &result
            && produced_fresh_evidence
        {
            self.decision_router_fresh_evidence = true;
            increment_metric(metrics, "fresh_evidence_observations", 1);
            self.log(
                "fresh_evidence_observed",
                json!({
                    "source": "decision_router",
                    "tool": call.name,
                    "captured_at": value.get("captured_at"),
                    "observed_url": value.get("observed_url"),
                    "target": value.get("target"),
                    "observation_id": value.get("observation_id"),
                }),
            )?;
        }
        increment_metric(metrics, "decision_router_executed", 1);
        increment_metric(
            metrics,
            decision_router_outcome_metric(&feedback.outcome, ok),
            1,
        );
        let candidate_fingerprint = decision_candidate_fingerprint(&candidate);
        if feedback.outcome == "failed" {
            self.decision_router_suppressed_candidates
                .insert(candidate_fingerprint.clone());
            self.decision_router_tool_failures
                .entry(call.name.clone())
                .or_default()
                .insert(candidate_fingerprint.clone());
            self.decision_router_cache_key = None;
        } else if feedback.outcome == "no_progress" {
            let count = self
                .decision_router_no_progress
                .entry(candidate_fingerprint.clone())
                .or_default();
            *count = count.saturating_add(1);
            if *count >= 3 {
                self.decision_router_suppressed_candidates
                    .insert(candidate_fingerprint.clone());
            }
        } else if feedback.outcome == "progress" {
            self.decision_router_refinement = DecisionRefinementState::default();
            self.decision_router_ambiguity_refreshes = 0;
            self.decision_router_tool_failures.clear();
            self.decision_router_completed_observations.clear();
        }
        if produced_fresh_evidence && feedback.outcome != "failed" {
            self.decision_router_completed_observations
                .insert(candidate_fingerprint.clone());
            self.decision_router_cache_key = None;
        }
        self.decision_router_actions.insert(signature);
        self.emit(AgentEvent::ToolFinished {
            call_id: call.id.clone(),
            name: call.name.clone(),
            ok,
            detail: tool_finished_detail(&call.name, &result),
            result: tool_finished_event_result(&call.name, &result),
        });
        self.emit(AgentEvent::DecisionRouterActionOutcome {
            turn,
            tool: call.name.clone(),
            outcome: feedback.outcome.clone(),
            ok,
        });
        let pipelined_window_id = if call.name == "activate_window" && ok {
            result
                .as_ref()
                .ok()
                .and_then(|val| val.get("id"))
                .and_then(Value::as_str)
                .map(str::to_owned)
        } else {
            None
        };
        self.messages
            .extend(tool_result_messages(&call, result, true));

        // Pipelined Action: When activate_window succeeds, automatically capture
        // fresh structured state for the activated window without an extra router roundtrip.
        if let Some(window_id) = pipelined_window_id {
            let capture_call = CompletedToolCall {
                id: format!("decision-router-pipeline-{}", Uuid::new_v4()),
                name: "capture_screen".into(),
                arguments: json!({
                    "scope": "window",
                    "window_id": window_id,
                    "include_ocr": true,
                    "include_ui_tree": true,
                    "enrichment": "fast"
                }),
            };
            self.emit(AgentEvent::ToolStarted {
                call_id: capture_call.id.clone(),
                name: capture_call.name.clone(),
                arguments: capture_call.arguments.clone(),
            });
            let capture_result = self
                .tools
                .call(
                    &capture_call.name,
                    capture_call.arguments.clone(),
                    &self.context,
                )
                .await;
            let capture_ok = tool_finished_ok(&capture_call.name, &capture_result);
            if let Ok(cap_val) = &capture_result {
                self.decision_router_last_result =
                    Some((capture_call.name.clone(), cap_val.clone()));
                if qualifies_as_fresh_evidence(&capture_call.name, cap_val, capture_ok) {
                    self.decision_router_fresh_evidence = true;
                    increment_metric(metrics, "fresh_evidence_observations", 1);
                }
            }
            self.emit(AgentEvent::ToolFinished {
                call_id: capture_call.id.clone(),
                name: capture_call.name.clone(),
                ok: capture_ok,
                detail: tool_finished_detail(&capture_call.name, &capture_result),
                result: tool_finished_event_result(&capture_call.name, &capture_result),
            });
            self.messages
                .extend(tool_result_messages(&capture_call, capture_result, true));
        }

        self.log(
            "decision_router_action",
            json!({
                "turn": turn,
                "tool": call.name,
                "arguments": call.arguments,
                "ok": ok,
                "outcome": feedback.outcome,
                "candidate_fingerprint": candidate_fingerprint,
            }),
        )?;
        Ok(DecisionRouterStep::Executed)
    }
}
