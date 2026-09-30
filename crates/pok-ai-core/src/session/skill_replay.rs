//! Direct skill execution: System 1 replays a trusted skill's opening steps before the planner's first call.

use super::*;

use crate::memory::ProcedureRecord;

/// Replay only a skill this relevant to the request (local score).
pub(super) const SKILL_REPLAY_MIN_RELEVANCE: f64 = 0.6;
/// Replay only a skill whose record earns this trust.
const SKILL_REPLAY_MIN_TRUST: f64 = 0.6;

/// How far a skill has earned replay: `(successes + 1) / (uses + 2)`, so a
/// new skill starts at one half and every verified run or failure moves it.
pub(super) fn skill_trust(skill: &ProcedureRecord) -> f64 {
    let successes = skill.success_count as f64;
    let failures = skill.failure_count as f64;
    (successes + 1.0) / (successes + failures + 2.0)
}

/// Quoted texts in their original case (straight or curly quotes).
fn quoted_texts(text: &str) -> Vec<String> {
    let mut texts = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find(['"', '\u{201c}']) {
        let open = rest[start..].chars().next().map_or(1, char::len_utf8);
        let after = &rest[start + open..];
        let Some(end) = after.find(['"', '\u{201d}']) else {
            break;
        };
        let quoted = after[..end].trim();
        if !quoted.is_empty() {
            texts.push(quoted.to_owned());
        }
        let close = after[end..].chars().next().map_or(1, char::len_utf8);
        rest = &after[end + close..];
    }
    texts
}

/// Words of a text normalized like a task signature (no length cap).
pub(super) fn signature_words(text: &str) -> BTreeSet<String> {
    text.split_whitespace()
        .map(task_signature)
        .filter(|word| !word.is_empty())
        .collect()
}

/// Whether a step may be replayed for this request: its labels are at least
/// two characters, and none is a value from the skill's original task (a
/// time zone, a file name) that this task does not ask for. Such a label has
/// a word from the original request that this one lacks and shares no word
/// with this one; "Change time zone" shares "time zone" with any time-zone
/// request, "Pacific (US & Canada)" only with one that asks for Pacific.
pub(super) fn fits_request(
    labels: &[String],
    original_request: &BTreeSet<String>,
    request: &BTreeSet<String>,
) -> bool {
    labels.iter().all(|label| {
        let words = signature_words(label);
        let old_value = words
            .iter()
            .any(|word| original_request.contains(word) && !request.contains(word));
        let in_request = words.iter().any(|word| request.contains(word));
        label.chars().count() >= 2 && (in_request || !old_value)
    })
}

/// The application a skill opens first, by the name its "Open ..." step
/// quotes ("Settings", "File Explorer"), when that step comes before any
/// action.
fn opening_application(steps: &[ProcedureStep]) -> Option<String> {
    for step in steps {
        match step.tool.as_str() {
            "open_application" => return quoted_texts(&step.instruction).into_iter().next(),
            "observe_desktop" | "list_windows" | "capture_screen" | "activate_window" => {}
            _ => return None,
        }
    }
    None
}

/// Whether a request asks for the same kind of task a skill's program was
/// recorded for: at least 40% of words shared (a reworded request, or the
/// same task with another value: "Add Paris" for "Add Kyoto"). A differing value is still caught where it matters:
/// request text the program types must be found in this request, and a label
/// from the original task stops the replay unless this request asks for it.
pub(super) fn same_task(original_request: &str, request: &str) -> bool {
    let original = original_request
        .split_whitespace()
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
    let current = task_signature(request)
        .split_whitespace()
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
    if original.is_empty() || current.is_empty() {
        return false;
    }
    let shared = original.intersection(&current).count() as f64;
    let all = original.union(&current).count() as f64;
    shared / all >= 0.4
}

/// Whether a window belongs to the skill's application: by process
/// (`Code.exe`), or by name for applications known by their window title
/// (Store apps such as Settings or Clock, whose process is a shared frame
/// host). A skill that names no application matches a short window title
/// that its own title or summary names ("... in the Clock app").
pub(super) fn skill_application_window(
    window: &WindowInfo,
    names: &[String],
    skill_text: &str,
) -> bool {
    let process = window.process_name.to_ascii_lowercase();
    let process = process.trim_end_matches(".exe");
    let title = window.title.trim().to_ascii_lowercase();
    let named = names.iter().any(|name| {
        let name = name.trim().to_ascii_lowercase();
        let name = name.trim_end_matches(".exe");
        name.len() >= 2
            && (name == process || title == name || title.ends_with(&format!(" - {name}")))
    });
    let title_words = signature_words(&title);
    let described = names.is_empty()
        && (3..=30).contains(&title.chars().count())
        && (1..=3).contains(&title_words.len())
        && title_words.is_subset(&signature_words(skill_text));
    // The desktop and taskbar belong to explorer.exe too, but are not File
    // Explorer windows.
    let shell = title.is_empty() || title == "program manager";
    window.visible && !window.minimized && !window.elevated && !shell && (named || described)
}

impl Session {
    /// Replays the opening steps of a trusted, relevant skill through
    /// `fast_actions` before the planner is asked anything, then tells the
    /// planner what already happened. Returns the action steps used (0 when
    /// nothing was replayed).
    pub(super) async fn replay_skill(
        &mut self,
        skill: &ProcedureRecord,
        relevance: f64,
        request: &str,
        metrics: &mut RunMetrics,
        workflow: &mut Vec<Value>,
    ) -> Result<u32> {
        if !self.delegated_decision_router() || relevance < SKILL_REPLAY_MIN_RELEVANCE {
            return Ok(0);
        }
        let trust = skill_trust(skill);
        // Only a whole recorded program is replayed: the same or a similar
        // task again. (Replaying just a skill's opening steps, without a
        // program, cost the planner more calls than it saved in v5.)
        let plan: Vec<Value> = skill
            .program
            .as_ref()
            .filter(|_| same_task(&skill.task_signature, request))
            .map(|program| program_nodes(program, &skill.title, &skill.task_signature, request))
            .unwrap_or_default();
        let from_program = !plan.is_empty();
        let describe = |node: &Value| {
            let mut parts = Vec::new();
            if let Some(hint) = node.get("target_hint").and_then(Value::as_str) {
                parts.push(format!("target_hint {hint}"));
            }
            if let Some(input) = node.get("input").and_then(Value::as_array) {
                parts.push(format!("input of {} keys and texts", input.len()));
            }
            parts.push(format!(
                "done_when {}",
                node.get("done_when")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
            ));
            parts.join("; ")
        };
        let skipped = if skill.success_count == 0 {
            Some("unverified")
        } else if trust < SKILL_REPLAY_MIN_TRUST {
            Some("low_trust")
        } else if plan.is_empty() {
            Some(if skill.program.is_some() {
                "program_does_not_fit"
            } else {
                "no_program"
            })
        } else {
            None
        };
        let base = json!({
            "id": skill.id,
            "title": skill.title,
            "relevance": relevance,
            "trust": trust,
            "success_count": skill.success_count,
            "failure_count": skill.failure_count,
        });
        if let Some(reason) = skipped {
            let mut record = base;
            record["reason"] = json!(reason);
            self.log("skill_replay_skipped", record)?;
            return Ok(0);
        }
        // Act on the skill's application, preferring the foreground window;
        // opening it when it is not running is part of the skill.
        let opening = opening_application(&skill.steps);
        let mut names = skill.applications.clone();
        names.extend(opening.clone());
        let skill_text = format!("{} {}", skill.title, skill.summary);
        let mut opened = None;
        let mut window = None;
        for attempt in 0..2 {
            let windows = self
                .context
                .platform
                .list_windows()
                .await
                .unwrap_or_default();
            let foreground = self
                .context
                .platform
                .foreground_window()
                .await
                .ok()
                .flatten();
            // The application's main window: the foreground one when it
            // matches and is not a small dialog beside a larger window of the
            // same application (a crash report or an options box).
            let area = |window: &WindowInfo| {
                i64::from(window.bounds.width) * i64::from(window.bounds.height)
            };
            let largest = windows
                .into_iter()
                .filter(|window| skill_application_window(window, &names, &skill_text))
                .max_by_key(area);
            window = match (
                foreground.filter(|window| skill_application_window(window, &names, &skill_text)),
                largest,
            ) {
                (Some(front), Some(large)) if area(&front) * 2 < area(&large) => Some(large),
                (Some(front), _) => Some(front),
                (None, large) => large,
            };
            let Some(name) = opening
                .as_ref()
                .filter(|_| window.is_none() && attempt == 0)
            else {
                break;
            };
            match self
                .tools
                .call("open_application", json!({"name": name}), &self.context)
                .await
            {
                Ok(_) => opened = Some(name.clone()),
                Err(PokError::Cancelled) => return Err(PokError::Cancelled),
                Err(error) => {
                    let mut record = base;
                    record["reason"] = json!(format!("open_application failed: {error}"));
                    self.log("skill_replay_skipped", record)?;
                    return Ok(0);
                }
            }
        }
        let Some(window) = window else {
            let mut record = base;
            record["reason"] = json!(if opened.is_some() {
                "application_did_not_open"
            } else {
                "application_not_open"
            });
            self.log("skill_replay_skipped", record)?;
            return Ok(0);
        };
        let capture = json!({"scope": "window", "window_id": window.id});
        let mut captured = match self
            .tools
            .call(
                "activate_window",
                json!({"window_id": window.id}),
                &self.context,
            )
            .await
        {
            Ok(_) => {
                self.tools
                    .call("capture_screen", capture.clone(), &self.context)
                    .await
            }
            Err(error) => Err(error),
        };
        // Choosing the window may leave another one in front (a capture then
        // reports it is not the foreground): bring it forward, as a person
        // would click it, and look again.
        if captured.is_err() && !matches!(captured, Err(PokError::Cancelled)) {
            let _ = self.context.platform.bring_to_front(&window.id).await;
            tokio::select! {
                () = self.context.cancellation.cancelled() => return Err(PokError::Cancelled),
                () = tokio::time::sleep(Duration::from_millis(400)) => {}
            }
            captured = self
                .tools
                .call("capture_screen", capture, &self.context)
                .await;
        }
        if let Ok(value) = &captured
            && qualifies_as_fresh_evidence("capture_screen", value, true)
        {
            self.decision_router_fresh_evidence = true;
        }
        if let Err(error) = captured {
            if matches!(error, PokError::Cancelled) {
                return Err(error);
            }
            let mut record = base;
            record["reason"] = json!(format!("capture failed: {error}"));
            self.log("skill_replay_skipped", record)?;
            return Ok(0);
        }
        // A window System 1 just opened may still be drawing its content:
        // look again (up to 4 s) until the first step's label is there.
        let first_labels = quoted_texts(
            plan[0]
                .get("target_hint")
                .and_then(Value::as_str)
                .unwrap_or_default(),
        )
        .iter()
        .map(|label| crate::decision::normalized_evidence(label))
        .collect::<Vec<_>>();
        let mut waits = 0;
        while opened.is_some() && !first_labels.is_empty() && waits < 4 {
            let visible =
                self.context
                    .latest_observation
                    .lock()
                    .as_ref()
                    .is_some_and(|observation| {
                        first_labels.iter().all(|label| {
                            observation.targets.iter().any(|target| {
                                crate::decision::normalized_evidence(&target.name)
                                    .contains(label.as_str())
                            })
                        })
                    });
            if visible {
                break;
            }
            waits += 1;
            tokio::select! {
                () = self.context.cancellation.cancelled() => return Err(PokError::Cancelled),
                () = tokio::time::sleep(Duration::from_secs(1)) => {}
            }
            if let Err(error) = self
                .tools
                .call(
                    "capture_screen",
                    json!({"scope": "window", "window_id": window.id}),
                    &self.context,
                )
                .await
                && matches!(error, PokError::Cancelled)
            {
                return Err(error);
            }
        }
        let mut arguments = plan[0].clone();
        if arguments.get("allowed_operations").is_none() {
            arguments["allowed_operations"] = json!(["click"]);
        }
        if plan.len() > 1 {
            arguments["then"] = Value::Array(plan[1..].to_vec());
        }
        increment_metric(metrics, "skill_replays", 1);
        self.skill_replay_active = true;
        let result = self.run_fast_actions(0, &arguments, metrics).await;
        self.skill_replay_active = false;
        let result = match result {
            Ok(result) => result,
            Err(PokError::Cancelled) => return Err(PokError::Cancelled),
            Err(error) => json!({"status": "failed", "reason": error.to_string()}),
        };
        let status = result["status"].as_str().unwrap_or("stalled").to_owned();
        let nodes = result
            .get("subgoals")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_else(|| vec![result.clone()]);
        let reached = |node: &Value| {
            matches!(
                node.get("status").and_then(Value::as_str),
                Some("done" | "unverified")
            )
        };
        let completed = nodes.iter().filter(|node| reached(node)).count();
        increment_metric(metrics, "skill_replay_steps_completed", completed as u64);
        // A replay that fit nothing changed nothing; the run's own outcome
        // (verified or not) is what moves the skill's trust.
        if completed > 0
            && let Some(step) = workflow_step("fast_actions", &arguments, &result)
        {
            push_workflow_step(workflow, step);
        }
        // A program that stopped before its end no longer fits the task: the
        // path this run finishes with replaces it.
        if from_program && completed < plan.len() {
            self.stale_program = Some(skill.id);
        }
        let lines = plan
            .iter()
            .enumerate()
            .map(|(index, node)| {
                let outcome = if index < completed {
                    "done"
                } else if index == completed && status != "done" {
                    status.as_str()
                } else {
                    "not run"
                };
                format!("{}. {} -> {outcome}", index + 1, describe(node))
            })
            .collect::<Vec<_>>()
            .join("\n");
        let stop_reason = nodes
            .iter()
            .find(|node| !reached(node))
            .and_then(|node| node.get("reason").and_then(Value::as_str))
            .map(|reason| format!("\nIt stopped because: {}", truncate_chars(reason, 300)))
            .unwrap_or_default();
        let lines = match &opened {
            Some(name) => format!("0. open_application \"{name}\" -> done\n{lines}"),
            None => lines,
        };
        let next = if completed == plan.len() {
            "These steps are done: do not repeat them. The screen after the replay follows; finish the rest of the task from it."
        } else if completed == 0 {
            "Nothing was changed by the replay; the screen may not match the skill. The current screen follows; plan from it."
        } else {
            "Do not repeat the finished steps. The screen after the replay follows; continue from where it stopped."
        };
        self.messages.push(BrainMessage::text_with_origin(
            "system",
            format!(
                "<system-reminder>System 1 replayed the opening steps of saved skill \"{}\" in window \"{}\" ({}) before this turn:\n{lines}{stop_reason}\n{next}</system-reminder>",
                truncate_chars(&skill.title, 120),
                truncate_chars(&window.title, 120),
                window.process_name,
            ),
            MessageOrigin::SystemReminder,
        ));
        let mut record = base;
        record["window"] = json!({"title": window.title, "application": window.process_name});
        record["opened"] = json!(opened);
        record["load_waits"] = json!(waits);
        record["steps"] = json!(plan.len());
        record["mode"] = json!(if from_program {
            "program"
        } else {
            "opening_steps"
        });
        record["completed"] = json!(completed);
        record["status"] = json!(status);
        self.log("skill_replay", record)?;
        Ok(1)
    }
}
