//! Motor programs: a verified run's exact actions, recorded as they happen, stored on its skill, and replayed end to end by System 1.

use super::*;

use sha2::{Digest, Sha256};

/// At most this many recorded steps are kept for one run.
const MOTOR_TAPE_LIMIT: usize = 200;
/// Longest text typed by a program step (longer text needs the planner).
const MOTOR_TEXT_LIMIT: usize = 500;

/// Tools that only look, plan, or remember: they leave no motor step.
fn looking_tool(tool: &str) -> bool {
    matches!(
        tool,
        "observe_desktop"
            | "list_windows"
            | "capture_screen"
            | "inspect_screen_region"
            | "query_screen_text"
            | "query_window_tree"
            | "locate_visual_target"
            | "managed_browser_snapshot"
            | "read_clipboard"
            | "update_task_plan"
            | "activate_window"
            | "fast_actions"
    ) || !super::run_loop::is_acting_tool(tool)
}

/// The motor steps one successful tool call performed: `{"click": label}`,
/// `{"key": chord}`, `{"text": text, "replace": bool}`, or `{"stop": tool}`
/// for an action a program cannot repeat exactly (a vision click, a command,
/// a drag, a scroll).
pub(super) fn motor_steps(tool: &str, arguments: &Value, result: &Value) -> Vec<Value> {
    if looking_tool(tool) || result.get("executed").and_then(Value::as_bool) == Some(false) {
        return Vec::new();
    }
    let action = result.get("action").unwrap_or(&Value::Null);
    let text_step = |text: &str, replace: bool| json!({"text": text, "replace": replace});
    match tool {
        "click_target" => {
            let label = action
                .get("label")
                .or_else(|| action.get("expected_label"))
                .or_else(|| arguments.get("expected_label"))
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|label| !label.is_empty());
            match label {
                Some(label) => {
                    let mut step = json!({"click": label});
                    if arguments.get("button").and_then(Value::as_str) == Some("right") {
                        step["button"] = json!("right");
                    }
                    if arguments.get("double_click").and_then(Value::as_bool) == Some(true) {
                        step["double"] = json!(true);
                    }
                    vec![step]
                }
                None => vec![json!({"stop": tool})],
            }
        }
        "simulate_input" => match action.get("kind").and_then(Value::as_str) {
            Some("key") => action
                .get("key")
                .and_then(Value::as_str)
                .map(|key| vec![json!({"key": key})])
                .unwrap_or_default(),
            Some("type_text") => action
                .get("text")
                .and_then(Value::as_str)
                .map(|text| {
                    vec![text_step(
                        text,
                        action.get("replace_existing").and_then(Value::as_bool) == Some(true),
                    )]
                })
                .unwrap_or_default(),
            _ => vec![json!({"stop": tool})],
        },
        "type_text" => action
            .get("text")
            .and_then(Value::as_str)
            .map(|text| {
                vec![text_step(
                    text,
                    action.get("mode").and_then(Value::as_str) == Some("replace"),
                )]
            })
            .unwrap_or_default(),
        "execute_action_batch" => {
            if result.get("failed_at_step").is_some()
                || result.get("success").and_then(Value::as_bool) == Some(false)
            {
                return Vec::new();
            }
            arguments
                .get("steps")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .map(|step| {
                    let kind = step.get("kind").and_then(Value::as_str).unwrap_or_default();
                    match crate::builtins::normalized_batch_kind(kind) {
                        "click_target" => step
                            .get("expected_label")
                            .and_then(Value::as_str)
                            .filter(|label| !label.trim().is_empty())
                            .map_or_else(|| json!({"stop": tool}), |label| json!({"click": label})),
                        "type_text" => step.get("text").and_then(Value::as_str).map_or_else(
                            || json!({"stop": tool}),
                            |text| {
                                text_step(
                                    text,
                                    step.get("replace_existing").and_then(Value::as_bool)
                                        == Some(true),
                                )
                            },
                        ),
                        "key" => step
                            .get("key")
                            .or_else(|| step.get("text"))
                            .and_then(Value::as_str)
                            .map_or_else(|| json!({"stop": tool}), |key| json!({"key": key})),
                        _ => json!({"stop": tool}),
                    }
                })
                .collect()
        }
        _ => vec![json!({"stop": tool})],
    }
}

fn text_fingerprint(text: &str) -> String {
    format!("{:x}", Sha256::digest(text.to_lowercase().as_bytes()))
}

/// The program a verified run leaves: its motor steps, where text the user's
/// request supplied is kept only as a fingerprint (found again in the next
/// request) and private-looking text ends the program. `None` when fewer than
/// two steps could be repeated.
pub(super) fn program_from_tape(tape: &[Value], request: &str) -> Option<Value> {
    let request_lower = request.to_lowercase();
    let steps = tape
        .iter()
        .map(|step| {
            let Some(text) = step.get("text").and_then(Value::as_str) else {
                return step.clone();
            };
            let trimmed = text.trim();
            let found = (!trimmed.is_empty())
                .then(|| request_lower.find(&trimmed.to_lowercase()))
                .flatten();
            if let Some(at) = found {
                // The words around the value locate its slot in a similar
                // request that asks for another value ("Add Paris to ...").
                let words = |text: &str| {
                    text.split(|character: char| !character.is_alphanumeric())
                        .filter(|word| !word.is_empty())
                        .map(str::to_owned)
                        .collect::<Vec<_>>()
                };
                let before = words(&request_lower[..at]);
                let after = words(&request_lower[at + trimmed.to_lowercase().len()..]);
                json!({
                    "request_text": {
                        "chars": trimmed.chars().count(),
                        "sha256": text_fingerprint(trimmed),
                        "before": before[before.len().saturating_sub(2)..],
                        "after": &after[..after.len().min(2)],
                    },
                    "replace": step.get("replace"),
                })
            } else if text.chars().count() > MOTOR_TEXT_LIMIT
                || looks_private(text)
                || memory_text_may_be_sensitive(text)
            {
                json!({"stop": "private_text"})
            } else {
                step.clone()
            }
        })
        .collect::<Vec<_>>();
    let replayable = steps
        .iter()
        .take_while(|step| step.get("stop").is_none())
        .count();
    (replayable >= 2).then(|| json!({"version": 1, "steps": steps}))
}

/// Key chords that commit a document or message (save, save as, send,
/// print), which a replay leaves to the planner.
fn commit_shortcut(key: &str) -> bool {
    let key = key.to_ascii_lowercase().replace(' ', "");
    matches!(
        key.as_str(),
        "ctrl+s"
            | "control+s"
            | "ctrl+shift+s"
            | "control+shift+s"
            | "ctrl+enter"
            | "control+enter"
            | "alt+s"
            | "ctrl+p"
            | "control+p"
            | "f12"
    )
}

/// The request text a fingerprint names, in the new request's own casing:
/// the same value when this request repeats it, or else the value in the same
/// slot (between the same surrounding words) of a similar request.
fn resolve_request_text(fingerprint: &Value, request: &str) -> Option<String> {
    let chars = usize::try_from(fingerprint.get("chars")?.as_u64()?).ok()?;
    let sha = fingerprint.get("sha256")?.as_str()?;
    let characters = request.chars().collect::<Vec<_>>();
    let same = (0..characters.len().saturating_sub(chars.saturating_sub(1)))
        .map(|start| characters[start..start + chars].iter().collect::<String>())
        .find(|candidate| text_fingerprint(candidate) == sha);
    same.or_else(|| slot_value(fingerprint, request))
}

/// The text between a slot's surrounding words in this request.
fn slot_value(fingerprint: &Value, request: &str) -> Option<String> {
    let words = |key: &str| {
        fingerprint
            .get(key)
            .and_then(Value::as_array)
            .map(|words| {
                words
                    .iter()
                    .filter_map(Value::as_str)
                    .map(regex::escape)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    };
    let (before, after) = (words("before"), words("after"));
    if before.is_empty() && after.is_empty() {
        return None;
    }
    let lead = if before.is_empty() {
        r"^\W*".to_owned()
    } else {
        format!(r"\b{}\W+", before.join(r"\W+"))
    };
    let tail = if after.is_empty() {
        r"\W*(?:[.;!?]|$)".to_owned()
    } else {
        format!(r"\W+{}\b", after.join(r"\W+"))
    };
    let pattern = regex::Regex::new(&format!("(?is){lead}(.+?){tail}")).ok()?;
    let value = pattern.captures(request)?.get(1)?.as_str().trim();
    let value = value.trim_matches(|character: char| "\"'\u{201c}\u{201d}".contains(character));
    (!value.is_empty() && value.chars().count() <= 200).then(|| value.to_owned())
}

/// The `fast_actions` nodes that replay a program for this request: each
/// click is a node, and keys and text between clicks are one input node typed
/// into the focused control, as recorded. The replay ends at the first step it cannot repeat (a stop, a
/// request value this request does not contain, a label from the original
/// task this request does not ask for). Each node's condition is the next
/// node's label becoming visible.
pub(super) fn program_nodes(
    program: &Value,
    skill_title: &str,
    original_request: &str,
    request: &str,
) -> Vec<Value> {
    let original_words = signature_words(original_request);
    let request_words = signature_words(request);
    let mut nodes: Vec<(Option<String>, Option<&'static str>, Vec<Value>)> = Vec::new();
    for step in program
        .get("steps")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        if let Some(label) = step.get("click").and_then(Value::as_str) {
            if !fits_request(&[label.to_owned()], &original_words, &request_words) {
                break;
            }
            let operation = if step.get("button").and_then(Value::as_str) == Some("right") {
                Some("right_click")
            } else if step.get("double").and_then(Value::as_bool) == Some(true) {
                Some("double_click")
            } else {
                None
            };
            nodes.push((Some(label.to_owned()), operation, Vec::new()));
            continue;
        }
        let entry = if let Some(key) = step.get("key").and_then(Value::as_str) {
            // A shortcut that saves, sends, or prints is a commit: it stays
            // with the planner, like a click on a Save or Send button.
            if commit_shortcut(key) {
                break;
            }
            json!({"key": key})
        } else if let Some(text) = step.get("text").and_then(Value::as_str) {
            json!({"text": text, "replace_existing": step.get("replace").and_then(Value::as_bool).unwrap_or(false)})
        } else if let Some(fingerprint) = step.get("request_text") {
            let Some(text) = resolve_request_text(fingerprint, request) else {
                break;
            };
            json!({"text": text, "replace_existing": step.get("replace").and_then(Value::as_bool).unwrap_or(false)})
        } else {
            break;
        };
        // Keys and text go where the recorded run typed them: into the
        // control focused after the previous step (a click on "Got it" moves
        // focus to the editor; the button itself takes no text).
        match nodes.last_mut() {
            Some((None, None, input)) if input.len() < 40 => input.push(entry),
            _ => nodes.push((None, None, vec![entry])),
        }
        if nodes.len() > FAST_MAX_NODES {
            break;
        }
    }
    nodes.truncate(FAST_MAX_NODES);
    let labels = nodes
        .iter()
        .map(|(label, _, _)| label.clone())
        .collect::<Vec<_>>();
    nodes
        .into_iter()
        .enumerate()
        .map(|(index, (label, operation, input))| {
            let done_when = match labels.get(index + 1).cloned().flatten() {
                Some(next) => format!("\"{next}\" is visible"),
                None => "this recorded step took effect".to_owned(),
            };
            let mut node = json!({
                "goal": format!("replay step {} of saved skill \"{}\"", index + 1, skill_title),
                "done_when": done_when,
            });
            if let Some(label) = label {
                node["target_hint"] = json!(format!("\"{label}\""));
            }
            if !input.is_empty() {
                node["input"] = Value::Array(input);
            }
            if let Some(operation) = operation {
                node["allowed_operations"] = json!([operation]);
            }
            node
        })
        .collect()
}

impl Session {
    /// Add the motor steps of a successful tool call to this run's tape.
    pub(super) fn record_motor(&mut self, tool: &str, arguments: &Value, result: &Result<Value>) {
        let Ok(value) = result else {
            return;
        };
        if self.motor_tape.len() >= MOTOR_TAPE_LIMIT {
            return;
        }
        let steps = motor_steps(tool, arguments, value);
        let room = MOTOR_TAPE_LIMIT - self.motor_tape.len();
        self.motor_tape.extend(steps.into_iter().take(room));
    }
}

/// Save a verified run's program on the skill it taught or reinforced. The
/// leaner program is kept, unless the stored one was replayed this run and
/// stopped before its end. Returns whether the program was stored.
pub(super) fn attach_program(
    memory: &crate::memory::MemoryStore,
    skill: Uuid,
    program: Option<&Value>,
    stale: Option<Uuid>,
) -> bool {
    program.is_some_and(|program| {
        memory
            .set_procedure_program(skill, program, stale == Some(skill))
            .unwrap_or(false)
    })
}
