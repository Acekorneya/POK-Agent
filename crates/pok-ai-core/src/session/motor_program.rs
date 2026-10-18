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
        // The user's own browser: opening a page and clicking a control by
        // its label replay through System 1; typing and choosing there do
        // not yet.
        // A scroll that moved the view replays as a System 1 scroll choice;
        // the planner's wheel point is not repeated, only the direction.
        "scroll_view" => {
            if result.get("viewport_changed").and_then(Value::as_bool) != Some(true) {
                return Vec::new();
            }
            let text = |key: &str| action.get(key).and_then(Value::as_str);
            match (text("direction"), text("amount")) {
                (Some(direction), Some(amount)) => vec![json!({"scroll": {
                    "direction": direction,
                    "amount": amount,
                    "repeat": action.get("repeat").and_then(Value::as_u64).unwrap_or(1).min(5),
                }})],
                _ => Vec::new(),
            }
        }
        "user_browser_navigate" => arguments
            .get("url")
            .and_then(Value::as_str)
            .filter(|url| url.starts_with("http"))
            .map_or_else(
                || vec![json!({"stop": tool})],
                |url| vec![json!({"browser_open": url})],
            ),
        "user_browser_click" => action
            .get("label")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|label| !label.is_empty())
            .map_or_else(
                || vec![json!({"stop": tool})],
                |label| vec![json!({"browser_click": label})],
            ),
        _ => vec![json!({"stop": tool})],
    }
}

/// Where a value the request supplied sits in it: its fingerprint and the
/// words around it, which locate the same slot in a similar request that
/// asks for another value ("Add Paris to ..."). None when the request does
/// not contain the value.
fn request_slot(value: &str, request: &str) -> Option<Value> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    let request_lower = request.to_lowercase();
    let at = request_lower.find(&trimmed.to_lowercase())?;
    let words = |text: &str| {
        text.split(|character: char| !character.is_alphanumeric())
            .filter(|word| !word.is_empty())
            .map(str::to_owned)
            .collect::<Vec<_>>()
    };
    let before = words(&request_lower[..at]);
    let after = words(&request_lower[at + trimmed.to_lowercase().len()..]);
    Some(json!({
        "chars": trimmed.chars().count(),
        "sha256": text_fingerprint(trimmed),
        "before": before[before.len().saturating_sub(2)..],
        "after": &after[..after.len().min(2)],
    }))
}

fn text_fingerprint(text: &str) -> String {
    format!("{:x}", Sha256::digest(text.to_lowercase().as_bytes()))
}

/// A private value a verified run used (an email address it typed, a mailbox
/// page it opened). It stays out of the program unless the user agrees to
/// let it be remembered in the secure store (`crate::secrets`).
#[derive(Debug, Clone)]
pub(super) struct PrivateValue {
    pub(super) page: bool,
    pub(super) value: String,
}

impl PrivateValue {
    /// What the question to the user shows: what kind of value, never the
    /// value itself.
    pub(super) fn hint(&self) -> String {
        if self.page {
            let site = reqwest::Url::parse(&self.value)
                .ok()
                .and_then(|url| url.host_str().map(str::to_owned))
                .unwrap_or_else(|| "a site".into());
            return format!("a page on {site}");
        }
        let kind = if self.value.contains('@') {
            "an email address"
        } else if self.value.contains(":\\") || self.value.contains('/') {
            "a file path"
        } else {
            "a number or code"
        };
        format!("{kind} ({} characters)", self.value.chars().count())
    }
}

/// A verified run's program before the user is asked about its private
/// values: those steps hold a placeholder (`{"private": n}`).
pub(super) struct ProgramDraft {
    steps: Vec<Value>,
    pub(super) private: Vec<PrivateValue>,
}

/// The motor steps of a verified run, ready to become a program. Text the
/// user's request supplied is kept only as a fingerprint (found again in the
/// next request); a credential or very long text ends the program; personal
/// values (an email address, a number, a private page) wait for the user's
/// answer.
pub(super) fn program_draft(tape: &[Value], request: &str) -> ProgramDraft {
    let mut private = Vec::new();
    let mut hold = |value: &str, page: bool| {
        private.push(PrivateValue {
            page,
            value: value.to_owned(),
        });
        private.len() - 1
    };
    let steps = tape
        .iter()
        .map(|step| {
            // A page the request named is a slot, so a similar request opens
            // its own page; any other address is kept as recorded.
            if let Some(url) = step.get("browser_open").and_then(Value::as_str) {
                return match request_slot(url, request) {
                    Some(slot) => json!({"browser_open_request": slot}),
                    None if memory_text_may_be_sensitive(url) => json!({"stop": "private_text"}),
                    // An address with a query or fragment (a mailbox message,
                    // a search, an account page) can carry private values.
                    None if url.contains(['?', '#']) || looks_private(url) => {
                        json!({"private_page": hold(url, true)})
                    }
                    None => step.clone(),
                };
            }
            let Some(text) = step.get("text").and_then(Value::as_str) else {
                return step.clone();
            };
            if let Some(slot) = request_slot(text, request) {
                json!({"request_text": slot, "replace": step.get("replace")})
            } else if text.chars().count() > MOTOR_TEXT_LIMIT || memory_text_may_be_sensitive(text)
            {
                json!({"stop": "private_text"})
            } else if looks_private(text) {
                json!({"private": hold(text, false), "replace": step.get("replace")})
            } else {
                step.clone()
            }
        })
        .collect();
    ProgramDraft { steps, private }
}

/// The program, with each private value replaced by the key it was saved
/// under (`names[n]`), or ending at the first one that was not saved. `None`
/// when fewer than two steps could be repeated.
pub(super) fn seal_program(draft: &ProgramDraft, names: &[Option<String>]) -> Option<Value> {
    let name = |step: &Value, key: &str| {
        step.get(key)
            .and_then(Value::as_u64)
            .and_then(|index| names.get(usize::try_from(index).ok()?).cloned().flatten())
    };
    let steps = draft
        .steps
        .iter()
        .map(|step| {
            if step.get("private").is_some() {
                name(step, "private").map_or_else(
                    || json!({"stop": "private_text"}),
                    |key| json!({"secret_text": key, "replace": step.get("replace")}),
                )
            } else if step.get("private_page").is_some() {
                name(step, "private_page").map_or_else(
                    || json!({"stop": "private_text"}),
                    |key| json!({"secret_page": key}),
                )
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

/// Ask the user whether System 1 may keep a skill's private values in the
/// secure store, and save them when they agree. The key each value was saved
/// under, or None (no store, declined, or the save failed).
pub(super) async fn remember_private_values(
    approvals: &Arc<dyn crate::policy::ApprovalHandler>,
    skill: Uuid,
    request: &str,
    private: &[PrivateValue],
) -> Vec<Option<String>> {
    if private.is_empty() {
        return Vec::new();
    }
    let Some(store) = crate::secrets::store() else {
        return Vec::new();
    };
    let hints = private.iter().map(PrivateValue::hint).collect::<Vec<_>>();
    let reason = format!(
        "To repeat this task by itself, System 1 needs {} private value{} it used: {}. Allow saving {} in {}? {} never written to POK-Agent's files; if you decline, a replay stops before {} and the planner continues.",
        private.len(),
        if private.len() == 1 { "" } else { "s" },
        hints.join(", "),
        if private.len() == 1 { "it" } else { "them" },
        store.name(),
        if private.len() == 1 {
            "It is"
        } else {
            "They are"
        },
        if private.len() == 1 { "it" } else { "them" },
    );
    let decision = approvals
        .approve(
            "remember_private_values",
            &json!({"task": crate::decision::bounded_text(request, 200), "values": hints}),
            &reason,
        )
        .await;
    if !matches!(
        decision,
        Ok(crate::policy::ApprovalDecision::AllowOnce
            | crate::policy::ApprovalDecision::AllowSession)
    ) {
        return vec![None; private.len()];
    }
    private
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let key = format!("skill-{skill}-{index}");
            store.save(&key, &value.value).ok().map(|()| key)
        })
        .collect()
}

/// The program a verified run leaves when nothing private is remembered.
#[cfg(test)]
pub(super) fn program_from_tape(tape: &[Value], request: &str) -> Option<Value> {
    seal_program(&program_draft(tape, request), &[])
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
    // The wording a scroll node adds to its goal (its direction and size).
    let mut scroll_goals: BTreeMap<usize, String> = BTreeMap::new();
    let steps = program
        .get("steps")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    // A page opened first is opened by the replay itself (program_page).
    let skip = usize::from(steps.first().is_some_and(opens_page));
    for step in &steps[skip..] {
        if let Some(label) = step.get("browser_click").and_then(Value::as_str) {
            if !fits_request(&[label.to_owned()], &original_words, &request_words) {
                break;
            }
            nodes.push((
                Some(label.to_owned()),
                Some("user_browser_click"),
                Vec::new(),
            ));
            continue;
        }
        if opens_page(step) {
            break;
        }
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
        if let Some(scroll) = step.get("scroll") {
            let word = |key: &str| scroll.get(key).and_then(Value::as_str);
            let (Some(direction @ ("up" | "down" | "left" | "right")), Some(amount)) =
                (word("direction"), word("amount"))
            else {
                break;
            };
            // One node per page: each System 1 pick performs a single scroll.
            let repeat = scroll
                .get("repeat")
                .and_then(Value::as_u64)
                .unwrap_or(1)
                .clamp(1, 5);
            for _ in 0..repeat {
                scroll_goals.insert(nodes.len(), format!("scroll {direction} by 1 x {amount}"));
                nodes.push((None, Some("scroll"), Vec::new()));
            }
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
        } else if let Some(key) = step.get("secret_text").and_then(Value::as_str) {
            // A value the user let System 1 remember; gone from the store, the
            // replay stops here.
            let Some(text) = crate::secrets::load(key) else {
                break;
            };
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
            if scroll_goals.contains_key(&index) {
                // One recorded scroll is one scroll, not a search.
                node["max_steps"] = json!(1);
            }
            if let Some(note) = scroll_goals.get(&index) {
                node["goal"] = json!(format!(
                    "{note} (step {} of saved skill \"{skill_title}\")",
                    index + 1
                ));
            }
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

/// Whether a program starts by opening a page in the user's browser.
pub(super) fn program_opens_page(program: &Value) -> bool {
    program
        .get("steps")
        .and_then(Value::as_array)
        .and_then(|steps| steps.first())
        .is_some_and(opens_page)
}

fn opens_page(step: &Value) -> bool {
    step.get("browser_open").is_some()
        || step.get("browser_open_request").is_some()
        || step.get("secret_page").is_some()
}

/// The page a program for the user's browser opens first, filled from this
/// request when the recorded page came from the request. None for a desktop
/// program, or when this request does not name a page in that place.
pub(super) fn program_page(program: &Value, request: &str) -> Option<String> {
    let first = program.get("steps")?.as_array()?.first()?;
    if let Some(url) = first.get("browser_open").and_then(Value::as_str) {
        return Some(url.to_owned());
    }
    if let Some(key) = first.get("secret_page").and_then(Value::as_str) {
        return crate::secrets::load(key);
    }
    let found = resolve_request_text(first.get("browser_open_request")?, request)?;
    // The whole address as written in the request: a slot can stop at
    // punctuation inside it.
    let url = request
        .find(&found)
        .and_then(|at| request[at..].split_whitespace().next())
        .unwrap_or(&found);
    // Sentence punctuation after the address is not part of it; a closing
    // parenthesis is, when it closes one in the address (Wikipedia titles).
    let mut url = url.trim_end_matches(['.', ',', ';']).to_owned();
    while url.ends_with(')') && url.matches(')').count() > url.matches('(').count() {
        url.pop();
    }
    url.starts_with("http").then_some(url)
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
