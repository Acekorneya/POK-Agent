//! Delegated fast actions: the planner's bounded plan trees (steps, branches, interrupts, reads) executed with quoted-label grounding and the fast decision model.

use super::*;

/// One normalized plan node, whatever level of the `fast_actions` tree it
/// came from.
#[derive(Debug, Clone)]
pub(super) struct FastPlanNode {
    pub(super) goal: String,
    pub(super) hint: String,
    pub(super) done_when: String,
    pub(super) allowed: Vec<FastOperation>,
    pub(super) avoid: Vec<String>,
    pub(super) branches: Vec<(String, Vec<FastPlanNode>)>,
    /// Keys and text the planner wrote for System 1 to enter.
    pub(super) input: Vec<crate::builtins::FastInputStep>,
}

/// A validated `fast_actions` plan: the root chain (the call's own subgoal
/// followed by `then`) and the branches chosen after it completes.
pub(super) struct FastPlan {
    pub(super) chain: Vec<FastPlanNode>,
    pub(super) branches: Vec<(String, Vec<FastPlanNode>)>,
    pub(super) total_nodes: usize,
}

/// State shared by every node of one `fast_actions` call.
pub(super) struct FastRunContext {
    pub(super) interrupts: Vec<crate::builtins::FastInterrupt>,
    pub(super) interrupts_used: u8,
}

/// A condition choice (branch or interrupt rule) that grounding could not
/// settle, waiting for the router.
pub(super) struct PendingFastCondition {
    pub(super) options: Vec<(String, String)>,
    pub(super) otherwise: Option<usize>,
    pub(super) count: usize,
    pub(super) state: Value,
    pub(super) purpose: String,
    pub(super) labels: Vec<String>,
}

pub(super) enum FastConditionOutcome {
    Picked(Option<usize>),
    NeedsRouter(PendingFastCondition),
}

pub(super) const FAST_MAX_NODES: usize = 64;
/// Most options a delegated run offers the decision model in one question.
pub(super) const FAST_MAX_OPTIONS: usize = 16;
pub(super) const FAST_MAX_BRANCHES: usize = 4;
pub(super) const FAST_MAX_INTERRUPTS: usize = 4;
pub(super) const FAST_MAX_INTERRUPT_ACTIONS: u8 = 3;
/// Target tier for read-only scrolling. Scrolling changes no application
/// state, so it uses the low-stakes tier from the TypeSafe confidence
/// guidance (act from 0.5) instead of the click gate.
pub(super) const FAST_SCROLL_MIN_PROBABILITY: f64 = 0.5;
/// An interrupt rule the fast model judged true ends or redirects the whole
/// run, so it needs stronger evidence than a completion check.
pub(super) const FAST_INTERRUPT_MIN_PROBABILITY: f64 = 0.8;

/// The dialogs on screen when a step started: a dialog already there then is
/// part of the task, not an interruption.
#[derive(Debug, Clone, Default)]
pub(super) struct PopupBaseline {
    window_id: Option<String>,
    window_area: u64,
    process: Option<String>,
    dialogs: HashSet<String>,
}

impl PopupBaseline {
    pub(super) fn of(observation: Option<&crate::types::Observation>) -> Self {
        let window = observation.and_then(|observation| observation.foreground_window.as_ref());
        Self {
            window_id: window.map(|window| window.id.clone()),
            window_area: window.map_or(0, |window| rect_area(&window.bounds)),
            process: window.map(|window| window.process_name.to_lowercase()),
            dialogs: observation
                .map(|observation| {
                    dialog_elements(observation)
                        .map(|element| element.name.trim().to_lowercase())
                        .collect()
                })
                .unwrap_or_default(),
        }
    }
}

fn rect_area(rect: &crate::types::Rect) -> u64 {
    u64::from(rect.width) * u64::from(rect.height)
}

fn rect_inside(inner: &crate::types::Rect, outer: &crate::types::Rect) -> bool {
    inner.x >= outer.x
        && inner.y >= outer.y
        && i64::from(inner.x) + i64::from(inner.width)
            <= i64::from(outer.x) + i64::from(outer.width)
        && i64::from(inner.y) + i64::from(inner.height)
            <= i64::from(outer.y) + i64::from(outer.height)
}

/// Named dialog windows inside the captured window (not the window itself).
fn dialog_elements(
    observation: &crate::types::Observation,
) -> impl Iterator<Item = &crate::types::UiElement> {
    let root_area = observation
        .foreground_window
        .as_ref()
        .map_or(u64::MAX, |window| rect_area(&window.bounds));
    observation.ui_elements.iter().filter(move |element| {
        let kind = element.control_type.to_ascii_lowercase();
        (kind == "dialog" || kind == "window")
            && !element.offscreen
            && !element.name.trim().is_empty()
            && rect_area(&element.bounds) < root_area
    })
}

/// A dialog System 1 noticed: what a person reads at a glance.
#[derive(Debug, Clone, Default, PartialEq)]
pub(super) struct Popup {
    pub(super) title: String,
    pub(super) text: String,
    pub(super) buttons: Vec<String>,
    /// The dialog's own window, when it is one, for a closer look.
    pub(super) window_id: Option<String>,
}

impl Popup {
    pub(super) fn value(&self) -> Value {
        json!({"title": self.title, "text": self.text, "buttons": self.buttons})
    }

    /// Fill in the message and buttons from the dialog's own elements.
    fn read(&mut self, elements: &[crate::types::UiElement], region: Option<&crate::types::Rect>) {
        let mut text = Vec::<String>::new();
        let mut buttons = Vec::<String>::new();
        for element in elements {
            let name = element.name.trim();
            if element.offscreen
                || name.is_empty()
                || region.is_some_and(|region| !rect_inside(&element.bounds, region))
            {
                continue;
            }
            let kind = element.control_type.to_ascii_lowercase();
            if kind == "text" && name != self.title && !text.iter().any(|seen| seen == name) {
                text.push(name.to_owned());
            } else if kind == "button"
                && buttons.len() < 8
                && !buttons.iter().any(|seen| seen == name)
            {
                buttons.push(name.to_owned());
            }
        }
        // The title bar's Close button says nothing about the dialog.
        if buttons.len() > 1 {
            buttons.retain(|button| !button.eq_ignore_ascii_case("close"));
        }
        if !text.is_empty() {
            self.text = crate::decision::bounded_text(&text.join(" "), 600);
        }
        if buttons.len() > self.buttons.len()
            || self.buttons.iter().all(|b| b.eq_ignore_ascii_case("close")) && !buttons.is_empty()
        {
            self.buttons = buttons;
        }
    }

    /// The capture that noticed the dialog may not have walked into it (a
    /// dialog is often its own window), leaving only its title.
    pub(super) fn needs_closer_look(&self) -> bool {
        self.text.is_empty()
            || self
                .buttons
                .iter()
                .all(|button| button.eq_ignore_ascii_case("close"))
    }
}

/// A dialog that opened during this step and that the plan did not mention,
/// read from UI Automation the way a person glances at a popup: its title,
/// message, and buttons. The planner then acts on what it says instead of
/// the step stalling behind it. A dialog the plan names (in its goal, target,
/// or completion condition) is expected and not reported.
pub(super) fn unexpected_popup(
    observation: &crate::types::Observation,
    baseline: &PopupBaseline,
    plan_text: &str,
) -> Option<Popup> {
    let window = observation.foreground_window.as_ref();
    let (mut popup, region) = if let Some(dialog) = dialog_elements(observation).find(|element| {
        !baseline
            .dialogs
            .contains(&element.name.trim().to_lowercase())
    }) {
        (
            Popup {
                title: dialog.name.trim().to_owned(),
                ..Popup::default()
            },
            dialog.bounds.clone(),
        )
    } else {
        // A smaller window of the same application took the foreground: a
        // dialog shown as its own window. Menus and pickers have no title.
        let window = window.filter(|window| {
            baseline
                .window_id
                .as_deref()
                .is_some_and(|id| id != window.id)
                && baseline.process.as_deref() == Some(window.process_name.to_lowercase().as_str())
                && rect_area(&window.bounds) < baseline.window_area
                && !window.title.trim().is_empty()
        })?;
        (
            Popup {
                title: window.title.trim().to_owned(),
                window_id: Some(window.id.clone()),
                ..Popup::default()
            },
            window.bounds.clone(),
        )
    };
    let plan = plan_text.to_lowercase();
    if plan.contains(&popup.title.to_lowercase()) {
        return None;
    }
    popup.read(&observation.ui_elements, Some(&region));
    let seen = format!("{} {}", popup.title, popup.text).to_lowercase();
    if crate::decision::quoted_labels(plan_text)
        .iter()
        .any(|label| seen.contains(&label.to_lowercase()))
    {
        return None;
    }
    Some(popup)
}

/// Delegated operations that only move the view.
pub(super) fn is_read_only_fast_tool(tool: &str) -> bool {
    matches!(tool, "scroll_view" | "managed_browser_scroll")
}

/// The default step when a named target is not on screen: continue down the
/// page by one page.
pub(super) fn default_scroll_down(candidates: &[DecisionCandidate]) -> Option<&DecisionCandidate> {
    candidates.iter().find(|candidate| {
        is_read_only_fast_tool(&candidate.tool)
            && candidate.arguments.get("direction").and_then(Value::as_str) == Some("down")
            && (candidate.arguments.get("amount").and_then(Value::as_str) == Some("page")
                || candidate.arguments.get("page").and_then(Value::as_bool) == Some(true))
    })
}

impl FastPlan {
    pub(super) fn from_args(args: &FastActionsArgs) -> Result<Self> {
        fn node(
            goal: &str,
            hint: Option<&str>,
            done_when: &str,
            allowed: &[FastOperation],
            avoid: &[String],
            branches: Vec<(String, Vec<FastPlanNode>)>,
            input: &[crate::builtins::FastInputStep],
        ) -> Result<FastPlanNode> {
            if input.len() > 40
                || input.iter().any(|step| {
                    step.key.is_some() == step.text.is_some()
                        || step
                            .key
                            .as_deref()
                            .is_some_and(|key| key.trim().is_empty() || key.chars().count() > 60)
                        || step
                            .text
                            .as_deref()
                            .is_some_and(|text| text.chars().count() > 4000)
                })
            {
                return Err(PokError::Tool(
                    "a fast_actions step's input holds at most 40 entries, each with exactly one of key (a named key or shortcut) or text (at most 4000 characters)".into(),
                ));
            }
            let goal = goal.trim();
            let done_when = done_when.trim();
            let hint = hint.map(str::trim).unwrap_or_default();
            if goal.is_empty()
                || done_when.is_empty()
                || goal.chars().count() > 500
                || done_when.chars().count() > 300
                || hint.chars().count() > 300
                || avoid.len() > 8
            {
                return Err(PokError::Tool(
                    "each fast_actions step requires goal (1-500 chars) and done_when (1-300 chars); target_hint is at most 300 chars and avoid at most 8 phrases".into(),
                ));
            }
            Ok(FastPlanNode {
                goal: goal.into(),
                hint: hint.into(),
                done_when: done_when.into(),
                allowed: allowed.to_vec(),
                avoid: avoid.to_vec(),
                branches,
                input: input.to_vec(),
            })
        }
        fn checked_when(when: &str) -> Result<String> {
            let when = when.trim();
            if when.is_empty() || when.chars().count() > 300 {
                return Err(PokError::Tool(
                    "each fast_actions branch or interrupt needs a when condition of 1-300 chars"
                        .into(),
                ));
            }
            Ok(when.into())
        }
        fn subgoal(step: &FastSubgoal) -> Result<FastPlanNode> {
            if step.branches.len() > FAST_MAX_BRANCHES {
                return Err(PokError::Tool(format!(
                    "a fast_actions step accepts at most {FAST_MAX_BRANCHES} branches"
                )));
            }
            let branches = step
                .branches
                .iter()
                .map(|branch| {
                    let leaves = branch
                        .then
                        .iter()
                        .map(|leaf| {
                            node(
                                &leaf.goal,
                                leaf.target_hint.as_deref(),
                                &leaf.done_when,
                                &leaf.allowed_operations,
                                &leaf.avoid,
                                Vec::new(),
                                &leaf.input,
                            )
                        })
                        .collect::<Result<Vec<_>>>()?;
                    Ok((checked_when(&branch.when)?, leaves))
                })
                .collect::<Result<Vec<_>>>()?;
            node(
                &step.goal,
                step.target_hint.as_deref(),
                &step.done_when,
                &step.allowed_operations,
                &step.avoid,
                branches,
                &step.input,
            )
        }
        fn count(nodes: &[FastPlanNode]) -> usize {
            nodes
                .iter()
                .map(|node| {
                    1 + node
                        .branches
                        .iter()
                        .map(|(_, children)| count(children))
                        .sum::<usize>()
                })
                .sum()
        }
        if args.on_interrupt.len() > FAST_MAX_INTERRUPTS {
            return Err(PokError::Tool(format!(
                "fast_actions accepts at most {FAST_MAX_INTERRUPTS} on_interrupt rules"
            )));
        }
        for rule in &args.on_interrupt {
            checked_when(&rule.when)?;
        }
        if args.read.len() > 8 {
            return Err(PokError::Tool(
                "fast_actions accepts at most 8 read requests".into(),
            ));
        }
        if args.branches.len() > FAST_MAX_BRANCHES {
            return Err(PokError::Tool(format!(
                "fast_actions accepts at most {FAST_MAX_BRANCHES} branches"
            )));
        }
        let mut chain = vec![node(
            &args.goal,
            args.target_hint.as_deref(),
            &args.done_when,
            &args.allowed_operations,
            &[],
            Vec::new(),
            &args.input,
        )?];
        for step in &args.then {
            chain.push(subgoal(step)?);
        }
        let branches = args
            .branches
            .iter()
            .map(|branch| {
                Ok((
                    checked_when(&branch.when)?,
                    branch
                        .then
                        .iter()
                        .map(subgoal)
                        .collect::<Result<Vec<_>>>()?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let total_nodes = count(&chain)
            + branches
                .iter()
                .map(|(_, children)| count(children))
                .sum::<usize>();
        if total_nodes > FAST_MAX_NODES {
            return Err(PokError::Tool(format!(
                "a fast_actions plan may hold at most {FAST_MAX_NODES} steps across then and branches"
            )));
        }
        Ok(Self {
            chain,
            branches,
            total_nodes,
        })
    }
}

/// True when every step of an `execute_action_batch` call only clicks or
/// scrolls, i.e. the batch is raw navigation rather than text or key input.
pub(super) fn pure_navigation_batch(arguments: &Value) -> bool {
    let Some(steps) = arguments.get("steps").and_then(Value::as_array) else {
        return false;
    };
    !steps.is_empty()
        && steps.iter().all(|step| {
            let kind = step.get("kind").and_then(Value::as_str).unwrap_or_default();
            matches!(
                crate::builtins::normalized_batch_kind(kind),
                "click_target" | "scroll" | "scroll_view"
            )
        })
}

/// Raw navigation tools withheld in delegated mode until `fast_actions`
/// returns a subgoal it could not complete.
/// Marks raw navigation tools as locked in the compact catalog while the
/// delegated router owns navigation, so the catalog never lists a tool as
/// active whose schema is withheld.
pub(super) fn delegated_catalog(catalog: String, navigation_locked: bool) -> String {
    if !navigation_locked {
        return catalog;
    }
    catalog
        .lines()
        .map(|line| {
            let locked = DELEGATED_NAVIGATION_TOOLS
                .iter()
                .any(|tool| line.starts_with(&format!("- {tool} [")));
            if locked {
                line.replacen("; active]", "; locked: use fast_actions]", 1)
            } else {
                line.to_owned()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub(super) const DELEGATED_NAVIGATION_TOOLS: [&str; 5] = [
    "click_target",
    "scroll_view",
    "activate_window",
    "managed_browser_click",
    "managed_browser_scroll",
];

/// Primary-model turns that may use raw navigation after `fast_actions`
/// hands a subgoal back, before the router is required again.
pub(super) const RAW_NAVIGATION_UNLOCK_TURNS: u8 = 2;

/// The managed-browser facts a completion condition usually names: URL,
/// title, and the start of the visible text. `Null` without a snapshot.
pub(super) fn fast_page_evidence(browser_state: &Value) -> Value {
    let text = |key: &str, limit: usize| {
        browser_state
            .get(key)
            .and_then(Value::as_str)
            .map(|value| value.chars().take(limit).collect::<String>())
            .unwrap_or_default()
    };
    let url = text("url", 300);
    if url.is_empty() {
        return Value::Null;
    }
    json!({
        "url": url,
        "title": text("title", 200),
        "text_start": text("visible_text", 200),
    })
}

/// Four fixed scroll candidates for a delegated run that allows scrolling.
pub(super) fn fast_scroll_candidates(
    observation: &crate::types::Observation,
) -> Vec<DecisionCandidate> {
    let observation_id = crate::builtins::observation_id_for(observation);
    [
        ("up", "small"),
        ("up", "page"),
        ("down", "small"),
        ("down", "page"),
    ]
    .into_iter()
    .map(|(direction, amount)| DecisionCandidate {
        id: format!("scroll_{direction}_{amount}"),
        tool: "scroll_view".into(),
        arguments: json!({
            "observation_id": observation_id,
            "direction": direction,
            "amount": amount,
            "repeat": 1,
        }),
        description: format!("Scroll the current view {direction} by one {amount}"),
        kind: DecisionCandidateKind::Action,
        local_score: 0.05,
    })
    .collect()
}

/// Activation candidates for every listed window other than the current
/// foreground window, ranked by label overlap with the subgoal.
pub(super) fn fast_window_candidates(
    windows: &Value,
    observation: Option<&crate::types::Observation>,
    query: &str,
) -> Vec<DecisionCandidate> {
    let foreground = observation
        .and_then(|observation| observation.foreground_window.as_ref())
        .map(|window| window.id.clone());
    let query = query.to_lowercase();
    windows
        .get("windows")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .take(40)
        .enumerate()
        .filter_map(|(index, window)| {
            let window_id = window.get("id").and_then(Value::as_str)?;
            if foreground.as_deref() == Some(window_id)
                || window.get("elevated").and_then(Value::as_bool) == Some(true)
            {
                return None;
            }
            let title = window.get("title").and_then(Value::as_str).unwrap_or("");
            let app = window.get("app").and_then(Value::as_str).unwrap_or("");
            let label = format!("{title} {app}").to_lowercase();
            let overlap = label
                .split(|character: char| !character.is_alphanumeric())
                .filter(|term| term.len() >= 3 && query.contains(term))
                .count();
            Some(DecisionCandidate {
                id: format!("activate_window_{index}"),
                tool: "activate_window".into(),
                arguments: json!({"window_id": window_id}),
                description: format!("Activate the {title:?} ({app} window)"),
                kind: DecisionCandidateKind::Action,
                local_score: 0.3 + (overlap.min(5) as f64) * 0.1,
            })
        })
        .collect()
}

/// A target hint that is only a short label ("OK", "Discard All") and names
/// a target on screen exactly counts as if the planner had quoted it; any
/// other unquoted hint is a description, left for the fast model.
pub(super) fn plain_label_as_quoted(
    hint: &str,
    observation: Option<&crate::types::Observation>,
) -> String {
    let trimmed = hint.trim();
    let plain = !trimmed.is_empty()
        && trimmed.chars().count() <= 40
        && trimmed.split_whitespace().count() <= 4
        && !trimmed.contains(['"', '\u{201c}', '\u{201d}']);
    let label = crate::decision::normalized_evidence(trimmed);
    let on_screen = !label.is_empty()
        && observation.is_some_and(|observation| {
            observation
                .targets
                .iter()
                .any(|target| crate::decision::normalized_evidence(&target.name) == label)
        });
    if plain && on_screen {
        format!("\"{trimmed}\"")
    } else {
        hint.to_owned()
    }
}

/// The single candidate whose label strongly matches the subgoal text when
/// every other candidate scores at most half as well. Scores come from
/// `desktop_candidates_for_task` term overlap against goal and target hint.
pub(super) fn unambiguous_label_match(
    candidates: &[DecisionCandidate],
) -> Option<&DecisionCandidate> {
    let mut ranked = candidates.iter().collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.local_score.total_cmp(&left.local_score));
    let top = ranked.first()?;
    let runner_up = ranked.get(1).map_or(0.0, |candidate| candidate.local_score);
    (top.kind == DecisionCandidateKind::Action
        && top.local_score >= 0.8
        && runner_up <= top.local_score * 0.5)
        .then_some(*top)
}

/// Target gate for a delegated pick. When every candidate shares one
/// operation, the operation stage is forced and only target certainty is a
/// real signal; otherwise the ordinary combined gate applies.
/// A plain input string that names a key or shortcut rather than text.
fn looks_like_key(text: &str) -> bool {
    let parts = text.trim().split('+').map(str::trim).collect::<Vec<_>>();
    let Some((last, modifiers)) = parts.split_last() else {
        return false;
    };
    let named = matches!(
        last.to_ascii_lowercase().as_str(),
        "enter"
            | "tab"
            | "escape"
            | "esc"
            | "backspace"
            | "delete"
            | "space"
            | "home"
            | "end"
            | "pageup"
            | "pagedown"
            | "up"
            | "down"
            | "left"
            | "right"
            | "insert"
    ) || (last.len() >= 2
        && last.len() <= 3
        && last.starts_with(['F', 'f'])
        && last[1..]
            .chars()
            .all(|character| character.is_ascii_digit()));
    let modified = !modifiers.is_empty()
        && modifiers.iter().all(|modifier| {
            matches!(
                modifier.to_ascii_lowercase().as_str(),
                "ctrl" | "control" | "alt" | "shift" | "win"
            )
        })
        && (named || last.chars().count() == 1);
    named || modified
}

/// Accept plan shapes models commonly produce for nested steps, which the
/// schema normalizer does not reach inside the recursive `then`/`branches`:
/// lists wrapped as `{"item": ...}` (an XML-style serialization) or nested in
/// lists, empty entries, plain strings as input, and a step given only its
/// goal or only its condition.
pub(super) fn repair_fast_plan(node: &mut Value, top_level: bool) {
    unwrap_item_lists(node);
    coerce_plan_scalars(node);
    let Some(object) = node.as_object_mut() else {
        return;
    };
    // A list field given one entry directly ("then": {...}) is that entry.
    for key in [
        "then",
        "branches",
        "input",
        "avoid",
        "allowed_operations",
        "read",
        "on_interrupt",
    ] {
        if let Some(field) = object.get_mut(key)
            && !field.is_array()
            && !field.is_null()
        {
            *field = Value::Array(vec![field.take()]);
        }
    }
    if let Some(branches) = object.get_mut("branches").and_then(Value::as_array_mut) {
        for branch in branches {
            if let Some(steps) = branch.get_mut("then")
                && steps.is_object()
            {
                *steps = Value::Array(vec![steps.take()]);
            }
        }
    }
    // A list given inside a list (`"input": [[...]]`, left by an unwrapped
    // wrapper) is one list; empty entries are dropped.
    for key in ["then", "input", "avoid", "allowed_operations"] {
        if let Some(items) = object.get_mut(key).and_then(Value::as_array_mut) {
            let flat = items
                .drain(..)
                .flat_map(|item| match item {
                    Value::Array(inner) => inner,
                    other => vec![other],
                })
                .filter(|item| {
                    !(item.is_null() || item.as_str().is_some_and(|text| text.trim().is_empty()))
                })
                .collect::<Vec<_>>();
            *items = flat;
        }
    }
    // A plain string in input is a key ("Ctrl+S", "Enter") or text to type.
    if let Some(items) = object.get_mut("input").and_then(Value::as_array_mut) {
        for item in items {
            if let Some(text) = item.as_str().map(str::to_owned) {
                *item = if looks_like_key(&text) {
                    json!({"key": text.trim()})
                } else {
                    json!({"text": text})
                };
            }
        }
    }
    if !top_level {
        let first_read = object
            .get("read")
            .and_then(Value::as_array)
            .and_then(|reads| reads.first())
            .and_then(|read| read.get("label"))
            .and_then(Value::as_str)
            .map(|label| {
                let label = label.trim();
                if label.starts_with('"') {
                    label.to_owned()
                } else {
                    format!("\"{label}\"")
                }
            });
        let hint = object
            .get("target_hint")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .filter(|hint| !hint.trim().is_empty());
        if !object.contains_key("goal") {
            let goal = match (&first_read, &hint) {
                (_, Some(hint)) => format!("act on {hint}"),
                (Some(_), None) => "read the requested values".to_owned(),
                (None, None) => String::new(),
            };
            if !goal.is_empty() {
                object.insert("goal".into(), json!(goal));
            }
        }
        if !object.contains_key("done_when")
            && let Some(label) = first_read
        {
            object.insert("done_when".into(), json!(format!("{label} is visible")));
        }
    }
    // A step given only its goal or only its condition: each stands in for
    // the other (an unquoted condition is then judged by the fast model).
    let text_field = |object: &serde_json::Map<String, Value>, key: &str| {
        object
            .get(key)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(str::to_owned)
    };
    match (text_field(object, "goal"), text_field(object, "done_when")) {
        (None, Some(done_when)) => {
            object.insert("goal".into(), json!(done_when));
        }
        (Some(goal), None) => {
            object.insert("done_when".into(), json!(goal));
        }
        _ => {}
    }
    for key in ["then", "branches"] {
        if let Some(children) = object.get_mut(key).and_then(Value::as_array_mut) {
            for child in children {
                if key == "branches" {
                    if let Some(steps) = child.get_mut("then").and_then(Value::as_array_mut) {
                        for step in steps {
                            repair_fast_plan(step, false);
                        }
                    }
                } else {
                    repair_fast_plan(child, false);
                }
            }
        }
    }
}

/// Flags and counts written as strings ("true", "2") anywhere in a plan,
/// including nested steps the schema normalizer does not reach.
pub(super) fn coerce_plan_scalars(value: &mut Value) {
    match value {
        Value::Object(object) => {
            for (key, field) in object.iter_mut() {
                match (key.as_str(), &*field) {
                    ("replace_existing" | "stop", Value::String(text)) => {
                        match text.trim().to_ascii_lowercase().as_str() {
                            "true" => *field = Value::Bool(true),
                            "false" => *field = Value::Bool(false),
                            _ => {}
                        }
                    }
                    ("max_steps", Value::String(text)) => {
                        if let Ok(number) = text.trim().parse::<u32>() {
                            *field = json!(number);
                        }
                    }
                    _ => coerce_plan_scalars(field),
                }
            }
        }
        Value::Array(items) => items.iter_mut().for_each(coerce_plan_scalars),
        _ => {}
    }
}

/// Replace every `{"item": x}` object with a list (`x` itself when it is a
/// list, otherwise `[x]`). No fast-actions field is named `item`.
pub(super) fn unwrap_item_lists(value: &mut Value) {
    match value {
        Value::Object(object) => {
            if object.len() == 1
                && let Some(item) = object.get_mut("item")
            {
                let mut item = item.take();
                // Nested wrappers ({"item": {"item": [...]}}) are one list.
                while let Some(inner) = item
                    .as_object_mut()
                    .filter(|inner| inner.len() == 1)
                    .and_then(|inner| inner.get_mut("item"))
                {
                    item = inner.take();
                }
                *value = match item {
                    Value::Array(items) => Value::Array(items),
                    other => Value::Array(vec![other]),
                };
                unwrap_item_lists(value);
                return;
            }
            for child in object.values_mut() {
                unwrap_item_lists(child);
            }
        }
        Value::Array(items) => items.iter_mut().for_each(unwrap_item_lists),
        _ => {}
    }
}

pub(super) fn fast_target_eligible(
    decision: &crate::decision::DecisionResult,
    config: &crate::config::DecisionRouterConfig,
    single_operation: bool,
    read_only: bool,
) -> bool {
    let valid = decision.selected_probability.is_finite()
        && (0.0..=1.0).contains(&decision.selected_probability)
        && decision
            .probabilities
            .values()
            .all(|probability| probability.is_finite() && (0.0..=1.0).contains(probability));
    if !valid {
        return false;
    }
    if read_only {
        let operation_ok = single_operation
            || decision
                .operation_probability
                .is_some_and(|score| score >= FAST_SCROLL_MIN_PROBABILITY);
        let target = if config.native_choice_probabilities() {
            decision.target_probability
        } else {
            decision.target_confidence
        };
        return operation_ok && target.is_some_and(|score| score >= FAST_SCROLL_MIN_PROBABILITY);
    }
    if !single_operation {
        return ordinary_decision_scores_eligible(decision, config);
    }
    if config.native_choice_probabilities() {
        decision
            .target_probability
            .is_some_and(|score| score >= config.laya.min_target_probability)
    } else {
        decision.selected_probability >= config.min_selected_probability
            && decision
                .target_confidence
                .is_some_and(|score| score >= config.min_confidence)
    }
}

impl Session {
    /// Delegated fast-action run for one primary-model `fast_actions` call.
    ///
    /// The primary model owns planning: it names the subgoal, an optional
    /// target hint, and an observable completion condition. The router only
    /// answers bounded questions it is good at: which offered target matches
    /// the subgoal, and whether the condition is satisfied. Every action runs
    /// through the normal registry (policy, approval, freshness) and the
    /// continuity ledger, and the whole run returns as one tool result so the
    /// provider's tool-call/tool-result ordering is preserved.
    pub(super) async fn run_fast_actions(
        &mut self,
        turn: u32,
        arguments: &Value,
        metrics: &mut RunMetrics,
    ) -> Result<Value> {
        if self.decision_router.is_none() || self.decision_router_config.is_none() {
            self.raw_navigation_unlocked = u8::MAX;
            return Ok(json!({
                "status": "unavailable",
                "reason": "no fast decision model is configured; continue with ordinary tools",
            }));
        }
        let mut arguments = self
            .tools
            .normalized_arguments("fast_actions", arguments.clone());
        repair_fast_plan(&mut arguments, true);
        let args: FastActionsArgs = serde_json::from_value(arguments)
            .map_err(|error| {
                PokError::Tool(format!(
                    "invalid fast_actions arguments: {error}. Shape: every step (the top level and each entry of then, and of each branch's then) is an object with goal and done_when; input is a list of {{\"key\": ...}} or {{\"text\": ...}} objects; allowed_operations, avoid, then, and branches are lists."
                ))
            })?;
        let plan = FastPlan::from_args(&args)?;
        increment_metric(metrics, "fast_actions_runs", 1);
        // A new plan supersedes a hand-back still waiting for its resolution.
        self.pending_handback = None;
        let target_question_before = self
            .training
            .as_ref()
            .and_then(|recorder| recorder.last_question("target_click"));
        let mut run = FastRunContext {
            interrupts: args.on_interrupt.clone(),
            interrupts_used: 0,
        };
        let mut queue = std::collections::VecDeque::from(plan.chain);
        let mut root_branches = Some(plan.branches);
        let mut results = Vec::new();
        let mut nodes_run = 0_usize;
        let mut completed = 0_usize;
        let mut executed_steps = 0_usize;
        let mut status = "done".to_owned();
        loop {
            let node = match queue.pop_front() {
                Some(node) => node,
                None => match root_branches.take().filter(|branches| !branches.is_empty()) {
                    Some(branches) => match self
                        .select_fast_branch(turn, &args.goal, &branches, metrics)
                        .await?
                    {
                        Some(index) => {
                            results.push(json!({"branch_taken": branches[index].0}));
                            let (_, nodes) = branches.into_iter().nth(index).expect("branch index");
                            queue.extend(nodes);
                            continue;
                        }
                        None => {
                            status = "uncertain_branch".into();
                            results.push(json!({
                                "status": "uncertain_branch",
                                "options": branches.iter().map(|(when, _)| when).collect::<Vec<_>>(),
                            }));
                            break;
                        }
                    },
                    None => break,
                },
            };
            let mut avoid = args.avoid.clone();
            avoid.extend(node.avoid.iter().cloned());
            let allowed = if node.allowed.is_empty() {
                args.allowed_operations.clone()
            } else {
                node.allowed.clone()
            };
            nodes_run += 1;
            let outcome = if !node.input.is_empty() {
                self.run_fast_input_node(turn, &node, &avoid, metrics)
                    .await?
            } else if allowed.contains(&FastOperation::Drag) {
                self.run_fast_drag_node(turn, &node, &avoid, metrics)
                    .await?
            } else {
                self.run_fast_subgoal(
                    turn,
                    &node.goal,
                    &node.hint,
                    &node.done_when,
                    allowed,
                    args.max_steps,
                    &avoid,
                    &mut run,
                    metrics,
                )
                .await?
            };
            executed_steps += outcome["steps"].as_array().map_or(0, Vec::len);
            status = outcome["status"].as_str().unwrap_or("stalled").to_owned();
            results.push(outcome);
            // An unverified node still clicked the planner's named target, so
            // the plan continues; the next node's own target check guards it.
            if !matches!(status.as_str(), "done" | "unverified") {
                break;
            }
            completed += 1;
            if !node.branches.is_empty() {
                match self
                    .select_fast_branch(turn, &node.goal, &node.branches, metrics)
                    .await?
                {
                    Some(index) => {
                        results.push(json!({"branch_taken": node.branches[index].0}));
                        for next in node.branches[index].1.iter().rev() {
                            queue.push_front(next.clone());
                        }
                    }
                    None => {
                        status = "uncertain_branch".into();
                        results.push(json!({
                            "status": "uncertain_branch",
                            "options": node.branches.iter().map(|(when, _)| when).collect::<Vec<_>>(),
                        }));
                        break;
                    }
                }
            }
        }
        // "unverified" means the named target was clicked; the next navigation
        // still goes through fast_actions.
        self.raw_navigation_unlocked = if matches!(status.as_str(), "done" | "unverified") {
            0
        } else {
            RAW_NAVIGATION_UNLOCK_TURNS
        };
        let observation = self.context.latest_observation.lock().clone();
        let mut result = if nodes_run == 1 && results.len() == 1 {
            results.pop().unwrap_or_else(|| json!({}))
        } else {
            json!({
                "status": status,
                "completed_subgoals": completed,
                "total_subgoals": plan.total_nodes,
                "subgoals": results,
            })
        };
        if self.raw_navigation_unlocked > 0 {
            result["next"] = json!(
                "raw navigation tools (click_target, scroll_view, activate_window, managed_browser_click, managed_browser_scroll) are available for two turns; finish this subgoal with them from the observation below, then return to fast_actions"
            );
        }
        // Read the values the planner asked for straight from grounded
        // evidence, beside the screen they came from.
        if !args.read.is_empty() && matches!(status.as_str(), "done" | "unverified") {
            let browser_state = crate::browser::decision_state(&self.context).await;
            let has_page = browser_state
                .as_object()
                .is_some_and(|state| !state.is_empty());
            let reads = crate::decision::extract_reads(
                observation.as_ref(),
                has_page.then_some(&browser_state),
                &args.read,
            );
            let reads_complete = reads
                .values()
                .all(|value| value.get("source").and_then(Value::as_str) != Some("not_found"));
            increment_metric(
                metrics,
                if reads_complete {
                    "fast_actions_reads_complete"
                } else {
                    "fast_actions_reads_partial"
                },
                1,
            );
            result["reads"] = Value::Object(reads);
        }
        // A hand-back is a question System 1 could not answer: keep what it
        // was asked and saw, so the primary model's resolving action can
        // become its answer.
        if !matches!(status.as_str(), "done" | "unverified")
            && let Some(recorder) = &self.training
        {
            let question = recorder
                .last_question("target_click")
                .filter(|id| Some(id) != target_question_before.as_ref());
            let failed = results
                .iter()
                .rev()
                .find(|node| node.get("goal").is_some())
                .cloned()
                .unwrap_or_else(|| json!({}));
            let visible = observation
                .as_ref()
                .map(|observation| {
                    crate::decision::condition_evidence_labels(
                        observation,
                        &format!(
                            "{} {}",
                            args.goal,
                            args.target_hint.clone().unwrap_or_default()
                        ),
                        40,
                    )
                })
                .unwrap_or_default();
            self.pending_handback = Some(json!({
                "status": status,
                "goal": failed.get("goal").cloned().unwrap_or_else(|| json!(args.goal)),
                "target_hint": args.target_hint,
                "question_id": question,
                "window": observation
                    .as_ref()
                    .and_then(|observation| observation.foreground_window.as_ref())
                    .map(|window| json!({"title": window.title, "application": window.process_name})),
                "visible": visible,
                "attempts": 0,
            }));
        }
        // A plan whose conditions already held did nothing; say so, so the
        // planner does not take the empty step list for a failure.
        if executed_steps == 0 && status == "done" {
            result["note"] = json!(
                "every done_when already held on the current screen (shown below), so nothing needed doing"
            );
        }
        // Return the newest verified observation so the primary model can
        // continue from it (including its image) without another capture. In
        // the arena, results without it were followed by a capture 67-91% of
        // the time, against 24% with it.
        if let Some(observation) = observation.as_ref()
            && let Ok(value) =
                crate::builtins::model_observation_value(observation, self.context.annotate_targets)
        {
            result["observation"] = value;
        } else if let Some(observation) = observation.as_ref() {
            result["observation_id"] = json!(crate::builtins::observation_id_for(observation));
        }
        Ok(result)
    }

    /// A drag between two labels the planner quoted ("drag \"A\" onto
    /// \"B\""). Both must be found by their exact visible label, once each;
    /// otherwise nothing is dragged and the step hands back.
    pub(super) async fn run_fast_drag_node(
        &mut self,
        turn: u32,
        node: &FastPlanNode,
        avoid: &[String],
        metrics: &mut RunMetrics,
    ) -> Result<Value> {
        let outcome = |status: &str, reason: Option<String>, steps: Vec<Value>| {
            json!({
                "goal": node.goal,
                "done_when": node.done_when,
                "status": status,
                "reason": reason,
                "steps": steps,
            })
        };
        let completion = std::slice::from_ref(&node.done_when);
        if !self.skill_replay_active
            && self
                .pick_fast_condition(turn, &node.goal, completion, "completion", metrics)
                .await?
                == Some(0)
        {
            return Ok(outcome("done", None, Vec::new()));
        }
        let labels = crate::decision::quoted_labels(&node.hint);
        if labels.len() < 2 {
            return Ok(outcome(
                "uncertain",
                Some(
                    "a drag step quotes what to drag and where to drop it, e.g. target_hint \"drag \\\"Sheet2\\\" onto \\\"Sheet1\\\"\""
                        .into(),
                ),
                Vec::new(),
            ));
        }
        let Some(observation) = self.context.latest_observation.lock().clone() else {
            return Ok(outcome(
                "no_candidates",
                Some("there is no current observation to drag in".into()),
                Vec::new(),
            ));
        };
        let find = |label: &String| {
            let found =
                crate::decision::exact_label_candidates(&observation, std::slice::from_ref(label))
                    .into_iter()
                    .filter(|candidate| {
                        !avoid.iter().any(|phrase| {
                            crate::decision::mentions_phrase(&candidate.description, phrase)
                        })
                    })
                    .collect::<Vec<_>>();
            (found.len() == 1).then(|| found[0].arguments.clone())
        };
        let (Some(source), Some(destination)) = (find(&labels[0]), find(&labels[1])) else {
            return Ok(outcome(
                "uncertain",
                Some(format!(
                    "could not find {:?} and {:?} exactly once each by their visible labels",
                    labels[0], labels[1]
                )),
                Vec::new(),
            ));
        };
        let baseline = PopupBaseline::of(Some(&observation));
        increment_metric(metrics, "fast_actions_drag_nodes", 1);
        let arguments = json!({
            "observation_id": source["observation_id"],
            "source_target_id": source["target_id"],
            "expected_source_label": source["expected_label"],
            "destination_target_id": destination["target_id"],
            "expected_destination_label": destination["expected_label"],
        });
        let result = tokio::select! {
            () = self.context.cancellation.cancelled() => return Err(PokError::Cancelled),
            result = self.tools.call("drag_target", arguments.clone(), &self.context) => result,
        };
        self.record_motor("drag_target", &arguments, &result);
        self.show_latest_observation();
        let error = result
            .as_ref()
            .err()
            .map(|error| crate::decision::bounded_text(&error.to_string(), 200));
        self.log(
            "fast_actions_drag",
            json!({"turn": turn, "from": labels[0], "to": labels[1], "ok": error.is_none(), "error": error}),
        )?;
        let steps = vec![json!({
            "tool": "drag_target",
            "ok": error.is_none(),
            "from": labels[0],
            "to": labels[1],
            "error": error,
        })];
        if let Some(error) = error {
            return Ok(outcome("input_failed", Some(error), steps));
        }
        let done = self
            .pick_fast_condition(turn, &node.goal, completion, "completion", metrics)
            .await?
            == Some(0);
        let current = self.context.latest_observation.lock().clone();
        if !done
            && let Some(found) = current.as_ref().and_then(|observation| {
                unexpected_popup(
                    observation,
                    &baseline,
                    &format!("{} {} {}", node.goal, node.hint, node.done_when),
                )
            })
        {
            let found = self.look_closer(found).await.value();
            increment_metric(metrics, "fast_actions_popups", 1);
            self.log("fast_actions_popup", json!({"turn": turn, "popup": found}))?;
            let mut result = outcome(
                "popup",
                Some("a dialog opened that the plan did not expect; its title, text, and buttons are in popup".into()),
                steps,
            );
            result["popup"] = found;
            return Ok(result);
        }
        Ok(outcome(
            if done { "done" } else { "unverified" },
            (!done).then(|| "dragged as planned; completion could not be confirmed, so check the observation".into()),
            steps,
        ))
    }

    /// The primary model's action after a System 1 hand-back: the first one
    /// that makes verified progress is how the step should have been done,
    /// and becomes a training record. Three actions without progress drop it.
    pub(super) fn record_handback(
        &mut self,
        name: &str,
        arguments: &Value,
        outcome: &str,
        result: &Result<Value>,
    ) {
        if name == "fast_actions" || !super::run_loop::is_acting_tool(name) {
            return;
        }
        let Some(pending) = self.pending_handback.as_mut() else {
            return;
        };
        if outcome != "progress" || result.is_err() {
            let attempts = pending["attempts"].as_u64().unwrap_or(0) + 1;
            pending["attempts"] = json!(attempts);
            if attempts >= 3 {
                self.pending_handback = None;
            }
            return;
        }
        let Some(mut record) = self.pending_handback.take() else {
            return;
        };
        let value = result.as_ref().ok();
        let label = ["expected_label", "label", "expected_source_label"]
            .iter()
            .find_map(|key| arguments.get(*key).and_then(Value::as_str))
            .or_else(|| {
                value.and_then(|value| {
                    ["/model_action/label", "/label", "/target/label"]
                        .iter()
                        .find_map(|pointer| value.pointer(pointer).and_then(Value::as_str))
                })
            })
            .map(|label| crate::decision::bounded_text(label, 120));
        let mut action = arguments.clone();
        if let Some(object) = action.as_object_mut() {
            object.remove("observation_id");
            object.remove("view_id");
        }
        record["resolution"] = json!({
            "tool": name,
            "label": label,
            "arguments": crate::decision::bounded_text(&action.to_string(), 600),
        });
        if let Some(recorder) = &self.training {
            recorder.handback(&record);
        }
    }

    /// A closer look at a dialog whose message the noticing capture did not
    /// include: read the dialog window's own UI tree (no screenshot, no model).
    pub(super) async fn look_closer(&self, mut popup: Popup) -> Popup {
        if !popup.needs_closer_look() {
            return popup;
        }
        let window_id = match popup.window_id.clone() {
            Some(id) => Some(id),
            None => self
                .context
                .platform
                .list_windows()
                .await
                .ok()
                .and_then(|windows| {
                    windows
                        .into_iter()
                        .find(|window| window.title.trim() == popup.title)
                        .map(|window| window.id)
                }),
        };
        let Some(window_id) = window_id else {
            return popup;
        };
        let request = crate::types::CaptureRequest {
            scope: crate::types::CaptureScope::Window,
            window_id: Some(window_id),
            monitor_id: None,
            region: None,
            max_edge: 640,
        };
        if let Ok(Ok(elements)) = tokio::time::timeout(
            std::time::Duration::from_secs(3),
            self.context.platform.query_ui_tree_target(&request),
        )
        .await
        {
            popup.read(&elements, None);
        }
        popup
    }

    /// A plan node with `input`: System 2 wrote the exact keys and text;
    /// System 1 finds the field to focus by local grounding only, performs
    /// the input as one batch through the ordinary input pipeline (policy,
    /// password guard, effect checks), and checks `done_when`. A field that
    /// is not grounded by its visible label is handed back, never guessed.
    pub(super) async fn run_fast_input_node(
        &mut self,
        turn: u32,
        node: &FastPlanNode,
        avoid: &[String],
        metrics: &mut RunMetrics,
    ) -> Result<Value> {
        let outcome = |status: &str, reason: Option<String>, steps: Vec<Value>| {
            json!({
                "goal": node.goal,
                "done_when": node.done_when,
                "status": status,
                "reason": reason,
                "steps": steps,
            })
        };
        let completion = std::slice::from_ref(&node.done_when);
        if self
            .pick_fast_condition(turn, &node.goal, completion, "completion", metrics)
            .await?
            == Some(0)
        {
            return Ok(outcome("done", None, Vec::new()));
        }
        let mut batch = Vec::new();
        let mut first_step_attached = false;
        let mut target_label = None;
        if !node.hint.is_empty() {
            let Some(observation) = self.context.latest_observation.lock().clone() else {
                return Ok(outcome(
                    "no_candidates",
                    Some("there is no current observation to find the field in".into()),
                    Vec::new(),
                ));
            };
            let limit = self
                .decision_router_config
                .as_ref()
                .map_or(60, |config| config.max_candidates.min(250));
            let mut candidates = crate::decision::exact_label_candidates(
                &observation,
                &crate::decision::quoted_labels(&node.hint),
            );
            candidates.extend(desktop_candidates_for_task(
                &observation,
                &node.goal,
                &node.hint,
                limit,
            ));
            candidates.retain(|candidate| {
                candidate.tool == "click_target"
                    && candidate.arguments.get("expected_label").is_some()
                    && !avoid.iter().any(|phrase| {
                        crate::decision::mentions_phrase(&candidate.description, phrase)
                    })
            });
            let matched = crate::decision::exact_quoted_match(
                &candidates,
                &format!("{} {}", node.hint, node.goal),
            )
            .or_else(|| unambiguous_label_match(&candidates))
            .cloned();
            let Some(matched) = matched else {
                return Ok(outcome(
                    "uncertain",
                    Some("the field for this input is not grounded by a visible label; focus it yourself or quote its exact label".into()),
                    Vec::new(),
                ));
            };
            target_label = Some(crate::decision::bounded_text(&matched.description, 160));
            let target_id = matched.arguments["target_id"].clone();
            match node.input.first() {
                // Focus and type in one step: the batch refuses to type into
                // anything that is not a text field.
                Some(first) if first.text.is_some() => {
                    batch.push(json!({
                        "kind": "type_text",
                        "target_id": target_id,
                        "text": first.text,
                        "replace_existing": first.replace_existing,
                    }));
                    first_step_attached = true;
                }
                _ => batch.push(json!({
                    "kind": "click_target",
                    "target_id": target_id,
                    "expected_label": matched.arguments["expected_label"],
                })),
            }
        }
        for (index, step) in node.input.iter().enumerate() {
            if index == 0 && first_step_attached {
                continue;
            }
            if let Some(key) = &step.key {
                batch.push(json!({"kind": "key", "key": key}));
            } else if let Some(text) = &step.text {
                batch.push(json!({
                    "kind": "type_text",
                    "text": text,
                    "replace_existing": step.replace_existing,
                }));
            }
        }
        increment_metric(metrics, "fast_actions_input_nodes", 1);
        let before = self.context.latest_observation.lock().clone();
        let baseline = PopupBaseline::of(before.as_ref());
        let before_evidence = before
            .as_ref()
            .map(crate::decision::observation_evidence_text)
            .unwrap_or_default();
        let started = Instant::now();
        let result = tokio::select! {
            () = self.context.cancellation.cancelled() => return Err(PokError::Cancelled),
            result = self.tools.call("execute_action_batch", json!({"steps": batch.clone()}), &self.context) => result,
        };
        self.record_motor("execute_action_batch", &json!({"steps": batch}), &result);
        self.show_latest_observation();
        if let Ok(value) = &result
            && qualifies_as_fresh_evidence("execute_action_batch", value, true)
        {
            self.decision_router_fresh_evidence = true;
        }
        // A batch reports a step that failed inside its result, not as an error.
        let error = match &result {
            Err(error) => Some(error.to_string()),
            Ok(value) if value.get("failed_at_step").is_some() => Some(format!(
                "input step {} failed: {}",
                value["failed_at_step"],
                value
                    .get("error")
                    .and_then(Value::as_str)
                    .unwrap_or("not verified")
            )),
            Ok(_) => None,
        }
        .map(|error| crate::decision::bounded_text(&error, 200));
        self.log(
            "fast_actions_input",
            json!({
                "turn": turn,
                "target": target_label,
                "input_steps": node.input.len(),
                "ok": error.is_none(),
                "error": error,
                "elapsed_ms": u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
            }),
        )?;
        let steps = vec![json!({
            "tool": "execute_action_batch",
            "ok": error.is_none(),
            "outcome": if error.is_none() { "progress" } else { "failed" },
            "target": target_label,
            "input_steps": node.input.len(),
            "error": error,
        })];
        if let Some(error) = error {
            return Ok(outcome("input_failed", Some(error), steps));
        }
        let done = self
            .pick_fast_condition(turn, &node.goal, completion, "completion", metrics)
            .await?
            == Some(0);
        // The typing raised a dialog the plan did not expect (an invalid
        // value, a confirmation): pass on what it says.
        let current = self.context.latest_observation.lock().clone();
        // Glance at what was typed: the last text entry should now show on
        // screen. Fields in some applications (LibreOffice cells, for one)
        // do not expose their text to UI Automation, so this reads the screen.
        let typed_visible = node
            .input
            .iter()
            .rev()
            .find_map(|step| step.text.as_deref())
            .zip(current.as_ref())
            .and_then(|(text, observation)| {
                crate::decision::typed_text_visible(
                    &before_evidence,
                    &crate::decision::observation_evidence_text(observation),
                    text,
                )
            });
        if !done
            && let Some(found) = current.as_ref().and_then(|observation| {
                unexpected_popup(
                    observation,
                    &baseline,
                    &format!("{} {} {}", node.goal, node.hint, node.done_when),
                )
            })
        {
            let found = self.look_closer(found).await.value();
            increment_metric(metrics, "fast_actions_popups", 1);
            self.log("fast_actions_popup", json!({"turn": turn, "popup": found}))?;
            let mut result = outcome(
                "popup",
                Some(
                    "a dialog opened that the plan did not expect; its title, text, and buttons are in popup"
                        .into(),
                ),
                steps,
            );
            result["popup"] = found;
            return Ok(result);
        }
        let reason = (!done).then(|| {
            match typed_visible {
                Some(true) => "typed as planned and the text now shows on screen; the completion condition could not be confirmed",
                Some(false) => "typed, but the text does not show on screen afterward (a formula shows its result, and Enter or Tab may have moved on); check the result",
                None => "typed as planned; completion could not be confirmed, so check the observation",
            }
            .into()
        });
        let mut result = outcome(if done { "done" } else { "unverified" }, reason, steps);
        if let Some(visible) = typed_visible {
            result["typed_text_visible"] = json!(visible);
        }
        Ok(result)
    }

    /// Evidence for local condition checks and the bounded state a fast
    /// model sees when a condition quotes nothing: the desktop observation
    /// and, when present, the managed-browser page.
    pub(super) async fn fast_condition_context(&self, query: &str) -> (String, Value) {
        let observation = self.context.latest_observation.lock().clone();
        let browser_state = crate::browser::decision_state(&self.context).await;
        let mut evidence = crate::decision::page_evidence_text(&browser_state);
        let mut state = json!({});
        let page = fast_page_evidence(&browser_state);
        if !page.is_null() {
            state["page"] = page;
        }
        if let Some(observation) = observation.as_ref() {
            evidence.push(' ');
            evidence.push_str(&crate::decision::observation_evidence_text(observation));
            state["window"] = observation
                .foreground_window
                .as_ref()
                .map(|window| json!({"application": window.process_name, "title": window.title}))
                .unwrap_or(Value::Null);
            state["visible"] = json!(crate::decision::condition_evidence_labels(
                observation,
                query,
                if state.get("page").is_some() { 8 } else { 24 },
            ));
        }
        (evidence, state)
    }

    /// Which planner condition holds now. Quoted conditions are checked
    /// locally (the first true one in planner order wins); only unquoted
    /// conditions go to the fast model as one bounded choice. Returns the
    /// index into `conditions`, or `None` when none holds or the choice is
    /// uncertain. An "otherwise" condition is taken when nothing else holds.
    pub(super) async fn pick_fast_condition(
        &mut self,
        turn: u32,
        goal: &str,
        conditions: &[String],
        purpose: &str,
        metrics: &mut RunMetrics,
    ) -> Result<Option<usize>> {
        match self
            .fast_condition_outcome(turn, conditions, purpose, metrics)
            .await?
        {
            FastConditionOutcome::Picked(picked) => Ok(picked),
            FastConditionOutcome::NeedsRouter(pending) => {
                self.ask_fast_condition(turn, goal, &pending, metrics).await
            }
        }
    }

    /// Settle a condition choice locally when grounding can, or describe the
    /// router question it needs. Quoted conditions are checked against the
    /// evidence; only unquoted ones reach the router.
    pub(super) async fn fast_condition_outcome(
        &mut self,
        turn: u32,
        conditions: &[String],
        purpose: &str,
        metrics: &mut RunMetrics,
    ) -> Result<FastConditionOutcome> {
        let (evidence, state) = self.fast_condition_context(&conditions.join(" ")).await;
        let otherwise = conditions
            .iter()
            .position(|condition| crate::decision::is_otherwise_condition(condition));
        let verdicts = conditions
            .iter()
            .enumerate()
            .map(|(index, condition)| {
                (Some(index) != otherwise)
                    .then(|| crate::decision::grounded_condition(condition, &evidence))
                    .flatten()
            })
            .collect::<Vec<_>>();
        let pending = PendingFastCondition {
            options: Vec::new(),
            otherwise,
            count: conditions.len(),
            state,
            purpose: purpose.to_owned(),
            labels: conditions.to_vec(),
        };
        if let Some(index) = verdicts.iter().position(|verdict| *verdict == Some(true)) {
            increment_metric(metrics, "fast_actions_grounded_conditions", 1);
            self.log_fast_condition(turn, &pending, Some(index), "grounding", None)?;
            return Ok(FastConditionOutcome::Picked(Some(index)));
        }
        let unquoted = (0..conditions.len())
            .filter(|index| Some(*index) != otherwise && verdicts[*index].is_none())
            .collect::<Vec<_>>();
        if unquoted.is_empty() {
            increment_metric(metrics, "fast_actions_grounded_conditions", 1);
            self.log_fast_condition(turn, &pending, otherwise, "grounding", None)?;
            return Ok(FastConditionOutcome::Picked(otherwise));
        }
        if self.decision_router.is_none() {
            return Ok(FastConditionOutcome::Picked(None));
        }
        if !self
            .decision_router_config
            .as_ref()
            .is_some_and(|config| config.model_conditions_trusted())
        {
            // Unquoted conditions need a trusted model; otherwise the choice
            // is left to the planner rather than guessed.
            self.log_fast_condition(turn, &pending, None, "untrusted_router", None)?;
            return Ok(FastConditionOutcome::Picked(None));
        }
        let mut options = unquoted
            .iter()
            .map(|index| (format!("c{index}"), conditions[*index].clone()))
            .collect::<Vec<_>>();
        options.push((
            "none".into(),
            "None of the other conditions is true right now".into(),
        ));
        Ok(FastConditionOutcome::NeedsRouter(PendingFastCondition {
            options,
            ..pending
        }))
    }

    /// Ask a pending condition question on its own.
    pub(super) async fn ask_fast_condition(
        &mut self,
        turn: u32,
        goal: &str,
        pending: &PendingFastCondition,
        metrics: &mut RunMetrics,
    ) -> Result<Option<usize>> {
        let Some(router) = self.decision_router.clone() else {
            return Ok(None);
        };
        increment_metric(metrics, "fast_actions_router_conditions", 1);
        let choice = tokio::select! {
            () = self.context.cancellation.cancelled() => return Err(PokError::Cancelled),
            choice = router.choose_condition(goal, &pending.options, &pending.state) => choice,
        };
        self.resolve_fast_condition(
            turn,
            pending,
            choice.as_ref().ok(),
            choice.is_ok(),
            "router",
        )
    }

    /// Map a router condition answer back to a condition index and log it.
    pub(super) fn resolve_fast_condition(
        &self,
        turn: u32,
        pending: &PendingFastCondition,
        choice: Option<&crate::decision::ConditionChoice>,
        answered: bool,
        source: &str,
    ) -> Result<Option<usize>> {
        let picked = match choice {
            Some(choice) if choice.accepted && choice.option_id == "none" => pending.otherwise,
            Some(choice)
                if pending.purpose == "interrupt"
                    && choice.probability < FAST_INTERRUPT_MIN_PROBABILITY =>
            {
                None
            }
            Some(choice) if choice.accepted => choice
                .option_id
                .strip_prefix('c')
                .and_then(|index| index.parse::<usize>().ok())
                .filter(|index| *index < pending.count),
            _ => None,
        };
        self.log_fast_condition(
            turn,
            pending,
            picked,
            if answered { source } else { "router_error" },
            choice.map(|choice| choice.probability),
        )?;
        Ok(picked)
    }

    pub(super) fn log_fast_condition(
        &self,
        turn: u32,
        pending: &PendingFastCondition,
        picked: Option<usize>,
        source: &str,
        probability: Option<f64>,
    ) -> Result<()> {
        self.log(
            "fast_actions_condition",
            json!({
                "turn": turn,
                "purpose": pending.purpose,
                "picked": picked.map(|index| crate::decision::bounded_text(&pending.labels[index], 160)),
                "source": source,
                "probability": probability,
            }),
        )
    }

    pub(super) async fn select_fast_branch(
        &mut self,
        turn: u32,
        goal: &str,
        branches: &[(String, Vec<FastPlanNode>)],
        metrics: &mut RunMetrics,
    ) -> Result<Option<usize>> {
        let conditions = branches
            .iter()
            .map(|(when, _)| when.clone())
            .collect::<Vec<_>>();
        self.pick_fast_condition(turn, goal, &conditions, "branch", metrics)
            .await
    }

    /// One node of a delegated plan: act until `done_when` holds, the
    /// router is uncertain, no target remains, or the step budget runs out.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn run_fast_subgoal(
        &mut self,
        turn: u32,
        goal: &str,
        hint: &str,
        done_when: &str,
        allowed: Vec<FastOperation>,
        max_steps: Option<u32>,
        avoid: &[String],
        run: &mut FastRunContext,
        metrics: &mut RunMetrics,
    ) -> Result<Value> {
        let (Some(router), Some(config)) = (
            self.decision_router.clone(),
            self.decision_router_config.clone(),
        ) else {
            return Ok(json!({"status": "unavailable", "steps": []}));
        };
        let goal = goal.to_owned();
        let hint = plain_label_as_quoted(
            hint,
            self.context.latest_observation.lock().clone().as_ref(),
        );
        let done_when = done_when.to_owned();
        // The default is target matching only: one operation family keeps
        // the router out of the weak "which operation" planning choice.
        // Scrolling and window switching must be requested explicitly.
        let allowed = if allowed.is_empty() {
            vec![FastOperation::Click, FastOperation::BrowserClick]
        } else {
            allowed
        };
        let allowed_tools = allowed
            .iter()
            .map(|operation| operation.tool())
            .collect::<HashSet<_>>();
        let browser_allowed = allowed.iter().any(|operation| {
            matches!(
                operation,
                FastOperation::BrowserClick | FastOperation::BrowserScroll
            )
        });
        let desktop_allowed = allowed.iter().any(|operation| {
            matches!(
                operation,
                FastOperation::Click
                    | FastOperation::DoubleClick
                    | FastOperation::RightClick
                    | FastOperation::Hover
                    | FastOperation::Scroll
                    | FastOperation::ActivateWindow
            )
        });
        let max_steps = max_steps
            .unwrap_or(6)
            .clamp(1, 20)
            .min(u32::try_from(config.max_step_actions).unwrap_or(20));
        let root_request = self.context.active_task.lock().root_request.clone();
        let evidence_query = format!("{goal} {hint} {done_when}");
        self.log(
            "fast_actions_started",
            json!({
                "turn": turn,
                "goal": crate::decision::bounded_text(&goal, 500),
                "target_hint": crate::decision::bounded_text(&hint, 300),
                "done_when": crate::decision::bounded_text(&done_when, 300),
                "allowed_operations": allowed,
                "max_steps": max_steps,
            }),
        )?;

        let mut steps: Vec<Value> = Vec::new();
        let mut attempted = HashSet::<String>::new();
        let mut consecutive_setbacks = 0_u8;
        let mut windows: Option<Value> = None;
        let mut status = "budget_exhausted";
        let mut reason: Option<String> = None;
        let mut completion: Option<crate::decision::ConditionVerdict> = None;
        let mut alternatives: Vec<Value> = Vec::new();
        let mut named_target_clicked = false;
        // The named target was a window to switch to: in background mode that
        // selects the agent's window without changing the screen.
        let mut named_target_was_window = false;
        let mut fired_interrupts = HashSet::<usize>::new();
        // An interrupt rule the router matched in a fan-out answer; acted on
        // at the start of the next step.
        let mut pending_interrupt: Option<usize> = None;
        // Surface evidence when the node started and at the previous step:
        // a click that leaves it unchanged proves nothing.
        let mut node_start_evidence: Option<String> = None;
        let mut last_step_evidence: Option<String> = None;
        // Dialogs on screen when the node started, and a new one found since.
        let mut popup_baseline: Option<PopupBaseline> = None;
        // Whether this node already took a fresh look for the input guard.
        let mut regrounded = false;
        let mut popup: Option<Value> = None;

        for step in 0..=max_steps {
            let mut deferred_completion = false;
            // An unquoted interrupt question waiting to ride along with this
            // step's target pick, with the rule index behind each option.
            let mut deferred_interrupt: Option<(PendingFastCondition, Vec<usize>)> = None;
            let observation = self.context.latest_observation.lock().clone();
            let baseline = popup_baseline
                .get_or_insert_with(|| PopupBaseline::of(observation.as_ref()))
                .clone();
            let browser_state = if browser_allowed {
                crate::browser::decision_state(&self.context).await
            } else {
                json!({})
            };
            let window = observation
                .as_ref()
                .and_then(|observation| observation.foreground_window.as_ref())
                .map(|window| json!({"application": window.process_name, "title": window.title}));
            // Completion evidence comes only from the surface this subgoal
            // acts on. Small decision models have short contexts; a stale
            // desktop capture beside a browser page drowned out the URL and
            // title the condition depends on.
            let mut condition_state = json!({"last_action": steps.last()});
            let page = fast_page_evidence(&browser_state);
            if browser_allowed && !page.is_null() {
                condition_state["page"] = page;
            }
            if (desktop_allowed || condition_state.get("page").is_none())
                && let Some(observation) = observation.as_ref()
            {
                condition_state["window"] = window.clone().unwrap_or(Value::Null);
                condition_state["visible"] = json!(crate::decision::condition_evidence_labels(
                    observation,
                    &evidence_query,
                    if condition_state.get("page").is_some() {
                        8
                    } else {
                        24
                    },
                ));
            }
            // Run a deferred completion check on its own. Used on every path
            // that exits or acts without a router pick, so completion is
            // always judged before anything is clicked.
            macro_rules! settle_completion {
                () => {
                    if std::mem::take(&mut deferred_completion) {
                        let settle_started = Instant::now();
                        let verdict = tokio::select! {
                            () = self.context.cancellation.cancelled() => return Err(PokError::Cancelled),
                            verdict = router.check_condition(&goal, &done_when, &condition_state) => verdict,
                        };
                        self.log(
                            "fast_actions_completion_check",
                            json!({
                                "turn": turn,
                                "step": step,
                                "source": "router",
                                "satisfied": verdict.as_ref().ok().map(|verdict| verdict.satisfied),
                                "answer": verdict.as_ref().ok().map(|verdict| verdict.answer.clone()),
                                "probability": verdict.as_ref().ok().map(|verdict| verdict.probability),
                                "error": verdict.as_ref().err().map(ToString::to_string),
                                "elapsed_ms": settle_started.elapsed().as_millis(),
                            }),
                        )?;
                        if let Ok(verdict) = verdict {
                            let satisfied = verdict.satisfied;
                            completion = Some(verdict);
                            if satisfied {
                                status = "done";
                                break;
                            }
                        }
                    }
                };
            }
            // Settle a deferred interrupt question on its own before a step
            // acts or exits without a router pick; a match restarts the step
            // with that rule's action.
            macro_rules! settle_interrupt {
                () => {
                    if let Some((pending, rules)) = deferred_interrupt.take()
                        && let Some(picked) = self
                            .ask_fast_condition(turn, &goal, &pending, metrics)
                            .await?
                    {
                        pending_interrupt = Some(rules[picked]);
                        continue;
                    }
                };
            }
            // Popup and dialog rules come first: a matching rule either
            // redirects this step to its quoted target or ends the run.
            let mut interrupt_hint: Option<String> = None;
            // Each rule acts at most once per node: a popup still present after
            // its action means the action did not work, and repeating it loops.
            let open_rules = (0..run.interrupts.len())
                .filter(|index| !fired_interrupts.contains(index))
                .collect::<Vec<_>>();
            let mut fire = pending_interrupt.take();
            if fire.is_none()
                && step < max_steps
                && !open_rules.is_empty()
                && run.interrupts_used < FAST_MAX_INTERRUPT_ACTIONS
            {
                let conditions = open_rules
                    .iter()
                    .map(|index| run.interrupts[*index].when.clone())
                    .collect::<Vec<_>>();
                match self
                    .fast_condition_outcome(turn, &conditions, "interrupt", metrics)
                    .await?
                {
                    FastConditionOutcome::Picked(picked) => {
                        fire = picked.map(|picked| open_rules[picked]);
                    }
                    // Fanned out with this step's target pick; a step that
                    // makes no router pick settles it on its own.
                    FastConditionOutcome::NeedsRouter(pending)
                        if config.batch_router_questions && step + 1 < max_steps =>
                    {
                        deferred_interrupt = Some((pending, open_rules.clone()));
                    }
                    FastConditionOutcome::NeedsRouter(pending) => {
                        fire = self
                            .ask_fast_condition(turn, &goal, &pending, metrics)
                            .await?
                            .map(|picked| open_rules[picked]);
                    }
                }
            }
            if let Some(index) = fire {
                fired_interrupts.insert(index);
                run.interrupts_used += 1;
                increment_metric(metrics, "fast_actions_interrupts", 1);
                let rule = run.interrupts[index].clone();
                match rule
                    .target_hint
                    .filter(|hint| !rule.stop && !hint.trim().is_empty())
                {
                    Some(hint) => interrupt_hint = Some(hint),
                    None => {
                        status = "interrupted";
                        reason = Some(format!(
                            "interrupt rule matched: {}",
                            crate::decision::bounded_text(&rule.when, 200)
                        ));
                        break;
                    }
                }
            }
            let check_started = Instant::now();
            // A done_when that quotes exact evidence is checked locally; only
            // an unquoted condition needs the fast model's judgment.
            let mut evidence = String::new();
            if condition_state.get("page").is_some() {
                evidence.push_str(&crate::decision::page_evidence_text(&browser_state));
            }
            if condition_state.get("visible").is_some()
                && let Some(observation) = observation.as_ref()
            {
                evidence.push(' ');
                evidence.push_str(&crate::decision::observation_evidence_text(observation));
            }
            let start_evidence = node_start_evidence.get_or_insert_with(|| evidence.clone());
            let unchanged_since_start = *start_evidence == evidence;
            let unchanged_since_last = last_step_evidence.as_ref() == Some(&evidence);
            last_step_evidence = Some(evidence.clone());
            if let Some(recorder) = &self.training {
                let window_info = observation
                    .as_ref()
                    .and_then(|observation| observation.foreground_window.as_ref());
                recorder.observe_window(
                    &format!(
                        "{} {} {}",
                        window_info.map_or("", |window| window.title.as_str()),
                        browser_state
                            .get("title")
                            .and_then(Value::as_str)
                            .unwrap_or_default(),
                        browser_state
                            .get("url")
                            .and_then(Value::as_str)
                            .unwrap_or_default(),
                    ),
                    window_info.map_or("", |window| window.process_name.as_str()),
                );
            }
            // Set only when the quoted evidence itself decided the condition:
            // a proven answer worth keeping as training data.
            let mut grounding_proof = None;
            // A pending interrupt action is handled before judging completion.
            let grounded = if interrupt_hint.is_some() {
                Some(false)
            } else if self.skill_replay_active && steps.is_empty() {
                // A replayed step is a recorded action: perform it, then judge
                // its condition (its end state may already show beforehand).
                Some(false)
            } else if (steps.is_empty() || unchanged_since_start)
                && crate::decision::condition_quotes_only_target(&done_when, &hint)
                && crate::decision::grounded_condition(
                    &done_when,
                    &crate::decision::title_evidence_text(observation.as_ref(), &browser_state),
                ) != Some(true)
            {
                // The quoted label may be the target itself: seeing it proves
                // only that the target is visible, which no model can resolve
                // from this screen either. Act on the target first, and count
                // it only once the screen has changed. A window or page title
                // that already shows it is real evidence.
                Some(false)
            } else {
                let proof = match crate::decision::grounded_condition(&done_when, &evidence) {
                    // The quotes are there, but the condition also asks for
                    // something they cannot show (a dialog closed, a folder
                    // opened): that part needs a judgment.
                    Some(true) if crate::decision::condition_has_unquoted_clause(&done_when) => {
                        None
                    }
                    grounded => grounded,
                };
                grounding_proof = proof;
                proof
            };
            if let (Some(recorder), Some(satisfied)) = (&self.training, grounding_proof)
                && let Some(body) =
                    router.completion_training_body(&goal, &done_when, &condition_state)
            {
                recorder.question("grounding", &body, None);
                recorder.label(
                    "satisfied",
                    if satisfied { "yes" } else { "no" },
                    "grounding",
                );
            }
            let verdict = if let Some(satisfied) = grounded {
                increment_metric(metrics, "fast_actions_grounded_checks", 1);
                Ok(crate::decision::ConditionVerdict {
                    satisfied,
                    answer: if satisfied { "yes" } else { "no" }.into(),
                    probability: if satisfied { 1.0 } else { 0.0 },
                    model: "grounding".into(),
                })
            } else if !config.model_conditions_trusted() {
                // An untrusted backend never declares completion; the node
                // keeps working and hands back unverified after its target.
                Ok(crate::decision::ConditionVerdict {
                    satisfied: false,
                    answer: "unknown".into(),
                    probability: 0.0,
                    model: "untrusted".into(),
                })
            } else if config.batch_router_questions {
                // Asked together with the target pick when this step needs
                // one; settled on its own before any other exit or action.
                deferred_completion = true;
                Ok(crate::decision::ConditionVerdict {
                    satisfied: false,
                    answer: "deferred".into(),
                    probability: 0.0,
                    model: "deferred".into(),
                })
            } else {
                tokio::select! {
                    () = self.context.cancellation.cancelled() => return Err(PokError::Cancelled),
                    verdict = router.check_condition(&goal, &done_when, &condition_state) => verdict,
                }
            };
            increment_metric(metrics, "fast_actions_completion_checks", 1);
            self.log(
                "fast_actions_completion_check",
                json!({
                    "turn": turn,
                    "step": step,
                    "source": if grounded.is_some() {
                        "grounding"
                    } else if deferred_completion {
                        "deferred_to_batch"
                    } else if config.model_conditions_trusted() {
                        "router"
                    } else {
                        "untrusted_router"
                    },
                    "satisfied": verdict.as_ref().ok().map(|verdict| verdict.satisfied),
                    "answer": verdict.as_ref().ok().map(|verdict| verdict.answer.clone()),
                    "probability": verdict.as_ref().ok().map(|verdict| verdict.probability),
                    "error": verdict.as_ref().err().map(ToString::to_string),
                    "elapsed_ms": check_started.elapsed().as_millis(),
                }),
            )?;
            if let Ok(verdict) = verdict {
                let satisfied = verdict.satisfied;
                completion = Some(verdict);
                if satisfied {
                    status = "done";
                    break;
                }
            }
            // A dialog opened that the plan did not mention (an error, a
            // warning, a question): read it and hand it back, rather than
            // acting behind it or stalling on a screen it covers.
            if interrupt_hint.is_none()
                && let Some(found) = observation.as_ref().and_then(|observation| {
                    unexpected_popup(
                        observation,
                        &baseline,
                        &format!("{goal} {hint} {done_when}"),
                    )
                })
            {
                settle_completion!();
                let found = self.look_closer(found).await.value();
                increment_metric(metrics, "fast_actions_popups", 1);
                self.log(
                    "fast_actions_popup",
                    json!({"turn": turn, "step": step, "popup": found}),
                )?;
                status = "popup";
                reason = Some(
                    "a dialog opened that the plan did not expect; its title, text, and buttons are in popup"
                        .into(),
                );
                popup = Some(found);
                break;
            }
            if step == max_steps {
                settle_interrupt!();
                settle_completion!();
                break;
            }
            // The planner's named target was already acted on and the
            // condition still is not confirmed: let the planner verify rather
            // than letting the router wander to other targets and undo it.
            if named_target_clicked && interrupt_hint.is_none() {
                settle_interrupt!();
                settle_completion!();
                if named_target_was_window {
                    status = "unverified";
                    reason =
                        Some("switched to the window you named; capture it to continue".into());
                } else if unchanged_since_last {
                    // Say so plainly: a click that changed nothing needs a
                    // different approach, not another identical click.
                    status = "stalled";
                    reason =
                        Some("clicked the target you named, but nothing on screen changed".into());
                } else {
                    status = "unverified";
                    reason = Some(
                        "clicked the target you named; completion could not be confirmed, so check the observation"
                            .into(),
                    );
                }
                break;
            }

            let elevated = observation.as_ref().is_some_and(|observation| {
                observation
                    .foreground_window
                    .as_ref()
                    .is_some_and(|window| window.elevated)
            });
            let limit = config.max_candidates.min(250);
            let mut candidates = Vec::new();
            if !elevated && let Some(observation) = observation.as_ref() {
                candidates.extend(desktop_candidates_for_task(
                    observation,
                    &goal,
                    &hint,
                    limit,
                ));
                if allowed.contains(&FastOperation::Scroll) {
                    candidates.extend(fast_scroll_candidates(observation));
                }
            }
            if allowed.contains(&FastOperation::ActivateWindow) {
                if windows.is_none() {
                    let listed = tokio::select! {
                        () = self.context.cancellation.cancelled() => return Err(PokError::Cancelled),
                        listed = self.tools.call("list_windows", json!({}), &self.context) => listed,
                    };
                    windows = Some(listed.unwrap_or_else(|_| json!({})));
                }
                candidates.extend(fast_window_candidates(
                    windows.as_ref().unwrap_or(&Value::Null),
                    observation.as_ref(),
                    &format!("{goal} {hint}"),
                ));
            }
            if browser_allowed {
                candidates.extend(
                    crate::browser::decision_candidates(&self.context, &goal, &hint, limit).await,
                );
            }
            // A quoted target may be any reliably grounded control (a button,
            // for example), not only the navigation controls offered above.
            if !elevated
                && allowed.iter().any(|operation| {
                    matches!(
                        operation,
                        FastOperation::Click
                            | FastOperation::DoubleClick
                            | FastOperation::RightClick
                            | FastOperation::Hover
                    )
                })
                && let Some(observation) = observation.as_ref()
            {
                // Without a target hint, a label the goal quotes names the
                // target ("open the \"Table\" menu").
                let named = if hint.trim().is_empty() { &goal } else { &hint };
                candidates.extend(crate::decision::exact_label_candidates(
                    observation,
                    &crate::decision::quoted_labels(named),
                ));
            }
            // With double_click allowed, desktop clicks open their target
            // (folders and files in Explorer only select on a single click).
            if allowed.contains(&FastOperation::DoubleClick) {
                for candidate in &mut candidates {
                    if candidate.tool == "click_target" {
                        candidate.arguments["double_click"] = json!(true);
                    }
                }
            }
            // A node's pointer action applies to whichever target is chosen:
            // a right-click opens its context menu, a hover rests on it.
            if allowed.contains(&FastOperation::RightClick) {
                for candidate in &mut candidates {
                    if candidate.tool == "click_target" {
                        candidate.arguments["button"] = json!("right");
                    }
                }
            } else if allowed.contains(&FastOperation::Hover)
                && !allowed.contains(&FastOperation::Click)
                && !allowed.contains(&FastOperation::DoubleClick)
            {
                for candidate in &mut candidates {
                    if candidate.tool == "click_target"
                        && let Some(arguments) = candidate.arguments.as_object_mut()
                    {
                        candidate.tool = "hover_target".into();
                        arguments.remove("button");
                        arguments.remove("double_click");
                    }
                }
            }
            if let Some(interrupt) = interrupt_hint.as_deref() {
                // An interrupt acts only on the label its rule quotes.
                candidates.clear();
                if !elevated && let Some(observation) = observation.as_ref() {
                    candidates.extend(crate::decision::exact_label_candidates(
                        observation,
                        &crate::decision::quoted_labels(interrupt),
                    ));
                }
                candidates.extend(
                    crate::browser::decision_candidates(&self.context, interrupt, interrupt, limit)
                        .await
                        .into_iter()
                        .filter(|candidate| candidate.tool == "managed_browser_click"),
                );
            }
            candidates.retain(|candidate| {
                candidate.kind == DecisionCandidateKind::Action
                    && (interrupt_hint.is_some() || allowed_tools.contains(candidate.tool.as_str()))
                    && !avoid
                        .iter()
                        .any(|phrase| crate::decision::mentions_phrase(&candidate.description, phrase))
                    // Scrolling the same way again is how a search proceeds;
                    // no-progress detection stops it at the end of the view.
                    && (is_read_only_fast_tool(&candidate.tool)
                        || !attempted.contains(&decision_candidate_fingerprint(candidate)))
            });
            // The commit that completes the user's request (save, send,
            // submit, ...) stays with the primary model and its commit
            // review. Hold it back, but keep it ready to hand over.
            let (held_commits, remaining): (Vec<_>, Vec<_>) =
                candidates.into_iter().partition(|candidate| {
                    requested_commit_label(&root_request, &candidate.description).is_some()
                });
            candidates = remaining;
            let named_commit = if interrupt_hint.is_none() {
                crate::decision::exact_quoted_match(&held_commits, &format!("{hint} {goal}"))
                    .or_else(|| {
                        (candidates.is_empty() && held_commits.len() == 1).then(|| &held_commits[0])
                    })
                    .cloned()
            } else {
                None
            };
            if let Some(commit) = named_commit {
                settle_interrupt!();
                settle_completion!();
                status = "commit_required";
                reason = Some(
                    "the step that completes the request is a commit, which you confirm: call this tool with these arguments"
                        .into(),
                );
                alternatives = vec![json!({
                    "tool": commit.tool,
                    "arguments": commit.arguments,
                    "description": crate::decision::bounded_text(&commit.description, 160),
                })];
                break;
            }
            // Candidate sources overlap (task candidates add scroll options
            // when the goal mentions scrolling); the router needs unique IDs.
            let mut seen_ids = HashSet::new();
            candidates.retain(|candidate| seen_ids.insert(candidate.id.clone()));
            // The named target is not on this screen yet, so no visible
            // control can be it: the small, well-posed question is which way
            // to scroll, not which unrelated control to click.
            let scroll_search = interrupt_hint.is_none()
                && crate::decision::quoted_labels_absent(&hint, &evidence)
                && candidates
                    .iter()
                    .any(|candidate| is_read_only_fast_tool(&candidate.tool));
            if scroll_search {
                candidates.retain(|candidate| is_read_only_fast_tool(&candidate.tool));
                increment_metric(metrics, "fast_actions_scroll_searches", 1);
                self.log(
                    "fast_actions_scroll_search",
                    json!({"turn": turn, "step": step, "options": candidates.len()}),
                )?;
            }
            if candidates.is_empty() {
                settle_interrupt!();
                settle_completion!();
                status = "no_candidates";
                reason = Some("no reversible target matching the subgoal is visible".into());
                break;
            }
            // Rank by the primary model's focused hint (or goal) so local
            // evidence reflects the named target, not incidental overlap with
            // a long goal sentence.
            let focus = if hint.is_empty() {
                goal.as_str()
            } else {
                hint.as_str()
            };
            for candidate in &mut candidates {
                candidate.local_score = crate::decision::hint_relevance(focus, candidate);
            }
            // Decision models are built for small, well-posed option menus
            // (2-16 options): offer only the 16 best-grounded candidates.
            let candidates = rank_and_limit_candidates(candidates, limit.min(FAST_MAX_OPTIONS));
            let best_local = candidates
                .iter()
                .map(|candidate| candidate.local_score)
                .fold(0.0_f64, f64::max);
            let pick_started = Instant::now();
            let mut picked_named_target = false;
            let candidate = if let Some(interrupt) = interrupt_hint.as_deref() {
                let Some(matched) = crate::decision::exact_quoted_match(&candidates, interrupt)
                else {
                    status = "interrupted";
                    reason = Some(format!(
                        "an interrupt rule matched but its target {} is not visible",
                        crate::decision::bounded_text(interrupt, 120)
                    ));
                    break;
                };
                self.log(
                    "fast_actions_interrupt_action",
                    json!({
                        "turn": turn,
                        "description": crate::decision::bounded_text(&matched.description, 160),
                    }),
                )?;
                matched.clone()
            } else if candidates.len() == 1 && !self.skill_replay_active {
                settle_interrupt!();
                settle_completion!();
                increment_metric(metrics, "fast_actions_structural_bypass", 1);
                // The only candidate may be exactly the label the planner
                // named; then it is the named target, as on any other path.
                picked_named_target =
                    crate::decision::exact_quoted_match(&candidates, &format!("{hint} {goal}"))
                        .is_some();
                candidates[0].clone()
            } else if let Some(matched) =
                crate::decision::exact_quoted_match(&candidates, &format!("{hint} {goal}")).or_else(
                    || {
                        (!self.skill_replay_active)
                            .then(|| unambiguous_label_match(&candidates))
                            .flatten()
                    },
                )
            {
                settle_interrupt!();
                settle_completion!();
                picked_named_target = true;
                // The primary model's hint already names one visible label and
                // nothing else comes close; asking the router would only add a
                // chance to override an exact grounding match.
                increment_metric(metrics, "fast_actions_local_matches", 1);
                if let Some(recorder) = &self.training
                    && let Some(body) = router.training_body(DecisionRequest {
                        purpose: DecisionPurpose::NextAction,
                        task: goal.clone(),
                        current_step: if hint.is_empty() {
                            done_when.clone()
                        } else {
                            hint.clone()
                        },
                        candidates: candidates.clone(),
                        state: json!({
                            "window": window,
                            "browser": browser_state,
                            "last_action": steps.last(),
                            "previous_outcome": self.continuity.current.as_ref().map(|state| &state.outcome),
                            "state_revision": self.continuity.state_revision,
                        }),
                    })
                {
                    let operation = crate::decision::candidate_operation(matched);
                    recorder.question("grounding", &body, None);
                    recorder.label("operation", &operation, "quoted_label_match");
                    recorder.label(&format!("target_{operation}"), &matched.id, "quoted_label_match");
                }
                self.log(
                    "fast_actions_local_match",
                    json!({
                        "turn": turn,
                        "candidate_id": matched.id,
                        "description": crate::decision::bounded_text(&matched.description, 160),
                        "local_score": matched.local_score,
                    }),
                )?;
                matched.clone()
            } else if scroll_search
                && !config.model_targets_trusted()
                && let Some(default) = default_scroll_down(&candidates)
            {
                settle_interrupt!();
                settle_completion!();
                // Without a trusted model, continue down the page: a
                // read-only step that reveals more of it.
                default.clone()
            } else if !config.model_targets_trusted() || self.skill_replay_active {
                settle_interrupt!();
                settle_completion!();
                // Grounding could not decide and this backend is not trusted
                // to pick targets (or a skill is being replayed, which acts
                // only on the labels it saved): hand the choice back with the
                // local ranking.
                status = "uncertain";
                reason =
                    Some("no visible label matches the hint exactly; choose one yourself".into());
                alternatives = candidates
                    .iter()
                    .take(3)
                    .map(|candidate| {
                        json!({
                            "tool": candidate.tool,
                            "arguments": candidate.arguments,
                            "description": crate::decision::bounded_text(&candidate.description, 160),
                            "local_score": candidate.local_score,
                        })
                    })
                    .collect();
                break;
            } else {
                let mut decision_state = json!({
                    "window": window,
                    "browser": browser_state,
                    "last_action": steps.last(),
                    "previous_outcome": self.continuity.current.as_ref().map(|state| &state.outcome),
                    "state_revision": self.continuity.state_revision,
                });
                let request = DecisionRequest {
                    purpose: DecisionPurpose::NextAction,
                    task: goal.clone(),
                    current_step: if hint.is_empty() {
                        done_when.clone()
                    } else {
                        hint.clone()
                    },
                    candidates: candidates.clone(),
                    state: Value::Null,
                };
                // Shown live in the dashboard ("Laya is evaluating options").
                self.emit(AgentEvent::DecisionRouterStarted {
                    turn,
                    purpose: "fast_actions".into(),
                    candidate_count: candidates.len(),
                });
                let fan_completion = std::mem::take(&mut deferred_completion);
                let fan_interrupt = deferred_interrupt.take();
                let decision = if fan_completion || fan_interrupt.is_some() {
                    // Speculative fan-out: every question this step may need
                    // goes in one request against one state; answers the step
                    // does not use are ignored.
                    for key in ["visible", "page"] {
                        if let Some(value) = condition_state.get(key) {
                            decision_state[key] = value.clone();
                        }
                    }
                    if let Some((pending, _)) = &fan_interrupt {
                        decision_state["interrupt_evidence"] = pending.state.clone();
                    }
                    let batch_started = Instant::now();
                    let answers = tokio::select! {
                        () = self.context.cancellation.cancelled() => return Err(PokError::Cancelled),
                        answers = router.decide_fanout(
                            DecisionRequest { state: decision_state.clone(), ..request },
                            &goal,
                            fan_completion.then_some((done_when.as_str(), &condition_state)),
                            fan_interrupt
                                .as_ref()
                                .map(|(pending, _)| (pending.options.as_slice(), &pending.state)),
                        ) => answers,
                    };
                    increment_metric(metrics, "fast_actions_batched_requests", 1);
                    match answers {
                        Ok(answers) => {
                            // An interrupt comes before the step's own work.
                            if let Some((pending, rules)) = &fan_interrupt {
                                increment_metric(metrics, "fast_actions_fanout_conditions", 1);
                                if let Some(picked) = self.resolve_fast_condition(
                                    turn,
                                    pending,
                                    answers.condition.as_ref(),
                                    answers.condition.is_some(),
                                    "router_batched",
                                )? {
                                    pending_interrupt = Some(rules[picked]);
                                    continue;
                                }
                            }
                            if let Some(verdict) = answers.completion {
                                self.log(
                                    "fast_actions_completion_check",
                                    json!({
                                        "turn": turn,
                                        "step": step,
                                        "source": "router_batched",
                                        "satisfied": verdict.satisfied,
                                        "answer": verdict.answer,
                                        "probability": verdict.probability,
                                        "elapsed_ms": batch_started.elapsed().as_millis(),
                                    }),
                                )?;
                                let satisfied = verdict.satisfied;
                                completion = Some(verdict);
                                if satisfied {
                                    status = "done";
                                    break;
                                }
                            }
                            Ok(answers.decision)
                        }
                        Err(error) => Err(error),
                    }
                } else {
                    tokio::select! {
                        () = self.context.cancellation.cancelled() => return Err(PokError::Cancelled),
                        decision = router.decide(DecisionRequest { state: decision_state.clone(), ..request }) => decision,
                    }
                };
                increment_metric(metrics, "fast_actions_router_decisions", 1);
                let decision = match decision {
                    Ok(decision) => decision,
                    Err(error) => {
                        status = "unavailable";
                        reason = Some(error.to_string());
                        break;
                    }
                };
                let selected = decision
                    .candidate_id
                    .as_deref()
                    .and_then(|id| candidates.iter().find(|candidate| candidate.id == id));
                let single_operation = candidates
                    .iter()
                    .map(|candidate| candidate.tool.as_str())
                    .collect::<HashSet<_>>()
                    .len()
                    == 1;
                // Router and grounding must agree: a confident pick that the
                // local label evidence clearly contradicts is handed back.
                let contradicted = selected.is_some_and(|selected| {
                    best_local >= 0.8 && selected.local_score < best_local * 0.5
                });
                let eligible = selected.is_some()
                    && decision.model == config.active_model()
                    && !contradicted
                    && fast_target_eligible(
                        &decision,
                        &config,
                        single_operation,
                        selected.is_some_and(|candidate| is_read_only_fast_tool(&candidate.tool)),
                    );
                self.log(
                    "fast_actions_pick",
                    json!({
                        "turn": turn,
                        "candidate_count": candidates.len(),
                        "candidate_id": decision.candidate_id,
                        "tool": selected.map(|candidate| &candidate.tool),
                        "description": selected.map(|candidate| crate::decision::bounded_text(&candidate.description, 200)),
                        "operation_probability": decision.operation_probability,
                        "target_probability": decision.target_probability,
                        "target_confidence": decision.target_confidence,
                        "single_operation": single_operation,
                        "contradicted_by_local_evidence": contradicted,
                        "eligible": eligible,
                        "top_candidates": candidates
                            .iter()
                            .take(10)
                            .map(|candidate| json!({
                                "id": candidate.id,
                                "description": crate::decision::bounded_text(&candidate.description, 80),
                                "local_score": candidate.local_score,
                                "probability": decision.probabilities.get(&candidate.id),
                            }))
                            .collect::<Vec<_>>(),
                        "elapsed_ms": pick_started.elapsed().as_millis(),
                    }),
                )?;
                // The pick disagreed with clear label evidence: the grounded
                // best candidate is the answer the router should have given.
                if contradicted
                    && let Some(recorder) = &self.training
                    && let Some(best) = candidates
                        .iter()
                        .max_by(|left, right| left.local_score.total_cmp(&right.local_score))
                {
                    let operation = crate::decision::candidate_operation(best);
                    recorder.label("operation", &operation, "local_evidence");
                    recorder.label(&format!("target_{operation}"), &best.id, "local_evidence");
                }
                let mut router_alternatives = decision
                    .probabilities
                    .iter()
                    .map(|(id, score)| (id.clone(), *score))
                    .collect::<Vec<_>>();
                router_alternatives.sort_by(|left, right| right.1.total_cmp(&left.1));
                router_alternatives.truncate(3);
                self.emit(AgentEvent::DecisionRouterEvaluated {
                    turn,
                    purpose: "fast_actions".into(),
                    candidate_count: candidates.len(),
                    candidate_id: decision.candidate_id.clone(),
                    tool: selected.map(|candidate| candidate.tool.clone()),
                    description: selected.map(|candidate| {
                        crate::decision::bounded_text(&candidate.description, 200)
                    }),
                    selected_probability: decision.selected_probability,
                    confidence: decision.confidence,
                    operation_probability: decision.operation_probability,
                    target_probability: decision.target_probability,
                    operation_confidence: decision.operation_confidence,
                    target_confidence: decision.target_confidence,
                    eligible,
                    rejection_reason: (!eligible).then(|| {
                        if selected.is_none() {
                            "no_candidate"
                        } else if decision.model != config.active_model() {
                            "model_mismatch"
                        } else if contradicted {
                            "contradicted_by_local_evidence"
                        } else {
                            "below_threshold"
                        }
                        .to_owned()
                    }),
                    probability_threshold: config.min_selected_probability,
                    confidence_threshold: None,
                    alternatives: router_alternatives,
                    elapsed_ms: pick_started.elapsed().as_millis() as u64,
                });
                // With a judge configured it reviews every router pick, not
                // only uncertain ones: the bench found local pickers
                // confidently wrong, so the judge acts as a veto as well as a
                // rescue.
                let promoted = if !contradicted
                    && config.judge.enabled
                    && let Some(selected) = selected
                {
                    increment_metric(metrics, "fast_actions_judge_calls", 1);
                    let verdict = tokio::select! {
                        () = self.context.cancellation.cancelled() => return Err(PokError::Cancelled),
                        verdict = router.judge_candidate(&goal, &hint, selected, &candidates, &decision_state) => verdict,
                    };
                    let promoted = verdict.as_ref().is_ok_and(|verdict| verdict.promoted);
                    self.log(
                        "fast_actions_judge",
                        json!({
                            "turn": turn,
                            "candidate_id": selected.id,
                            "promoted": promoted,
                            "votes": verdict.as_ref().ok().map(|verdict| verdict.votes.clone()),
                            "error": verdict.as_ref().err().map(ToString::to_string),
                        }),
                    )?;
                    promoted.then(|| selected.clone())
                } else {
                    None
                };
                match (eligible && !config.judge.enabled, selected, promoted) {
                    (true, Some(selected), _) => selected.clone(),
                    (_, _, Some(promoted)) => {
                        increment_metric(metrics, "fast_actions_judge_promotions", 1);
                        promoted
                    }
                    // Only scrolls are offered and the model is unsure of the
                    // direction: continue down, which is read-only.
                    _ if scroll_search && default_scroll_down(&candidates).is_some() => {
                        default_scroll_down(&candidates)
                            .expect("checked above")
                            .clone()
                    }
                    _ => {
                        status = "uncertain";
                        reason = Some(
                            "the fast model could not confidently match a target; choose one yourself"
                                .into(),
                        );
                        let mut ranked = decision
                            .probabilities
                            .iter()
                            .filter_map(|(id, score)| {
                                candidates
                                    .iter()
                                    .find(|candidate| &candidate.id == id)
                                    .map(|candidate| (candidate, *score))
                            })
                            .collect::<Vec<_>>();
                        ranked.sort_by(|left, right| right.1.total_cmp(&left.1));
                        alternatives = ranked
                            .into_iter()
                            .take(3)
                            .map(|(candidate, score)| {
                                json!({
                                    "tool": candidate.tool,
                                    "arguments": candidate.arguments,
                                    "description": crate::decision::bounded_text(&candidate.description, 160),
                                    "probability": score,
                                })
                            })
                            .collect();
                        break;
                    }
                }
            };

            let fingerprint = decision_candidate_fingerprint(&candidate);
            attempted.insert(fingerprint.clone());
            let call = CompletedToolCall {
                id: format!("fast-actions-{}", Uuid::new_v4()),
                name: candidate.tool.clone(),
                arguments: candidate.arguments.clone(),
            };
            let (signature, prior_attempts) = match self.continuity.before_call(
                &call.name,
                &call.arguments,
                observation.as_ref(),
            ) {
                PreCallDecision::Execute {
                    signature,
                    prior_attempts,
                } => (signature, prior_attempts),
                PreCallDecision::Suppress { reason: why, .. } => {
                    self.log(
                        "fast_actions_step_suppressed",
                        json!({"turn": turn, "tool": call.name, "reason": why}),
                    )?;
                    if regrounded {
                        status = "stalled";
                        reason = Some(crate::decision::bounded_text(&why, 300));
                        break;
                    }
                    // The repeat-input guard asks for a fresh look before
                    // more input: System 1 takes one, then decides again
                    // from the new screen.
                    regrounded = true;
                    attempted.remove(&fingerprint);
                    let look = CompletedToolCall {
                        id: format!("fast-actions-{}", Uuid::new_v4()),
                        name: "capture_screen".into(),
                        arguments: json!({}),
                    };
                    if let PreCallDecision::Execute {
                        signature,
                        prior_attempts,
                    } = self.continuity.before_call(
                        &look.name,
                        &look.arguments,
                        observation.as_ref(),
                    ) {
                        let captured = tokio::select! {
                            () = self.context.cancellation.cancelled() => return Err(PokError::Cancelled),
                            captured = self.tools.call(&look.name, look.arguments.clone(), &self.context) => captured,
                        };
                        self.continuity.after_call(
                            &look.name,
                            &look.arguments,
                            &signature,
                            prior_attempts,
                            &captured,
                        );
                        increment_metric(metrics, "fast_actions_regrounds", 1);
                    }
                    continue;
                }
            };
            self.emit(AgentEvent::ToolStarted {
                call_id: call.id.clone(),
                name: call.name.clone(),
                arguments: call.arguments.clone(),
            });
            let result = tokio::select! {
                () = self.context.cancellation.cancelled() => return Err(PokError::Cancelled),
                result = self.tools.call(&call.name, call.arguments.clone(), &self.context) => result,
            };
            self.record_motor(&call.name, &call.arguments, &result);
            self.show_latest_observation();
            let feedback = self.continuity.after_call(
                &call.name,
                &call.arguments,
                &signature,
                prior_attempts,
                &result,
            );
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
            }
            // A window activation changes the authoritative surface; capture it
            // so the next pick and completion check see its targets.
            if call.name == "activate_window"
                && ok
                && let Some(window_id) = result
                    .as_ref()
                    .ok()
                    .and_then(|value| value.get("id"))
                    .and_then(Value::as_str)
            {
                let capture = tokio::select! {
                    () = self.context.cancellation.cancelled() => return Err(PokError::Cancelled),
                    capture = self.tools.call(
                        "capture_screen",
                        json!({
                            "scope": "window",
                            "window_id": window_id,
                            "include_ocr": true,
                            "include_ui_tree": true,
                            "enrichment": "fast"
                        }),
                        &self.context,
                    ) => capture,
                };
                if let Ok(value) = &capture
                    && qualifies_as_fresh_evidence("capture_screen", value, true)
                {
                    self.decision_router_fresh_evidence = true;
                }
                windows = None;
            }
            if picked_named_target {
                if ok && feedback.outcome != "failed" {
                    named_target_clicked = true;
                    named_target_was_window = call.name == "activate_window";
                } else {
                    // The planner's exact target did not take the click; say
                    // so instead of letting the run drift to other targets.
                    status = "stalled";
                    reason = Some(format!(
                        "the named target could not be clicked: {}",
                        result
                            .as_ref()
                            .err()
                            .map(|error| crate::decision::bounded_text(&error.to_string(), 160))
                            .unwrap_or_else(|| feedback.outcome.clone())
                    ));
                }
            }
            increment_metric(metrics, "fast_actions_steps", 1);
            metrics.tool_calls = metrics.tool_calls.saturating_add(1);
            steps.push(json!({
                "tool": call.name,
                "target": crate::decision::bounded_text(&candidate.description, 160),
                "outcome": feedback.outcome,
                "ok": ok,
                "error": result.as_ref().err().map(|error| crate::decision::bounded_text(&error.to_string(), 200)),
            }));
            self.log(
                "fast_actions_step",
                json!({
                    "turn": turn,
                    "tool": call.name,
                    "arguments": call.arguments,
                    "ok": ok,
                    "input_method": result.as_ref().ok().and_then(|value| value.get("input_method")),
                    "outcome": feedback.outcome,
                    "elapsed_ms": pick_started.elapsed().as_millis(),
                }),
            )?;
            if status == "stalled" {
                break;
            }
            if ok && matches!(feedback.outcome.as_str(), "progress" | "observed") {
                consecutive_setbacks = 0;
            } else {
                consecutive_setbacks = consecutive_setbacks.saturating_add(1);
                if consecutive_setbacks >= 2 {
                    status = "stalled";
                    reason = Some("two consecutive actions failed or made no progress".into());
                    break;
                }
            }
        }

        increment_metric(metrics, &format!("fast_actions_{status}"), 1);
        self.log(
            "fast_actions_finished",
            json!({"turn": turn, "status": status, "steps": steps.len(), "reason": reason}),
        )?;
        let mut result = json!({
            "status": status,
            "goal": goal,
            "done_when": done_when,
            "steps": steps,
            "completion": completion.map(|verdict| json!({
                "answer": verdict.answer,
                "probability": verdict.probability,
            })),
        });
        if let Some(reason) = reason {
            result["reason"] = json!(reason);
        }
        if !alternatives.is_empty() {
            result["alternatives"] = json!(alternatives);
        }
        if let Some(popup) = popup {
            result["popup"] = popup;
        }
        Ok(result)
    }
}
