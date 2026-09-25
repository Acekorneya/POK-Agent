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

pub(super) const FAST_MAX_NODES: usize = 16;
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
        ) -> Result<FastPlanNode> {
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
/// Accept plan shapes models commonly produce for nested steps, which the
/// schema normalizer does not reach inside the recursive `then`/`branches`:
/// lists wrapped as `{"item": ...}` (an XML-style serialization), and a step
/// that only reads values, without its own `goal` or `done_when`.
pub(super) fn repair_fast_plan(node: &mut Value, top_level: bool) {
    unwrap_item_lists(node);
    let Some(object) = node.as_object_mut() else {
        return;
    };
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

/// Replace every `{"item": x}` object with a list (`x` itself when it is a
/// list, otherwise `[x]`). No fast-actions field is named `item`.
pub(super) fn unwrap_item_lists(value: &mut Value) {
    match value {
        Value::Object(object) => {
            if object.len() == 1
                && let Some(item) = object.get_mut("item")
            {
                let item = item.take();
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
            .map_err(|error| PokError::Tool(format!("invalid fast_actions arguments: {error}")))?;
        let plan = FastPlan::from_args(&args)?;
        increment_metric(metrics, "fast_actions_runs", 1);
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
            let outcome = self
                .run_fast_subgoal(
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
                .await?;
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
        // evidence, so the final answer turn does not need a screenshot.
        let mut reads_complete = false;
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
            reads_complete = reads
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
        // Return the newest verified observation so the primary model can
        // continue from it (including its image) without another capture.
        if executed_steps > 0
            && !reads_complete
            && let Some(observation) = observation.as_ref()
            && let Ok(value) =
                crate::builtins::model_observation_value(observation, self.context.annotate_targets)
        {
            result["observation"] = value;
        } else if let Some(observation) = observation.as_ref() {
            result["observation_id"] = json!(crate::builtins::observation_id_for(observation));
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
        let hint = hint.to_owned();
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

        for step in 0..=max_steps {
            let mut deferred_completion = false;
            // An unquoted interrupt question waiting to ride along with this
            // step's target pick, with the rule index behind each option.
            let mut deferred_interrupt: Option<(PendingFastCondition, Vec<usize>)> = None;
            let observation = self.context.latest_observation.lock().clone();
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
                && (allowed.contains(&FastOperation::Click)
                    || allowed.contains(&FastOperation::DoubleClick))
                && let Some(observation) = observation.as_ref()
            {
                candidates.extend(crate::decision::exact_label_candidates(
                    observation,
                    &crate::decision::quoted_labels(&hint),
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
            } else if candidates.len() == 1 {
                settle_interrupt!();
                settle_completion!();
                increment_metric(metrics, "fast_actions_structural_bypass", 1);
                candidates[0].clone()
            } else if let Some(matched) =
                crate::decision::exact_quoted_match(&candidates, &format!("{hint} {goal}"))
                    .or_else(|| unambiguous_label_match(&candidates))
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
            } else if !config.model_targets_trusted() {
                settle_interrupt!();
                settle_completion!();
                // Grounding could not decide and this backend is not trusted
                // to pick targets: hand the choice back with the local ranking.
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
            let (signature, prior_attempts) =
                match self
                    .continuity
                    .before_call(&call.name, &call.arguments, observation.as_ref())
                {
                    PreCallDecision::Execute {
                        signature,
                        prior_attempts,
                    } => (signature, prior_attempts),
                    PreCallDecision::Suppress { reason: why, .. } => {
                        self.log(
                            "fast_actions_step_suppressed",
                            json!({"turn": turn, "tool": call.name, "reason": why}),
                        )?;
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
        Ok(result)
    }
}
