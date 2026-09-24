//! Offline benchmark of a decision-router backend on the question shapes the
//! delegated `fast_actions` loop actually asks: target picks, completion
//! checks, and condition choices (branches and interrupts).
//!
//! Every case goes through the real `DecisionRouter` methods, so payloads are
//! byte-for-byte what production sends; earlier Python benches drifted from
//! production formatting and changed model answers. Each case is also scored
//! against local grounding alone (quoted labels and hint ranking), which shows
//! whether a model adds anything beyond the free local checks.

use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{
    Result,
    config::DecisionRouterConfig,
    decision::{
        DecisionCandidate, DecisionPurpose, DecisionRequest, DecisionRouter, exact_quoted_match,
        grounded_condition, hint_relevance,
    },
};

/// Same bound as the delegated `fast_actions` loop.
const MAX_OFFERED_OPTIONS: usize = 16;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RouterSuite {
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub description: String,
    pub cases: Vec<RouterCase>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RouterCase {
    /// Pick one candidate for the planner's goal and target hint.
    Target {
        id: String,
        goal: String,
        #[serde(default)]
        hint: String,
        candidates: Vec<DecisionCandidate>,
        #[serde(default)]
        state: Value,
        expected: String,
    },
    /// Is `condition` already true in `state`? `expected` is "yes" or "no".
    Completion {
        id: String,
        goal: String,
        condition: String,
        #[serde(default)]
        state: Value,
        expected: String,
    },
    /// Which of the ordered `options` holds; `expected` is an option id.
    ConditionChoice {
        id: String,
        goal: String,
        options: Vec<(String, String)>,
        #[serde(default)]
        state: Value,
        expected: String,
    },
}

impl RouterCase {
    fn id(&self) -> &str {
        match self {
            Self::Target { id, .. }
            | Self::Completion { id, .. }
            | Self::ConditionChoice { id, .. } => id,
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            Self::Target { .. } => "target",
            Self::Completion { .. } => "completion",
            Self::ConditionChoice { .. } => "condition_choice",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CaseResult {
    pub id: String,
    pub kind: String,
    pub expected: String,
    pub answer: Option<String>,
    pub probability: Option<f64>,
    pub correct: bool,
    /// The answer cleared the production gate and would be acted on.
    pub accepted: bool,
    /// What local grounding alone decides, when it can decide.
    pub grounding_answer: Option<String>,
    pub latency_ms: u64,
    pub error: Option<String>,
}

/// All visible strings of a bench state, normalized the same way production
/// builds condition evidence.
fn state_evidence(state: &Value) -> String {
    fn collect(value: &Value, output: &mut Vec<String>) {
        match value {
            Value::String(text) => output.push(text.clone()),
            Value::Array(values) => values.iter().for_each(|value| collect(value, output)),
            Value::Object(fields) => fields.values().for_each(|value| collect(value, output)),
            _ => {}
        }
    }
    let mut parts = Vec::new();
    collect(state, &mut parts);
    crate::decision::normalized_evidence(&parts.join(" \n "))
}

fn target_accepted(
    decision: &crate::decision::DecisionResult,
    config: &DecisionRouterConfig,
) -> bool {
    if config.native_choice_probabilities() {
        decision
            .target_probability
            .is_some_and(|score| score >= config.laya.min_target_probability)
    } else {
        decision
            .target_confidence
            .is_some_and(|score| score >= config.min_confidence)
    }
}

pub async fn run_case(
    router: &dyn DecisionRouter,
    config: &DecisionRouterConfig,
    case: &RouterCase,
) -> CaseResult {
    let started = Instant::now();
    let mut result = CaseResult {
        id: case.id().to_owned(),
        kind: case.kind().to_owned(),
        expected: String::new(),
        answer: None,
        probability: None,
        correct: false,
        accepted: false,
        grounding_answer: None,
        latency_ms: 0,
        error: None,
    };
    match case {
        RouterCase::Target {
            goal,
            hint,
            candidates,
            state,
            expected,
            ..
        } => {
            result.expected = expected.clone();
            let focus = if hint.is_empty() { goal } else { hint };
            result.grounding_answer = exact_quoted_match(candidates, &format!("{hint} {goal}"))
                .or_else(|| {
                    candidates.iter().max_by(|left, right| {
                        hint_relevance(focus, left).total_cmp(&hint_relevance(focus, right))
                    })
                })
                .map(|candidate| candidate.id.clone());
            // Mirror production: the model sees at most the 16 candidates
            // local grounding ranks highest for the hint.
            let mut offered = candidates.clone();
            offered.sort_by(|left, right| {
                hint_relevance(focus, right).total_cmp(&hint_relevance(focus, left))
            });
            offered.truncate(MAX_OFFERED_OPTIONS);
            match router
                .decide(DecisionRequest {
                    purpose: DecisionPurpose::NextAction,
                    task: goal.clone(),
                    current_step: hint.clone(),
                    candidates: offered,
                    state: state.clone(),
                })
                .await
            {
                Ok(decision) => {
                    result.accepted = target_accepted(&decision, config);
                    result.probability = decision.target_probability.or(decision.target_confidence);
                    result.answer = decision.candidate_id;
                }
                Err(error) => result.error = Some(error.to_string()),
            }
        }
        RouterCase::Completion {
            goal,
            condition,
            state,
            expected,
            ..
        } => {
            result.expected = expected.clone();
            result.grounding_answer = grounded_condition(condition, &state_evidence(state))
                .map(|satisfied| if satisfied { "yes" } else { "no" }.to_owned());
            match router.check_condition(goal, condition, state).await {
                Ok(verdict) => {
                    // The gate only ever acts on a confident "yes"; a "no" or
                    // "unknown" keeps working, which is the safe default.
                    result.accepted = verdict.satisfied;
                    result.probability = Some(verdict.probability);
                    result.answer = Some(verdict.answer);
                }
                Err(error) => result.error = Some(error.to_string()),
            }
        }
        RouterCase::ConditionChoice {
            goal,
            options,
            state,
            expected,
            ..
        } => {
            result.expected = expected.clone();
            let evidence = state_evidence(state);
            result.grounding_answer = options
                .iter()
                .find(|(_, text)| grounded_condition(text, &evidence) == Some(true))
                .map(|(id, _)| id.clone());
            match router.choose_condition(goal, options, state).await {
                Ok(choice) => {
                    result.accepted = choice.accepted;
                    result.probability = Some(choice.probability);
                    result.answer = Some(choice.option_id);
                }
                Err(error) => result.error = Some(error.to_string()),
            }
        }
    }
    result.correct = result.answer.as_deref() == Some(result.expected.as_str());
    result.latency_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
    result
}

/// Latency and agreement of asking a target pick and a completion check as
/// two requests versus one batched request, on one target case. The check is
/// "is the goal already complete?" against the pre-action state, so its
/// correct answer is "no".
#[derive(Debug, Clone, Serialize)]
pub struct BatchComparison {
    pub id: String,
    pub pick_ms: u64,
    pub check_ms: u64,
    pub batched_ms: u64,
    pub same_pick: bool,
    pub same_verdict: bool,
    pub batched_verdict_correct: bool,
    /// Speculative fan-out: an interrupt-style condition choice asked on
    /// its own, then all three questions in one request.
    pub condition_ms: u64,
    pub fanout_ms: u64,
    pub fanout_same_pick: bool,
    pub fanout_same_verdict: bool,
    pub fanout_same_condition: bool,
    pub fanout_condition_correct: bool,
    pub error: Option<String>,
}

/// Interrupt-style rules for the fan-out comparison. None of them holds on
/// the suite's ordinary screens, so the correct answer is "none".
fn fanout_condition_options() -> Vec<(String, String)> {
    vec![
        (
            "c0".into(),
            "A sign-in or permission dialog is blocking the window".into(),
        ),
        (
            "c1".into(),
            "An error message dialog is covering the window".into(),
        ),
        (
            "none".into(),
            "None of the other conditions is true right now".into(),
        ),
    ]
}

pub async fn compare_batching(
    router: &dyn DecisionRouter,
    case: &RouterCase,
) -> Option<BatchComparison> {
    let RouterCase::Target {
        id,
        goal,
        hint,
        candidates,
        state,
        ..
    } = case
    else {
        return None;
    };
    let focus = if hint.is_empty() { goal } else { hint };
    let mut offered = candidates.clone();
    offered.sort_by(|left, right| {
        hint_relevance(focus, right).total_cmp(&hint_relevance(focus, left))
    });
    offered.truncate(MAX_OFFERED_OPTIONS);
    let request = DecisionRequest {
        purpose: DecisionPurpose::NextAction,
        task: goal.clone(),
        current_step: hint.clone(),
        candidates: offered,
        state: state.clone(),
    };
    let condition = format!("the goal is already complete: {goal}");
    let mut comparison = BatchComparison {
        id: id.clone(),
        pick_ms: 0,
        check_ms: 0,
        batched_ms: 0,
        same_pick: false,
        same_verdict: false,
        batched_verdict_correct: false,
        condition_ms: 0,
        fanout_ms: 0,
        fanout_same_pick: false,
        fanout_same_verdict: false,
        fanout_same_condition: false,
        fanout_condition_correct: false,
        error: None,
    };
    let options = fanout_condition_options();
    let elapsed =
        |started: Instant| u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
    let started = Instant::now();
    let pick = router.decide(request.clone()).await;
    comparison.pick_ms = elapsed(started);
    let started = Instant::now();
    let check = router.check_condition(goal, &condition, state).await;
    comparison.check_ms = elapsed(started);
    let started = Instant::now();
    let batched = router
        .decide_with_completion(request.clone(), goal, &condition, state)
        .await;
    comparison.batched_ms = elapsed(started);
    let started = Instant::now();
    let chosen = router.choose_condition(goal, &options, state).await;
    comparison.condition_ms = elapsed(started);
    let started = Instant::now();
    let fanout = router
        .decide_fanout(
            request,
            goal,
            Some((&condition, state)),
            Some((&options, state)),
        )
        .await;
    comparison.fanout_ms = elapsed(started);
    match (pick, check, batched, chosen, fanout) {
        (Ok(pick), Ok(check), Ok((batched_pick, batched_check)), Ok(chosen), Ok(fanout)) => {
            comparison.same_pick = pick.candidate_id == batched_pick.candidate_id;
            comparison.same_verdict = check.answer == batched_check.answer;
            comparison.batched_verdict_correct = batched_check.answer == "no";
            comparison.fanout_same_pick = pick.candidate_id == fanout.decision.candidate_id;
            comparison.fanout_same_verdict = fanout
                .completion
                .as_ref()
                .is_some_and(|verdict| verdict.answer == check.answer);
            comparison.fanout_same_condition = fanout
                .condition
                .as_ref()
                .is_some_and(|choice| choice.option_id == chosen.option_id);
            comparison.fanout_condition_correct = fanout
                .condition
                .as_ref()
                .is_some_and(|choice| choice.option_id == "none");
        }
        (pick, check, batched, chosen, fanout) => {
            comparison.error = [
                pick.err(),
                check.err(),
                batched.err(),
                chosen.err(),
                fanout.err(),
            ]
            .into_iter()
            .flatten()
            .next()
            .map(|error| error.to_string());
        }
    }
    Some(comparison)
}

pub async fn run_suite(
    router: &dyn DecisionRouter,
    config: &DecisionRouterConfig,
    suite: &RouterSuite,
) -> Vec<CaseResult> {
    let mut results = Vec::with_capacity(suite.cases.len());
    for case in &suite.cases {
        results.push(run_case(router, config, case).await);
    }
    results
}

/// Per-kind accuracy, gate outcomes, grounding comparison, latency, and a
/// calibration table (accuracy by reported probability bucket).
pub fn summarize(results: &[CaseResult]) -> Value {
    let mut kinds = std::collections::BTreeMap::<String, Vec<&CaseResult>>::new();
    for result in results {
        kinds.entry(result.kind.clone()).or_default().push(result);
    }
    let by_kind = kinds
        .iter()
        .map(|(kind, cases)| {
            let n = cases.len();
            let count = |predicate: fn(&CaseResult) -> bool| {
                cases.iter().filter(|case| predicate(case)).count()
            };
            let grounded = cases
                .iter()
                .filter(|case| case.grounding_answer.is_some())
                .collect::<Vec<_>>();
            let mut latencies = cases.iter().map(|case| case.latency_ms).collect::<Vec<_>>();
            latencies.sort_unstable();
            (
                kind.clone(),
                json!({
                    "cases": n,
                    "correct": count(|case| case.correct),
                    "accepted_correct": count(|case| case.accepted && case.correct),
                    "accepted_wrong": count(|case| case.accepted && !case.correct),
                    "deferred": count(|case| !case.accepted),
                    "errors": count(|case| case.error.is_some()),
                    "grounding_decided": grounded.len(),
                    "grounding_correct": grounded
                        .iter()
                        .filter(|case| case.grounding_answer.as_deref() == Some(case.expected.as_str()))
                        .count(),
                    "latency_ms_p50": latencies.get(n / 2).copied().unwrap_or(0),
                    "latency_ms_max": latencies.last().copied().unwrap_or(0),
                }),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    let buckets = [(0.0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.01)];
    let calibration = buckets
        .iter()
        .map(|(low, high)| {
            let cases = results
                .iter()
                .filter(|case| case.probability.is_some_and(|p| p >= *low && p < *high))
                .collect::<Vec<_>>();
            json!({
                "bucket": format!("{low:.2}-{:.2}", high.min(1.0)),
                "cases": cases.len(),
                "correct": cases.iter().filter(|case| case.correct).count(),
            })
        })
        .collect::<Vec<_>>();
    json!({"by_kind": by_kind, "calibration": calibration})
}

pub fn load_suite(path: &std::path::Path) -> Result<RouterSuite> {
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;

    use super::*;
    use crate::decision::{
        ConditionChoice, ConditionVerdict, DecisionCandidateKind, DecisionResult,
    };

    struct FixedRouter;

    #[async_trait]
    impl DecisionRouter for FixedRouter {
        async fn decide(&self, request: DecisionRequest) -> Result<DecisionResult> {
            Ok(DecisionResult {
                candidate_id: request
                    .candidates
                    .last()
                    .map(|candidate| candidate.id.clone()),
                selected_probability: 0.95,
                confidence: Some(0.95),
                operation: None,
                operation_probability: Some(1.0),
                target_probability: Some(0.95),
                operation_confidence: Some(1.0),
                target_confidence: Some(0.95),
                redacted_state_fields: 0,
                dropped_candidates: 0,
                model: "fixed".into(),
                probabilities: Default::default(),
                progress_probability: None,
                backend_metadata: None,
            })
        }

        async fn check_condition(&self, _: &str, _: &str, _: &Value) -> Result<ConditionVerdict> {
            Ok(ConditionVerdict {
                satisfied: true,
                answer: "yes".into(),
                probability: 0.9,
                model: "fixed".into(),
            })
        }

        async fn choose_condition(
            &self,
            _: &str,
            options: &[(String, String)],
            _: &Value,
        ) -> Result<ConditionChoice> {
            Ok(ConditionChoice {
                option_id: options[0].0.clone(),
                probability: 0.6,
                accepted: false,
                model: "fixed".into(),
            })
        }
    }

    fn candidate(id: &str, label: &str) -> DecisionCandidate {
        DecisionCandidate {
            id: id.into(),
            tool: "click_target".into(),
            arguments: json!({"expected_label": label}),
            description: format!("{label:?} (list item, enabled)"),
            kind: DecisionCandidateKind::Action,
            local_score: 0.0,
        }
    }

    #[tokio::test]
    async fn suite_scores_router_answers_gates_and_grounding() {
        let suite: RouterSuite = serde_json::from_value(json!({
            "cases": [
                {"kind": "target", "id": "t", "goal": "open System", "hint": "\"System\"",
                 "candidates": [candidate("a", "System"), candidate("b", "Display")],
                 "expected": "a"},
                {"kind": "completion", "id": "c", "goal": "open About",
                 "condition": "\"About\" is visible", "state": {"visible": ["Home", "System"]},
                 "expected": "no"},
                {"kind": "condition_choice", "id": "b", "goal": "handle popup",
                 "options": [["c0", "\"Sign in\" is visible"], ["none", "none"]],
                 "state": {"visible": ["Sign in"]}, "expected": "c0"},
            ]
        }))
        .unwrap();
        let config = DecisionRouterConfig {
            backend: crate::config::DecisionRouterBackend::Laya,
            ..Default::default()
        };
        let results = run_suite(&FixedRouter, &config, &suite).await;
        // The router picked the wrong target confidently; grounding got it.
        assert!(!results[0].correct && results[0].accepted);
        assert_eq!(results[0].grounding_answer.as_deref(), Some("a"));
        // A confident wrong "yes" is an accepted error; grounding says no.
        assert!(!results[1].correct && results[1].accepted);
        assert_eq!(results[1].grounding_answer.as_deref(), Some("no"));
        // Right answer below the gate is a deferral, not an acceptance.
        assert!(results[2].correct && !results[2].accepted);
        let summary = summarize(&results);
        assert_eq!(summary["by_kind"]["target"]["accepted_wrong"], 1);
        assert_eq!(summary["by_kind"]["condition_choice"]["deferred"], 1);
    }
}
