//! Agent events and the observer interface the dashboard and CLI subscribe to.

use super::*;

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    RunStarted {
        session_id: Uuid,
        model: String,
    },
    TurnStarted {
        turn: u32,
    },
    ModelRequestDispatched {
        turn: u32,
        estimated_prompt_tokens: u64,
        tool_count: usize,
    },
    ProviderFirstEvent {
        turn: u32,
        elapsed_ms: u64,
        event_kind: String,
    },
    DecisionRouterStarted {
        turn: u32,
        purpose: String,
        candidate_count: usize,
    },
    DecisionRouterCacheHit {
        turn: u32,
        purpose: String,
        candidate_count: usize,
    },
    DecisionRouterEvaluated {
        turn: u32,
        purpose: String,
        candidate_count: usize,
        candidate_id: Option<String>,
        tool: Option<String>,
        description: Option<String>,
        selected_probability: f64,
        confidence: Option<f64>,
        operation_probability: Option<f64>,
        target_probability: Option<f64>,
        operation_confidence: Option<f64>,
        target_confidence: Option<f64>,
        eligible: bool,
        rejection_reason: Option<String>,
        probability_threshold: f64,
        confidence_threshold: Option<f64>,
        alternatives: Vec<(String, f64)>,
        elapsed_ms: u64,
    },
    DecisionRouterJudgeAttempted {
        turn: u32,
        tool: String,
        candidate_id: String,
        reason: Option<String>,
        promoted: bool,
        reused: bool,
        votes: std::collections::BTreeMap<String, String>,
    },
    DecisionRouterActionOutcome {
        turn: u32,
        tool: String,
        outcome: String,
        ok: bool,
    },
    DecisionRouterMemoryOutcome {
        session_id: Uuid,
        disposition: String,
        related_memory_id: Option<Uuid>,
        used_jev: bool,
    },
    EnvironmentChanged {
        revision: u64,
        previous_window: Option<String>,
        current_window: Option<String>,
    },
    EnvironmentWaiting {
        revision: u64,
        quiet_period_ms: u64,
    },
    /// Mouse or keyboard input is on hold because the user is using the
    /// machine (`waiting`), or has resumed after they stopped.
    UserActivityWait {
        waiting: bool,
    },
    EnvironmentRecovered {
        revision: u64,
        action: String,
        used_jev: bool,
        elapsed_ms: u64,
    },
    CompressionStarted {
        sequence: usize,
        trigger: String,
        transcript_chars: usize,
    },
    CompressionCompleted {
        sequence: usize,
        elapsed_ms: u64,
        summary_source: String,
        attempts: u32,
        prompt_tokens: u64,
        completion_tokens: u64,
        reasoning_chars: usize,
    },
    CompressionFailed {
        sequence: usize,
        elapsed_ms: u64,
        error: String,
    },
    ReasoningDelta {
        text: String,
    },
    TextDelta {
        text: String,
    },
    ToolStarted {
        call_id: String,
        name: String,
        arguments: Value,
    },
    ToolDelayed {
        call_id: String,
        name: String,
        elapsed_ms: u64,
        stage: String,
    },
    ToolFinished {
        call_id: String,
        name: String,
        ok: bool,
        detail: String,
        result: Option<Value>,
    },
    CommandProgress {
        event: crate::commands::CommandEvent,
    },
    Usage {
        prompt_tokens: u64,
        completion_tokens: u64,
        cumulative_prompt_tokens: u64,
        cumulative_completion_tokens: u64,
    },
    ModelFinished {
        reason: Option<String>,
    },
    ModelOutputLimitReached {
        attempt: u32,
        next_max_tokens: u32,
    },
    PauseRequested,
    Paused {
        interrupted_phase: String,
    },
    Resumed {
        paused_ms: u64,
        note: Option<String>,
        previous_foreground: Option<String>,
        current_foreground: Option<String>,
    },
    ModelTurnInterrupted {
        reason: String,
    },
    RunCompleted {
        answer: String,
        completion_status: CompletionStatus,
        warnings: Vec<CompletionWarning>,
        deliverables: Vec<ArtifactReference>,
    },
    RunFailed {
        error: String,
    },
    GuidanceInjected {
        text: String,
    },
    TaskStateChanged {
        root_request: String,
        status: TaskItemStatus,
        current_step: String,
        steps: Vec<TaskItem>,
    },
    ActionBlocked {
        tool: String,
        reason: String,
    },
    ContextStatus {
        budget: crate::context::ContextBudget,
        prompt_tokens: u64,
        conversation_tokens: u64,
        working_set_target_tokens: u64,
        remaining_tokens: u64,
        archived_entries: u64,
        archived_tokens: u64,
        approximate_turns_remaining: Option<u64>,
        compactions: usize,
    },
    ProviderStallRetry {
        attempt: u32,
        maximum: u32,
    },
    ProviderRetry {
        attempt: u32,
        maximum: Option<u32>,
        delay_ms: u64,
        error: String,
    },
    CompletionCandidateRejected {
        attempt: u32,
        maximum: u32,
        reason: String,
    },
    ClarificationRequested {
        root_request: String,
        question: String,
    },
}

pub trait SessionObserver: Send + Sync {
    fn emit(&self, event: AgentEvent);
}

pub(super) struct ObserverCommandSink {
    pub(super) observer: Arc<dyn SessionObserver>,
}

impl crate::commands::CommandEventSink for ObserverCommandSink {
    fn emit(&self, event: crate::commands::CommandEvent) {
        self.observer.emit(AgentEvent::CommandProgress { event });
    }
}
