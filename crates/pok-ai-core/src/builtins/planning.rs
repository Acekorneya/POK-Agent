//! Planning and conversation tools: tool discovery, the task plan, the clock, and questions to the user.

use super::*;

#[derive(Debug, Deserialize, JsonSchema)]
pub(super) struct DiscoverToolsArgs {
    /// Capability families to enable for subsequent turns. Supported values:
    /// desktop, browser (the managed browser and the user's own browser),
    /// system, coding, memory, archive, generated, subagent, mcp, other.
    /// Group names, not tool names.
    pub(super) groups: Vec<String>,
}

pub(super) struct DiscoverToolsTool;

#[async_trait]
impl Tool for DiscoverToolsTool {
    fn name(&self) -> &'static str {
        "discover_tools"
    }

    fn description(&self) -> &'static str {
        "Enable additional tool-schema families when the current compact tool set lacks a required capability."
    }

    fn input_schema(&self) -> Value {
        schema::<DiscoverToolsArgs>()
    }

    fn risk(&self) -> RiskClass {
        RiskClass::ReadOnly
    }

    async fn execute(&self, args: Value, _context: &ToolContext) -> Result<Value> {
        let args: DiscoverToolsArgs = serde_json::from_value(args)?;
        Ok(json!({"enabled_for_next_turn": args.groups}))
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
pub(super) struct UpdateTaskPlanArgs {
    #[serde(default)]
    pub(super) status: Option<TaskItemStatus>,
    #[serde(default)]
    #[schemars(length(min = 1, max = 500))]
    pub(super) current_step: Option<String>,
    #[serde(default)]
    pub(super) steps: Vec<TaskItem>,
    #[serde(default)]
    #[schemars(skip)]
    pub(super) plan: Option<String>,
}

pub(super) struct UpdateTaskPlanTool;

/// Some small models send `steps` as a string that holds the JSON list
/// (`"steps": "[{\"id\": \"1\", ...}]"`) instead of the list itself. The
/// call then failed with a type error on every attempt (17 times in one
/// reported run); the meaning is unambiguous, so read it.
/// Read a plan the way small models write it. Besides a string holding the
/// list, they name the fields their own way: `{"step": 1, "action": "...",
/// "completed": true}` instead of `{"id", "content", "status"}` (seen in a
/// live Nemotron run, where every such call failed). Only the names are mapped;
/// what the plan says is untouched, and a correct plan passes through as it is.
pub(super) fn normalize_plan_arguments(args: &mut Value) {
    unwrap_stringified_steps(args);
    let Some(Value::Array(steps)) = args.get_mut("steps") else {
        return;
    };
    for step in steps {
        let Some(item) = step.as_object_mut() else {
            continue;
        };
        if !item.contains_key("id") {
            for alias in ["step", "index", "number", "n"] {
                if let Some(value) = item.remove(alias) {
                    item.insert("id".into(), value);
                    break;
                }
            }
        }
        if !item.contains_key("content") && !item.contains_key("description") {
            for alias in ["action", "title", "text", "task", "name", "summary"] {
                if let Some(value) = item.remove(alias) {
                    item.insert("content".into(), value);
                    break;
                }
            }
        }
        if !item.contains_key("status") {
            let done = ["completed", "done", "complete"]
                .iter()
                .find_map(|key| item.get(*key).and_then(Value::as_bool));
            if let Some(done) = done {
                item.insert(
                    "status".into(),
                    Value::String(if done { "completed" } else { "pending" }.into()),
                );
            }
        }
        for key in ["completed", "done", "complete"] {
            item.remove(key);
        }
    }
}

pub(super) fn unwrap_stringified_steps(args: &mut Value) {
    let Some(steps) = args.get_mut("steps") else {
        return;
    };
    let Some(text) = steps.as_str() else {
        return;
    };
    if let Ok(parsed @ Value::Array(_)) = serde_json::from_str::<Value>(text.trim()) {
        *steps = parsed;
    }
}

#[async_trait]
impl Tool for UpdateTaskPlanTool {
    fn name(&self) -> &'static str {
        "update_task_plan"
    }

    fn description(&self) -> &'static str {
        "Update the structured plan for the active user request. Use on tasks that take several tool calls, keep exactly one step in_progress while working, and mark the task completed only after tool evidence supports the requested outcome. Omit current_step to preserve or infer it; never send an empty string. To finish an existing plan, status=\"completed\" is sufficient and preserves its steps. For supplied steps use id, content, and status; description is also accepted as an alias for content."
    }

    fn input_schema(&self) -> Value {
        schema::<UpdateTaskPlanArgs>()
    }

    fn risk(&self) -> RiskClass {
        RiskClass::ReadOnly
    }

    async fn execute(&self, mut args: Value, context: &ToolContext) -> Result<Value> {
        normalize_plan_arguments(&mut args);
        let mut args: UpdateTaskPlanArgs = serde_json::from_value(args)?;
        hydrate_plan_update(&mut args, &context.active_task.lock());
        if args.steps.is_empty() {
            return Err(PokError::Tool(
                "task plan must contain at least one step".into(),
            ));
        }
        if args.steps.len() > 32 {
            return Err(PokError::Tool(
                "task plan cannot contain more than 32 steps".into(),
            ));
        }
        let status = resolved_plan_status(args.status, &args.steps);
        let supplied_current =
            resolved_current_step(status, args.current_step.as_deref(), &args.steps);
        let current_step = args
            .steps
            .iter()
            .find(|step| step.id == supplied_current)
            .map_or(supplied_current.as_str(), |step| step.content.trim());
        if current_step.is_empty() || current_step.chars().count() > 500 {
            return Err(PokError::Tool(
                "current_step must contain between 1 and 500 characters".into(),
            ));
        }
        let mut ids = HashSet::new();
        for item in &args.steps {
            if item.id.trim().is_empty()
                || item.id.chars().count() > 80
                || item.content.trim().is_empty()
                || item.content.chars().count() > 500
                || !ids.insert(item.id.trim().to_owned())
            {
                return Err(PokError::Tool(
                    "task steps require unique non-empty ids and content within size limits".into(),
                ));
            }
        }
        let in_progress = args
            .steps
            .iter()
            .filter(|item| item.status == TaskItemStatus::InProgress)
            .count();
        match status {
            TaskItemStatus::Completed
                if args
                    .steps
                    .iter()
                    .any(|item| item.status != TaskItemStatus::Completed) =>
            {
                return Err(PokError::Tool(
                    "all task steps must be completed before completing the task".into(),
                ));
            }
            TaskItemStatus::InProgress if in_progress != 1 => {
                return Err(PokError::Tool(
                    "an in-progress task plan must contain exactly one in_progress step".into(),
                ));
            }
            TaskItemStatus::Pending => {
                return Err(PokError::Tool(
                    "the active task must be in_progress or completed".into(),
                ));
            }
            _ => {}
        }

        let mut state = context.active_task.lock();
        let changed = task_plan_changed(&state, status, current_step, &args.steps);
        state.status = status;
        state.current_step = current_step.to_owned();
        state.steps = args.steps;
        state.plan_updated = true;
        let mut result = serde_json::to_value(&*state)?;
        result["plan_updated"] = json!(changed);
        if !changed {
            result["recovery_guidance"] = json!(
                "This plan is unchanged. Do not update it again now; execute a concrete tool, call discover_tools for an inactive capability, or answer the user."
            );
        }
        Ok(result)
    }
}

pub(super) fn task_plan_changed(
    state: &ActiveTaskState,
    status: TaskItemStatus,
    current_step: &str,
    steps: &[TaskItem],
) -> bool {
    state.status != status || state.current_step != current_step || state.steps != steps
}

pub(super) fn hydrate_plan_update(args: &mut UpdateTaskPlanArgs, existing: &ActiveTaskState) {
    if !args.steps.is_empty() {
        return;
    }
    if let Some(plan) = args.plan.as_deref() {
        args.steps =
            parse_structured_plan_string(plan).unwrap_or_else(|| parse_plan_shorthand(plan));
        if args.status == Some(TaskItemStatus::Completed) {
            for step in &mut args.steps {
                step.status = TaskItemStatus::Completed;
            }
            if args.current_step.as_deref().is_none_or(str::is_empty) {
                args.current_step = args.steps.last().map(|step| step.content.clone());
            }
        } else {
            if args.status.is_none() {
                args.status = Some(resolved_plan_status(None, &args.steps));
            }
            if args.current_step.as_deref().is_none_or(str::is_empty) {
                args.current_step = args
                    .steps
                    .iter()
                    .find(|step| step.status == TaskItemStatus::InProgress)
                    .or_else(|| args.steps.first())
                    .map(|step| step.content.clone());
            }
        }
    } else if args.status.is_some() {
        args.steps = existing.steps.clone();
        if args.status == Some(TaskItemStatus::Completed) {
            for step in &mut args.steps {
                step.status = TaskItemStatus::Completed;
            }
            args.current_step = args.steps.last().map(|step| step.content.clone());
        }
    }
}

pub(super) fn parse_structured_plan_string(plan: &str) -> Option<Vec<TaskItem>> {
    let value: Value = serde_json::from_str(plan.trim()).ok()?;
    let steps = match value {
        Value::Array(steps) => steps,
        Value::Object(mut object) => object.remove("steps")?.as_array()?.clone(),
        _ => return None,
    };
    let steps = steps
        .into_iter()
        .map(serde_json::from_value::<TaskItem>)
        .collect::<std::result::Result<Vec<_>, _>>()
        .ok()?;
    (!steps.is_empty() && steps.len() <= 32).then_some(steps)
}

pub(super) fn resolved_current_step(
    status: TaskItemStatus,
    supplied: Option<&str>,
    steps: &[TaskItem],
) -> String {
    let inferred = match status {
        TaskItemStatus::InProgress => steps
            .iter()
            .find(|step| step.status == TaskItemStatus::InProgress),
        TaskItemStatus::Completed => steps.last(),
        TaskItemStatus::Pending => None,
    };
    supplied
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .or_else(|| inferred.map(|step| step.content.trim()))
        .unwrap_or_default()
        .to_owned()
}

pub(super) fn resolved_plan_status(
    supplied: Option<TaskItemStatus>,
    steps: &[TaskItem],
) -> TaskItemStatus {
    supplied.unwrap_or_else(|| {
        if steps
            .iter()
            .all(|step| step.status == TaskItemStatus::Completed)
        {
            TaskItemStatus::Completed
        } else if steps
            .iter()
            .any(|step| step.status == TaskItemStatus::InProgress)
        {
            TaskItemStatus::InProgress
        } else {
            TaskItemStatus::Pending
        }
    })
}

pub(super) fn parse_plan_shorthand(plan: &str) -> Vec<TaskItem> {
    plan.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(|line| line.trim_start_matches(['-', '*', '•', ' ']))
        .map(|line| {
            let without_number = line
                .split_once('.')
                .and_then(|(prefix, rest)| {
                    prefix
                        .chars()
                        .all(|c| c.is_ascii_digit())
                        .then_some(rest.trim())
                })
                .unwrap_or(line);
            without_number.to_owned()
        })
        .filter(|line| !line.is_empty())
        .take(32)
        .enumerate()
        .map(|(index, content)| TaskItem {
            id: (index + 1).to_string(),
            content,
            status: if index == 0 {
                TaskItemStatus::InProgress
            } else {
                TaskItemStatus::Pending
            },
        })
        .collect()
}

pub(super) struct CurrentTimeTool;
#[async_trait]
impl Tool for CurrentTimeTool {
    fn name(&self) -> &'static str {
        "get_current_time"
    }
    fn description(&self) -> &'static str {
        "Return the exact current Windows-local and UTC date/time. Use when a task depends on the current time, timezone, deadline, or relative date; the session prompt already contains today's date."
    }
    fn input_schema(&self) -> Value {
        json!({"type": "object", "properties": {}})
    }
    fn risk(&self) -> RiskClass {
        RiskClass::ReadOnly
    }
    async fn execute(&self, _args: Value, _context: &ToolContext) -> Result<Value> {
        let local = Local::now();
        Ok(json!({
            "local_iso8601": local.to_rfc3339(),
            "local_date": local.format("%Y-%m-%d").to_string(),
            "weekday": local.format("%A").to_string(),
            "timezone_abbreviation": local.format("%Z").to_string(),
            "utc_offset": local.format("%:z").to_string(),
            "utc_iso8601": Utc::now().to_rfc3339(),
        }))
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
pub(super) struct AskUserQuestionArgs {
    pub(super) questions: Vec<AskUserQuestionItem>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub(super) struct AskUserQuestionItem {
    #[serde(default)]
    #[schemars(skip)]
    pub(super) id: Option<String>,
    #[serde(default)]
    pub(super) header: String,
    pub(super) question: String,
    #[serde(default)]
    pub(super) options: Vec<AskUserQuestionOption>,
    #[serde(default, alias = "multiSelect")]
    pub(super) multi_select: bool,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub(super) struct AskUserQuestionOption {
    pub(super) label: String,
    #[serde(default)]
    pub(super) description: String,
}

pub(super) struct AskUserQuestionTool;

#[async_trait]
impl Tool for AskUserQuestionTool {
    fn name(&self) -> &'static str {
        "ask_user_question"
    }
    fn description(&self) -> &'static str {
        "Ask the user when missing information, ambiguity, or a preference materially changes the next action. Group all currently known related questions into one call; use additional rounds if answers reveal new uncertainty. Do not ask for information already available from context or tools, and do not interrupt for trivial choices you can reasonably decide. Options are optional for open-ended questions. When choices are useful, keep them concise, put the recommended option first, and suffix its label with '(Recommended)'. The UI always provides a custom Other answer, so never add an Other option yourself. After the user answers, continue the task and do not repeat answered questions."
    }
    fn input_schema(&self) -> Value {
        schema::<AskUserQuestionArgs>()
    }
    fn risk(&self) -> RiskClass {
        RiskClass::ReadOnly
    }
    async fn execute(&self, args: Value, context: &ToolContext) -> Result<Value> {
        let args: AskUserQuestionArgs = serde_json::from_value(args)?;
        if args.questions.is_empty() {
            return Err(PokError::Tool(
                "ask_user_question requires at least one question".into(),
            ));
        }
        let handler = context.questions.as_ref().ok_or_else(|| {
            PokError::Tool("interactive questions are unavailable in this session; ask the user in your response instead".into())
        })?;
        let questions = prepare_user_questions(args.questions)?;
        match handler.ask(questions.clone()).await? {
            UserQuestionOutcome::Answered(answers) => {
                let question_text = questions
                    .iter()
                    .map(|question| (question.id.as_str(), question.question.as_str()))
                    .collect::<HashMap<_, _>>();
                let answers = answers
                    .into_iter()
                    .map(|answer| {
                        let question = question_text.get(answer.id.as_str()).copied();
                        json!({
                            "id": answer.id,
                            "question": question,
                            "selected": answer.selected,
                            "custom": answer.custom,
                        })
                    })
                    .collect::<Vec<_>>();
                Ok(json!({"status": "answered", "answers": answers}))
            }
            UserQuestionOutcome::Dismissed => Ok(json!({
                "status": "dismissed",
                "answers": [],
                "instruction": "The user chose not to answer. Continue using available context and best judgment; do not repeat the same questions."
            })),
        }
    }
}

pub(super) fn prepare_user_questions(items: Vec<AskUserQuestionItem>) -> Result<Vec<UserQuestion>> {
    let mut ids = HashSet::new();
    items
        .into_iter()
        .enumerate()
        .map(|(index, item)| {
            let question = item.question.trim().to_owned();
            if question.is_empty() {
                return Err(PokError::Tool(format!(
                    "question {} must contain question text",
                    index + 1
                )));
            }
            let preferred = item.id.unwrap_or_default().trim().to_owned();
            let base = if preferred.is_empty() {
                format!("question_{}", index + 1)
            } else {
                preferred
            };
            let mut id = base.clone();
            let mut suffix = 2;
            while !ids.insert(id.clone()) {
                id = format!("{base}_{suffix}");
                suffix += 1;
            }
            Ok(UserQuestion {
                id,
                header: item.header.trim().to_owned(),
                question,
                options: item
                    .options
                    .into_iter()
                    .filter_map(|option| {
                        let label = option.label.trim().to_owned();
                        (!label.is_empty()).then(|| UserQuestionOption {
                            label,
                            description: option.description.trim().to_owned(),
                        })
                    })
                    .collect(),
                multi_select: item.multi_select,
            })
        })
        .collect()
}

#[derive(Debug, Deserialize, JsonSchema)]
pub(super) struct StandingOrdersArgs {
    /// "add", "list", or "remove".
    pub(super) action: String,
    /// For add: the instruction exactly as the user means it to be kept up, in
    /// the user's terms ("answer every unread message in the chat app").
    #[serde(default)]
    pub(super) text: Option<String>,
    /// For add: how often, in minutes, the desktop app should re-run it (5 or more, default 10).
    #[serde(default)]
    pub(super) every_minutes: Option<u32>,
    /// For remove: the order's id.
    #[serde(default)]
    pub(super) id: Option<String>,
}

/// Words that show the user asked for something to be kept up, not done once.
const STANDING_REQUEST_WORDS: &[&str] = &[
    "always",
    "keep ",
    "monitor",
    "whenever",
    "every time",
    "from now on",
    "continuously",
    "stay ",
    "standing",
    "watch for",
    "all the time",
    "until i say",
];

pub(super) struct StandingOrdersTool;

#[async_trait]
impl Tool for StandingOrdersTool {
    fn name(&self) -> &'static str {
        "standing_orders"
    }

    fn description(&self) -> &'static str {
        "Keep an instruction alive across runs. When the user asks for something to be done always, continuously, or whenever it comes up (\"always answer unread messages\", \"stay in the voice channel\", \"monitor the DM\"), add it as a standing order: it is shown to you at the start of every run and the app re-runs it on a timer. Only add one the user asked for; remove one only when they cancel it. Actions: add, list, remove."
    }

    fn input_schema(&self) -> Value {
        schema::<StandingOrdersArgs>()
    }

    fn risk(&self) -> RiskClass {
        RiskClass::WorkspaceWrite
    }

    async fn execute(&self, args: Value, context: &ToolContext) -> Result<Value> {
        let args: StandingOrdersArgs = serde_json::from_value(args)?;
        let store = crate::standing_orders::StandingOrders::open(
            &context.data_dir.join("standing-orders.json"),
        );
        match args.action.as_str() {
            "list" => Ok(json!({"orders": store.list()})),
            "add" => {
                let request = context.active_task.lock().root_request.to_lowercase();
                if !STANDING_REQUEST_WORDS
                    .iter()
                    .any(|word| request.contains(word))
                {
                    return Err(PokError::Tool(
                        "the user did not ask for this to be kept up; do it now, or ask whether it should be a standing order".into(),
                    ));
                }
                let (order, created) =
                    store.add(args.text.as_deref().unwrap_or_default(), args.every_minutes)?;
                Ok(json!({"order": order, "created": created}))
            }
            "remove" => {
                let id = args.id.unwrap_or_default();
                Ok(json!({"removed": store.remove(&id)?, "id": id}))
            }
            other => Err(PokError::Tool(format!(
                "unknown standing_orders action {other:?}; use add, list, or remove"
            ))),
        }
    }
}
