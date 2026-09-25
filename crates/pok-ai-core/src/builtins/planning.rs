//! Planning and conversation tools: tool discovery, the task plan, the clock, and questions to the user.

use super::*;

#[derive(Debug, Deserialize, JsonSchema)]
pub(super) struct DiscoverToolsArgs {
    /// Capability families to enable for subsequent turns. Supported values:
    /// desktop, system, coding, memory, archive, generated, subagent, other.
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

    async fn execute(&self, args: Value, context: &ToolContext) -> Result<Value> {
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
