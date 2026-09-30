use std::{
    path::{Path, PathBuf},
    process::Command,
    sync::Arc,
};

use async_trait::async_trait;
use schemars::{JsonSchema, schema_for};
use serde::Deserialize;
use serde_json::{Value, json};
use uuid::Uuid;

use crate::{
    PokError, Result,
    brain::Brain,
    coding::register_coding_tools,
    policy::{AllowApprovals, Policy, PolicyMode, RiskClass},
    process_window,
    session::Session,
    tool::{Tool, ToolContext, ToolRegistry},
};

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SpawnSubagentArgs {
    pub subtask_title: String,
    pub instruction: String,
    pub target_directory: PathBuf,
    #[serde(default = "default_turns")]
    pub max_turns: u32,
}
const fn default_turns() -> u32 {
    12
}

pub struct SpawnSubagentTool {
    brain: Arc<dyn Brain>,
    model: String,
}

impl SpawnSubagentTool {
    pub fn new(brain: Arc<dyn Brain>, model: impl Into<String>) -> Self {
        Self {
            brain,
            model: model.into(),
        }
    }
}

#[async_trait]
impl Tool for SpawnSubagentTool {
    fn name(&self) -> &'static str {
        "spawn_subagent"
    }
    fn description(&self) -> &'static str {
        "Run a focused coding child in an isolated Git worktree; returns its commit, diff, tests, and artifacts without merging."
    }
    fn input_schema(&self) -> Value {
        serde_json::to_value(schema_for!(SpawnSubagentArgs)).expect("schema serializes")
    }
    fn risk(&self) -> RiskClass {
        RiskClass::WorkspaceWrite
    }
    fn target_path(&self, args: &Value) -> Option<PathBuf> {
        serde_json::from_value::<SpawnSubagentArgs>(args.clone())
            .ok()
            .map(|args| args.target_directory)
    }

    async fn execute(&self, args: Value, parent: &ToolContext) -> Result<Value> {
        let args: SpawnSubagentArgs = serde_json::from_value(args)?;
        let target = parent.resolve_workspace_path(args.target_directory)?;
        let repo = git_output(&target, ["rev-parse", "--show-toplevel"])?;
        let repo = PathBuf::from(repo.trim());
        if !git_output(&repo, ["status", "--porcelain"])?
            .trim()
            .is_empty()
        {
            return Err(PokError::Tool(
                "subagents require a clean Git worktree".into(),
            ));
        }
        let child_id = Uuid::new_v4();
        let branch = format!("pok-ai/{}", child_id.simple());
        let worktree = parent.data_dir.join("worktrees").join(child_id.to_string());
        std::fs::create_dir_all(worktree.parent().expect("worktree has parent"))?;
        git_status(
            Command::new("git").current_dir(&repo).args([
                "worktree",
                "add",
                "-b",
                &branch,
                &worktree.display().to_string(),
                "HEAD",
            ]),
            "create worktree",
        )?;

        let artifact_dir = parent
            .artifact_dir
            .join("subagents")
            .join(child_id.to_string());
        let mut registry = ToolRegistry::new();
        register_coding_tools(&mut registry);
        let context = ToolContext {
            session_id: child_id,
            workspace: worktree.clone(),
            data_dir: parent.data_dir.clone(),
            artifact_dir: artifact_dir.clone(),
            policy: Policy {
                mode: PolicyMode::Exam {
                    sandbox: worktree.clone(),
                    allowed_window_title: String::new(),
                    allowed_process: String::new(),
                },
            },
            approvals: Arc::new(AllowApprovals),
            platform: parent.platform.clone(),
            memory: parent.memory.clone(),
            session_archive: crate::session_archive::SessionArchive::open(&artifact_dir)?,
            cancellation: parent.cancellation.child_token(),
            pause: parent.pause.clone(),
            latest_observation: Default::default(),
            latest_observation_view: Default::default(),
            pending_visual_localization: Default::default(),
            task_hint: parent.task_hint.clone(),
            input_ledger: Default::default(),
            artifact_evidence: Default::default(),
            attached_paths: Default::default(),
            session_files: Default::default(),
            active_task: Default::default(),
            focused_control: Default::default(),
            command_timeout_seconds: parent.command_timeout_seconds,
            input_text_inter_key_pause_ms: parent.input_text_inter_key_pause_ms,
            vision_max_edge: parent.vision_max_edge,
            prompt_token_target: parent.prompt_token_target,
            context_budget: parent.context_budget.clone(),
            visual_history_limit: parent.visual_history_limit,
            uia_element_limit: parent.uia_element_limit,
            desktop_enrichment_timeout_ms: parent.desktop_enrichment_timeout_ms,
            desktop_deep_enrichment_timeout_ms: parent.desktop_deep_enrichment_timeout_ms,
            model_target_limit: parent.model_target_limit,
            fusion_iou_threshold: parent.fusion_iou_threshold,
            ocr_containment_threshold: parent.ocr_containment_threshold,
            annotate_targets: parent.annotate_targets,
            approval_cache: Default::default(),
            user_guidance_queue: parent.user_guidance_queue.clone(),
            questions: None,
            command_manager: Arc::new(crate::commands::CommandManager::new(artifact_dir.clone())),
            current_tool_call_id: Default::default(),
        };
        let mut session = Session::new(
            self.brain.clone(),
            Arc::new(registry),
            context,
            self.model.clone(),
            args.max_turns.min(30),
        )?;
        let run = session.run(format!(
            "You are a restricted coding subagent. Work only in this worktree. Task: {}\n{}\nRun relevant tests and report exactly what changed.",
            args.subtask_title, args.instruction
        )).await;
        let summary = run
            .as_ref()
            .map(|run| run.answer.clone())
            .unwrap_or_else(|error| format!("subagent failed: {error}"));
        let diff = git_output(&worktree, ["diff", "--", "."])?;
        if !diff.trim().is_empty() {
            git_status(
                Command::new("git")
                    .current_dir(&worktree)
                    .args(["add", "-A"]),
                "stage subagent work",
            )?;
            git_status(
                Command::new("git").current_dir(&worktree).args([
                    "-c",
                    "user.name=POK-Agent Subagent",
                    "-c",
                    "user.email=pok-ai@local",
                    "commit",
                    "-m",
                    &format!("POK-Agent subagent: {}", args.subtask_title),
                ]),
                "commit subagent work",
            )?;
        }
        let commit = git_output(&worktree, ["rev-parse", "HEAD"])?;
        Ok(json!({
            "subagent_id": child_id, "branch": branch, "commit": commit.trim(),
            "worktree": worktree, "summary": summary, "diff": diff,
            "artifacts": artifact_dir, "succeeded": run.is_ok(),
            "integration": "Review the diff, then cherry-pick the returned commit only after explicit approval."
        }))
    }
}

fn git_output<const N: usize>(cwd: &Path, args: [&str; N]) -> Result<String> {
    let mut command = Command::new("git");
    command.current_dir(cwd).args(args);
    process_window::hide_std(&mut command);
    let output = command.output()?;
    if !output.status.success() {
        return Err(PokError::Tool(format!(
            "git failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn git_status(command: &mut Command, operation: &str) -> Result<()> {
    process_window::hide_std(command);
    let output = command.output()?;
    if !output.status.success() {
        return Err(PokError::Tool(format!(
            "failed to {operation}: {}",
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    Ok(())
}
