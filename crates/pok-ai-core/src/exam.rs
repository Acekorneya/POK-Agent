use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use axum::{Json, Router, extract::State, routing::post};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::{
    PokError, Result,
    brain::Brain,
    memory::MemoryStore,
    platform::DesktopPlatform,
    policy::{AllowApprovals, Policy, PolicyMode},
    session::Session,
    tool::{ToolContext, ToolRegistry},
    types::{RunMetrics, RunSummary},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExamKind {
    Coding,
    Desktop,
    Memory,
    Orchestration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExamScenario {
    pub id: String,
    pub title: String,
    pub kind: ExamKind,
    pub prompt: String,
    pub workspace: PathBuf,
    #[serde(default = "default_weight")]
    pub weight: f64,
    #[serde(default = "default_turns")]
    pub max_turns: u32,
    #[serde(default)]
    pub allowed_window_title: String,
    #[serde(default)]
    pub allowed_process: String,
    #[serde(default)]
    pub assertions: Vec<ExamAssertion>,
}
const fn default_weight() -> f64 {
    1.0
}
const fn default_turns() -> u32 {
    20
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ExamAssertion {
    FileContains {
        path: PathBuf,
        text: String,
    },
    FileEquals {
        path: PathBuf,
        text: String,
    },
    FileExists {
        path: PathBuf,
    },
    JsonEquals {
        path: PathBuf,
        pointer: String,
        value: Value,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssertionResult {
    pub assertion: ExamAssertion,
    pub passed: bool,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioResult {
    pub scenario_id: String,
    pub model: String,
    pub score: f64,
    pub max_score: f64,
    pub assertions: Vec<AssertionResult>,
    pub metrics: RunMetrics,
    pub answer: String,
    pub artifacts: PathBuf,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelScore {
    pub model: String,
    pub earned: f64,
    pub possible: f64,
    pub pass_percent: f64,
    pub results: Vec<ScenarioResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExamReport {
    pub run_id: Uuid,
    pub scores: Vec<ModelScore>,
}

pub struct ExamRunner {
    pub brain: Arc<dyn Brain>,
    pub platform: Arc<dyn DesktopPlatform>,
    pub data_dir: PathBuf,
    pub memory: Arc<MemoryStore>,
    pub command_timeout_seconds: u64,
}

impl ExamRunner {
    pub async fn run<F>(
        &self,
        models: &[String],
        scenarios: &[ExamScenario],
        registry_factory: F,
    ) -> Result<ExamReport>
    where
        F: Fn(Arc<dyn Brain>, &str) -> ToolRegistry,
    {
        let run_id = Uuid::new_v4();
        let mut scores = Vec::new();
        for model in models {
            let mut results = Vec::new();
            for scenario in scenarios {
                let source_workspace = dunce::canonicalize(&scenario.workspace)?;
                let session_id = Uuid::new_v4();
                let scenario_root = self
                    .data_dir
                    .join("exams")
                    .join(run_id.to_string())
                    .join(model_safe(model))
                    .join(&scenario.id);
                let workspace = scenario_root.join("workspace");
                copy_fixture(&source_workspace, &workspace)?;
                let artifact_dir = scenario_root.join("artifacts");
                let desktop_fixture = if matches!(scenario.kind, ExamKind::Desktop) {
                    Some(DesktopFixture::start(workspace.clone()).await?)
                } else {
                    None
                };
                let prompt = desktop_fixture.as_ref().map_or_else(
                    || scenario.prompt.clone(),
                    |fixture| scenario.prompt.replace("{{EXAM_URL}}", &fixture.url),
                );
                let context = ToolContext {
                    session_id,
                    workspace: workspace.clone(),
                    data_dir: self.data_dir.clone(),
                    artifact_dir: artifact_dir.clone(),
                    policy: Policy {
                        mode: PolicyMode::Exam {
                            sandbox: workspace.clone(),
                            allowed_window_title: scenario.allowed_window_title.clone(),
                            allowed_process: scenario.allowed_process.clone(),
                        },
                    },
                    approvals: Arc::new(AllowApprovals),
                    platform: self.platform.clone(),
                    memory: self.memory.clone(),
                    session_archive: crate::session_archive::SessionArchive::open(&artifact_dir)?,
                    cancellation: CancellationToken::new(),
                    pause: Arc::new(crate::pause::PauseController::default()),
                    latest_observation: Default::default(),
                    latest_observation_view: Default::default(),
                    pending_visual_localization: Default::default(),
                    task_hint: Default::default(),
                    input_ledger: Default::default(),
                    artifact_evidence: Default::default(),
                    session_files: Default::default(),
                    active_task: Default::default(),
                    focused_control: Default::default(),
                    command_timeout_seconds: self.command_timeout_seconds,
                    input_text_inter_key_pause_ms: 50,
                    vision_max_edge: 1_600,
                    prompt_token_target: 16_000,
                    context_budget: Arc::new(parking_lot::Mutex::new(
                        crate::context::ContextBudget::new(
                            "exam".into(),
                            model.clone(),
                            20_000,
                            80,
                            crate::context::ContextSource::FallbackUnknown,
                        ),
                    )),
                    visual_history_limit: 1,
                    uia_element_limit: 500,
                    desktop_enrichment_timeout_ms: 2_000,
                    desktop_deep_enrichment_timeout_ms: 10_000,
                    model_target_limit: 120,
                    fusion_iou_threshold: 0.1,
                    ocr_containment_threshold: 0.6,
                    annotate_targets: true,
                    approval_cache: Default::default(),
                    user_guidance_queue: Default::default(),
                    questions: None,
                    command_manager: Arc::new(crate::commands::CommandManager::new(artifact_dir)),
                    current_tool_call_id: Default::default(),
                };
                let registry = Arc::new(registry_factory(self.brain.clone(), model));
                let mut session = Session::new(
                    self.brain.clone(),
                    registry,
                    context,
                    model.clone(),
                    scenario.max_turns,
                )?;
                let run = session.run(prompt).await;
                results.push(score_scenario(model, scenario, &workspace, run));
            }
            let earned = results.iter().map(|result| result.score).sum();
            let possible = results.iter().map(|result| result.max_score).sum();
            scores.push(ModelScore {
                model: model.clone(),
                earned,
                possible,
                pass_percent: if possible > 0.0 {
                    earned / possible * 100.0
                } else {
                    0.0
                },
                results,
            });
        }
        scores.sort_by(|a, b| {
            b.pass_percent
                .total_cmp(&a.pass_percent)
                .then_with(|| total_errors(&a.results).cmp(&total_errors(&b.results)))
                .then_with(|| total_actions(&a.results).cmp(&total_actions(&b.results)))
        });
        let report = ExamReport { run_id, scores };
        let root = self.data_dir.join("exams").join(run_id.to_string());
        std::fs::create_dir_all(&root)?;
        crate::memory::atomic_write(
            &root.join("report.json"),
            &serde_json::to_vec_pretty(&report)?,
        )?;
        crate::memory::atomic_write(&root.join("report.md"), render_markdown(&report).as_bytes())?;
        Ok(report)
    }
}

struct DesktopFixture {
    url: String,
    server: tokio::task::JoinHandle<()>,
}

impl DesktopFixture {
    async fn start(root: PathBuf) -> Result<Self> {
        let state_root = root.clone();
        let app = Router::new()
            .route(
                "/api/state",
                post(
                    move |State(root): State<PathBuf>, Json(value): Json<Value>| async move {
                        crate::memory::atomic_write(
                            &root.join("state.json"),
                            &serde_json::to_vec_pretty(&value).map_err(|error| {
                                (axum::http::StatusCode::BAD_REQUEST, error.to_string())
                            })?,
                        )
                        .map_err(|error| {
                            (
                                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                                error.to_string(),
                            )
                        })?;
                        Ok::<_, (axum::http::StatusCode, String)>(Json(
                            serde_json::json!({"ok": true}),
                        ))
                    },
                ),
            )
            .fallback_service(tower_http::services::ServeDir::new(root))
            .with_state(state_root);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let url = format!("http://{address}/index.html");
        let server = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        launch_exam_browser(&url)?;
        Ok(Self { url, server })
    }
}

impl Drop for DesktopFixture {
    fn drop(&mut self) {
        self.server.abort();
    }
}

fn launch_exam_browser(url: &str) -> Result<()> {
    #[cfg(windows)]
    {
        let mut command = std::process::Command::new("cmd.exe");
        command.args([
            "/C",
            "start",
            "",
            "msedge.exe",
            "--new-window",
            "--app",
            url,
        ]);
        crate::process_window::hide_std(&mut command);
        command.spawn()?;
        Ok(())
    }
    #[cfg(not(windows))]
    {
        let _ = url;
        Err(PokError::Unsupported(
            "desktop exams must run from the native Windows build".into(),
        ))
    }
}

fn score_scenario(
    model: &str,
    scenario: &ExamScenario,
    workspace: &Path,
    run: Result<RunSummary>,
) -> ScenarioResult {
    let (answer, metrics, artifacts, error) = match run {
        Ok(summary) => (summary.answer, summary.metrics, summary.artifact_dir, None),
        Err(error) => (
            String::new(),
            RunMetrics::default(),
            PathBuf::new(),
            Some(error.to_string()),
        ),
    };
    let assertions: Vec<_> = scenario
        .assertions
        .iter()
        .cloned()
        .map(|assertion| evaluate(workspace, assertion))
        .collect();
    let passed = assertions.iter().filter(|result| result.passed).count();
    let fraction = if assertions.is_empty() {
        if error.is_none() { 1.0 } else { 0.0 }
    } else {
        passed as f64 / assertions.len() as f64
    };
    ScenarioResult {
        scenario_id: scenario.id.clone(),
        model: model.into(),
        score: scenario.weight * fraction,
        max_score: scenario.weight,
        assertions,
        metrics,
        answer,
        artifacts,
        error,
    }
}

fn evaluate(workspace: &Path, assertion: ExamAssertion) -> AssertionResult {
    let outcome: Result<(bool, String)> = (|| {
        let assertion_path = match &assertion {
            ExamAssertion::FileContains { path, .. }
            | ExamAssertion::FileEquals { path, .. }
            | ExamAssertion::FileExists { path }
            | ExamAssertion::JsonEquals { path, .. } => path,
        };
        if assertion_path.is_absolute()
            || assertion_path
                .components()
                .any(|part| matches!(part, std::path::Component::ParentDir))
        {
            return Err(PokError::Tool(
                "exam assertion path escapes workspace".into(),
            ));
        }
        let (passed, detail) = match &assertion {
            ExamAssertion::FileContains { path, text } => {
                let content = std::fs::read_to_string(workspace.join(path))?;
                (
                    content.contains(text),
                    format!("expected file to contain {text:?}"),
                )
            }
            ExamAssertion::FileEquals { path, text } => {
                let content = std::fs::read_to_string(workspace.join(path))?;
                (content == *text, "expected exact file content".into())
            }
            ExamAssertion::FileExists { path } => (
                workspace.join(path).is_file(),
                "expected file to exist".into(),
            ),
            ExamAssertion::JsonEquals {
                path,
                pointer,
                value,
            } => {
                let document: Value =
                    serde_json::from_slice(&std::fs::read(workspace.join(path))?)?;
                (
                    document.pointer(pointer) == Some(value),
                    format!("expected {pointer} to equal {value}"),
                )
            }
        };
        Ok((passed, detail))
    })();
    match outcome {
        Ok((passed, detail)) => AssertionResult {
            assertion,
            passed,
            detail,
        },
        Err(error) => AssertionResult {
            assertion,
            passed: false,
            detail: error.to_string(),
        },
    }
}

fn model_safe(model: &str) -> String {
    model
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn copy_fixture(source: &Path, destination: &Path) -> Result<()> {
    std::fs::create_dir_all(destination)?;
    for entry in ignore::WalkBuilder::new(source)
        .hidden(false)
        .git_ignore(false)
        .build()
        .filter_map(std::result::Result::ok)
    {
        let relative = entry
            .path()
            .strip_prefix(source)
            .map_err(|error| PokError::Other(error.into()))?;
        if relative.as_os_str().is_empty() {
            continue;
        }
        let target = destination.join(relative);
        if entry.file_type().is_some_and(|kind| kind.is_dir()) {
            std::fs::create_dir_all(target)?;
        } else if entry.file_type().is_some_and(|kind| kind.is_file()) {
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::copy(entry.path(), target)?;
        }
    }
    Ok(())
}
fn total_errors(results: &[ScenarioResult]) -> u32 {
    results
        .iter()
        .map(|result| result.metrics.malformed_tool_calls)
        .sum()
}
fn total_actions(results: &[ScenarioResult]) -> u32 {
    results.iter().map(|result| result.metrics.tool_calls).sum()
}

fn render_markdown(report: &ExamReport) -> String {
    let mut output = format!(
        "# POK-Ai Brain Exam {}\n\n| Rank | Model | Score |\n|---:|---|---:|\n",
        report.run_id
    );
    for (index, score) in report.scores.iter().enumerate() {
        output.push_str(&format!(
            "| {} | {} | {:.1}% |\n",
            index + 1,
            score.model,
            score.pass_percent
        ));
    }
    output
}
