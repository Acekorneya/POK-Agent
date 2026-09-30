use std::{
    io::{self, Write},
    path::{Path, PathBuf},
    sync::Arc,
};

use async_trait::async_trait;
use clap::{Parser, Subcommand};
use pok_ai_core::{
    Result,
    brain::{Brain, from_provider},
    builtins::{register_desktop_tools, register_memory_tools},
    coding::register_coding_tools,
    config::{Config, ProviderConfig, ProviderDataBoundary},
    conversation_store::ConversationStore,
    exam::{ExamRunner, ExamScenario},
    memory::MemoryStore,
    policy::{ApprovalDecision, ApprovalHandler, Policy},
    session::{AgentEvent, Session, SessionObserver},
    subagent::SpawnSubagentTool,
    tool::{ToolContext, ToolRegistry},
};
use pok_ai_windows::WindowsDesktop;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

struct ConsoleObserver {
    /// Display name of the active decision-router backend ("Laya", "JEV"),
    /// so console lines never claim JEV while the local router is running.
    router_name: String,
}

impl SessionObserver for ConsoleObserver {
    fn emit(&self, event: AgentEvent) {
        let router = &self.router_name;
        match event {
            AgentEvent::ToolStarted {
                name, arguments, ..
            } if name == "run_command" => {
                if let Some(command) = arguments.get("command").and_then(serde_json::Value::as_str)
                {
                    eprintln!("\n$ {command}");
                }
            }
            AgentEvent::CommandProgress { event } if event.kind == "output" => {
                if let Some(chunk) = event.chunk {
                    eprint!("{chunk}");
                }
            }
            AgentEvent::CommandProgress { event }
                if event.kind == "status" && event.status != "running" =>
            {
                eprintln!(
                    "\n[command {}: {}]",
                    &event.task_id[..8.min(event.task_id.len())],
                    event.status
                );
            }
            AgentEvent::DecisionRouterStarted {
                purpose,
                candidate_count,
                ..
            } => eprintln!(
                "[{router} evaluating {candidate_count} bounded candidates for {purpose}]"
            ),
            AgentEvent::DecisionRouterEvaluated {
                purpose,
                tool,
                eligible,
                rejection_reason,
                elapsed_ms,
                ..
            } => {
                if eligible {
                    eprintln!(
                        "[{router} selected {} for {purpose} in {elapsed_ms} ms]",
                        tool.as_deref().unwrap_or("a candidate")
                    );
                } else {
                    eprintln!(
                        "[{router} used local fallback for {purpose} in {elapsed_ms} ms: {}]",
                        rejection_reason.as_deref().unwrap_or("not eligible")
                    );
                }
            }
            AgentEvent::DecisionRouterCacheHit { purpose, .. } => {
                eprintln!("[{router} reused the unchanged {purpose} decision]");
            }
            AgentEvent::DecisionRouterMemoryOutcome {
                disposition,
                used_jev,
                ..
            } => eprintln!(
                "[Memory {disposition}; verified {}]",
                if used_jev {
                    format!("by {router}")
                } else {
                    "locally".to_string()
                }
            ),
            _ => {}
        }
    }
}

#[derive(Parser)]
#[command(
    name = "pok-ai",
    version,
    about = "Local-first Windows computer-use and coding harness"
)]
struct Cli {
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Doctor,
    Models {
        #[arg(long)]
        provider: Option<String>,
        /// Also print each model's reported capabilities: vision, tool use,
        /// and reasoning levels with their default.
        #[arg(long)]
        details: bool,
    },
    Run {
        prompt: String,
        #[arg(long)]
        workspace: Option<PathBuf>,
        #[arg(long)]
        provider: Option<String>,
        #[arg(long)]
        model: Option<String>,
        #[arg(long)]
        yes: bool,
        #[arg(long = "continue", conflicts_with = "session")]
        continue_session: bool,
        #[arg(long, conflicts_with = "continue_session")]
        session: Option<Uuid>,
    },
    Sessions {
        #[command(subcommand)]
        command: SessionCommands,
    },
    Exam {
        scenario_file: PathBuf,
        #[arg(long, value_delimiter = ',')]
        models: Vec<String>,
        #[arg(long)]
        provider: Option<String>,
        #[arg(long)]
        yes: bool,
    },
    Memory {
        query: String,
        #[arg(long, default_value_t = 10)]
        limit: usize,
    },
    MemoryDrafts {
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    MemoryApprove {
        id: Uuid,
    },
    /// Benchmark a decision-router backend on a question suite using the
    /// production request code (see docs/decision-router-backends.md).
    RouterBench {
        suite: PathBuf,
        /// Backend override: jev, laya, or llm_choice.
        #[arg(long)]
        backend: Option<String>,
        /// Model name override for the selected backend.
        #[arg(long)]
        model: Option<String>,
        /// Write per-case results and the summary as JSON.
        #[arg(long)]
        json: Option<PathBuf>,
        /// Instead of accuracy, time a target pick plus a completion check as
        /// two requests versus one batched request on every target case.
        #[arg(long)]
        batch_ab: bool,
    },
}

#[derive(Subcommand)]
enum SessionCommands {
    List {
        #[arg(long, default_value_t = 20)]
        limit: usize,
        #[arg(long)]
        workspace: Option<PathBuf>,
    },
}

fn main() -> anyhow::Result<()> {
    // Agent runs nest deep async state machines (a delegated plan inside a
    // tool call inside the run loop). Run them on a worker with a roomy stack
    // instead of the main thread, whose stack is 1 MiB on Windows.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_stack_size(16 * 1024 * 1024)
        .build()?;
    runtime.block_on(async { tokio::spawn(run_cli()).await? })
}

async fn run_cli() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let cli = Cli::parse();
    let config_path = cli.config.or_else(|| {
        Path::new("pok-ai.toml")
            .is_file()
            .then(|| PathBuf::from("pok-ai.toml"))
    });
    let config = Config::load(config_path.as_deref())?;
    match cli.command {
        Commands::Doctor => doctor(&config).await?,
        Commands::RouterBench {
            suite,
            backend,
            model,
            json,
            batch_ab,
        } => {
            let mut router_config = config.decision_router.clone();
            router_config.enabled = true;
            if let Some(backend) = backend {
                router_config.backend = serde_json::from_value(serde_json::Value::String(backend))?;
            }
            if let Some(model) = model {
                match router_config.backend {
                    pok_ai_core::config::DecisionRouterBackend::Laya => {
                        router_config.laya.model = model
                    }
                    pok_ai_core::config::DecisionRouterBackend::LlmChoice => {
                        router_config.llm_choice.model = model
                    }
                    _ => router_config.model = model,
                }
            }
            let suite = pok_ai_core::router_bench::load_suite(&suite)?;
            if batch_ab {
                let mut rows = Vec::new();
                for case in &suite.cases {
                    let router =
                        pok_ai_core::decision::TypeSafeDecisionRouter::new(router_config.clone())?;
                    if let Some(row) =
                        pok_ai_core::router_bench::compare_batching(&router, case).await
                    {
                        rows.push(row);
                    }
                }
                let ok = rows
                    .iter()
                    .filter(|row| row.error.is_none())
                    .collect::<Vec<_>>();
                let median = |mut values: Vec<u64>| {
                    values.sort_unstable();
                    values.get(values.len() / 2).copied().unwrap_or(0)
                };
                println!(
                    "backend={:?} model={} target_cases={} errors={}",
                    router_config.backend,
                    router_config.active_model(),
                    rows.len(),
                    rows.len() - ok.len()
                );
                println!(
                    "separate (pick + check) p50: {} ms   batched p50: {} ms   pick alone p50: {} ms",
                    median(ok.iter().map(|row| row.pick_ms + row.check_ms).collect()),
                    median(ok.iter().map(|row| row.batched_ms).collect()),
                    median(ok.iter().map(|row| row.pick_ms).collect()),
                );
                println!(
                    "same pick: {}/{}   same verdict: {}/{}   batched verdict correct: {}/{}",
                    ok.iter().filter(|row| row.same_pick).count(),
                    ok.len(),
                    ok.iter().filter(|row| row.same_verdict).count(),
                    ok.len(),
                    ok.iter().filter(|row| row.batched_verdict_correct).count(),
                    ok.len(),
                );
                println!(
                    "fan-out: separate (pick + check + condition) p50: {} ms   one request p50: {} ms",
                    median(
                        ok.iter()
                            .map(|row| row.pick_ms + row.check_ms + row.condition_ms)
                            .collect()
                    ),
                    median(ok.iter().map(|row| row.fanout_ms).collect()),
                );
                println!(
                    "fan-out same pick: {}/{}   same verdict: {}/{}   same condition: {}/{}   condition correct: {}/{}",
                    ok.iter().filter(|row| row.fanout_same_pick).count(),
                    ok.len(),
                    ok.iter().filter(|row| row.fanout_same_verdict).count(),
                    ok.len(),
                    ok.iter().filter(|row| row.fanout_same_condition).count(),
                    ok.len(),
                    ok.iter().filter(|row| row.fanout_condition_correct).count(),
                    ok.len(),
                );
                if let Some(path) = json {
                    std::fs::write(path, serde_json::to_vec_pretty(&rows)?)?;
                }
                return Ok(());
            }
            // A fresh router per case: the production circuit breaker would
            // otherwise turn one slow answer into errors for every later case.
            let mut results = Vec::with_capacity(suite.cases.len());
            for case in &suite.cases {
                let router =
                    pok_ai_core::decision::TypeSafeDecisionRouter::new(router_config.clone())?;
                results
                    .push(pok_ai_core::router_bench::run_case(&router, &router_config, case).await);
            }
            let summary = pok_ai_core::router_bench::summarize(&results);
            println!(
                "backend={:?} model={} cases={}",
                router_config.backend,
                router_config.active_model(),
                results.len()
            );
            println!(
                "{:<18}{:>6}{:>9}{:>9}{:>8}{:>9}{:>9}{:>9}{:>8}",
                "kind",
                "cases",
                "correct",
                "acc_ok",
                "acc_bad",
                "deferred",
                "ground_n",
                "ground_ok",
                "p50_ms"
            );
            if let Some(kinds) = summary["by_kind"].as_object() {
                for (kind, row) in kinds {
                    println!(
                        "{:<18}{:>6}{:>9}{:>9}{:>8}{:>9}{:>9}{:>9}{:>8}",
                        kind,
                        row["cases"].as_u64().unwrap_or(0),
                        row["correct"].as_u64().unwrap_or(0),
                        row["accepted_correct"].as_u64().unwrap_or(0),
                        row["accepted_wrong"].as_u64().unwrap_or(0),
                        row["deferred"].as_u64().unwrap_or(0),
                        row["grounding_decided"].as_u64().unwrap_or(0),
                        row["grounding_correct"].as_u64().unwrap_or(0),
                        row["latency_ms_p50"].as_u64().unwrap_or(0),
                    );
                }
            }
            for bucket in summary["calibration"].as_array().into_iter().flatten() {
                println!(
                    "calibration {}: {}/{} correct",
                    bucket["bucket"].as_str().unwrap_or_default(),
                    bucket["correct"],
                    bucket["cases"]
                );
            }
            for result in results
                .iter()
                .filter(|result| result.error.is_some())
                .take(5)
            {
                eprintln!(
                    "error in {}: {}",
                    result.id,
                    result.error.as_deref().unwrap_or_default()
                );
            }
            if let Some(path) = json {
                std::fs::write(
                    path,
                    serde_json::to_vec_pretty(&serde_json::json!({
                        "backend": router_config.backend,
                        "model": router_config.active_model(),
                        "summary": summary,
                        "results": results,
                    }))?,
                )?;
            }
        }
        Commands::Models { provider, details } => {
            let brain = brain(&config, provider.as_deref())?;
            if details {
                let flag = |value: Option<bool>| match value {
                    Some(true) => "yes",
                    Some(false) => "no",
                    None => "?",
                };
                for model in brain.model_info().await? {
                    let reasoning = if model.reasoning_efforts.is_empty() {
                        flag(model.reasoning).to_owned()
                    } else {
                        format!(
                            "{} (default {})",
                            model.reasoning_efforts.join("/"),
                            model.reasoning_default.as_deref().unwrap_or("provider")
                        )
                    };
                    println!(
                        "{}\tvision {}\ttools {}\treasoning {reasoning}",
                        model.id,
                        flag(model.vision),
                        flag(model.tool_use)
                    );
                }
            } else {
                for model in brain.list_models().await? {
                    println!("{model}");
                }
            }
        }
        Commands::Run {
            prompt,
            workspace,
            provider,
            model,
            yes,
            continue_session,
            session: session_id,
        } => {
            let conversation_store = ConversationStore::open(&config.data_dir)?;
            conversation_store.import_legacy_diagnostics(&config.diagnostics_dir)?;
            let resume_snapshot = if let Some(id) = session_id {
                Some(
                    conversation_store
                        .load(id)?
                        .ok_or_else(|| anyhow::anyhow!("conversation {id} was not found"))?,
                )
            } else if continue_session {
                Some(
                    conversation_store
                        .latest(None)?
                        .ok_or_else(|| anyhow::anyhow!("no saved conversation is available"))?,
                )
            } else {
                None
            };
            let provider_name = resume_snapshot.as_ref().map_or_else(
                || {
                    provider
                        .clone()
                        .unwrap_or_else(|| config.default_provider.clone())
                },
                |snapshot| snapshot.summary.provider.clone(),
            );
            let provider_config = config.provider(Some(&provider_name))?;
            confirm_external_inference(&provider_name, provider_config, yes)?;
            let mut decision_router_config = config.decision_router.clone();
            if let Some(saved_preference) = resume_snapshot
                .as_ref()
                .and_then(|snapshot| snapshot.state.get("decision_router_enabled"))
                .and_then(serde_json::Value::as_bool)
            {
                decision_router_config.enabled = saved_preference;
            }
            if decision_router_config.enabled
                && let Some(key_env) = decision_router_config.active_key_env()
                && pok_ai_core::brain::get_api_key(key_env).is_none_or(|key| key.trim().is_empty())
            {
                eprintln!(
                    "Decision router unavailable: {key_env} is not configured; continuing with the complete local path."
                );
                decision_router_config.enabled = false;
            }
            confirm_external_decision_router(&decision_router_config, yes)?;
            let brain = brain(&config, Some(&provider_name))?;
            let model = if let Some(snapshot) = &resume_snapshot {
                snapshot.summary.model.clone()
            } else {
                select_model(&config, model, brain.clone()).await?
            };
            let provider_contexts = brain.model_info().await.ok().and_then(|models| {
                models
                    .into_iter()
                    .find(|item| item.id.eq_ignore_ascii_case(&model))
                    .map(|item| (item.loaded_context_length, item.max_context_length))
            });
            if config.prompt_token_target.is_some() {
                eprintln!(
                    "Warning: prompt_token_target is deprecated and ignored; POK-Agent now sizes the model working set automatically."
                );
            }
            let context_budget = pok_ai_core::context::resolve_context_budget(
                &config.data_dir,
                &provider_name,
                &model,
                provider_contexts.and_then(|value| value.0),
                provider_contexts.and_then(|value| value.1),
                32_768,
            )
            .await;
            let workspace = if let Some(snapshot) = &resume_snapshot {
                snapshot.summary.workspace.clone()
            } else {
                dunce::canonicalize(workspace.unwrap_or(std::env::current_dir()?))?
            };
            let session_id = resume_snapshot
                .as_ref()
                .map_or_else(Uuid::new_v4, |snapshot| snapshot.summary.id);
            let memory = MemoryStore::open(config.data_dir.join("memory"))?;
            let approvals: Arc<dyn ApprovalHandler> = if yes {
                Arc::new(YesApproval)
            } else {
                Arc::new(ConsoleApproval)
            };
            let cancellation = CancellationToken::new();
            let stop = cancellation.clone();
            tokio::spawn(async move {
                if tokio::signal::ctrl_c().await.is_ok() {
                    stop.cancel();
                }
            });
            let platform = pok_ai_windows::desktop_platform(config.background_desktop_work);
            let mut tools = standard_tools(brain.clone(), &model);
            pok_ai_core::generated_tools::register_generated_tool_tools(&mut tools);
            let artifact_dir = resume_snapshot.as_ref().map_or_else(
                || {
                    config
                        .diagnostics_dir
                        .join("sessions")
                        .join(session_id.to_string())
                },
                |snapshot| snapshot.summary.artifact_dir.clone(),
            );
            let context = ToolContext {
                session_id,
                workspace,
                data_dir: config.data_dir.clone(),
                artifact_dir: artifact_dir.clone(),
                policy: Policy::interactive(),
                approvals,
                platform,
                memory,
                session_archive: pok_ai_core::session_archive::SessionArchive::open(&artifact_dir)?,
                cancellation,
                pause: Arc::new(pok_ai_core::pause::PauseController::default()),
                latest_observation: Default::default(),
                latest_observation_view: Default::default(),
                pending_visual_localization: Default::default(),
                task_hint: Default::default(),
                input_ledger: Default::default(),
                artifact_evidence: Default::default(),
                attached_paths: Default::default(),
                session_files: Default::default(),
                active_task: Default::default(),
                focused_control: Default::default(),
                command_timeout_seconds: config.command_timeout_seconds,
                input_text_inter_key_pause_ms: config.input_text_inter_key_pause_ms,
                vision_max_edge: config.vision_max_edge,
                prompt_token_target: context_budget.compact_at_tokens,
                context_budget: Arc::new(parking_lot::Mutex::new(context_budget)),
                visual_history_limit: config.visual_history_limit,
                uia_element_limit: config.uia_element_limit,
                desktop_enrichment_timeout_ms: config.desktop_enrichment_timeout_ms,
                desktop_deep_enrichment_timeout_ms: config.desktop_deep_enrichment_timeout_ms,
                model_target_limit: config.model_target_limit,
                fusion_iou_threshold: config.fusion_iou_threshold,
                ocr_containment_threshold: config.ocr_containment_threshold,
                annotate_targets: config.annotate_targets,
                approval_cache: Default::default(),
                user_guidance_queue: Default::default(),
                questions: None,
                command_manager: Arc::new(pok_ai_core::commands::CommandManager::new(artifact_dir)),
                current_tool_call_id: Default::default(),
            };
            let router_name = decision_router_config.backend.display_name().to_string();
            let mut session = Session::new(
                brain,
                Arc::new(std::mem::take(&mut tools)),
                context,
                model,
                config.max_turns,
            )?
            .with_conversation_store(conversation_store, provider_name)?
            .with_decision_router(
                pok_ai_core::decision::TypeSafeDecisionRouter::from_config(
                    &decision_router_config,
                )?,
                decision_router_config,
            );
            if let Some(snapshot) = resume_snapshot {
                session.restore_conversation(snapshot)?;
            }
            let mut session = session
                .with_temperature(config.agent_temperature.value())
                .with_action_step_budget(config.action_step_budget)
                .with_standing_instructions(config.standing_instructions.as_deref())
                .with_observer(Arc::new(ConsoleObserver { router_name }));
            let result = session.run(prompt).await?;
            println!("{}", result.answer);
            session
                .finish_background_work(std::time::Duration::from_secs(300))
                .await;
            eprintln!("\nArtifacts: {}", result.artifact_dir.display());
        }
        Commands::Sessions { command } => match command {
            SessionCommands::List { limit, workspace } => {
                let store = ConversationStore::open(&config.data_dir)?;
                store.import_legacy_diagnostics(&config.diagnostics_dir)?;
                let workspace = workspace.map(dunce::canonicalize).transpose()?;
                for item in store.list(workspace.as_deref(), limit)? {
                    println!(
                        "{}  {}  {} / {}  {}  {}",
                        item.id,
                        item.updated_at,
                        item.provider,
                        item.model,
                        item.status,
                        item.title.replace('\n', " ")
                    );
                }
            }
        },
        Commands::Exam {
            scenario_file,
            models,
            provider,
            yes,
        } => {
            let provider_name = provider
                .clone()
                .unwrap_or_else(|| config.default_provider.clone());
            confirm_external_inference(
                &provider_name,
                config.provider(Some(&provider_name))?,
                yes,
            )?;
            let scenarios: Vec<ExamScenario> =
                serde_json::from_slice(&std::fs::read(&scenario_file)?)?;
            let brain = brain(&config, provider.as_deref())?;
            let models = if models.is_empty() {
                vec![select_model(&config, None, brain.clone()).await?]
            } else {
                models
            };
            let runner = ExamRunner {
                brain: brain.clone(),
                platform: Arc::new(WindowsDesktop::new()),
                data_dir: config.data_dir.clone(),
                memory: MemoryStore::open(config.data_dir.join("exam-memory"))?,
                command_timeout_seconds: config.command_timeout_seconds,
            };
            let report = runner.run(&models, &scenarios, standard_tools).await?;
            for (index, score) in report.scores.iter().enumerate() {
                println!(
                    "{}. {} — {:.1}% ({:.2}/{:.2})",
                    index + 1,
                    score.model,
                    score.pass_percent,
                    score.earned,
                    score.possible
                );
            }
            println!(
                "Report: {}",
                config
                    .data_dir
                    .join("exams")
                    .join(report.run_id.to_string())
                    .join("report.md")
                    .display()
            );
        }
        Commands::Memory { query, limit } => {
            let context =
                MemoryStore::open(config.data_dir.join("memory"))?.retrieve_context(&query)?;
            for hit in context.all().take(limit) {
                println!(
                    "[{:.3}] {} {}: {}",
                    hit.score, hit.category, hit.source, hit.text
                );
            }
        }
        Commands::MemoryDrafts { limit } => {
            for draft in MemoryStore::open(config.data_dir.join("memory"))?.list_drafts(limit)? {
                println!("{} [{}] {}", draft.id, draft.source, draft.text);
            }
        }
        Commands::MemoryApprove { id } => {
            let approved = MemoryStore::open(config.data_dir.join("memory"))?.approve(id)?;
            println!(
                "{}",
                if approved {
                    "approved"
                } else {
                    "draft not found"
                }
            );
        }
    }
    Ok(())
}

fn brain(config: &Config, provider: Option<&str>) -> Result<Arc<dyn Brain>> {
    from_provider(config.provider(provider)?)
}

fn confirm_external_inference(
    provider_name: &str,
    provider: &ProviderConfig,
    assume_yes: bool,
) -> Result<()> {
    if provider.resolved_data_boundary(provider_name) == ProviderDataBoundary::LocalDevice {
        return Ok(());
    }
    eprintln!(
        "External provider warning: {provider_name} ({}) may receive full prompts, workspace paths, tool results, screenshots/OCR, and document contents.",
        provider.base_url
    );
    if assume_yes {
        return Ok(());
    }
    print!("Continue with full-fidelity external inference? [y/N] ");
    io::stdout().flush()?;
    let mut answer = String::new();
    io::stdin().read_line(&mut answer)?;
    if matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes") {
        Ok(())
    } else {
        Err(pok_ai_core::PokError::Provider(
            "external provider request cancelled before task context was sent".into(),
        ))
    }
}

fn confirm_external_decision_router(
    config: &pok_ai_core::config::DecisionRouterConfig,
    assume_yes: bool,
) -> Result<()> {
    if !config.enabled || config.backend != pok_ai_core::config::DecisionRouterBackend::Jev {
        // Laya is a loopback-only local backend and never sends data externally,
        // so it needs no acknowledgment here (matches the Tauri app, which only
        // gates this confirmation for the hosted Jev backend).
        return Ok(());
    }
    eprintln!(
        "External JEV warning: {} may receive compact task/UI metadata, bounded context or memory snippets, and locally shortlisted source-code previews. Screenshots, full OCR, unrestricted history, ignored files, and detected secrets are excluded.",
        config.endpoint
    );
    if assume_yes {
        return Ok(());
    }
    print!("Continue with optional external JEV decisions? [y/N] ");
    io::stdout().flush()?;
    let mut answer = String::new();
    io::stdin().read_line(&mut answer)?;
    if matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes") {
        Ok(())
    } else {
        Err(pok_ai_core::PokError::Provider(
            "external JEV request cancelled before task metadata was sent".into(),
        ))
    }
}

async fn select_model(
    config: &Config,
    requested: Option<String>,
    brain: Arc<dyn Brain>,
) -> Result<String> {
    if let Some(model) = requested.filter(|model| !model.is_empty()) {
        return Ok(model);
    }
    if !config.default_model.is_empty() {
        return Ok(config.default_model.clone());
    }
    brain
        .list_models()
        .await?
        .into_iter()
        .next()
        .ok_or_else(|| pok_ai_core::PokError::Provider("provider returned no models".into()))
}

fn standard_tools(brain: Arc<dyn Brain>, model: &str) -> ToolRegistry {
    let mut tools = ToolRegistry::new();
    register_coding_tools(&mut tools);
    register_desktop_tools(&mut tools);
    register_memory_tools(&mut tools);
    tools.register(SpawnSubagentTool::new(brain, model));
    tools
}

async fn doctor(config: &Config) -> Result<()> {
    println!("POK-Agent data: {}", config.data_dir.display());
    println!("Host: {}", std::env::consts::OS);
    if !cfg!(windows) {
        println!("Desktop: unavailable in this WSL/Linux build; use the native Windows binary");
    }
    let provider = config.provider(None)?;
    println!(
        "Provider: {} ({})",
        config.default_provider, provider.base_url
    );
    match from_provider(provider)?.model_info().await {
        Ok(models) => {
            println!("Provider reachable: yes ({} models)", models.len());
            println!(
                "Default model present: {}",
                models.iter().any(|model| model.id == config.default_model)
            );
            for model in models {
                println!(
                    "  - {} [context={}, vision={}, tools={}, reasoning={}, loaded={}]",
                    model.id,
                    model
                        .context_length
                        .map_or_else(|| "unknown".into(), |value| value.to_string()),
                    model
                        .vision
                        .map_or_else(|| "unknown".into(), |value| value.to_string()),
                    model
                        .tool_use
                        .map_or_else(|| "unknown".into(), |value| value.to_string()),
                    model
                        .reasoning
                        .map_or_else(|| "unknown".into(), |value| value.to_string()),
                    model
                        .loaded
                        .map_or_else(|| "unknown".into(), |value| value.to_string()),
                );
            }
        }
        Err(error) => println!("Provider reachable: no ({error})"),
    }
    let memory = MemoryStore::open(config.data_dir.join("memory"))?;
    println!("Memory: {}", memory.root().display());
    Ok(())
}

struct YesApproval;
#[async_trait]
impl ApprovalHandler for YesApproval {
    async fn approve(
        &self,
        _tool: &str,
        _arguments: &serde_json::Value,
        _reason: &str,
    ) -> Result<ApprovalDecision> {
        Ok(ApprovalDecision::AllowOnce)
    }
}

struct ConsoleApproval;
#[async_trait]
impl ApprovalHandler for ConsoleApproval {
    async fn approve(
        &self,
        tool: &str,
        arguments: &serde_json::Value,
        reason: &str,
    ) -> Result<ApprovalDecision> {
        let tool = tool.to_owned();
        let arguments = arguments.clone();
        let reason = reason.to_owned();
        tokio::task::spawn_blocking(move || {
            println!("\nApproval required: {tool}\nReason: {reason}\nArguments: {arguments}");
            print!("Allow? [y/N] ");
            io::stdout().flush()?;
            let mut answer = String::new();
            io::stdin().read_line(&mut answer)?;
            Ok(
                if matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes") {
                    ApprovalDecision::AllowOnce
                } else {
                    ApprovalDecision::Deny
                },
            )
        })
        .await
        .map_err(|error| pok_ai_core::PokError::Other(error.into()))?
    }
}
