use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use directories::ProjectDirs;
use serde::{Deserialize, Serialize};

use crate::{PokError, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderProtocol {
    OpenaiCompatible,
    AnthropicMessages,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderDataBoundary {
    LocalDevice,
    ExternalService,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub protocol: ProviderProtocol,
    pub base_url: String,
    #[serde(default)]
    pub data_boundary: Option<ProviderDataBoundary>,
    #[serde(default)]
    pub api_key_env: Option<String>,
    #[serde(default = "default_provider_max_retries")]
    pub max_retries: u32,
}

/// How the fast decision model is used.
///
/// `RouterFirst` runs the router before every primary-model turn and lets it
/// choose the next operation for the whole task. `Delegated` keeps planning in
/// the primary model: it calls `fast_actions` with a subgoal and a completion
/// condition, and the router only matches targets and checks completion.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionRouterMode {
    RouterFirst,
    #[default]
    Delegated,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionRouterBackend {
    Off,
    #[default]
    Jev,
    Laya,
    /// A small OpenAI-compatible LLM behind the local `llm_choice_sidecar.py`
    /// adapter, answering the same typed choice questions.
    LlmChoice,
    /// kev-4b (jaredpalmer/kev, Apache-2.0): a local Jev-style decision model
    /// that serves the `/v1/systemone` contract natively (`scripts/setup-kev.ps1`).
    Kev,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayaDevicePreference {
    #[default]
    Auto,
    Gpu,
    Cpu,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayaRouterConfig {
    #[serde(default = "default_laya_endpoint")]
    pub endpoint: String,
    #[serde(default = "default_laya_model")]
    pub model: String,
    #[serde(default = "default_laya_token_env")]
    pub token_env: String,
    #[serde(default = "default_laya_min_free_vram_mb")]
    pub min_free_vram_mb: u64,
    #[serde(default)]
    pub device: LayaDevicePreference,
    #[serde(default = "default_laya_operation_probability")]
    pub min_operation_probability: f64,
    #[serde(default = "default_laya_target_probability")]
    pub min_target_probability: f64,
    /// Laya-specific override for the shared `min_terminal_probability` gate
    /// (config.rs's top-level field, which also governs JEV and stays
    /// unchanged). Real traffic showed correct DONE/BLOCKED decisions
    /// landing right at the shared 0.60 default (e.g. 0.5986, missed by
    /// 0.0014) — see docs/decision-router-backends.md for the evidence.
    #[serde(default = "default_laya_terminal_probability")]
    pub min_terminal_probability: f64,
}

impl Default for LayaRouterConfig {
    fn default() -> Self {
        Self {
            endpoint: default_laya_endpoint(),
            model: default_laya_model(),
            token_env: default_laya_token_env(),
            min_free_vram_mb: default_laya_min_free_vram_mb(),
            device: LayaDevicePreference::Auto,
            min_operation_probability: default_laya_operation_probability(),
            min_target_probability: default_laya_target_probability(),
            min_terminal_probability: default_laya_terminal_probability(),
        }
    }
}

/// Cross-model second opinion for a Laya decision the combined probability
/// gate (`LayaRouterConfig::min_operation_probability`/
/// `min_target_probability`) didn't clear. Offline+live testing
/// (docs/decision-router-backends.md) found a *different* local model
/// judging Laya's specific picked candidate with three independent
/// yes/no/unknown sub-questions (reversible/cheap/evidenced, promoting only
/// on unanimous agreement) resolved 90% of cases correctly with zero
/// dangerous false positives — a different backend judging beats both
/// Laya judging itself (58.3%, worse than chance on single-candidate
/// scenes) and JEV judging itself (70%, including two cases where JEV
/// confidently re-endorsed its own wrong pick). Disabled by default: it
/// requires a second local sidecar (e.g. `php-ai/zeiger-0.6b`) to be
/// running independently; nothing in this harness launches or manages it
/// yet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgeConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_judge_endpoint")]
    pub endpoint: String,
    #[serde(default = "default_judge_model")]
    pub model: String,
    #[serde(default = "default_judge_token_env")]
    pub token_env: String,
    #[serde(default = "default_judge_timeout_ms")]
    pub timeout_ms: u64,
}

impl Default for JudgeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            endpoint: default_judge_endpoint(),
            model: default_judge_model(),
            token_env: default_judge_token_env(),
            timeout_ms: default_judge_timeout_ms(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionRouterConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub backend: DecisionRouterBackend,
    #[serde(default)]
    pub mode: DecisionRouterMode,
    #[serde(default = "default_decision_router_endpoint")]
    pub endpoint: String,
    #[serde(default = "default_decision_router_model")]
    pub model: String,
    #[serde(default = "default_decision_router_key_env")]
    pub api_key_env: String,
    #[serde(default = "default_decision_router_timeout_ms")]
    pub timeout_ms: u64,
    #[serde(default = "default_decision_router_candidate_limit")]
    pub max_candidates: usize,
    #[serde(default = "default_decision_router_probability")]
    pub min_selected_probability: f64,
    /// Minimum confidence that the selected operation family is appropriate.
    /// Target confidence remains independently controlled by `min_confidence`.
    #[serde(default = "default_decision_router_operation_confidence")]
    pub min_operation_confidence: f64,
    /// Minimum confidence in the exact bounded target. Kept under the original
    /// name for configuration compatibility.
    #[serde(default = "default_decision_router_confidence")]
    pub min_confidence: f64,
    #[serde(default = "default_decision_router_terminal_probability")]
    pub min_terminal_probability: f64,
    #[serde(default = "default_decision_router_terminal_confidence")]
    pub min_terminal_confidence: f64,
    /// Maximum number of progressively narrower JEV evaluations for one
    /// unchanged action state before bounded primary-model assistance.
    #[serde(default = "default_decision_router_refinement_attempts")]
    pub max_refinement_attempts: usize,
    /// Stop refinement early after this many consecutive evaluations fail to
    /// improve confidence by `min_refinement_confidence_gain`.
    #[serde(default = "default_decision_router_stagnant_refinements")]
    pub max_stagnant_refinements: usize,
    #[serde(default = "default_decision_router_refinement_gain")]
    pub min_refinement_confidence_gain: f64,
    /// Maximum consecutive harness-owned actions JEV may select before the
    /// primary model is consulted again. Terminal DONE/BLOCKED decisions end
    /// the loop earlier.
    #[serde(default = "default_decision_router_step_actions")]
    pub max_step_actions: usize,
    /// HTTPS endpoint used for deterministic managed-browser searches. The
    /// active request is appended as the `q` query parameter.
    #[serde(default = "default_decision_router_search_url")]
    pub search_url: String,
    /// Minimum probability of the "yes" answer before a delegated
    /// `fast_actions` run treats its `done_when` condition as satisfied.
    #[serde(default = "default_decision_router_completion_probability")]
    pub min_completion_probability: f64,
    /// Whether delegated `fast_actions` may act on a target the fast model
    /// picks when local grounding cannot decide it. Unset means true only for
    /// calibrated hosted backends (JEV): the router question bench found the
    /// local backends confidently wrong on target picks.
    #[serde(default)]
    pub trust_model_targets: Option<bool>,
    /// Whether the fast model may judge unquoted done_when, branch and
    /// interrupt conditions. Unset means true only for JEV, for the same
    /// reason; quoted conditions are always checked locally.
    #[serde(default)]
    pub trust_model_conditions: Option<bool>,
    /// Ask a fast-action step's completion check and target pick in one
    /// request when both need the model, so the state is sent once.
    #[serde(default = "default_true")]
    pub batch_router_questions: bool,
    #[serde(default)]
    pub laya: LayaRouterConfig,
    #[serde(default)]
    pub llm_choice: LlmChoiceRouterConfig,
    #[serde(default)]
    pub kev: KevRouterConfig,
    #[serde(default)]
    pub judge: JudgeConfig,
}

/// Local kev-4b server (`python -m kev.serve`), loopback only. It answers the
/// typed contract directly, so no adapter sits in between. Its gates reuse the
/// Laya probability thresholds; unlike Laya it is trusted by default because
/// the router question bench found it as accurate as JEV on target picks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KevRouterConfig {
    #[serde(default = "default_kev_endpoint")]
    pub endpoint: String,
    #[serde(default = "default_kev_model")]
    pub model: String,
    #[serde(default = "default_kev_token_env")]
    pub token_env: String,
}

impl Default for KevRouterConfig {
    fn default() -> Self {
        Self {
            endpoint: default_kev_endpoint(),
            model: default_kev_model(),
            token_env: default_kev_token_env(),
        }
    }
}

fn default_kev_endpoint() -> String {
    "http://127.0.0.1:43216/v1/systemone".into()
}
fn default_kev_model() -> String {
    "jaredpalmer/kev-4b".into()
}
fn default_kev_token_env() -> String {
    "POK_KEV_TOKEN".into()
}

/// Loopback sidecar that turns typed choice questions into single-token
/// calls against a small OpenAI-compatible model (see
/// `scripts/llm_choice_sidecar.py`). Its gates reuse the Laya probability
/// thresholds, since both return native per-option probabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmChoiceRouterConfig {
    #[serde(default = "default_llm_choice_endpoint")]
    pub endpoint: String,
    #[serde(default = "default_llm_choice_model")]
    pub model: String,
    #[serde(default = "default_llm_choice_token_env")]
    pub token_env: String,
}

impl Default for LlmChoiceRouterConfig {
    fn default() -> Self {
        Self {
            endpoint: default_llm_choice_endpoint(),
            model: default_llm_choice_model(),
            token_env: default_llm_choice_token_env(),
        }
    }
}

fn default_llm_choice_endpoint() -> String {
    "http://127.0.0.1:43211/v1/systemone".into()
}
fn default_llm_choice_model() -> String {
    "lfm2.5-8b-a1b".into()
}
fn default_llm_choice_token_env() -> String {
    "POK_LLM_CHOICE_TOKEN".into()
}

impl Default for DecisionRouterConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            backend: DecisionRouterBackend::Jev,
            mode: DecisionRouterMode::Delegated,
            endpoint: default_decision_router_endpoint(),
            model: default_decision_router_model(),
            api_key_env: default_decision_router_key_env(),
            timeout_ms: default_decision_router_timeout_ms(),
            max_candidates: default_decision_router_candidate_limit(),
            min_selected_probability: default_decision_router_probability(),
            min_operation_confidence: default_decision_router_operation_confidence(),
            min_confidence: default_decision_router_confidence(),
            min_terminal_probability: default_decision_router_terminal_probability(),
            min_terminal_confidence: default_decision_router_terminal_confidence(),
            max_refinement_attempts: default_decision_router_refinement_attempts(),
            max_stagnant_refinements: default_decision_router_stagnant_refinements(),
            min_refinement_confidence_gain: default_decision_router_refinement_gain(),
            max_step_actions: default_decision_router_step_actions(),
            search_url: default_decision_router_search_url(),
            min_completion_probability: default_decision_router_completion_probability(),
            trust_model_targets: None,
            trust_model_conditions: None,
            batch_router_questions: true,
            laya: LayaRouterConfig::default(),
            llm_choice: LlmChoiceRouterConfig::default(),
            kev: KevRouterConfig::default(),
            judge: JudgeConfig::default(),
        }
    }
}

impl DecisionRouterConfig {
    pub fn active_backend(&self) -> DecisionRouterBackend {
        if self.enabled {
            self.backend
        } else {
            DecisionRouterBackend::Off
        }
    }

    pub fn active_endpoint(&self) -> &str {
        match self.backend {
            DecisionRouterBackend::Laya => &self.laya.endpoint,
            DecisionRouterBackend::LlmChoice => &self.llm_choice.endpoint,
            DecisionRouterBackend::Kev => &self.kev.endpoint,
            _ => &self.endpoint,
        }
    }

    pub fn active_model(&self) -> &str {
        match self.backend {
            DecisionRouterBackend::Laya => &self.laya.model,
            DecisionRouterBackend::LlmChoice => &self.llm_choice.model,
            DecisionRouterBackend::Kev => &self.kev.model,
            _ => &self.model,
        }
    }

    pub fn active_key_env(&self) -> Option<&str> {
        match self.backend {
            DecisionRouterBackend::Jev => Some(&self.api_key_env),
            DecisionRouterBackend::Laya => Some(&self.laya.token_env),
            DecisionRouterBackend::LlmChoice => Some(&self.llm_choice.token_env),
            DecisionRouterBackend::Kev => Some(&self.kev.token_env),
            DecisionRouterBackend::Off => None,
        }
    }

    /// True for local backends that answer with native per-option choice
    /// probabilities (Laya, LLM choice) rather than JEV's calibrated
    /// confidences; they share the Laya probability gates and local timeouts.
    pub fn model_targets_trusted(&self) -> bool {
        self.trust_model_targets.unwrap_or(matches!(
            self.backend,
            DecisionRouterBackend::Jev | DecisionRouterBackend::Kev
        ))
    }

    pub fn model_conditions_trusted(&self) -> bool {
        self.trust_model_conditions.unwrap_or(matches!(
            self.backend,
            DecisionRouterBackend::Jev | DecisionRouterBackend::Kev
        ))
    }

    pub fn native_choice_probabilities(&self) -> bool {
        matches!(
            self.backend,
            DecisionRouterBackend::Laya
                | DecisionRouterBackend::LlmChoice
                | DecisionRouterBackend::Kev
        )
    }
}

fn default_laya_endpoint() -> String {
    "http://127.0.0.1:43127/v1/systemone".into()
}
fn default_laya_model() -> String {
    "convaiinnovations/laya-typed-decisions".into()
}
fn default_laya_token_env() -> String {
    "POK_LAYA_TOKEN".into()
}
const fn default_laya_min_free_vram_mb() -> u64 {
    2_048
}
const fn default_laya_operation_probability() -> f64 {
    // Lowered again from 0.45: offline+live testing of a combined
    // operation+target gate (crossing operation_probability with
    // target_probability instead of gating them independently at similar
    // thresholds) showed a confident target can safely compensate a
    // clustering-band operation score. `op >= 0.35 and target >= 0.90` beat
    // both the prior independent 0.45/0.50 gate (which deferred ~100% of
    // real clustering-band decisions to the LLM) and a 3-question
    // speculative-committee alternative (58.3% correlation, worse than
    // chance on single-candidate scenes) on a 12-case real+synthetic suite.
    // See docs/decision-router-backends.md for the full evidence.
    0.35
}
const fn default_laya_target_probability() -> f64 {
    // Raised from 0.50 to 0.90 as the other half of the combined gate
    // above: since operation_probability alone is no longer required to
    // clear a high bar, target_probability now carries more of the real
    // discriminative weight, so it needs a correspondingly higher floor.
    0.90
}
const fn default_laya_terminal_probability() -> f64 {
    0.55
}

fn default_judge_endpoint() -> String {
    "http://127.0.0.1:43399/v1/systemone".into()
}
fn default_judge_model() -> String {
    "php-ai/zeiger-0.6b".into()
}
fn default_judge_token_env() -> String {
    "POK_JUDGE_TOKEN".into()
}
const fn default_judge_timeout_ms() -> u64 {
    5_000
}

fn default_decision_router_step_actions() -> usize {
    60
}

fn default_decision_router_search_url() -> String {
    "https://www.google.com/search".into()
}

fn default_decision_router_completion_probability() -> f64 {
    0.60
}

impl ProviderConfig {
    pub fn resolved_data_boundary(&self, provider_name: &str) -> ProviderDataBoundary {
        self.data_boundary.unwrap_or_else(|| {
            if matches!(
                provider_name.to_ascii_lowercase().as_str(),
                "lm_studio" | "ollama"
            ) || reqwest::Url::parse(&self.base_url).is_ok_and(|url| {
                url.host_str().is_some_and(|host| {
                    host.eq_ignore_ascii_case("localhost")
                        || host
                            .parse::<std::net::IpAddr>()
                            .is_ok_and(|ip| ip.is_loopback())
                })
            }) {
                ProviderDataBoundary::LocalDevice
            } else {
                ProviderDataBoundary::ExternalService
            }
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TemperatureSetting {
    Value(f32),
    Mode(TemperatureMode),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemperatureMode {
    ServerDefault,
}

impl TemperatureSetting {
    pub const fn value(self) -> Option<f32> {
        match self {
            Self::Value(value) => Some(value),
            Self::Mode(TemperatureMode::ServerDefault) => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub data_dir: PathBuf,
    #[serde(default)]
    pub diagnostics_dir: PathBuf,
    /// Deprecated legacy fixed working-set setting. Context sizing is automatic.
    #[serde(default)]
    pub prompt_token_target: Option<u64>,
    #[serde(default = "default_compaction_threshold_percent")]
    pub compaction_threshold_percent: u8,
    #[serde(default = "default_vision_max_edge")]
    pub vision_max_edge: u32,
    #[serde(default = "default_visual_history_limit")]
    pub visual_history_limit: usize,
    #[serde(default = "default_uia_element_limit")]
    pub uia_element_limit: usize,
    /// Maximum time spent waiting for optional OCR/UI Automation enrichment
    /// during an ordinary desktop capture.
    #[serde(default = "default_desktop_enrichment_timeout_ms")]
    pub desktop_enrichment_timeout_ms: u64,
    /// Opt-in timeout for a capture that explicitly requests deep enrichment.
    #[serde(default = "default_desktop_deep_enrichment_timeout_ms")]
    pub desktop_deep_enrichment_timeout_ms: u64,
    /// Let the agent work in its own window while the user keeps using
    /// another: choosing a window does not bring it forward, captures render
    /// that window even when covered, and accessibility actions reach it
    /// without the cursor. Real mouse or keyboard input still brings the
    /// window forward, once the user has stopped using the machine.
    #[serde(default = "default_true")]
    pub background_desktop_work: bool,
    #[serde(default = "default_model_target_limit")]
    pub model_target_limit: usize,
    #[serde(default = "default_fusion_iou_threshold")]
    pub fusion_iou_threshold: f32,
    #[serde(default = "default_ocr_containment_threshold")]
    pub ocr_containment_threshold: f32,
    #[serde(default = "default_true")]
    pub annotate_targets: bool,
    #[serde(default = "default_provider")]
    pub default_provider: String,
    #[serde(default)]
    pub default_model: String,
    #[serde(default = "default_max_turns")]
    pub max_turns: u32,
    #[serde(default = "default_command_timeout")]
    pub command_timeout_seconds: u64,
    /// Delay between oversized native text-input batches. Ordinary text is sent
    /// as one batch.
    #[serde(default = "default_input_text_inter_key_pause_ms")]
    pub input_text_inter_key_pause_ms: u64,
    #[serde(default = "default_agent_temperature")]
    pub agent_temperature: TemperatureSetting,
    #[serde(default = "default_providers")]
    pub providers: BTreeMap<String, ProviderConfig>,
    #[serde(default)]
    pub decision_router: DecisionRouterConfig,
}

fn default_decision_router_endpoint() -> String {
    "https://api.typesafe.ai/v1/systemone".into()
}
fn default_decision_router_model() -> String {
    "jev-1.13.0".into()
}
fn default_decision_router_key_env() -> String {
    "TYPESAFE_API_KEY".into()
}
const fn default_decision_router_timeout_ms() -> u64 {
    750
}
const fn default_decision_router_candidate_limit() -> usize {
    30
}
const fn default_decision_router_probability() -> f64 {
    0.30
}
const fn default_decision_router_confidence() -> f64 {
    0.75
}
const fn default_decision_router_operation_confidence() -> f64 {
    0.60
}
const fn default_decision_router_terminal_probability() -> f64 {
    0.60
}
const fn default_decision_router_terminal_confidence() -> f64 {
    0.60
}
const fn default_decision_router_refinement_attempts() -> usize {
    3
}
const fn default_decision_router_stagnant_refinements() -> usize {
    2
}
const fn default_decision_router_refinement_gain() -> f64 {
    0.05
}

fn default_provider() -> String {
    "lm_studio".into()
}
const fn default_max_turns() -> u32 {
    128
}
const fn default_command_timeout() -> u64 {
    30
}
const fn default_input_text_inter_key_pause_ms() -> u64 {
    50
}
const fn default_provider_max_retries() -> u32 {
    3
}
const fn default_agent_temperature() -> TemperatureSetting {
    TemperatureSetting::Value(0.0)
}
const fn default_compaction_threshold_percent() -> u8 {
    80
}
const fn default_vision_max_edge() -> u32 {
    1_280
}
const fn default_visual_history_limit() -> usize {
    1
}
const fn default_uia_element_limit() -> usize {
    500
}
const fn default_desktop_enrichment_timeout_ms() -> u64 {
    2_000
}
const fn default_desktop_deep_enrichment_timeout_ms() -> u64 {
    10_000
}
const fn default_model_target_limit() -> usize {
    120
}
const fn default_fusion_iou_threshold() -> f32 {
    0.1
}
const fn default_ocr_containment_threshold() -> f32 {
    0.6
}
const fn default_true() -> bool {
    true
}

fn default_providers() -> BTreeMap<String, ProviderConfig> {
    BTreeMap::from([
        (
            "lm_studio".into(),
            ProviderConfig {
                protocol: ProviderProtocol::OpenaiCompatible,
                base_url: "http://127.0.0.1:1234/v1".into(),
                data_boundary: Some(ProviderDataBoundary::LocalDevice),
                api_key_env: Some("LM_STUDIO_API_KEY".into()),
                max_retries: default_provider_max_retries(),
            },
        ),
        (
            "ollama".into(),
            ProviderConfig {
                protocol: ProviderProtocol::OpenaiCompatible,
                base_url: "http://127.0.0.1:11434/v1".into(),
                data_boundary: Some(ProviderDataBoundary::LocalDevice),
                api_key_env: Some("OLLAMA_API_KEY".into()),
                max_retries: default_provider_max_retries(),
            },
        ),
        (
            "gemini".into(),
            ProviderConfig {
                protocol: ProviderProtocol::OpenaiCompatible,
                base_url: "https://generativelanguage.googleapis.com/v1beta/openai/".into(),
                data_boundary: Some(ProviderDataBoundary::ExternalService),
                api_key_env: Some("GEMINI_API_KEY".into()),
                max_retries: default_provider_max_retries(),
            },
        ),
        (
            "openai".into(),
            ProviderConfig {
                protocol: ProviderProtocol::OpenaiCompatible,
                base_url: "https://api.openai.com/v1".into(),
                data_boundary: Some(ProviderDataBoundary::ExternalService),
                api_key_env: Some("OPENAI_API_KEY".into()),
                max_retries: default_provider_max_retries(),
            },
        ),
        (
            "openrouter".into(),
            ProviderConfig {
                protocol: ProviderProtocol::OpenaiCompatible,
                base_url: "https://openrouter.ai/api/v1".into(),
                data_boundary: Some(ProviderDataBoundary::ExternalService),
                api_key_env: Some("OPENROUTER_API_KEY".into()),
                max_retries: default_provider_max_retries(),
            },
        ),
        (
            "groq".into(),
            ProviderConfig {
                protocol: ProviderProtocol::OpenaiCompatible,
                base_url: "https://api.groq.com/openai/v1".into(),
                data_boundary: Some(ProviderDataBoundary::ExternalService),
                api_key_env: Some("GROQ_API_KEY".into()),
                max_retries: default_provider_max_retries(),
            },
        ),
        (
            "grok".into(),
            ProviderConfig {
                protocol: ProviderProtocol::OpenaiCompatible,
                base_url: "https://api.x.ai/v1".into(),
                data_boundary: Some(ProviderDataBoundary::ExternalService),
                api_key_env: Some("GROK_API_KEY".into()),
                max_retries: default_provider_max_retries(),
            },
        ),
        (
            "anthropic".into(),
            ProviderConfig {
                protocol: ProviderProtocol::AnthropicMessages,
                base_url: "https://api.anthropic.com/v1".into(),
                data_boundary: Some(ProviderDataBoundary::ExternalService),
                api_key_env: Some("ANTHROPIC_API_KEY".into()),
                max_retries: default_provider_max_retries(),
            },
        ),
    ])
}

impl Default for Config {
    fn default() -> Self {
        let data_dir = default_data_dir();
        Self {
            diagnostics_dir: data_dir.join("diagnostics"),
            data_dir,
            prompt_token_target: None,
            compaction_threshold_percent: default_compaction_threshold_percent(),
            vision_max_edge: default_vision_max_edge(),
            visual_history_limit: default_visual_history_limit(),
            uia_element_limit: default_uia_element_limit(),
            desktop_enrichment_timeout_ms: default_desktop_enrichment_timeout_ms(),
            desktop_deep_enrichment_timeout_ms: default_desktop_deep_enrichment_timeout_ms(),
            background_desktop_work: true,
            model_target_limit: default_model_target_limit(),
            fusion_iou_threshold: default_fusion_iou_threshold(),
            ocr_containment_threshold: default_ocr_containment_threshold(),
            annotate_targets: true,
            default_provider: default_provider(),
            default_model: String::new(),
            max_turns: default_max_turns(),
            command_timeout_seconds: default_command_timeout(),
            input_text_inter_key_pause_ms: default_input_text_inter_key_pause_ms(),
            agent_temperature: default_agent_temperature(),
            providers: default_providers(),
            decision_router: DecisionRouterConfig::default(),
        }
    }
}

impl Config {
    pub fn load(path: Option<&Path>) -> Result<Self> {
        let path = path
            .map(Path::to_path_buf)
            .unwrap_or_else(default_config_path);
        let mut config = if path.exists() {
            toml::from_str::<Self>(&std::fs::read_to_string(&path)?)
                .map_err(|error| PokError::Config(format!("{}: {error}", path.display())))?
        } else {
            Self::default()
        };
        if config.data_dir.as_os_str().is_empty() {
            config.data_dir = default_data_dir();
        }
        if config.diagnostics_dir.as_os_str().is_empty() {
            config.diagnostics_dir = config.data_dir.join("diagnostics");
        } else if config.diagnostics_dir.is_relative() && path.exists() {
            config.diagnostics_dir = path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(&config.diagnostics_dir);
        }
        for (name, provider_config) in default_providers() {
            config.providers.entry(name).or_insert(provider_config);
        }
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if !self.providers.contains_key(&self.default_provider) {
            return Err(PokError::Config(format!(
                "default provider {:?} is not configured",
                self.default_provider
            )));
        }
        if self.max_turns == 0 {
            return Err(PokError::Config(
                "max_turns must be greater than zero".into(),
            ));
        }
        if self.decision_router.enabled {
            match self.decision_router.backend {
                DecisionRouterBackend::Off => {
                    return Err(PokError::Config(
                        "decision_router cannot be enabled with backend=off".into(),
                    ));
                }
                DecisionRouterBackend::Jev
                    if !self.decision_router.endpoint.starts_with("https://") =>
                {
                    return Err(PokError::Config(
                        "JEV decision_router.endpoint must use https".into(),
                    ));
                }
                DecisionRouterBackend::Laya
                | DecisionRouterBackend::LlmChoice
                | DecisionRouterBackend::Kev => {
                    let endpoint = reqwest::Url::parse(self.decision_router.active_endpoint())
                        .map_err(|_| {
                            PokError::Config(
                                "Laya endpoint must be a valid loopback HTTP URL".into(),
                            )
                        })?;
                    let loopback = endpoint.host_str().is_some_and(|host| {
                        host.eq_ignore_ascii_case("localhost")
                            || host
                                .parse::<std::net::IpAddr>()
                                .is_ok_and(|ip| ip.is_loopback())
                    });
                    if endpoint.scheme() != "http" || !loopback {
                        return Err(PokError::Config(
                            "Laya endpoint must use HTTP on localhost or a loopback IP".into(),
                        ));
                    }
                }
                _ => {}
            }
            if !self.decision_router.search_url.starts_with("https://") {
                return Err(PokError::Config(
                    "decision_router.search_url must use https".into(),
                ));
            }
            if self.decision_router.judge.enabled {
                let endpoint =
                    reqwest::Url::parse(&self.decision_router.judge.endpoint).map_err(|_| {
                        PokError::Config("Judge endpoint must be a valid loopback HTTP URL".into())
                    })?;
                let loopback = endpoint.host_str().is_some_and(|host| {
                    host.eq_ignore_ascii_case("localhost")
                        || host
                            .parse::<std::net::IpAddr>()
                            .is_ok_and(|ip| ip.is_loopback())
                });
                if endpoint.scheme() != "http" || !loopback {
                    return Err(PokError::Config(
                        "Judge endpoint must use HTTP on localhost or a loopback IP".into(),
                    ));
                }
                if !(100..=30_000).contains(&self.decision_router.judge.timeout_ms) {
                    return Err(PokError::Config(
                        "Judge timeout_ms must be between 100 and 30000".into(),
                    ));
                }
            }
            if !(100..=10_000).contains(&self.decision_router.timeout_ms)
                || !(1..=250).contains(&self.decision_router.max_candidates)
                || !(1..=60).contains(&self.decision_router.max_step_actions)
                || !(0.0..=1.0).contains(&self.decision_router.min_selected_probability)
                || !(0.0..=1.0).contains(&self.decision_router.min_operation_confidence)
                || !(0.0..=1.0).contains(&self.decision_router.min_confidence)
                || !(0.0..=1.0).contains(&self.decision_router.min_terminal_probability)
                || !(0.0..=1.0).contains(&self.decision_router.min_terminal_confidence)
                || !(0.0..=1.0).contains(&self.decision_router.min_completion_probability)
                || !(0.0..=1.0).contains(&self.decision_router.laya.min_operation_probability)
                || !(0.0..=1.0).contains(&self.decision_router.laya.min_target_probability)
                || !(1..=10).contains(&self.decision_router.max_refinement_attempts)
                || !(1..=5).contains(&self.decision_router.max_stagnant_refinements)
                || !(0.0..=1.0).contains(&self.decision_router.min_refinement_confidence_gain)
            {
                return Err(PokError::Config(
                    "decision router timeout, candidate/action limits, or thresholds are invalid"
                        .into(),
                ));
            }
        }
        if self.input_text_inter_key_pause_ms > 250 {
            return Err(PokError::Config(
                "input_text_inter_key_pause_ms must be between 0 and 250".into(),
            ));
        }
        if self
            .agent_temperature
            .value()
            .is_some_and(|value| !value.is_finite() || !(0.0..=2.0).contains(&value))
        {
            return Err(PokError::Config(
                "agent_temperature must be between 0 and 2, or \"server_default\"".into(),
            ));
        }
        if !(50..=95).contains(&self.compaction_threshold_percent) {
            return Err(PokError::Config(
                "compaction_threshold_percent must be between 50 and 95".into(),
            ));
        }
        if self.vision_max_edge < 640 {
            return Err(PokError::Config(
                "vision_max_edge must be at least 640".into(),
            ));
        }
        if self.visual_history_limit == 0 {
            return Err(PokError::Config(
                "visual_history_limit must be greater than zero".into(),
            ));
        }
        if self.uia_element_limit == 0 || self.model_target_limit == 0 {
            return Err(PokError::Config(
                "UIA and model target limits must be greater than zero".into(),
            ));
        }
        if !(250..=10_000).contains(&self.desktop_enrichment_timeout_ms) {
            return Err(PokError::Config(
                "desktop_enrichment_timeout_ms must be between 250 and 10000".into(),
            ));
        }
        if self.desktop_deep_enrichment_timeout_ms < self.desktop_enrichment_timeout_ms
            || self.desktop_deep_enrichment_timeout_ms > 30_000
        {
            return Err(PokError::Config(
                "desktop_deep_enrichment_timeout_ms must be at least the normal timeout and no more than 30000".into(),
            ));
        }
        if !(0.0..=1.0).contains(&self.fusion_iou_threshold)
            || !(0.0..=1.0).contains(&self.ocr_containment_threshold)
        {
            return Err(PokError::Config(
                "grounding overlap thresholds must be between 0 and 1".into(),
            ));
        }
        for (name, provider) in &self.providers {
            reqwest::Url::parse(&provider.base_url)
                .map_err(|error| PokError::Config(format!("provider {name}: {error}")))?;
            if provider.max_retries > 10 {
                return Err(PokError::Config(format!(
                    "provider {name}: max_retries must be between 0 and 10"
                )));
            }
        }
        Ok(())
    }

    pub fn provider(&self, name: Option<&str>) -> Result<&ProviderConfig> {
        let name = name.unwrap_or(&self.default_provider);
        self.providers
            .get(name)
            .ok_or_else(|| PokError::Config(format!("unknown provider {name:?}")))
    }
}

pub fn default_data_dir() -> PathBuf {
    ProjectDirs::from("ai", "POK-Ai", "POK-Ai").map_or_else(
        || PathBuf::from(".pok-ai"),
        |dirs| dirs.data_local_dir().to_path_buf(),
    )
}

pub fn default_config_path() -> PathBuf {
    default_data_dir().join("pok-ai.toml")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_retry_limit_defaults_to_three() {
        let provider: ProviderConfig = toml::from_str(
            r#"
protocol = "openai_compatible"
base_url = "https://openrouter.ai/api/v1"
"#,
        )
        .unwrap();
        assert_eq!(provider.max_retries, 3);
    }

    #[test]
    fn provider_data_boundary_is_explicit_with_safe_legacy_fallbacks() {
        let local: ProviderConfig = toml::from_str(
            r#"
protocol = "openai_compatible"
base_url = "http://localhost:1234/v1"
"#,
        )
        .unwrap();
        assert_eq!(
            local.resolved_data_boundary("custom"),
            ProviderDataBoundary::LocalDevice
        );

        let remote: ProviderConfig = toml::from_str(
            r#"
protocol = "openai_compatible"
base_url = "https://models.example.test/v1"
"#,
        )
        .unwrap();
        assert_eq!(
            remote.resolved_data_boundary("custom"),
            ProviderDataBoundary::ExternalService
        );
        assert_eq!(
            remote.resolved_data_boundary("lm_studio"),
            ProviderDataBoundary::LocalDevice
        );
    }

    #[test]
    fn provider_retry_limit_can_be_disabled_but_is_bounded() {
        let mut config = Config::default();
        config.providers.get_mut("openrouter").unwrap().max_retries = 0;
        assert!(config.validate().is_ok());

        config.providers.get_mut("openrouter").unwrap().max_retries = 11;
        assert!(config.validate().is_err());
    }

    #[test]
    fn temperature_can_use_a_value_or_server_default() {
        let numeric: Config = toml::from_str("agent_temperature = 0.2").unwrap();
        assert_eq!(numeric.agent_temperature.value(), Some(0.2));

        let omitted: Config = toml::from_str(r#"agent_temperature = "server_default""#).unwrap();
        assert_eq!(omitted.agent_temperature.value(), None);

        let defaulted: Config = toml::from_str("").unwrap();
        assert_eq!(defaulted.agent_temperature.value(), Some(0.0));
    }

    #[test]
    fn temperature_validation_rejects_values_outside_portable_range() {
        let config = Config {
            agent_temperature: TemperatureSetting::Value(2.1),
            ..Config::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn text_input_batch_pause_keeps_compatible_default_and_is_bounded() {
        let defaulted: Config = toml::from_str("").unwrap();
        assert_eq!(defaulted.input_text_inter_key_pause_ms, 50);

        let configured: Config = toml::from_str("input_text_inter_key_pause_ms = 25").unwrap();
        assert_eq!(configured.input_text_inter_key_pause_ms, 25);
        assert!(configured.validate().is_ok());

        let invalid: Config = toml::from_str("input_text_inter_key_pause_ms = 251").unwrap();
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn legacy_fixed_prompt_target_is_accepted_only_for_migration_notice() {
        let legacy: Config = toml::from_str("prompt_token_target = 16000").unwrap();
        assert_eq!(legacy.prompt_token_target, Some(16_000));
        assert!(Config::default().prompt_token_target.is_none());
    }

    #[test]
    fn laya_router_is_restricted_to_loopback_http() {
        let mut config = Config::default();
        config.decision_router.enabled = true;
        config.decision_router.backend = DecisionRouterBackend::Laya;
        config.decision_router.laya.endpoint = "http://127.0.0.1:43127/v1/systemone".into();
        assert!(config.validate().is_ok());

        config.decision_router.laya.endpoint = "https://example.test/v1/systemone".into();
        assert!(config.validate().is_err());
    }

    #[test]
    fn disabled_router_preserves_the_all_llm_path() {
        let config = DecisionRouterConfig {
            backend: DecisionRouterBackend::Laya,
            ..DecisionRouterConfig::default()
        };
        assert_eq!(config.active_backend(), DecisionRouterBackend::Off);
        let config = DecisionRouterConfig {
            enabled: true,
            ..config
        };
        assert_eq!(config.active_backend(), DecisionRouterBackend::Laya);
    }

    #[test]
    fn laya_combined_gate_defaults_let_a_confident_target_offset_a_clustering_band_operation() {
        // Real+synthetic testing (docs/decision-router-backends.md) found a
        // confident target compensates a clustering-band operation score
        // better than gating both independently at similar thresholds.
        let laya = LayaRouterConfig::default();
        assert_eq!(laya.min_operation_probability, 0.35);
        assert_eq!(laya.min_target_probability, 0.90);
        // A representative clustering-band decision (operation_probability
        // in the 0.44-0.57 band that used to defer 100% of the time, target
        // certain) must now clear the gate.
        assert!(0.45 >= laya.min_operation_probability);
        assert!(1.0 >= laya.min_target_probability);
        // A representative wrong-candidate decision with a merely-moderate
        // target (not near-certain) must still be rejected.
        assert!(!(0.45 >= laya.min_operation_probability && 0.60 >= laya.min_target_probability));
    }
}
