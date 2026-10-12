use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::{Duration, SystemTime},
};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{Result, brain::ModelInfo, memory::atomic_write};

const CATALOG_URL: &str = "https://models.dev/api.json";
const CATALOG_TTL: Duration = Duration::from_secs(60 * 60);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ContextSource {
    #[serde(alias = "provider")]
    ProviderLoaded,
    ProviderMax,
    ModelsDev,
    Manual,
    FallbackUnknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextBudget {
    pub provider: String,
    pub model: String,
    pub context_window_tokens: u64,
    pub threshold_percent: u8,
    pub compact_at_tokens: u64,
    pub post_compaction_target_tokens: u64,
    pub output_reserve_tokens: u64,
    pub source: ContextSource,
    pub known: bool,
}

impl ContextBudget {
    pub fn new(
        provider: String,
        model: String,
        window: u64,
        threshold: u8,
        source: ContextSource,
    ) -> Self {
        let threshold = threshold.clamp(50, 95);
        let output_reserve = 4_096_u64.min(window / 4);
        let safety_reserve = (window / 20).max(2_048).min(window / 4);
        let percent_limit = window.saturating_mul(u64::from(threshold)) / 100;
        let compact_at = percent_limit
            .min(window.saturating_sub(output_reserve + safety_reserve))
            .max(4_000);
        Self {
            provider,
            model,
            context_window_tokens: window,
            threshold_percent: threshold,
            compact_at_tokens: compact_at,
            post_compaction_target_tokens: (compact_at / 2).max(4_000),
            output_reserve_tokens: output_reserve,
            known: source != ContextSource::FallbackUnknown,
            source,
        }
    }

    pub fn reserve_output_tokens(&mut self, tokens: u64) {
        self.output_reserve_tokens = tokens.clamp(1_024, self.context_window_tokens / 4);
        let safety_reserve = (self.context_window_tokens / 20)
            .max(2_048)
            .min(self.context_window_tokens / 4);
        let percent_limit = self
            .context_window_tokens
            .saturating_mul(u64::from(self.threshold_percent))
            / 100;
        self.compact_at_tokens = percent_limit
            .min(
                self.context_window_tokens
                    .saturating_sub(self.output_reserve_tokens + safety_reserve),
            )
            .max(4_000);
        self.post_compaction_target_tokens = (self.compact_at_tokens / 2).max(4_000);
    }
}

/// Select a latency-oriented model working set independently from the hard
/// context-window compaction boundary. The fixed cost includes system
/// instructions, tool schemas, and the active-task reminder. Recent history is
/// allowed to scale with the model window, but is deliberately capped so a
/// large local context does not make every turn expensive to prefill.
pub fn automatic_working_set_target(budget: &ContextBudget, fixed_tokens: u64) -> u64 {
    // Scale the working set with the model's window but keep it
    // latency-bounded: a model with a very large window gets room to keep the
    // files and tool results a multi-step analysis is working from instead of
    // evicting them and re-reading the same inputs every turn.
    let recent_tokens = (budget.context_window_tokens.saturating_mul(8) / 100).clamp(4_000, 32_000);
    let percentage_cap = budget.context_window_tokens.saturating_mul(35) / 100;
    let safety_cap = budget
        .compact_at_tokens
        .saturating_sub(budget.output_reserve_tokens);
    let base_cap = (budget.context_window_tokens / 16).clamp(32_000, 96_000);
    let cap = base_cap.min(percentage_cap).min(safety_cap).max(4_000);
    fixed_tokens
        .saturating_add(recent_tokens)
        .min(cap)
        .max(4_000)
}

pub fn automatic_recent_tail_target(working_set_target: u64) -> u64 {
    (working_set_target / 2).clamp(4_000, 8_000)
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelContextOverride {
    pub context_window_tokens: Option<u64>,
    pub threshold_percent: Option<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextSettings {
    #[serde(default = "default_threshold")]
    pub threshold_percent: u8,
    #[serde(default)]
    pub models: BTreeMap<String, ModelContextOverride>,
}

const fn default_threshold() -> u8 {
    80
}
impl Default for ContextSettings {
    fn default() -> Self {
        Self {
            threshold_percent: default_threshold(),
            models: BTreeMap::new(),
        }
    }
}

impl ContextSettings {
    pub fn path(data_dir: &Path) -> PathBuf {
        data_dir.join("context-settings.json")
    }
    pub fn load(data_dir: &Path) -> Self {
        std::fs::read(Self::path(data_dir))
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .unwrap_or_default()
    }
    pub fn save(&self, data_dir: &Path) -> Result<()> {
        atomic_write(&Self::path(data_dir), &serde_json::to_vec_pretty(self)?)
    }
    pub fn key(provider: &str, model: &str) -> String {
        format!(
            "{}::{}",
            provider.to_ascii_lowercase(),
            model.to_ascii_lowercase()
        )
    }
}

pub async fn resolve_context_budget(
    data_dir: &Path,
    provider: &str,
    model: &str,
    provider_context: Option<u64>,
    provider_max_context: Option<u64>,
    legacy_fallback: u64,
) -> ContextBudget {
    let settings = ContextSettings::load(data_dir);
    let override_value = settings.models.get(&ContextSettings::key(provider, model));
    let threshold = override_value
        .and_then(|value| value.threshold_percent)
        .unwrap_or(settings.threshold_percent);
    if let Some(window) = override_value
        .and_then(|value| value.context_window_tokens)
        .filter(|value| *value >= 4_000)
    {
        return ContextBudget::new(
            provider.into(),
            model.into(),
            window,
            threshold,
            ContextSource::Manual,
        );
    }
    if let Some(window) = provider_context.filter(|value| *value >= 4_000) {
        return ContextBudget::new(
            provider.into(),
            model.into(),
            window,
            threshold,
            ContextSource::ProviderLoaded,
        );
    }
    if let Some(window) = provider_max_context.filter(|value| *value >= 4_000) {
        return ContextBudget::new(
            provider.into(),
            model.into(),
            window,
            threshold,
            ContextSource::ProviderMax,
        );
    }
    if let Some(window) = catalog_context(data_dir, provider, model).await {
        return ContextBudget::new(
            provider.into(),
            model.into(),
            window,
            threshold,
            ContextSource::ModelsDev,
        );
    }
    ContextBudget::new(
        provider.into(),
        model.into(),
        legacy_fallback.max(4_000),
        threshold,
        ContextSource::FallbackUnknown,
    )
}

async fn catalog_context(data_dir: &Path, provider: &str, model: &str) -> Option<u64> {
    let catalog = models_dev_catalog(data_dir).await?;
    lookup_catalog(&catalog, provider, model)
}

/// The models.dev catalog, cached under `data_dir` and refreshed hourly.
async fn models_dev_catalog(data_dir: &Path) -> Option<Value> {
    let cache_path = data_dir.join("models-dev-cache.json");
    let fresh = cache_path
        .metadata()
        .ok()
        .and_then(|meta| meta.modified().ok())
        .and_then(|modified| SystemTime::now().duration_since(modified).ok())
        .is_some_and(|age| age < CATALOG_TTL);
    let mut catalog = std::fs::read(&cache_path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok());
    if !fresh {
        let fetched =
            match tokio::time::timeout(Duration::from_secs(5), reqwest::get(CATALOG_URL)).await {
                Ok(Ok(response)) => match response.error_for_status() {
                    Ok(response) => response.json::<Value>().await.ok(),
                    Err(_) => None,
                },
                _ => None,
            };
        if let Some(value) = fetched.filter(|value| value.is_object()) {
            let _ = atomic_write(&cache_path, &serde_json::to_vec(&value).ok()?);
            catalog = Some(value);
        }
    }
    catalog
}

fn catalog_provider_id(provider: &str) -> String {
    match provider.to_ascii_lowercase().as_str() {
        "gemini" => "google".into(),
        "grok" => "xai".into(),
        other => other.into(),
    }
}

fn lookup_catalog(catalog: &Value, provider: &str, model: &str) -> Option<u64> {
    let provider_id = catalog_provider_id(provider);
    let models = catalog.get(&provider_id)?.get("models")?.as_object()?;
    models
        .iter()
        .find(|(id, _)| id.eq_ignore_ascii_case(model))
        .and_then(|(_, entry)| entry.pointer("/limit/context")?.as_u64())
        .filter(|value| *value > 0)
}

/// Fill the capability fields a provider's own model list leaves out.
/// OpenCode's `/models` returns only ids; models.dev (which OpenCode and most
/// providers publish) carries the context limit, vision/tools/reasoning
/// flags, and the reasoning levels a model accepts. Values the provider
/// itself reported are kept: this only fills gaps.
pub async fn enrich_model_capabilities(data_dir: &Path, provider: &str, models: &mut [ModelInfo]) {
    if models.is_empty() {
        return;
    }
    let Some(catalog) = models_dev_catalog(data_dir).await else {
        return;
    };
    let provider_id = catalog_provider_id(provider);
    let Some(entries) = catalog
        .get(&provider_id)
        .and_then(|entry| entry.get("models"))
        .and_then(Value::as_object)
    else {
        return;
    };
    for model in models {
        if let Some(entry) = entries
            .iter()
            .find(|(id, _)| id.eq_ignore_ascii_case(&model.id))
            .map(|(_, entry)| entry)
        {
            apply_catalog_entry(model, entry);
        }
    }
}

/// Every model a models.dev provider lists, as `ModelInfo` with the
/// capabilities the catalog reports. Empty when the catalog is unavailable.
pub async fn catalog_model_infos(data_dir: &Path, provider: &str) -> Vec<ModelInfo> {
    let Some(catalog) = models_dev_catalog(data_dir).await else {
        return Vec::new();
    };
    let provider_id = catalog_provider_id(provider);
    let Some(entries) = catalog
        .get(&provider_id)
        .and_then(|entry| entry.get("models"))
        .and_then(Value::as_object)
    else {
        return Vec::new();
    };
    entries
        .iter()
        .map(|(id, entry)| {
            let mut model = ModelInfo {
                id: id.clone(),
                context_length: None,
                loaded_context_length: None,
                max_context_length: None,
                vision: None,
                tool_use: None,
                reasoning: None,
                supported_parameters: Vec::new(),
                reasoning_efforts: Vec::new(),
                reasoning_default: None,
                loaded: None,
            };
            apply_catalog_entry(&mut model, entry);
            model
        })
        .collect()
}

/// Fill one model's missing capability fields from its catalog entry.
fn apply_catalog_entry(model: &mut ModelInfo, entry: &Value) {
    let context = entry
        .pointer("/limit/context")
        .and_then(Value::as_u64)
        .filter(|value| *value > 0);
    if model.context_length.is_none() {
        model.context_length = context;
    }
    if model.max_context_length.is_none() {
        model.max_context_length = context;
    }
    if model.vision.is_none() {
        model.vision = entry
            .pointer("/modalities/input")
            .and_then(Value::as_array)
            .map(|modalities| {
                modalities
                    .iter()
                    .any(|modality| modality.as_str() == Some("image"))
            });
    }
    if model.tool_use.is_none() {
        model.tool_use = entry.get("tool_call").and_then(Value::as_bool);
    }
    if model.reasoning.is_none() {
        model.reasoning = entry.get("reasoning").and_then(Value::as_bool);
    }
    if model.reasoning_efforts.is_empty() {
        let (efforts, default) = reasoning_options(entry);
        model.reasoning_efforts = efforts;
        if model.reasoning_default.is_none() {
            model.reasoning_default = default;
        }
    }
}

/// The reasoning levels a models.dev entry offers: named efforts when the
/// model has them, an off/on switch when it only has a toggle, and "off"
/// alongside named levels when a toggle says reasoning can be disabled.
fn reasoning_options(entry: &Value) -> (Vec<String>, Option<String>) {
    let mut toggle = false;
    let mut efforts = Vec::<String>::new();
    for option in entry
        .get("reasoning_options")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        match option.get("type").and_then(Value::as_str) {
            Some("toggle") => toggle = true,
            Some("effort") => {
                for value in option
                    .get("values")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_str)
                {
                    let effort = crate::brain::canonical_reasoning_effort(value.trim());
                    if !effort.is_empty() && !efforts.iter().any(|known| known == effort) {
                        efforts.push(effort.to_owned());
                    }
                }
            }
            _ => {}
        }
    }
    if efforts.is_empty() && toggle {
        return (vec!["off".into(), "on".into()], Some("on".into()));
    }
    if !efforts.is_empty() && toggle && !efforts.iter().any(|effort| effort == "off") {
        efforts.push("off".into());
    }
    efforts.sort_by_key(|effort| crate::brain::reasoning_rank(effort));
    (efforts, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn budget_reserves_output_and_safety_space() {
        let mut budget = ContextBudget::new(
            "p".into(),
            "m".into(),
            100_000,
            80,
            ContextSource::ProviderLoaded,
        );
        assert_eq!(budget.compact_at_tokens, 80_000);
        assert_eq!(budget.post_compaction_target_tokens, 40_000);
        budget.reserve_output_tokens(16_384);
        assert_eq!(budget.output_reserve_tokens, 16_384);
        assert!(budget.compact_at_tokens <= 80_000);
    }

    #[test]
    fn automatic_working_set_scales_but_stays_latency_bounded() {
        let target = |window, fixed| {
            automatic_working_set_target(
                &ContextBudget::new(
                    "p".into(),
                    "m".into(),
                    window,
                    80,
                    ContextSource::ProviderLoaded,
                ),
                fixed,
            )
        };
        assert_eq!(target(32_000, 8_000), 11_200);
        assert_eq!(target(70_000, 8_000), 13_600);
        assert_eq!(target(100_000, 8_000), 16_000);
        assert_eq!(target(200_000, 8_000), 24_000);
        // A very large window keeps a proportionally larger recent working set
        // so multi-file work is not re-read every turn.
        assert_eq!(target(1_000_000, 15_000), 47_000);
        assert_eq!(automatic_recent_tail_target(16_000), 8_000);
    }
    #[tokio::test]
    async fn sparse_model_capabilities_are_filled_from_the_catalog() {
        let temp = tempfile::tempdir().unwrap();
        let catalog = serde_json::json!({
            "opencode-go": {"models": {
                "longcat-2.5-preview-free": {
                    "limit": {"context": 1_000_000, "output": 131_072},
                    "modalities": {"input": ["text", "image"]},
                    "tool_call": true,
                    "reasoning": true,
                    "reasoning_options": [{"type": "toggle"}]
                },
                "deepseek-v4-flash": {
                    "limit": {"context": 1_000_000},
                    "modalities": {"input": ["text"]},
                    "tool_call": true,
                    "reasoning": true,
                    "reasoning_options": [{"type": "effort", "values": ["low", "high", "max"]}]
                }
            }}
        });
        std::fs::write(
            temp.path().join("models-dev-cache.json"),
            serde_json::to_vec(&catalog).unwrap(),
        )
        .unwrap();
        let mut models = vec![
            test_model_info("longcat-2.5-preview-free"),
            test_model_info("deepseek-v4-flash"),
        ];
        // The catalog fills gaps but never overwrites provider-reported values.
        models[1].context_length = Some(123);
        models[1].reasoning_efforts = vec!["custom".into()];
        enrich_model_capabilities(temp.path(), "opencode-go", &mut models).await;
        assert_eq!(models[0].context_length, Some(1_000_000));
        assert_eq!(models[0].max_context_length, Some(1_000_000));
        assert_eq!(models[0].vision, Some(true));
        assert_eq!(models[0].tool_use, Some(true));
        assert_eq!(models[0].reasoning, Some(true));
        assert_eq!(models[0].reasoning_efforts, vec!["off", "on"]);
        assert_eq!(models[0].reasoning_default.as_deref(), Some("on"));
        assert_eq!(models[1].context_length, Some(123));
        assert_eq!(models[1].reasoning_efforts, vec!["custom"]);
        assert_eq!(models[1].reasoning_default, None);
        assert_eq!(models[1].tool_use, Some(true));
    }

    fn test_model_info(id: &str) -> ModelInfo {
        ModelInfo {
            id: id.into(),
            context_length: None,
            loaded_context_length: None,
            max_context_length: None,
            vision: None,
            tool_use: None,
            reasoning: None,
            supported_parameters: Vec::new(),
            reasoning_efforts: Vec::new(),
            reasoning_default: None,
            loaded: None,
        }
    }

    #[test]
    fn catalog_lookup_is_provider_aware() {
        let data =
            serde_json::json!({"google":{"models":{"gemini-x":{"limit":{"context":123456}}}}});
        assert_eq!(lookup_catalog(&data, "gemini", "GEMINI-X"), Some(123456));
    }

    #[tokio::test]
    async fn loaded_context_wins_over_theoretical_provider_maximum() {
        let temp = tempfile::tempdir().unwrap();
        let budget = resolve_context_budget(
            temp.path(),
            "lm_studio",
            "local-model",
            Some(100_000),
            Some(262_144),
            16_000,
        )
        .await;
        assert_eq!(budget.context_window_tokens, 100_000);
        assert_eq!(budget.compact_at_tokens, 80_000);
        assert_eq!(budget.source, ContextSource::ProviderLoaded);
    }
}
