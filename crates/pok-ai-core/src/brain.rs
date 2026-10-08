use std::{collections::BTreeMap, pin::Pin, sync::Arc};

use async_trait::async_trait;
use base64::Engine;
use futures::{Stream, StreamExt};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{
    PokError, Result,
    config::{ProviderConfig, ProviderProtocol},
};

pub type BrainStream = Pin<Box<dyn Stream<Item = Result<BrainEvent>> + Send>>;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessageContent {
    Text { text: String },
    ImagePng { base64: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct BrainMessage {
    pub role: String,
    pub content: Vec<MessageContent>,
    #[serde(default)]
    pub origin: MessageOrigin,
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<CompletedToolCall>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MessageOrigin {
    #[default]
    Unknown,
    System,
    UserInput,
    UserGuidance,
    Assistant,
    ToolResult,
    ToolImage,
    HistorySummary,
    SystemReminder,
    /// Query-specific archive, memory, skill, and capability recall. Replaced
    /// on the next user request instead of becoming permanent prompt history.
    RetrievedContext,
}

impl BrainMessage {
    pub fn text(role: impl Into<String>, text: impl Into<String>) -> Self {
        let role = role.into();
        let origin = match role.as_str() {
            "system" => MessageOrigin::System,
            "user" => MessageOrigin::UserInput,
            "assistant" => MessageOrigin::Assistant,
            "tool" => MessageOrigin::ToolResult,
            _ => MessageOrigin::Unknown,
        };
        Self {
            role,
            content: vec![MessageContent::Text { text: text.into() }],
            origin,
            tool_call_id: None,
            tool_calls: Vec::new(),
        }
    }

    pub fn text_with_origin(
        role: impl Into<String>,
        text: impl Into<String>,
        origin: MessageOrigin,
    ) -> Self {
        let mut message = Self::text(role, text);
        message.origin = origin;
        message
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct BrainTool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct BrainRequest {
    pub model: String,
    pub messages: Vec<BrainMessage>,
    pub tools: Vec<BrainTool>,
    pub temperature: Option<f32>,
    /// Provider output limit. `None` asks providers that support omission to
    /// use their native context/output allowance.
    pub max_tokens: Option<u32>,
    pub seed: Option<u64>,
    #[serde(default)]
    pub reasoning_effort: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CompletedToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ModelInfo {
    pub id: String,
    /// Effective context for a live model instance, falling back to the
    /// provider-advertised maximum when no instance is loaded.
    pub context_length: Option<u64>,
    #[serde(default)]
    pub loaded_context_length: Option<u64>,
    #[serde(default)]
    pub max_context_length: Option<u64>,
    pub vision: Option<bool>,
    pub tool_use: Option<bool>,
    pub reasoning: Option<bool>,
    #[serde(default)]
    pub supported_parameters: Vec<String>,
    #[serde(default)]
    pub reasoning_efforts: Vec<String>,
    #[serde(default)]
    pub reasoning_default: Option<String>,
    pub loaded: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BrainEvent {
    TextDelta {
        text: String,
    },
    ReasoningDelta {
        text: String,
    },
    ToolCall {
        call: CompletedToolCall,
    },
    MalformedToolCall {
        name: String,
        raw_arguments: String,
        error: String,
    },
    Usage {
        prompt_tokens: u64,
        completion_tokens: u64,
    },
    Finished {
        reason: Option<String>,
    },
}

#[async_trait]
pub trait Brain: Send + Sync {
    async fn list_models(&self) -> Result<Vec<String>>;
    fn max_retries(&self) -> u32 {
        3
    }
    fn requires_max_tokens(&self) -> bool {
        false
    }
    async fn model_info(&self) -> Result<Vec<ModelInfo>> {
        Ok(self
            .list_models()
            .await?
            .into_iter()
            .map(|id| ModelInfo {
                id,
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
            })
            .collect())
    }
    fn stream(&self, request: BrainRequest) -> BrainStream;
}

/// Totals for every primary-model request made through one `CountingBrain`,
/// including bounded helpers, compaction, and curation, not only agent turns.
#[derive(Debug, Default)]
pub struct BrainUsageCounter {
    requests: std::sync::atomic::AtomicU64,
    prompt_tokens: std::sync::atomic::AtomicU64,
    completion_tokens: std::sync::atomic::AtomicU64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct BrainUsageSnapshot {
    pub requests: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
}

impl BrainUsageCounter {
    pub fn snapshot(&self) -> BrainUsageSnapshot {
        use std::sync::atomic::Ordering;
        BrainUsageSnapshot {
            requests: self.requests.load(Ordering::Relaxed),
            prompt_tokens: self.prompt_tokens.load(Ordering::Relaxed),
            completion_tokens: self.completion_tokens.load(Ordering::Relaxed),
        }
    }
}

impl BrainUsageSnapshot {
    pub fn since(self, earlier: Self) -> Self {
        Self {
            requests: self.requests.saturating_sub(earlier.requests),
            prompt_tokens: self.prompt_tokens.saturating_sub(earlier.prompt_tokens),
            completion_tokens: self
                .completion_tokens
                .saturating_sub(earlier.completion_tokens),
        }
    }
}

/// Transparent `Brain` wrapper that counts provider requests and reported usage.
pub struct CountingBrain {
    inner: Arc<dyn Brain>,
    counter: Arc<BrainUsageCounter>,
}

impl CountingBrain {
    pub fn new(inner: Arc<dyn Brain>) -> Self {
        Self {
            inner,
            counter: Arc::new(BrainUsageCounter::default()),
        }
    }

    pub fn counter(&self) -> Arc<BrainUsageCounter> {
        self.counter.clone()
    }
}

#[async_trait]
impl Brain for CountingBrain {
    async fn list_models(&self) -> Result<Vec<String>> {
        self.inner.list_models().await
    }
    fn max_retries(&self) -> u32 {
        self.inner.max_retries()
    }
    fn requires_max_tokens(&self) -> bool {
        self.inner.requires_max_tokens()
    }
    async fn model_info(&self) -> Result<Vec<ModelInfo>> {
        self.inner.model_info().await
    }
    fn stream(&self, request: BrainRequest) -> BrainStream {
        use std::sync::atomic::Ordering;
        self.counter.requests.fetch_add(1, Ordering::Relaxed);
        let counter = self.counter.clone();
        Box::pin(self.inner.stream(request).inspect(move |event| {
            if let Ok(BrainEvent::Usage {
                prompt_tokens,
                completion_tokens,
            }) = event
            {
                counter
                    .prompt_tokens
                    .fetch_add(*prompt_tokens, Ordering::Relaxed);
                counter
                    .completion_tokens
                    .fetch_add(*completion_tokens, Ordering::Relaxed);
            }
        }))
    }
}

pub static SAVED_KEYS: parking_lot::Mutex<BTreeMap<String, String>> =
    parking_lot::Mutex::new(BTreeMap::new());

pub fn set_api_key(name: impl Into<String>, value: impl Into<String>) {
    SAVED_KEYS.lock().insert(name.into(), value.into());
}

pub fn get_api_key(name: &str) -> Option<String> {
    SAVED_KEYS
        .lock()
        .get(name)
        .cloned()
        .or_else(|| std::env::var(name).ok())
}

pub fn from_provider(config: &ProviderConfig) -> Result<Arc<dyn Brain>> {
    let api_key = config.api_key_env.as_deref().and_then(get_api_key);
    match config.protocol {
        ProviderProtocol::OpenaiCompatible => Ok(Arc::new(
            OpenAiCompatibleBrain::new(config.base_url.clone(), api_key)?
                .with_max_retries(config.max_retries),
        )),
        ProviderProtocol::AnthropicMessages => Ok(Arc::new(
            AnthropicBrain::new(config.base_url.clone(), api_key)?
                .with_max_retries(config.max_retries),
        )),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum LocalModelPhase {
    Checking,
    ServerOffline,
    Unloaded,
    Unloading,
    Loading,
    Loaded,
    Unsupported,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct LocalModelRuntimeStatus {
    pub provider: String,
    pub model: String,
    pub phase: LocalModelPhase,
    pub detail: Option<String>,
    pub instance_ids: Vec<String>,
    #[serde(default)]
    pub context_length: Option<u64>,
}

impl LocalModelRuntimeStatus {
    fn stable(
        provider: &str,
        model: &str,
        instance_ids: Vec<String>,
        context_length: Option<u64>,
    ) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
            phase: if instance_ids.is_empty() {
                LocalModelPhase::Unloaded
            } else {
                LocalModelPhase::Loaded
            },
            detail: None,
            instance_ids,
            context_length,
        }
    }

    fn with_phase(provider: &str, model: &str, phase: LocalModelPhase, detail: String) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
            phase,
            detail: Some(detail),
            instance_ids: Vec::new(),
            context_length: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LocalProviderKind {
    LmStudio,
    Ollama,
}

fn local_provider_kind(provider: &str) -> Option<LocalProviderKind> {
    match provider.to_ascii_lowercase().as_str() {
        "lm_studio" => Some(LocalProviderKind::LmStudio),
        "ollama" => Some(LocalProviderKind::Ollama),
        _ => None,
    }
}

fn local_server_root(base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    trimmed
        .strip_suffix("/v1")
        .unwrap_or(trimmed)
        .trim_end_matches('/')
        .to_owned()
}

fn ollama_model_matches(running: &str, selected: &str) -> bool {
    if running.eq_ignore_ascii_case(selected) {
        return true;
    }
    let running_without_latest = running.strip_suffix(":latest").unwrap_or(running);
    let selected_without_latest = selected.strip_suffix(":latest").unwrap_or(selected);
    running_without_latest.eq_ignore_ascii_case(selected_without_latest)
}

fn local_client(timeout: std::time::Duration) -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .timeout(timeout)
        .build()
        .map_err(|error| PokError::Provider(error.to_string()))
}

fn authorize_local(
    request: reqwest::RequestBuilder,
    config: &ProviderConfig,
) -> reqwest::RequestBuilder {
    if let Some(key) = config.api_key_env.as_deref().and_then(get_api_key) {
        request.bearer_auth(key)
    } else {
        request
    }
}

async fn local_response_json(request: reqwest::RequestBuilder, operation: &str) -> Result<Value> {
    let response = request
        .send()
        .await
        .map_err(|error| PokError::Provider(format!("{operation}: {error}")))?;
    let status = response.status();
    let text = response
        .text()
        .await
        .map_err(|error| PokError::Provider(format!("{operation}: {error}")))?;
    if !status.is_success() {
        return Err(PokError::Provider(format!(
            "{operation} returned {status}: {}",
            text.trim()
        )));
    }
    serde_json::from_str(&text)
        .map_err(|error| PokError::Provider(format!("{operation} returned invalid JSON: {error}")))
}

pub async fn local_model_status(
    provider: &str,
    config: &ProviderConfig,
    model: &str,
) -> LocalModelRuntimeStatus {
    let Some(kind) = local_provider_kind(provider) else {
        return LocalModelRuntimeStatus::with_phase(
            provider,
            model,
            LocalModelPhase::Unsupported,
            "model lifecycle management is unavailable for this provider".into(),
        );
    };
    if model.trim().is_empty() {
        return LocalModelRuntimeStatus::with_phase(
            provider,
            model,
            LocalModelPhase::Unloaded,
            "select a model".into(),
        );
    }
    let client = match local_client(std::time::Duration::from_secs(5)) {
        Ok(client) => client,
        Err(error) => {
            return LocalModelRuntimeStatus::with_phase(
                provider,
                model,
                LocalModelPhase::Error,
                error.to_string(),
            );
        }
    };
    let root = local_server_root(&config.base_url);
    let request = match kind {
        LocalProviderKind::LmStudio => client.get(format!("{root}/api/v1/models")),
        LocalProviderKind::Ollama => client.get(format!("{root}/api/ps")),
    };
    let body = match local_response_json(authorize_local(request, config), "model status").await {
        Ok(body) => body,
        Err(error) => {
            return LocalModelRuntimeStatus::with_phase(
                provider,
                model,
                LocalModelPhase::ServerOffline,
                error.to_string(),
            );
        }
    };

    match kind {
        LocalProviderKind::LmStudio => {
            let Some(models) = body.get("models").and_then(Value::as_array) else {
                return LocalModelRuntimeStatus::with_phase(
                    provider,
                    model,
                    LocalModelPhase::Unsupported,
                    "LM Studio did not return the native v1 models format".into(),
                );
            };
            let Some(found) = models.iter().find(|item| {
                item.get("key")
                    .and_then(Value::as_str)
                    .is_some_and(|key| key.eq_ignore_ascii_case(model))
            }) else {
                return LocalModelRuntimeStatus::with_phase(
                    provider,
                    model,
                    LocalModelPhase::Error,
                    format!("model {model:?} was not found by LM Studio"),
                );
            };
            let instances = found
                .get("loaded_instances")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|instance| instance.get("id").and_then(Value::as_str))
                .map(str::to_owned)
                .collect();
            let context_length = found
                .get("loaded_instances")
                .and_then(Value::as_array)
                .and_then(|instances| instances.first())
                .and_then(|instance| instance.pointer("/config/context_length"))
                .and_then(Value::as_u64);
            LocalModelRuntimeStatus::stable(provider, model, instances, context_length)
        }
        LocalProviderKind::Ollama => {
            let Some(models) = body.get("models").and_then(Value::as_array) else {
                return LocalModelRuntimeStatus::with_phase(
                    provider,
                    model,
                    LocalModelPhase::Unsupported,
                    "Ollama did not return the /api/ps models format".into(),
                );
            };
            let instances = models
                .iter()
                .filter_map(|item| {
                    item.get("name")
                        .or_else(|| item.get("model"))
                        .and_then(Value::as_str)
                })
                .filter(|name| ollama_model_matches(name, model))
                .map(str::to_owned)
                .collect();
            LocalModelRuntimeStatus::stable(provider, model, instances, None)
        }
    }
}

pub async fn load_local_model(
    provider: &str,
    config: &ProviderConfig,
    model: &str,
) -> Result<LocalModelRuntimeStatus> {
    load_local_model_with_context(provider, config, model, None).await
}

pub async fn load_local_model_with_context(
    provider: &str,
    config: &ProviderConfig,
    model: &str,
    context_length: Option<u64>,
) -> Result<LocalModelRuntimeStatus> {
    let current = local_model_status(provider, config, model).await;
    load_local_model_from_status_with_context(provider, config, model, current, context_length)
        .await
}

pub async fn load_local_model_from_status(
    provider: &str,
    config: &ProviderConfig,
    model: &str,
    current: LocalModelRuntimeStatus,
) -> Result<LocalModelRuntimeStatus> {
    load_local_model_from_status_with_context(provider, config, model, current, None).await
}

pub async fn load_local_model_from_status_with_context(
    provider: &str,
    config: &ProviderConfig,
    model: &str,
    current: LocalModelRuntimeStatus,
    context_length: Option<u64>,
) -> Result<LocalModelRuntimeStatus> {
    let kind = local_provider_kind(provider).ok_or_else(|| {
        PokError::Provider(format!(
            "model lifecycle management is unavailable for provider {provider:?}"
        ))
    })?;
    let current = if current.phase == LocalModelPhase::Loaded
        && context_length.is_some_and(|desired| current.context_length != Some(desired))
    {
        unload_local_model_from_status(provider, config, model, current).await?
    } else if current.phase == LocalModelPhase::Loaded {
        return Ok(current);
    } else {
        current
    };
    if matches!(
        current.phase,
        LocalModelPhase::ServerOffline | LocalModelPhase::Unsupported
    ) {
        return Err(PokError::Provider(
            current
                .detail
                .unwrap_or_else(|| "local model server is unavailable".into()),
        ));
    }
    let client = local_client(std::time::Duration::from_secs(600))?;
    let root = local_server_root(&config.base_url);
    let request = match kind {
        LocalProviderKind::LmStudio => {
            let mut payload = json!({"model": model, "echo_load_config": true});
            if let Some(context_length) = context_length {
                payload["context_length"] = json!(context_length);
            }
            client
                .post(format!("{root}/api/v1/models/load"))
                .json(&payload)
        }
        LocalProviderKind::Ollama => client.post(format!("{root}/api/generate")).json(&json!({
            "model": model,
            "prompt": "",
            "stream": false,
            "keep_alive": -1
        })),
    };
    let response = local_response_json(authorize_local(request, config), "model load").await?;
    let echoed_context = response
        .pointer("/load_config/context_length")
        .and_then(Value::as_u64);
    let mut loaded = local_model_status(provider, config, model).await;
    loaded.context_length = loaded.context_length.or(echoed_context);
    if loaded.phase != LocalModelPhase::Loaded {
        return Err(PokError::Provider(format!(
            "{model:?} did not report as loaded after the load request"
        )));
    }
    Ok(loaded)
}

pub async fn unload_local_model(
    provider: &str,
    config: &ProviderConfig,
    model: &str,
) -> Result<LocalModelRuntimeStatus> {
    let current = local_model_status(provider, config, model).await;
    unload_local_model_from_status(provider, config, model, current).await
}

pub async fn unload_local_model_from_status(
    provider: &str,
    config: &ProviderConfig,
    model: &str,
    current: LocalModelRuntimeStatus,
) -> Result<LocalModelRuntimeStatus> {
    let kind = local_provider_kind(provider).ok_or_else(|| {
        PokError::Provider(format!(
            "model lifecycle management is unavailable for provider {provider:?}"
        ))
    })?;
    if current.phase == LocalModelPhase::Unloaded {
        return Ok(current);
    }
    if current.phase != LocalModelPhase::Loaded {
        return Err(PokError::Provider(current.detail.unwrap_or_else(|| {
            "could not determine loaded model state".into()
        })));
    }
    let client = local_client(std::time::Duration::from_secs(60))?;
    let root = local_server_root(&config.base_url);
    match kind {
        LocalProviderKind::LmStudio => {
            for instance_id in current.instance_ids {
                let request = client
                    .post(format!("{root}/api/v1/models/unload"))
                    .json(&json!({"instance_id": instance_id}));
                let _ =
                    local_response_json(authorize_local(request, config), "model unload").await?;
            }
        }
        LocalProviderKind::Ollama => {
            let request = client.post(format!("{root}/api/generate")).json(&json!({
                "model": model,
                "prompt": "",
                "stream": false,
                "keep_alive": 0
            }));
            let _ = local_response_json(authorize_local(request, config), "model unload").await?;
        }
    }
    let unloaded = local_model_status(provider, config, model).await;
    if unloaded.phase != LocalModelPhase::Unloaded {
        return Err(PokError::Provider(format!(
            "{model:?} still reports as loaded after the unload request"
        )));
    }
    Ok(unloaded)
}

#[derive(Clone)]
pub struct OpenAiCompatibleBrain {
    client: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
    max_retries: u32,
}

impl OpenAiCompatibleBrain {
    pub fn new(base_url: String, api_key: Option<String>) -> Result<Self> {
        reqwest::Url::parse(&base_url).map_err(|error| PokError::Config(error.to_string()))?;
        Ok(Self {
            client: reqwest::Client::new(),
            base_url: base_url.trim_end_matches('/').into(),
            api_key,
            max_retries: 3,
        })
    }

    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    fn authorized(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.api_key {
            Some(key) => request.bearer_auth(key),
            None => request,
        }
    }

    fn host(&self) -> Option<String> {
        reqwest::Url::parse(&self.base_url)
            .ok()?
            .host_str()
            .map(str::to_ascii_lowercase)
    }

    fn is_openrouter(&self) -> bool {
        self.host()
            .is_some_and(|host| host.ends_with("openrouter.ai"))
    }

    async fn standard_model_info(&self) -> Result<Vec<ModelInfo>> {
        let response = self
            .authorized(self.client.get(format!("{}/models", self.base_url)))
            .send()
            .await
            .map_err(provider_error)?;
        let response = require_success(response).await?;
        let body: Value = response.json().await.map_err(provider_error)?;
        let mut models = body
            .get("data")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(openai_model_info)
            .collect::<Vec<_>>();
        // xAI lists reasoning levels only in its `language-models` catalog.
        if self.host().is_some_and(|host| host.ends_with("x.ai")) {
            self.merge_xai_reasoning(&mut models).await;
        }
        Ok(models)
    }

    async fn merge_xai_reasoning(&self, models: &mut [ModelInfo]) {
        let Ok(response) = self
            .authorized(
                self.client
                    .get(format!("{}/language-models", self.base_url)),
            )
            .send()
            .await
        else {
            return;
        };
        let Ok(body) = response.json::<Value>().await else {
            return;
        };
        for entry in body
            .get("models")
            .or_else(|| body.get("data"))
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let (efforts, default) = reasoning_metadata(entry);
            if efforts.is_empty() {
                continue;
            }
            let ids = std::iter::once(entry.get("id"))
                .chain(
                    entry
                        .get("aliases")
                        .and_then(Value::as_array)
                        .into_iter()
                        .flatten()
                        .map(Some),
                )
                .flatten()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>();
            for model in models
                .iter_mut()
                .filter(|model| ids.contains(&model.id.as_str()))
            {
                model.reasoning = Some(true);
                model.reasoning_efforts = efforts.clone();
                model.reasoning_default = default.clone();
            }
        }
    }
}

#[async_trait]
impl Brain for OpenAiCompatibleBrain {
    fn max_retries(&self) -> u32 {
        self.max_retries
    }

    async fn list_models(&self) -> Result<Vec<String>> {
        let response = self
            .authorized(self.client.get(format!("{}/models", self.base_url)))
            .send()
            .await
            .map_err(provider_error)?;
        let response = require_success(response).await?;
        let body: Value = response.json().await.map_err(provider_error)?;
        Ok(body
            .get("data")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|item| item.get("id").and_then(Value::as_str).map(str::to_owned))
            .collect())
    }

    async fn model_info(&self) -> Result<Vec<ModelInfo>> {
        let Some(server_root) = self.base_url.strip_suffix("/v1") else {
            return self.standard_model_info().await;
        };
        let response = self
            .authorized(self.client.get(format!("{server_root}/api/v1/models")))
            .send()
            .await
            .map_err(provider_error)?;
        if !response.status().is_success() {
            return self.standard_model_info().await;
        }
        let body: Value = response.json().await.map_err(provider_error)?;
        Ok(body
            .get("models")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|model| {
                let id = model.get("key")?.as_str()?.to_owned();
                let capabilities = model.get("capabilities").unwrap_or(&Value::Null);
                let loaded_instances = model.get("loaded_instances").and_then(Value::as_array);
                let loaded_context = loaded_instances
                    .and_then(|instances| instances.first())
                    .and_then(|instance| instance.get("config"))
                    .and_then(|config| config.get("context_length"))
                    .and_then(Value::as_u64);
                let (reasoning_efforts, reasoning_default) = reasoning_metadata(model);
                Some(ModelInfo {
                    id,
                    // LM Studio exposes both the model's theoretical maximum and the
                    // context actually allocated to a loaded instance. The latter is
                    // the useful limit for planning a live session.
                    context_length: loaded_context
                        .or_else(|| model.get("max_context_length").and_then(Value::as_u64)),
                    loaded_context_length: loaded_context,
                    max_context_length: model.get("max_context_length").and_then(Value::as_u64),
                    vision: capabilities.get("vision").and_then(Value::as_bool),
                    tool_use: capabilities
                        .get("trained_for_tool_use")
                        .and_then(Value::as_bool),
                    reasoning: capabilities
                        .get("reasoning")
                        .and_then(Value::as_bool)
                        .or_else(|| {
                            capabilities
                                .get("reasoning")
                                .is_some_and(Value::is_object)
                                .then_some(true)
                        }),
                    supported_parameters: model
                        .get("supported_parameters")
                        .and_then(Value::as_array)
                        .into_iter()
                        .flatten()
                        .filter_map(|item| item.as_str().map(str::to_owned))
                        .collect(),
                    reasoning_efforts,
                    reasoning_default,
                    loaded: loaded_instances.map(|instances| !instances.is_empty()),
                })
            })
            .collect())
    }

    fn stream(&self, request: BrainRequest) -> BrainStream {
        let this = self.clone();
        Box::pin(async_stream::try_stream! {
            let body = if this.is_openrouter() {
                openrouter_request(&request)
            } else {
                openai_request(&request)
            };
            let response = this.authorized(this.client.post(format!("{}/chat/completions", this.base_url)))
                .json(&body).send().await.map_err(provider_error)?;
            let response = require_success(response).await?;
            let mut bytes = response.bytes_stream();
            let mut buffer = String::new();
            let mut calls: BTreeMap<u64, ToolCallAccumulator> = BTreeMap::new();
            while let Some(chunk) = bytes.next().await {
                let chunk = chunk.map_err(provider_error)?;
                buffer.push_str(&String::from_utf8_lossy(&chunk));
                while let Some(end) = buffer.find("\n\n") {
                    let event = buffer[..end].to_owned();
                    buffer.drain(..end + 2);
                    for line in event.lines().filter_map(|line| line.strip_prefix("data:")) {
                        let data = line.trim();
                        if data == "[DONE]" { continue; }
                        let value: Value = serde_json::from_str(data)?;
                        if let Some(error) = openai_stream_error(&value) {
                            Err(error)?;
                        }
                        if let Some(usage) = value.get("usage") {
                            yield BrainEvent::Usage {
                                prompt_tokens: usage.get("prompt_tokens").and_then(Value::as_u64).unwrap_or(0),
                                completion_tokens: usage.get("completion_tokens").and_then(Value::as_u64).unwrap_or(0),
                            };
                        }
                        for choice in value.get("choices").and_then(Value::as_array).into_iter().flatten() {
                            let delta = choice.get("delta").unwrap_or(&Value::Null);
                            if let Some(text) = delta.get("content").and_then(Value::as_str) {
                                if !text.is_empty() { yield BrainEvent::TextDelta { text: text.into() }; }
                            }
                            if let Some(text) = delta.get("reasoning").or_else(|| delta.get("reasoning_content")).and_then(Value::as_str) {
                                if !text.is_empty() { yield BrainEvent::ReasoningDelta { text: text.into() }; }
                            }
                            for call in delta.get("tool_calls").and_then(Value::as_array).into_iter().flatten() {
                                let index = call.get("index").and_then(Value::as_u64).unwrap_or(0);
                                let accumulator = calls.entry(index).or_default();
                                if let Some(id) = call.get("id").and_then(Value::as_str) { accumulator.id.push_str(id); }
                                if let Some(function) = call.get("function") {
                                    if let Some(name) = function.get("name").and_then(Value::as_str) { accumulator.name.push_str(name); }
                                    if let Some(arguments) = function.get("arguments").and_then(Value::as_str) { accumulator.arguments.push_str(arguments); }
                                }
                            }
                            if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
                                for (_, call) in std::mem::take(&mut calls) {
                                    match serde_json::from_str(&call.arguments) {
                                        Ok(arguments) => yield BrainEvent::ToolCall { call: CompletedToolCall {
                                            id: if call.id.is_empty() { uuid::Uuid::new_v4().to_string() } else { call.id },
                                            name: call.name, arguments,
                                        }},
                                        Err(error) => yield BrainEvent::MalformedToolCall {
                                            name: call.name, raw_arguments: call.arguments, error: error.to_string(),
                                        },
                                    }
                                }
                                yield BrainEvent::Finished { reason: Some(reason.into()) };
                            }
                        }
                    }
                }
            }
        })
    }
}

fn openai_model_info(model: &Value) -> Option<ModelInfo> {
    let id = model.get("id")?.as_str()?.to_owned();
    let (reasoning_efforts, reasoning_default) = reasoning_metadata(model);
    let input_modalities = model
        .pointer("/architecture/input_modalities")
        .or_else(|| model.get("input_modalities"))
        .and_then(Value::as_array);
    let supported_parameters = model.get("supported_parameters").and_then(Value::as_array);
    let supports_parameter = |name: &str| {
        supported_parameters.is_some_and(|parameters| {
            parameters
                .iter()
                .any(|parameter| parameter.as_str() == Some(name))
        })
    };
    Some(ModelInfo {
        id,
        context_length: model
            .get("context_length")
            .or_else(|| model.get("max_context_length"))
            .and_then(Value::as_u64),
        loaded_context_length: None,
        max_context_length: model
            .get("max_context_length")
            .or_else(|| model.get("context_length"))
            .and_then(Value::as_u64),
        vision: model
            .pointer("/capabilities/vision")
            .and_then(Value::as_bool)
            .or_else(|| {
                input_modalities.map(|modalities| {
                    modalities
                        .iter()
                        .any(|modality| modality.as_str() == Some("image"))
                })
            }),
        tool_use: model
            .pointer("/capabilities/trained_for_tool_use")
            .and_then(Value::as_bool)
            .or_else(|| supported_parameters.map(|_| supports_parameter("tools"))),
        reasoning: model
            .pointer("/capabilities/reasoning")
            .and_then(Value::as_bool)
            .or_else(|| {
                model
                    .pointer("/capabilities/reasoning")
                    .is_some_and(Value::is_object)
                    .then_some(true)
            })
            .or_else(|| supported_parameters.map(|_| supports_parameter("reasoning"))),
        supported_parameters: supported_parameters
            .into_iter()
            .flatten()
            .filter_map(|item| item.as_str().map(str::to_owned))
            .collect(),
        reasoning_efforts,
        reasoning_default,
        loaded: None,
    })
}

/// A model's reasoning levels and default, from whichever catalog shape the
/// provider uses:
/// - LM Studio `capabilities.reasoning.{allowed_options, default}` (`off`/`on`
///   or named levels);
/// - OpenRouter `reasoning.{supported_efforts, default_effort, mandatory,
///   default_enabled}`;
/// - xAI `capabilities.{reasoning_effort, default_reasoning_effort}`;
/// - OpenAI-style `reasoning_efforts` / `supported_reasoning_efforts`.
///
/// A model that can only switch reasoning on or off gets `off`/`on`.
fn reasoning_metadata(model: &Value) -> (Vec<String>, Option<String>) {
    let options = model
        .pointer("/capabilities/reasoning/allowed_options")
        .or_else(|| model.pointer("/capabilities/reasoning/efforts"))
        .or_else(|| model.pointer("/capabilities/reasoning_effort"))
        .or_else(|| model.pointer("/reasoning/supported_efforts"))
        .or_else(|| model.get("reasoning_efforts"))
        .or_else(|| model.get("supported_reasoning_efforts"))
        .and_then(Value::as_array);
    let mut efforts = Vec::new();
    for effort in options
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::trim)
        .filter(|effort| !effort.is_empty())
    {
        let effort = canonical_reasoning_effort(effort);
        if !efforts.iter().any(|known| known == effort) {
            efforts.push(effort.to_owned());
        }
    }
    // OpenRouter: a reasoning model without named levels is an on/off switch,
    // and one whose reasoning is mandatory cannot be switched off.
    let openrouter = model.get("reasoning").filter(|value| value.is_object());
    let mandatory = openrouter
        .and_then(|reasoning| reasoning.get("mandatory"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if let Some(reasoning) = openrouter
        && efforts.is_empty()
        && !mandatory
    {
        efforts = vec!["off".into(), "on".into()];
        let enabled = reasoning
            .get("default_enabled")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        return (efforts, Some(if enabled { "on" } else { "off" }.into()));
    }
    if mandatory {
        efforts.retain(|effort| effort != "none" && effort != "off");
    }
    efforts.sort_by_key(|effort| reasoning_rank(effort));
    let default = model
        .pointer("/capabilities/reasoning/default")
        .or_else(|| model.pointer("/capabilities/default_reasoning_effort"))
        .or_else(|| model.pointer("/reasoning/default_effort"))
        .or_else(|| model.get("reasoning_default"))
        .or_else(|| model.get("default_reasoning_effort"))
        .and_then(Value::as_str)
        .map(str::trim)
        .map(canonical_reasoning_effort)
        .filter(|effort| efforts.iter().any(|allowed| allowed == effort))
        .map(str::to_owned);
    (efforts, default)
}

/// Display order, from no reasoning to the most.
fn reasoning_rank(effort: &str) -> u8 {
    match effort {
        "off" => 0,
        "none" => 1,
        // The model's own default, offered next to switching it off.
        "on" => 2,
        "minimal" => 3,
        "low" => 4,
        "medium" => 5,
        "high" => 6,
        "xhigh" => 7,
        "max" => 8,
        _ => 9,
    }
}

fn canonical_reasoning_effort(effort: &str) -> &str {
    match effort {
        // Catalog spellings of the same level.
        "extra_high" | "extra-high" | "x-high" => "xhigh",
        "maximum" => "max",
        "disabled" => "off",
        "enabled" => "on",
        other => other,
    }
}

#[derive(Default)]
struct ToolCallAccumulator {
    id: String,
    name: String,
    arguments: String,
}

fn openai_request(request: &BrainRequest) -> Value {
    let scalar_tool_types = model_requires_scalar_tool_types(&request.model);
    let messages: Vec<Value> = request
        .messages
        .iter()
        .map(|message| {
            let content: Vec<Value> = message
                .content
                .iter()
                .filter_map(|part| match part {
                    MessageContent::Text { text } if !text.trim().is_empty() => {
                        Some(json!({"type": "text", "text": text}))
                    }
                    MessageContent::Text { .. } => None,
                    MessageContent::ImagePng { base64 } => Some(json!({
                        "type": "image_url", "image_url": {"url": format!("data:image/png;base64,{base64}")}
                    })),
                })
                .collect();
            let tool_only_assistant = content.is_empty()
                && message.role == "assistant"
                && !message.tool_calls.is_empty();
            let mut value = json!({"role": message.role});
            // LM Studio's OpenAI-compatible tool history omits `content` for a
            // tool-only assistant message. It rejects both null and empty text.
            if message.role == "tool" {
                let text = message
                    .content
                    .iter()
                    .filter_map(|part| match part {
                        MessageContent::Text { text } if !text.trim().is_empty() => {
                            Some(text.as_str())
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                value["content"] = Value::String(text);
            } else if !tool_only_assistant {
                value["content"] = Value::Array(content);
            }
            if let Some(id) = &message.tool_call_id {
                value["tool_call_id"] = json!(id);
            }
            if !message.tool_calls.is_empty() {
                value["tool_calls"] =
                    json!(message.tool_calls.iter().map(|call| json!({
                "id": call.id, "type": "function",
                "function": {"name": call.name, "arguments": call.arguments.to_string()}
            })).collect::<Vec<_>>());
            }
            value
        })
        .collect();
    let mut body = json!({
        "model": request.model,
        "messages": messages,
        "stream": true,
        "stream_options": {"include_usage": true}
    });
    if let Some(max_tokens) = request.max_tokens {
        body["max_tokens"] = json!(max_tokens);
    }
    if !request.tools.is_empty() {
        body["tools"] = json!(
            request
                .tools
                .iter()
                .map(|tool| json!({"type": "function", "function": {
                    "name": tool.name,
                    "description": tool.description,
                    // Some native Jinja tool templates (notably Nemotron 3) apply the
                    // `string` filter to every extra JSON-Schema value. Schemars emits
                    // `"default": null` for optional fields, and LM Studio's Jinja
                    // renderer rejects converting that NullValue. Null schema entries
                    // carry no useful constraint, so omit them at the provider boundary.
                    "parameters": if scalar_tool_types {
                        scalarize_schema_types(&tool_parameters_schema(&tool.input_schema))
                    } else {
                        tool_parameters_schema(&tool.input_schema)
                    }
                }}))
                .collect::<Vec<_>>()
        );
        body["tool_choice"] = json!("auto");
    }
    if let Some(temperature) = request.temperature {
        body["temperature"] = json!(temperature);
    }
    if let Some(seed) = request.seed {
        body["seed"] = json!(seed);
    }
    // `on` means the model's own default reasoning, so nothing is sent; `off`
    // is the OpenAI-style `none`.
    match request.reasoning_effort.as_deref() {
        None | Some("on") => {}
        Some("off") => body["reasoning_effort"] = json!("none"),
        Some(effort) => body["reasoning_effort"] = json!(effort),
    }
    body
}

/// OpenRouter takes its unified `reasoning` object; sending the OpenAI-style
/// `reasoning_effort` as well is rejected for some models.
fn openrouter_request(request: &BrainRequest) -> Value {
    let mut body = openai_request(request);
    if let Some(object) = body.as_object_mut() {
        object.remove("reasoning_effort");
    }
    match request.reasoning_effort.as_deref() {
        None => {}
        Some("on") => body["reasoning"] = json!({"enabled": true}),
        Some("off") => body["reasoning"] = json!({"enabled": false}),
        Some(effort) => body["reasoning"] = json!({"effort": effort}),
    }
    body
}

fn model_requires_scalar_tool_types(model: &str) -> bool {
    let model = model.to_ascii_lowercase();
    model.contains("seed-oss")
}

fn scalarize_schema_types(value: &Value) -> Value {
    fn primitive_type(value: &Value) -> Option<&'static str> {
        match value {
            Value::String(_) => Some("string"),
            Value::Bool(_) => Some("boolean"),
            Value::Number(number) if number.is_i64() || number.is_u64() => Some("integer"),
            Value::Number(_) => Some("number"),
            Value::Array(_) => Some("array"),
            Value::Object(_) => Some("object"),
            Value::Null => None,
        }
    }

    let Some(source) = value.as_object() else {
        return json!({"type": primitive_type(value).unwrap_or("object")});
    };
    let mut schema = source.clone();

    if let Some(types) = schema.get("type").and_then(Value::as_array) {
        let scalar = types
            .iter()
            .find(|schema_type| schema_type.as_str() != Some("null"))
            .cloned()
            .unwrap_or_else(|| json!("object"));
        schema.insert("type".into(), scalar);
    }

    for key in ["properties", "patternProperties", "dependentSchemas"] {
        if let Some(properties) = schema.get_mut(key).and_then(Value::as_object_mut) {
            for property in properties.values_mut() {
                *property = scalarize_schema_types(property);
            }
        }
    }
    for key in [
        "items",
        "contains",
        "not",
        "if",
        "then",
        "else",
        "propertyNames",
        "additionalProperties",
        "unevaluatedProperties",
    ] {
        if let Some(child) = schema.get_mut(key) {
            if child.is_object() || child.is_boolean() {
                *child = scalarize_schema_types(child);
            }
        }
    }
    for key in ["allOf", "anyOf", "oneOf", "prefixItems"] {
        if let Some(branches) = schema.get_mut(key).and_then(Value::as_array_mut) {
            for branch in branches {
                *branch = scalarize_schema_types(branch);
            }
        }
    }

    if !schema.get("type").is_some_and(Value::is_string) {
        let inferred = if schema.contains_key("properties") {
            Some("object")
        } else if schema.contains_key("items") || schema.contains_key("prefixItems") {
            Some("array")
        } else {
            schema
                .get("enum")
                .and_then(Value::as_array)
                .and_then(|values| values.iter().find_map(primitive_type))
                .or_else(|| schema.get("const").and_then(primitive_type))
                .or_else(|| {
                    ["anyOf", "oneOf", "allOf"].iter().find_map(|key| {
                        schema
                            .get(*key)
                            .and_then(Value::as_array)
                            .into_iter()
                            .flatten()
                            .filter_map(|branch| branch.get("type").and_then(Value::as_str))
                            .find(|schema_type| *schema_type != "null")
                    })
                })
        }
        .unwrap_or("object")
        .to_owned();
        schema.insert("type".into(), json!(inferred));
    }
    Value::Object(schema)
}

pub(crate) fn reference_free_schema(value: &Value) -> Value {
    fn resolve(value: &Value, root: &Value, stack: &mut Vec<String>, depth: usize) -> Value {
        if depth > 32 {
            return json!({});
        }
        if let Some(reference) = value.get("$ref").and_then(Value::as_str) {
            if stack.iter().any(|item| item == reference) {
                return json!({});
            }
            if let Some(pointer) = reference.strip_prefix('#') {
                if let Some(target) = root.pointer(pointer) {
                    stack.push(reference.to_owned());
                    let resolved = resolve(target, root, stack, depth + 1);
                    stack.pop();
                    return resolved;
                }
            }
            return json!({});
        }
        match value {
            Value::Object(object) => Value::Object(
                object
                    .iter()
                    .filter(|(key, _)| *key != "$defs" && *key != "definitions")
                    .filter(|(_, child)| !child.is_null())
                    .map(|(key, child)| (key.clone(), resolve(child, root, stack, depth + 1)))
                    .collect(),
            ),
            Value::Array(values) => Value::Array(
                values
                    .iter()
                    .map(|child| resolve(child, root, stack, depth + 1))
                    .collect(),
            ),
            _ => value.clone(),
        }
    }
    openai_compatible_schema(&resolve(value, value, &mut Vec::new(), 0))
}

/// Convert standards-compliant JSON Schema into the conservative subset used by
/// OpenAI-compatible local runtimes. JSON Schema permits `true` as shorthand for
/// an unconstrained schema, but LM Studio's automatic tool parser currently only
/// accepts object-form schemas.
/// Root `parameters` object for a tool. Strict servers (and MCP tools without
/// arguments) require `properties` to be present on the root object.
fn tool_parameters_schema(value: &Value) -> Value {
    let mut schema = openai_compatible_schema(value);
    if !schema.is_object() {
        schema = json!({});
    }
    if let Some(object) = schema.as_object_mut() {
        object.entry("type").or_insert_with(|| json!("object"));
        if object.get("type") == Some(&json!("object")) {
            object.entry("properties").or_insert_with(|| json!({}));
        }
    }
    schema
}

fn openai_compatible_schema(value: &Value) -> Value {
    match value {
        Value::Bool(true) => json!({"type": "object"}),
        Value::Bool(false) => json!({"not": {}}),
        Value::Object(object) => {
            let mut compatible = serde_json::Map::new();
            for (key, child) in object {
                if child.is_null() {
                    continue;
                }
                let child = match key.as_str() {
                    "properties" | "patternProperties" | "$defs" | "definitions"
                    | "dependentSchemas" => schema_map(child),
                    "items" | "contains" | "not" | "if" | "then" | "else" | "propertyNames" => {
                        openai_compatible_schema(child)
                    }
                    "additionalProperties" | "unevaluatedProperties"
                        if child == &Value::Bool(true) =>
                    {
                        // `true` is the JSON Schema default, so omitting it keeps
                        // the same meaning without upsetting strict local parsers.
                        continue;
                    }
                    "additionalProperties" | "unevaluatedProperties" if child.is_object() => {
                        openai_compatible_schema(child)
                    }
                    "allOf" | "anyOf" | "oneOf" | "prefixItems" => schema_array(child),
                    _ => value_without_nulls(child),
                };
                compatible.insert(key.clone(), child);
            }
            Value::Object(compatible)
        }
        _ => value.clone(),
    }
}

fn schema_map(value: &Value) -> Value {
    let Some(object) = value.as_object() else {
        return value_without_nulls(value);
    };
    Value::Object(
        object
            .iter()
            .filter(|(_, value)| !value.is_null())
            .map(|(key, value)| (key.clone(), openai_compatible_schema(value)))
            .collect(),
    )
}

fn schema_array(value: &Value) -> Value {
    let Some(values) = value.as_array() else {
        return value_without_nulls(value);
    };
    Value::Array(
        values
            .iter()
            .filter(|value| !value.is_null())
            .map(openai_compatible_schema)
            .collect(),
    )
}

fn value_without_nulls(value: &Value) -> Value {
    match value {
        Value::Object(object) => Value::Object(
            object
                .iter()
                .filter(|(_, value)| !value.is_null())
                .map(|(key, value)| (key.clone(), value_without_nulls(value)))
                .collect(),
        ),
        Value::Array(values) => Value::Array(
            values
                .iter()
                .filter(|value| !value.is_null())
                .map(value_without_nulls)
                .collect(),
        ),
        _ => value.clone(),
    }
}

fn openai_stream_error(value: &Value) -> Option<PokError> {
    let error = value.get("error")?;
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .or_else(|| error.as_str())
        .unwrap_or("OpenAI-compatible provider returned an unknown streaming error");
    let code = error.get("code").and_then(|code| {
        code.as_str()
            .map(str::to_owned)
            .or_else(|| code.as_i64().map(|code| code.to_string()))
    });
    let detail = code.map_or_else(|| message.to_owned(), |code| format!("{code}: {message}"));
    let error_type = error
        .pointer("/metadata/error_type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let status_code = error
        .get("code")
        .and_then(Value::as_u64)
        .and_then(|code| u16::try_from(code).ok());
    let capacity_error = status_code.is_some_and(capacity_status_code)
        || matches!(
            error_type,
            "rate_limit_exceeded" | "provider_overloaded" | "provider_unavailable"
        );
    let retryable = status_code.is_some_and(transient_status_code)
        || capacity_error
        || matches!(error_type, "server" | "timeout");
    Some(if retryable {
        PokError::ProviderTransient {
            message: detail,
            retry_after_ms: None,
            retry_until_cancelled: capacity_error,
        }
    } else {
        PokError::Provider(detail)
    })
}

async fn require_success(response: reqwest::Response) -> Result<reqwest::Response> {
    if response.status().is_success() {
        return Ok(response);
    }
    let status = response.status();
    let retry_after_ms = retry_after_ms(response.headers());
    let body = response
        .text()
        .await
        .unwrap_or_else(|error| format!("<could not read provider response: {error}>"));
    let message = format!("HTTP {status}: {body}");
    if transient_status_code(status.as_u16()) {
        Err(PokError::ProviderTransient {
            message,
            retry_after_ms,
            retry_until_cancelled: capacity_status_code(status.as_u16()),
        })
    } else {
        Err(PokError::Provider(message))
    }
}

fn transient_status_code(status: u16) -> bool {
    matches!(status, 408 | 429 | 500 | 502 | 503 | 504 | 529)
}

fn capacity_status_code(status: u16) -> bool {
    matches!(status, 429 | 503 | 529)
}

fn retry_after_ms(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    const MAX_RETRY_AFTER_SECONDS: u64 = 600;
    headers
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse::<f64>()
        .ok()
        .filter(|seconds| seconds.is_finite() && *seconds > 0.0)
        .map(|seconds| {
            (seconds * 1_000.0)
                .ceil()
                .min((MAX_RETRY_AFTER_SECONDS * 1_000) as f64) as u64
        })
}

#[derive(Clone)]
pub struct AnthropicBrain {
    client: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
    max_retries: u32,
}

impl AnthropicBrain {
    pub fn new(base_url: String, api_key: Option<String>) -> Result<Self> {
        reqwest::Url::parse(&base_url).map_err(|error| PokError::Config(error.to_string()))?;
        Ok(Self {
            client: reqwest::Client::new(),
            base_url: base_url.trim_end_matches('/').into(),
            api_key,
            max_retries: 3,
        })
    }

    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }
}

#[async_trait]
impl Brain for AnthropicBrain {
    fn max_retries(&self) -> u32 {
        self.max_retries
    }

    fn requires_max_tokens(&self) -> bool {
        true
    }

    async fn list_models(&self) -> Result<Vec<String>> {
        let mut request = self
            .client
            .get(format!("{}/models", self.base_url))
            .header("anthropic-version", "2023-06-01");
        if let Some(key) = &self.api_key {
            request = request.header("x-api-key", key);
        }
        let response = request.send().await.map_err(provider_error)?;
        let response = require_success(response).await?;
        let body: Value = response.json().await.map_err(provider_error)?;
        Ok(body
            .get("data")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|item| item.get("id").and_then(Value::as_str).map(str::to_owned))
            .collect())
    }

    async fn model_info(&self) -> Result<Vec<ModelInfo>> {
        let mut request = self
            .client
            .get(format!("{}/models?limit=1000", self.base_url))
            .header("anthropic-version", "2023-06-01");
        if let Some(key) = &self.api_key {
            request = request.header("x-api-key", key);
        }
        let response = request.send().await.map_err(provider_error)?;
        let response = require_success(response).await?;
        let body: Value = response.json().await.map_err(provider_error)?;
        Ok(body
            .get("data")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(anthropic_model_info)
            .collect())
    }

    fn stream(&self, request: BrainRequest) -> BrainStream {
        let this = self.clone();
        Box::pin(async_stream::try_stream! {
            let body = anthropic_request(&request);
            let mut http = this.client.post(format!("{}/messages", this.base_url))
                .header("anthropic-version", "2023-06-01").json(&body);
            if let Some(key) = &this.api_key { http = http.header("x-api-key", key); }
            let response = http.send().await.map_err(provider_error)?;
            let response = require_success(response).await?;
            let value: Value = response.json().await.map_err(provider_error)?;
            for block in value.get("content").and_then(Value::as_array).into_iter().flatten() {
                match block.get("type").and_then(Value::as_str) {
                    Some("text") => if let Some(text) = block.get("text").and_then(Value::as_str) { yield BrainEvent::TextDelta { text: text.into() }; },
                    Some("tool_use") => yield BrainEvent::ToolCall { call: CompletedToolCall {
                        id: block.get("id").and_then(Value::as_str).unwrap_or_default().into(),
                        name: block.get("name").and_then(Value::as_str).unwrap_or_default().into(),
                        arguments: block.get("input").cloned().unwrap_or_else(|| json!({})),
                    }},
                    _ => {}
                }
            }
            if let Some(usage) = value.get("usage") {
                yield BrainEvent::Usage {
                    prompt_tokens: usage.get("input_tokens").and_then(Value::as_u64).unwrap_or(0),
                    completion_tokens: usage.get("output_tokens").and_then(Value::as_u64).unwrap_or(0),
                };
            }
            yield BrainEvent::Finished { reason: value.get("stop_reason").and_then(Value::as_str).map(str::to_owned) };
        })
    }
}

fn anthropic_request(request: &BrainRequest) -> Value {
    let system = request
        .messages
        .iter()
        .filter(|message| message.role == "system")
        .flat_map(|message| &message.content)
        .filter_map(|part| match part {
            MessageContent::Text { text } => Some(text.as_str()),
            MessageContent::ImagePng { .. } => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    let messages = request
        .messages
        .iter()
        .filter(|message| message.role != "system")
        .map(anthropic_message)
        .collect::<Vec<_>>();
    let mut body = json!({
        "model": request.model,
        "system": system,
        "messages": messages,
        "tools": request.tools.iter().map(|tool| json!({
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema
        })).collect::<Vec<_>>(),
        "max_tokens": request.max_tokens.unwrap_or(64_000)
    });
    if let Some(temperature) = request.temperature {
        body["temperature"] = json!(temperature);
    }
    match request.reasoning_effort.as_deref() {
        None | Some("off" | "none") => {}
        // Models without effort levels think with a fixed token budget.
        Some("on") => {
            let max_tokens = request.max_tokens.unwrap_or(64_000);
            body["thinking"] = json!({
                "type": "enabled",
                "budget_tokens": (max_tokens / 2).clamp(1_024, 32_000),
            });
        }
        Some(effort) => body["output_config"] = json!({"effort": effort}),
    }
    body
}

/// Model capabilities from the Anthropic Models API: effort levels where the
/// model has them, otherwise an on/off thinking switch.
fn anthropic_model_info(model: &Value) -> Option<ModelInfo> {
    let id = model.get("id")?.as_str()?.to_owned();
    let capabilities = model.get("capabilities").unwrap_or(&Value::Null);
    let supported = |pointer: &str| {
        capabilities
            .pointer(pointer)
            .and_then(|value| value.get("supported"))
            .and_then(Value::as_bool)
            == Some(true)
    };
    let mut efforts = ["low", "medium", "high", "xhigh", "max"]
        .into_iter()
        .filter(|level| supported(&format!("/effort/{level}")))
        .map(str::to_owned)
        .collect::<Vec<_>>();
    if efforts.is_empty() && supported("/thinking/types/enabled") {
        efforts = vec!["off".into(), "on".into()];
    }
    Some(ModelInfo {
        id,
        context_length: model.get("max_input_tokens").and_then(Value::as_u64),
        loaded_context_length: None,
        max_context_length: model.get("max_input_tokens").and_then(Value::as_u64),
        vision: capabilities
            .pointer("/image_input/supported")
            .and_then(Value::as_bool),
        tool_use: Some(true),
        reasoning: capabilities
            .pointer("/thinking/supported")
            .and_then(Value::as_bool),
        supported_parameters: Vec::new(),
        reasoning_default: None,
        reasoning_efforts: efforts,
        loaded: None,
    })
}

fn anthropic_message(message: &BrainMessage) -> Value {
    let mut content: Vec<Value> = message
        .content
        .iter()
        .filter_map(|part| match part {
            MessageContent::Text { text } if !text.trim().is_empty() => {
                Some(json!({"type": "text", "text": text}))
            }
            MessageContent::Text { .. } => None,
            MessageContent::ImagePng { base64 } => Some(json!({"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": base64
            }})),
        })
        .collect();
    content.extend(message.tool_calls.iter().map(|call| {
        json!({
            "type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments
        })
    }));
    if let Some(id) = &message.tool_call_id {
        let text = message
            .content
            .iter()
            .filter_map(|part| match part {
                MessageContent::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        content = vec![json!({"type": "tool_result", "tool_use_id": id, "content": text})];
    }
    json!({"role": if message.role == "assistant" { "assistant" } else { "user" }, "content": content})
}

fn provider_error(error: reqwest::Error) -> PokError {
    let message = error.to_string();
    if error.is_timeout()
        || error.is_connect()
        || error
            .status()
            .is_some_and(|status| transient_status_code(status.as_u16()))
    {
        PokError::ProviderTransient {
            message,
            retry_after_ms: None,
            retry_until_cancelled: error
                .status()
                .is_some_and(|status| capacity_status_code(status.as_u16())),
        }
    } else {
        PokError::Provider(message)
    }
}

pub fn encode_png(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        Json, Router,
        extract::State,
        routing::{get, post},
    };
    use std::sync::{
        Mutex as StdMutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    };

    fn local_test_config(base_url: String) -> ProviderConfig {
        ProviderConfig {
            protocol: ProviderProtocol::OpenaiCompatible,
            base_url,
            data_boundary: None,
            api_key_env: None,
            max_retries: 0,
        }
    }

    async fn spawn_local_server(router: Router) -> (String, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        (format!("http://{address}"), server)
    }

    #[tokio::test]
    async fn lm_studio_lifecycle_uses_native_v1_endpoints() {
        let loaded = Arc::new(AtomicBool::new(false));
        let router = Router::new()
            .route(
                "/api/v1/models",
                get(|State(loaded): State<Arc<AtomicBool>>| async move {
                    Json(if loaded.load(Ordering::SeqCst) {
                        json!({"models": [{
                            "key": "test-model",
                            "loaded_instances": [{"id": "instance-1"}]
                        }]})
                    } else {
                        json!({"models": [{
                            "key": "test-model",
                            "loaded_instances": []
                        }]})
                    })
                }),
            )
            .route(
                "/api/v1/models/load",
                post(|State(loaded): State<Arc<AtomicBool>>| async move {
                    loaded.store(true, Ordering::SeqCst);
                    Json(json!({"instance_id": "instance-1"}))
                }),
            )
            .route(
                "/api/v1/models/unload",
                post(|State(loaded): State<Arc<AtomicBool>>| async move {
                    loaded.store(false, Ordering::SeqCst);
                    Json(json!({"success": true}))
                }),
            )
            .with_state(loaded);
        let (root, server) = spawn_local_server(router).await;
        let config = local_test_config(format!("{root}/v1"));

        assert_eq!(
            local_model_status("lm_studio", &config, "test-model")
                .await
                .phase,
            LocalModelPhase::Unloaded
        );
        assert_eq!(
            load_local_model("lm_studio", &config, "test-model")
                .await
                .unwrap()
                .phase,
            LocalModelPhase::Loaded
        );
        assert_eq!(
            unload_local_model("lm_studio", &config, "test-model")
                .await
                .unwrap()
                .phase,
            LocalModelPhase::Unloaded
        );
        server.abort();
    }

    #[tokio::test]
    async fn lm_studio_load_requests_and_reports_configured_context() {
        #[derive(Clone)]
        struct LoadState {
            loaded: Arc<AtomicBool>,
            request: Arc<StdMutex<Option<Value>>>,
        }
        let state = LoadState {
            loaded: Arc::new(AtomicBool::new(false)),
            request: Arc::new(StdMutex::new(None)),
        };
        let router = Router::new()
            .route(
                "/api/v1/models",
                get(|State(state): State<LoadState>| async move {
                    Json(json!({"models": [{
                        "key": "test-model",
                        "max_context_length": 262144,
                        "loaded_instances": if state.loaded.load(Ordering::SeqCst) {
                            json!([{"id": "instance-1", "config": {"context_length": 100000}}])
                        } else {
                            json!([])
                        }
                    }]}))
                }),
            )
            .route(
                "/api/v1/models/load",
                post(
                    |State(state): State<LoadState>, Json(body): Json<Value>| async move {
                        *state.request.lock().unwrap() = Some(body);
                        state.loaded.store(true, Ordering::SeqCst);
                        Json(json!({
                            "instance_id": "instance-1",
                            "load_config": {"context_length": 100000}
                        }))
                    },
                ),
            )
            .with_state(state.clone());
        let (root, server) = spawn_local_server(router).await;
        let config = local_test_config(format!("{root}/v1"));

        let loaded =
            load_local_model_with_context("lm_studio", &config, "test-model", Some(100_000))
                .await
                .unwrap();
        assert_eq!(loaded.context_length, Some(100_000));
        let request = state.request.lock().unwrap().clone().unwrap();
        assert_eq!(request["context_length"], 100_000);
        assert_eq!(request["echo_load_config"], true);
        server.abort();
    }

    #[tokio::test]
    async fn lm_studio_known_status_skips_a_redundant_inventory_request() {
        let inventory_requests = Arc::new(AtomicUsize::new(0));
        let router = Router::new()
            .route(
                "/api/v1/models",
                get(|State(requests): State<Arc<AtomicUsize>>| async move {
                    let request = requests.fetch_add(1, Ordering::SeqCst);
                    Json(json!({"models": [{
                        "key": "test-model",
                        "loaded_instances": if request == 0 {
                            json!([])
                        } else {
                            json!([{"id": "instance-1"}])
                        }
                    }]}))
                }),
            )
            .route(
                "/api/v1/models/load",
                post(|| async { Json(json!({"instance_id": "instance-1"})) }),
            )
            .with_state(inventory_requests.clone());
        let (root, server) = spawn_local_server(router).await;
        let config = local_test_config(format!("{root}/v1"));

        let known = local_model_status("lm_studio", &config, "test-model").await;
        assert_eq!(known.phase, LocalModelPhase::Unloaded);
        let loaded = load_local_model_from_status("lm_studio", &config, "test-model", known)
            .await
            .unwrap();
        assert_eq!(loaded.phase, LocalModelPhase::Loaded);
        assert_eq!(inventory_requests.load(Ordering::SeqCst), 2);
        server.abort();
    }

    #[tokio::test]
    async fn ollama_status_requires_the_selected_model_to_be_running() {
        let loaded = Arc::new(AtomicBool::new(false));
        let router = Router::new()
            .route(
                "/api/ps",
                get(|State(loaded): State<Arc<AtomicBool>>| async move {
                    let mut models = vec![json!({"name": "another-model:latest"})];
                    if loaded.load(Ordering::SeqCst) {
                        models.push(json!({"name": "selected-model:latest"}));
                    }
                    Json(json!({"models": models}))
                }),
            )
            .route(
                "/api/generate",
                post(
                    |State(loaded): State<Arc<AtomicBool>>, Json(body): Json<Value>| async move {
                        let keep_alive = body.get("keep_alive").and_then(Value::as_i64);
                        loaded.store(keep_alive == Some(-1), Ordering::SeqCst);
                        Json(json!({"done": true}))
                    },
                ),
            )
            .with_state(loaded);
        let (root, server) = spawn_local_server(router).await;
        let config = local_test_config(format!("{root}/v1"));

        assert_eq!(
            local_model_status("ollama", &config, "selected-model")
                .await
                .phase,
            LocalModelPhase::Unloaded
        );
        assert_eq!(
            load_local_model("ollama", &config, "selected-model")
                .await
                .unwrap()
                .phase,
            LocalModelPhase::Loaded
        );
        assert_eq!(
            unload_local_model("ollama", &config, "selected-model")
                .await
                .unwrap()
                .phase,
            LocalModelPhase::Unloaded
        );
        server.abort();
    }

    #[test]
    fn openrouter_metadata_detects_text_and_vision_models() {
        let text = openai_model_info(&json!({
            "id": "text-model",
            "context_length": 32768,
            "architecture": {"input_modalities": ["text"]},
            "supported_parameters": ["tools"]
        }))
        .unwrap();
        assert_eq!(text.vision, Some(false));
        assert_eq!(text.tool_use, Some(true));

        let vision = openai_model_info(&json!({
            "id": "vision-model",
            "architecture": {"input_modalities": ["text", "image"]},
            "supported_parameters": ["tools"]
        }))
        .unwrap();
        assert_eq!(vision.vision, Some(true));
    }

    #[test]
    fn lm_studio_reasoning_metadata_maps_to_chat_completion_efforts() {
        let model = json!({
            "id": "reasoning-model",
            "capabilities": {
                "reasoning": {
                    "allowed_options": ["off", "low", "medium", "xhigh", "on", "low"],
                    "default": "xhigh"
                }
            }
        });
        let info = openai_model_info(&model).unwrap();
        assert_eq!(info.reasoning, Some(true));
        assert_eq!(
            info.reasoning_efforts,
            ["off", "on", "low", "medium", "xhigh"]
        );
        assert_eq!(info.reasoning_default.as_deref(), Some("xhigh"));

        let binary = json!({
            "capabilities": {
                "reasoning": {
                    "allowed_options": ["off", "on"],
                    "default": "on"
                }
            }
        });
        let (efforts, default) = reasoning_metadata(&binary);
        assert_eq!(efforts, ["off", "on"]);
        assert_eq!(default.as_deref(), Some("on"));
        // Chat completions: off is "none"; on leaves the model's default.
        let mut request = BrainRequest {
            model: "reasoning-model".into(),
            messages: Vec::new(),
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            seed: None,
            reasoning_effort: Some("off".into()),
        };
        assert_eq!(openai_request(&request)["reasoning_effort"], "none");
        request.reasoning_effort = Some("on".into());
        assert!(openai_request(&request).get("reasoning_effort").is_none());
    }

    #[test]
    fn provider_catalogs_report_their_reasoning_levels() {
        // OpenRouter: named levels, a default, and mandatory reasoning.
        let (efforts, default) = reasoning_metadata(&json!({"reasoning": {
            "mandatory": true, "default_effort": "max",
            "supported_efforts": ["max", "xhigh", "high", "medium", "low"]
        }}));
        assert_eq!(efforts, ["low", "medium", "high", "xhigh", "max"]);
        assert_eq!(default.as_deref(), Some("max"));
        // OpenRouter: reasoning that can only be switched on or off.
        let (efforts, default) = reasoning_metadata(&json!({"reasoning": {
            "mandatory": false, "default_enabled": false
        }}));
        assert_eq!(efforts, ["off", "on"]);
        assert_eq!(default.as_deref(), Some("off"));
        // Mandatory reasoning without levels offers nothing to choose.
        assert!(
            reasoning_metadata(&json!({"reasoning": {"mandatory": true}}))
                .0
                .is_empty()
        );
        // xAI language-models catalog.
        let (efforts, default) = reasoning_metadata(&json!({"capabilities": {
            "reasoning_effort": ["none", "low", "medium", "high", "xhigh"],
            "default_reasoning_effort": "low"
        }}));
        assert_eq!(efforts, ["none", "low", "medium", "high", "xhigh"]);
        assert_eq!(default.as_deref(), Some("low"));
        // Anthropic Models API: effort levels, or a thinking switch.
        let opus = anthropic_model_info(&json!({"id": "claude-opus-4-6", "capabilities": {
            "thinking": {"supported": true, "types": {"enabled": {"supported": true}, "adaptive": {"supported": true}}},
            "effort": {"supported": true, "low": {"supported": true}, "medium": {"supported": true},
                       "high": {"supported": true}, "xhigh": {"supported": false}, "max": {"supported": true}}
        }})).unwrap();
        assert_eq!(opus.reasoning_efforts, ["low", "medium", "high", "max"]);
        let haiku = anthropic_model_info(&json!({"id": "claude-haiku-4-5", "capabilities": {
            "thinking": {"supported": true, "types": {"enabled": {"supported": true}, "adaptive": {"supported": false}}},
            "effort": {"supported": false}
        }})).unwrap();
        assert_eq!(haiku.reasoning_efforts, ["off", "on"]);
    }

    #[test]
    fn each_protocol_sends_the_level_its_own_way() {
        let mut request = BrainRequest {
            model: "model".into(),
            messages: Vec::new(),
            tools: Vec::new(),
            temperature: None,
            max_tokens: Some(20_000),
            seed: None,
            reasoning_effort: Some("high".into()),
        };
        let openrouter = openrouter_request(&request);
        assert_eq!(openrouter["reasoning"], json!({"effort": "high"}));
        assert!(openrouter.get("reasoning_effort").is_none());
        assert_eq!(
            anthropic_request(&request)["output_config"],
            json!({"effort": "high"})
        );
        request.reasoning_effort = Some("off".into());
        assert_eq!(
            openrouter_request(&request)["reasoning"],
            json!({"enabled": false})
        );
        assert!(anthropic_request(&request).get("thinking").is_none());
        request.reasoning_effort = Some("on".into());
        assert_eq!(
            anthropic_request(&request)["thinking"],
            json!({"type": "enabled", "budget_tokens": 10_000})
        );
    }

    #[test]
    fn openrouter_stream_rate_limit_is_retryable() {
        let error = openai_stream_error(&json!({
            "error": {
                "code": 429,
                "message": "Provider returned error",
                "metadata": {"error_type": "rate_limit_exceeded"}
            }
        }))
        .unwrap();
        assert!(error.is_transient_provider_error());
        assert!(error.provider_retry_until_cancelled());
        assert!(error.to_string().contains("429"));
    }

    #[test]
    fn openrouter_stream_auth_error_is_not_retryable() {
        let error = openai_stream_error(&json!({
            "error": {
                "code": 401,
                "message": "Invalid API key"
            }
        }))
        .unwrap();
        assert!(!error.is_transient_provider_error());
    }

    #[test]
    fn retry_after_seconds_are_parsed_and_capped() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(reqwest::header::RETRY_AFTER, "1.5".parse().unwrap());
        assert_eq!(retry_after_ms(&headers), Some(1_500));
        headers.insert(reqwest::header::RETRY_AFTER, "900".parse().unwrap());
        assert_eq!(retry_after_ms(&headers), Some(600_000));
        headers.insert(reqwest::header::RETRY_AFTER, "tomorrow".parse().unwrap());
        assert_eq!(retry_after_ms(&headers), None);
    }

    #[test]
    fn openai_tool_only_assistant_omits_content() {
        let request = BrainRequest {
            model: "local-model".into(),
            messages: vec![BrainMessage {
                role: "assistant".into(),
                content: vec![MessageContent::Text {
                    text: String::new(),
                }],
                origin: MessageOrigin::Assistant,
                tool_call_id: None,
                tool_calls: vec![CompletedToolCall {
                    id: "call-1".into(),
                    name: "capture_screen".into(),
                    arguments: json!({}),
                }],
            }],
            tools: Vec::new(),
            temperature: Some(0.0),
            max_tokens: Some(100),
            seed: None,
            reasoning_effort: None,
        };
        let body = openai_request(&request);
        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none());
        assert!(body["messages"][0].get("content").is_none());
        assert_eq!(
            body["messages"][0]["tool_calls"][0]["function"]["name"],
            "capture_screen"
        );
    }

    #[test]
    fn openai_tool_output_is_text_and_followup_images_use_user_content() {
        let request = BrainRequest {
            model: "local-model".into(),
            messages: vec![
                BrainMessage {
                    role: "tool".into(),
                    content: vec![MessageContent::Text {
                        text: r#"{"screenshots":[{"monitor":"primary"}]}"#.into(),
                    }],
                    origin: MessageOrigin::ToolResult,
                    tool_call_id: Some("call-1".into()),
                    tool_calls: Vec::new(),
                },
                BrainMessage {
                    role: "user".into(),
                    content: vec![
                        MessageContent::Text {
                            text: "Current desktop screenshot.".into(),
                        },
                        MessageContent::ImagePng {
                            base64: "iVBORw0KGgo=".into(),
                        },
                    ],
                    origin: MessageOrigin::ToolImage,
                    tool_call_id: None,
                    tool_calls: Vec::new(),
                },
            ],
            tools: Vec::new(),
            temperature: Some(0.0),
            max_tokens: Some(100),
            seed: None,
            reasoning_effort: None,
        };
        let body = openai_request(&request);
        assert!(body["messages"][0]["content"].is_string());
        assert_eq!(body["messages"][0]["tool_call_id"], "call-1");
        assert!(body["messages"][1]["content"].is_array());
        assert_eq!(body["messages"][1]["content"][1]["type"], "image_url");
    }

    #[test]
    fn openai_tool_schemas_omit_null_values_for_strict_jinja_templates() {
        let request = BrainRequest {
            model: "local-model".into(),
            messages: vec![BrainMessage::text("user", "test")],
            tools: vec![BrainTool {
                name: "example".into(),
                description: "Example tool".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "optional": {"type": ["string", "null"], "default": null},
                        "nested": {"examples": [null, "kept"]}
                    }
                }),
            }],
            temperature: Some(0.0),
            max_tokens: Some(100),
            seed: None,
            reasoning_effort: None,
        };

        let body = openai_request(&request);
        let parameters = &body["tools"][0]["function"]["parameters"];
        assert!(
            parameters["properties"]["optional"]
                .get("default")
                .is_none()
        );
        assert_eq!(
            parameters["properties"]["optional"]["type"],
            json!(["string", "null"])
        );
        assert_eq!(
            parameters["properties"]["nested"]["examples"],
            json!(["kept"])
        );
        assert!(body.get("seed").is_none());
    }

    #[test]
    fn seed_oss_tool_schemas_use_scalar_types_for_its_jinja_parser() {
        let request = BrainRequest {
            model: "bytedance/seed-oss-36b".into(),
            messages: vec![BrainMessage::text("user", "test")],
            tools: vec![BrainTool {
                name: "example".into(),
                description: "Example tool".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "optional_text": {"type": ["string", "null"]},
                        "optional_count": {"type": ["integer", "null"]},
                        "mode": {"enum": ["fast", "safe"]},
                        "choice": {"anyOf": [{"type": "null"}, {"type": "string"}]},
                        "unconstrained": {}
                    }
                }),
            }],
            temperature: None,
            max_tokens: Some(100),
            seed: None,
            reasoning_effort: None,
        };

        let body = openai_request(&request);
        let properties = &body["tools"][0]["function"]["parameters"]["properties"];
        assert_eq!(properties["optional_text"]["type"], "string");
        assert_eq!(properties["optional_count"]["type"], "integer");
        assert_eq!(properties["mode"]["type"], "string");
        assert_eq!(properties["choice"]["type"], "string");
        assert_eq!(properties["unconstrained"]["type"], "object");
    }

    #[test]
    fn openai_tool_schemas_replace_boolean_schema_shorthand() {
        let request = BrainRequest {
            model: "local-model".into(),
            messages: vec![BrainMessage::text("user", "test")],
            tools: vec![BrainTool {
                name: "generated_tool".into(),
                description: "Tool containing arbitrary JSON arguments".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "input": true,
                        "enabled": {"type": "boolean", "default": true}
                    },
                    "additionalProperties": true
                }),
            }],
            temperature: Some(0.0),
            max_tokens: Some(100),
            seed: None,
            reasoning_effort: None,
        };

        let body = openai_request(&request);
        let parameters = &body["tools"][0]["function"]["parameters"];
        assert_eq!(parameters["properties"]["input"], json!({"type": "object"}));
        assert_eq!(parameters["properties"]["enabled"]["default"], true);
        assert!(parameters.get("additionalProperties").is_none());
    }

    #[test]
    fn provider_requests_omit_temperature_when_server_default_is_requested() {
        let request = BrainRequest {
            model: "server-sampled-model".into(),
            messages: vec![BrainMessage::text("user", "hello")],
            tools: Vec::new(),
            temperature: None,
            max_tokens: Some(100),
            seed: None,
            reasoning_effort: None,
        };

        assert!(openai_request(&request).get("temperature").is_none());
        assert!(anthropic_request(&request).get("temperature").is_none());

        let mut explicit = request;
        explicit.temperature = Some(0.2);
        for body in [openai_request(&explicit), anthropic_request(&explicit)] {
            let temperature = body["temperature"].as_f64().unwrap();
            assert!((temperature - 0.2).abs() < 0.000_001);
        }

        explicit.reasoning_effort = Some("xhigh".into());
        assert_eq!(openai_request(&explicit)["reasoning_effort"], "xhigh");
    }

    #[test]
    fn openai_compatible_requests_can_omit_the_output_limit() {
        let request = BrainRequest {
            model: "local-reasoning-model".into(),
            messages: vec![BrainMessage::text("user", "summarize")],
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            seed: None,
            reasoning_effort: None,
        };
        assert!(openai_request(&request).get("max_tokens").is_none());
        assert_eq!(anthropic_request(&request)["max_tokens"], 64_000);
    }

    #[test]
    fn openai_stream_errors_are_not_misreported_as_empty_responses() {
        let value = json!({
            "error": {
                "code": 400,
                "message": "Unable to generate parser: Unrecognized schema: true"
            }
        });
        let error = openai_stream_error(&value).unwrap();
        assert_eq!(
            error.to_string(),
            "provider error: 400: Unable to generate parser: Unrecognized schema: true"
        );
        assert!(openai_stream_error(&json!({"choices": []})).is_none());
    }

    #[test]
    fn tool_parameters_schema_always_has_root_properties() {
        for input in [
            json!({"type": "object", "additionalProperties": true}),
            json!({"type": "object"}),
            json!({}),
            json!(true),
        ] {
            let schema = tool_parameters_schema(&input);
            assert_eq!(schema["type"], "object");
            assert!(schema["properties"].is_object());
        }
    }

    #[test]
    fn reference_free_schema_resolves_local_definitions() {
        let schema = json!({
            "type": "object",
            "properties": {"target": {"$ref": "#/$defs/Target"}},
            "$defs": {"Target": {
                "type": "object",
                "properties": {"id": {"type": "string"}}
            }}
        });
        let flattened = reference_free_schema(&schema);
        assert!(flattened.get("$defs").is_none());
        assert_eq!(
            flattened.pointer("/properties/target/properties/id/type"),
            Some(&json!("string"))
        );
    }

    #[test]
    fn reference_free_schema_removes_recursive_references() {
        let schema = json!({
            "type": "object",
            "properties": {"node": {"$ref": "#/$defs/Node"}},
            "$defs": {"Node": {
                "type": "object",
                "properties": {"child": {"$ref": "#/$defs/Node"}}
            }}
        });
        let flattened = reference_free_schema(&schema);
        assert!(!flattened.to_string().contains("$ref"));
        assert_eq!(
            flattened.pointer("/properties/node/properties/child"),
            Some(&json!({}))
        );
    }
}
