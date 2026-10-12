//! The OpenCode Go model plane. One subscription endpoint serves models over
//! three wires — OpenAI chat completions, the Responses API, and Anthropic
//! messages — and answers `ModelProtocolUnsupported` when a model is asked on
//! the wrong one. The brain tries the protocols in a family-aware order per
//! model, remembers the wire that worked, and passes everything else through
//! to the same adapters the rest of the harness uses.
//!
//! Zen's app-only free tier on `/zen/v1` stays unavailable: this is the Go
//! subscription endpoint (`/zen/go/v1`), which is the one meant for coding
//! agents.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};

use async_trait::async_trait;
use futures::StreamExt;
use parking_lot::Mutex;

use crate::{
    PokError, Result,
    brain::{AnthropicBrain, Brain, BrainRequest, BrainStream, ModelInfo, OpenAiCompatibleBrain},
};

/// The wire a model answers on at the Go endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WireProtocol {
    Chat,
    Responses,
    Messages,
}

/// The order to try a model's wire: known families first, then the rest.
/// Wrong-wire attempts fail fast with a 400 before any tokens are billed.
fn protocol_order(model: &str) -> [WireProtocol; 3] {
    let id = model
        .rsplit('/')
        .next()
        .unwrap_or(model)
        .to_ascii_lowercase();
    if id.starts_with("claude") {
        [
            WireProtocol::Messages,
            WireProtocol::Responses,
            WireProtocol::Chat,
        ]
    } else if id.starts_with("gpt-")
        || id.starts_with("grok")
        || id.starts_with("muse-")
        || id.starts_with("mai-code")
        || id.starts_with("codex")
    {
        [
            WireProtocol::Responses,
            WireProtocol::Chat,
            WireProtocol::Messages,
        ]
    } else {
        [
            WireProtocol::Chat,
            WireProtocol::Responses,
            WireProtocol::Messages,
        ]
    }
}

/// Whether a provider error means "this model does not speak that wire".
fn is_protocol_error(error: &PokError) -> bool {
    let text = error.to_string();
    text.contains("ModelProtocolUnsupported") || text.contains("does not support this protocol")
}

#[derive(Clone)]
pub struct OpenCodeGoBrain {
    chat: OpenAiCompatibleBrain,
    messages: AnthropicBrain,
    client: reqwest::Client,
    endpoint: String,
    api_key: Option<String>,
    headers: Vec<(String, String)>,
    data_dir: PathBuf,
    /// The wire that worked for each model, once one has.
    resolved: Arc<Mutex<HashMap<String, WireProtocol>>>,
    max_retries: u32,
}

impl OpenCodeGoBrain {
    pub fn new(
        base_url: &str,
        api_key: Option<String>,
        session: Option<&str>,
        data_dir: &Path,
    ) -> Result<Self> {
        let base_url = base_url.trim_end_matches('/');
        let headers = crate::brain::opencode_client_headers(base_url, session);
        let chat = OpenAiCompatibleBrain::new(base_url.to_owned(), api_key.clone())?
            .with_headers(&headers)?;
        let messages =
            AnthropicBrain::new(base_url.to_owned(), api_key.clone())?.with_headers(&headers)?;
        Ok(Self {
            chat,
            messages,
            client: reqwest::Client::new(),
            endpoint: format!("{base_url}/responses"),
            api_key,
            headers,
            data_dir: data_dir.to_path_buf(),
            resolved: Arc::new(Mutex::new(HashMap::new())),
            max_retries: 3,
        })
    }

    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Open the stream for one wire. Chat and messages open lazily (their
    /// first poll carries the request); the Responses wire sends here so a
    /// wrong-wire rejection surfaces before the caller consumes anything.
    async fn open(&self, protocol: WireProtocol, request: &BrainRequest) -> Result<BrainStream> {
        match protocol {
            WireProtocol::Chat => Ok(self.chat.stream(request.clone())),
            WireProtocol::Messages => Ok(self.messages.stream(request.clone())),
            WireProtocol::Responses => {
                let body = crate::codex::responses_request(request);
                let mut http = self
                    .client
                    .post(&self.endpoint)
                    .header("Content-Type", "application/json")
                    .json(&body);
                for (name, value) in &self.headers {
                    http = http.header(name, value);
                }
                if let Some(key) = &self.api_key {
                    http = http.bearer_auth(key);
                }
                let response = http.send().await.map_err(opencode_error)?;
                let response = crate::brain::require_success(response).await?;
                Ok(crate::codex::responses_events(response))
            }
        }
    }
}

#[async_trait]
impl Brain for OpenCodeGoBrain {
    fn max_retries(&self) -> u32 {
        self.max_retries
    }

    async fn list_models(&self) -> Result<Vec<String>> {
        self.chat.list_models().await
    }

    async fn model_info(&self) -> Result<Vec<ModelInfo>> {
        // The gateway's list names the models; models.dev (OpenCode's own
        // catalog) describes them, including the ones the gateway just added.
        let mut models = crate::context::catalog_model_infos(&self.data_dir, "opencode-go").await;
        let listed = self.chat.list_models().await.unwrap_or_default();
        for id in listed {
            if !models.iter().any(|model| model.id == id) {
                models.push(ModelInfo {
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
                });
            }
        }
        crate::context::enrich_model_capabilities(&self.data_dir, "opencode-go", &mut models).await;
        Ok(models)
    }

    fn stream(&self, request: BrainRequest) -> BrainStream {
        let this = self.clone();
        Box::pin(async_stream::try_stream! {
            let model = request.model.clone();
            let order: Vec<WireProtocol> = match this.resolved.lock().get(&model).copied() {
                Some(protocol) => vec![protocol],
                None => protocol_order(&model).to_vec(),
            };
            let attempts = order.len();
            for (index, protocol) in order.into_iter().enumerate() {
                let last = index + 1 == attempts;
                match this.open(protocol, &request).await {
                    Ok(mut events) => {
                        let mut rejection = None;
                        while let Some(event) = events.next().await {
                            match event {
                                Ok(event) => yield event,
                                Err(error) if is_protocol_error(&error) => {
                                    rejection = Some(error);
                                    break;
                                }
                                Err(error) => Err(error)?,
                            }
                        }
                        if rejection.is_none() {
                            // This wire answered the model; use it from now on.
                            this.resolved.lock().insert(model, protocol);
                            return;
                        }
                        if last {
                            Err(rejection.expect("a rejection was recorded"))?;
                        }
                    }
                    Err(error) => {
                        if !is_protocol_error(&error) || last {
                            Err(error)?;
                        }
                    }
                }
            }
            Err(PokError::Provider(format!(
                "{model} did not accept any protocol on this endpoint"
            )))?;
        })
    }
}

fn opencode_error(error: reqwest::Error) -> PokError {
    PokError::Provider(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_model_family_tries_its_wire_first() {
        assert_eq!(
            protocol_order("claude-haiku-5-5")[0],
            WireProtocol::Messages
        );
        assert_eq!(
            protocol_order("muse-spark-1.3-contributor")[0],
            WireProtocol::Responses
        );
        assert_eq!(protocol_order("gpt-5.6-luna")[0], WireProtocol::Responses);
        assert_eq!(protocol_order("deepseek-v4-flash")[0], WireProtocol::Chat);
        assert_eq!(
            protocol_order("longcat-2.5-preview-free")[0],
            WireProtocol::Chat
        );
        // Every model still falls back to the other two wires, so a wrong
        // family guess costs one rejected request, not a failed run.
        for model in [
            "claude-opus-5",
            "muse-spark-1.3-contributor",
            "deepseek-v4-flash",
        ] {
            let order = protocol_order(model);
            assert_eq!(order.len(), 3, "{model}");
            assert!(
                order.contains(&WireProtocol::Chat)
                    && order.contains(&WireProtocol::Responses)
                    && order.contains(&WireProtocol::Messages),
                "{model}"
            );
        }
    }

    #[test]
    fn protocol_rejections_are_recognized() {
        assert!(is_protocol_error(&PokError::Provider(
            "HTTP 400 Bad Request: {\"type\":\"error\",\"error\":{\"type\":\"ModelProtocolUnsupported\",\"message\":\"Model does not support this protocol.\"}}".into()
        )));
        assert!(!is_protocol_error(&PokError::Provider(
            "HTTP 429 Too Many Requests".into()
        )));
    }
}
