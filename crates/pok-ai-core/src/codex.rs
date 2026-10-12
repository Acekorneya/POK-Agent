//! The ChatGPT subscription model plane: OpenAI's Responses API served by the
//! ChatGPT backend, with the headers the Codex client sends. The subscription
//! access token is refreshed here when it expires; models are the
//! Codex-eligible GPT family from the models.dev catalog, matching what
//! ChatGPT plans authorize.
//!
//! Only the family ChatGPT plans serve is offered: newer major versions
//! (GPT-6, GPT-5.5+ and later point releases), minus the pro/research
//! variants and the legacy `gpt-5.6` alias, exactly the Codex client rules.

use std::{collections::BTreeMap, path::PathBuf, sync::Arc};

use async_trait::async_trait;
use futures::StreamExt;
use parking_lot::Mutex;
use serde_json::{Value, json};

use crate::{
    PokError, Result,
    brain::{
        Brain, BrainEvent, BrainRequest, BrainStream, CompletedToolCall, MessageContent, ModelInfo,
    },
    oauth::OpenAiCredential,
};

/// Prompt and window limits the Codex client reports for ChatGPT plans.
const CODEX_PROMPT_TOKENS: u64 = 272_000;
const CODEX_WINDOW_TOKENS: u64 = 400_000;
/// Models ChatGPT plans serve, even when the version filter alone would not
/// decide them.
const CODEX_ALLOWED: &[&str] = &["gpt-5.5", "gpt-5.3-codex-spark"];
/// Models ChatGPT plans lost or never served.
const CODEX_DISALLOWED: &[&str] = &["gpt-5.5-pro", "gpt-5.6"];

/// Whether a ChatGPT plan authorizes this model over the Codex backend.
fn codex_model_eligible(id: &str) -> bool {
    let id = id.rsplit('/').next().unwrap_or(id);
    if CODEX_ALLOWED.contains(&id) {
        return true;
    }
    if CODEX_DISALLOWED.contains(&id) {
        return false;
    }
    let Some(rest) = id.strip_prefix("gpt-") else {
        return false;
    };
    // The leading numeric version: "5.6-luna" -> 5.6, "6.1-sol" -> 6.1.
    let version = rest
        .chars()
        .take_while(|character| character.is_ascii_digit() || *character == '.')
        .collect::<String>();
    let mut numbers = version.split('.');
    let major: u32 = numbers
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    let minor: u32 = numbers
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    major > 5 || (major == 5 && minor > 4)
}

#[derive(Clone)]
pub struct CodexBrain {
    client: reqwest::Client,
    base_url: String,
    credential: Arc<Mutex<OpenAiCredential>>,
    /// The conversation id, sent as `session-id` so ChatGPT can route prompt
    /// caching (children share their parent's value upstream).
    session: String,
    data_dir: PathBuf,
    max_retries: u32,
}

impl CodexBrain {
    pub fn new(
        credential: OpenAiCredential,
        session: Option<&str>,
        data_dir: &std::path::Path,
    ) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: crate::oauth::CODEX_BASE_URL.to_owned(),
            credential: Arc::new(Mutex::new(credential)),
            session: session
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map_or_else(|| uuid::Uuid::new_v4().to_string(), str::to_owned),
            data_dir: data_dir.to_path_buf(),
            max_retries: 3,
        }
    }

    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// A fresh access token, refreshing (and persisting) an expired one.
    async fn access_token(&self) -> Result<String> {
        let current = self.credential.lock().clone();
        if !current.is_expired() {
            return Ok(current.access_token);
        }
        let refreshed = crate::oauth::refresh_codex_credential(&current).await?;
        let stored = refreshed.to_stored()?;
        *self.credential.lock() = refreshed.clone();
        let _ = crate::oauth::save_codex_credential(&self.data_dir, &refreshed);
        crate::brain::set_api_key(crate::oauth::CODEX_KEY_ENV, stored);
        Ok(refreshed.access_token)
    }

    async fn request(&self, body: &Value, token: &str) -> Result<reqwest::Response> {
        let mut request = self
            .client
            .post(format!("{}/responses", self.base_url))
            .bearer_auth(token)
            .header("originator", "pok-agent")
            .header("session-id", &self.session)
            .header("Content-Type", "application/json");
        if let Some(account) = self.credential.lock().account_id.clone() {
            request = request.header("chatgpt-account-id", account);
        }
        let response = request.json(body).send().await.map_err(codex_error)?;
        crate::brain::require_success(response).await
    }
}

#[async_trait]
impl Brain for CodexBrain {
    fn max_retries(&self) -> u32 {
        self.max_retries
    }

    async fn list_models(&self) -> Result<Vec<String>> {
        Ok(self
            .model_info()
            .await?
            .into_iter()
            .map(|model| model.id)
            .collect())
    }

    async fn model_info(&self) -> Result<Vec<ModelInfo>> {
        let mut models = crate::context::catalog_model_infos(&self.data_dir, "openai").await;
        models.retain(|model| codex_model_eligible(&model.id));
        for model in &mut models {
            // Match the Codex client so context accounting and subscription
            // usage agree between clients.
            model.context_length = Some(CODEX_PROMPT_TOKENS);
            model.max_context_length = Some(CODEX_WINDOW_TOKENS);
        }
        Ok(models)
    }

    fn stream(&self, request: BrainRequest) -> BrainStream {
        let this = self.clone();
        Box::pin(async_stream::try_stream! {
            let body = codex_request(&request);
            let mut token = this.access_token().await?;
            let response = match this.request(&body, &token).await {
                Ok(response) => response,
                Err(error) if is_unauthorized(&error) => {
                    // The backend revoked the token early: refresh once.
                    let current = this.credential.lock().clone();
                    let refreshed = crate::oauth::refresh_codex_credential(&current).await?;
                    token = refreshed.access_token.clone();
                    *this.credential.lock() = refreshed.clone();
                    let _ = crate::oauth::save_codex_credential(&this.data_dir, &refreshed);
                    this.request(&body, &token).await?
                }
                Err(error) => Err(error)?,
            };
            let mut bytes = response.bytes_stream();
            let mut buffer = String::new();
            let mut calls: BTreeMap<String, CodexToolCall> = BTreeMap::new();
            while let Some(chunk) = bytes.next().await {
                let chunk = chunk.map_err(codex_error)?;
                buffer.push_str(&String::from_utf8_lossy(&chunk));
                while let Some(end) = buffer.find("\n\n") {
                    let event = buffer[..end].to_owned();
                    buffer.drain(..end + 2);
                    for line in event.lines().filter_map(|line| line.strip_prefix("data:")) {
                        let data = line.trim();
                        if data == "[DONE]" { continue; }
                        let value: Value = serde_json::from_str(data)?;
                        match value.get("type").and_then(Value::as_str).unwrap_or_default() {
                            "response.output_text.delta" => {
                                if let Some(text) = value.get("delta").and_then(Value::as_str)
                                    && !text.is_empty()
                                {
                                    yield BrainEvent::TextDelta { text: text.into() };
                                }
                            }
                            "response.reasoning.delta" | "response.reasoning_summary_text.delta" => {
                                if let Some(text) = value.get("delta").and_then(Value::as_str)
                                    && !text.is_empty()
                                {
                                    yield BrainEvent::ReasoningDelta { text: text.into() };
                                }
                            }
                            "response.output_item.added" => {
                                if let Some(item) = value.get("item")
                                    && item.get("type").and_then(Value::as_str) == Some("function_call")
                                    && let Some(item_id) = item.get("id").and_then(Value::as_str)
                                {
                                    let call = calls.entry(item_id.to_owned()).or_default();
                                    if let Some(call_id) = item.get("call_id").and_then(Value::as_str) {
                                        call.call_id = call_id.to_owned();
                                    }
                                    if let Some(name) = item.get("name").and_then(Value::as_str) {
                                        call.name = name.to_owned();
                                    }
                                    if let Some(arguments) = item.get("arguments").and_then(Value::as_str) {
                                        call.arguments.push_str(arguments);
                                    }
                                }
                            }
                            "response.function_call_arguments.delta" => {
                                if let Some(item_id) = value.get("item_id").and_then(Value::as_str)
                                    && let Some(delta) = value.get("delta").and_then(Value::as_str)
                                {
                                    calls.entry(item_id.to_owned()).or_default().arguments.push_str(delta);
                                }
                            }
                            "response.function_call_arguments.done" => {
                                if let Some(item_id) = value.get("item_id").and_then(Value::as_str) {
                                    let mut call = calls.remove(item_id).unwrap_or_default();
                                    if let Some(arguments) = value.get("arguments").and_then(Value::as_str) {
                                        call.arguments = arguments.to_owned();
                                    }
                                    if call.name.is_empty() {
                                        call.name = value.get("name").and_then(Value::as_str).unwrap_or_default().to_owned();
                                    }
                                    if !call.name.is_empty() {
                                        yield call.into_event();
                                    }
                                }
                            }
                            "response.output_item.done" => {
                                if let Some(item) = value.get("item")
                                    && item.get("type").and_then(Value::as_str) == Some("function_call")
                                    && let Some(item_id) = item.get("id").and_then(Value::as_str)
                                    && let Some(mut call) = calls.remove(item_id)
                                {
                                    if let Some(call_id) = item.get("call_id").and_then(Value::as_str) {
                                        call.call_id = call_id.to_owned();
                                    }
                                    if let Some(name) = item.get("name").and_then(Value::as_str) {
                                        call.name = name.to_owned();
                                    }
                                    if let Some(arguments) = item.get("arguments").and_then(Value::as_str) {
                                        call.arguments = arguments.to_owned();
                                    }
                                    if !call.name.is_empty() {
                                        yield call.into_event();
                                    }
                                }
                            }
                            "response.completed" | "response.incomplete" => {
                                if let Some(usage) = value.pointer("/response/usage") {
                                    yield BrainEvent::Usage {
                                        prompt_tokens: usage.get("input_tokens").and_then(Value::as_u64).unwrap_or(0),
                                        completion_tokens: usage.get("output_tokens").and_then(Value::as_u64).unwrap_or(0),
                                    };
                                }
                                let reason = value.get("type").and_then(Value::as_str).unwrap_or("completed");
                                yield BrainEvent::Finished { reason: Some(reason.into()) };
                            }
                            "response.failed" | "error" => {
                                let message = value
                                    .pointer("/response/error/message")
                                    .or_else(|| value.pointer("/error/message"))
                                    .or_else(|| value.pointer("/response/incomplete_details/reason"))
                                    .and_then(Value::as_str)
                                    .unwrap_or("the ChatGPT backend reported a failed response");
                                Err(PokError::Provider(message.to_owned()))?;
                            }
                            _ => {}
                        }
                    }
                }
            }
        })
    }
}

#[derive(Default)]
struct CodexToolCall {
    call_id: String,
    name: String,
    arguments: String,
}

impl CodexToolCall {
    fn into_event(self) -> BrainEvent {
        match serde_json::from_str(&self.arguments) {
            Ok(arguments) => BrainEvent::ToolCall {
                call: CompletedToolCall {
                    id: if self.call_id.is_empty() {
                        uuid::Uuid::new_v4().to_string()
                    } else {
                        self.call_id
                    },
                    name: self.name,
                    arguments,
                },
            },
            Err(error) => BrainEvent::MalformedToolCall {
                name: self.name,
                raw_arguments: self.arguments,
                error: error.to_string(),
            },
        }
    }
}

/// The Responses request for one turn: instructions from the system messages,
/// the conversation as input items, and the tools in Responses format.
fn codex_request(request: &BrainRequest) -> Value {
    let mut instructions = Vec::new();
    let mut input = Vec::new();
    for message in &request.messages {
        let text = message
            .content
            .iter()
            .filter_map(|part| match part {
                MessageContent::Text { text } => Some(text.as_str()),
                MessageContent::ImagePng { .. } => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        match message.role.as_str() {
            "system" | "developer" => {
                if !text.trim().is_empty() {
                    instructions.push(text);
                }
            }
            "user" => {
                let mut content = Vec::new();
                if !text.is_empty() {
                    content.push(json!({"type": "input_text", "text": text}));
                }
                for part in &message.content {
                    if let MessageContent::ImagePng { base64 } = part {
                        content.push(json!({
                            "type": "input_image",
                            "image_url": format!("data:image/png;base64,{base64}"),
                        }));
                    }
                }
                if !content.is_empty() {
                    input.push(json!({"type": "message", "role": "user", "content": content}));
                }
            }
            "assistant" => {
                if !text.trim().is_empty() {
                    input.push(json!({
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text}],
                    }));
                }
                for call in &message.tool_calls {
                    input.push(json!({
                        "type": "function_call",
                        "call_id": call.id,
                        "name": call.name,
                        "arguments": call.arguments.to_string(),
                    }));
                }
            }
            "tool" => {
                let call_id = message
                    .tool_call_id
                    .clone()
                    .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
                input.push(json!({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": text,
                }));
            }
            _ => {}
        }
    }
    let tools = request
        .tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            })
        })
        .collect::<Vec<_>>();
    let mut body = json!({
        "model": request.model,
        "input": input,
        "store": false,
        "stream": true,
    });
    if !instructions.is_empty() {
        body["instructions"] = json!(instructions.join("\n\n"));
    }
    if !tools.is_empty() {
        body["tools"] = json!(tools);
        body["tool_choice"] = json!("auto");
        body["parallel_tool_calls"] = json!(true);
    }
    // The ChatGPT backend rejects a requested output limit, so `max_tokens`
    // stays out of the request on purpose.
    if let Some(effort) = request
        .reasoning_effort
        .as_deref()
        .map(str::trim)
        .filter(|effort| !effort.is_empty())
    {
        match effort {
            "off" => body["reasoning"] = json!({"effort": "none"}),
            "on" => {}
            other => body["reasoning"] = json!({"effort": other}),
        }
    }
    body
}

fn is_unauthorized(error: &PokError) -> bool {
    let text = error.to_string();
    text.contains("401") || text.contains("403")
}

fn codex_error(error: reqwest::Error) -> PokError {
    PokError::Provider(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain::{BrainMessage, BrainTool};

    #[test]
    fn chatgpt_plans_serve_only_the_codex_family() {
        for eligible in [
            "gpt-5.5",
            "gpt-5.6-luna",
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-6-luna",
            "gpt-6.1-sol",
            "openai/gpt-6-sol",
        ] {
            assert!(codex_model_eligible(eligible), "{eligible}");
        }
        for skipped in [
            "gpt-5.4",
            "gpt-5.4-mini",
            "gpt-5.5-pro",
            "gpt-5.6",
            "gpt-4o",
            "o3",
            "text-embedding-3-small",
        ] {
            assert!(!codex_model_eligible(skipped), "{skipped}");
        }
    }

    #[test]
    fn the_responses_request_carries_instructions_input_and_tools() {
        let request = BrainRequest {
            model: "gpt-5.5".into(),
            messages: vec![
                BrainMessage::text("system", "You are POK-Agent."),
                BrainMessage::text("user", "Read the file."),
                BrainMessage {
                    role: "assistant".into(),
                    content: vec![MessageContent::Text {
                        text: "Reading it.".into(),
                    }],
                    origin: Default::default(),
                    tool_call_id: None,
                    tool_calls: vec![CompletedToolCall {
                        id: "call-1".into(),
                        name: "read_file".into(),
                        arguments: json!({"path": "notes.txt"}),
                    }],
                },
                BrainMessage {
                    role: "tool".into(),
                    content: vec![MessageContent::Text {
                        text: "the contents".into(),
                    }],
                    origin: Default::default(),
                    tool_call_id: Some("call-1".into()),
                    tool_calls: Vec::new(),
                },
            ],
            tools: vec![BrainTool {
                name: "read_file".into(),
                description: "Read a file".into(),
                input_schema: json!({"type": "object", "properties": {"path": {"type": "string"}}}),
            }],
            temperature: None,
            max_tokens: Some(16384),
            seed: None,
            reasoning_effort: Some("high".into()),
        };
        let body = codex_request(&request);
        assert_eq!(body["model"], "gpt-5.5");
        assert_eq!(body["instructions"], "You are POK-Agent.");
        assert_eq!(body["store"], false);
        assert_eq!(body["stream"], true);
        assert!(
            body.get("max_output_tokens").is_none(),
            "the backend rejects output limits: {body}"
        );
        assert_eq!(body["reasoning"]["effort"], "high");
        assert_eq!(body["input"][0]["type"], "message");
        assert_eq!(body["input"][0]["role"], "user");
        assert_eq!(body["input"][0]["content"][0]["type"], "input_text");
        assert_eq!(body["input"][1]["type"], "message");
        assert_eq!(body["input"][1]["role"], "assistant");
        assert_eq!(body["input"][2]["type"], "function_call");
        assert_eq!(body["input"][2]["call_id"], "call-1");
        assert_eq!(body["input"][2]["arguments"], "{\"path\":\"notes.txt\"}");
        assert_eq!(body["input"][3]["type"], "function_call_output");
        assert_eq!(body["input"][3]["output"], "the contents");
        assert_eq!(body["tools"][0]["type"], "function");
        assert_eq!(body["tools"][0]["name"], "read_file");
        assert_eq!(body["tool_choice"], "auto");
    }

    #[test]
    fn reasoning_off_maps_to_none_and_on_stays_default() {
        let mut request = BrainRequest {
            model: "gpt-5.5".into(),
            messages: vec![BrainMessage::text("user", "hi")],
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            seed: None,
            reasoning_effort: Some("off".into()),
        };
        assert_eq!(codex_request(&request)["reasoning"]["effort"], "none");
        request.reasoning_effort = Some("on".into());
        assert!(codex_request(&request).get("reasoning").is_none());
    }
}
