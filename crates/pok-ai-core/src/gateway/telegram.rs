//! Telegram Bot API client: long-poll updates in, text messages out.

use std::time::Duration;

use serde::Deserialize;
use serde_json::{Value, json};

use crate::error::{PokError, Result};

/// Telegram rejects messages over 4096 characters; stay under it.
const MESSAGE_CHARS: usize = 3_900;

#[derive(Debug, Clone, Deserialize)]
pub struct Update {
    pub update_id: i64,
    #[serde(default)]
    pub message: Option<Message>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Message {
    pub chat: Chat,
    #[serde(default)]
    pub from: Option<User>,
    #[serde(default)]
    pub text: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Chat {
    pub id: i64,
    #[serde(rename = "type")]
    pub kind: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct User {
    pub id: i64,
    #[serde(default)]
    pub first_name: String,
    #[serde(default)]
    pub username: Option<String>,
    #[serde(default)]
    pub is_bot: bool,
}

impl User {
    pub fn display_name(&self) -> String {
        match (&self.username, self.first_name.trim()) {
            (Some(username), _) => format!("@{username}"),
            (None, name) if !name.is_empty() => name.to_owned(),
            _ => format!("user {}", self.id),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct BotInfo {
    pub username: String,
}

#[derive(Clone)]
pub struct TelegramClient {
    http: reqwest::Client,
    /// `https://api.telegram.org/bot<token>`; it holds the secret, so it is
    /// never printed and request errors are stripped of their URL.
    base: String,
}

impl TelegramClient {
    pub fn new(token: &str) -> Result<Self> {
        Self::with_api_root(token, "https://api.telegram.org")
    }

    pub fn with_api_root(token: &str, api_root: &str) -> Result<Self> {
        let token = token.trim();
        if token.is_empty() {
            return Err(PokError::Config("the Telegram bot token is empty".into()));
        }
        let http = reqwest::Client::builder()
            // Long polls hold the connection open; the per-call timeout is
            // set from the poll length instead.
            .connect_timeout(Duration::from_secs(15))
            .build()
            .map_err(|error| PokError::Provider(error.without_url().to_string()))?;
        Ok(Self {
            http,
            base: format!("{}/bot{token}", api_root.trim_end_matches('/')),
        })
    }

    async fn call(&self, method: &str, body: Value, timeout: Duration) -> Result<Value> {
        let response = self
            .http
            .post(format!("{}/{method}", self.base))
            .timeout(timeout)
            .json(&body)
            .send()
            .await
            .map_err(|error| PokError::Provider(format!("Telegram: {}", error.without_url())))?;
        let status = response.status();
        let payload: Value = response
            .json()
            .await
            .map_err(|error| PokError::Provider(format!("Telegram: {}", error.without_url())))?;
        if !status.is_success() || payload.get("ok") != Some(&Value::Bool(true)) {
            let description = payload
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or("request failed");
            return Err(PokError::Provider(format!(
                "Telegram {method}: {description} (HTTP {status})"
            )));
        }
        Ok(payload.get("result").cloned().unwrap_or(Value::Null))
    }

    pub async fn get_me(&self) -> Result<BotInfo> {
        let result = self
            .call("getMe", json!({}), Duration::from_secs(15))
            .await?;
        serde_json::from_value(result).map_err(|error| PokError::Provider(error.to_string()))
    }

    /// Long-poll for messages after `offset`; waits up to `wait_seconds`.
    pub async fn get_updates(&self, offset: i64, wait_seconds: u64) -> Result<Vec<Update>> {
        let result = self
            .call(
                "getUpdates",
                json!({"offset": offset, "timeout": wait_seconds, "allowed_updates": ["message"]}),
                Duration::from_secs(wait_seconds + 15),
            )
            .await?;
        serde_json::from_value(result).map_err(|error| PokError::Provider(error.to_string()))
    }

    /// Send plain text, split into several messages when it is long.
    pub async fn send_message(&self, chat_id: i64, text: &str) -> Result<()> {
        for part in split_message(text) {
            self.call(
                "sendMessage",
                json!({"chat_id": chat_id, "text": part, "disable_web_page_preview": true}),
                Duration::from_secs(30),
            )
            .await?;
        }
        Ok(())
    }

    pub async fn send_typing(&self, chat_id: i64) {
        let _ = self
            .call(
                "sendChatAction",
                json!({"chat_id": chat_id, "action": "typing"}),
                Duration::from_secs(10),
            )
            .await;
    }
}

/// Split `text` into chunks Telegram accepts, preferring line breaks.
pub fn split_message(text: &str) -> Vec<String> {
    let text = text.trim();
    if text.is_empty() {
        return vec!["(no output)".into()];
    }
    let mut parts = Vec::new();
    let mut current = String::new();
    for line in text.lines() {
        // A single line longer than a message is cut on character bounds.
        let mut line = line;
        while line.chars().count() > MESSAGE_CHARS {
            let cut = line
                .char_indices()
                .nth(MESSAGE_CHARS)
                .map_or(line.len(), |(index, _)| index);
            if !current.is_empty() {
                parts.push(std::mem::take(&mut current));
            }
            parts.push(line[..cut].to_owned());
            line = &line[cut..];
        }
        if current.chars().count() + line.chars().count() + 1 > MESSAGE_CHARS && !current.is_empty()
        {
            parts.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push('\n');
        }
        current.push_str(line);
    }
    if !current.is_empty() {
        parts.push(current);
    }
    parts
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Json, Router, extract::State, routing::post};
    use std::sync::{Arc, Mutex};

    #[test]
    fn long_text_is_split_on_lines_and_never_over_the_limit() {
        let line = "x".repeat(1_000);
        let text = std::iter::repeat_n(line.as_str(), 9)
            .collect::<Vec<_>>()
            .join("\n");
        let parts = split_message(&text);
        assert!(parts.len() >= 3);
        assert!(
            parts
                .iter()
                .all(|part| part.chars().count() <= MESSAGE_CHARS)
        );
        assert_eq!(parts.join("\n"), text);
        let one_long_line = "é".repeat(MESSAGE_CHARS * 2 + 5);
        let parts = split_message(&one_long_line);
        assert_eq!(parts.len(), 3);
        assert_eq!(parts.concat(), one_long_line);
        assert_eq!(split_message("   "), vec!["(no output)".to_owned()]);
    }

    #[test]
    fn user_names_prefer_the_handle() {
        let user = |username: Option<&str>, first: &str| User {
            id: 7,
            first_name: first.into(),
            username: username.map(str::to_owned),
            is_bot: false,
        };
        assert_eq!(user(Some("sam"), "Sam").display_name(), "@sam");
        assert_eq!(user(None, "Sam").display_name(), "Sam");
        assert_eq!(user(None, " ").display_name(), "user 7");
    }

    #[tokio::test]
    async fn updates_and_messages_use_the_bot_api_shape() {
        type Calls = Arc<Mutex<Vec<(String, Value)>>>;
        async fn handle(
            State(calls): State<Calls>,
            axum::extract::Path((_token, method)): axum::extract::Path<(String, String)>,
            Json(body): Json<Value>,
        ) -> Json<Value> {
            calls.lock().unwrap().push((method.clone(), body));
            let result = match method.as_str() {
                "getMe" => json!({"id": 1, "is_bot": true, "username": "pok_test_bot"}),
                "getUpdates" => json!([{
                    "update_id": 41,
                    "message": {
                        "message_id": 1,
                        "chat": {"id": 900, "type": "private"},
                        "from": {"id": 900, "is_bot": false, "first_name": "Sam"},
                        "text": "hello"
                    }
                }]),
                _ => json!(true),
            };
            Json(json!({"ok": true, "result": result}))
        }
        let calls = Calls::default();
        let app = Router::new()
            .route("/{token}/{method}", post(handle))
            .with_state(calls.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let root = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let client = TelegramClient::with_api_root("123:secret", &root).unwrap();

        assert_eq!(client.get_me().await.unwrap().username, "pok_test_bot");
        let updates = client.get_updates(41, 0).await.unwrap();
        assert_eq!(updates[0].update_id, 41);
        let message = updates[0].message.as_ref().unwrap();
        assert_eq!(message.chat.kind, "private");
        assert_eq!(message.text.as_deref(), Some("hello"));
        client.send_message(900, "done").await.unwrap();

        let calls = calls.lock().unwrap();
        let send = calls
            .iter()
            .find(|(method, _)| method == "sendMessage")
            .unwrap();
        assert_eq!(send.1["chat_id"], 900);
        assert_eq!(send.1["text"], "done");
        let poll = calls
            .iter()
            .find(|(method, _)| method == "getUpdates")
            .unwrap();
        assert_eq!(poll.1["offset"], 41);
    }

    #[tokio::test]
    async fn an_unreachable_api_error_never_contains_the_token() {
        let client =
            TelegramClient::with_api_root("999:very-secret-token", "http://127.0.0.1:1").unwrap();
        let error = client.get_me().await.unwrap_err().to_string();
        assert!(!error.contains("very-secret-token"), "{error}");
    }

    #[test]
    fn an_empty_token_is_refused() {
        assert!(TelegramClient::new("  ").is_err());
    }
}
