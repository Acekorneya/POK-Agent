//! Telegram Bot API client and channel: long-poll updates in (text, pictures,
//! documents, voice notes), text messages out.

use std::{sync::Arc, time::Duration};

use async_trait::async_trait;
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use super::{
    Channel, Gateway, Inbound,
    media::{Download, MediaKind, MediaSource},
};
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
    /// The words sent with a picture, document or video.
    #[serde(default)]
    pub caption: Option<String>,
    /// The same picture at several sizes.
    #[serde(default)]
    pub photo: Vec<FileInfo>,
    #[serde(default)]
    pub document: Option<FileInfo>,
    #[serde(default)]
    pub voice: Option<FileInfo>,
    #[serde(default)]
    pub audio: Option<FileInfo>,
    #[serde(default)]
    pub video: Option<FileInfo>,
    #[serde(default)]
    pub video_note: Option<FileInfo>,
    #[serde(default)]
    pub animation: Option<FileInfo>,
    #[serde(default)]
    pub sticker: Option<FileInfo>,
}

/// The fields every Telegram file object shares (photo size, document,
/// voice note, video, ...).
#[derive(Debug, Clone, Deserialize)]
pub struct FileInfo {
    pub file_id: String,
    #[serde(default)]
    pub file_name: Option<String>,
    #[serde(default)]
    pub mime_type: Option<String>,
    #[serde(default)]
    pub file_size: Option<u64>,
    #[serde(default)]
    pub width: Option<u32>,
    #[serde(default)]
    pub height: Option<u32>,
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
    /// `https://api.telegram.org/file/bot<token>`: where files are downloaded.
    file_base: String,
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
        let root = api_root.trim_end_matches('/');
        Ok(Self {
            http,
            base: format!("{root}/bot{token}"),
            file_base: format!("{root}/file/bot{token}"),
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

    /// Download a file a user sent, refusing anything over `max_bytes`.
    pub async fn download_file(&self, file_id: &str, max_bytes: u64) -> Result<Vec<u8>> {
        let info = self
            .call(
                "getFile",
                json!({"file_id": file_id}),
                Duration::from_secs(20),
            )
            .await?;
        let path = info
            .get("file_path")
            .and_then(Value::as_str)
            .ok_or_else(|| PokError::Provider("Telegram sent no download path".into()))?;
        if info
            .get("file_size")
            .and_then(Value::as_u64)
            .is_some_and(|size| size > max_bytes)
        {
            return Err(PokError::Provider("the file is too large".into()));
        }
        let response = self
            .http
            .get(format!("{}/{path}", self.file_base))
            .timeout(Duration::from_secs(120))
            .send()
            .await
            .map_err(|error| PokError::Provider(format!("Telegram: {}", error.without_url())))?;
        if !response.status().is_success() {
            return Err(PokError::Provider(format!(
                "Telegram file download failed (HTTP {})",
                response.status()
            )));
        }
        let mut bytes = Vec::new();
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|error| {
                PokError::Provider(format!("Telegram: {}", error.without_url()))
            })?;
            if bytes.len() as u64 + chunk.len() as u64 > max_bytes {
                return Err(PokError::Provider("the file is too large".into()));
            }
            bytes.extend_from_slice(&chunk);
        }
        Ok(bytes)
    }
}

/// One attachment of a Telegram message.
struct TelegramMedia {
    client: TelegramClient,
    kind: MediaKind,
    file: FileInfo,
}

#[async_trait]
impl MediaSource for TelegramMedia {
    fn kind(&self) -> MediaKind {
        self.kind
    }

    fn label(&self) -> String {
        self.file.file_name.clone().unwrap_or_else(|| {
            match self.kind {
                MediaKind::Image => "a picture",
                MediaKind::Document => "a file",
                MediaKind::Audio => "a voice note",
                MediaKind::Video => "a video",
                MediaKind::Sticker => "a sticker",
            }
            .to_owned()
        })
    }

    fn size_hint(&self) -> Option<u64> {
        self.file.file_size
    }

    async fn fetch(&self, max_bytes: u64) -> Result<Download> {
        let bytes = self
            .client
            .download_file(&self.file.file_id, max_bytes)
            .await?;
        Ok(Download {
            kind: self.kind,
            name: self.file.file_name.clone(),
            mime: self.file.mime_type.clone(),
            bytes,
        })
    }
}

impl Message {
    /// The attachments of this message, one entry each. Telegram sends a
    /// photo as several sizes; the largest one is used.
    fn attachments(&self, client: &TelegramClient) -> Vec<Arc<dyn MediaSource>> {
        let source = |kind, file: &FileInfo| -> Arc<dyn MediaSource> {
            Arc::new(TelegramMedia {
                client: client.clone(),
                kind,
                file: file.clone(),
            })
        };
        let mut found = Vec::new();
        if let Some(largest) = self.photo.iter().max_by_key(|size| {
            u64::from(size.width.unwrap_or(0)) * u64::from(size.height.unwrap_or(0))
                + size.file_size.unwrap_or(0)
        }) {
            found.push(source(MediaKind::Image, largest));
        }
        if let Some(document) = &self.document {
            // A picture sent "as a file" keeps full quality; read it as a picture.
            let kind = if document
                .mime_type
                .as_deref()
                .is_some_and(|mime| mime.starts_with("image/"))
            {
                MediaKind::Image
            } else {
                MediaKind::Document
            };
            found.push(source(kind, document));
        }
        for (file, kind) in [
            (&self.voice, MediaKind::Audio),
            (&self.audio, MediaKind::Audio),
            (&self.video, MediaKind::Video),
            (&self.video_note, MediaKind::Video),
            (&self.animation, MediaKind::Video),
            (&self.sticker, MediaKind::Sticker),
        ] {
            if let Some(file) = file {
                found.push(source(kind, file));
            }
        }
        found
    }
}

/// Telegram as a gateway channel.
pub struct TelegramChannel {
    client: TelegramClient,
    poll_seconds: u64,
}

impl TelegramChannel {
    pub fn new(client: TelegramClient, poll_seconds: u64) -> Self {
        Self {
            client,
            poll_seconds: poll_seconds.clamp(0, 50),
        }
    }

    /// One incoming update. Public so the behavior can be tested without a
    /// network poll.
    pub async fn handle_update(&self, gateway: &Arc<Gateway>, update: Update) {
        let Some(message) = update.message else {
            return;
        };
        // Groups and channels are never a command surface.
        if message.chat.kind != "private" {
            return;
        }
        let Some(user) = message.from.as_ref() else {
            return;
        };
        if user.is_bot {
            return;
        }
        let text = message
            .text
            .clone()
            .or_else(|| message.caption.clone())
            .unwrap_or_default();
        let media = message.attachments(&self.client);
        if text.trim().is_empty() && media.is_empty() {
            return;
        }
        gateway
            .handle_inbound(Inbound {
                channel: "telegram",
                chat: message.chat.id.to_string(),
                user: user.id.to_string(),
                name: user.display_name(),
                text,
                media,
                trusted: false,
            })
            .await;
    }

    async fn poll_loop(&self, gateway: &Arc<Gateway>, cancel: &CancellationToken) {
        // Skip whatever piled up while the gateway was off: acting on a
        // stale command hours later is never what the sender meant.
        let mut offset = loop {
            let attempt = tokio::select! {
                () = cancel.cancelled() => return,
                result = self.client.get_updates(-1, 0) => result,
            };
            match attempt {
                Ok(updates) => break updates.last().map_or(0, |update| update.update_id + 1),
                Err(error) => {
                    tracing::warn!("gateway cannot reach Telegram yet: {error}");
                    tokio::select! {
                        () = cancel.cancelled() => return,
                        () = tokio::time::sleep(Duration::from_secs(5)) => {}
                    }
                }
            }
        };
        let mut backoff = Duration::from_secs(2);
        loop {
            let polled = tokio::select! {
                () = cancel.cancelled() => return,
                result = self.client.get_updates(offset, self.poll_seconds) => result,
            };
            match polled {
                Ok(updates) => {
                    backoff = Duration::from_secs(2);
                    for update in updates {
                        offset = offset.max(update.update_id + 1);
                        self.handle_update(gateway, update).await;
                    }
                }
                Err(error) => {
                    tracing::warn!("gateway poll failed: {error}");
                    tokio::select! {
                        () = cancel.cancelled() => return,
                        () = tokio::time::sleep(backoff) => {}
                    }
                    backoff = (backoff * 2).min(Duration::from_secs(30));
                }
            }
        }
    }
}

#[async_trait]
impl Channel for TelegramChannel {
    fn name(&self) -> &'static str {
        "telegram"
    }

    async fn send_text(&self, chat: &str, text: &str) -> Result<()> {
        let chat_id: i64 = chat
            .parse()
            .map_err(|_| PokError::Provider(format!("not a Telegram chat: {chat}")))?;
        self.client.send_message(chat_id, text).await
    }

    async fn send_typing(&self, chat: &str) {
        if let Ok(chat_id) = chat.parse() {
            self.client.send_typing(chat_id).await;
        }
    }

    async fn listen(self: Arc<Self>, gateway: Arc<Gateway>, cancel: CancellationToken) {
        self.poll_loop(&gateway, &cancel).await;
    }
}

/// Split `text` into chunks Telegram accepts, preferring line breaks.
pub fn split_message(text: &str) -> Vec<String> {
    super::split_text(text, MESSAGE_CHARS)
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
