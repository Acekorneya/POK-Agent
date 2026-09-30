use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use chrono::Utc;
use parking_lot::Mutex;
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use uuid::Uuid;

use crate::{
    Result,
    brain::{BrainMessage, CompletedToolCall, MessageContent, MessageOrigin},
};

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    workspace TEXT NOT NULL,
    artifact_dir TEXT NOT NULL,
    status TEXT NOT NULL,
    format_version INTEGER NOT NULL,
    legacy_imported INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS conversations_updated ON conversations(updated_at DESC);
CREATE INDEX IF NOT EXISTS conversations_workspace_updated ON conversations(workspace, updated_at DESC);
CREATE TABLE IF NOT EXISTS conversation_messages (
    session_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    message_json TEXT NOT NULL,
    PRIMARY KEY(session_id, sequence),
    FOREIGN KEY(session_id) REFERENCES conversations(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS conversation_state (
    session_id TEXT PRIMARY KEY,
    state_json TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES conversations(id) ON DELETE CASCADE
);
"#;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationSummary {
    pub id: Uuid,
    pub title: String,
    pub provider: String,
    pub model: String,
    pub workspace: PathBuf,
    pub artifact_dir: PathBuf,
    pub status: String,
    pub legacy_imported: bool,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationDisplayMessage {
    pub sequence: u64,
    pub subsequence: u32,
    pub sender: String,
    pub kind: String,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationHistoryPage {
    pub messages: Vec<ConversationDisplayMessage>,
    pub next_before_sequence: Option<u64>,
    pub has_more: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationSnapshot {
    pub summary: ConversationSummary,
    pub messages: Vec<BrainMessage>,
    #[serde(default)]
    pub state: Value,
}

pub struct ConversationStore {
    connection: Mutex<Connection>,
}

impl ConversationStore {
    pub fn open(data_dir: &Path) -> Result<Arc<Self>> {
        std::fs::create_dir_all(data_dir)?;
        let connection = Connection::open(data_dir.join("conversations.db"))?;
        connection.execute_batch("PRAGMA foreign_keys=ON;")?;
        connection.execute_batch(SCHEMA)?;
        Ok(Arc::new(Self {
            connection: Mutex::new(connection),
        }))
    }

    pub fn checkpoint(&self, snapshot: &ConversationSnapshot) -> Result<()> {
        let mut connection = self.connection.lock();
        let transaction = connection.transaction()?;
        transaction.execute(
            "INSERT INTO conversations(id,title,provider,model,workspace,artifact_dir,status,format_version,legacy_imported,created_at,updated_at)
             VALUES (?1,?2,?3,?4,?5,?6,?7,1,?8,?9,?10)
             ON CONFLICT(id) DO UPDATE SET title=excluded.title,provider=excluded.provider,model=excluded.model,
             workspace=excluded.workspace,artifact_dir=excluded.artifact_dir,status=excluded.status,
             legacy_imported=excluded.legacy_imported,updated_at=excluded.updated_at",
            params![
                snapshot.summary.id.to_string(), snapshot.summary.title, snapshot.summary.provider,
                snapshot.summary.model, snapshot.summary.workspace.to_string_lossy(),
                snapshot.summary.artifact_dir.to_string_lossy(), snapshot.summary.status,
                snapshot.summary.legacy_imported, snapshot.summary.created_at, snapshot.summary.updated_at,
            ],
        )?;
        transaction.execute(
            "DELETE FROM conversation_messages WHERE session_id=?1",
            params![snapshot.summary.id.to_string()],
        )?;
        for (sequence, message) in snapshot.messages.iter().enumerate() {
            transaction.execute(
                "INSERT INTO conversation_messages(session_id,sequence,message_json) VALUES (?1,?2,?3)",
                params![snapshot.summary.id.to_string(), u64::try_from(sequence).unwrap_or(u64::MAX), serde_json::to_string(message)?],
            )?;
        }
        transaction.execute(
            "INSERT INTO conversation_state(session_id,state_json) VALUES (?1,?2)
             ON CONFLICT(session_id) DO UPDATE SET state_json=excluded.state_json",
            params![
                snapshot.summary.id.to_string(),
                serde_json::to_string(&snapshot.state)?
            ],
        )?;
        transaction.commit()?;
        Ok(())
    }

    pub fn list(&self, workspace: Option<&Path>, limit: usize) -> Result<Vec<ConversationSummary>> {
        let connection = self.connection.lock();
        let limit = i64::try_from(limit.clamp(1, 200)).unwrap_or(50);
        let mut summaries = Vec::new();
        if let Some(workspace) = workspace {
            let mut statement = connection.prepare(
                "SELECT id,title,provider,model,workspace,artifact_dir,status,legacy_imported,created_at,updated_at
                 FROM conversations WHERE workspace=?1 ORDER BY updated_at DESC LIMIT ?2")?;
            let rows = statement.query_map(
                params![workspace.to_string_lossy(), limit],
                summary_from_row,
            )?;
            for row in rows {
                summaries.push(parse_summary(row?)?);
            }
        } else {
            let mut statement = connection.prepare(
                "SELECT id,title,provider,model,workspace,artifact_dir,status,legacy_imported,created_at,updated_at
                 FROM conversations ORDER BY updated_at DESC LIMIT ?1")?;
            let rows = statement.query_map(params![limit], summary_from_row)?;
            for row in rows {
                summaries.push(parse_summary(row?)?);
            }
        }
        Ok(summaries)
    }

    pub fn load(&self, id: Uuid) -> Result<Option<ConversationSnapshot>> {
        let connection = self.connection.lock();
        let row = connection.query_row(
            "SELECT id,title,provider,model,workspace,artifact_dir,status,legacy_imported,created_at,updated_at
             FROM conversations WHERE id=?1",
            params![id.to_string()], summary_from_row).optional()?;
        let Some(row) = row else {
            return Ok(None);
        };
        let summary = parse_summary(row)?;
        let mut statement = connection.prepare(
            "SELECT message_json FROM conversation_messages WHERE session_id=?1 ORDER BY sequence",
        )?;
        let messages = statement
            .query_map(params![id.to_string()], |row| row.get::<_, String>(0))?
            .map(|row| Ok(serde_json::from_str::<BrainMessage>(&row?)?))
            .collect::<Result<Vec<_>>>()?;
        let state = connection
            .query_row(
                "SELECT state_json FROM conversation_state WHERE session_id=?1",
                params![id.to_string()],
                |row| row.get::<_, String>(0),
            )
            .optional()?
            .map(|value| serde_json::from_str(&value))
            .transpose()?
            .unwrap_or_else(|| json!({}));
        Ok(Some(ConversationSnapshot {
            summary,
            messages,
            state,
        }))
    }

    pub fn metadata(&self, id: Uuid) -> Result<Option<(ConversationSummary, Value)>> {
        let connection = self.connection.lock();
        let row = connection
            .query_row(
                "SELECT id,title,provider,model,workspace,artifact_dir,status,legacy_imported,created_at,updated_at
                 FROM conversations WHERE id=?1",
                params![id.to_string()],
                summary_from_row,
            )
            .optional()?;
        let Some(row) = row else {
            return Ok(None);
        };
        let summary = parse_summary(row)?;
        let state = connection
            .query_row(
                "SELECT state_json FROM conversation_state WHERE session_id=?1",
                params![id.to_string()],
                |row| row.get::<_, String>(0),
            )
            .optional()?
            .map(|value| serde_json::from_str(&value))
            .transpose()?
            .unwrap_or_else(|| json!({}));
        Ok(Some((summary, state)))
    }

    pub fn latest(&self, workspace: Option<&Path>) -> Result<Option<ConversationSnapshot>> {
        let Some(summary) = self.list(workspace, 1)?.into_iter().next() else {
            return Ok(None);
        };
        self.load(summary.id)
    }

    pub fn display_messages(&self, id: Uuid) -> Result<Vec<ConversationDisplayMessage>> {
        let Some(snapshot) = self.load(id)? else {
            return Ok(Vec::new());
        };
        Ok(snapshot
            .messages
            .iter()
            .enumerate()
            .flat_map(|(sequence, message)| {
                display_events_for_message(u64::try_from(sequence).unwrap_or(u64::MAX), message)
            })
            .collect())
    }

    pub fn display_messages_page(
        &self,
        id: Uuid,
        before_sequence: Option<u64>,
        limit: usize,
    ) -> Result<ConversationHistoryPage> {
        let connection = self.connection.lock();
        let before = before_sequence
            .and_then(|value| i64::try_from(value).ok())
            .unwrap_or(i64::MAX);
        let raw_limit = i64::try_from(limit.clamp(20, 200)).unwrap_or(100);
        let mut statement = connection.prepare(
            "SELECT sequence,message_json FROM conversation_messages
             WHERE session_id=?1 AND sequence<?2 ORDER BY sequence DESC LIMIT ?3",
        )?;
        let mut rows = statement
            .query_map(params![id.to_string(), before, raw_limit], |row| {
                Ok((row.get::<_, u64>(0)?, row.get::<_, String>(1)?))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        rows.reverse();
        let oldest_sequence = rows.first().map(|(sequence, _)| *sequence);
        let mut messages = Vec::new();
        for (sequence, encoded) in rows {
            let message = serde_json::from_str::<BrainMessage>(&encoded)?;
            messages.extend(display_events_for_message(sequence, &message));
        }
        let has_more = oldest_sequence.is_some_and(|sequence| {
            connection
                .query_row(
                    "SELECT EXISTS(SELECT 1 FROM conversation_messages WHERE session_id=?1 AND sequence<?2)",
                    params![id.to_string(), sequence],
                    |row| row.get::<_, bool>(0),
                )
                .unwrap_or(false)
        });
        Ok(ConversationHistoryPage {
            messages,
            next_before_sequence: has_more.then_some(oldest_sequence.unwrap_or_default()),
            has_more,
        })
    }

    pub fn import_legacy_diagnostics(&self, diagnostics_dir: &Path) -> Result<usize> {
        let sessions_dir = diagnostics_dir.join("sessions");
        let Ok(entries) = std::fs::read_dir(&sessions_dir) else {
            return Ok(0);
        };
        let mut imported = 0;
        for entry in entries.flatten() {
            let path = entry.path();
            let Some(id) = path
                .file_name()
                .and_then(|name| name.to_str())
                .and_then(|name| Uuid::parse_str(name).ok())
            else {
                continue;
            };
            if self.load(id)?.is_some() {
                continue;
            }
            let Ok(trace) = std::fs::read_to_string(path.join("trace.jsonl")) else {
                continue;
            };
            let mut messages = Vec::new();
            let mut provider = String::new();
            let mut model = String::new();
            let mut workspace = PathBuf::new();
            let mut title = String::new();
            let mut created_at = String::new();
            let mut updated_at = String::new();
            for line in trace.lines() {
                let Ok(event) = serde_json::from_str::<Value>(line) else {
                    continue;
                };
                let kind = event
                    .get("kind")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let timestamp = event
                    .get("timestamp")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if created_at.is_empty() {
                    created_at = timestamp.into();
                }
                if !timestamp.is_empty() {
                    updated_at = timestamp.into();
                }
                let payload = event.get("payload").unwrap_or(&Value::Null);
                match kind {
                    "run_started" => {
                        if provider.is_empty() {
                            provider = payload
                                .pointer("/context_budget/provider")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .into();
                        }
                        if model.is_empty() {
                            model = payload
                                .get("model")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .into();
                        }
                        if workspace.as_os_str().is_empty() {
                            workspace = PathBuf::from(
                                payload
                                    .get("workspace")
                                    .and_then(Value::as_str)
                                    .unwrap_or_default(),
                            );
                        }
                        if let Some(prompt) = payload.get("prompt").and_then(Value::as_str) {
                            if title.is_empty() {
                                title = prompt.chars().take(80).collect();
                            }
                            messages.push(BrainMessage::text("user", prompt));
                        }
                    }
                    "model_response" => {
                        let text = payload
                            .get("text")
                            .and_then(Value::as_str)
                            .unwrap_or_default();
                        let calls = payload
                            .get("tool_calls")
                            .cloned()
                            .and_then(|value| {
                                serde_json::from_value::<Vec<CompletedToolCall>>(value).ok()
                            })
                            .unwrap_or_default();
                        messages.push(BrainMessage {
                            role: "assistant".into(),
                            content: (!text.is_empty())
                                .then(|| MessageContent::Text { text: text.into() })
                                .into_iter()
                                .collect(),
                            origin: MessageOrigin::Assistant,
                            tool_call_id: None,
                            tool_calls: calls,
                        });
                    }
                    "tool_result" => {
                        let Some(call_id) = payload.get("id").and_then(Value::as_str) else {
                            continue;
                        };
                        let content = payload
                            .get("result")
                            .cloned()
                            .unwrap_or_else(|| json!({"error": payload.get("error")}));
                        messages.push(BrainMessage {
                            role: "tool".into(),
                            content: vec![MessageContent::Text {
                                text: content.to_string(),
                            }],
                            origin: MessageOrigin::ToolResult,
                            tool_call_id: Some(call_id.into()),
                            tool_calls: Vec::new(),
                        });
                    }
                    _ => {}
                }
            }
            if messages.is_empty() || model.is_empty() || workspace.as_os_str().is_empty() {
                continue;
            }
            let now = Utc::now().to_rfc3339();
            self.checkpoint(&ConversationSnapshot {
                summary: ConversationSummary {
                    id,
                    title: if title.is_empty() {
                        "Imported conversation".into()
                    } else {
                        title
                    },
                    provider,
                    model,
                    workspace,
                    artifact_dir: path,
                    status: "legacy_reconstructed".into(),
                    legacy_imported: true,
                    created_at: if created_at.is_empty() {
                        now.clone()
                    } else {
                        created_at
                    },
                    updated_at: if updated_at.is_empty() {
                        now
                    } else {
                        updated_at
                    },
                },
                messages,
                state: json!({"legacy_reconstructed": true}),
            })?;
            imported += 1;
        }
        Ok(imported)
    }
}

fn display_events_for_message(
    sequence: u64,
    message: &BrainMessage,
) -> Vec<ConversationDisplayMessage> {
    let (sender, kind) = match message.origin {
        MessageOrigin::UserInput => ("user", "prompt"),
        MessageOrigin::UserGuidance => ("user", "guidance"),
        MessageOrigin::Assistant => ("agent", "response"),
        _ => return Vec::new(),
    };
    let text = message
        .content
        .iter()
        .filter_map(|part| match part {
            MessageContent::Text { text } => Some(text.as_str()),
            MessageContent::ImagePng { .. } => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    // Text the model wrote alongside its tool calls came first when the turn
    // was streamed, so it comes first when the conversation is restored.
    let mut events = Vec::new();
    if !text.trim().is_empty() {
        events.push(ConversationDisplayMessage {
            sequence,
            subsequence: 0,
            sender: sender.into(),
            kind: kind.into(),
            text,
            tool: None,
            arguments: None,
        });
    }
    if message.origin == MessageOrigin::Assistant {
        let offset = events.len();
        events.extend(message.tool_calls.iter().enumerate().map(|(index, call)| {
            ConversationDisplayMessage {
                sequence,
                subsequence: u32::try_from(offset + index).unwrap_or(u32::MAX),
                sender: "system".into(),
                kind: "activity".into(),
                text: format!("Running {}", call.name),
                tool: Some(call.name.clone()),
                arguments: Some(call.arguments.clone()),
            }
        }));
    }
    events
}

type SummaryRow = (
    String,
    String,
    String,
    String,
    String,
    String,
    String,
    bool,
    String,
    String,
);

fn summary_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<SummaryRow> {
    Ok((
        row.get(0)?,
        row.get(1)?,
        row.get(2)?,
        row.get(3)?,
        row.get(4)?,
        row.get(5)?,
        row.get(6)?,
        row.get(7)?,
        row.get(8)?,
        row.get(9)?,
    ))
}

fn parse_summary(row: SummaryRow) -> Result<ConversationSummary> {
    Ok(ConversationSummary {
        id: Uuid::parse_str(&row.0).map_err(|error| crate::PokError::Other(error.into()))?,
        title: row.1,
        provider: row.2,
        model: row.3,
        workspace: PathBuf::from(row.4),
        artifact_dir: PathBuf::from(row.5),
        status: row.6,
        legacy_imported: row.7,
        created_at: row.8,
        updated_at: row.9,
    })
}

pub fn sanitized_persisted_messages(messages: &[BrainMessage]) -> Vec<BrainMessage> {
    messages.iter().cloned().map(|mut message| {
        let had_images = message.content.iter().any(|part| matches!(part, MessageContent::ImagePng { .. }));
        message.content.retain(|part| !matches!(part, MessageContent::ImagePng { .. }));
        if had_images { message.content.push(MessageContent::Text { text: "[image omitted from durable conversation; re-observe current state before acting]".into() }); }
        message
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_round_trip_and_latest_preserve_metadata() {
        let temp = tempfile::tempdir().unwrap();
        let store = ConversationStore::open(temp.path()).unwrap();
        let id = Uuid::new_v4();
        let snapshot = ConversationSnapshot {
            summary: ConversationSummary {
                id,
                title: "Resume this task".into(),
                provider: "openrouter".into(),
                model: "example/model".into(),
                workspace: temp.path().join("workspace"),
                artifact_dir: temp.path().join("diagnostics").join(id.to_string()),
                status: "active".into(),
                legacy_imported: false,
                created_at: "2026-08-29T00:00:00Z".into(),
                updated_at: "2026-08-29T00:01:00Z".into(),
            },
            messages: vec![BrainMessage::text("user", "keep going")],
            state: json!({"turn": 2}),
        };
        store.checkpoint(&snapshot).unwrap();

        let restored = store.latest(None).unwrap().unwrap();
        assert_eq!(restored.summary.id, id);
        assert_eq!(restored.summary.provider, "openrouter");
        assert_eq!(restored.summary.model, "example/model");
        assert_eq!(restored.messages.len(), 1);
        assert_eq!(restored.state, json!({"turn": 2}));
    }

    #[test]
    fn persisted_messages_strip_image_payloads() {
        let messages = vec![BrainMessage {
            role: "user".into(),
            content: vec![MessageContent::ImagePng {
                base64: "secret-screen".into(),
            }],
            origin: MessageOrigin::ToolImage,
            tool_call_id: None,
            tool_calls: Vec::new(),
        }];
        let sanitized = sanitized_persisted_messages(&messages);
        let encoded = serde_json::to_string(&sanitized).unwrap();
        assert!(!encoded.contains("secret-screen"));
        assert!(encoded.contains("image omitted"));
    }

    #[test]
    fn display_history_pages_preserve_message_and_tool_order() {
        let temp = tempfile::tempdir().unwrap();
        let store = ConversationStore::open(temp.path()).unwrap();
        let id = Uuid::new_v4();
        let mut messages = Vec::new();
        for index in 0..120 {
            messages.push(BrainMessage::text("user", format!("request {index}")));
            messages.push(BrainMessage {
                role: "assistant".into(),
                content: Vec::new(),
                origin: MessageOrigin::Assistant,
                tool_call_id: None,
                tool_calls: vec![CompletedToolCall {
                    id: format!("call-{index}"),
                    name: "capture_screen".into(),
                    arguments: json!({"index": index}),
                }],
            });
            messages.push(BrainMessage::text("assistant", format!("answer {index}")));
        }
        store
            .checkpoint(&ConversationSnapshot {
                summary: ConversationSummary {
                    id,
                    title: "Paged history".into(),
                    provider: "test".into(),
                    model: "test".into(),
                    workspace: temp.path().into(),
                    artifact_dir: temp.path().join("diagnostics"),
                    status: "completed".into(),
                    legacy_imported: false,
                    created_at: "2026-09-20T00:00:00Z".into(),
                    updated_at: "2026-09-20T00:01:00Z".into(),
                },
                messages,
                state: json!({}),
            })
            .unwrap();

        let latest = store.display_messages_page(id, None, 100).unwrap();
        assert!(latest.has_more);
        assert_eq!(latest.next_before_sequence, Some(260));
        assert_eq!(latest.messages.first().unwrap().sequence, 260);
        assert_eq!(latest.messages.last().unwrap().sequence, 359);
        assert!(latest.messages.iter().any(|message| {
            message.kind == "activity" && message.tool.as_deref() == Some("capture_screen")
        }));

        let previous = store
            .display_messages_page(id, latest.next_before_sequence, 100)
            .unwrap();
        assert_eq!(previous.messages.first().unwrap().sequence, 160);
        assert_eq!(previous.messages.last().unwrap().sequence, 259);
        assert!(
            previous
                .messages
                .windows(2)
                .all(|pair| pair[0].sequence <= pair[1].sequence)
        );
    }

    #[test]
    fn restored_turns_keep_the_text_written_before_their_tool_calls() {
        let message = BrainMessage {
            role: "assistant".into(),
            content: vec![MessageContent::Text {
                text: "Opening Settings first.".into(),
            }],
            origin: MessageOrigin::Assistant,
            tool_call_id: None,
            tool_calls: vec![
                CompletedToolCall {
                    id: "call-1".into(),
                    name: "open_application".into(),
                    arguments: json!({"name": "Settings"}),
                },
                CompletedToolCall {
                    id: "call-2".into(),
                    name: "capture_screen".into(),
                    arguments: json!({}),
                },
            ],
        };
        let events = display_events_for_message(7, &message);
        let shape = events
            .iter()
            .map(|event| {
                (
                    event.subsequence,
                    event.kind.as_str(),
                    event.tool.as_deref(),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            shape,
            [
                (0, "response", None),
                (1, "activity", Some("open_application")),
                (2, "activity", Some("capture_screen")),
            ]
        );
    }
}
