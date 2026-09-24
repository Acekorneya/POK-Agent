use std::{path::Path, sync::Arc};

use chrono::Utc;
use parking_lot::Mutex;
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{Result, memory::sanitize_fts_query};

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS context_entries (
    id TEXT PRIMARY KEY,
    sequence INTEGER NOT NULL,
    role TEXT NOT NULL,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    metadata TEXT NOT NULL,
    token_estimate INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS context_entries_fts USING fts5(id UNINDEXED, text);
CREATE INDEX IF NOT EXISTS context_entries_sequence ON context_entries(sequence);
"#;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchivedContextEntry {
    pub id: Uuid,
    pub sequence: u64,
    pub role: String,
    pub kind: String,
    pub text: String,
    pub metadata: serde_json::Value,
    pub token_estimate: u64,
    pub created_at: String,
}

pub struct SessionArchive {
    connection: Mutex<Connection>,
}

impl SessionArchive {
    pub fn open(artifact_dir: &Path) -> Result<Arc<Self>> {
        std::fs::create_dir_all(artifact_dir)?;
        let connection = Connection::open(artifact_dir.join("context.db"))?;
        connection.execute_batch(SCHEMA)?;
        Ok(Arc::new(Self {
            connection: Mutex::new(connection),
        }))
    }

    pub fn append(
        &self,
        role: &str,
        kind: &str,
        text: &str,
        metadata: serde_json::Value,
    ) -> Result<ArchivedContextEntry> {
        let text = text.trim();
        let mut connection = self.connection.lock();
        let sequence = connection.query_row(
            "SELECT COALESCE(MAX(sequence), 0) + 1 FROM context_entries",
            [],
            |row| row.get::<_, u64>(0),
        )?;
        let archived_text = text.chars().take(64_000).collect::<String>();
        let entry = ArchivedContextEntry {
            id: Uuid::new_v4(),
            sequence,
            role: role.into(),
            kind: kind.into(),
            token_estimate: u64::try_from(archived_text.len().div_ceil(4)).unwrap_or(u64::MAX),
            text: archived_text,
            metadata,
            created_at: Utc::now().to_rfc3339(),
        };
        let transaction = connection.transaction()?;
        transaction.execute(
            "INSERT INTO context_entries(id, sequence, role, kind, text, metadata, token_estimate, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                entry.id.to_string(), entry.sequence, entry.role, entry.kind, entry.text,
                serde_json::to_string(&entry.metadata)?, entry.token_estimate, entry.created_at,
            ],
        )?;
        transaction.execute(
            "INSERT INTO context_entries_fts(id, text) VALUES (?1, ?2)",
            params![entry.id.to_string(), entry.text],
        )?;
        transaction.commit()?;
        Ok(entry)
    }

    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<ArchivedContextEntry>> {
        let query = sanitize_fts_query(query);
        if query.is_empty() {
            return Ok(Vec::new());
        }
        let connection = self.connection.lock();
        let mut statement = connection.prepare(
            "SELECT e.id, e.sequence, e.role, e.kind, e.text, e.metadata, e.token_estimate, e.created_at
             FROM context_entries_fts f JOIN context_entries e ON e.id = f.id
             WHERE context_entries_fts MATCH ?1 ORDER BY bm25(context_entries_fts), e.sequence DESC LIMIT ?2",
        )?;
        let rows = statement.query_map(
            params![query, i64::try_from(limit.min(20)).unwrap_or(20)],
            context_entry_from_row,
        )?;
        rows.map(|row| parse_context_entry(row?)).collect()
    }

    pub fn read(&self, id: Uuid) -> Result<Option<ArchivedContextEntry>> {
        let connection = self.connection.lock();
        let mut statement = connection.prepare(
            "SELECT id, sequence, role, kind, text, metadata, token_estimate, created_at
             FROM context_entries WHERE id = ?1",
        )?;
        let mut rows = statement.query(params![id.to_string()])?;
        rows.next()?
            .map(context_entry_from_row)
            .transpose()?
            .map(parse_context_entry)
            .transpose()
    }

    pub fn stats(&self) -> Result<(u64, u64)> {
        Ok(self.connection.lock().query_row(
            "SELECT COUNT(*), COALESCE(SUM(token_estimate), 0) FROM context_entries",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?)
    }
}

type ContextRow = (String, u64, String, String, String, String, u64, String);

fn context_entry_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ContextRow> {
    Ok((
        row.get(0)?,
        row.get(1)?,
        row.get(2)?,
        row.get(3)?,
        row.get(4)?,
        row.get(5)?,
        row.get(6)?,
        row.get(7)?,
    ))
}

fn parse_context_entry(row: ContextRow) -> Result<ArchivedContextEntry> {
    Ok(ArchivedContextEntry {
        id: Uuid::parse_str(&row.0).map_err(|error| crate::PokError::Other(error.into()))?,
        sequence: row.1,
        role: row.2,
        kind: row.3,
        text: row.4,
        metadata: serde_json::from_str(&row.5)?,
        token_estimate: row.6,
        created_at: row.7,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn archive_is_searchable_and_reopens() {
        let temp = tempfile::tempdir().unwrap();
        let archive = SessionArchive::open(temp.path()).unwrap();
        let entry = archive
            .append(
                "user",
                "conversation",
                "play electronic music",
                serde_json::json!({}),
            )
            .unwrap();
        assert_eq!(
            archive.search("electronic music", 5).unwrap()[0].id,
            entry.id
        );
        drop(archive);
        assert_eq!(
            SessionArchive::open(temp.path())
                .unwrap()
                .read(entry.id)
                .unwrap()
                .unwrap()
                .text,
            "play electronic music"
        );
    }
}
