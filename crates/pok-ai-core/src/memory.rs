use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use rusqlite::{Connection, OptionalExtension, params};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::{PokError, Result};

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    text TEXT NOT NULL,
    approved INTEGER NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    fingerprint TEXT,
    updated_at TEXT,
    kind TEXT NOT NULL DEFAULT 'fact',
    reinforcement_count INTEGER NOT NULL DEFAULT 1,
    review_status TEXT NOT NULL DEFAULT 'independent',
    related_memory_id TEXT,
    version INTEGER NOT NULL DEFAULT 1,
    provenance TEXT NOT NULL DEFAULT 'legacy'
);
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(id UNINDEXED, text);
CREATE TABLE IF NOT EXISTS procedures (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    task_signature TEXT NOT NULL,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    applications TEXT NOT NULL,
    steps TEXT NOT NULL,
    command_template TEXT,
    evidence TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    success_count INTEGER NOT NULL DEFAULT 1,
    retrieval_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    fingerprint TEXT NOT NULL UNIQUE
);
CREATE VIRTUAL TABLE IF NOT EXISTS procedures_fts USING fts5(
    id UNINDEXED, task_signature, title, summary, applications, command_template
);
CREATE TABLE IF NOT EXISTS memory_tombstones (
    fingerprint TEXT PRIMARY KEY,
    item_type TEXT NOT NULL,
    deleted_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memory_schema (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memory_merges (
    id TEXT PRIMARY KEY,
    canonical_id TEXT NOT NULL,
    merged_records_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    undone INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS memory_semantic_suppressions (
    id TEXT PRIMARY KEY,
    normalized_text TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"#;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: Uuid,
    pub source: String,
    pub text: String,
    pub approved: bool,
    pub enabled: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub reinforcement_count: u64,
    pub review_status: String,
    pub related_memory_id: Option<Uuid>,
    pub version: u64,
    pub provenance: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MemoryWriteProvenance {
    ExplicitUser,
    InferredCuration,
    Legacy,
}

impl MemoryWriteProvenance {
    fn as_str(self) -> &'static str {
        match self {
            Self::ExplicitUser => "explicit_user",
            Self::InferredCuration => "inferred_curation",
            Self::Legacy => "legacy",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySaveOutcome {
    pub record: MemoryRecord,
    pub disposition: String,
    pub related_memory_id: Option<Uuid>,
    pub local_similarity: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryDuplicateGroup {
    pub id: String,
    pub records: Vec<MemoryRecord>,
    pub similarity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryHit {
    pub id: Uuid,
    pub source: String,
    pub text: String,
    pub score: f64,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcedureKind {
    Workflow,
    Command,
}

impl ProcedureKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Workflow => "workflow",
            Self::Command => "command",
        }
    }

    fn parse(value: &str) -> Result<Self> {
        match value {
            "workflow" => Ok(Self::Workflow),
            "command" => Ok(Self::Command),
            _ => Err(PokError::Tool(format!("unknown procedure kind `{value}`"))),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct ProcedureStep {
    pub tool: String,
    pub instruction: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcedureRecord {
    pub id: Uuid,
    pub kind: ProcedureKind,
    pub task_signature: String,
    pub title: String,
    pub summary: String,
    pub applications: Vec<String>,
    pub steps: Vec<ProcedureStep>,
    pub command_template: Option<String>,
    pub evidence: String,
    pub enabled: bool,
    pub success_count: u64,
    pub retrieval_count: u64,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub fingerprint: String,
}

#[derive(Debug, Clone)]
pub struct NewProcedure {
    pub kind: ProcedureKind,
    pub task_signature: String,
    pub title: String,
    pub summary: String,
    pub applications: Vec<String>,
    pub steps: Vec<ProcedureStep>,
    pub command_template: Option<String>,
    pub evidence: String,
    pub fingerprint: String,
    pub verified_successes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRetrieval {
    pub id: Uuid,
    pub category: String,
    pub source: String,
    pub text: String,
    pub score: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryContext {
    pub workflows: Vec<MemoryRetrieval>,
    pub commands: Vec<MemoryRetrieval>,
    pub facts: Vec<MemoryRetrieval>,
}

impl MemoryContext {
    pub fn is_empty(&self) -> bool {
        self.workflows.is_empty() && self.commands.is_empty() && self.facts.is_empty()
    }

    pub fn all(&self) -> impl Iterator<Item = &MemoryRetrieval> {
        self.workflows
            .iter()
            .chain(self.commands.iter())
            .chain(self.facts.iter())
    }
}

pub struct MemoryStore {
    connection: Mutex<Connection>,
    root: PathBuf,
}

impl MemoryStore {
    pub fn open(root: impl AsRef<Path>) -> Result<Arc<Self>> {
        let root = root.as_ref().to_path_buf();
        std::fs::create_dir_all(root.join("skills"))?;
        let connection = Connection::open(root.join("memory.db"))?;
        connection.execute_batch(SCHEMA)?;
        migrate_memory_schema(&connection, &root)?;
        Ok(Arc::new(Self {
            connection: Mutex::new(connection),
            root,
        }))
    }

    pub fn save(&self, source: &str, text: &str, approved: bool) -> Result<MemoryRecord> {
        self.save_with_provenance(
            source,
            text,
            approved,
            if approved {
                MemoryWriteProvenance::ExplicitUser
            } else {
                MemoryWriteProvenance::InferredCuration
            },
        )
        .map(|outcome| outcome.record)
    }

    pub fn save_with_provenance(
        &self,
        source: &str,
        text: &str,
        approved: bool,
        provenance: MemoryWriteProvenance,
    ) -> Result<MemorySaveOutcome> {
        let fingerprint = memory_fingerprint(text);
        let mut connection = self.connection.lock();
        if approved {
            connection.execute(
                "DELETE FROM memory_tombstones WHERE fingerprint = ?1 AND item_type = 'fact'",
                params![fingerprint],
            )?;
        } else if tombstone_exists(&connection, &fingerprint, "fact")? {
            return Err(PokError::Tool(
                "this inferred fact was previously deleted by the user".into(),
            ));
        }
        if provenance == MemoryWriteProvenance::InferredCuration
            && semantic_suppression_exists(&connection, text)?
        {
            return Err(PokError::Tool(
                "this inferred fact is similar to content the user explicitly chose to forget"
                    .into(),
            ));
        }
        if let Some(id) = connection
            .query_row(
                "SELECT id FROM memories WHERE fingerprint = ?1",
                params![fingerprint],
                |row| row.get::<_, String>(0),
            )
            .optional()?
        {
            let now = Utc::now().to_rfc3339();
            connection.execute(
                "UPDATE memories SET updated_at = ?2,
                 reinforcement_count = reinforcement_count + 1,
                 version = version + 1 WHERE id = ?1",
                params![id, now],
            )?;
            let record = load_memory_record(&connection, &id)?;
            return Ok(MemorySaveOutcome {
                related_memory_id: Some(record.id),
                record,
                disposition: "reinforced_exact".into(),
                local_similarity: Some(1.0),
            });
        }
        let related = related_memory_candidates(&connection, text, 16)?;
        let best = related
            .iter()
            .map(|record| (record, memory_similarity(text, &record.text)))
            .max_by(|left, right| left.1.total_cmp(&right.1));
        if let Some((existing, similarity)) = best
            && similarity >= 0.92
            && protected_terms_match(text, &existing.text)
        {
            let now = Utc::now().to_rfc3339();
            connection.execute(
                "UPDATE memories SET updated_at = ?2,
                 reinforcement_count = reinforcement_count + 1,
                 version = version + 1 WHERE id = ?1",
                params![existing.id.to_string(), now],
            )?;
            let record = load_memory_record(&connection, &existing.id.to_string())?;
            return Ok(MemorySaveOutcome {
                related_memory_id: Some(record.id),
                record,
                disposition: "reinforced_local".into(),
                local_similarity: Some(similarity),
            });
        }
        let possible_duplicate = best
            .filter(|(_, similarity)| *similarity >= 0.58)
            .map(|(record, similarity)| (record.id, similarity));
        let record = MemoryRecord {
            id: Uuid::new_v4(),
            source: source.into(),
            text: text.trim().into(),
            approved,
            enabled: true,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            reinforcement_count: 1,
            review_status: possible_duplicate
                .map_or_else(|| "independent".into(), |_| "possible_duplicate".into()),
            related_memory_id: possible_duplicate.map(|value| value.0),
            version: 1,
            provenance: provenance.as_str().into(),
        };
        let transaction = connection.transaction()?;
        transaction.execute(
            "INSERT INTO memories(id, source, text, approved, enabled, created_at, fingerprint, updated_at, kind,
             reinforcement_count, review_status, related_memory_id, version, provenance)
             VALUES (?1, ?2, ?3, ?4, 1, ?5, ?6, ?5, 'fact', 1, ?7, ?8, 1, ?9)",
            params![record.id.to_string(), record.source, record.text, record.approved, record.created_at.to_rfc3339(), fingerprint,
                record.review_status, record.related_memory_id.map(|id| id.to_string()), record.provenance],
        )?;
        transaction.execute(
            "INSERT INTO memories_fts(id, text) VALUES (?1, ?2)",
            params![record.id.to_string(), record.text],
        )?;
        transaction.commit()?;
        Ok(MemorySaveOutcome {
            related_memory_id: record.related_memory_id,
            record,
            disposition: if possible_duplicate.is_some() {
                "possible_duplicate".into()
            } else {
                "created".into()
            },
            local_similarity: possible_duplicate.map(|value| value.1),
        })
    }

    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<MemoryHit>> {
        let fts_query = sanitize_fts_query(query);
        if fts_query.is_empty() {
            return Ok(Vec::new());
        }
        let connection = self.connection.lock();
        let mut statement = connection.prepare(
            "SELECT m.id, m.source, m.text, bm25(memories_fts), m.created_at
             FROM memories_fts JOIN memories m ON m.id = memories_fts.id
             WHERE memories_fts MATCH ?1 AND m.approved = 1 AND m.enabled = 1
             ORDER BY bm25(memories_fts), m.created_at DESC LIMIT ?2",
        )?;
        let rows = statement.query_map(
            params![fts_query, i64::try_from(limit).unwrap_or(i64::MAX)],
            |row| {
                let id: String = row.get(0)?;
                let created_at: String = row.get(4)?;
                Ok((
                    id,
                    row.get(1)?,
                    row.get(2)?,
                    row.get::<_, f64>(3)?,
                    created_at,
                ))
            },
        )?;
        rows.map(|row| {
            let (id, source, text, score, created_at) = row?;
            Ok(MemoryHit {
                id: Uuid::parse_str(&id).map_err(|error| PokError::Other(error.into()))?,
                source,
                text,
                score: -score,
                created_at: DateTime::parse_from_rfc3339(&created_at)
                    .map_err(|error| PokError::Other(error.into()))?
                    .with_timezone(&Utc),
            })
        })
        .collect()
    }

    pub fn list_drafts(&self, limit: usize) -> Result<Vec<MemoryRecord>> {
        let connection = self.connection.lock();
        let mut statement = connection.prepare(
            "SELECT id, source, text, approved, enabled, created_at,
                    COALESCE(updated_at, created_at), reinforcement_count, review_status,
                    related_memory_id, version, provenance FROM memories
             WHERE approved = 0 ORDER BY created_at DESC LIMIT ?1",
        )?;
        let rows = statement.query_map(
            params![i64::try_from(limit).unwrap_or(i64::MAX)],
            memory_record_from_row,
        )?;
        rows.map(parse_memory_record).collect()
    }

    pub fn approve(&self, id: Uuid) -> Result<bool> {
        Ok(self.connection.lock().execute(
            "UPDATE memories SET approved = 1 WHERE id = ?1 AND approved = 0",
            params![id.to_string()],
        )? == 1)
    }

    pub fn reject_draft(&self, id: Uuid) -> Result<bool> {
        let mut connection = self.connection.lock();
        let transaction = connection.transaction()?;
        let fingerprint = transaction
            .query_row(
                "SELECT fingerprint FROM memories WHERE id = ?1 AND approved = 0",
                params![id.to_string()],
                |row| row.get::<_, String>(0),
            )
            .optional()?;
        let Some(fingerprint) = fingerprint else {
            return Ok(false);
        };
        transaction.execute(
            "INSERT OR REPLACE INTO memory_tombstones(fingerprint, item_type, deleted_at)
             VALUES (?1, 'fact', ?2)",
            params![fingerprint, Utc::now().to_rfc3339()],
        )?;
        transaction.execute(
            "DELETE FROM memories_fts WHERE id = ?1",
            params![id.to_string()],
        )?;
        transaction.execute(
            "DELETE FROM memories WHERE id = ?1",
            params![id.to_string()],
        )?;
        transaction.commit()?;
        Ok(true)
    }

    pub fn list_approved(&self, source: &str, limit: usize) -> Result<Vec<MemoryRecord>> {
        let connection = self.connection.lock();
        let mut statement = connection.prepare(
            "SELECT id, source, text, approved, enabled, created_at,
                    COALESCE(updated_at, created_at), reinforcement_count, review_status,
                    related_memory_id, version, provenance FROM memories
             WHERE source = ?1 AND approved = 1 AND enabled = 1
             ORDER BY created_at DESC LIMIT ?2",
        )?;
        let rows = statement.query_map(
            params![source, i64::try_from(limit).unwrap_or(i64::MAX)],
            memory_record_from_row,
        )?;
        rows.map(parse_memory_record).collect()
    }

    pub fn save_skill(&self, name: &str, markdown: &str) -> Result<PathBuf> {
        if name.is_empty()
            || !name
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_'))
        {
            return Err(PokError::Tool(
                "skill name may contain only letters, numbers, '-' and '_'".into(),
            ));
        }
        let dir = self.root.join("skills").join(name);
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("SKILL.md");
        atomic_write(&path, markdown.as_bytes())?;
        let source = format!("skill:{name}");
        let mut connection = self.connection.lock();
        let existing = {
            let mut statement = connection.prepare("SELECT id FROM memories WHERE source = ?1")?;
            statement
                .query_map(params![source], |row| row.get::<_, String>(0))?
                .collect::<std::result::Result<Vec<_>, _>>()?
        };
        let transaction = connection.transaction()?;
        for id in existing {
            transaction.execute("DELETE FROM memories_fts WHERE id = ?1", params![id])?;
            transaction.execute("DELETE FROM memories WHERE id = ?1", params![id])?;
        }
        let record = MemoryRecord {
            id: Uuid::new_v4(),
            source,
            text: markdown.into(),
            approved: true,
            enabled: true,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            reinforcement_count: 1,
            review_status: "independent".into(),
            related_memory_id: None,
            version: 1,
            provenance: "user_skill".into(),
        };
        transaction.execute(
            "INSERT INTO memories(id, source, text, approved, enabled, created_at) VALUES (?1, ?2, ?3, 1, 1, ?4)",
            params![record.id.to_string(), record.source, record.text, record.created_at.to_rfc3339()],
        )?;
        transaction.execute(
            "INSERT INTO memories_fts(id, text) VALUES (?1, ?2)",
            params![record.id.to_string(), record.text],
        )?;
        transaction.commit()?;
        Ok(path)
    }

    pub fn list_memories(&self, limit: usize) -> Result<Vec<MemoryRecord>> {
        let connection = self.connection.lock();
        let mut statement = connection.prepare(
            "SELECT id, source, text, approved, enabled, created_at,
                    COALESCE(updated_at, created_at), reinforcement_count, review_status,
                    related_memory_id, version, provenance FROM memories
             ORDER BY created_at DESC LIMIT ?1",
        )?;
        let rows = statement.query_map(
            params![i64::try_from(limit).unwrap_or(i64::MAX)],
            memory_record_from_row,
        )?;
        rows.map(parse_memory_record).collect()
    }

    pub fn related_memories(&self, text: &str, limit: usize) -> Result<Vec<MemoryRecord>> {
        related_memory_candidates(&self.connection.lock(), text, limit.min(32))
    }

    pub fn reinforce_memory(
        &self,
        id: Uuid,
        _approved: bool,
        _provenance: MemoryWriteProvenance,
    ) -> Result<MemoryRecord> {
        let connection = self.connection.lock();
        let updated = connection.execute(
            "UPDATE memories SET reinforcement_count = reinforcement_count + 1,
             updated_at = ?2, version = version + 1 WHERE id = ?1",
            params![id.to_string(), Utc::now().to_rfc3339()],
        )?;
        if updated != 1 {
            return Err(PokError::Tool(format!("memory {id} was not found")));
        }
        load_memory_record(&connection, &id.to_string())
    }

    pub fn mark_memory_review(
        &self,
        id: Uuid,
        status: &str,
        related_memory_id: Option<Uuid>,
    ) -> Result<MemoryRecord> {
        if !matches!(
            status,
            "possible_duplicate" | "conflict" | "proposed_update" | "independent"
        ) {
            return Err(PokError::Tool("invalid memory review status".into()));
        }
        let connection = self.connection.lock();
        connection.execute(
            "UPDATE memories SET review_status = ?2, related_memory_id = ?3,
             updated_at = ?4, version = version + 1 WHERE id = ?1",
            params![
                id.to_string(),
                status,
                related_memory_id.map(|id| id.to_string()),
                Utc::now().to_rfc3339()
            ],
        )?;
        load_memory_record(&connection, &id.to_string())
    }

    pub fn find_duplicate_groups(&self, limit: usize) -> Result<Vec<MemoryDuplicateGroup>> {
        let records = self
            .list_memories(limit.min(500))?
            .into_iter()
            .filter(|record| !record.source.starts_with("skill:"))
            .collect::<Vec<_>>();
        let mut groups = Vec::new();
        let mut used = std::collections::HashSet::new();
        for (index, record) in records.iter().enumerate() {
            if used.contains(&record.id) {
                continue;
            }
            let mut related = vec![record.clone()];
            let mut strongest: f64 = 0.0;
            for candidate in records.iter().skip(index + 1) {
                if used.contains(&candidate.id)
                    || !protected_terms_match(&record.text, &candidate.text)
                {
                    continue;
                }
                let similarity = memory_similarity(&record.text, &candidate.text);
                if similarity >= 0.58 {
                    strongest = strongest.max(similarity);
                    related.push(candidate.clone());
                }
            }
            if related.len() > 1 {
                for item in &related {
                    used.insert(item.id);
                }
                groups.push(MemoryDuplicateGroup {
                    id: format!("duplicate:{}", record.id),
                    records: related,
                    similarity: strongest,
                });
            }
        }
        Ok(groups)
    }

    pub fn merge_memories(&self, canonical_id: Uuid, expected: &[(Uuid, u64)]) -> Result<Uuid> {
        if expected.len() < 2 || !expected.iter().any(|(id, _)| *id == canonical_id) {
            return Err(PokError::Tool(
                "a memory merge requires a canonical record and at least one duplicate".into(),
            ));
        }
        let mut connection = self.connection.lock();
        let mut records = Vec::new();
        for (id, version) in expected {
            let record = load_memory_record(&connection, &id.to_string())?;
            if record.version != *version {
                return Err(PokError::Tool(format!(
                    "memory {id} changed after the duplicate scan; scan again before merging"
                )));
            }
            records.push(record);
        }
        let reinforcement_total = records
            .iter()
            .map(|record| record.reinforcement_count)
            .sum::<u64>();
        let merge_id = Uuid::new_v4();
        let now = Utc::now().to_rfc3339();
        let transaction = connection.transaction()?;
        transaction.execute(
            "INSERT INTO memory_merges(id, canonical_id, merged_records_json, created_at, undone)
             VALUES (?1, ?2, ?3, ?4, 0)",
            params![
                merge_id.to_string(),
                canonical_id.to_string(),
                serde_json::to_string(&records)?,
                now
            ],
        )?;
        transaction.execute(
            "UPDATE memories SET reinforcement_count = ?2, updated_at = ?3,
             review_status = 'independent', related_memory_id = NULL, version = version + 1
             WHERE id = ?1",
            params![canonical_id.to_string(), reinforcement_total, now],
        )?;
        for record in records.iter().filter(|record| record.id != canonical_id) {
            transaction.execute(
                "DELETE FROM memories_fts WHERE id = ?1",
                params![record.id.to_string()],
            )?;
            transaction.execute(
                "DELETE FROM memories WHERE id = ?1",
                params![record.id.to_string()],
            )?;
        }
        transaction.commit()?;
        Ok(merge_id)
    }

    pub fn undo_memory_merge(&self, merge_id: Uuid) -> Result<bool> {
        let mut connection = self.connection.lock();
        let snapshot = connection
            .query_row(
                "SELECT canonical_id, merged_records_json FROM memory_merges WHERE id = ?1 AND undone = 0",
                params![merge_id.to_string()],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
            )
            .optional()?;
        let Some((canonical_id, snapshot)) = snapshot else {
            return Ok(false);
        };
        let records: Vec<MemoryRecord> = serde_json::from_str(&snapshot)?;
        let original_canonical = records
            .iter()
            .find(|record| record.id.to_string() == canonical_id)
            .ok_or_else(|| {
                PokError::Tool("memory merge snapshot has no canonical record".into())
            })?;
        let current_canonical = load_memory_record(&connection, &canonical_id)?;
        if current_canonical.version != original_canonical.version.saturating_add(1) {
            return Err(PokError::Tool(
                "canonical memory changed after the merge; undo would overwrite newer changes"
                    .into(),
            ));
        }
        let transaction = connection.transaction()?;
        for record in records {
            transaction.execute(
                "DELETE FROM memories_fts WHERE id = ?1",
                params![record.id.to_string()],
            )?;
            transaction.execute(
                "DELETE FROM memories WHERE id = ?1",
                params![record.id.to_string()],
            )?;
            transaction.execute(
                "INSERT INTO memories(id, source, text, approved, enabled, created_at, fingerprint, updated_at,
                 kind, reinforcement_count, review_status, related_memory_id, version, provenance)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 'fact', ?9, ?10, ?11, ?12, ?13)",
                params![record.id.to_string(), record.source, record.text, record.approved, record.enabled,
                    record.created_at.to_rfc3339(), memory_fingerprint(&record.text), record.updated_at.to_rfc3339(),
                    record.reinforcement_count, record.review_status,
                    record.related_memory_id.map(|id| id.to_string()), record.version, record.provenance],
            )?;
            transaction.execute(
                "INSERT INTO memories_fts(id, text) VALUES (?1, ?2)",
                params![record.id.to_string(), record.text],
            )?;
        }
        transaction.execute(
            "UPDATE memory_merges SET undone = 1 WHERE id = ?1",
            params![merge_id.to_string()],
        )?;
        transaction.commit()?;
        Ok(true)
    }

    pub fn set_memory_enabled(&self, id: Uuid, enabled: bool) -> Result<bool> {
        Ok(self.connection.lock().execute(
            "UPDATE memories SET enabled = ?2 WHERE id = ?1",
            params![id.to_string(), enabled],
        )? == 1)
    }

    pub fn delete_memory(&self, id: Uuid) -> Result<bool> {
        let mut connection = self.connection.lock();
        let transaction = connection.transaction()?;
        if let Some(fingerprint) = transaction
            .query_row(
                "SELECT fingerprint FROM memories WHERE id = ?1",
                params![id.to_string()],
                |row| row.get::<_, Option<String>>(0),
            )
            .optional()?
            .flatten()
        {
            transaction.execute(
                "INSERT OR REPLACE INTO memory_tombstones(fingerprint, item_type, deleted_at)
                 VALUES (?1, 'fact', ?2)",
                params![fingerprint, Utc::now().to_rfc3339()],
            )?;
        }
        transaction.execute(
            "DELETE FROM memories_fts WHERE id = ?1",
            params![id.to_string()],
        )?;
        let deleted = transaction.execute(
            "DELETE FROM memories WHERE id = ?1",
            params![id.to_string()],
        )? == 1;
        transaction.commit()?;
        Ok(deleted)
    }

    pub fn forget_memory_and_prevent_relearning(&self, id: Uuid) -> Result<bool> {
        let mut connection = self.connection.lock();
        let record = connection
            .query_row(
                "SELECT text, fingerprint FROM memories WHERE id = ?1",
                params![id.to_string()],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?)),
            )
            .optional()?;
        let Some((text, fingerprint)) = record else {
            return Ok(false);
        };
        let transaction = connection.transaction()?;
        transaction.execute(
            "INSERT INTO memory_semantic_suppressions(id, normalized_text, created_at)
             VALUES (?1, ?2, ?3)",
            params![
                Uuid::new_v4().to_string(),
                canonical_memory_text(&text),
                Utc::now().to_rfc3339()
            ],
        )?;
        if let Some(fingerprint) = fingerprint {
            transaction.execute(
                "INSERT OR REPLACE INTO memory_tombstones(fingerprint, item_type, deleted_at)
                 VALUES (?1, 'fact', ?2)",
                params![fingerprint, Utc::now().to_rfc3339()],
            )?;
        }
        transaction.execute(
            "DELETE FROM memories_fts WHERE id = ?1",
            params![id.to_string()],
        )?;
        transaction.execute(
            "DELETE FROM memories WHERE id = ?1",
            params![id.to_string()],
        )?;
        transaction.commit()?;
        Ok(true)
    }

    pub fn save_or_reinforce_procedure(
        &self,
        procedure: NewProcedure,
    ) -> Result<(ProcedureRecord, bool)> {
        let mut connection = self.connection.lock();
        if procedure.kind == ProcedureKind::Command {
            return Err(PokError::Tool(
                "one-off commands are not eligible for durable memory".into(),
            ));
        }
        if tombstone_exists(&connection, &procedure.fingerprint, "skill")? {
            return Err(PokError::Tool(
                "this learned skill was previously deleted by the user".into(),
            ));
        }
        let existing_id = connection
            .query_row(
                "SELECT id FROM procedures WHERE fingerprint = ?1",
                params![procedure.fingerprint],
                |row| row.get::<_, String>(0),
            )
            .optional()?;
        if let Some(id) = existing_id {
            connection.execute(
                "UPDATE procedures SET success_count = success_count + ?2, updated_at = ?3,
                 evidence = ?4, enabled = 1 WHERE id = ?1",
                params![
                    id,
                    procedure.verified_successes,
                    Utc::now().to_rfc3339(),
                    procedure.evidence
                ],
            )?;
            return Ok((load_procedure(&connection, &id)?, false));
        }

        let id = Uuid::new_v4();
        let now = Utc::now();
        let applications = serde_json::to_string(&procedure.applications)?;
        let steps = serde_json::to_string(&procedure.steps)?;
        let searchable = procedure_searchable_text(&procedure);
        let transaction = connection.transaction()?;
        transaction.execute(
            "INSERT INTO procedures(
                id, kind, task_signature, title, summary, applications, steps,
                command_template, evidence, enabled, success_count, retrieval_count,
                created_at, updated_at, fingerprint
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 1, ?10, 0, ?11, ?11, ?12)",
            params![
                id.to_string(),
                procedure.kind.as_str(),
                procedure.task_signature,
                procedure.title,
                procedure.summary,
                applications,
                steps,
                procedure.command_template,
                procedure.evidence,
                procedure.verified_successes,
                now.to_rfc3339(),
                procedure.fingerprint,
            ],
        )?;
        transaction.execute(
            "INSERT INTO procedures_fts(id, task_signature, title, summary, applications, command_template)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                id.to_string(),
                searchable.0,
                searchable.1,
                searchable.2,
                searchable.3,
                searchable.4,
            ],
        )?;
        transaction.commit()?;
        let record = load_procedure(&connection, &id.to_string())?;
        write_skill_file(&self.root, &record)?;
        Ok((record, true))
    }

    pub fn list_procedures(
        &self,
        kind: Option<ProcedureKind>,
        limit: usize,
    ) -> Result<Vec<ProcedureRecord>> {
        let connection = self.connection.lock();
        let mut records = Vec::new();
        if let Some(kind) = kind {
            let mut statement = connection.prepare(
                "SELECT id FROM procedures WHERE kind = ?1 ORDER BY updated_at DESC LIMIT ?2",
            )?;
            let ids = statement
                .query_map(
                    params![kind.as_str(), i64::try_from(limit).unwrap_or(i64::MAX)],
                    |row| row.get::<_, String>(0),
                )?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            for id in ids {
                records.push(load_procedure(&connection, &id)?);
            }
        } else {
            let mut statement = connection
                .prepare("SELECT id FROM procedures ORDER BY updated_at DESC LIMIT ?1")?;
            let ids = statement
                .query_map(params![i64::try_from(limit).unwrap_or(i64::MAX)], |row| {
                    row.get::<_, String>(0)
                })?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            for id in ids {
                records.push(load_procedure(&connection, &id)?);
            }
        }
        Ok(records)
    }

    pub fn search_skills(&self, query: &str, limit: usize) -> Result<Vec<ProcedureRecord>> {
        let terms = meaningful_terms(query);
        let mut skills = self
            .list_procedures(Some(ProcedureKind::Workflow), 200)?
            .into_iter()
            .filter(|skill| skill.enabled)
            .filter_map(|skill| {
                deterministic_relevance(
                    &terms,
                    &format!(
                        "{} {} {} {}",
                        skill.title,
                        skill.summary,
                        skill.task_signature,
                        skill.applications.join(" ")
                    ),
                )
                .map(|score| (score, skill))
            })
            .collect::<Vec<_>>();
        skills.sort_by(|left, right| {
            right
                .0
                .partial_cmp(&left.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| right.1.success_count.cmp(&left.1.success_count))
        });
        Ok(skills
            .into_iter()
            .take(limit.min(20))
            .map(|(_, skill)| skill)
            .collect())
    }

    pub fn load_skill(&self, id: Uuid) -> Result<(ProcedureRecord, String)> {
        let connection = self.connection.lock();
        let record = load_procedure(&connection, &id.to_string())?;
        if record.kind != ProcedureKind::Workflow || !record.enabled {
            return Err(PokError::Tool("skill is disabled or unavailable".into()));
        }
        connection.execute(
            "UPDATE procedures SET retrieval_count = retrieval_count + 1 WHERE id = ?1",
            params![id.to_string()],
        )?;
        drop(connection);
        let markdown = std::fs::read_to_string(
            self.root
                .join("skills")
                .join(id.to_string())
                .join("SKILL.md"),
        )?;
        Ok((record, markdown))
    }

    pub fn set_procedure_enabled(&self, id: Uuid, enabled: bool) -> Result<bool> {
        Ok(self.connection.lock().execute(
            "UPDATE procedures SET enabled = ?2, updated_at = ?3 WHERE id = ?1",
            params![id.to_string(), enabled, Utc::now().to_rfc3339()],
        )? == 1)
    }

    pub fn mark_skill_verified(&self, id: Uuid, evidence: &str) -> Result<bool> {
        Ok(self.connection.lock().execute(
            "UPDATE procedures SET success_count = success_count + 1, evidence = ?2,
             updated_at = ?3 WHERE id = ?1 AND kind = 'workflow' AND enabled = 1",
            params![id.to_string(), evidence, Utc::now().to_rfc3339()],
        )? == 1)
    }

    pub fn delete_procedure(&self, id: Uuid) -> Result<bool> {
        let mut connection = self.connection.lock();
        let transaction = connection.transaction()?;
        let record = transaction
            .query_row(
                "SELECT fingerprint FROM procedures WHERE id = ?1",
                params![id.to_string()],
                |row| row.get::<_, String>(0),
            )
            .optional()?;
        if let Some(fingerprint) = &record {
            transaction.execute(
                "INSERT OR REPLACE INTO memory_tombstones(fingerprint, item_type, deleted_at)
                 VALUES (?1, 'skill', ?2)",
                params![fingerprint, Utc::now().to_rfc3339()],
            )?;
        }
        transaction.execute(
            "DELETE FROM procedures_fts WHERE id = ?1",
            params![id.to_string()],
        )?;
        let deleted = transaction.execute(
            "DELETE FROM procedures WHERE id = ?1",
            params![id.to_string()],
        )? == 1;
        transaction.commit()?;
        let skill_dir = self.root.join("skills").join(id.to_string());
        if skill_dir.is_dir() {
            std::fs::remove_dir_all(skill_dir)?;
        }
        Ok(deleted)
    }

    pub fn retrieve_context(&self, query: &str) -> Result<MemoryContext> {
        let fts_query = sanitize_fts_query(query);
        if fts_query.is_empty() {
            return Ok(MemoryContext::default());
        }
        let mut context = MemoryContext::default();
        let connection = self.connection.lock();
        let mut statement = connection.prepare(
            "SELECT p.id, p.kind, p.title, p.summary, p.command_template,
                    p.success_count, bm25(procedures_fts)
             FROM procedures_fts JOIN procedures p ON p.id = procedures_fts.id
             WHERE procedures_fts MATCH ?1 AND p.enabled = 1 AND p.kind = 'workflow'
             ORDER BY bm25(procedures_fts), p.success_count DESC, p.updated_at DESC LIMIT 48",
        )?;
        let rows = statement
            .query_map(params![fts_query], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, Option<String>>(4)?,
                    row.get::<_, u64>(5)?,
                    row.get::<_, f64>(6)?,
                ))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        drop(statement);
        let query_terms = meaningful_terms(query);
        for (id, kind, title, summary, command, successes, bm25) in rows {
            let searchable = format!("{title} {summary}");
            let Some(relevance) = deterministic_relevance(&query_terms, &searchable) else {
                continue;
            };
            let retrieval = MemoryRetrieval {
                id: Uuid::parse_str(&id).map_err(|error| PokError::Other(error.into()))?,
                category: kind.clone(),
                source: format!("procedure:{kind}"),
                text: command.map_or_else(
                    || format!("{title}: {summary}"),
                    |template| format!("{title}: {summary}\nCommand template: {template}"),
                ),
                score: relevance + (-bm25).max(0.0) + (successes as f64).ln_1p() * 0.15,
            };
            if relevance >= 0.5 && retrieval.score >= 0.5 && context.workflows.len() < 2 {
                context.workflows.push(retrieval);
            }
        }
        drop(connection);
        context.facts = self
            .search(query, 30)?
            .into_iter()
            .filter(|hit| !hit.source.starts_with("skill:") && hit.source != "learned_command")
            .filter_map(|mut hit| {
                let relevance = deterministic_relevance(&query_terms, &hit.text)?;
                hit.score = relevance;
                (hit.score >= 0.5).then_some(hit)
            })
            .map(|hit| MemoryRetrieval {
                id: hit.id,
                category: "fact".into(),
                source: hit.source,
                text: hit.text,
                score: hit.score,
            })
            .collect();
        Ok(context)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
}

fn migrate_memory_schema(connection: &Connection, root: &Path) -> Result<()> {
    let mut statement = connection.prepare("PRAGMA table_info(memories)")?;
    let columns = statement
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if !columns.iter().any(|column| column == "enabled") {
        connection.execute(
            "ALTER TABLE memories ADD COLUMN enabled INTEGER NOT NULL DEFAULT 1",
            [],
        )?;
    }
    for (name, definition) in [
        ("fingerprint", "TEXT"),
        ("updated_at", "TEXT"),
        ("kind", "TEXT NOT NULL DEFAULT 'fact'"),
        ("reinforcement_count", "INTEGER NOT NULL DEFAULT 1"),
        ("review_status", "TEXT NOT NULL DEFAULT 'independent'"),
        ("related_memory_id", "TEXT"),
        ("version", "INTEGER NOT NULL DEFAULT 1"),
        ("provenance", "TEXT NOT NULL DEFAULT 'legacy'"),
    ] {
        if !columns.iter().any(|column| column == name) {
            connection.execute(
                &format!("ALTER TABLE memories ADD COLUMN {name} {definition}"),
                [],
            )?;
        }
    }
    let rows = {
        let mut statement =
            connection.prepare("SELECT id, text, created_at, fingerprint FROM memories")?;
        statement
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, Option<String>>(3)?,
                ))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?
    };
    for (id, text, created_at, fingerprint) in rows {
        if fingerprint.as_deref().is_none_or(str::is_empty) {
            connection.execute(
                "UPDATE memories SET fingerprint = ?2,
                 updated_at = COALESCE(updated_at, ?3) WHERE id = ?1",
                params![id, memory_fingerprint(&text), created_at],
            )?;
        }
    }
    let command_ids = {
        let mut statement =
            connection.prepare("SELECT id FROM procedures WHERE kind = 'command'")?;
        statement
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<std::result::Result<Vec<_>, _>>()?
    };
    for id in command_ids {
        connection.execute("DELETE FROM procedures_fts WHERE id = ?1", params![id])?;
        connection.execute("DELETE FROM procedures WHERE id = ?1", params![id])?;
    }
    let workflow_ids = {
        let mut statement =
            connection.prepare("SELECT id FROM procedures WHERE kind = 'workflow'")?;
        statement
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<std::result::Result<Vec<_>, _>>()?
    };
    for id in workflow_ids {
        if let Ok(record) = load_procedure(connection, &id) {
            let _ = write_skill_file(root, &record);
        }
    }
    connection.execute(
        "INSERT OR REPLACE INTO memory_schema(key, value) VALUES ('version', '4')",
        [],
    )?;
    Ok(())
}

fn memory_fingerprint(text: &str) -> String {
    let normalized = canonical_memory_text(text);
    format!("{:x}", Sha256::digest(normalized.as_bytes()))
}

fn canonical_memory_text(text: &str) -> String {
    text.chars()
        .map(|ch| {
            if ch.is_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn memory_similarity(left: &str, right: &str) -> f64 {
    let left = canonical_memory_text(left);
    let right = canonical_memory_text(right);
    if left == right {
        return 1.0;
    }
    let left_terms = left
        .split_whitespace()
        .collect::<std::collections::BTreeSet<_>>();
    let right_terms = right
        .split_whitespace()
        .collect::<std::collections::BTreeSet<_>>();
    if left_terms.is_empty() || right_terms.is_empty() {
        return 0.0;
    }
    let intersection = left_terms.intersection(&right_terms).count() as f64;
    let dice = 2.0 * intersection / (left_terms.len() + right_terms.len()) as f64;
    let containment = intersection / left_terms.len().min(right_terms.len()) as f64;
    dice * 0.65 + containment * 0.35
}

fn protected_terms_match(left: &str, right: &str) -> bool {
    fn protected(value: &str) -> std::collections::BTreeSet<String> {
        canonical_memory_text(value)
            .split_whitespace()
            .filter(|term| {
                term.chars().any(|ch| ch.is_ascii_digit())
                    || matches!(
                        *term,
                        "not" | "never" | "no" | "without" | "before" | "after"
                    )
            })
            .map(str::to_owned)
            .collect()
    }
    protected(left) == protected(right)
}

fn semantic_suppression_exists(connection: &Connection, text: &str) -> Result<bool> {
    let mut statement = connection.prepare(
        "SELECT normalized_text FROM memory_semantic_suppressions ORDER BY created_at DESC LIMIT 500",
    )?;
    let suppressed = statement
        .query_map([], |row| row.get::<_, String>(0))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(suppressed.iter().any(|existing| {
        protected_terms_match(text, existing) && memory_similarity(text, existing) >= 0.72
    }))
}

fn related_memory_candidates(
    connection: &Connection,
    text: &str,
    limit: usize,
) -> Result<Vec<MemoryRecord>> {
    let terms = canonical_memory_text(text)
        .split_whitespace()
        .filter(|term| term.len() >= 3)
        .take(8)
        .map(|term| format!("\"{}\"", term.replace('"', "")))
        .collect::<Vec<_>>();
    if terms.is_empty() {
        return Ok(Vec::new());
    }
    let query = terms.join(" OR ");
    let mut statement = connection.prepare(
        "SELECT m.id, m.source, m.text, m.approved, m.enabled, m.created_at,
                COALESCE(m.updated_at, m.created_at), m.reinforcement_count, m.review_status,
                m.related_memory_id, m.version, m.provenance
         FROM memories_fts JOIN memories m ON m.id = memories_fts.id
         WHERE memories_fts MATCH ?1 AND m.kind = 'fact'
         ORDER BY bm25(memories_fts), m.updated_at DESC LIMIT ?2",
    )?;
    let rows = statement.query_map(
        params![query, i64::try_from(limit).unwrap_or(i64::MAX)],
        memory_record_from_row,
    )?;
    rows.map(parse_memory_record).collect()
}

fn tombstone_exists(connection: &Connection, fingerprint: &str, item_type: &str) -> Result<bool> {
    Ok(connection
        .query_row(
            "SELECT 1 FROM memory_tombstones WHERE fingerprint = ?1 AND item_type = ?2",
            params![fingerprint, item_type],
            |_| Ok(()),
        )
        .optional()?
        .is_some())
}

fn load_memory_record(connection: &Connection, id: &str) -> Result<MemoryRecord> {
    parse_memory_record(connection.query_row(
        "SELECT id, source, text, approved, enabled, created_at,
                COALESCE(updated_at, created_at), reinforcement_count, review_status,
                related_memory_id, version, provenance FROM memories WHERE id = ?1",
        params![id],
        memory_record_from_row,
    ))
}

fn write_skill_file(root: &Path, record: &ProcedureRecord) -> Result<PathBuf> {
    let dir = root.join("skills").join(record.id.to_string());
    std::fs::create_dir_all(&dir)?;
    let applications = if record.applications.is_empty() {
        String::new()
    } else {
        format!("applications: [{}]\n", record.applications.join(", "))
    };
    let steps = record
        .steps
        .iter()
        .enumerate()
        .map(|(index, step)| format!("{}. `{}` — {}", index + 1, step.tool, step.instruction))
        .collect::<Vec<_>>()
        .join("\n");
    let markdown = format!(
        "---\nname: learned-{}\ndescription: {}\nwhen_to_use: {}\n{}---\n\n# {}\n\n{}\n",
        &record.id.to_string()[..8],
        yaml_scalar(&record.summary),
        yaml_scalar(&record.task_signature),
        applications,
        record.title,
        steps,
    );
    let path = dir.join("SKILL.md");
    atomic_write(&path, markdown.as_bytes())?;
    Ok(path)
}

fn yaml_scalar(value: &str) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "\"learned skill\"".into())
}

fn meaningful_terms(text: &str) -> Vec<String> {
    const GENERIC: &[&str] = &[
        "a", "an", "and", "are", "can", "could", "data", "file", "for", "from", "get", "i", "in",
        "is", "it", "me", "my", "of", "on", "open", "please", "read", "the", "this", "to", "use",
        "using", "want", "with", "you",
    ];
    let mut terms = text
        .split(|character: char| !character.is_alphanumeric() && character != '_')
        .map(str::to_ascii_lowercase)
        .filter(|term| term.len() >= 3 && !GENERIC.contains(&term.as_str()))
        .collect::<Vec<_>>();
    terms.sort();
    terms.dedup();
    terms
}

fn deterministic_relevance(query_terms: &[String], candidate: &str) -> Option<f64> {
    if query_terms.is_empty() {
        return None;
    }
    let candidate_terms = meaningful_terms(candidate);
    let matched = query_terms
        .iter()
        .filter(|term| candidate_terms.contains(term))
        .collect::<Vec<_>>();
    let high_signal = matched
        .iter()
        .any(|term| term.len() >= 6 || matches!(term.as_str(), "pdf" | "mkv" | "docx" | "xlsx"));
    (matched.len() >= 2 || (matched.len() == 1 && high_signal))
        .then_some(matched.len() as f64 / query_terms.len().max(1) as f64)
}

type MemoryRow = (
    String,
    String,
    String,
    bool,
    bool,
    String,
    String,
    u64,
    String,
    Option<String>,
    u64,
    String,
);

fn memory_record_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<MemoryRow> {
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
        row.get(10)?,
        row.get(11)?,
    ))
}

fn parse_memory_record(row: rusqlite::Result<MemoryRow>) -> Result<MemoryRecord> {
    let (
        id,
        source,
        text,
        approved,
        enabled,
        created_at,
        updated_at,
        reinforcement_count,
        review_status,
        related_memory_id,
        version,
        provenance,
    ) = row?;
    Ok(MemoryRecord {
        id: Uuid::parse_str(&id).map_err(|error| PokError::Other(error.into()))?,
        source,
        text,
        approved,
        enabled,
        created_at: DateTime::parse_from_rfc3339(&created_at)
            .map_err(|error| PokError::Other(error.into()))?
            .with_timezone(&Utc),
        updated_at: DateTime::parse_from_rfc3339(&updated_at)
            .map_err(|error| PokError::Other(error.into()))?
            .with_timezone(&Utc),
        reinforcement_count,
        review_status,
        related_memory_id: related_memory_id
            .map(|id| Uuid::parse_str(&id))
            .transpose()
            .map_err(|error| PokError::Other(error.into()))?,
        version,
        provenance,
    })
}

fn procedure_searchable_text(
    procedure: &NewProcedure,
) -> (String, String, String, String, Option<String>) {
    (
        procedure.task_signature.clone(),
        procedure.title.clone(),
        procedure.summary.clone(),
        procedure.applications.join(" "),
        procedure.command_template.clone(),
    )
}

fn load_procedure(connection: &Connection, id: &str) -> Result<ProcedureRecord> {
    let values = connection.query_row(
        "SELECT id, kind, task_signature, title, summary, applications, steps,
                command_template, evidence, enabled, success_count, retrieval_count,
                created_at, updated_at, fingerprint
         FROM procedures WHERE id = ?1",
        params![id],
        |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, String>(6)?,
                row.get::<_, Option<String>>(7)?,
                row.get::<_, String>(8)?,
                row.get::<_, bool>(9)?,
                row.get::<_, u64>(10)?,
                row.get::<_, u64>(11)?,
                row.get::<_, String>(12)?,
                row.get::<_, String>(13)?,
                row.get::<_, String>(14)?,
            ))
        },
    )?;
    Ok(ProcedureRecord {
        id: Uuid::parse_str(&values.0).map_err(|error| PokError::Other(error.into()))?,
        kind: ProcedureKind::parse(&values.1)?,
        task_signature: values.2,
        title: values.3,
        summary: values.4,
        applications: serde_json::from_str(&values.5)?,
        steps: serde_json::from_str(&values.6)?,
        command_template: values.7,
        evidence: values.8,
        enabled: values.9,
        success_count: values.10,
        retrieval_count: values.11,
        created_at: DateTime::parse_from_rfc3339(&values.12)
            .map_err(|error| PokError::Other(error.into()))?
            .with_timezone(&Utc),
        updated_at: DateTime::parse_from_rfc3339(&values.13)
            .map_err(|error| PokError::Other(error.into()))?
            .with_timezone(&Utc),
        fingerprint: values.14,
    })
}

pub fn sanitize_fts_query(query: &str) -> String {
    query
        .split_whitespace()
        .map(|term| {
            let clean: String = term
                .chars()
                .filter(|ch| ch.is_alphanumeric() || *ch == '_')
                .collect();
            if clean.is_empty() {
                String::new()
            } else {
                format!("\"{clean}\"")
            }
        })
        .filter(|term| !term.is_empty())
        .collect::<Vec<_>>()
        .join(" OR ")
}

pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| PokError::Tool("path has no parent".into()))?;
    std::fs::create_dir_all(parent)?;
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    std::io::Write::write_all(&mut temporary, bytes)?;
    temporary.as_file().sync_all()?;
    temporary
        .persist(path)
        .map_err(|error| PokError::Io(error.error))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn approved_memory_is_searchable_and_drafts_are_hidden() {
        let dir = tempfile::tempdir().unwrap();
        let memory = MemoryStore::open(dir.path()).unwrap();
        memory
            .save("user", "default printer is OfficeJet", true)
            .unwrap();
        memory.save("draft", "secret draft printer", false).unwrap();
        let hits = memory.search("printer", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].text.contains("OfficeJet"));
        let draft = memory.list_drafts(10).unwrap().pop().unwrap();
        assert!(memory.approve(draft.id).unwrap());
        assert_eq!(memory.search("secret", 10).unwrap().len(), 1);
    }

    #[test]
    fn rejected_memory_draft_is_deleted_without_touching_approved_memory() {
        let dir = tempfile::tempdir().unwrap();
        let memory = MemoryStore::open(dir.path()).unwrap();
        let draft = memory.save("environment", "unwanted guess", false).unwrap();
        let approved = memory.save("user", "keep this fact", true).unwrap();

        assert!(memory.reject_draft(draft.id).unwrap());
        assert!(!memory.reject_draft(approved.id).unwrap());
        assert!(memory.list_drafts(10).unwrap().is_empty());
        assert_eq!(memory.search("keep", 10).unwrap().len(), 1);
        assert!(memory.save("environment", "unwanted guess", false).is_err());
        assert!(
            memory
                .save("user", "unwanted guess", true)
                .expect("an explicit user save overrides the tombstone")
                .approved
        );
    }

    #[test]
    fn saved_skill_is_approved_searchable_and_replaced() {
        let dir = tempfile::tempdir().unwrap();
        let memory = MemoryStore::open(dir.path()).unwrap();
        memory
            .save_skill("discord-message", "Use the obsolete Discord sequence")
            .unwrap();
        memory
            .save_skill("discord-message", "Use the verified Discord workflow")
            .unwrap();
        let hits = memory.search("Discord workflow", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].text.contains("verified"));
    }

    #[test]
    fn list_approved_returns_only_matching_approved_memories() {
        let dir = tempfile::tempdir().unwrap();
        let memory = MemoryStore::open(dir.path()).unwrap();
        memory.save("user", "fact 1", true).unwrap();
        memory.save("user", "fact 2", true).unwrap();
        memory.save("user", "draft fact", false).unwrap();
        memory.save("project", "project fact", true).unwrap();

        let user_facts = memory.list_approved("user", 10).unwrap();
        assert_eq!(user_facts.len(), 2);
        assert!(user_facts.iter().any(|item| item.text == "fact 1"));
        assert!(user_facts.iter().any(|item| item.text == "fact 2"));
    }

    #[test]
    fn local_memory_gate_reinforces_safe_paraphrases() {
        let dir = tempfile::tempdir().unwrap();
        let memory = MemoryStore::open(dir.path()).unwrap();
        let first = memory
            .save("user", "The user prefers the dark color theme", true)
            .unwrap();
        let outcome = memory
            .save_with_provenance(
                "user",
                "User prefers dark color theme",
                false,
                MemoryWriteProvenance::InferredCuration,
            )
            .unwrap();
        assert_eq!(outcome.record.id, first.id);
        assert_eq!(outcome.disposition, "reinforced_local");
        assert_eq!(outcome.record.reinforcement_count, 2);
        assert_eq!(memory.list_memories(10).unwrap().len(), 1);
    }

    #[test]
    fn local_memory_gate_preserves_numeric_and_negative_differences() {
        let dir = tempfile::tempdir().unwrap();
        let memory = MemoryStore::open(dir.path()).unwrap();
        memory
            .save("environment", "The server uses port 512", true)
            .unwrap();
        memory
            .save("environment", "The server uses port 513", false)
            .unwrap();
        memory
            .save("user", "The user likes notifications", true)
            .unwrap();
        memory
            .save("user", "The user does not like notifications", false)
            .unwrap();
        assert_eq!(memory.list_memories(10).unwrap().len(), 4);
    }

    #[test]
    fn inferred_reinforcement_does_not_reenable_disabled_memory() {
        let dir = tempfile::tempdir().unwrap();
        let memory = MemoryStore::open(dir.path()).unwrap();
        let record = memory
            .save("user", "User prefers compact layout", true)
            .unwrap();
        memory.set_memory_enabled(record.id, false).unwrap();
        let outcome = memory
            .save_with_provenance(
                "user",
                "The user prefers compact layout",
                false,
                MemoryWriteProvenance::InferredCuration,
            )
            .unwrap();
        assert_eq!(outcome.record.id, record.id);
        assert!(!outcome.record.enabled);
    }

    #[test]
    fn explicit_reinforcement_preserves_disabled_and_review_state() {
        let dir = tempfile::tempdir().unwrap();
        let memory = MemoryStore::open(dir.path()).unwrap();
        let record = memory
            .save("curator", "User prefers compact layout", false)
            .unwrap();
        memory.set_memory_enabled(record.id, false).unwrap();
        let outcome = memory
            .save_with_provenance(
                "user",
                "User prefers compact layout",
                true,
                MemoryWriteProvenance::ExplicitUser,
            )
            .unwrap();
        assert_eq!(outcome.record.id, record.id);
        assert!(!outcome.record.enabled);
        assert!(!outcome.record.approved);
        assert_eq!(outcome.record.provenance, "inferred_curation");
    }

    #[test]
    fn semantic_forget_is_explicit_and_blocks_only_inferred_relearning() {
        let dir = tempfile::tempdir().unwrap();
        let memory = MemoryStore::open(dir.path()).unwrap();
        let ordinary = memory
            .save("user", "User prefers the ocean blue theme", true)
            .unwrap();
        assert!(memory.delete_memory(ordinary.id).unwrap());
        assert!(
            memory
                .save("curator", "The user prefers an ocean blue theme", false)
                .is_ok(),
            "ordinary deletion retains only an exact fingerprint"
        );

        let forgotten = memory
            .save("user", "User prefers compact navigation panels", true)
            .unwrap();
        assert!(
            memory
                .forget_memory_and_prevent_relearning(forgotten.id)
                .unwrap()
        );
        assert!(
            memory
                .save(
                    "curator",
                    "The user prefers compact navigation panels",
                    false
                )
                .is_err()
        );
    }

    #[test]
    fn reviewed_duplicate_merge_is_versioned_and_undoable() {
        let dir = tempfile::tempdir().unwrap();
        let memory = MemoryStore::open(dir.path()).unwrap();
        let first = memory
            .save("user", "User prefers dark theme", true)
            .unwrap();
        let second = memory
            .save("user", "User strongly prefers dark themes", true)
            .unwrap();
        let expected = vec![(first.id, first.version), (second.id, second.version)];
        let merge_id = memory.merge_memories(first.id, &expected).unwrap();
        assert_eq!(memory.list_memories(10).unwrap().len(), 1);
        assert!(memory.undo_memory_merge(merge_id).unwrap());
        assert_eq!(memory.list_memories(10).unwrap().len(), 2);
    }

    fn procedure(signature: &str, fingerprint: &str, kind: ProcedureKind) -> NewProcedure {
        NewProcedure {
            kind,
            task_signature: signature.into(),
            title: format!("Complete {signature}"),
            summary: format!("Verified approach for {signature}"),
            applications: vec!["example.exe".into()],
            steps: vec![ProcedureStep {
                tool: "click_target".into(),
                instruction: "Resolve and click the fresh target.".into(),
            }],
            command_template: (kind == ProcedureKind::Command).then(|| "tool --check".into()),
            evidence: "verified_test".into(),
            fingerprint: fingerprint.into(),
            verified_successes: 1,
        }
    }

    #[test]
    fn duplicate_procedure_reinforces_one_record() {
        let dir = tempfile::tempdir().unwrap();
        let memory = MemoryStore::open(dir.path()).unwrap();
        let (first, created) = memory
            .save_or_reinforce_procedure(procedure(
                "send channel message",
                "same-fingerprint",
                ProcedureKind::Workflow,
            ))
            .unwrap();
        assert!(created);
        let (second, created) = memory
            .save_or_reinforce_procedure(procedure(
                "send channel message",
                "same-fingerprint",
                ProcedureKind::Workflow,
            ))
            .unwrap();
        assert!(!created);
        assert_eq!(first.id, second.id);
        assert_eq!(second.success_count, 2);
        assert_eq!(memory.list_procedures(None, 10).unwrap().len(), 1);
    }

    #[test]
    fn user_requested_skill_starts_unverified_and_is_verified_after_success() {
        let dir = tempfile::tempdir().unwrap();
        let memory = MemoryStore::open(dir.path()).unwrap();
        let mut requested = procedure(
            "play music in the requested application",
            "user-requested-skill",
            ProcedureKind::Workflow,
        );
        requested.evidence = "explicit_user_request".into();
        requested.verified_successes = 0;
        let (record, created) = memory.save_or_reinforce_procedure(requested).unwrap();
        assert!(created);
        assert_eq!(record.success_count, 0);
        assert!(
            memory
                .mark_skill_verified(record.id, "grounded_successful_use")
                .unwrap()
        );
        let (verified, _) = memory.load_skill(record.id).unwrap();
        assert_eq!(verified.success_count, 1);
        assert_eq!(verified.evidence, "grounded_successful_use");
    }

    #[test]
    fn commands_are_rejected_and_skills_and_facts_are_retrieved() {
        let dir = tempfile::tempdir().unwrap();
        let memory = MemoryStore::open(dir.path()).unwrap();
        memory
            .save_or_reinforce_procedure(procedure(
                "send channel message",
                "workflow-a",
                ProcedureKind::Workflow,
            ))
            .unwrap();
        assert!(
            memory
                .save_or_reinforce_procedure(procedure(
                    "export channel history",
                    "command-b",
                    ProcedureKind::Command,
                ))
                .is_err()
        );
        memory
            .save("user", "Preferred channel is general", true)
            .unwrap();

        let context = memory
            .retrieve_context("send general channel message")
            .unwrap();
        assert_eq!(context.workflows.len(), 1);
        assert!(context.commands.is_empty());
        assert_eq!(context.facts.len(), 1);
    }

    #[test]
    fn weak_single_term_workflow_matches_are_not_injected() {
        let dir = tempfile::tempdir().unwrap();
        let memory = MemoryStore::open(dir.path()).unwrap();
        memory
            .save_or_reinforce_procedure(procedure(
                "send channel message",
                "workflow-a",
                ProcedureKind::Workflow,
            ))
            .unwrap();

        let context = memory.retrieve_context("channel monthly report").unwrap();
        assert!(context.workflows.is_empty());
    }

    #[test]
    fn disabled_and_deleted_records_are_not_retrieved() {
        let dir = tempfile::tempdir().unwrap();
        let memory = MemoryStore::open(dir.path()).unwrap();
        let (record, _) = memory
            .save_or_reinforce_procedure(procedure(
                "open browser settings",
                "disable-me",
                ProcedureKind::Workflow,
            ))
            .unwrap();
        assert!(memory.set_procedure_enabled(record.id, false).unwrap());
        assert!(
            memory
                .retrieve_context("open browser settings")
                .unwrap()
                .workflows
                .is_empty()
        );
        assert!(memory.delete_procedure(record.id).unwrap());
        assert!(memory.list_procedures(None, 10).unwrap().is_empty());
        assert!(
            memory
                .save_or_reinforce_procedure(procedure(
                    "open browser settings",
                    "disable-me",
                    ProcedureKind::Workflow,
                ))
                .is_err()
        );
    }
}
