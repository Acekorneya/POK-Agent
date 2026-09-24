//! Local-first retrieval shared by JEV-on and JEV-off sessions.

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    path::{Path, PathBuf},
};

use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{Result, memory::sanitize_fts_query};

const SCHEMA_VERSION: i64 = 1;
const MAX_FILE_BYTES: u64 = 256 * 1024;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RetrievalIntent {
    Implementation,
    Explanation,
    #[default]
    General,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceCandidate {
    pub id: String,
    pub path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub text: String,
    pub local_score: f64,
}

pub struct RetrievalService {
    workspace: PathBuf,
    connection: Connection,
}

impl RetrievalService {
    pub fn open(data_dir: &Path, workspace: &Path) -> Result<Self> {
        let canonical = dunce::canonicalize(workspace).unwrap_or_else(|_| workspace.to_path_buf());
        let mut hasher = Sha256::new();
        hasher.update(canonical.to_string_lossy().as_bytes());
        let key = format!("{:x}", hasher.finalize());
        let directory = data_dir.join("retrieval").join(&key[..24]);
        std::fs::create_dir_all(&directory)?;
        let connection = Connection::open(directory.join("source-index.db"))?;
        connection.execute_batch(
            "PRAGMA journal_mode=WAL;
             CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
             CREATE TABLE IF NOT EXISTS files(
                 path TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, modified_ns INTEGER NOT NULL
             );
             CREATE TABLE IF NOT EXISTS chunks(
                 id INTEGER PRIMARY KEY, path TEXT NOT NULL, start_line INTEGER NOT NULL,
                 end_line INTEGER NOT NULL, symbol TEXT NOT NULL, content TEXT NOT NULL,
                 is_source INTEGER NOT NULL, fingerprint TEXT NOT NULL
             );
             CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                 path, symbol, content, content='chunks', content_rowid='id',
                 tokenize='porter unicode61'
             );
             CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                 INSERT INTO chunks_fts(rowid,path,symbol,content)
                 VALUES(new.id,new.path,new.symbol,new.content);
             END;
             CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                 INSERT INTO chunks_fts(chunks_fts,rowid,path,symbol,content)
                 VALUES('delete',old.id,old.path,old.symbol,old.content);
             END;",
        )?;
        let version = connection
            .query_row(
                "SELECT value FROM metadata WHERE key='schema_version'",
                [],
                |row| row.get::<_, String>(0),
            )
            .optional()?
            .and_then(|value| value.parse::<i64>().ok());
        if version != Some(SCHEMA_VERSION) {
            connection.execute_batch(
                "DELETE FROM chunks; DELETE FROM files;
                 INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild');",
            )?;
            connection.execute(
                "INSERT OR REPLACE INTO metadata(key,value) VALUES('schema_version',?1)",
                [SCHEMA_VERSION.to_string()],
            )?;
        }
        Ok(Self {
            workspace: canonical,
            connection,
        })
    }

    pub fn refresh(&mut self) -> Result<usize> {
        let mut seen = HashSet::new();
        let mut changed = 0usize;
        let walker = ignore::WalkBuilder::new(&self.workspace)
            .hidden(false)
            .git_ignore(true)
            .git_global(true)
            .git_exclude(true)
            .build();
        for entry in walker.filter_map(std::result::Result::ok) {
            let path = entry.path();
            let Ok(metadata) = entry.metadata() else {
                continue;
            };
            if !path.is_file()
                || metadata.len() > MAX_FILE_BYTES
                || sensitive_path(path)
                || generated_path(path)
            {
                continue;
            }
            let relative = path
                .strip_prefix(&self.workspace)
                .unwrap_or(path)
                .to_string_lossy()
                .replace('\\', "/");
            seen.insert(relative.clone());
            let modified_ns = metadata
                .modified()
                .ok()
                .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(0, |duration| {
                    duration.as_nanos().min(i64::MAX as u128) as i64
                });
            let existing = self
                .connection
                .query_row(
                    "SELECT fingerprint, modified_ns FROM files WHERE path=?1",
                    [&relative],
                    |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?)),
                )
                .optional()?;
            if existing
                .as_ref()
                .is_some_and(|(_, modified)| *modified == modified_ns)
            {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(path) else {
                continue;
            };
            let fingerprint = format!("{:x}", Sha256::digest(text.as_bytes()));
            if existing
                .as_ref()
                .is_some_and(|(known, _)| known == &fingerprint)
            {
                self.connection.execute(
                    "UPDATE files SET modified_ns=?2 WHERE path=?1",
                    params![relative, modified_ns],
                )?;
                continue;
            }
            let transaction = self.connection.transaction()?;
            transaction.execute("DELETE FROM chunks WHERE path=?1", [&relative])?;
            for chunk in chunk_source(&relative, &text) {
                transaction.execute(
                    "INSERT INTO chunks(path,start_line,end_line,symbol,content,is_source,fingerprint) VALUES(?1,?2,?3,?4,?5,?6,?7)",
                    params![relative, chunk.0 as i64, chunk.1 as i64, chunk.2, chunk.3, is_source_path(Path::new(&relative)) as i64, fingerprint],
                )?;
            }
            transaction.execute(
                "INSERT OR REPLACE INTO files(path,fingerprint,modified_ns) VALUES(?1,?2,?3)",
                params![relative, fingerprint, modified_ns],
            )?;
            transaction.commit()?;
            changed += 1;
        }
        let stored = {
            let mut statement = self.connection.prepare("SELECT path FROM files")?;
            statement
                .query_map([], |row| row.get::<_, String>(0))?
                .collect::<std::result::Result<Vec<_>, _>>()?
        };
        for path in stored.into_iter().filter(|path| !seen.contains(path)) {
            let transaction = self.connection.transaction()?;
            transaction.execute("DELETE FROM chunks WHERE path=?1", [&path])?;
            transaction.execute("DELETE FROM files WHERE path=?1", [&path])?;
            transaction.commit()?;
            changed += 1;
        }
        Ok(changed)
    }

    pub fn search(
        &self,
        query: &str,
        intent: RetrievalIntent,
        limit: usize,
    ) -> Result<Vec<SourceCandidate>> {
        let query = sanitize_fts_query(query);
        if query.is_empty() {
            return Ok(Vec::new());
        }
        let broad = self.query_ranking(&query, "bm25(chunks_fts, 0.3, 1.2, 1.0)", 100)?;
        let symbols = self.query_ranking(&query, "bm25(chunks_fts, 0.4, 2.5, 0.8)", 100)?;
        let mut fused = HashMap::<i64, f64>::new();
        for ranking in [&broad, &symbols] {
            for (rank, row) in ranking.iter().enumerate() {
                *fused.entry(row.0).or_default() += 1.0 / (60.0 + rank as f64 + 1.0);
            }
        }
        let rows = broad
            .into_iter()
            .chain(symbols)
            .map(|row| (row.0, row))
            .collect::<BTreeMap<_, _>>();
        let mut ranked = fused
            .into_iter()
            .filter_map(|(id, score)| rows.get(&id).cloned().map(|row| (score, row)))
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| left.1.1.cmp(&right.1.1))
                .then_with(|| left.1.2.cmp(&right.1.2))
        });
        let mut selected = Vec::new();
        if intent == RetrievalIntent::Implementation {
            selected.extend(ranked.iter().filter(|(_, row)| row.6).take(15).cloned());
        }
        for candidate in ranked {
            if selected.iter().any(|(_, row)| row.0 == candidate.1.0) {
                continue;
            }
            selected.push(candidate);
            if selected.len() >= limit.min(30) {
                break;
            }
        }
        let mut accepted: Vec<SourceCandidate> = Vec::new();
        for (score, row) in selected {
            if accepted.iter().any(|current| {
                current.path == row.1
                    && overlap_at_least_half(current.start_line, current.end_line, row.2, row.3)
            }) {
                continue;
            }
            accepted.push(SourceCandidate {
                id: format!("code_context_{}", row.0),
                path: row.1,
                start_line: row.2,
                end_line: row.3,
                text: row.5,
                local_score: score,
            });
            if accepted.len() >= limit.min(30) {
                break;
            }
        }
        Ok(accepted)
    }

    fn query_ranking(&self, query: &str, order: &str, limit: usize) -> Result<Vec<ChunkRow>> {
        let sql = format!(
            "SELECT c.id,c.path,c.start_line,c.end_line,c.symbol,c.content,c.is_source FROM chunks_fts f JOIN chunks c ON c.id=f.rowid WHERE chunks_fts MATCH ?1 ORDER BY {order} LIMIT ?2"
        );
        let mut statement = self.connection.prepare(&sql)?;
        let rows = statement.query_map(params![query, limit as i64], |row| {
            Ok((
                row.get(0)?,
                row.get(1)?,
                row.get::<_, i64>(2)? as usize,
                row.get::<_, i64>(3)? as usize,
                row.get(4)?,
                row.get(5)?,
                row.get::<_, i64>(6)? != 0,
            ))
        })?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }
}

type ChunkRow = (i64, String, usize, usize, String, String, bool);

fn chunk_source(path: &str, text: &str) -> Vec<(usize, usize, String, String)> {
    let lines = text.lines().collect::<Vec<_>>();
    if lines.is_empty() {
        return Vec::new();
    }
    let source = is_source_path(Path::new(path));
    let declaration = regex::Regex::new(r"^\s*(?:pub\s+)?(?:async\s+)?(?:fn|struct|enum|trait|impl|class|def|function|interface|type)\s+([A-Za-z_][A-Za-z0-9_]*)").expect("valid declaration regex");
    let starts = lines
        .iter()
        .enumerate()
        .filter_map(|(index, line)| {
            declaration.captures(line).map(|captures| {
                (
                    index,
                    captures
                        .get(1)
                        .map_or("", |value| value.as_str())
                        .to_owned(),
                )
            })
        })
        .collect::<Vec<_>>();
    if source && !starts.is_empty() {
        let mut chunks = Vec::new();
        for (position, (start, symbol)) in starts.iter().enumerate() {
            let section_end = starts.get(position + 1).map_or(lines.len(), |next| next.0);
            for window_start in (*start..section_end).step_by(120) {
                let end = (window_start + 120).min(section_end);
                chunks.push((
                    window_start + 1,
                    end,
                    symbol.clone(),
                    lines[window_start..end].join("\n"),
                ));
            }
        }
        return chunks;
    }
    (0..lines.len())
        .step_by(30)
        .map(|start| {
            let end = (start + 40).min(lines.len());
            (start + 1, end, String::new(), lines[start..end].join("\n"))
        })
        .collect()
}

fn is_source_path(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|value| value.to_str())
            .unwrap_or("")
            .to_ascii_lowercase()
            .as_str(),
        "rs" | "ts"
            | "tsx"
            | "js"
            | "jsx"
            | "py"
            | "go"
            | "java"
            | "cs"
            | "c"
            | "cc"
            | "cpp"
            | "h"
            | "hpp"
            | "kt"
            | "swift"
    )
}

fn sensitive_path(path: &Path) -> bool {
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    name.starts_with(".env")
        || name.ends_with(".pem")
        || name.ends_with(".key")
        || name.ends_with(".p12")
}

fn generated_path(path: &Path) -> bool {
    path.components().any(|component| {
        matches!(
            component
                .as_os_str()
                .to_string_lossy()
                .to_ascii_lowercase()
                .as_str(),
            ".git" | "node_modules" | "target" | "dist" | "build" | ".next" | "vendor"
        )
    })
}

fn overlap_at_least_half(a_start: usize, a_end: usize, b_start: usize, b_end: usize) -> bool {
    let overlap = a_end
        .min(b_end)
        .saturating_sub(a_start.max(b_start))
        .saturating_add(1);
    overlap * 2
        >= (a_end.saturating_sub(a_start).saturating_add(1))
            .min(b_end.saturating_sub(b_start).saturating_add(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persistent_index_ranks_symbols_and_refreshes_changes() {
        let root = tempfile::tempdir().unwrap();
        let data = tempfile::tempdir().unwrap();
        std::fs::write(
            root.path().join("auth.rs"),
            "fn verify_token() {\n    validate_signature();\n}\n",
        )
        .unwrap();
        std::fs::write(root.path().join("notes.md"), "authentication overview").unwrap();
        std::fs::write(root.path().join(".env"), "TOKEN=verify_token_signature").unwrap();
        let mut service = RetrievalService::open(data.path(), root.path()).unwrap();
        assert_eq!(service.refresh().unwrap(), 2);
        let results = service
            .search(
                "verify token signature",
                RetrievalIntent::Implementation,
                10,
            )
            .unwrap();
        assert_eq!(results[0].path, "auth.rs");
        assert!(results.iter().all(|result| result.path != ".env"));
        assert_eq!(service.refresh().unwrap(), 0);
        std::fs::remove_file(root.path().join("auth.rs")).unwrap();
        assert_eq!(service.refresh().unwrap(), 1);
        assert!(
            service
                .search(
                    "verify token signature",
                    RetrievalIntent::Implementation,
                    10
                )
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn overlapping_ranges_are_detected() {
        assert!(overlap_at_least_half(1, 10, 6, 12));
        assert!(!overlap_at_least_half(1, 10, 10, 20));
    }
}
