//! Memory hygiene: what an inferred fact must look like to be kept, how
//! near-duplicates fold into one, and the budgets that keep the store small
//! without the user pruning it by hand.

use super::*;

/// Two inferred facts this alike are one fact; the repeat reinforces.
pub(super) const FACT_MERGE_SIMILARITY: f64 = 0.62;
/// An explicit statement this alike to a stored fact replaces its wording.
pub(super) const EXPLICIT_FACT_MERGE_SIMILARITY: f64 = 0.78;
/// Two swarm playbooks for the same kind of goal are one playbook.
const PLAYBOOK_MERGE_SIMILARITY: f64 = 0.55;
/// Facts the harness inferred and nobody has reviewed yet.
const MAX_FACT_DRAFTS: usize = 15;
/// An unreinforced draft nobody reviewed expires after this many days.
const DRAFT_TTL_DAYS: i64 = 14;
/// Inferred facts that are in use (approved, or promoted by recurrence).
const MAX_INFERRED_FACTS: usize = 60;
/// A draft seen again in this many separate runs is promoted to a fact.
const PROMOTE_AT_REINFORCEMENTS: u64 = 2;
const MAX_PLAYBOOKS: usize = 8;
const MAX_LEARNED_SKILLS: usize = 60;
const UNUSED_SKILL_DAYS: i64 = 45;
const MAX_INFERRED_FACT_CHARS: usize = 320;

/// The source label of swarm playbooks (see `swarm.rs`).
pub(crate) const PLAYBOOK_SOURCE: &str = "swarm_playbook";

/// Words that mark a note about how some website or web fetch behaved.
const WEB_BEHAVIOUR_MARKERS: &[&str] = &[
    "captcha",
    "redirect",
    "managed browser",
    "web fetch",
    "web_fetch",
    "website",
    "web page",
    "webpage",
    "search result",
    "web interface",
    "sign-in page",
    "login page",
    "http 4",
    "http 5",
    "rate limit",
];

/// Why an inferred fact is not worth keeping, or `None` when it is.
///
/// Durable memory holds what stays true about the user, their machine, and
/// how they like work done. What a task found out (prices, rates, store or
/// website behaviour, links) belongs in that task's answer and the session
/// archive, not in the list the user has to review.
pub(crate) fn inferred_fact_rejection(source: &str, text: &str) -> Option<&'static str> {
    let trimmed = text.trim();
    let length = trimmed.chars().count();
    if length < 12 {
        return Some("too short to mean anything on its own");
    }
    if length > MAX_INFERRED_FACT_CHARS {
        return Some("too long for a durable fact; it should be one short sentence");
    }
    let lower = trimmed.to_ascii_lowercase();
    if lower.starts_with("swarm playbook") {
        return Some("playbooks are saved by the swarm, not as facts");
    }
    if lower.contains("http://") || lower.contains("https://") || lower.contains("www.") {
        return Some("links belong in the task answer, not in memory");
    }
    if mentions_price_or_rate(trimmed) {
        return Some("prices and rates change; they belong in the task answer");
    }
    if source == "environment"
        && WEB_BEHAVIOUR_MARKERS
            .iter()
            .any(|marker| lower.contains(marker))
    {
        return Some("how a website behaved on one run is not a durable fact");
    }
    None
}

fn mentions_price_or_rate(text: &str) -> bool {
    let chars: Vec<char> = text.chars().collect();
    chars.windows(2).any(|pair| {
        (pair[0] == '$' && pair[1].is_ascii_digit()) || (pair[0].is_ascii_digit() && pair[1] == '%')
    })
}

fn delete_fact_rows(connection: &Connection, ids: &[String]) -> Result<usize> {
    let mut removed = 0;
    for id in ids {
        connection.execute("DELETE FROM memories_fts WHERE id = ?1", params![id])?;
        removed += connection.execute("DELETE FROM memories WHERE id = ?1", params![id])?;
    }
    Ok(removed)
}

fn ids_of(
    connection: &Connection,
    sql: &str,
    params: impl rusqlite::Params,
) -> Result<Vec<String>> {
    let mut statement = connection.prepare(sql)?;
    let rows = statement
        .query_map(params, |row| row.get::<_, String>(0))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(rows)
}

/// Keep the fact store small on its own: promote what keeps recurring,
/// expire drafts nobody confirmed, and cap what the harness inferred.
/// Facts the user wrote or approved by hand are never touched.
pub(super) fn enforce_fact_budgets(connection: &mut Connection) -> Result<()> {
    const INFERRED: &str = "kind = 'fact' AND source NOT LIKE 'skill:%'
        AND provenance IN ('inferred_curation', 'legacy')";
    connection.execute(
        &format!(
            "UPDATE memories SET approved = 1 WHERE approved = 0 AND {INFERRED}
             AND provenance = 'inferred_curation' AND review_status = 'independent'
             AND reinforcement_count >= ?1 AND source IN ('user', 'project', 'lesson')"
        ),
        params![PROMOTE_AT_REINFORCEMENTS],
    )?;
    let cutoff = (Utc::now() - chrono::Duration::days(DRAFT_TTL_DAYS)).to_rfc3339();
    let expired = ids_of(
        connection,
        &format!(
            "SELECT id FROM memories WHERE approved = 0 AND {INFERRED}
             AND reinforcement_count <= 1 AND COALESCE(updated_at, created_at) < ?1"
        ),
        params![cutoff],
    )?;
    delete_fact_rows(connection, &expired)?;
    let overflow = |sql_where: String, keep: usize| -> Result<Vec<String>> {
        ids_of(
            connection,
            &format!(
                "SELECT id FROM memories WHERE {sql_where}
                 ORDER BY reinforcement_count DESC, COALESCE(updated_at, created_at) DESC
                 LIMIT -1 OFFSET {keep}"
            ),
            [],
        )
    };
    let drafts = overflow(
        format!("approved = 0 AND {INFERRED} AND source != '{PLAYBOOK_SOURCE}'"),
        MAX_FACT_DRAFTS,
    )?;
    delete_fact_rows(connection, &drafts)?;
    let kept = overflow(
        format!(
            "approved = 1 AND {INFERRED} AND provenance = 'inferred_curation'
             AND source != '{PLAYBOOK_SOURCE}'"
        ),
        MAX_INFERRED_FACTS,
    )?;
    delete_fact_rows(connection, &kept)?;
    let playbooks = overflow(format!("source = '{PLAYBOOK_SOURCE}'"), MAX_PLAYBOOKS)?;
    delete_fact_rows(connection, &playbooks)?;
    Ok(())
}

type FactRow = (String, String, String, bool, i64, String);

impl MemoryStore {
    /// Fold what earlier versions piled up into the shape new saves keep:
    /// drop drafts that were never durable facts, merge the same fact worded
    /// several ways, collapse repeated swarm playbooks, then apply the
    /// budgets. Returns how many rows were removed.
    pub fn consolidate_facts(&self) -> Result<usize> {
        let mut connection = self.connection.lock();
        let rows = {
            let mut statement = connection.prepare(
                "SELECT id, source, text, approved, reinforcement_count, provenance
                 FROM memories
                 WHERE kind = 'fact' AND source NOT LIKE 'skill:%'
                   AND provenance IN ('inferred_curation', 'legacy')
                 ORDER BY approved DESC, reinforcement_count DESC, created_at ASC
                 LIMIT 800",
            )?;
            statement
                .query_map([], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, bool>(3)?,
                        row.get::<_, i64>(4)?,
                        row.get::<_, String>(5)?,
                    ))
                })?
                .collect::<std::result::Result<Vec<FactRow>, _>>()?
        };
        let mut doomed: Vec<String> = Vec::new();
        // (keeper id, keeper text, extra reinforcement gathered)
        let mut keepers: Vec<(String, String, i64)> = Vec::new();
        let mut playbooks: Vec<(String, String, i64)> = Vec::new();
        for (id, source, text, approved, count, provenance) in rows {
            let is_playbook = source == PLAYBOOK_SOURCE;
            if !approved && !is_playbook && inferred_fact_rejection(&source, &text).is_some() {
                doomed.push(id);
                continue;
            }
            // Only what the harness inferred (or an unreviewed legacy draft)
            // may be folded away; a fact the user approved or wrote stays.
            let loseable = !approved || provenance == "inferred_curation";
            let pool = if is_playbook {
                &mut playbooks
            } else {
                &mut keepers
            };
            let threshold = if is_playbook {
                PLAYBOOK_MERGE_SIMILARITY
            } else {
                FACT_MERGE_SIMILARITY
            };
            let twin = pool.iter().position(|(_, kept, _)| {
                (is_playbook || protected_terms_match(kept, &text))
                    && memory_similarity(kept, &text) >= threshold
            });
            match twin {
                Some(index) if loseable => {
                    pool[index].2 += count.max(1);
                    doomed.push(id);
                }
                _ => pool.push((id, text, 0)),
            }
        }
        let transaction = connection.transaction()?;
        for (id, _, extra) in keepers.iter().chain(playbooks.iter()) {
            if *extra > 0 {
                transaction.execute(
                    "UPDATE memories SET reinforcement_count = MIN(reinforcement_count + ?2, 99),
                     version = version + 1 WHERE id = ?1",
                    params![id, extra],
                )?;
            }
        }
        let mut removed = delete_fact_rows(&transaction, &doomed)?;
        transaction.commit()?;
        let before = connection.query_row("SELECT COUNT(*) FROM memories", [], |row| {
            row.get::<_, i64>(0)
        })?;
        enforce_fact_budgets(&mut connection)?;
        let after = connection.query_row("SELECT COUNT(*) FROM memories", [], |row| {
            row.get::<_, i64>(0)
        })?;
        removed += usize::try_from(before - after).unwrap_or(0);
        Ok(removed)
    }

    /// Save a swarm playbook. A run for the same kind of goal updates the
    /// stored playbook (and reinforces it) instead of adding another.
    pub(crate) fn save_playbook(&self, text: &str, approved: bool) -> Result<MemoryRecord> {
        let mut connection = self.connection.lock();
        let fingerprint = memory_fingerprint(text);
        let now = Utc::now().to_rfc3339();
        let existing = {
            let mut statement = connection.prepare(
                "SELECT id, text FROM memories WHERE source = ?1 AND kind = 'fact'
                 ORDER BY COALESCE(updated_at, created_at) DESC LIMIT 60",
            )?;
            let rows = statement
                .query_map(params![PLAYBOOK_SOURCE], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            rows.into_iter()
                .map(|(id, kept)| (id, memory_similarity(text, &kept)))
                .filter(|(_, similarity)| *similarity >= PLAYBOOK_MERGE_SIMILARITY)
                .max_by(|left, right| left.1.total_cmp(&right.1))
        };
        let id = match existing {
            Some((id, _)) => {
                let transaction = connection.transaction()?;
                transaction.execute(
                    "UPDATE memories SET text = ?2, fingerprint = ?3, updated_at = ?4,
                     approved = MAX(approved, ?5), reinforcement_count = reinforcement_count + 1,
                     version = version + 1 WHERE id = ?1",
                    params![id, text.trim(), fingerprint, now, approved],
                )?;
                transaction.execute("DELETE FROM memories_fts WHERE id = ?1", params![id])?;
                transaction.execute(
                    "INSERT INTO memories_fts(id, text) VALUES (?1, ?2)",
                    params![id, text.trim()],
                )?;
                transaction.commit()?;
                id
            }
            None => {
                let id = Uuid::new_v4().to_string();
                let transaction = connection.transaction()?;
                transaction.execute(
                    "INSERT INTO memories(id, source, text, approved, enabled, created_at,
                     fingerprint, updated_at, kind, reinforcement_count, review_status,
                     related_memory_id, version, provenance)
                     VALUES (?1, ?2, ?3, ?4, 1, ?5, ?6, ?5, 'fact', 1, 'independent', NULL, 1,
                     'inferred_curation')",
                    params![id, PLAYBOOK_SOURCE, text.trim(), approved, now, fingerprint],
                )?;
                transaction.execute(
                    "INSERT INTO memories_fts(id, text) VALUES (?1, ?2)",
                    params![id, text.trim()],
                )?;
                transaction.commit()?;
                id
            }
        };
        enforce_fact_budgets(&mut connection)?;
        load_memory_record(&connection, &id)
    }

    /// Retire learned skills nobody used again: a skill the harness learned
    /// once, never loaded and never repeated, goes after a while, and the
    /// learned set stays under a cap (weakest first). Skills the user asked
    /// for are never retired this way. Returns how many were removed.
    pub fn prune_learned_skills(&self) -> Result<usize> {
        let mut connection = self.connection.lock();
        let rows = {
            let mut statement = connection.prepare(
                "SELECT id, success_count, retrieval_count, updated_at FROM procedures
                 WHERE evidence != 'explicit_user_request'
                 ORDER BY success_count ASC, retrieval_count ASC, updated_at ASC",
            )?;
            statement
                .query_map([], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, i64>(2)?,
                        row.get::<_, String>(3)?,
                    ))
                })?
                .collect::<std::result::Result<Vec<_>, _>>()?
        };
        let cutoff = (Utc::now() - chrono::Duration::days(UNUSED_SKILL_DAYS)).to_rfc3339();
        let mut doomed: Vec<String> = rows
            .iter()
            .filter(|(_, successes, retrievals, updated)| {
                *successes <= 1 && *retrievals == 0 && *updated < cutoff
            })
            .map(|(id, ..)| id.clone())
            .collect();
        let remaining: Vec<&String> = rows
            .iter()
            .map(|(id, ..)| id)
            .filter(|id| !doomed.contains(id))
            .collect();
        if remaining.len() > MAX_LEARNED_SKILLS {
            let excess = remaining.len() - MAX_LEARNED_SKILLS;
            doomed.extend(remaining.into_iter().take(excess).cloned());
        }
        if doomed.is_empty() {
            return Ok(0);
        }
        let transaction = connection.transaction()?;
        for id in &doomed {
            transaction.execute("DELETE FROM procedures_fts WHERE id = ?1", params![id])?;
            transaction.execute("DELETE FROM procedures WHERE id = ?1", params![id])?;
        }
        transaction.commit()?;
        drop(connection);
        for id in &doomed {
            let dir = self.root.join("skills").join(id);
            if dir.is_dir() {
                let _ = std::fs::remove_dir_all(dir);
            }
        }
        Ok(doomed.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> (tempfile::TempDir, Arc<MemoryStore>) {
        let dir = tempfile::tempdir().unwrap();
        let memory = MemoryStore::open(dir.path()).unwrap();
        (dir, memory)
    }

    fn inferred(memory: &MemoryStore, source: &str, text: &str) -> Result<MemorySaveOutcome> {
        memory.save_with_provenance(source, text, false, MemoryWriteProvenance::InferredCuration)
    }

    #[test]
    fn task_findings_are_not_saved_as_inferred_facts() {
        let (_dir, memory) = store();
        for (source, text) in [
            (
                "environment",
                "The sales tax rate for the buyer's city is 8.75%, which is the rate to use.",
            ),
            (
                "environment",
                "Shop pages return an HTTP 403 to plain web fetch, so read them through the managed browser.",
            ),
            (
                "environment",
                "The listing on https://example.com/item/1 costs $184 plus shipping.",
            ),
            ("user", &"The user wants thorough answers. ".repeat(20)),
            ("user", "ok"),
        ] {
            assert!(inferred(&memory, source, text).is_err(), "{text}");
        }
        assert!(memory.list_memories(10).unwrap().is_empty());
    }

    #[test]
    fn real_preferences_are_still_saved() {
        let (_dir, memory) = store();
        let outcome = inferred(
            &memory,
            "user",
            "The user prefers direct links to individual items so they can inspect them.",
        )
        .unwrap();
        assert_eq!(outcome.disposition, "created");
        assert!(!outcome.record.approved);
    }

    #[test]
    fn the_same_fact_worded_differently_reinforces_instead_of_adding() {
        let (_dir, memory) = store();
        inferred(
            &memory,
            "user",
            "The user prefers direct links to individual listings so they can inspect the photos.",
        )
        .unwrap();
        let again = inferred(
            &memory,
            "user",
            "When comparing items the user prefers direct links to individual listings to inspect the photos.",
        )
        .unwrap();
        assert_eq!(again.disposition, "reinforced_local");
        assert_eq!(memory.list_memories(10).unwrap().len(), 1);
        assert_eq!(again.record.reinforcement_count, 2);
    }

    #[test]
    fn a_draft_seen_in_a_second_run_becomes_a_fact_without_review() {
        let (_dir, memory) = store();
        let text = "The user keeps project notes in the docs folder of each repository.";
        let first = inferred(&memory, "user", text).unwrap();
        assert!(!first.record.approved);
        let second = inferred(&memory, "user", text).unwrap();
        assert_eq!(second.record.reinforcement_count, 2);
        let promoted = memory.list_memories(10).unwrap();
        assert!(promoted[0].approved, "recurrence confirms the fact");
        assert!(
            !memory
                .search("project notes docs folder", 5)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn an_explicit_statement_replaces_the_older_wording() {
        let (_dir, memory) = store();
        memory
            .save("user", "The user lives in the Eastern time zone area", true)
            .unwrap();
        let outcome = memory
            .save("user", "The user lives in the Pacific time zone area", true)
            .map(|record| record.text)
            .unwrap();
        assert!(outcome.contains("Pacific"));
        let all = memory.list_memories(10).unwrap();
        assert_eq!(all.len(), 1, "a corrected fact replaces the old one");
        assert!(all[0].text.contains("Pacific"));
    }

    #[test]
    fn distinct_explicit_facts_are_not_merged() {
        let (_dir, memory) = store();
        memory
            .save("user", "My work email is on the company domain", true)
            .unwrap();
        memory
            .save("user", "My home email is on a personal domain", true)
            .unwrap();
        assert_eq!(memory.list_memories(10).unwrap().len(), 2);
    }

    #[test]
    fn unreviewed_drafts_stay_under_the_cap() {
        let (_dir, memory) = store();
        let topics = [
            "tea", "cats", "rowing", "chess", "jazz", "maps", "tools", "paint", "wine", "bikes",
            "ferns", "coins", "kites", "bread", "opera", "yoga", "lego", "inks", "birds", "salt",
            "moss", "dice", "silk", "peas", "rope", "clay",
        ];
        for topic in topics {
            let _ = inferred(
                &memory,
                "user",
                &format!("The user enjoys reading long articles about {topic} on weekends."),
            );
        }
        assert!(memory.list_drafts(100).unwrap().len() <= MAX_FACT_DRAFTS);
    }

    #[test]
    fn user_written_facts_survive_every_budget() {
        let (_dir, memory) = store();
        for index in 0..80 {
            memory
                .save(
                    "user",
                    &format!("Fact number {index} about zebra{index}"),
                    true,
                )
                .unwrap();
        }
        let _ = memory.consolidate_facts().unwrap();
        assert_eq!(memory.list_memories(200).unwrap().len(), 80);
    }

    #[test]
    fn consolidation_clears_old_research_drafts_and_folds_repeats() {
        let (_dir, memory) = store();
        {
            let connection = memory.connection.lock();
            let rows = [
                (
                    "environment",
                    "Rates for the buyer's city sit at 8.75% right now.",
                ),
                (
                    "environment",
                    "Outlet stock is refurbished and sold with a 1-year warranty, see https://example.com/outlet.",
                ),
                (
                    "user",
                    "The user prefers dark themes in every editor they use.",
                ),
                (
                    "user",
                    "In every editor they use the user prefers dark themes.",
                ),
                (
                    "user",
                    "The user prefers dark themes in all editors they use.",
                ),
            ];
            for (index, (source, text)) in rows.iter().enumerate() {
                connection
                    .execute(
                        "INSERT INTO memories(id, source, text, approved, enabled, created_at,
                         fingerprint, updated_at, kind, reinforcement_count, review_status,
                         version, provenance)
                         VALUES (?1, ?2, ?3, 0, 1, ?4, ?5, ?4, 'fact', 1, 'independent', 1, 'legacy')",
                        params![
                            Uuid::new_v4().to_string(),
                            source,
                            text,
                            Utc::now().to_rfc3339(),
                            format!("fp{index}")
                        ],
                    )
                    .unwrap();
                connection
                    .execute(
                        "INSERT INTO memories_fts(id, text)
                         SELECT id, text FROM memories WHERE fingerprint = ?1",
                        params![format!("fp{index}")],
                    )
                    .unwrap();
            }
        }
        let removed = memory.consolidate_facts().unwrap();
        assert!(removed >= 4, "removed {removed}");
        let left = memory.list_memories(10).unwrap();
        assert_eq!(left.len(), 1);
        assert!(left[0].text.contains("dark themes"));
        assert!(left[0].reinforcement_count >= 3, "repeats add up");
    }

    #[test]
    fn swarm_playbooks_update_in_place_and_stay_capped() {
        let (_dir, memory) = store();
        let goal = |n: &str| {
            format!(
                "Swarm playbook: for a goal like \"find {n} for a client\", 3 agents worked in parallel and all completed: research agent \"{n} search\"; review agent \"check {n}\"."
            )
        };
        memory.save_playbook(&goal("laptops"), true).unwrap();
        let again = memory.save_playbook(&goal("laptops"), true).unwrap();
        assert_eq!(again.reinforcement_count, 2);
        for topic in [
            "monitors", "chairs", "desks", "cables", "phones", "routers", "tablets", "printers",
            "speakers", "cameras", "drives", "lamps",
        ] {
            memory
                .save_playbook(
                    &format!(
                        "Swarm playbook: for a goal like \"audit {topic} coverage\", {} agents worked.",
                        topic.len()
                    ),
                    true,
                )
                .unwrap();
        }
        let playbooks = memory
            .list_memories(100)
            .unwrap()
            .into_iter()
            .filter(|record| record.source == PLAYBOOK_SOURCE)
            .count();
        assert!(playbooks <= MAX_PLAYBOOKS, "{playbooks}");
    }

    #[test]
    fn learned_skills_nobody_used_are_retired_but_requested_ones_stay() {
        let (_dir, memory) = store();
        let old = (Utc::now() - chrono::Duration::days(UNUSED_SKILL_DAYS + 5)).to_rfc3339();
        {
            let connection = memory.connection.lock();
            for (index, evidence) in ["verified_workflow", "explicit_user_request"]
                .iter()
                .enumerate()
            {
                connection
                    .execute(
                        "INSERT INTO procedures(id, kind, task_signature, title, summary,
                         applications, steps, evidence, enabled, success_count, retrieval_count,
                         created_at, updated_at, fingerprint)
                         VALUES (?1, 'workflow', ?2, 'Old skill', 'summary', '[]', '[]', ?3, 1, 1, 0,
                         ?4, ?4, ?5)",
                        params![
                            Uuid::new_v4().to_string(),
                            format!("sig{index}"),
                            evidence,
                            old,
                            format!("fp{index}")
                        ],
                    )
                    .unwrap();
            }
        }
        assert_eq!(memory.prune_learned_skills().unwrap(), 1);
        let left = memory.list_procedures(None, 10).unwrap();
        assert_eq!(left.len(), 1);
        assert_eq!(left[0].evidence, "explicit_user_request");
    }
}
