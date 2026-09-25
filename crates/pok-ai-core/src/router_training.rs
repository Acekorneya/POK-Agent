//! Opt-in log of decision-router questions for training a System 1 model.
//!
//! Every question the router is asked (and, as "shadow" questions, the ones
//! local grounding answered instead) is appended to
//! `<data_dir>/router-training/<session>.jsonl`
//! in Laya's typed-decision shape, `{state, questions}`, exactly as the
//! backend receives it after outbound sanitization. Correct answers arrive
//! later as separate `label` records that point at a question id, from
//! sources that proved them: grounding, a click that made progress, or local
//! evidence that contradicted a pick. `scripts/build_router_dataset.py`
//! joins questions and labels from many sessions into a training set.
//!
//! Nothing is written unless the user enables it, files never leave the
//! machine, and windows matching the exclude list (by title or process) are
//! never recorded.

use std::{
    collections::HashMap,
    io::Write,
    path::{Path, PathBuf},
};

use parking_lot::Mutex;
use serde_json::{Value, json};
use uuid::Uuid;

pub struct TrainingRecorder {
    path: PathBuf,
    session_id: Uuid,
    excluded: Vec<String>,
    inner: Mutex<RecorderState>,
}

#[derive(Default)]
struct RecorderState {
    /// The latest record id per question key ("satisfied", "target_click", ...).
    last: HashMap<String, String>,
    /// The current window matches the exclude list.
    excluded_window: bool,
}

impl TrainingRecorder {
    pub fn new(directory: &Path, session_id: Uuid, excluded: &[String]) -> Self {
        Self {
            path: directory.join(format!("{session_id}.jsonl")),
            session_id,
            excluded: excluded
                .iter()
                .map(|entry| entry.trim().to_lowercase())
                .filter(|entry| !entry.is_empty())
                .collect(),
            inner: Mutex::new(RecorderState::default()),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Note the window the agent is working in; questions about an excluded
    /// application are not recorded.
    pub fn observe_window(&self, title: &str, process: &str) {
        let haystack = format!("{title} {process}").to_lowercase();
        self.inner.lock().excluded_window = self
            .excluded
            .iter()
            .any(|entry| haystack.contains(entry.as_str()));
    }

    /// Record one request body (`{model, state, questions}`) and, when the
    /// backend answered, its answers. `source` is `router` for a real request
    /// or `grounding` for a question local evidence answered. Returns the
    /// record id.
    pub fn question(&self, source: &str, body: &Value, answers: Option<&Value>) -> Option<String> {
        let questions = body.get("questions")?.as_object()?;
        if contains_private_text(body) {
            return None;
        }
        let mut inner = self.inner.lock();
        if inner.excluded_window {
            return None;
        }
        let id = Uuid::new_v4().to_string();
        for key in questions.keys() {
            inner.last.insert(key.clone(), id.clone());
        }
        drop(inner);
        self.append(&json!({
            "kind": "question",
            "id": id,
            "session_id": self.session_id,
            "recorded_at": chrono::Utc::now().to_rfc3339(),
            "source": source,
            "model": body.get("model"),
            "state": body.get("state"),
            "questions": body.get("questions"),
            "answers": answers,
        }));
        Some(id)
    }

    /// Attach the proven answer to the latest recorded question with this key.
    pub fn label(&self, question: &str, gold: &str, source: &str) {
        let Some(id) = self.inner.lock().last.get(question).cloned() else {
            return;
        };
        self.label_record(&id, question, gold, source);
    }

    /// Attach the proven answer to a specific recorded question.
    pub fn label_record(&self, id: &str, question: &str, gold: &str, source: &str) {
        self.append(&json!({
            "kind": "label",
            "id": id,
            "question": question,
            "label": gold,
            "source": source,
            "recorded_at": chrono::Utc::now().to_rfc3339(),
        }));
    }

    fn append(&self, record: &Value) {
        let Ok(line) = serde_json::to_string(record) else {
            return;
        };
        if let Some(parent) = self.path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
        {
            let _ = writeln!(file, "{line}");
        }
    }
}

/// Email addresses and long digit runs (phone, account, tracking, or order
/// numbers) never go into a training file: the whole record is skipped.
fn contains_private_text(value: &Value) -> bool {
    match value {
        Value::String(text) => {
            let mut digits = 0;
            for character in text.chars() {
                digits = if character.is_ascii_digit() {
                    digits + 1
                } else {
                    0
                };
                if digits >= 6 {
                    return true;
                }
            }
            text.split_whitespace().any(|word| {
                word.split_once('@')
                    .is_some_and(|(user, domain)| !user.is_empty() && domain.contains('.'))
            })
        }
        Value::Array(values) => values.iter().any(contains_private_text),
        Value::Object(object) => object.values().any(contains_private_text),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn records(recorder: &TrainingRecorder) -> Vec<Value> {
        std::fs::read_to_string(recorder.path())
            .unwrap_or_default()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    #[test]
    fn questions_and_their_labels_are_appended_and_linked() {
        let dir = tempfile::tempdir().unwrap();
        let recorder = TrainingRecorder::new(dir.path(), Uuid::new_v4(), &[]);
        let body = json!({
            "model": "laya",
            "state": {"goal": "open display settings"},
            "questions": {"satisfied": {"type": "choice", "instructions": "done?", "criteria": {"yes": "y", "no": "n"}}},
        });
        let id = recorder.question("grounding", &body, None).unwrap();
        recorder.label("satisfied", "yes", "grounding");
        let records = records(&recorder);
        assert_eq!(records.len(), 2);
        assert_eq!(records[0]["kind"], "question");
        assert_eq!(records[0]["questions"]["satisfied"]["type"], "choice");
        assert_eq!(records[1]["id"], id.as_str());
        assert_eq!(records[1]["label"], "yes");
    }

    #[test]
    fn records_with_emails_or_long_numbers_are_skipped() {
        let dir = tempfile::tempdir().unwrap();
        let recorder = TrainingRecorder::new(dir.path(), Uuid::new_v4(), &[]);
        for text in [
            "mail alice@example.com",
            "Tracking 9400111202555",
            "call 5550100123",
        ] {
            let body =
                json!({"model": "laya", "state": {"labels": [text]}, "questions": {"q": {}}});
            assert!(recorder.question("router", &body, None).is_none(), "{text}");
        }
        let body = json!({"model": "laya", "state": {"labels": ["Refresh rate 165 Hz", "@home"]}, "questions": {"q": {}}});
        assert!(recorder.question("router", &body, None).is_some());
    }

    #[test]
    fn excluded_applications_are_never_recorded() {
        let dir = tempfile::tempdir().unwrap();
        let recorder = TrainingRecorder::new(
            dir.path(),
            Uuid::new_v4(),
            &["Discord".into(), "example_pos".into()],
        );
        let body = json!({"model": "laya", "state": {}, "questions": {"q": {}}});
        recorder.observe_window("#general | Server - Discord", "Discord.exe");
        assert!(recorder.question("router", &body, None).is_none());
        recorder.label("q", "a", "grounding");
        assert!(records(&recorder).is_empty());
        recorder.observe_window("Settings", "SystemSettings.exe");
        assert!(recorder.question("router", &body, None).is_some());
    }
}
