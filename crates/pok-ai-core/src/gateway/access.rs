//! Who may message the agent: a default-deny allowlist, with one-time pairing
//! codes that only the owner can approve on the computer itself.

use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::error::Result;

/// A pairing code stops working after this long.
const PAIRING_TTL: Duration = Duration::from_secs(60 * 60);
/// Strangers can only queue this many open codes; the rest are ignored.
const MAX_PENDING: usize = 8;
/// Unambiguous characters only (no 0/O, 1/I/L).
const CODE_ALPHABET: &[u8] = b"ABCDEFGHJKMNPQRSTUVWXYZ23456789";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AllowedUser {
    pub id: i64,
    pub name: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct PendingPairing {
    pub code: String,
    pub user_id: i64,
    pub chat_id: i64,
    pub name: String,
    /// Seconds until the code expires.
    pub expires_in: u64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct AccessFile {
    #[serde(default)]
    allowed: Vec<AllowedUser>,
    /// Where scheduled results go.
    #[serde(default)]
    home_chat: Option<i64>,
}

struct Pending {
    code: String,
    user_id: i64,
    chat_id: i64,
    name: String,
    created: Instant,
}

pub struct AccessControl {
    path: PathBuf,
    saved: Mutex<AccessFile>,
    pending: Mutex<Vec<Pending>>,
}

impl AccessControl {
    /// Load the allowlist stored at `path`; a missing or damaged file starts
    /// empty, which denies everyone.
    pub fn open(path: &Path) -> Self {
        let saved = std::fs::read_to_string(path)
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default();
        Self {
            path: path.to_owned(),
            saved: Mutex::new(saved),
            pending: Mutex::new(Vec::new()),
        }
    }

    fn save(&self, saved: &AccessFile) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let temporary = self.path.with_extension("json.tmp");
        std::fs::write(&temporary, serde_json::to_vec_pretty(saved)?)?;
        std::fs::rename(&temporary, &self.path)?;
        Ok(())
    }

    pub fn is_allowed(&self, user_id: i64) -> bool {
        self.saved
            .lock()
            .allowed
            .iter()
            .any(|user| user.id == user_id)
    }

    pub fn allowed(&self) -> Vec<AllowedUser> {
        self.saved.lock().allowed.clone()
    }

    pub fn home_chat(&self) -> Option<i64> {
        self.saved.lock().home_chat
    }

    pub fn set_home_chat(&self, chat_id: i64) -> Result<()> {
        let mut saved = self.saved.lock();
        saved.home_chat = Some(chat_id);
        self.save(&saved)
    }

    /// A one-time code for an unknown sender, or `None` when too many are
    /// already open. Asking again returns the same open code.
    pub fn request_pairing(&self, user_id: i64, chat_id: i64, name: &str) -> Option<String> {
        let mut pending = self.pending.lock();
        pending.retain(|entry| entry.created.elapsed() < PAIRING_TTL);
        if let Some(existing) = pending.iter().find(|entry| entry.user_id == user_id) {
            return Some(existing.code.clone());
        }
        if pending.len() >= MAX_PENDING {
            return None;
        }
        let code = new_code();
        pending.push(Pending {
            code: code.clone(),
            user_id,
            chat_id,
            name: name.to_owned(),
            created: Instant::now(),
        });
        Some(code)
    }

    pub fn pending(&self) -> Vec<PendingPairing> {
        let mut pending = self.pending.lock();
        pending.retain(|entry| entry.created.elapsed() < PAIRING_TTL);
        pending
            .iter()
            .map(|entry| PendingPairing {
                code: entry.code.clone(),
                user_id: entry.user_id,
                chat_id: entry.chat_id,
                name: entry.name.clone(),
                expires_in: PAIRING_TTL
                    .saturating_sub(entry.created.elapsed())
                    .as_secs(),
            })
            .collect()
    }

    /// Approve an open code (case and dash insensitive). The first person
    /// approved becomes the home chat for scheduled results.
    pub fn approve(&self, code: &str) -> Result<Option<AllowedUser>> {
        let wanted = normalize_code(code);
        let entry = {
            let mut pending = self.pending.lock();
            pending.retain(|entry| entry.created.elapsed() < PAIRING_TTL);
            let Some(index) = pending
                .iter()
                .position(|entry| normalize_code(&entry.code) == wanted)
            else {
                return Ok(None);
            };
            pending.remove(index)
        };
        let user = AllowedUser {
            id: entry.user_id,
            name: entry.name,
        };
        let mut saved = self.saved.lock();
        if !saved.allowed.iter().any(|existing| existing.id == user.id) {
            saved.allowed.push(user.clone());
        }
        if saved.home_chat.is_none() {
            saved.home_chat = Some(entry.chat_id);
        }
        self.save(&saved)?;
        Ok(Some(user))
    }

    pub fn deny(&self, code: &str) -> bool {
        let wanted = normalize_code(code);
        let mut pending = self.pending.lock();
        let before = pending.len();
        pending.retain(|entry| normalize_code(&entry.code) != wanted);
        pending.len() != before
    }

    pub fn remove(&self, user_id: i64) -> Result<bool> {
        let mut saved = self.saved.lock();
        let before = saved.allowed.len();
        saved.allowed.retain(|user| user.id != user_id);
        let removed = saved.allowed.len() != before;
        if removed {
            self.save(&saved)?;
        }
        Ok(removed)
    }
}

fn normalize_code(code: &str) -> String {
    code.chars()
        .filter(char::is_ascii_alphanumeric)
        .map(|c| c.to_ascii_uppercase())
        .collect()
}

/// `XXXX-XXXX` from the OS-seeded random bytes of two v4 UUIDs.
fn new_code() -> String {
    let bytes = [
        uuid::Uuid::new_v4().into_bytes(),
        uuid::Uuid::new_v4().into_bytes(),
    ]
    .concat();
    let letters: String = bytes
        .iter()
        .take(8)
        .map(|byte| CODE_ALPHABET[usize::from(*byte) % CODE_ALPHABET.len()] as char)
        .collect();
    format!("{}-{}", &letters[..4], &letters[4..])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> (tempfile::TempDir, AccessControl) {
        let dir = tempfile::tempdir().unwrap();
        let access = AccessControl::open(&dir.path().join("gateway").join("access.json"));
        (dir, access)
    }

    #[test]
    fn strangers_are_denied_until_the_owner_approves_their_code() {
        let (_dir, access) = store();
        assert!(!access.is_allowed(11));
        let code = access.request_pairing(11, 11, "@sam").unwrap();
        assert_eq!(code.len(), 9);
        assert_eq!(access.request_pairing(11, 11, "@sam"), Some(code.clone()));
        assert!(!access.is_allowed(11), "a code alone grants nothing");
        assert_eq!(access.pending().len(), 1);

        let lowercase_without_dash = code.replace('-', "").to_lowercase();
        let user = access.approve(&lowercase_without_dash).unwrap().unwrap();
        assert_eq!(user.id, 11);
        assert!(access.is_allowed(11));
        assert_eq!(
            access.home_chat(),
            Some(11),
            "the first person becomes home"
        );
        assert!(access.pending().is_empty());
        assert!(
            access.approve(&code).unwrap().is_none(),
            "a code works once"
        );
    }

    #[test]
    fn the_allowlist_survives_a_restart_and_removal_revokes_it() {
        let (dir, access) = store();
        let code = access.request_pairing(5, 50, "Sam").unwrap();
        access.approve(&code).unwrap();
        let reopened = AccessControl::open(&dir.path().join("gateway").join("access.json"));
        assert!(reopened.is_allowed(5));
        assert_eq!(reopened.home_chat(), Some(50));
        assert!(reopened.remove(5).unwrap());
        assert!(!reopened.is_allowed(5));
        assert!(
            !AccessControl::open(&dir.path().join("gateway").join("access.json")).is_allowed(5)
        );
    }

    #[test]
    fn denied_codes_and_a_flood_of_strangers_are_contained() {
        let (_dir, access) = store();
        let code = access.request_pairing(1, 1, "a").unwrap();
        assert!(access.deny(&code));
        assert!(access.approve(&code).unwrap().is_none());
        for id in 100..100 + MAX_PENDING as i64 {
            assert!(access.request_pairing(id, id, "x").is_some());
        }
        assert!(access.request_pairing(999, 999, "late").is_none());
    }

    #[test]
    fn a_damaged_file_denies_everyone() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("access.json");
        std::fs::write(&path, "{ not json").unwrap();
        assert!(!AccessControl::open(&path).is_allowed(1));
    }

    #[test]
    fn codes_use_the_unambiguous_alphabet() {
        for _ in 0..50 {
            let code = new_code();
            assert!(
                code.chars()
                    .all(|c| c == '-' || CODE_ALPHABET.contains(&(c as u8)))
            );
        }
    }
}
