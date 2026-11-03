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

/// A person or chat on a channel, written `<channel>:<id>`, for example
/// `telegram:900` or `whatsapp:15550100`.
pub fn address(channel: &str, id: &str) -> String {
    format!("{channel}:{id}")
}

/// Split `<channel>:<id>`; an address without a channel is returned whole.
pub fn split_address(address: &str) -> (&str, &str) {
    address.split_once(':').unwrap_or(("", address))
}

/// Files written before WhatsApp existed hold bare Telegram numbers.
fn address_or_legacy_number<'de, D>(deserializer: D) -> std::result::Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error;
    match serde_json::Value::deserialize(deserializer)? {
        serde_json::Value::Number(number) => Ok(address("telegram", &number.to_string())),
        serde_json::Value::String(text) if text.contains(':') => Ok(text),
        serde_json::Value::String(text) => Ok(address("telegram", &text)),
        other => Err(D::Error::custom(format!("not an address: {other}"))),
    }
}

fn optional_address<'de, D>(deserializer: D) -> std::result::Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    match serde_json::Value::deserialize(deserializer)? {
        serde_json::Value::Null => Ok(None),
        serde_json::Value::Number(number) => Ok(Some(address("telegram", &number.to_string()))),
        serde_json::Value::String(text) if text.contains(':') => Ok(Some(text)),
        serde_json::Value::String(text) => Ok(Some(address("telegram", &text))),
        _ => Ok(None),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AllowedUser {
    /// `<channel>:<id>`.
    #[serde(deserialize_with = "address_or_legacy_number")]
    pub id: String,
    pub name: String,
    /// A trusted person answers permission requests and may use every
    /// command, like the owner. A friend (not trusted) can only ask for tasks;
    /// their permission requests go to the owner. People saved before this
    /// existed are trusted, as they always were.
    #[serde(default = "trusted_by_default")]
    pub trusted: bool,
}

const fn trusted_by_default() -> bool {
    true
}

impl AllowedUser {
    pub fn channel(&self) -> &str {
        split_address(&self.id).0
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct PendingPairing {
    pub code: String,
    pub user_id: String,
    pub chat_id: String,
    pub channel: String,
    pub name: String,
    /// A friend asked for access without being told a code: approving makes
    /// them a friend, not a trusted person.
    pub guest: bool,
    /// Seconds until the code expires.
    pub expires_in: u64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct AccessFile {
    #[serde(default)]
    allowed: Vec<AllowedUser>,
    /// Where scheduled results go.
    #[serde(default, deserialize_with = "optional_address")]
    home_chat: Option<String>,
}

struct Pending {
    code: String,
    user_id: String,
    chat_id: String,
    name: String,
    guest: bool,
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

    pub fn is_allowed(&self, user_id: &str) -> bool {
        self.saved
            .lock()
            .allowed
            .iter()
            .any(|user| user.id == user_id)
    }

    /// Whether an approved person is trusted (see [`AllowedUser::trusted`]).
    pub fn is_trusted(&self, user_id: &str) -> bool {
        self.saved
            .lock()
            .allowed
            .iter()
            .any(|user| user.id == user_id && user.trusted)
    }

    pub fn set_trusted(&self, user_id: &str, trusted: bool) -> Result<bool> {
        let mut saved = self.saved.lock();
        let Some(user) = saved.allowed.iter_mut().find(|user| user.id == user_id) else {
            return Ok(false);
        };
        user.trusted = trusted;
        self.save(&saved)?;
        Ok(true)
    }

    pub fn allowed(&self) -> Vec<AllowedUser> {
        self.saved.lock().allowed.clone()
    }

    pub fn home_chat(&self) -> Option<String> {
        self.saved.lock().home_chat.clone()
    }

    pub fn set_home_chat(&self, chat: &str) -> Result<()> {
        let mut saved = self.saved.lock();
        saved.home_chat = Some(chat.to_owned());
        self.save(&saved)
    }

    /// A one-time code for an unknown sender, or `None` when too many are
    /// already open. Asking again returns the same open code.
    pub fn request_pairing(&self, user_id: &str, chat_id: &str, name: &str) -> Option<String> {
        self.request(user_id, chat_id, name, false)
    }

    /// Record that someone asked to use the agent, without telling them
    /// anything. The owner sees the request in Settings; approving it makes
    /// the person a friend (not trusted).
    pub fn request_guest_access(&self, user_id: &str, chat_id: &str, name: &str) -> Option<String> {
        self.request(user_id, chat_id, name, true)
    }

    fn request(&self, user_id: &str, chat_id: &str, name: &str, guest: bool) -> Option<String> {
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
            user_id: user_id.to_owned(),
            chat_id: chat_id.to_owned(),
            name: name.to_owned(),
            guest,
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
                user_id: entry.user_id.clone(),
                chat_id: entry.chat_id.clone(),
                channel: split_address(&entry.user_id).0.to_owned(),
                name: entry.name.clone(),
                guest: entry.guest,
                expires_in: PAIRING_TTL
                    .saturating_sub(entry.created.elapsed())
                    .as_secs(),
            })
            .collect()
    }

    /// An open request by its code, so the person can be told they were
    /// approved and how to talk to the agent.
    pub fn pending_request(&self, code: &str) -> Option<PendingPairing> {
        let wanted = normalize_code(code);
        self.pending()
            .into_iter()
            .find(|entry| normalize_code(&entry.code) == wanted)
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
            trusted: !entry.guest,
        };
        let mut saved = self.saved.lock();
        if !saved.allowed.iter().any(|existing| existing.id == user.id) {
            saved.allowed.push(user.clone());
        }
        // Scheduled results go to the first trusted person, never to a friend.
        if saved.home_chat.is_none() && !entry.guest {
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

    pub fn remove(&self, user_id: &str) -> Result<bool> {
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
        assert!(!access.is_allowed("telegram:11"));
        let code = access
            .request_pairing("telegram:11", "telegram:11", "@sam")
            .unwrap();
        assert_eq!(code.len(), 9);
        assert_eq!(
            access.request_pairing("telegram:11", "telegram:11", "@sam"),
            Some(code.clone())
        );
        assert!(
            !access.is_allowed("telegram:11"),
            "a code alone grants nothing"
        );
        assert_eq!(access.pending().len(), 1);
        assert_eq!(access.pending()[0].channel, "telegram");

        let lowercase_without_dash = code.replace('-', "").to_lowercase();
        let user = access.approve(&lowercase_without_dash).unwrap().unwrap();
        assert_eq!(user.id, "telegram:11");
        assert!(access.is_allowed("telegram:11"));
        assert_eq!(
            access.home_chat().as_deref(),
            Some("telegram:11"),
            "the first person becomes home"
        );
        assert!(access.pending().is_empty());
        assert!(
            access.approve(&code).unwrap().is_none(),
            "a code works once"
        );
    }

    #[test]
    fn the_same_number_on_two_channels_is_two_different_people() {
        let (_dir, access) = store();
        let code = access
            .request_pairing("whatsapp:15550100", "whatsapp:15550100", "Sam")
            .unwrap();
        access.approve(&code).unwrap();
        assert!(access.is_allowed("whatsapp:15550100"));
        assert!(
            !access.is_allowed("telegram:15550100"),
            "approval on one channel never carries to another"
        );
        assert_eq!(access.allowed()[0].channel(), "whatsapp");
    }

    #[test]
    fn the_allowlist_survives_a_restart_and_removal_revokes_it() {
        let (dir, access) = store();
        let code = access
            .request_pairing("telegram:5", "telegram:50", "Sam")
            .unwrap();
        access.approve(&code).unwrap();
        let reopened = AccessControl::open(&dir.path().join("gateway").join("access.json"));
        assert!(reopened.is_allowed("telegram:5"));
        assert_eq!(reopened.home_chat().as_deref(), Some("telegram:50"));
        assert!(reopened.remove("telegram:5").unwrap());
        assert!(!reopened.is_allowed("telegram:5"));
        assert!(
            !AccessControl::open(&dir.path().join("gateway").join("access.json"))
                .is_allowed("telegram:5")
        );
    }

    #[test]
    fn an_access_file_from_before_whatsapp_keeps_its_telegram_people() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("access.json");
        std::fs::write(
            &path,
            r#"{"allowed": [{"id": 900, "name": "Sam"}], "home_chat": 900}"#,
        )
        .unwrap();
        let access = AccessControl::open(&path);
        assert!(access.is_allowed("telegram:900"));
        assert_eq!(access.home_chat().as_deref(), Some("telegram:900"));
        // Saving rewrites it in the new form, and that form reads back.
        access.set_home_chat("telegram:900").unwrap();
        let written = std::fs::read_to_string(&path).unwrap();
        assert!(written.contains("\"telegram:900\""), "{written}");
        assert!(AccessControl::open(&path).is_allowed("telegram:900"));
    }

    #[test]
    fn denied_codes_and_a_flood_of_strangers_are_contained() {
        let (_dir, access) = store();
        let code = access
            .request_pairing("telegram:1", "telegram:1", "a")
            .unwrap();
        assert!(access.deny(&code));
        assert!(access.approve(&code).unwrap().is_none());
        for id in 100..100 + MAX_PENDING {
            let key = format!("telegram:{id}");
            assert!(access.request_pairing(&key, &key, "x").is_some());
        }
        assert!(
            access
                .request_pairing("telegram:999", "telegram:999", "late")
                .is_none()
        );
    }

    #[test]
    fn a_damaged_file_denies_everyone() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("access.json");
        std::fs::write(&path, "{ not json").unwrap();
        assert!(!AccessControl::open(&path).is_allowed("telegram:1"));
    }

    #[test]
    fn an_open_code_knows_which_chat_asked() {
        let (_dir, access) = store();
        let code = access
            .request_pairing(
                "whatsapp:15550100",
                "whatsapp:15550100@s.whatsapp.net",
                "Sam",
            )
            .unwrap();
        let request = access.pending_request(&code.to_lowercase()).unwrap();
        assert_eq!(request.chat_id, "whatsapp:15550100@s.whatsapp.net");
        assert!(!request.guest);
        assert!(access.pending_request("NOPE-NOPE").is_none());
        access.approve(&code).unwrap();
        assert!(
            access.pending_request(&code).is_none(),
            "gone once approved"
        );
    }

    #[test]
    fn a_friend_is_approved_without_trust_and_never_becomes_home() {
        let (_dir, access) = store();
        // The owner is first, so they are home.
        let owner = access
            .request_pairing("telegram:1", "telegram:1", "Owner")
            .unwrap();
        access.approve(&owner).unwrap();
        assert!(access.is_trusted("telegram:1"));

        let code = access
            .request_guest_access(
                "whatsapp:15550100",
                "whatsapp:15550100@s.whatsapp.net",
                "Sam",
            )
            .unwrap();
        assert!(access.pending_request(&code).unwrap().guest);
        let friend = access.approve(&code).unwrap().unwrap();
        assert!(!friend.trusted);
        assert!(access.is_allowed("whatsapp:15550100"));
        assert!(!access.is_trusted("whatsapp:15550100"));
        assert_eq!(access.home_chat().as_deref(), Some("telegram:1"));

        // The owner can promote them, and demote them again.
        assert!(access.set_trusted("whatsapp:15550100", true).unwrap());
        assert!(access.is_trusted("whatsapp:15550100"));
        assert!(access.set_trusted("whatsapp:15550100", false).unwrap());
        assert!(!access.set_trusted("whatsapp:nobody", true).unwrap());
    }

    #[test]
    fn a_friend_alone_does_not_become_the_home_chat() {
        let (_dir, access) = store();
        let code = access
            .request_guest_access(
                "whatsapp:15550100",
                "whatsapp:15550100@s.whatsapp.net",
                "Sam",
            )
            .unwrap();
        access.approve(&code).unwrap();
        assert!(
            access.home_chat().is_none(),
            "scheduled results are never sent to a friend by default"
        );
    }

    #[test]
    fn trust_survives_a_restart_and_older_files_stay_trusted() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("access.json");
        std::fs::write(
            &path,
            r#"{"allowed": [{"id": "telegram:9", "name": "Old"}, {"id": "whatsapp:1", "name": "Friend", "trusted": false}]}"#,
        )
        .unwrap();
        let access = AccessControl::open(&path);
        assert!(
            access.is_trusted("telegram:9"),
            "saved before the flag existed"
        );
        assert!(!access.is_trusted("whatsapp:1"));
        access.set_trusted("whatsapp:1", true).unwrap();
        assert!(AccessControl::open(&path).is_trusted("whatsapp:1"));
    }

    #[test]
    fn addresses_split_into_channel_and_id() {
        assert_eq!(address("whatsapp", "1555"), "whatsapp:1555");
        assert_eq!(split_address("telegram:900"), ("telegram", "900"));
        assert_eq!(
            split_address("whatsapp:1555@s.whatsapp.net"),
            ("whatsapp", "1555@s.whatsapp.net")
        );
        assert_eq!(split_address("bare"), ("", "bare"));
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
