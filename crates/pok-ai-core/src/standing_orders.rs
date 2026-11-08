//! Standing orders: instructions the user wants kept up across runs ("always
//! answer unread messages", "stay in the voice channel"). They are stored on
//! disk, shown to the model at the start of every run, and re-run on a timer by
//! the desktop app, so a long-horizon task does not end with a single reply.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use chrono::{DateTime, Duration, Utc};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::error::{PokError, Result};

/// Shorter intervals would let a typo keep the agent busy all the time.
pub const MIN_INTERVAL_MINUTES: u32 = 5;
pub const DEFAULT_INTERVAL_MINUTES: u32 = 10;
const MAX_ORDERS: usize = 20;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StandingOrder {
    pub id: String,
    pub text: String,
    pub every_minutes: u32,
    pub enabled: bool,
    pub created: DateTime<Utc>,
    #[serde(default)]
    pub last_run: Option<DateTime<Utc>>,
}

impl StandingOrder {
    fn is_due(&self, now: DateTime<Utc>) -> bool {
        let base = self.last_run.unwrap_or(self.created);
        self.enabled && base + Duration::minutes(i64::from(self.every_minutes)) <= now
    }
}

/// The user's standing orders, shared between the agent's tool and the desktop
/// timer through one JSON file (re-read whenever it changes on disk).
pub struct StandingOrders {
    path: PathBuf,
    state: Mutex<(Option<SystemTime>, Vec<StandingOrder>)>,
}

impl StandingOrders {
    pub fn open(path: &Path) -> Self {
        let store = Self {
            path: path.to_owned(),
            state: Mutex::new((None, Vec::new())),
        };
        store.refresh();
        store
    }

    fn refresh(&self) {
        let modified = std::fs::metadata(&self.path)
            .and_then(|metadata| metadata.modified())
            .ok();
        let mut state = self.state.lock();
        if state.0 == modified && modified.is_some() {
            return;
        }
        state.1 = std::fs::read_to_string(&self.path)
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default();
        state.0 = modified;
    }

    fn save(&self, orders: &[StandingOrder]) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let temporary = self.path.with_extension("json.tmp");
        std::fs::write(&temporary, serde_json::to_vec_pretty(orders)?)?;
        std::fs::rename(&temporary, &self.path)?;
        let modified = std::fs::metadata(&self.path)
            .and_then(|metadata| metadata.modified())
            .ok();
        *self.state.lock() = (modified, orders.to_vec());
        Ok(())
    }

    pub fn list(&self) -> Vec<StandingOrder> {
        self.refresh();
        self.state.lock().1.clone()
    }

    /// Add an order, or update the existing one that says the same thing so a
    /// repeated instruction never piles up copies. Returns the order and
    /// whether it is new.
    pub fn add(&self, text: &str, every_minutes: Option<u32>) -> Result<(StandingOrder, bool)> {
        let text = text.trim();
        if text.is_empty() {
            return Err(PokError::Tool("a standing order needs text".into()));
        }
        let every = every_minutes
            .unwrap_or(DEFAULT_INTERVAL_MINUTES)
            .max(MIN_INTERVAL_MINUTES);
        self.refresh();
        let mut orders = self.list();
        if let Some(existing) = orders
            .iter_mut()
            .find(|order| same_order(&order.text, text))
        {
            existing.text = text.to_owned();
            existing.every_minutes = every;
            existing.enabled = true;
            let updated = existing.clone();
            self.save(&orders)?;
            return Ok((updated, false));
        }
        if orders.len() >= MAX_ORDERS {
            return Err(PokError::Tool(format!(
                "there are already {MAX_ORDERS} standing orders; remove one first"
            )));
        }
        let order = StandingOrder {
            id: uuid::Uuid::new_v4().simple().to_string()[..6].to_owned(),
            text: text.to_owned(),
            every_minutes: every,
            enabled: true,
            created: Utc::now(),
            last_run: None,
        };
        orders.push(order.clone());
        self.save(&orders)?;
        Ok((order, true))
    }

    pub fn remove(&self, id: &str) -> Result<bool> {
        let mut orders = self.list();
        let before = orders.len();
        orders.retain(|order| order.id != id);
        let removed = orders.len() != before;
        if removed {
            self.save(&orders)?;
        }
        Ok(removed)
    }

    /// The first order whose time has come.
    pub fn due(&self, now: DateTime<Utc>) -> Option<StandingOrder> {
        self.list().into_iter().find(|order| order.is_due(now))
    }

    pub fn mark_run(&self, id: &str, at: DateTime<Utc>) -> Result<()> {
        let mut orders = self.list();
        if let Some(order) = orders.iter_mut().find(|order| order.id == id) {
            order.last_run = Some(at);
            self.save(&orders)?;
        }
        Ok(())
    }

    /// The block shown to the model at the start of a run.
    pub fn prompt_block(&self) -> Option<String> {
        let orders = self
            .list()
            .into_iter()
            .filter(|order| order.enabled)
            .collect::<Vec<_>>();
        if orders.is_empty() {
            return None;
        }
        let lines = orders
            .iter()
            .map(|order| format!("- [{}] {}", order.id, order.text))
            .collect::<Vec<_>>()
            .join("\n");
        Some(format!(
            "<standing_orders>\nThe user gave these instructions to keep up until they cancel them. Carry them out whenever they apply, including after this request, and do not finish while one still has work waiting (for example unread messages). Remove one with the standing_orders tool only when the user cancels it.\n{lines}\n</standing_orders>"
        ))
    }
}

/// Two orders say the same thing when most of their meaningful words agree.
fn same_order(left: &str, right: &str) -> bool {
    let terms = |text: &str| {
        text.to_lowercase()
            .split(|character: char| !character.is_alphanumeric())
            .filter(|word| word.len() >= 4)
            .map(str::to_owned)
            .collect::<std::collections::BTreeSet<_>>()
    };
    let (left, right) = (terms(left), terms(right));
    if left.is_empty() || right.is_empty() {
        return false;
    }
    let shared = left.intersection(&right).count() as f64;
    shared / left.union(&right).count() as f64 >= 0.6
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orders_persist_dedupe_and_come_due() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("standing-orders.json");
        let store = StandingOrders::open(&path);
        let (first, created) = store
            .add("Answer all unread chat messages", Some(1))
            .unwrap();
        assert!(created);
        assert_eq!(first.every_minutes, MIN_INTERVAL_MINUTES);
        // The same instruction again updates it instead of adding a copy.
        let (again, created) = store
            .add("answer all unread chat messages always", Some(15))
            .unwrap();
        assert!(!created);
        assert_eq!(again.id, first.id);
        assert_eq!(again.every_minutes, 15);
        assert_eq!(store.list().len(), 1);
        // A second handle (the desktop timer) sees the agent's change.
        let other = StandingOrders::open(&path);
        assert_eq!(other.list().len(), 1);
        assert!(other.due(Utc::now()).is_none());
        assert!(other.due(Utc::now() + Duration::minutes(16)).is_some());
        other.mark_run(&first.id, Utc::now()).unwrap();
        assert!(other.due(Utc::now() + Duration::minutes(10)).is_none());
        assert!(store.prompt_block().unwrap().contains(&first.id));
        assert!(store.remove(&first.id).unwrap());
        assert!(other.prompt_block().is_none());
    }
}
