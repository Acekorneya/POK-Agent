//! Values the user agreed to let the agent remember, kept in a secure store
//! outside POK-Agent's files: Windows Credential Manager in the desktop app,
//! which installs it at startup. Without an installed store nothing private
//! is remembered.

use std::sync::{Arc, OnceLock};

use parking_lot::RwLock;

use crate::Result;

pub trait SecretStore: Send + Sync {
    /// Where values go, for the question to the user ("Windows Credential Manager").
    fn name(&self) -> &str;
    fn load(&self, key: &str) -> Option<String>;
    fn save(&self, key: &str, value: &str) -> Result<()>;
}

static STORE: OnceLock<RwLock<Option<Arc<dyn SecretStore>>>> = OnceLock::new();

/// Install the store values are saved to (replacing any earlier one).
pub fn install(store: Arc<dyn SecretStore>) {
    *STORE.get_or_init(Default::default).write() = Some(store);
}

/// The installed store, when there is one.
pub fn store() -> Option<Arc<dyn SecretStore>> {
    STORE.get_or_init(Default::default).read().clone()
}

/// A remembered value.
pub fn load(key: &str) -> Option<String> {
    store()?.load(key).filter(|value| !value.is_empty())
}
