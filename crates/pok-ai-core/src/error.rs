use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum PokError {
    #[error("configuration error: {0}")]
    Config(String),
    #[error("provider error: {0}")]
    Provider(String),
    #[error("transient provider error: {message}")]
    ProviderTransient {
        message: String,
        retry_after_ms: Option<u64>,
        retry_until_cancelled: bool,
    },
    #[error("tool error: {0}")]
    Tool(String),
    #[error("policy denied {tool}: {reason}")]
    PolicyDenied { tool: String, reason: String },
    #[error("path is outside workspace: {0}")]
    OutsideWorkspace(PathBuf),
    #[error("operation cancelled")]
    Cancelled,
    #[error("unsupported on this platform: {0}")]
    Unsupported(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("database error: {0}")]
    Database(#[from] rusqlite::Error),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl PokError {
    pub fn is_transient_provider_error(&self) -> bool {
        matches!(self, Self::ProviderTransient { .. })
    }

    pub fn provider_retry_after_ms(&self) -> Option<u64> {
        match self {
            Self::ProviderTransient { retry_after_ms, .. } => *retry_after_ms,
            _ => None,
        }
    }

    pub fn provider_retry_until_cancelled(&self) -> bool {
        matches!(
            self,
            Self::ProviderTransient {
                retry_until_cancelled: true,
                ..
            }
        )
    }
}

pub type Result<T> = std::result::Result<T, PokError>;
