//! The JSON-lines protocol spoken with the agent over stdin/stdout.
//!
//! Events go out one object per line on stdout (`emit`). Commands arrive one
//! object per line on stdin. A command that carries a `request` id is
//! answered by a `result` event with the same id.

use std::{io::Write, path::PathBuf};

use serde::Deserialize;
use serde_json::{Value, json};

/// What the agent can ask for.
#[derive(Debug, Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
pub enum Command {
    SendText {
        request: String,
        chat: String,
        text: String,
    },
    /// Show "typing..." in a chat. No answer is expected.
    Typing {
        chat: String,
    },
    /// Download the `index`-th attachment of a received message into `dir`.
    Download {
        request: String,
        message: String,
        index: usize,
        max_bytes: u64,
        dir: PathBuf,
    },
    /// Unlink this device from the phone.
    Logout {
        #[serde(default)]
        request: Option<String>,
    },
    Shutdown,
}

/// Print one event line. Stdout is the protocol channel: nothing else may
/// write to it.
pub fn emit(event: &Value) {
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{event}");
    let _ = out.flush();
}

pub fn log(level: &str, message: impl std::fmt::Display) {
    emit(&json!({"event": "log", "level": level, "message": message.to_string()}));
}

pub fn reply_ok(request: &str, extra: Value) {
    let mut event = json!({"event": "result", "request": request, "ok": true});
    if let (Some(target), Some(extra)) = (event.as_object_mut(), extra.as_object()) {
        target.extend(extra.clone());
    }
    emit(&event);
}

pub fn reply_error(request: &str, error: impl std::fmt::Display) {
    emit(&json!({"event": "result", "request": request, "ok": false, "error": error.to_string()}));
}

/// The part of a file name that is safe to use on disk.
pub fn safe_file_stem(text: &str) -> String {
    let cleaned: String = text
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .take(64)
        .collect();
    if cleaned.is_empty() {
        "media".into()
    } else {
        cleaned
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commands_parse_the_way_the_agent_writes_them() {
        let send: Command = serde_json::from_str(
            r#"{"cmd":"send_text","request":"r1","chat":"15550100@s.whatsapp.net","text":"hi"}"#,
        )
        .unwrap();
        assert!(matches!(send, Command::SendText { ref text, .. } if text == "hi"));
        // The agent adds a request id to every command, typing included.
        let typing: Command =
            serde_json::from_str(r#"{"cmd":"typing","chat":"1@s.whatsapp.net","request":"r2"}"#)
                .unwrap();
        assert!(matches!(typing, Command::Typing { .. }));
        let download: Command = serde_json::from_str(
            r#"{"cmd":"download","request":"r3","message":"M1","index":2,"max_bytes":100,"dir":"/tmp/x"}"#,
        )
        .unwrap();
        assert!(matches!(download, Command::Download { index: 2, .. }));
        assert!(matches!(
            serde_json::from_str::<Command>(r#"{"cmd":"logout"}"#).unwrap(),
            Command::Logout { request: None }
        ));
        assert!(serde_json::from_str::<Command>(r#"{"cmd":"explode"}"#).is_err());
    }

    #[test]
    fn media_names_cannot_leave_the_download_folder() {
        assert_eq!(safe_file_stem("../../x"), "______x");
        assert_eq!(safe_file_stem("3EB0A1-b2"), "3EB0A1-b2");
        assert_eq!(safe_file_stem(""), "media");
        assert!(safe_file_stem(&"a".repeat(300)).len() <= 64);
    }
}
