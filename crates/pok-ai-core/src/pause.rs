use std::sync::atomic::{AtomicU8, Ordering};

use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use serde::Serialize;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use crate::{PokError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PauseState {
    Running,
    PauseRequested,
    Paused,
}

impl PauseState {
    const fn as_u8(self) -> u8 {
        match self {
            Self::Running => 0,
            Self::PauseRequested => 1,
            Self::Paused => 2,
        }
    }

    const fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::PauseRequested,
            2 => Self::Paused,
            _ => Self::Running,
        }
    }
}

/// Receives `true` when the agent starts holding input because the user is
/// using the mouse or keyboard, and `false` when it resumes.
pub type UserActivityListener = std::sync::Arc<dyn Fn(bool) + Send + Sync>;

#[derive(Default)]
pub struct PauseController {
    state: AtomicU8,
    notify: Notify,
    requested_at: Mutex<Option<DateTime<Utc>>>,
    resume_note: Mutex<Option<String>>,
    user_activity_listener: Mutex<Option<UserActivityListener>>,
}

impl PauseController {
    /// Register the observer told when input waits for the user.
    pub fn set_user_activity_listener(&self, listener: UserActivityListener) {
        *self.user_activity_listener.lock() = Some(listener);
    }

    /// Report that input is (or is no longer) waiting for the user.
    pub fn notify_waiting_for_user(&self, waiting: bool) {
        let listener = self.user_activity_listener.lock().clone();
        if let Some(listener) = listener {
            listener(waiting);
        }
    }

    pub fn state(&self) -> PauseState {
        PauseState::from_u8(self.state.load(Ordering::Acquire))
    }

    pub fn request_pause(&self) -> bool {
        let changed = self
            .state
            .compare_exchange(
                PauseState::Running.as_u8(),
                PauseState::PauseRequested.as_u8(),
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok();
        if changed {
            *self.requested_at.lock() = Some(Utc::now());
            self.notify.notify_waiters();
        }
        changed
    }

    pub fn mark_paused(&self) {
        if self.state() != PauseState::Running {
            self.state
                .store(PauseState::Paused.as_u8(), Ordering::Release);
            self.notify.notify_waiters();
        }
    }

    pub fn resume(&self, note: Option<String>) -> bool {
        if self.state() == PauseState::Running {
            return false;
        }
        *self.resume_note.lock() = note.filter(|value| !value.trim().is_empty());
        self.state
            .store(PauseState::Running.as_u8(), Ordering::Release);
        self.notify.notify_waiters();
        true
    }

    pub fn reset(&self) {
        self.state
            .store(PauseState::Running.as_u8(), Ordering::Release);
        *self.requested_at.lock() = None;
        *self.resume_note.lock() = None;
        self.notify.notify_waiters();
    }

    pub fn ensure_action_allowed(&self) -> Result<()> {
        if self.state() == PauseState::Running {
            Ok(())
        } else {
            Err(PokError::Tool(
                "agent paused by user before physical input; wait for resume and reobserve".into(),
            ))
        }
    }

    pub async fn wait_until_resumed(&self, cancellation: &CancellationToken) -> Result<()> {
        while self.state() != PauseState::Running {
            let notified = self.notify.notified();
            if self.state() == PauseState::Running {
                break;
            }
            tokio::select! {
                () = cancellation.cancelled() => return Err(PokError::Cancelled),
                () = notified => {}
            }
        }
        Ok(())
    }

    pub async fn pause_requested(&self) {
        while self.state() == PauseState::Running {
            let notified = self.notify.notified();
            if self.state() != PauseState::Running {
                break;
            }
            notified.await;
        }
    }

    pub fn take_resume_metadata(&self) -> (Option<DateTime<Utc>>, Option<String>) {
        (
            self.requested_at.lock().take(),
            self.resume_note.lock().take(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn pause_blocks_actions_until_manual_resume() {
        let controller = PauseController::default();
        assert!(controller.request_pause());
        assert!(controller.ensure_action_allowed().is_err());
        controller.mark_paused();
        assert_eq!(controller.state(), PauseState::Paused);
        assert!(controller.resume(Some("opened Discord".into())));
        controller
            .wait_until_resumed(&CancellationToken::new())
            .await
            .unwrap();
        assert!(controller.ensure_action_allowed().is_ok());
        assert_eq!(
            controller.take_resume_metadata().1.as_deref(),
            Some("opened Discord")
        );
    }
}
