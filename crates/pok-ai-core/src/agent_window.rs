//! Lets the agent work in its own window while the user keeps working in
//! another one.
//!
//! [`AgentWindowDesktop`] wraps a [`DesktopPlatform`] and tracks the window
//! the agent is working in. While that window is open and not minimized, the
//! rest of the runtime sees it as the foreground window, so every existing
//! observation, freshness and input check applies to the agent's window, not
//! to whatever the user has in front. Choosing a window (`activate_window`)
//! only moves the agent's attention; nothing is brought to the front.
//! Captures render the window itself, and accessibility (pattern) actions
//! reach it without the cursor. Only real mouse or keyboard input needs the
//! window in front: it is brought forward immediately before that input,
//! after the caller has waited for the user to stop using the machine.
//!
//! When the agent's window closes or is minimized, the wrapper reverts to the
//! real foreground window, so the runtime's normal desktop-change recovery
//! takes over.

use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;

use crate::{
    PokError, Result,
    platform::{
        DesktopActivitySnapshot, DesktopPlatform, PatternActionOutcome, PatternActionRequest,
        PatternScrollRequest, PatternTextOutcome, PatternTextRequest,
    },
    types::{
        CaptureRequest, CaptureScope, DesktopCapture, InputAction, MonitorInfo, OcrBlock,
        Screenshot, UiElement, WindowInfo,
    },
};

pub struct AgentWindowDesktop {
    inner: Arc<dyn DesktopPlatform>,
    agent_window: Mutex<Option<String>>,
    borrow: Arc<Mutex<FocusBorrow>>,
}

/// The user's window the agent took the foreground from for physical input,
/// and a counter of physical inputs so a return is only made once the agent
/// has finished its burst of input.
#[derive(Default)]
struct FocusBorrow {
    user_window: Option<String>,
    inputs: u64,
}

/// How long after its last physical input the agent returns the foreground
/// to the window the user had in front. Long enough that a run of steps
/// (type, then save, then confirm) keeps the window in front instead of
/// switching back and forth between the steps.
const FOCUS_RETURN_DELAY: std::time::Duration = std::time::Duration::from_secs(8);

/// Whether two native window ids (`pid:HANDLE(...)`) belong to one process.
fn same_process(left: &str, right: &str) -> bool {
    match (left.split_once(':'), right.split_once(':')) {
        (Some((left_pid, _)), Some((right_pid, _))) => {
            !left_pid.is_empty() && left_pid == right_pid
        }
        _ => false,
    }
}

impl AgentWindowDesktop {
    pub fn new(inner: Arc<dyn DesktopPlatform>) -> Arc<Self> {
        Arc::new(Self {
            inner,
            agent_window: Mutex::new(None),
            borrow: Arc::new(Mutex::new(FocusBorrow::default())),
        })
    }

    /// After a burst of physical input, give the foreground back to the
    /// window the user had in front, unless the user has already moved on.
    fn schedule_focus_return(&self) {
        let input = {
            let mut borrow = self.borrow.lock();
            if borrow.user_window.is_none() {
                return;
            }
            borrow.inputs = borrow.inputs.wrapping_add(1);
            borrow.inputs
        };
        let Some(agent_window) = self.agent_window.lock().clone() else {
            return;
        };
        let inner = self.inner.clone();
        let borrow = self.borrow.clone();
        tokio::spawn(async move {
            tokio::time::sleep(FOCUS_RETURN_DELAY).await;
            let user_window = {
                let mut borrow = borrow.lock();
                if borrow.inputs != input {
                    // More input followed; the latest input returns focus.
                    return;
                }
                borrow.user_window.take()
            };
            let Some(user_window) = user_window else {
                return;
            };
            // Only while the agent's window (or its dialog) is still in
            // front: if the user already switched elsewhere, leave it.
            let agent_in_front =
                inner
                    .foreground_window()
                    .await
                    .ok()
                    .flatten()
                    .is_some_and(|front| {
                        front.id == agent_window || same_process(&front.id, &agent_window)
                    });
            if agent_in_front {
                let _ = inner.activate_window(&user_window).await;
            }
        });
    }

    /// The agent's window, if it is still open and not minimized. A window
    /// of the same application that is really in front (a dialog, Save As,
    /// or popup the app opened) becomes the agent's window, and so does the
    /// main window again once the dialog closes.
    async fn agent_window(&self) -> Result<Option<WindowInfo>> {
        let Some(id) = self.agent_window.lock().clone() else {
            return Ok(None);
        };
        if let Some(front) = self.inner.foreground_window().await?
            && front.id != id
            && front.visible
            && !front.minimized
            && same_process(&front.id, &id)
        {
            let mut current = self.agent_window.lock();
            if current.as_deref() == Some(id.as_str()) {
                *current = Some(front.id.clone());
            }
            return Ok(Some(front));
        }
        let windows = self.inner.list_windows().await?;
        let usable = |window: &WindowInfo| window.visible && !window.minimized;
        if let Some(window) = windows
            .iter()
            .find(|window| window.id == id && usable(window))
        {
            return Ok(Some(window.clone()));
        }
        // A dialog that closed hands the work back to the application window
        // it belonged to (a message box over Save As returns to Save As).
        let owner = windows
            .into_iter()
            .find(|window| window.id != id && usable(window) && same_process(&window.id, &id));
        let mut current = self.agent_window.lock();
        if current.as_deref() == Some(id.as_str()) {
            // Gone or minimized with no other window of the application: the
            // real foreground is authoritative again.
            *current = owner.as_ref().map(|window| window.id.clone());
        }
        Ok(owner)
    }

    /// Bring the agent's window in front for real mouse or keyboard input.
    async fn ensure_real_focus(&self) -> Result<()> {
        let Some(window) = self.agent_window().await? else {
            return Ok(());
        };
        let in_front =
            |current: Option<&WindowInfo>| current.is_some_and(|current| current.id == window.id);
        let front = self.inner.foreground_window().await?;
        if in_front(front.as_ref()) {
            return Ok(());
        }
        // Remember the user's window so it can be given back afterwards.
        if let Some(front) = front.filter(|front| !same_process(&front.id, &window.id)) {
            let mut borrow = self.borrow.lock();
            if borrow.user_window.is_none() {
                borrow.user_window = Some(front.id);
            }
        }
        self.inner.activate_window(&window.id).await?;
        if !in_front(self.inner.foreground_window().await?.as_ref()) {
            return Err(PokError::Tool(format!(
                "could not bring {:?} to the front for mouse or keyboard input",
                window.title
            )));
        }
        Ok(())
    }

    /// Point "active window" requests at the agent's window.
    fn agent_request(&self, request: &CaptureRequest) -> CaptureRequest {
        let mut request = request.clone();
        if matches!(request.scope, CaptureScope::ActiveWindow)
            && let Some(id) = self.agent_window.lock().clone()
        {
            request.scope = CaptureScope::Window;
            request.window_id = Some(id);
        }
        request
    }
}

#[async_trait]
impl DesktopPlatform for AgentWindowDesktop {
    async fn capture_screens(&self) -> Result<Vec<Screenshot>> {
        self.inner.capture_screens().await
    }

    async fn launch_application(&self, name: &str) -> Result<()> {
        // A newly launched application opens in front; follow the real
        // foreground until the agent chooses a window again.
        *self.agent_window.lock() = None;
        self.inner.launch_application(name).await
    }

    async fn capture_target(&self, request: &CaptureRequest) -> Result<DesktopCapture> {
        let request = self.agent_request(request);
        let capture = self.inner.capture_target(&request).await?;
        if matches!(request.scope, CaptureScope::Window) {
            // Looking at a window makes it the one the agent works in.
            *self.agent_window.lock() = Some(capture.target.id.clone());
        }
        Ok(capture)
    }

    async fn query_ocr(&self) -> Result<Vec<OcrBlock>> {
        self.inner.query_ocr().await
    }

    async fn query_ocr_target(&self, request: &CaptureRequest) -> Result<Vec<OcrBlock>> {
        self.inner
            .query_ocr_target(&self.agent_request(request))
            .await
    }

    async fn query_ocr_capture(
        &self,
        capture: &DesktopCapture,
        request: &CaptureRequest,
    ) -> Result<Vec<OcrBlock>> {
        self.inner
            .query_ocr_capture(capture, &self.agent_request(request))
            .await
    }

    async fn query_ui_tree(&self) -> Result<Vec<UiElement>> {
        self.inner.query_ui_tree().await
    }

    async fn query_ui_tree_target(&self, request: &CaptureRequest) -> Result<Vec<UiElement>> {
        self.inner
            .query_ui_tree_target(&self.agent_request(request))
            .await
    }

    async fn query_ui_tree_target_limited(
        &self,
        request: &CaptureRequest,
        limit: usize,
    ) -> Result<Vec<UiElement>> {
        self.inner
            .query_ui_tree_target_limited(&self.agent_request(request), limit)
            .await
    }

    async fn list_monitors(&self) -> Result<Vec<MonitorInfo>> {
        self.inner.list_monitors().await
    }

    async fn list_windows(&self) -> Result<Vec<WindowInfo>> {
        self.inner.list_windows().await
    }

    async fn activate_window(&self, window_id: &str) -> Result<WindowInfo> {
        let window = self
            .inner
            .list_windows()
            .await?
            .into_iter()
            .find(|window| window.id == window_id);
        match window {
            // Choosing a visible window only moves the agent's attention.
            Some(window) if window.visible && !window.minimized => {
                *self.agent_window.lock() = Some(window.id.clone());
                Ok(window)
            }
            // A minimized window has to be restored, which brings it forward.
            _ => {
                let window = self.inner.activate_window(window_id).await?;
                *self.agent_window.lock() = Some(window.id.clone());
                Ok(window)
            }
        }
    }

    async fn bring_to_front(&self, window_id: &str) -> Result<WindowInfo> {
        let window = self.inner.activate_window(window_id).await?;
        *self.agent_window.lock() = Some(window.id.clone());
        Ok(window)
    }

    async fn foreground_window(&self) -> Result<Option<WindowInfo>> {
        match self.agent_window().await? {
            Some(window) => Ok(Some(window)),
            None => self.inner.foreground_window().await,
        }
    }

    async fn desktop_activity_snapshot(&self) -> Result<DesktopActivitySnapshot> {
        let mut snapshot = self.inner.desktop_activity_snapshot().await?;
        if let Some(window) = self.agent_window().await? {
            snapshot.foreground_window = Some(window);
        }
        Ok(snapshot)
    }

    async fn cursor_position(&self) -> Result<Option<(i32, i32)>> {
        self.inner.cursor_position().await
    }

    async fn simulate_input(&self, action: &InputAction) -> Result<()> {
        self.ensure_real_focus().await?;
        let result = self.inner.simulate_input(action).await;
        self.schedule_focus_return();
        result
    }

    async fn simulate_text(&self, text: &str, inter_key_pause_ms: u64) -> Result<()> {
        self.ensure_real_focus().await?;
        let result = self.inner.simulate_text(text, inter_key_pause_ms).await;
        self.schedule_focus_return();
        result
    }

    async fn focused_text(&self, max_chars: usize) -> Result<Option<String>> {
        // The focused control belongs to the window in front; read it only
        // from the agent's window, never from the user's.
        self.ensure_real_focus().await?;
        self.inner.focused_text(max_chars).await
    }

    async fn replace_focused_text(&self, text: &str) -> Result<bool> {
        self.ensure_real_focus().await?;
        let result = self.inner.replace_focused_text(text).await;
        self.schedule_focus_return();
        result
    }

    async fn perform_pattern_action(
        &self,
        request: &PatternActionRequest,
    ) -> Result<Option<PatternActionOutcome>> {
        self.inner.perform_pattern_action(request).await
    }

    async fn perform_pattern_scroll(
        &self,
        request: &PatternScrollRequest,
    ) -> Result<Option<PatternActionOutcome>> {
        self.inner.perform_pattern_scroll(request).await
    }

    async fn perform_pattern_text(
        &self,
        request: &PatternTextRequest,
    ) -> Result<Option<PatternTextOutcome>> {
        self.inner.perform_pattern_text(request).await
    }
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use uuid::Uuid;

    use super::*;
    use crate::{
        platform::{MockDesktop, PatternAction},
        types::{MouseButton, Observation, Rect},
    };

    fn window(id: &str, title: &str) -> WindowInfo {
        WindowInfo {
            id: id.into(),
            title: title.into(),
            process_name: format!("{title}.exe"),
            bounds: Rect {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
            elevated: false,
            visible: true,
            minimized: false,
        }
    }

    fn desktop() -> (Arc<MockDesktop>, Arc<AgentWindowDesktop>) {
        let mock = MockDesktop::new(Observation {
            version: Uuid::new_v4(),
            captured_at: Utc::now(),
            foreground_window: Some(window("user", "Editor")),
            target: None,
            cursor: None,
            screenshots: Vec::new(),
            ocr: Vec::new(),
            ui_elements: Vec::new(),
            targets: Vec::new(),
            timings_ms: Default::default(),
            warnings: Vec::new(),
        });
        mock.add_windows(vec![window("agent", "Settings")]);
        let wrapped = AgentWindowDesktop::new(mock.clone());
        (mock, wrapped)
    }

    #[tokio::test]
    async fn choosing_a_window_does_not_steal_focus() {
        let (mock, desktop) = desktop();
        desktop.activate_window("agent").await.unwrap();
        assert!(mock.activations().is_empty(), "nothing was brought forward");
        // The runtime works against the agent's window...
        let seen = desktop.foreground_window().await.unwrap().unwrap();
        assert_eq!(seen.id, "agent");
        // ...while the user's window stays in front.
        let real = mock.foreground_window().await.unwrap().unwrap();
        assert_eq!(real.id, "user");
    }

    #[tokio::test]
    async fn bringing_a_window_to_front_really_focuses_it() {
        let (mock, desktop) = desktop();
        desktop.bring_to_front("agent").await.unwrap();
        assert_eq!(mock.activations(), vec!["agent".to_owned()]);
        let seen = desktop.foreground_window().await.unwrap().unwrap();
        assert_eq!(seen.id, "agent");
    }

    #[tokio::test]
    async fn switching_windows_does_not_look_like_a_desktop_change() {
        let (mock, desktop) = desktop();
        desktop.activate_window("agent").await.unwrap();
        mock.switch_foreground("user");
        let snapshot = desktop.desktop_activity_snapshot().await.unwrap();
        assert_eq!(snapshot.foreground_window.unwrap().id, "agent");
    }

    #[tokio::test]
    async fn physical_input_brings_the_agent_window_forward_first() {
        let (mock, desktop) = desktop();
        desktop.activate_window("agent").await.unwrap();
        desktop
            .simulate_input(&InputAction::Click {
                x: 10,
                y: 10,
                button: MouseButton::Left,
            })
            .await
            .unwrap();
        assert_eq!(mock.activations(), vec!["agent".to_owned()]);
        assert_eq!(mock.actions().len(), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn the_users_window_comes_back_after_the_agents_input() {
        let (mock, desktop) = desktop();
        desktop.activate_window("agent").await.unwrap();
        desktop
            .simulate_input(&InputAction::Click {
                x: 10,
                y: 10,
                button: MouseButton::Left,
            })
            .await
            .unwrap();
        assert_eq!(mock.activations(), vec!["agent".to_owned()]);
        tokio::time::sleep(FOCUS_RETURN_DELAY + std::time::Duration::from_millis(300)).await;
        assert_eq!(
            mock.activations(),
            vec!["agent".to_owned(), "user".to_owned()],
            "the user's window is given back"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn steps_in_a_row_keep_the_window_in_front() {
        let (mock, desktop) = desktop();
        desktop.activate_window("agent").await.unwrap();
        let click = InputAction::Click {
            x: 10,
            y: 10,
            button: MouseButton::Left,
        };
        desktop.simulate_input(&click).await.unwrap();
        // The next step arrives while the planner is still working.
        tokio::time::sleep(FOCUS_RETURN_DELAY / 2).await;
        desktop.simulate_input(&click).await.unwrap();
        tokio::time::sleep(FOCUS_RETURN_DELAY / 2).await;
        assert_eq!(
            mock.activations(),
            vec!["agent".to_owned()],
            "no switch between steps"
        );
        tokio::time::sleep(FOCUS_RETURN_DELAY).await;
        assert_eq!(
            mock.activations(),
            vec!["agent".to_owned(), "user".to_owned()],
            "one return after the last step"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn focus_is_not_returned_when_the_user_already_moved_on() {
        let (mock, desktop) = desktop();
        mock.add_windows(vec![window("video", "YouTube")]);
        desktop.activate_window("agent").await.unwrap();
        desktop
            .simulate_input(&InputAction::Click {
                x: 10,
                y: 10,
                button: MouseButton::Left,
            })
            .await
            .unwrap();
        // The user clicks their video before the agent returns focus.
        mock.switch_foreground("video");
        tokio::time::sleep(FOCUS_RETURN_DELAY + std::time::Duration::from_millis(300)).await;
        assert_eq!(mock.activations(), vec!["agent".to_owned()]);
    }

    #[tokio::test]
    async fn pattern_actions_reach_the_agent_window_without_focus() {
        let (mock, desktop) = desktop();
        mock.enable_pattern_actions();
        desktop.activate_window("agent").await.unwrap();
        let outcome = desktop
            .perform_pattern_action(&PatternActionRequest {
                window_id: "agent".into(),
                name: "Display".into(),
                control_type: "button".into(),
                bounds: Rect {
                    x: 0,
                    y: 0,
                    width: 10,
                    height: 10,
                },
                point: (5, 5),
                action: PatternAction::Activate,
            })
            .await
            .unwrap();
        assert!(outcome.is_some());
        assert!(mock.activations().is_empty());
    }

    #[tokio::test]
    async fn active_window_requests_target_the_agent_window() {
        let (_mock, desktop) = desktop();
        desktop.activate_window("agent").await.unwrap();
        let request = desktop.agent_request(&CaptureRequest {
            scope: CaptureScope::ActiveWindow,
            window_id: None,
            monitor_id: None,
            region: None,
            max_edge: 640,
        });
        assert!(matches!(request.scope, CaptureScope::Window));
        assert_eq!(request.window_id.as_deref(), Some("agent"));
    }

    #[tokio::test]
    async fn a_dialog_the_app_opens_becomes_the_agent_window() {
        let (mock, desktop) = desktop();
        mock.add_windows(vec![
            window("7:HANDLE(0x1)", "Document1 - Word"),
            window("7:HANDLE(0x2)", "Save As"),
            window("9:HANDLE(0x3)", "Other app"),
        ]);
        desktop.activate_window("7:HANDLE(0x1)").await.unwrap();
        // The app opens its Save As dialog in front.
        mock.switch_foreground("7:HANDLE(0x2)");
        let seen = desktop.foreground_window().await.unwrap().unwrap();
        assert_eq!(seen.title, "Save As");
        // The dialog closes and the main window is in front again.
        mock.switch_foreground("7:HANDLE(0x1)");
        let seen = desktop.foreground_window().await.unwrap().unwrap();
        assert_eq!(seen.title, "Document1 - Word");
        // A different application in front is the user's, not the agent's.
        mock.switch_foreground("9:HANDLE(0x3)");
        let seen = desktop.foreground_window().await.unwrap().unwrap();
        assert_eq!(seen.title, "Document1 - Word");
    }

    #[test]
    fn process_identity_comes_from_the_window_id() {
        assert!(same_process("7:HANDLE(0x1)", "7:HANDLE(0x2)"));
        assert!(!same_process("7:HANDLE(0x1)", "9:HANDLE(0x1)"));
        assert!(!same_process("agent", "user"));
    }

    #[tokio::test]
    async fn a_closed_agent_window_hands_back_to_the_real_foreground() {
        let (mock, desktop) = desktop();
        desktop.activate_window("agent").await.unwrap();
        // The agent's window disappears from the window list.
        *desktop.agent_window.lock() = Some("closed".into());
        let seen = desktop.foreground_window().await.unwrap().unwrap();
        assert_eq!(seen.id, "user");
        assert!(desktop.agent_window.lock().is_none());
        assert!(mock.activations().is_empty());
    }

    #[tokio::test]
    async fn a_closed_message_box_hands_back_to_its_applications_window() {
        let (mock, desktop) = desktop();
        mock.add_windows(vec![
            window("7:HANDLE(0x2)", "Save As"),
            window("9:HANDLE(0x3)", "Other app"),
        ]);
        mock.switch_foreground("9:HANDLE(0x3)");
        // The agent followed a message box over Save As, which has closed.
        *desktop.agent_window.lock() = Some("7:HANDLE(0x9)".into());
        let seen = desktop.foreground_window().await.unwrap().unwrap();
        assert_eq!(seen.title, "Save As");
        assert_eq!(
            desktop.agent_window.lock().as_deref(),
            Some("7:HANDLE(0x2)")
        );
    }
}
