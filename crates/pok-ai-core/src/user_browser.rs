//! The user's own Chrome or Edge, attached over the Chrome DevTools Protocol
//! so the agent works in the sessions the user is already signed in to. No
//! cookies or credentials are read or copied: they stay inside the browser,
//! which simply sends the user's session with each request the agent makes.
//!
//! Unlike the managed browser, nothing here launches, restarts, or closes the
//! browser. The user starts it with `--remote-debugging-port` (the status
//! tool and `pok-ai doctor` print the exact command when it is not running),
//! and then:
//!
//! - The agent works in tabs it opens itself, in the background, so the
//!   user's tabs and focus are left alone. It enters one of the user's tabs
//!   only when switched there explicitly, and never navigates one away or
//!   closes it.
//! - Input goes to the page, never through the system cursor or keyboard, so
//!   the user keeps working. A tab in front gets native DevTools input; a
//!   background tab, which the browser neither draws nor sends native input
//!   to, gets the equivalent page events (pointer and mouse events and a
//!   click, text insertion, Enter and form submission).
//! - System 2 is the permission arbiter: a click, submission, or navigation
//!   that deletes, buys, sends, or changes account settings pauses for the
//!   user's explicit confirmation (`guardrail`) in every policy mode.
//!   System 1 (`fast_actions`) never receives those controls as candidates;
//!   a plan step that names one hands it back as `commit_required`.
//! - Password fields are never filled, and their values never leave the page.

use std::{
    collections::{HashMap, HashSet},
    io::Cursor,
    path::{Path, PathBuf},
    sync::{Arc, OnceLock},
    time::Duration,
};

use async_trait::async_trait;
use base64::Engine;
use parking_lot::Mutex as SyncMutex;
use schemars::{JsonSchema, schema_for};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::{
    PokError, Result,
    browser::{
        BrowserConnection, ManagedBrowser, bounded_label, devtools_http_client,
        snapshot_decision_state,
    },
    config::UserBrowserConfig,
    decision::{DecisionCandidate, DecisionCandidateKind},
    policy::{PolicyMode, RiskClass},
    tool::{Tool, ToolContext, ToolRegistry},
};

/// HTML returned by `user_browser_extract` with `include_html`.
const HTML_LIMIT: usize = 24_000;
/// Candidate ids for System 1 clicks that the guardrail flags. They are
/// withheld from System 1 and handed back to System 2 instead.
const COMMIT_CANDIDATE_PREFIX: &str = "user_browser_commit_";
const CLICK_CANDIDATE_PREFIX: &str = "user_browser_click_";

#[derive(Default)]
struct UserBrowserState {
    connection: Option<BrowserConnection>,
    /// Tabs the agent opened, or that opened from its clicks: the only tabs
    /// it navigates away from or closes.
    owned_tabs: HashSet<String>,
    /// Tabs holding a form the agent filled and did not submit, by address:
    /// opening another page goes to a new tab, so the filled form stays.
    filled_tabs: HashMap<String, String>,
    /// The form fields on each tab the agent has had open, as its latest read
    /// of that tab showed them: the run's record of what is filled where.
    forms: HashMap<String, FormPage>,
}

/// The fields of a form on one page, read from the page itself.
#[derive(Debug, Clone, PartialEq)]
struct FormPage {
    url: String,
    title: String,
    fields: Vec<FormField>,
}

#[derive(Debug, Clone, PartialEq)]
struct FormField {
    label: String,
    filled: bool,
    /// A check box: unchecked can be the right answer, so it is not "empty".
    check_box: bool,
}

pub struct UserBrowser {
    port: u16,
    user_data_dir: PathBuf,
}

/// One tab in the user's browser.
#[derive(Debug, Clone)]
struct Tab {
    id: String,
    title: String,
    url: String,
}

/// The control an action resolved to, pinned to the snapshot it came from
/// so the action is refused if the control changed in the meantime.
struct Resolved {
    snapshot_id: String,
    target_id: String,
    element: Value,
    page_url: String,
}

/// The agent's state per DevTools port, shared by every registry built in
/// this process so a conversation keeps working in the tab it opened. A
/// swarm worker has its own entry (keyed by its session), so each worker
/// works in its own tab.
type StateKey = (u16, Option<Uuid>);
static STATES: OnceLock<SyncMutex<HashMap<StateKey, Arc<Mutex<UserBrowserState>>>>> =
    OnceLock::new();
/// The most recently registered user browser, read by System 1.
static ACTIVE: OnceLock<SyncMutex<Option<Arc<UserBrowser>>>> = OnceLock::new();

fn active() -> Option<Arc<UserBrowser>> {
    ACTIVE.get().and_then(|active| active.lock().clone())
}

/// The profile folder the suggested launch command names by default.
pub fn default_profile_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("browser").join("profiles").join("personal")
}

impl UserBrowser {
    fn new(config: &UserBrowserConfig, data_dir: &Path) -> Self {
        Self {
            port: config.port,
            user_data_dir: config
                .user_data_dir
                .clone()
                .unwrap_or_else(|| default_profile_dir(data_dir)),
        }
    }

    /// The agent's state for this session: shared by the main agent's
    /// conversations, separate for each swarm worker.
    fn state(&self, session: Uuid) -> Arc<Mutex<UserBrowserState>> {
        let lane = crate::swarm::is_worker(session).then_some(session);
        STATES
            .get_or_init(Default::default)
            .lock()
            .entry((self.port, lane))
            .or_default()
            .clone()
    }

    /// The DevTools `/json/version` answer, or why the browser is not there.
    async fn version(&self) -> std::result::Result<Value, String> {
        let client = devtools_http_client().map_err(|error| error.to_string())?;
        let response = client
            .get(format!("http://127.0.0.1:{}/json/version", self.port))
            .send()
            .await
            .map_err(|error| {
                if error.is_connect() {
                    "nothing is listening on that port".to_owned()
                } else if error.is_timeout() {
                    "it did not answer in time".to_owned()
                } else {
                    error.to_string()
                }
            })?;
        if !response.status().is_success() {
            return Err(format!("it answered HTTP {}", response.status()));
        }
        let version = response
            .json::<Value>()
            .await
            .map_err(|_| "it did not answer like a Chrome DevTools endpoint".to_owned())?;
        if version.get("webSocketDebuggerUrl").is_none() {
            return Err("it did not answer like a Chrome DevTools endpoint".into());
        }
        Ok(version)
    }

    /// The command that starts the user's browser so it can be attached.
    fn launch_command(&self) -> String {
        launch_command(
            self.port,
            &self.user_data_dir,
            &crate::browser::chromium_candidates(),
        )
    }

    fn not_running(&self, reason: &str) -> PokError {
        PokError::Tool(format!(
            "the user's browser is not reachable on DevTools port {} ({reason}). Tell the user to start it with remote debugging by running this command (if that profile is already open without remote debugging, they close its windows first):\n{}\nand then to sign in to their sites in that window once; it remembers them next time. Chrome 136 and later allow remote debugging only with a separate --user-data-dir, which is why the command names one. Do not retry until the user says the browser is running.",
            self.port,
            self.launch_command()
        ))
    }

    /// Check the browser is there and keep a connection to it. The user's
    /// browser is never started from here.
    async fn attach(&self, state: &mut UserBrowserState) -> Result<()> {
        self.version()
            .await
            .map_err(|reason| self.not_running(&reason))?;
        if state.connection.is_none() {
            state.connection = Some(BrowserConnection::user(self.port));
        }
        Ok(())
    }

    async fn tabs(&self) -> Result<Vec<Tab>> {
        let pages = devtools_http_client()?
            .get(format!("http://127.0.0.1:{}/json/list", self.port))
            .send()
            .await
            .map_err(|error| PokError::Tool(format!("could not list the browser's tabs: {error}")))?
            .json::<Vec<Value>>()
            .await
            .map_err(|error| PokError::Tool(format!("invalid tab list: {error}")))?;
        Ok(pages
            .iter()
            .filter(|page| page.get("type").and_then(Value::as_str) == Some("page"))
            .filter_map(|page| {
                let text = |key: &str| {
                    page.get(key)
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned()
                };
                Some(Tab {
                    id: page.get("id")?.as_str()?.to_owned(),
                    title: text("title"),
                    url: text("url"),
                })
            })
            .collect())
    }

    /// A new tab that opens behind the user's current tab, so their focus
    /// and what they are looking at stay as they were.
    async fn open_background_tab(&self) -> Result<String> {
        self.version()
            .await
            .map_err(|reason| self.not_running(&reason))?;
        crate::browser::open_background_tab(self.port).await
    }

    fn tab_listing(&self, tabs: &[Tab], state: &UserBrowserState) -> Vec<Value> {
        let current = state
            .connection
            .as_ref()
            .and_then(|connection| connection.page_target.as_deref());
        tabs.iter()
            .map(|tab| {
                json!({
                    "tab_id": tab.id,
                    "title": crate::decision::bounded_text(&tab.title, 160),
                    "url": crate::decision::bounded_text(&tab.url, 300),
                    "opened_by_agent": state.owned_tabs.contains(&tab.id),
                    "agent_is_here": current == Some(tab.id.as_str()),
                })
            })
            .collect()
    }

    /// The tab the agent is working in, or an error that lists the tabs to
    /// choose from. The agent never picks one of the user's tabs by itself.
    async fn require_tab(&self, state: &UserBrowserState) -> Result<()> {
        let current = state
            .connection
            .as_ref()
            .and_then(|connection| connection.page_target.clone());
        let tabs = self.tabs().await?;
        if current.is_some_and(|id| tabs.iter().any(|tab| tab.id == id)) {
            return Ok(());
        }
        Err(PokError::Tool(format!(
            "the agent has no tab open in the user's browser. Open a page with user_browser_navigate, or call user_browser_tabs with action \"switch\" and one of these tab ids to work in a tab the user has open: {}",
            Value::Array(self.tab_listing(&tabs, state))
        )))
    }

    /// Label a page result with the tab it came from.
    fn decorate(&self, mut result: Value, state: &UserBrowserState) -> Value {
        let tab = state
            .connection
            .as_ref()
            .and_then(|connection| connection.page_target.clone());
        if let Some(object) = result.as_object_mut() {
            object.insert(
                "tab".into(),
                json!({
                    "tab_id": tab,
                    "opened_by_agent": tab.as_ref().is_some_and(|id| state.owned_tabs.contains(id)),
                }),
            );
        }
        result
    }

    async fn status(&self, context: &ToolContext) -> Value {
        let state = self.state(context.session_id);
        let mut guard = state.lock().await;
        let state = &mut *guard;
        match self.version().await {
            Ok(version) => {
                if state.connection.is_none() {
                    state.connection = Some(BrowserConnection::user(self.port));
                }
                let tabs = self.tabs().await.unwrap_or_default();
                json!({
                    "connected": true,
                    "port": self.port,
                    "browser": version.get("Browser"),
                    "tabs": self.tab_listing(&tabs, state),
                    "how_it_works": "The agent works in the user's signed-in sessions, in background tabs it opens, without using their mouse or keyboard. Open a page with user_browser_navigate, or switch to a tab the user has open with user_browser_tabs.",
                })
            }
            Err(reason) => json!({
                "connected": false,
                "port": self.port,
                "reason": reason,
                "launch_command": self.launch_command(),
                "profile_folder": self.user_data_dir,
                "instructions": "Give the user the launch_command to run (in PowerShell on Windows). If that profile is already open without remote debugging, they close its windows first. They sign in to their sites in that window once, and it remembers them. Chrome 136 and later allow remote debugging only with a separate --user-data-dir, which is why the command names one. Wait for the user to say it is running before trying again.",
            }),
        }
    }

    async fn navigate(&self, context: &ToolContext, url: &str) -> Result<Value> {
        let parsed = reqwest::Url::parse(url)
            .ok()
            .filter(|parsed| matches!(parsed.scheme(), "http" | "https"))
            .ok_or_else(|| {
                PokError::Tool("user_browser_navigate needs an absolute http(s) URL".into())
            })?;
        let state = self.state(context.session_id);
        let mut guard = state.lock().await;
        let state = &mut *guard;
        self.attach(state).await?;
        if let Some((consequence, evidence)) = guardrail::navigation(parsed.as_str()) {
            confirm(
                context,
                "user_browser_navigate",
                json!({"url": parsed.as_str()}),
                consequence,
                &evidence,
                parsed.as_str(),
            )
            .await?;
        }
        let tabs = self.tabs().await?;
        state
            .filled_tabs
            .retain(|id, _| tabs.iter().any(|tab| &tab.id == id));
        let current = state
            .connection
            .as_ref()
            .and_then(|connection| connection.page_target.clone());
        // A form filled and not submitted would be lost by loading another
        // page over it: that tab is kept, and the page opens in a new one.
        let kept = current.as_ref().and_then(|id| {
            state
                .filled_tabs
                .get(id)
                .map(|url| (id.clone(), url.clone()))
        });
        // Only a tab the agent opened is navigated away from its page.
        let reusable = current
            .filter(|id| state.owned_tabs.contains(id) && tabs.iter().any(|tab| &tab.id == id))
            .filter(|_| kept.is_none());
        let tab_id = match reusable {
            Some(id) => id,
            None => {
                let id = self.open_background_tab().await?;
                state.owned_tabs.insert(id.clone());
                id
            }
        };
        let connection = state.connection.as_mut().expect("attached");
        connection.page_target = Some(tab_id);
        let navigation =
            ManagedBrowser::cdp(connection, "Page.navigate", json!({"url": parsed.as_str()}))
                .await?;
        if let Some(error) = navigation
            .get("errorText")
            .and_then(Value::as_str)
            .filter(|error| !error.is_empty())
        {
            return Err(PokError::Tool(format!(
                "the page {parsed} did not load: {error}"
            )));
        }
        tokio::time::sleep(Duration::from_millis(300)).await;
        let mut snapshot = ManagedBrowser::settled_snapshot(connection, true).await?;
        if let Some((id, url)) = kept {
            snapshot["kept_filled_form"] = json!({
                "tab_id": id,
                "url": url,
                "note": "The form you filled there and did not submit stays open in that tab; this page opened in a new tab. Switch back with user_browser_tabs to finish or check it.",
            });
        }
        Ok(self.decorate(snapshot, state))
    }

    /// Resolve `target`, ask the user first when the guardrail flags the
    /// action, then act and return the settled page.
    async fn act(
        &self,
        context: &ToolContext,
        tool: &str,
        target: &str,
        action: Action<'_>,
    ) -> Result<Value> {
        let state = self.state(context.session_id);
        let mut guard = state.lock().await;
        let state = &mut *guard;
        self.attach(state).await?;
        self.require_tab(state).await?;
        let connection = state.connection.as_mut().expect("attached");
        let resolved = resolve_target(connection, target).await?;
        let flagged = match action {
            Action::Click => guardrail::click(&resolved.element, &resolved.page_url),
            Action::Type { submit: true, .. } => {
                guardrail::submit(&resolved.element, &resolved.page_url)
            }
            Action::Type { .. } => None,
            Action::Select { .. } => guardrail::select(&resolved.element, &resolved.page_url),
        };
        // A click that sends or submits (confirmed below) leaves no unsent form.
        let flagged_send = flagged.as_ref().is_some_and(|(consequence, _)| {
            matches!(consequence, guardrail::Consequence::SendOrSubmit)
        });
        if let Action::Type { .. } = action
            && resolved.element.get("sensitive").and_then(Value::as_bool) == Some(true)
        {
            return Err(PokError::PolicyDenied {
                tool: tool.into(),
                reason: "password fields are never filled by the agent; the user's signed-in session usually makes a password unnecessary, otherwise ask the user to enter it".into(),
            });
        }
        if let Some((consequence, evidence)) = flagged {
            let request = match action {
                Action::Click => json!({"target": target}),
                Action::Type { text, submit } => {
                    json!({"target": target, "text": text, "submit": submit})
                }
                Action::Select { option } => json!({"target": target, "option": option}),
            };
            confirm(
                context,
                tool,
                request,
                consequence,
                &evidence,
                &resolved.page_url,
            )
            .await?;
        }
        let before = connection.page_target.clone();
        let (operation, value) = match action {
            Action::Click => ("click", None),
            Action::Type { text, .. } => ("type", Some(text)),
            Action::Select { option } => ("select", Some(option)),
        };
        let mut result = ManagedBrowser::act_locked(
            connection,
            &resolved.snapshot_id,
            &resolved.target_id,
            operation,
            value,
        )
        .await?;
        if let Action::Type { submit: true, .. } = action
            && result.get("snapshot_error").is_none()
        {
            // Enter goes to the field just filled, which still has focus. A
            // background tab drops native keys, so there the page presses it.
            let pressed = ManagedBrowser::cdp(
                connection,
                "Runtime.evaluate",
                json!({"expression": PRESS_ENTER_SCRIPT, "returnByValue": true}),
            )
            .await?;
            if pressed.pointer("/result/value/ok").and_then(Value::as_bool) != Some(true) {
                return Err(PokError::Tool(format!(
                    "the text was entered but Enter was not pressed: {}",
                    pressed
                        .pointer("/result/value/reason")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown")
                )));
            }
            if pressed
                .pointer("/result/value/native")
                .and_then(Value::as_bool)
                == Some(true)
            {
                for kind in ["keyDown", "keyUp"] {
                    let mut key = json!({
                        "type": kind, "key": "Enter", "code": "Enter",
                        "windowsVirtualKeyCode": 13, "nativeVirtualKeyCode": 13
                    });
                    if kind == "keyDown" {
                        key["text"] = json!("\r");
                    }
                    ManagedBrowser::cdp(connection, "Input.dispatchKeyEvent", key).await?;
                }
            }
            tokio::time::sleep(Duration::from_millis(400)).await;
            result = ManagedBrowser::snapshot_after_action(
                connection,
                json!({"operation": "type", "target_id": resolved.target_id, "submitted": true, "ok": true}),
            )
            .await;
        }
        // A click that opened a new tab moved the agent there; that tab is
        // the agent's own.
        if connection.page_target != before
            && let Some(id) = connection.page_target.clone()
        {
            state.owned_tabs.insert(id);
        }
        // Entering a value leaves an unsent form in this tab; pressing Enter
        // to submit it, or a confirmed send, ends that.
        if let Some(id) = connection.page_target.clone() {
            let checkbox = matches!(
                resolved.element.get("input_type").and_then(Value::as_str),
                Some("checkbox" | "radio")
            );
            match action {
                Action::Type { submit: true, .. } => {
                    state.filled_tabs.remove(&id);
                }
                Action::Type { .. } | Action::Select { .. } => {
                    state.filled_tabs.insert(id, resolved.page_url.clone());
                }
                Action::Click if checkbox => {
                    state.filled_tabs.insert(id, resolved.page_url.clone());
                }
                Action::Click if flagged_send => {
                    state.filled_tabs.remove(&id);
                }
                Action::Click => {}
            }
        }
        // The control's label, so a verified run can be replayed by label
        // (motor programs) on a page whose element ids differ.
        if let Some(label) = resolved
            .element
            .get("name")
            .and_then(Value::as_str)
            .filter(|label| !label.trim().is_empty())
            && let Some(action) = result.get_mut("action").and_then(Value::as_object_mut)
        {
            action.insert("label".into(), json!(label));
        }
        Ok(self.decorate(result, state))
    }

    async fn extract(
        &self,
        context: &ToolContext,
        selector: Option<&str>,
        include_html: bool,
        page: Option<(usize, usize)>,
    ) -> Result<Value> {
        let state = self.state(context.session_id);
        let mut guard = state.lock().await;
        let state = &mut *guard;
        self.attach(state).await?;
        self.require_tab(state).await?;
        let connection = state.connection.as_mut().expect("attached");
        let mut snapshot = ManagedBrowser::settled_snapshot(connection, true).await?;
        if selector.is_some() || include_html {
            let expression = EXTRACT_SCRIPT
                .replace("__SELECTOR__", &serde_json::to_string(&selector)?)
                .replace("__LIMIT__", &HTML_LIMIT.to_string());
            let section = ManagedBrowser::cdp(
                connection,
                "Runtime.evaluate",
                json!({"expression": expression, "returnByValue": true}),
            )
            .await?;
            let section = section
                .pointer("/result/value")
                .cloned()
                .unwrap_or(Value::Null);
            if section.get("found").and_then(Value::as_bool) != Some(true) {
                return Err(PokError::Tool(format!(
                    "no element on the page matches the selector {:?}",
                    selector.unwrap_or_default()
                )));
            }
            if include_html {
                for key in ["html", "html_chars", "html_truncated"] {
                    snapshot[key] = section.get(key).cloned().unwrap_or(Value::Null);
                }
            }
            if selector.is_some() {
                snapshot["section_text"] = section.get("text").cloned().unwrap_or(Value::Null);
            }
        }
        // The whole page text, read in parts; the snapshot shows its start.
        let full = ManagedBrowser::cdp(
            connection,
            "Runtime.evaluate",
            json!({"expression": PAGE_TEXT_SCRIPT, "returnByValue": true}),
        )
        .await?;
        let full = full
            .pointer("/result/value")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let total = full.chars().count();
        let (offset, shown) = match page {
            Some((offset, max)) => {
                let part = full.chars().skip(offset).take(max).collect::<String>();
                let shown = part.chars().count();
                snapshot["text"] = json!(part);
                (offset, shown)
            }
            None => (
                0,
                snapshot
                    .get("text")
                    .and_then(Value::as_str)
                    .map_or(0, |text| text.chars().count()),
            ),
        };
        snapshot["text_offset"] = json!(offset);
        snapshot["total_chars"] = json!(total);
        snapshot["next_offset"] = json!((offset + shown < total).then_some(offset + shown));
        Ok(self.decorate(snapshot, state))
    }

    async fn scroll(&self, context: &ToolContext, direction: &str, amount: &str) -> Result<Value> {
        let state = self.state(context.session_id);
        let mut guard = state.lock().await;
        let state = &mut *guard;
        self.attach(state).await?;
        self.require_tab(state).await?;
        let connection = state.connection.as_mut().expect("attached");
        let script = SCROLL_SCRIPT
            .replace("__DIRECTION__", &serde_json::to_string(direction)?)
            .replace("__AMOUNT__", &serde_json::to_string(amount)?);
        let moved = ManagedBrowser::cdp(
            connection,
            "Runtime.evaluate",
            json!({"expression": script, "returnByValue": true}),
        )
        .await?;
        let moved = moved
            .pointer("/result/value")
            .cloned()
            .unwrap_or(Value::Null);
        tokio::time::sleep(Duration::from_millis(500)).await;
        let at = moved.get("at").and_then(Value::as_u64);
        if moved.get("hidden").and_then(Value::as_bool) == Some(true)
            && at.is_some()
            && at != moved.get("from").and_then(Value::as_u64)
        {
            // A background tab does not run the rendering steps that fire
            // visibility observers (lazy images, infinite feeds); a capture of
            // the visible area renders it once without showing it, so what a
            // person scrolling there would see can load, and no more. The
            // browser draws a background tab only when something moved, so a
            // scroll that did not move is not rendered, and a capture that
            // does not come back in time is left.
            let clip = json!({
                "x": 0, "y": at, "scale": 1,
                "width": moved.get("width").and_then(Value::as_u64).unwrap_or(1280),
                "height": moved.get("view").and_then(Value::as_u64).unwrap_or(800),
            });
            let _ = tokio::time::timeout(
                Duration::from_secs(3),
                ManagedBrowser::cdp(
                    connection,
                    "Page.captureScreenshot",
                    json!({"format": "jpeg", "quality": 1, "captureBeyondViewport": true, "clip": clip}),
                ),
            )
            .await;
            tokio::time::sleep(Duration::from_millis(700)).await;
            if let Some(at) = at {
                restore_scroll(connection, at).await;
            }
        }
        let mut snapshot = ManagedBrowser::settled_snapshot(connection, true).await?;
        let height = snapshot
            .pointer("/metrics/page_height")
            .and_then(Value::as_u64)
            .unwrap_or_default();
        let before = moved
            .get("height")
            .and_then(Value::as_u64)
            .unwrap_or_default();
        let to = snapshot
            .pointer("/metrics/scroll_y")
            .and_then(Value::as_u64)
            .unwrap_or_default();
        let viewport = snapshot
            .pointer("/metrics/viewport_height")
            .and_then(Value::as_u64)
            .unwrap_or_default();
        snapshot["scroll"] = json!({
            "from": moved.get("from"),
            "to": to,
            "page_height": height,
            "more_loaded": height > before,
            "at_end": to + viewport + 2 >= height,
            "at_top": to == 0,
        });
        Ok(self.decorate(snapshot, state))
    }

    async fn wait_for(
        &self,
        context: &ToolContext,
        text: Option<&str>,
        selector: Option<&str>,
        gone: bool,
        timeout: Duration,
    ) -> Result<Value> {
        if text.is_none() && selector.is_none() {
            return Err(PokError::Tool(
                "wait for text, a CSS selector, or both".into(),
            ));
        }
        let state = self.state(context.session_id);
        let mut guard = state.lock().await;
        let state = &mut *guard;
        self.attach(state).await?;
        self.require_tab(state).await?;
        let connection = state.connection.as_mut().expect("attached");
        let script = WAIT_SCRIPT
            .replace("__TEXT__", &serde_json::to_string(&text)?)
            .replace("__SELECTOR__", &serde_json::to_string(&selector)?);
        let started = tokio::time::Instant::now();
        let mut last = Value::Null;
        while started.elapsed() < timeout {
            if context.cancellation.is_cancelled() {
                return Err(PokError::Cancelled);
            }
            // A page replaced by a navigation cannot be read for a moment.
            if let Ok(found) = ManagedBrowser::cdp(
                connection,
                "Runtime.evaluate",
                json!({"expression": script, "returnByValue": true}),
            )
            .await
            {
                last = found
                    .pointer("/result/value")
                    .cloned()
                    .unwrap_or(Value::Null);
                if last.get("present").and_then(Value::as_bool) == Some(!gone) {
                    let mut snapshot = ManagedBrowser::settled_snapshot(connection, false).await?;
                    snapshot["waited_ms"] = json!(started.elapsed().as_millis() as u64);
                    return Ok(self.decorate(snapshot, state));
                }
            }
            tokio::time::sleep(Duration::from_millis(300)).await;
        }
        Err(PokError::Tool(format!(
            "{} {} within {} seconds; the page shows: {}",
            match (text, selector) {
                (Some(text), Some(selector)) => format!("{text:?} in {selector:?}"),
                (Some(text), None) => format!("{text:?}"),
                (None, Some(selector)) => format!("{selector:?}"),
                (None, None) => unreachable!("checked above"),
            },
            if gone {
                "did not go away"
            } else {
                "did not appear"
            },
            timeout.as_secs(),
            crate::decision::bounded_text(
                last.get("excerpt").and_then(Value::as_str).unwrap_or(""),
                300
            )
        )))
    }

    async fn screenshot(&self, context: &ToolContext) -> Result<Value> {
        let max_edge = context.vision_max_edge;
        let state = self.state(context.session_id);
        let mut guard = state.lock().await;
        let state = &mut *guard;
        self.attach(state).await?;
        self.require_tab(state).await?;
        let connection = state.connection.as_mut().expect("attached");
        let page = ManagedBrowser::cdp(
            connection,
            "Runtime.evaluate",
            json!({"expression": "({url: location.href, title: document.title, y: Math.round(scrollY), hidden: document.visibilityState !== 'visible'})", "returnByValue": true}),
        )
        .await?;
        // A background tab is not drawn; capturing beyond the viewport makes
        // the browser render it for the capture without showing it.
        let hidden = page
            .pointer("/result/value/hidden")
            .and_then(Value::as_bool)
            == Some(true);
        let shot = ManagedBrowser::cdp(
            connection,
            "Page.captureScreenshot",
            json!({"format": "png", "captureBeyondViewport": hidden}),
        )
        .await?;
        if hidden && let Some(y) = page.pointer("/result/value/y").and_then(Value::as_u64) {
            restore_scroll(connection, y).await;
        }
        let data = shot
            .get("data")
            .and_then(Value::as_str)
            .ok_or_else(|| PokError::Tool("the browser returned no screenshot".into()))?;
        let (png_base64, width, height) = bounded_png(data, max_edge)?;
        let result = json!({
            "png_base64": png_base64,
            "width": width,
            "height": height,
            "url": page.pointer("/result/value/url"),
            "title": page.pointer("/result/value/title"),
        });
        Ok(self.decorate(result, state))
    }

    async fn tabs_action(
        &self,
        context: &ToolContext,
        action: TabAction,
        tab_id: Option<&str>,
    ) -> Result<Value> {
        let state = self.state(context.session_id);
        let mut guard = state.lock().await;
        let state = &mut *guard;
        self.attach(state).await?;
        let tabs = self.tabs().await?;
        let named = || {
            let id = tab_id.ok_or_else(|| PokError::Tool("this action needs a tab_id".into()))?;
            tabs.iter()
                .find(|tab| tab.id == id)
                .cloned()
                .ok_or_else(|| PokError::Tool(format!("no open tab has the id {id:?}")))
        };
        match action {
            TabAction::List => Ok(json!({"tabs": self.tab_listing(&tabs, state)})),
            TabAction::Switch => {
                let tab = named()?;
                let connection = state.connection.as_mut().expect("attached");
                connection.page_target = Some(tab.id);
                let snapshot = ManagedBrowser::settled_snapshot(connection, true).await?;
                Ok(self.decorate(snapshot, state))
            }
            TabAction::Close => {
                let tab = named()?;
                if !state.owned_tabs.contains(&tab.id) {
                    return Err(PokError::PolicyDenied {
                        tool: "user_browser_tabs".into(),
                        reason:
                            "the agent closes only tabs it opened; the user's own tabs stay open"
                                .into(),
                    });
                }
                devtools_http_client()?
                    .get(format!(
                        "http://127.0.0.1:{}/json/close/{}",
                        self.port, tab.id
                    ))
                    .send()
                    .await
                    .map_err(|error| PokError::Tool(format!("could not close the tab: {error}")))?;
                state.owned_tabs.remove(&tab.id);
                state.filled_tabs.remove(&tab.id);
                state.forms.remove(&tab.id);
                if let Some(connection) = state.connection.as_mut()
                    && connection.page_target.as_deref() == Some(tab.id.as_str())
                {
                    connection.page_target = None;
                }
                let tabs = self.tabs().await?;
                Ok(json!({"closed": tab.id, "tabs": self.tab_listing(&tabs, state)}))
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Action<'a> {
    Click,
    Type { text: &'a str, submit: bool },
    Select { option: &'a str },
}

/// The launch command for a browser listening on `port` with the profile
/// in `user_data_dir`, preferring an installed Chrome, then Edge.
fn launch_command(port: u16, user_data_dir: &Path, installed: &[PathBuf]) -> String {
    let dir = user_data_dir.display();
    if cfg!(target_os = "macos") {
        return format!(
            "open -na \"Google Chrome\" --args --remote-debugging-port={port} --user-data-dir=\"{dir}\""
        );
    }
    if !cfg!(windows) {
        return format!("google-chrome --remote-debugging-port={port} --user-data-dir=\"{dir}\"");
    }
    let executable = installed
        .iter()
        .find(|path| {
            path.file_name()
                .is_some_and(|name| name.eq_ignore_ascii_case("chrome.exe"))
        })
        .or_else(|| installed.first());
    match executable {
        Some(path) => format!(
            "& \"{}\" --remote-debugging-port={port} --user-data-dir=\"{dir}\"",
            path.display()
        ),
        None => format!(
            "Start-Process chrome -ArgumentList '--remote-debugging-port={port}', '--user-data-dir=\"{dir}\"'"
        ),
    }
}

/// Lowercase words separated by single spaces, with a space at each end, so
/// a phrase matches only whole words (" send " is not in " sender ").
fn words(text: &str) -> String {
    let words = text
        .to_lowercase()
        .chars()
        .map(|character| {
            if character.is_alphanumeric() {
                character
            } else {
                ' '
            }
        })
        .collect::<String>();
    format!(
        " {} ",
        words.split_whitespace().collect::<Vec<_>>().join(" ")
    )
}

/// A control's visible name. A submit button made with `<input>` has its
/// label only in its value.
fn control_label(element: &Value) -> String {
    let name = element.get("name").and_then(Value::as_str).unwrap_or("");
    let button_input = matches!(
        element.get("input_type").and_then(Value::as_str),
        Some("submit" | "button" | "reset" | "image")
    );
    match element.get("value").and_then(Value::as_str) {
        Some(value) if button_input && !value.trim().is_empty() && !name.contains(value) => {
            format!("{name} {value}").trim().to_owned()
        }
        _ => name.to_owned(),
    }
}

fn is_actionable(element: &Value) -> bool {
    element.get("actionable").and_then(Value::as_bool) == Some(true)
        && element.get("disabled").and_then(Value::as_bool) != Some(true)
}

enum TextMatch {
    One(String),
    Many(Vec<Value>),
    None,
}

/// The enabled control whose visible text is `text`: an exact match first,
/// then one whose text contains `text` as whole words. Several matches are
/// returned for the caller to choose among by id, never guessed.
fn match_text(elements: &[Value], text: &str) -> TextMatch {
    let wanted = words(text);
    if wanted.trim().is_empty() {
        return TextMatch::None;
    }
    let pick = |found: Vec<&Value>| match found.as_slice() {
        [] => TextMatch::None,
        [only] => only
            .get("id")
            .and_then(Value::as_str)
            .map_or(TextMatch::None, |id| TextMatch::One(id.to_owned())),
        many => TextMatch::Many(many.iter().map(|element| (*element).clone()).collect()),
    };
    let actionable = elements
        .iter()
        .filter(|element| is_actionable(element))
        .collect::<Vec<_>>();
    match pick(
        actionable
            .iter()
            .copied()
            .filter(|element| words(&control_label(element)) == wanted)
            .collect(),
    ) {
        TextMatch::None => pick(
            actionable
                .into_iter()
                .filter(|element| words(&control_label(element)).contains(&wanted))
                .collect(),
        ),
        found => found,
    }
}

fn element_by_id<'a>(snapshot: &'a Value, id: &str) -> Option<&'a Value> {
    snapshot
        .get("elements")
        .and_then(Value::as_array)?
        .iter()
        .find(|element| element.get("id").and_then(Value::as_str) == Some(id))
}

fn ambiguous(target: &str, matches: &[Value]) -> PokError {
    let listed = matches
        .iter()
        .take(10)
        .map(|element| {
            format!(
                "{} ({} {:?})",
                element.get("id").and_then(Value::as_str).unwrap_or("?"),
                element
                    .get("role")
                    .and_then(Value::as_str)
                    .unwrap_or("control"),
                bounded_label(&control_label(element))
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    PokError::Tool(format!(
        "{target:?} matches {} controls; pass the id of the one you mean as target: {listed}",
        matches.len()
    ))
}

/// Resolve a target given as an element id from the latest page read, a
/// control's visible text, or a CSS selector, to exactly one control.
async fn resolve_target(connection: &mut BrowserConnection, target: &str) -> Result<Resolved> {
    let target = target.trim();
    if target.is_empty() {
        return Err(PokError::Tool("target is empty".into()));
    }
    let pinned = |snapshot: &Value, id: &str, element: &Value| Resolved {
        snapshot_id: snapshot
            .get("snapshot_id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        target_id: id.to_owned(),
        element: element.clone(),
        page_url: snapshot
            .get("url")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
    };
    // An id from the page read the model is acting on keeps that read's
    // check that the control has not changed since.
    if let Some(snapshot) = connection.last_snapshot.as_ref()
        && let Some(element) = element_by_id(snapshot, target)
    {
        return Ok(pinned(snapshot, target, element));
    }
    let snapshot = ManagedBrowser::snapshot_locked(connection).await?;
    let elements = snapshot
        .get("elements")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let exact = elements
        .iter()
        .filter(|element| is_actionable(element) && words(&control_label(element)) == words(target))
        .count();
    let by_text = match match_text(&elements, target) {
        TextMatch::Many(found) => return Err(ambiguous(target, &found)),
        TextMatch::One(id) if exact == 1 => return finish(&snapshot, &id, pinned),
        TextMatch::One(id) => Some(id),
        TextMatch::None => None,
    };
    // A CSS selector, when the text matched nothing exactly.
    let selected = ManagedBrowser::cdp(
        connection,
        "Runtime.evaluate",
        json!({"expression": SELECTOR_SCRIPT.replace("__SELECTOR__", &serde_json::to_string(target)?), "returnByValue": true}),
    )
    .await?;
    let selected = selected
        .pointer("/result/value")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .filter_map(|id| element_by_id(&snapshot, id))
        .cloned()
        .collect::<Vec<_>>();
    match (selected.as_slice(), by_text) {
        ([only], _) => {
            let id = only.get("id").and_then(Value::as_str).unwrap_or_default();
            finish(&snapshot, id, pinned)
        }
        ([], Some(id)) => finish(&snapshot, &id, pinned),
        ([], None) => Err(PokError::Tool(format!(
            "no visible control matches {target:?} by id, text, or CSS selector; read the page with user_browser_extract and pass an element id"
        ))),
        (many, _) => Err(ambiguous(target, many)),
    }
}

fn finish(
    snapshot: &Value,
    id: &str,
    pinned: impl Fn(&Value, &str, &Value) -> Resolved,
) -> Result<Resolved> {
    let element = element_by_id(snapshot, id)
        .ok_or_else(|| PokError::Tool(format!("control {id} is no longer on the page")))?;
    Ok(pinned(snapshot, id, element))
}

/// Press Enter in the focused field. In a visible tab the caller sends a
/// native key press (`native`); a background tab drops those, so the page
/// gets the key events and, unless a handler cancels the key, a single-line
/// field submits its form the way a browser does: by clicking the form's
/// default button, else submitting the form.
const PRESS_ENTER_SCRIPT: &str = r#"(() => {
  const field = document.activeElement;
  if (!field || field === document.body) return {ok: false, reason: 'no field has focus'};
  if (document.visibilityState === 'visible') return {ok: true, native: true};
  const key = {key: 'Enter', code: 'Enter', keyCode: 13, which: 13, bubbles: true, cancelable: true, composed: true};
  const proceed = field.dispatchEvent(new KeyboardEvent('keydown', key));
  if (proceed) field.dispatchEvent(new KeyboardEvent('keypress', key));
  field.dispatchEvent(new KeyboardEvent('keyup', key));
  const form = field.form;
  if (proceed && form && field instanceof HTMLInputElement) {
    const button = form.querySelector('button[type=submit], input[type=submit], button:not([type])');
    if (button && !button.disabled) button.click(); else form.requestSubmit();
  }
  return {ok: true, native: false};
})()"#;

/// The page's whole visible text, spaced like the snapshot's.
const PAGE_TEXT_SCRIPT: &str =
    r#"String(document.body?.innerText || '').replace(/\s+/g, ' ').trim().slice(0, 2000000)"#;

/// Scroll the page and report where it was. A scroll listener (common for
/// infinite feeds) is also told, because a background tab does not
/// deliver scroll events on its own.
const SCROLL_SCRIPT: &str = r#"(() => {
  const direction = __DIRECTION__, amount = __AMOUNT__;
  const from = Math.round(scrollY);
  const height = document.documentElement?.scrollHeight || 0;
  if (amount === 'end') scrollTo(0, direction === 'up' ? 0 : height);
  else scrollBy(0, (direction === 'up' ? -1 : 1) * innerHeight * (amount === 'page' ? 0.85 : 0.35));
  for (const target of [window, document]) target.dispatchEvent(new Event('scroll'));
  return {from, at: Math.round(scrollY), height, width: innerWidth, view: innerHeight, hidden: document.visibilityState !== 'visible'};
})()"#;

/// Put a background tab back at `y` after a capture that rendered it: the
/// whole page is laid out for the capture, so content that loads meanwhile
/// (an infinite feed) can leave it scrolled somewhere else.
async fn restore_scroll(connection: &mut BrowserConnection, y: u64) {
    let _ = ManagedBrowser::cdp(
        connection,
        "Runtime.evaluate",
        json!({"expression": format!("scrollTo(0, {y})"), "returnByValue": true}),
    )
    .await;
}

/// Whether the awaited text (case-insensitive) and element are present.
const WAIT_SCRIPT: &str = r#"(() => {
  const text = __TEXT__, selector = __SELECTOR__;
  const body = String(document.body?.innerText || '');
  let present = true;
  if (text) present = present && body.toLowerCase().includes(String(text).toLowerCase());
  if (selector) {
    let element = null;
    try { element = document.querySelector(selector); } catch (_) { return {present: false, excerpt: 'invalid CSS selector'}; }
    const box = element && element.getBoundingClientRect();
    present = present && !!element && box.width > 0 && box.height > 0;
  }
  return {present, excerpt: body.replace(/\s+/g, ' ').trim().slice(0, 300)};
})()"#;

const SELECTOR_SCRIPT: &str = r#"(() => {
  let nodes;
  try { nodes = Array.from(document.querySelectorAll(__SELECTOR__)); } catch (_) { return []; }
  const ids = globalThis.__pokBrowserIds;
  return nodes.map(node => ids && ids.get(node)).filter(Boolean).slice(0, 50);
})()"#;

/// The HTML of the page or one element, for working out complex forms:
/// scripts, styles, and inline handlers removed, long attributes cut, the
/// values of password, hidden, payment-card, and one-time-code inputs
/// removed, and each control marked with the id the agent acts on
/// (`data-pok-id`). Only a copy is changed; the user's page is untouched.
const EXTRACT_SCRIPT: &str = r#"(() => {
  const selector = __SELECTOR__;
  const limit = __LIMIT__;
  let root;
  try { root = selector ? document.querySelector(selector) : document.documentElement; } catch (_) { root = null; }
  if (!root) return {found: false};
  const ids = globalThis.__pokBrowserIds;
  const copy = root.cloneNode(true);
  const originals = [root, ...root.querySelectorAll('*')];
  const copies = [copy, ...copy.querySelectorAll('*')];
  originals.forEach((element, index) => {
    const id = ids && ids.get(element);
    if (id && copies[index]) copies[index].setAttribute('data-pok-id', id);
  });
  copy.querySelectorAll('script,style,noscript,template,link,meta').forEach(node => node.remove());
  copy.querySelectorAll('svg').forEach(node => node.replaceChildren());
  for (const element of [copy, ...copy.querySelectorAll('*')]) {
    for (const attribute of Array.from(element.attributes)) {
      const name = attribute.name.toLowerCase();
      if (name === 'style' || name.startsWith('on')) element.removeAttribute(attribute.name);
      else if (attribute.value.length > 200) element.setAttribute(attribute.name, attribute.value.slice(0, 200) + '…');
    }
    if (element.tagName === 'INPUT') {
      const type = String(element.getAttribute('type') || '').toLowerCase();
      const autocomplete = String(element.getAttribute('autocomplete') || '').toLowerCase();
      if (type === 'password' || type === 'hidden' || autocomplete.includes('password') || autocomplete.startsWith('cc-') || autocomplete.includes(' cc-') || autocomplete === 'one-time-code') element.removeAttribute('value');
    }
  }
  const html = copy.outerHTML.replace(/\s+/g, ' ');
  return {
    found: true,
    html: html.slice(0, limit),
    html_chars: html.length,
    html_truncated: html.length > limit,
    text: String(root.innerText || '').replace(/\s+/g, ' ').trim().slice(0, 16000),
  };
})()"#;

/// The screenshot as PNG, scaled down to `max_edge` like desktop captures.
fn bounded_png(data: &str, max_edge: u32) -> Result<(String, u32, u32)> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(data)
        .map_err(|error| PokError::Tool(format!("invalid screenshot data: {error}")))?;
    let image = image::load_from_memory_with_format(&bytes, image::ImageFormat::Png)
        .map_err(|error| PokError::Tool(format!("invalid screenshot image: {error}")))?
        .to_rgba8();
    let (width, height) = image.dimensions();
    let max_edge = max_edge.max(1);
    let longest = width.max(height).max(1);
    if longest <= max_edge {
        return Ok((data.to_owned(), width, height));
    }
    let scaled = |length: u32| {
        ((u64::from(length) * u64::from(max_edge) / u64::from(longest)) as u32).max(1)
    };
    let resized = image::imageops::resize(
        &image,
        scaled(width),
        scaled(height),
        image::imageops::FilterType::Triangle,
    );
    let mut cursor = Cursor::new(Vec::new());
    let (width, height) = resized.dimensions();
    image::DynamicImage::ImageRgba8(resized)
        .write_to(&mut cursor, image::ImageFormat::Png)
        .map_err(|error| PokError::Tool(format!("failed to encode screenshot: {error}")))?;
    Ok((
        base64::engine::general_purpose::STANDARD.encode(cursor.into_inner()),
        width,
        height,
    ))
}

/// System 2's permission gate: what an action in the user's own browser
/// would commit, judged from the control's visible label, its role, and the
/// page address. Flagged actions wait for the user's confirmation.
pub(crate) mod guardrail {
    use serde_json::Value;

    use super::{control_label, words};

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) enum Consequence {
        Destructive,
        Purchase,
        SendOrSubmit,
        AccountChange,
    }

    impl Consequence {
        pub(crate) fn key(self) -> &'static str {
            match self {
                Self::Destructive => "destructive",
                Self::Purchase => "purchase",
                Self::SendOrSubmit => "send",
                Self::AccountChange => "account",
            }
        }

        pub(crate) fn describe(self) -> &'static str {
            match self {
                Self::Destructive => "deletes or removes something",
                Self::Purchase => "buys, pays, or subscribes",
                Self::SendOrSubmit => "sends a message, posts, or submits a form",
                Self::AccountChange => "changes account settings or signs out",
            }
        }
    }

    const PURCHASE: &[&str] = &[
        "buy",
        "purchase",
        "pay",
        "place order",
        "place your order",
        "complete order",
        "confirm order",
        "submit order",
        "confirm and pay",
        "subscribe",
        "upgrade",
        "donate",
        "book now",
        "reserve",
        "rent",
        "start trial",
        "start free trial",
        "add payment",
        "send money",
        "transfer",
    ];
    const DESTRUCTIVE: &[&str] = &[
        "delete",
        "remove",
        "discard",
        "erase",
        "trash",
        "destroy",
        "wipe",
        "purge",
        "permanently",
        "deactivate",
        "terminate",
        "revoke",
        "unsubscribe",
        "cancel subscription",
        "cancel order",
        "cancel membership",
        "cancel plan",
        "cancel booking",
        "cancel reservation",
        "close account",
        "clear history",
        "clear all",
        "clear data",
        "reset",
        "block",
        "unfriend",
        "leave group",
        "leave channel",
        "leave conversation",
    ];
    const SEND: &[&str] = &[
        "send", "post", "publish", "tweet", "retweet", "repost", "reply", "comment", "share",
        "submit", "invite", "confirm",
    ];
    const ACCOUNT: &[&str] = &[
        "sign out",
        "signout",
        "log out",
        "logout",
        "change password",
        "update password",
        "reset password",
        "change email",
        "save",
        "save changes",
    ];
    /// On a settings page these commit a change.
    const SETTINGS_COMMIT: &[&str] = &[
        "save", "update", "apply", "change", "confirm", "enable", "disable", "turn on", "turn off",
        "set",
    ];
    /// On a checkout or payment page these complete the purchase.
    const CHECKOUT_COMMIT: &[&str] = &[
        "continue", "next", "submit", "confirm", "complete", "finish", "place", "proceed",
    ];
    const SETTINGS_PAGES: &[&str] = &[
        "settings",
        "setting",
        "account",
        "accounts",
        "myaccount",
        "preferences",
        "prefs",
        "security",
        "privacy",
        "billing",
        "password",
        "profile",
    ];
    const CHECKOUT_PAGES: &[&str] = &["checkout", "payment", "payments", "pay", "purchase"];
    const SIGN_OUT_PAGES: &[&str] = &["logout", "signout", "log out", "sign out", "logoff"];

    fn first(text: &str, phrases: &[&str]) -> Option<String> {
        phrases
            .iter()
            .find(|phrase| text.contains(&format!(" {phrase} ")))
            .map(|phrase| (*phrase).to_owned())
    }

    /// The words of a control's label. A long label is page content (an
    /// email subject, an article link), and only its opening words say what
    /// activating it does.
    fn label_words(element: &Value) -> (String, String) {
        let label = control_label(element);
        let all = words(&label);
        let count = all.split_whitespace().count();
        let text = if count > 10 {
            format!(
                " {} ",
                all.split_whitespace().take(6).collect::<Vec<_>>().join(" ")
            )
        } else {
            all
        };
        (crate::browser::bounded_label(&label), text)
    }

    /// The words of a page address's host and path (not its query).
    fn page_words(url: &str) -> String {
        reqwest::Url::parse(url).map_or_else(
            |_| words(url),
            |url| words(&format!("{} {}", url.host_str().unwrap_or(""), url.path())),
        )
    }

    fn evidence(element: &Value, label: &str, matched: &str) -> String {
        let role = element
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("control");
        format!("{role} {label:?} (matched \"{matched}\")")
    }

    pub(crate) fn click(element: &Value, page_url: &str) -> Option<(Consequence, String)> {
        let (label, text) = label_words(element);
        for (consequence, phrases) in [
            (Consequence::Purchase, PURCHASE),
            (Consequence::Destructive, DESTRUCTIVE),
            (Consequence::SendOrSubmit, SEND),
            (Consequence::AccountChange, ACCOUNT),
        ] {
            if let Some(matched) = first(&text, phrases) {
                return Some((consequence, evidence(element, &label, &matched)));
            }
        }
        let page = page_words(page_url);
        if first(&page, CHECKOUT_PAGES).is_some()
            && let Some(matched) = first(&text, CHECKOUT_COMMIT)
        {
            return Some((
                Consequence::Purchase,
                format!("{} on a checkout page", evidence(element, &label, &matched)),
            ));
        }
        if first(&page, SETTINGS_PAGES).is_some() {
            if let Some(matched) = first(&text, SETTINGS_COMMIT) {
                return Some((
                    Consequence::AccountChange,
                    format!("{} on a settings page", evidence(element, &label, &matched)),
                ));
            }
            // A switch or check box on a settings page applies at once.
            let toggle = matches!(
                element.get("role").and_then(Value::as_str),
                Some("switch" | "checkbox" | "radio" | "menuitemcheckbox" | "menuitemradio")
            ) || matches!(
                element.get("input_type").and_then(Value::as_str),
                Some("checkbox" | "radio")
            );
            if toggle {
                return Some((
                    Consequence::AccountChange,
                    format!("toggle {label:?} on a settings page"),
                ));
            }
        }
        None
    }

    /// Pressing Enter in a field submits its form; a search only searches.
    pub(crate) fn submit(element: &Value, page_url: &str) -> Option<(Consequence, String)> {
        let (label, text) = label_words(element);
        let search = element.get("input_type").and_then(Value::as_str) == Some("search")
            || element.get("role").and_then(Value::as_str) == Some("searchbox")
            || element.get("search").and_then(Value::as_bool) == Some(true)
            || first(&text, &["search", "find", "filter", "look up", "lookup"]).is_some();
        if search {
            return None;
        }
        let consequence = if first(&page_words(page_url), CHECKOUT_PAGES).is_some() {
            Consequence::Purchase
        } else {
            Consequence::SendOrSubmit
        };
        Some((
            consequence,
            format!("pressing Enter in the field {label:?} submits it"),
        ))
    }

    /// A list on a settings page usually applies its choice at once.
    pub(crate) fn select(element: &Value, page_url: &str) -> Option<(Consequence, String)> {
        let (label, _) = label_words(element);
        first(&page_words(page_url), SETTINGS_PAGES).map(|_| {
            (
                Consequence::AccountChange,
                format!("list {label:?} on a settings page"),
            )
        })
    }

    /// Opening a sign-out address ends the user's session.
    pub(crate) fn navigation(url: &str) -> Option<(Consequence, String)> {
        let address = reqwest::Url::parse(url).map_or_else(
            |_| words(url),
            |url| words(&format!("{} {}", url.path(), url.query().unwrap_or(""))),
        );
        first(&address, SIGN_OUT_PAGES).map(|matched| {
            (
                Consequence::AccountChange,
                format!("the address {url:?} (matched \"{matched}\")"),
            )
        })
    }
}

/// Ask the user before a flagged action. "Allow for this session" covers the
/// same kind of action on the same site only. Exams never reach the user's
/// browser at all.
async fn confirm(
    context: &ToolContext,
    tool: &str,
    request: Value,
    consequence: guardrail::Consequence,
    evidence: &str,
    page_url: &str,
) -> Result<()> {
    let site = reqwest::Url::parse(page_url)
        .ok()
        .and_then(|url| url.host_str().map(str::to_owned))
        .unwrap_or_else(|| page_url.to_owned());
    let key = format!("user_browser:{}:{site}", consequence.key());
    if context.approval_cache.lock().contains(&key) {
        return Ok(());
    }
    let reason = format!(
        "This {} in your own signed-in browser: {evidence} on {site}. Allow it only if you asked for it. Allowing it for this session covers only this kind of action on {site}.",
        consequence.describe()
    );
    let mut shown = request;
    shown["page"] = json!(page_url);
    shown["control"] = json!(evidence);
    let decision = context.approvals.approve(tool, &shown, &reason).await?;
    if crate::tool::apply_approval_decision(decision, Some(&key), &context.approval_cache) {
        Ok(())
    } else {
        Err(PokError::PolicyDenied {
            tool: tool.into(),
            reason: format!(
                "the user declined this action, which {} ({evidence}). Do not retry it or do it another way; tell the user it was not done.",
                consequence.describe()
            ),
        })
    }
}

fn allowed(context: &ToolContext, tool: &str) -> Result<()> {
    if matches!(context.policy.mode, PolicyMode::Exam { .. }) {
        return Err(PokError::PolicyDenied {
            tool: tool.into(),
            reason: "the user's own browser is outside every exam sandbox".into(),
        });
    }
    Ok(())
}

/// System 1 candidates: clicks on the visible controls of the agent's tab,
/// in the same form as managed-browser candidates. Controls the guardrail
/// flags get the commit id prefix, so `fast_actions` withholds them and
/// hands a named one back to System 2 (`needs_confirmation`).
pub async fn decision_candidates(
    context: &ToolContext,
    task: &str,
    current_step: &str,
    limit: usize,
) -> Vec<DecisionCandidate> {
    let Some(browser) = active() else {
        return Vec::new();
    };
    let state = browser.state(context.session_id);
    let state = state.lock().await;
    let Some(snapshot) = agent_snapshot(&state) else {
        return Vec::new();
    };
    click_candidates(snapshot, &format!("{task} {current_step}"), limit)
}

/// Whether a System 1 candidate needs the user's confirmation and so must be
/// performed by System 2, never by System 1.
pub fn needs_confirmation(candidate: &DecisionCandidate) -> bool {
    candidate.tool == "user_browser_click" && candidate.id.starts_with(COMMIT_CANDIDATE_PREFIX)
}

/// The bounded page state System 1 checks conditions against: the agent's
/// tab only, with sensitive fields withheld.
pub async fn decision_state(context: &ToolContext) -> Value {
    let Some(browser) = active() else {
        return json!({});
    };
    let state = browser.state(context.session_id);
    let state = state.lock().await;
    agent_snapshot(&state).map_or_else(|| json!({}), snapshot_decision_state)
}

fn agent_snapshot(state: &UserBrowserState) -> Option<&Value> {
    state
        .connection
        .as_ref()
        .filter(|connection| connection.page_target.is_some())
        .and_then(|connection| connection.last_snapshot.as_ref())
}

fn click_candidates(snapshot: &Value, request: &str, limit: usize) -> Vec<DecisionCandidate> {
    let query = request.to_ascii_lowercase();
    let terms = query
        .split(|character: char| !character.is_alphanumeric())
        .filter(|term| term.len() > 2)
        .collect::<Vec<_>>();
    let page_url = snapshot.get("url").and_then(Value::as_str).unwrap_or("");
    let mut candidates = snapshot
        .get("elements")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|element| {
            is_actionable(element)
                && element.get("sensitive").and_then(Value::as_bool) != Some(true)
        })
        .filter_map(|element| {
            let id = element.get("id")?.as_str()?;
            let label = control_label(element);
            let role = element
                .get("role")
                .and_then(Value::as_str)
                .unwrap_or("control");
            let normalized = label.to_ascii_lowercase();
            let matches = terms
                .iter()
                .filter(|term| normalized.contains(**term))
                .count();
            if matches == 0 && !terms.is_empty() {
                return None;
            }
            let flagged = guardrail::click(element, page_url);
            let prefix = if flagged.is_some() {
                COMMIT_CANDIDATE_PREFIX
            } else {
                CLICK_CANDIDATE_PREFIX
            };
            Some(DecisionCandidate {
                id: format!("{prefix}{id}"),
                tool: "user_browser_click".into(),
                arguments: json!({"target": id}),
                // The managed-browser description form the decision models
                // were validated on.
                description: format!("{:?} (browser {role}, visible)", bounded_label(&label)),
                kind: DecisionCandidateKind::Action,
                local_score: (0.35 + matches as f64 * 0.2).min(1.0),
            })
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| right.local_score.total_cmp(&left.local_score));
    candidates.truncate(limit.min(250));
    candidates
}

fn schema<T: JsonSchema>() -> Value {
    serde_json::to_value(schema_for!(T)).expect("schema serializes")
}

#[derive(Debug, Deserialize, JsonSchema)]
struct NavigateArgs {
    /// Absolute http(s) URL.
    url: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ClickArgs {
    /// An element id from the latest page read (e.g. "b12"), the control's
    /// exact visible text, or a CSS selector.
    target: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct TypeArgs {
    /// An element id from the latest page read (e.g. "b12"), the field's
    /// visible label, or a CSS selector.
    target: String,
    /// The exact text; it replaces what the field holds. Dates use the
    /// field's format, e.g. 2026-10-10.
    text: String,
    /// Press Enter afterwards, e.g. to run a search.
    #[serde(default)]
    submit: bool,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SelectArgs {
    /// An element id from the latest page read (e.g. "b12"), the list's
    /// visible label, or a CSS selector.
    target: String,
    /// The option's value or visible label.
    option: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ExtractArgs {
    /// A CSS selector for one part of the page: its text is returned as
    /// section_text (and its HTML with include_html).
    #[serde(default)]
    selector: Option<String>,
    /// Also return the HTML (scripts and styles removed; each control marked
    /// with its id as data-pok-id), for working out complex forms.
    #[serde(default)]
    include_html: bool,
    /// Read the page text from this character on (a previous result's
    /// next_offset), for long pages.
    #[serde(default)]
    offset: Option<usize>,
    /// Characters of page text to return when reading by offset (default
    /// 12000, at most 40000).
    #[serde(default)]
    max_chars: Option<usize>,
}

#[derive(Debug, Clone, Copy, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum ScrollDirection {
    Up,
    Down,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum ScrollAmount {
    /// About a third of the window.
    Small,
    /// Most of the window.
    #[default]
    Page,
    /// The top or bottom of the page.
    End,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ScrollArgs {
    direction: ScrollDirection,
    #[serde(default)]
    amount: ScrollAmount,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct WaitArgs {
    /// Text to wait for anywhere on the page (case-insensitive).
    #[serde(default)]
    text: Option<String>,
    /// A CSS selector for an element to wait for (it must be visible).
    #[serde(default)]
    selector: Option<String>,
    /// Wait for the text or element to go away instead (a spinner, "Loading").
    #[serde(default)]
    gone: bool,
    /// Seconds to wait (default 10, at most 30).
    #[serde(default)]
    timeout_seconds: Option<u64>,
}

#[derive(Debug, Clone, Copy, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum TabAction {
    List,
    Switch,
    Close,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct TabsArgs {
    action: TabAction,
    /// The tab for switch or close, from a list.
    #[serde(default)]
    tab_id: Option<String>,
}

struct StatusTool(Arc<UserBrowser>);
struct NavigateTool(Arc<UserBrowser>);
struct ClickTool(Arc<UserBrowser>);
struct TypeTool(Arc<UserBrowser>);
struct SelectTool(Arc<UserBrowser>);
struct ExtractTool(Arc<UserBrowser>);
struct ScrollTool(Arc<UserBrowser>);
struct WaitForTool(Arc<UserBrowser>);
struct ScreenshotTool(Arc<UserBrowser>);
struct TabsTool(Arc<UserBrowser>);

#[async_trait]
impl Tool for StatusTool {
    fn name(&self) -> &'static str {
        "user_browser_status"
    }
    fn description(&self) -> &'static str {
        "Check whether the user's own Chrome or Edge (their signed-in browser) is reachable on its DevTools port, and list its tabs. When it is not, the result has the exact command for the user to start it; relay it and wait."
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object","properties":{}})
    }
    fn risk(&self) -> RiskClass {
        RiskClass::ReadOnly
    }
    async fn execute(&self, _arguments: Value, context: &ToolContext) -> Result<Value> {
        allowed(context, self.name())?;
        Ok(self.0.status(context).await)
    }
}

#[async_trait]
impl Tool for NavigateTool {
    fn name(&self) -> &'static str {
        "user_browser_navigate"
    }
    fn description(&self) -> &'static str {
        "Open an absolute http(s) URL in the user's own signed-in browser, where they are already logged in (their cookies, SSO, and 2FA sessions), in a background tab the agent owns, and return the page as structured controls and text. Use the user_browser tools when a task needs the user's own accounts or they ask for their browser; for anonymous browsing use managed_browser_open. The user's mouse, keyboard, and tabs are left alone."
    }
    fn input_schema(&self) -> Value {
        schema::<NavigateArgs>()
    }
    fn risk(&self) -> RiskClass {
        RiskClass::DesktopInput
    }
    async fn execute(&self, arguments: Value, context: &ToolContext) -> Result<Value> {
        allowed(context, self.name())?;
        let args: NavigateArgs = serde_json::from_value(arguments)?;
        let result = self.0.navigate(context, &args.url).await;
        self.0.note_form(context).await;
        result
    }
}

#[async_trait]
impl Tool for ClickTool {
    fn name(&self) -> &'static str {
        "user_browser_click"
    }
    fn description(&self) -> &'static str {
        "Click one control (button, link, check box, tab, menu item) in the agent's tab of the user's browser and return the settled page. target is an element id from the latest page read (\"b12\"), the control's exact visible text, or a CSS selector; text that matches several controls is refused with their ids. Deleting, buying, sending, posting, and account changes wait for the user's confirmation."
    }
    fn input_schema(&self) -> Value {
        schema::<ClickArgs>()
    }
    fn risk(&self) -> RiskClass {
        RiskClass::DesktopInput
    }
    async fn execute(&self, arguments: Value, context: &ToolContext) -> Result<Value> {
        allowed(context, self.name())?;
        let args: ClickArgs = serde_json::from_value(arguments)?;
        let result = self
            .0
            .act(context, self.name(), &args.target, Action::Click)
            .await;
        self.0.note_form(context).await;
        result
    }
}

#[async_trait]
impl Tool for TypeTool {
    fn name(&self) -> &'static str {
        "user_browser_type"
    }
    fn description(&self) -> &'static str {
        "Fill one field (text, search, email, number, text area, editable area, date or time) in the agent's tab of the user's browser with exactly this text, replacing what it holds; with submit, press Enter afterwards. Password fields are never filled. Submitting anything other than a search waits for the user's confirmation."
    }
    fn input_schema(&self) -> Value {
        schema::<TypeArgs>()
    }
    fn risk(&self) -> RiskClass {
        RiskClass::DesktopInput
    }
    async fn execute(&self, arguments: Value, context: &ToolContext) -> Result<Value> {
        allowed(context, self.name())?;
        let args: TypeArgs = serde_json::from_value(arguments)?;
        let result = self
            .0
            .act(
                context,
                self.name(),
                &args.target,
                Action::Type {
                    text: &args.text,
                    submit: args.submit,
                },
            )
            .await;
        self.0.note_form(context).await;
        result
    }
}

#[async_trait]
impl Tool for SelectTool {
    fn name(&self) -> &'static str {
        "user_browser_select"
    }
    fn description(&self) -> &'static str {
        "Choose an option, by its value or visible label, in a dropdown list (select) in the agent's tab of the user's browser. For a custom menu that is not a select, click it and then click the option."
    }
    fn input_schema(&self) -> Value {
        schema::<SelectArgs>()
    }
    fn risk(&self) -> RiskClass {
        RiskClass::DesktopInput
    }
    async fn execute(&self, arguments: Value, context: &ToolContext) -> Result<Value> {
        allowed(context, self.name())?;
        let args: SelectArgs = serde_json::from_value(arguments)?;
        let result = self
            .0
            .act(
                context,
                self.name(),
                &args.target,
                Action::Select {
                    option: &args.option,
                },
            )
            .await;
        self.0.note_form(context).await;
        result
    }
}

#[async_trait]
impl Tool for ExtractTool {
    fn name(&self) -> &'static str {
        "user_browser_extract"
    }
    fn description(&self) -> &'static str {
        "Read the agent's tab of the user's browser: URL, title, visible text, and its controls with the ids to act on. With selector, also that part's text; with include_html, its HTML (each control marked with its id as data-pok-id), for working out complex forms."
    }
    fn input_schema(&self) -> Value {
        schema::<ExtractArgs>()
    }
    fn risk(&self) -> RiskClass {
        RiskClass::ReadOnly
    }
    async fn execute(&self, arguments: Value, context: &ToolContext) -> Result<Value> {
        allowed(context, self.name())?;
        let args: ExtractArgs = serde_json::from_value(arguments)?;
        let result = self
            .0
            .extract(
                context,
                args.selector.as_deref(),
                args.include_html,
                (args.offset.is_some() || args.max_chars.is_some()).then(|| {
                    (
                        args.offset.unwrap_or(0),
                        args.max_chars.unwrap_or(12_000).clamp(500, 40_000),
                    )
                }),
            )
            .await;
        self.0.note_form(context).await;
        result
    }
}

#[async_trait]
impl Tool for ScreenshotTool {
    fn name(&self) -> &'static str {
        "user_browser_screenshot"
    }
    fn description(&self) -> &'static str {
        "Capture the visible part of the agent's tab of the user's browser as an image, even while that tab is in the background. Use it for layouts, images, or canvas content the page text does not describe."
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object","properties":{}})
    }
    fn risk(&self) -> RiskClass {
        RiskClass::ReadOnly
    }
    async fn execute(&self, _arguments: Value, context: &ToolContext) -> Result<Value> {
        allowed(context, self.name())?;
        self.0.screenshot(context).await
    }
}

#[async_trait]
impl Tool for TabsTool {
    fn name(&self) -> &'static str {
        "user_browser_tabs"
    }
    fn description(&self) -> &'static str {
        "List the tabs open in the user's browser, switch the agent to one of them (to read or act on a page the user already has open; it is not brought to the front), or close a tab the agent opened. The user's own tabs are never closed."
    }
    fn input_schema(&self) -> Value {
        schema::<TabsArgs>()
    }
    fn risk(&self) -> RiskClass {
        RiskClass::DesktopInput
    }
    fn risk_for_arguments(&self, arguments: &Value) -> RiskClass {
        if arguments.get("action").and_then(Value::as_str) == Some("list") {
            RiskClass::ReadOnly
        } else {
            RiskClass::DesktopInput
        }
    }
    async fn execute(&self, arguments: Value, context: &ToolContext) -> Result<Value> {
        allowed(context, self.name())?;
        let args: TabsArgs = serde_json::from_value(arguments)?;
        let result = self
            .0
            .tabs_action(context, args.action, args.tab_id.as_deref())
            .await;
        self.0.note_form(context).await;
        result
    }
}

#[async_trait]
impl Tool for ScrollTool {
    fn name(&self) -> &'static str {
        "user_browser_scroll"
    }
    fn description(&self) -> &'static str {
        "Scroll the agent's tab of the user's browser up or down (small, page, or to the end) so more of a feed or list loads, and return the settled page with where it scrolled and whether more content loaded. Works in the background tab."
    }
    fn input_schema(&self) -> Value {
        schema::<ScrollArgs>()
    }
    fn risk(&self) -> RiskClass {
        RiskClass::DesktopInput
    }
    async fn execute(&self, arguments: Value, context: &ToolContext) -> Result<Value> {
        allowed(context, self.name())?;
        let args: ScrollArgs = serde_json::from_value(arguments)?;
        let direction = match args.direction {
            ScrollDirection::Up => "up",
            ScrollDirection::Down => "down",
        };
        let amount = match args.amount {
            ScrollAmount::Small => "small",
            ScrollAmount::Page => "page",
            ScrollAmount::End => "end",
        };
        self.0.scroll(context, direction, amount).await
    }
}

#[async_trait]
impl Tool for WaitForTool {
    fn name(&self) -> &'static str {
        "user_browser_wait_for"
    }
    fn description(&self) -> &'static str {
        "Wait until text or an element (CSS selector) appears, or with gone, disappears, in the agent's tab of the user's browser, then return the page. Use it after an action that loads results or shows a spinner, instead of reading the page repeatedly."
    }
    fn input_schema(&self) -> Value {
        schema::<WaitArgs>()
    }
    fn risk(&self) -> RiskClass {
        RiskClass::ReadOnly
    }
    async fn execute(&self, arguments: Value, context: &ToolContext) -> Result<Value> {
        allowed(context, self.name())?;
        let args: WaitArgs = serde_json::from_value(arguments)?;
        let timeout = Duration::from_secs(args.timeout_seconds.unwrap_or(10).clamp(1, 30));
        self.0
            .wait_for(
                context,
                args.text.as_deref().filter(|text| !text.trim().is_empty()),
                args.selector
                    .as_deref()
                    .filter(|selector| !selector.trim().is_empty()),
                args.gone,
                timeout,
            )
            .await
    }
}

impl UserBrowser {
    /// Record the form in the agent's tab as the page now shows it.
    async fn note_form(&self, context: &ToolContext) {
        let state = self.state(context.session_id);
        let mut state = state.lock().await;
        let Some((tab, form)) = state.connection.as_ref().and_then(|connection| {
            Some((
                connection.page_target.clone()?,
                connection.last_snapshot.as_ref().map(form_page),
            ))
        }) else {
            return;
        };
        match form.flatten() {
            Some(form) => {
                state.forms.insert(tab, form);
            }
            None => {
                state.forms.remove(&tab);
            }
        }
    }
}

/// The form on a page read: its fields to fill (not search boxes, buttons,
/// radio buttons, or password fields), when there are at least two.
fn form_page(snapshot: &Value) -> Option<FormPage> {
    let text = |value: &Value, key: &str| {
        value
            .get(key)
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_owned()
    };
    let fields = snapshot
        .get("elements")?
        .as_array()?
        .iter()
        .filter_map(|element| {
            let input_type = element
                .get("input_type")
                .and_then(Value::as_str)
                .unwrap_or("text");
            if !matches!(
                element.get("tag").and_then(Value::as_str),
                Some("input" | "textarea" | "select")
            ) || matches!(
                input_type,
                "hidden"
                    | "submit"
                    | "button"
                    | "reset"
                    | "image"
                    | "file"
                    | "radio"
                    | "search"
                    | "password"
            ) || element.get("search").and_then(Value::as_bool) == Some(true)
                || element.get("sensitive").and_then(Value::as_bool) == Some(true)
            {
                return None;
            }
            // A list's accessible name ends with its options' labels.
            let options = element
                .get("options")
                .and_then(Value::as_array)
                .map(|options| {
                    options
                        .iter()
                        .filter_map(|option| option.get("label")?.as_str())
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                .unwrap_or_default();
            let name = text(element, "name");
            let label = name
                .strip_suffix(options.as_str())
                .filter(|_| !options.is_empty())
                .unwrap_or(&name)
                .trim()
                .chars()
                .take(80)
                .collect::<String>();
            if label.is_empty() {
                return None;
            }
            let check_box = input_type == "checkbox";
            let filled = if check_box {
                element.get("checked").and_then(Value::as_bool) == Some(true)
            } else {
                !text(element, "value").is_empty()
            };
            Some(FormField {
                label,
                filled,
                check_box,
            })
        })
        .collect::<Vec<_>>();
    (fields.len() >= 2).then(|| FormPage {
        url: text(snapshot, "url"),
        title: text(snapshot, "title"),
        fields,
    })
}

/// What a form holds, in a line: what is filled, and by name what is not.
fn describe_form(form: &FormPage) -> String {
    let (check_boxes, entries): (Vec<_>, Vec<_>) =
        form.fields.iter().partition(|field| field.check_box);
    let filled = entries.iter().filter(|field| field.filled).count();
    let empty = entries
        .iter()
        .filter(|field| !field.filled)
        .map(|field| field.label.as_str())
        .collect::<Vec<_>>()
        .join("; ");
    let mut line = if empty.is_empty() {
        format!("all {} fields filled", entries.len())
    } else if filled == 0 {
        format!("not started; empty: {empty}")
    } else {
        format!(
            "{filled} of {} fields filled; empty: {empty}",
            entries.len()
        )
    };
    let unchecked = check_boxes
        .iter()
        .filter(|field| !field.filled)
        .map(|field| field.label.as_str())
        .collect::<Vec<_>>();
    if !unchecked.is_empty() {
        line.push_str(&format!("; unchecked: {}", unchecked.join("; ")));
    }
    line
}

/// The forms in the agent's tabs, one line each, while it is filling forms.
/// `unfinished` keeps only forms with empty fields that it filled part of or
/// opened itself.
async fn form_lines(session: Uuid, unfinished: bool) -> Vec<String> {
    let Some(browser) = active() else {
        return Vec::new();
    };
    let state = browser.state(session);
    let state = state.lock().await;
    if state.filled_tabs.is_empty() {
        return Vec::new();
    }
    let here = state
        .connection
        .as_ref()
        .and_then(|connection| connection.page_target.as_deref());
    let mut forms = state
        .forms
        .iter()
        .filter(|(tab, form)| {
            !unfinished
                || ((state.filled_tabs.contains_key(*tab) || state.owned_tabs.contains(*tab))
                    && form
                        .fields
                        .iter()
                        .any(|field| !field.check_box && !field.filled))
        })
        .collect::<Vec<_>>();
    forms.sort_by(|left, right| left.1.url.cmp(&right.1.url));
    forms
        .into_iter()
        .map(|(tab, form)| {
            format!(
                "- \"{}\" ({}, tab_id {tab}{}): {}",
                form.title,
                form.url,
                if here == Some(tab.as_str()) {
                    ", you are here"
                } else {
                    ""
                },
                describe_form(form)
            )
        })
        .collect()
}

/// What the forms in the agent's tabs hold, read from the pages, for the turn
/// reminder while it fills forms: weaker models otherwise lose track of
/// which tab still needs what, switch back and forth, and report forms done
/// that are not.
pub(crate) async fn form_notes(session: Uuid) -> Option<String> {
    let lines = form_lines(session, false).await;
    (!lines.is_empty()).then(|| {
        format!(
            "FORMS, as the pages show them (nothing sent):\n{}",
            lines.join("\n")
        )
    })
}

/// The forms still missing entries, checked before the agent reports the
/// task finished.
pub(crate) async fn unfinished_forms(session: Uuid) -> Option<String> {
    let lines = form_lines(session, true).await;
    (!lines.is_empty()).then(|| lines.join("\n"))
}

/// The page the main agent last had open in its tab of the user's browser,
/// for swarm workers, which work in tabs of their own.
pub(crate) async fn lead_page() -> Option<String> {
    let leads = STATES
        .get_or_init(Default::default)
        .lock()
        .iter()
        .filter(|((_, lane), _)| lane.is_none())
        .map(|(_, state)| state.clone())
        .collect::<Vec<_>>();
    for state in leads {
        let state = state.lock().await;
        if let Some(url) = state
            .connection
            .as_ref()
            .and_then(|connection| connection.last_snapshot.as_ref())
            .and_then(|snapshot| snapshot.get("url"))
            .and_then(Value::as_str)
            .filter(|url| url.starts_with("http"))
        {
            return Some(url.to_owned());
        }
    }
    None
}

/// Close the tabs a finished swarm worker opened in the user's browser and
/// forget its state. The user's own tabs are never touched.
pub(crate) async fn release_worker(session_id: Uuid) {
    let lanes = {
        let mut states = STATES.get_or_init(Default::default).lock();
        let keys = states
            .keys()
            .filter(|(_, lane)| *lane == Some(session_id))
            .copied()
            .collect::<Vec<_>>();
        keys.into_iter()
            .filter_map(|key| states.remove(&key).map(|state| (key.0, state)))
            .collect::<Vec<_>>()
    };
    for (port, state) in lanes {
        let state = state.lock().await;
        for tab in &state.owned_tabs {
            // A form the worker filled and did not send stays open: the user
            // reviews it there, and submitting it is still their decision.
            if state.filled_tabs.contains_key(tab) {
                continue;
            }
            crate::browser::close_tab(port, tab).await;
        }
    }
}

/// Register the user-browser tools, unless disabled in configuration.
pub fn register_user_browser_tools(
    registry: &mut ToolRegistry,
    config: &UserBrowserConfig,
    data_dir: &Path,
) {
    if !config.enabled {
        return;
    }
    let browser = Arc::new(UserBrowser::new(config, data_dir));
    *ACTIVE.get_or_init(Default::default).lock() = Some(browser.clone());
    registry.register(StatusTool(browser.clone()));
    registry.register(NavigateTool(browser.clone()));
    registry.register(ClickTool(browser.clone()));
    registry.register(TypeTool(browser.clone()));
    registry.register(SelectTool(browser.clone()));
    registry.register(ExtractTool(browser.clone()));
    registry.register(ScreenshotTool(browser.clone()));
    registry.register(ScrollTool(browser.clone()));
    registry.register(WaitForTool(browser.clone()));
    registry.register(TabsTool(browser));
}

/// Make `snapshot` the page in the agent's tab, for System 1 tests. It uses
/// the default port's shared state, so every test that registers the tools
/// sees the same page.
#[cfg(test)]
pub(crate) async fn install_test_page(snapshot: Value) {
    let browser = Arc::new(UserBrowser::new(
        &UserBrowserConfig::default(),
        Path::new("."),
    ));
    let mut connection = BrowserConnection::user(browser.port);
    connection.page_target = Some("agent-tab".into());
    connection.last_snapshot = Some(snapshot);
    browser.state(Uuid::nil()).lock().await.connection = Some(connection);
    *ACTIVE.get_or_init(Default::default).lock() = Some(browser);
}

/// One line for `pok-ai doctor`: whether the user's browser is attachable,
/// and the command to start it when it is not.
pub async fn health_check(config: &UserBrowserConfig, data_dir: &Path) -> String {
    if !config.enabled {
        return "Your browser: attaching is disabled ([user_browser] enabled = false)".into();
    }
    let browser = UserBrowser::new(config, data_dir);
    match browser.version().await {
        Ok(version) => format!(
            "Your browser: attached on port {} ({})",
            config.port,
            version
                .get("Browser")
                .and_then(Value::as_str)
                .unwrap_or("Chromium")
        ),
        Err(reason) => format!(
            "Your browser: not detected on port {} ({reason}). To let the agent use your signed-in browser, start it with:\n  {}",
            config.port,
            browser.launch_command()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::guardrail::Consequence;
    use super::*;

    fn element(id: &str, role: &str, name: &str) -> Value {
        json!({"id": id, "role": role, "name": name, "actionable": true, "disabled": false})
    }

    fn flagged(role: &str, name: &str, url: &str) -> Option<Consequence> {
        guardrail::click(&element("b1", role, name), url).map(|(consequence, _)| consequence)
    }

    #[test]
    fn a_form_is_read_from_the_page_as_filled_and_empty_fields() {
        let page = json!({
            "url": "https://jobs.example/apply",
            "title": "Apply: Platform Engineer",
            "elements": [
                {"tag": "input", "input_type": "search", "name": "Search jobs", "value": ""},
                {"tag": "input", "input_type": "text", "name": "Full name", "value": "Jordan Avery"},
                {"tag": "input", "input_type": "email", "name": "Email", "value": ""},
                {"tag": "select", "name": "Years Choose 0-2 3-5", "value": "",
                 "options": [{"label": "Choose"}, {"label": "0-2"}, {"label": "3-5"}]},
                {"tag": "input", "input_type": "checkbox", "name": "Remote", "checked": false},
                {"tag": "input", "input_type": "password", "name": "Password", "value": ""},
                {"tag": "textarea", "name": "Why us?", "value": "Because"},
                {"tag": "button", "name": "Submit application"}
            ]
        });
        let form = form_page(&page).expect("a form");
        assert_eq!(form.fields.len(), 5);
        assert_eq!(
            describe_form(&form),
            "2 of 4 fields filled; empty: Email; Years; unchecked: Remote"
        );
        // A page with one search box is not a form.
        let search = json!({"url": "u", "title": "t", "elements": [
            {"tag": "input", "input_type": "search", "name": "Search", "value": ""}
        ]});
        assert!(form_page(&search).is_none());
    }

    #[test]
    fn consequential_clicks_are_flagged_by_what_they_commit() {
        let page = "https://example.com/cart";
        assert_eq!(
            flagged("button", "Place your order", page),
            Some(Consequence::Purchase)
        );
        assert_eq!(
            flagged("button", "Buy now", page),
            Some(Consequence::Purchase)
        );
        assert_eq!(
            flagged("button", "Delete conversation", page),
            Some(Consequence::Destructive)
        );
        assert_eq!(
            flagged("button", "Move to Trash", page),
            Some(Consequence::Destructive)
        );
        assert_eq!(
            flagged("button", "Cancel subscription", page),
            Some(Consequence::Destructive)
        );
        assert_eq!(
            flagged("button", "Send", page),
            Some(Consequence::SendOrSubmit)
        );
        assert_eq!(
            flagged("button", "Post", page),
            Some(Consequence::SendOrSubmit)
        );
        assert_eq!(
            flagged("link", "Sign out", page),
            Some(Consequence::AccountChange)
        );
        // A submit button made with <input> is labeled by its value.
        let input = json!({"id": "b2", "role": "textbox", "name": "", "input_type": "submit",
            "value": "Place order", "actionable": true});
        assert_eq!(
            guardrail::click(&input, page).map(|(consequence, _)| consequence),
            Some(Consequence::Purchase)
        );
    }

    #[test]
    fn ordinary_navigation_and_safe_buttons_are_not_flagged() {
        let page = "https://mail.example.com/inbox";
        for (role, name) in [
            ("link", "Inbox"),
            ("link", "Your Orders"),
            ("button", "Cancel"),
            ("button", "Add to cart"),
            ("link", "Sender settings help"),
            ("button", "Next page"),
            ("link", "PayPal"),
            ("link", "Posts"),
        ] {
            assert_eq!(flagged(role, name, page), None, "{name}");
        }
        // A long label is content; words deep inside it do not commit.
        assert_eq!(
            flagged(
                "link",
                "Weekly digest from the team about what we plan to delete or remove next quarter",
                page
            ),
            None
        );
    }

    #[test]
    fn checkout_and_settings_pages_flag_their_commit_buttons() {
        assert_eq!(
            flagged(
                "button",
                "Continue",
                "https://shop.example.com/checkout/review"
            ),
            Some(Consequence::Purchase)
        );
        assert_eq!(
            flagged("button", "Continue", "https://shop.example.com/search"),
            None
        );
        assert_eq!(
            flagged("button", "Apply", "https://example.com/account/settings"),
            Some(Consequence::AccountChange)
        );
        let switch = json!({"id": "b3", "role": "switch", "name": "Two-step verification",
            "actionable": true});
        assert_eq!(
            guardrail::click(&switch, "https://myaccount.example.com/security")
                .map(|(consequence, _)| consequence),
            Some(Consequence::AccountChange)
        );
        assert!(
            guardrail::select(
                &element("b4", "combobox", "Language"),
                "https://example.com/settings"
            )
            .is_some()
        );
        assert!(
            guardrail::select(
                &element("b4", "combobox", "Sort by"),
                "https://shop.example.com/search"
            )
            .is_none()
        );
    }

    #[test]
    fn enter_submits_except_in_a_search_field() {
        let page = "https://mail.example.com/";
        assert!(guardrail::submit(&element("b1", "textbox", "Search mail"), page).is_none());
        let search = json!({"id": "b1", "role": "textbox", "name": "q", "input_type": "search"});
        assert!(guardrail::submit(&search, page).is_none());
        // An unlabeled box the page marks as a search (its form, landmark, or name).
        let unlabeled = json!({"id": "b3", "role": "textbox", "name": "", "input_type": "text", "search": true});
        assert!(guardrail::submit(&unlabeled, "https://duckduckgo.com/").is_none());
        assert_eq!(
            guardrail::submit(&element("b2", "textbox", "Message"), page)
                .map(|(consequence, _)| consequence),
            Some(Consequence::SendOrSubmit)
        );
    }

    #[test]
    fn sign_out_addresses_need_confirmation() {
        assert!(guardrail::navigation("https://example.com/logout").is_some());
        assert!(guardrail::navigation("https://example.com/auth/sign-out?next=/").is_some());
        assert!(guardrail::navigation("https://example.com/inbox").is_none());
    }

    #[test]
    fn text_targets_resolve_to_one_control_or_list_the_choices() {
        let elements = vec![
            element("b1", "button", "Send"),
            element("b2", "link", "Sender settings"),
            element("b3", "button", "Archive"),
            element("b4", "link", "Archive"),
            element("b5", "button", "Reply all"),
            json!({"id": "b6", "role": "button", "name": "Disabled", "actionable": true, "disabled": true}),
        ];
        assert!(matches!(match_text(&elements, "send"), TextMatch::One(id) if id == "b1"));
        assert!(
            matches!(match_text(&elements, "  Reply   all "), TextMatch::One(id) if id == "b5")
        );
        assert!(matches!(match_text(&elements, "all"), TextMatch::One(id) if id == "b5"));
        assert!(
            matches!(match_text(&elements, "Archive"), TextMatch::Many(found) if found.len() == 2)
        );
        assert!(matches!(match_text(&elements, "Disabled"), TextMatch::None));
        assert!(matches!(
            match_text(&elements, "Settings page"),
            TextMatch::None
        ));
    }

    #[test]
    fn system_one_never_gets_consequential_controls_as_ordinary_clicks() {
        let snapshot = json!({
            "url": "https://mail.example.com/compose",
            "elements": [
                element("b1", "button", "Send"),
                element("b2", "link", "Inbox"),
                element("b3", "button", "Discard draft"),
                json!({"id": "b4", "role": "textbox", "name": "Password", "actionable": true, "sensitive": true}),
            ],
        });
        let candidates = click_candidates(&snapshot, "", 30);
        let ids = candidates
            .iter()
            .map(|candidate| candidate.id.as_str())
            .collect::<Vec<_>>();
        assert!(ids.contains(&"user_browser_click_b2"));
        assert!(ids.contains(&"user_browser_commit_b1"));
        assert!(ids.contains(&"user_browser_commit_b3"));
        assert!(!ids.iter().any(|id| id.ends_with("b4")));
        let send = candidates
            .iter()
            .find(|candidate| candidate.id.ends_with("b1"))
            .unwrap();
        assert!(needs_confirmation(send));
        assert_eq!(send.arguments, json!({"target": "b1"}));
        assert_eq!(send.description, "\"Send\" (browser button, visible)");
        // Task words narrow the candidates like the managed browser's.
        let narrowed = click_candidates(&snapshot, "open the inbox", 30);
        assert_eq!(narrowed.len(), 1);
        assert!(!needs_confirmation(&narrowed[0]));
    }

    #[test]
    fn the_launch_command_names_the_port_and_a_separate_profile() {
        let profile = Path::new("/profiles/personal");
        let command = launch_command(9333, profile, &[]);
        assert!(
            command.contains("--remote-debugging-port=9333"),
            "{command}"
        );
        assert!(
            command.contains("--user-data-dir=\"/profiles/personal\""),
            "{command}"
        );
        if cfg!(windows) {
            let edge = PathBuf::from(r"C:\Edge\msedge.exe");
            let chrome = PathBuf::from(r"C:\Chrome\chrome.exe");
            let command = launch_command(9222, profile, &[edge, chrome]);
            assert!(
                command.starts_with(r#"& "C:\Chrome\chrome.exe""#),
                "{command}"
            );
        }
    }

    #[tokio::test]
    async fn a_browser_that_is_not_running_gets_the_launch_command() {
        let port = std::net::TcpListener::bind(("127.0.0.1", 0))
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let data = tempfile::tempdir().unwrap();
        let config = UserBrowserConfig {
            enabled: true,
            port,
            user_data_dir: None,
        };
        let browser = UserBrowser::new(&config, data.path());
        let platform = Arc::new(crate::platform::MockDesktop::default());
        let context = crate::builtins::tests::mock_input_context(platform, &data);
        let status = browser.status(&context).await;
        assert_eq!(status["connected"], false);
        let command = status["launch_command"].as_str().unwrap();
        assert!(command.contains(&format!("--remote-debugging-port={port}")));
        assert!(command.contains("personal"));
        let line = health_check(&config, data.path()).await;
        assert!(line.contains("not detected"), "{line}");
        let error = browser
            .navigate(&context, "https://example.com/")
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("--remote-debugging-port"), "{error}");
        assert!(error.contains("Do not retry"), "{error}");
    }

    struct Recording {
        decision: crate::policy::ApprovalDecision,
        asked: SyncMutex<Vec<(String, Value, String)>>,
    }

    #[async_trait]
    impl crate::policy::ApprovalHandler for Recording {
        async fn approve(
            &self,
            tool: &str,
            arguments: &Value,
            reason: &str,
        ) -> Result<crate::policy::ApprovalDecision> {
            self.asked
                .lock()
                .push((tool.into(), arguments.clone(), reason.into()));
            Ok(self.decision)
        }
    }

    fn approval_context(
        decision: crate::policy::ApprovalDecision,
        data: &tempfile::TempDir,
    ) -> (ToolContext, Arc<Recording>) {
        let platform = Arc::new(crate::platform::MockDesktop::default());
        let mut context = crate::builtins::tests::mock_input_context(platform, data);
        let recording = Arc::new(Recording {
            decision,
            asked: SyncMutex::default(),
        });
        context.approvals = recording.clone();
        // The gate holds even where the policy itself would ask nothing.
        context.policy = crate::policy::Policy::autonomous();
        (context, recording)
    }

    #[tokio::test]
    async fn the_user_confirms_each_kind_of_consequential_action_per_site() {
        let data = tempfile::tempdir().unwrap();
        let (context, recording) =
            approval_context(crate::policy::ApprovalDecision::AllowSession, &data);
        let send = || {
            confirm(
                &context,
                "user_browser_click",
                json!({"target": "b1"}),
                Consequence::SendOrSubmit,
                "button \"Send\"",
                "https://mail.example.com/compose",
            )
        };
        send().await.unwrap();
        send().await.unwrap();
        assert_eq!(recording.asked.lock().len(), 1, "allowed for the session");
        let (tool, shown, reason) = recording.asked.lock()[0].clone();
        assert_eq!(tool, "user_browser_click");
        assert_eq!(shown["page"], "https://mail.example.com/compose");
        assert!(reason.contains("mail.example.com"), "{reason}");
        // Another kind of action, or another site, asks again.
        confirm(
            &context,
            "user_browser_click",
            json!({"target": "b9"}),
            Consequence::Purchase,
            "button \"Buy now\"",
            "https://mail.example.com/store",
        )
        .await
        .unwrap();
        confirm(
            &context,
            "user_browser_click",
            json!({"target": "b1"}),
            Consequence::SendOrSubmit,
            "button \"Send\"",
            "https://chat.example.com/",
        )
        .await
        .unwrap();
        assert_eq!(recording.asked.lock().len(), 3);
    }

    #[tokio::test]
    async fn a_declined_action_is_refused_and_not_remembered() {
        let data = tempfile::tempdir().unwrap();
        let (context, recording) = approval_context(crate::policy::ApprovalDecision::Deny, &data);
        let error = confirm(
            &context,
            "user_browser_click",
            json!({"target": "b1"}),
            Consequence::Destructive,
            "button \"Delete\"",
            "https://mail.example.com/",
        )
        .await
        .unwrap_err();
        assert!(matches!(error, PokError::PolicyDenied { .. }), "{error}");
        assert!(error.to_string().contains("Do not retry"), "{error}");
        assert!(context.approval_cache.lock().is_empty());
        assert_eq!(recording.asked.lock().len(), 1);
    }

    #[tokio::test]
    async fn exams_never_reach_the_users_browser() {
        let data = tempfile::tempdir().unwrap();
        let platform = Arc::new(crate::platform::MockDesktop::default());
        let mut context = crate::builtins::tests::mock_input_context(platform, &data);
        context.policy = crate::policy::Policy {
            mode: PolicyMode::Exam {
                sandbox: data.path().to_path_buf(),
                allowed_window_title: "Exam".into(),
                allowed_process: "exam.exe".into(),
            },
        };
        let mut registry = ToolRegistry::new();
        register_user_browser_tools(&mut registry, &UserBrowserConfig::default(), data.path());
        let error = registry
            .call("user_browser_status", json!({}), &context)
            .await
            .unwrap_err();
        assert!(matches!(error, PokError::PolicyDenied { .. }), "{error}");
    }

    #[test]
    fn disabled_configuration_registers_nothing() {
        let data = tempfile::tempdir().unwrap();
        let mut registry = ToolRegistry::new();
        register_user_browser_tools(
            &mut registry,
            &UserBrowserConfig {
                enabled: false,
                ..UserBrowserConfig::default()
            },
            data.path(),
        );
        assert!(registry.names().is_empty());
    }

    #[test]
    fn screenshots_are_scaled_to_the_vision_edge() {
        let image = image::RgbaImage::from_pixel(2000, 1000, image::Rgba([10, 20, 30, 255]));
        let mut cursor = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(image)
            .write_to(&mut cursor, image::ImageFormat::Png)
            .unwrap();
        let data = base64::engine::general_purpose::STANDARD.encode(cursor.into_inner());
        let (_, width, height) = bounded_png(&data, 1000).unwrap();
        assert_eq!((width, height), (1000, 500));
        let (same, width, _) = bounded_png(&data, 4000).unwrap();
        assert_eq!((same.as_str(), width), (data.as_str(), 2000));
    }

    /// A form the agent filled and did not submit survives opening another
    /// page: set POK_AI_USER_BROWSER_TEST_PORT, POK_AI_USER_BROWSER_TEST_URL
    /// (the test form), and POK_AI_USER_BROWSER_TEST_FEED_URL.
    #[tokio::test]
    #[ignore = "needs a Chromium started with --remote-debugging-port; set POK_AI_USER_BROWSER_TEST_PORT, POK_AI_USER_BROWSER_TEST_URL and POK_AI_USER_BROWSER_TEST_FEED_URL"]
    async fn a_filled_form_is_kept_when_another_page_opens() {
        let port: u16 = std::env::var("POK_AI_USER_BROWSER_TEST_PORT")
            .expect("POK_AI_USER_BROWSER_TEST_PORT")
            .parse()
            .unwrap();
        let form =
            std::env::var("POK_AI_USER_BROWSER_TEST_URL").expect("POK_AI_USER_BROWSER_TEST_URL");
        let feed = std::env::var("POK_AI_USER_BROWSER_TEST_FEED_URL")
            .expect("POK_AI_USER_BROWSER_TEST_FEED_URL");
        let data = tempfile::tempdir().unwrap();
        let (context, _) = approval_context(crate::policy::ApprovalDecision::Deny, &data);
        let config = UserBrowserConfig {
            enabled: true,
            port,
            user_data_dir: None,
        };
        let mut registry = ToolRegistry::new();
        register_user_browser_tools(&mut registry, &config, data.path());
        let call = |name: &'static str, arguments: Value| {
            let registry = &registry;
            let context = &context;
            async move { registry.call(name, arguments, context).await }
        };
        let opened = call("user_browser_navigate", json!({"url": form}))
            .await
            .unwrap();
        let form_tab = opened["tab"]["tab_id"].as_str().unwrap().to_owned();
        // Reading a page and opening another reuses the tab.
        let reused = call("user_browser_navigate", json!({"url": form}))
            .await
            .unwrap();
        assert_eq!(reused["tab"]["tab_id"], form_tab.as_str());
        assert!(reused.get("kept_filled_form").is_none());
        call(
            "user_browser_type",
            json!({"target": "Search mail", "text": "kept value"}),
        )
        .await
        .unwrap();
        // With an unsent value in it, the form's tab stays and the next page
        // opens in a new tab.
        let next = call("user_browser_navigate", json!({"url": feed}))
            .await
            .unwrap();
        assert_eq!(
            next["kept_filled_form"]["tab_id"],
            form_tab.as_str(),
            "{next}"
        );
        assert_ne!(next["tab"]["tab_id"], form_tab.as_str());
        let tabs = call("user_browser_tabs", json!({"action": "list"}))
            .await
            .unwrap();
        assert!(tabs.to_string().contains(&form_tab), "{tabs}");
        // The value is still there.
        call(
            "user_browser_tabs",
            json!({"action": "switch", "tab_id": form_tab}),
        )
        .await
        .unwrap();
        let back = call("user_browser_extract", json!({"include_html": true}))
            .await
            .unwrap();
        assert!(back.to_string().contains("kept value"), "{back}");
        // The turn reminder's record of the form, read from the page.
        let notes = form_notes(context.session_id).await.unwrap_or_default();
        assert!(
            notes.contains(&format!("tab_id {form_tab}, you are here")),
            "{notes}"
        );
        for tab in [form_tab.as_str(), next["tab"]["tab_id"].as_str().unwrap()] {
            let _ = call(
                "user_browser_tabs",
                json!({"action": "close", "tab_id": tab}),
            )
            .await;
        }
        assert!(form_notes(context.session_id).await.is_none());
    }

    /// Scrolling, waiting, and paging in a background tab of a real browser:
    /// set POK_AI_USER_BROWSER_TEST_PORT and POK_AI_USER_BROWSER_TEST_FEED_URL.
    #[tokio::test]
    #[ignore = "needs a Chromium started with --remote-debugging-port; set POK_AI_USER_BROWSER_TEST_PORT and POK_AI_USER_BROWSER_TEST_FEED_URL"]
    async fn scrolls_waits_and_pages_in_a_background_tab() {
        let port: u16 = std::env::var("POK_AI_USER_BROWSER_TEST_PORT")
            .expect("POK_AI_USER_BROWSER_TEST_PORT")
            .parse()
            .unwrap();
        let feed = std::env::var("POK_AI_USER_BROWSER_TEST_FEED_URL")
            .expect("POK_AI_USER_BROWSER_TEST_FEED_URL");
        let data = tempfile::tempdir().unwrap();
        let (context, _) = approval_context(crate::policy::ApprovalDecision::Deny, &data);
        let config = UserBrowserConfig {
            enabled: true,
            port,
            user_data_dir: None,
        };
        let mut registry = ToolRegistry::new();
        register_user_browser_tools(&mut registry, &config, data.path());
        let call = |name: &'static str, arguments: Value| {
            let registry = &registry;
            let context = &context;
            async move { registry.call(name, arguments, context).await }
        };
        let page = call("user_browser_navigate", json!({"url": feed}))
            .await
            .unwrap();
        let text = |value: &Value| value["text"].as_str().unwrap_or("").to_owned();
        assert!(
            text(&page).contains("item 20") && !text(&page).contains("item 41"),
            "{}",
            text(&page)
        );

        // An infinite feed loads more as it is scrolled, even in the background.
        let scrolled = call(
            "user_browser_scroll",
            json!({"direction": "down", "amount": "end"}),
        )
        .await
        .unwrap();
        assert_eq!(
            scrolled["scroll"]["more_loaded"], true,
            "{}",
            scrolled["scroll"]
        );
        assert!(
            text(&scrolled).contains("item 21"),
            "{}",
            scrolled["scroll"]
        );
        // The tab stays where it was scrolled, so reading on continues from there.
        let at = scrolled["scroll"]["to"].as_u64().unwrap();
        assert!(at > 0, "{}", scrolled["scroll"]);
        let up = call(
            "user_browser_scroll",
            json!({"direction": "up", "amount": "page"}),
        )
        .await
        .unwrap();
        assert_eq!(up["scroll"]["from"].as_u64(), Some(at), "{}", up["scroll"]);
        assert!(
            up["scroll"]["to"].as_u64().unwrap() < at,
            "{}",
            up["scroll"]
        );

        // Waiting for results that appear after a delay, and for ones that never do.
        call("user_browser_click", json!({"target": "Load results"}))
            .await
            .unwrap();
        let ready = call(
            "user_browser_wait_for",
            json!({"text": "results ready", "timeout_seconds": 8}),
        )
        .await
        .unwrap();
        assert!(
            ready["waited_ms"].as_u64().unwrap() >= 500,
            "{}",
            ready["waited_ms"]
        );
        let missing = call(
            "user_browser_wait_for",
            json!({"text": "never shown", "timeout_seconds": 1}),
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(missing.contains("did not appear"), "{missing}");
        let gone = call(
            "user_browser_wait_for",
            json!({"text": "idle", "gone": true, "timeout_seconds": 3}),
        )
        .await
        .unwrap();
        assert!(gone.get("waited_ms").is_some());

        // A long page is read in parts that join up.
        let first = call(
            "user_browser_extract",
            json!({"offset": 0, "max_chars": 2000}),
        )
        .await
        .unwrap();
        assert_eq!(first["next_offset"], 2000, "{}", first["next_offset"]);
        let total = first["total_chars"].as_u64().unwrap();
        assert!(total > 20_000, "{total}");
        let second = call(
            "user_browser_extract",
            json!({"offset": 2000, "max_chars": 2000}),
        )
        .await
        .unwrap();
        assert_eq!(second["text_offset"], 2000);
        assert_ne!(first["text"], second["text"]);
        let tab = page["tab"]["tab_id"].as_str().unwrap().to_owned();
        call(
            "user_browser_tabs",
            json!({"action": "close", "tab_id": tab}),
        )
        .await
        .unwrap();
    }

    /// End to end against a real browser started with
    /// `--remote-debugging-port`: set `POK_AI_USER_BROWSER_TEST_PORT`.
    #[tokio::test]
    #[ignore = "needs a Chromium started with --remote-debugging-port; set POK_AI_USER_BROWSER_TEST_PORT"]
    async fn works_in_a_background_tab_of_a_running_browser() {
        let port: u16 = std::env::var("POK_AI_USER_BROWSER_TEST_PORT")
            .expect("POK_AI_USER_BROWSER_TEST_PORT")
            .parse()
            .unwrap();
        let page_url = std::env::var("POK_AI_USER_BROWSER_TEST_URL")
            .expect("POK_AI_USER_BROWSER_TEST_URL: an http page with the test form");
        let data = tempfile::tempdir().unwrap();
        let (context, recording) = approval_context(crate::policy::ApprovalDecision::Deny, &data);
        let config = UserBrowserConfig {
            enabled: true,
            port,
            user_data_dir: None,
        };
        let mut registry = ToolRegistry::new();
        register_user_browser_tools(&mut registry, &config, data.path());
        let status = registry
            .call("user_browser_status", json!({}), &context)
            .await
            .unwrap();
        assert_eq!(status["connected"], true, "{status}");
        let user_tabs = status["tabs"].as_array().unwrap().len();

        let page = registry
            .call("user_browser_navigate", json!({"url": page_url}), &context)
            .await
            .unwrap();
        assert_eq!(page["tab"]["opened_by_agent"], true, "{page}");
        let tabs = registry
            .call("user_browser_tabs", json!({"action": "list"}), &context)
            .await
            .unwrap();
        assert_eq!(tabs["tabs"].as_array().unwrap().len(), user_tabs + 1);

        // Click by visible text, fill by label and by CSS selector, choose
        // an option by label, and search with Enter: none of it asks.
        let clicked = registry
            .call(
                "user_browser_click",
                json!({"target": "Press me"}),
                &context,
            )
            .await
            .unwrap();
        // The user's tab stayed in front the whole time.
        assert_eq!(clicked["action"]["background_tab"], true, "{clicked}");
        // Icon controls: the point under the pointer is an SVG shape.
        for target in ["Star", "Inbox"] {
            registry
                .call("user_browser_click", json!({"target": target}), &context)
                .await
                .unwrap();
        }
        registry
            .call(
                "user_browser_type",
                json!({"target": "#date", "text": "2026-10-10"}),
                &context,
            )
            .await
            .unwrap();
        registry
            .call(
                "user_browser_select",
                json!({"target": "Size", "option": "Large"}),
                &context,
            )
            .await
            .unwrap();
        let searched = registry
            .call(
                "user_browser_type",
                json!({"target": "Search mail", "text": "invoice march", "submit": true}),
                &context,
            )
            .await
            .unwrap();
        assert_eq!(searched["action"]["submitted"], true, "{searched}");
        let read = registry
            .call(
                "user_browser_extract",
                json!({"selector": "#result", "include_html": true}),
                &context,
            )
            .await
            .unwrap();
        let text = read["section_text"].as_str().unwrap();
        assert!(text.contains("clicked 1"), "{text}");
        assert!(text.contains("date 2026-10-10"), "{text}");
        assert!(text.contains("size large"), "{text}");
        assert!(text.contains("searched invoice march"), "{text}");
        assert!(text.contains("starred"), "{text}");
        assert!(text.contains("link #inbox"), "{text}");
        assert!(
            read["html"].as_str().unwrap().contains("data-pok-id"),
            "{read}"
        );
        assert!(recording.asked.lock().is_empty());

        // Sending asks first; declined, nothing is sent.
        let error = registry
            .call("user_browser_click", json!({"target": "Send"}), &context)
            .await
            .unwrap_err();
        assert!(matches!(error, PokError::PolicyDenied { .. }), "{error}");
        assert_eq!(recording.asked.lock().len(), 1);
        let read = registry
            .call(
                "user_browser_extract",
                json!({"selector": "#result"}),
                &context,
            )
            .await
            .unwrap();
        assert!(
            !read["section_text"].as_str().unwrap().contains("sent"),
            "{read}"
        );
        // Password fields are never filled.
        let error = registry
            .call(
                "user_browser_type",
                json!({"target": "Password", "text": "hunter2"}),
                &context,
            )
            .await
            .unwrap_err();
        assert!(matches!(error, PokError::PolicyDenied { .. }), "{error}");

        // When the user looks at the agent's tab, native input is used.
        let agent_tab = page["tab"]["tab_id"].as_str().unwrap().to_owned();
        devtools_http_client()
            .unwrap()
            .get(format!("http://127.0.0.1:{port}/json/activate/{agent_tab}"))
            .send()
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(500)).await;
        let clicked = registry
            .call(
                "user_browser_click",
                json!({"target": "Press me"}),
                &context,
            )
            .await
            .unwrap();
        assert!(
            clicked["action"].get("background_tab").is_none(),
            "{clicked}"
        );
        registry
            .call(
                "user_browser_type",
                json!({"target": "Search mail", "text": "receipts", "submit": true}),
                &context,
            )
            .await
            .unwrap();
        let read = registry
            .call(
                "user_browser_extract",
                json!({"selector": "#result"}),
                &context,
            )
            .await
            .unwrap();
        let text = read["section_text"].as_str().unwrap();
        assert!(text.contains("clicked 2"), "{text}");
        assert!(text.contains("searched receipts"), "{text}");

        let shot = registry
            .call("user_browser_screenshot", json!({}), &context)
            .await
            .unwrap();
        assert!(
            shot["png_base64"]
                .as_str()
                .is_some_and(|data| !data.is_empty())
        );
        assert!(shot["width"].as_u64().unwrap() > 0, "{}", shot["width"]);
        if let Ok(path) = std::env::var("POK_AI_USER_BROWSER_TEST_SHOT") {
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(shot["png_base64"].as_str().unwrap())
                .unwrap();
            std::fs::write(path, bytes).unwrap();
        }
        registry
            .call(
                "user_browser_tabs",
                json!({"action": "close", "tab_id": agent_tab}),
                &context,
            )
            .await
            .unwrap();
        let tabs = registry
            .call("user_browser_tabs", json!({"action": "list"}), &context)
            .await
            .unwrap();
        assert_eq!(tabs["tabs"].as_array().unwrap().len(), user_tabs);
    }
}
