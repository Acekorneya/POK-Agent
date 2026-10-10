//! Managed Chromium fast path. It is intentionally separate from the native
//! desktop platform: personal browser windows continue through UIA/OCR, while
//! this module owns only the isolated automation profile it launches.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, OnceLock},
    time::Duration,
};

use async_trait::async_trait;
use futures::{SinkExt, StreamExt};
use parking_lot::Mutex as SyncMutex;
#[cfg(windows)]
use process_wrap::tokio::JobObject;
use process_wrap::tokio::{ChildWrapper, CommandWrap, KillOnDrop};
use schemars::{JsonSchema, schema_for};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::sync::Mutex;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use uuid::Uuid;

use crate::{
    PokError, Result,
    policy::RiskClass,
    tool::{Tool, ToolContext, ToolRegistry},
};

const SNAPSHOT_SCRIPT: &str = r#"(() => {
  const root = globalThis;
  root.__pokBrowserIds ||= new WeakMap();
  root.__pokBrowserNextId ||= 1;
  const ids = root.__pokBrowserIds;
  const seen = [];
  const visit = (node, frame = 'main') => {
    if (!node || !node.querySelectorAll) return;
    for (const el of node.querySelectorAll('*')) {
      const rect = el.getBoundingClientRect();
      const style = getComputedStyle(el);
      if (rect.width < 1 || rect.height < 1 || style.visibility === 'hidden' || style.display === 'none') continue;
      const tag = el.tagName.toLowerCase();
      const role = el.getAttribute('role') || ({a:'link',button:'button',input:'textbox',select:'combobox',textarea:'textbox'}[tag] || tag);
      const label = el.getAttribute('aria-label') || el.getAttribute('title') ||
        (el.labels && el.labels[0] && el.labels[0].innerText) || el.innerText || el.getAttribute('placeholder') || el.getAttribute('alt') || '';
      const actionable = ['a','button','input','select','textarea','summary'].includes(tag) ||
        ['button','link','textbox','combobox','checkbox','radio','menuitem','tab','option'].includes(role) ||
        el.tabIndex >= 0 || typeof el.onclick === 'function';
      if (actionable || label.trim()) {
        const inputType = tag === 'input' ? String(el.type || 'text').toLowerCase() : null;
        const sensitive = inputType === 'password' || el.autocomplete === 'current-password' || el.autocomplete === 'new-password';
        if (!ids.has(el)) ids.set(el, `b${root.__pokBrowserNextId++}`);
        const guard = JSON.stringify([location.href, role, String(label).replace(/\s+/g,' ').trim().slice(0,300),
          'value' in el && !sensitive ? String(el.value).slice(0,300) : null, 'checked' in el ? !!el.checked : null,
          !!el.disabled || el.getAttribute('aria-disabled') === 'true']);
        seen.push({
          id: ids.get(el), frame, tag, role,
          name: String(label).replace(/\s+/g,' ').trim().slice(0,300),
          input_type: inputType, sensitive,
          value: 'value' in el && !sensitive ? String(el.value).slice(0,300) : null,
          options: tag === 'select' ? Array.from(el.options).slice(0,100).map(o => ({value:String(o.value).slice(0,300), label:String(o.textContent || '').trim().slice(0,300), selected:!!o.selected})) : null,
          checked: 'checked' in el ? !!el.checked : null,
          disabled: !!el.disabled || el.getAttribute('aria-disabled') === 'true',
          selected: 'selected' in el ? !!el.selected : null,
          actionable, guard,
          bounds: {x: Math.round(rect.x), y: Math.round(rect.y), width: Math.round(rect.width), height: Math.round(rect.height)}
        });
      }
      if (el.shadowRoot) visit(el.shadowRoot, frame);
      if (tag === 'iframe') {
        try { if (el.contentDocument) visit(el.contentDocument, `${frame}/iframe`); } catch (_) {}
      }
    }
  };
  visit(document);
  return {
    url: location.href,
    title: document.title,
    loading: document.readyState !== 'complete',
    elements: seen.slice(0, 2500),
    text: String(document.body?.innerText || '').replace(/\s+/g,' ').trim().slice(0,16000),
    metrics: { element_count: seen.length, scroll_y: Math.round(scrollY), page_height: document.documentElement?.scrollHeight || 0, viewport_height: innerHeight }
  };
})()"#;

const DEVTOOLS_STARTUP_TIMEOUT: Duration = Duration::from_secs(10);
const DEVTOOLS_HTTP_TIMEOUT: Duration = Duration::from_millis(750);
const CDP_OPERATION_TIMEOUT: Duration = Duration::from_secs(10);

pub(crate) fn devtools_http_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .no_proxy()
        .connect_timeout(Duration::from_millis(500))
        .timeout(DEVTOOLS_HTTP_TIMEOUT)
        .build()
        .map_err(|error| PokError::Tool(format!("failed to configure DevTools client: {error}")))
}

pub(crate) struct BrowserConnection {
    // Chromium's multi-process architecture spawns renderer, GPU, utility,
    // and network-service processes as children of the browser process we
    // launch; those children are never visible to a plain `tokio::process::Child`
    // handle, so `child.kill()` / `kill_on_drop` only terminate the one
    // process we're tracking and orphan the rest (confirmed in practice:
    // dozens of leaked chrome.exe processes accumulated across repeated
    // managed-browser tasks). On Windows, wrapping the spawn with
    // `process_wrap`'s `JobObject` assigns the process (and everything it
    // spawns, since job membership is inherited) to a job with
    // `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, so the whole tree dies together
    // when `child.kill()` is called or the job handle closes.
    /// `None` when attached to a managed browser that was already running
    /// with this profile (left open by an earlier session); it is alive while
    /// its DevTools endpoint answers.
    child: Option<Box<dyn ChildWrapper>>,
    pub(crate) port: u16,
    /// The tab every DevTools call goes to. A restored or reused profile can
    /// hold several tabs whose listing order changes, so the first tab used
    /// is kept, and a click that opens a new tab moves here to it.
    pub(crate) page_target: Option<String>,
    pub(crate) snapshot_id: Option<String>,
    target_guards: HashMap<String, String>,
    pub(crate) last_snapshot: Option<Value>,
    /// False for the user's own browser, attached over its DevTools port
    /// (see `user_browser`); true for the isolated profile launched here.
    managed: bool,
    /// Act only in a tab this connection opened or was given: the user's
    /// browser, and each swarm worker's lane of the managed browser.
    own_tabs_only: bool,
    /// Tabs this connection opened, closed when a swarm worker finishes.
    opened_tabs: Vec<String>,
}

impl BrowserConnection {
    fn attached(child: Option<Box<dyn ChildWrapper>>, port: u16) -> Self {
        Self {
            child,
            port,
            page_target: None,
            snapshot_id: None,
            target_guards: HashMap::new(),
            last_snapshot: None,
            managed: true,
            own_tabs_only: false,
            opened_tabs: Vec::new(),
        }
    }

    /// The user's own running browser. It is never launched or closed here.
    pub(crate) fn user(port: u16) -> Self {
        Self {
            managed: false,
            own_tabs_only: true,
            ..Self::attached(None, port)
        }
    }
}

/// The browser-wide DevTools WebSocket (tab creation, the target list).
async fn browser_endpoint(port: u16) -> Result<String> {
    let version = devtools_http_client()?
        .get(format!("http://127.0.0.1:{port}/json/version"))
        .send()
        .await
        .map_err(|error| {
            PokError::Tool(format!(
                "the browser's DevTools endpoint did not answer: {error}"
            ))
        })?
        .json::<Value>()
        .await
        .map_err(|error| PokError::Tool(format!("invalid DevTools version answer: {error}")))?;
    version
        .get("webSocketDebuggerUrl")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| PokError::Tool("the browser gave no DevTools endpoint".into()))
}

/// Tabs opened by `opener` (a link with a new-tab target, `window.open`),
/// `rel=noopener` included.
async fn tabs_opened_by(port: u16, opener: &str) -> Result<Vec<String>> {
    let endpoint = browser_endpoint(port).await?;
    let targets = tokio::time::timeout(
        CDP_OPERATION_TIMEOUT,
        cdp_call(&endpoint, "Target.getTargets", json!({})),
    )
    .await
    .map_err(|_| PokError::Tool("the browser did not list its tabs in time".into()))??;
    Ok(targets
        .get("targetInfos")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|target| target.get("type").and_then(Value::as_str) == Some("page"))
        .filter(|target| target.get("openerId").and_then(Value::as_str) == Some(opener))
        .filter_map(|target| Some(target.get("targetId")?.as_str()?.to_owned()))
        .collect())
}

/// Open a tab behind the current one through the browser's DevTools
/// endpoint, and wait until the browser lists it.
pub(crate) async fn open_background_tab(port: u16) -> Result<String> {
    let endpoint = browser_endpoint(port).await?;
    let created = tokio::time::timeout(
        CDP_OPERATION_TIMEOUT,
        cdp_call(
            &endpoint,
            "Target.createTarget",
            json!({"url": "about:blank", "background": true}),
        ),
    )
    .await
    .map_err(|_| PokError::Tool("the browser did not open a tab in time".into()))??;
    let id = created
        .get("targetId")
        .and_then(Value::as_str)
        .ok_or_else(|| PokError::Tool("the browser opened no tab".into()))?
        .to_owned();
    // The new tab can take a moment to appear in the tab listing.
    for _ in 0..20 {
        if ManagedBrowser::page_targets(port)
            .await?
            .iter()
            .any(|(tab, _, _)| *tab == id)
        {
            return Ok(id);
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    Err(PokError::Tool(
        "the browser opened a tab but does not list it".into(),
    ))
}

/// Close one tab; a tab already gone is not an error.
pub(crate) async fn close_tab(port: u16, id: &str) {
    if let Ok(client) = devtools_http_client() {
        let _ = client
            .get(format!("http://127.0.0.1:{port}/json/close/{id}"))
            .send()
            .await;
    }
}

/// The port in a Chromium profile's `DevToolsActivePort` file (first line).
fn devtools_active_port(text: &str) -> Option<u16> {
    text.lines()
        .next()?
        .trim()
        .parse()
        .ok()
        .filter(|port| *port != 0)
}

async fn devtools_ready(client: &reqwest::Client, port: u16) -> bool {
    client
        .get(format!("http://127.0.0.1:{port}/json/version"))
        .send()
        .await
        .is_ok_and(|response| response.status().is_success())
}

pub struct ManagedBrowser {
    profile_dir: PathBuf,
    executable_override: Option<PathBuf>,
    state: Mutex<Option<BrowserConnection>>,
    /// The main agent's browser for this data folder, when this one is a
    /// swarm worker's lane: the worker attaches to the browser that one
    /// started and works only in tabs it opens.
    primary: Option<Arc<ManagedBrowser>>,
}

impl ManagedBrowser {
    fn new(data_dir: &Path) -> Self {
        Self {
            profile_dir: data_dir.join("browser").join("profiles").join("default"),
            executable_override: std::env::var_os("POK_AI_BROWSER_EXECUTABLE").map(PathBuf::from),
            state: Mutex::new(None),
            primary: None,
        }
    }

    async fn ensure_started(&self) -> Result<()> {
        if let Some(primary) = &self.primary {
            return self.attach_lane(primary).await;
        }
        let mut state = self.state.lock().await;
        let client = devtools_http_client()?;
        if let Some(connection) = state.as_mut() {
            let alive = match connection.child.as_mut() {
                Some(child) => child.try_wait().ok().flatten().is_none(),
                None => devtools_ready(&client, connection.port).await,
            };
            if alive {
                return Ok(());
            }
        }
        std::fs::create_dir_all(&self.profile_dir)?;
        // A browser already running with this profile (from an earlier
        // conversation) is reused: a second launch would only hand its
        // request to that window and exit.
        if let Some(port) = self.running_port(&client).await {
            *state = Some(BrowserConnection::attached(None, port));
            return Ok(());
        }
        let executables = match &self.executable_override {
            Some(executable) => vec![executable.clone()],
            None => chromium_candidates(),
        };
        if executables.is_empty() {
            return Err(PokError::Unsupported("managed browser requires Microsoft Edge or Google Chrome; set POK_AI_BROWSER_EXECUTABLE to override discovery".into()));
        }
        let mut last_error = None;
        for executable in executables {
            match self.launch(&executable, &client).await {
                Ok(connection) => {
                    *state = Some(connection);
                    return Ok(());
                }
                Err(error) => {
                    // Exit code 0 before DevTools usually means the launch was
                    // handed to a copy already using this profile: attach.
                    if let Some(port) = self.running_port(&client).await {
                        *state = Some(BrowserConnection::attached(None, port));
                        return Ok(());
                    }
                    last_error = Some(match error {
                        PokError::Tool(message) => message,
                        other => other.to_string(),
                    });
                }
            }
        }
        let error = last_error.unwrap_or_else(|| "managed browser could not be started".into());
        Err(PokError::Tool(format!(
            "{error}. Do not retry the managed browser in this turn; instead open the page in the user's own browser (run_command `Start-Process \"<url>\"`, or open_application) and read it with the desktop tools (activate_window, capture_screen)."
        )))
    }

    /// A worker's lane: the browser is launched (once) by the main agent's
    /// instance, which owns the process, so a finishing worker never closes
    /// the browser the others use.
    async fn attach_lane(&self, primary: &ManagedBrowser) -> Result<()> {
        let mut state = self.state.lock().await;
        let client = devtools_http_client()?;
        if let Some(connection) = state.as_ref()
            && devtools_ready(&client, connection.port).await
        {
            return Ok(());
        }
        Box::pin(primary.ensure_started()).await?;
        let port = primary
            .state
            .lock()
            .await
            .as_ref()
            .map(|connection| connection.port)
            .ok_or_else(|| PokError::Tool("managed browser is not running".into()))?;
        let mut connection = BrowserConnection::attached(None, port);
        connection.own_tabs_only = true;
        *state = Some(connection);
        Ok(())
    }

    /// Give a worker's lane its own tab the first time it opens a page.
    async fn ensure_own_tab(connection: &mut BrowserConnection) -> Result<()> {
        if !connection.managed || !connection.own_tabs_only {
            return Ok(());
        }
        let open = Self::page_targets(connection.port).await?;
        if connection
            .page_target
            .as_ref()
            .is_some_and(|id| open.iter().any(|(tab, _, _)| tab == id))
        {
            return Ok(());
        }
        let id = open_background_tab(connection.port).await?;
        connection.opened_tabs.push(id.clone());
        connection.page_target = Some(id);
        Ok(())
    }

    /// The DevTools port of a browser already running with this profile, when
    /// it answers.
    async fn running_port(&self, client: &reqwest::Client) -> Option<u16> {
        let text = std::fs::read_to_string(self.profile_dir.join("DevToolsActivePort")).ok()?;
        let port = devtools_active_port(&text)?;
        devtools_ready(client, port).await.then_some(port)
    }

    async fn launch(
        &self,
        executable: &Path,
        client: &reqwest::Client,
    ) -> Result<BrowserConnection> {
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0))?;
        let port = listener.local_addr()?.port();
        drop(listener);
        let mut command = CommandWrap::with_new(executable, |command| {
            command
                .arg(format!("--remote-debugging-port={port}"))
                .arg("--remote-debugging-address=127.0.0.1")
                .arg(format!("--user-data-dir={}", self.profile_dir.display()))
                .arg("--no-first-run")
                .arg("--no-default-browser-check")
                // The agent works while this window is behind others or on
                // another monitor. Chromium otherwise pauses rendering, timers,
                // and input for windows it considers hidden, so clicks sent
                // over DevTools would wait until the user brought it up.
                .arg("--disable-backgrounding-occluded-windows")
                .arg("--disable-renderer-backgrounding")
                .arg("--disable-background-timer-throttling")
                .arg("--disable-features=CalculateNativeWinOcclusion")
                .arg("about:blank");
        });
        command.wrap(KillOnDrop);
        #[cfg(windows)]
        command.wrap(JobObject);
        let mut child = command.spawn().map_err(|error| {
            PokError::Tool(format!(
                "failed to launch managed browser {}: {error}",
                executable.display()
            ))
        })?;
        let endpoint = format!("http://127.0.0.1:{port}/json/version");
        let deadline = tokio::time::Instant::now() + DEVTOOLS_STARTUP_TIMEOUT;
        let mut last_error = None;
        while tokio::time::Instant::now() < deadline {
            match client.get(&endpoint).send().await {
                Ok(response) if response.status().is_success() => {
                    return Ok(BrowserConnection::attached(Some(child), port));
                }
                Ok(response) => {
                    last_error = Some(format!("HTTP {}", response.status()));
                }
                Err(error) => {
                    last_error = Some(error.to_string());
                }
            }
            if let Some(status) = child.try_wait()? {
                return Err(PokError::Tool(format!(
                    "managed browser {} exited before exposing DevTools (status {status})",
                    executable.display()
                )));
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
        let _ = Box::into_pin(child.kill()).await;
        let _ = child.wait().await;
        Err(PokError::Tool(format!(
            "managed browser did not expose its local DevTools endpoint within {} seconds{}",
            DEVTOOLS_STARTUP_TIMEOUT.as_secs(),
            last_error.map_or_else(String::new, |error| format!(": {error}"))
        )))
    }

    /// Open tabs as (target id, DevTools WebSocket URL, page URL).
    pub(crate) async fn page_targets(port: u16) -> Result<Vec<(String, String, String)>> {
        let pages = devtools_http_client()?
            .get(format!("http://127.0.0.1:{port}/json/list"))
            .send()
            .await
            .map_err(|error| PokError::Tool(format!("CDP target discovery failed: {error}")))?
            .json::<Vec<Value>>()
            .await
            .map_err(|error| PokError::Tool(format!("invalid CDP target list: {error}")))?;
        Ok(pages
            .iter()
            .filter(|page| page.get("type").and_then(Value::as_str) == Some("page"))
            .filter_map(|page| {
                Some((
                    page.get("id")?.as_str()?.to_owned(),
                    page.get("webSocketDebuggerUrl")?.as_str()?.to_owned(),
                    page.get("url")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                ))
            })
            .collect())
    }

    /// The pinned tab's WebSocket, pinning a web page (else any tab) when
    /// none is pinned or the pinned tab was closed.
    async fn page_websocket(connection: &mut BrowserConnection) -> Result<String> {
        let pages = Self::page_targets(connection.port).await?;
        let pinned = connection
            .page_target
            .as_ref()
            .and_then(|id| pages.iter().find(|(page_id, _, _)| page_id == id));
        // In the user's own browser the agent never picks a tab by itself:
        // falling back would mean acting in a tab the user is using.
        if connection.own_tabs_only && pinned.is_none() {
            connection.page_target = None;
            return Err(PokError::Tool(if connection.managed {
                "this agent has no open tab of its own in the managed browser yet; open a page with managed_browser_open first".into()
            } else {
                "the agent has no open tab in the user's browser (it was closed or none was chosen); open one with user_browser_navigate or choose one with user_browser_tabs".into()
            }));
        }
        let (id, socket, _) = pinned
            .or_else(|| pages.iter().find(|(_, _, url)| url.starts_with("http")))
            .or_else(|| pages.first())
            .ok_or_else(|| PokError::Tool("managed browser has no page target".into()))?;
        connection.page_target = Some(id.clone());
        Ok(socket.clone())
    }

    pub(crate) async fn cdp(
        connection: &mut BrowserConnection,
        method: &str,
        params: Value,
    ) -> Result<Value> {
        match tokio::time::timeout(
            CDP_OPERATION_TIMEOUT,
            Self::cdp_inner(connection, method, params),
        )
        .await
        {
            Ok(result) => result,
            Err(_) => Err(PokError::Tool(format!(
                "CDP {method} exceeded the {} second operation timeout",
                CDP_OPERATION_TIMEOUT.as_secs()
            ))),
        }
    }

    async fn cdp_inner(
        connection: &mut BrowserConnection,
        method: &str,
        params: Value,
    ) -> Result<Value> {
        let endpoint = Self::page_websocket(connection).await?;
        cdp_call(&endpoint, method, params).await
    }

    pub(crate) async fn snapshot_locked(connection: &mut BrowserConnection) -> Result<Value> {
        let result = Self::cdp(
            connection,
            "Runtime.evaluate",
            json!({"expression": SNAPSHOT_SCRIPT, "returnByValue": true, "awaitPromise": true}),
        )
        .await?;
        let mut page = result
            .pointer("/result/value")
            .cloned()
            .ok_or_else(|| PokError::Tool("managed browser snapshot returned no value".into()))?;
        let encoded = serde_json::to_vec(&page)?;
        let fingerprint = format!("{:x}", Sha256::digest(&encoded));
        let snapshot_id = Uuid::new_v4().to_string();
        connection.target_guards = page
            .get("elements")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|element| {
                Some((
                    element.get("id")?.as_str()?.to_owned(),
                    element.get("guard")?.as_str()?.to_owned(),
                ))
            })
            .collect();
        connection.snapshot_id = Some(snapshot_id.clone());
        page["snapshot_id"] = json!(snapshot_id);
        page["fingerprint"] = json!(fingerprint);
        page["managed_profile"] = json!(connection.managed);
        if !connection.managed {
            page["user_profile"] = json!(true);
        }
        connection.last_snapshot = Some(page.clone());
        Ok(page)
    }

    async fn snapshot(&self) -> Result<Value> {
        self.ensure_started().await?;
        let mut state = self.state.lock().await;
        Self::snapshot_locked(state.as_mut().expect("browser started")).await
    }

    async fn navigate(&self, url: &str) -> Result<Value> {
        let parsed = reqwest::Url::parse(url).map_err(|_| {
            PokError::Tool("managed browser URL must be an absolute http(s) URL".into())
        })?;
        if !matches!(parsed.scheme(), "http" | "https") {
            return Err(PokError::Tool(
                "managed browser permits only http(s) navigation".into(),
            ));
        }
        self.ensure_started().await?;
        let mut state = self.state.lock().await;
        let connection = state.as_mut().expect("browser started");
        Self::ensure_own_tab(connection).await?;
        Self::cdp(connection, "Page.navigate", json!({"url": parsed.as_str()})).await?;
        tokio::time::sleep(Duration::from_millis(400)).await;
        Self::snapshot_locked(connection).await
    }

    async fn act(
        &self,
        snapshot_id: &str,
        target_id: &str,
        operation: &str,
        value: Option<&str>,
    ) -> Result<Value> {
        self.ensure_started().await?;
        let mut state = self.state.lock().await;
        let connection = state.as_mut().expect("browser started");
        Self::act_locked(connection, snapshot_id, target_id, operation, value).await
    }

    /// Perform one freshness- and semantics-checked action on a snapshot
    /// target, then return the settled snapshot. Input goes to the page,
    /// never through the system cursor or keyboard: native DevTools input,
    /// or page events in a background tab of the user's own browser.
    pub(crate) async fn act_locked(
        connection: &mut BrowserConnection,
        snapshot_id: &str,
        target_id: &str,
        operation: &str,
        value: Option<&str>,
    ) -> Result<Value> {
        let Some(expected_guard) = connection
            .target_guards
            .get(target_id)
            .cloned()
            .filter(|_| connection.snapshot_id.as_deref() == Some(snapshot_id))
        else {
            return Err(PokError::Tool(
                "stale managed-browser target; take a new browser_snapshot".into(),
            ));
        };
        let target = serde_json::to_string(target_id)?;
        let operation_json = serde_json::to_string(operation)?;
        let value_json = serde_json::to_string(value.unwrap_or_default())?;
        let guard = serde_json::to_string(&expected_guard)?;
        // A browser draws only the tab in front, and drops native pointer and
        // key input sent to a background tab (it is not queued: nothing fires
        // later when the tab is shown). In a background tab (the user's
        // browser, or a swarm worker's tab) the action is performed with
        // page events instead.
        let expression = format!(
            r#"(() => {{
          const ids = globalThis.__pokBrowserIds;
          if (!ids) return {{ok:false, reason:'snapshot state missing'}};
          const all = [];
          const visit = n => {{ if (!n?.querySelectorAll) return; for (const e of n.querySelectorAll('*')) {{ all.push(e); if (e.shadowRoot) visit(e.shadowRoot); try {{ if (e.tagName === 'IFRAME' && e.contentDocument) visit(e.contentDocument); }} catch (_) {{}} }} }};
          visit(document);
          const el = all.find(e => ids.get(e) === {target});
          if (!el || !el.isConnected) return {{ok:false, reason:'target disappeared'}};
          const r = el.getBoundingClientRect(); const s = getComputedStyle(el);
          if (r.width < 1 || r.height < 1 || s.visibility === 'hidden' || s.display === 'none' || el.disabled) return {{ok:false, reason:'target is not actionable'}};
          const role = el.getAttribute('role') || ({{a:'link',button:'button',input:'textbox',select:'combobox',textarea:'textbox'}}[el.tagName.toLowerCase()] || el.tagName.toLowerCase());
          const label = el.getAttribute('aria-label') || el.getAttribute('title') || (el.labels && el.labels[0] && el.labels[0].innerText) || el.innerText || el.getAttribute('placeholder') || el.getAttribute('alt') || '';
          const currentGuard = JSON.stringify([location.href, role, String(label).replace(/\s+/g,' ').trim().slice(0,300),
            'value' in el ? String(el.value).slice(0,300) : null, 'checked' in el ? !!el.checked : null,
            !!el.disabled || el.getAttribute('aria-disabled') === 'true']);
          if (currentGuard !== {guard}) return {{ok:false, reason:'target semantics changed'}};
          let box=r, x=r.x+r.width/2, y=r.y+r.height/2;
          const outside = () => x < 0 || y < 0 || x >= innerWidth || y >= innerHeight;
          // A target below or beside the visible area is scrolled into view
          // first, as a person would; one still covered (a banner, a dialog) is refused.
          if (outside()) {{ el.scrollIntoView({{block:'center', inline:'center', behavior:'instant'}}); box = el.getBoundingClientRect(); x=box.x+box.width/2; y=box.y+box.height/2; }}
          if (outside()) return {{ok:false, reason:'target is outside the viewport'}};
          if (!el.contains(document.elementFromPoint(x,y))) return {{ok:false, reason:'target is covered by another element (a banner or dialog); dismiss it first'}};
          const op = {operation_json}; const v = {value_json};
          const inputType = el.tagName === 'INPUT' ? String(el.type || 'text').toLowerCase() : '';
          if (op === 'type' && (inputType === 'password' || el.autocomplete === 'current-password' || el.autocomplete === 'new-password')) return {{ok:false, reason:'password-field typing is prohibited'}};
          // The prototype setter, not `el.value =`, so frameworks that track
          // the value on the element (React) still see the change.
          const setValue = (next) => {{
            const proto = el instanceof HTMLSelectElement ? HTMLSelectElement.prototype : el instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
            Object.getOwnPropertyDescriptor(proto, 'value').set.call(el, next);
            el.dispatchEvent(new Event('input', {{bubbles:true}})); el.dispatchEvent(new Event('change', {{bubbles:true}}));
          }};
          // Date, time, color, and range inputs do not take typed text.
          if (op === 'type' && ['date','time','datetime-local','month','week','color','range'].includes(inputType)) {{ el.focus(); setValue(v); return {{ok:true,x,y,filled:true}}; }}
          if (document.visibilityState !== 'visible') {{
            const hit = document.elementFromPoint(x, y) || el;
            const at = {{bubbles:true, cancelable:true, composed:true, view:window, clientX:x, clientY:y}};
            const pointer = {{...at, pointerId:1, pointerType:'mouse', isPrimary:true}};
            if (op === 'click' || op === 'hover') {{
              hit.dispatchEvent(new PointerEvent('pointerover', pointer)); hit.dispatchEvent(new MouseEvent('mouseover', at));
              el.dispatchEvent(new MouseEvent('mouseenter', {{...at, bubbles:false}}));
              hit.dispatchEvent(new PointerEvent('pointermove', pointer)); hit.dispatchEvent(new MouseEvent('mousemove', at));
            }}
            if (op === 'click') {{
              hit.dispatchEvent(new PointerEvent('pointerdown', {{...pointer, button:0, buttons:1}})); hit.dispatchEvent(new MouseEvent('mousedown', {{...at, button:0, buttons:1}}));
              if (typeof el.focus === 'function') el.focus({{preventScroll:true}});
              hit.dispatchEvent(new PointerEvent('pointerup', {{...pointer, button:0}})); hit.dispatchEvent(new MouseEvent('mouseup', {{...at, button:0}}));
              // An icon's SVG has no click(); its click event bubbles to the
              // button or link around it.
              if (typeof hit.click === 'function') hit.click(); else hit.dispatchEvent(new MouseEvent('click', {{...at, button:0, detail:1}}));
              return {{ok:true,x,y,page_events:true}};
            }}
            if (op === 'hover') return {{ok:true,x,y,page_events:true}};
            if (op === 'type') {{
              el.focus();
              if (typeof el.select === 'function') el.select();
              else {{ const range = document.createRange(); range.selectNodeContents(el); const selection = getSelection(); selection.removeAllRanges(); selection.addRange(range); }}
              return {{ok:true,x,y,page_events:true}};
            }}
          }}
          if (op === 'type') el.focus();
          else if (op === 'select') {{
            if (!(el instanceof HTMLSelectElement)) return {{ok:false, reason:'target is not a list (select); click it and then click the option instead'}};
            const norm = s => String(s).replace(/\s+/g,' ').trim().toLowerCase();
            const option = Array.from(el.options).find(o => o.value === v) || Array.from(el.options).find(o => norm(o.textContent) === norm(v));
            if (!option) return {{ok:false, reason:'no option has that value or label'}};
            setValue(option.value);
          }}
          else if (op !== 'click' && op !== 'hover') return {{ok:false, reason:'unsupported operation'}};
          return {{ok:true,x,y}};
        }})()"#
        );
        // The tabs open before the action: in a background tab the page
        // performs a click itself, so a tab it opens exists once it returns.
        let before = if operation == "click" {
            Self::page_targets(connection.port).await?
        } else {
            Vec::new()
        };
        // As a user's click: without it, a page-event click on a link that
        // opens a new tab is dropped by the popup blocker.
        let result = Self::cdp(
            connection,
            "Runtime.evaluate",
            json!({"expression": expression, "returnByValue": true, "userGesture": true}),
        )
        .await?;
        if result.pointer("/result/value/ok").and_then(Value::as_bool) != Some(true) {
            return Err(PokError::Tool(format!(
                "{} action was rejected: {}",
                if connection.managed {
                    "managed browser"
                } else {
                    "browser"
                },
                result
                    .pointer("/result/value/reason")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
            )));
        }
        let x = result
            .pointer("/result/value/x")
            .and_then(Value::as_f64)
            .unwrap_or_default();
        let y = result
            .pointer("/result/value/y")
            .and_then(Value::as_f64)
            .unwrap_or_default();
        // The page already performed the action with page events.
        let performed = result
            .pointer("/result/value/page_events")
            .and_then(Value::as_bool)
            == Some(true);
        match operation {
            "click" => {
                let opener = connection.page_target.clone();
                for event in ["mousePressed", "mouseReleased"]
                    .into_iter()
                    .filter(|_| !performed)
                {
                    Self::cdp(
                        connection,
                        "Input.dispatchMouseEvent",
                        json!({
                            "type": event, "x": x, "y": y, "button": "left", "clickCount": 1
                        }),
                    )
                    .await?;
                }
                // A link that opens a new tab: continue in that tab. Only a
                // tab this one opened: a tab the user or another swarm worker
                // opens meanwhile is never taken over.
                tokio::time::sleep(Duration::from_millis(400)).await;
                if let Some(opener) = opener
                    && let Some(id) = tabs_opened_by(connection.port, &opener)
                        .await?
                        .into_iter()
                        .rev()
                        .find(|id| !before.iter().any(|(known, _, _)| known == id))
                {
                    if connection.managed && connection.own_tabs_only {
                        connection.opened_tabs.push(id.clone());
                    }
                    connection.page_target = Some(id);
                }
            }
            "hover" if performed => {}
            "hover" => {
                Self::cdp(
                    connection,
                    "Input.dispatchMouseEvent",
                    json!({
                        "type": "mouseMoved", "x": x, "y": y
                    }),
                )
                .await?;
            }
            "type"
                if result
                    .pointer("/result/value/filled")
                    .and_then(Value::as_bool)
                    == Some(true) => {}
            "type" => {
                // In a background tab the page selected the field's content;
                // text insertion still reaches it there.
                if !performed {
                    Self::cdp(connection, "Input.dispatchKeyEvent", json!({
                        "type": "keyDown", "key": "a", "code": "KeyA", "modifiers": 2, "commands": ["selectAll"]
                    })).await?;
                    Self::cdp(
                        connection,
                        "Input.dispatchKeyEvent",
                        json!({
                            "type": "keyUp", "key": "a", "code": "KeyA", "modifiers": 2
                        }),
                    )
                    .await?;
                }
                Self::cdp(
                    connection,
                    "Input.insertText",
                    json!({"text": value.unwrap_or_default()}),
                )
                .await?;
            }
            "select" => {}
            _ => unreachable!("validated above"),
        }
        tokio::time::sleep(Duration::from_millis(400)).await;
        let mut action = json!({"operation": operation, "target_id": target_id, "ok": true});
        if performed {
            action["background_tab"] = json!(true);
        }
        Ok(Self::snapshot_after_action(connection, action).await)
    }

    /// A snapshot once a page has settled. A navigation replaces the page
    /// while it is read, so a failed read is retried; with `wait_for_load`,
    /// so is a page still loading. Gives up after about three seconds.
    pub(crate) async fn settled_snapshot(
        connection: &mut BrowserConnection,
        wait_for_load: bool,
    ) -> Result<Value> {
        let mut snapshot = Self::snapshot_locked(connection).await;
        for _ in 0..7 {
            let settled = match &snapshot {
                Ok(page) => {
                    !wait_for_load || page.get("loading").and_then(Value::as_bool) != Some(true)
                }
                Err(_) => false,
            };
            if settled {
                break;
            }
            tokio::time::sleep(Duration::from_millis(400)).await;
            snapshot = Self::snapshot_locked(connection).await;
        }
        snapshot
    }

    /// The page after an action that already happened. A page that cannot be
    /// read yet must not report the action as failed, or it could be repeated.
    pub(crate) async fn snapshot_after_action(
        connection: &mut BrowserConnection,
        action: Value,
    ) -> Value {
        let wait_for_load = !connection.managed;
        match Self::settled_snapshot(connection, wait_for_load).await {
            Ok(mut snapshot) => {
                snapshot["action"] = action;
                snapshot
            }
            Err(error) => json!({
                "action": action,
                "snapshot_error": error.to_string(),
                "next": "The action was performed but the page could not be read afterwards. Read the page again before acting further; do not repeat the action.",
            }),
        }
    }
}

/// One DevTools request on a page or browser WebSocket endpoint.
pub(crate) async fn cdp_call(endpoint: &str, method: &str, params: Value) -> Result<Value> {
    let (mut socket, _) = connect_async(endpoint)
        .await
        .map_err(|error| PokError::Tool(format!("CDP connection failed: {error}")))?;
    socket
        .send(Message::Text(
            serde_json::to_string(&json!({"id": 1, "method": method, "params": params}))?.into(),
        ))
        .await
        .map_err(|error| PokError::Tool(format!("CDP request failed: {error}")))?;
    while let Some(message) = socket.next().await {
        let message =
            message.map_err(|error| PokError::Tool(format!("CDP response failed: {error}")))?;
        let Some(text) = (match message {
            Message::Text(text) => Some(text),
            _ => None,
        }) else {
            continue;
        };
        let value: Value = serde_json::from_str(&text)?;
        if value.get("id").and_then(Value::as_u64) != Some(1) {
            continue;
        }
        if let Some(error) = value.get("error") {
            return Err(PokError::Tool(format!("CDP {method} failed: {error}")));
        }
        return Ok(value.get("result").cloned().unwrap_or(Value::Null));
    }
    Err(PokError::Tool(
        "CDP connection closed without a response".into(),
    ))
}

/// Bound a snapshot element's accessible name for decision descriptions. A
/// long link text (repository cards, article links) previously arrived as a
/// mid-word-truncated page-text blob, and the cross-model judge answered
/// `reversible: unknown` because it could not tell what the action was.
/// Keep a readable label so evidence/reversibility questions are answerable.
pub(crate) fn bounded_label(label: &str) -> String {
    let normalized = label.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut result = String::new();
    for word in normalized.split(' ') {
        if !result.is_empty() && result.chars().count() + word.chars().count() + 1 > 72 {
            break;
        }
        if !result.is_empty() {
            result.push(' ');
        }
        result.push_str(word);
    }
    if result.chars().count() < normalized.chars().count() {
        result.push('…');
    }
    result
}

/// Produce only actions whose arguments are already complete and whose target
/// identity is tied to the most recent structured snapshot. Values for typing
/// are deliberately never invented here.
pub async fn decision_candidates(
    context: &ToolContext,
    task: &str,
    current_step: &str,
    limit: usize,
) -> Vec<crate::decision::DecisionCandidate> {
    use crate::decision::{DecisionCandidate, DecisionCandidateKind};

    let request_text = format!("{task} {current_step}");
    let query = request_text.to_ascii_lowercase();
    let explicit_url = explicit_http_url(&request_text);
    let mut candidates = Vec::new();

    let managed = browser(context);
    let state = managed.state.lock().await;
    let Some(snapshot) = state
        .as_ref()
        .and_then(|state| state.last_snapshot.as_ref())
    else {
        if let Some(url) = explicit_url {
            candidates.push(DecisionCandidate {
                id: "browser_open_explicit_url".into(),
                tool: "managed_browser_open".into(),
                arguments: json!({"url": url}),
                description:
                    "Open the explicit user-requested HTTP(S) URL in the isolated managed browser"
                        .into(),
                kind: DecisionCandidateKind::Action,
                local_score: 1.0,
            });
        }
        candidates.truncate(limit.min(250));
        return candidates;
    };
    let Some(snapshot_id) = snapshot.get("snapshot_id").and_then(Value::as_str) else {
        candidates.truncate(limit.min(250));
        return candidates;
    };
    let terms = query
        .split(|ch: char| !ch.is_alphanumeric())
        .filter(|term| term.len() > 2)
        .collect::<Vec<_>>();
    if terms.iter().any(|term| {
        matches!(
            *term,
            "inspect" | "read" | "snapshot" | "title" | "text" | "page" | "visible"
        )
    }) {
        candidates.push(DecisionCandidate {
            id: format!("browser_snapshot_{snapshot_id}"),
            tool: "managed_browser_snapshot".into(),
            arguments: json!({}),
            description: "Read a fresh structured snapshot of the current managed-browser page"
                .into(),
            kind: DecisionCandidateKind::Action,
            local_score: 0.95,
        });
    }
    candidates.extend(
        snapshot
            .get("elements")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter(|element| {
                element.get("actionable").and_then(Value::as_bool) == Some(true)
                    && element.get("disabled").and_then(Value::as_bool) != Some(true)
            })
            .filter_map(|element| {
                let id = element.get("id")?.as_str()?;
                let label = element.get("name").and_then(Value::as_str).unwrap_or("");
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
                Some(DecisionCandidate {
                    id: format!("browser_click_{id}"),
                    tool: "managed_browser_click".into(),
                    arguments: json!({"snapshot_id": snapshot_id, "target_id": id}),
                    description: format!("{:?} (browser {role}, visible)", bounded_label(label)),
                    kind: DecisionCandidateKind::Action,
                    local_score: (0.35 + matches as f64 * 0.2).min(1.0),
                })
            })
            .collect::<Vec<_>>(),
    );
    candidates.extend(
        snapshot
            .get("elements")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter(|element| {
                element.get("actionable").and_then(Value::as_bool) == Some(true)
                    && element.get("disabled").and_then(Value::as_bool) != Some(true)
                    && element.get("sensitive").and_then(Value::as_bool) != Some(true)
                    && matches!(
                        element.get("role").and_then(Value::as_str),
                        Some("textbox" | "searchbox")
                    )
            })
            .filter_map(|element| {
                let id = element.get("id")?.as_str()?;
                let label = element.get("name").and_then(Value::as_str).unwrap_or("");
                Some(DecisionCandidate {
                    id: format!("browser_type_{id}"),
                    tool: "managed_browser_type".into(),
                    arguments: json!({
                        "snapshot_id": snapshot_id,
                        "target_id": id,
                        "value": "",
                        "generate_value_with_primary_model": true
                    }),
                    description: format!("{label:?} (browser field, text input)"),
                    kind: DecisionCandidateKind::Action,
                    local_score: 0.45,
                })
            }),
    );
    for element in snapshot
        .get("elements")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|element| {
            element.get("disabled").and_then(Value::as_bool) != Some(true)
                && element.get("role").and_then(Value::as_str) == Some("combobox")
        })
    {
        let Some(id) = element.get("id").and_then(Value::as_str) else {
            continue;
        };
        let label = element.get("name").and_then(Value::as_str).unwrap_or("");
        for option in element
            .get("options")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter(|option| option.get("selected").and_then(Value::as_bool) != Some(true))
            .take(30)
        {
            let Some(value) = option.get("value").and_then(Value::as_str) else {
                continue;
            };
            let option_label = option.get("label").and_then(Value::as_str).unwrap_or(value);
            candidates.push(DecisionCandidate {
                id: format!("browser_select_{}_{}", id, candidates.len()),
                tool: "managed_browser_select".into(),
                arguments: json!({"snapshot_id": snapshot_id, "target_id": id, "value": value}),
                description: format!(
                    "Select option {option_label:?} in the visible managed-browser control {label:?}"
                ),
                kind: DecisionCandidateKind::Action,
                local_score: 0.4,
            });
        }
    }
    if terms.iter().any(|term| {
        matches!(
            *term,
            "scroll" | "below" | "more" | "next" | "down" | "above" | "previous" | "up"
        )
    }) {
        for (direction, page) in [("down", false), ("down", true), ("up", false)] {
            candidates.push(DecisionCandidate {
                id: format!("browser_scroll_{direction}_{}", if page { "page" } else { "small" }),
                tool: "managed_browser_scroll".into(),
                arguments: json!({"snapshot_id": snapshot_id, "direction": direction, "page": page}),
                description: format!(
                    "Scroll the managed browser {direction} by {}",
                    if page { "one page" } else { "a small amount" }
                ),
                kind: DecisionCandidateKind::Action,
                local_score: 0.4,
            });
        }
    }
    candidates.sort_by(|left, right| right.local_score.total_cmp(&left.local_score));
    candidates.truncate(limit.min(250));
    candidates
}

/// Return a bounded, screenshot-free page projection for semantic action and
/// terminal decisions. Sensitive fields and suspicious text are withheld.
pub async fn decision_state(context: &ToolContext) -> Value {
    let managed = browser(context);
    let state = managed.state.lock().await;
    state
        .as_ref()
        .and_then(|connection| connection.last_snapshot.as_ref())
        .map_or_else(|| json!({}), snapshot_decision_state)
}

/// The bounded decision projection of one page snapshot.
pub(crate) fn snapshot_decision_state(snapshot: &Value) -> Value {
    let safe = |text: &str, limit: usize| {
        if crate::decision::outbound_text_may_be_sensitive(text) {
            "[withheld]".into()
        } else {
            text.chars().take(limit).collect::<String>()
        }
    };
    let elements = snapshot
        .get("elements")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|element| element.get("sensitive").and_then(Value::as_bool) != Some(true))
        .take(24)
        .map(|element| {
            let name = element.get("name").and_then(Value::as_str).unwrap_or("");
            let value = element.get("value").and_then(Value::as_str).unwrap_or("");
            json!({
                "id": element.get("id"),
                "role": element.get("role"),
                "name": safe(name, 160),
                "value": safe(value, 160),
                "checked": element.get("checked"),
                "selected": element.get("selected"),
                "disabled": element.get("disabled"),
            })
        })
        .collect::<Vec<_>>();
    json!({
        "url": safe(snapshot.get("url").and_then(Value::as_str).unwrap_or(""), 1000),
        "title": safe(snapshot.get("title").and_then(Value::as_str).unwrap_or(""), 500),
        "visible_text": safe(snapshot.get("text").and_then(Value::as_str).unwrap_or(""), 4000),
        "loading": snapshot.get("loading"),
        "metrics": snapshot.get("metrics"),
        "elements": elements,
    })
}

fn explicit_http_url(value: &str) -> Option<String> {
    value
        .split_whitespace()
        .map(|token| token.trim_matches(|ch: char| ",;:!?()[]{}<>.\"'".contains(ch)))
        .find_map(|token| {
            let url = reqwest::Url::parse(token).ok()?;
            matches!(url.scheme(), "http" | "https").then(|| url.to_string())
        })
}

/// Installed Chromium browsers to try, Edge first, then Chrome.
pub(crate) fn chromium_candidates() -> Vec<PathBuf> {
    #[cfg_attr(not(windows), allow(unused_mut))]
    let mut found = Vec::new();
    #[cfg(windows)]
    {
        let roots = [
            std::env::var_os("PROGRAMFILES").map(PathBuf::from),
            std::env::var_os("PROGRAMFILES(X86)").map(PathBuf::from),
            std::env::var_os("LOCALAPPDATA").map(PathBuf::from),
        ];
        for relative in [
            "Microsoft/Edge/Application/msedge.exe",
            "Google/Chrome/Application/chrome.exe",
        ] {
            if let Some(candidate) = roots
                .iter()
                .flatten()
                .map(|root| root.join(relative))
                .find(|candidate| candidate.is_file())
            {
                found.push(candidate);
            }
        }
    }
    found
}

/// Managed browsers by data folder and, for a swarm worker, its session: one
/// browser process per data folder, one tab per worker.
type BrowserKey = (PathBuf, Option<Uuid>);
static BROWSERS: OnceLock<SyncMutex<HashMap<BrowserKey, Arc<ManagedBrowser>>>> = OnceLock::new();

fn browser(context: &ToolContext) -> Arc<ManagedBrowser> {
    let browsers = BROWSERS.get_or_init(Default::default);
    let mut browsers = browsers.lock();
    let primary = browsers
        .entry((context.data_dir.clone(), None))
        .or_insert_with(|| Arc::new(ManagedBrowser::new(&context.data_dir)))
        .clone();
    if !crate::swarm::is_worker(context.session_id) {
        return primary;
    }
    browsers
        .entry((context.data_dir.clone(), Some(context.session_id)))
        .or_insert_with(|| {
            Arc::new(ManagedBrowser {
                primary: Some(primary),
                ..ManagedBrowser::new(&context.data_dir)
            })
        })
        .clone()
}

/// Open `url` in this agent's managed-browser tab and evaluate `script`
/// there, returning its value.
pub(crate) async fn open_and_evaluate(
    context: &ToolContext,
    url: &str,
    script: &str,
) -> Result<Value> {
    let managed = browser(context);
    managed.navigate(url).await?;
    let mut state = managed.state.lock().await;
    let connection = state.as_mut().expect("browser started");
    let result = ManagedBrowser::cdp(
        connection,
        "Runtime.evaluate",
        json!({"expression": script, "returnByValue": true}),
    )
    .await?;
    Ok(result
        .pointer("/result/value")
        .cloned()
        .unwrap_or(Value::Null))
}

/// Close the tabs a finished swarm worker opened in the managed browser
/// and forget its lane. The browser itself keeps running.
pub(crate) async fn release_worker(data_dir: &Path, session_id: Uuid) {
    let lane = BROWSERS
        .get_or_init(Default::default)
        .lock()
        .remove(&(data_dir.to_path_buf(), Some(session_id)));
    let Some(lane) = lane else {
        return;
    };
    let state = lane.state.lock().await;
    if let Some(connection) = state.as_ref() {
        for tab in &connection.opened_tabs {
            close_tab(connection.port, tab).await;
        }
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct BrowserOpenArgs {
    url: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct BrowserTargetArgs {
    snapshot_id: String,
    target_id: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct BrowserValueArgs {
    snapshot_id: String,
    target_id: String,
    value: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct BrowserScrollArgs {
    snapshot_id: String,
    direction: String,
    #[serde(default)]
    page: bool,
}

struct BrowserOpenTool;
struct BrowserSnapshotTool;
struct BrowserClickTool;
struct BrowserTypeTool;
struct BrowserSelectTool;
struct BrowserHoverTool;
struct BrowserScrollTool;

fn schema<T: JsonSchema>() -> Value {
    serde_json::to_value(schema_for!(T)).expect("schema serializes")
}

#[async_trait]
impl Tool for BrowserOpenTool {
    fn name(&self) -> &'static str {
        "managed_browser_open"
    }
    fn description(&self) -> &'static str {
        "Open an absolute http(s) URL in the isolated POK-Agent managed browser and return a structured snapshot. Use the existing desktop browser tools only when the user asks for their personal browser."
    }
    fn input_schema(&self) -> Value {
        schema::<BrowserOpenArgs>()
    }
    fn risk(&self) -> RiskClass {
        RiskClass::DesktopInput
    }
    async fn execute(&self, arguments: Value, context: &ToolContext) -> Result<Value> {
        let args: BrowserOpenArgs = serde_json::from_value(arguments)?;
        browser(context).navigate(&args.url).await
    }
}

#[async_trait]
impl Tool for BrowserSnapshotTool {
    fn name(&self) -> &'static str {
        "managed_browser_snapshot"
    }
    fn description(&self) -> &'static str {
        "Read the current managed browser page as bounded structured controls and visible text without taking a screenshot."
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object","properties":{}})
    }
    fn risk(&self) -> RiskClass {
        RiskClass::ReadOnly
    }
    async fn execute(&self, _arguments: Value, context: &ToolContext) -> Result<Value> {
        browser(context).snapshot().await
    }
}

macro_rules! target_tool {
    ($type:ty, $name:literal, $description:literal, $operation:literal) => {
        #[async_trait]
        impl Tool for $type {
            fn name(&self) -> &'static str {
                $name
            }
            fn description(&self) -> &'static str {
                $description
            }
            fn input_schema(&self) -> Value {
                schema::<BrowserTargetArgs>()
            }
            fn risk(&self) -> RiskClass {
                RiskClass::DesktopInput
            }
            async fn execute(&self, arguments: Value, context: &ToolContext) -> Result<Value> {
                let args: BrowserTargetArgs = serde_json::from_value(arguments)?;
                browser(context)
                    .act(&args.snapshot_id, &args.target_id, $operation, None)
                    .await
            }
        }
    };
}

target_tool!(
    BrowserClickTool,
    "managed_browser_click",
    "Click one target from the latest managed-browser snapshot and return the settled next snapshot.",
    "click"
);
target_tool!(
    BrowserHoverTool,
    "managed_browser_hover",
    "Hover one target from the latest managed-browser snapshot and return the settled next snapshot.",
    "hover"
);

macro_rules! value_tool {
    ($type:ty, $name:literal, $description:literal, $operation:literal) => {
        #[async_trait]
        impl Tool for $type {
            fn name(&self) -> &'static str {
                $name
            }
            fn description(&self) -> &'static str {
                $description
            }
            fn input_schema(&self) -> Value {
                schema::<BrowserValueArgs>()
            }
            fn risk(&self) -> RiskClass {
                RiskClass::DesktopInput
            }
            async fn execute(&self, arguments: Value, context: &ToolContext) -> Result<Value> {
                let args: BrowserValueArgs = serde_json::from_value(arguments)?;
                browser(context)
                    .act(
                        &args.snapshot_id,
                        &args.target_id,
                        $operation,
                        Some(&args.value),
                    )
                    .await
            }
        }
    };
}

value_tool!(
    BrowserTypeTool,
    "managed_browser_type",
    "Type an exact supplied value into one target from the latest managed-browser snapshot and return the settled next snapshot.",
    "type"
);

#[async_trait]
impl Tool for BrowserScrollTool {
    fn name(&self) -> &'static str {
        "managed_browser_scroll"
    }
    fn description(&self) -> &'static str {
        "Scroll the latest managed-browser page up or down, then return a settled structured snapshot."
    }
    fn input_schema(&self) -> Value {
        schema::<BrowserScrollArgs>()
    }
    fn risk(&self) -> RiskClass {
        RiskClass::DesktopInput
    }
    async fn execute(&self, arguments: Value, context: &ToolContext) -> Result<Value> {
        let args: BrowserScrollArgs = serde_json::from_value(arguments)?;
        if !matches!(args.direction.as_str(), "up" | "down") {
            return Err(PokError::Tool(
                "browser scroll direction must be up or down".into(),
            ));
        }
        let managed = browser(context);
        managed.ensure_started().await?;
        let mut state = managed.state.lock().await;
        let connection = state.as_mut().expect("browser started");
        if connection.snapshot_id.as_deref() != Some(args.snapshot_id.as_str()) {
            return Err(PokError::Tool(
                "stale managed-browser snapshot; take a new snapshot before scrolling".into(),
            ));
        }
        let multiplier = if args.direction == "up" { -1.0 } else { 1.0 };
        let fraction = if args.page { 0.85 } else { 0.35 };
        ManagedBrowser::cdp(
            connection,
            "Runtime.evaluate",
            json!({"expression": format!("scrollBy(0, innerHeight * {})", multiplier * fraction)}),
        )
        .await?;
        tokio::time::sleep(Duration::from_millis(300)).await;
        ManagedBrowser::snapshot_locked(connection).await
    }
}
value_tool!(
    BrowserSelectTool,
    "managed_browser_select",
    "Select an exact supplied option value in one target from the latest managed-browser snapshot and return the settled next snapshot.",
    "select"
);

pub fn register_managed_browser_tools(registry: &mut ToolRegistry) {
    registry.register(BrowserOpenTool);
    registry.register(BrowserSnapshotTool);
    registry.register(BrowserClickTool);
    registry.register(BrowserTypeTool);
    registry.register(BrowserSelectTool);
    registry.register(BrowserHoverTool);
    registry.register(BrowserScrollTool);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn devtools_active_port_reads_the_first_line() {
        assert_eq!(
            devtools_active_port("9222\n/devtools/browser/abc\n"),
            Some(9222)
        );
        assert_eq!(devtools_active_port("0\n"), None);
        assert_eq!(devtools_active_port(""), None);
    }

    #[tokio::test]
    async fn a_browser_already_running_with_the_profile_is_reused() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                let mut buffer = [0; 1024];
                let _ = socket.read(&mut buffer).await;
                let body = "{\"Browser\":\"Edg/153\"}";
                let _ = socket
                    .write_all(
                        format!(
                            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                            body.len()
                        )
                        .as_bytes(),
                    )
                    .await;
            }
        });
        let data = tempfile::tempdir().unwrap();
        let mut browser = ManagedBrowser::new(data.path());
        // Launching would fail: the running browser must be reused instead.
        browser.executable_override = Some(data.path().join("missing-browser.exe"));
        std::fs::create_dir_all(&browser.profile_dir).unwrap();
        std::fs::write(
            browser.profile_dir.join("DevToolsActivePort"),
            format!("{port}\n/devtools/browser/id\n"),
        )
        .unwrap();
        browser.ensure_started().await.unwrap();
        let state = browser.state.lock().await;
        let connection = state.as_ref().unwrap();
        assert_eq!(connection.port, port);
        assert!(connection.child.is_none());
    }

    #[tokio::test]
    async fn a_browser_that_cannot_start_suggests_the_users_own_browser() {
        let data = tempfile::tempdir().unwrap();
        let mut browser = ManagedBrowser::new(data.path());
        browser.executable_override = Some(data.path().join("missing-browser.exe"));
        let error = browser.ensure_started().await.unwrap_err().to_string();
        assert!(error.contains("Start-Process"), "{error}");
        assert!(!error.contains("tool error: tool error"), "{error}");
    }

    /// Two swarm workers drive their own tabs of one managed browser at the
    /// same time. Set POK_AI_BROWSER_EXECUTABLE, POK_AI_LANE_A_URL, and
    /// POK_AI_LANE_B_URL (pages with "Mark A" / "Mark B" buttons).
    #[tokio::test]
    #[ignore = "needs a Chromium; set POK_AI_BROWSER_EXECUTABLE, POK_AI_LANE_A_URL, POK_AI_LANE_B_URL"]
    async fn swarm_workers_get_their_own_managed_tabs() {
        let a = std::env::var("POK_AI_LANE_A_URL").expect("POK_AI_LANE_A_URL");
        let b = std::env::var("POK_AI_LANE_B_URL").expect("POK_AI_LANE_B_URL");
        let data = tempfile::tempdir().unwrap();
        let platform = Arc::new(crate::platform::MockDesktop::default());
        let main = crate::builtins::tests::mock_input_context(platform.clone(), &data);
        let first = crate::builtins::tests::mock_input_context(platform.clone(), &data);
        let second = crate::builtins::tests::mock_input_context(platform, &data);
        // The main agent's browser is running with a page of its own.
        browser(&main).navigate(&a).await.unwrap();
        crate::swarm::set_test_worker(first.session_id, Some("worker a"));
        crate::swarm::set_test_worker(second.session_id, Some("worker b"));
        let (lane_a, lane_b) = (browser(&first), browser(&second));
        let (page_a, page_b) = tokio::join!(lane_a.navigate(&a), lane_b.navigate(&b));
        let (page_a, page_b) = (page_a.unwrap(), page_b.unwrap());
        assert_eq!(page_a["title"], "Lane A");
        assert_eq!(page_b["title"], "Lane B");
        let tab = |context: &ToolContext| {
            let managed = browser(context);
            async move {
                managed
                    .state
                    .lock()
                    .await
                    .as_ref()
                    .unwrap()
                    .page_target
                    .clone()
                    .unwrap()
            }
        };
        let control = |page: &Value, role: &str, name: &str| {
            page["elements"]
                .as_array()
                .unwrap()
                .iter()
                .find(|element| element["role"] == role && element["name"] == name)
                .map(|element| element["id"].as_str().unwrap().to_owned())
                .unwrap()
        };
        let mut registry = ToolRegistry::new();
        register_managed_browser_tools(&mut registry);
        let click = |page: &Value, role: &str, name: &str| json!({"snapshot_id": page["snapshot_id"], "target_id": control(page, role, name)});
        let (clicked_a, clicked_b) = tokio::join!(
            registry.call(
                "managed_browser_click",
                click(&page_a, "button", "Mark A"),
                &first
            ),
            registry.call(
                "managed_browser_click",
                click(&page_b, "button", "Mark B"),
                &second
            ),
        );
        let (clicked_a, clicked_b) = (clicked_a.unwrap(), clicked_b.unwrap());
        assert_eq!(clicked_a["title"], "Lane A clicked");
        assert_eq!(clicked_b["title"], "Lane B clicked");
        // A link that opens a new tab, from a background tab: the worker
        // follows it, and not a tab someone else opens at the same moment.
        let port = browser(&main).state.lock().await.as_ref().unwrap().port;
        let first_tab = tab(&first).await;
        let (followed, other) = tokio::join!(
            registry.call(
                "managed_browser_click",
                click(&clicked_a, "link", "Open B"),
                &first
            ),
            async {
                tokio::time::sleep(Duration::from_millis(100)).await;
                open_background_tab(port).await.unwrap()
            },
        );
        let followed = followed.unwrap();
        assert!(
            followed["url"]
                .as_str()
                .unwrap()
                .ends_with("lane-b.html?from=a"),
            "{followed}"
        );
        let popup = tab(&first).await;
        assert_ne!(popup, other);
        let (main_tab, tab_a, tab_b) = (tab(&main).await, popup, tab(&second).await);
        assert!(main_tab != tab_a && tab_a != tab_b && main_tab != tab_b);
        // A finished worker's tab is closed; the main agent's stays.
        release_worker(data.path(), first.session_id).await;
        release_worker(data.path(), second.session_id).await;
        crate::swarm::set_test_worker(first.session_id, None);
        crate::swarm::set_test_worker(second.session_id, None);
        tokio::time::sleep(Duration::from_millis(300)).await;
        let open = ManagedBrowser::page_targets(port).await.unwrap();
        assert!(open.iter().any(|(id, _, _)| *id == main_tab));
        assert!(
            !open
                .iter()
                .any(|(id, _, _)| [&first_tab, &tab_a, &tab_b].contains(&id))
        );
        close_tab(port, &other).await;
    }

    #[test]
    fn managed_browser_tools_have_a_dedicated_contract() {
        let mut registry = ToolRegistry::new();
        register_managed_browser_tools(&mut registry);
        assert!(registry.names().contains(&"managed_browser_open".into()));
        assert!(
            registry
                .names()
                .contains(&"managed_browser_snapshot".into())
        );
        assert!(registry.input_schema("managed_browser_click").is_some());
        assert!(registry.input_schema("managed_browser_scroll").is_some());
    }

    #[test]
    fn explicit_http_url_candidates_preserve_user_url_and_reject_other_schemes() {
        assert_eq!(
            explicit_http_url("Open https://Example.com/Docs/Index.html."),
            Some("https://example.com/Docs/Index.html".into())
        );
        assert_eq!(explicit_http_url("Open javascript:alert(1)"), None);
    }
}
