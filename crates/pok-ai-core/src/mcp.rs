//! Local Model Context Protocol servers over stdio: spawn a configured server,
//! complete the JSON-RPC handshake, list its tools, and expose each of them as
//! a typed harness tool under the `mcp` group.

use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    process::Stdio,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use async_trait::async_trait;
use parking_lot::Mutex as SyncMutex;
use serde_json::{Value, json};
use tokio::{
    io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader},
    process::{Child, Command},
    sync::{Mutex as AsyncMutex, oneshot},
};

use crate::{
    PokError, Result,
    config::{McpConfig, McpRisk, McpServerConfig},
    policy::RiskClass,
    tool::{Tool, ToolContext, ToolRegistry},
};

/// MCP revision the client announces; every 2024-11+ server understands it.
const PROTOCOL_VERSION: &str = "2025-06-18";
const MAX_TOOLS: usize = 128;
const MAX_TOOL_PAGES: usize = 16;
const MAX_CONTENT_ITEMS: usize = 64;
const MAX_RESULT_CHARS: usize = 64 * 1024;
const MAX_TEXT_CHARS: usize = 32 * 1024;
const MAX_SCHEMA_CHARS: usize = 16 * 1024;
const MAX_DESCRIPTION_CHARS: usize = 600;

/// What one server did during registration, for the settings page.
#[derive(Debug, Clone)]
pub struct McpServerHealth {
    pub name: String,
    pub ok: bool,
    pub tools: usize,
    pub detail: Option<String>,
}

/// What `register_mcp_tools` did, so the caller can report failures without
/// failing the session.
#[derive(Debug, Default)]
pub struct McpRegistration {
    pub servers: usize,
    pub tools: usize,
    pub warnings: Vec<String>,
    pub statuses: Vec<McpServerHealth>,
}

/// Connect every enabled server and register its tools as `mcp__<server>__<tool>`.
/// A server that does not start is reported as a warning; the rest still load.
pub async fn register_mcp_tools(
    registry: &mut ToolRegistry,
    config: &McpConfig,
) -> McpRegistration {
    let mut registration = McpRegistration::default();
    if !config.enabled {
        return registration;
    }
    let enabled = config
        .servers
        .iter()
        .filter(|spec| spec.enabled)
        .collect::<Vec<_>>();
    if enabled.is_empty() {
        return registration;
    }
    let connected =
        futures::future::join_all(enabled.iter().map(|spec| connect_server(spec, config))).await;
    for (spec, result) in enabled.into_iter().zip(connected) {
        match result {
            Ok(server) => {
                registration.servers += 1;
                let risk = risk_from_config(spec.risk, spec.auto_approve);
                let call_timeout = Duration::from_secs(
                    spec.call_timeout_seconds
                        .unwrap_or(config.call_timeout_seconds),
                );
                registration.statuses.push(McpServerHealth {
                    name: spec.name.clone(),
                    ok: true,
                    tools: server.tools.len(),
                    detail: None,
                });
                for descriptor in server.tools {
                    registration.tools += 1;
                    registry.register(McpTool::new(&server.client, descriptor, risk, call_timeout));
                }
            }
            Err(error) => {
                registration.warnings.push(format!(
                    "MCP server {:?} is unavailable: {error}",
                    spec.name
                ));
                registration.statuses.push(McpServerHealth {
                    name: spec.name.clone(),
                    ok: false,
                    tools: 0,
                    detail: Some(error.to_string()),
                });
            }
        }
    }
    registration
}

/// Connect, list tools, and disconnect: the settings page's "Test" action.
pub async fn probe_server(spec: &McpServerConfig, config: &McpConfig) -> Result<Vec<String>> {
    let server = connect_server(spec, config).await?;
    Ok(server.tools.into_iter().map(|tool| tool.name).collect())
}

/// Desktop-managed MCP servers, stored beside the conversation database. The
/// settings page edits this file; `pok-ai.toml` servers still apply and an
/// app-managed server with the same name overrides the file entry.
pub struct McpServerStore {
    path: PathBuf,
}

impl McpServerStore {
    pub fn open(data_dir: &Path) -> Self {
        Self {
            path: data_dir.join("mcp-servers.json"),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn load(&self) -> Vec<McpServerConfig> {
        std::fs::read_to_string(&self.path)
            .ok()
            .and_then(|text| serde_json::from_str::<Vec<McpServerConfig>>(&text).ok())
            .unwrap_or_default()
    }

    pub fn save(&self, servers: &[McpServerConfig]) -> Result<()> {
        let encoded = serde_json::to_string_pretty(servers)?;
        crate::memory::atomic_write(&self.path, encoded.as_bytes())
    }
}

/// Merge the config-file servers with the desktop-managed store.
pub fn merge_servers(config: &McpConfig, stored: &[McpServerConfig]) -> McpConfig {
    let mut merged = config.clone();
    for server in stored {
        match merged
            .servers
            .iter_mut()
            .find(|item| item.name == server.name)
        {
            Some(existing) => *existing = server.clone(),
            None => merged.servers.push(server.clone()),
        }
    }
    merged
}

async fn connect_server(spec: &McpServerConfig, config: &McpConfig) -> Result<McpServer> {
    let startup = Duration::from_secs(
        spec.startup_timeout_seconds
            .unwrap_or(config.startup_timeout_seconds),
    );
    let client = Arc::new(McpClient::connect(spec).await?);
    client.initialize(startup).await?;
    let mut tools = client.list_tools(startup).await?;
    if !spec.tools.is_empty() {
        tools.retain(|tool| spec.tools.iter().any(|name| name == &tool.name));
    }
    Ok(McpServer { client, tools })
}

struct McpServer {
    client: Arc<McpClient>,
    tools: Vec<McpToolDescriptor>,
}

struct McpToolDescriptor {
    name: String,
    description: String,
    input_schema: Value,
}

fn risk_from_config(risk: McpRisk, auto_approve: bool) -> RiskClass {
    if auto_approve {
        return RiskClass::ReadOnly;
    }
    match risk {
        McpRisk::ReadOnly => RiskClass::ReadOnly,
        McpRisk::WorkspaceWrite => RiskClass::WorkspaceWrite,
        McpRisk::ProcessExecution => RiskClass::ProcessExecution,
        McpRisk::DesktopInput => RiskClass::DesktopInput,
        McpRisk::HighImpact => RiskClass::HighImpact,
    }
}

fn tool_name(server: &str, tool: &str) -> String {
    let server = sanitize(server);
    let tool = sanitize(tool);
    let name = format!("mcp__{server}__{tool}");
    name.chars().take(64).collect()
}

fn sanitize(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '_' || character == '-' {
                character.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect()
}

/// Intern a dynamic name/description once so the `Tool` trait's `&'static str`
/// accessors stay cheap. A session registers a bounded number of MCP tools, and
/// identical strings are shared across sessions.
fn intern(value: &str) -> &'static str {
    static NAMES: OnceLock<SyncMutex<HashSet<&'static str>>> = OnceLock::new();
    let mut names = NAMES.get_or_init(|| SyncMutex::new(HashSet::new())).lock();
    if let Some(existing) = names.get(value) {
        return existing;
    }
    let leaked: &'static str = Box::leak(value.to_owned().into_boxed_str());
    names.insert(leaked);
    leaked
}

struct McpTool {
    name: &'static str,
    description: &'static str,
    input_schema: Value,
    tool: String,
    server: String,
    risk: RiskClass,
    client: Arc<McpClient>,
    call_timeout: Duration,
}

impl McpTool {
    fn new(
        client: &Arc<McpClient>,
        descriptor: McpToolDescriptor,
        risk: RiskClass,
        call_timeout: Duration,
    ) -> Self {
        let name = intern(&tool_name(&client.name, &descriptor.name));
        let description = descriptor.description.trim();
        let description = if description.is_empty() {
            format!(
                "{} ({} tool, no description provided)",
                descriptor.name, client.name
            )
        } else {
            format!("[{}] {description}", client.name)
        };
        Self {
            name,
            description: intern(&truncate(&description, MAX_DESCRIPTION_CHARS)),
            input_schema: sanitize_schema(&descriptor.input_schema),
            tool: descriptor.name,
            server: client.name.clone(),
            risk,
            client: client.clone(),
            call_timeout,
        }
    }
}

#[async_trait]
impl Tool for McpTool {
    fn name(&self) -> &'static str {
        self.name
    }

    fn description(&self) -> &'static str {
        self.description
    }

    fn input_schema(&self) -> Value {
        self.input_schema.clone()
    }

    fn risk(&self) -> RiskClass {
        self.risk
    }

    /// One approved MCP call can be remembered for the whole server, matching
    /// how the dashboard remembers a tool for a session.
    fn approval_key(&self, _arguments: &Value) -> Option<String> {
        Some(format!("mcp:{}", self.server))
    }

    async fn execute(&self, arguments: Value, context: &ToolContext) -> Result<Value> {
        if context.cancellation.is_cancelled() {
            return Err(PokError::Cancelled);
        }
        tokio::select! {
            () = context.cancellation.cancelled() => Err(PokError::Cancelled),
            result = self.client.call_tool(&self.tool, arguments, self.call_timeout) => result,
        }
    }
}

/// A JSON-RPC 2.0 client for one stdio MCP server. Requests are correlated by
/// id; the stderr stream is discarded because server diagnostics are not part
/// of the protocol and can contain private data.
pub struct McpClient {
    name: String,
    writer: AsyncMutex<Box<dyn AsyncWrite + Send + Unpin>>,
    pending: Arc<SyncMutex<HashMap<u64, oneshot::Sender<Value>>>>,
    next_id: AtomicU64,
    /// Kept alive so the child process lives as long as the session; dropping
    /// it kills the server.
    _child: SyncMutex<Option<McpChild>>,
}

/// Owns the server process and tears down its whole tree. A Windows npm
/// launcher runs through `cmd.exe`, so the real server is a grandchild; the
/// shim must be killed with `/T` before it disappears.
struct McpChild {
    child: Child,
}

impl Drop for McpChild {
    fn drop(&mut self) {
        #[cfg(windows)]
        if let Some(pid) = self.child.id() {
            let _ = std::process::Command::new("taskkill")
                .args(["/PID", &pid.to_string(), "/T", "/F"])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
        }
        let _ = self.child.start_kill();
    }
}

/// Windows `npx`, `npm`, and friends are `.cmd` batch shims that
/// `CreateProcess` cannot execute, and `npx` has no `.cmd` extension in the
/// command line as typed. Route those launchers, and any explicit `.cmd`/`.bat`
/// command, through `cmd.exe` so the configured command works as written.
fn launcher(command: &str, args: &[String]) -> (String, Vec<String>) {
    #[cfg(windows)]
    {
        let lower = command.trim().to_ascii_lowercase();
        let known_shim = matches!(
            lower.as_str(),
            "npx" | "npm" | "npm.cmd" | "pnpm" | "pnpm.cmd" | "yarn" | "yarn.cmd" | "bunx" | "uvx"
        );
        if known_shim || lower.ends_with(".cmd") || lower.ends_with(".bat") {
            let mut wrapped = Vec::with_capacity(args.len() + 4);
            wrapped.extend(["/d".to_owned(), "/s".to_owned(), "/c".to_owned()]);
            wrapped.push(command.to_owned());
            wrapped.extend(args.iter().cloned());
            return ("cmd.exe".to_owned(), wrapped);
        }
    }
    (command.to_owned(), args.to_vec())
}

impl McpClient {
    pub async fn connect(spec: &McpServerConfig) -> Result<Self> {
        let (program, args) = launcher(&spec.command, &spec.args);
        let mut command = Command::new(&program);
        command
            .args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        if let Some(cwd) = &spec.cwd {
            command.current_dir(cwd);
        }
        for (key, value) in &spec.env {
            if value.is_empty() {
                // `NAME = ""` passes the process environment through, so
                // secrets stay out of the configuration file.
                if let Ok(passthrough) = std::env::var(key) {
                    command.env(key, passthrough);
                }
            } else {
                command.env(key, value);
            }
        }
        #[cfg(windows)]
        crate::process_window::hide_tokio(&mut command);
        let mut child = command.spawn().map_err(|error| {
            PokError::Tool(format!(
                "could not start MCP server {:?}: {error}",
                spec.name
            ))
        })?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| PokError::Tool("MCP server stdin was not piped".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| PokError::Tool("MCP server stdout was not piped".into()))?;
        Ok(Self::from_io(
            spec.name.clone(),
            Box::new(stdin),
            Box::new(stdout),
            Some(McpChild { child }),
        ))
    }

    fn from_io(
        name: String,
        writer: Box<dyn AsyncWrite + Send + Unpin>,
        reader: Box<dyn AsyncRead + Send + Unpin>,
        child: Option<McpChild>,
    ) -> Self {
        let pending = Arc::new(SyncMutex::new(HashMap::new()));
        let pending_for_reader = pending.clone();
        tokio::spawn(async move {
            reader_loop(reader, pending_for_reader).await;
        });
        Self {
            name,
            writer: AsyncMutex::new(writer),
            pending,
            next_id: AtomicU64::new(0),
            _child: SyncMutex::new(child),
        }
    }

    async fn initialize(&self, timeout: Duration) -> Result<()> {
        self.request(
            "initialize",
            json!({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "pok-ai", "version": env!("CARGO_PKG_VERSION")},
            }),
            timeout,
        )
        .await?;
        self.notify("notifications/initialized", json!({})).await
    }

    async fn list_tools(&self, timeout: Duration) -> Result<Vec<McpToolDescriptor>> {
        let mut tools = Vec::new();
        let mut cursor: Option<String> = None;
        for _ in 0..MAX_TOOL_PAGES {
            let params = cursor
                .as_ref()
                .map_or_else(|| json!({}), |cursor| json!({"cursor": cursor}));
            let result = self.request("tools/list", params, timeout).await?;
            for item in result
                .get("tools")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                let Some(name) = item.get("name").and_then(Value::as_str) else {
                    continue;
                };
                tools.push(McpToolDescriptor {
                    name: name.to_owned(),
                    description: item
                        .get("description")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    input_schema: item
                        .get("inputSchema")
                        .cloned()
                        .unwrap_or_else(|| json!({"type": "object"})),
                });
                if tools.len() >= MAX_TOOLS {
                    return Ok(tools);
                }
            }
            cursor = result
                .get("nextCursor")
                .and_then(Value::as_str)
                .map(str::to_owned);
            if cursor.is_none() {
                break;
            }
        }
        Ok(tools)
    }

    async fn call_tool(&self, tool: &str, arguments: Value, timeout: Duration) -> Result<Value> {
        let result = self
            .request(
                "tools/call",
                json!({"name": tool, "arguments": arguments}),
                timeout,
            )
            .await?;
        project_tool_call(&self.name, tool, result)
    }

    async fn notify(&self, method: &str, params: Value) -> Result<()> {
        self.write_line(&json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }))
        .await
    }

    async fn request(&self, method: &str, params: Value, timeout: Duration) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed) + 1;
        let (sender, receiver) = oneshot::channel();
        self.pending.lock().insert(id, sender);
        self.write_line(&json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        }))
        .await?;
        match tokio::time::timeout(timeout, receiver).await {
            Ok(Ok(value)) => response_result(value),
            Ok(Err(_)) => Err(PokError::Tool(format!(
                "MCP server {} closed the connection",
                self.name
            ))),
            Err(_) => {
                self.pending.lock().remove(&id);
                Err(PokError::Tool(format!(
                    "MCP server {} did not answer {method} within {}s",
                    self.name,
                    timeout.as_secs()
                )))
            }
        }
    }

    async fn write_line(&self, payload: &Value) -> Result<()> {
        let mut encoded = serde_json::to_vec(payload)
            .map_err(|error| PokError::Tool(format!("MCP request encoding failed: {error}")))?;
        encoded.push(b'\n');
        let mut writer = self.writer.lock().await;
        writer.write_all(&encoded).await.map_err(|error| {
            PokError::Tool(format!("MCP server {} write failed: {error}", self.name))
        })?;
        writer.flush().await.map_err(|error| {
            PokError::Tool(format!("MCP server {} flush failed: {error}", self.name))
        })
    }
}

async fn reader_loop(
    reader: Box<dyn AsyncRead + Send + Unpin>,
    pending: Arc<SyncMutex<HashMap<u64, oneshot::Sender<Value>>>>,
) {
    let mut lines = BufReader::new(reader).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        let Ok(value) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        let Some(id) = value.get("id").and_then(Value::as_u64) else {
            // Notifications and server-initiated requests (sampling, roots)
            // are not supported by the harness; ignoring them is valid MCP.
            continue;
        };
        if let Some(sender) = pending.lock().remove(&id) {
            let _ = sender.send(value);
        }
    }
    // The server exited: fail every in-flight request instead of waiting for
    // its timeout.
    pending.lock().clear();
}

fn response_result(value: Value) -> Result<Value> {
    if let Some(error) = value.get("error") {
        let code = error.get("code").and_then(Value::as_i64).unwrap_or(0);
        let message = error
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("unknown error");
        return Err(PokError::Tool(format!(
            "MCP error {code}: {}",
            truncate(message, 500)
        )));
    }
    Ok(value.get("result").cloned().unwrap_or(Value::Null))
}

fn project_tool_call(server: &str, tool: &str, result: Value) -> Result<Value> {
    let is_error = result
        .get("isError")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let mut content = Vec::new();
    let mut total = 0usize;
    for item in result
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .take(MAX_CONTENT_ITEMS)
    {
        if total >= MAX_RESULT_CHARS {
            break;
        }
        let kind = item.get("type").and_then(Value::as_str).unwrap_or("text");
        let projected = match kind {
            "text" => {
                let text = truncate(
                    item.get("text").and_then(Value::as_str).unwrap_or_default(),
                    MAX_TEXT_CHARS,
                );
                total += text.len();
                json!({"type": "text", "text": text})
            }
            "image" => {
                let mime = item.get("mimeType").and_then(Value::as_str).unwrap_or("");
                let data = item.get("data").and_then(Value::as_str).unwrap_or_default();
                if !mime.eq_ignore_ascii_case("image/png") || data.is_empty() {
                    continue;
                }
                total += data.len();
                json!({"type": "image", "png_base64": data})
            }
            "resource" => {
                let resource = item.get("resource").cloned().unwrap_or(Value::Null);
                let text = resource
                    .get("text")
                    .and_then(Value::as_str)
                    .map(|text| truncate(text, MAX_TEXT_CHARS))
                    .unwrap_or_else(|| truncate(&resource.to_string(), 4_000));
                total += text.len();
                json!({"type": "text", "text": text})
            }
            other => {
                let encoded = truncate(&item.to_string(), 4_000);
                total += encoded.len();
                json!({"type": other, "json": encoded})
            }
        };
        content.push(projected);
    }
    if is_error {
        let message = content
            .iter()
            .filter_map(|item| item.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n");
        return Err(PokError::Tool(format!(
            "{server} {tool} failed: {}",
            truncate(message.trim(), 2_000)
        )));
    }
    let mut value = json!({
        "server": server,
        "tool": tool,
        "is_error": false,
        "content": content,
    });
    if let Some(structured) = result.get("structuredContent") {
        value["structured_content"] = json!(truncate(&structured.to_string(), 8_000));
    }
    Ok(value)
}

fn sanitize_schema(schema: &Value) -> Value {
    if schema.is_object() && schema.to_string().len() <= MAX_SCHEMA_CHARS {
        schema.clone()
    } else {
        json!({"type": "object", "additionalProperties": true})
    }
}

fn truncate(value: &str, limit: usize) -> String {
    if value.chars().count() <= limit {
        return value.to_owned();
    }
    let mut output = value
        .chars()
        .take(limit.saturating_sub(1))
        .collect::<String>();
    output.push('…');
    output
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    fn mock_client<F>(respond: F) -> McpClient
    where
        F: Fn(&str, &Value, u64) -> Value + Send + Sync + 'static,
    {
        let (client_writer, server_reader) = tokio::io::duplex(256 * 1024);
        let (mut server_writer, client_reader) = tokio::io::duplex(256 * 1024);
        tokio::spawn(async move {
            let mut lines = BufReader::new(server_reader).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let Ok(value) = serde_json::from_str::<Value>(&line) else {
                    continue;
                };
                let Some(id) = value.get("id").and_then(Value::as_u64) else {
                    continue;
                };
                let method = value
                    .get("method")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let params = value.get("params").cloned().unwrap_or(json!({}));
                let response = respond(method, &params, id);
                let mut encoded = serde_json::to_vec(&response).unwrap();
                encoded.push(b'\n');
                if server_writer.write_all(&encoded).await.is_err() {
                    break;
                }
                let _ = server_writer.flush().await;
            }
        });
        McpClient::from_io(
            "mock".to_owned(),
            Box::new(client_writer),
            Box::new(client_reader),
            None,
        )
    }

    fn ok(id: u64, result: Value) -> Value {
        json!({"jsonrpc": "2.0", "id": id, "result": result})
    }

    #[tokio::test]
    async fn handshake_and_paginated_tool_listing_work() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_for_mock = calls.clone();
        let client = mock_client(move |method, params, id| match method {
            "initialize" => ok(
                id,
                json!({"protocolVersion": PROTOCOL_VERSION, "capabilities": {}, "serverInfo": {"name": "mock", "version": "1"}}),
            ),
            "tools/list" => {
                calls_for_mock.fetch_add(1, Ordering::Relaxed);
                if params.get("cursor").is_none() {
                    ok(
                        id,
                        json!({"tools": [{"name": "first", "description": "First", "inputSchema": {"type": "object"}}], "nextCursor": "page-2"}),
                    )
                } else {
                    ok(
                        id,
                        json!({"tools": [{"name": "second", "description": "Second", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}}}]}),
                    )
                }
            }
            other => panic!("unexpected method {other}"),
        });
        client
            .initialize(Duration::from_secs(5))
            .await
            .expect("initialize");
        let tools = client
            .list_tools(Duration::from_secs(5))
            .await
            .expect("list");
        assert_eq!(
            tools
                .iter()
                .map(|tool| tool.name.as_str())
                .collect::<Vec<_>>(),
            ["first", "second"]
        );
        assert_eq!(calls.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn tool_calls_project_text_images_and_errors() {
        let client = mock_client(|method, _params, id| match method {
            "tools/call" => ok(
                id,
                json!({
                    "content": [
                        {"type": "text", "text": "pong"},
                        {"type": "image", "mimeType": "image/png", "data": "QUJD"},
                        {"type": "image", "mimeType": "image/jpeg", "data": "zzz"},
                    ],
                    "isError": false,
                }),
            ),
            other => panic!("unexpected method {other}"),
        });
        let result = client
            .call_tool("ping", json!({}), Duration::from_secs(5))
            .await
            .expect("call");
        assert_eq!(result["content"][0]["text"], "pong");
        assert_eq!(result["content"][1]["png_base64"], "QUJD");
        // Non-PNG images cannot ride the PNG-only image channel.
        assert_eq!(result["content"].as_array().unwrap().len(), 2);

        let failing = mock_client(|_method, _params, id| {
            ok(
                id,
                json!({"content": [{"type": "text", "text": "engine not installed"}], "isError": true}),
            )
        });
        let error = failing
            .call_tool("ping", json!({}), Duration::from_secs(5))
            .await
            .expect_err("isError becomes a tool failure");
        assert!(error.to_string().contains("engine not installed"));
    }

    #[tokio::test]
    async fn protocol_errors_and_disconnects_fail_fast() {
        let failing = mock_client(
            |_method, _params, id| json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32601, "message": "method not found"}}),
        );
        let error = failing
            .request("tools/list", json!({}), Duration::from_secs(5))
            .await
            .expect_err("json-rpc error");
        assert!(error.to_string().contains("-32601"));

        // Server closes immediately: the request must not wait for its timeout.
        let (client_writer, server_reader) = tokio::io::duplex(4096);
        let (server_writer, client_reader) = tokio::io::duplex(4096);
        drop(server_reader);
        drop(server_writer);
        let closed = McpClient::from_io(
            "closed".to_owned(),
            Box::new(client_writer),
            Box::new(client_reader),
            None,
        );
        let started = std::time::Instant::now();
        let error = closed
            .request("tools/list", json!({}), Duration::from_secs(30))
            .await
            .expect_err("closed server");
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "waited for the timeout"
        );
        let message = error.to_string();
        assert!(
            message.contains("closed the connection")
                || message.contains("write failed")
                || message.contains("flush failed"),
            "unexpected error: {message}"
        );
    }

    #[cfg(windows)]
    #[test]
    fn windows_batch_launchers_route_through_cmd() {
        let (program, args) = launcher("npx", &["-y".into(), "package".into(), "mcp".into()]);
        assert_eq!(program, "cmd.exe");
        assert_eq!(
            args.iter().map(String::as_str).collect::<Vec<_>>(),
            ["/d", "/s", "/c", "npx", "-y", "package", "mcp"]
        );
        assert_eq!(launcher("C:\\tools\\server.cmd", &[]).0, "cmd.exe");
        assert_eq!(launcher("powershell", &[]).0, "powershell");
        assert_eq!(launcher("node.exe", &[]).0, "node.exe");
    }

    /// Manual end-to-end check: downloads and speaks MCP to a real server.
    /// Run with `cargo test -p pok-ai-core --lib npx_mcp_server_smoke_test --
    /// --ignored --nocapture` on a machine with Node 22.19+ and network.
    #[cfg(windows)]
    #[tokio::test]
    #[ignore = "downloads and launches a real MCP server over npx"]
    async fn npx_mcp_server_smoke_test() {
        let spec = McpServerConfig {
            name: "everything".into(),
            command: "npx".into(),
            enabled: true,
            args: vec![
                "-y".into(),
                "@modelcontextprotocol/server-everything".into(),
            ],
            env: Default::default(),
            cwd: None,
            auto_approve: false,
            risk: McpRisk::ProcessExecution,
            tools: Vec::new(),
            startup_timeout_seconds: Some(180),
            call_timeout_seconds: Some(60),
        };
        let mut registry = ToolRegistry::new();
        let registration = register_mcp_tools(
            &mut registry,
            &McpConfig {
                servers: vec![spec],
                startup_timeout_seconds: 180,
                ..McpConfig::default()
            },
        )
        .await;
        assert!(
            registration.warnings.is_empty(),
            "warnings: {:?}",
            registration.warnings
        );
        assert!(registration.tools > 0, "no tools registered");
        assert!(
            registry
                .names()
                .iter()
                .any(|name| name.starts_with("mcp__everything__"))
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn connect_spawns_a_stdio_server_and_registers_its_tools() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let script_path = dir.path().join("mock-mcp.sh");
        std::fs::write(
            &script_path,
            r#"#!/bin/sh
while IFS= read -r line; do
  id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9]*\).*/\1/p')
  case "$line" in
    *initialize*) result='{"protocolVersion":"2025-06-18","capabilities":{},"serverInfo":{"name":"mock","version":"0"}}' ;;
    *tools/list*) result='{"tools":[{"name":"ping","description":"Ping","inputSchema":{"type":"object"}}]}' ;;
    *tools/call*) result='{"content":[{"type":"text","text":"pong"}],"isError":false}' ;;
    *) result='{}' ;;
  esac
  printf '{"jsonrpc":"2.0","id":%s,"result":%s}\n' "$id" "$result"
done
"#,
        )
        .unwrap();
        std::fs::set_permissions(&script_path, std::fs::Permissions::from_mode(0o755)).unwrap();

        let spec = McpServerConfig {
            name: "mock".into(),
            command: script_path.display().to_string(),
            enabled: true,
            args: Vec::new(),
            env: Default::default(),
            cwd: None,
            auto_approve: false,
            risk: McpRisk::ProcessExecution,
            tools: Vec::new(),
            startup_timeout_seconds: None,
            call_timeout_seconds: None,
        };
        let mut registry = ToolRegistry::new();
        let registration = register_mcp_tools(
            &mut registry,
            &McpConfig {
                servers: vec![spec],
                ..McpConfig::default()
            },
        )
        .await;
        assert!(
            registration.warnings.is_empty(),
            "{:?}",
            registration.warnings
        );
        assert_eq!(registration.servers, 1);
        assert_eq!(registration.tools, 1);
        assert!(registration.statuses[0].ok);
        assert!(
            registry
                .names()
                .iter()
                .any(|name| name == "mcp__mock__ping")
        );
    }

    #[test]
    fn tool_names_are_sanitized_and_grouped() {
        assert_eq!(
            tool_name("Rea Tools", "open/binary"),
            "mcp__rea_tools__open_binary"
        );
        assert_eq!(crate::tool::tool_group("mcp__rea__open_binary"), "mcp");
    }

    #[test]
    fn risk_defaults_to_process_execution_and_trust_maps_to_read_only() {
        assert_eq!(
            risk_from_config(McpRisk::ProcessExecution, false),
            RiskClass::ProcessExecution
        );
        assert_eq!(
            risk_from_config(McpRisk::HighImpact, false),
            RiskClass::HighImpact
        );
        assert_eq!(
            risk_from_config(McpRisk::ProcessExecution, true),
            RiskClass::ReadOnly
        );
    }

    #[test]
    fn oversized_schemas_fall_back_to_a_permissive_object() {
        let huge = json!({"type": "object", "x": "y".repeat(MAX_SCHEMA_CHARS + 1)});
        assert_eq!(
            sanitize_schema(&huge),
            json!({"type": "object", "additionalProperties": true})
        );
        assert_eq!(
            sanitize_schema(&json!({"type": "object"})),
            json!({"type": "object"})
        );
    }
}
