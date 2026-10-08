use std::{
    collections::{HashMap, VecDeque},
    io::{Read as _, Write as _},
    path::{Path, PathBuf},
    process::Stdio,
    sync::Arc,
    time::Duration,
};

use chrono::{DateTime, Utc};
use parking_lot::Mutex as SyncMutex;
use portable_pty::{CommandBuilder, PtySize, native_pty_system};
use serde::{Deserialize, Serialize};
use tokio::{
    fs::{self, OpenOptions},
    io::{AsyncRead, AsyncReadExt, AsyncWriteExt},
    process::{Child, ChildStdin, Command},
    sync::{Mutex, Notify, mpsc},
};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

#[cfg(windows)]
use crate::process_window;
use crate::{PokError, Result};

const READ_CHUNK: usize = 8 * 1024;
const PREVIEW_HEAD: usize = 4 * 1024;
const PREVIEW_TAIL: usize = 12 * 1024;
const DEFAULT_LOG_LIMIT: u64 = 128 * 1024 * 1024;
const MAX_ACTIVE: usize = 16;
const MAX_FINISHED: usize = 64;

#[derive(Debug, Clone, Serialize)]
pub struct CommandEvent {
    pub call_id: String,
    pub task_id: String,
    pub kind: String,
    pub stream: Option<String>,
    pub chunk: Option<String>,
    pub total_bytes: u64,
    pub status: String,
    pub exit_code: Option<i32>,
    pub termination_reason: Option<String>,
}

pub trait CommandEventSink: Send + Sync {
    fn emit(&self, event: CommandEvent);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommandStatus {
    Running,
    Completed,
    Failed,
    TimedOut,
    Cancelled,
    Killed,
}

impl CommandStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::TimedOut => "timed_out",
            Self::Cancelled => "cancelled",
            Self::Killed => "killed",
        }
    }

    fn terminal(self) -> bool {
        self != Self::Running
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CommandSnapshot {
    pub task_id: String,
    pub call_id: String,
    pub command: String,
    pub cwd: PathBuf,
    pub status: CommandStatus,
    pub pid: Option<u32>,
    pub started_at: DateTime<Utc>,
    pub finished_at: Option<DateTime<Utc>>,
    pub exit_code: Option<i32>,
    pub termination_reason: Option<String>,
    pub stdout: String,
    pub stderr: String,
    pub output: String,
    pub total_bytes: u64,
    pub log_bytes: u64,
    pub log_truncated: bool,
    pub output_file: PathBuf,
    pub next_cursor: u64,
}

#[derive(Debug)]
struct RecordState {
    status: CommandStatus,
    pid: Option<u32>,
    finished_at: Option<DateTime<Utc>>,
    exit_code: Option<i32>,
    termination_reason: Option<String>,
    stdout: Preview,
    stderr: Preview,
    combined: Preview,
    total_bytes: u64,
    log_bytes: u64,
    log_truncated: bool,
}

#[derive(Debug, Default)]
struct Preview {
    head: Vec<u8>,
    tail: VecDeque<u8>,
}

impl Preview {
    fn push(&mut self, bytes: &[u8]) {
        if self.head.len() < PREVIEW_HEAD {
            let take = (PREVIEW_HEAD - self.head.len()).min(bytes.len());
            self.head.extend_from_slice(&bytes[..take]);
        }
        self.tail.extend(bytes.iter().copied());
        while self.tail.len() > PREVIEW_TAIL {
            self.tail.pop_front();
        }
    }

    fn text(&self) -> String {
        let mut value = String::from_utf8_lossy(&self.head).into_owned();
        let tail = self.tail.iter().copied().collect::<Vec<_>>();
        if tail.len() > self.head.len() || !tail.starts_with(&self.head) {
            if !value.is_empty() {
                value.push_str("\n... output omitted; use manage_command read ...\n");
            }
            value.push_str(&String::from_utf8_lossy(&tail));
        }
        value
    }
}

struct CommandRecord {
    task_id: String,
    call_id: String,
    command: String,
    cwd: PathBuf,
    started_at: DateTime<Utc>,
    output_file: PathBuf,
    state: Mutex<RecordState>,
    stdin: Mutex<Option<CommandInput>>,
    finished: Notify,
}

enum CommandInput {
    Pipe(ChildStdin),
    Pty(Arc<std::sync::Mutex<Box<dyn std::io::Write + Send>>>),
}

impl CommandRecord {
    async fn snapshot(&self) -> CommandSnapshot {
        let state = self.state.lock().await;
        CommandSnapshot {
            task_id: self.task_id.clone(),
            call_id: self.call_id.clone(),
            command: self.command.clone(),
            cwd: self.cwd.clone(),
            status: state.status,
            pid: state.pid,
            started_at: self.started_at,
            finished_at: state.finished_at,
            exit_code: state.exit_code,
            termination_reason: state.termination_reason.clone(),
            stdout: state.stdout.text(),
            stderr: state.stderr.text(),
            output: state.combined.text(),
            total_bytes: state.total_bytes,
            log_bytes: state.log_bytes,
            log_truncated: state.log_truncated,
            output_file: self.output_file.clone(),
            next_cursor: state.log_bytes,
        }
    }
}

#[derive(Clone)]
pub struct CommandManager {
    inner: Arc<ManagerInner>,
}

struct ManagerInner {
    records: Mutex<HashMap<String, Arc<CommandRecord>>>,
    finished_order: Mutex<VecDeque<String>>,
    root: PathBuf,
    shutdown: CancellationToken,
    event_sink: SyncMutex<Option<Arc<dyn CommandEventSink>>>,
}

impl CommandManager {
    pub fn new(root: PathBuf) -> Self {
        Self {
            inner: Arc::new(ManagerInner {
                records: Mutex::new(HashMap::new()),
                finished_order: Mutex::new(VecDeque::new()),
                root: root.join("commands"),
                shutdown: CancellationToken::new(),
                event_sink: SyncMutex::new(None),
            }),
        }
    }

    pub fn set_event_sink(&self, sink: Arc<dyn CommandEventSink>) {
        *self.inner.event_sink.lock() = Some(sink);
    }

    pub fn request_shutdown(&self) {
        self.inner.shutdown.cancel();
    }

    pub async fn spawn(
        &self,
        call_id: String,
        command: String,
        cwd: PathBuf,
        hard_timeout: Option<Duration>,
        pty: bool,
    ) -> Result<CommandSnapshot> {
        let active = {
            let records = self.inner.records.lock().await;
            let entries = records.values().cloned().collect::<Vec<_>>();
            drop(records);
            let mut active = 0;
            for record in entries {
                if !record.state.lock().await.status.terminal() {
                    active += 1;
                }
            }
            active
        };
        if active >= MAX_ACTIVE {
            return Err(PokError::Tool(format!(
                "maximum of {MAX_ACTIVE} active commands reached; wait for or kill an existing command"
            )));
        }
        fs::create_dir_all(&self.inner.root).await?;
        let task_id = Uuid::new_v4().to_string();
        let output_file = self.inner.root.join(format!("{task_id}.log"));
        if pty {
            return self
                .spawn_pty(task_id, call_id, command, cwd, output_file, hard_timeout)
                .await;
        }
        let mut process = platform_command(&command);
        process
            .current_dir(&cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .env("POK_AI_AGENT", "1");
        let mut child = process.spawn()?;
        let pid = child.id();
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| PokError::Tool("stdout pipe unavailable".into()))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| PokError::Tool("stderr pipe unavailable".into()))?;
        let stdin = child.stdin.take();
        let record = Arc::new(CommandRecord {
            task_id: task_id.clone(),
            call_id,
            command,
            cwd,
            started_at: Utc::now(),
            output_file,
            state: Mutex::new(RecordState {
                status: CommandStatus::Running,
                pid,
                finished_at: None,
                exit_code: None,
                termination_reason: None,
                stdout: Preview::default(),
                stderr: Preview::default(),
                combined: Preview::default(),
                total_bytes: 0,
                log_bytes: 0,
                log_truncated: false,
            }),
            stdin: Mutex::new(stdin.map(CommandInput::Pipe)),
            finished: Notify::new(),
        });
        self.inner
            .records
            .lock()
            .await
            .insert(task_id.clone(), record.clone());
        self.emit_status(&record).await;
        let manager = self.clone();
        tokio::spawn(async move {
            manager
                .run_record(record, child, stdout, stderr, hard_timeout)
                .await;
        });
        Ok(self.get(&task_id).await.expect("new command record exists"))
    }

    async fn spawn_pty(
        &self,
        task_id: String,
        call_id: String,
        command: String,
        cwd: PathBuf,
        output_file: PathBuf,
        hard_timeout: Option<Duration>,
    ) -> Result<CommandSnapshot> {
        let pair = native_pty_system()
            .openpty(PtySize {
                rows: 30,
                cols: 120,
                pixel_width: 0,
                pixel_height: 0,
            })
            .map_err(|error| PokError::Tool(format!("failed to create PTY: {error}")))?;
        let mut builder = pty_command(&command);
        builder.cwd(&cwd);
        builder.env("POK_AI_AGENT", "1");
        let child = pair
            .slave
            .spawn_command(builder)
            .map_err(|error| PokError::Tool(format!("failed to start PTY command: {error}")))?;
        let pid = child.process_id();
        let reader = pair
            .master
            .try_clone_reader()
            .map_err(|error| PokError::Tool(format!("failed to open PTY output: {error}")))?;
        let writer = pair
            .master
            .take_writer()
            .map_err(|error| PokError::Tool(format!("failed to open PTY input: {error}")))?;
        let record = Arc::new(CommandRecord {
            task_id: task_id.clone(),
            call_id,
            command,
            cwd,
            started_at: Utc::now(),
            output_file,
            state: Mutex::new(RecordState {
                status: CommandStatus::Running,
                pid,
                finished_at: None,
                exit_code: None,
                termination_reason: None,
                stdout: Preview::default(),
                stderr: Preview::default(),
                combined: Preview::default(),
                total_bytes: 0,
                log_bytes: 0,
                log_truncated: false,
            }),
            stdin: Mutex::new(Some(CommandInput::Pty(Arc::new(std::sync::Mutex::new(
                writer,
            ))))),
            finished: Notify::new(),
        });
        self.inner
            .records
            .lock()
            .await
            .insert(task_id.clone(), record.clone());
        self.emit_status(&record).await;
        let manager = self.clone();
        tokio::spawn(async move {
            manager
                .run_pty_record(record, reader, child, hard_timeout)
                .await;
        });
        Ok(self.get(&task_id).await.expect("new PTY record exists"))
    }

    async fn run_pty_record(
        &self,
        record: Arc<CommandRecord>,
        mut reader: Box<dyn std::io::Read + Send>,
        mut child: Box<dyn portable_pty::Child + Send + Sync>,
        hard_timeout: Option<Duration>,
    ) {
        let (sender, receiver) = mpsc::channel::<(&'static str, Vec<u8>)>(64);
        let read_task = tokio::task::spawn_blocking(move || {
            let mut buffer = vec![0; READ_CHUNK];
            loop {
                match reader.read(&mut buffer) {
                    Ok(0) | Err(_) => break,
                    Ok(read) => {
                        if sender
                            .blocking_send(("pty", buffer[..read].to_vec()))
                            .is_err()
                        {
                            break;
                        }
                    }
                }
            }
        });
        let writer = tokio::spawn(self.clone().write_output(record.clone(), receiver));
        let mut wait = tokio::task::spawn_blocking(move || child.wait());
        let (mut status, terminal_reason) = if let Some(limit) = hard_timeout {
            tokio::select! {
                result = &mut wait => (result.ok().and_then(|value| value.ok()), None),
                () = self.inner.shutdown.cancelled() => (None, Some((CommandStatus::Cancelled, "session_shutdown"))),
                () = tokio::time::sleep(limit) => (None, Some((CommandStatus::TimedOut, "hard_timeout"))),
            }
        } else {
            tokio::select! {
                result = &mut wait => (result.ok().and_then(|value| value.ok()), None),
                () = self.inner.shutdown.cancelled() => (None, Some((CommandStatus::Cancelled, "session_shutdown"))),
            }
        };
        if let Some((state, reason)) = terminal_reason {
            if let Some(pid) = record.state.lock().await.pid {
                terminate_process_tree(pid).await;
            }
            let mut current = record.state.lock().await;
            current.status = state;
            current.termination_reason = Some(reason.into());
            status = None;
        }
        if let Some(exit) = status {
            let code = exit.exit_code() as i32;
            let mut state = record.state.lock().await;
            if !state.status.terminal() {
                state.exit_code = Some(code);
                state.status = if code == 0 {
                    CommandStatus::Completed
                } else {
                    CommandStatus::Failed
                };
            }
        }
        let _ = read_task.await;
        let _ = writer.await;
        *record.stdin.lock().await = None;
        record.state.lock().await.finished_at = Some(Utc::now());
        record.finished.notify_waiters();
        self.emit_status(&record).await;
        self.remember_finished(&record.task_id).await;
    }

    async fn run_record(
        &self,
        record: Arc<CommandRecord>,
        mut child: Child,
        stdout: impl AsyncRead + Unpin + Send + 'static,
        stderr: impl AsyncRead + Unpin + Send + 'static,
        hard_timeout: Option<Duration>,
    ) {
        let (sender, receiver) = mpsc::channel::<(&'static str, Vec<u8>)>(64);
        let out_task = tokio::spawn(read_stream(stdout, "stdout", sender.clone()));
        let err_task = tokio::spawn(read_stream(stderr, "stderr", sender.clone()));
        drop(sender);
        let writer = tokio::spawn(self.clone().write_output(record.clone(), receiver));

        let wait = child.wait();
        tokio::pin!(wait);
        let outcome = if let Some(limit) = hard_timeout {
            tokio::select! {
                status = &mut wait => (status.ok(), None),
                () = self.inner.shutdown.cancelled() => (None, Some((CommandStatus::Cancelled, "session_shutdown"))),
                () = tokio::time::sleep(limit) => (None, Some((CommandStatus::TimedOut, "hard_timeout"))),
            }
        } else {
            tokio::select! {
                status = &mut wait => (status.ok(), None),
                () = self.inner.shutdown.cancelled() => (None, Some((CommandStatus::Cancelled, "session_shutdown"))),
            }
        };
        if let Some((status, reason)) = outcome.1 {
            let _ = self.terminate_record(&record).await;
            let mut state = record.state.lock().await;
            state.status = status;
            state.termination_reason = Some(reason.into());
        } else if let Some(status) = outcome.0 {
            let mut state = record.state.lock().await;
            if !state.status.terminal() {
                state.exit_code = status.code();
                state.status = if status.success() {
                    CommandStatus::Completed
                } else {
                    CommandStatus::Failed
                };
                if !status.success() {
                    state.termination_reason = Some("nonzero_exit".into());
                }
            }
        }
        let _ = out_task.await;
        let _ = err_task.await;
        let _ = writer.await;
        *record.stdin.lock().await = None;
        {
            let mut state = record.state.lock().await;
            state.finished_at = Some(Utc::now());
        }
        record.finished.notify_waiters();
        self.emit_status(&record).await;
        self.remember_finished(&record.task_id).await;
    }

    async fn write_output(
        self,
        record: Arc<CommandRecord>,
        mut receiver: mpsc::Receiver<(&'static str, Vec<u8>)>,
    ) {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&record.output_file)
            .await
            .ok();
        while let Some((stream, bytes)) = receiver.recv().await {
            {
                let mut state = record.state.lock().await;
                state.total_bytes = state.total_bytes.saturating_add(bytes.len() as u64);
                state.combined.push(&bytes);
                if stream == "stderr" {
                    state.stderr.push(&bytes);
                } else {
                    state.stdout.push(&bytes);
                }
                if state.log_bytes < DEFAULT_LOG_LIMIT {
                    let allowed =
                        (DEFAULT_LOG_LIMIT - state.log_bytes).min(bytes.len() as u64) as usize;
                    if let Some(file) = file.as_mut() {
                        let _ = file.write_all(&bytes[..allowed]).await;
                    }
                    state.log_bytes += allowed as u64;
                    state.log_truncated |= allowed < bytes.len();
                } else {
                    state.log_truncated = true;
                }
            }
            self.emit_chunk(&record, stream, &bytes).await;
        }
        if let Some(file) = file.as_mut() {
            let _ = file.flush().await;
        }
    }

    async fn remember_finished(&self, task_id: &str) {
        let mut order = self.inner.finished_order.lock().await;
        order.push_back(task_id.to_owned());
        while order.len() > MAX_FINISHED {
            if let Some(oldest) = order.pop_front() {
                self.inner.records.lock().await.remove(&oldest);
            }
        }
    }

    pub async fn get(&self, task_id: &str) -> Option<CommandSnapshot> {
        let record = self.inner.records.lock().await.get(task_id).cloned()?;
        Some(record.snapshot().await)
    }

    pub async fn list(&self) -> Vec<CommandSnapshot> {
        let records = self
            .inner
            .records
            .lock()
            .await
            .values()
            .cloned()
            .collect::<Vec<_>>();
        let mut result = Vec::with_capacity(records.len());
        for record in records {
            result.push(record.snapshot().await);
        }
        result.sort_by_key(|item| std::cmp::Reverse(item.started_at));
        result
    }

    pub async fn wait(&self, task_id: &str, timeout: Duration) -> Result<CommandSnapshot> {
        let record = self.record(task_id).await?;
        let notified = record.finished.notified();
        if !record.state.lock().await.status.terminal() {
            let _ = tokio::time::timeout(timeout, notified).await;
        }
        Ok(record.snapshot().await)
    }

    pub async fn kill(&self, task_id: &str, reason: &str) -> Result<CommandSnapshot> {
        let record = self.record(task_id).await?;
        if !record.state.lock().await.status.terminal() {
            self.terminate_record(&record).await?;
            let mut state = record.state.lock().await;
            state.status = if reason == "cancelled" {
                CommandStatus::Cancelled
            } else {
                CommandStatus::Killed
            };
            state.termination_reason = Some(reason.into());
        }
        self.emit_status(&record).await;
        Ok(record.snapshot().await)
    }

    pub async fn write(&self, task_id: &str, data: &[u8], newline: bool) -> Result<()> {
        let record = self.record(task_id).await?;
        let mut stdin = record.stdin.lock().await;
        let input = stdin
            .as_mut()
            .ok_or_else(|| PokError::Tool("command stdin is closed".into()))?;
        match input {
            CommandInput::Pipe(input) => {
                input.write_all(data).await?;
                if newline {
                    input.write_all(b"\n").await?;
                }
                input.flush().await?;
            }
            CommandInput::Pty(writer) => {
                let writer = writer.clone();
                let mut bytes = data.to_vec();
                if newline {
                    bytes.push(b'\r');
                }
                tokio::task::spawn_blocking(move || -> std::io::Result<()> {
                    let mut writer = writer
                        .lock()
                        .map_err(|_| std::io::Error::other("PTY input lock poisoned"))?;
                    writer.write_all(&bytes)?;
                    writer.flush()
                })
                .await
                .map_err(|error| PokError::Other(error.into()))??;
            }
        }
        Ok(())
    }

    pub async fn close_stdin(&self, task_id: &str) -> Result<()> {
        let record = self.record(task_id).await?;
        *record.stdin.lock().await = None;
        Ok(())
    }

    pub async fn read(
        &self,
        task_id: &str,
        offset: u64,
        max_bytes: usize,
    ) -> Result<(String, u64, bool)> {
        let record = self.record(task_id).await?;
        let bytes = fs::read(&record.output_file).await.unwrap_or_default();
        let start = usize::try_from(offset)
            .unwrap_or(usize::MAX)
            .min(bytes.len());
        let end = start
            .saturating_add(max_bytes.clamp(1, 64 * 1024))
            .min(bytes.len());
        Ok((
            String::from_utf8_lossy(&bytes[start..end]).into_owned(),
            end as u64,
            end < bytes.len(),
        ))
    }

    async fn record(&self, task_id: &str) -> Result<Arc<CommandRecord>> {
        self.inner
            .records
            .lock()
            .await
            .get(task_id)
            .cloned()
            .ok_or_else(|| PokError::Tool(format!("unknown command task id {task_id}")))
    }

    async fn terminate_record(&self, record: &CommandRecord) -> Result<()> {
        let pid = record.state.lock().await.pid;
        if let Some(pid) = pid {
            terminate_process_tree(pid).await;
        }
        Ok(())
    }

    async fn emit_chunk(&self, record: &CommandRecord, stream: &str, bytes: &[u8]) {
        let Some(sink) = self.inner.event_sink.lock().clone() else {
            return;
        };
        let state = record.state.lock().await;
        sink.emit(CommandEvent {
            call_id: record.call_id.clone(),
            task_id: record.task_id.clone(),
            kind: "output".into(),
            stream: Some(stream.into()),
            chunk: Some(String::from_utf8_lossy(bytes).into_owned()),
            total_bytes: state.total_bytes,
            status: state.status.as_str().into(),
            exit_code: state.exit_code,
            termination_reason: state.termination_reason.clone(),
        });
    }

    async fn emit_status(&self, record: &CommandRecord) {
        let Some(sink) = self.inner.event_sink.lock().clone() else {
            return;
        };
        let state = record.state.lock().await;
        sink.emit(CommandEvent {
            call_id: record.call_id.clone(),
            task_id: record.task_id.clone(),
            kind: "status".into(),
            stream: None,
            chunk: None,
            total_bytes: state.total_bytes,
            status: state.status.as_str().into(),
            exit_code: state.exit_code,
            termination_reason: state.termination_reason.clone(),
        });
    }
}

async fn read_stream(
    mut reader: impl AsyncRead + Unpin,
    stream: &'static str,
    sender: mpsc::Sender<(&'static str, Vec<u8>)>,
) {
    let mut buffer = vec![0; READ_CHUNK];
    loop {
        match reader.read(&mut buffer).await {
            Ok(0) | Err(_) => break,
            Ok(read) => {
                if sender
                    .send((stream, buffer[..read].to_vec()))
                    .await
                    .is_err()
                {
                    break;
                }
            }
        }
    }
}

fn platform_command(command: &str) -> Command {
    #[cfg(windows)]
    {
        let mut process = Command::new("powershell.exe");
        process.args([
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-WindowStyle",
            "Hidden",
            "-Command",
            &powershell_script(command),
        ]);
        process_window::hide_tokio(&mut process);
        process
    }
    #[cfg(not(windows))]
    {
        let mut process = Command::new("setsid");
        process.args(["bash", "--noprofile", "--norc", "-c", command]);
        process
    }
}

fn pty_command(command: &str) -> CommandBuilder {
    #[cfg(windows)]
    {
        let mut builder = CommandBuilder::new("powershell.exe");
        for arg in [
            "-NoLogo",
            "-NoProfile",
            "-WindowStyle",
            "Hidden",
            "-Command",
            &powershell_script(command),
        ] {
            builder.arg(arg);
        }
        builder
    }
    #[cfg(not(windows))]
    {
        let mut builder = CommandBuilder::new("bash");
        for arg in ["--noprofile", "--norc", "-c", command] {
            builder.arg(arg);
        }
        builder
    }
}

/// The PowerShell wrapper around every Windows command. Kept
/// platform-independent so tests on any host exercise the exact string the
/// Windows spawn path uses.
///
/// `$ErrorActionPreference = 'Stop'` turns the first line a native command
/// writes to stderr into a terminating NativeCommandError. When the command
/// pipes or redirects stderr (as debugging commands commonly do), that aborts
/// the stream: only "Traceback (most recent call last):" survives and the
/// actual exception is lost. Keep 'Continue' so the whole native stderr
/// reaches the model, take native failures from the exit code, and still fail
/// on terminating script errors via the catch.
#[allow(dead_code)]
fn powershell_utf8_wrapper(command: &str) -> String {
    format!(
        "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); $OutputEncoding = [Console]::OutputEncoding; $global:LASTEXITCODE = 0; $ErrorActionPreference = 'Continue'; $pok_ok = $true; try {{ & {{ {command} }} }} catch {{ [Console]::Error.WriteLine(($_ | Out-String)); $pok_ok = $false }}; if (-not $pok_ok) {{ exit 1 }}; exit $global:LASTEXITCODE"
    )
}

#[cfg(windows)]
fn powershell_script(command: &str) -> String {
    powershell_utf8_wrapper(command)
}

async fn terminate_process_tree(pid: u32) {
    #[cfg(windows)]
    {
        let mut command = Command::new("taskkill");
        command.args(["/PID", &pid.to_string(), "/T", "/F"]);
        process_window::hide_tokio(&mut command);
        let _ = command.status().await;
    }
    #[cfg(not(windows))]
    {
        let group = format!("-{pid}");
        let _ = Command::new("kill")
            .args(["-TERM", "--", &group])
            .status()
            .await;
        tokio::time::sleep(Duration::from_millis(500)).await;
        let _ = Command::new("kill")
            .args(["-KILL", "--", &group])
            .status()
            .await;
    }
}

pub fn snapshot_json(snapshot: &CommandSnapshot) -> serde_json::Value {
    let success = match snapshot.status {
        CommandStatus::Running => serde_json::Value::Null,
        CommandStatus::Completed => serde_json::Value::Bool(snapshot.exit_code == Some(0)),
        _ => serde_json::Value::Bool(false),
    };
    serde_json::json!({
        "task_id": snapshot.task_id, "call_id": snapshot.call_id, "command": snapshot.command,
        "cwd": snapshot.cwd, "status": snapshot.status.as_str(),
        "success": success,
        "exit_code": snapshot.exit_code, "termination_reason": snapshot.termination_reason,
        "stdout": snapshot.stdout, "stderr": snapshot.stderr, "output": snapshot.output,
        "total_bytes": snapshot.total_bytes, "log_bytes": snapshot.log_bytes,
        "output_truncated": snapshot.log_truncated, "output_file": snapshot.output_file,
        "next_cursor": snapshot.next_cursor, "started_at": snapshot.started_at,
        "finished_at": snapshot.finished_at,
    })
}

pub fn output_path_is_within(path: &Path, root: &Path) -> bool {
    path.starts_with(root)
}

#[cfg(all(test, not(windows)))]
mod tests {
    use super::*;

    #[tokio::test]
    async fn captures_output_and_paginates_the_full_log() {
        let root = tempfile::tempdir().unwrap();
        let manager = CommandManager::new(root.path().to_path_buf());
        let started = manager
            .spawn(
                "call-1".into(),
                "printf 'first\\nsecond\\nthird\\n'".into(),
                root.path().to_path_buf(),
                Some(Duration::from_secs(5)),
                false,
            )
            .await
            .unwrap();
        let done = manager
            .wait(&started.task_id, Duration::from_secs(5))
            .await
            .unwrap();
        assert_eq!(done.status, CommandStatus::Completed);
        assert_eq!(done.exit_code, Some(0));
        let (first, cursor, more) = manager.read(&done.task_id, 0, 8).await.unwrap();
        assert_eq!(first, "first\nse");
        assert!(more);
        let (rest, _, more) = manager.read(&done.task_id, cursor, 1024).await.unwrap();
        assert!(rest.contains("cond\nthird"));
        assert!(!more);
    }

    #[tokio::test]
    async fn background_stdin_submit_and_wait_work() {
        let root = tempfile::tempdir().unwrap();
        let manager = CommandManager::new(root.path().to_path_buf());
        let started = manager
            .spawn(
                "call-2".into(),
                "read value; printf 'received:%s\\n' \"$value\"".into(),
                root.path().to_path_buf(),
                Some(Duration::from_secs(5)),
                false,
            )
            .await
            .unwrap();
        manager
            .write(&started.task_id, b"hello", true)
            .await
            .unwrap();
        let done = manager
            .wait(&started.task_id, Duration::from_secs(5))
            .await
            .unwrap();
        assert_eq!(done.status, CommandStatus::Completed);
        assert!(done.stdout.contains("received:hello"));
    }

    #[tokio::test]
    async fn hard_timeout_keeps_partial_output() {
        let root = tempfile::tempdir().unwrap();
        let manager = CommandManager::new(root.path().to_path_buf());
        let started = manager
            .spawn(
                "call-3".into(),
                "printf 'before-timeout\\n'; sleep 30".into(),
                root.path().to_path_buf(),
                Some(Duration::from_millis(200)),
                false,
            )
            .await
            .unwrap();
        let done = manager
            .wait(&started.task_id, Duration::from_secs(5))
            .await
            .unwrap();
        assert_eq!(done.status, CommandStatus::TimedOut);
        assert!(done.output.contains("before-timeout"));
        assert_eq!(done.termination_reason.as_deref(), Some("hard_timeout"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn pty_accepts_submitted_input() {
        let root = tempfile::tempdir().unwrap();
        let manager = CommandManager::new(root.path().to_path_buf());
        let started = manager
            .spawn(
                "call-pty".into(),
                "read value; printf 'pty:%s\\n' \"$value\"".into(),
                root.path().to_path_buf(),
                Some(Duration::from_secs(5)),
                true,
            )
            .await
            .unwrap();
        manager
            .write(&started.task_id, b"hello", true)
            .await
            .unwrap();
        let done = manager
            .wait(&started.task_id, Duration::from_secs(5))
            .await
            .unwrap();
        assert_eq!(done.status, CommandStatus::Completed);
        assert!(done.output.contains("pty:hello"), "{}", done.output);
    }

    #[test]
    fn powershell_wrapper_keeps_native_stderr_and_exit_codes() {
        let script = powershell_utf8_wrapper("python probe.py 2>&1 | Select-Object -First 5");
        assert!(script.contains("[Console]::OutputEncoding"));
        assert!(script.contains("$OutputEncoding"));
        assert!(script.contains("python probe.py 2>&1 | Select-Object -First 5"));
        // 'Stop' aborts a native command at its first stderr line, hiding the
        // rest of the traceback; keep 'Continue' and use the exit code.
        assert!(!script.contains("$ErrorActionPreference = 'Stop'"));
        assert!(script.contains("$ErrorActionPreference = 'Continue'"));
        assert!(script.contains("exit $global:LASTEXITCODE"));
        assert!(script.contains("catch"));
        assert!(!script.contains("powershell -Command"));
    }
}
