//! WhatsApp as a gateway channel, through a linked-device sidecar.
//!
//! WhatsApp has no bot API for personal accounts. Like Hermes Agent and
//! OpenClaw, POK-Agent signs in as a linked device (Settings, Gateway, scan a
//! QR code in WhatsApp under Linked devices). The unofficial WhatsApp Web
//! client lives in a separate program, `pok-ai-whatsapp`, so its dependencies
//! and toolchain stay out of the agent and it exists only on computers that
//! opted in. The two talk with one JSON object per line over stdin/stdout.
//!
//! Sidecar to agent (`event`): `qr`, `connected`, `disconnected`, `message`,
//! `result`, `log`. Agent to sidecar (`cmd`): `send_text`, `typing`,
//! `download`, `logout`, `shutdown`. A command with a `request` id is
//! answered by a `result` event carrying the same id.
//!
//! The unofficial client can get an account suspended, so a dedicated number
//! (`bot` mode) is the safer choice; `self_chat` mode talks to the owner's
//! own number through "Message yourself" and never answers anyone else.

use std::{
    collections::{HashMap, VecDeque},
    path::PathBuf,
    process::Stdio,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use async_trait::async_trait;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader},
    sync::{mpsc, oneshot},
};
use tokio_util::sync::CancellationToken;

use super::{
    Channel, Gateway, Inbound, Strangers,
    media::{Download, MediaKind, MediaSource, safe_component},
};
use crate::error::{PokError, Result};

/// WhatsApp cuts messages near 65 000 characters, but long ones are hard to
/// read on a phone: send paragraphs of at most this many characters.
const MESSAGE_CHARS: usize = 4_000;
/// A message older than this when it arrives (the phone was off, the link was
/// down) is dropped rather than acted on late.
const MAX_MESSAGE_AGE_SECS: i64 = 120;
/// Answers to commands that expect one.
const REPLY_TIMEOUT: Duration = Duration::from_secs(120);
/// Marks the agent's own replies inside the owner's self-chat, which would
/// otherwise come back as new messages.
pub const SELF_CHAT_PREFIX: &str = "POK-Agent: ";
/// How many of our own message ids are remembered for the echo check.
const REMEMBERED_SENDS: usize = 256;
/// The word a friend starts a request with in friends mode.
pub const WAKE_WORD: &str = "pok";
/// After a friend's request, their next messages in that chat join it without
/// the wake word for this long: the rest of a photo album, or a message the
/// app split in two.
const FOLLOW_UP_WINDOW: Duration = Duration::from_secs(20);

/// Whose messages the agent answers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WhatsAppMode {
    /// The owner's own number, writing in "Message yourself". Nobody else is
    /// ever answered.
    #[default]
    SelfChat,
    /// A dedicated number that approved people message; strangers get a
    /// pairing code like on Telegram.
    Bot,
    /// The owner's own number, shared with friends. "Message yourself" works
    /// as in `SelfChat`. A friend asks by starting a message with "pok"
    /// (everything else they write is a normal chat with the owner and is
    /// never touched); someone not yet approved gets no reply at all, and the
    /// owner sees the request in Settings. Friends cannot answer permission
    /// requests or use commands.
    Friends,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkState {
    /// Not started.
    #[default]
    Off,
    /// The sidecar is starting or reconnecting.
    Connecting,
    /// Waiting for the owner to scan the QR code.
    WaitingForScan,
    Connected,
    /// The device was unlinked from the phone; a new scan is needed.
    LoggedOut,
    /// The sidecar program is missing or keeps failing.
    Failed,
}

/// What the settings page shows about the WhatsApp link.
#[derive(Debug, Clone, Default, Serialize)]
pub struct WhatsAppStatus {
    pub state: LinkState,
    /// The QR code to scan, as an SVG image.
    pub qr_svg: Option<String>,
    /// The linked number.
    pub account: Option<String>,
    pub error: Option<String>,
}

/// How to start the sidecar.
#[derive(Debug, Clone)]
pub struct WhatsAppConfig {
    pub sidecar: PathBuf,
    /// Holds the linked-device session (keys): treat it like a password.
    pub data_dir: PathBuf,
    pub mode: WhatsAppMode,
}

// ---- protocol -------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
enum Event {
    Qr {
        code: String,
    },
    Connected {
        #[serde(default)]
        phone: Option<String>,
    },
    Disconnected {
        #[serde(default)]
        reason: String,
        #[serde(default)]
        logged_out: bool,
    },
    Message(Box<WaMessage>),
    Result {
        request: String,
        #[serde(default)]
        ok: bool,
        #[serde(default)]
        error: Option<String>,
        #[serde(default)]
        message_id: Option<String>,
        #[serde(default)]
        path: Option<String>,
    },
    Log {
        #[serde(default)]
        level: String,
        #[serde(default)]
        message: String,
    },
}

#[derive(Debug, Clone, Deserialize)]
pub struct WaMessage {
    pub id: String,
    /// The chat to answer in.
    pub chat: String,
    /// `direct`, `group`, `broadcast` or `newsletter`.
    #[serde(default)]
    pub chat_kind: String,
    /// The sender as a phone number (digits), when WhatsApp revealed it.
    #[serde(default)]
    pub sender: String,
    #[serde(default)]
    pub name: String,
    /// Sent from the linked account itself.
    #[serde(default)]
    pub from_me: bool,
    /// The chat is the linked account's own "Message yourself" chat.
    #[serde(default)]
    pub self_chat: bool,
    #[serde(default)]
    pub text: String,
    /// Unix seconds.
    #[serde(default)]
    pub timestamp: i64,
    #[serde(default)]
    pub media: Vec<WaMedia>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WaMedia {
    pub index: usize,
    pub kind: String,
    #[serde(default)]
    pub mime: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub size: Option<u64>,
}

#[derive(Debug)]
struct Reply {
    ok: bool,
    error: Option<String>,
    message_id: Option<String>,
    path: Option<String>,
}

/// The live connection to one running sidecar.
struct Link {
    commands: mpsc::UnboundedSender<String>,
    pending: Mutex<HashMap<String, oneshot::Sender<Reply>>>,
}

// ---- the channel -----------------------------------------------------------

pub struct WhatsAppChannel {
    config: WhatsAppConfig,
    status: Mutex<WhatsAppStatus>,
    link: Mutex<Option<Arc<Link>>>,
    sent_ids: Mutex<VecDeque<String>>,
    /// Friends mode: when each chat last made a request.
    engaged: Mutex<HashMap<String, Instant>>,
    /// The saved session is deleted once the sidecar has exited: set when the
    /// device was unlinked, so a revoked session is never tried again.
    wipe_session: AtomicBool,
}

impl WhatsAppChannel {
    pub fn new(config: WhatsAppConfig) -> Arc<Self> {
        Arc::new(Self {
            config,
            status: Mutex::new(WhatsAppStatus::default()),
            link: Mutex::new(None),
            sent_ids: Mutex::new(VecDeque::new()),
            engaged: Mutex::new(HashMap::new()),
            wipe_session: AtomicBool::new(false),
        })
    }

    pub fn status(&self) -> WhatsAppStatus {
        self.status.lock().clone()
    }

    pub fn mode(&self) -> WhatsAppMode {
        self.config.mode
    }

    fn update(&self, change: impl FnOnce(&mut WhatsAppStatus)) {
        change(&mut self.status.lock());
    }

    fn live_link(&self) -> Result<Arc<Link>> {
        self.link
            .lock()
            .clone()
            .ok_or_else(|| PokError::Provider("WhatsApp is not connected".into()))
    }

    /// Send a command and, when it has a request id, wait for its result.
    async fn command(&self, mut command: Value, wait: bool) -> Result<Reply> {
        let link = self.live_link()?;
        let request = uuid::Uuid::new_v4().simple().to_string();
        let receiver = wait.then(|| {
            let (sender, receiver) = oneshot::channel();
            link.pending.lock().insert(request.clone(), sender);
            receiver
        });
        command["request"] = Value::String(request.clone());
        if link.commands.send(command.to_string()).is_err() {
            link.pending.lock().remove(&request);
            return Err(PokError::Provider("WhatsApp is not connected".into()));
        }
        let Some(receiver) = receiver else {
            return Ok(Reply {
                ok: true,
                error: None,
                message_id: None,
                path: None,
            });
        };
        let reply = tokio::time::timeout(REPLY_TIMEOUT, receiver).await;
        link.pending.lock().remove(&request);
        match reply {
            Ok(Ok(reply)) if reply.ok => Ok(reply),
            Ok(Ok(reply)) => Err(PokError::Provider(
                reply
                    .error
                    .unwrap_or_else(|| "WhatsApp refused that".into()),
            )),
            Ok(Err(_)) => Err(PokError::Provider("WhatsApp disconnected".into())),
            Err(_) => Err(PokError::Provider("WhatsApp did not answer in time".into())),
        }
    }

    /// Unlink this computer from the phone and forget the saved session. With
    /// a running sidecar it is asked to unlink first, and the session is
    /// deleted after the program has exited and released its files.
    pub async fn logout(&self) -> Result<()> {
        if self.live_link().is_ok() {
            self.wipe_session.store(true, Ordering::SeqCst);
            let _ = self.command(json!({"cmd": "logout"}), true).await;
        } else {
            forget_session(&self.config.data_dir);
        }
        self.update(|status| {
            status.state = LinkState::LoggedOut;
            status.qr_svg = None;
            status.account = None;
        });
        Ok(())
    }

    fn remember_send(&self, id: String) {
        let mut sent = self.sent_ids.lock();
        sent.push_back(id);
        while sent.len() > REMEMBERED_SENDS {
            sent.pop_front();
        }
    }

    fn sent_by_us(&self, id: &str) -> bool {
        self.sent_ids.lock().iter().any(|sent| sent == id)
    }

    /// Whether a message is for the agent, and if so who it is from.
    /// Everything else is ignored without a reply.
    fn route(&self, message: &WaMessage, now: i64) -> Option<Sender> {
        if message.chat_kind != "direct" {
            return None;
        }
        if message.timestamp > 0 && now - message.timestamp > MAX_MESSAGE_AGE_SECS {
            return None;
        }
        if self.sent_by_us(&message.id) {
            return None;
        }
        match self.config.mode {
            WhatsAppMode::SelfChat => {
                // Only the owner's own "Message yourself" chat; anything the
                // agent wrote there carries the prefix and is skipped.
                if !(message.from_me && message.self_chat)
                    || message
                        .text
                        .trim_start()
                        .starts_with(SELF_CHAT_PREFIX.trim_end())
                {
                    return None;
                }
                Some(Sender::Owner)
            }
            WhatsAppMode::Bot => {
                if message.from_me || message.sender.trim().is_empty() {
                    return None;
                }
                Some(Sender::Person)
            }
            WhatsAppMode::Friends => {
                if message.from_me {
                    // The owner's own chat, as in self-chat mode. What the
                    // owner writes to friends is never a request, and the
                    // agent's replies there carry the prefix.
                    if !message.self_chat
                        || message
                            .text
                            .trim_start()
                            .starts_with(SELF_CHAT_PREFIX.trim_end())
                    {
                        return None;
                    }
                    return Some(Sender::Owner);
                }
                if message.sender.trim().is_empty() {
                    return None;
                }
                Some(Sender::Person)
            }
        }
    }

    /// Whether `chat` made a request a moment ago (friends mode).
    fn is_engaged(&self, chat: &str) -> bool {
        self.engaged
            .lock()
            .get(chat)
            .is_some_and(|since| since.elapsed() < FOLLOW_UP_WINDOW)
    }

    fn mark_engaged(&self, chat: &str) {
        let mut engaged = self.engaged.lock();
        engaged.retain(|_, since| since.elapsed() < FOLLOW_UP_WINDOW * 3);
        engaged.insert(chat.to_owned(), Instant::now());
    }

    async fn on_event(self: &Arc<Self>, event: Event, link: &Arc<Link>, gateway: &Arc<Gateway>) {
        match event {
            Event::Qr { code } => {
                let svg = qr_svg(&code);
                self.update(|status| {
                    status.state = LinkState::WaitingForScan;
                    status.qr_svg = svg;
                    status.error = None;
                });
                gateway.notify_changed();
            }
            Event::Connected { phone } => {
                self.update(|status| {
                    status.state = LinkState::Connected;
                    status.qr_svg = None;
                    status.account = phone.map(|phone| format!("+{phone}"));
                    status.error = None;
                });
                gateway.notify_changed();
            }
            Event::Disconnected { reason, logged_out } => {
                if logged_out {
                    self.wipe_session.store(true, Ordering::SeqCst);
                }
                self.update(|status| {
                    status.state = if logged_out {
                        LinkState::LoggedOut
                    } else {
                        LinkState::Connecting
                    };
                    status.qr_svg = None;
                    if logged_out {
                        status.account = None;
                    }
                    status.error = (!reason.is_empty()).then_some(reason);
                });
                gateway.notify_changed();
            }
            Event::Result {
                request,
                ok,
                error,
                message_id,
                path,
            } => {
                if let Some(id) = &message_id {
                    self.remember_send(id.clone());
                }
                if let Some(waiting) = link.pending.lock().remove(&request) {
                    let _ = waiting.send(Reply {
                        ok,
                        error,
                        message_id,
                        path,
                    });
                }
            }
            Event::Log { level, message } => match level.as_str() {
                "error" => tracing::warn!("whatsapp sidecar: {message}"),
                _ => tracing::debug!("whatsapp sidecar: {message}"),
            },
            Event::Message(message) => {
                let now = chrono::Utc::now().timestamp();
                let Some(sender) = self.route(&message, now) else {
                    return;
                };
                let channel = self.clone();
                let gateway = gateway.clone();
                // Fetching attachments needs this read loop to keep running.
                tokio::spawn(async move {
                    if let Some(inbound) = channel.to_inbound(*message, sender) {
                        gateway.handle_inbound(inbound).await;
                    }
                });
            }
        }
    }

    /// The message for the dispatcher, or `None` when it is not a request:
    /// in friends mode a friend's message counts only when it starts with the
    /// wake word (or follows one closely).
    fn to_inbound(self: &Arc<Self>, message: WaMessage, sender: Sender) -> Option<Inbound> {
        let owner = sender == Sender::Owner;
        let mut text = message.text.clone();
        if self.config.mode == WhatsAppMode::Friends && !owner {
            match strip_wake_word(&text) {
                Some(request) => text = request,
                None if self.is_engaged(&message.chat) => {}
                None => return None,
            }
            self.mark_engaged(&message.chat);
        }
        let media = message
            .media
            .iter()
            .map(|media| {
                Arc::new(WhatsAppMedia {
                    channel: self.clone(),
                    message_id: message.id.clone(),
                    info: media.clone(),
                }) as Arc<dyn MediaSource>
            })
            .collect();
        Some(Inbound {
            channel: "whatsapp",
            chat: message.chat.clone(),
            user: if owner {
                self.status
                    .lock()
                    .account
                    .clone()
                    .unwrap_or_else(|| "owner".into())
                    .trim_start_matches('+')
                    .to_owned()
            } else {
                message.sender.clone()
            },
            name: if owner {
                "you".to_owned()
            } else if message.name.trim().is_empty() {
                format!("+{}", message.sender)
            } else {
                format!("{} (+{})", message.name.trim(), message.sender)
            },
            text,
            media,
            trusted: owner,
            strangers: if self.config.mode == WhatsAppMode::Friends {
                Strangers::Quiet
            } else {
                Strangers::Pair
            },
        })
    }

    /// Read events and write commands until the sidecar's pipes close.
    pub async fn run_session<R, W>(
        self: &Arc<Self>,
        reader: R,
        writer: W,
        gateway: Arc<Gateway>,
        cancel: &CancellationToken,
    ) where
        R: AsyncRead + Unpin,
        W: AsyncWrite + Unpin + Send + 'static,
    {
        let (commands, mut queue) = mpsc::unbounded_channel::<String>();
        let link = Arc::new(Link {
            commands,
            pending: Mutex::new(HashMap::new()),
        });
        *self.link.lock() = Some(link.clone());
        let writer_task = tokio::spawn(async move {
            let mut writer = writer;
            while let Some(line) = queue.recv().await {
                if writer
                    .write_all(format!("{line}\n").as_bytes())
                    .await
                    .is_err()
                    || writer.flush().await.is_err()
                {
                    break;
                }
            }
        });
        let mut lines = BufReader::new(reader).lines();
        loop {
            let line = tokio::select! {
                () = cancel.cancelled() => break,
                line = lines.next_line() => line,
            };
            match line {
                Ok(Some(line)) if line.trim().is_empty() => {}
                Ok(Some(line)) => match serde_json::from_str::<Event>(&line) {
                    Ok(event) => self.on_event(event, &link, &gateway).await,
                    Err(error) => tracing::debug!("whatsapp sidecar sent an unknown line: {error}"),
                },
                Ok(None) | Err(_) => break,
            }
        }
        *self.link.lock() = None;
        // Dropping the senders wakes every waiting command with an error.
        link.pending.lock().clear();
        writer_task.abort();
    }

    async fn run_sidecar(
        self: &Arc<Self>,
        gateway: &Arc<Gateway>,
        cancel: &CancellationToken,
    ) -> std::io::Result<()> {
        let mut command = tokio::process::Command::new(&self.config.sidecar);
        command
            .arg("--data-dir")
            .arg(&self.config.data_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        #[cfg(windows)]
        command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
        let mut child = command.spawn()?;
        let (Some(stdin), Some(stdout)) = (child.stdin.take(), child.stdout.take()) else {
            return Err(std::io::Error::other("the sidecar has no pipes"));
        };
        self.run_session(stdout, stdin, gateway.clone(), cancel)
            .await;
        if cancel.is_cancelled() {
            // Ask politely first so the session is saved, then make sure.
            let _ = tokio::time::timeout(Duration::from_secs(2), child.wait()).await;
        }
        let _ = child.kill().await;
        Ok(())
    }
}

/// Who a routed message is from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Sender {
    /// The linked account itself.
    Owner,
    /// Someone else, subject to the allowlist.
    Person,
}

struct WhatsAppMedia {
    channel: Arc<WhatsAppChannel>,
    message_id: String,
    info: WaMedia,
}

#[async_trait]
impl MediaSource for WhatsAppMedia {
    fn kind(&self) -> MediaKind {
        match self.info.kind.as_str() {
            "image" => MediaKind::Image,
            "audio" => MediaKind::Audio,
            "video" => MediaKind::Video,
            "sticker" => MediaKind::Sticker,
            _ => MediaKind::Document,
        }
    }

    fn label(&self) -> String {
        self.info.name.clone().unwrap_or_else(|| {
            match self.kind() {
                MediaKind::Image => "a picture",
                MediaKind::Document => "a file",
                MediaKind::Audio => "a voice note",
                MediaKind::Video => "a video",
                MediaKind::Sticker => "a sticker",
            }
            .to_owned()
        })
    }

    fn size_hint(&self) -> Option<u64> {
        self.info.size
    }

    async fn fetch(&self, max_bytes: u64) -> Result<Download> {
        let directory = self.channel.config.data_dir.join("downloads");
        std::fs::create_dir_all(&directory)?;
        let reply = self
            .channel
            .command(
                json!({
                    "cmd": "download",
                    "message": self.message_id,
                    "index": self.info.index,
                    "max_bytes": max_bytes,
                    "dir": directory,
                }),
                true,
            )
            .await?;
        let path = reply
            .path
            .map(PathBuf::from)
            .ok_or_else(|| PokError::Provider("WhatsApp returned no file".into()))?;
        // Only a file the sidecar wrote into the download folder is read.
        let path = dunce::canonicalize(&path)?;
        if !path.starts_with(dunce::canonicalize(&directory)?) {
            return Err(PokError::Provider(
                "WhatsApp returned an unexpected path".into(),
            ));
        }
        let bytes = std::fs::read(&path);
        let _ = std::fs::remove_file(&path);
        Ok(Download {
            kind: self.kind(),
            name: self.info.name.as_deref().map(safe_component),
            mime: self.info.mime.clone(),
            bytes: bytes?,
        })
    }
}

#[async_trait]
impl Channel for WhatsAppChannel {
    fn name(&self) -> &'static str {
        "whatsapp"
    }

    async fn send_text(&self, chat: &str, text: &str) -> Result<()> {
        let prefix = reply_prefix(self.config.mode);
        for part in super::split_text(&to_whatsapp_markup(text), MESSAGE_CHARS) {
            let reply = self
                .command(
                    json!({"cmd": "send_text", "chat": chat, "text": format!("{prefix}{part}")}),
                    true,
                )
                .await?;
            if let Some(id) = reply.message_id {
                self.remember_send(id);
            }
        }
        Ok(())
    }

    async fn send_typing(&self, chat: &str) {
        let _ = self
            .command(json!({"cmd": "typing", "chat": chat}), false)
            .await;
    }

    async fn listen(self: Arc<Self>, gateway: Arc<Gateway>, cancel: CancellationToken) {
        let this = self;
        let mut backoff = Duration::from_secs(2);
        loop {
            this.update(|status| {
                if status.state != LinkState::LoggedOut {
                    status.state = LinkState::Connecting;
                }
            });
            gateway.notify_changed();
            let started = std::time::Instant::now();
            let ran = this.run_sidecar(&gateway, &cancel).await;
            if this.wipe_session.swap(false, Ordering::SeqCst) {
                forget_session(&this.config.data_dir);
            }
            match ran {
                Ok(()) => {}
                Err(error) => {
                    let missing = error.kind() == std::io::ErrorKind::NotFound;
                    this.update(|status| {
                        status.state = LinkState::Failed;
                        status.error = Some(if missing {
                            "The WhatsApp component file is missing. Turn WhatsApp off and on again, or reinstall POK-Agent."
                                .to_owned()
                        } else {
                            format!("Could not start the WhatsApp component: {error}")
                        });
                    });
                    gateway.notify_changed();
                    if missing {
                        // Retrying cannot help until the file is back.
                        cancel.cancelled().await;
                        return;
                    }
                }
            }
            if cancel.is_cancelled() {
                return;
            }
            // A healthy run resets the delay; a crash loop backs off.
            if started.elapsed() > Duration::from_secs(60) {
                backoff = Duration::from_secs(2);
            }
            tokio::select! {
                () = cancel.cancelled() => return,
                () = tokio::time::sleep(backoff) => {}
            }
            backoff = (backoff * 2).min(Duration::from_secs(60));
        }
    }
}

/// Delete the saved linked-device session, so the next start asks for a scan.
pub fn forget_session(data_dir: &std::path::Path) {
    let _ = std::fs::remove_dir_all(data_dir.join("session"));
}

/// Whether a session is saved, which means this computer has been linked.
pub fn session_exists(data_dir: &std::path::Path) -> bool {
    data_dir.join("session").join("whatsapp.db").is_file()
}

// ---- text and QR helpers ---------------------------------------------------

/// What every reply starts with. On the owner's own number it says the agent
/// wrote it: that keeps the self-chat from answering itself, and tells friends
/// it is not the owner typing. A dedicated number needs none.
pub fn reply_prefix(mode: WhatsAppMode) -> &'static str {
    match mode {
        WhatsAppMode::Bot => "",
        WhatsAppMode::SelfChat | WhatsAppMode::Friends => SELF_CHAT_PREFIX,
    }
}

/// The request after the wake word ("pok", "@pok", "pok agent", "pok-agent",
/// any case), or `None` when the text does not start with it. "pokemon" does
/// not count. `Some("")` means the wake word alone.
pub fn strip_wake_word(text: &str) -> Option<String> {
    let trimmed = text.trim_start();
    let rest = trimmed.strip_prefix('@').unwrap_or(trimmed);
    // ASCII lowercasing keeps every byte offset, so slicing `rest` is safe.
    let lower = rest.to_ascii_lowercase();
    let after_word = lower.strip_prefix(WAKE_WORD)?;
    let mut consumed = rest.len() - after_word.len();
    for suffix in ["-agent", " agent"] {
        if lower[consumed..].starts_with(suffix) {
            consumed += suffix.len();
            break;
        }
    }
    let remainder = &rest[consumed..];
    // The word must end here.
    if remainder
        .chars()
        .next()
        .is_some_and(|c| !(c.is_whitespace() || matches!(c, ',' | ':' | ';' | '!' | '.' | '?')))
    {
        return None;
    }
    Some(
        remainder
            .trim_start_matches(|c: char| c.is_whitespace() || matches!(c, ',' | ':' | ';' | '!'))
            .trim()
            .to_owned(),
    )
}

/// The QR code to scan, as an SVG image, or `None` if it cannot be drawn.
pub fn qr_svg(code: &str) -> Option<String> {
    let qr = qrcode::QrCode::new(code.as_bytes()).ok()?;
    Some(
        qr.render::<qrcode::render::svg::Color>()
            .min_dimensions(256, 256)
            .quiet_zone(true)
            .build(),
    )
}

/// Chat Markdown in WhatsApp's own markup: `**bold**` becomes `*bold*`,
/// `~~gone~~` becomes `~gone~`, headings become bold lines and links keep
/// their address. Code stays as it is.
pub fn to_whatsapp_markup(text: &str) -> String {
    let mut out = Vec::new();
    let mut in_code = false;
    for line in text.lines() {
        if line.trim_start().starts_with("```") {
            in_code = !in_code;
            out.push(line.to_owned());
            continue;
        }
        if in_code {
            out.push(line.to_owned());
            continue;
        }
        let trimmed = line.trim_start();
        let heading = trimmed.trim_start_matches('#');
        let converted = if trimmed.starts_with('#')
            && heading.starts_with(' ')
            && trimmed.len() - heading.len() <= 6
        {
            format!("*{}*", heading.trim().trim_matches('*'))
        } else {
            line.to_owned()
        };
        out.push(convert_inline(&converted));
    }
    out.join("\n")
}

fn convert_inline(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let chars: Vec<char> = line.chars().collect();
    let mut index = 0;
    let mut in_code = false;
    while index < chars.len() {
        let c = chars[index];
        if c == '`' {
            in_code = !in_code;
        }
        if !in_code {
            if c == '*' && chars.get(index + 1) == Some(&'*') {
                out.push('*');
                index += 2;
                continue;
            }
            if c == '~' && chars.get(index + 1) == Some(&'~') {
                out.push('~');
                index += 2;
                continue;
            }
            // [label](https://example.com) -> label (https://example.com)
            if c == '['
                && let Some(close) = chars[index..].iter().position(|c| *c == ']')
                && chars.get(index + close + 1) == Some(&'(')
                && let Some(end) = chars[index + close + 1..].iter().position(|c| *c == ')')
            {
                let label: String = chars[index + 1..index + close].iter().collect();
                let url: String = chars[index + close + 2..index + close + 1 + end]
                    .iter()
                    .collect();
                out.push_str(&format!("{label} ({url})"));
                index += close + 1 + end + 1;
                continue;
            }
        }
        out.push(c);
        index += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markdown_becomes_whatsapp_markup() {
        assert_eq!(
            to_whatsapp_markup("This is **bold** and ~~gone~~."),
            "This is *bold* and ~gone~."
        );
        assert_eq!(to_whatsapp_markup("## Results\nline"), "*Results*\nline");
        assert_eq!(
            to_whatsapp_markup("See [the docs](https://example.com/a) now"),
            "See the docs (https://example.com/a) now"
        );
        // Code is left alone, inline and fenced.
        assert_eq!(
            to_whatsapp_markup("run `a **b** c` now"),
            "run `a **b** c` now"
        );
        assert_eq!(
            to_whatsapp_markup("```\n**keep** ## this\n```"),
            "```\n**keep** ## this\n```"
        );
        // A hash that is not a heading stays.
        assert_eq!(to_whatsapp_markup("#hashtag"), "#hashtag");
        assert_eq!(to_whatsapp_markup("plain"), "plain");
    }

    #[test]
    fn the_qr_code_is_drawn_as_an_svg() {
        let svg = qr_svg("2@abcdefghijklmnop,xyz==,123").unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("viewBox") || svg.contains("width"));
    }

    fn channel(mode: WhatsAppMode) -> Arc<WhatsAppChannel> {
        WhatsAppChannel::new(WhatsAppConfig {
            sidecar: PathBuf::from("unused"),
            data_dir: PathBuf::from("unused"),
            mode,
        })
    }

    fn message(id: &str, chat_kind: &str, from_me: bool, self_chat: bool, text: &str) -> WaMessage {
        WaMessage {
            id: id.into(),
            chat: "15550100@s.whatsapp.net".into(),
            chat_kind: chat_kind.into(),
            sender: "15550100".into(),
            name: "Sam".into(),
            from_me,
            self_chat,
            text: text.into(),
            timestamp: 1_000,
            media: Vec::new(),
        }
    }

    #[test]
    fn self_chat_mode_answers_only_the_owner_writing_to_themselves() {
        let channel = channel(WhatsAppMode::SelfChat);
        let now = 1_010;
        assert_eq!(
            channel.route(&message("1", "direct", true, true, "research x"), now),
            Some(Sender::Owner)
        );
        // Someone else writing to the owner's number is never answered.
        assert_eq!(
            channel.route(&message("2", "direct", false, false, "hi"), now),
            None
        );
        // The owner writing to a friend is not a request either.
        assert_eq!(
            channel.route(&message("3", "direct", true, false, "hi"), now),
            None
        );
        // Groups and broadcast lists are never a command surface.
        assert_eq!(
            channel.route(&message("4", "group", true, true, "hi"), now),
            None
        );
        assert_eq!(
            channel.route(&message("5", "broadcast", true, true, "hi"), now),
            None
        );
    }

    #[test]
    fn the_agents_own_replies_in_the_self_chat_are_not_new_tasks() {
        let channel = channel(WhatsAppMode::SelfChat);
        let now = 1_010;
        // By the prefix every reply carries...
        let echo = message(
            "9",
            "direct",
            true,
            true,
            &format!("{SELF_CHAT_PREFIX}Done."),
        );
        assert_eq!(channel.route(&echo, now), None);
        // ...and by the id of a message we sent, even without it.
        channel.remember_send("7".into());
        assert_eq!(
            channel.route(&message("7", "direct", true, true, "Done."), now),
            None
        );
        assert_eq!(
            channel.route(&message("8", "direct", true, true, "Done."), now),
            Some(Sender::Owner)
        );
    }

    #[test]
    fn bot_mode_routes_other_people_and_never_its_own_account() {
        let channel = channel(WhatsAppMode::Bot);
        let now = 1_010;
        assert_eq!(
            channel.route(&message("1", "direct", false, false, "hi"), now),
            Some(Sender::Person)
        );
        assert_eq!(
            channel.route(&message("2", "direct", true, false, "hi"), now),
            None
        );
        assert_eq!(
            channel.route(&message("3", "direct", true, true, "hi"), now),
            None
        );
        assert_eq!(
            channel.route(&message("4", "group", false, false, "hi"), now),
            None
        );
        let mut anonymous = message("5", "direct", false, false, "hi");
        anonymous.sender.clear();
        assert_eq!(
            channel.route(&anonymous, now),
            None,
            "no identity, no answer"
        );
    }

    #[test]
    fn messages_replayed_after_a_reconnect_are_dropped() {
        let channel = channel(WhatsAppMode::Bot);
        let sent_at = 1_000;
        assert!(
            channel
                .route(&message("1", "direct", false, false, "hi"), sent_at + 60)
                .is_some()
        );
        assert!(
            channel
                .route(
                    &message("2", "direct", false, false, "hi"),
                    sent_at + MAX_MESSAGE_AGE_SECS + 1
                )
                .is_none()
        );
    }

    #[test]
    fn friends_mode_serves_the_owners_own_chat_and_friends_only() {
        let channel = channel(WhatsAppMode::Friends);
        let now = 1_010;
        assert_eq!(
            channel.route(&message("1", "direct", true, true, "research x"), now),
            Some(Sender::Owner)
        );
        assert_eq!(
            channel.route(&message("2", "direct", false, false, "pok hi"), now),
            Some(Sender::Person)
        );
        // What the owner types to a friend is a normal chat, never a request.
        assert_eq!(
            channel.route(&message("3", "direct", true, false, "pok hi"), now),
            None
        );
        // The agent's own replies in the self-chat are not new requests.
        let echo = message(
            "4",
            "direct",
            true,
            true,
            &format!("{SELF_CHAT_PREFIX}Done."),
        );
        assert_eq!(channel.route(&echo, now), None);
        // Groups stay out.
        assert_eq!(
            channel.route(&message("5", "group", false, false, "pok hi"), now),
            None
        );
    }

    #[test]
    fn the_wake_word_is_recognised_in_its_usual_spellings_only() {
        let request = |text: &str| strip_wake_word(text);
        assert_eq!(
            request("pok what is on today").as_deref(),
            Some("what is on today")
        );
        assert_eq!(request("  Pok, research X").as_deref(), Some("research X"));
        assert_eq!(request("@POK: do it").as_deref(), Some("do it"));
        assert_eq!(
            request("pok agent summarize this").as_deref(),
            Some("summarize this")
        );
        assert_eq!(
            request("POK-Agent open the calendar").as_deref(),
            Some("open the calendar")
        );
        assert_eq!(request("pok").as_deref(), Some(""), "the word alone");
        // Ordinary chat that merely begins with the same letters is not a request.
        assert_eq!(request("pokemon is great"), None);
        assert_eq!(request("poke me later"), None);
        assert_eq!(request("pok-man"), None);
        assert_eq!(
            request("are we on for dinner? pok"),
            None,
            "it must come first"
        );
        assert_eq!(request(""), None);
        // Non-ASCII text in front is simply not the wake word.
        assert_eq!(request("é pok x"), None);
    }

    #[test]
    fn in_friends_mode_only_requests_with_the_wake_word_reach_the_agent() {
        let channel = channel(WhatsAppMode::Friends);
        let friend = |text: &str| message("9", "direct", false, false, text);
        // A normal chat with the owner is never touched.
        assert!(
            channel
                .to_inbound(friend("are we still on for dinner?"), Sender::Person)
                .is_none()
        );
        let request = channel
            .to_inbound(friend("Pok, what is on my calendar?"), Sender::Person)
            .unwrap();
        assert_eq!(request.text, "what is on my calendar?");
        assert_eq!(
            request.strangers,
            Strangers::Quiet,
            "strangers get no reply"
        );
        assert!(!request.trusted, "a friend is never the owner");

        // The rest of an album, or a message the app split, rides along...
        assert_eq!(
            channel
                .to_inbound(friend("and the second part"), Sender::Person)
                .unwrap()
                .text,
            "and the second part"
        );
        // ...but a while later it is an ordinary chat again.
        channel.engaged.lock().insert(
            "15550100@s.whatsapp.net".into(),
            Instant::now().checked_sub(Duration::from_secs(30)).unwrap(),
        );
        assert!(
            channel
                .to_inbound(friend("thanks!"), Sender::Person)
                .is_none()
        );

        // The owner never needs the word.
        let owner = channel
            .to_inbound(
                message("10", "direct", true, true, "just do it"),
                Sender::Owner,
            )
            .unwrap();
        assert_eq!(owner.text, "just do it");
        assert!(owner.trusted);
    }

    #[test]
    fn a_dedicated_number_does_not_use_a_wake_word_and_pairs_strangers() {
        let channel = channel(WhatsAppMode::Bot);
        let inbound = channel
            .to_inbound(
                message("1", "direct", false, false, "hello"),
                Sender::Person,
            )
            .unwrap();
        assert_eq!(inbound.text, "hello");
        assert_eq!(inbound.strangers, Strangers::Pair);
    }

    #[test]
    fn replies_name_the_agent_except_on_a_dedicated_number() {
        assert_eq!(reply_prefix(WhatsAppMode::SelfChat), SELF_CHAT_PREFIX);
        assert_eq!(reply_prefix(WhatsAppMode::Friends), SELF_CHAT_PREFIX);
        assert_eq!(reply_prefix(WhatsAppMode::Bot), "");
    }

    #[test]
    fn events_parse_from_the_sidecar_lines() {
        let line = r#"{"event":"message","id":"A1","chat":"15550100@s.whatsapp.net","chat_kind":"direct","sender":"15550100","from_me":false,"text":"hello","timestamp":5,"media":[{"index":0,"kind":"image","mime":"image/jpeg","size":2048}]}"#;
        let Event::Message(message) = serde_json::from_str(line).unwrap() else {
            panic!("not a message");
        };
        assert_eq!(message.text, "hello");
        assert_eq!(message.media[0].kind, "image");
        assert!(matches!(
            serde_json::from_str::<Event>(r#"{"event":"qr","code":"2@x"}"#).unwrap(),
            Event::Qr { .. }
        ));
        assert!(serde_json::from_str::<Event>(r#"{"event":"nonsense"}"#).is_err());
    }
}
