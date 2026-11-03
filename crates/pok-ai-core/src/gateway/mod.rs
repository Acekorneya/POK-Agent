//! Always-on messaging gateway: the owner messages the agent from Telegram or
//! WhatsApp (text, pictures, files), and scheduled jobs run on their own.
//! Messages become tasks for a `GatewayHost` (the desktop app), which runs
//! them like any other prompt and hands back the answer.
//!
//! Each messaging app is a [`Channel`]. Chats and people are addressed as
//! `<channel>:<id>`, so one allowlist, one set of commands and one scheduler
//! serve every channel.
//!
//! Safe by default: only people the owner approved on the computer can
//! message the agent (a stranger just gets a one-time pairing code), only
//! private chats count, attachments are downloaded only for approved people,
//! and messages that arrived while the gateway was off are dropped rather
//! than acted on late.

pub mod access;
pub mod media;
pub mod schedule;
pub mod sidecar;
pub mod telegram;
pub mod whatsapp;

use std::{collections::HashMap, path::PathBuf, sync::Arc, time::Duration};

use async_trait::async_trait;
use chrono::{Local, Utc};
use parking_lot::Mutex;
use tokio_util::sync::CancellationToken;

use crate::error::Result;
use access::{AccessControl, address, split_address};
use media::{MediaSource, Prepared};
use schedule::{JobStore, parse_schedule};
use telegram::TelegramClient;

/// Where a task came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskOrigin {
    Message { channel: String, user: String },
    Schedule { job: String },
}

#[derive(Debug, Clone)]
pub struct GatewayTask {
    pub id: String,
    pub origin: TaskOrigin,
    pub prompt: String,
    /// The chat that gets the answer, as `<channel>:<id>`.
    pub chat: String,
    /// Pictures the sender attached, as base64 PNG for a vision model.
    pub images: Vec<String>,
    /// Files the sender attached (pictures included), saved in the inbox.
    pub files: Vec<PathBuf>,
}

/// A message from a messaging app, ready for the dispatcher.
pub struct Inbound {
    pub channel: &'static str,
    /// The chat's id on that channel.
    pub chat: String,
    /// The sender's id on that channel; this is what the allowlist checks.
    pub user: String,
    /// A name to show the owner when the sender asks to be approved.
    pub name: String,
    pub text: String,
    pub media: Vec<Arc<dyn MediaSource>>,
    /// The channel itself vouches for the sender, without an allowlist entry:
    /// the owner's own account writing to itself.
    pub trusted: bool,
}

/// A messaging app the gateway listens on and answers through.
#[async_trait]
pub trait Channel: Send + Sync {
    /// The lowercase name used in addresses, such as `telegram`.
    fn name(&self) -> &'static str;
    /// Send plain text to a chat (its channel-local id).
    async fn send_text(&self, chat: &str, text: &str) -> Result<()>;
    async fn send_typing(&self, chat: &str);
    /// Receive messages and pass each to [`Gateway::handle_inbound`] until
    /// `cancel` fires.
    async fn listen(self: Arc<Self>, gateway: Arc<Gateway>, cancel: CancellationToken);
}

/// The agent behind the gateway.
#[async_trait]
pub trait GatewayHost: Send + Sync {
    /// Start a task. `Err` carries a sentence for the sender, for example
    /// that the agent is busy.
    async fn submit(&self, task: GatewayTask) -> std::result::Result<(), String>;
    async fn stop(&self) -> String;
    async fn new_conversation(&self) -> String;
    async fn status(&self) -> String;
    /// Answer the approval the agent is waiting on.
    async fn approve(&self, allow: bool) -> String;
    /// Pairing requests or jobs changed; the settings page should refresh.
    fn changed(&self) {}
}

const HELP: &str = "POK-Agent is listening. Send a message and it becomes a task. Pictures and files you attach go with it.\n\n\
/status - what it is doing\n\
/stop - stop the current task\n\
/new - start a fresh conversation\n\
/approve, /deny - answer a permission request\n\
/schedule daily 09:00 <task> - run a task every day\n\
/schedule every 30m <task> - run a task on an interval\n\
/jobs - list scheduled tasks\n\
/unschedule <id> - remove one\n\
/sethome - send scheduled results to this chat\n\
/whoami - who you are to the bot";

#[derive(Debug, PartialEq, Eq)]
pub enum Command {
    Help,
    WhoAmI,
    Status,
    Stop,
    New,
    SetHome,
    Jobs,
    Approve,
    Deny,
    Schedule(String),
    Unschedule(String),
    Unknown(String),
    Task(String),
}

pub fn parse_command(text: &str) -> Command {
    let text = text.trim();
    let Some(rest) = text.strip_prefix('/') else {
        return Command::Task(text.to_owned());
    };
    let (word, argument) = rest
        .split_once(char::is_whitespace)
        .map_or((rest, ""), |(word, argument)| (word, argument.trim()));
    // `/status@my_bot` in a chat with several bots.
    let word = word.split('@').next().unwrap_or(word).to_lowercase();
    match word.as_str() {
        "start" | "help" => Command::Help,
        "whoami" => Command::WhoAmI,
        "status" => Command::Status,
        "stop" | "cancel" => Command::Stop,
        "new" | "reset" => Command::New,
        "sethome" => Command::SetHome,
        "jobs" => Command::Jobs,
        "approve" => Command::Approve,
        "deny" => Command::Deny,
        "schedule" => Command::Schedule(argument.to_owned()),
        "unschedule" => Command::Unschedule(argument.to_owned()),
        other => Command::Unknown(other.to_owned()),
    }
}

pub struct Gateway {
    channels: Vec<Arc<dyn Channel>>,
    access: Arc<AccessControl>,
    jobs: Arc<JobStore>,
    host: Arc<dyn GatewayHost>,
    /// Where attachments are saved, one folder per task.
    inbox: PathBuf,
    /// Messages from one chat that arrive within a window become one task:
    /// the photos of an album, or a long text the app split in two.
    text_window: Duration,
    media_window: Duration,
    batches: Mutex<HashMap<String, Batch>>,
}

struct Batch {
    name: String,
    texts: Vec<String>,
    media: Vec<Arc<dyn MediaSource>>,
}

/// How often scheduled jobs are checked.
const SCHEDULE_TICK: Duration = Duration::from_secs(30);
const TEXT_WINDOW: Duration = Duration::from_millis(400);
const MEDIA_WINDOW: Duration = Duration::from_millis(1_500);

impl Gateway {
    pub fn new(
        channels: Vec<Arc<dyn Channel>>,
        access: Arc<AccessControl>,
        jobs: Arc<JobStore>,
        host: Arc<dyn GatewayHost>,
        inbox: PathBuf,
    ) -> Self {
        Self {
            channels,
            access,
            jobs,
            host,
            inbox,
            text_window: TEXT_WINDOW,
            media_window: MEDIA_WINDOW,
            batches: Mutex::new(HashMap::new()),
        }
    }

    /// Change how long a chat's messages are gathered into one task; zero
    /// starts every message at once.
    #[must_use]
    pub fn with_batch_windows(mut self, text: Duration, media: Duration) -> Self {
        self.text_window = text;
        self.media_window = media;
        self
    }

    pub fn access(&self) -> &Arc<AccessControl> {
        &self.access
    }

    pub fn jobs(&self) -> &Arc<JobStore> {
        &self.jobs
    }

    /// Tell the host that something the settings page shows has changed.
    pub fn notify_changed(&self) {
        self.host.changed();
    }

    fn channel_of(&self, chat: &str) -> Option<&Arc<dyn Channel>> {
        let (name, _) = split_address(chat);
        self.channels.iter().find(|channel| channel.name() == name)
    }

    /// Send text to a chat (`<channel>:<id>`), ignoring delivery errors the
    /// caller cannot fix.
    pub async fn send(&self, chat: &str, text: &str) {
        let Some(channel) = self.channel_of(chat) else {
            tracing::warn!("gateway has no channel for {chat}");
            return;
        };
        if let Err(error) = channel.send_text(split_address(chat).1, text).await {
            tracing::warn!("gateway could not deliver a message: {error}");
        }
    }

    async fn send_typing(&self, chat: &str) {
        if let Some(channel) = self.channel_of(chat) {
            channel.send_typing(split_address(chat).1).await;
        }
    }

    /// Listen on every channel and run the scheduler until `cancel` fires.
    pub async fn run(self: Arc<Self>, cancel: CancellationToken) {
        let inbox = self.inbox.clone();
        tokio::task::spawn_blocking(move || media::sweep_inbox(&inbox, media::INBOX_RETENTION));
        let mut workers = vec![tokio::spawn({
            let gateway = self.clone();
            let cancel = cancel.clone();
            async move { gateway.schedule_loop(cancel).await }
        })];
        for channel in &self.channels {
            let channel = channel.clone();
            let gateway = self.clone();
            let cancel = cancel.clone();
            workers.push(tokio::spawn(async move {
                channel.listen(gateway, cancel).await;
            }));
        }
        cancel.cancelled().await;
        for worker in workers {
            worker.abort();
        }
    }

    async fn schedule_loop(&self, cancel: CancellationToken) {
        loop {
            tokio::select! {
                () = cancel.cancelled() => return,
                () = tokio::time::sleep(SCHEDULE_TICK) => {}
            }
            self.run_due_jobs().await;
        }
    }

    /// Start the first due job; the agent does one task at a time, so the
    /// rest wait for a later tick.
    pub async fn run_due_jobs(&self) {
        let Some(chat) = self.access.home_chat() else {
            return;
        };
        let Some(job) = self.jobs.due(Local::now()).into_iter().next() else {
            return;
        };
        let task = GatewayTask {
            id: uuid::Uuid::new_v4().simple().to_string(),
            origin: TaskOrigin::Schedule {
                job: job.name.clone(),
            },
            prompt: job.prompt.clone(),
            chat: chat.clone(),
            images: Vec::new(),
            files: Vec::new(),
        };
        if self.host.submit(task).await.is_ok() {
            if let Err(error) = self.jobs.mark_run(&job.id, Utc::now()) {
                tracing::warn!("could not record a job run: {error}");
            }
            self.send(&chat, &format!("Running scheduled task: {}", job.name))
                .await;
            self.host.changed();
        }
    }

    /// One incoming message from any channel.
    pub async fn handle_inbound(self: &Arc<Self>, inbound: Inbound) {
        let user = address(inbound.channel, &inbound.user);
        let chat = address(inbound.channel, &inbound.chat);
        if !inbound.trusted && !self.access.is_allowed(&user) {
            self.greet_stranger(&user, &chat, &inbound.name).await;
            return;
        }
        // The owner writing from their own linked account: make that chat
        // the home for scheduled results if none is chosen yet.
        if inbound.trusted && self.access.home_chat().is_none() {
            let _ = self.access.set_home_chat(&chat);
            self.host.changed();
        }
        let text = inbound.text.trim().to_owned();
        // A caption that starts with a slash is still a request, not a command.
        if inbound.media.is_empty() && text.starts_with('/') {
            self.run_command(&chat, &inbound.name, &text).await;
            return;
        }
        self.queue_task(chat, inbound, text).await;
    }

    async fn greet_stranger(&self, user: &str, chat: &str, name: &str) {
        match self.access.request_pairing(user, chat, name) {
            Some(code) => {
                self.send(
                    chat,
                    &format!(
                        "This assistant is private. To use it, the owner has to approve your pairing code in POK-Agent (Settings, Gateway): {code}\n\nThe code works for one hour."
                    ),
                )
                .await;
                self.host.changed();
            }
            None => tracing::info!("gateway ignored a stranger: too many pairing requests open"),
        }
    }

    async fn run_command(&self, chat: &str, name: &str, text: &str) {
        let reply = match parse_command(text) {
            Command::Help => HELP.to_owned(),
            Command::WhoAmI => format!("You are {name}. This chat is {chat}."),
            Command::Status => self.host.status().await,
            Command::Stop => self.host.stop().await,
            Command::New => self.host.new_conversation().await,
            Command::Approve => self.host.approve(true).await,
            Command::Deny => self.host.approve(false).await,
            Command::SetHome => match self.access.set_home_chat(chat) {
                Ok(()) => "Scheduled results will be sent to this chat.".into(),
                Err(error) => format!("Could not save that: {error}"),
            },
            Command::Jobs => self.describe_jobs(),
            Command::Schedule(spec) => self.add_job(chat, &spec),
            Command::Unschedule(id) => match self.jobs.remove(id.trim()) {
                Ok(true) => {
                    self.host.changed();
                    "Removed.".into()
                }
                Ok(false) => "No scheduled task has that id (see /jobs).".into(),
                Err(error) => format!("Could not remove it: {error}"),
            },
            Command::Unknown(word) => format!("I do not know /{word}. Try /help."),
            Command::Task(prompt) if prompt.is_empty() => return,
            // Not reachable: only slash-prefixed text is parsed here.
            Command::Task(_) => return,
        };
        self.send(chat, &reply).await;
    }

    /// A task message. Messages from the same chat that follow within the
    /// batching window join it.
    async fn queue_task(self: &Arc<Self>, chat: String, inbound: Inbound, text: String) {
        let window = if inbound.media.is_empty() {
            self.text_window
        } else {
            self.media_window
        };
        if window.is_zero() {
            self.start_task(&chat, &inbound.name, vec![text], inbound.media)
                .await;
            return;
        }
        let first = {
            let mut batches = self.batches.lock();
            if let Some(batch) = batches.get_mut(&chat) {
                batch.texts.push(text);
                batch.media.extend(inbound.media);
                false
            } else {
                batches.insert(
                    chat.clone(),
                    Batch {
                        name: inbound.name,
                        texts: vec![text],
                        media: inbound.media,
                    },
                );
                true
            }
        };
        if first {
            let gateway = self.clone();
            tokio::spawn(async move {
                tokio::time::sleep(window).await;
                let batch = gateway.batches.lock().remove(&chat);
                if let Some(batch) = batch {
                    gateway
                        .start_task(&chat, &batch.name, batch.texts, batch.media)
                        .await;
                }
            });
        }
    }

    async fn start_task(
        &self,
        chat: &str,
        name: &str,
        texts: Vec<String>,
        sources: Vec<Arc<dyn MediaSource>>,
    ) {
        let id = uuid::Uuid::new_v4().simple().to_string();
        let mut prepared = Prepared::default();
        if !sources.is_empty() {
            // Downloads can take a while: show that the agent is on it.
            self.send_typing(chat).await;
            prepared = media::prepare(&self.inbox, &id, &sources).await;
        }
        let text = texts
            .iter()
            .map(|text| text.trim())
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        let prompt = media::compose_prompt(&text, &prepared);
        if prompt.is_empty() {
            // Only a sticker, or attachments that all failed: tell the sender
            // why nothing happened rather than staying silent.
            if !prepared.notes.is_empty() {
                self.send(chat, &prepared.notes.join("\n")).await;
            }
            return;
        }
        let acknowledgement = acknowledgement(&prepared);
        let task = GatewayTask {
            id,
            origin: TaskOrigin::Message {
                channel: split_address(chat).0.to_owned(),
                user: name.to_owned(),
            },
            prompt,
            chat: chat.to_owned(),
            images: prepared.images,
            files: prepared.files,
        };
        match self.host.submit(task).await {
            Ok(()) => {
                self.send_typing(chat).await;
                self.send(chat, &acknowledgement).await;
            }
            Err(reason) => self.send(chat, &reason).await,
        }
    }

    fn describe_jobs(&self) -> String {
        let jobs = self.jobs.list();
        if jobs.is_empty() {
            return "No scheduled tasks. Try: /schedule daily 09:00 summarize my day".into();
        }
        jobs.iter()
            .map(|job| {
                format!(
                    "{} - {} ({}{})",
                    job.id,
                    job.name,
                    job.schedule.describe(),
                    if job.enabled { "" } else { ", paused" }
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn add_job(&self, chat: &str, spec: &str) -> String {
        let Some((schedule, prompt)) = parse_schedule(spec) else {
            return "Use: /schedule daily 09:00 <task> or /schedule every 30m <task> (at least every 5m).".into();
        };
        match self.jobs.add("", &prompt, schedule) {
            Ok(job) => {
                self.host.changed();
                let home = if self.access.home_chat().as_deref() == Some(chat) {
                    ""
                } else {
                    " Results go to the home chat; send /sethome here to change it."
                };
                format!("Scheduled {} ({}).{home}", job.id, job.schedule.describe())
            }
            Err(error) => format!("Could not schedule that: {error}"),
        }
    }
}

/// What the sender hears right after a task starts.
fn acknowledgement(prepared: &Prepared) -> String {
    let mut text = String::from("On it. I will message you when it is done.");
    if !prepared.images.is_empty() {
        text = format!(
            "On it, with your {}. I will message you when it is done.",
            if prepared.images.len() == 1 {
                "picture".to_owned()
            } else {
                format!("{} pictures", prepared.images.len())
            }
        );
    }
    if !prepared.notes.is_empty() {
        text.push_str("\n\n");
        text.push_str(&prepared.notes.join("\n"));
    }
    text
}

/// Split `text` into chunks of at most `limit` characters, preferring line
/// breaks. Empty text becomes a short placeholder so something is sent.
pub fn split_text(text: &str, limit: usize) -> Vec<String> {
    let text = text.trim();
    if text.is_empty() {
        return vec!["(no output)".into()];
    }
    let mut parts = Vec::new();
    let mut current = String::new();
    for line in text.lines() {
        // A single line longer than a message is cut on character bounds.
        let mut line = line;
        while line.chars().count() > limit {
            let cut = line
                .char_indices()
                .nth(limit)
                .map_or(line.len(), |(index, _)| index);
            if !current.is_empty() {
                parts.push(std::mem::take(&mut current));
            }
            parts.push(line[..cut].to_owned());
            line = &line[cut..];
        }
        if current.chars().count() + line.chars().count() + 1 > limit && !current.is_empty() {
            parts.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push('\n');
        }
        current.push_str(line);
    }
    if !current.is_empty() {
        parts.push(current);
    }
    parts
}

/// Validate a bot token by asking Telegram who it is.
pub async fn verify_token(token: &str) -> Result<String> {
    Ok(TelegramClient::new(token)?.get_me().await?.username)
}

#[cfg(test)]
mod tests;
