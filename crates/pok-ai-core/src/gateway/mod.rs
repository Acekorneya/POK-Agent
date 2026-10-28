//! Always-on messaging gateway: the owner messages the agent from Telegram,
//! and scheduled jobs run on their own. Messages become tasks for a
//! `GatewayHost` (the desktop app), which runs them like any other prompt and
//! hands back the answer.
//!
//! Safe by default: only people the owner approved on the computer can
//! message the agent (a stranger just gets a one-time pairing code), only
//! private chats count, and messages that arrived while the gateway was off
//! are dropped rather than acted on late.

pub mod access;
pub mod schedule;
pub mod telegram;

use std::{sync::Arc, time::Duration};

use async_trait::async_trait;
use chrono::{Local, Utc};
use tokio_util::sync::CancellationToken;

use crate::error::Result;
use access::AccessControl;
use schedule::{JobStore, parse_schedule};
use telegram::{Message, TelegramClient, Update};

/// Where a task came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskOrigin {
    Message { user: String },
    Schedule { job: String },
}

#[derive(Debug, Clone)]
pub struct GatewayTask {
    pub id: String,
    pub origin: TaskOrigin,
    pub prompt: String,
    /// The chat that gets the answer.
    pub chat_id: i64,
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

const HELP: &str = "POK-Agent is listening. Send a message and it becomes a task.\n\n\
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
    telegram: TelegramClient,
    access: Arc<AccessControl>,
    jobs: Arc<JobStore>,
    host: Arc<dyn GatewayHost>,
    poll_seconds: u64,
}

/// How often scheduled jobs are checked.
const SCHEDULE_TICK: Duration = Duration::from_secs(30);

impl Gateway {
    pub fn new(
        telegram: TelegramClient,
        access: Arc<AccessControl>,
        jobs: Arc<JobStore>,
        host: Arc<dyn GatewayHost>,
        poll_seconds: u64,
    ) -> Self {
        Self {
            telegram,
            access,
            jobs,
            host,
            poll_seconds: poll_seconds.clamp(0, 50),
        }
    }

    pub fn access(&self) -> &Arc<AccessControl> {
        &self.access
    }

    pub fn jobs(&self) -> &Arc<JobStore> {
        &self.jobs
    }

    /// Send text to a chat, ignoring delivery errors the caller cannot fix.
    pub async fn send(&self, chat_id: i64, text: &str) {
        if let Err(error) = self.telegram.send_message(chat_id, text).await {
            tracing::warn!("gateway could not deliver a message: {error}");
        }
    }

    /// Poll Telegram and run the scheduler until `cancel` fires.
    pub async fn run(self: Arc<Self>, cancel: CancellationToken) {
        let scheduler = tokio::spawn({
            let gateway = self.clone();
            let cancel = cancel.clone();
            async move { gateway.schedule_loop(cancel).await }
        });
        self.poll_loop(&cancel).await;
        scheduler.abort();
    }

    async fn poll_loop(&self, cancel: &CancellationToken) {
        // Skip whatever piled up while the gateway was off: acting on a
        // stale command hours later is never what the sender meant.
        let mut offset = loop {
            let attempt = tokio::select! {
                () = cancel.cancelled() => return,
                result = self.telegram.get_updates(-1, 0) => result,
            };
            match attempt {
                Ok(updates) => break updates.last().map_or(0, |update| update.update_id + 1),
                Err(error) => {
                    tracing::warn!("gateway cannot reach Telegram yet: {error}");
                    tokio::select! {
                        () = cancel.cancelled() => return,
                        () = tokio::time::sleep(Duration::from_secs(5)) => {}
                    }
                }
            }
        };
        let mut backoff = Duration::from_secs(2);
        loop {
            let polled = tokio::select! {
                () = cancel.cancelled() => return,
                result = self.telegram.get_updates(offset, self.poll_seconds) => result,
            };
            match polled {
                Ok(updates) => {
                    backoff = Duration::from_secs(2);
                    for update in updates {
                        offset = offset.max(update.update_id + 1);
                        self.handle_update(update).await;
                    }
                }
                Err(error) => {
                    tracing::warn!("gateway poll failed: {error}");
                    tokio::select! {
                        () = cancel.cancelled() => return,
                        () = tokio::time::sleep(backoff) => {}
                    }
                    backoff = (backoff * 2).min(Duration::from_secs(30));
                }
            }
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
        let Some(chat_id) = self.access.home_chat() else {
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
            chat_id,
        };
        if self.host.submit(task).await.is_ok() {
            if let Err(error) = self.jobs.mark_run(&job.id, Utc::now()) {
                tracing::warn!("could not record a job run: {error}");
            }
            self.send(chat_id, &format!("Running scheduled task: {}", job.name))
                .await;
            self.host.changed();
        }
    }

    /// One incoming update. Public so the behavior can be tested without a
    /// network poll.
    pub async fn handle_update(&self, update: Update) {
        let Some(message) = update.message else {
            return;
        };
        // Groups and channels are never a command surface.
        if message.chat.kind != "private" {
            return;
        }
        let (Some(user), Some(text)) = (message.from.as_ref(), message.text.as_deref()) else {
            return;
        };
        if user.is_bot {
            return;
        }
        let chat_id = message.chat.id;
        if !self.access.is_allowed(user.id) {
            self.greet_stranger(&message).await;
            return;
        }
        self.dispatch(chat_id, &user.display_name(), text).await;
    }

    async fn greet_stranger(&self, message: &Message) {
        let Some(user) = &message.from else { return };
        match self
            .access
            .request_pairing(user.id, message.chat.id, &user.display_name())
        {
            Some(code) => {
                self.send(
                    message.chat.id,
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

    async fn dispatch(&self, chat_id: i64, name: &str, text: &str) {
        let reply = match parse_command(text) {
            Command::Help => HELP.to_owned(),
            Command::WhoAmI => format!("You are {name}. This chat is {chat_id}."),
            Command::Status => self.host.status().await,
            Command::Stop => self.host.stop().await,
            Command::New => self.host.new_conversation().await,
            Command::Approve => self.host.approve(true).await,
            Command::Deny => self.host.approve(false).await,
            Command::SetHome => match self.access.set_home_chat(chat_id) {
                Ok(()) => "Scheduled results will be sent to this chat.".into(),
                Err(error) => format!("Could not save that: {error}"),
            },
            Command::Jobs => self.describe_jobs(),
            Command::Schedule(spec) => self.add_job(chat_id, &spec),
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
            Command::Task(prompt) => {
                let task = GatewayTask {
                    id: uuid::Uuid::new_v4().simple().to_string(),
                    origin: TaskOrigin::Message {
                        user: name.to_owned(),
                    },
                    prompt,
                    chat_id,
                };
                match self.host.submit(task).await {
                    Ok(()) => {
                        self.telegram.send_typing(chat_id).await;
                        "On it. I will message you when it is done.".into()
                    }
                    Err(reason) => reason,
                }
            }
        };
        self.send(chat_id, &reply).await;
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

    fn add_job(&self, chat_id: i64, spec: &str) -> String {
        let Some((schedule, prompt)) = parse_schedule(spec) else {
            return "Use: /schedule daily 09:00 <task> or /schedule every 30m <task> (at least every 5m).".into();
        };
        match self.jobs.add("", &prompt, schedule) {
            Ok(job) => {
                self.host.changed();
                let home = if self.access.home_chat() == Some(chat_id) {
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

/// Validate a bot token by asking Telegram who it is.
pub async fn verify_token(token: &str) -> Result<String> {
    Ok(TelegramClient::new(token)?.get_me().await?.username)
}

#[cfg(test)]
mod tests;
