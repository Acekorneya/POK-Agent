//! pok-ai-whatsapp: a WhatsApp linked-device client for POK-Agent.
//!
//! It signs in like WhatsApp Web (scan a QR code under Linked devices), then
//! relays messages to the agent and sends the agent's replies. See
//! `protocol.rs` for the JSON-lines protocol on stdin/stdout.
//!
//! Unofficial: WhatsApp's terms do not allow custom clients and an account
//! can be suspended. A dedicated number is the safer choice.

mod protocol;

use std::{
    collections::{HashMap, VecDeque},
    path::PathBuf,
    sync::{Arc, Mutex, OnceLock},
};

use anyhow::{Context, Result};
use protocol::{Command, emit, log, reply_error, reply_ok, safe_file_stem};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncBufReadExt, BufReader},
    sync::Notify,
};
use whatsapp_rust::{
    Jid,
    prelude::*,
    wacore::{download::Downloadable, proto_helpers::MessageExt, types::message::MessageSource},
    wacore_binary::JidExt,
    waproto::whatsapp as wa,
};

/// How many received messages keep their attachments available for download.
const HELD_MESSAGES: usize = 200;
/// The name shown under Linked devices on the phone.
const DEVICE_NAME: &str = "POK-Agent";

/// One downloadable attachment of a received message.
#[derive(Clone)]
enum Media {
    Image(wa::message::ImageMessage),
    Document(wa::message::DocumentMessage),
    Audio(wa::message::AudioMessage),
    Video(wa::message::VideoMessage),
    Sticker(wa::message::StickerMessage),
}

impl Media {
    fn downloadable(&self) -> &dyn Downloadable {
        match self {
            Self::Image(m) => m,
            Self::Document(m) => m,
            Self::Audio(m) => m,
            Self::Video(m) => m,
            Self::Sticker(m) => m,
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            Self::Image(_) => "image",
            Self::Document(_) => "document",
            Self::Audio(_) => "audio",
            Self::Video(_) => "video",
            Self::Sticker(_) => "sticker",
        }
    }

    fn mime(&self) -> Option<String> {
        match self {
            Self::Image(m) => m.mimetype.clone(),
            Self::Document(m) => m.mimetype.clone(),
            Self::Audio(m) => m.mimetype.clone(),
            Self::Video(m) => m.mimetype.clone(),
            Self::Sticker(m) => m.mimetype.clone(),
        }
    }

    fn name(&self) -> Option<String> {
        match self {
            Self::Document(m) => m.file_name.clone(),
            _ => None,
        }
    }

    fn size(&self) -> Option<u64> {
        match self {
            Self::Image(m) => m.file_length,
            Self::Document(m) => m.file_length,
            Self::Audio(m) => m.file_length,
            Self::Video(m) => m.file_length,
            Self::Sticker(m) => m.file_length,
        }
    }
}

/// Attachments of recent messages, kept so the agent can download them after
/// it decided to answer (and only for people it approved).
#[derive(Default)]
struct Held {
    order: VecDeque<String>,
    items: HashMap<String, Vec<Media>>,
}

impl Held {
    fn put(&mut self, message: String, media: Vec<Media>) {
        if self.items.insert(message.clone(), media).is_none() {
            self.order.push_back(message);
        }
        while self.order.len() > HELD_MESSAGES {
            if let Some(old) = self.order.pop_front() {
                self.items.remove(&old);
            }
        }
    }

    fn get(&self, message: &str, index: usize) -> Option<Media> {
        self.items.get(message)?.get(index).cloned()
    }
}

#[derive(Default)]
struct Shared {
    client: OnceLock<Arc<Client>>,
    held: Mutex<Held>,
    shutdown: Notify,
}

impl Shared {
    fn client(&self) -> Result<Arc<Client>> {
        self.client
            .get()
            .cloned()
            .context("WhatsApp is still starting")
    }
}

fn data_dir() -> Result<PathBuf> {
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        if argument == "--data-dir" {
            return args
                .next()
                .map(PathBuf::from)
                .context("--data-dir needs a folder");
        }
    }
    anyhow::bail!("usage: pok-ai-whatsapp --data-dir <folder>")
}

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        log("error", format!("{error:#}"));
        emit(
            &json!({"event": "disconnected", "reason": format!("{error:#}"), "logged_out": false}),
        );
        std::process::exit(1);
    }
}

async fn run() -> Result<()> {
    let data_dir = data_dir()?;
    let session_dir = data_dir.join("session");
    std::fs::create_dir_all(&session_dir).context("could not create the session folder")?;
    // The session holds the linked-device keys: keep it private to this user.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&session_dir, std::fs::Permissions::from_mode(0o700))
            .context("could not protect the session folder")?;
    }
    let store = SqliteStore::new(
        session_dir
            .join("whatsapp.db")
            .to_str()
            .context("the session path is not valid text")?,
    )
    .await
    .context("could not open the WhatsApp session store")?;

    let shared = Arc::new(Shared::default());
    let bot = Bot::builder()
        .with_backend(store)
        .with_device_props(
            whatsapp_rust::wacore::store::DevicePropsOverride::new().with_os(DEVICE_NAME),
        )
        // The agent acts on new messages only; old chats are never replayed.
        .skip_history_sync()
        .on_qr_code(|code, _timeout| async move {
            emit(&json!({"event": "qr", "code": code}));
        })
        .on_connected({
            let shared = shared.clone();
            move |client| {
                let shared = shared.clone();
                async move {
                    let phone = client.pn().map(|jid| jid.user.to_string());
                    let _ = shared.client.set(client);
                    emit(&json!({"event": "connected", "phone": phone}));
                }
            }
        })
        .on_logged_out({
            let shared = shared.clone();
            move |_info| {
                let shared = shared.clone();
                async move {
                    emit(&json!({
                        "event": "disconnected",
                        "reason": "This device was unlinked from WhatsApp.",
                        "logged_out": true
                    }));
                    shared.shutdown.notify_one();
                }
            }
        })
        .on_message({
            let shared = shared.clone();
            move |ctx| {
                let shared = shared.clone();
                async move { handle_message(&shared, &ctx).await }
            }
        })
        .build()
        .await
        .context("could not start the WhatsApp client")?;

    let _ = shared.client.set(bot.client());
    let handle = bot.spawn();
    tokio::spawn(command_loop(shared.clone()));
    shared.shutdown.notified().await;
    if let Ok(client) = shared.client() {
        client.disconnect().await;
    }
    handle.abort();
    Ok(())
}

/// Turn one received message into an event for the agent.
async fn handle_message(shared: &Arc<Shared>, ctx: &MessageContext) {
    let info = &ctx.info;
    let chat = &info.source.chat;
    let kind = if info.source.is_group || chat.is_group() {
        "group"
    } else if chat.is_newsletter() {
        "newsletter"
    } else if chat.is_status_broadcast() || chat.is_broadcast_list() {
        "broadcast"
    } else {
        "direct"
    };

    let base = ctx.message.get_base_message();
    let mut held = Vec::new();
    if let Some(m) = base.image_message.as_option() {
        held.push(Media::Image(m.clone()));
    }
    if let Some(m) = base.document_message.as_option() {
        held.push(Media::Document(m.clone()));
    }
    if let Some(m) = base.audio_message.as_option() {
        held.push(Media::Audio(m.clone()));
    }
    if let Some(m) = base.video_message.as_option() {
        held.push(Media::Video(m.clone()));
    }
    if let Some(m) = base.sticker_message.as_option() {
        held.push(Media::Sticker(m.clone()));
    }
    let text = ctx
        .message
        .text_content()
        .or_else(|| ctx.message.get_caption())
        .unwrap_or_default()
        .to_owned();
    if text.is_empty() && held.is_empty() {
        return;
    }

    let media: Vec<Value> = held
        .iter()
        .enumerate()
        .map(|(index, item)| {
            json!({
                "index": index, "kind": item.kind(), "mime": item.mime(),
                "name": item.name(), "size": item.size(),
            })
        })
        .collect();
    if !held.is_empty()
        && let Ok(mut store) = shared.held.lock()
    {
        store.put(info.id.clone(), held);
    }

    let (own_pn, own_lid) = shared
        .client()
        .map(|client| (client.pn(), client.lid()))
        .unwrap_or((None, None));
    let is_own = |jid: &Jid| {
        own_pn.as_ref().is_some_and(|own| own.user == jid.user)
            || own_lid.as_ref().is_some_and(|own| own.user == jid.user)
    };
    let self_chat = kind == "direct" && is_own(chat);

    emit(&json!({
        "event": "message",
        "id": info.id,
        "chat": chat.to_string(),
        "chat_kind": kind,
        "sender": sender_identity(&info.source),
        "name": info.push_name,
        "from_me": info.source.is_from_me,
        "self_chat": self_chat,
        "text": text,
        "timestamp": info.timestamp.timestamp(),
        "media": media,
    }));
}

/// A stable identity for the sender: the phone number when WhatsApp reveals
/// it, otherwise the account's LID, marked so it is never mistaken for a number.
fn sender_identity(source: &MessageSource) -> String {
    if source.sender.is_pn() {
        return source.sender.user.to_string();
    }
    if let Some(pn) = source.sender_alt.as_ref().filter(|jid| jid.is_pn()) {
        return pn.user.to_string();
    }
    format!("lid-{}", source.sender.user)
}

async fn command_loop(shared: Arc<Shared>) {
    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<Command>(&line) {
            Ok(Command::Shutdown) => break,
            Ok(command) => {
                let shared = shared.clone();
                tokio::spawn(async move { run_command(&shared, command).await });
            }
            Err(error) => log("error", format!("unknown command: {error}")),
        }
    }
    // The agent closed the pipe or asked to stop.
    shared.shutdown.notify_one();
}

async fn run_command(shared: &Arc<Shared>, command: Command) {
    match command {
        Command::SendText {
            request,
            chat,
            text,
        } => match send_text(shared, &chat, text).await {
            Ok(id) => reply_ok(&request, json!({"message_id": id})),
            Err(error) => reply_error(&request, format!("{error:#}")),
        },
        Command::Typing { chat } => {
            if let (Ok(client), Ok(jid)) = (shared.client(), chat.parse::<Jid>()) {
                let _ = client.chatstate().send_composing(&jid).await;
            }
        }
        Command::Download {
            request,
            message,
            index,
            max_bytes,
            dir,
        } => match download(shared, &message, index, max_bytes, &dir).await {
            Ok(path) => reply_ok(&request, json!({"path": path})),
            Err(error) => reply_error(&request, format!("{error:#}")),
        },
        Command::Logout { request } => {
            if let Ok(client) = shared.client() {
                client.logout().await;
            }
            if let Some(request) = request {
                reply_ok(&request, json!({}));
            }
            shared.shutdown.notify_one();
        }
        Command::Shutdown => shared.shutdown.notify_one(),
    }
}

async fn send_text(shared: &Arc<Shared>, chat: &str, text: String) -> Result<String> {
    let client = shared.client()?;
    let jid: Jid = chat
        .parse()
        .map_err(|_| anyhow::anyhow!("not a WhatsApp chat: {chat}"))?;
    let sent = client
        .send_message(&jid, wa::Message::text(text))
        .await
        .map_err(|error| anyhow::anyhow!("WhatsApp would not send that: {error}"))?;
    Ok(sent.message_id)
}

async fn download(
    shared: &Arc<Shared>,
    message: &str,
    index: usize,
    max_bytes: u64,
    dir: &std::path::Path,
) -> Result<PathBuf> {
    let media = shared
        .held
        .lock()
        .map_err(|_| anyhow::anyhow!("internal error"))?
        .get(message, index)
        .context("that attachment is no longer available")?;
    if media.size().is_some_and(|size| size > max_bytes) {
        anyhow::bail!("the file is larger than the allowed size");
    }
    let client = shared.client()?;
    let bytes = client
        .download(media.downloadable())
        .await
        .map_err(|error| anyhow::anyhow!("WhatsApp would not give the file: {error}"))?;
    if bytes.len() as u64 > max_bytes {
        anyhow::bail!("the file is larger than the allowed size");
    }
    tokio::fs::create_dir_all(dir).await?;
    let path = dir.join(format!("{}-{index}.bin", safe_file_stem(message)));
    tokio::fs::write(&path, bytes).await?;
    Ok(path)
}
