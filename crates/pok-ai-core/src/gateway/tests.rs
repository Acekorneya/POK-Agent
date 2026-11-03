use std::{
    collections::{HashMap, VecDeque},
    io::Cursor,
    sync::{
        Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use axum::{
    Json, Router,
    extract::State,
    routing::{get, post},
};
use serde_json::{Value, json};

use super::{
    telegram::{TelegramChannel, Update},
    *,
};

/// A fake Telegram API: records every call, replays scripted updates and
/// serves the files it was given.
#[derive(Default)]
struct FakeTelegram {
    calls: Mutex<Vec<(String, Value)>>,
    updates: Mutex<VecDeque<Value>>,
    /// file_id -> (download path, content)
    files: Mutex<HashMap<String, (String, Vec<u8>)>>,
}

impl FakeTelegram {
    fn sent(&self) -> Vec<(i64, String)> {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .filter(|(method, _)| method == "sendMessage")
            .map(|(_, body)| {
                (
                    body["chat_id"].as_i64().unwrap(),
                    body["text"].as_str().unwrap().to_owned(),
                )
            })
            .collect()
    }

    fn calls_to(&self, wanted: &str) -> usize {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .filter(|(method, _)| method == wanted)
            .count()
    }

    fn add_file(&self, file_id: &str, path: &str, bytes: Vec<u8>) {
        self.files
            .lock()
            .unwrap()
            .insert(file_id.to_owned(), (path.to_owned(), bytes));
    }
}

async fn fake_api(
    State(fake): State<Arc<FakeTelegram>>,
    axum::extract::Path((_token, method)): axum::extract::Path<(String, String)>,
    Json(body): Json<Value>,
) -> Json<Value> {
    fake.calls
        .lock()
        .unwrap()
        .push((method.clone(), body.clone()));
    let result = if method == "getUpdates" {
        // The first poll asks for the latest update only (offset -1).
        let next = fake.updates.lock().unwrap().pop_front();
        match next {
            Some(update) => json!([update]),
            None => {
                tokio::time::sleep(Duration::from_millis(40)).await;
                json!([])
            }
        }
    } else if method == "getFile" {
        let id = body["file_id"].as_str().unwrap_or_default();
        match fake.files.lock().unwrap().get(id) {
            Some((path, bytes)) => json!({
                "file_id": id, "file_path": path, "file_size": bytes.len()
            }),
            None => {
                return Json(json!({"ok": false, "description": "file not found"}));
            }
        }
    } else {
        json!(true)
    };
    Json(json!({"ok": true, "result": result}))
}

async fn fake_download(
    State(fake): State<Arc<FakeTelegram>>,
    axum::extract::Path((_token, path)): axum::extract::Path<(String, String)>,
) -> Vec<u8> {
    fake.files
        .lock()
        .unwrap()
        .values()
        .find(|(known, _)| *known == path)
        .map(|(_, bytes)| bytes.clone())
        .unwrap_or_default()
}

async fn fake_server() -> (Arc<FakeTelegram>, TelegramClient) {
    let fake = Arc::new(FakeTelegram::default());
    let app = Router::new()
        .route("/{token}/{method}", post(fake_api))
        .route("/file/{token}/{*path}", get(fake_download))
        .with_state(fake.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let root = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (
        fake,
        TelegramClient::with_api_root("1:test-token", &root).unwrap(),
    )
}

fn jpeg_bytes(width: u32, height: u32) -> Vec<u8> {
    let canvas = image::RgbImage::from_fn(width, height, |x, y| {
        image::Rgb([(x % 256) as u8, (y % 256) as u8, 40])
    });
    let mut out = Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(canvas)
        .write_to(&mut out, image::ImageFormat::Jpeg)
        .unwrap();
    out.into_inner()
}

/// A second channel, to show that nothing is tied to Telegram.
struct FakeChannel {
    name: &'static str,
    sent: Mutex<Vec<(String, String)>>,
}

#[async_trait]
impl Channel for FakeChannel {
    fn name(&self) -> &'static str {
        self.name
    }
    async fn send_text(&self, chat: &str, text: &str) -> Result<()> {
        self.sent
            .lock()
            .unwrap()
            .push((chat.to_owned(), text.to_owned()));
        Ok(())
    }
    async fn send_typing(&self, _chat: &str) {}
    async fn listen(self: Arc<Self>, _gateway: Arc<Gateway>, cancel: CancellationToken) {
        cancel.cancelled().await;
    }
}

#[derive(Default)]
struct MockHost {
    tasks: Mutex<Vec<GatewayTask>>,
    busy: AtomicBool,
    stops: AtomicUsize,
    approvals: Mutex<Vec<bool>>,
    changes: AtomicUsize,
}

#[async_trait]
impl GatewayHost for MockHost {
    async fn submit(&self, task: GatewayTask) -> std::result::Result<(), String> {
        if self.busy.load(Ordering::SeqCst) {
            return Err("Busy with another task.".into());
        }
        self.tasks.lock().unwrap().push(task);
        Ok(())
    }
    async fn stop(&self) -> String {
        self.stops.fetch_add(1, Ordering::SeqCst);
        "Stopped.".into()
    }
    async fn new_conversation(&self) -> String {
        "Fresh conversation.".into()
    }
    async fn status(&self) -> String {
        "Idle.".into()
    }
    async fn approve(&self, allow: bool) -> String {
        self.approvals.lock().unwrap().push(allow);
        "Done.".into()
    }
    fn changed(&self) {
        self.changes.fetch_add(1, Ordering::SeqCst);
    }
}

struct Fixture {
    dir: tempfile::TempDir,
    fake: Arc<FakeTelegram>,
    telegram: Arc<TelegramChannel>,
    host: Arc<MockHost>,
    gateway: Arc<Gateway>,
}

impl Fixture {
    async fn update(&self, update: Update) {
        self.telegram.handle_update(&self.gateway, update).await;
    }

    fn approve(&self, user: &str, chat: &str, name: &str) {
        let code = self
            .gateway
            .access()
            .request_pairing(user, chat, name)
            .unwrap();
        self.gateway.access().approve(&code).unwrap();
    }

    fn inbox(&self) -> std::path::PathBuf {
        self.dir.path().join("inbox")
    }
}

async fn fixture() -> Fixture {
    fixture_with(Duration::ZERO, Duration::ZERO).await
}

async fn fixture_with(text_window: Duration, media_window: Duration) -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let (fake, client) = fake_server().await;
    let telegram = Arc::new(TelegramChannel::new(client, 0));
    let host = Arc::new(MockHost::default());
    let gateway = Arc::new(
        Gateway::new(
            vec![telegram.clone()],
            Arc::new(AccessControl::open(&dir.path().join("access.json"))),
            Arc::new(JobStore::open(&dir.path().join("jobs.json"))),
            host.clone(),
            dir.path().join("inbox"),
        )
        .with_batch_windows(text_window, media_window),
    );
    Fixture {
        dir,
        fake,
        telegram,
        host,
        gateway,
    }
}

fn private_message(update_id: i64, user_id: i64, text: &str) -> Update {
    serde_json::from_value(json!({
        "update_id": update_id,
        "message": {
            "chat": {"id": user_id, "type": "private"},
            "from": {"id": user_id, "first_name": "Sam", "username": "sam"},
            "text": text
        }
    }))
    .unwrap()
}

#[test]
fn commands_parse_with_bot_suffixes_and_arguments() {
    assert_eq!(parse_command("/status@pok_bot"), Command::Status);
    assert_eq!(parse_command("/STOP"), Command::Stop);
    assert_eq!(
        parse_command("/schedule daily 09:00 brief me"),
        Command::Schedule("daily 09:00 brief me".into())
    );
    assert_eq!(
        parse_command("/unschedule  ab12 "),
        Command::Unschedule("ab12".into())
    );
    assert_eq!(parse_command("/dance"), Command::Unknown("dance".into()));
    assert_eq!(
        parse_command("  open the calendar "),
        Command::Task("open the calendar".into())
    );
    assert_eq!(parse_command("/start"), Command::Help);
}

#[tokio::test]
async fn a_stranger_gets_a_pairing_code_and_starts_nothing() {
    let fixture = fixture().await;
    fixture
        .update(private_message(1, 77, "delete all my files"))
        .await;
    assert!(fixture.host.tasks.lock().unwrap().is_empty());
    let sent = fixture.fake.sent();
    assert_eq!(sent.len(), 1);
    assert!(sent[0].1.contains("private"), "{}", sent[0].1);
    let pending = fixture.gateway.access().pending();
    assert_eq!(pending.len(), 1);
    assert!(sent[0].1.contains(&pending[0].code));
    assert!(fixture.host.changes.load(Ordering::SeqCst) >= 1);
    // Commands are no way around the allowlist either.
    fixture.update(private_message(2, 77, "/stop")).await;
    assert_eq!(fixture.host.stops.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn an_approved_person_can_run_tasks_and_use_commands() {
    let fixture = fixture().await;
    let code = fixture
        .gateway
        .access()
        .request_pairing("telegram:5", "telegram:5", "@sam")
        .unwrap();
    fixture.gateway.access().approve(&code).unwrap();

    fixture
        .update(private_message(1, 5, "open the calendar and read today"))
        .await;
    {
        let tasks = fixture.host.tasks.lock().unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].prompt, "open the calendar and read today");
        assert_eq!(tasks[0].chat, "telegram:5");
        assert_eq!(
            tasks[0].origin,
            TaskOrigin::Message {
                channel: "telegram".into(),
                user: "@sam".into()
            }
        );
    }
    assert!(fixture.fake.sent().last().unwrap().1.starts_with("On it"));

    fixture.update(private_message(2, 5, "/stop")).await;
    assert_eq!(fixture.host.stops.load(Ordering::SeqCst), 1);
    fixture.update(private_message(3, 5, "/approve")).await;
    fixture.update(private_message(4, 5, "/deny")).await;
    assert_eq!(*fixture.host.approvals.lock().unwrap(), vec![true, false]);

    fixture
        .update(private_message(
            5,
            5,
            "/schedule daily 09:00 summarize my day",
        ))
        .await;
    let jobs = fixture.gateway.jobs().list();
    assert_eq!(jobs.len(), 1);
    assert_eq!(jobs[0].prompt, "summarize my day");
    fixture.update(private_message(6, 5, "/jobs")).await;
    assert!(
        fixture
            .fake
            .sent()
            .last()
            .unwrap()
            .1
            .contains("daily at 09:00")
    );
    fixture
        .update(private_message(
            7,
            5,
            &format!("/unschedule {}", jobs[0].id),
        ))
        .await;
    assert!(fixture.gateway.jobs().list().is_empty());
    fixture
        .update(private_message(8, 5, "/schedule whenever"))
        .await;
    assert!(fixture.fake.sent().last().unwrap().1.starts_with("Use:"));
}

#[tokio::test]
async fn a_busy_agent_tells_the_sender_instead_of_queueing() {
    let fixture = fixture().await;
    let code = fixture
        .gateway
        .access()
        .request_pairing("telegram:5", "telegram:5", "Sam")
        .unwrap();
    fixture.gateway.access().approve(&code).unwrap();
    fixture.host.busy.store(true, Ordering::SeqCst);
    fixture.update(private_message(1, 5, "do it")).await;
    assert!(fixture.host.tasks.lock().unwrap().is_empty());
    assert_eq!(
        fixture.fake.sent().last().unwrap().1,
        "Busy with another task."
    );
}

#[tokio::test]
async fn groups_bots_and_messages_without_content_are_ignored() {
    let fixture = fixture().await;
    let code = fixture
        .gateway
        .access()
        .request_pairing("telegram:5", "telegram:5", "Sam")
        .unwrap();
    fixture.gateway.access().approve(&code).unwrap();
    let group: Update = serde_json::from_value(json!({
        "update_id": 1,
        "message": {"chat": {"id": -100, "type": "supergroup"},
            "from": {"id": 5, "first_name": "Sam"}, "text": "hello team"}
    }))
    .unwrap();
    let from_bot: Update = serde_json::from_value(json!({
        "update_id": 2,
        "message": {"chat": {"id": 5, "type": "private"},
            "from": {"id": 5, "first_name": "Bot", "is_bot": true}, "text": "hi"}
    }))
    .unwrap();
    // A location or a poll carries neither words nor an attachment.
    let location: Update = serde_json::from_value(json!({
        "update_id": 3,
        "message": {"chat": {"id": 5, "type": "private"}, "from": {"id": 5, "first_name": "Sam"},
            "location": {"latitude": 1.0, "longitude": 2.0}}
    }))
    .unwrap();
    for update in [group, from_bot, location] {
        fixture.update(update).await;
    }
    assert!(fixture.host.tasks.lock().unwrap().is_empty());
    assert!(fixture.fake.sent().is_empty());
}

#[tokio::test]
async fn the_poll_loop_drops_the_backlog_and_acts_on_new_messages() {
    let fixture = fixture().await;
    let code = fixture
        .gateway
        .access()
        .request_pairing("telegram:5", "telegram:5", "Sam")
        .unwrap();
    fixture.gateway.access().approve(&code).unwrap();
    {
        let mut updates = fixture.fake.updates.lock().unwrap();
        // Sent while the gateway was off: must not run.
        updates.push_back(json!({"update_id": 10, "message": {
            "chat": {"id": 5, "type": "private"},
            "from": {"id": 5, "first_name": "Sam"}, "text": "stale command"}}));
        updates.push_back(json!({"update_id": 11, "message": {
            "chat": {"id": 5, "type": "private"},
            "from": {"id": 5, "first_name": "Sam"}, "text": "fresh command"}}));
    }
    let cancel = CancellationToken::new();
    let running = tokio::spawn(fixture.gateway.clone().run(cancel.clone()));
    for _ in 0..100 {
        if !fixture.host.tasks.lock().unwrap().is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(30)).await;
    }
    cancel.cancel();
    tokio::time::timeout(Duration::from_secs(5), running)
        .await
        .expect("the loop stops promptly")
        .unwrap();
    let tasks = fixture.host.tasks.lock().unwrap();
    assert_eq!(tasks.len(), 1);
    assert_eq!(tasks[0].prompt, "fresh command");
    let polls: Vec<i64> = fixture
        .fake
        .calls
        .lock()
        .unwrap()
        .iter()
        .filter(|(method, _)| method == "getUpdates")
        .map(|(_, body)| body["offset"].as_i64().unwrap())
        .collect();
    assert_eq!(polls[0], -1, "the first poll only skips the backlog");
    assert!(polls.contains(&12), "later polls confirm what was handled");
}

#[tokio::test]
async fn due_jobs_run_on_the_home_chat_and_wait_while_the_agent_is_busy() {
    let fixture = fixture().await;
    let dir = tempfile::tempdir().unwrap();
    // A job created yesterday, so its daily run is due now.
    let yesterday = Utc::now() - chrono::Duration::days(2);
    let path = dir.path().join("jobs.json");
    std::fs::write(
        &path,
        json!([{
            "id": "abc123", "name": "Morning brief", "prompt": "brief me",
            "schedule": {"kind": "every", "minutes": 5}, "enabled": true,
            "created": yesterday
        }])
        .to_string(),
    )
    .unwrap();
    let gateway = Gateway::new(
        vec![fixture.telegram.clone()],
        fixture.gateway.access().clone(),
        Arc::new(JobStore::open(&path)),
        fixture.host.clone(),
        dir.path().join("inbox"),
    );
    // Nobody is paired yet, so there is nowhere to send results.
    gateway.run_due_jobs().await;
    assert!(fixture.host.tasks.lock().unwrap().is_empty());

    let code = gateway
        .access()
        .request_pairing("telegram:5", "telegram:55", "Sam")
        .unwrap();
    gateway.access().approve(&code).unwrap();
    fixture.host.busy.store(true, Ordering::SeqCst);
    gateway.run_due_jobs().await;
    assert!(fixture.host.tasks.lock().unwrap().is_empty());
    assert!(
        gateway.jobs().list()[0].last_run.is_none(),
        "a busy agent keeps the job due"
    );

    fixture.host.busy.store(false, Ordering::SeqCst);
    gateway.run_due_jobs().await;
    let tasks = fixture.host.tasks.lock().unwrap();
    assert_eq!(tasks.len(), 1);
    assert_eq!(tasks[0].chat, "telegram:55");
    assert_eq!(tasks[0].prompt, "brief me");
    assert_eq!(
        tasks[0].origin,
        TaskOrigin::Schedule {
            job: "Morning brief".into()
        }
    );
    assert!(gateway.jobs().list()[0].last_run.is_some());
}

fn photo_message(update_id: i64, user_id: i64, file_id: &str, caption: Option<&str>) -> Update {
    let mut message = json!({
        "chat": {"id": user_id, "type": "private"},
        "from": {"id": user_id, "first_name": "Sam"},
        // Telegram sends the same photo at several sizes; the largest wins.
        "photo": [
            {"file_id": format!("{file_id}-small"), "width": 90, "height": 60, "file_size": 900},
            {"file_id": file_id, "width": 1280, "height": 960, "file_size": 90_000},
        ]
    });
    if let Some(caption) = caption {
        message["caption"] = json!(caption);
    }
    serde_json::from_value(json!({"update_id": update_id, "message": message})).unwrap()
}

#[tokio::test]
async fn a_photo_with_a_caption_reaches_the_task_as_a_png_for_the_model() {
    let fixture = fixture().await;
    fixture.approve("telegram:5", "telegram:5", "Sam");
    fixture
        .fake
        .add_file("big", "photos/file_1.jpg", jpeg_bytes(1280, 960));
    fixture
        .update(photo_message(
            1,
            5,
            "big",
            Some("what plant is this? research care tips"),
        ))
        .await;

    let tasks = fixture.host.tasks.lock().unwrap();
    assert_eq!(tasks.len(), 1);
    let task = &tasks[0];
    assert_eq!(task.prompt, "what plant is this? research care tips");
    assert_eq!(task.images.len(), 1);
    // The phone's JPEG arrived as a PNG the session accepts.
    use base64::Engine as _;
    let png = base64::engine::general_purpose::STANDARD
        .decode(&task.images[0])
        .unwrap();
    assert!(png.starts_with(b"\x89PNG\r\n\x1a\n"));
    // It is also saved where the agent may read it, inside the inbox.
    assert_eq!(task.files.len(), 1);
    assert!(task.files[0].starts_with(fixture.inbox()));
    assert!(task.files[0].exists());
    assert!(
        fixture
            .fake
            .sent()
            .last()
            .unwrap()
            .1
            .contains("your picture")
    );
    drop(tasks);
    // Only the largest size was fetched.
    assert_eq!(fixture.fake.calls_to("getFile"), 1);
}

#[tokio::test]
async fn a_picture_without_words_becomes_a_default_question() {
    let fixture = fixture().await;
    fixture.approve("telegram:5", "telegram:5", "Sam");
    fixture
        .fake
        .add_file("only", "photos/file_2.jpg", jpeg_bytes(64, 48));
    fixture.update(photo_message(1, 5, "only", None)).await;
    let tasks = fixture.host.tasks.lock().unwrap();
    assert_eq!(tasks[0].prompt, "What is in this picture?");
}

#[tokio::test]
async fn a_strangers_attachment_is_never_downloaded() {
    let fixture = fixture().await;
    fixture
        .fake
        .add_file("big", "photos/file_3.jpg", jpeg_bytes(64, 48));
    fixture
        .update(photo_message(1, 77, "big", Some("hello")))
        .await;
    assert!(fixture.host.tasks.lock().unwrap().is_empty());
    assert_eq!(
        fixture.fake.calls_to("getFile"),
        0,
        "nothing is fetched before the owner approves the sender"
    );
    assert_eq!(fixture.gateway.access().pending().len(), 1);
    assert!(!fixture.inbox().exists());
}

#[tokio::test]
async fn a_document_is_saved_for_the_agent_and_a_voice_note_is_flagged() {
    let fixture = fixture().await;
    fixture.approve("telegram:5", "telegram:5", "Sam");
    fixture
        .fake
        .add_file("doc", "documents/file_4.csv", b"name,qty\npen,3\n".to_vec());
    fixture
        .fake
        .add_file("voice", "voice/file_5.oga", vec![1, 2, 3, 4]);
    let update: Update = serde_json::from_value(json!({
        "update_id": 1,
        "message": {
            "chat": {"id": 5, "type": "private"}, "from": {"id": 5, "first_name": "Sam"},
            "caption": "total the quantities",
            "document": {"file_id": "doc", "file_name": "stock.csv", "mime_type": "text/csv", "file_size": 17},
            "voice": {"file_id": "voice", "mime_type": "audio/ogg", "file_size": 4}
        }
    }))
    .unwrap();
    fixture.update(update).await;

    let tasks = fixture.host.tasks.lock().unwrap();
    let task = &tasks[0];
    assert!(task.images.is_empty());
    let names: Vec<_> = task
        .files
        .iter()
        .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert_eq!(names[0], "stock.csv");
    assert_eq!(std::fs::read(&task.files[0]).unwrap(), b"name,qty\npen,3\n");
    // The voice note cannot be transcribed yet: the agent and the sender are told.
    assert!(task.prompt.contains("total the quantities"));
    assert!(
        task.prompt.contains("cannot be transcribed"),
        "{}",
        task.prompt
    );
    assert!(
        fixture
            .fake
            .sent()
            .last()
            .unwrap()
            .1
            .contains("cannot be transcribed")
    );
}

#[tokio::test]
async fn an_oversized_file_is_declined_without_a_download() {
    let fixture = fixture().await;
    fixture.approve("telegram:5", "telegram:5", "Sam");
    let update: Update = serde_json::from_value(json!({
        "update_id": 1,
        "message": {
            "chat": {"id": 5, "type": "private"}, "from": {"id": 5, "first_name": "Sam"},
            "caption": "read this",
            "document": {"file_id": "huge", "file_name": "big.iso", "file_size": 90_000_000}
        }
    }))
    .unwrap();
    fixture.update(update).await;
    assert_eq!(fixture.fake.calls_to("getFile"), 0);
    let tasks = fixture.host.tasks.lock().unwrap();
    assert!(tasks[0].files.is_empty());
    assert!(tasks[0].prompt.contains("big.iso"));
    assert!(tasks[0].prompt.contains("not downloaded"));
}

#[tokio::test]
async fn a_failed_download_still_runs_the_request_and_says_so() {
    let fixture = fixture().await;
    fixture.approve("telegram:5", "telegram:5", "Sam");
    // No such file on the fake server: Telegram answers "file not found".
    fixture
        .update(photo_message(1, 5, "missing", Some("describe it")))
        .await;
    let tasks = fixture.host.tasks.lock().unwrap();
    assert_eq!(tasks.len(), 1);
    assert!(tasks[0].images.is_empty());
    assert!(tasks[0].prompt.contains("could not be downloaded"));
}

#[tokio::test]
async fn a_sticker_alone_starts_nothing() {
    let fixture = fixture().await;
    fixture.approve("telegram:5", "telegram:5", "Sam");
    let update: Update = serde_json::from_value(json!({
        "update_id": 1,
        "message": {
            "chat": {"id": 5, "type": "private"}, "from": {"id": 5, "first_name": "Sam"},
            "sticker": {"file_id": "st", "file_size": 100}
        }
    }))
    .unwrap();
    fixture.update(update).await;
    assert!(fixture.host.tasks.lock().unwrap().is_empty());
    assert_eq!(fixture.fake.calls_to("getFile"), 0);
}

#[tokio::test]
async fn the_photos_of_an_album_become_one_task() {
    let fixture = fixture_with(Duration::from_millis(40), Duration::from_millis(150)).await;
    fixture.approve("telegram:5", "telegram:5", "Sam");
    for (n, id) in ["a", "b", "c"].into_iter().enumerate() {
        fixture.fake.add_file(
            id,
            &format!("photos/album_{n}.jpg"),
            jpeg_bytes(32 + n as u32, 24),
        );
    }
    // Telegram delivers an album as separate messages; the caption is on one.
    fixture
        .update(photo_message(1, 5, "a", Some("compare these")))
        .await;
    fixture.update(photo_message(2, 5, "b", None)).await;
    fixture.update(photo_message(3, 5, "c", None)).await;
    assert!(
        fixture.host.tasks.lock().unwrap().is_empty(),
        "nothing starts before the window closes"
    );
    for _ in 0..100 {
        if !fixture.host.tasks.lock().unwrap().is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let tasks = fixture.host.tasks.lock().unwrap();
    assert_eq!(tasks.len(), 1, "one task, not one rejection per photo");
    assert_eq!(tasks[0].images.len(), 3);
    assert_eq!(tasks[0].prompt, "compare these");
}

#[tokio::test]
async fn a_text_split_by_the_app_is_joined_but_commands_are_not_delayed() {
    let fixture = fixture_with(Duration::from_millis(60), Duration::from_millis(60)).await;
    fixture.approve("telegram:5", "telegram:5", "Sam");
    fixture
        .update(private_message(1, 5, "first half of the request"))
        .await;
    fixture.update(private_message(2, 5, "second half")).await;
    fixture.update(private_message(3, 5, "/stop")).await;
    assert_eq!(
        fixture.host.stops.load(Ordering::SeqCst),
        1,
        "a command runs at once"
    );
    for _ in 0..100 {
        if !fixture.host.tasks.lock().unwrap().is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let tasks = fixture.host.tasks.lock().unwrap();
    assert_eq!(tasks.len(), 1);
    assert_eq!(tasks[0].prompt, "first half of the request\nsecond half");
}

fn other_channel(name: &'static str) -> Arc<FakeChannel> {
    Arc::new(FakeChannel {
        name,
        sent: Mutex::new(Vec::new()),
    })
}

fn inbound(channel: &'static str, user: &str, text: &str, trusted: bool) -> Inbound {
    Inbound {
        channel,
        chat: user.to_owned(),
        user: user.to_owned(),
        name: "Sam".into(),
        text: text.to_owned(),
        media: Vec::new(),
        trusted,
    }
}

#[tokio::test]
async fn a_second_channel_has_its_own_allowlist_and_answers_on_itself() {
    let dir = tempfile::tempdir().unwrap();
    let whatsapp = other_channel("whatsapp");
    let telegram = other_channel("telegram");
    let host = Arc::new(MockHost::default());
    let gateway = Arc::new(
        Gateway::new(
            vec![telegram.clone(), whatsapp.clone()],
            Arc::new(AccessControl::open(&dir.path().join("access.json"))),
            Arc::new(JobStore::open(&dir.path().join("jobs.json"))),
            host.clone(),
            dir.path().join("inbox"),
        )
        .with_batch_windows(Duration::ZERO, Duration::ZERO),
    );
    // Approved on Telegram only.
    let code = gateway
        .access()
        .request_pairing("telegram:15550100", "telegram:15550100", "Sam")
        .unwrap();
    gateway.access().approve(&code).unwrap();

    // The same number on WhatsApp is a stranger and gets a pairing code.
    gateway
        .handle_inbound(inbound("whatsapp", "15550100", "hi", false))
        .await;
    assert!(host.tasks.lock().unwrap().is_empty());
    let sent = whatsapp.sent.lock().unwrap().clone();
    assert_eq!(sent.len(), 1);
    assert_eq!(
        sent[0].0, "15550100",
        "the reply goes to the WhatsApp chat id"
    );
    assert!(sent[0].1.contains("pairing code"));
    assert!(telegram.sent.lock().unwrap().is_empty());
    assert_eq!(gateway.access().pending()[0].channel, "whatsapp");

    // After approval the task runs, and its answer address names WhatsApp.
    let code = gateway.access().pending()[0].code.clone();
    gateway.access().approve(&code).unwrap();
    gateway
        .handle_inbound(inbound("whatsapp", "15550100", "research this", false))
        .await;
    {
        let tasks = host.tasks.lock().unwrap();
        assert_eq!(tasks[0].chat, "whatsapp:15550100");
        assert_eq!(
            tasks[0].origin,
            TaskOrigin::Message {
                channel: "whatsapp".into(),
                user: "Sam".into()
            }
        );
    }
    gateway.send("whatsapp:15550100", "done").await;
    assert_eq!(whatsapp.sent.lock().unwrap().last().unwrap().1, "done");
    assert!(
        telegram.sent.lock().unwrap().is_empty(),
        "an answer never leaks to the other channel"
    );
}

#[tokio::test]
async fn the_owners_own_account_needs_no_pairing_and_becomes_home() {
    let dir = tempfile::tempdir().unwrap();
    let whatsapp = other_channel("whatsapp");
    let host = Arc::new(MockHost::default());
    let gateway = Arc::new(
        Gateway::new(
            vec![whatsapp.clone()],
            Arc::new(AccessControl::open(&dir.path().join("access.json"))),
            Arc::new(JobStore::open(&dir.path().join("jobs.json"))),
            host.clone(),
            dir.path().join("inbox"),
        )
        .with_batch_windows(Duration::ZERO, Duration::ZERO),
    );
    assert!(gateway.access().home_chat().is_none());
    gateway
        .handle_inbound(inbound("whatsapp", "15550199", "check my calendar", true))
        .await;
    assert_eq!(host.tasks.lock().unwrap()[0].prompt, "check my calendar");
    assert!(gateway.access().pending().is_empty());
    assert_eq!(
        gateway.access().home_chat().as_deref(),
        Some("whatsapp:15550199"),
        "scheduled results have somewhere to go"
    );
    // An existing home is not replaced.
    gateway.access().set_home_chat("telegram:1").unwrap();
    gateway
        .handle_inbound(inbound("whatsapp", "15550199", "again", true))
        .await;
    assert_eq!(gateway.access().home_chat().as_deref(), Some("telegram:1"));
}

#[tokio::test]
async fn sethome_and_whoami_use_channel_addresses() {
    let fixture = fixture().await;
    fixture.approve("telegram:5", "telegram:5", "@sam");
    fixture.update(private_message(1, 5, "/sethome")).await;
    assert_eq!(
        fixture.gateway.access().home_chat().as_deref(),
        Some("telegram:5")
    );
    fixture.update(private_message(2, 5, "/whoami")).await;
    assert!(fixture.fake.sent().last().unwrap().1.contains("telegram:5"));
}

/// The sidecar side of a WhatsApp link, scripted. It prints the events the
/// test pushes into `events`, and answers every command the agent sends the
/// way the real program does, recording them in `commands`.
fn scripted_sidecar(
    from_agent: tokio::io::DuplexStream,
    to_agent: tokio::io::DuplexStream,
    commands: Arc<Mutex<Vec<Value>>>,
    download: Vec<u8>,
) -> tokio::sync::mpsc::UnboundedSender<Value> {
    use tokio::io::{AsyncBufReadExt as _, AsyncWriteExt as _, BufReader};
    let (events, mut queue) = tokio::sync::mpsc::unbounded_channel::<Value>();
    tokio::spawn(async move {
        let mut lines = BufReader::new(from_agent).lines();
        let mut writer = to_agent;
        let mut sent = 0;
        loop {
            let output = tokio::select! {
                event = queue.recv() => match event {
                    Some(event) => Some(event),
                    None => return,
                },
                line = lines.next_line() => {
                    let Ok(Some(line)) = line else { return };
                    let command: Value = serde_json::from_str(&line).unwrap();
                    commands.lock().unwrap().push(command.clone());
                    let request = command["request"].clone();
                    match command["cmd"].as_str().unwrap() {
                        "download" => {
                            let target = std::path::Path::new(command["dir"].as_str().unwrap())
                                .join("incoming.bin");
                            std::fs::write(&target, &download).unwrap();
                            Some(json!({"event": "result", "request": request, "ok": true, "path": target}))
                        }
                        "send_text" => {
                            sent += 1;
                            Some(json!({"event": "result", "request": request, "ok": true,
                                "message_id": format!("SENT{sent}")}))
                        }
                        _ => None,
                    }
                }
            };
            if let Some(output) = output {
                writer
                    .write_all(format!("{output}\n").as_bytes())
                    .await
                    .unwrap();
            }
        }
    });
    events
}

async fn eventually(what: &str, mut condition: impl FnMut() -> bool) {
    for _ in 0..200 {
        if condition() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(15)).await;
    }
    panic!("timed out waiting for {what}");
}

#[tokio::test]
async fn whatsapp_round_trip_over_the_sidecar_protocol() {
    use whatsapp::{LinkState, SELF_CHAT_PREFIX, WhatsAppChannel, WhatsAppConfig, WhatsAppMode};

    let dir = tempfile::tempdir().unwrap();
    let channel = WhatsAppChannel::new(WhatsAppConfig {
        sidecar: "unused".into(),
        data_dir: dir.path().join("whatsapp"),
        mode: WhatsAppMode::SelfChat,
    });
    let host = Arc::new(MockHost::default());
    let gateway = Arc::new(
        Gateway::new(
            vec![channel.clone()],
            Arc::new(AccessControl::open(&dir.path().join("access.json"))),
            Arc::new(JobStore::open(&dir.path().join("jobs.json"))),
            host.clone(),
            dir.path().join("inbox"),
        )
        .with_batch_windows(Duration::ZERO, Duration::ZERO),
    );
    let (agent_reads, sidecar_writes) = tokio::io::duplex(1 << 20);
    let (sidecar_reads, agent_writes) = tokio::io::duplex(1 << 20);
    let cancel = CancellationToken::new();
    let session = tokio::spawn({
        let (channel, gateway, cancel) = (channel.clone(), gateway.clone(), cancel.clone());
        async move {
            channel
                .run_session(agent_reads, agent_writes, gateway, &cancel)
                .await;
        }
    });
    let commands = Arc::new(Mutex::new(Vec::<Value>::new()));
    let events = scripted_sidecar(
        sidecar_reads,
        sidecar_writes,
        commands.clone(),
        jpeg_bytes(200, 100),
    );

    // 1. The link shows a QR code to scan, then reports the linked number.
    events
        .send(json!({"event": "qr", "code": "2@scan-me,abc,def"}))
        .unwrap();
    eventually("the QR code", || channel.status().qr_svg.is_some()).await;
    assert_eq!(channel.status().state, LinkState::WaitingForScan);
    events
        .send(json!({"event": "connected", "phone": "15550199"}))
        .unwrap();
    eventually("the link", || {
        channel.status().state == LinkState::Connected
    })
    .await;
    assert_eq!(channel.status().account.as_deref(), Some("+15550199"));
    assert!(
        channel.status().qr_svg.is_none(),
        "the QR code goes away once linked"
    );

    // 2. The owner sends themselves a photo with a question.
    let now = chrono::Utc::now().timestamp();
    events
        .send(json!({
            "event": "message", "id": "IN1", "chat": "15550199@s.whatsapp.net",
            "chat_kind": "direct", "sender": "15550199", "from_me": true, "self_chat": true,
            "text": "what is this? find where to buy it", "timestamp": now,
            "media": [{"index": 0, "kind": "image", "mime": "image/jpeg", "size": 4096}]
        }))
        .unwrap();
    eventually("the task", || !host.tasks.lock().unwrap().is_empty()).await;
    {
        let tasks = host.tasks.lock().unwrap();
        let task = &tasks[0];
        assert_eq!(task.prompt, "what is this? find where to buy it");
        assert_eq!(task.chat, "whatsapp:15550199@s.whatsapp.net");
        assert_eq!(
            task.images.len(),
            1,
            "the JPEG was fetched and turned into a PNG"
        );
        assert!(task.files[0].starts_with(dir.path().join("inbox")));
    }
    // The agent asked the sidecar for the file, inside its own folder.
    let download = commands
        .lock()
        .unwrap()
        .iter()
        .find(|command| command["cmd"] == "download")
        .cloned()
        .expect("a download command");
    assert_eq!(download["message"], "IN1");
    assert!(download["dir"].as_str().unwrap().contains("whatsapp"));
    assert!(
        !dir.path()
            .join("whatsapp")
            .join("downloads")
            .join("incoming.bin")
            .exists(),
        "the temporary download is removed after it is read"
    );

    // 3. The answer goes back through the sidecar, marked as the agent's.
    gateway
        .send("whatsapp:15550199@s.whatsapp.net", "It is a **desk lamp**.")
        .await;
    let reply = commands
        .lock()
        .unwrap()
        .iter()
        .filter(|command| command["cmd"] == "send_text")
        .next_back()
        .cloned()
        .unwrap();
    assert_eq!(reply["chat"], "15550199@s.whatsapp.net");
    assert_eq!(
        reply["text"],
        format!("{SELF_CHAT_PREFIX}It is a *desk lamp*.")
    );

    // 4. The echo of that very reply, and someone else's message, start nothing.
    let started = host.tasks.lock().unwrap().len();
    events
        .send(json!({
            "event": "message", "id": "SENT2", "chat": "15550199@s.whatsapp.net",
            "chat_kind": "direct", "sender": "15550199", "from_me": true, "self_chat": true,
            "text": "It is a desk lamp.", "timestamp": now
        }))
        .unwrap();
    events
        .send(json!({
            "event": "message", "id": "IN2", "chat": "15550142@s.whatsapp.net",
            "chat_kind": "direct", "sender": "15550142", "from_me": false, "self_chat": false,
            "text": "delete everything", "timestamp": now
        }))
        .unwrap();
    tokio::time::sleep(Duration::from_millis(150)).await;
    assert_eq!(host.tasks.lock().unwrap().len(), started);
    assert!(
        gateway.access().pending().is_empty(),
        "no pairing in self-chat mode"
    );

    // 5. Unlinking on the phone is shown, and a closed pipe ends the session.
    events
        .send(json!({"event": "disconnected", "reason": "logged out", "logged_out": true}))
        .unwrap();
    eventually("the logout", || {
        channel.status().state == LinkState::LoggedOut
    })
    .await;
    drop(events);
    cancel.cancel();
    tokio::time::timeout(Duration::from_secs(5), session)
        .await
        .unwrap()
        .unwrap();
    assert!(
        channel.send_text("x", "late").await.is_err(),
        "nothing can be sent once the link is gone"
    );
}

/// Runs the real `pok-ai-whatsapp` program through the real channel code:
/// spawn, pipes, events, QR code, shutdown. It needs the built program and
/// internet access, so it is opt-in:
///
/// ```text
/// POK_WA_SIDECAR=sidecars/whatsapp/target/debug/pok-ai-whatsapp \
///   cargo test -p pok-ai-core --lib -- --ignored real_sidecar
/// ```
#[tokio::test]
#[ignore = "needs the built WhatsApp component (POK_WA_SIDECAR) and internet access"]
async fn real_sidecar_reaches_whatsapp_and_offers_a_qr_code() {
    use whatsapp::{LinkState, WhatsAppChannel, WhatsAppConfig, WhatsAppMode};

    let Ok(program) = std::env::var("POK_WA_SIDECAR") else {
        eprintln!("POK_WA_SIDECAR is not set; nothing to run");
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let channel = WhatsAppChannel::new(WhatsAppConfig {
        sidecar: std::path::PathBuf::from(&program),
        data_dir: dir.path().join("whatsapp"),
        mode: WhatsAppMode::SelfChat,
    });
    let host = Arc::new(MockHost::default());
    let gateway = Arc::new(Gateway::new(
        vec![channel.clone()],
        Arc::new(AccessControl::open(&dir.path().join("access.json"))),
        Arc::new(JobStore::open(&dir.path().join("jobs.json"))),
        host.clone(),
        dir.path().join("inbox"),
    ));
    let cancel = CancellationToken::new();
    let running = tokio::spawn(gateway.clone().run(cancel.clone()));

    // Not linked yet, so WhatsApp offers a QR code to scan.
    for _ in 0..600 {
        if channel.status().qr_svg.is_some() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let status = channel.status();
    assert_eq!(status.state, LinkState::WaitingForScan, "{status:?}");
    assert!(
        status
            .qr_svg
            .as_deref()
            .is_some_and(|svg| svg.contains("<svg"))
    );
    assert!(
        host.changes.load(Ordering::SeqCst) >= 1,
        "the page is told to refresh"
    );
    assert!(
        dir.path().join("whatsapp").join("session").exists(),
        "the session folder is created for the keys"
    );

    cancel.cancel();
    tokio::time::timeout(Duration::from_secs(10), running)
        .await
        .expect("the gateway stops promptly and takes the program down with it")
        .unwrap();
}

/// A missing program is reported, not retried forever.
#[tokio::test]
async fn a_missing_whatsapp_component_is_reported_clearly() {
    use whatsapp::{LinkState, WhatsAppChannel, WhatsAppConfig, WhatsAppMode};

    let dir = tempfile::tempdir().unwrap();
    let channel = WhatsAppChannel::new(WhatsAppConfig {
        sidecar: dir.path().join("not-installed").join("pok-ai-whatsapp"),
        data_dir: dir.path().join("whatsapp"),
        mode: WhatsAppMode::SelfChat,
    });
    let gateway = Arc::new(Gateway::new(
        vec![channel.clone()],
        Arc::new(AccessControl::open(&dir.path().join("access.json"))),
        Arc::new(JobStore::open(&dir.path().join("jobs.json"))),
        Arc::new(MockHost::default()),
        dir.path().join("inbox"),
    ));
    let cancel = CancellationToken::new();
    let running = tokio::spawn(gateway.clone().run(cancel.clone()));
    eventually("the failure", || {
        channel.status().state == LinkState::Failed
    })
    .await;
    let error = channel.status().error.unwrap();
    assert!(error.contains("file is missing"), "{error}");
    cancel.cancel();
    tokio::time::timeout(Duration::from_secs(5), running)
        .await
        .expect("a missing program does not keep the gateway from stopping")
        .unwrap();
}
