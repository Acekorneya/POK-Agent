use std::{
    collections::VecDeque,
    sync::{
        Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use axum::{Json, Router, extract::State, routing::post};
use serde_json::{Value, json};

use super::*;

/// A fake Telegram API: records every call and replays scripted updates.
#[derive(Default)]
struct FakeTelegram {
    calls: Mutex<Vec<(String, Value)>>,
    updates: Mutex<VecDeque<Value>>,
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
    } else {
        json!(true)
    };
    Json(json!({"ok": true, "result": result}))
}

async fn fake_server() -> (Arc<FakeTelegram>, TelegramClient) {
    let fake = Arc::new(FakeTelegram::default());
    let app = Router::new()
        .route("/{token}/{method}", post(fake_api))
        .with_state(fake.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let root = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (
        fake,
        TelegramClient::with_api_root("1:test-token", &root).unwrap(),
    )
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
    _dir: tempfile::TempDir,
    fake: Arc<FakeTelegram>,
    host: Arc<MockHost>,
    gateway: Arc<Gateway>,
}

async fn fixture() -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let (fake, client) = fake_server().await;
    let host = Arc::new(MockHost::default());
    let gateway = Arc::new(Gateway::new(
        client,
        Arc::new(AccessControl::open(&dir.path().join("access.json"))),
        Arc::new(JobStore::open(&dir.path().join("jobs.json"))),
        host.clone(),
        0,
    ));
    Fixture {
        _dir: dir,
        fake,
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
        .gateway
        .handle_update(private_message(1, 77, "delete all my files"))
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
    fixture
        .gateway
        .handle_update(private_message(2, 77, "/stop"))
        .await;
    assert_eq!(fixture.host.stops.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn an_approved_person_can_run_tasks_and_use_commands() {
    let fixture = fixture().await;
    let code = fixture
        .gateway
        .access()
        .request_pairing(5, 5, "@sam")
        .unwrap();
    fixture.gateway.access().approve(&code).unwrap();

    fixture
        .gateway
        .handle_update(private_message(1, 5, "open the calendar and read today"))
        .await;
    {
        let tasks = fixture.host.tasks.lock().unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].prompt, "open the calendar and read today");
        assert_eq!(tasks[0].chat_id, 5);
        assert_eq!(
            tasks[0].origin,
            TaskOrigin::Message {
                user: "@sam".into()
            }
        );
    }
    assert!(fixture.fake.sent().last().unwrap().1.starts_with("On it"));

    fixture
        .gateway
        .handle_update(private_message(2, 5, "/stop"))
        .await;
    assert_eq!(fixture.host.stops.load(Ordering::SeqCst), 1);
    fixture
        .gateway
        .handle_update(private_message(3, 5, "/approve"))
        .await;
    fixture
        .gateway
        .handle_update(private_message(4, 5, "/deny"))
        .await;
    assert_eq!(*fixture.host.approvals.lock().unwrap(), vec![true, false]);

    fixture
        .gateway
        .handle_update(private_message(
            5,
            5,
            "/schedule daily 09:00 summarize my day",
        ))
        .await;
    let jobs = fixture.gateway.jobs().list();
    assert_eq!(jobs.len(), 1);
    assert_eq!(jobs[0].prompt, "summarize my day");
    fixture
        .gateway
        .handle_update(private_message(6, 5, "/jobs"))
        .await;
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
        .gateway
        .handle_update(private_message(
            7,
            5,
            &format!("/unschedule {}", jobs[0].id),
        ))
        .await;
    assert!(fixture.gateway.jobs().list().is_empty());
    fixture
        .gateway
        .handle_update(private_message(8, 5, "/schedule whenever"))
        .await;
    assert!(fixture.fake.sent().last().unwrap().1.starts_with("Use:"));
}

#[tokio::test]
async fn a_busy_agent_tells_the_sender_instead_of_queueing() {
    let fixture = fixture().await;
    let code = fixture
        .gateway
        .access()
        .request_pairing(5, 5, "Sam")
        .unwrap();
    fixture.gateway.access().approve(&code).unwrap();
    fixture.host.busy.store(true, Ordering::SeqCst);
    fixture
        .gateway
        .handle_update(private_message(1, 5, "do it"))
        .await;
    assert!(fixture.host.tasks.lock().unwrap().is_empty());
    assert_eq!(
        fixture.fake.sent().last().unwrap().1,
        "Busy with another task."
    );
}

#[tokio::test]
async fn groups_bots_and_non_text_messages_are_ignored() {
    let fixture = fixture().await;
    let code = fixture
        .gateway
        .access()
        .request_pairing(5, 5, "Sam")
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
    let photo: Update = serde_json::from_value(json!({
        "update_id": 3,
        "message": {"chat": {"id": 5, "type": "private"}, "from": {"id": 5, "first_name": "Sam"}}
    }))
    .unwrap();
    for update in [group, from_bot, photo] {
        fixture.gateway.handle_update(update).await;
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
        .request_pairing(5, 5, "Sam")
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
        TelegramClient::with_api_root("1:t", "http://127.0.0.1:1").unwrap(),
        fixture.gateway.access().clone(),
        Arc::new(JobStore::open(&path)),
        fixture.host.clone(),
        0,
    );
    // Nobody is paired yet, so there is nowhere to send results.
    gateway.run_due_jobs().await;
    assert!(fixture.host.tasks.lock().unwrap().is_empty());

    let code = gateway.access().request_pairing(5, 55, "Sam").unwrap();
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
    assert_eq!(tasks[0].chat_id, 55);
    assert_eq!(tasks[0].prompt, "brief me");
    assert_eq!(
        tasks[0].origin,
        TaskOrigin::Schedule {
            job: "Morning brief".into()
        }
    );
    assert!(gateway.jobs().list()[0].last_run.is_some());
}
