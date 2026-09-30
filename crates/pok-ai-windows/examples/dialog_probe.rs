//! Read-only check of what the agent's UI Automation sees in the foreground
//! window, through the same platform code a capture uses. Performs no actions.
//!
//!   `dialog_probe [seconds to wait first] [window title part]`
//!
//! Prints the foreground window, how many UI Automation elements its capture
//! returns, the count per control type, and the names of its buttons (a
//! dialog's OK, Cancel, ...). Open the dialog to check, then run it.

#[cfg(windows)]
fn main() {
    use pok_ai_core::types::{CaptureRequest, CaptureScope};

    let wait = std::env::args()
        .nth(1)
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(0);
    std::thread::sleep(std::time::Duration::from_secs(wait));
    let runtime = tokio::runtime::Runtime::new().expect("runtime");
    runtime.block_on(async {
        let platform = pok_ai_windows::desktop_platform(false);
        let title = std::env::args().nth(2);
        let window = match &title {
            Some(title) => platform
                .list_windows()
                .await
                .unwrap_or_default()
                .into_iter()
                .find(|window| window.title.contains(title.as_str())),
            None => platform.foreground_window().await.ok().flatten(),
        };
        let Some(window) = window else {
            println!("no matching window");
            return;
        };
        if title.is_some() {
            // Bring the named window forward first (focus only, no input).
            let _ = platform.bring_to_front(&window.id).await;
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }
        let started = std::time::Instant::now();
        println!(
            "foreground: {:?} ({}) id {}",
            window.title, window.process_name, window.id
        );
        let request = CaptureRequest {
            scope: CaptureScope::Window,
            window_id: Some(window.id.clone()),
            monitor_id: None,
            region: None,
            max_edge: 1280,
        };
        let tree = platform.query_ui_tree_target(&request).await;
        println!(
            "UI Automation query took {} ms",
            started.elapsed().as_millis()
        );
        match tree {
            Err(error) => println!("UI Automation: ERROR {error}"),
            Ok(elements) => {
                let mut kinds = std::collections::BTreeMap::<String, usize>::new();
                for element in &elements {
                    *kinds
                        .entry(element.control_type.to_lowercase())
                        .or_default() += 1;
                }
                println!("UI Automation elements: {}", elements.len());
                println!("by control type: {kinds:?}");
                let buttons = elements
                    .iter()
                    .filter(|element| element.control_type.eq_ignore_ascii_case("button"))
                    .map(|element| element.name.as_str())
                    .filter(|name| !name.is_empty())
                    .collect::<Vec<_>>();
                println!("buttons: {buttons:?}");
            }
        }
    });
}

#[cfg(not(windows))]
fn main() {
    eprintln!("dialog_probe runs on Windows only");
}
