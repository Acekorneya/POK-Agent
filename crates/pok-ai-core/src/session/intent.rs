//! Classifying the user's request: requested apps, URLs and commits, whether it needs live information, the desktop, or a visual or artifact outcome, and how attempts failed.

use super::*;

/// Known application labels mapped to their launch names. Only this bounded
/// list can produce an automatic launch candidate, so a free-form task never
/// turns into "run an arbitrary executable".
pub(super) const KNOWN_APPLICATIONS: &[(&str, &str)] = &[
    ("notepad", "notepad"),
    ("calculator", "calc"),
    ("paint", "mspaint"),
    ("file explorer", "explorer"),
    ("command prompt", "cmd"),
    ("powershell", "powershell"),
    ("task manager", "taskmgr"),
    ("visual studio code", "code"),
    ("vscode", "code"),
    ("word", "winword"),
    ("excel", "excel"),
    ("powerpoint", "powerpnt"),
    ("chrome", "chrome"),
    ("edge", "msedge"),
    ("firefox", "firefox"),
    ("discord", "discord"),
    ("spotify", "spotify"),
];

/// The launch name of an application the task asks for that is not currently
/// visible. Matching is token-boundary based so "password" never matches
/// "word"; the visible-window and last-listing checks avoid offering a launch
/// for an application that is already open.
pub(super) fn requested_application(
    task: &str,
    current_step: &str,
    observation: Option<&crate::types::Observation>,
    last_result: Option<&(String, Value)>,
    launched: &HashSet<String>,
) -> Option<String> {
    let text = format!("{task} {current_step}").to_ascii_lowercase();
    if !["open", "launch", "start", "bring up"]
        .iter()
        .any(|verb| text.contains(verb))
    {
        return None;
    }
    fn tokenize(value: &str) -> Vec<String> {
        value
            .to_ascii_lowercase()
            .split(|character: char| !character.is_alphanumeric())
            .filter(|token| !token.is_empty())
            .map(str::to_owned)
            .collect()
    }
    fn phrase_in(haystack: &[String], phrase: &str) -> bool {
        let words = phrase.split(' ').collect::<Vec<_>>();
        haystack
            .windows(words.len())
            .any(|window| window.iter().map(String::as_str).eq(words.iter().copied()))
    }
    let task_tokens = tokenize(&text);
    let visible = observation
        .and_then(|observation| observation.foreground_window.as_ref())
        .map(|window| tokenize(&format!("{} {}", window.title, window.process_name)));
    let listed = last_result
        .filter(|(tool, _)| tool == "list_windows")
        .map(|(_, result)| tokenize(&serde_json::to_string(result).unwrap_or_default()));
    for (label, executable) in KNOWN_APPLICATIONS {
        if !phrase_in(&task_tokens, label) {
            continue;
        }
        if launched.contains(*executable) {
            continue;
        }
        if visible
            .as_deref()
            .is_some_and(|tokens| phrase_in(tokens, label))
        {
            continue;
        }
        if listed
            .as_deref()
            .is_some_and(|tokens| phrase_in(tokens, label))
        {
            continue;
        }
        return Some((*executable).to_string());
    }
    None
}

/// An explicit URL or domain the task asks to visit. Requires an http(s)
/// scheme or a navigation verb, and rejects file-like tokens so
/// "open pok-ai.example.toml" never becomes a website navigation.
pub(super) fn requested_url(task: &str, current_step: &str) -> Option<String> {
    let text = format!("{task} {current_step}");
    let lower = text.to_ascii_lowercase();
    let navigation_intent = ["go to", "open", "visit", "navigate", "browse to"]
        .iter()
        .any(|verb| lower.contains(verb));
    const FILE_EXTENSIONS: &[&str] = &[
        "toml", "txt", "md", "json", "rs", "py", "exe", "dll", "png", "jpg", "jpeg", "pdf", "yaml",
        "yml", "lock", "log", "csv", "xml", "html", "htm", "js", "ts", "tsx", "css",
    ];
    for raw in text.split_whitespace() {
        let token = raw
            .trim_matches(|character: char| {
                matches!(
                    character,
                    '"' | '\'' | '(' | ')' | ',' | ';' | '!' | '?' | '`' | '<' | '>'
                )
            })
            .trim_end_matches('.');
        let candidate = if token.starts_with("http://") || token.starts_with("https://") {
            token.to_string()
        } else {
            if !navigation_intent {
                continue;
            }
            let host = token.split('/').next().unwrap_or_default();
            let labels = host.split('.').collect::<Vec<_>>();
            if labels.len() < 2
                || labels.iter().any(|label| {
                    label.is_empty()
                        || !label
                            .chars()
                            .all(|character| character.is_ascii_alphanumeric() || character == '-')
                })
            {
                continue;
            }
            let tld = labels.last().copied().unwrap_or_default();
            if tld.len() < 2
                || tld.len() > 24
                || !tld.chars().all(|character| character.is_ascii_alphabetic())
                || FILE_EXTENSIONS.contains(&tld)
            {
                continue;
            }
            format!("https://{token}")
        };
        return Some(candidate);
    }
    None
}

pub(super) fn requested_commit_label(root_request: &str, label: &str) -> Option<String> {
    let root_terms = root_request
        .to_ascii_lowercase()
        .split(|character: char| !character.is_ascii_alphanumeric())
        .filter(|term| !term.is_empty())
        .filter_map(canonical_commit_term)
        .collect::<HashSet<_>>();
    // Button labels are capitalized ("Save", "Send"); compare like the request.
    label
        .to_ascii_lowercase()
        .split(|character: char| !character.is_ascii_alphanumeric())
        .filter_map(canonical_commit_term)
        .find(|term| root_terms.contains(*term))
        .map(str::to_owned)
}

pub(super) fn canonical_commit_term(term: &str) -> Option<&'static str> {
    match term {
        "delete" | "deleted" | "deleting" => Some("delete"),
        "download" | "downloaded" | "downloading" => Some("download"),
        "export" | "exported" | "exporting" => Some("export"),
        "finish" | "finished" | "finishing" => Some("finish"),
        "pay" | "paid" | "paying" => Some("pay"),
        "print" | "printed" | "printing" => Some("print"),
        "publish" | "published" | "publishing" => Some("publish"),
        "record" | "recorded" | "recording" => Some("record"),
        "save" | "saved" | "saving" => Some("save"),
        "send" | "sent" | "sending" => Some("send"),
        "submit" | "submitted" | "submitting" => Some("submit"),
        "upload" | "uploaded" | "uploading" => Some("upload"),
        _ => None,
    }
}

pub(super) fn requests_artifact_outcome(request: &str) -> bool {
    let request = request_for_classification(request);
    let action = [
        "create", "make", "write", "edit", "update", "generate", "save", "export", "build",
    ]
    .iter()
    .any(|marker| request.contains(marker));
    let object = [
        "file",
        "document",
        "letter",
        "spreadsheet",
        "workbook",
        "presentation",
        "slides",
        "chart",
        "report",
        "image",
        "code",
        "script",
        "note",
        "pdf",
    ]
    .iter()
    .any(|marker| request.contains(marker));
    let explicit_destination = [
        "desktop",
        "downloads",
        "save it to",
        "save this to",
        "folder",
        "file path",
    ]
    .iter()
    .any(|marker| request.contains(marker));
    action && (object || explicit_destination)
}

pub(super) fn requests_visual_outcome(request: &str) -> bool {
    let request = request_for_classification(request);
    [
        "open it",
        "show me",
        "so i can see",
        "visually",
        "look like",
        "appearance",
        "format",
        "layout",
        "chart",
        "presentation",
        "slides",
        "design",
    ]
    .iter()
    .any(|marker| request.contains(marker))
}

pub(super) fn guidance_requests_repeat(guidance: &str) -> bool {
    let guidance = guidance.to_ascii_lowercase();
    if ["do not", "don't", "dont", "no more", "not again"]
        .iter()
        .any(|marker| guidance.contains(marker))
    {
        return false;
    }
    [
        "try again",
        "do it again",
        "repeat the action",
        "redo",
        "reprint",
        "another copy",
        "another one",
    ]
    .iter()
    .any(|marker| guidance.contains(marker))
        || canonical_commit_terms(&guidance)
            .iter()
            .any(|term| guidance.contains(&format!("{term} again")))
}

pub(super) fn guidance_confirms_completion(guidance: &str) -> bool {
    let guidance = guidance.to_ascii_lowercase();
    [
        "it worked",
        "task was successful",
        "task is complete",
        "task was complete",
        "you accomplished the task",
        "you completed the task",
        "already completed",
        "already finished",
    ]
    .iter()
    .any(|marker| guidance.contains(marker))
}

pub(super) fn canonical_commit_terms(request: &str) -> HashSet<&'static str> {
    request
        .split(|character: char| !character.is_ascii_alphanumeric())
        .filter_map(canonical_commit_term)
        .collect()
}

pub(super) fn classify_attempt_failure(name: &str, error: &PokError) -> &'static str {
    let message = error.to_string().to_ascii_lowercase();
    if message.contains("\"failure_attribution\":\"caller_input\"")
        || message.contains("invalid or ambiguous window_id")
        || message.contains("unknown target")
        || message.contains("stale observation")
    {
        "caller_input"
    } else if message.contains("\"failure_attribution\":\"environment_failure\"") {
        "environment_failure"
    } else if message.contains("desktop control changed before input") {
        "grounding_gap"
    } else if matches!(
        error,
        PokError::OutsideWorkspace(_) | PokError::PolicyDenied { .. }
    ) {
        "policy_constraint"
    } else if matches!(
        name,
        "capture_screen"
            | "click_target"
            | "locate_visual_target"
            | "click_localized"
            | "move_pointer"
            | "drag_pointer"
            | "drag_target"
            | "query_screen_text"
            | "query_window_tree"
            | "scroll_until_text"
            | "scroll_view"
    ) {
        "grounding_gap"
    } else if message.contains("not found")
        || message.contains("no such file")
        || message.contains("missing")
        || message.contains("unavailable")
    {
        "environment_failure"
    } else if name == "inspect_artifact" {
        "verification_gap"
    } else if matches!(
        name,
        "activate_window"
            | "click_localized"
            | "move_pointer"
            | "drag_pointer"
            | "drag_target"
            | "simulate_input"
    ) {
        "application_failure"
    } else {
        "tool_failure"
    }
}

pub(super) fn classify_unsuccessful_result(name: &str, value: &Value) -> &'static str {
    let detail = format!(
        "{} {}",
        value.get("stdout").and_then(Value::as_str).unwrap_or(""),
        value.get("stderr").and_then(Value::as_str).unwrap_or("")
    )
    .to_ascii_lowercase();
    if detail.contains("not found")
        || detail.contains("no such file")
        || detail.contains("is not recognized")
        || detail.contains("modulenotfound")
        || detail.contains("missing")
    {
        "environment_failure"
    } else if is_guarded_action(name) {
        "application_failure"
    } else {
        "tool_failure"
    }
}

pub(super) fn is_clarification_request(text: &str) -> bool {
    let trimmed = text.trim();
    if !trimmed.ends_with('?') || trimmed.chars().count() > 800 {
        return false;
    }
    let lower = trimmed.to_ascii_lowercase();
    let explicit_dependency = [
        "i need to know",
        "i need your",
        "please provide",
        "could you provide",
        "can you provide",
        "which one",
        "which file",
        "which window",
        "what location",
        "what city",
        "what should",
        "where should",
        "before i continue",
        "to continue",
        "could you tell me",
        "can you tell me",
        "please specify",
    ]
    .iter()
    .any(|phrase| lower.contains(phrase));
    let short_direct_question = trimmed.chars().count() <= 300
        && ["what ", "which ", "where ", "who ", "how "]
            .iter()
            .any(|prefix| lower.starts_with(prefix));
    explicit_dependency || short_direct_question
}

pub(super) fn substantive_candidate_answer(text: &str) -> bool {
    let trimmed = text.trim();
    trimmed.chars().count() >= 80
        && ![
            "i can't",
            "i cannot",
            "unable to",
            "don't have access",
            "do not have access",
        ]
        .iter()
        .any(|phrase| trimmed.to_ascii_lowercase().contains(phrase))
}

pub(super) fn is_unresolved_task_failure(text: &str) -> bool {
    let text = text.to_ascii_lowercase();
    [
        "couldn't find",
        "could not find",
        "wasn't able to",
        "was not able to",
        "unable to complete",
        "failed to complete",
        "not showing up",
        "couldn't complete",
        "could not complete",
    ]
    .iter()
    .any(|phrase| text.contains(phrase))
}

pub(super) fn requests_live_information(request: &str) -> bool {
    let request = request_for_classification(request);
    let explicit_live_marker = [
        "current time",
        "time right now",
        "right now",
        "currently",
        "today",
        "this week",
        "latest",
        "live ",
        "stock price",
        "exchange rate",
    ]
    .iter()
    .any(|phrase| request.contains(phrase));
    explicit_live_marker
        || (request.contains("weather")
            && ["current", "check", "find", "look up", "tonight", "tomorrow"]
                .iter()
                .any(|marker| request.contains(marker)))
        || (request.contains("news")
            && ["check", "find", "look up", "headlines"]
                .iter()
                .any(|marker| request.contains(marker)))
}

pub(super) fn requests_managed_web_search(request: &str) -> bool {
    let request = request_for_classification(request);
    [
        "browser",
        "website",
        "webpage",
        "online",
        "search",
        "wikipedia",
        "news",
        "headline",
        "headlines",
        "current events",
        "latest",
        "look up",
        "weather",
        "forecast",
        "temperature",
        "stock",
        "ticker",
        "share price",
        "market close",
        "exchange rate",
        "sports score",
    ]
    .iter()
    .any(|term| request.contains(term))
}

pub(super) fn requests_desktop_interaction(request: &str) -> bool {
    let request = request_for_classification(request);
    [
        "on the screen",
        "this screen",
        "current screen",
        "application",
        " app ",
        "window",
        "desktop",
        "click",
        "select",
        "type into",
        "channel",
        "chat",
        "streaming",
        "what do you see",
        "currently selected",
    ]
    .iter()
    .any(|term| request.contains(term))
}

pub(super) fn requests_desktop_action(request: &str) -> bool {
    let request = request_for_classification(request);
    [
        "open ", "launch ", "close ", "play ", "pause ", "scroll ", "drag ", "press ",
    ]
    .iter()
    .any(|term| request.starts_with(term) || request.contains(&format!(" {term}")))
}

pub(super) fn requests_current_screen_observation(request: &str) -> bool {
    let request = request_for_classification(request);
    [
        "on the screen",
        "on screen",
        "this screen",
        "current screen",
        "what do you see",
        "can you see",
        "look at the screen",
        "visible right now",
        "currently selected",
        "selected item",
        "top of the list",
    ]
    .iter()
    .any(|phrase| request.contains(phrase))
}

pub(super) fn request_for_classification(request: &str) -> String {
    let mut normalized = request
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase();
    loop {
        normalized = normalized
            .trim_start_matches(|character: char| {
                character.is_whitespace() || matches!(character, ',' | ':' | ';' | '-' | '—')
            })
            .to_owned();
        let mut removed = false;
        for prefix in ["and", "also", "okay", "ok", "now", "please", "then", "so"] {
            let Some(remainder) = normalized.strip_prefix(prefix) else {
                continue;
            };
            if remainder.is_empty()
                || remainder.starts_with(|character: char| {
                    character.is_whitespace() || matches!(character, ',' | ':' | ';' | '-' | '—')
                })
            {
                normalized = remainder.to_owned();
                removed = true;
                break;
            }
        }
        if !removed {
            return normalized;
        }
    }
}

pub(super) fn is_safety_refusal(text: &str) -> bool {
    let text = text.to_ascii_lowercase();
    [
        "password",
        "secure desktop",
        "user account control",
        "uac prompt",
    ]
    .iter()
    .any(|phrase| text.contains(phrase))
}
