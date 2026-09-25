//! Managed-browser navigation and waiting until the destination page is ready.

use super::*;

#[derive(Debug, Deserialize, JsonSchema)]
pub(super) struct BrowserNavigateArgs {
    pub(super) window_id: String,
    #[serde(alias = "url", alias = "query", alias = "destination")]
    pub(super) query_or_url: String,
}

pub(super) struct BrowserNavigateTool;

pub(super) const NAVIGATION_SETTLE_TIMEOUT: Duration = Duration::from_secs(6);
pub(super) const NAVIGATION_SETTLE_DELAYS_MS: [u64; 5] = [500, 400, 700, 1_100, 1_600];

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum NavigationReadinessStatus {
    Ready,
    Loading,
    ErrorPage,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct NavigationReadiness {
    pub(super) status: NavigationReadinessStatus,
    pub(super) elapsed_ms: u64,
    pub(super) attempts: u8,
    pub(super) url_match: bool,
    pub(super) title_changed: bool,
    pub(super) content_changed: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct BrowserPageIdentity {
    pub(super) title: String,
    pub(super) url: Option<String>,
    pub(super) content: String,
}

#[async_trait]
impl Tool for BrowserNavigateTool {
    fn name(&self) -> &'static str {
        "browser_navigate"
    }
    fn description(&self) -> &'static str {
        "Activate a browser window and safely submit one user-facing webpage URL or search query using Ctrl+L, text, and Enter, then return a fresh capture. Prefer this over manually clicking the address bar. This performs browser navigation (GET), not an API request; do not navigate to inference endpoints such as /v1/chat/completions to call a model."
    }
    fn input_schema(&self) -> Value {
        schema::<BrowserNavigateArgs>()
    }
    fn risk(&self) -> RiskClass {
        RiskClass::DesktopInput
    }
    async fn execute(&self, args: Value, context: &ToolContext) -> Result<Value> {
        let mut args: BrowserNavigateArgs = serde_json::from_value(args)?;
        args.window_id = resolve_window_id(context, &args.window_id).await?;
        let destination = args.query_or_url.trim();
        if destination.is_empty() || destination.chars().count() > 4096 {
            return Err(PokError::Tool(
                "query_or_url must contain between 1 and 4096 characters".into(),
            ));
        }
        let window = context.platform.activate_window(&args.window_id).await?;
        let app = window.process_name.to_ascii_lowercase();
        if !matches!(
            app.as_str(),
            "chrome.exe" | "msedge.exe" | "firefox.exe" | "brave.exe"
        ) {
            return Err(PokError::Tool(format!(
                "window {:?} is not a supported browser",
                window.process_name
            )));
        }
        let focus = FocusedControl {
            app: window.process_name.clone(),
            label: "Address and search bar".into(),
            control_type: "Edit".into(),
        };
        validate_browser_navigation(destination, &context.task_hint.lock(), Some(&focus))?;
        for action in [
            InputAction::Key {
                key: "Ctrl+L".into(),
            },
            InputAction::TypeText {
                text: destination.into(),
                replace_existing: true,
            },
            InputAction::Key {
                key: "Enter".into(),
            },
        ] {
            simulate_input_for_window(context, &action, &args.window_id).await?;
            tokio::time::sleep(std::time::Duration::from_millis(120)).await;
        }
        *context.focused_control.lock() = Some(focus);
        let request = CaptureRequest {
            scope: CaptureScope::Window,
            window_id: Some(args.window_id.clone()),
            monitor_id: None,
            region: None,
            max_edge: context.vision_max_edge,
        };
        let before = context
            .latest_observation
            .lock()
            .as_ref()
            .filter(|observation| {
                observation
                    .target
                    .as_ref()
                    .is_some_and(|target| target.id == args.window_id)
            })
            .map(browser_page_identity)
            .unwrap_or_else(|| BrowserPageIdentity {
                title: normalized_text(&window.title),
                ..BrowserPageIdentity::default()
            });
        let (mut observation, readiness) = settle_browser_navigation(
            context,
            &request,
            &before,
            destination,
            NAVIGATION_SETTLE_TIMEOUT,
        )
        .await?;
        observation.targets = build_targets(
            &observation,
            context.model_target_limit,
            context.fusion_iou_threshold,
            context.ocr_containment_threshold,
            &context.task_hint.lock(),
        );
        *context.latest_observation.lock() = Some(observation.clone());
        *context.latest_observation_view.lock() = None;
        *context.pending_visual_localization.lock() = None;
        context.write_artifact(
            &format!("observation-{}.json", observation.version),
            &crate::tool::diagnostic_value(&serde_json::to_value(&observation)?),
        )?;
        save_observation_visuals(context, &observation)?;
        let mut value = model_observation_value(&observation, context.annotate_targets)?;
        value["navigation_submitted"] = json!(true);
        value["query_or_url"] = json!(destination);
        value["navigation_readiness"] = serde_json::to_value(&readiness)?;
        value["page_status"] = json!(match readiness.status {
            NavigationReadinessStatus::Ready => "loaded",
            NavigationReadinessStatus::Loading => "loading",
            NavigationReadinessStatus::ErrorPage => "error_page",
        });
        Ok(value)
    }
}

pub(super) async fn settle_browser_navigation(
    context: &ToolContext,
    request: &CaptureRequest,
    before: &BrowserPageIdentity,
    destination: &str,
    timeout: Duration,
) -> Result<(Observation, NavigationReadiness)> {
    let started = Instant::now();
    let direct_url = looks_like_browser_url(destination);
    let mut attempts = 0_u8;
    let mut latest = None;
    let mut previous_changed_content = None::<String>;

    for delay_ms in NAVIGATION_SETTLE_DELAYS_MS {
        tokio::select! {
            () = context.cancellation.cancelled() => return Err(PokError::Cancelled),
            () = tokio::time::sleep(Duration::from_millis(delay_ms)) => {}
        }
        attempts = attempts.saturating_add(1);
        let observation = context
            .platform
            .observe(
                request,
                true,
                true,
                context.uia_element_limit,
                Duration::from_millis(context.desktop_enrichment_timeout_ms),
            )
            .await?;
        let identity = browser_page_identity(&observation);
        let (ready, error_page, url_match, title_changed, content_changed) =
            navigation_sample_readiness(
                before,
                &identity,
                destination,
                direct_url,
                previous_changed_content.as_deref(),
                attempts,
            );
        let readiness = NavigationReadiness {
            status: if error_page {
                NavigationReadinessStatus::ErrorPage
            } else if ready {
                NavigationReadinessStatus::Ready
            } else {
                NavigationReadinessStatus::Loading
            },
            elapsed_ms: u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
            attempts,
            url_match,
            title_changed,
            content_changed,
        };
        if content_changed {
            previous_changed_content = Some(identity.content);
        }
        latest = Some((observation, readiness.clone()));
        if ready || error_page || started.elapsed() >= timeout {
            break;
        }
    }

    latest.ok_or_else(|| PokError::Tool("browser readiness produced no observation".into()))
}

pub(super) fn navigation_sample_readiness(
    before: &BrowserPageIdentity,
    identity: &BrowserPageIdentity,
    destination: &str,
    direct_url: bool,
    previous_changed_content: Option<&str>,
    attempts: u8,
) -> (bool, bool, bool, bool, bool) {
    let url_match = navigation_url_matches(destination, identity.url.as_deref(), direct_url);
    let title_changed = !identity.title.is_empty() && identity.title != before.title;
    let content_changed = !identity.content.is_empty() && identity.content != before.content;
    let content_confirmed = content_changed
        && previous_changed_content.is_some_and(|previous| previous == identity.content);
    let error_page = looks_like_browser_error(identity);
    let loading = looks_like_loading_page(identity);
    let same_destination = before
        .url
        .as_deref()
        .zip(identity.url.as_deref())
        .is_some_and(|(before, after)| {
            normalized_browser_location(before) == normalized_browser_location(after)
        });
    let ready = url_match
        && !loading
        && !error_page
        && (title_changed || content_confirmed || (same_destination && attempts >= 2));
    (ready, error_page, url_match, title_changed, content_changed)
}

pub(super) fn browser_page_identity(observation: &Observation) -> BrowserPageIdentity {
    let Some(target) = observation.target.as_ref() else {
        return BrowserPageIdentity::default();
    };
    let mut content = observation
        .ui_elements
        .iter()
        .filter(|element| overlaps(&element.bounds, &target.bounds))
        .filter(|element| !element.control_type.eq_ignore_ascii_case("edit"))
        .map(|element| element.name.trim())
        .filter(|text| !text.is_empty())
        .take(24)
        .collect::<Vec<_>>()
        .join(" ");
    if content.is_empty() {
        content = observation
            .ocr
            .iter()
            .filter(|block| overlaps(&block.bounds, &target.bounds))
            .map(|block| block.text.trim())
            .filter(|text| !text.is_empty())
            .take(24)
            .collect::<Vec<_>>()
            .join(" ");
    }
    BrowserPageIdentity {
        title: normalized_text(&target.title),
        url: browser_address_url(observation, target),
        content: normalized_text(&content),
    }
}

pub(super) fn navigation_url_matches(
    destination: &str,
    observed: Option<&str>,
    direct_url: bool,
) -> bool {
    let Some(observed) = observed else {
        return false;
    };
    if !direct_url {
        return !observed.trim().is_empty();
    }
    let destination = normalized_browser_location(destination);
    let observed = normalized_browser_location(observed);
    observed == destination
        || observed.starts_with(&format!("{destination}?"))
        || observed.starts_with(&format!("{destination}/"))
}

pub(super) fn normalized_browser_location(value: &str) -> String {
    let mut normalized = value.trim().trim_end_matches('/').to_ascii_lowercase();
    if let Some(rest) = normalized.strip_prefix("https://") {
        normalized = rest.to_owned();
    } else if let Some(rest) = normalized.strip_prefix("http://") {
        normalized = rest.to_owned();
    }
    normalized
        .trim_start_matches("www.")
        .trim_end_matches('/')
        .to_owned()
}

pub(super) fn looks_like_loading_page(identity: &BrowserPageIdentity) -> bool {
    let text = format!("{} {}", identity.title, identity.content);
    [
        "loading",
        "please wait",
        "just a moment",
        "checking your browser",
    ]
    .iter()
    .any(|marker| text.contains(marker))
}

pub(super) fn looks_like_browser_error(identity: &BrowserPageIdentity) -> bool {
    let text = format!("{} {}", identity.title, identity.content);
    [
        "site can't be reached",
        "site cannot be reached",
        "page unavailable",
        "page not found",
        "404 not found",
    ]
    .iter()
    .any(|marker| text.contains(marker))
}
