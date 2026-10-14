//! User-added OpenAI-compatible endpoints: saved in the data directory and merged into the provider catalog.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{
    PokError, Result,
    brain::ModelInfo,
    config::{
        Config, ProviderConfig, ProviderDataBoundary, ProviderProtocol,
        default_provider_max_retries,
    },
    memory::atomic_write,
};

const MAX_NAME_LEN: usize = 32;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CustomProvider {
    /// Catalog key: lowercase letters, digits, `-` and `_`.
    pub name: String,
    /// Name shown in the provider list.
    pub label: String,
    pub base_url: String,
    /// `None` decides from the address: loopback is this device, anything else external.
    #[serde(default)]
    pub data_boundary: Option<ProviderDataBoundary>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct CustomProviderFile {
    providers: Vec<CustomProvider>,
}

fn path(data_dir: &Path) -> PathBuf {
    data_dir.join("custom-providers.json")
}

/// A catalog key from a display name: `"My Strata #2"` becomes `"my-strata-2"`.
pub fn slugify(label: &str) -> String {
    let mut slug = String::new();
    for ch in label.trim().chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch.to_ascii_lowercase());
        } else if !slug.is_empty() && !slug.ends_with('-') {
            slug.push('-');
        }
    }
    slug.truncate(MAX_NAME_LEN);
    slug.trim_end_matches('-').to_owned()
}

/// Only plain `http(s)` addresses without embedded credentials are accepted.
fn validate_url(base_url: &str) -> Result<reqwest::Url> {
    let url = reqwest::Url::parse(base_url)
        .map_err(|error| PokError::Config(format!("invalid endpoint URL: {error}")))?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
        return Err(PokError::Config(
            "the endpoint URL must start with http:// or https://".into(),
        ));
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(PokError::Config(
            "put the API key in the key field, not in the endpoint URL".into(),
        ));
    }
    Ok(url)
}

impl CustomProvider {
    /// Validate user input. Only plain `http(s)` addresses without embedded credentials are accepted.
    pub fn new(
        label: &str,
        base_url: &str,
        data_boundary: Option<ProviderDataBoundary>,
    ) -> Result<Self> {
        let label = label.trim();
        let name = slugify(label);
        if name.is_empty() {
            return Err(PokError::Config(
                "a custom endpoint needs a name with at least one letter or digit".into(),
            ));
        }
        let base_url = base_url.trim();
        validate_url(base_url)?;
        Ok(Self {
            name,
            label: label.chars().take(60).collect(),
            base_url: base_url.to_owned(),
            data_boundary,
        })
    }

    /// Environment/credential name that holds this endpoint's key.
    pub fn api_key_env(&self) -> String {
        format!(
            "CUSTOM_{}_API_KEY",
            self.name.to_uppercase().replace('-', "_")
        )
    }

    pub fn provider_config(&self) -> ProviderConfig {
        ProviderConfig {
            protocol: ProviderProtocol::OpenaiCompatible,
            base_url: self.base_url.clone(),
            data_boundary: self.data_boundary,
            api_key_env: Some(self.api_key_env()),
            max_retries: default_provider_max_retries(),
        }
    }
}

/// Saved custom endpoints. A missing file is an empty list; an unreadable one is an error.
pub fn load(data_dir: &Path) -> Result<Vec<CustomProvider>> {
    match std::fs::read(path(data_dir)) {
        Ok(bytes) => Ok(serde_json::from_slice::<CustomProviderFile>(&bytes)
            .map_err(|error| PokError::Config(format!("{}: {error}", path(data_dir).display())))?
            .providers),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(error) => Err(error.into()),
    }
}

fn save(data_dir: &Path, providers: Vec<CustomProvider>) -> Result<()> {
    atomic_write(
        &path(data_dir),
        &serde_json::to_vec_pretty(&CustomProviderFile { providers })?,
    )
}

/// Merge saved endpoints into the catalog. A name already configured in the TOML file or
/// built in wins, so a saved endpoint can never replace a real provider.
pub fn apply(config: &mut Config) -> Result<()> {
    for provider in load(&config.data_dir)? {
        config
            .providers
            .entry(provider.name.clone())
            .or_insert_with(|| provider.provider_config());
    }
    Ok(())
}

/// Save a custom endpoint and add it to the live catalog. Saving a name that is already a
/// custom endpoint replaces it; a built-in or TOML-configured name is refused.
pub fn add(config: &mut Config, provider: CustomProvider) -> Result<()> {
    let mut saved = load(&config.data_dir)?;
    let replaces_custom = saved.iter().any(|item| item.name == provider.name);
    if !replaces_custom && config.providers.contains_key(&provider.name) {
        return Err(PokError::Config(format!(
            "{:?} is already a built-in provider; pick another name",
            provider.name
        )));
    }
    saved.retain(|item| item.name != provider.name);
    saved.push(provider.clone());
    save(&config.data_dir, saved)?;
    config
        .providers
        .insert(provider.name.clone(), provider.provider_config());
    Ok(())
}

/// Forget a custom endpoint. Built-in providers cannot be removed. Returns the removed entry.
pub fn remove(config: &mut Config, name: &str) -> Result<CustomProvider> {
    let mut saved = load(&config.data_dir)?;
    let index = saved
        .iter()
        .position(|item| item.name == name)
        .ok_or_else(|| PokError::Config(format!("{name:?} is not a custom endpoint")))?;
    let removed = saved.remove(index);
    save(&config.data_dir, saved)?;
    config.providers.remove(name);
    forget_capabilities(&config.data_dir, name);
    Ok(removed)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProbeOutcome {
    /// The server answered with a model list.
    Ready,
    /// The server wants an API key and none was given.
    NeedsKey,
    /// The server refused the key that was given.
    BadKey,
    /// Something answered, but not like an OpenAI-compatible server.
    NotCompatible,
    /// Nothing answered at that address.
    Unreachable,
    /// The address itself is not usable.
    InvalidUrl,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProbeResult {
    pub outcome: ProbeOutcome,
    /// The address that worked: the typed one, or the typed one with `/v1` added.
    pub resolved_url: String,
    pub models: Vec<String>,
    pub model_count: usize,
    pub latency_ms: u64,
    /// One plain sentence for the person setting this up. Never contains the key.
    pub detail: String,
}

const PROBE_MODEL_LIMIT: usize = 50;

fn probe_result(
    outcome: ProbeOutcome,
    url: &str,
    detail: impl Into<String>,
    started: std::time::Instant,
) -> ProbeResult {
    ProbeResult {
        outcome,
        resolved_url: url.to_owned(),
        models: Vec::new(),
        model_count: 0,
        latency_ms: started.elapsed().as_millis() as u64,
        detail: detail.into(),
    }
}

/// Ask an OpenAI-compatible server for its model list and say what the answer means.
///
/// A missing `/v1` is added when that is what makes the server answer. Redirects are not followed,
/// so the key can never be forwarded to another host.
pub async fn probe(base_url: &str, api_key: Option<&str>) -> ProbeResult {
    let started = std::time::Instant::now();
    let typed = base_url.trim().trim_end_matches('/').to_owned();
    let url = match validate_url(&typed) {
        Ok(url) => url,
        Err(error) => {
            return probe_result(ProbeOutcome::InvalidUrl, &typed, error.to_string(), started);
        }
    };
    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .redirect(reqwest::redirect::Policy::none())
        .build()
    {
        Ok(client) => client,
        Err(error) => {
            return probe_result(
                ProbeOutcome::Unreachable,
                &typed,
                format!("could not start a connection: {error}"),
                started,
            );
        }
    };
    let key = api_key.map(str::trim).filter(|key| !key.is_empty());
    // An OpenAI-compatible base address ends in /v1. Some servers also answer /models at the bare root,
    // so a bare host must be tried with /v1 first or chat requests would later 404.
    let path = url.path().trim_end_matches('/');
    let with_v1 = format!("{typed}/v1");
    let candidates = if path.ends_with("/v1") {
        vec![typed.clone()]
    } else if path.is_empty() {
        vec![with_v1, typed.clone()]
    } else {
        vec![typed.clone(), with_v1]
    };
    let mut last = probe_result(
        ProbeOutcome::NotCompatible,
        &typed,
        "no answer in the OpenAI format",
        started,
    );
    for candidate in candidates {
        let mut request = client.get(format!("{candidate}/models"));
        if let Some(key) = key {
            request = request.bearer_auth(key);
        }
        let response = match request.send().await {
            Ok(response) => response,
            Err(error) => {
                let detail = if error.is_timeout() {
                    "no answer within 10 seconds: check the address, that the server is running, and any VPN or firewall in between".to_owned()
                } else if error.is_connect() {
                    format!(
                        "nothing is listening at {}: start the server or check the address and port",
                        url.host_str()
                            .map(|host| match url.port() {
                                Some(port) => format!("{host}:{port}"),
                                None => host.to_owned(),
                            })
                            .unwrap_or_default()
                    )
                } else {
                    "the connection failed before the server answered".to_owned()
                };
                return probe_result(ProbeOutcome::Unreachable, &candidate, detail, started);
            }
        };
        let status = response.status();
        if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
            return if key.is_some() {
                probe_result(
                    ProbeOutcome::BadKey,
                    &candidate,
                    "the server refused this API key",
                    started,
                )
            } else {
                probe_result(
                    ProbeOutcome::NeedsKey,
                    &candidate,
                    "the server needs an API key",
                    started,
                )
            };
        }
        if !status.is_success() {
            last = probe_result(
                ProbeOutcome::NotCompatible,
                &candidate,
                format!(
                    "the server answered HTTP {} at {candidate}/models",
                    status.as_u16()
                ),
                started,
            );
            continue;
        }
        let bytes = response.bytes().await.unwrap_or_default();
        let ids: Option<Vec<String>> = serde_json::from_slice::<serde_json::Value>(&bytes)
            .ok()
            .and_then(|value| value.get("data").and_then(|data| data.as_array()).cloned())
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| item.get("id").and_then(|id| id.as_str()).map(str::to_owned))
                    .collect()
            });
        match ids {
            Some(ids) => {
                let count = ids.len();
                let detail = match count {
                    0 => "connected, but the server lists no models yet".to_owned(),
                    1 => "connected, 1 model available".to_owned(),
                    n => format!("connected, {n} models available"),
                };
                return ProbeResult {
                    outcome: ProbeOutcome::Ready,
                    resolved_url: candidate,
                    models: ids.into_iter().take(PROBE_MODEL_LIMIT).collect(),
                    model_count: count,
                    latency_ms: started.elapsed().as_millis() as u64,
                    detail,
                };
            }
            None => {
                last = probe_result(
                    ProbeOutcome::NotCompatible,
                    &candidate,
                    "something answered, but not with an OpenAI-style model list",
                    started,
                );
            }
        }
    }
    last
}

/// What a model on a compatible server was seen to do when asked. `None` or empty means it could not
/// be told, which is different from "no".
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CapabilityProbe {
    #[serde(default)]
    pub tool_use: Option<bool>,
    #[serde(default)]
    pub vision: Option<bool>,
    #[serde(default)]
    pub reasoning: Option<bool>,
    /// Levels the server accepted for `reasoning_effort`, with `off` first when thinking can be switched off.
    #[serde(default)]
    pub reasoning_efforts: Vec<String>,
    /// One plain sentence per finding. Never contains the key.
    #[serde(default)]
    pub notes: Vec<String>,
}

impl CapabilityProbe {
    /// Fill what the server's own model list left unknown. Facts the server reported always win.
    pub fn apply_to(&self, info: &mut ModelInfo) {
        info.vision = info.vision.or(self.vision);
        info.tool_use = info.tool_use.or(self.tool_use);
        info.reasoning = info.reasoning.or(self.reasoning);
        if info.reasoning_efforts.is_empty() {
            info.reasoning_efforts = self.reasoning_efforts.clone();
        }
    }
}

fn capability_path(data_dir: &Path) -> PathBuf {
    data_dir.join("custom-capabilities.json")
}

fn capability_key(provider: &str, model: &str) -> String {
    format!(
        "{}\n{}",
        provider.to_ascii_lowercase(),
        model.to_ascii_lowercase()
    )
}

fn read_capabilities(data_dir: &Path) -> std::collections::BTreeMap<String, CapabilityProbe> {
    std::fs::read(capability_path(data_dir))
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_default()
}

/// The saved result of an earlier capability check for this model, if any.
pub fn load_capability(data_dir: &Path, provider: &str, model: &str) -> Option<CapabilityProbe> {
    read_capabilities(data_dir).remove(&capability_key(provider, model))
}

pub fn save_capability(
    data_dir: &Path,
    provider: &str,
    model: &str,
    probe: &CapabilityProbe,
) -> Result<()> {
    let mut all = read_capabilities(data_dir);
    all.insert(capability_key(provider, model), probe.clone());
    atomic_write(
        &capability_path(data_dir),
        &serde_json::to_vec_pretty(&all)?,
    )
}

fn forget_capabilities(data_dir: &Path, provider: &str) {
    let prefix = format!("{}\n", provider.to_ascii_lowercase());
    let mut all = read_capabilities(data_dir);
    let before = all.len();
    all.retain(|key, _| !key.starts_with(&prefix));
    if all.len() != before
        && let Ok(bytes) = serde_json::to_vec_pretty(&all)
    {
        let _ = atomic_write(&capability_path(data_dir), &bytes);
    }
}

/// A 64x64 solid red PNG: any model that can see should call it red.
const RED_SQUARE_PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAS0lEQVR42u3PQQkAAAgAsetfWiP4FgYrsKZeS0BAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEDgsqnc8OJg6Ln3AAAAAElFTkSuQmCC";

/// The server's own explanation of a refusal, shortened: `HTTP 400: unknown reasoning effort ...`.
fn refusal(status: reqwest::StatusCode, body: &serde_json::Value) -> String {
    let message = body
        .pointer("/error/message")
        .or_else(|| body.get("error"))
        .and_then(|value| value.as_str())
        .unwrap_or("");
    let message: String = message.chars().take(120).collect();
    if message.is_empty() {
        format!("HTTP {}", status.as_u16())
    } else {
        format!("HTTP {}: {message}", status.as_u16())
    }
}

fn reasoning_text(message: &serde_json::Value) -> String {
    ["reasoning_content", "reasoning"]
        .iter()
        .filter_map(|key| message.get(*key).and_then(|value| value.as_str()))
        .collect::<Vec<_>>()
        .join("")
}

/// Ask the model itself what it can do: call a tool, look at a picture, and think with and without
/// `reasoning_effort`. Sends about seven small requests, one at a time. The key is only ever used as a
/// bearer header on the typed address (redirects are not followed).
pub async fn probe_capabilities(
    base_url: &str,
    api_key: Option<&str>,
    model: &str,
) -> CapabilityProbe {
    use serde_json::{Value, json};

    let mut probe = CapabilityProbe::default();
    let base = base_url.trim().trim_end_matches('/').to_owned();
    if validate_url(&base).is_err() {
        probe
            .notes
            .push("the endpoint address is not usable".into());
        return probe;
    }
    let Ok(client) = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .redirect(reqwest::redirect::Policy::none())
        .build()
    else {
        return probe;
    };
    let key = api_key.map(str::trim).filter(|key| !key.is_empty());
    let chat = |extra: Value| {
        let client = client.clone();
        let url = format!("{base}/chat/completions");
        let mut body = json!({"model": model, "temperature": 0});
        if let (Some(target), Some(extra)) = (body.as_object_mut(), extra.as_object()) {
            target.extend(extra.clone());
        }
        async move {
            let mut request = client.post(url).json(&body);
            if let Some(key) = key {
                request = request.bearer_auth(key);
            }
            let response = request.send().await.ok()?;
            let status = response.status();
            let bytes = response.bytes().await.ok()?;
            Some((
                status,
                serde_json::from_slice::<Value>(&bytes).unwrap_or(Value::Null),
            ))
        }
    };
    let first_message = |body: &Value| {
        body.pointer("/choices/0/message")
            .cloned()
            .unwrap_or(Value::Null)
    };

    // Tools.
    let weather = json!({"type": "function", "function": {
        "name": "get_weather", "description": "Get the current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}});
    match chat(
        json!({"max_tokens": 512, "tools": [weather], "tool_choice": "auto", "messages": [
        {"role": "user", "content": "What is the weather in Paris? Use the get_weather tool."}]}),
    )
    .await
    {
        Some((status, body)) if status.is_success() => {
            let called = first_message(&body)
                .get("tool_calls")
                .and_then(Value::as_array)
                .is_some_and(|calls| !calls.is_empty());
            if called {
                probe.tool_use = Some(true);
            } else {
                probe.notes.push(
                    "it answered without calling the test tool, so tool use is not confirmed"
                        .into(),
                );
            }
        }
        Some((status, body)) if status.is_client_error() => {
            probe.tool_use = Some(false);
            probe.notes.push(format!(
                "the server rejected a request with tools ({})",
                refusal(status, &body)
            ));
        }
        _ => probe
            .notes
            .push("the tool check got no usable answer".into()),
    }

    // Vision.
    let image = format!("data:image/png;base64,{RED_SQUARE_PNG_BASE64}");
    match chat(
        json!({"max_tokens": 512, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": image}},
        {"type": "text", "text": "What color is this image? Answer with one word."}]}]}),
    )
    .await
    {
        Some((status, body)) if status.is_success() => {
            let message = first_message(&body);
            let said = format!(
                "{} {}",
                message.get("content").and_then(Value::as_str).unwrap_or(""),
                reasoning_text(&message)
            )
            .to_lowercase();
            probe.vision = Some(said.contains("red"));
            if !said.contains("red") {
                probe
                    .notes
                    .push("it accepted a picture but did not describe it correctly".into());
            }
        }
        Some((status, body)) if status.is_client_error() => {
            probe.vision = Some(false);
            probe.notes.push(format!(
                "the server rejected a request with an image ({})",
                refusal(status, &body)
            ));
        }
        _ => probe
            .notes
            .push("the image check got no usable answer".into()),
    }

    // Thinking: does the model think, and can `reasoning_effort` change that?
    let question = json!([{"role": "user", "content": "What is 17 times 23? Answer briefly."}]);
    let default_run = chat(json!({"max_tokens": 300, "messages": question})).await;
    let thinks = default_run
        .as_ref()
        .filter(|(status, _)| status.is_success())
        .map(|(_, body)| !reasoning_text(&first_message(body)).is_empty());
    match thinks {
        Some(false) => {
            probe.reasoning = Some(false);
            probe.notes.push("it does not show any thinking".into());
        }
        Some(true) => {
            probe.reasoning = Some(true);
            let off =
                chat(json!({"max_tokens": 300, "reasoning_effort": "none", "messages": question}))
                    .await;
            match off {
                Some((status, body)) if status.is_success() => {
                    if reasoning_text(&first_message(&body)).is_empty() {
                        probe.reasoning_efforts.push("off".into());
                        for level in ["low", "medium", "high"] {
                            if let Some((status, _)) = chat(json!({"max_tokens": 200, "reasoning_effort": level, "messages": question})).await
                                && status.is_success()
                            {
                                probe.reasoning_efforts.push(level.into());
                            }
                        }
                    } else {
                        probe.notes.push(
                            "it thinks, but reasoning_effort does not switch the thinking off"
                                .into(),
                        );
                    }
                }
                _ => probe.notes.push(
                    "the server rejected reasoning_effort, so thinking cannot be controlled".into(),
                ),
            }
        }
        None => probe
            .notes
            .push("the thinking check got no usable answer".into()),
    }
    probe
}

/// Ask a server that loads models on demand (POK-Serve's gateway, Strata) to load `model` now, so the
/// first message does not wait for it: `POST {base}/load`.
///
/// `Ok(true)`: the server loaded it. `Ok(false)`: the server has no such endpoint (nothing to do).
/// `Err`: it tried and failed, or could not be reached. Waits up to 20 minutes, as big models take minutes.
pub async fn warm_up(
    base_url: &str,
    api_key: Option<&str>,
    model: &str,
) -> std::result::Result<bool, String> {
    let base = base_url.trim().trim_end_matches('/');
    validate_url(base).map_err(|error| error.to_string())?;
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(1200))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|error| error.to_string())?;
    let mut request = client
        .post(format!("{base}/load"))
        .json(&serde_json::json!({"model": model}));
    if let Some(key) = api_key.map(str::trim).filter(|key| !key.is_empty()) {
        request = request.bearer_auth(key);
    }
    let response = request
        .send()
        .await
        .map_err(|_| "could not reach the server to load the model".to_owned())?;
    let status = response.status();
    if status.is_success() {
        return Ok(true);
    }
    if matches!(status.as_u16(), 404 | 405 | 501) {
        return Ok(false);
    }
    let body = response
        .bytes()
        .await
        .ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .unwrap_or(serde_json::Value::Null);
    Err(refusal(status, &body))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_in(dir: &Path) -> Config {
        Config {
            data_dir: dir.to_path_buf(),
            ..Config::default()
        }
    }

    #[test]
    fn names_become_catalog_keys() {
        assert_eq!(slugify("  My Strata #2 "), "my-strata-2");
        assert_eq!(slugify("***"), "");
        assert_eq!(slugify(&"a".repeat(80)).len(), MAX_NAME_LEN);
    }

    #[test]
    fn rejects_unusable_input() {
        assert!(CustomProvider::new("   ", "http://127.0.0.1:8080/v1", None).is_err());
        assert!(CustomProvider::new("x", "ftp://example.com/v1", None).is_err());
        assert!(CustomProvider::new("x", "not a url", None).is_err());
        assert!(CustomProvider::new("x", "https://user:secret@example.com/v1", None).is_err());
        assert!(CustomProvider::new("x", "http://127.0.0.1:8080/v1", None).is_ok());
    }

    #[test]
    fn key_env_is_namespaced_so_it_never_hits_a_builtin_key() {
        let provider = CustomProvider::new("My Strata", "http://127.0.0.1:8080/v1", None).unwrap();
        assert_eq!(provider.api_key_env(), "CUSTOM_MY_STRATA_API_KEY");
    }

    #[test]
    fn boundary_follows_the_address_unless_chosen() {
        let local = CustomProvider::new("a", "http://127.0.0.1:8080/v1", None).unwrap();
        assert_eq!(
            local.provider_config().resolved_data_boundary(&local.name),
            ProviderDataBoundary::LocalDevice
        );
        let lan = CustomProvider::new("b", "http://192.168.1.20:8080/v1", None).unwrap();
        assert_eq!(
            lan.provider_config().resolved_data_boundary(&lan.name),
            ProviderDataBoundary::ExternalService
        );
        let trusted = CustomProvider::new(
            "c",
            "http://192.168.1.20:8080/v1",
            Some(ProviderDataBoundary::LocalDevice),
        )
        .unwrap();
        assert_eq!(
            trusted
                .provider_config()
                .resolved_data_boundary(&trusted.name),
            ProviderDataBoundary::LocalDevice
        );
    }

    #[test]
    fn add_persists_and_reloads_into_a_fresh_config() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = config_in(dir.path());
        let provider =
            CustomProvider::new("Home Server", "http://127.0.0.1:8080/v1", None).unwrap();
        add(&mut config, provider.clone()).unwrap();
        assert_eq!(config.providers["home-server"].base_url, provider.base_url);

        let mut reloaded = config_in(dir.path());
        apply(&mut reloaded).unwrap();
        let entry = &reloaded.providers["home-server"];
        assert_eq!(
            entry.api_key_env.as_deref(),
            Some("CUSTOM_HOME_SERVER_API_KEY")
        );
        assert!(matches!(entry.protocol, ProviderProtocol::OpenaiCompatible));
        reloaded.validate().unwrap();
    }

    #[test]
    fn add_replaces_a_custom_entry_but_refuses_a_builtin_name() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = config_in(dir.path());
        add(
            &mut config,
            CustomProvider::new("box", "http://127.0.0.1:1/v1", None).unwrap(),
        )
        .unwrap();
        add(
            &mut config,
            CustomProvider::new("box", "http://127.0.0.1:2/v1", None).unwrap(),
        )
        .unwrap();
        assert_eq!(load(dir.path()).unwrap().len(), 1);
        assert_eq!(config.providers["box"].base_url, "http://127.0.0.1:2/v1");

        let builtin = CustomProvider::new("Ollama", "http://127.0.0.1:3/v1", None).unwrap();
        config
            .providers
            .insert("ollama".into(), builtin.provider_config());
        assert!(add(&mut config, builtin).is_err());
    }

    #[test]
    fn saved_entry_never_replaces_a_configured_provider() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = config_in(dir.path());
        config.providers.insert(
            "ollama".into(),
            CustomProvider::new("x", "http://127.0.0.1:11434/v1", None)
                .unwrap()
                .provider_config(),
        );
        save(
            dir.path(),
            vec![CustomProvider::new("ollama", "https://evil.example.com/v1", None).unwrap()],
        )
        .unwrap();
        apply(&mut config).unwrap();
        assert_eq!(
            config.providers["ollama"].base_url,
            "http://127.0.0.1:11434/v1"
        );
    }

    #[test]
    fn remove_deletes_only_custom_entries() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = config_in(dir.path());
        add(
            &mut config,
            CustomProvider::new("box", "http://127.0.0.1:1/v1", None).unwrap(),
        )
        .unwrap();
        assert!(remove(&mut config, "lm_studio").is_err());
        assert!(config.providers.contains_key("lm_studio"));
        assert_eq!(remove(&mut config, "box").unwrap().name, "box");
        assert!(!config.providers.contains_key("box"));
        assert!(load(dir.path()).unwrap().is_empty());
    }

    #[test]
    fn unreadable_file_is_an_error_not_silently_empty() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(path(dir.path()), b"{not json").unwrap();
        assert!(load(dir.path()).is_err());
    }

    mod probe_tests {
        use super::super::*;
        use axum::{Json, Router, http::HeaderMap, http::StatusCode, routing::get};

        async fn models(
            headers: HeaderMap,
        ) -> std::result::Result<Json<serde_json::Value>, StatusCode> {
            match headers
                .get("authorization")
                .and_then(|value| value.to_str().ok())
            {
                Some("Bearer good-key") => Ok(Json(serde_json::json!({
                    "data": [{"id": "model-a"}, {"id": "model-b"}]
                }))),
                _ => Err(StatusCode::UNAUTHORIZED),
            }
        }

        async fn serve() -> String {
            let app = Router::new()
                .route("/v1/models", get(models))
                .route("/models", get(models))
                .route("/html/models", get(|| async { "<html>hi</html>" }))
                .route(
                    "/empty/v1/models",
                    get(|| async { Json(serde_json::json!({"data": []})) }),
                );
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let port = listener.local_addr().unwrap().port();
            tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            format!("http://127.0.0.1:{port}")
        }

        #[tokio::test]
        async fn ready_with_the_right_key_lists_models() {
            let base = serve().await;
            let result = probe(&format!("{base}/v1"), Some("good-key")).await;
            assert_eq!(result.outcome, ProbeOutcome::Ready);
            assert_eq!(result.models, ["model-a", "model-b"]);
            assert_eq!(result.model_count, 2);
            assert_eq!(result.resolved_url, format!("{base}/v1"));
        }

        #[tokio::test]
        async fn missing_and_wrong_keys_are_told_apart() {
            let base = serve().await;
            let url = format!("{base}/v1/");
            assert_eq!(probe(&url, None).await.outcome, ProbeOutcome::NeedsKey);
            assert_eq!(
                probe(&url, Some("  ")).await.outcome,
                ProbeOutcome::NeedsKey
            );
            let wrong = probe(&url, Some("wrong-secret")).await;
            assert_eq!(wrong.outcome, ProbeOutcome::BadKey);
            assert!(!wrong.detail.contains("wrong-secret"));
        }

        #[tokio::test]
        async fn a_missing_v1_is_added_when_that_makes_it_work() {
            let base = serve().await;
            let result = probe(&base, Some("good-key")).await;
            assert_eq!(result.outcome, ProbeOutcome::Ready);
            assert_eq!(result.resolved_url, format!("{base}/v1"));
        }

        #[tokio::test]
        async fn an_empty_model_list_is_ready_but_says_so() {
            let base = serve().await;
            let result = probe(&format!("{base}/empty"), None).await;
            assert_eq!(result.outcome, ProbeOutcome::Ready);
            assert_eq!(result.model_count, 0);
            assert!(result.detail.contains("no models"));
        }

        #[tokio::test]
        async fn something_that_is_not_an_openai_server_is_not_compatible() {
            let base = serve().await;
            let result = probe(&format!("{base}/html"), None).await;
            assert_eq!(result.outcome, ProbeOutcome::NotCompatible);
        }

        #[tokio::test]
        async fn a_closed_port_is_unreachable_and_a_bad_address_is_invalid() {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let port = listener.local_addr().unwrap().port();
            drop(listener);
            let closed = probe(&format!("http://127.0.0.1:{port}/v1"), None).await;
            assert_eq!(closed.outcome, ProbeOutcome::Unreachable);
            assert!(closed.detail.contains("nothing is listening"));
            for bad in [
                "ftp://example.com/v1",
                "not a url",
                "https://user:pw@example.com/v1",
            ] {
                assert_eq!(
                    probe(bad, None).await.outcome,
                    ProbeOutcome::InvalidUrl,
                    "{bad}"
                );
            }
        }
    }

    mod capability_tests {
        use super::super::*;
        use axum::{Json, Router, http::StatusCode, routing::post};
        use serde_json::{Value, json};

        fn asks_for_image(body: &Value) -> bool {
            body.to_string().contains("image_url")
        }

        /// A model that calls tools, sees images, thinks by default, and honours reasoning_effort
        /// (but rejects "high").
        async fn capable(Json(body): Json<Value>) -> (StatusCode, Json<Value>) {
            let message = if body.get("tools").is_some() {
                json!({"role": "assistant", "content": null, "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}]})
            } else if asks_for_image(&body) {
                json!({"role": "assistant", "content": "Red"})
            } else {
                match body.get("reasoning_effort").and_then(Value::as_str) {
                    Some("none") => json!({"role": "assistant", "content": "391"}),
                    Some("high") => {
                        return (StatusCode::BAD_REQUEST, Json(json!({"error": "no high"})));
                    }
                    _ => {
                        json!({"role": "assistant", "content": "391", "reasoning_content": "17*23=391"})
                    }
                }
            };
            (
                StatusCode::OK,
                Json(json!({"choices": [{"message": message}]})),
            )
        }

        /// A plain text model: rejects tools and images, never shows thinking.
        async fn plain(Json(body): Json<Value>) -> (StatusCode, Json<Value>) {
            if body.get("tools").is_some() || asks_for_image(&body) {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({"error": "unsupported"})),
                );
            }
            (
                StatusCode::OK,
                Json(json!({"choices": [{"message": {"role": "assistant", "content": "391"}}]})),
            )
        }

        async fn serve() -> String {
            let app = Router::new()
                .route("/capable/v1/chat/completions", post(capable))
                .route("/plain/v1/chat/completions", post(plain));
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let port = listener.local_addr().unwrap().port();
            tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            format!("http://127.0.0.1:{port}")
        }

        #[tokio::test]
        async fn a_capable_model_is_detected_including_the_levels_it_accepts() {
            let base = serve().await;
            let probe = probe_capabilities(&format!("{base}/capable/v1"), None, "m").await;
            assert_eq!(probe.tool_use, Some(true));
            assert_eq!(probe.vision, Some(true));
            assert_eq!(probe.reasoning, Some(true));
            assert_eq!(probe.reasoning_efforts, ["off", "low", "medium"]);
        }

        #[tokio::test]
        async fn a_plain_text_model_is_reported_as_unable_not_unknown() {
            let base = serve().await;
            let probe = probe_capabilities(&format!("{base}/plain/v1"), None, "m").await;
            assert_eq!(probe.tool_use, Some(false));
            assert_eq!(probe.vision, Some(false));
            assert_eq!(probe.reasoning, Some(false));
            assert!(probe.reasoning_efforts.is_empty());
        }

        #[tokio::test]
        async fn an_unreachable_server_leaves_everything_unknown() {
            let probe = probe_capabilities("http://127.0.0.1:1/v1", None, "m").await;
            assert_eq!(
                (probe.tool_use, probe.vision, probe.reasoning),
                (None, None, None)
            );
            assert!(!probe.notes.is_empty());
        }

        #[test]
        fn a_probe_fills_only_what_the_server_left_unknown() {
            let probe = CapabilityProbe {
                tool_use: Some(true),
                vision: Some(false),
                reasoning: Some(true),
                reasoning_efforts: vec!["off".into(), "low".into()],
                notes: Vec::new(),
            };
            let mut info = ModelInfo {
                id: "m".into(),
                context_length: None,
                loaded_context_length: None,
                max_context_length: None,
                vision: Some(true), // the server said so: the probe must not overrule it
                tool_use: None,
                reasoning: None,
                supported_parameters: Vec::new(),
                reasoning_efforts: Vec::new(),
                reasoning_default: None,
                loaded: None,
            };
            probe.apply_to(&mut info);
            assert_eq!(info.vision, Some(true));
            assert_eq!(info.tool_use, Some(true));
            assert_eq!(info.reasoning, Some(true));
            assert_eq!(info.reasoning_efforts, ["off", "low"]);
            info.reasoning_efforts = vec!["high".into()];
            probe.apply_to(&mut info);
            assert_eq!(info.reasoning_efforts, ["high"]);
        }

        #[test]
        fn results_are_kept_per_model_case_insensitively_and_dropped_with_the_endpoint() {
            let dir = tempfile::tempdir().unwrap();
            let probe = CapabilityProbe {
                tool_use: Some(true),
                ..CapabilityProbe::default()
            };
            save_capability(dir.path(), "Box", "Model-A", &probe).unwrap();
            assert_eq!(load_capability(dir.path(), "box", "model-a"), Some(probe));
            assert_eq!(load_capability(dir.path(), "box", "other"), None);

            let mut config = Config {
                data_dir: dir.path().to_path_buf(),
                ..Config::default()
            };
            add(
                &mut config,
                CustomProvider::new("box", "http://127.0.0.1:1/v1", None).unwrap(),
            )
            .unwrap();
            remove(&mut config, "box").unwrap();
            assert_eq!(load_capability(dir.path(), "box", "model-a"), None);
        }
    }

    /// Against a real server: `POKSERVE_URL=http://host:8085/v1 POKSERVE_KEY=... cargo test -p pok-ai-core
    /// --lib live_server -- --ignored --nocapture`. Skipped unless run on purpose.
    #[tokio::test]
    #[ignore = "needs a running OpenAI-compatible server"]
    async fn live_server_probe_and_capabilities() {
        let url = std::env::var("POKSERVE_URL").expect("POKSERVE_URL");
        let key = std::env::var("POKSERVE_KEY").ok();
        let probed = probe(&url, key.as_deref()).await;
        println!(
            "probe: {:?} {} ({} ms) models={:?}",
            probed.outcome, probed.detail, probed.latency_ms, probed.models
        );
        assert_eq!(probed.outcome, ProbeOutcome::Ready);
        let started = std::time::Instant::now();
        let found =
            probe_capabilities(&probed.resolved_url, key.as_deref(), &probed.models[0]).await;
        println!("capabilities after {:?}: {found:#?}", started.elapsed());
    }

    mod warm_up_tests {
        use super::super::*;
        use axum::{Json, Router, http::StatusCode, routing::post};
        use serde_json::{Value, json};

        async fn serve() -> String {
            let app =
                Router::new()
                    .route(
                        "/ok/v1/load",
                        post(|Json(body): Json<Value>| async move {
                            Json(json!({"model": body["model"]}))
                        }),
                    )
                    .route(
                        "/busy/v1/load",
                        post(|| async {
                            (
                                StatusCode::SERVICE_UNAVAILABLE,
                                Json(json!({"error": {"message": "out of memory"}})),
                            )
                        }),
                    );
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let port = listener.local_addr().unwrap().port();
            tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            format!("http://127.0.0.1:{port}")
        }

        #[tokio::test]
        async fn loaded_unsupported_and_failed_are_told_apart() {
            let base = serve().await;
            assert_eq!(warm_up(&format!("{base}/ok/v1"), None, "m").await, Ok(true));
            // No /load route at all: a plain server, nothing to do and nothing wrong.
            assert_eq!(
                warm_up(&format!("{base}/none/v1"), None, "m").await,
                Ok(false)
            );
            let failed = warm_up(&format!("{base}/busy/v1"), None, "m")
                .await
                .unwrap_err();
            assert!(
                failed.contains("503") && failed.contains("out of memory"),
                "{failed}"
            );
            assert!(warm_up("http://127.0.0.1:1/v1", None, "m").await.is_err());
            assert!(warm_up("ftp://x/v1", None, "m").await.is_err());
        }
    }
}
