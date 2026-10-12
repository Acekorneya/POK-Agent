//! Subscription sign-in (OAuth) for providers whose plans are meant for
//! coding agents, starting with GitHub Copilot: the device flow, the
//! entitlement check, and the credential the provider adapter consumes.
//!
//! The device flow is the same one the Copilot client uses (a public client
//! id, no redirect): show a code, the user approves it in the browser, and
//! the polling response carries the GitHub OAuth token that
//! `api.githubcopilot.com` accepts as a bearer token.
//!
//! Credentials live where provider API keys live: Windows Credential Manager
//! through the desktop app (stored as the `COPILOT_API_KEY` value), or a
//! data-directory file written by `pok-ai login copilot`.

use std::{path::Path, time::Duration};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{PokError, Result};

/// The public client id the GitHub Copilot device flow uses.
pub const COPILOT_CLIENT_ID: &str = "Ov23li8tweQw6odWQebz";
/// API version the Copilot endpoints are called with.
pub const COPILOT_API_VERSION: &str = "2026-08-01";
/// Where Copilot's chat API lives (enterprise deployments may differ).
pub const COPILOT_DEFAULT_ENDPOINT: &str = "https://api.githubcopilot.com";
/// The provider id and API-key name the desktop stores this credential under.
pub const COPILOT_PROVIDER: &str = "copilot";
pub const COPILOT_KEY_ENV: &str = "COPILOT_API_KEY";
const TOKEN_FILE_NAME: &str = "copilot-oauth.json";
const DEVICE_CODE_URL: &str = "https://github.com/login/device/code";
const TOKEN_URL: &str = "https://github.com/login/oauth/access_token";
const USER_URL: &str = "https://api.github.com/copilot_internal/user";

/// A GitHub Copilot sign-in: the token the API accepts, the account that
/// granted it, and the API endpoint (enterprise deployments differ).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CopilotCredential {
    pub access_token: String,
    #[serde(default)]
    pub account: Option<String>,
    #[serde(default)]
    pub api_endpoint: Option<String>,
}

impl CopilotCredential {
    /// Where this credential's chat API lives.
    pub fn api_endpoint(&self) -> &str {
        self.api_endpoint
            .as_deref()
            .map(str::trim)
            .filter(|endpoint| !endpoint.is_empty())
            .unwrap_or(COPILOT_DEFAULT_ENDPOINT)
    }

    /// The credential as stored: a JSON bundle (the desktop and the login
    /// command both keep it in one string, the way API keys are stored).
    pub fn to_stored(&self) -> Result<String> {
        serde_json::to_string(self).map_err(|error| PokError::Config(error.to_string()))
    }

    /// Read a stored credential: a JSON bundle, or a bare token (someone who
    /// pasted a GitHub token directly).
    pub fn from_stored(stored: &str) -> Option<Self> {
        let stored = stored.trim();
        if stored.is_empty() {
            return None;
        }
        if stored.starts_with('{') {
            return serde_json::from_str::<Self>(stored)
                .ok()
                .filter(|credential| !credential.access_token.trim().is_empty());
        }
        Some(Self {
            access_token: stored.to_owned(),
            account: None,
            api_endpoint: None,
        })
    }
}

/// A device flow waiting for the user to approve it in the browser.
#[derive(Debug, Clone, Serialize)]
pub struct CopilotLogin {
    pub user_code: String,
    pub verification_uri: String,
    pub interval_seconds: u64,
    #[serde(skip)]
    device_code: String,
}

/// What one poll of the device flow found.
#[derive(Debug)]
pub enum LoginPoll {
    /// The user has not approved the code yet; wait `interval_seconds`.
    Pending,
    /// GitHub asked us to slow down; wait the (increased) interval.
    SlowDown,
    Authorized(Box<CopilotCredential>),
    /// The code expired or was denied; start a new login.
    Expired,
}

fn github_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .user_agent(format!("POK-Agent/{}", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|error| PokError::Provider(error.to_string()))
}

/// Start a device login: the code to show and the URL to open.
pub async fn begin_copilot_login() -> Result<CopilotLogin> {
    let client = github_client()?;
    let response = client
        .post(DEVICE_CODE_URL)
        .header("Accept", "application/json")
        .json(&serde_json::json!({
            "client_id": COPILOT_CLIENT_ID,
            "scope": "read:user",
        }))
        .send()
        .await
        .map_err(provider_error)?;
    let response = response.error_for_status().map_err(provider_error)?;
    let body: Value = response.json().await.map_err(provider_error)?;
    let user_code = string_field(&body, "user_code")?;
    let verification_uri = string_field(&body, "verification_uri")?;
    let device_code = string_field(&body, "device_code")?;
    let interval = body
        .get("interval")
        .and_then(Value::as_u64)
        .unwrap_or(5)
        .max(1);
    Ok(CopilotLogin {
        user_code,
        verification_uri,
        interval_seconds: interval,
        device_code,
    })
}

/// Poll once for the login's outcome. The caller keeps the user code visible
/// until this returns `Authorized`.
pub async fn poll_copilot_login(login: &CopilotLogin) -> Result<LoginPoll> {
    let client = github_client()?;
    let response = client
        .post(TOKEN_URL)
        .header("Accept", "application/json")
        .json(&serde_json::json!({
            "client_id": COPILOT_CLIENT_ID,
            "device_code": login.device_code,
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        }))
        .send()
        .await
        .map_err(provider_error)?;
    let response = response.error_for_status().map_err(provider_error)?;
    let body: Value = response.json().await.map_err(provider_error)?;
    if let Some(access) = body.get("access_token").and_then(Value::as_str) {
        return Ok(LoginPoll::Authorized(Box::new(
            credential_from_token(access).await?,
        )));
    }
    match body.get("error").and_then(Value::as_str) {
        Some("authorization_pending") => Ok(LoginPoll::Pending),
        Some("slow_down") => Ok(LoginPoll::SlowDown),
        Some("expired_token" | "access_denied" | "incorrect_device_code") | None => {
            Ok(LoginPoll::Expired)
        }
        Some(_) => Ok(LoginPoll::Expired),
    }
}

/// Check the account's Copilot entitlement and build the credential. OAuth
/// itself succeeds for any GitHub account, so this is the only signal that
/// the account can actually chat.
async fn credential_from_token(access_token: &str) -> Result<CopilotCredential> {
    let client = github_client()?;
    let user = client
        .get(USER_URL)
        .header("Accept", "application/json")
        .header("Authorization", format!("Bearer {access_token}"))
        .header("X-GitHub-Api-Version", "2025-04-01")
        .send()
        .await
        .map_err(provider_error)?
        .error_for_status()
        .map_err(provider_error)?
        .json::<Value>()
        .await
        .ok();
    if let Some(user) = &user
        && user.get("chat_enabled").and_then(Value::as_bool) == Some(false)
    {
        let limited = user
            .get("can_signup_for_limited")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        return Err(PokError::Provider(if limited {
            "this GitHub account is not signed up for GitHub Copilot; sign up for Copilot Free at https://github.com/features/copilot/plans and connect again".into()
        } else {
            "this GitHub account does not have GitHub Copilot access; it needs an active Copilot subscription or a seat assigned by an organization".into()
        }));
    }
    let account = user
        .as_ref()
        .and_then(|user| {
            user.get("login")
                .or_else(|| user.pointer("/user/login"))
                .and_then(Value::as_str)
        })
        .map(str::to_owned);
    let api_endpoint = user
        .as_ref()
        .and_then(|user| user.pointer("/endpoints/api"))
        .and_then(Value::as_str)
        .map(|endpoint| endpoint.trim_end_matches('/').to_owned())
        .filter(|endpoint| !endpoint.is_empty());
    Ok(CopilotCredential {
        access_token: access_token.to_owned(),
        account,
        api_endpoint,
    })
}

/// Run the whole login: start it, then poll until the user approves it or
/// the code expires. `on_code` receives the code the moment it is available.
pub async fn copilot_login(mut on_code: impl FnMut(&CopilotLogin)) -> Result<CopilotCredential> {
    let login = begin_copilot_login().await?;
    on_code(&login);
    loop {
        match poll_copilot_login(&login).await? {
            LoginPoll::Pending => {
                tokio::time::sleep(Duration::from_secs(login.interval_seconds)).await;
            }
            LoginPoll::SlowDown => {
                tokio::time::sleep(Duration::from_secs(login.interval_seconds + 5)).await;
            }
            LoginPoll::Authorized(credential) => return Ok(*credential),
            LoginPoll::Expired => {
                return Err(PokError::Provider(
                    "the GitHub device code expired or was denied; start the login again".into(),
                ));
            }
        }
    }
}

/// The stored Copilot credential: the app's key store first (Windows
/// Credential Manager through the desktop, or an environment variable), then
/// the file `pok-ai login copilot` writes in the data directory.
pub fn stored_copilot_credential(data_dir: Option<&Path>) -> Option<CopilotCredential> {
    if let Some(stored) = crate::brain::get_api_key(COPILOT_KEY_ENV)
        && let Some(credential) = CopilotCredential::from_stored(&stored)
    {
        return Some(credential);
    }
    let stored = std::fs::read_to_string(token_file(data_dir?)).ok()?;
    CopilotCredential::from_stored(&stored)
}

/// Save the credential where `stored_copilot_credential` finds it. The
/// desktop stores it in Credential Manager instead (the login command uses
/// this for a machine without the app).
pub fn save_copilot_credential(data_dir: &Path, credential: &CopilotCredential) -> Result<()> {
    let path = token_file(data_dir);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    crate::memory::atomic_write(&path, credential.to_stored()?.as_bytes())?;
    restrict_permissions(&path);
    Ok(())
}

/// Remove the stored credential, if any.
pub fn clear_copilot_credential(data_dir: &Path) -> Result<()> {
    match std::fs::remove_file(token_file(data_dir)) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn token_file(data_dir: &Path) -> std::path::PathBuf {
    data_dir.join(TOKEN_FILE_NAME)
}

#[cfg(unix)]
fn restrict_permissions(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
}

#[cfg(not(unix))]
fn restrict_permissions(_path: &Path) {}

fn string_field(body: &Value, name: &str) -> Result<String> {
    body.get(name)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| PokError::Provider(format!("the device login response has no {name}")))
}

fn provider_error(error: reqwest::Error) -> PokError {
    PokError::Provider(error.to_string())
}

// ---------------------------------------------------------------------------
// ChatGPT / Codex (OpenAI subscription)

/// The public client id the Codex login uses.
pub const OPENAI_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
pub const OPENAI_ISSUER: &str = "https://auth.openai.com";
/// The ChatGPT backend the subscription token authorizes.
pub const CODEX_BASE_URL: &str = "https://chatgpt.com/backend-api/codex";
pub const CODEX_PROVIDER: &str = "chatgpt";
pub const CODEX_KEY_ENV: &str = "CHATGPT_API_KEY";
const CODEX_TOKEN_FILE_NAME: &str = "chatgpt-oauth.json";
const CODEX_DEVICE_URL: &str = "https://auth.openai.com/codex/device";
const CODEX_USERCODE_URL: &str = "https://auth.openai.com/api/accounts/deviceauth/usercode";
const CODEX_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
const CODEX_DEVICE_TOKEN_URL: &str = "https://auth.openai.com/api/accounts/deviceauth/token";
const CODEX_DEVICE_REDIRECT: &str = "https://auth.openai.com/deviceauth/callback";

/// A ChatGPT subscription sign-in: the short-lived access token, the
/// long-lived refresh token, the account the plan belongs to, and the access
/// token's expiry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OpenAiCredential {
    pub access_token: String,
    pub refresh_token: String,
    #[serde(default)]
    pub account_id: Option<String>,
    /// Unix milliseconds after which the access token is refreshed.
    #[serde(default)]
    pub expires_at: Option<u64>,
}

impl OpenAiCredential {
    fn from_tokens(
        access_token: String,
        refresh_token: String,
        id_token: Option<&str>,
        expires_in: Option<u64>,
    ) -> Self {
        Self {
            account_id: account_id_from_tokens(&access_token, id_token),
            access_token,
            refresh_token,
            expires_at: expires_in.map(|seconds| {
                now_ms()
                    .saturating_add(seconds.saturating_mul(1_000))
                    .saturating_sub(60_000)
            }),
        }
    }

    /// Whether the access token is at (or near) its expiry and must be
    /// refreshed before the next request. An unknown expiry (`None`) tries the
    /// token and refreshes on rejection.
    pub fn is_expired(&self) -> bool {
        self.expires_at.is_some_and(|at| now_ms() >= at)
    }

    pub fn to_stored(&self) -> Result<String> {
        serde_json::to_string(self).map_err(|error| PokError::Config(error.to_string()))
    }

    pub fn from_stored(stored: &str) -> Option<Self> {
        let stored = stored.trim();
        if !stored.starts_with('{') {
            return None;
        }
        serde_json::from_str::<Self>(stored)
            .ok()
            .filter(|credential| !credential.access_token.trim().is_empty())
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

/// The account id (`chatgpt_account_id`) from the id or access token's JWT
/// claims, top-level or under the API auth namespace.
fn account_id_from_tokens(access_token: &str, id_token: Option<&str>) -> Option<String> {
    id_token
        .and_then(account_id_from_jwt)
        .or_else(|| account_id_from_jwt(access_token))
}

fn account_id_from_jwt(token: &str) -> Option<String> {
    let payload = token.split('.').nth(1)?;
    let decoded =
        base64::Engine::decode(&base64::engine::general_purpose::URL_SAFE_NO_PAD, payload).ok()?;
    let claims: Value = serde_json::from_slice(&decoded).ok()?;
    claims
        .get("chatgpt_account_id")
        .and_then(Value::as_str)
        .or_else(|| {
            claims
                .get("https://api.openai.com/auth")
                .and_then(|auth| auth.get("chatgpt_account_id"))
                .and_then(Value::as_str)
        })
        .or_else(|| {
            claims
                .get("organizations")
                .and_then(Value::as_array)?
                .first()?
                .get("id")?
                .as_str()
        })
        .map(str::to_owned)
}

/// A ChatGPT device flow waiting for the user to approve it in the browser.
#[derive(Debug, Clone, Serialize)]
pub struct CodexLogin {
    pub user_code: String,
    pub verification_uri: String,
    pub interval_seconds: u64,
    #[serde(skip)]
    device_auth_id: String,
}

/// What one poll of the ChatGPT device flow found.
#[derive(Debug)]
pub enum CodexPoll {
    Pending,
    Authorized(Box<OpenAiCredential>),
    Expired,
}

/// Start a ChatGPT device login: the code to show and the URL to open.
pub async fn begin_codex_login() -> Result<CodexLogin> {
    let client = github_client()?;
    let response = client
        .post(CODEX_USERCODE_URL)
        .header("Content-Type", "application/json")
        .json(&serde_json::json!({ "client_id": OPENAI_CLIENT_ID }))
        .send()
        .await
        .map_err(provider_error)?;
    let response = response.error_for_status().map_err(provider_error)?;
    let body: Value = response.json().await.map_err(provider_error)?;
    let user_code = string_field(&body, "user_code")?;
    let device_auth_id = string_field(&body, "device_auth_id")?;
    let interval = body
        .get("interval")
        .and_then(|value| match value {
            Value::String(text) => text.parse::<u64>().ok(),
            other => other.as_u64(),
        })
        .unwrap_or(5)
        .max(1);
    Ok(CodexLogin {
        user_code,
        verification_uri: CODEX_DEVICE_URL.into(),
        interval_seconds: interval,
        device_auth_id,
    })
}

/// Poll once for the login's outcome; a 403/404 means the user has not
/// approved the code yet.
pub async fn poll_codex_login(login: &CodexLogin) -> Result<CodexPoll> {
    let client = github_client()?;
    let response = client
        .post(CODEX_DEVICE_TOKEN_URL)
        .header("Content-Type", "application/json")
        .json(&serde_json::json!({
            "device_auth_id": login.device_auth_id,
            "user_code": login.user_code,
        }))
        .send()
        .await
        .map_err(provider_error)?;
    if matches!(response.status().as_u16(), 403 | 404) {
        return Ok(CodexPoll::Pending);
    }
    let response = response.error_for_status().map_err(provider_error)?;
    let body: Value = response.json().await.map_err(provider_error)?;
    let Some(code) = body.get("authorization_code").and_then(Value::as_str) else {
        return Ok(CodexPoll::Expired);
    };
    let verifier = body
        .get("code_verifier")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let credential = exchange_codex_code(code, verifier).await?;
    Ok(CodexPoll::Authorized(Box::new(credential)))
}

/// Trade an authorization code for tokens (`redirect_uri` and verifier match
/// what the device flow issued).
async fn exchange_codex_code(
    authorization_code: &str,
    code_verifier: &str,
) -> Result<OpenAiCredential> {
    let client = github_client()?;
    let response = client
        .post(CODEX_TOKEN_URL)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .form(&[
            ("grant_type", "authorization_code"),
            ("code", authorization_code),
            ("redirect_uri", CODEX_DEVICE_REDIRECT),
            ("client_id", OPENAI_CLIENT_ID),
            ("code_verifier", code_verifier),
        ])
        .send()
        .await
        .map_err(provider_error)?;
    let response = response.error_for_status().map_err(provider_error)?;
    let body: Value = response.json().await.map_err(provider_error)?;
    let access_token = string_field(&body, "access_token")?;
    let refresh_token = string_field(&body, "refresh_token")?;
    let id_token = body.get("id_token").and_then(Value::as_str);
    let expires_in = body.get("expires_in").and_then(Value::as_u64);
    Ok(OpenAiCredential::from_tokens(
        access_token,
        refresh_token,
        id_token,
        expires_in,
    ))
}

/// Refresh an expired access token, keeping the account and refresh token.
pub async fn refresh_codex_credential(credential: &OpenAiCredential) -> Result<OpenAiCredential> {
    let client = github_client()?;
    let response = client
        .post(CODEX_TOKEN_URL)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .form(&[
            ("grant_type", "refresh_token"),
            ("refresh_token", credential.refresh_token.as_str()),
            ("client_id", OPENAI_CLIENT_ID),
        ])
        .send()
        .await
        .map_err(provider_error)?;
    let response = response.error_for_status().map_err(provider_error)?;
    let body: Value = response.json().await.map_err(provider_error)?;
    let access_token = string_field(&body, "access_token")?;
    let refresh_token = body
        .get("refresh_token")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| credential.refresh_token.clone());
    let id_token = body.get("id_token").and_then(Value::as_str);
    let expires_in = body.get("expires_in").and_then(Value::as_u64);
    let next = OpenAiCredential::from_tokens(access_token, refresh_token, id_token, expires_in);
    Ok(OpenAiCredential {
        account_id: next.account_id.or_else(|| credential.account_id.clone()),
        ..next
    })
}

/// Run the whole ChatGPT login: start it, then poll until the user approves
/// it or the code expires. `on_code` receives the code when it is available.
pub async fn codex_login(mut on_code: impl FnMut(&CodexLogin)) -> Result<OpenAiCredential> {
    let login = begin_codex_login().await?;
    on_code(&login);
    loop {
        match poll_codex_login(&login).await? {
            CodexPoll::Pending => {
                tokio::time::sleep(Duration::from_secs(login.interval_seconds)).await;
            }
            CodexPoll::Authorized(credential) => return Ok(*credential),
            CodexPoll::Expired => {
                return Err(PokError::Provider(
                    "the ChatGPT device code expired or was denied; start the login again".into(),
                ));
            }
        }
    }
}

/// The stored ChatGPT credential: the app's key store first (Windows
/// Credential Manager through the desktop, or an environment variable), then
/// the file `pok-ai login chatgpt` writes in the data directory.
pub fn stored_codex_credential(data_dir: Option<&Path>) -> Option<OpenAiCredential> {
    if let Some(stored) = crate::brain::get_api_key(CODEX_KEY_ENV)
        && let Some(credential) = OpenAiCredential::from_stored(&stored)
    {
        return Some(credential);
    }
    let stored = std::fs::read_to_string(codex_token_file(data_dir?)).ok()?;
    OpenAiCredential::from_stored(&stored)
}

/// Save the credential where `stored_codex_credential` finds it.
pub fn save_codex_credential(data_dir: &Path, credential: &OpenAiCredential) -> Result<()> {
    let path = codex_token_file(data_dir);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    crate::memory::atomic_write(&path, credential.to_stored()?.as_bytes())?;
    restrict_permissions(&path);
    Ok(())
}

/// Remove the stored credential, if any.
pub fn clear_codex_credential(data_dir: &Path) -> Result<()> {
    match std::fs::remove_file(codex_token_file(data_dir)) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn codex_token_file(data_dir: &Path) -> std::path::PathBuf {
    data_dir.join(CODEX_TOKEN_FILE_NAME)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stored_credentials_accept_bundles_and_bare_tokens() {
        let credential = CopilotCredential {
            access_token: "gho_example".into(),
            account: Some("example-user".into()),
            api_endpoint: Some("https://copilot-api.example.com".into()),
        };
        let stored = credential.to_stored().unwrap();
        assert_eq!(CopilotCredential::from_stored(&stored).unwrap(), credential);
        // A bare token pasted by a user still works, with the default endpoint.
        let bare = CopilotCredential::from_stored("gho_pasted_token").unwrap();
        assert_eq!(bare.access_token, "gho_pasted_token");
        assert_eq!(bare.api_endpoint(), COPILOT_DEFAULT_ENDPOINT);
        assert!(CopilotCredential::from_stored("  ").is_none());
        assert!(CopilotCredential::from_stored("{\"account\":null}").is_none());
    }

    #[test]
    fn the_token_file_round_trips_and_clears() {
        let temp = tempfile::tempdir().unwrap();
        assert!(stored_copilot_credential(Some(temp.path())).is_none());
        let credential = CopilotCredential {
            access_token: "gho_file".into(),
            account: None,
            api_endpoint: None,
        };
        save_copilot_credential(temp.path(), &credential).unwrap();
        assert_eq!(
            stored_copilot_credential(Some(temp.path())).unwrap(),
            credential
        );
        clear_copilot_credential(temp.path()).unwrap();
        assert!(stored_copilot_credential(Some(temp.path())).is_none());
        // Clearing a missing file is not an error.
        clear_copilot_credential(temp.path()).unwrap();
    }

    fn jwt(payload: &str) -> String {
        let encoded = base64::Engine::encode(
            &base64::engine::general_purpose::URL_SAFE_NO_PAD,
            payload.as_bytes(),
        );
        format!("header.{encoded}.signature")
    }

    #[test]
    fn chatgpt_account_ids_come_from_the_token_claims() {
        let namespaced = OpenAiCredential::from_tokens(
            jwt(r#"{"https://api.openai.com/auth":{"chatgpt_account_id":"acc-2"}}"#),
            "refresh".into(),
            None,
            Some(3600),
        );
        assert_eq!(namespaced.account_id.as_deref(), Some("acc-2"));
        assert!(!namespaced.is_expired(), "an hour-long token is fresh");
        let top_level = OpenAiCredential::from_tokens(
            jwt(r#"{"chatgpt_account_id":"acc-1"}"#),
            "r".into(),
            None,
            None,
        );
        assert_eq!(top_level.account_id.as_deref(), Some("acc-1"));
        // An unknown expiry tries the token; a past expiry forces a refresh.
        assert!(!top_level.is_expired());
        let mut stale = top_level.clone();
        stale.expires_at = Some(now_ms().saturating_sub(1));
        assert!(stale.is_expired());
        let organization = OpenAiCredential::from_tokens(
            jwt(r#"{"organizations":[{"id":"org-9"}]}"#),
            "r".into(),
            None,
            None,
        );
        assert_eq!(organization.account_id.as_deref(), Some("org-9"));
        // The id token wins when both tokens carry claims.
        let id_wins = OpenAiCredential::from_tokens(
            jwt(r#"{"chatgpt_account_id":"access"}"#),
            "r".into(),
            Some(&jwt(r#"{"chatgpt_account_id":"id"}"#)),
            None,
        );
        assert_eq!(id_wins.account_id.as_deref(), Some("id"));
        let stored = namespaced.to_stored().unwrap();
        assert_eq!(OpenAiCredential::from_stored(&stored).unwrap(), namespaced);
        assert!(OpenAiCredential::from_stored("bare-token").is_none());
    }

    #[test]
    fn codex_credentials_round_trip_through_the_token_file() {
        let temp = tempfile::tempdir().unwrap();
        assert!(stored_codex_credential(Some(temp.path())).is_none());
        let credential = OpenAiCredential {
            access_token: "a".into(),
            refresh_token: "r".into(),
            account_id: Some("acc".into()),
            expires_at: Some(now_ms() + 60_000),
        };
        save_codex_credential(temp.path(), &credential).unwrap();
        assert_eq!(
            stored_codex_credential(Some(temp.path())).unwrap(),
            credential
        );
        clear_codex_credential(temp.path()).unwrap();
        assert!(stored_codex_credential(Some(temp.path())).is_none());
    }
}
