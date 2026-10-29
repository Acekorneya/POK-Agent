//! Settings, Web search: which provider answers `web_search`, the Tavily key,
//! and a try-it search. A saved Tavily key makes Tavily the default.

use pok_ai_core::{config::SearchProvider, web};
use serde::Serialize;
use tauri::State;

use super::{AppRuntime, persist_provider_key};

const TAVILY_ENV: &str = "TAVILY_API_KEY";

#[derive(Serialize)]
pub struct WebSearchStatus {
    /// What Settings or `[web]` asks for (`auto` follows the saved key).
    configured: SearchProvider,
    /// What `web_search` uses right now.
    effective: SearchProvider,
    has_tavily_key: bool,
}

fn status(runtime: &AppRuntime) -> WebSearchStatus {
    let config = runtime.config.lock().web.clone();
    WebSearchStatus {
        configured: web::search_provider_override().unwrap_or(config.search_provider),
        effective: web::effective_search_provider(&config),
        has_tavily_key: pok_ai_core::brain::get_api_key(TAVILY_ENV)
            .is_some_and(|key| !key.trim().is_empty()),
    }
}

#[tauri::command]
pub fn get_web_search_status(runtime: State<'_, AppRuntime>) -> WebSearchStatus {
    status(runtime.inner())
}

/// Choose the provider. It applies to the conversation in progress too.
#[tauri::command]
pub fn set_web_search_provider(
    provider: SearchProvider,
    runtime: State<'_, AppRuntime>,
) -> WebSearchStatus {
    web::set_search_provider_override(Some(provider));
    runtime.config.lock().web.search_provider = provider;
    status(runtime.inner())
}

/// Check the key with Tavily first, so a typo is never saved, then keep it in
/// Windows Credential Manager like a provider key.
#[tauri::command]
pub async fn save_tavily_key(
    api_key: String,
    runtime: State<'_, AppRuntime>,
) -> Result<WebSearchStatus, String> {
    let key = api_key.trim();
    if key.is_empty() {
        return Err("The Tavily API key cannot be empty.".into());
    }
    web::verify_tavily_key(key)
        .await
        .map_err(|error| error.to_string())?;
    persist_provider_key(TAVILY_ENV, key)?;
    pok_ai_core::brain::set_api_key(TAVILY_ENV, key);
    Ok(status(runtime.inner()))
}

#[tauri::command]
pub fn clear_tavily_key(runtime: State<'_, AppRuntime>) -> Result<WebSearchStatus, String> {
    persist_provider_key(TAVILY_ENV, "")?;
    pok_ai_core::brain::set_api_key(TAVILY_ENV, "");
    Ok(status(runtime.inner()))
}

/// One real search with whatever `web_search` would use now.
#[tauri::command]
pub async fn try_web_search(
    query: String,
    runtime: State<'_, AppRuntime>,
) -> Result<serde_json::Value, String> {
    let query = query.trim().to_owned();
    if query.is_empty() {
        return Err("Type something to search for.".into());
    }
    let config = runtime.config.lock().web.clone();
    web::try_search(&config, &query)
        .await
        .map_err(|error| error.to_string())
}
