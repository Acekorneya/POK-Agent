//! Finding and installing the WhatsApp component (`pok-ai-whatsapp`).
//!
//! The unofficial WhatsApp client is not part of the installer. It is a
//! separate program published with each release, downloaded on request, and
//! checked against the release's `SHA256SUMS.txt` before it is kept.

use std::path::{Path, PathBuf};

use futures::StreamExt;
use sha2::{Digest, Sha256};

use crate::error::{PokError, Result};

/// The release file the component is published as.
pub const RELEASE_ASSET: &str = "POK-Agent-whatsapp-windows-x64.exe";
const CHECKSUMS: &str = "SHA256SUMS.txt";
/// The component is about 20 MB; refuse anything absurd.
const MAX_BYTES: u64 = 150 * 1024 * 1024;

/// The program's file name on this platform.
pub const fn program_name() -> &'static str {
    if cfg!(windows) {
        "pok-ai-whatsapp.exe"
    } else {
        "pok-ai-whatsapp"
    }
}

/// Where an installed copy lives.
pub fn installed_path(data_dir: &Path) -> PathBuf {
    data_dir.join("whatsapp").join("bin").join(program_name())
}

/// The component, if it exists: the installed copy first, then any of
/// `folders` (next to the app, or a development build).
pub fn find(data_dir: &Path, folders: &[PathBuf]) -> Option<PathBuf> {
    std::iter::once(installed_path(data_dir))
        .chain(folders.iter().map(|folder| folder.join(program_name())))
        .find(|path| path.is_file())
}

/// Download the component from `release_base` (a folder URL holding
/// [`RELEASE_ASSET`] and `SHA256SUMS.txt`) to `destination`. The file is kept
/// only if its SHA-256 matches the published checksum.
pub async fn install(release_base: &str, destination: &Path) -> Result<()> {
    let base = release_base.trim_end_matches('/');
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(20))
        .timeout(std::time::Duration::from_secs(600))
        .user_agent("POK-Agent")
        .build()
        .map_err(|error| PokError::Provider(error.without_url().to_string()))?;

    let sums = fetch_text(&client, &format!("{base}/{CHECKSUMS}")).await?;
    let expected = expected_hash(&sums, RELEASE_ASSET).ok_or_else(|| {
        PokError::Provider(
            "This version of POK-Agent has no WhatsApp component. Update the app first.".into(),
        )
    })?;

    let response = client
        .get(format!("{base}/{RELEASE_ASSET}"))
        .send()
        .await
        .map_err(|error| PokError::Provider(error.without_url().to_string()))?;
    if !response.status().is_success() {
        return Err(PokError::Provider(format!(
            "Could not download the WhatsApp component (HTTP {})",
            response.status()
        )));
    }
    let mut hasher = Sha256::new();
    let mut bytes = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|error| PokError::Provider(error.without_url().to_string()))?;
        if bytes.len() as u64 + chunk.len() as u64 > MAX_BYTES {
            return Err(PokError::Provider(
                "The WhatsApp component is larger than expected; it was not installed.".into(),
            ));
        }
        hasher.update(&chunk);
        bytes.extend_from_slice(&chunk);
    }
    let actual = hex(&hasher.finalize());
    if actual != expected {
        return Err(PokError::Provider(
            "The downloaded WhatsApp component does not match its published checksum, so it was not installed."
                .into(),
        ));
    }

    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent)?;
    }
    // Write beside the target and rename, so a half-written file is never run.
    let partial = destination.with_extension("download");
    std::fs::write(&partial, &bytes)?;
    std::fs::rename(&partial, destination)?;
    Ok(())
}

async fn fetch_text(client: &reqwest::Client, url: &str) -> Result<String> {
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|error| PokError::Provider(error.without_url().to_string()))?;
    if !response.status().is_success() {
        return Err(PokError::Provider(format!(
            "This version of POK-Agent has no published WhatsApp component (HTTP {}). Update the app first.",
            response.status()
        )));
    }
    response
        .text()
        .await
        .map_err(|error| PokError::Provider(error.without_url().to_string()))
}

/// The hash listed for `name` in a `sha256sum`-style file (`<hash>  <name>`).
fn expected_hash(sums: &str, name: &str) -> Option<String> {
    sums.lines().find_map(|line| {
        let mut parts = line.split_whitespace();
        let (hash, file) = (parts.next()?, parts.next()?);
        (file.trim_start_matches('*') == name
            && hash.len() == 64
            && hash.chars().all(|c| c.is_ascii_hexdigit()))
        .then(|| hash.to_ascii_lowercase())
    })
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Router, http::StatusCode, routing::get};

    /// A release folder served from memory: (path, status, body).
    async fn serve(files: Vec<(&'static str, Vec<u8>)>) -> String {
        let mut app = Router::new();
        for (name, body) in files {
            app = app.route(
                &format!("/{name}"),
                get(move || {
                    let body = body.clone();
                    async move { (StatusCode::OK, body) }
                }),
            );
        }
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        base
    }

    fn sums_for(name: &str, content: &[u8]) -> Vec<u8> {
        let hash = hex(&Sha256::digest(content));
        format!("{hash}  POK-Agent-windows-x64.exe\n{hash}  {name}\n").into_bytes()
    }

    fn destination(dir: &tempfile::TempDir) -> PathBuf {
        dir.path().join("bin").join("pok-ai-whatsapp")
    }

    #[tokio::test]
    async fn a_download_that_matches_its_checksum_is_installed() {
        let program = b"MZ pretend program".to_vec();
        let base = serve(vec![
            (CHECKSUMS, sums_for(RELEASE_ASSET, &program)),
            (RELEASE_ASSET, program.clone()),
        ])
        .await;
        let dir = tempfile::tempdir().unwrap();
        let target = destination(&dir);
        install(&base, &target).await.unwrap();
        assert_eq!(std::fs::read(&target).unwrap(), program);
        assert!(!target.with_extension("download").exists());
    }

    #[tokio::test]
    async fn a_tampered_download_is_refused_and_leaves_nothing_behind() {
        let base = serve(vec![
            (CHECKSUMS, sums_for(RELEASE_ASSET, b"the real program")),
            (RELEASE_ASSET, b"something else entirely".to_vec()),
        ])
        .await;
        let dir = tempfile::tempdir().unwrap();
        let target = destination(&dir);
        let error = install(&base, &target).await.unwrap_err().to_string();
        assert!(error.contains("checksum"), "{error}");
        assert!(!target.exists());
        assert!(!target.with_extension("download").exists());
    }

    #[tokio::test]
    async fn a_release_without_the_component_says_to_update() {
        // Older releases publish checksums that do not list it.
        let base = serve(vec![(
            CHECKSUMS,
            b"0000000000000000000000000000000000000000000000000000000000000000  POK-Agent-windows-x64.exe\n"
                .to_vec(),
        )])
        .await;
        let dir = tempfile::tempdir().unwrap();
        let error = install(&base, &destination(&dir))
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("Update the app"), "{error}");

        // And a release that has no checksum file at all.
        let empty = serve(vec![]).await;
        let error = install(&empty, &destination(&dir))
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("Update the app"), "{error}");
    }

    #[test]
    fn checksums_are_read_from_sha256sum_style_lines() {
        let sums = "AAAA  other.zip\n\
                    9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08 *POK-Agent-whatsapp-windows-x64.exe\n";
        assert_eq!(
            expected_hash(sums, RELEASE_ASSET).as_deref(),
            Some("9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08")
        );
        assert!(
            expected_hash(sums, "other.zip").is_none(),
            "a short hash is not a checksum"
        );
        assert!(expected_hash("", RELEASE_ASSET).is_none());
    }

    #[test]
    fn the_installed_copy_wins_over_other_folders() {
        let data = tempfile::tempdir().unwrap();
        let beside = tempfile::tempdir().unwrap();
        assert!(find(data.path(), &[beside.path().to_owned()]).is_none());
        std::fs::write(beside.path().join(program_name()), "x").unwrap();
        assert_eq!(
            find(data.path(), &[beside.path().to_owned()]),
            Some(beside.path().join(program_name()))
        );
        let installed = installed_path(data.path());
        std::fs::create_dir_all(installed.parent().unwrap()).unwrap();
        std::fs::write(&installed, "y").unwrap();
        assert_eq!(
            find(data.path(), &[beside.path().to_owned()]),
            Some(installed)
        );
    }
}
