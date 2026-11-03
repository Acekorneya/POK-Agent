//! Finding and unpacking the WhatsApp component (`pok-ai-whatsapp`).
//!
//! The unofficial WhatsApp client is a separate program (see
//! `sidecars/whatsapp`), but it ships inside the app: the desktop build embeds
//! it and [`extract`] unpacks it the first time WhatsApp is switched on, so
//! users never download or manage a second file. Nothing runs until they
//! turn WhatsApp on and scan the QR code.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::error::{PokError, Result};

/// The program's file name on this platform.
pub const fn program_name() -> &'static str {
    if cfg!(windows) {
        "pok-ai-whatsapp.exe"
    } else {
        "pok-ai-whatsapp"
    }
}

/// Where the unpacked program lives.
pub fn installed_path(data_dir: &Path) -> PathBuf {
    data_dir.join("whatsapp").join("bin").join(program_name())
}

/// A copy of the program that is already on disk: the unpacked one first,
/// then any of `folders` (next to the app, or a development build).
pub fn find(data_dir: &Path, folders: &[PathBuf]) -> Option<PathBuf> {
    std::iter::once(installed_path(data_dir))
        .chain(folders.iter().map(|folder| folder.join(program_name())))
        .find(|path| path.is_file())
}

/// Write the embedded program to `destination`, unless an identical copy is
/// already there. A different copy (an older version of the app unpacked it)
/// is replaced. The file is written beside the target and renamed, so a
/// half-written program is never run.
pub fn extract(embedded: &[u8], destination: &Path) -> Result<()> {
    if embedded.is_empty() {
        return Err(PokError::Unsupported(
            "this build of POK-Agent does not include the WhatsApp component".into(),
        ));
    }
    if is_identical(embedded, destination) {
        return Ok(());
    }
    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let partial = destination.with_extension("unpacking");
    std::fs::write(&partial, embedded)?;
    if let Err(error) = std::fs::rename(&partial, destination) {
        let _ = std::fs::remove_file(&partial);
        // On Windows a program that is running cannot be replaced.
        return Err(PokError::Io(error));
    }
    Ok(())
}

/// Whether `path` already holds exactly `expected`. The length is compared
/// first so an unrelated file is rejected without reading it all.
fn is_identical(expected: &[u8], path: &Path) -> bool {
    let Ok(metadata) = std::fs::metadata(path) else {
        return false;
    };
    if metadata.len() != expected.len() as u64 {
        return false;
    }
    std::fs::read(path)
        .map(|existing| Sha256::digest(&existing) == Sha256::digest(expected))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target(dir: &tempfile::TempDir) -> PathBuf {
        installed_path(dir.path())
    }

    #[test]
    fn the_embedded_program_is_unpacked_where_the_app_looks_for_it() {
        let dir = tempfile::tempdir().unwrap();
        let destination = target(&dir);
        assert!(find(dir.path(), &[]).is_none());
        extract(b"MZ pretend program", &destination).unwrap();
        assert_eq!(std::fs::read(&destination).unwrap(), b"MZ pretend program");
        assert_eq!(find(dir.path(), &[]), Some(destination.clone()));
        assert!(!destination.with_extension("unpacking").exists());
    }

    #[test]
    fn an_identical_copy_is_left_alone_and_a_different_one_is_replaced() {
        let dir = tempfile::tempdir().unwrap();
        let destination = target(&dir);
        extract(b"version one", &destination).unwrap();
        let first = std::fs::metadata(&destination).unwrap().modified().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(30));

        // The same program again: not rewritten.
        extract(b"version one", &destination).unwrap();
        assert_eq!(
            std::fs::metadata(&destination).unwrap().modified().unwrap(),
            first
        );

        // A new version of the app carries a new program: it replaces the old.
        extract(b"version two!", &destination).unwrap();
        assert_eq!(std::fs::read(&destination).unwrap(), b"version two!");

        // Same length, different content, is still different.
        extract(b"version twoX", &destination).unwrap();
        assert_eq!(std::fs::read(&destination).unwrap(), b"version twoX");
    }

    #[test]
    fn a_build_without_the_component_says_so_instead_of_writing_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let error = extract(&[], &target(&dir)).unwrap_err().to_string();
        assert!(
            error.contains("does not include the WhatsApp component"),
            "{error}"
        );
        assert!(!target(&dir).exists());
    }

    #[test]
    fn the_unpacked_copy_wins_over_other_folders() {
        let data = tempfile::tempdir().unwrap();
        let beside = tempfile::tempdir().unwrap();
        assert!(find(data.path(), &[beside.path().to_owned()]).is_none());
        std::fs::write(beside.path().join(program_name()), "x").unwrap();
        assert_eq!(
            find(data.path(), &[beside.path().to_owned()]),
            Some(beside.path().join(program_name()))
        );
        let unpacked = installed_path(data.path());
        std::fs::create_dir_all(unpacked.parent().unwrap()).unwrap();
        std::fs::write(&unpacked, "y").unwrap();
        assert_eq!(
            find(data.path(), &[beside.path().to_owned()]),
            Some(unpacked)
        );
    }
}
