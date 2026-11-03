//! Pictures and files that arrive with a chat message.
//!
//! Every attachment is saved in an inbox folder the agent may read. Pictures
//! are also decoded (JPEG, PNG, WebP, GIF, BMP), turned upright from their
//! EXIF orientation, scaled to the size the dashboard uses, and encoded as PNG
//! for a vision model: the session only accepts PNG, and phones send JPEG.
//! Voice notes and video are kept as files with a note, since they cannot be
//! transcribed or watched yet.

use std::{
    io::Cursor,
    path::{Path, PathBuf},
    time::{Duration, SystemTime},
};

use async_trait::async_trait;
use base64::Engine as _;
use image::{DynamicImage, ImageFormat, imageops::FilterType};

use crate::error::{PokError, Result};

/// Longest edge of a picture shown to the model; the dashboard composer uses
/// the same value.
pub const MAX_IMAGE_EDGE: u32 = 1568;
/// Largest attachment taken from a chat, in bytes. Telegram bots cannot
/// download more than 20 MB anyway.
pub const MAX_DOWNLOAD_BYTES: u64 = 20 * 1024 * 1024;
/// A picture with more pixels than this is refused instead of decoded.
const MAX_PIXELS: u64 = 64_000_000;
/// How long a message's saved files stay in the inbox.
pub const INBOX_RETENTION: Duration = Duration::from_secs(60 * 60 * 24 * 7);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaKind {
    Image,
    Document,
    /// A voice note or other audio.
    Audio,
    Video,
    /// Stickers carry no request; they are skipped.
    Sticker,
}

/// A downloaded attachment.
#[derive(Debug, Clone)]
pub struct Download {
    pub kind: MediaKind,
    pub name: Option<String>,
    pub mime: Option<String>,
    pub bytes: Vec<u8>,
}

/// An attachment a channel can fetch on request. The dispatcher fetches only
/// after the sender passed the allowlist, so a stranger cannot make the
/// agent download anything.
#[async_trait]
pub trait MediaSource: Send + Sync {
    fn kind(&self) -> MediaKind;
    /// A short label for notes, such as the file name.
    fn label(&self) -> String;
    /// The size the channel announced, when it did.
    fn size_hint(&self) -> Option<u64>;
    async fn fetch(&self, max_bytes: u64) -> Result<Download>;
}

/// What a message's attachments became.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Prepared {
    /// PNG images, base64, for a vision model.
    pub images: Vec<String>,
    /// Every saved file (pictures included), readable by the agent.
    pub files: Vec<PathBuf>,
    /// Things that were skipped or cannot be used yet, in plain words.
    pub notes: Vec<String>,
}

impl Prepared {
    pub fn is_empty(&self) -> bool {
        self.images.is_empty() && self.files.is_empty()
    }

    pub fn merge(&mut self, other: Prepared) {
        self.images.extend(other.images);
        self.files.extend(other.files);
        self.notes.extend(other.notes);
        self.images.truncate(crate::session::MAX_USER_IMAGES);
        self.files.truncate(crate::session::MAX_ATTACHED_FILES);
    }
}

/// Fetch and save a message's attachments under `inbox/<folder>/`.
pub async fn prepare(
    inbox: &Path,
    folder: &str,
    sources: &[std::sync::Arc<dyn MediaSource>],
) -> Prepared {
    let mut prepared = Prepared::default();
    let directory = inbox.join(safe_component(folder));
    for source in sources {
        let label = source.label();
        match source.kind() {
            MediaKind::Sticker => continue,
            MediaKind::Audio => prepared.notes.push(format!(
                "A voice note or audio file ({label}) came with the message. Audio cannot be transcribed yet, so it is only saved."
            )),
            MediaKind::Video => prepared.notes.push(format!(
                "A video ({label}) came with the message. It cannot be watched, so it is only saved."
            )),
            MediaKind::Image | MediaKind::Document => {}
        }
        if source
            .size_hint()
            .is_some_and(|size| size > MAX_DOWNLOAD_BYTES)
        {
            prepared.notes.push(format!(
                "{label} is larger than {} MB, so it was not downloaded.",
                MAX_DOWNLOAD_BYTES / (1024 * 1024)
            ));
            continue;
        }
        let download = match source.fetch(MAX_DOWNLOAD_BYTES).await {
            Ok(download) => download,
            Err(error) => {
                tracing::warn!("gateway could not download an attachment: {error}");
                prepared
                    .notes
                    .push(format!("{label} could not be downloaded."));
                continue;
            }
        };
        if prepared.files.len() >= crate::session::MAX_ATTACHED_FILES {
            prepared
                .notes
                .push("Too many attachments; the rest were ignored.".into());
            break;
        }
        match save_download(&directory, &download, prepared.files.len() + 1) {
            Ok(path) => prepared.files.push(path),
            Err(error) => {
                tracing::warn!("gateway could not save an attachment: {error}");
                prepared.notes.push(format!("{label} could not be saved."));
                continue;
            }
        }
        if download.kind == MediaKind::Image {
            if prepared.images.len() >= crate::session::MAX_USER_IMAGES {
                prepared.notes.push(format!(
                    "Only the first {} pictures are shown to the model; {label} is saved as a file.",
                    crate::session::MAX_USER_IMAGES
                ));
                continue;
            }
            match image_to_png_base64(&download.bytes) {
                Ok(png) => prepared.images.push(png),
                Err(error) => prepared.notes.push(format!(
                    "{label} could not be read as a picture ({error}); it is saved as a file."
                )),
            }
        }
    }
    prepared
}

fn save_download(directory: &Path, download: &Download, index: usize) -> Result<PathBuf> {
    std::fs::create_dir_all(directory)?;
    let name = file_name_for(download, index);
    let mut path = directory.join(&name);
    // Two attachments may share a name ("image.jpg"): keep both.
    let mut copy = 1;
    while path.exists() {
        copy += 1;
        path = directory.join(format!("{copy}-{name}"));
    }
    std::fs::write(&path, &download.bytes)?;
    Ok(path)
}

/// A harmless file name: no directories, no control characters, bounded
/// length, and an extension that matches the content type when the sender
/// gave none.
fn file_name_for(download: &Download, index: usize) -> String {
    let given = download
        .name
        .as_deref()
        .map(safe_component)
        .filter(|name| !name.is_empty() && name != "_");
    let extension = given
        .as_deref()
        .and_then(|name| Path::new(name).extension())
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase);
    match given {
        Some(name) if extension.is_some() => name,
        other => {
            let stem = other.unwrap_or_else(|| match download.kind {
                MediaKind::Image => format!("picture-{index}"),
                MediaKind::Audio => format!("audio-{index}"),
                MediaKind::Video => format!("video-{index}"),
                MediaKind::Document | MediaKind::Sticker => format!("file-{index}"),
            });
            match extension_for(download.mime.as_deref(), &download.bytes) {
                Some(extension) => format!("{stem}.{extension}"),
                None => stem,
            }
        }
    }
}

fn extension_for(mime: Option<&str>, bytes: &[u8]) -> Option<&'static str> {
    if let Ok(format) = image::guess_format(bytes) {
        return Some(match format {
            ImageFormat::Png => "png",
            ImageFormat::Jpeg => "jpg",
            ImageFormat::Gif => "gif",
            ImageFormat::WebP => "webp",
            ImageFormat::Bmp => "bmp",
            _ => return None,
        });
    }
    match mime?.split(';').next()?.trim() {
        "application/pdf" => Some("pdf"),
        "text/plain" => Some("txt"),
        "audio/ogg" | "audio/opus" => Some("ogg"),
        "audio/mpeg" => Some("mp3"),
        "audio/mp4" | "audio/aac" => Some("m4a"),
        "video/mp4" => Some("mp4"),
        _ => None,
    }
}

/// One path component made safe: separators, control characters and leading
/// dots become `_`, and the length is capped.
pub fn safe_component(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| {
            if c.is_control() || matches!(c, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|') {
                '_'
            } else {
                c
            }
        })
        .collect();
    let cleaned = cleaned.trim().trim_start_matches('.');
    let mut kept: String = cleaned.chars().take(80).collect();
    if kept.is_empty() {
        kept.push('_');
    }
    kept
}

/// Decode any supported picture, apply its EXIF rotation, scale it to
/// [`MAX_IMAGE_EDGE`], and encode it as base64 PNG.
pub fn image_to_png_base64(bytes: &[u8]) -> Result<String> {
    let reader = image::io::Reader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|error| PokError::Tool(error.to_string()))?;
    let (width, height) = reader
        .into_dimensions()
        .map_err(|_| PokError::Tool("not a picture this build can read".into()))?;
    if u64::from(width) * u64::from(height) > MAX_PIXELS {
        return Err(PokError::Tool("the picture is too large".into()));
    }
    let mut picture = image::load_from_memory(bytes)
        .map_err(|_| PokError::Tool("the picture could not be decoded".into()))?;
    picture = apply_orientation(picture, jpeg_orientation(bytes));
    if picture.width().max(picture.height()) > MAX_IMAGE_EDGE {
        picture = picture.resize(MAX_IMAGE_EDGE, MAX_IMAGE_EDGE, FilterType::Triangle);
    }
    let mut png = Cursor::new(Vec::new());
    picture
        .write_to(&mut png, ImageFormat::Png)
        .map_err(|error| PokError::Tool(error.to_string()))?;
    let png = png.into_inner();
    if png.len() > crate::session::MAX_USER_IMAGE_BYTES {
        return Err(PokError::Tool("the picture is still over 10 MB".into()));
    }
    Ok(base64::engine::general_purpose::STANDARD.encode(png))
}

fn apply_orientation(picture: DynamicImage, orientation: u16) -> DynamicImage {
    match orientation {
        2 => picture.fliph(),
        3 => picture.rotate180(),
        4 => picture.flipv(),
        5 => picture.rotate90().fliph(),
        6 => picture.rotate90(),
        7 => picture.rotate270().fliph(),
        8 => picture.rotate270(),
        _ => picture,
    }
}

/// The EXIF orientation (1-8) of a JPEG; 1 (upright) when there is none.
fn jpeg_orientation(bytes: &[u8]) -> u16 {
    if !bytes.starts_with(&[0xFF, 0xD8]) {
        return 1;
    }
    let mut at = 2;
    while at + 4 <= bytes.len() {
        if bytes[at] != 0xFF {
            return 1;
        }
        let marker = bytes[at + 1];
        match marker {
            0xFF => {
                at += 1;
                continue;
            }
            // Markers without a length.
            0x01 | 0xD0..=0xD8 => {
                at += 2;
                continue;
            }
            // Start of scan or end of image: no more headers.
            0xD9 | 0xDA => return 1,
            _ => {}
        }
        let length = usize::from(u16::from_be_bytes([bytes[at + 2], bytes[at + 3]]));
        if length < 2 || at + 2 + length > bytes.len() {
            return 1;
        }
        if marker == 0xE1 {
            let segment = &bytes[at + 4..at + 2 + length];
            if let Some(tiff) = segment.strip_prefix(b"Exif\0\0") {
                return tiff_orientation(tiff).unwrap_or(1);
            }
        }
        at += 2 + length;
    }
    1
}

fn tiff_orientation(tiff: &[u8]) -> Option<u16> {
    let little = match tiff.get(0..2)? {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    let short = |at: usize| -> Option<u16> {
        let pair = [*tiff.get(at)?, *tiff.get(at + 1)?];
        Some(if little {
            u16::from_le_bytes(pair)
        } else {
            u16::from_be_bytes(pair)
        })
    };
    let long = |at: usize| -> Option<u32> {
        let quad = [
            *tiff.get(at)?,
            *tiff.get(at + 1)?,
            *tiff.get(at + 2)?,
            *tiff.get(at + 3)?,
        ];
        Some(if little {
            u32::from_le_bytes(quad)
        } else {
            u32::from_be_bytes(quad)
        })
    };
    if short(2)? != 42 {
        return None;
    }
    let directory = usize::try_from(long(4)?).ok()?;
    let entries = usize::from(short(directory)?);
    (0..entries.min(64))
        .map(|entry| directory + 2 + entry * 12)
        .find(|at| short(*at) == Some(0x0112))
        .and_then(|at| short(at + 8))
        .filter(|value| (1..=8).contains(value))
}

/// The request the agent receives: the sender's words (or a default question
/// when they sent only a picture or file), then notes about what was skipped.
pub fn compose_prompt(text: &str, prepared: &Prepared) -> String {
    let text = text.trim();
    let mut prompt = if !text.is_empty() {
        text.to_owned()
    } else if prepared.images.len() > 1 {
        "What is in these pictures?".to_owned()
    } else if !prepared.images.is_empty() {
        "What is in this picture?".to_owned()
    } else if !prepared.files.is_empty() {
        "Look at this and tell me what it contains.".to_owned()
    } else {
        String::new()
    };
    if !prompt.is_empty() && !prepared.notes.is_empty() {
        prompt.push_str("\n\n[Attachment notes: ");
        prompt.push_str(&prepared.notes.join(" "));
        prompt.push(']');
    }
    prompt
}

/// Delete saved attachments older than `max_age`. Failures are ignored: a
/// file in use will go on the next sweep.
pub fn sweep_inbox(inbox: &Path, max_age: Duration) {
    let Ok(entries) = std::fs::read_dir(inbox) else {
        return;
    };
    let now = SystemTime::now();
    for entry in entries.flatten() {
        let old = entry
            .metadata()
            .and_then(|metadata| metadata.modified())
            .ok()
            .and_then(|modified| now.duration_since(modified).ok())
            .is_some_and(|age| age > max_age);
        if old && entry.path().is_dir() {
            let _ = std::fs::remove_dir_all(entry.path());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn picture(width: u32, height: u32, format: ImageFormat) -> Vec<u8> {
        let canvas = image::RgbImage::from_fn(width, height, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 90])
        });
        let mut out = Cursor::new(Vec::new());
        DynamicImage::ImageRgb8(canvas)
            .write_to(&mut out, format)
            .unwrap();
        out.into_inner()
    }

    /// Put an EXIF APP1 segment with one orientation entry after the JPEG's SOI.
    fn with_exif_orientation(jpeg: &[u8], orientation: u16, little: bool) -> Vec<u8> {
        let short = |value: u16| {
            if little {
                value.to_le_bytes()
            } else {
                value.to_be_bytes()
            }
        };
        let mut tiff = Vec::new();
        tiff.extend_from_slice(if little { b"II" } else { b"MM" });
        tiff.extend_from_slice(&short(42));
        tiff.extend_from_slice(&if little {
            8_u32.to_le_bytes()
        } else {
            8_u32.to_be_bytes()
        });
        tiff.extend_from_slice(&short(1)); // one entry
        tiff.extend_from_slice(&short(0x0112));
        tiff.extend_from_slice(&short(3)); // SHORT
        tiff.extend_from_slice(&if little {
            1_u32.to_le_bytes()
        } else {
            1_u32.to_be_bytes()
        });
        tiff.extend_from_slice(&short(orientation));
        tiff.extend_from_slice(&[0, 0]);
        tiff.extend_from_slice(&[0, 0, 0, 0]); // no next directory
        let mut segment = b"Exif\0\0".to_vec();
        segment.extend_from_slice(&tiff);
        let length = u16::try_from(segment.len() + 2).unwrap();
        let mut out = vec![0xFF, 0xD8, 0xFF, 0xE1];
        out.extend_from_slice(&length.to_be_bytes());
        out.extend_from_slice(&segment);
        out.extend_from_slice(&jpeg[2..]);
        out
    }

    fn decoded(base64_png: &str) -> DynamicImage {
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(base64_png)
            .unwrap();
        assert!(bytes.starts_with(b"\x89PNG\r\n\x1a\n"));
        image::load_from_memory(&bytes).unwrap()
    }

    #[test]
    fn phone_formats_become_png_the_session_accepts() {
        for format in [
            ImageFormat::Jpeg,
            ImageFormat::Png,
            ImageFormat::Gif,
            ImageFormat::Bmp,
        ] {
            let png = image_to_png_base64(&picture(40, 30, format)).unwrap();
            let image = decoded(&png);
            assert_eq!((image.width(), image.height()), (40, 30), "{format:?}");
        }
    }

    #[test]
    fn big_pictures_are_scaled_to_the_dashboard_edge() {
        let png = image_to_png_base64(&picture(3200, 1600, ImageFormat::Jpeg)).unwrap();
        let image = decoded(&png);
        assert_eq!(image.width(), MAX_IMAGE_EDGE);
        assert_eq!(image.height(), MAX_IMAGE_EDGE / 2);
    }

    #[test]
    fn exif_rotation_turns_a_sideways_phone_photo_upright() {
        let upright = picture(60, 20, ImageFormat::Jpeg);
        for little in [true, false] {
            // Orientation 6 means "rotate 90 degrees clockwise to display".
            let rotated = image_to_png_base64(&with_exif_orientation(&upright, 6, little)).unwrap();
            let image = decoded(&rotated);
            assert_eq!((image.width(), image.height()), (20, 60), "little={little}");
        }
        let kept = image_to_png_base64(&with_exif_orientation(&upright, 1, true)).unwrap();
        assert_eq!(decoded(&kept).width(), 60);
        assert_eq!(jpeg_orientation(&upright), 1, "no EXIF means upright");
        assert_eq!(jpeg_orientation(b"not a jpeg"), 1);
    }

    #[test]
    fn broken_or_oversized_pictures_are_refused_not_decoded() {
        assert!(image_to_png_base64(b"definitely not a picture").is_err());
        let mut truncated = picture(100, 100, ImageFormat::Png);
        truncated.truncate(40);
        assert!(image_to_png_base64(&truncated).is_err());
    }

    #[test]
    fn file_names_cannot_escape_the_inbox() {
        assert_eq!(safe_component("../../etc/passwd"), "_.._etc_passwd");
        assert_eq!(
            safe_component("C:\\Windows\\system32\\cmd.exe"),
            "C__Windows_system32_cmd.exe"
        );
        assert_eq!(safe_component(".hidden"), "hidden");
        assert_eq!(safe_component(""), "_");
        assert_eq!(safe_component(&"a".repeat(500)).chars().count(), 80);
        assert!(!safe_component("bell\u{7}name").contains('\u{7}'));
    }

    #[test]
    fn unnamed_attachments_get_an_extension_from_their_content() {
        let jpeg = Download {
            kind: MediaKind::Image,
            name: None,
            mime: None,
            bytes: picture(8, 8, ImageFormat::Jpeg),
        };
        assert_eq!(file_name_for(&jpeg, 1), "picture-1.jpg");
        let pdf = Download {
            kind: MediaKind::Document,
            name: Some("report".into()),
            mime: Some("application/pdf".into()),
            bytes: b"%PDF-1.7".to_vec(),
        };
        assert_eq!(file_name_for(&pdf, 2), "report.pdf");
        let named = Download {
            name: Some("notes.TXT".into()),
            ..pdf
        };
        assert_eq!(file_name_for(&named, 3), "notes.TXT");
    }

    struct Fake {
        kind: MediaKind,
        name: &'static str,
        bytes: Vec<u8>,
        announced: Option<u64>,
        fail: bool,
    }

    #[async_trait]
    impl MediaSource for Fake {
        fn kind(&self) -> MediaKind {
            self.kind
        }
        fn label(&self) -> String {
            self.name.to_owned()
        }
        fn size_hint(&self) -> Option<u64> {
            self.announced
        }
        async fn fetch(&self, _max_bytes: u64) -> Result<Download> {
            if self.fail {
                return Err(PokError::Provider("offline".into()));
            }
            Ok(Download {
                kind: self.kind,
                name: Some(self.name.to_owned()),
                mime: None,
                bytes: self.bytes.clone(),
            })
        }
    }

    fn source(kind: MediaKind, name: &'static str, bytes: Vec<u8>) -> Arc<dyn MediaSource> {
        Arc::new(Fake {
            kind,
            name,
            bytes,
            announced: None,
            fail: false,
        })
    }

    #[tokio::test]
    async fn a_mixed_message_is_saved_and_prepared() {
        let inbox = tempfile::tempdir().unwrap();
        let sources = vec![
            source(
                MediaKind::Image,
                "photo.jpg",
                picture(64, 48, ImageFormat::Jpeg),
            ),
            source(MediaKind::Document, "budget.csv", b"a,b\n1,2\n".to_vec()),
            source(MediaKind::Audio, "voice.ogg", vec![1, 2, 3]),
            source(MediaKind::Sticker, "sticker.webp", vec![9]),
            Arc::new(Fake {
                kind: MediaKind::Document,
                name: "huge.bin",
                bytes: Vec::new(),
                announced: Some(MAX_DOWNLOAD_BYTES + 1),
                fail: false,
            }),
            Arc::new(Fake {
                kind: MediaKind::Document,
                name: "gone.pdf",
                bytes: Vec::new(),
                announced: None,
                fail: true,
            }),
        ];
        let prepared = prepare(inbox.path(), "task-1", &sources).await;

        assert_eq!(prepared.images.len(), 1);
        decoded(&prepared.images[0]);
        // Picture, spreadsheet and voice note are saved; the sticker, the
        // oversized file and the failed download are not.
        let names: Vec<_> = prepared
            .files
            .iter()
            .map(|path| path.file_name().unwrap().to_str().unwrap().to_owned())
            .collect();
        assert_eq!(names, ["photo.jpg", "budget.csv", "voice.ogg"]);
        assert!(
            prepared
                .files
                .iter()
                .all(|path| path.starts_with(inbox.path()))
        );
        assert!(prepared.notes.iter().any(|note| note.contains("voice.ogg")));
        assert!(
            prepared
                .notes
                .iter()
                .any(|note| note.contains("huge.bin") && note.contains("20 MB"))
        );
        assert!(
            prepared
                .notes
                .iter()
                .any(|note| note.contains("gone.pdf") && note.contains("could not be downloaded"))
        );
        assert_eq!(
            std::fs::read(&prepared.files[1]).unwrap(),
            b"a,b\n1,2\n",
            "files are saved byte for byte"
        );
    }

    #[tokio::test]
    async fn a_picture_that_will_not_decode_is_still_kept_as_a_file() {
        let inbox = tempfile::tempdir().unwrap();
        let sources = vec![source(MediaKind::Image, "broken.jpg", b"garbage".to_vec())];
        let prepared = prepare(inbox.path(), "task-2", &sources).await;
        assert!(prepared.images.is_empty());
        assert_eq!(prepared.files.len(), 1);
        assert!(prepared.notes[0].contains("broken.jpg"));
    }

    #[tokio::test]
    async fn only_eight_pictures_reach_the_model() {
        let inbox = tempfile::tempdir().unwrap();
        let sources: Vec<_> = (0..10)
            .map(|_| source(MediaKind::Image, "p.png", picture(8, 8, ImageFormat::Png)))
            .collect();
        let prepared = prepare(inbox.path(), "task-3", &sources).await;
        assert_eq!(prepared.images.len(), crate::session::MAX_USER_IMAGES);
        assert_eq!(
            prepared.files.len(),
            10,
            "all are saved, names kept distinct"
        );
        let unique: std::collections::HashSet<_> = prepared.files.iter().collect();
        assert_eq!(unique.len(), 10);
    }

    #[test]
    fn the_prompt_uses_the_senders_words_or_a_sensible_default() {
        let one = Prepared {
            images: vec!["x".into()],
            ..Prepared::default()
        };
        let two = Prepared {
            images: vec!["x".into(), "y".into()],
            ..Prepared::default()
        };
        let file = Prepared {
            files: vec![PathBuf::from("a.pdf")],
            ..Prepared::default()
        };
        assert_eq!(compose_prompt("  fix this  ", &one), "fix this");
        assert_eq!(compose_prompt("", &one), "What is in this picture?");
        assert_eq!(compose_prompt("", &two), "What is in these pictures?");
        assert_eq!(
            compose_prompt("", &file),
            "Look at this and tell me what it contains."
        );
        assert_eq!(compose_prompt("", &Prepared::default()), "");
        let noted = Prepared {
            notes: vec!["The voice note is only saved.".into()],
            ..one
        };
        assert!(compose_prompt("listen", &noted).ends_with("only saved.]"));
    }

    #[test]
    fn the_sweep_removes_only_old_message_folders() {
        let inbox = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(inbox.path().join("fresh")).unwrap();
        std::fs::write(inbox.path().join("fresh").join("a.txt"), "x").unwrap();
        // Nothing is older than a week, so nothing is removed...
        sweep_inbox(inbox.path(), INBOX_RETENTION);
        assert!(inbox.path().join("fresh").exists());
        // ...and with a zero retention everything is.
        std::thread::sleep(Duration::from_millis(20));
        sweep_inbox(inbox.path(), Duration::ZERO);
        assert!(!inbox.path().join("fresh").exists());
    }
}
