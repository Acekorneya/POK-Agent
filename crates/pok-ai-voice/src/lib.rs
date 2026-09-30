//! Offline voice input: live captions from a small streaming model, each
//! phrase refined by an accurate model at the next pause. Audio stays on the
//! device and is never stored.
//!
//! Two modes share one pipeline:
//!
//! - **English (live)**: a streaming Zipformer shows words while the user
//!   speaks; at each pause (endpoint) Parakeet re-reads the phrase and its
//!   text, with punctuation and casing, replaces the live caption.
//! - **Multilingual**: Silero VAD finds each phrase and Parakeet v3
//!   transcribes it (25 European languages); there is no live caption.
//!
//! The native engine exists on Windows only; elsewhere [`VoiceEngine::load`]
//! reports that voice input is unsupported.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

#[cfg(windows)]
mod native;

#[derive(Debug, thiserror::Error)]
pub enum VoiceError {
    #[error("voice models are not installed")]
    NotInstalled,
    #[error("voice input is only supported on Windows")]
    Unsupported,
    #[error("no microphone was found")]
    NoMicrophone,
    #[error("{0}")]
    Audio(String),
    #[error("could not load the speech model: {0}")]
    Model(String),
}

pub type Result<T> = std::result::Result<T, VoiceError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum VoiceMode {
    /// Live captions (streaming English) refined by Parakeet at each pause.
    #[default]
    English,
    /// Parakeet v3 per phrase, found by voice activity detection.
    Multilingual,
}

/// What the dashboard receives while listening.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum VoiceEvent {
    /// Microphone level, 0.0 to 1.0, about ten times a second.
    Level { level: f32 },
    /// The live caption for the phrase being spoken; replaced by `Final`.
    Partial { segment: u64, text: String },
    /// The accurate text for a finished phrase (may be empty for noise).
    Final { segment: u64, text: String },
    /// Listening ended; `error` says why when it was not requested.
    Stopped { error: Option<String> },
}

pub type EventSink = Arc<dyn Fn(VoiceEvent) + Send + Sync>;

/// One downloadable model: an archive (or single file) from the
/// sherpa-onnx release, and the files kept after extraction.
#[derive(Debug, Clone, Copy)]
pub struct ModelDownload {
    pub name: &'static str,
    pub url: &'static str,
    /// Folder (or file) name under the voice directory once installed.
    pub installed_as: &'static str,
    /// Files needed at run time; everything else in the folder is removed.
    pub keep: &'static [&'static str],
    pub approximate_mb: u32,
}

/// Models come from the sherpa-onnx release on GitHub.
macro_rules! release_url {
    ($file:literal) => {
        concat!(
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/",
            $file
        )
    };
}

const STREAMING_DIR: &str = "sherpa-onnx-streaming-zipformer-en-2023-06-26";
const PARAKEET_DIR: &str = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8";
const VAD_FILE: &str = "silero_vad.onnx";

pub const STREAMING_MODEL: ModelDownload = ModelDownload {
    name: "Live captions (streaming Zipformer, English)",
    url: release_url!("sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2"),
    installed_as: STREAMING_DIR,
    keep: &[
        "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
        "decoder-epoch-99-avg-1-chunk-16-left-128.onnx",
        "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
        "tokens.txt",
    ],
    approximate_mb: 296,
};

pub const ACCURATE_MODEL: ModelDownload = ModelDownload {
    name: "Accurate text (NVIDIA Parakeet TDT 0.6B v3, 25 languages, CC-BY-4.0)",
    url: release_url!("sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2"),
    installed_as: PARAKEET_DIR,
    keep: &[
        "encoder.int8.onnx",
        "decoder.int8.onnx",
        "joiner.int8.onnx",
        "tokens.txt",
    ],
    approximate_mb: 464,
};

pub const VAD_MODEL: ModelDownload = ModelDownload {
    name: "Voice activity detection (Silero VAD)",
    url: release_url!("silero_vad.onnx"),
    installed_as: VAD_FILE,
    keep: &[],
    approximate_mb: 1,
};

/// The models each mode needs.
pub fn downloads(mode: VoiceMode) -> &'static [ModelDownload] {
    match mode {
        VoiceMode::English => &[STREAMING_MODEL, ACCURATE_MODEL],
        VoiceMode::Multilingual => &[ACCURATE_MODEL, VAD_MODEL],
    }
}

/// Where the voice models live: `<data_dir>/voice`.
#[derive(Debug, Clone)]
pub struct VoiceModels {
    root: PathBuf,
}

impl VoiceModels {
    pub fn new(data_dir: &Path) -> Self {
        Self {
            root: data_dir.join("voice"),
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn is_installed(&self, download: &ModelDownload) -> bool {
        let path = self.root.join(download.installed_as);
        if download.keep.is_empty() {
            path.is_file()
        } else {
            download.keep.iter().all(|file| path.join(file).is_file())
        }
    }

    pub fn ready(&self, mode: VoiceMode) -> bool {
        downloads(mode)
            .iter()
            .all(|download| self.is_installed(download))
    }

    /// Remove files an extracted archive brought that are not needed at run
    /// time (full-precision copies, test recordings, scripts).
    pub fn prune(&self, download: &ModelDownload) -> std::io::Result<()> {
        if download.keep.is_empty() {
            return Ok(());
        }
        let folder = self.root.join(download.installed_as);
        for entry in std::fs::read_dir(&folder)? {
            let entry = entry?;
            let name = entry.file_name();
            if download
                .keep
                .iter()
                .any(|keep| name.to_str() == Some(*keep))
            {
                continue;
            }
            let path = entry.path();
            if path.is_dir() {
                std::fs::remove_dir_all(path)?;
            } else {
                std::fs::remove_file(path)?;
            }
        }
        Ok(())
    }

    #[cfg(windows)]
    fn streaming(&self) -> PathBuf {
        self.root.join(STREAMING_DIR)
    }

    #[cfg(windows)]
    fn accurate(&self) -> PathBuf {
        self.root.join(PARAKEET_DIR)
    }

    #[cfg(windows)]
    fn vad(&self) -> PathBuf {
        self.root.join(VAD_FILE)
    }
}

/// Microphones the system reports, by name; the first is the default.
pub fn input_devices() -> Vec<String> {
    #[cfg(windows)]
    {
        native::input_devices()
    }
    #[cfg(not(windows))]
    {
        Vec::new()
    }
}

/// Loaded speech models, kept between listening sessions so the microphone
/// button responds at once after the first use.
pub struct VoiceEngine {
    mode: VoiceMode,
    #[cfg(windows)]
    inner: Arc<native::Engine>,
}

impl VoiceEngine {
    /// Load the models for `mode` (a few seconds, once).
    pub fn load(models: &VoiceModels, mode: VoiceMode) -> Result<Self> {
        if !models.ready(mode) {
            return Err(VoiceError::NotInstalled);
        }
        #[cfg(windows)]
        {
            Ok(Self {
                mode,
                inner: Arc::new(native::Engine::load(models, mode)?),
            })
        }
        #[cfg(not(windows))]
        {
            let _ = mode;
            Err(VoiceError::Unsupported)
        }
    }

    pub fn mode(&self) -> VoiceMode {
        self.mode
    }

    /// Start listening on `device` (or the default microphone). Events go to
    /// `sink` until the returned session is stopped or dropped.
    pub fn listen(&self, device: Option<&str>, sink: EventSink) -> Result<VoiceSession> {
        #[cfg(windows)]
        {
            Ok(VoiceSession {
                inner: Some(native::Session::start(self.inner.clone(), device, sink)?),
            })
        }
        #[cfg(not(windows))]
        {
            let _ = (device, sink);
            Err(VoiceError::Unsupported)
        }
    }
}

impl VoiceEngine {
    /// Feed recorded audio through the live pipeline instead of the
    /// microphone, returning when every event has been sent.
    #[doc(hidden)]
    pub fn replay(&self, rate: i32, samples: Vec<f32>, sink: EventSink) -> Result<()> {
        #[cfg(windows)]
        {
            native::Session::replay(self.inner.clone(), rate, samples, sink);
            Ok(())
        }
        #[cfg(not(windows))]
        {
            let _ = (rate, samples, sink);
            Err(VoiceError::Unsupported)
        }
    }
}

/// An active microphone session.
pub struct VoiceSession {
    #[cfg(windows)]
    inner: Option<native::Session>,
}

impl VoiceSession {
    /// Stop listening; the phrase in progress is still transcribed and sent
    /// as a final event before `Stopped`.
    pub fn stop(mut self) {
        self.finish();
    }

    fn finish(&mut self) {
        #[cfg(windows)]
        if let Some(session) = self.inner.take() {
            session.stop();
        }
    }
}

impl Drop for VoiceSession {
    fn drop(&mut self) {
        self.finish();
    }
}

/// The streaming model writes upper case without punctuation; show it as a
/// sentence until the accurate text arrives.
pub fn caption_case(text: &str) -> String {
    let lower = text.trim().to_lowercase();
    let mut characters = lower.chars();
    let Some(first) = characters.next() else {
        return String::new();
    };
    let mut caption: String = first.to_uppercase().collect();
    caption.push_str(characters.as_str());
    // A lone "i" is always capitalised in English.
    caption
        .split(' ')
        .map(|word| match word {
            "i" => "I",
            "i'm" => "I'm",
            "i've" => "I've",
            "i'll" => "I'll",
            "i'd" => "I'd",
            other => other,
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Root-mean-square level of `samples`, scaled so normal speech reads about
/// 0.3 to 0.8 and silence near 0.
pub fn level(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let mean_square =
        samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len() as f32;
    (mean_square.sqrt() * 8.0).clamp(0.0, 1.0)
}

/// Down-mix interleaved frames to mono.
pub fn to_mono(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    interleaved
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captions_read_as_a_sentence() {
        assert_eq!(
            caption_case("ASK NOT WHAT YOUR COUNTRY"),
            "Ask not what your country"
        );
        assert_eq!(caption_case("  I THINK I'M READY "), "I think I'm ready");
        assert_eq!(caption_case(""), "");
    }

    #[test]
    fn levels_and_mono_mix() {
        assert_eq!(level(&[]), 0.0);
        assert_eq!(level(&[0.0; 160]), 0.0);
        assert!(level(&[0.1; 160]) > 0.5);
        assert_eq!(to_mono(&[1.0, 0.0, 0.5, 0.5], 2), vec![0.5, 0.5]);
        assert_eq!(to_mono(&[0.2, 0.4], 1), vec![0.2, 0.4]);
    }

    #[test]
    fn a_mode_is_ready_only_with_all_its_models() {
        let dir = tempfile::tempdir().unwrap();
        let models = VoiceModels::new(dir.path());
        assert!(!models.ready(VoiceMode::English));
        for download in downloads(VoiceMode::English) {
            let folder = models.root().join(download.installed_as);
            std::fs::create_dir_all(&folder).unwrap();
            for file in download.keep {
                std::fs::write(folder.join(file), b"x").unwrap();
            }
        }
        assert!(models.ready(VoiceMode::English));
        // Multilingual also needs the VAD file.
        assert!(!models.ready(VoiceMode::Multilingual));
        std::fs::write(models.root().join(VAD_MODEL.installed_as), b"x").unwrap();
        assert!(models.ready(VoiceMode::Multilingual));
    }

    #[test]
    fn pruning_keeps_only_the_runtime_files() {
        let dir = tempfile::tempdir().unwrap();
        let models = VoiceModels::new(dir.path());
        let folder = models.root().join(ACCURATE_MODEL.installed_as);
        std::fs::create_dir_all(folder.join("test_wavs")).unwrap();
        for file in ACCURATE_MODEL.keep {
            std::fs::write(folder.join(file), b"x").unwrap();
        }
        std::fs::write(folder.join("encoder.onnx"), b"full precision").unwrap();
        models.prune(&ACCURATE_MODEL).unwrap();
        assert!(models.is_installed(&ACCURATE_MODEL));
        assert!(!folder.join("encoder.onnx").exists());
        assert!(!folder.join("test_wavs").exists());
    }

    #[test]
    fn download_urls_point_at_the_sherpa_release() {
        for download in [STREAMING_MODEL, ACCURATE_MODEL, VAD_MODEL] {
            assert!(
                download.url.starts_with(
                    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
                ),
                "{}",
                download.url
            );
        }
    }
}
