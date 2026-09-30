//! Windows engine: microphone capture (cpal/WASAPI) and sherpa-onnx models.
//!
//! Three threads per session keep the caption live while accurate text is
//! computed: the audio thread only copies samples, the worker runs the
//! streaming model (or VAD) and cuts phrases, and the finisher runs Parakeet
//! on each finished phrase, in order.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use sherpa_onnx::{
    LinearResampler, OfflineRecognizer, OfflineRecognizerConfig, OnlineRecognizer,
    OnlineRecognizerConfig, SileroVadModelConfig, VadModelConfig, VoiceActivityDetector,
};

use crate::{
    EventSink, Result, VoiceError, VoiceEvent, VoiceMode, VoiceModels, caption_case, level, to_mono,
};

/// A phrase longer than this is cut even without a pause, so accurate text
/// keeps arriving during long dictation.
const MAX_PHRASE_SECONDS: f32 = 20.0;
/// Silence kept before a phrase while nothing has been said.
const IDLE_AUDIO_SECONDS: f32 = 1.0;
const VAD_RATE: i32 = 16_000;
const VAD_WINDOW: usize = 512;

pub(crate) fn input_devices() -> Vec<String> {
    let host = cpal::default_host();
    let default = host
        .default_input_device()
        .and_then(|device| device.description().ok())
        .map(|description| description.name().to_owned());
    let mut names: Vec<String> = host
        .input_devices()
        .map(|devices| {
            devices
                .filter_map(|device| device.description().ok())
                .map(|description| description.name().to_owned())
                .collect()
        })
        .unwrap_or_default();
    if let Some(default) = default {
        names.retain(|name| name != &default);
        names.insert(0, default);
    }
    names
}

pub(crate) struct Engine {
    mode: VoiceMode,
    streaming: Option<OnlineRecognizer>,
    accurate: OfflineRecognizer,
    vad_model: String,
}

fn path(path: std::path::PathBuf) -> Option<String> {
    Some(path.display().to_string())
}

impl Engine {
    pub(crate) fn load(models: &VoiceModels, mode: VoiceMode) -> Result<Self> {
        let threads = std::thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(4)
            .clamp(2, 4) as i32;
        let streaming = match mode {
            VoiceMode::English => {
                let folder = models.streaming();
                let mut config = OnlineRecognizerConfig::default();
                let model = &mut config.model_config;
                model.transducer.encoder =
                    path(folder.join("encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx"));
                model.transducer.decoder =
                    path(folder.join("decoder-epoch-99-avg-1-chunk-16-left-128.onnx"));
                model.transducer.joiner =
                    path(folder.join("joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx"));
                model.tokens = path(folder.join("tokens.txt"));
                model.num_threads = 2;
                config.decoding_method = Some("greedy_search".into());
                // A pause of about 0.6 s after speech ends a phrase.
                config.enable_endpoint = true;
                config.rule1_min_trailing_silence = 2.4;
                config.rule2_min_trailing_silence = 0.6;
                config.rule3_min_utterance_length = MAX_PHRASE_SECONDS;
                Some(
                    OnlineRecognizer::create(&config)
                        .ok_or_else(|| VoiceError::Model("live caption model".into()))?,
                )
            }
            VoiceMode::Multilingual => None,
        };
        let folder = models.accurate();
        let mut config = OfflineRecognizerConfig::default();
        let model = &mut config.model_config;
        model.transducer.encoder = path(folder.join("encoder.int8.onnx"));
        model.transducer.decoder = path(folder.join("decoder.int8.onnx"));
        model.transducer.joiner = path(folder.join("joiner.int8.onnx"));
        model.tokens = path(folder.join("tokens.txt"));
        model.model_type = Some("nemo_transducer".into());
        model.num_threads = threads;
        let accurate = OfflineRecognizer::create(&config)
            .ok_or_else(|| VoiceError::Model("Parakeet".into()))?;
        Ok(Self {
            mode,
            streaming,
            accurate,
            vad_model: models.vad().display().to_string(),
        })
    }

    fn transcribe(&self, rate: i32, samples: &[f32]) -> String {
        let stream = self.accurate.create_stream();
        stream.accept_waveform(rate, samples);
        self.accurate.decode(&stream);
        stream
            .get_result()
            .map(|result| result.text.trim().to_owned())
            .unwrap_or_default()
    }
}

enum Audio {
    Samples(Vec<f32>),
    Failed(String),
}

struct Phrase {
    segment: u64,
    rate: i32,
    samples: Vec<f32>,
}

pub(crate) struct Session {
    stop: Arc<AtomicBool>,
    audio: Option<JoinHandle<()>>,
}

impl Session {
    pub(crate) fn start(
        engine: Arc<Engine>,
        device: Option<&str>,
        sink: EventSink,
    ) -> Result<Self> {
        let stop = Arc::new(AtomicBool::new(false));
        let (audio_tx, audio_rx) = mpsc::channel::<Audio>();
        let (ready_tx, ready_rx) = mpsc::sync_channel::<Result<i32>>(1);
        let device = device.map(str::to_owned);
        let audio_stop = stop.clone();
        // cpal streams are kept on the thread that built them.
        let audio = std::thread::Builder::new()
            .name("voice-audio".into())
            .spawn(move || capture(device.as_deref(), audio_tx, audio_stop, ready_tx))
            .map_err(|error| VoiceError::Audio(error.to_string()))?;
        let rate = match ready_rx.recv_timeout(Duration::from_secs(10)) {
            Ok(Ok(rate)) => rate,
            Ok(Err(error)) => {
                let _ = audio.join();
                return Err(error);
            }
            Err(_) => {
                stop.store(true, Ordering::SeqCst);
                return Err(VoiceError::Audio("the microphone did not start".into()));
            }
        };
        let (phrase_tx, phrase_rx) = mpsc::channel::<Phrase>();
        let finisher_engine = engine.clone();
        let finisher_sink = sink.clone();
        let finisher = std::thread::Builder::new()
            .name("voice-finish".into())
            .spawn(move || {
                for phrase in phrase_rx {
                    let text = finisher_engine.transcribe(phrase.rate, &phrase.samples);
                    finisher_sink(VoiceEvent::Final {
                        segment: phrase.segment,
                        text,
                    });
                }
            })
            .map_err(|error| VoiceError::Audio(error.to_string()))?;
        std::thread::Builder::new()
            .name("voice-listen".into())
            .spawn(move || {
                let error = match engine.mode {
                    VoiceMode::English => {
                        listen_streaming(&engine, rate, audio_rx, &phrase_tx, &sink)
                    }
                    VoiceMode::Multilingual => {
                        listen_vad(&engine, rate, audio_rx, &phrase_tx, &sink)
                    }
                };
                // The finisher drains every phrase before listening is over.
                drop(phrase_tx);
                let _ = finisher.join();
                sink(VoiceEvent::Stopped { error });
            })
            .map_err(|error| VoiceError::Audio(error.to_string()))?;
        Ok(Self {
            stop,
            audio: Some(audio),
        })
    }

    /// Run `samples` through the same pipeline as the microphone, in 100 ms
    /// chunks, and wait for it to finish (tests and the replay example).
    pub(crate) fn replay(engine: Arc<Engine>, rate: i32, samples: Vec<f32>, sink: EventSink) {
        let (audio_tx, audio_rx) = mpsc::channel::<Audio>();
        for chunk in samples.chunks((rate / 10).max(1) as usize) {
            let _ = audio_tx.send(Audio::Samples(chunk.to_vec()));
        }
        drop(audio_tx);
        let (phrase_tx, phrase_rx) = mpsc::channel::<Phrase>();
        let finisher_engine = engine.clone();
        let finisher_sink = sink.clone();
        let finisher = std::thread::spawn(move || {
            for phrase in phrase_rx {
                let text = finisher_engine.transcribe(phrase.rate, &phrase.samples);
                finisher_sink(VoiceEvent::Final {
                    segment: phrase.segment,
                    text,
                });
            }
        });
        let error = match engine.mode {
            VoiceMode::English => listen_streaming(&engine, rate, audio_rx, &phrase_tx, &sink),
            VoiceMode::Multilingual => listen_vad(&engine, rate, audio_rx, &phrase_tx, &sink),
        };
        drop(phrase_tx);
        let _ = finisher.join();
        sink(VoiceEvent::Stopped { error });
    }

    pub(crate) fn stop(mut self) {
        self.halt();
    }

    fn halt(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(audio) = self.audio.take() {
            let _ = audio.join();
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.halt();
    }
}

/// Open the microphone and forward mono samples until `stop` is set.
fn capture(
    device: Option<&str>,
    tx: mpsc::Sender<Audio>,
    stop: Arc<AtomicBool>,
    ready: mpsc::SyncSender<Result<i32>>,
) {
    let host = cpal::default_host();
    let chosen = match device {
        Some(name) => host.input_devices().ok().and_then(|mut devices| {
            devices.find(|device| {
                device
                    .description()
                    .map(|description| description.name() == name)
                    .unwrap_or(false)
            })
        }),
        None => None,
    }
    .or_else(|| host.default_input_device());
    let Some(device) = chosen else {
        let _ = ready.send(Err(VoiceError::NoMicrophone));
        return;
    };
    let supported = match device.default_input_config() {
        Ok(config) => config,
        Err(error) => {
            let _ = ready.send(Err(VoiceError::Audio(error.to_string())));
            return;
        }
    };
    let channels = usize::from(supported.channels());
    let rate = supported.sample_rate() as i32;
    let format = supported.sample_format();
    let config = supported.config();
    let data_tx = tx.clone();
    let error_tx = tx;
    let stream = device.build_input_stream_raw(
        config,
        format,
        move |data: &cpal::Data, _: &cpal::InputCallbackInfo| {
            let samples: Vec<f32> = if let Some(values) = data.as_slice::<f32>() {
                values.to_vec()
            } else if let Some(values) = data.as_slice::<i16>() {
                values
                    .iter()
                    .map(|value| f32::from(*value) / 32_768.0)
                    .collect()
            } else if let Some(values) = data.as_slice::<i32>() {
                values
                    .iter()
                    .map(|value| *value as f32 / 2_147_483_648.0)
                    .collect()
            } else if let Some(values) = data.as_slice::<u16>() {
                values
                    .iter()
                    .map(|value| (f32::from(*value) - 32_768.0) / 32_768.0)
                    .collect()
            } else {
                return;
            };
            let _ = data_tx.send(Audio::Samples(to_mono(&samples, channels)));
        },
        move |error| {
            // These notifications leave the stream active. WASAPI can report
            // an Xrun after a brief discontinuity and then deliver more data.
            if matches!(
                error.kind(),
                cpal::ErrorKind::Xrun | cpal::ErrorKind::DeviceChanged
            ) {
                return;
            }
            let _ = error_tx.send(Audio::Failed(error.to_string()));
        },
        None,
    );
    let stream = match stream {
        Ok(stream) => stream,
        Err(error) => {
            let _ = ready.send(Err(VoiceError::Audio(error.to_string())));
            return;
        }
    };
    if let Err(error) = stream.play() {
        let _ = ready.send(Err(VoiceError::Audio(error.to_string())));
        return;
    }
    let _ = ready.send(Ok(rate));
    while !stop.load(Ordering::SeqCst) {
        std::thread::sleep(Duration::from_millis(30));
    }
    drop(stream);
}

struct LevelMeter {
    last: Instant,
}

impl LevelMeter {
    fn report(&mut self, samples: &[f32], sink: &EventSink) {
        if self.last.elapsed() >= Duration::from_millis(100) {
            self.last = Instant::now();
            sink(VoiceEvent::Level {
                level: level(samples),
            });
        }
    }
}

/// English: the streaming model captions; each endpoint sends the phrase's
/// audio to the finisher. Returns an error message when audio failed.
fn listen_streaming(
    engine: &Engine,
    rate: i32,
    audio: mpsc::Receiver<Audio>,
    phrases: &mpsc::Sender<Phrase>,
    sink: &EventSink,
) -> Option<String> {
    let Some(recognizer) = engine.streaming.as_ref() else {
        return Some("the live caption model is not loaded".into());
    };
    let stream = recognizer.create_stream();
    let mut segment = 0_u64;
    let mut buffer: Vec<f32> = Vec::new();
    let mut caption = String::new();
    let mut meter = LevelMeter {
        last: Instant::now(),
    };
    let idle_keep = (IDLE_AUDIO_SECONDS * rate as f32) as usize;
    let mut failure = None;
    let cut = |buffer: &mut Vec<f32>, caption: &mut String, segment: &mut u64| {
        if !caption.is_empty() {
            let _ = phrases.send(Phrase {
                segment: *segment,
                rate,
                samples: std::mem::take(buffer),
            });
            *segment += 1;
        }
        buffer.clear();
        caption.clear();
    };
    for message in audio {
        let samples = match message {
            Audio::Samples(samples) => samples,
            Audio::Failed(error) => {
                failure = Some(error);
                break;
            }
        };
        meter.report(&samples, sink);
        buffer.extend_from_slice(&samples);
        stream.accept_waveform(rate, &samples);
        while recognizer.is_ready(&stream) {
            recognizer.decode(&stream);
        }
        let text = recognizer
            .get_result(&stream)
            .map(|result| caption_case(&result.text))
            .unwrap_or_default();
        if text != caption {
            caption = text;
            if !caption.is_empty() {
                sink(VoiceEvent::Partial {
                    segment,
                    text: caption.clone(),
                });
            }
        }
        if caption.is_empty() && buffer.len() > idle_keep * 3 {
            // Nothing said yet: keep only the latest second as lead-in.
            let excess = buffer.len() - idle_keep;
            buffer.drain(..excess);
        }
        if recognizer.is_endpoint(&stream) {
            cut(&mut buffer, &mut caption, &mut segment);
            recognizer.reset(&stream);
        }
    }
    // Stopped: finish the phrase in progress.
    stream.input_finished();
    while recognizer.is_ready(&stream) {
        recognizer.decode(&stream);
    }
    if let Some(result) = recognizer.get_result(&stream) {
        caption = caption_case(&result.text);
    }
    cut(&mut buffer, &mut caption, &mut segment);
    failure
}

/// Multilingual: voice activity detection cuts phrases for Parakeet v3.
fn listen_vad(
    engine: &Engine,
    rate: i32,
    audio: mpsc::Receiver<Audio>,
    phrases: &mpsc::Sender<Phrase>,
    sink: &EventSink,
) -> Option<String> {
    let config = VadModelConfig {
        silero_vad: SileroVadModelConfig {
            model: Some(engine.vad_model.clone()),
            threshold: 0.5,
            min_silence_duration: 0.5,
            min_speech_duration: 0.25,
            window_size: VAD_WINDOW as i32,
            max_speech_duration: MAX_PHRASE_SECONDS,
        },
        sample_rate: VAD_RATE,
        num_threads: 1,
        ..Default::default()
    };
    let Some(vad) = VoiceActivityDetector::create(&config, 60.0) else {
        return Some("voice activity detection could not start".into());
    };
    let resampler = (rate != VAD_RATE)
        .then(|| LinearResampler::create(rate, VAD_RATE))
        .flatten();
    let mut pending: Vec<f32> = Vec::new();
    let mut segment = 0_u64;
    let mut speaking = false;
    let mut meter = LevelMeter {
        last: Instant::now(),
    };
    let mut failure = None;
    let drain = |segment: &mut u64| {
        while !vad.is_empty() {
            if let Some(front) = vad.front() {
                let _ = phrases.send(Phrase {
                    segment: *segment,
                    rate: VAD_RATE,
                    samples: front.samples().to_vec(),
                });
                *segment += 1;
            }
            vad.pop();
        }
    };
    for message in audio {
        let samples = match message {
            Audio::Samples(samples) => samples,
            Audio::Failed(error) => {
                failure = Some(error);
                break;
            }
        };
        meter.report(&samples, sink);
        match &resampler {
            Some(resampler) => pending.extend(resampler.resample(&samples, false)),
            None => pending.extend_from_slice(&samples),
        }
        while pending.len() >= VAD_WINDOW {
            let window: Vec<f32> = pending.drain(..VAD_WINDOW).collect();
            vad.accept_waveform(&window);
        }
        let detected = vad.detected();
        if detected && !speaking {
            sink(VoiceEvent::Partial {
                segment,
                text: "…".into(),
            });
        }
        speaking = detected;
        drain(&mut segment);
    }
    vad.flush();
    drain(&mut segment);
    failure
}
