//! Replay a WAV through the live voice pipeline and print its events:
//! `cargo run -p pok-ai-voice --example replay_wav -- <data_dir> <file.wav> [english|multilingual]`
//! Or listen to the microphone for a few seconds: `... -- <data_dir> --mic 8`.

use std::sync::Arc;
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let data_dir = std::path::PathBuf::from(args.next().expect("data dir"));
    let source = args.next().expect("wav file or --mic");
    let third = args.next();
    let mode = if third.as_deref() == Some("multilingual") {
        pok_ai_voice::VoiceMode::Multilingual
    } else {
        pok_ai_voice::VoiceMode::English
    };
    let models = pok_ai_voice::VoiceModels::new(&data_dir);
    let loading = Instant::now();
    let engine = pok_ai_voice::VoiceEngine::load(&models, mode).expect("load models");
    println!("models loaded in {:?}", loading.elapsed());
    let started = Instant::now();
    let sink: pok_ai_voice::EventSink = Arc::new(move |event| {
        if !matches!(event, pok_ai_voice::VoiceEvent::Level { .. }) {
            println!("{:>6} ms  {event:?}", started.elapsed().as_millis());
        }
    });
    if source == "--mic" {
        let seconds: u64 = third.and_then(|value| value.parse().ok()).unwrap_or(6);
        println!("microphones: {:?}", pok_ai_voice::input_devices());
        let session = engine.listen(None, sink).expect("start microphone");
        std::thread::sleep(std::time::Duration::from_secs(seconds));
        session.stop();
        std::thread::sleep(std::time::Duration::from_secs(2));
        return;
    }
    let (rate, samples) = read_wav(&source);
    // A second of silence between two copies: two phrases, one pause.
    let mut audio = samples.clone();
    audio.extend(std::iter::repeat_n(0.0, rate as usize));
    audio.extend(samples);
    println!(
        "{:.1} s of audio at {rate} Hz",
        audio.len() as f32 / rate as f32
    );
    engine.replay(rate, audio, sink).expect("replay");
}

/// 16-bit PCM WAV, mono or stereo.
fn read_wav(path: &str) -> (i32, Vec<f32>) {
    let bytes = std::fs::read(path).expect("read wav");
    let channels = u16::from_le_bytes([bytes[22], bytes[23]]) as usize;
    let rate = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]) as i32;
    let data = bytes
        .windows(4)
        .position(|window| window == b"data")
        .expect("data chunk")
        + 8;
    let samples: Vec<f32> = bytes[data..]
        .chunks_exact(2)
        .map(|pair| f32::from(i16::from_le_bytes([pair[0], pair[1]])) / 32_768.0)
        .collect();
    (rate, pok_ai_voice::to_mono(&samples, channels))
}
