import React, { useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { SettingRow, SettingsCard, StatusPill } from "./settings-drawer";

export type VoiceMode = "english" | "multilingual";
export type VoiceStatus = {
  supported: boolean; mode: VoiceMode; device: string | null; devices: string[];
  english_ready: boolean; multilingual_ready: boolean; listening: boolean; installing: boolean;
  english_download_mb: number; multilingual_download_mb: number;
};
type VoiceEventPayload =
  | { kind: "level"; level: number }
  | { kind: "partial"; segment: number; text: string }
  | { kind: "final"; segment: number; text: string }
  | { kind: "stopped"; error: string | null }
  | { kind: "loading" } | { kind: "listening" };
type InstallProgress = { label: string; detail: string; done: boolean; error: string | null };
export type VoicePhase = "off" | "loading" | "listening" | "stopping";

/** Append dictated text to what is already in the composer. */
export function appendDictation(current: string, text: string): string {
  const addition = text.trim();
  if (!addition) return current;
  if (!current.trim()) return addition;
  return /\s$/.test(current) ? current + addition : `${current} ${addition}`;
}

/** Voice input state: listens to the backend and feeds finished phrases
    into the composer through `onText`. */
export function useVoice(onText: (text: string) => void) {
  const [status, setStatus] = useState<VoiceStatus | null>(null);
  const [phase, setPhase] = useState<VoicePhase>("off");
  const [level, setLevel] = useState(0);
  const [caption, setCaption] = useState<{ segment: number; text: string } | null>(null);
  const [error, setError] = useState("");
  const [install, setInstall] = useState<InstallProgress | null>(null);
  const onTextRef = useRef(onText);
  onTextRef.current = onText;

  const refresh = useCallback(async () => {
    try { setStatus(await invoke<VoiceStatus>("voice_status")); } catch { setStatus(null); }
  }, []);

  useEffect(() => {
    void refresh();
    const voice = listen<VoiceEventPayload>("voice_event", ({ payload }) => {
      switch (payload.kind) {
        case "loading": setPhase("loading"); break;
        case "listening": setPhase("listening"); setError(""); break;
        case "level": setLevel(payload.level); break;
        case "partial": setCaption({ segment: payload.segment, text: payload.text }); break;
        case "final":
          if (payload.text.trim()) onTextRef.current(payload.text);
          setCaption((current) => current && current.segment <= payload.segment ? null : current);
          break;
        case "stopped":
          setPhase("off"); setLevel(0); setCaption(null);
          if (payload.error) setError(payload.error);
          // Clear the session on the backend when the microphone went away.
          void invoke("stop_voice").catch(() => undefined);
          break;
      }
    });
    const progress = listen<InstallProgress>("voice_install", ({ payload }) => {
      setInstall(payload.done ? null : payload);
      if (payload.error) setError(payload.error);
      if (payload.done) void refresh();
    });
    return () => { void voice.then((stop) => stop()); void progress.then((stop) => stop()); };
  }, [refresh]);

  const ready = status ? (status.mode === "english" ? status.english_ready : status.multilingual_ready) : false;

  const start = useCallback(async () => {
    setError("");
    setPhase("loading");
    try { await invoke("start_voice"); } catch (reason) { setPhase("off"); setError(String(reason)); }
  }, []);
  const stop = useCallback(async () => {
    setPhase("stopping");
    try { await invoke("stop_voice"); } catch { /* The stopped event resets the state. */ }
  }, []);
  const toggle = useCallback(() => {
    if (phase === "listening" || phase === "loading") void stop(); else void start();
  }, [phase, start, stop]);

  const installModels = useCallback(async (mode: VoiceMode) => {
    setError("");
    setInstall({ label: "Starting the download", detail: "", done: false, error: null });
    try { await invoke("install_voice_models", { mode }); } catch (reason) { setError(String(reason)); setInstall(null); }
    void refresh();
  }, [refresh]);
  const saveSettings = useCallback(async (mode: VoiceMode, device: string | null) => {
    try { await invoke("set_voice_settings", { mode, device }); } catch (reason) { setError(String(reason)); }
    void refresh();
  }, [refresh]);

  return { status, phase, level, caption, error, install, ready, toggle, stop, refresh, installModels, saveSettings };
}
export type Voice = ReturnType<typeof useVoice>;

function MicIcon() {
  return <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="9" y="3" width="6" height="11" rx="3" /><path d="M5 11a7 7 0 0 0 14 0" /><path d="M12 18v3" />
  </svg>;
}

/** The composer's microphone toggle. Without models it opens Voice settings. */
export function MicButton({ voice, onSetup }: { voice: Voice; onSetup: () => void }) {
  const active = voice.phase === "listening" || voice.phase === "loading";
  const unavailable = voice.status !== null && !voice.status.supported;
  const label = unavailable ? "Voice input needs Windows"
    : !voice.ready ? "Set up voice input"
    : active ? "Stop voice input (Ctrl+M)" : "Start voice input (Ctrl+M)";
  return <button type="button" className={`mic-button phase-${voice.phase}`} aria-label={label} title={label}
    aria-pressed={active} disabled={unavailable || voice.phase === "stopping"}
    style={{ "--mic-level": String(voice.phase === "listening" ? voice.level : 0) } as React.CSSProperties}
    onClick={() => (voice.ready ? voice.toggle() : onSetup())}>
    <span className="mic-ring" aria-hidden="true" /><MicIcon />
  </button>;
}

/** What is being heard right now, shown above the composer's controls. */
export function VoiceCaption({ voice }: { voice: Voice }) {
  if (voice.phase === "off" && !voice.error) return null;
  const text = voice.phase === "loading" ? "Loading the speech models…"
    : voice.phase === "stopping" ? "Finishing the last phrase…"
    : voice.caption?.text || (voice.phase === "listening" ? "Listening…" : "");
  return <div className={`voice-caption ${voice.error && voice.phase === "off" ? "error" : ""}`} role="status" aria-live="polite">
    {voice.phase === "off" ? <span>Voice input stopped: {voice.error}</span> : <>
      <span className="voice-bars" aria-hidden="true">{[0.5, 1, 0.7].map((scale, index) =>
        <i key={index} style={{ transform: `scaleY(${Math.max(0.15, Math.min(1, voice.level * scale * 1.6))})` }} />)}</span>
      <span className={voice.caption ? "voice-caption-text live" : "voice-caption-text"}>{text}</span>
    </>}
  </div>;
}

/** Settings → Voice. */
export function VoiceSettingsPage({ voice }: { voice: Voice }) {
  const status = voice.status;
  if (!status) return <SettingsCard title="Voice input" description="Checking voice input…" />;
  if (!status.supported) return <SettingsCard title="Voice input" description="Voice input runs on Windows." status={<StatusPill tone="off">Unavailable</StatusPill>} />;
  const modes: [VoiceMode, string, string, boolean, number][] = [
    ["english", "English, live", "Words appear as you speak, then each phrase is corrected with punctuation.", status.english_ready, status.english_download_mb],
    ["multilingual", "Multilingual", "25 European languages. Each phrase appears when you pause.", status.multilingual_ready, status.multilingual_download_mb],
  ];
  return <>
    <SettingsCard title="Voice input"
      status={voice.ready ? <StatusPill tone="ok">Ready</StatusPill> : <StatusPill tone="off">Not installed</StatusPill>}
      description="Speak into the message box with the microphone button or Ctrl+M. Speech is transcribed on this computer; audio never leaves it and is not saved. Nothing is sent until you press Enter.">
      <div className="choice-cards" role="radiogroup" aria-label="Voice mode">
        {modes.map(([mode, title, text, ready, download]) => <button key={mode} type="button" role="radio" aria-checked={status.mode === mode}
          className={status.mode === mode ? "selected" : ""} disabled={voice.phase !== "off"}
          onClick={() => void voice.saveSettings(mode, status.device)}>
          <strong>{title}</strong><small>{text}</small>
          <small className="choice-meta">{ready ? "Installed" : `Download about ${download} MB`}</small>
        </button>)}
      </div>
      {!voice.ready && <SettingRow label="Install" help="The models download once to the data folder.">
        <button type="button" className="primary" disabled={status.installing || voice.install !== null}
          onClick={() => void voice.installModels(status.mode)}>{voice.install ? "Installing…" : "Download and install"}</button>
      </SettingRow>}
      {voice.install && <div className="install-progress" role="status" aria-live="polite">
        <div className="install-progress-track"><div className="install-progress-fill" /></div>
        <span className="install-progress-label">{voice.install.label}{voice.install.detail ? ` — ${voice.install.detail}` : ""}</span>
      </div>}
      {voice.error && <p className="error-text">{voice.error}</p>}
    </SettingsCard>
    <SettingsCard title="Microphone" actions={<button type="button" onClick={() => void voice.refresh()}>Refresh</button>}>
      <SettingRow label="Input device">
        <select aria-label="Microphone" value={status.device ?? ""} disabled={voice.phase !== "off"}
          onChange={(event) => void voice.saveSettings(status.mode, event.target.value || null)}>
          <option value="">System default{status.devices[0] ? ` (${status.devices[0]})` : ""}</option>
          {status.devices.map((name) => <option key={name} value={name}>{name}</option>)}
        </select>
      </SettingRow>
      {status.devices.length === 0 && <p className="settings-note">No microphone found. Connect one and press Refresh.</p>}
    </SettingsCard>
    <SettingsCard title="Models" description={<>Live captions use a streaming Zipformer model (Apache-2.0). Accurate text uses NVIDIA Parakeet TDT 0.6B v3 (CC-BY-4.0). Both run with sherpa-onnx (Apache-2.0).</>} />
  </>;
}
