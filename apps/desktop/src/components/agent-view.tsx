import React, { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";

export type FrameTarget = { id: string; label: string; source: string; x: number; y: number; width: number; height: number };
export type ObservationFrame = { id: string; imagePath: string; windowTitle?: string; targets: FrameTarget[]; capturedAt: number };

/** Frames kept for stepping back; images load from disk only when shown. */
export const AGENT_VIEW_FRAMES = 40;

/** A floating panel with the still frames the agent saw and the numbered
    targets it could act on. */
export function AgentView({ frames, onClose }: { frames: ObservationFrame[]; onClose: () => void }) {
  const [index, setIndex] = useState<number | null>(null); // null follows the newest frame
  const [showTargets, setShowTargets] = useState(true);
  const [large, setLarge] = useState(false);
  const [image, setImage] = useState<string | null>(null);
  const [hovered, setHovered] = useState<string | null>(null);
  const cache = useRef(new Map<string, string>());
  const position = index === null ? frames.length - 1 : Math.min(index, frames.length - 1);
  const frame = frames[position];

  useEffect(() => {
    if (!frame) { setImage(null); return; }
    const cached = cache.current.get(frame.imagePath);
    if (cached) { setImage(cached); return; }
    let live = true;
    invoke<string>("read_observation_frame", { path: frame.imagePath })
      .then((url) => {
        cache.current.set(frame.imagePath, url);
        if (cache.current.size > 12) cache.current.delete(cache.current.keys().next().value!);
        if (live) setImage(url);
      })
      .catch(() => { if (live) setImage(null); });
    return () => { live = false; };
  }, [frame?.imagePath]);

  const step = (delta: number) => {
    const next = Math.max(0, Math.min(frames.length - 1, position + delta));
    setIndex(next === frames.length - 1 ? null : next);
  };

  return <aside className={`agent-view ${large ? "large" : ""}`} aria-label="Agent view">
    <header>
      <div className="agent-view-title">
        <strong>What the agent sees</strong>
        <small title={frame?.windowTitle}>{frame ? frame.windowTitle || "Screen" : "Waiting for the first look"}</small>
      </div>
      <div className="agent-view-actions">
        <button type="button" aria-pressed={showTargets} title="Show the numbered targets" onClick={() => setShowTargets((value) => !value)}>Targets</button>
        <button type="button" title={large ? "Smaller" : "Larger"} aria-label={large ? "Smaller agent view" : "Larger agent view"} onClick={() => setLarge((value) => !value)}>{large ? "⤡" : "⤢"}</button>
        <button type="button" aria-label="Close agent view" onClick={onClose}>×</button>
      </div>
    </header>
    <div className="agent-view-frame">
      {image && frame ? <div className="agent-view-canvas">
        <img src={image} alt={`Frame of ${frame.windowTitle || "the screen"}`} />
        {showTargets && frame.targets.map((target) => <div key={target.id}
          className={`agent-view-target source-${target.source} ${hovered === target.id ? "hovered" : ""}`}
          style={{ left: `${target.x * 100}%`, top: `${target.y * 100}%`, width: `${target.width * 100}%`, height: `${target.height * 100}%` }}
          onMouseEnter={() => setHovered(target.id)} onMouseLeave={() => setHovered(null)} title={`${target.id} · ${target.label}`}>
          <span>{target.id}</span>
        </div>)}
      </div> : <p className="agent-view-empty">{frame ? "Loading frame…" : "Frames appear here each time the agent looks at the screen."}</p>}
    </div>
    <footer>
      <button type="button" aria-label="Previous frame" disabled={position <= 0} onClick={() => step(-1)}>‹</button>
      <span>{frames.length ? `${position + 1} / ${frames.length}` : "0 / 0"}{index === null && frames.length > 0 ? " · live" : ""}</span>
      <button type="button" aria-label="Next frame" disabled={position >= frames.length - 1} onClick={() => step(1)}>›</button>
      <small>{hovered && frame ? frame.targets.find((target) => target.id === hovered)?.label : frame ? `${frame.targets.length} targets` : ""}</small>
    </footer>
  </aside>;
}
