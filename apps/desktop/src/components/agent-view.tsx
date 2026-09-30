import React, { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";

export type FrameTarget = { id: string; label: string; source: string; x: number; y: number; width: number; height: number };
export type ObservationFrame = { id: string; imagePath: string; windowTitle?: string; targets: FrameTarget[]; capturedAt: number };

/** A floating panel showing the agent's newest saved screenshot and targets. */
export function AgentView({ frame, onClose }: { frame: ObservationFrame | null; onClose: () => void }) {
  const [showTargets, setShowTargets] = useState(true);
  const [large, setLarge] = useState(false);
  const [image, setImage] = useState<{ path: string; url: string } | null>(null);
  const [hovered, setHovered] = useState<string | null>(null);
  const panel = useRef<HTMLElement>(null);
  // Where the user dragged the panel (top-left corner, in window pixels).
  const [position, setPosition] = useState<{ x: number; y: number } | null>(() => {
    try { return JSON.parse(localStorage.getItem("pok_agent_view_position") ?? "null"); } catch { return null; }
  });
  const drag = useRef<{ dx: number; dy: number } | null>(null);
  const clamp = (x: number, y: number) => {
    const box = panel.current?.getBoundingClientRect();
    const width = box?.width ?? 380, height = box?.height ?? 240;
    return { x: Math.max(8, Math.min(window.innerWidth - width - 8, x)), y: Math.max(8, Math.min(window.innerHeight - Math.min(height, 120), y)) };
  };
  const startDrag = (event: React.PointerEvent<HTMLElement>) => {
    if ((event.target as HTMLElement).closest("button") || !panel.current) return;
    const box = panel.current.getBoundingClientRect();
    drag.current = { dx: event.clientX - box.left, dy: event.clientY - box.top };
    event.currentTarget.setPointerCapture(event.pointerId);
  };
  const moveDrag = (event: React.PointerEvent<HTMLElement>) => {
    if (drag.current) setPosition(clamp(event.clientX - drag.current.dx, event.clientY - drag.current.dy));
  };
  const endDrag = () => {
    if (!drag.current) return;
    drag.current = null;
    try { localStorage.setItem("pok_agent_view_position", JSON.stringify(position)); } catch { /* Storage can be disabled. */ }
  };
  // Keep a remembered position on screen when the window gets smaller.
  useEffect(() => {
    const keep = () => setPosition((current) => current && clamp(current.x, current.y));
    window.addEventListener("resize", keep);
    return () => window.removeEventListener("resize", keep);
  }, []);
  useEffect(() => {
    if (!frame) { setImage(null); return; }
    let live = true;
    invoke<string>("read_observation_frame", { path: frame.imagePath })
      .then((url) => { if (live) setImage({ path: frame.imagePath, url }); })
      .catch(() => { if (live) setImage(null); });
    return () => { live = false; };
  }, [frame?.imagePath]);

  return <aside ref={panel} className={`agent-view ${large ? "large" : ""} ${position ? "placed" : ""}`} aria-label="Agent view"
    style={position ? { left: position.x, top: position.y } : undefined}>
    <header className="agent-view-handle" title="Drag to move" onPointerDown={startDrag} onPointerMove={moveDrag} onPointerUp={endDrag} onPointerCancel={endDrag}>
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
      {image && frame && image.path === frame.imagePath ? <div className="agent-view-canvas">
        <img src={image.url} alt={`Frame of ${frame.windowTitle || "the screen"}`} />
        {showTargets && frame.targets.map((target) => <div key={target.id}
          className={`agent-view-target source-${target.source} ${hovered === target.id ? "hovered" : ""}`}
          style={{ left: `${target.x * 100}%`, top: `${target.y * 100}%`, width: `${target.width * 100}%`, height: `${target.height * 100}%` }}
          onMouseEnter={() => setHovered(target.id)} onMouseLeave={() => setHovered(null)} title={`${target.id} · ${target.label}`}>
          <span>{target.id}</span>
        </div>)}
      </div> : <p className="agent-view-empty">{frame ? "Loading frame…" : "Waiting for the agent's first capture."}</p>}
    </div>
    <footer>
      <span>{frame ? "Latest capture" : "Waiting for capture"}</span>
      <small>{hovered && frame ? frame.targets.find((target) => target.id === hovered)?.label : frame ? `${frame.targets.length} targets` : ""}</small>
    </footer>
  </aside>;
}
