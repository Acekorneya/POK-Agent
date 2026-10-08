import React, { useEffect, useRef, useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

export type Theme = "teal" | "dark" | "light";
export function applyTheme(theme: Theme) {
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem("pok_theme", theme); } catch { /* Storage can be disabled. */ }
}
export function Appearance() {
  const [theme, setTheme] = useState(document.documentElement.dataset.theme || "teal");
  return <fieldset className="theme-picker"><legend>Appearance</legend>{([
    ["teal", "POK dark teal"], ["dark", "Neutral dark"], ["light", "Light"],
  ] as const).map(([value, label]) => <button key={value} type="button" aria-pressed={theme === value} onClick={() => { applyTheme(value); setTheme(value); }}>
    <span className={`theme-swatch swatch-${value}`} />{label}
  </button>)}</fieldset>;
}

type Conversation = { id: string; title: string; workspace: string; model: string; updated_at: string };
export function ConversationSidebar({ conversations, activeId, disabled, onSelect, onNew, onSettings, onClose }: {
  conversations: Conversation[]; activeId: string; disabled: boolean;
  onSelect: (id: string) => void; onNew: () => void; onSettings: (section: string) => void; onClose: () => void;
}) {
  const [query, setQuery] = useState("");
  const groups = new Map<string, Conversation[]>();
  conversations.filter(c => `${c.title} ${c.workspace} ${c.model}`.toLowerCase().includes(query.toLowerCase())).forEach(c => {
    const group = c.workspace || "Default workspace";
    groups.set(group, [...(groups.get(group) || []), c]);
  });
  return <aside className="conversation-sidebar" aria-label="Conversations">
    <div className="sidebar-brand"><img className="brand-logo" src="/pok-logo.png" alt="POK" /><strong>Agent</strong><button aria-label="Close navigation" onClick={onClose}>‹</button></div>
    <button className="new-conversation" disabled={disabled} onClick={onNew}>＋ New conversation</button>
    <input aria-label="Search recent conversations" placeholder="Search conversations…" value={query} onChange={e => setQuery(e.target.value)} />
    <div className="conversation-list">{[...groups].map(([workspace, items]) => <section key={workspace}>
      <h2 title={workspace}>{workspace.split(/[\\/]/).filter(Boolean).pop() || workspace}</h2>
      {items.map(c => <button key={c.id} className={`conversation-item ${activeId === c.id ? "active" : ""}`} aria-current={activeId === c.id ? "page" : undefined} disabled={disabled} onClick={() => onSelect(c.id)}>
        <strong>{c.title || "Untitled conversation"}</strong><span>{c.model}</span><time dateTime={c.updated_at}>{new Date(c.updated_at).toLocaleDateString(undefined, { month: "short", day: "numeric" })}</time>
      </button>)}
    </section>)}{groups.size === 0 && <p className="sidebar-empty">{query ? "No matching conversations" : "Your conversations will appear here."}</p>}</div>
    <small className="history-limit">Latest 50 conversations</small>
    <nav className="sidebar-footer"><button onClick={() => onSettings("memory-library")}>◇ Memory & skills</button><button onClick={() => onSettings("generated-tools")}>⌘ Generated tools</button><button onClick={() => onSettings("settings-heading")}>⚙ Settings</button></nav>
  </aside>;
}

function CopyButton({ text }: { text: string }) {
  const [label, setLabel] = useState("Copy");
  return <button className="copy-button" onClick={() => { void navigator.clipboard.writeText(text).then(() => setLabel("Copied"), () => setLabel("Copy unavailable")); }}>{label}</button>;
}
function CodeBlock({ children, ...props }: React.ComponentProps<"pre">) {
  const ref = useRef<HTMLPreElement>(null);
  return <div className="code-block"><button className="copy-button" onClick={e => { const button = e.currentTarget; void navigator.clipboard.writeText(ref.current?.textContent || "").then(() => { button.textContent = "Copied"; }, () => { button.textContent = "Copy unavailable"; }); }}>Copy code</button><pre ref={ref} {...props}>{children}</pre></div>;
}
function MarkdownMessageView({ text }: { text: string }) {
  return <div className="markdown-message"><Markdown remarkPlugins={[remarkGfm]} skipHtml components={{
    pre: CodeBlock,
    img: ({ alt }) => <span className="image-placeholder">[Image: {alt || "not loaded"}]</span>,
    a: ({ children, href }) => <a href={href && /^https?:\/\//i.test(href) ? href : undefined} target="_blank" rel="noreferrer noopener">{children}</a>,
  }}>{text}</Markdown><CopyButton text={text} /></div>;
}
/** Markdown parsing is the most expensive part of the feed; skip it when the
 * text did not change (every composer keystroke re-renders the session). */
export const MarkdownMessage = React.memo(MarkdownMessageView);

export function ActivityRow({ label, detail, status, durationMs, children }: {
  label: string; detail?: string; status?: string; durationMs?: number; children: React.ReactNode;
}) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    if (status !== "running") return;
    const start = Date.now();
    const timer = window.setInterval(() => setElapsed(Math.floor((Date.now() - start) / 1000)), 1000);
    return () => window.clearInterval(timer);
  }, [status]);
  return <details className={`tool-disclosure activity-${status}`}><summary>
    <span className="tool-state" aria-label={status || "activity"}>{status === "running" ? "◌" : status === "failed" ? "×" : status === "done" ? "✓" : "•"}</span>
    <strong>{label}</strong><span className="tool-summary">{detail}</span><span className="tool-duration">{status === "running" ? `Running · ${elapsed}s` : durationMs !== undefined ? `${Math.floor(durationMs / 1000)}s` : ""}</span>
  </summary><div className="tool-body">{children}</div></details>;
}

export function Composer({ children, prompt, onChange, onKeyDown, onPaste, placeholder, above }: {
  children: React.ReactNode; prompt: string; onChange: (value: string) => void;
  onKeyDown: (event: React.KeyboardEvent<HTMLTextAreaElement>) => void; placeholder: string;
  onPaste?: (event: React.ClipboardEvent<HTMLTextAreaElement>) => void;
  /** Shown above the message box (attached images). */
  above?: React.ReactNode;
}) {
  const ref = useRef<HTMLTextAreaElement>(null);
  useEffect(() => { if (ref.current) { ref.current.style.height = "0px"; ref.current.style.height = `${Math.min(180, Math.max(56, ref.current.scrollHeight))}px`; } }, [prompt]);
  return <div className="composer">{above}<textarea ref={ref} aria-label="Message the agent" value={prompt} onChange={e => onChange(e.target.value)} onKeyDown={onKeyDown} onPaste={onPaste} placeholder={placeholder} rows={2} />{children}</div>;
}

/** Focus the topmost dialog, contain Tab, and restore the triggering control. */
export function useDialogFocus(dependency: unknown) {
  useEffect(() => {
    const dialogs = document.querySelectorAll<HTMLElement>('.modal .dialog, .settings-open .settings-panel');
    const dialog = dialogs[dialogs.length - 1];
    if (!dialog) return;
    const previous = document.activeElement as HTMLElement | null;
    const focusable = () => [...dialog.querySelectorAll<HTMLElement>('button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled), a[href], [tabindex="0"]')].filter(el => el.getClientRects().length > 0);
    dialog.setAttribute("role", "dialog"); dialog.setAttribute("aria-modal", "true");
    (dialog.querySelector<HTMLElement>('[aria-current="page"]') ?? focusable()[0])?.focus({ focusVisible: false } as FocusOptions);
    const handle = (event: KeyboardEvent) => {
      if (event.key !== "Tab") return;
      const items = focusable(); const first = items[0]; const last = items[items.length - 1];
      if (!first) { event.preventDefault(); return; }
      if (event.shiftKey && (document.activeElement === first || !dialog.contains(document.activeElement))) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && (document.activeElement === last || !dialog.contains(document.activeElement))) { event.preventDefault(); first.focus(); }
    };
    document.addEventListener("keydown", handle);
    return () => { document.removeEventListener("keydown", handle); if (previous?.isConnected) previous.focus(); };
  }, [dependency]);
}
