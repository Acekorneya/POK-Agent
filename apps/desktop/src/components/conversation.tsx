import React from "react";
import type { ChatMessage } from "../session-display";
import { ActivityRow, MarkdownMessage } from "./workspace";
import { FileChangeCard, TerminalExecutionCard } from "./activity-cards";

const FRIENDLY_CANDIDATES: Record<string, string> = {
  jev_done: "Router done",
  jev_blocked: "Router blocked",
  search_web_for_request: "Search the web",
  list_visible_windows: "List windows",
  list_windows: "List windows",
  get_current_time: "Get current time",
};

export type MessageActionKind = "copy" | "fork" | "revert";
export type MessageActionHandler = (kind: MessageActionKind, message: ChatMessage) => void;

const MessageActionContext = React.createContext<MessageActionHandler | null>(null);

/** Supplies the copy/fork/revert handler to prompt and response rows. The
 * value must be referentially stable (a ref-backed callback) so memoized
 * history rows do not re-render on every session event. */
export function MessageActionProvider({ onAction, children }: { onAction: MessageActionHandler; children: React.ReactNode }) {
  return <MessageActionContext.Provider value={onAction}>{children}</MessageActionContext.Provider>;
}

function MessageTools({ msg }: { msg: ChatMessage }) {
  const onAction = React.useContext(MessageActionContext);
  if (!onAction) return null;
  return (
    <span className="message-tools" role="toolbar" aria-label="Message actions" onClick={(event) => event.stopPropagation()}>
      <button type="button" aria-label="Copy message" onClick={() => onAction("copy", msg)}>Copy</button>
      <button type="button" aria-label="Fork from this message" title="Continue in a new conversation from here" onClick={() => onAction("fork", msg)}>Fork</button>
      <button type="button" aria-label="Revert to this message" title="Erase everything after this message" onClick={() => onAction("revert", msg)}>Revert</button>
    </span>
  );
}

function friendlyCandidate(id: string): string {
  if (FRIENDLY_CANDIDATES[id]) return FRIENDLY_CANDIDATES[id];
  if (id.startsWith("activate_window")) return "Activate window";
  if (id.startsWith("click_")) return "Click target";
  if (id.startsWith("evidence_")) return "Focus evidence";
  if (id.startsWith("refresh_ambiguous")) return "Refresh state";
  if (id.startsWith("browser_click_")) return "Click link";
  return id;
}

function DecisionRouterCard({ msg }: { msg: ChatMessage }) {
  const result = msg.activityResult ?? {};
  const candidateId = result.candidate_id ? String(result.candidate_id) : null;
  const selected = result.selected_candidate ? String(result.selected_candidate) : null;
  const op = typeof result.operation_probability === "number" ? Math.round(result.operation_probability * 100) : null;
  const tgt = typeof result.target_probability === "number" ? Math.round(result.target_probability * 100) : null;
  const reason = result.rejection_reason ? String(result.rejection_reason).replaceAll("_", " ") : null;
  const alternatives = Array.isArray(result.alternatives)
    ? (result.alternatives as Array<[string, number]>).filter(([id]) => !id.startsWith("operation:") && !id.startsWith("target:") && !id.startsWith("candidate_"))
    : [];
  const isTerminal = candidateId === "jev_done" || candidateId === "jev_blocked";
  return (
    <div className="decision-card">
      {isTerminal ? (
        <p className="decision-card-pick"><strong>{candidateId === "jev_done" ? "Router done" : "Router blocked"}</strong>{op !== null ? ` — ${op}%` : ""}</p>
      ) : selected ? (
        <p className="decision-card-pick"><strong>{friendlyCandidate(candidateId ?? "")}</strong>{op !== null && tgt !== null ? ` — operation ${op}% · target ${tgt}%` : op !== null ? ` — ${op}%` : ""}</p>
      ) : null}
      {selected && isTerminal && <p className="decision-card-note">{selected}</p>}
      {reason && <p className="decision-card-reason">{reason}</p>}
      {alternatives.length > 0 && (
        <ul className="decision-card-alternatives">
          {alternatives.slice(0, 3).map(([id, score]) => (
            <li key={id}>{friendlyCandidate(id)} <span>{Math.round(score * 100)}%</span></li>
          ))}
        </ul>
      )}
      {!selected && !isTerminal && !reason && alternatives.length === 0 && <p className="decision-card-note">No bounded candidate was selected.</p>}
    </div>
  );
}

function JudgeCard({ msg }: { msg: ChatMessage }) {
  const result = msg.activityResult ?? {};
  const votes = (result.votes ?? {}) as Record<string, string>;
  const checks = ["reversible", "cheap", "evidenced"] as const;
  return (
    <div className="decision-card">
      <p className="decision-card-verdict">
        {result.promoted
          ? "Approved — safe, cheap and evidenced; executed without the primary model."
          : "Held — at least one check was not a clear yes; handed to the primary model."}
      </p>
      <ul className="decision-card-checks">
        {checks.map((question) => (
          <li key={question}><span>{question}</span> <strong>{votes[question] ?? "—"}</strong></li>
        ))}
      </ul>
      {votes.comparative && (
        <p className="decision-card-note">
          vs runner-up: {votes.comparative === "b" ? "runner-up preferred" : "pick preferred"}
        </p>
      )}
      {!!result.reused && <p className="decision-card-note">Cached verdict reused — no new model call.</p>}
    </div>
  );
}

function ConversationMessageView({ message: msg }: { message: ChatMessage }) {
  switch (msg.type) {
    case "prompt":
      return <div className="console-line line-prompt">
        {msg.images && msg.images.length > 0 && <div className="prompt-images">
          {msg.images.map((image, index) => <img key={index} src={image} alt={`Attached image ${index + 1}`} />)}
        </div>}
        {msg.files && msg.files.length > 0 && <div className="prompt-files">
          {msg.files.map((path) => <span key={path} className="file-chip" title={path}>
            <span className="file-chip-icon" aria-hidden="true">▤</span><span className="file-chip-name">{path.split(/[\\/]/).filter(Boolean).pop()}</span>
          </span>)}
        </div>}
        {msg.text}
        <MessageTools msg={msg} />
      </div>;
    case "guidance":
      return <div className="console-line line-guidance"><small>Guidance</small>{msg.text}</div>;
    case "reasoning":
      return <details className="reasoning-disclosure"><summary>Reasoning <span>{msg.text.trim().split("\n").filter(Boolean).at(-1)?.slice(0, 110)}</span></summary><MarkdownMessage text={msg.text} /></details>;
    case "activity": {
      if (!msg.activityTool) return <div className="session-notice">{msg.text}</div>;
      const args = msg.activityArgs ?? {};
      const result = msg.activityResult ?? {};
      const hasArgs = Object.keys(args).length > 0;
      const hasResult = Object.keys(result).length > 0;
      return <ActivityRow label={msg.activityLabel ?? msg.text} detail={msg.activityDetail} status={msg.activityStatus} durationMs={msg.durationMs}>
        {msg.activityTool === "run_command" ? <TerminalExecutionCard msg={msg} />
          : ["edit_file", "write_file", "replace_file_content", "undo_edit"].includes(msg.activityTool) ? <FileChangeCard msg={msg} />
          : msg.activityTool === "jev_decision" ? <DecisionRouterCard msg={msg} />
          : msg.activityTool === "jev_judge" ? <JudgeCard msg={msg} />
          : hasArgs || hasResult ? (
              <>
                {hasArgs && <><h3>Arguments</h3><pre>{JSON.stringify(args, null, 2)}</pre></>}
                {hasResult && <><h3>Result</h3><pre>{JSON.stringify(result, null, 2)}</pre></>}
              </>
            )
          : <p className="decision-card-note">Completed.</p>}
      </ActivityRow>;
    }
    case "response":
      return <div className="console-line line-response"><MarkdownMessage text={msg.text} /><MessageTools msg={msg} /></div>;
    case "error":
      return <div className="console-line line-error">{msg.text}</div>;
  }
}

/**
 * Messages are immutable once appended (streaming replaces only the last one),
 * so a keystroke in the composer must not re-parse markdown or re-stringify
 * every historical tool payload in a long conversation.
 */
export const ConversationMessage = React.memo(ConversationMessageView);