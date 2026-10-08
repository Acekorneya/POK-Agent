import React, { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { asRecord, formatElapsed } from "../display-utils";
import type { ChatMessage } from "../session-display";

export function TerminalExecutionCard({ msg }: { msg: ChatMessage }) {
  const [expanded, setExpanded] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [fullOutput, setFullOutput] = useState("");
  const [fullOutputOffset, setFullOutputOffset] = useState(0);
  const [moreFullOutput, setMoreFullOutput] = useState(false);
  const [loadingFullOutput, setLoadingFullOutput] = useState(false);
  const command = String(msg.activityArgs?.command ?? msg.activityDetail ?? msg.text);
  const stdout = String(msg.activityResult?.stdout ?? msg.activityResult?.output ?? "");
  const stderr = String(msg.activityResult?.stderr ?? "");
  const output = [
    stdout.trimEnd(),
    stderr.trimEnd() ? `${stdout.trimEnd() ? "stderr:\n" : ""}${stderr.trimEnd()}` : "",
  ].filter(Boolean).join("\n");
  const lines = output ? output.split(/\r?\n/) : [];
  const hasMore = lines.length > 4;
  const preview = expanded ? output : lines.slice(0, 4).join("\n");
  const durationStr = msg.durationMs !== undefined ? formatElapsed(msg.durationMs) : "";
  const exitCode = msg.activityResult?.exit_code;
  const uiTruncated = msg.activityResult?.ui_output_truncated === true;
  const backendTruncated = msg.activityResult?.output_truncated === true;
  const failed = msg.activityStatus === "failed";
  const taskId = typeof msg.activityResult?.task_id === "string" ? msg.activityResult.task_id : undefined;
  const totalBytes = Number(msg.activityResult?.total_bytes ?? 0);
  const status = String(msg.activityResult?.status ?? (msg.activityStatus === "running" ? "running" : "completed"));
  const running = status === "running";
  const loadFullOutput = () => {
    if (!taskId || loadingFullOutput) return;
    setLoadingFullOutput(true);
    void invoke<{ output: string; next_offset: number; truncated: boolean }>("read_command_output", {
      taskId,
      offset: fullOutputOffset,
      maxBytes: 64 * 1024,
    }).then(page => {
      setFullOutput(current => current + page.output);
      setFullOutputOffset(page.next_offset);
      setMoreFullOutput(page.truncated);
      setExpanded(true);
    }).finally(() => setLoadingFullOutput(false));
  };
  const generatedArtifacts = Array.isArray(msg.activityResult?.artifacts)
    ? msg.activityResult.artifacts
      .map(asRecord)
      .filter(artifact => typeof artifact.path === "string")
    : [];

  return (
    <div className={`rich-activity-card terminal-card ${failed ? "terminal-failed" : ""}`}>
      <div className="terminal-header">
        <span><code className="terminal-cmd" title={command}>{command}</code></span>
        <span className="terminal-meta">
          {taskId && <span title={taskId}>{taskId.slice(0, 8)}</span>}
          {totalBytes > 0 && <span>{totalBytes.toLocaleString()} B</span>}
          {exitCode !== undefined && exitCode !== null && <span>exit {String(exitCode)}</span>}
          {durationStr && <span>({durationStr})</span>}
          {running && taskId && <button className="terminal-stop" disabled={stopping} onClick={() => {
            setStopping(true);
            void invoke("stop_command", { taskId }).finally(() => setStopping(false));
          }}>{stopping ? "Stopping…" : "Stop"}</button>}
        </span>
      </div>
      {output ? (
        <>
          <pre className="terminal-drawer">{fullOutput || preview}</pre>
          {hasMore && (
            <button className="terminal-drawer-toggle" onClick={() => setExpanded(!expanded)}>
              {expanded ? "▲ Collapse output" : `... +${lines.length - 4} lines (click to view)`}
            </button>
          )}
          {(uiTruncated || backendTruncated) && (
            <div className="terminal-output-note">
              Output truncated here; the complete captured result remains in diagnostics.
              {taskId && (
                <button className="terminal-drawer-toggle" disabled={loadingFullOutput} onClick={loadFullOutput}>
                  {loadingFullOutput ? "Loading…" : fullOutput ? (moreFullOutput ? "Load more output" : "Full output loaded") : "Load complete output"}
                </button>
              )}
            </div>
          )}
        </>
      ) : (
        <div className="terminal-empty-output">
          {msg.activityStatus === "running"
            ? "Command is running…"
            : failed ? "Command failed with no captured output" : "Completed with no output"}
        </div>
      )}
      {generatedArtifacts.length > 0 && (
        <div className="terminal-artifacts">
          <strong>Generated</strong>
          {generatedArtifacts.map((artifact, index) => (
            <span key={`${String(artifact.path)}-${index}`}>
              <code title={String(artifact.path)}>{String(artifact.path)}</code>
              {typeof artifact.artifact_type === "string" && <small>{artifact.artifact_type}</small>}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

const MAX_FILE_PREVIEW_LINES = 500;

export function FileChangeCard({ msg }: { msg: ChatMessage }) {
  const tool = msg.activityTool ?? "";
  const path = String(
    msg.activityResult?.path
      ?? msg.activityArgs?.filepath
      ?? msg.activityArgs?.path
      ?? msg.activityDetail
      ?? "file"
  );
  const isCreate = tool === "write_file";
  const isRewrite = msg.activityResult?.operation === "rewritten";
  const isUndo = tool === "undo_edit";
  const target = isCreate ? "" : String(msg.activityArgs?.target_content ?? "");
  const replacement = isCreate
    ? String(msg.activityArgs?.content ?? "")
    : String(msg.activityArgs?.replacement_content ?? msg.activityArgs?.code_content ?? "");
  const startLine = Number(msg.activityArgs?.start_line ?? 1);

  const targetLines = target ? target.split("\n") : [];
  const replacementLines = replacement ? replacement.split("\n") : [];
  const visibleTargetLines = targetLines.slice(0, MAX_FILE_PREVIEW_LINES);
  const remainingPreviewLines = Math.max(
    0,
    targetLines.length + replacementLines.length - MAX_FILE_PREVIEW_LINES,
  );
  const visibleReplacementLines = replacementLines.slice(
    0,
    Math.max(0, MAX_FILE_PREVIEW_LINES - visibleTargetLines.length),
  );
  const action = msg.activityStatus === "failed"
    ? isRewrite ? "Failed to rewrite" : isCreate ? "Failed to create" : isUndo ? "Failed to restore" : "Failed to edit"
    : msg.activityStatus === "running"
      ? isRewrite ? "Rewriting" : isCreate ? "Creating" : isUndo ? "Restoring" : "Editing"
      : isRewrite ? "Rewritten" : isCreate ? "Created" : isUndo ? "Restored" : "Edited";

  return (
    <div className={`rich-activity-card diff-card ${msg.activityStatus === "failed" ? "diff-failed" : ""}`}>
      <div className="diff-header">
        <span>{action} <code title={path}>{path}</code></span>
        <span className="diff-stat">
          {replacementLines.length > 0 && <span className="add">+{replacementLines.length}</span>}
          {targetLines.length > 0 && <span className="del">-{targetLines.length}</span>}
        </span>
      </div>
      <div className="diff-body">
        {visibleTargetLines.map((line, index) => (
          <div key={`del-${index}`} className="diff-row diff-del">
            <span className="diff-ln">{startLine + index} -</span>
            <span>{line}</span>
          </div>
        ))}
        {visibleReplacementLines.map((line, index) => (
          <div key={`add-${index}`} className="diff-row diff-add">
            <span className="diff-ln">{startLine + index} +</span>
            <span>{line}</span>
          </div>
        ))}
        {remainingPreviewLines > 0 && (
          <div className="diff-preview-note">
            {remainingPreviewLines.toLocaleString()} additional lines are preserved in diagnostics.
          </div>
        )}
        {!target && !replacement && (
          <div className="diff-preview-note">
            {msg.activityStatus === "running" ? "Waiting for file result…" : msg.activityDetail ?? "File operation completed."}
          </div>
        )}
      </div>
    </div>
  );
}
