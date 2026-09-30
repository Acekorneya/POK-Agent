import React, { useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";

/** An image attached to the next message, as a PNG data URL. */
export type ImageAttachment = { id: string; name: string; dataUrl: string };
/** Any other file or folder, sent to the agent by its path. */
export type FileAttachment = { id: string; name: string; path: string };

const IMAGE_EXTENSIONS = /\.(png|jpe?g|gif|webp|bmp)$/i;
export const fileName = (path: string) => path.split(/[\\/]/).filter(Boolean).pop() ?? path;

/** Longest edge sent to the model; larger pictures are scaled down. */
export const ATTACHMENT_MAX_EDGE = 1568;
export const MAX_ATTACHMENTS = 8;
const MAX_INPUT_BYTES = 30 * 1024 * 1024;

function readAsDataUrl(file: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error ?? new Error("could not read the file"));
    reader.readAsDataURL(file);
  });
}

/** Decode any image the browser understands (PNG, JPEG, WebP, GIF, BMP),
    scale it to fit the model, and encode it as PNG. */
export async function imageFileToPng(file: File, maxEdge = ATTACHMENT_MAX_EDGE): Promise<string> {
  if (!file.type.startsWith("image/")) throw new Error(`${file.name || "That file"} is not an image`);
  if (file.size > MAX_INPUT_BYTES) throw new Error(`${file.name || "The image"} is larger than 30 MB`);
  let bitmap: ImageBitmap | null = null;
  try { bitmap = typeof createImageBitmap === "function" ? await createImageBitmap(file) : null; } catch { bitmap = null; }
  const canvas = document.createElement("canvas");
  const context = bitmap ? canvas.getContext("2d") : null;
  if (!bitmap || !context) {
    // No canvas (tests, very old engines): a PNG can still be sent as is.
    if (file.type === "image/png") return readAsDataUrl(file);
    throw new Error(`${file.name || "The image"} could not be decoded`);
  }
  const scale = Math.min(1, maxEdge / Math.max(bitmap.width, bitmap.height));
  canvas.width = Math.max(1, Math.round(bitmap.width * scale));
  canvas.height = Math.max(1, Math.round(bitmap.height * scale));
  context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  bitmap.close();
  return canvas.toDataURL("image/png");
}

/** Images waiting to be sent with the next message. */
export function useAttachments() {
  const [items, setItems] = useState<ImageAttachment[]>([]);
  const [files, setFiles] = useState<FileAttachment[]>([]);
  const [error, setError] = useState("");
  const add = useCallback(async (files: Iterable<File>) => {
    const images = [...files].filter((file) => file.type.startsWith("image/"));
    if (images.length === 0) { setError("Only images can be attached."); return; }
    setError("");
    for (const file of images) {
      try {
        const dataUrl = await imageFileToPng(file);
        let full = false;
        setItems((current) => {
          if (current.length >= MAX_ATTACHMENTS) { full = true; return current; }
          return [...current, { id: crypto.randomUUID(), name: file.name || "Pasted image", dataUrl }];
        });
        if (full) { setError(`Attach up to ${MAX_ATTACHMENTS} images per message.`); break; }
      } catch (reason) {
        setError(reason instanceof Error ? reason.message : String(reason));
      }
    }
  }, []);
  /** Dropped paths: images become vision attachments, everything else
      (documents, archives, folders) is handed to the agent by path. */
  const addPaths = useCallback(async (paths: string[]) => {
    setError("");
    const images: File[] = [];
    const others: FileAttachment[] = [];
    for (const path of paths) {
      if (IMAGE_EXTENSIONS.test(path)) {
        try {
          const dataUrl = await invoke<string>("read_dropped_image", { path });
          const blob = await (await fetch(dataUrl)).blob();
          images.push(new File([blob], fileName(path), { type: blob.type }));
          continue;
        } catch { /* Unreadable as an image: attach it as a file instead. */ }
      }
      others.push({ id: crypto.randomUUID(), name: fileName(path), path });
    }
    if (others.length > 0) {
      setFiles((current) => {
        const known = new Set(current.map((file) => file.path));
        return [...current, ...others.filter((file) => !known.has(file.path))].slice(0, 20);
      });
    }
    if (images.length > 0) await add(images);
  }, [add]);
  const remove = useCallback((id: string) => {
    setItems((current) => current.filter((item) => item.id !== id));
    setFiles((current) => current.filter((file) => file.id !== id));
  }, []);
  const clear = useCallback(() => { setItems([]); setFiles([]); setError(""); }, []);
  return { items, files, error, setError, add, addPaths, remove, clear };
}
export type Attachments = ReturnType<typeof useAttachments>;

/** Thumbnails above the message box, each removable. */
export function AttachmentStrip({ attachments }: { attachments: Attachments }) {
  if (attachments.items.length === 0 && attachments.files.length === 0 && !attachments.error) return null;
  return <div className="attachment-strip" aria-label="Attachments">
    {attachments.items.map((item) => <figure key={item.id} className="attachment">
      <img src={item.dataUrl} alt={item.name} />
      <button type="button" aria-label={`Remove ${item.name}`} title="Remove" onClick={() => attachments.remove(item.id)}>×</button>
    </figure>)}
    {attachments.files.map((file) => <span key={file.id} className="file-chip" title={file.path}>
      <span className="file-chip-icon" aria-hidden="true">{/\.[a-z0-9]{1,5}$/i.test(file.name) ? "▤" : "▣"}</span>
      <span className="file-chip-name">{file.name}</span>
      <button type="button" aria-label={`Remove ${file.name}`} title="Remove" onClick={() => attachments.remove(file.id)}>×</button>
    </span>)}
    {attachments.error && <span className="attachment-error" role="alert">{attachments.error}</span>}
  </div>;
}

function PaperclipIcon() {
  return <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 11.5 12.5 20a5 5 0 0 1-7-7L14 4.5a3.5 3.5 0 0 1 5 5L10.5 18a2 2 0 0 1-3-3L15 7.5" />
  </svg>;
}

/** The composer's attach button; opens a file picker for images. */
export function AttachButton({ attachments, disabled, title }: { attachments: Attachments; disabled?: boolean; title?: string }) {
  const input = useRef<HTMLInputElement>(null);
  const label = title ?? "Attach images";
  return <>
    <button type="button" className="attach-button" aria-label={label} title={label} disabled={disabled} onClick={() => input.current?.click()}>
      <PaperclipIcon />
    </button>
    <input ref={input} type="file" accept="image/*" multiple hidden aria-hidden="true" tabIndex={-1}
      onChange={(event) => { if (event.target.files) void attachments.add(event.target.files); event.target.value = ""; }} />
  </>;
}

/** Drop target over the conversation: files dragged anywhere on it attach. */
export function useImageDrop(attachments: Attachments, enabled: boolean) {
  const [dragging, setDragging] = useState(false);
  const depth = useRef(0);
  const hasFiles = (event: React.DragEvent) => [...event.dataTransfer.types].includes("Files");
  return {
    dragging,
    handlers: {
      onDragEnter: (event: React.DragEvent) => { if (!enabled || !hasFiles(event)) return; event.preventDefault(); depth.current += 1; setDragging(true); },
      onDragOver: (event: React.DragEvent) => { if (!enabled || !hasFiles(event)) return; event.preventDefault(); event.dataTransfer.dropEffect = "copy"; },
      onDragLeave: () => { depth.current = Math.max(0, depth.current - 1); if (depth.current === 0) setDragging(false); },
      onDrop: (event: React.DragEvent) => {
        if (!enabled || !hasFiles(event)) return;
        event.preventDefault(); depth.current = 0; setDragging(false);
        void attachments.add(event.dataTransfer.files);
      },
    },
  };
}

/** Files and folders dropped anywhere on the window (Tauri gives their
    paths; the page itself would only see anonymous file contents). */
export function useNativeDrop(attachments: Attachments, enabled: boolean) {
  const [dragging, setDragging] = useState(false);
  const enabledRef = useRef(enabled);
  enabledRef.current = enabled;
  const addPaths = attachments.addPaths;
  useEffect(() => {
    if (!("__TAURI_INTERNALS__" in window)) return;
    let stop: (() => void) | undefined;
    let cancelled = false;
    void import("@tauri-apps/api/webview").then(({ getCurrentWebview }) =>
      getCurrentWebview().onDragDropEvent(({ payload }) => {
        if (payload.type === "enter" || payload.type === "over") setDragging(enabledRef.current);
        else if (payload.type === "leave") setDragging(false);
        else if (payload.type === "drop") {
          setDragging(false);
          if (enabledRef.current && payload.paths.length > 0) void addPaths(payload.paths);
        }
      })).then((unlisten) => { if (cancelled) unlisten(); else stop = unlisten; }).catch(() => undefined);
    return () => { cancelled = true; stop?.(); };
  }, [addPaths]);
  return dragging;
}
