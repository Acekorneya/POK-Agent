/**
 * The markdown renderer, loaded on demand with a dynamic import so the
 * initial bundle carries only what the first paint needs (the same pattern
 * OpenCode uses for its lazy screens and dialogs). `MarkdownMessage` in
 * workspace.tsx renders this behind a Suspense boundary and falls back to the
 * plain text until the chunk is ready.
 */
import React, { useRef, useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

function CopyButton({ text }: { text: string }) {
  const [label, setLabel] = useState("Copy");
  return <button className="copy-button" onClick={() => { void navigator.clipboard.writeText(text).then(() => setLabel("Copied"), () => setLabel("Copy unavailable")); }}>{label}</button>;
}

function CodeBlock({ children, ...props }: React.ComponentProps<"pre">) {
  const ref = useRef<HTMLPreElement>(null);
  return <div className="code-block"><button className="copy-button" onClick={e => { const button = e.currentTarget; void navigator.clipboard.writeText(ref.current?.textContent || "").then(() => { button.textContent = "Copied"; }, () => { button.textContent = "Copy unavailable"; }); }}>Copy code</button><pre ref={ref} {...props}>{children}</pre></div>;
}

export default function MarkdownRenderer({ text }: { text: string }) {
  return <div className="markdown-message"><Markdown remarkPlugins={[remarkGfm]} skipHtml components={{
    pre: CodeBlock,
    img: ({ alt }) => <span className="image-placeholder">[Image: {alt || "not loaded"}]</span>,
    a: ({ children, href }) => <a href={href && /^https?:\/\//i.test(href) ? href : undefined} target="_blank" rel="noreferrer noopener">{children}</a>,
  }}>{text}</Markdown><CopyButton text={text} /></div>;
}
