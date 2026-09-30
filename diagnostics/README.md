# POK-Agent internal diagnostics

Interactive development runs create one directory per session:

```text
diagnostics/
  sessions/
    <session-uuid>/
      trace.jsonl
      observation-<uuid>-monitor-<id>.png
      observation-<uuid>-monitor-<id>-source.png
      observation-<uuid>-monitor-<id>-overlay.svg
      observation-<uuid>-ocr.txt
      observation-<uuid>.json
      tool-<timestamp>-<name>.json
```

`trace.jsonl` records request shapes, estimated/current/cumulative token usage,
context compactions, streamed reasoning and response text,
latency, token usage, tool activity, and provider failures. PNG files are the
exact resized images supplied to the vision model; `-source.png` files retain
the native-resolution target crop. Overlay SVGs show UI Automation rectangles
in green and OCR rectangles in orange.

Captured screens, OCR, prompts, and tool results may contain private data. The
generated contents of this directory are ignored by Git; only this README is
tracked. Delete individual session directories when they are no longer needed.
These local diagnostics are full-fidelity. JSON records omit duplicated image
base64 only because the exact model images are stored as neighboring PNG files,
not as a privacy-masking step.
