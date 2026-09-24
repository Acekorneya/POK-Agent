# LM Studio API Compatibility

POK-Ai uses LM Studio's OpenAI-compatible `POST /v1/chat/completions` endpoint
for its client-side Rust tools. The adapter sends `tools`, `tool_choice: auto`,
`stream: true`, and `stream_options.include_usage: true`.

## Conversation rules

- Normal message content is an array of non-empty text/image objects.
- An assistant message containing only `tool_calls` omits `content`. LM Studio
  rejects empty text objects and, in current versions, also rejects `null`.
- Every assistant tool-call message is followed by its corresponding `tool`
  result before the next model request.
- Default-tool-format models may emit `[TOOL_REQUEST]` blocks; POK-Ai retains a
  recovery parser for this LM Studio fallback.

## Streaming

The OpenAI adapter consumes SSE deltas for:

- `delta.content`
- `delta.reasoning` and `delta.reasoning_content`
- fragmented `delta.tool_calls`
- token usage supplied through `stream_options.include_usage`

The Tauri backend forwards normalized reasoning, response text, turn changes,
tool execution, completion, and usage events to the dashboard immediately.

LM Studio's newer native `POST /api/v1/chat` endpoint provides additional named
events such as prompt-processing progress and tool-call boundaries. It is most
directly suited to LM Studio-managed plugins and MCP integrations. POK-Ai keeps
the OpenAI-compatible endpoint for its in-process Windows and coding tools while
using `/api/v1/models` for richer model capability discovery.

## Automatic vision routing

Before a session starts, POK-Ai reads the selected model from `GET /api/v1/models`.
When `capabilities.vision` is `true`, capture results include the annotated PNG and
the compact target list. When it is `false`, POK-Ai strips all image content from
the conversation while retaining OCR/UIA targets; full screenshots are still saved
to the diagnostic bundle. Unknown capability metadata preserves image delivery for
compatibility with non-LM-Studio providers. The chosen mode and discovered tool-use
capability are recorded in the session's `run_started` trace event.

## Official references

- https://lmstudio.ai/docs/developer/openai-compat/chat-completions
- https://lmstudio.ai/docs/developer/openai-compat/tools
- https://lmstudio.ai/docs/developer/rest/chat
- https://lmstudio.ai/docs/developer/rest/streaming-events
- https://lmstudio.ai/docs/developer/api-changelog

For raw server/model diagnostics, LM Studio documents `lms log stream` with
source filters, JSON output, and token-speed statistics.
