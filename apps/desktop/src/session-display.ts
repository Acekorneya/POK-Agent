export type ChatMessage = {
  id: string;
  sender: "user" | "agent" | "system";
  type: "prompt" | "guidance" | "reasoning" | "activity" | "response" | "error";
  text: string;
  timestamp: Date;
  activityTool?: string;
  activityLabel?: string;
  activityDetail?: string;
  activityStatus?: "running" | "done" | "failed";
  activityCallId?: string;
  activityArgs?: Record<string, unknown>;
  activityResult?: Record<string, unknown>;
  activityAttempts?: number;
  durationMs?: number;
};

export function appendStreamDelta(
  messages: ChatMessage[],
  type: "reasoning" | "response",
  delta: string,
): ChatMessage[] {
  if (!delta) return messages;
  const last = messages[messages.length - 1];
  if (last?.sender === "agent" && last.type === type) {
    return [
      ...messages.slice(0, -1),
      { ...last, text: last.text + delta },
    ];
  }
  return [
    ...messages,
    {
      id: `${type}-${crypto.randomUUID()}`,
      sender: "agent",
      type,
      text: delta,
      timestamp: new Date(),
    },
  ];
}
