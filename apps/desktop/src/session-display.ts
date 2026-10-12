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
  /** Live workers of a `run_agents` swarm, for its per-agent status card. */
  activitySwarm?: SwarmRunState;
  activityAttempts?: number;
  durationMs?: number;
  /** Images the user attached to a prompt (PNG data URLs). */
  images?: string[];
  /** Paths of files or folders the user attached to a prompt. */
  files?: string[];
};

export type SwarmAgentStatus =
  | "starting"
  | "running"
  | "done"
  | "failed"
  | "timed_out"
  | "cancelled";

export type SwarmAgentState = {
  id: string;
  title: string;
  role: string;
  task?: string;
  status: SwarmAgentStatus;
  activity?: string;
  detail?: string;
  turn?: number;
  startedAt?: number;
  seconds?: number;
  report?: string;
};

export type SwarmRunState = {
  runId: string;
  agents: SwarmAgentState[];
};

/** A swarm progress event, shaped like the `agent_event` payload. */
export type SwarmEventPayload = {
  type: string;
  run_id?: string;
  agent_id?: string;
  title?: string;
  role?: string;
  task?: string;
  activity?: string;
  detail?: string;
  turn?: number;
  status?: string;
  report?: string;
  seconds?: number;
};

function swarmAgentStatus(status: string | undefined): SwarmAgentStatus {
  switch (status) {
    case "completed":
      return "done";
    case "timed_out":
      return "timed_out";
    case "cancelled":
      return "cancelled";
    default:
      return "failed";
  }
}

/**
 * Fold one swarm event into the run's agent list. Workers are keyed by id so
 * activity and the final report land on the agent that reported them; the
 * card also matches by title against the tool arguments before the first
 * event arrives.
 */
export function appendSwarmEvent(
  run: SwarmRunState | null,
  event: SwarmEventPayload,
  now = Date.now(),
): SwarmRunState {
  const agents = run?.agents ? [...run.agents] : [];
  const id = event.agent_id ?? "agent";
  const index = agents.findIndex((agent) => agent.id === id);
  if (event.type === "swarm_agent_started") {
    const started: SwarmAgentState = {
      ...(index >= 0 ? agents[index] : { id, title: event.title ?? "agent", role: event.role ?? "general" }),
      title: event.title ?? (index >= 0 ? agents[index].title : "agent"),
      role: event.role ?? (index >= 0 ? agents[index].role : "general"),
      task: event.task,
      status: "running",
      startedAt: index >= 0 ? (agents[index].startedAt ?? now) : now,
    };
    if (index >= 0) agents[index] = started;
    else agents.push(started);
  } else if (event.type === "swarm_agent_activity") {
    if (index >= 0) {
      agents[index] = {
        ...agents[index],
        status: "running",
        activity: event.activity ?? agents[index].activity,
        detail: event.detail,
        turn: event.turn ?? agents[index].turn,
      };
    }
  } else if (event.type === "swarm_agent_finished" && index >= 0) {
    agents[index] = {
      ...agents[index],
      status: swarmAgentStatus(event.status),
      seconds: event.seconds ?? agents[index].seconds,
      report: event.report ?? agents[index].report,
    };
  }
  return { runId: run?.runId ?? event.run_id ?? "swarm", agents };
}

const SWARM_ACTIVITY_LABELS: Record<string, string> = {
  thinking: "thinking",
  web_search: "searching the web",
  web_fetch: "reading a page",
  read_file: "reading a file",
  list_directory: "listing a folder",
  grep_search: "searching code",
  source_outline: "outlining source",
  inspect_artifact: "inspecting a file",
  capture_screen: "looking at the screen",
  inspect_screen_region: "inspecting the screen",
  query_screen_text: "reading screen text",
  query_window_tree: "reading a window",
  list_windows: "listing windows",
  locate_visual_target: "locating a target",
  run_command: "running a command",
  manage_command: "managing a command",
  write_file: "writing a file",
  edit_file: "editing a file",
  undo_edit: "undoing an edit",
  team_post: "posting to the team board",
  team_read: "reading the team board",
  update_task_plan: "updating the plan",
  managed_browser_open: "opening a page",
  managed_browser_snapshot: "reading a page",
  user_browser_navigate: "opening a browser tab",
  fast_actions: "acting on the screen",
};

/** A short label for the tool or phase a swarm worker is in. */
export function swarmActivityLabel(activity: string | undefined): string {
  if (!activity) return "working";
  return SWARM_ACTIVITY_LABELS[activity] ?? activity.replaceAll("_", " ");
}

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
