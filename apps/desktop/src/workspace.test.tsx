import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Appearance, MarkdownMessage } from "./components/workspace";
import { appendStreamDelta } from "./session-display";
import { App } from "./main";

const mock = vi.hoisted(() => ({ invoke: vi.fn(), listeners: new Map<string, (event: { payload: any }) => void>() }));
vi.mock("@tauri-apps/api/core", () => ({ invoke: mock.invoke }));
vi.mock("@tauri-apps/api/event", () => ({ listen: vi.fn((name, fn) => { mock.listeners.set(name, fn); return Promise.resolve(() => mock.listeners.delete(name)); }) }));
const conversation = { id: "saved", title: "Explore the project", workspace: "C:/project", model: "test-model", provider: "openai", status: "completed", updated_at: "2026-09-19T10:00:00Z" };
beforeEach(() => {
  localStorage.clear(); localStorage.setItem("pok_selected_provider", "openai"); localStorage.setItem("pok_selected_model", "test-model");
  document.documentElement.dataset.theme = "teal";
  Object.defineProperty(window, "matchMedia", { configurable: true, value: () => ({ matches: true, addEventListener() {}, removeEventListener() {} }) });
  HTMLElement.prototype.scrollIntoView = vi.fn(); HTMLElement.prototype.scrollTo = vi.fn();
  mock.listeners.clear(); mock.invoke.mockReset();
  mock.invoke.mockImplementation(async (name: string) => {
    switch (name) {
      case "get_status": return { provider: "openai", model: "test-model", base_url: "https://example.test", platform: "test", data_dir: "test", has_api_key: true, data_boundary: "external_service" };
      case "list_providers": return ["openai"];
      case "list_models": return ["test-model"];
      case "list_conversations": return [conversation];
      case "resume_conversation": return { summary: conversation, messages: [{ sequence: 1, sender: "user", text: "Earlier question" }, { sequence: 2, sender: "agent", text: "Earlier answer" }], next_before_sequence: null, has_more: false, decision_router_enabled: false, decision_activity: [] };
      case "list_memory_library": return { facts: [], skills: [] };
      case "list_generated_tools": case "list_generated_tool_candidates_command": return [];
      case "get_model_capabilities": return { id: "test-model", supported_parameters: [], reasoning_efforts: [] };
      case "get_decision_router_status": return {
        enabled: false,
        backend: "off",
        model: "jev-1.13.0",
        has_api_key: true,
        laya: { installed: false, enabled: false, running: false, phase: "not_installed", detail: "not installed", model: "laya", checkpoint: "", preference: "auto" },
        judge: { installed: false, enabled: false, running: false, phase: "not_installed", detail: "not installed", model: "zeiger", checkpoint: "" },
      };
      case "set_decision_router_backend": return {
        enabled: true,
        backend: "jev",
        model: "jev-1.13.0",
        has_api_key: true,
        laya: { installed: false, enabled: false, running: false, phase: "not_installed", detail: "not installed", model: "laya", checkpoint: "", preference: "auto" },
        judge: { installed: false, enabled: false, running: false, phase: "not_installed", detail: "not installed", model: "zeiger", checkpoint: "" },
      };
      default: return null;
    }
  });
});
afterEach(cleanup);
function emit(payload: Record<string, unknown>, name = "agent_event") { act(() => mock.listeners.get(name)?.({ payload })); }
async function renderApp() { render(<App />); await screen.findByText("Explore the project"); }

describe("workspace presentation", () => {
  it("persists each theme and restores the selected appearance", async () => {
    const user = userEvent.setup(); const view = render(<Appearance />);
    for (const [label, value] of [["Light", "light"], ["Neutral dark", "dark"], ["POK dark teal", "teal"]]) {
      await user.click(screen.getByRole("button", { name: label }));
      expect(document.documentElement.dataset.theme).toBe(value); expect(localStorage.getItem("pok_theme")).toBe(value);
    }
    view.unmount(); render(<Appearance />); expect(screen.getByRole("button", { name: "POK dark teal" }).getAttribute("aria-pressed")).toBe("true");
  });
  it("renders tables and code without raw HTML, remote images, or unsafe links", () => {
    const { container } = render(<MarkdownMessage text={'| A | B |\n|---|---|\n| 1 | 2 |\n\n```rust\nfn main() {}\n```\n\n<script>alert(1)</script>\n\n![private](https://example.test/track)\n\n[bad](javascript:alert(1))'} />);
    expect(container.querySelector("table")).not.toBeNull(); expect(container.querySelector("pre")?.textContent).toContain("fn main");
    expect(container.querySelector("script, img, a[href^='javascript:']")).toBeNull();
  });
  it("keeps streaming blocks ordered across reasoning and answers", () => {
    let messages = appendStreamDelta([], "reasoning", "Check ");
    messages = appendStreamDelta(messages, "reasoning", "the file");
    messages = appendStreamDelta(messages, "response", "Done");
    expect(messages.map(m => m.text)).toEqual(["Check the file", "Done"]);
    expect(new Set(messages.map(m => m.id)).size).toBe(2);
  });
  it("searches recent conversations and resumes the chosen session", async () => {
    await renderApp(); const user = userEvent.setup();
    await user.type(screen.getByRole("textbox", { name: "Search recent conversations" }), "missing");
    expect(screen.queryByText("Explore the project")).toBeNull();
    await user.clear(screen.getByRole("textbox", { name: "Search recent conversations" }));
    await user.click(screen.getByRole("button", { name: /Explore the project/ }));
    await screen.findByText("Earlier answer");
    expect(mock.invoke).toHaveBeenCalledWith("resume_conversation", { sessionId: "saved" });
  });
  it("restores tool activity in its original chronological position", async () => {
    const previous = mock.invoke.getMockImplementation()!;
    mock.invoke.mockImplementation((name, args) => name === "resume_conversation"
      ? Promise.resolve({ summary: conversation, messages: [
        { sequence: 1, subsequence: 0, sender: "user", kind: "prompt", text: "Earlier question" },
        { sequence: 2, subsequence: 0, sender: "system", kind: "activity", text: "Running capture_screen", tool: "capture_screen", arguments: {} },
        { sequence: 3, subsequence: 0, sender: "agent", kind: "response", text: "Earlier answer" },
      ], next_before_sequence: null, has_more: false, decision_router_enabled: true, decision_activity: [] })
      : previous(name, args));
    await renderApp();
    await userEvent.click(screen.getByRole("button", { name: /Explore the project/ }));
    const question = await screen.findByText("Earlier question");
    const tool = await screen.findByText("Ran capture_screen");
    const answer = await screen.findByText("Earlier answer");
    expect(question.compareDocumentPosition(tool) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(tool.compareDocumentPosition(answer) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });
  it("matches concurrent tool results by call ID and exposes failures when collapsed", async () => {
    await renderApp();
    emit({ type: "tool_started", name: "read_file", call_id: "a", arguments: { path: "one.rs" } });
    emit({ type: "tool_started", name: "read_file", call_id: "b", arguments: { path: "two.rs" } });
    emit({ type: "tool_finished", name: "read_file", call_id: "a", ok: false, detail: "File missing", result: {} });
    const rows = document.querySelectorAll(".tool-disclosure");
    expect(rows[0].className).toContain("activity-failed"); expect(rows[0].querySelector("summary")?.textContent).toContain("File missing");
    expect(rows[1].className).toContain("activity-running");
  });
  it("updates bounded command output and final status", async () => {
    await renderApp();
    emit({ type: "tool_started", name: "run_command", call_id: "cmd", arguments: { command: "cargo check" } });
    emit({ type: "command_progress", event: { call_id: "cmd", task_id: "task", kind: "output", chunk: "Checking workspace", total_bytes: 18, status: "running" } });
    expect(document.querySelector(".terminal-drawer")?.textContent).toContain("Checking workspace");
    emit({ type: "command_progress", event: { call_id: "cmd", task_id: "task", kind: "status", total_bytes: 18, status: "completed", exit_code: 0 } });
    expect(document.querySelector(".tool-disclosure")?.className).toContain("activity-done");
  });
  it("preserves approval decisions and emergency stop commands", async () => {
    await renderApp();
    emit({ id: "approval-1", tool: "run_command", reason: "Run the command?", arguments: { command: "cargo check" } }, "approval_requested");
    await userEvent.click(screen.getByRole("button", { name: "Allow once" }));
    expect(mock.invoke).toHaveBeenCalledWith("resolve_approval", { id: "approval-1", allow: true, rememberAlways: false });
    await userEvent.click(screen.getByRole("button", { name: /Emergency stop/ }));
    expect(mock.invoke).toHaveBeenCalledWith("emergency_stop");
  });
  it("does not submit while composing text", async () => {
    await renderApp(); const input = screen.getByRole("textbox", { name: "Message the agent" });
    fireEvent.change(input, { target: { value: "hello" } });
    fireEvent.keyDown(input, { key: "Enter", isComposing: true, keyCode: 229 });
    expect(mock.invoke.mock.calls.some(c => c[0] === "run_prompt")).toBe(false);
  });
  it("submits selected answers using the existing question contract", async () => {
    await renderApp();
    emit({ id: "request", questions: [{ id: "target", header: "Target", question: "Which folder?", options: [{ label: "Current folder", description: "Use the workspace" }], multi_select: false }] }, "question_requested");
    await userEvent.click(screen.getByRole("button", { name: /Current folder/ }));
    await userEvent.click(screen.getByRole("button", { name: "Continue" }));
    expect(mock.invoke).toHaveBeenCalledWith("resolve_question", { id: "request", answers: [{ id: "target", selected: ["Current folder"], custom: null }] });
  });
  it("retains retry and key setup when the connection needs attention", async () => {
    const previous = mock.invoke.getMockImplementation()!;
    mock.invoke.mockImplementation((name, args) => name === "get_status" ? Promise.reject(new Error("offline")) : previous(name, args));
    await renderApp();
    await userEvent.click(await screen.findByRole("button", { name: "Retry" }));
    expect(mock.invoke.mock.calls.filter(c => c[0] === "get_status").length).toBeGreaterThan(1);
    mock.invoke.mockImplementation((name, args) => name === "get_status" ? Promise.resolve({ provider: "openai", model: "test-model", requires_api_key: true, has_api_key: false }) : previous(name, args));
    await userEvent.click(screen.getByRole("button", { name: "Retry" }));
    await userEvent.click(await screen.findByRole("button", { name: /Connect Key/ }));
    expect(screen.getByText("Connect OPENAI API Key")).not.toBeNull();
  });
  it("keeps guidance, pause, and resume wired during an active task", async () => {
    await renderApp();
    const previous = mock.invoke.getMockImplementation()!;
    let finish!: (value: string) => void;
    mock.invoke.mockImplementation((name, args) => name === "run_prompt" ? new Promise<string>(resolve => { finish = resolve; }) : previous(name, args));
    const input = screen.getByRole("textbox", { name: "Message the agent" });
    fireEvent.change(input, { target: { value: "Check the project" } });
    await userEvent.click(screen.getByRole("button", { name: "Run task" }));
    expect((screen.getByRole("button", { name: /Explore the project/ }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.change(input, { target: { value: "Start with the README" } });
    await userEvent.click(screen.getByRole("button", { name: "Send Guidance" }));
    expect(mock.invoke).toHaveBeenCalledWith("send_guidance", { message: "Start with the README" });
    await userEvent.click(screen.getByRole("button", { name: "Pause agent" }));
    expect(mock.invoke).toHaveBeenCalledWith("pause_agent");
    emit({ type: "paused" });
    fireEvent.change(input, { target: { value: "I closed the other window" } });
    await userEvent.click(screen.getByRole("button", { name: "Resume agent" }));
    expect(mock.invoke).toHaveBeenCalledWith("resume_agent", { note: "I closed the other window" });
    await act(async () => finish("Done"));
  });
  it("keeps sending disabled during a local model transition", async () => {
    await renderApp();
    emit({ provider: "openai", model: "test-model", phase: "loading", instance_ids: [] }, "model_runtime_event");
    fireEvent.change(screen.getByRole("textbox", { name: "Message the agent" }), { target: { value: "hello" } });
    expect((screen.getByRole("button", { name: "Run task" }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.keyDown(screen.getByRole("textbox", { name: "Message the agent" }), { key: "Enter" });
    expect(mock.invoke.mock.calls.some(c => c[0] === "run_prompt")).toBe(false);
  });
  it("keeps model and permission selections readable and selectable", async () => {
    await renderApp();
    const modelSelect = screen.getByRole("combobox", { name: "Model" }) as HTMLSelectElement;
    const permissionSelect = screen.getByRole("combobox", { name: "Permission mode" }) as HTMLSelectElement;
    expect(modelSelect.value).toBe("test-model");
    expect(permissionSelect.value).toBe("interactive");
    await userEvent.selectOptions(permissionSelect, "autonomous");
    expect(permissionSelect.value).toBe("autonomous");
    expect(screen.getByRole("option", { name: "Autonomous" }).getAttribute("value")).toBe("autonomous");
  });
  it("switches the router per conversation and shows bounded decision activity", async () => {
    await renderApp();
    const router = screen.getByRole("combobox", { name: "Decision router" }) as HTMLSelectElement;
    await userEvent.selectOptions(router, "jev");
    await waitFor(() => expect(router.value).toBe("jev"));
    emit({ type: "decision_router_started", turn: 1, purpose: "next_action", candidate_count: 7 });
    expect(await screen.findByText("JEV is evaluating options")).not.toBeNull();
    emit({ type: "decision_router_evaluated", turn: 1, purpose: "next_action", eligible: false, tool: "fallback", selected_probability: 0.5, confidence: 0.6, probability_threshold: 0.8, confidence_threshold: 0.75, elapsed_ms: 42, rejection_reason: "low_confidence", alternatives: [] });
    expect(await screen.findByText("JEV deferred to the main model")).not.toBeNull();
    expect(screen.getByText(/Deferred this step/)).not.toBeNull();
    emit({ type: "decision_router_cache_hit", turn: 2, purpose: "next_action", candidate_count: 7 });
    expect(screen.queryByText(/Deferred this step/)).toBeNull();
  });
  it("labels decision activity with real trace payloads (done, context routing, judge)", async () => {
    await renderApp();
    const router = screen.getByRole("combobox", { name: "Decision router" }) as HTMLSelectElement;
    await userEvent.selectOptions(router, "jev");
    await waitFor(() => expect(router.value).toBe("jev"));

    // Real next_action pick: friendly label, no internal ids leaked.
    emit({ type: "decision_router_evaluated", turn: 1, purpose: "next_action", eligible: true, tool: "list_windows", candidate_id: "list_visible_windows", description: "List visible desktop windows", selected_probability: 1.0, operation_probability: 1.0, target_probability: 1.0, confidence: 1.0, operation_confidence: 1.0, target_confidence: 1.0, probability_threshold: 0.3, confidence_threshold: 0.75, alternatives: [["list_visible_windows", 1.0], ["get_current_time", 0.2]], elapsed_ms: 0 });
    expect(await screen.findByText("JEV selected a candidate")).not.toBeNull();

    // Real terminal DONE decision: "Router done", never "jev_done"/"operation:DONE".
    emit({ type: "decision_router_evaluated", turn: 1, purpose: "next_action", eligible: true, tool: "__done__", candidate_id: "jev_done", description: "Current verified evidence is sufficient", selected_probability: 1.0, operation_probability: 0.61, target_probability: 1.0, confidence: 0.04, probability_threshold: 0.3, confidence_threshold: 0.6, alternatives: [["jev_done", 1.0], ["operation:DONE", 0.61], ["operation:BLOCKED", 0.39]], elapsed_ms: 12 });
    expect(await screen.findByText("JEV finished (evidence sufficient)")).not.toBeNull();
    expect(screen.getAllByText("Router done").length).toBeGreaterThan(0);
    expect(screen.queryByText("jev_done")).toBeNull();
    expect(screen.queryByText("operation:DONE")).toBeNull();

    // Optional-context routing that stays local: no confusing "deferred" row.
    emit({ type: "decision_router_evaluated", turn: 1, purpose: "retrieval_intent", eligible: false, selected_probability: 0.0, probability_threshold: 0.3, elapsed_ms: 146, rejection_reason: "local_intent_fallback", alternatives: [] });
    expect(screen.queryByText(/deferred to the main model/i)).toBeNull();
    expect(screen.queryByText(/local intent ranking/i)).toBeNull();

    // Context routing that actually injected context: positive announcement.
    emit({ type: "decision_router_evaluated", turn: 1, purpose: "context_selection", eligible: true, tool: "include_context", candidate_id: "archive_0", description: "Workspace archive context", selected_probability: 0.9, probability_threshold: 0.3, elapsed_ms: 20, alternatives: [] });
    expect(await screen.findByText("JEV added optional context")).not.toBeNull();

    // Judge promotion: approved row plus verdict card, not "JEV done".
    emit({ type: "decision_router_judge_attempted", turn: 1, tool: "click_target", candidate_id: "click_9", promoted: true, reused: false, votes: { reversible: "yes", cheap: "yes", evidenced: "yes", comparative: "a" }, reason: "target_below_probability_threshold" });
    expect(await screen.findByText("Zeiger approved: click_target")).not.toBeNull();
    expect(screen.getByText(/safe, cheap and evidenced/i)).not.toBeNull();
  });
  it("stops following output when scrolled back and offers jump to latest", async () => {
    await renderApp(); const feed = document.querySelector(".console-container")!;
    Object.defineProperties(feed, { scrollHeight: { configurable: true, value: 1500 }, clientHeight: { configurable: true, value: 400 }, scrollTop: { configurable: true, writable: true, value: 0 } });
    fireEvent.scroll(feed); await screen.findByRole("button", { name: /Jump to latest/ });
    emit({ type: "tool_started", name: "read_file", call_id: "a", arguments: {} });
    expect(feed.scrollTop).toBe(0);
    await userEvent.click(screen.getByRole("button", { name: /Jump to latest/ }));
    await waitFor(() => expect(screen.queryByRole("button", { name: /Jump to latest/ })).toBeNull());
  });
});
