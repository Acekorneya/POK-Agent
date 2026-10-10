# The agent swarm

POK-Agent's planner (System 2) works swarm-first. When a task has two or
more parts, or something to find out first (options, documentation, where
something is in a code base, which of several approaches works), it calls
`run_agents`: one worker agent per part or angle, all at the same time. The
workers report back, and the planner puts their findings together and
finishes the task. Only a single obvious step, such as opening one
application or running one known command, is done directly, because starting
agents for it would take longer than doing it.

Nothing about the swarm is configured per task. The planner decides how many
agents a task needs (up to 16) and what each one does.

## Roles

Each worker is a complete agent session: its own plan, its own System 1
router, the same safety gates, and the tools of its role.

| Role | For | Tools |
| --- | --- | --- |
| `research` | Finding things out on the web | `web_search`, `web_fetch`, the managed browser |
| `browse` | Pages in your own signed-in browser | research tools plus the `user_browser_*` tools, in a tab of its own |
| `desktop` | Work in one application window | desktop observation and input, the managed browser |
| `scout` | Studying an application, or how to do something in it, without changing anything | desktop observation (no input), the managed browser |
| `explore` | Reading and mapping code or documents | file reading and search |
| `code` | Editing code and running tests | file editing and commands, in its own git worktree |
| `review` | Checking work or a change | file reading and search |
| `general` | Anything else that does not edit files | research, browsing, desktop, and reading |

Scouts serve tasks in a single application: several scouts look at the
options at once (menus, settings, documentation) and report them, and the
planner chooses one and carries it out.

A code worker works in a git worktree of its own, on a `pok-ai/swarm-...`
branch, so several of them can change code at once without touching your
checkout. It returns its commit, branch, and diff. The planner merges or
cherry-picks it only after you agree. Asking for the change in your request
counts as agreeing.

## The team board

Workers talk to each other through a team board:

- **Findings, claims, questions, and answers.** A worker posts what a
  teammate or the planner needs (with its evidence), claims work before
  starting it so it is not done twice, and asks or answers a teammate by
  name. Posts reach teammates with their next tool result, so nobody has to
  stop and check.
- **Leads, like pheromone trails.** A promising direction is posted as a lead
  with a strength from 0 to 1. A lead's strength halves every two minutes
  unless teammates reinforce it by following it, so attention gathers on what
  keeps paying off and fades from what does not.
- **Recruitment, like foraging bees.** When a worker finishes, the strongest
  lead that is still unexplored (strength 0.55 or more) recruits a new worker
  to follow it up. A swarm recruits at most half its starting size, and never
  beyond 16 agents.

Board posts and reports come from the agents and from the pages and
applications they read, so the planner treats them as evidence to weigh, not
as instructions.

## Running many agents at once

- **Model requests.** A model on this computer (LM Studio, Ollama) gets four
  requests at once. A hosted model gets as many as there are workers, and
  whenever the provider says to slow down (a rate limit or overload) the
  swarm makes one fewer request at a time, down to one.
- **A smaller model for the workers.** `[swarm] worker_model` names a smaller,
  cheaper model on the same provider for the workers, while the planner keeps
  the main model. With a System 1 router, each worker also runs its plan
  trees through `fast_actions`, so most steps cost no planner call at all.
- **The desktop.** Reading the screen runs in parallel. Mouse and keyboard
  input takes turns: one action at a time across the swarm, and a window one
  worker is acting in is off limits to the others until it finishes. Input
  never moves while you are using the computer; it waits until you stop,
  as everywhere else in POK-Agent.
- **Browsers.** Every worker gets its own tab, in the managed browser and in
  your browser, and a finished worker's tabs are closed.
- **Limits.** A worker stops after its turn limit (by role, 25 to 50 turns
  unless the planner sets one) or after 20 minutes.

```toml
[swarm]
# worker_model = "qwen3-4b"   # unset: the workers use the main model
```

## Learning

A swarm run in which every agent completed is saved to memory as a playbook:
the goal, and how it was split into roles and tasks. The next time a similar
goal comes up, the two most relevant playbooks are recalled into the
planner's context at the start of the run (approved, enabled ones only), so
it splits the work the way that worked. In the desktop app a new playbook
waits for your approval in Memory, like other learned facts; the CLI saves it
approved. Each worker also
learns skills and motor programs from its own verified runs, exactly as a
single agent does.

## Safety

Workers run under the same policy as the planner, with one exception: a code
worker may write files and run commands only inside its own worktree, without
asking. In your own browser, a send, purchase, deletion, or account change by
any worker asks you first, and a worker whose action you decline reports it
instead of retrying. Workers cannot ask you questions; they act on the most
reasonable reading of their task and say what they assumed.
