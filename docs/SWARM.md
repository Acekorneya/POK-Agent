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

- **Closing ranks.** Before a worker's report is accepted, it hears what
  its teammates posted that bears on its part (the latest eight findings and
  leads, each cut to a few hundred characters, with the author and source)
  and any question addressed to it that it has not answered. It revises once:
  it keeps what its own sources support, says which teammate's finding adds
  to, confirms, or contradicts its part, and, when two sources disagree,
  reports both and which it trusts instead of deferring. A worker that has
  nothing waiting for it finishes without an extra model call. Turn it off
  with `[swarm] closing_ranks = false`.
- **The shield.** A worker that fails, is cancelled, or runs out of time hands
  back the findings it had already posted (its last six) with its report, so
  its work is not lost with it. The planner is told how many reports were
  rebuilt this way.

`run_agents` returns a `hive` summary (posts, findings, how many agents
posted, closing reviews sent, salvaged reports) and `grounded_agents`, the
number of finished workers that looked something up or read a page
themselves. The planner relays grounded reports instead of repeating the
lookups, and still checks for itself when no worker was grounded.

Why the board works this way:

- Agents that only write to a shared board and act on its contents work with
  fewer tokens than agents that message each other
  ([LbMAS](https://arxiv.org/abs/2507.01701)), so the board stays the only
  channel and it is bounded.
- Agents that read each other's answers and revise are more accurate
  ([multiagent debate](https://arxiv.org/abs/2305.14325)), but not everyone
  needs to hear everyone ([sparse topologies](https://arxiv.org/abs/2406.11776)):
  the revise round happens once, on bounded intel.
- An orchestrator that checks the workers' output contains errors far better
  than independent agents do (4.4x against 17.2x amplification), and parallel
  work gains while sequential work loses
  ([scaling study](https://research.google/blog/towards-a-science-of-scaling-agent-systems-when-and-why-agent-systems-work/)):
  workers stay on parallel parts, and the planner stays the validator.
- Duplicated work, vague tasks, and noisy updates are the common failures
  ([Anthropic's research system](https://www.anthropic.com/engineering/multi-agent-research-system)):
  claims, bounded notes, and findings kept on the board address them.

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
  unless the planner sets one) or after 20 minutes. At the turn limit its
  tools are withdrawn and it reports what it found, so a model that loops
  cannot run on; a worker that reads the team board over and over with
  nothing done in between is stopped the same way, and `run_agents` returns
  how many turns each worker used.

```toml
[swarm]
# worker_model = "qwen3-4b"   # unset: the workers use the main model
```

## Testing a swarm live

Unit tests cover the board, the guards and the limits with scripted models.
What they cannot show is how a real model behaves in a swarm, so a change to
the swarm is also tried live and read back from the traces.

**Run it.**

- Use the desktop app or the CLI with a hosted model that is cheap to repeat
  (for example Nebius Token Factory), and a task whose parts can run side by
  side. Two that work well: a broad request such as "any news today I should
  know about" (the planner splits it into lanes such as world, local and
  financial news), and a request whose lanes depend on each other, such as
  "find the current stable Rust release, the minimum supported Rust version of
  the latest tokio, and of the latest axum, and say whether the first
  satisfies both".
- To compare two settings (for example `[swarm] closing_ranks` on and off),
  run each from its own copy of `pok-ai.toml` whose `data_dir` and
  `diagnostics_dir` point at scratch folders, so memory, playbooks and traces
  never carry over from one arm to the other. Change one setting per arm, use
  the same task and models, and repeat: one run per arm is a signal, not a
  result.
- Keep API keys in the environment or the credential manager and out of
  configuration files and logs. Use tasks that carry no private information.
- To stop a swarm that misbehaves, press the stop button or `Ctrl+Alt+Esc`.

**Read it back.** Every run writes `trace.jsonl` in its session folder under
the diagnostics directory, and every worker writes its own beside the
planner's at `swarm/<run>/<agent>/trace.jsonl`.

- In the planner's trace, the `run_agents` `tool_result` holds one entry per
  worker (`status`, `grounded_evidence`, `turns`, `report`), the `hive`
  summary (posts, findings, agents posting, closing reviews, salvaged
  reports), `grounded_agents`, and the team board.
- In a worker's trace, count `model_response` events per tool: a worker that
  is working searches, reads and posts; one that is stuck repeats a single
  tool. Also look for `completion_review` (a closing review was sent),
  `turn_limit_reached`, `repeated_lookup`, `plan_loop_recovery`, and
  `fresh_evidence_observed`. `execution_epoch_extended` should never appear in
  a worker.
- In the planner's trace, check that a swarm with grounded workers is
  followed by an answer and not by `completion_execution_incomplete`, which
  would mean the planner went back to repeat the lookups.

**What a live run found (November 2026).** A broad news request, with one
small model as both planner and workers, showed three problems that no
scripted test had: workers polled the team board 130 times and rewrote their
plan 79 times instead of finishing; the turn limit did not stop them,
because reaching it only started a new execution epoch; and one worker's
"final answer" was a run of garbage tokens. The fixes are in this document
under Limits (a hard turn limit for workers, and a board-polling loop that
withdraws the worker's tools). The garbage answer was sent back to work only
because a closing review happened to fire on it; nothing yet checks that a
worker's report is readable. A swarm's weak spot is a weak model looping,
so judge a change by what the slowest, least reliable worker does, not by
the best one.

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
