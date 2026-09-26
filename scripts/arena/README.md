# POK-Ai on Windows Agent Arena

This folder runs POK-Ai on [Windows Agent Arena](https://github.com/microsoft/WindowsAgentArena)
(WAA): 154 tasks across 12 Windows applications, each scored automatically,
in a clean Windows 11 VM. It serves three purposes:

1. **Comparable numbers.** Success rate, primary-LLM calls, and time per task, with and
   without a System 1 decision model, on a public benchmark.
2. **Self-improvement.** The same tasks run in several passes while POK-Ai keeps its
   memory, so later passes show whether learned skills make it faster and more reliable.
3. **Training data.** Router questions and their answers from every task, joined with
   the task's result, become a System 1 training set. The VM holds no personal data, so
   the set can be shared.

## How it works

```text
WSL host                        WAA container (Docker)         Windows 11 VM (QEMU/KVM)
run_arena.py ── docker run ──▶  WAA client run.py
                                  env.reset(task)  ─────────▶  task setup
                                  PokAiAgent.predict ─upload─▶ C:\pokai\runs\<id>\
                                                   ─execute─▶  run-task.ps1 → pok-ai.exe run
                                  (waits for exit.txt)  ◀────  \\host.lan\Data\pokai\out\<id>\
                                  env.evaluate()   ─────────▶  WAA evaluator checks the result
```

- POK-Ai runs **inside** the VM and observes and acts through UI Automation there.
  WAA still prepares and scores every task the way it does for any agent.
- The agent (`waa_agent/agent.py`) is installed into WAA as `mm_agents/pokai`.
  `setup_waa.py` adds a five-line patch to WAA's `run.py`: one to register the agent,
  one to tell it which task it is running.
- Each task's instruction gets one sentence telling POK-Ai how to report an impossible
  task (`INFEASIBLE:`). The agent turns that into WAA's `FAIL` action, which is the
  expected answer for WAA's 13 infeasible tasks. WAA's own agent has the same option.
- Every pass starts from a fresh copy of the golden VM disk. POK-Ai's memory lives on
  the arena share, not in the VM, so it carries across passes with `--memory persist`.
- API keys are read from Windows Credential Manager on the host. They pass through a
  private temporary env file into the container, and are uploaded to the VM per task
  (WAA's server does not log uploads). The in-VM wrapper deletes them after reading.
  Nothing with a key is written to the repository or to results.

## Setup (once)

Requirements: WSL2 with `/dev/kvm`, Docker, about 40 GB disk per golden image plus one
working copy per running pass, and 8 GB RAM per VM.

```bash
python3 scripts/arena/setup_waa.py install   # clone WAA (pinned), install the agent
python3 scripts/arena/setup_waa.py images    # pull windowsarena/winarena
# Download the Windows 11 Enterprise evaluation ISO from the Microsoft Evaluation
# Center and save it as <waa>/src/win-arena-container/vm/image/setup.iso
python3 scripts/arena/setup_waa.py prepare   # ~20 min; watch http://localhost:8006
python3 scripts/arena/setup_waa.py status
```

WAA is cloned to `~/arena/WindowsAgentArena` (override with `POKAI_WAA_DIR`).

## Running

```bash
# Two passes with memory kept: the second pass measures self-improvement.
python3 scripts/arena/run_arena.py run --llm bunny --arms none,jev \
    --tasks settings,notepad,windows_calc,clock --passes 2 --tag waa1

# The control: the same tasks with fresh memory for every task.
python3 scripts/arena/run_arena.py run --llm bunny --arms jev \
    --tasks settings,notepad,windows_calc,clock --passes 2 --memory fresh --tag waa1-fresh

python3 scripts/arena/run_arena.py summary --tag waa1
```

- `--tasks` accepts `all`, domain names (`settings`, `file_explorer`, `msedge`, …), or
  `domain/id`.
- `--llm` and `--arms` are the same names as `scripts/live_bench.py`. Arms that need a
  local decision-model sidecar (Laya, Zeiger, llm_choice) cannot reach the VM yet;
  `none` and `jev` work.
- Release `pok-ai.exe` is built automatically (`build-cli.ps1`), or pass `--exe`.

Results go to `~/arena/runs/<tag>/`:

| Path | Contents |
| --- | --- |
| `results.jsonl` | One row per task and pass: score, LLM calls, seconds, skills loaded, learning outcomes, skill counts |
| `<llm>-<arm>/share/out/<run>/` | `log.txt`, `trace.jsonl`, observation JSON/OCR, `router-training.jsonl` |
| `<llm>-<arm>/share/memory/main/` | The agent's memory after the latest pass (`memory.db`, `skills/`) |
| `<llm>-<arm>/pass<k>/container.log` | WAA client and VM log |

`summary` prints success, LLM calls, time, how often a skill was used, the skill count,
and duplicates (skills beyond one per task), then pass 1 against the last pass on the
same tasks.

## Training data

```bash
python3 scripts/arena/run_arena.py dataset --tag waa1 --output dataset/waa1
```

This joins every task's router records with the task's arena score. It builds a Laya
typed-decisions set (`scripts/build_router_dataset.py`) from proven labels, plus JEV's
confident answers from tasks that passed. Each row keeps its `outcome`, for further
filtering.
