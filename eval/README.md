# Harness evaluation

A small, repeatable test of the agent harness with real models. It runs fixed
tasks with models of different sizes, with and without System 1, and scores each
run by checking the outcome: the final answer, or the state of the pages.

It answers questions like: does a small model finish the same task as a capable
one, how many planner calls and tokens does System 1 save, and does a learned
program replay correctly?

## Setup

1. Build the CLI: `cargo build -p pok-ai-cli`.
2. Set the API keys in your shell. They are passed to the CLI as environment
   variables and never written to a file:
   - `OPENROUTER_API_KEY` (all runs)
   - `TYPESAFE_API_KEY` (the System 1 modes)
3. For the managed-browser task, point `POK_AI_BROWSER_EXECUTABLE` at a script
   that starts Chromium. As root, Chromium needs `--no-sandbox`:

   ```sh
   #!/bin/sh
   exec /path/to/chrome --no-sandbox "$@"
   ```

4. For the user-browser tasks, start a Chrome with remote debugging on port 9333,
   using its own profile folder (any folder works):

   ```sh
   chrome --remote-debugging-port=9333 --user-data-dir=/path/to/eval-profile
   ```

   Log into nothing in it: the tasks use only fixture pages and public Wikipedia.
   The runner starts its own fixture server on port 8765 when it is not running.

## Run

```sh
python3 eval/run.py --out ~/pok-eval                # everything: 3 models x 3 tasks x 3 modes
python3 eval/run.py --out ~/pok-eval --dry-run      # print the plan only
python3 eval/run.py --out ~/pok-eval --task job_forms --mode s2
python3 eval/run.py --out ~/pok-eval --model small=nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free
```

Keep `--out` outside the repository. The folder holds each run's config, log,
workspace, and the page state, and the runner never deletes it.

## Tasks

| Task | Browser | What it checks |
| --- | --- | --- |
| `wiki_firefox` | user's browser, live Wikipedia | Follow links from Rust to Mozilla to Firefox; the answer gives Firefox's initial release date (November 9, 2004). |
| `job_forms` | user's browser, fixture job board | Fill two applications from `fixtures/resume.md`. Scored on 20 checks: six exact fields, two long text answers, the remote box, and "not submitted", for each form. |
| `shop_cheapest` | managed browser, fixture shop | The cheapest laptop across the shop (Aero 14, $899). |

All names, addresses and products in the fixtures are fictional.

## Modes

- `s2`: System 2 only (the planner calls the tools one at a time). Fresh memory.
- `s1`: System 1 (JEV, delegated) plus System 2. Fresh memory, so this run learns.
- `s1_replay`: the same task again in the `s1` folder, so System 1 can replay what
  it learned.

## Output

- `results.jsonl`: one record per run: model, task, mode, score, outcome, planner
  calls, tool calls, prompt and completion tokens, System 1 steps, wall time, the
  answer (first 500 characters), and the per-check detail.
- `summary.md`: one table of the same, printed at the end.
- Per run: `config.toml`, `run.log`, `ws/` (the workspace), `data/` (memory and
  traces), and `tabs.json` for the user-browser tasks.

## What it does not show

- **Small sample.** Three tasks, one attempt each. A single run can swing either
  way, so treat differences of one task as noise. Add repeats or tasks before
  drawing conclusions.
- **Live pages.** The Wikipedia task depends on the live article and on the
  network. Recheck the expected answer if the article changes.
- **Free models change.** OpenRouter's free list is rate-limited and may change
  without notice. Results are tied to the model IDs and dates they were run on.
- **Not local models.** The small tiers are small hosted models, used as stand-ins
  for models that run on a laptop. They are not the same as a local runtime.
- **Accuracy of the job answers.** The check verifies the fields and that nothing
  was sent. It does not judge whether the tailored answers are good.
