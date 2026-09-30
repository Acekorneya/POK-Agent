# Brain Exam

The brain exam compares local models with identical tools, prompts, starting
files, temperature, seed (when honored), and turn limits. Models run
sequentially so they do not compete for GPU memory.

## Coding exam

```bash
cargo run -p pok-ai-cli -- --config pok-ai.toml exam \
  exams/coding-basic.json \
  --models google/gemma-4-26b-a4b-qat,google/gemma-4-12b-qat
```

Every run receives a private copy of the fixture. Assertions examine that copy,
not the model's prose. Reports are written as JSON and Markdown below the POK-Agent
data directory.

## Desktop exam

Run this from native Windows:

```powershell
cargo run -p pok-ai-cli -- --config pok-ai.toml exam `
  exams/desktop-form.json --models google/gemma-4-26b-a4b-qat
```

POK-Agent starts a loopback-only fixture server, opens the page in a dedicated Edge
app window, and allows unattended input only while that window and process are
foreground. The page writes submitted values to `state.json`; assertions grade
those values directly.

## Ranking

The primary rank is weighted assertion completion. Ties use fewer malformed tool
calls, then fewer total actions. Reports retain latency, tokens, retries, answer,
tool traces, command output, diffs, and screenshots for diagnosis.

The first validated run of `google/gemma-4-26b-a4b-qat` completed the calculator
repair, recovered from unavailable `pytest` and `python` commands, used `python3`,
applied an exact edit, and passed both tests.

