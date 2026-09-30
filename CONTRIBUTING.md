# Contributing to POK-Agent

Thanks for helping. Bug reports, trace-backed fixes, new tests, and benchmark
results are all welcome.

## Before you start

- For anything larger than a small fix, open an issue first so the approach can
  be agreed before you spend time on it.
- Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) ("Where the code lives")
  to find the module that owns the behavior you want to change.

## Building

WSL or Linux (core logic, mock desktop, frontend):

```bash
./scripts/setup-wsl.sh
./scripts/check-wsl.sh
```

Native Windows (capture, OCR, UI Automation, input, installers):

```powershell
.\scripts\setup-windows.ps1
.\scripts\check-windows.ps1
.\scripts\run-windows.ps1 -Mode Dev
```

## Before opening a pull request

Run the same checks as CI:

```bash
cargo fmt --all
cargo clippy -p pok-ai-core -p pok-ai-cli -p pok-ai-windows -p pok-ai-voice --all-targets -- -D warnings
cargo test -p pok-ai-core --lib
cargo test -p pok-ai-cli
npm --prefix apps/desktop test
npm --prefix apps/desktop run build
```

If you changed Windows-facing behavior, also run `.\scripts\check-windows.ps1`
and say in the pull request what you tested on a real desktop.

## Ground rules

- **Keep behavior generic.** Fix the underlying cause, not one website, app,
  window title, or model. Reproduction-specific names belong in tests only.
- **Add a regression test** that reproduces the bug's exact shape (parsing,
  grounding, input verification, scrolling, policy, router thresholds,
  compaction).
- **Never weaken safety**: approvals, workspace containment, observation
  freshness, password/elevation checks, and the call guard stay intact.
- **Use fake test data only**: `example.com`, `555-01xx` phone numbers,
  "Example Guild". Never real names, emails, or screen text from a session.
- **Never commit** API keys, traces (`diagnostics/sessions`), screenshots,
  OCR dumps, training logs, or documents.
- **Benchmarks must stay fair**: fix harness bugs, never a model's mistakes,
  and rerun affected arms from the start.
- Agent reasoning, tools, and policy live in Rust; React only displays state
  and sends commands.

## Licence

By contributing you agree that your work is licensed under the
[Apache License 2.0](LICENSE).
