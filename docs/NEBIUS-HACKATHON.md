# POK-Agent on NVIDIA Nemotron and Nebius Token Factory

Entry for the Nebius x NVIDIA Global AI Hackathon, track **Best Apps and Agents**.

POK-Agent is a Windows computer-use agent. A large model plans (System 2). A small, fast model
makes the bounded on-screen decisions (System 1): which button, is this step done, which branch.
The large model is called as rarely as possible, and repeated tasks become recorded "muscle
memory" that System 1 replays without the planner. The hackathon's own track text, "Nemotron 3
Ultra for serious reasoning, Nano or Super for the fast, everyday calls", is this design.

## How the entry uses the platform

| Piece | What runs there |
| --- | --- |
| Nebius Token Factory, `nvidia/Nemotron-3.5-Super-VL` | The planner. Reads window screenshots plus the numbered UI Automation/OCR targets and calls typed tools. |
| Nebius Token Factory, `nvidia/Nemotron-3-Ultra-550b-a55b` | Alternative planner for harder reasoning (text only, so it relies on the numbered targets rather than the screenshot). |
| Nebius Token Factory, `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`, `nvidia/nemotron-3-super-120b-a12b`, `nvidia/Nemotron-3_5-Lightning` | System 1. Typed choice questions (pick a target, is the condition true, is the task done) answered one letter at a time with a JSON-schema constrained reply, asked twice with the options rotated so position bias shows up as disagreement. |
| Tavily | `web_search` for research tasks (bonus track). |

Rust keeps everything that must stay deterministic: candidate construction, redaction of sensitive
fields, confidence gates, policy and approvals, input execution and post-action verification. A
model never acts directly; it only answers questions the harness asked.

## What is measured on Nemotron

`pok-ai router-bench` replays 57 questions from real captures (27 completion checks, 19 target
picks, 11 condition choices) through the production request code against Token Factory. Accepted
wrong means the confidence gates let a wrong answer through; the rest of the cases go back to the
planner.

| Model as System 1 | Targets | Completion | Conditions | Accepted wrong | Median latency |
| --- | --- | --- | --- | --- | --- |
| Nemotron-3_5-Lightning | 18 / 19 | 16 / 27 | 10 / 11 | 2 | 0.8 to 1.6 s |
| NVIDIA-Nemotron-3-Nano-30B-A3B | 18 / 19 | 23 / 27 | 10 / 11 | 4 | 1.0 to 2.1 s |
| nemotron-3-super-120b-a12b | 19 / 19 | 27 / 27 | 10 / 11 | 1 | 1.2 to 2.7 s |

Lightning is the fastest, Super the most accurate, Nano the middle. A separate smoke run had
`Nemotron-3.5-Super-VL` plan and finish a research task end to end (Tavily search, page fetch,
file written and verified, memory saved) in 9 model requests.

Earlier results that carry over, with a different planner and System 1: on
[Windows Agent Arena](https://github.com/microsoft/WindowsAgentArena) (154 tasks, clean Windows 11
VM) a recorded program replayed by System 1 cut planner calls 28% on average at the same success
(see the README). A full Arena run on Nemotron has not been done.

## What was built during the submission period

The repository's first commit is 2026-09-24 and v0.1.0 shipped 2026-09-30; v0.1.10 shipped
2026-10-17. For this entry specifically:

- `nebius` provider in the catalog (`crates/pok-ai-core/src/config.rs`) and in the desktop
  Settings; `pok-ai models --provider nebius --details` lists its models with vision, tools and
  reasoning levels.
- `scripts/llm_choice_sidecar.py --chat --api-key-env`: hosted chat endpoints as the System 1
  backend, with thinking switched off so a reasoning model answers instead of spending the token
  budget (see feedback below).
- `scripts/live_bench.py` arms `nemotron` (planner) and `nemotron_nano` (System 1) to compare
  planner calls with and without System 1 on live Windows tasks.
- Measured router-bench results above and the setup steps in the README.

## Reproduce

Setup is in the README section "NVIDIA Nemotron on Nebius Token Factory". In short:

```bash
export NEBIUS_API_KEY=...            # and optionally TAVILY_API_KEY
cargo run -p pok-ai-cli -- models --provider nebius --details
cargo run -p pok-ai-cli -- run --provider nebius --model nvidia/Nemotron-3.5-Super-VL "<task>"
```

Real screen capture, UI Automation and input need Windows. On Linux the same CLI runs as a coding
and research agent and the desktop tools report "unsupported".

## Where Token Factory helped

The planner and System 1 share one OpenAI-compatible endpoint and one key, so swapping Nano,
Lightning and Super under the same gates was a one-line model change, which is how the table above
was produced. Answers in about a second for a few-token reply are what make a hosted System 1
usable at all.

## Feedback on Nebius Token Factory and NVIDIA models

- **Capabilities in `/v1/models`.** The listing gives context length and a `modality` for some
  models, but not tool calling, vision, reasoning modes or structured-output support. Our client
  still shows "unknown" for Nano. A capabilities field would remove guesswork.
- **Switching thinking off differs by model.** `chat_template_kwargs: {"enable_thinking": false}`
  worked on Nano, Lightning and Super. `reasoning_effort: "none"` worked on Lightning and Super, but
  on Nano the answer landed in the reasoning field and `content` came back empty. With thinking on
  and a small `max_tokens`, `content` was empty on Nano and Lightning. A documented per-model
  switch would help.
- **Model ids.** Casing and punctuation vary (`nvidia/nemotron-3-super-120b-a12b`,
  `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`, `nvidia/Nemotron-3_5-Lightning`,
  `nvidia/Nemotron-3.5-Super-VL`). Consistent ids would prevent typos.
- **Availability.** During a short probe, a `nemotron-3-super-120b-a12b` call returned HTTP 529
  "worker overloaded" while other calls to it succeeded, so clients should retry.
- **What worked well.** JSON-schema constrained output on all three models, low latency, and
  Super's accuracy on completion checks (27 of 27 here). Nano misjudged
  "is it done?" more often than Super (23 of 27), which is exactly why the confidence gates stay
  high and uncertain answers go back to the planner.
