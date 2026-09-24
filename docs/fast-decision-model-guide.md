# Fast Decision Models: How They Work and How to Build With Them

A self-contained guide for building systems where a large LLM plans and a small,
fast "decision model" makes the step-by-step choices. It is written from a
working Windows computer-use agent (POK-Ai) and its measured results, and
generalizes to games, bots, triage, routing and any loop that repeats small
decisions.

---

## 1. What a fast decision model is

A fast decision model (also called a typed decision model or "System 1" model)
does not generate text. You give it:

- a **state**: text or JSON describing the current situation, and
- one or more **typed questions**, each with a fixed set of options.

It returns a **probability distribution over the options you supplied**, and
nothing else. It cannot invent an answer outside your options, and it answers in
about 100–1000 ms on one GPU, or on a hosted API.

Question types:

| Type | Asks | Returns |
|---|---|---|
| `choice` | Pick one option from a map of `{option_id: description}` | `choice`, `probabilities` per option, `confidence` |
| `noul` | Is this statement true? | `noul`: a probability from 0 to 1 |
| `score` | Where does the state sit on an ordered scale of levels? | `score`, `legend`, `probabilities`, `confidence` |

Known models (checked September 2026; confirm licenses before shipping):

| Model | Where it runs | Notes |
|---|---|---|
| JEV (TypeSafe) | Hosted API `https://api.typesafe.ai/v1/systemone` | Most accurate in our tests; answers several questions in parallel |
| kev-4b (`jaredpalmer/kev-4b`) | Local, about 4B parameters, Apache-2.0 | Closest local match to JEV; needs a lot of VRAM |
| OpenJev v5 | Local, MIT | Accurate but slow (about 3.7 s per pick) |
| SemIf | Local, MIT; turns any chat LLM into a letter-logit scorer | Good accuracy; one forward pass per question |
| Zeiger 0.6B, Laya | Local, very small and very fast | Weaker alone; fine for yes/no checks and small menus |
| AgentJev 0.6B | Local, Apache-2.0 | Middling accuracy |

Official JEV docs: https://docs.typesafe.ai/introduction (primitives,
confidence, and patterns pages).

---

## 2. The API contract

Every backend in this guide sits behind the same contract. Put local models
behind a small loopback HTTP "sidecar" that speaks it, so you can swap models
without code changes.

**Request**

`POST /v1/systemone` with the header `Authorization: Bearer <key>`:

```json
{
  "model": "jev-1.13.0",
  "state": {
    "goal": "Open Display settings",
    "window": {"title": "Settings", "application": "SystemSettings.exe"},
    "visible": ["System", "Bluetooth & devices", "Display", "Sound"]
  },
  "questions": {
    "target": {
      "type": "choice",
      "instructions": "Which control should be clicked next to reach the goal? Judge only the supplied state.",
      "criteria": {
        "c0": "\"System\" (list item)",
        "c1": "\"Display\" (list item)",
        "c2": "\"Sound\" (list item)"
      }
    },
    "satisfied": {
      "type": "choice",
      "instructions": "Does the state show that this is already true: the Display page is open?",
      "criteria": {"yes": "Clearly true", "no": "Clearly not yet true", "unknown": "Not enough evidence"}
    }
  }
}
```

**Response**

```json
{
  "model": "jev-1.13.0",
  "answers": {
    "target":    {"type": "choice", "choice": "c1", "confidence": 0.93, "probabilities": {"c0": 0.04, "c1": 0.95, "c2": 0.01}},
    "satisfied": {"type": "choice", "choice": "no", "confidence": 0.90, "probabilities": {"yes": 0.03, "no": 0.94, "unknown": 0.03}}
  }
}
```

**Rules for the contract**

- Questions are independent: one answer never feeds into another. Evaluate
  them all against the same state.
- The sidecar must echo the requested `model` name. Reject any answer whose
  model does not match what you configured.
- Keep the request bounded (we cap it at 32 KiB). Send only the state the
  question needs.
- Filter out sensitive text (passwords, tokens) before anything leaves the
  process.

---

## 3. The architecture: planner plus fast executor

```text
Big LLM (planner, slow, expensive; runs every few steps)
   │  writes a short plan: steps, exact target labels, done-conditions,
   │  branches, things to avoid, popups to handle
   ▼
Executor loop (your code)
   ├─ build the current state from real data (UI tree, OCR, game state, API)
   ├─ answer everything it can LOCALLY first (exact matches, rules)
   ├─ ask the fast model only small fuzzy questions, all in ONE request
   ├─ act only when confidence clears the gate for that action's risk
   ├─ verify the action changed something
   └─ hand control back to the planner when uncertain, stuck or done
```

The planner is the expensive part. The goal is fewer planner calls per
successful task. In our agent the executor also does the clicking itself, so
the planner never spends a turn on a routine click.

### The plan format (condition language)

Let the planner write conditions the executor can check cheaply:

- **Quoted strings are exact evidence.** `done_when: heading "Advanced display"
  is visible` is checked locally by looking for that exact text. No model call
  is needed.
- **Unquoted conditions are fuzzy.** `done_when: the settings page for the
  monitor is open` is sent to the fast model as a yes/no/unknown question.
- The plan tree can hold: `then` (next steps), `branches` (a `when` condition
  plus steps, with an `otherwise` default), `on_interrupt` (e.g. "if a cookie
  banner is visible, click its quoted button, or stop"), `avoid` (phrases never
  to act on), and `read` (values to extract so the planner can answer without
  another look).
- Bound it: depth ≤ 3, ≤ 16 nodes, ≤ 4 branches and ≤ 4 interrupt rules.

Example:

```json
{
  "goal": "Open Advanced display",
  "target_hint": "list item \"System\"",
  "done_when": "breadcrumb \"System\" is visible",
  "then": [
    {"goal": "Open Display", "target_hint": "\"Display\"", "done_when": "breadcrumb \"Display\" is visible"},
    {"goal": "Open Advanced display", "target_hint": "\"Advanced display\"",
     "done_when": "heading \"Advanced display\" is visible", "allowed_operations": ["click", "scroll"]}
  ],
  "avoid": ["voice channel"],
  "on_interrupt": [{"when": "\"Accept cookies\" is visible", "target_hint": "\"Accept\""}]
}
```

---

## 4. Principles that made it work (measured)

1. **Ask small, well-posed questions.** These models are strong at choosing
   among 2–16 clear options and weak at "pick 1 of 30 noisy options". In our
   tests, a small model rated the wrong item 0.87 and the right one 0.07 on a
   crowded Settings sidebar. Rank the candidates locally, then offer at most 16.
2. **Ground first, ask second.** Check exact labels, rules and legal moves in
   code. Ask the model only what code cannot decide. In our agent, local checks
   settled 164 of 187 completion checks, with no model call.
3. **Require agreement.** If the model confidently picks something the local
   evidence clearly contradicts (for example, the best local match scores ≥ 0.8
   and the pick scores < half of that), hand it back instead of acting.
4. **Fan out.** Send every question a step might need in one request: the
   target pick, "is it already done?", and "is a popup in the way?". Ignore the
   answers you don't use. This is the TypeSafe "speculative fan-out" pattern,
   and it helped every backend we tested (section 7).
5. **Gate by risk.** Use the TypeSafe confidence tiers:
   - **Below 0.5:** hand back.
   - **0.5 and up:** fine for read-only actions (scroll, look, read).
   - **Stricter gates (0.75–0.9):** for state-changing actions.
   - **Irreversible or consequential actions** (send, submit, buy, delete):
     never done by the fast model; they go back to the planner.
6. **Keep the operation choice out of the model's hands.** If one step allows
   only one kind of action (e.g. click), don't ask "which operation?". That
   question is weak and adds noise.
7. **Make "target not visible" a different, smaller question.** If the planner
   names a target that is not on screen, offer only scroll directions, not the
   visible unrelated controls. If the model is unsure, default to a safe
   read-only move (scroll down one page).
8. **Verify after acting, and be honest about no-ops.** Compare the state
   before and after. If nothing changed, report it ("clicked, but nothing
   changed") instead of success.
9. **Hand back with context.** When the executor stops (uncertain, stalled, no
   candidates, budget used up), return the reason, the top alternatives with
   their probabilities, and the fresh state. The planner can then decide in one
   call.
10. **Text state is enough.** None of our fast models use vision. The screen
    became text (UI Automation tree + OCR + window titles, or the page URL,
    title and visible text), and that was sufficient.

---

## 5. Traps we hit (and the fixes)

Each of these cost more planner calls than the gap between any two models.
Check for them first:

| Trap | Symptom | Fix |
|---|---|---|
| The done-condition quotes the target's own label | "Done" before clicking, because the link text is already on screen | Before acting, a quoted label that only names the target is not evidence. After acting, require the screen to have changed, or the label to appear in a window or page title. |
| Click reported OK but nothing happened | The planner told "done" repeatedly (one run took 76 planner calls) | Compare the state before and after; report "stalled: nothing changed". |
| Duplicate option IDs | The backend rejects the request | De-duplicate candidates by ID before sending. |
| Used options removed from the menu | The scroll search had to scroll back up | Let read-only moves repeat; stop on no progress instead. |
| Local validation errors counted as backend failures | The circuit breaker suspended a healthy model | Count only real transport and response failures. |
| Timeout too short for a hosted model | Random "unavailable" steps | Give hosted backends about 3–10 s; network latency spikes happen. |
| The tool catalog lists withheld tools as active | The planner tried workarounds with the wrong tools | Label locked tools explicitly ("locked: use fast_actions"). |
| A finished browser step returns no state | The planner spends a call taking a snapshot | Return a compact state with every "done". |

---

## 6. Implementation recipe (pseudocode)

```python
def run_plan_node(node, router, env, budget):
    start = env.state()
    for step in range(node.max_steps + 1):
        state = env.state()

        # 1. Popup rules: quoted ones are checked locally, fuzzy ones deferred
        #    so they join this step's single request
        interrupt_q = fuzzy_rules(node.on_interrupt, state)

        # 2. Completion: quoted conditions checked locally
        done = grounded(node.done_when, state)
        if done is None:
            completion_q = node.done_when        # fuzzy: ask the model
        elif done and not only_names_target(node, state, start):
            return "done"
        else:
            completion_q = None

        # 3. Candidates: legal actions, filtered by avoid, commits and risk,
        #    ranked locally; at most 16
        cands = rank(legal_actions(state, node.allowed), node.target_hint)[:16]
        if target_off_screen(node.target_hint, state):
            cands = [c for c in cands if c.read_only]   # scroll only
        exact = exact_match(cands, node.target_hint)
        if exact:
            act(exact)
            continue                             # no model needed

        # 4. One fan-out request
        ans = router.ask(state, pick=cands, completion=completion_q, condition=interrupt_q)
        if ans.condition_fired:
            handle_interrupt()
            continue
        if ans.completion_yes:
            return "done"
        pick = ans.pick
        gate = 0.5 if pick.read_only else 0.75
        if pick.prob < gate or contradicts_local(pick, cands):
            return hand_back("uncertain", alternatives=top3(ans))

        before = env.state()
        act(pick)
        if env.state() == before:
            return hand_back("stalled", "action changed nothing")
    return hand_back("budget_exhausted")
```

Sidecar pattern for local models: a small FastAPI/uvicorn server on
`127.0.0.1` with a bearer token. It translates `/v1/systemone` to the model's
native call and echoes the model name. Load one model onto the GPU at a time,
and unload it before loading the next.

---

## 7. Measured results (POK-Ai, September 2026)

**Offline question bench:** 57 questions captured from real agent runs (19
target picks, 27 completion checks, 11 condition choices). Results are the
number correct, plus wrong answers given with high confidence.

| Model | Correct | Confident wrong |
|---|---|---|
| JEV | 57/57 | 0 |
| kev-4b | 53/57 | 2 |
| OpenJev v5 | 53/57 | 2 |
| SemIf | 51/57 | 3 |
| AgentJev | 40/57 | 12 |
| Zeiger R17 | 34/57 | 13 |
| Laya | 18/57 | 14 |

General chat LLMs used as choosers (2–8B, via logprobs): 26–39/57. That's
worse than the purpose-built models.

**Fan-out:** the same three questions sent as 3 calls versus 1 request, over 19
real states. Picks were identical in every case.

| Model | 3 calls | 1 request | Saved |
|---|---|---|---|
| JEV | 520 ms | 151 ms | 71% |
| kev-4b | 973 ms | 417 ms | 57% |
| Zeiger | 234 ms | 169 ms | 28% |
| Laya | 168 ms | 132 ms | 21% |

**Live agent:** real Windows tasks (a Rust Book page in a managed browser, and
File Explorer to Program Files), averaged per task. 24 of 24 runs succeeded.

| Setup | Planner calls | Time | Prompt tokens |
|---|---|---|---|
| No fast model | 9.3 | 44 s | 191k |
| Local checks only, no model | 7.5 | 39 s | 151k |
| JEV | 6.5 | 36 s | 134k |
| Laya | 6.5 | 35 s | 130k |
| Zeiger | 6.8 | 37 s | 157k |
| kev-4b | 7.0 | 38 s | 131k |

What these numbers show:
- **Planner calls:** up to 30% fewer. Wall time fell up to 20% and tokens up to 32%.
- **Where the gain comes from:** plan structure plus local checks account for
  much of it. The fast model adds the rest.
- **Model choice in live use:** with 2 runs per task, the models are roughly
  tied. The offline bench separates them more clearly.

**Fair-test rules we used:**
- Close windows left from the previous run, so every run starts from the same state.
- Load one local model at a time.
- Fix only harness bugs, never model mistakes.
- After any harness fix, rerun every model from scratch on the same build.

---

## 8. Applying this to games and other projects

**Games** (turn-based, RPG, strategy, or real-time at a fixed tick):

- **State:** compact JSON from the game engine, not pixels. Include HP,
  position, visible enemies, inventory, cooldowns and objective progress.
- **Candidates:** the engine lists the legal moves. That is your grounding,
  and the model only chooses among real actions. Rank them by simple
  heuristics and send at most 16.
- **Planner (LLM):** runs every N turns or on a new situation. It writes the
  strategy as a plan tree, for example:
  - goal: "clear the camp";
  - `done_when`: "no enemies within 10 tiles";
  - `branches`: "if HP < 30%: retreat and heal";
  - `avoid`: "attack the boss";
  - `on_interrupt`: "if a trade window is open: close it".
- **Per tick, one fan-out request:**
  - a `choice` of the next action from the legal moves;
  - a `noul` for "am I in danger?";
  - a `score` for threat level;
  - a `choice` for which branch holds.
- **Risk tiers:**
  - movement and looking: act at ≥ 0.5;
  - using consumables or attacking: ≥ 0.75;
  - irreversible choices (selling rare items, story choices, spending premium
    currency): back to the planner.
- **Verify:** if the action didn't change the game state, count it as a
  setback. After two setbacks, hand back to the planner.

**Other uses:**
- **Support triage:** one fan-out gives category, urgency, refund intent and
  frustration.
- **Content moderation:** yes/no checks for each policy clause.
- **Intent routing:** choose which handler or agent gets a request.
- **Code agents:** "is the test output a pass?" or "which file is relevant?",
  given a small ranked list.

**Picking a model:**
- **Best accuracy, network allowed:** JEV.
- **Best local accuracy:** kev-4b (needs a lot of VRAM).
- **Cheapest, fastest local:** Zeiger or Laya. Use them for yes/no questions
  and small menus, backed by strong local checks.

Whichever model you pick, measure with the same two tools: an offline question
bench built from your real states, and a live A/B that counts planner calls per
successful task.
