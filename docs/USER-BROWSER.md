# Your own signed-in browser

POK-Agent can work in the Chrome or Edge you are already signed in to: your
mail, orders, documents, and dashboards, with your cookies, single sign-on,
and two-factor state. It attaches to the running browser over the Chrome
DevTools Protocol (CDP), the same mechanism as Playwright's
`connect_over_cdp`. Nothing is exported or copied: cookies and saved
passwords stay inside the browser, which sends your session with each request
the agent makes, exactly as it does for you.

The isolated managed browser (`managed_browser_*`) is still the default for
anonymous web work. The `user_browser_*` tools are for tasks that need your
accounts, or when you ask for your own browser.

## Start your browser with remote debugging

The agent never launches, restarts, or closes your browser. Start it yourself
with a DevTools port (PowerShell):

```powershell
& "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="$env:LOCALAPPDATA\POK-Ai\POK-Ai\data\browser\profiles\personal"
```

Run `pok-ai doctor` to print the exact command for your machine (it finds
your Chrome or Edge and fills in the profile folder from your configuration).
When the browser is not reachable, the agent's `user_browser_status` tool
gives the same command, and the agent relays it to you instead of retrying.

Chrome 136 and later ignore `--remote-debugging-port` on the default profile,
so the command names a separate `--user-data-dir`. Sign in to your sites in
that window once; the profile remembers them next time. If that profile is
already open without remote debugging, close its windows first, or the new
launch only opens another window of the running browser and the port never
opens.

Configuration (`pok-ai.toml`):

```toml
[user_browser]
enabled = true                 # false removes the user_browser tools
port = 9222                    # the --remote-debugging-port, on 127.0.0.1
# user_data_dir = "D:\\Profiles\\agent"   # the profile the command names
```

The DevTools port gives full control of the browser to any program on this
computer, so start the browser this way only when you want the agent to use
it, and close it afterwards if other software on the machine is not trusted.

## How the agent works in it

- **Its own tabs, in the background.** The agent opens its pages in new
  background tabs (`Target.createTarget` with `background: true`), so the tab
  you are looking at and your keyboard focus stay as they are. It works in
  one of your tabs only after switching there explicitly
  (`user_browser_tabs`), never navigates your tab away from its page, and
  closes only tabs it opened.
- **Never your mouse or keyboard.** Input goes to the page, not through the
  operating system. A tab in front gets native DevTools input (trusted mouse
  and key events). A background tab is neither drawn nor sent native input by
  Chrome; those events are dropped, not queued, so nothing fires later when
  you look at the tab. There the agent performs the same action with page
  events: the pointer and mouse sequence and a click on the element under the
  point, text insertion into the selected field, and Enter followed by the
  form's default submission. Screenshots of a background tab are rendered for
  the capture without showing the tab.
- **Any field a form needs.** Text, search, email, number, text areas,
  editable areas, date and time inputs, dropdown lists (by value or visible
  label), check boxes, radio buttons, and custom controls. Values are set so
  that framework-managed inputs (React and similar) see the change. Password
  fields are never filled; your signed-in session usually makes them
  unnecessary.
- **It can read the whole page.** `user_browser_extract` returns the URL,
  title, visible text, and every visible control with the id to act on, and
  with `include_html` the page's HTML (scripts and styles removed, each
  control marked with its id as `data-pok-id`) for working out complex forms.
  The values of password, hidden, payment-card, and one-time-code inputs are
  removed from that HTML; only a copy is changed, never your page.

The tools (the fast execution layer: fixed functions, no shell scripts):

| Tool | Does |
| --- | --- |
| `user_browser_status` | Health check: attached or not, the open tabs, and the launch command when it is not running |
| `user_browser_navigate` | Open an http(s) URL in the agent's background tab |
| `user_browser_click` | Click one control by element id (`b12`), exact visible text, or CSS selector; ambiguous text is refused with the matching ids |
| `user_browser_type` | Fill one field with exact text, optionally pressing Enter |
| `user_browser_select` | Choose an option in a dropdown list |
| `user_browser_extract` | Read the page (text, controls, optionally HTML) |
| `user_browser_screenshot` | Capture the agent's tab as an image |
| `user_browser_tabs` | List tabs, switch to one, or close one the agent opened |

Every action is checked against the page the agent read: if the control's
page, label, value, or state changed since then (you clicked something, the
page reloaded), the action is refused rather than landing on something else.

## System 2 decides, System 1 executes

The user browser follows POK-Agent's
[System 1 / System 2 split](SYSTEM1-SYSTEM2-ROADMAP.md):

- **System 2 (the planner) is the permission arbiter.** It breaks your
  request into steps, writes every value that gets typed, and is the only
  layer that commits. A click, Enter submission, list choice, or navigation
  that **deletes or removes**, **buys, pays, or subscribes**, **sends,
  posts, or submits**, or **changes account settings or signs out** pauses
  for your explicit confirmation, in every policy mode, including autonomous.
  The confirmation shows the control, the page, and what it would do.
  Choosing "allow for this session" covers only the same kind of action on
  the same site. A declined action is reported back, not retried. The CLI's
  `pok-ai run --yes` approves every prompt in advance, these included. The gate
  is enforced in Rust before the action runs (`user_browser.rs`,
  `guardrail`), not in the prompt; exams never reach your browser at all.
- **System 1 (`fast_actions` with JEV or Laya) navigates.** With the decision
  router in delegated mode, the planner's plan trees can act in your browser
  with `allowed_operations: ["user_browser_click"]`, with the same quoted-label
  grounding, `done_when` checks, interrupts, and hand-backs as everywhere
  else. Controls the guardrail flags are never System 1 candidates: a step
  that names one comes back as `commit_required` with the exact call, and
  System 2 performs it once you confirm. As with `managed_browser_click`, raw
  `user_browser_click` is withheld from the planner in delegated mode until
  `fast_actions` hands a step back.

What leaves the computer: the planner provider receives page text and
controls like any other tool result. With the hosted JEV router, System 1
receives the candidate labels and the page's URL, title, and opening text
(suspicious text is withheld) the same way it receives desktop labels; use
the local Laya router to keep that on the machine. The opt-in router training
log never records questions about your browser, whatever the page. Every
tool call is also written to the local diagnostics folder, as for all tools.

## Limits

- The guardrail reads labels and addresses. An icon-only button with no
  accessible name (a bare trash-can icon) cannot be classified; System 2's
  plan and the page's own confirmation dialogs remain the check there.
- In a background tab, page events are untrusted (`isTrusted` is false). Most
  sites accept them; a few security-sensitive widgets do not. Links that open
  a new tab may be stopped by the popup blocker there; open their address with
  `user_browser_navigate` instead. When you switch to the agent's tab, native
  input is used again.
- Native `<select>` popups, file pickers, and browser dialogs (print, save)
  are outside the page and cannot be operated; `user_browser_select` sets list
  values directly.
- Shadow DOM and same-origin frames are read; cross-origin frames are not.
- Muscle memory (motor programs) does not record user-browser steps yet.
