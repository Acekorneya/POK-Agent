# Web search and page reading

For information on the web, the agent searches with `web_search` and reads
pages with `web_fetch` before it opens a browser. Both are plain HTTP
requests, so they are fast, need no browser tab, and many agents in a
[swarm](SWARM.md) can run them at once. A page that only works with
JavaScript, or one behind a sign-in, is read in the managed browser or in
[your own browser](USER-BROWSER.md) instead.

## `web_search`

Returns ranked results (title, address, snippet) for a query, 8 by default
and at most 20. The provider is configurable:

| `search_provider` | Key | Notes |
| --- | --- | --- |
| `auto` (default) | `TAVILY_API_KEY` (optional) | Tavily as soon as a Tavily key is saved (Settings, Web search, or the variable), with DuckDuckGo as the fallback if Tavily cannot be reached; DuckDuckGo when there is no key |
| `tavily` | `TAVILY_API_KEY` | Always Tavily (an error if Tavily fails) |
| `duckduckgo` | none | DuckDuckGo's HTML results. When a direct request is refused, the same results page is read in the managed browser. |
| `brave` | `BRAVE_SEARCH_API_KEY` | Brave Search API |
| `searxng` | none | Your own SearXNG instance at `searxng_url`, with its JSON output format enabled |

```toml
[web]
search_provider = "auto"   # or "tavily", "duckduckgo", "brave", "searxng"
# api_key_env = "BRAVE_SEARCH_API_KEY"   # another variable or saved key name
# searxng_url = "http://localhost:8888"
```

### Tavily

[Tavily](https://tavily.com) is a search API built for AI agents, and the default as soon as a key is
saved: paste it under *Settings → Web search* (it is checked with a test search before it is stored in
Windows Credential Manager) or set `TAVILY_API_KEY`. The agent then gets, beyond ranked results:

- **A short answer** composed by Tavily from the results (`answer`), which the agent is told to check
  against the sources.
- **News and time limits:** `topic = "news"` and `time_range` of `day`, `week`, `month` or `year`.
- **Deep search:** `deep = true` uses Tavily's advanced depth for harder questions.
- **Pages it cannot read:** `web_fetch` asks Tavily Extract for the text of a page that is blocked,
  fails, or comes back empty because it needs JavaScript (the result says `via: tavily_extract`).
  Addresses on this computer's network are never sent, and a PDF is not.

Only the query, and the addresses of pages the agent could not read itself, go to Tavily. If Tavily is
down or the key was revoked, `auto` quietly falls back to DuckDuckGo and the result says so (`fallback`).

## `web_fetch`

Reads one public page as clean text: navigation menus, footers, scripts, and
styles are left out; headings, list items, and table cells are kept, code
blocks keep their indentation, and the page's links are listed with their
addresses. Long pages are read in parts: each
result gives `total_chars` and, while there is more, `next_offset` to pass
back as `offset` (12,000 characters per part by default, at most 40,000).
A page is fetched once and kept for ten minutes, so reading on does not
download it again.

Plain text, JSON, and XML are returned as they are. A PDF is not converted;
the agent downloads it and reads it with its file tools.

## Safety

- Page text is untrusted data. Results are marked as such, and an
  instruction written on a page is not an instruction from you.
- Fetching never reaches this computer or its local network: loopback,
  private, link-local, carrier-grade NAT, and unique-local addresses are
  refused, and redirects are followed one hop at a time with the same check
  at every hop (at most five). A page therefore cannot point the agent at a
  local service, such as a browser's DevTools port or a router's admin page.
- Responses are limited to 8 MB and 20 seconds.
