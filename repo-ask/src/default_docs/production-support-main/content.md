# Production Support Main Skill

## When to Use

- You have a **scan plan** from `production-support-plan` (confirmed services, resolved log URLs, time ranges, extracted identifiers).
- You need to perform the actual keyword-driven log investigation.
- Log data is **confidential** and must NOT be sent to the LLM verbatim.

## Input Contract

This skill expects a scan plan JSON produced by `production-support-plan`.
The key change from the flat format is that **keywords and time windows live inside each `scan_tasks` entry**, so different logs can be scanned for different terms simultaneously.

```json
{
  "incident_summary": "<one-sentence description of the incident>",
  "environment": "<environment-name>",
  "original_query": "<verbatim user query>",
  "extracted_identifiers": ["<raw-identifier-from-query>"],
  "scan_tasks": [
    {
      "log_url": "<base-url>/logTail/<prefix>?file=<prefix>-<timestamp>.log",
      "words": ["<token-from-query>", "<log-message-literal-from-source-code>"],
      "time_ranges": [
        {"start": "<YYYY-MM-DD HH:MM:SS.mmm>", "end": "<YYYY-MM-DD HH:MM:SS.mmm>"}
      ]
    }
  ]
}
```

`main.py` validates `extracted_identifiers` against `original_query`, converts them with `guess_pattern()`, and appends the resulting patterns to every task's word list automatically.

If no plan is available, run `production-support-plan` first.

## Privacy Constraint — What May Be Sent to LLM

| Allowed | Forbidden |
|---|---|
| Log filenames / URLs | Raw log lines |
| Incident time range (start / end) | Trade IDs, account numbers, prices |
| Static string literals from source code | PII, secrets, tokens |
| Tool-returned JSON hit-lists (timestamps only) | Full log file content |

All sensitive data is filtered server-side by the scanning tool before returning results. The LLM only ever sees timestamps and keyword hit counts.

---

## Workflow

### Step 1 — Build Keyword List per Log

For each entry in `scan_tasks`, populate `words` **exclusively** from the two sources below. Do **NOT** invent, guess, or add any word that does not literally come from one of these two sources — no domain assumptions, no synonyms, no extra context.

**Source A — User query terms**  
Extract meaningful noun phrases and identifiers directly from `original_query`. Keep only tokens likely to appear verbatim in log output (route paths, action verbs, entity names, service names). Discard filler words (e.g. "the", "is", "a").

**Source B — Project source code scan**  
Search the project source tree for the following patterns and extract their **literal string fragments**:

| Pattern to search | What to extract |
|---|---|
| `logger.error(` / `log.error(` | First string argument (the message template) |
| `logger.warn(` / `log.warn(` | First string argument |
| `throw new …Exception("…")` | Exception message literal |
| `@RequestMapping` / `@GetMapping` / `@PostMapping` value | Route path string |

Strip format placeholders (`{}`, `%s`) from extracted strings. Keep only fragments whose topic is semantically relevant to the selected log's description (e.g. `<domain>-related error messages for the <domain> log, <service> route paths for the <service-name> log`).

Do **not** include `COMMON_ERROR_WORDS` in the plan — pass `--seed` to `main.py` instead so they are prepended automatically.

`extracted_identifiers` go at the top level of the plan (not inside each task); `main.py` validates them against `original_query` and applies `guess_pattern()` automatically:

| Raw value | `guess_pattern()` result |
|---|---|
| `<ISIN-like alphanumeric>` | `[A-Z]{2}[A-Z0-9]{9}[0-9]` |
| `<ORDER-prefix-ID>` | `<PREFIX>(?:ER)?-\d+` |
| `<TRADE-prefix-ID>` | `<PREFIX>-\d+` |
| `<decimal-currency-amount>` | `\$?\d+\.\d{2}` |

---

### Step 2 — Run the Scanner

Save the plan JSON as `plan.json` (or pass inline) and invoke `main.py` once with `--plan`:

```bash
python scripts/main.py --plan plan.json --seed
```

`main.py` reads every `scan_tasks` entry, expands each entry's `time_ranges`, and launches all `(log_url × window)` combinations in one async pool — returning a single merged hit-list JSON.  No per-log argument assembly is needed.

---

### Step 3 — Iterative Investigate (max 5 rounds, configurable)

This is a **think → scan → think** loop.  Each pass = one round.  Stop as soon as the root cause is located or you have reached `--max-rounds` (default **5**).

#### 3a — Reconstruct the event timeline

From the hit-list `log_name → keyword → [timestamps]`:

1. **Sort all keywords by timestamp** within each log.  Present as:
   ```
   <log-name>
     HH:MM:SS.mmm  keyword_A   (first)
     HH:MM:SS.mmm  keyword_B
     HH:MM:SS.mmm  keyword_C   (same line as keyword_B → correlated)
   ```
2. Keywords that **share the exact same timestamp** came from the same log line — treat them as a single event.
3. Note the **ordering** across different logs to reveal call-chain direction (e.g. service A errored before service B).

#### 3b — Search source code for each correlated event

You have workspace tools available (e.g. `vscode_searchInFiles`, `vscode_readFile`, `vscode_listDirectory`).  **Use them freely** to locate code that produces the correlated log entries — there is no fixed list of patterns you must follow.

Start from the most specific evidence in the hit-list (exact timestamp clusters, keyword strings) and let those guide where you search.  Useful starting points:

- Search for an exact quoted fragment of a keyword that looks like a log message
- Find the class or method that owns the route matching a hit path keyword
- Read the file once you locate a candidate, and inspect the surrounding call chain (±30 lines)
- Follow downstream calls, queue publishes, or exception throw sites to adjacent services

Extract for each event: **file path, line number, class/method, upstream trigger, downstream impact**.  Reason about the full call chain path that produces all correlated keywords in the observed order.

#### 3c — Confidence gate

After code-search reasoning, assess confidence:

- **High confidence** (root cause clearly traceable to a specific code path + all keyword events explained): **stop iterating**.  Proceed to Step 4.
- **Low/medium confidence** (some keywords unexplained, alternative paths possible): continue to step 3d.

#### 3d — Propose next scan (new round)

Build a refined plan covering **only the new keywords** derived from the code-search findings (e.g. the function-level log string you just found, the downstream service it calls, the queue / topic name).  Pass:

```bash
python scripts/main.py --plan <new_plan.json> --seed \
    --round <current_round> --max-rounds <max_rounds>
```

Replace `<current_round>` with the round number (starting at 1 for the very first call after the initial scan).  `main.py` will echo `Round N/M` in stderr and attach `_meta.round`, `_meta.max_rounds`, and `_meta.budget_remaining` to its JSON output.

When `budget_remaining == 0` (emitted by main.py), **do not start another round** — proceed directly to Step 4 with a best-effort summary.

> **Early termination rule**: as soon as a round produces a hit-list that, in combination with the code search, fully explains the incident (clear root cause + traced code path), write the summary immediately without exhausting remaining rounds.

---

### Step 4 — Interpret and Summarize

The scanner returns grouped dictionary `log_name → keyword → [timestamps]`. Correlate keywords that share timestamps; they indicate the same log line or transaction:

```json
{
  "<log-filename>.log": {
    "<identifier-from-query>": ["<YYYY-MM-DD HH:MM:SS.mmm>"],
    "<keyword-A>": ["<YYYY-MM-DD HH:MM:SS.mmm>", "<YYYY-MM-DD HH:MM:SS.mmm>"],
    "<keyword-B>": ["<YYYY-MM-DD HH:MM:SS.mmm>"]
  }
}
```

Write the incident summary:

```
Root cause: <one sentence>  [CONFIRMED | LIKELY | POSSIBLE — state confidence]

Timeline:
  HH:MM:SS.mmm — <keyword> hit in <log-name> → <what this means>
  ...  (across all rounds)

Rounds used: N / M

Traced URLs:
  <log URLs used>

Recommended action: <next step>
Related docs: <any RepoAsk hits>
```

If max rounds were exhausted without a confirmed root cause, list all **possible causes** ranked by supporting evidence count.

---

## Scripts Reference

All scripts live in `scripts/` next to this document.

### Log Search URL Parameters

The log server (`/logTail/{component}?file={filename}`) accepts the following query parameters:

| Param | Format | Description |
|---|---|---|
| `n` | integer | Return last N lines (tail) before any other filter |
| `i` | text | Include only lines containing this text (case-insensitive); response is JSON |
| `e` | text | Exclude lines containing this text (case-insensitive) |
| `f` | `HH:MM` | Return lines from this time onwards (inclusive) |
| `t` | `HH:MM` | Return lines up to this time (inclusive) |

Parameters can be combined, e.g. `?file=<logfile>&f=<HH:MM>&t=<HH:MM>&i=<include-text>&n=<N>`.
`f`/`t` are applied before `n`; `i` and `e` both return plain-text output.

### Scripts

| Script | Purpose | Runnable? |
|---|---|---|
| `main.py` | Async pool runner — accepts `--plan` (full plan JSON) or flat `--log-urls` args; each `ScanTask` carries its own keywords and window; all tasks run concurrently via `asyncio.gather`; returns merged hit-list JSON | **Yes** — sole entry-point |

### Running locally

```bash
# Plan mode — preferred; all logs/windows/keywords in one call
python scripts/main.py --plan plan.json --seed

# With iteration tracking (round 2 of 5)
python scripts/main.py --plan plan.json --seed --round 2 --max-rounds 5

# Plan mode — inline JSON
python scripts/main.py --plan '{"scan_tasks": [{"log_url": "<base-url>/logTail/<prefix>?file=<prefix>-<timestamp>.log", "words": ["<keyword>"], "time_ranges": [{"start": "<YYYY-MM-DD HH:MM:SS.mmm>", "end": "<YYYY-MM-DD HH:MM:SS.mmm>"}]}], "original_query": "<query>", "extracted_identifiers": []}' --seed

# Flat mode — ad-hoc, all logs share one keyword list
python scripts/main.py \
  --log-urls \
      <base-url>/logTail/<prefix-1>?file=<prefix-1>-<timestamp>.log \
      <base-url>/logTail/<prefix-2>?file=<prefix-2>-<timestamp>.log \
  --time-ranges '[{"start": "<YYYY-MM-DD HH:MM:SS.mmm>", "end": "<YYYY-MM-DD HH:MM:SS.mmm>"}, {"start": "<YYYY-MM-DD HH:MM:SS.mmm>", "end": "<YYYY-MM-DD HH:MM:SS.mmm>"}]' \
  --seed
```

### Environment variables

| Variable | Description |
|---|---|
| `LOG_SCANNER_BASE_URL` | Base URL of the log-scanning service (default: `<logtail-base-url>`) |
| `LOG_SCANNER_TOKEN` | Bearer token for the scanning service (optional) |

---

## Security Notes

- **Never** pass raw log lines, trade IDs, prices, or account numbers as CLI arguments or in the JSON payload.
- All `guess_pattern` substitutions must be reviewed before sending to ensure no sensitive literal escapes.
- The scanning service should be deployed behind your internal network; do not expose it publicly.
- Credentials are read from environment variables only — do not hard-code them.
