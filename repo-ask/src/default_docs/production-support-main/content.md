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
  "incident_summary": "<one-sentence description>",
  "environment": "<environment>",
  "original_query": "<verbatim user query>",
  "extracted_identifiers": ["US0378331005"],
  "scan_tasks": [
    {
      "log_url": "http://<log-server>/api/logs/trading-system-<timestamp>.log",
      "words": ["/orders/", "oms system", "routing", "fill"],
      "time_ranges": [
        {"start": "2026-04-01 09:00:00.000", "end": "2026-04-01 13:00:00.000"}
      ]
    },
    {
      "log_url": "<LOG_SCANNER_BASE_URL>/api/logs/error-scenarios-<timestamp>.log",
      "words": ["unhandled", "fault", "rejection"],
      "time_ranges": [
        {"start": "2026-04-01 09:00:00.000", "end": "2026-04-01 13:00:00.000"}
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

For each entry in `scan_tasks`, populate `words` with domain terms inferred from `incident_summary` and the log's own description:

| Domain | Per-log `words` examples |
|---|---|
| Order lifecycle | `"/orders/"`, `"oms system"`, `"routing"`, `"fill"` |
| Settlement | `"settlement"`, `"ledger"`, `"reconcil"`, `"custody"` |
| Risk | `"var"`, `"limit breach"`, `"exposure"`, `"concentration"` |
| Market data | `"pricing feed"`, `"quote"`, `"fx rate"`, `"snapshot"` |

Do **not** include `COMMON_ERROR_WORDS` in the plan — pass `--seed` to `main.py` instead so they are prepended automatically.

`extracted_identifiers` go at the top level of the plan (not inside each task); `main.py` validates them against `original_query` and applies `guess_pattern()` automatically:

| Raw value | `guess_pattern()` result |
|---|---|
| `US0378331005` | `[A-Z]{2}[A-Z0-9]{9}[0-9]` |
| `ORD-00412` | `ORD(?:ER)?-\d+` |
| `TRD-20260401001` | `TRD-\d+` |
| `$102.50` | `\$?\d+\.\d{2}` |

Optionally search the workspace for `log.error(`, `logger.warn(`, route paths, and queue names to add relevant static strings to each task's `words`.

---

### Step 2 — Run the Scanner

Save the plan JSON as `plan.json` (or pass inline) and invoke `main.py` once with `--plan`:

```bash
python scripts/main.py --plan plan.json --seed
```

`main.py` reads every `scan_tasks` entry, expands each entry's `time_ranges`, and launches all `(log_url × window)` combinations in one async pool — returning a single merged hit-list JSON.  No per-log argument assembly is needed.

---

### Step 3 — Interpret and Summarize

The scanner returns `keyword → [timestamps]`. Correlate keywords that share timestamps; they indicate the same log line or transaction:

```json
{
  "US0378331005": ["2026-04-01 09:44:03.666"],
  "error":        ["2026-04-01 09:44:03.666", "2026-04-01 09:45:55.817"],
  "/orders/":     ["2026-04-01 09:45:55.817"]
}
```

If a hit reveals a new entity (queue name, route), add it to `--words` and re-run Step 2.

Write the incident summary:

```
Root cause: <one sentence>

Timeline:
  HH:MM:SS.mmm — <keyword> hit → <what this means>
  ...

Traced URLs:
  <log URLs used>

Recommended action: <next step>
Related docs: <any RepoAsk hits>
```

---

## Scripts Reference

All scripts live in `scripts/` next to this document.

| Script | Purpose | Runnable? |
|---|---|---|
| `main.py` | Async pool runner — accepts `--plan` (full plan JSON) or flat `--log-urls` args; each `ScanTask` carries its own keywords and window; all tasks run concurrently via `asyncio.gather`; returns merged hit-list JSON | **Yes** — sole entry-point |
| `rules.py` | Common error seed words; `guess_pattern()`; `tokenize_query()` | Library only |

### Running locally

```bash
# Install dependencies
pip install httpx

# Plan mode — preferred; all logs/windows/keywords in one call
python scripts/main.py --plan plan.json --seed

# Plan mode — inline JSON
python scripts/main.py --plan '{"scan_tasks": [{"log_url": "http://127.0.0.1:8093/api/logs/trading-system-202604011300.log", "words": ["/orders/"], "time_ranges": [{"start": "2026-04-01 09:00:00.000", "end": "2026-04-01 13:00:00.000"}]}], "original_query": "", "extracted_identifiers": []}' --seed

# Flat mode — ad-hoc, all logs share one keyword list
python scripts/main.py \
  --log-urls \
      http://127.0.0.1:8093/api/logs/trading-system-202604011300.log \
      http://127.0.0.1:8093/api/logs/error-scenarios-202604011515.log \
  --start-time "2026-04-01 09:00:00.000" \
  --end-time "2026-04-01 13:00:00.000" \
  --words error timeout --seed

# Flat mode — multiple time windows (2 logs × 2 ranges = 4 parallel tasks)
python scripts/main.py \
  --log-urls \
      http://127.0.0.1:8093/api/logs/trading-system-202604011300.log \
      http://127.0.0.1:8093/api/logs/error-scenarios-202604011515.log \
  --time-ranges '[{"start": "2026-04-01 09:00:00.000", "end": "2026-04-01 12:00:00.000"}, {"start": "2026-04-01 14:00:00.000", "end": "2026-04-01 17:00:00.000"}]' \
  --seed
```

### Environment variables

| Variable | Description |
|---|---|
| `LOG_SCANNER_BASE_URL` | Base URL of the log-scanning service (default: `http://localhost:8080`) |
| `LOG_SCANNER_TOKEN` | Bearer token for the scanning service (optional) |

---

## Security Notes

- **Never** pass raw log lines, trade IDs, prices, or account numbers as CLI arguments or in the JSON payload.
- All `guess_pattern` substitutions must be reviewed before sending to ensure no sensitive literal escapes.
- The scanning service should be deployed behind your internal network; do not expose it publicly.
- Credentials are read from environment variables only — do not hard-code them.
