---
name: production-support-plan
description: Pre-investigation planning — list available logs and summaries, optionally check internal docs, then produce a scan plan for production-support-main
user-invocable: true
---

# Production Support Plan Skill

---

## Configuration

### Environment Base URLs

| Environment | Base URL |
|-------------|----------|
| local | `http://127.0.0.1:8093/api/logs` |

> To add environments, extend this table (e.g. `staging`, `prod`).

### Available Logs

Log file URLs follow the pattern: `{base_url}/{prefix}-{timestamp}.log`  
Timestamp format: `YYYYMMDDHHmm`

| Prefix | Description |
|--------|-------------|
| `trading-system` | Core OMS and order-lifecycle events — order submissions, fills, cancellations, and routing decisions. |
| `error-scenarios` | Captured operational failure scenarios — unhandled exceptions, validation rejections, and system fault injections. |
| `settlement-processing` | Trade settlement lifecycle — cash movements, custody reconciliation, and ledger commit confirmations. |
| `risk-management` | Portfolio risk metrics and limit checks — VaR calculations, stress-test results, concentration breaches, and counterparty exposure. |
| `market-data-sync` | Market data ingestion pipeline — equity quotes, FX rates, index snapshots, and pricing feed health. |

---

## When to Use

- A production incident needs investigation and you are at the **beginning** of triage.
- Output is a human-approved scan plan JSON for `production-support-main`.

## Output Plan Format

```json
{
  "incident_summary": "<one-sentence description>",
  "environment": "<environment>",
  "original_query": "<verbatim user query>",
  "extracted_identifiers": ["US0378331005"],
  "scan_tasks": [
    {
      "log_url": "<base>/api/logs/trading-system-<timestamp>.log",
      "words": ["/orders/", "oms system", "routing", "fill"],
      "time_ranges": [
        {"start": "2026-04-01 09:00:00.000", "end": "2026-04-01 13:00:00.000"}
      ]
    },
    {
      "log_url": "<base>/api/logs/error-scenarios-<timestamp>.log",
      "words": ["unhandled", "fault", "rejection"],
      "time_ranges": [
        {"start": "2026-04-01 09:00:00.000", "end": "2026-04-01 13:00:00.000"}
      ]
    }
  ],
  "related_docs": [
    {"id": "<doc_id>", "title": "<title>", "summary": "<one-line>"}
  ]
}
```

**`words` per task** — use domain-specific terms for that log (see Step 1 keyword table in `production-support-main`).  
**`time_ranges` per task** — can differ across logs if the incident window is service-specific.  
**`extracted_identifiers`** — raw sensitive tokens from `original_query`; `main.py` validates and converts them with `guess_pattern()` before scanning.

---

## Workflow

### Step 1 — List Available Logs and Optionally Check Internal Docs

Call `list_log_descriptions(env)` from `scripts/urls.py`. It reads the **Configuration** tables in this file (base URL and known prefixes) then queries the logtail server, returning which files are actually present and their timestamp suffixes:

```python
from scripts.urls import list_log_descriptions
logs = list_log_descriptions(env="local")  # returns prefix, description, available [{timestamp, url}]
```

Select prefixes whose descriptions match the incident. Use the `url` values from `available` directly in `scan_tasks`. If `available` is empty for a relevant prefix, the file is not yet on the server — note this in the plan.

**If the logtail server is unreachable** (connection refused, timeout, or any HTTP error), skip log scanning entirely and fall back to a plain doc search — answer the user's question directly as if `@repoask` was invoked with no plan context:

```python
repoask_doc_check(searchTerms=[<key terms from user query>], mode="id_2_metadata_4_summary", limit=5)
```

Return a natural-language answer based on matching docs. Do not emit a plan JSON or proceed to Step 2.

Optionally run `repoask_doc_check` for related runbooks (when server is available):

```python
repoask_doc_check(searchTerms=[<key terms from user query>], mode="id_2_metadata_4_summary", limit=5)
```

Include relevant docs in `related_docs`; omit if none found.

---

### Step 2 — Produce Plan and Hand Off

Present the selected logs and time window to the user for review.  Once approved:

1. Emit the plan JSON (populated `scan_tasks[]` with per-log `words` and `time_ranges`).
2. Save it as `plan.json` (or pass inline) and invoke `production-support-main`:

```bash
python scripts/main.py --plan plan.json --seed
```

`main.py` reads every `scan_tasks` entry, expands each entry's `time_ranges`, and launches all combinations in one async pool — no further argument assembly needed.
