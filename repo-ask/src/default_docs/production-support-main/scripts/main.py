#!/usr/bin/env python3
"""
main.py — Async production log scanner (the ONLY runnable script in this skill).

Takes a scan plan produced by production-support-plan and runs all log ×
time-range combinations in one async task pool, returning a merged JSON map
of keyword -> [timestamp, ...] for every match.

No sensitive data is sent to any remote LLM.  Only static keyword strings
and the returned timestamp hit-list are produced; all log content stays
server-side.

Plan mode (recommended)
-----------------------
Pass the full scan plan from production-support-plan as --plan.  Each entry
in plan["scan_tasks"] carries its own log_url, words, and time_ranges, so
different logs can be searched for different terms simultaneously.

    python scripts/main.py --plan plan.json [--seed] [--pretty] [--raw]

The plan JSON structure is:

    {
      "incident_summary": "...",
      "original_query": "...",
      "extracted_identifiers": ["US0378331005"],
      "scan_tasks": [
        {
          "log_url": "http://host/api/logs/trading-system-202604011300.log",
          "words": ["/orders/", "fill", "routing"],
          "time_ranges": [{"start": "2026-04-01 09:00:00.000",
                           "end":   "2026-04-01 13:00:00.000"}]
        }
      ]
    }

main.py validates every extracted_identifier against original_query
(injection guard), applies guess_pattern() to convert raw tokens into safe
regexes, and merges the results into one keyword -> [timestamps] dict.

Flat mode (fallback)
--------------------
For ad-hoc use without a plan file, pass --log-urls, --words, and
--start-time/--end-time (or --time-ranges).  All logs share the same keyword
list and time windows.

    python scripts/main.py \\
        --log-urls http://127.0.0.1:8093/api/logs/trading-system-202604011300.log \\
        --start-time "2026-04-01 09:00:00.000" \\
        --end-time   "2026-04-01 13:00:00.000" \\
        --words error timeout --seed

Environment variables
---------------------
    LOG_SCANNER_TOKEN      Optional Bearer token for the service
"""

import argparse
import asyncio
import json
import sys

try:
    import httpx  # noqa: F401 — imported here so the error message is user-friendly
except ImportError:
    print("ERROR: 'httpx' is required. Install with:  pip install httpx", file=sys.stderr)
    sys.exit(2)

from utils import (
    build_tasks_from_args,
    build_tasks_from_plan,
    merge_hits,
    scan_all,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Async production log keyword scanner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Plan mode -----------------------------------------------------------
    p.add_argument(
        "--plan",
        default=None,
        metavar="JSON_OR_FILE",
        help=(
            "Scan plan produced by production-support-plan.  Accepts either a "
            "file path (plan.json) or an inline JSON string.  When provided, "
            "--log-urls / --time-ranges / --words are ignored."
        ),
    )

    # --- Flat mode (fallback) ------------------------------------------------
    p.add_argument(
        "--log-urls",
        nargs="+",
        default=[],
        metavar="URL",
        help="(Flat mode) Pre-resolved log file URLs",
    )
    p.add_argument(
        "--start-time",
        default=None,
        help='(Flat mode) Single-range start  e.g. "2026-04-01 09:00:00.000"',
    )
    p.add_argument(
        "--end-time",
        default=None,
        help='(Flat mode) Single-range end    e.g. "2026-04-01 17:00:00.000"',
    )
    p.add_argument(
        "--time-ranges",
        default=None,
        metavar="JSON",
        help=(
            '(Flat mode) JSON array of {"start": "...", "end": "..."} objects. '
            'Example: \'[{"start": "2026-04-01 09:00:00.000", "end": "2026-04-01 12:00:00.000"}]\''
        ),
    )
    p.add_argument(
        "--original-query",
        default="",
        metavar="TEXT",
        help="Raw user query text (validates --tokenized-query tokens)",
    )
    p.add_argument(
        "--tokenized-query",
        nargs="+",
        default=[],
        metavar="TOKEN",
        help=(
            "Sensitive tokens extracted from the user query "
            "(e.g. ISINs, order IDs). Each must appear in --original-query."
        ),
    )
    p.add_argument(
        "--words",
        nargs="+",
        default=[],
        metavar="KW",
        help="Static keyword strings / safe regex patterns to search for",
    )
    p.add_argument(
        "--seed",
        action="store_true",
        help="Prepend COMMON_ERROR_WORDS to every task's keyword list",
    )
    p.add_argument(
        "--query",
        default="",
        help="(Flat mode) Free-text query to auto-tokenize into keywords",
    )
    p.add_argument(
        "--pretty",
        action="store_true",
        default=True,
        help="Pretty-print JSON output (default: True)",
    )
    p.add_argument(
        "--raw",
        action="store_true",
        help="Emit one JSON object per task instead of a merged hit-list",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    # ------------------------------------------------------------------
    # Build task list — plan mode OR flat mode
    # ------------------------------------------------------------------
    plan: dict | None = None
    if args.plan:
        plan_text = args.plan
        try:
            if not plan_text.lstrip().startswith("{"):
                with open(plan_text, encoding="utf-8") as fh:
                    plan_text = fh.read()
            plan = json.loads(plan_text)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"ERROR: could not load --plan: {exc}", file=sys.stderr)
            sys.exit(1)

        tasks = build_tasks_from_plan(plan, seed=args.seed)
        if not tasks:
            print("ERROR: plan contains no scan_tasks.", file=sys.stderr)
            sys.exit(1)
    else:
        if not args.log_urls:
            print(
                "ERROR: provide --plan (plan mode) or --log-urls (flat mode).",
                file=sys.stderr,
            )
            sys.exit(1)
        if not args.start_time and not args.end_time and not args.time_ranges:
            print(
                "ERROR: provide --start-time/--end-time or --time-ranges.",
                file=sys.stderr,
            )
            sys.exit(1)

        tasks = build_tasks_from_args(
            log_urls=args.log_urls,
            words=args.words,
            tokenized_query_tokens=args.tokenized_query,
            original_query=args.original_query,
            seed=args.seed,
            query=args.query,
            time_ranges_json=args.time_ranges,
            start_time=args.start_time,
            end_time=args.end_time,
        )

        if not any(t.words for t in tasks):
            print(
                "ERROR: no keywords to search for. "
                "Pass --words, --seed, or --query.",
                file=sys.stderr,
            )
            sys.exit(1)

    unique_logs = len({t.log_url for t in tasks})
    print(
        f"Scanning {unique_logs} log(s), {len(tasks)} parallel task(s) ...",
        file=sys.stderr,
    )

    results = asyncio.run(scan_all(tasks))

    indent = 2 if args.pretty else None

    if args.raw:
        print(json.dumps(results, indent=indent))
    else:
        merged = merge_hits(results)
        print(json.dumps(merged, indent=indent))

    if all("_error" in v for v in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()

