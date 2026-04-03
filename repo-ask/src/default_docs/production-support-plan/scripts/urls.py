#!/usr/bin/env python3
"""
urls.py — Discover available production log files by querying the logtail server.

Config (base URLs and log prefixes/descriptions) is read from data/urls.csv
and data/config.csv in the same directory as this script.

Call list_log_descriptions(env) to get each known log prefix, its summary,
and all timestamped files currently available on the server.
Use this in Step 1 of the plan before handing off to production-support-main.
"""

import csv
import json
import re
import urllib.request
from pathlib import Path

_DATA_DIR = Path(__file__).parent / "data"

# ---------------------------------------------------------------------------
# Read config from CSV files
# ---------------------------------------------------------------------------

def _parse_config() -> tuple[dict[str, str], dict[str, str]]:
    """
    Read data/urls.csv and data/config.csv and return:
      env_bases        — {env_name: base_url}
      log_descriptions — {prefix: description}
    """
    env_bases: dict[str, str] = {}
    with (_DATA_DIR / "urls.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            env_bases[row["environment"].strip().lower()] = row["base_url"].strip()

    log_descriptions: dict[str, str] = {}
    with (_DATA_DIR / "config.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            log_descriptions[row["prefix"].strip()] = row["description"].strip()

    return env_bases, log_descriptions


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_log_descriptions(env: str = "local") -> list[dict]:
    """
    Query the logtail server and return one entry per known log prefix
    with its description and all timestamped files currently available.

    Returns list of:
        {
          "prefix": str,
          "description": str,
          "available": [{"timestamp": str, "url": str}, ...]
        }

    "available" is sorted by timestamp ascending; empty list means the
    prefix is configured but no matching file exists on the server yet.
    """
    env_bases, log_descriptions = _parse_config()

    base = env_bases.get(env.lower().strip())
    if not base:
        raise KeyError(f"Unknown environment {env!r}. Known: {sorted(env_bases)}")

    with urllib.request.urlopen(base, timeout=5) as resp:
        data = json.loads(resp.read())

    # Map prefix -> available [{timestamp, url}] from server file list
    ts_re = re.compile(r"^(.+)-(\d{12})\.log$")
    prefix_files: dict[str, list[dict]] = {}
    for entry in data.get("logs", []):
        m = ts_re.match(entry["name"])
        if not m:
            continue
        prefix, ts = m.group(1), m.group(2)
        if prefix not in log_descriptions:
            continue
        prefix_files.setdefault(prefix, []).append(
            {"timestamp": ts, "url": f"{base}/{entry['name']}"}
        )

    return [
        {
            "prefix":      prefix,
            "description": log_descriptions[prefix],
            "available":   sorted(
                prefix_files.get(prefix, []), key=lambda x: x["timestamp"]
            ),
        }
        for prefix in sorted(log_descriptions)
    ]
