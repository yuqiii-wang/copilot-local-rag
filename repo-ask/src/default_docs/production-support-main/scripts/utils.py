"""
utils.py — Log-search utilities used by main.py.

Provides:
  ScanTask                  NamedTuple representing one atomic scan unit
  validate_tokenized_query  Injection-guard: verify tokens against original query
  scan_log                  Async POST for a single ScanTask
  scan_all                  Async pool: launch all ScanTasks concurrently
  merge_hits                Flatten per-task results into keyword -> [timestamps]
  build_tasks_from_plan     Expand a production-support-plan JSON into ScanTask list
  build_tasks_from_args     Build ScanTask list from flat CLI args (fallback)

Import only — run main.py as the sole entry-point script.
"""

import asyncio
import json
import re
import sys
from typing import Any, NamedTuple

import httpx

from rules import COMMON_ERROR_WORDS, guess_pattern, tokenize_query

_TIMEOUT = httpx.Timeout(30.0, connect=5.0)
_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})")


# ---------------------------------------------------------------------------
# ScanTask — one atomic unit of work dispatched to the async pool
# ---------------------------------------------------------------------------

class ScanTask(NamedTuple):
    """One (log, window, keywords) triplet dispatched to the async pool."""
    log_url: str
    start: str
    end: str
    words: list[str]
 

def _task_key(task: ScanTask) -> str:
    """Stable dict key for the results map."""
    return f"{task.log_url}[{task.start}~{task.end}]"


# ---------------------------------------------------------------------------
# Token validation
# ---------------------------------------------------------------------------

def validate_tokenized_query(
    tokens: list[str], original_query: str
) -> tuple[list[str], list[str]]:
    """
    Verify each *token* appears literally (case-insensitive) in *original_query*.

    Returns (valid_tokens, rejected_tokens).  Only valid tokens are safe to
    forward; rejected ones are dropped and reported as warnings so the LLM
    cannot inject arbitrary identifiers.
    """
    valid: list[str] = []
    rejected: list[str] = []
    for tok in tokens:
        if tok.lower() in original_query.lower():
            valid.append(tok)
        else:
            rejected.append(tok)
    return valid, rejected


# ---------------------------------------------------------------------------
# Core async scan
# ---------------------------------------------------------------------------

async def _search_word(
    client: httpx.AsyncClient,
    log_url: str,
    word: str,
    start: str,
    end: str,
) -> tuple[str, list[str]]:
    """GET {log_url}/search?q=word — returns (word, [timestamps])."""
    try:
        resp = await client.get(
            f"{log_url}/search",
            params={"q": word, "start_time": start, "end_time": end},
            timeout=_TIMEOUT,
        )
    except (httpx.TransportError, httpx.TimeoutException) as exc:
        print(f"WARN: search failed for {log_url!r} word={word!r}: {exc}", file=sys.stderr)
        return word, []

    if resp.status_code != 200:
        print(f"WARN: HTTP {resp.status_code} for {log_url!r} word={word!r}", file=sys.stderr)
        return word, []

    try:
        matches = resp.json().get("matches", [])
    except Exception:
        return word, []

    timestamps: list[str] = []
    for m in matches:
        hit = _TS_RE.match(m.get("content", ""))
        if hit:
            timestamps.append(hit.group(1))
    return word, timestamps


async def scan_log(
    client: httpx.AsyncClient,
    task: ScanTask,
) -> dict[str, Any]:
    """
    Search all words in *task* against the log in parallel.

    Fires one GET per word to ``{log_url}/search`` and collects timestamps.
    Returns ``{word: [timestamps]}``; never raises.
    """
    word_results = await asyncio.gather(
        *[_search_word(client, task.log_url, w, task.start, task.end) for w in task.words]
    )
    return {word: ts for word, ts in word_results}


async def scan_all(tasks: list[ScanTask]) -> dict[str, dict[str, Any]]:
    """
    Launch every ScanTask in a single async p ool and await all results.

    Each task fires one GET per keyword against its log URL — all in parallel.
    Returns a dict keyed by ``"{log_url}[{start}~{end}]"``.
    """
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *[scan_log(client, task) for task in tasks]
        )
    return {_task_key(task): result for task, result in zip(tasks, results)}


# ---------------------------------------------------------------------------
# Merge helper
# ---------------------------------------------------------------------------

def merge_hits(results: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """
    Flatten per-task results into a single keyword -> sorted timestamp list.

    Error entries (``_error`` key) are printed to stderr and skipped.
    """
    merged: dict[str, list[str]] = {}
    for task_key, hit_map in results.items():
        if "_error" in hit_map:
            print(
                f"WARN: scan failed for {task_key!r}: {hit_map['_error']}",
                file=sys.stderr,
            )
            continue
        for keyword, timestamps in hit_map.items():
            if not isinstance(timestamps, list):
                continue
            merged.setdefault(keyword, []).extend(timestamps)

    for kw in merged:
        merged[kw] = sorted(set(merged[kw]))

    return merged


# ---------------------------------------------------------------------------
# Task builders
# ---------------------------------------------------------------------------

def _dedup(words: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for w in words:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def _resolve_identifiers(
    identifiers: list[str], original_query: str
) -> list[str]:
    """
    Validate identifiers against original_query and return safe raw values.

    Each token must appear literally in original_query (injection guard).
    Returns only the validated raw token strings for use as search terms.
    """
    if not identifiers:
        return []
    valid, rejected = validate_tokenized_query(identifiers, original_query)
    if rejected:
        print(
            f"WARN: identifiers not found in original_query, dropped: {rejected}",
            file=sys.stderr,
        )
    return valid


def build_tasks_from_plan(plan: dict, seed: bool = False) -> list[ScanTask]:
    """
    Expand a production-support-plan JSON into a flat list of ScanTask objects.

    For each entry in plan["scan_tasks"] × its time_ranges one ScanTask is
    produced.  ``seed=True`` prepends COMMON_ERROR_WORDS to every task's word
    list.  ``extracted_identifiers`` are validated against ``original_query``
    and appended as raw search terms.
    """
    original_query = plan.get("original_query", "")
    identifier_words = _resolve_identifiers(
        plan.get("extracted_identifiers", []), original_query
    )

    tasks: list[ScanTask] = []
    for entry in plan.get("scan_tasks", []):
        base_words: list[str] = []
        if seed:
            base_words.extend(COMMON_ERROR_WORDS)
        base_words.extend(entry.get("words", []))
        seen = set(base_words)
        for tok in identifier_words:
            if tok not in seen:
                base_words.append(tok)
                seen.add(tok)
        task_words = _dedup(base_words)

        for tr in entry.get("time_ranges", []):
            tasks.append(ScanTask(
                log_url=entry["log_url"],
                start=tr["start"],
                end=tr["end"],
                words=task_words,
            ))

    return tasks


def build_tasks_from_args(
    log_urls: list[str],
    words: list[str],
    tokenized_query_tokens: list[str],
    original_query: str,
    seed: bool,
    query: str,
    time_ranges_json: str | None,
    start_time: str | None,
    end_time: str | None,
) -> list[ScanTask]:
    """
    Build a flat list of ScanTask objects from flat CLI values.

    All log URLs share one keyword list and one set of time windows
    (cartesian product: N logs × M windows = N×M tasks).
    """
    log_words: list[str] = list(words)
    tokenized_query: list[str] = []

    if tokenized_query_tokens:
        if not original_query:
            print(
                "ERROR: --original-query is required when using --tokenized-query.",
                file=sys.stderr,
            )
            sys.exit(1)
        valid, rejected = validate_tokenized_query(tokenized_query_tokens, original_query)
        if rejected:
            print(
                f"WARN: tokens not found in original query, dropped: {rejected}",
                file=sys.stderr,
            )
        tokenized_query = valid
        for tok in valid:
            log_words.append(tok)  # use raw value — server does literal search

    if seed:
        log_words = COMMON_ERROR_WORDS + log_words

    if query:
        extra = tokenize_query(query)
        if not tokenized_query:
            tokenized_query = extra
        log_words = extra + log_words

    log_words = _dedup(log_words)

    time_ranges: list[tuple[str, str]] = []
    if time_ranges_json:
        try:
            raw = json.loads(time_ranges_json)
            if not isinstance(raw, list) or not raw:
                raise ValueError("Expected a non-empty JSON array")
            time_ranges = [(r["start"], r["end"]) for r in raw]
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            print(
                f'ERROR: --time-ranges must be a JSON array of {{"start": "...", "end": "..."}} '
                f"objects: {exc}",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        time_ranges = [(start_time, end_time)]

    return [
        ScanTask(
            log_url=url,
            start=start,
            end=end,
            words=log_words,
        )
        for url in log_urls
        for start, end in time_ranges
    ]
