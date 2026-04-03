#!/usr/bin/env python3
"""
rules.py — Common log-analysis utilities (library module, no CLI).

Provides:
  - COMMON_ERROR_WORDS   : seed keyword list for any log scan
  - tokenize_query()     : split a free-text query into plain-text log keywords
  - guess_pattern()      : replace a sensitive dynamic value with a safe regex

Import only — run main.py as the sole entry-point script.
"""

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Seed keywords — always included in every log scan
# ---------------------------------------------------------------------------

COMMON_ERROR_WORDS: list[str] = [
    # Severity / level markers
    "error",
    "exception",
    "warn",
    "warning",
    "fatal",
    "critical",
    "severe",

    # Connectivity / latency
    "timeout",
    "timed out",
    "connection refused",
    "connection reset",
    "socket",
    "unreachable",
    "retry",
    "retrying",

    # Database
    "sql error",
    "constraint violation",
    "duplicate key",
    "could not execute",
    "transaction",
]

# ---------------------------------------------------------------------------
# Pattern rules for guess_pattern()
# Each rule is (compiled_regex_to_detect_value, replacement_pattern_template).
# Rules are tried in order; the first match wins.
# ---------------------------------------------------------------------------

_GUESS_RULES: list[tuple[re.Pattern, str]] = [
    # ISIN  (e.g. US0378331005)
    (re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$"), r"[A-Z]{2}[A-Z0-9]{9}[0-9]"),

    # CUSIP  (e.g. 037833100)
    (re.compile(r"^[0-9]{3}[A-Z0-9]{5}[0-9]$"), r"[0-9]{3}[A-Z0-9]{5}[0-9]"),

    # Price  (e.g. $100.51 or 100.51)
    (re.compile(r"^\$?\d{1,10}\.\d{2}$"), r"\$?\d+\.\d{2}"),

    # Order ID  (e.g. ORD-00123 or ORDER-12345)
    (re.compile(r"^ORD(?:ER)?-\d+$", re.IGNORECASE), r"ORD(?:ER)?-\d+"),

    # Trade ID  (e.g. TRD-2026041100042)
    (re.compile(r"^TRD-\d+$", re.IGNORECASE), r"TRD-\d+"),

    # UUID  (e.g. 550e8400-e29b-41d4-a716-446655440000)
    (
        re.compile(
            r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
            r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
        ),
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    ),

    # IPv4 address  (e.g. 10.0.1.42)
    (
        re.compile(r"^(\d{1,3}\.){3}\d{1,3}$"),
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",
    ),

    # Date-time stamp
    (
        re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}"),
        r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}",
    ),

    # Generic long numeric ID  (>= 6 digits)
    (re.compile(r"^\d{6,}$"), r"\d+"),

    # Generic alphanumeric token  (>= 8 chars mixed case+digits)
    (re.compile(r"^[A-Za-z0-9]{8,}$"), r"[A-Za-z0-9]+"),
]


def guess_pattern(value: str) -> Optional[str]:
    """
    Given a sensitive *value* found in a log or query, return a safe regex
    pattern that can be used in the scanner payload instead of the raw value.

    Returns None if no rule matches (caller should decide to drop the value).
    """
    for detector, pattern in _GUESS_RULES:
        if detector.search(value):
            return pattern
    return None


# ---------------------------------------------------------------------------
# Query tokenizer
# ---------------------------------------------------------------------------

# Words that carry no log-search signal
_STOP_WORDS: frozenset[str] = frozenset(
    [
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "shall",
        "should", "may", "might", "must", "can", "could",
        "i", "we", "you", "he", "she", "it", "they", "them", "their",
        "and", "or", "not", "but", "if", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "up", "about", "into", "through",
        "during", "before", "after", "above", "below", "between",
        "what", "which", "who", "when", "where", "why", "how",
    ]
)


def tokenize_query(query: str) -> list[str]:
    """
    Split *query* into a list of candidate log keywords.

    - Strips noise words.
    - Preserves multi-word technical phrases (path segments like /orders/).
    - Does NOT include raw sensitive values; use guess_pattern() for those.
    """
    tokens: list[str] = []

    # Extract path-like segments  (e.g. /orders/placed)
    for match in re.finditer(r"/[a-zA-Z0-9_/-]+", query):
        tokens.append(match.group())

    # Split remaining words
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_.-]*", query.lower())
    for w in words:
        if w not in _STOP_WORDS and len(w) >= 3:
            tokens.append(w)

    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result



