"""
Logtail dummy server – hosts log files from the dummy-servers/logs/ directory.

Endpoints
---------
GET /                              – HTML home / index of available logs
GET /api/logs                      – JSON list of available log files
GET /api/logs/{filename}           – Return full log content (plain text)
GET /api/logs/{filename}?start_time=YYYY-MM-DD HH:MM:SS&end_time=YYYY-MM-DD HH:MM:SS
                                   – Return time-filtered log lines
GET /api/logs/{filename}/tail?lines=N
                                   – Return last N lines (default 100)
GET /api/logs/{filename}/search?q=keyword
                                   – Return lines matching keyword (case-insensitive)

Runs on port 8093.
"""

from fastapi import FastAPI, Request, Query, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, HTMLResponse
import os
import re
from datetime import datetime
from typing import Optional
from template_utils import render_home_template

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
LOG_TIMESTAMP_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})"
)
TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S.%f"

app = FastAPI(title="Logtail Dummy Server", version="1.0.0")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _list_logs() -> list[str]:
    if not os.path.isdir(LOGS_DIR):
        return []
    return sorted(
        f for f in os.listdir(LOGS_DIR)
        if os.path.isfile(os.path.join(LOGS_DIR, f))
    )


def _read_log(filename: str) -> list[str]:
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = os.path.join(LOGS_DIR, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"Log file '{filename}' not found")
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.readlines()


def _parse_ts(line: str) -> Optional[datetime]:
    m = LOG_TIMESTAMP_RE.match(line)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), TIMESTAMP_FMT)
    except ValueError:
        return None


def _filter_by_time(lines: list[str], start: Optional[str], end: Optional[str]) -> list[str]:
    start_dt = datetime.strptime(start, TIMESTAMP_FMT) if start else None
    end_dt = datetime.strptime(end, TIMESTAMP_FMT) if end else None
    result = []
    current_ts: Optional[datetime] = None
    for line in lines:
        ts = _parse_ts(line)
        if ts is not None:
            current_ts = ts
        if current_ts is None:
            continue
        if start_dt and current_ts < start_dt:
            continue
        if end_dt and current_ts > end_dt:
            continue
        result.append(line)
    return result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home():
    logs = _list_logs()
    links = [
        {"text": name, "href": f"/api/logs/{name}"}
        for name in logs
    ]
    sections = [
        {
            "title": "Available Log Files",
            "items": [
                {
                    "title": name,
                    "description": f"Full log · <a href='/api/logs/{name}?start_time=2026-04-01 09:00:00.000&end_time=2026-04-01 17:00:00.000'>time filter</a> · <a href='/api/logs/{name}/tail'>tail 100</a> · <a href='/api/logs/{name}/search?q=ERROR'>search ERROR</a>",
                    "url": f"/api/logs/{name}",
                } for name in logs
            ],
        }
    ]
    return render_home_template(
        title="Logtail Dummy Server",
        description="Serves log files for the trading system simulation.",
        port=8093,
        sections=sections,
    )


@app.get("/api/logs", response_class=JSONResponse)
async def list_logs():
    logs = _list_logs()
    return JSONResponse({
        "logs": [
            {
                "name": name,
                "url": f"/api/logs/{name}",
                "tail_url": f"/api/logs/{name}/tail",
                "search_url": f"/api/logs/{name}/search",
            }
            for name in logs
        ],
        "count": len(logs),
    })


@app.get("/api/logs/{filename}", response_class=PlainTextResponse)
async def get_log(
    filename: str,
    start_time: Optional[str] = Query(default=None, description="YYYY-MM-DD HH:MM:SS.mmm"),
    end_time:   Optional[str] = Query(default=None, description="YYYY-MM-DD HH:MM:SS.mmm"),
):
    lines = _read_log(filename)
    if start_time or end_time:
        lines = _filter_by_time(lines, start_time, end_time)
    return PlainTextResponse("".join(lines))


@app.get("/api/logs/{filename}/tail", response_class=PlainTextResponse)
async def tail_log(filename: str, lines: int = Query(default=100, ge=1, le=10000)):
    all_lines = _read_log(filename)
    return PlainTextResponse("".join(all_lines[-lines:]))


@app.get("/api/logs/{filename}/search", response_class=JSONResponse)
async def search_log(
    filename: str,
    q: str = Query(..., description="Keyword to search for (case-insensitive)"),
    start_time: Optional[str] = Query(default=None),
    end_time:   Optional[str] = Query(default=None),
):
    all_lines = _read_log(filename)
    if start_time or end_time:
        all_lines = _filter_by_time(all_lines, start_time, end_time)
    pattern = re.compile(re.escape(q), re.IGNORECASE)
    matches = [
        {"line_number": i + 1, "content": line.rstrip("\n")}
        for i, line in enumerate(all_lines)
        if pattern.search(line)
    ]
    return JSONResponse({"keyword": q, "matches": matches, "count": len(matches)})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8093)
