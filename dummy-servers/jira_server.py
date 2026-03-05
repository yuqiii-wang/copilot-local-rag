from urllib.parse import parse_qs, urlparse

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()


def build_issue(
    issue_id: str,
    key: str,
    project: str,
    summary: str,
    description: str,
    issue_type: str,
    status: str,
    priority: str,
    reporter: str,
    assignee: str,
    updated: str,
    labels: list[str] | None = None,
):
    issue = {
        "id": issue_id,
        "key": key,
        "self": f"/rest/api/2/issue/{issue_id}",
        "fields": {
            "project": {
                "key": project,
            },
            "summary": summary,
            "description": description,
            "issuetype": {
                "name": issue_type,
            },
            "status": {
                "name": status,
            },
            "priority": {
                "name": priority,
            },
            "reporter": {
                "displayName": reporter,
            },
            "assignee": {
                "displayName": assignee,
            },
            "updated": updated,
            "labels": labels or [],
        },
        "_links": {
            "webui": f"/browse/{key}",
        },
    }
    return issue


dummy_issues = {
    "10001": build_issue(
        issue_id="10001",
        key="PROJECT-1001",
        project="ENG",
        summary="Set up service health endpoint",
        description=(
            "Implement /health endpoint returning service status, version, and timestamp. "
            "Add basic smoke checks for dependencies."
        ),
        issue_type="Task",
        status="In Progress",
        priority="High",
        reporter="John Doe",
        assignee="Alice Chen",
        updated="2026-03-04T10:20:00.000+0000",
        labels=["backend", "observability"],
    ),
    "10002": build_issue(
        issue_id="10002",
        key="PROJECT-1002",
        project="PMO",
        summary="Publish project kickoff checklist",
        description=(
            "Create a reusable kickoff checklist template and align with delivery governance. "
            "Include stakeholder mapping and milestone planning sections."
        ),
        issue_type="Story",
        status="To Do",
        priority="Medium",
        reporter="Jane Smith",
        assignee="Liam Patel",
        updated="2026-03-03T15:45:00.000+0000",
        labels=["process", "template"],
    ),
    "10003": build_issue(
        issue_id="10003",
        key="PROJECT-1003",
        project="DEV",
        summary="Fix markdown rendering in docs preview",
        description=(
            "Code fences with language tags are not rendered correctly in preview mode. "
            "Update parser options and add regression tests."
        ),
        issue_type="Bug",
        status="Done",
        priority="Highest",
        reporter="Bob Johnson",
        assignee="Nina Garcia",
        updated="2026-03-02T08:10:00.000+0000",
        labels=["docs", "frontend", "bugfix"],
    ),
}


def issue_for_rest_api(issue: dict):
    response_issue = dict(issue)
    response_issue["description"] = issue["fields"]["description"]
    response_issue["summary"] = issue["fields"]["summary"]
    return response_issue


def resolve_issue_by_arg(arg_value: str):
    if arg_value in dummy_issues:
        return dummy_issues[arg_value]

    normalized = arg_value.strip().upper()
    for issue in dummy_issues.values():
        if issue["key"].upper() == normalized:
            return issue

    parsed = urlparse(arg_value)
    if parsed.scheme and parsed.netloc:
        query_params = parse_qs(parsed.query)

        issue_key_candidates = query_params.get("issueKey", [])
        if issue_key_candidates:
            key = issue_key_candidates[0].strip().upper()
            for issue in dummy_issues.values():
                if issue["key"].upper() == key:
                    return issue

        issue_id_candidates = query_params.get("issueId", [])
        if issue_id_candidates and issue_id_candidates[0] in dummy_issues:
            return dummy_issues[issue_id_candidates[0]]

        path_parts = [segment for segment in parsed.path.split("/") if segment]
        if path_parts and path_parts[-1].upper() in {
            issue["key"].upper() for issue in dummy_issues.values()
        }:
            key = path_parts[-1].upper()
            for issue in dummy_issues.values():
                if issue["key"].upper() == key:
                    return issue

    lowered = arg_value.strip().lower()
    for issue in dummy_issues.values():
        if issue["fields"]["summary"].lower() == lowered:
            return issue

    raise HTTPException(status_code=404, detail="Issue not found")


@app.get("/rest/api/2/issue/{issue_arg}")
async def get_issue(issue_arg: str):
    """Get a Jira issue by ID or key"""
    if issue_arg in dummy_issues:
        return issue_for_rest_api(dummy_issues[issue_arg])

    for issue in dummy_issues.values():
        if issue["key"].upper() == issue_arg.upper():
            return issue_for_rest_api(issue)

    raise HTTPException(status_code=404, detail="Issue not found")


@app.get("/rest/api/2/search")
async def search_issues(project: str | None = Query(default=None)):
    """Search dummy Jira issues"""
    issues = [issue_for_rest_api(issue) for issue in dummy_issues.values()]
    if project:
        issues = [
            issue
            for issue in issues
            if issue.get("fields", {}).get("project", {}).get("key") == project
        ]

    return {
        "startAt": 0,
        "maxResults": len(issues),
        "total": len(issues),
        "issues": issues,
    }


@app.get("/rest/api/2/issue/resolve")
async def resolve_issue(arg: str = Query(...)):
    """Resolve issue by id, key, URL, or exact summary"""
    return issue_for_rest_api(resolve_issue_by_arg(arg))


@app.get("/browse/{issue_key}", response_class=HTMLResponse)
async def get_issue_html(issue_key: str):
    """Get a simple Jira issue HTML view"""
    for issue in dummy_issues.values():
        if issue["key"].upper() == issue_key.upper():
            fields = issue["fields"]
            return f"""
            <h1>{issue['key']}: {fields['summary']}</h1>
            <p><strong>Project:</strong> {fields['project']['key']}</p>
            <p><strong>Type:</strong> {fields['issuetype']['name']}</p>
            <p><strong>Status:</strong> {fields['status']['name']}</p>
            <p><strong>Priority:</strong> {fields['priority']['name']}</p>
            <p><strong>Reporter:</strong> {fields['reporter']['displayName']}</p>
            <p><strong>Assignee:</strong> {fields['assignee']['displayName']}</p>
            <p><strong>Updated:</strong> {fields['updated']}</p>
            <h2>Description</h2>
            <p>{fields['description']}</p>
            <p><strong>Labels:</strong> {', '.join(fields.get('labels', [])) or 'None'}</p>
            """

    raise HTTPException(status_code=404, detail="Issue not found")


@app.get("/", response_class=HTMLResponse)
async def index():
    cards = []
    for issue in dummy_issues.values():
        fields = issue["fields"]
        cards.append(
            f"<li><a href='{issue['_links']['webui']}'>{issue['key']}</a>: "
            f"{fields['summary']} ({fields['status']['name']})</li>"
        )

    return (
        "<h1>Dummy Jira Server</h1>"
        "<p>Use <code>/rest/api/2/search</code> for JSON API or links below for HTML issue pages.</p>"
        f"<ul>{''.join(cards)}</ul>"
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)