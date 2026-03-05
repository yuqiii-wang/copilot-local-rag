from urllib.parse import urlparse, parse_qs

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()


def build_page(
    page_id: str,
    space: str,
    title: str,
    author: str,
    last_updated: str,
    parent_topic: str,
    content: str,
    keywords: list[str] | None = None,
    summary: str | None = None,
):
    page = {
        "id": page_id,
        "space": space,
        "title": title,
        "author": author,
        "last_updated": last_updated,
        "parent_confluence_topic": parent_topic,
        "_links": {
            "self": f"/rest/api/content/{page_id}",
            "webui": f"/wiki/spaces/{space}/pages/{page_id}",
        },
        "body": {
            "storage": {
                "value": content,
                "representation": "storage",
            }
        },
    }

    if keywords is not None:
        page["keywords"] = keywords
    if summary is not None:
        page["summary"] = summary

    return page


# Dummy Confluence pages data
# Intentionally omits "keywords" and "summary" to exercise extension post-processing.
dummy_pages = {
    "1": {
        **build_page(
            page_id="1",
            space="ENG",
            title="Getting Started with Confluence",
            author="John Doe",
            last_updated="2026-03-01",
            parent_topic="Team Collaboration",
            content="""
        <h1>Getting Started with Confluence</h1>
        <p>Welcome to Confluence! This page will help you get started with using Confluence for your team collaboration.</p>
        <h2>Key Features</h2>
        <ul>
            <li>Create and edit pages</li>
            <li>Organize content in spaces</li>
            <li>Collaborate with team members</li>
            <li>Attach files and media</li>
        </ul>
        <p>To create a new page, click on the \"Create\" button in the top right corner.</p>
        """,
        )
    },
    "2": {
        **build_page(
            page_id="2",
            space="PMO",
            title="Project Management Best Practices",
            author="Jane Smith",
            last_updated="2026-03-02",
            parent_topic="Delivery Excellence",
            content="""
        <h1>Project Management Best Practices</h1>
        <p>This page outlines the best practices for project management using Confluence.</p>
        <h2>Planning Phase</h2>
        <ul>
            <li>Define project scope</li>
            <li>Set clear objectives</li>
            <li>Identify stakeholders</li>
        </ul>
        <h2>Execution Phase</h2>
        <ul>
            <li>Regular status updates</li>
            <li>Risk management</li>
            <li>Resource allocation</li>
        </ul>
        """,
        )
    },
    "3": {
        **build_page(
            page_id="3",
            space="DEV",
            title="Technical Documentation Guide",
            author="Bob Johnson",
            last_updated="2026-03-03",
            parent_topic="Engineering Handbook",
            content="""
        <h1>Technical Documentation Guide</h1>
        <p>This guide helps technical writers create effective documentation in Confluence.</p>
        <h2>Documentation Structure</h2>
        <ul>
            <li>Introduction</li>
            <li>Installation instructions</li>
            <li>Usage examples</li>
            <li>Troubleshooting</li>
        </ul>
        <h2>Formatting Tips</h2>
        <ul>
            <li>Use headings to organize content</li>
            <li>Include code snippets where relevant</li>
            <li>Add screenshots for visual guidance</li>
        </ul>
        """,
        )
    }
}


def page_for_rest_api(page: dict):
    rest_page = dict(page)
    rest_page["content"] = page["body"]["storage"]["value"]
    return rest_page


def resolve_page_by_arg(arg_value: str):
    if arg_value in dummy_pages:
        return dummy_pages[arg_value]

    parsed = urlparse(arg_value)
    if parsed.scheme and parsed.netloc:
        query_params = parse_qs(parsed.query)
        page_id_candidates = query_params.get("pageId", [])
        if page_id_candidates and page_id_candidates[0] in dummy_pages:
            return dummy_pages[page_id_candidates[0]]

        path_parts = [segment for segment in parsed.path.split("/") if segment]
        if path_parts and path_parts[-1] in dummy_pages:
            return dummy_pages[path_parts[-1]]

    lowered = arg_value.strip().lower()
    for page in dummy_pages.values():
        if page["title"].lower() == lowered:
            return page

    raise HTTPException(status_code=404, detail="Page not found")


@app.get("/rest/api/content/{page_id}")
async def get_page(page_id: str):
    """Get a Confluence page by ID"""
    if page_id not in dummy_pages:
        raise HTTPException(status_code=404, detail="Page not found")
    return page_for_rest_api(dummy_pages[page_id])


@app.get("/rest/api/content")
async def get_pages(space: str | None = Query(default=None)):
    """Get all Confluence pages"""
    pages = [page_for_rest_api(page) for page in dummy_pages.values()]
    if space:
        pages = [page for page in pages if page.get("space") == space]
    return pages


@app.get("/rest/api/content/resolve")
async def resolve_page(arg: str = Query(...)):
    """Resolve page by id, URL, or exact title"""
    return page_for_rest_api(resolve_page_by_arg(arg))


@app.get("/wiki/spaces/{space}/pages/{page_id}", response_class=HTMLResponse)
async def get_page_html(space: str, page_id: str):
    """Get a Confluence page as HTML"""
    if page_id not in dummy_pages:
        raise HTTPException(status_code=404, detail="Page not found")
    page = dummy_pages[page_id]
    if page["space"] != space:
        raise HTTPException(status_code=404, detail="Page not found in given space")
    return page["body"]["storage"]["value"]


@app.get("/", response_class=HTMLResponse)
async def index():
    cards = []
    for page in dummy_pages.values():
        cards.append(
            f"<li><a href='{page['_links']['webui']}'>{page['title']}</a> "
            f"({page['space']}) — updated {page['last_updated']}</li>"
        )
    return (
        "<h1>Dummy Confluence Server</h1>"
        "<p>Use <code>/rest/api/content</code> for JSON API or links below for HTML pages.</p>"
        f"<ul>{''.join(cards)}</ul>"
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
