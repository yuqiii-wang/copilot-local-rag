from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
import json
import os
from template_utils import render_home_template

app = FastAPI()

def load_dummy_data():
    data_path = os.path.join(os.path.dirname(__file__), 'dummy_data.json')
    if os.path.exists(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

try:
    dummy_data = load_dummy_data()
    DUMMY_JIRA_ISSUES = dummy_data.get('jira', {})
    DUMMY_CONFLUENCE_PAGES = dummy_data.get('confluence', {})
except Exception:
    DUMMY_JIRA_ISSUES = {}
    DUMMY_CONFLUENCE_PAGES = {}

@app.get('/')
async def home():
    html_content = render_home_template({}, DUMMY_JIRA_ISSUES)
    return HTMLResponse(content=html_content)

@app.get('/browse/{issue_key}')
async def jira_ui_view(issue_key: str):
    issue = DUMMY_JIRA_ISSUES.get(issue_key)
    if issue:
        summary = issue.get('fields', {}).get('summary', 'No summary')
        description = issue.get('fields', {}).get('description', 'No description')
        html_content = f'''
        <html>
            <head>
                <title>Jira Issue: {issue_key}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .issue-container {{ border: 1px solid #ddd; padding: 20px; border-radius: 5px; }}
                    .issue-key {{ font-size: 24px; font-weight: bold; color: #1a73e8; }}
                    .issue-summary {{ font-size: 18px; margin: 10px 0; }}
                    .issue-description {{ margin: 20px 0; }}
                    .back-link {{ margin-top: 20px; }}
                </style>
            </head>
            <body>
                <div class="issue-container">
                    <div class="issue-key">{issue_key}</div>
                    <div class="issue-summary">{summary}</div>
                    <div class="issue-description">{description}</div>
                    <div class="back-link">
                        <a href="/">Back to Home</a>
                    </div>
                </div>
            </body>
        </html>
        '''
        return HTMLResponse(content=html_content)
    return HTMLResponse(status_code=404, content='<h1>Issue not found</h1><a href="/">Back to Home</a>')

@app.get('/rest/api/2/issue/resolve')
async def resolve_issue(arg: str):
    issue = DUMMY_JIRA_ISSUES.get(arg)
    if not issue:
        for v in DUMMY_JIRA_ISSUES.values():
            if v['id'] == arg:
                issue = v
                break
    if issue:
        return JSONResponse(content=issue)
    return JSONResponse(status_code=404, content={'message': 'Issue not found'})

@app.get('/rest/api/2/issue/{issue_id_or_key}')
async def get_issue(issue_id_or_key: str):
    issue = DUMMY_JIRA_ISSUES.get(issue_id_or_key)
    if not issue:
        for v in DUMMY_JIRA_ISSUES.values():
            if v['id'] == issue_id_or_key:
                issue = v
                break
    if issue:
        return JSONResponse(content=issue)
    return JSONResponse(status_code=404, content={'message': 'Issue not found'})

@app.get('/rest/api/2/search')
async def search_issues(project: str = None):
    results = list(DUMMY_JIRA_ISSUES.values())
    if project:
        results = [issue for issue in results if issue['key'].startswith(project)]
    return JSONResponse(content={'issues': results, 'total': len(results)})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8092)
