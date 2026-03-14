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
    DUMMY_CONFLUENCE_PAGES = dummy_data.get('confluence', {})
    DUMMY_JIRA_ISSUES = dummy_data.get('jira', {})
except Exception:
    DUMMY_CONFLUENCE_PAGES = {}
    DUMMY_JIRA_ISSUES = {}

@app.get('/')
async def home():
    html_content = render_home_template(DUMMY_CONFLUENCE_PAGES, {})
    return HTMLResponse(content=html_content)

@app.get('/confluence/spaces/{spaceKey}/pages/{pageId}/{title}')
async def confluence_ui_view(spaceKey: str, pageId: str, title: str):
    page = DUMMY_CONFLUENCE_PAGES.get(pageId)
    if page:
        page_title = page.get('title', 'Untitled')
        body_content = page.get('body', {}).get('storage', {}).get('value', 'No content')
        html_content = f'''
        <html>
            <head>
                <title>Confluence Page: {page_title}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .page-container {{ border: 1px solid #ddd; padding: 20px; border-radius: 5px; }}
                    .page-title {{ font-size: 24px; font-weight: bold; color: #1a73e8; margin-bottom: 20px; }}
                    .page-content {{ margin: 20px 0; }}
                    .back-link {{ margin-top: 20px; }}
                </style>
            </head>
            <body>
                <div class="page-container">
                    <div class="page-title">{page_title}</div>
                    <div class="page-content">{body_content}</div>
                    <div class="back-link">
                        <a href="/">Back to Home</a>
                    </div>
                </div>
            </body>
        </html>
        '''
        return HTMLResponse(content=html_content)
    return HTMLResponse(status_code=404, content='<h1>Page not found</h1><a href="/">Back to Home</a>')

@app.get('/confluence/rest/api/content/{page_id}')
async def get_page(page_id: str, expand: str = None):
    page = DUMMY_CONFLUENCE_PAGES.get(page_id)
    if page:
        return JSONResponse(content=page)
    return JSONResponse(status_code=404, content={'message': 'Page not found'})

@app.get('/rest/api/content')
async def get_all_pages(expand: str = None):
    return JSONResponse(content={'results': list(DUMMY_CONFLUENCE_PAGES.values())})

@app.get('/confluence/rest/api/content/{page_id}/child/page')
async def get_page_children(page_id: str, expand: str = None):
    return JSONResponse(content={'results': []})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8091)
