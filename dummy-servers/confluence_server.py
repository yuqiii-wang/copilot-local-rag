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
                    .page-content table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
                    .page-content th, .page-content td {{ border: 1px solid #d1d5db; padding: 8px; text-align: left; vertical-align: top; }}
                    .page-content th {{ background: #f3f4f6; }}
                    .page-content ul, .page-content ol {{ margin: 8px 0 8px 20px; padding: 0; }}
                    .page-content li {{ margin: 4px 0; }}
                    .page-content pre {{ background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 4px; padding: 8px; overflow-x: auto; }}
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

@app.put('/confluence/rest/api/content/{page_id}')
async def update_page(page_id: str, request: Request):
    page = DUMMY_CONFLUENCE_PAGES.get(page_id)
    if not page:
        return JSONResponse(status_code=404, content={'message': 'Page not found'})
    
    try:
        update_data = await request.json()
        # Update page content
        if 'body' in update_data and 'storage' in update_data['body']:
            page['body']['storage'] = update_data['body']['storage']
        # Update version
        if 'version' in update_data:
            page['version'] = update_data['version']
        # Update title
        if 'title' in update_data:
            page['title'] = update_data['title']
        
        return JSONResponse(content=page)
    except Exception as e:
        return JSONResponse(status_code=400, content={'message': f'Error updating page: {str(e)}'})

@app.get('/rest/api/content')
async def get_all_pages(expand: str = None):
    return JSONResponse(content={'results': list(DUMMY_CONFLUENCE_PAGES.values())})

@app.get('/confluence/rest/api/content/{page_id}/child/page')
async def get_page_children(page_id: str, expand: str = None):
    return JSONResponse(content={'results': []})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8091)
