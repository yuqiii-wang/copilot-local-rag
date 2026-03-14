import os

def render_home_template(confluence_pages, jira_issues):
    """
    Render the home page template with the provided Confluence pages and Jira issues.
    """
    template_path = os.path.join(os.path.dirname(__file__), 'home_template.html')
    
    with open(template_path, 'r', encoding='utf-8') as f:
        template_content = f.read()
    
    # Determine server type and update title
    if confluence_pages and not jira_issues:
        # Confluence server
        page_title = "Dummy Confluence Server"
        main_heading = "Confluence Pages"
        # Generate Confluence pages HTML
        confluence_html = ''
        for page_id, page in confluence_pages.items():
            title = page.get('title', 'Untitled')
            space_key = 'PROJ'  # Default space key
            title_url = title.replace(' ', '+')
            confluence_html += f"""
                    <li>
                        <a href="/confluence/spaces/{space_key}/pages/{page_id}/{title_url}">{title}</a>
                        <div class="page-info">ID: {page_id}</div>
                    </li>
            """
        # Replace placeholders
        rendered_content = template_content.replace('Dummy Confluence & Jira Server', page_title)
        rendered_content = rendered_content.replace('<h2>Confluence Pages</h2>\n        <ul>\n            {{ confluence_pages }}\n        </ul>\n        \n        <h2>Jira Issues</h2>\n        <ul>\n            {{ jira_issues }}\n        </ul>', 
                                                  f'<h2>{main_heading}</h2>\n        <ul>\n            {confluence_html}\n        </ul>')
    elif jira_issues and not confluence_pages:
        # Jira server
        page_title = "Dummy Jira Server"
        main_heading = "Jira Issues"
        # Generate Jira issues HTML
        jira_html = ''
        for issue_key, issue in jira_issues.items():
            summary = issue.get('fields', {}).get('summary', 'No summary')
            jira_html += f"""
                    <li>
                        <a href="/browse/{issue_key}">{issue_key}: {summary}</a>
                        <div class="page-info">ID: {issue.get('id', 'N/A')}</div>
                    </li>
            """
        # Replace placeholders
        rendered_content = template_content.replace('Dummy Confluence & Jira Server', page_title)
        rendered_content = rendered_content.replace('<h2>Confluence Pages</h2>\n        <ul>\n            {{ confluence_pages }}\n        </ul>\n        \n        <h2>Jira Issues</h2>\n        <ul>\n            {{ jira_issues }}\n        </ul>', 
                                                  f'<h2>{main_heading}</h2>\n        <ul>\n            {jira_html}\n        </ul>')
    else:
        # Default behavior (both types)
        # Generate Confluence pages HTML
        confluence_html = ''
        for page_id, page in confluence_pages.items():
            title = page.get('title', 'Untitled')
            space_key = 'PROJ'  # Default space key
            title_url = title.replace(' ', '+')
            confluence_html += f"""
                    <li>
                        <a href="/confluence/spaces/{space_key}/pages/{page_id}/{title_url}">{title}</a>
                        <div class="page-info">ID: {page_id}</div>
                    </li>
            """
        
        # Generate Jira issues HTML
        jira_html = ''
        for issue_key, issue in jira_issues.items():
            summary = issue.get('fields', {}).get('summary', 'No summary')
            jira_html += f"""
                    <li>
                        <a href="/browse/{issue_key}">{issue_key}: {summary}</a>
                        <div class="page-info">ID: {issue.get('id', 'N/A')}</div>
                    </li>
            """
        
        # Replace placeholders in the template
        rendered_content = template_content.replace('{{ confluence_pages }}', confluence_html)
        rendered_content = rendered_content.replace('{{ jira_issues }}', jira_html)
    
    return rendered_content