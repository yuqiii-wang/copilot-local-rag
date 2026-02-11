import os
import urllib.request
from urllib.parse import unquote, urlparse
from services.knowledge_graph import tokenization
from services.utils.html_cleaner import HTMLContentCleaner

class DownloadService:
    def _resolve_path(self, path: str):
        """
        Try to resolve a potential file path from a text/url string.
        Checks:
        1. file:// URI decoding
        2. Absolute paths
        3. Relative to CWD
        4. Relative to Workspace Root (Parent of CWD)
        """
        # Handle file:// prefix
        if path.startswith("file://"):
            parsed = urlparse(path)
            path = unquote(parsed.path)
            # Handle Windows paths starting with / like /C:/...
            if os.name == 'nt' and path.startswith('/') and ':' in path[1:3]:
                path = path[1:]
        
        candidates = [path]
        
        if not os.path.isabs(path):
            cwd = os.getcwd()
            # 1. Try relative to current working directory (e.g. backend/)
            candidates.append(os.path.join(cwd, path))
            # 2. Try relative to workspace root (e.g. backend/../)
            candidates.append(os.path.join(os.path.dirname(cwd), path))
        
        for cand in candidates:
            if os.path.exists(cand) and os.path.isfile(cand):
                return cand
        
        return None

    def _extract_relevant_snippet(self, content: str, query: str) -> str:
        if not query or not content:
            return content
        
        lines = content.split('\n')
        # Simple tokenization of unique keywords (lowercase, length > 2)
        # Exclude common stopwords could be good but let's stick to length > 2
        keywords = set(w.lower() for w in query.split() if len(w) > 2)
        if not keywords:
            return content

        relevant_indices = set()
        
        # 1. Find Search Hits
        for i, line in enumerate(lines):
            lower_line = line.lower()
            if any(k in lower_line for k in keywords):
                # Add window +/- 10
                start = max(0, i - 10)
                end = min(len(lines), i + 11)
                for j in range(start, end):
                    # Filter boilerplate imports if they are not the target line itself
                    # (Allow if the keyword actually matches the import line, e.g. searching for "java.util")
                    if j != i and tokenization.is_boilerplate_line(lines[j]):
                        continue
                    relevant_indices.add(j)
                
                # 2. Add Structure Context (Indentation Scan)
                if not line.strip(): continue

                # Count indentation (tabs as 4 spaces for heuristic)
                current_indent = len(line.replace('\t', '    ')) - len(line.replace('\t', '    ').lstrip())
                
                # Scan backwards for parents
                for j in range(i - 1, -1, -1):
                    parent_line = lines[j]
                    if not parent_line.strip(): continue
                    
                    parent_indent = len(parent_line.replace('\t', '    ')) - len(parent_line.replace('\t', '    ').lstrip())
                    if parent_indent < current_indent:
                        # Avoid adding boilerplate parents (e.g. package pkg;)
                        if not tokenization.is_boilerplate_line(parent_line):
                            relevant_indices.add(j)
                            current_indent = parent_indent
                        # Stop if we hit root or close to it
                        if current_indent <= 0: break

        if not relevant_indices:
            # If no keywords match, return full content (fallback)
            return content
        
        # 3. Reconstruct
        sorted_indices = sorted(list(relevant_indices))
        result_lines = []
        last_idx = -1
        
        for idx in sorted_indices:
            if last_idx != -1 and idx > last_idx + 1:
                result_lines.append(f"... [skipped {idx - last_idx - 1} lines] ...")
            result_lines.append(lines[idx])
            last_idx = idx
            
        return "\n".join(result_lines)

    def _is_html(self, url: str, content: str) -> bool:
        if isinstance(url, str) and url.lower().endswith(('.html', '.htm', '.xhtml')):
            return True
        # Check first 1000 chars for doctype or html tag
        header = content[:1000].lower() if content else ""
        if "<!doctype html" in header or "<html" in header:
            return True
        return False

    def fetch_content(self, urls: list, query: str = None):
        """urls may be a list of strings or a list of objects { url: str, comment?: str }
        Returns results with 'url', 'content', 'found', and preserves 'comment' if provided.
        If query is provided, extracts relevant code snippets."""
        results = []
        for entry in urls:
            # Support both string and dict inputs
            if isinstance(entry, dict):
                url = entry.get('url') or entry.get('source') or entry.get('sourceUrl')
                provided_comment = entry.get('comment', None)
            else:
                url = entry
                provided_comment = None

            found = False
            try:
                if isinstance(url, str) and (url.startswith("http://") or url.startswith("https://")):
                    with urllib.request.urlopen(url) as response:
                        content = response.read().decode('utf-8', errors='replace')
                    found = True
                elif isinstance(url, str):
                    # Resolve local file path
                    real_path = self._resolve_path(url)
                    
                    if real_path:
                        with open(real_path, "r", encoding="utf-8", errors="replace") as f:
                            content = f.read()
                        found = True
                    else:
                        # File not found
                        content = f"File not found: {url}"
                else:
                    content = f"Invalid URL type: {entry}"
            except Exception as e:
                content = f"Error reading file {url}: {e}"
            
            # HTML Cleaning
            if found and self._is_html(url, content):
                try:
                    cleaner = HTMLContentCleaner()
                    cleaner.feed(content)
                    content = cleaner.get_text()
                except Exception as e:
                    print(f"HTML cleanup failed for {url}: {e}")

            # Smart filtering enabled if query is present and file was found
            if found and query:
                content = self._extract_relevant_snippet(content, query)

            result = {"url": url, "content": content, "found": found}
            if provided_comment:
                result['comment'] = provided_comment
            results.append(result)
        return results

download_service = DownloadService()
