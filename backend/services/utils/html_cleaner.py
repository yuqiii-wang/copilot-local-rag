from html.parser import HTMLParser
from urllib.parse import urljoin

class HTMLContentCleaner(HTMLParser):
    def __init__(self, base_url=None):
        super().__init__()
        self.result = []
        self.skip_tags = {'script', 'style', 'noscript', 'iframe', 'svg', 'head', 'meta', 'link'}
        self.current_skip = 0
        self.base_url = base_url
        self.image_urls = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() in self.skip_tags:
            self.current_skip += 1
        elif tag.lower() == 'br':
            self.result.append('\n')
        elif tag.lower() == 'img':
            if self.current_skip == 0:
                self._handle_img(attrs)

    def _handle_img(self, attrs):
        attr_dict = dict((k.lower(), v) for k, v in attrs)
        src = attr_dict.get('src')
        alt = attr_dict.get('alt', 'image')
        
        if src:
            # Resolve relative URL if base_url is present
            if self.base_url:
                try:
                    src = urljoin(self.base_url, src)
                except Exception:
                    pass
            
            self.image_urls.append(src)
            # Markdown format
            self.result.append(f"\n![{alt}]({src})\n")

    def handle_endtag(self, tag):
        if tag.lower() in self.skip_tags:
            self.current_skip = max(0, self.current_skip - 1)
        # Add newlines for block elements to preserve some structure
        if tag.lower() in {'p', 'div', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'tr', 'table', 'section', 'article', 'header', 'footer'}:
            self.result.append('\n')

    def handle_data(self, data):
        if self.current_skip == 0:
            # Preserve reasonable whitespace but avoid excessive empty lines
            text = data.strip()
            if text:
                self.result.append(text + " ")
            elif data.count('\n') > 0:
                # If the original data was just whitespace with newlines, we might want to keep one?
                # Actually, handle_endtag taking care of newlines is safer for block structure.
                pass

    def get_text(self):
        return "".join(self.result).strip()
