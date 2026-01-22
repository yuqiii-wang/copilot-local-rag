class DownloadService:
    def fetch_content(self, urls: list[str]):
        results = []
        for url in urls:
            # Dummy content generation
            content = f"Start of content for {url}\nThis is some dummy content retrieved from the document.\nIt contains relevant information for the user's query.\nEnd of content for {url}"
            results.append({"url": url, "content": content})
        return results

download_service = DownloadService()
