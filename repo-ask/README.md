# @RepoAsk VS Code Extension

A VS Code plugin for searching, managing, and querying Confluence and Jira documentation directly within your IDE. Connects to your local API simulators for realistic document indexing and retrieval functionality.

## Features & Chat Participants

This extension provides two chat participants to assist you in VS Code:

### `@repoask`
The Doc Agent helps you answer general questions using your local document store.
*   **How it works**: It relies on built-in search ranking logic (BM25 with 1-4 n-grams / IDF fallback) and `repoask_doc_check`, using a two-round LLM workflow to provide clean answers without hallucination.
*   **Capabilities**: Bases answers solely on retrieved text (no hallucination). It provides citations, doc IDs, and titles for referenced material. Includes a "Log Action" feedback button linked to the prime referenced document.


