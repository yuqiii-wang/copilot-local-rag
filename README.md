# RepoAsk - Copilot Local RAG Extension

RepoAsk is a Visual Studio Code extension that supercharges your coding experience by integrating **Local Retrieval-Augmented Generation (RAG)** directly into GitHub Copilot. It allows you to chat with your repository's knowledge base, including external documentation (Confluence, Jira), code dependencies, and even text extracted from images via Vision services.

## Features

*   **🤖 Custom Chat Participant**: Use `@repo-ask` in the Copilot Chat window to contextually interact with your retrieved data.
*   **📚 Multi-Source Context**: Seamlessly pull information from specific URLs (e.g., Jira tickets, documentation pages) to answer your queries.
*   **🖼️ Vision & Image Support**: Extract text from images and use it as context for your questions.
*   **🕸️ Knowledge Graph Power**: (Backend) Utilizes a Hypergraph-based Knowledge Graph to find deep semantic links between your code and documentation.
*   **📝 Feedback Loop**: Built-in commands to Accept or Reject answers, and even push valuable insights directly back to Confluence.
*   **Sidebar Interface**: Dedicated "RepoAsk Chat" sidebar for managing your context, inputs, and uploads.

## Prerequisites

1.  **VS Code** (version 1.96.0 or higher).
2.  **GitHub Copilot Chat** extension installed and active.
3.  **RepoAsk Backend Service**: This extension requires the accompanying Python backend to be running locally to handle indexing, tokenization, and RAG operations.

## Getting Started

### 1. Start the Backend
Navigate to the `backend/` directory and start the service:
```bash
cd backend
./start.sh
```
*By default, the backend runs on `http://localhost:14321`.*

### 2. Configure the Extension
Open VS Code Settings (`Ctrl+,`) and search for `RepoAsk`. Configure the following:
*   `repo-ask.backendUrl`: URL of your local backend (Default: `http://localhost:14321`).
*   `repo-ask.jira.urls`: List of Jira instance URLs to index/query.
*   `repo-ask.jira.personalToken`: Your Jira Personal Access Token.
*   `repo-ask.confluence.urls`: List of Confluence base URLs.
*   `repo-ask.confluence.personalToken`: Your Confluence Personal Access Token.

### 3. Use the Extension
1.  Click the **RepoAsk** icon in the Activity Bar to open the sidebar.
2.  Add context (URLs to docs, or upload screenshots).
3.  Type your query in the sidebar or switch to the **Chat** view.
4.  Invoke the agent by typing `@repo-ask` followed by your question (e.g., `@repo-ask How does the tokenization strategy handle Java files?`).

## Extension Settings

| Setting | Description | Default |
|Str|Str|Str|
| `repo-ask.backendUrl` | URL of the local RAG backend server | `http://localhost:14321` |
| `repo-ask.jira.urls` | List of allowed Jira URLs | `["http://localhost:14329"]` |
| `repo-ask.confluence.urls` | List of allowed Confluence URLs | `["http://localhost:14329"]` |
| `repo-ask.jira.personalToken` | Auth token for Jira | `""` |
| `repo-ask.confluence.personalToken` | Auth token for Confluence | `""` |

## Commands

*   `RepoAsk: Open Chat Window`: Opens the chat interface.
*   `RepoAsk: Accept Answer`: Provide positive feedback on a generated response.
*   `RepoAsk: Reject Answer`: Provide negative feedback.
*   `RepoAsk: Add to Confluence`: Push the current answer/insight to a Confluence page.

## Architecture

This extension acts as the frontend client for a sophisticated Knowledge Graph backend:
*   **Frontend**: VS Code Extension (TypeScript) handling UI, Chat API integration, and user input.
*   **Backend**: Python (FastAPI) handling:
    *   **Vision**: Extracting text from uploaded images.    *   **Indexing**: Parsing Java code and Documentation into a Hypergraph.
    *   **GNN**: Running a Graph Neural Network to find semantically related files.
    *   **Retrieval**: Fetching relevant context for the LLM.

## Contributing

1.  Open the project in VS Code.
2.  Run `npm install` to install dependencies.
3.  Press `F5` to start debugging the extension.
