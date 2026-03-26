# Getting Started with @repoask

Welcome to `@repoask`! This documentation serves as a quick-start guide to understanding the features and commands offered by the extension to effectively sync, query, and rank documentation from Confluence and Jira directly in VS Code.

## What is @repoask?

`@repoask` uses Retrieval-Augmented Generation (RAG) concepts by fetching remote knowledge bases (like your Confluence spaces or Jira tasks), storing them locally as easily readable markdown, and allowing you or AI language models to interact with that context quickly inside VS Code.

## Commands & Features

### 1. Refresh (`repo-ask.refresh`)
Used to download or resynchronize documentation from Confluence or Jira.
- **Refresh Single Doc:** Run the `Refresh` command with the ID, Title, or URL of the page/issue.
- **Refresh All Docs:** Trigger the `Refresh All` command or click the Refresh button in the `Doc Store` Sidebar.
- *Tip:* This handles converting HTML or Confluence formats into standard Markdown and parsing metadata automatically.

### 2. Check (`repoask_doc_check`)
Validates a user question against your local synchronized metadata to provide contextually relevant references and excerpts.
- Use this to cross-reference if a specific technical question covers anything stored in the RAG store.
- **Modes:** search (default), metadata, content, all

### 3. Rank (`repoask_rank`)
Rank queries the local `doc-store` indexing logic directly using a BM25 or Inverse Document Frequency (IDF) scoring system. 
- You can find the best matching documents based on your query terms.
- The sidebar utilizes `rank` under the hood to perform search across indexed docs.

### 4. Annotate
Use this functionality to automatically generate summaries and parse technical keywords using Language Models and Regex logic.
- You can annotate a single document or all your docs to enrich the `metadata.json` for better subsequent searches.

### 5. Add to Prompts
Found in the `Doc Store` sidebar context menu or detail window, "Add to Prompts" integrates your documentation directly into the `.github/prompts/*.prompt.md` files. This is extremely useful for explicitly giving project specifications, AI instructions, or a Jira task to your GitHub Copilot Agent workspace.

## Using the Chat Agent

The extension provides one chat participant in GitHub Copilot Chat:

### `@repoask`
Use this for general questions about your documentation. It can automatically:
- Recognize when you want to `refresh` by analyzing your prompt structure.
- Answer queries using local documentation contexts directly.
- Recommend actions or point you directly to local synced pages.

## Tool Flow Integration

Behind the scenes, the extension provides multiple LLM Tools:
- `repoask_rank`: Rank local documents using IDF scoring over local metadata and markdown content.
- `repoask_doc_check`: Find relevant local documents for a question using metadata relevance and include local content references.

## Local Store

The local document store is kept in your VS Code global storage directory. Depending on your operating system, the path is:

- **Windows:** `%APPDATA%\Code\User\globalStorage\repo-ask.repo-ask`
- **macOS:** `~/Library/Application Support/Code/User/globalStorage/repo-ask.repo-ask`
- **Linux:** `~/.config/Code/User/globalStorage/repo-ask.repo-ask`

Happy coding with `@repoask`!