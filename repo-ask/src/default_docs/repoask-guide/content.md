# Getting Started with @repoask

Welcome to `@repoask`! This documentation serves as a quick-start guide to understanding the features and commands offered by the extension to effectively sync, query, and rank documentation from Confluence and Jira directly in VS Code.

## What is @repoask?

`@repoask` uses Retrieval-Augmented Generation (RAG) concepts by fetching remote knowledge bases (like your Confluence spaces or Jira tasks), storing them locally as easily readable markdown, and allowing you or AI language models to interact with that context quickly inside VS Code.

## Commands & Features

### 1. Refresh (`repoask.refresh`)
Used to download or resynchronize documentation from Confluence or Jira.
- **Refresh Single Doc:** Run the `Refresh` command with the ID, Title, or URL of the page/issue.
- **Refresh All Docs:** Trigger the `Refresh All` command or click the Refresh button in the `Doc Store` Sidebar.
- *Tip:* This handles converting HTML or Confluence formats into standard Markdown and parsing metadata automatically.

### 2. Check (`repoask.check`)
Validates a user question against your local synchronized metadata to provide contextually relevant references and excerpts.
- Use this to cross-reference if a specific technical question covers anything stored in the RAG store.

### 3. Rank (`repoask.rank`)
Rank queries the local `doc-store` indexing logic directly using a BM25 or Inverse Document Frequency (IDF) scoring system. 
- You can find the best matching documents based on your query terms.
- The sidebar utilizes `rank` under the hood to perform search across indexed docs.

### 4. Annotate (`repoask.annotate`)
Use this command to automatically generate summaries and parse technical keywords using Language Models and Regex logic.
- You can annotate a single document or all your docs to enrich the `metadata.json` for better subsequent searches.

### 5. Add to Prompts
Found in the `Doc Store` sidebar context menu or detail window, "Add to Prompts" integrates your documentation directly into the `.github/prompts/*.prompt.md` files. This is extremely useful for explicitly giving project specifications, AI instructions, or a Jira task to your GitHub Copilot Agent workspace.

## Using the Chat Agent

By typing `@repoask` in the GitHub Copilot Chat, you engage with the local RAG participant. It can automatically:
- Recognize when you want to `refresh` by analyzing your prompt structure.
- Answer queries using local documentation contexts directly.
- Recommend actions or point you directly to local synced pages.

## Tool Flow Integration

Behind the scenes, the extension provides multiple LLM Tools:
- `repoask_check`: Searches locally.
- `repoask_refresh`: Syncs dynamically.
- `repoask_new_code_check`: Auto-checks PR diffs and searches `.github/prompts/` to start working on new Jira tasks directly upon codebase synchronization.

Happy coding with `@repoask`!