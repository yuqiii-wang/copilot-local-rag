---
name: project-agent
description: Project implementation and maintenance agent for the local Confluence simulator and the @RepoAsk VS Code extension.
argument-hint: A concrete implementation task, bug report, or feature request for this repository.
tools: [vscode, execute, read, agent, edit, search, web, todo]
---
This agent is responsible for end-to-end work in this repository, including the FastAPI dummy Confluence server, the VS Code extension `@RepoAsk`, and supporting tokenization utilities.

## When to use this agent
- Implement or fix extension commands (`check`, `rank`, `refresh`, `annotate`) and chat participant behavior.
- Update sync logic between local store and dummy Confluence/Jira APIs.
- Improve preprocessing/postprocessing logic for keyword extraction, summary generation, and HTML-to-markdown conversion.
- Maintain sidebar behavior (search, preview, metadata panel, delete, add-to-prompts).
- Validate compile/package/runtime health and update project docs.

## Behavioral contract
- Work directly in the existing codebase with minimal, targeted edits.
- Prefer root-cause fixes over superficial patches.
- Preserve current architecture and naming unless the request explicitly requires a redesign.
- Keep metadata schema stable: `title`, `author`, `last_updated`, `parent_confluence_topic`, `keywords`, `summary`.
- Ensure local store contains one markdown/plain-text content file (`<id>.md`) and one metadata file (`<id>.json`) per document.

## Capability expectations
- Understand and modify FastAPI routes and dummy data models in `dummy-servers/confluence_server.py` and `dummy-servers/jira_server.py`.
- Understand and modify VS Code extension runtime in `repo-ask/src/extension.js` (with `repo-ask/index.js` as entrypoint).
- Reuse tokenization helpers from `repo-ask/src/tokenization/*` and text-processing helpers from `repo-ask/src/textProcessing.js`.
- Understand document lifecycle in `repo-ask/src/extension/documentService.js` and LLM parsing helpers in `repo-ask/src/extension/llm.js`.
- Run project commands for verification (`npm run compile`, `npx vsce package`, and relevant Python syntax checks).

## Implementation guidance
1. Scan relevant files before editing.
2. Create/update a short TODO plan for multi-step tasks.
3. Apply minimal patches and keep existing style.
4. Run focused validation commands.
5. Report what changed, what was validated, and any remaining risks.

Implementation additions:
- Prefer adding small, well-scoped helper functions (e.g. `parseRefreshArg`, `selectToolAndArg`) that separate LLM prompts from execution logic.
- Validate page arguments by resolving with Confluence (`fetchConfluencePage`) before single-page refresh where the flow is ambiguous.
- Offer a safe fallback for ambiguous or unresolved args (e.g. prompt user with `Refresh All Docs` or perform full refresh only after explicit confirmation).
- Keep LLM output handling defensive: parse with `extractJsonObject`, validate shape, and fall back to heuristics.
- Keep sidebar interactions local to the webview (doc selection updates preview pane; no forced editor open from sidebar click).

## Success criteria for typical tasks
- `refresh` can sync one Confluence page by id/title/link, one Jira issue by key/id/link, and all Confluence pages when no argument is provided.
- `check` evaluates metadata relevance and returns references backed by local markdown/plain-text content.
- `rank` uses IDF scoring over local metadata + content and is reused by sidebar search.
- Pre-process/post-process steps are integrated into refresh/annotate (`htmlToMarkdown`, tokenization keywords, LLM summary/keywords with fallback).
- Sidebar supports sync status, search, content preview, metadata view, delete doc, and add-to-prompts.
- Documentation reflects current behavior and commands.

Additional acceptance criteria (recent additions):
- Chat LLM selector (`selectToolAndArg`) is used to choose tools (`refresh`, `annotate`, `rank`, `check`) and returns a validated `arg` when appropriate.
- `parseRefreshArg` extracts links/pageIds/titles using local heuristics first and falls back to LLM parsing only for extraction, not execution.
- Full refresh is only triggered explicitly (via user command or explicit clickable action), and ambiguous results provide a safe `Refresh All Docs` option rather than silently attempting an unresolved single-page refresh.
- For chat refresh flows, unresolved Confluence args are surfaced with an actionable fallback button instead of auto-downloading all docs.

Quick validation checklist before merging any agent-driven change:
- Run `npm run compile` in `repo-ask` and ensure no JS diagnostics in the changed files.
- When changing dummy servers, run `python -m py_compile dummy-servers/confluence_server.py` and `python -m py_compile dummy-servers/jira_server.py`.
- Package the extension with `npx vsce package` to confirm manifest and contribution validity.

Current code map (verified):
- Extension runtime and commands: `repo-ask/src/extension.js`
- Document orchestration (refresh/annotate/rank): `repo-ask/src/extension/documentService.js`
- LLM tool/arg parsing helpers: `repo-ask/src/extension/llm.js`
- Confluence/Jira API adapters: `repo-ask/src/confluenceApi.js`, `repo-ask/src/jiraApi.js`
- Local storage contract (`.md` + `.json`): `repo-ask/src/storage.js`
- Relevance + IDF ranking: `repo-ask/src/relevance.js`
- Text conversion + keyword/summary fallback: `repo-ask/src/textProcessing.js`
- Sidebar UI: `repo-ask/src/sidebar/index.html`, `repo-ask/src/sidebar/styles.css`
- Dummy servers: `dummy-servers/confluence_server.py`, `dummy-servers/jira_server.py`

Current behavior snapshot:
- Chat participant supports `refresh`, `annotate`, `rank`, `check`, with `selectToolAndArg` decisioning.
- `handleRefreshFromSource` parses args via `parseRefreshArg`, supports Jira issue extraction, resolves Confluence args before refresh, and provides safe fallback UI.
- `refresh` command currently refreshes one item (Confluence/Jira arg path) or all Confluence docs when input is empty.
- `annotate` command updates local metadata only (single doc or all docs).
- Sidebar search uses IDF ranking; selecting a doc updates embedded preview + metadata panel; delete removes local `.md/.json`; Add to Prompts writes `.github/prompts/*.prompt.md`.

Risks and notes:
- LLM responses can be noisy: always use `extractJsonObject` and validate outputs before executing commands.
- The extension relies on `vscode.lm` APIs; if not present, the code falls back to heuristics.
- Keep `engines.vscode` and `@types/vscode` aligned to avoid activation/typing mismatches.
- `repo-ask.refresh` command path currently calls `refreshDocument(arg)` for non-Jira inputs (not the parsed `arg`), while chat refresh uses parsed/validated arg flow; preserve or align behavior intentionally when modifying refresh logic.