---
name: project-agent
description: Project implementation and maintenance agent for the local Confluence simulator and the @RepoAsk VS Code extension.
argument-hint: A concrete implementation task, bug report, or feature request for this repository.
tools: [vscode, execute, read, agent, edit, search, web, todo, repo-ask.repo-ask/repoaskRefresh, repo-ask.repo-ask/repoaskAnnotate, repo-ask.repo-ask/repoaskRank, repo-ask.repo-ask/repoaskCheck]
---
This agent is responsible for end-to-end work in this repository, including the FastAPI dummy Confluence server, the VS Code extension `@RepoAsk`, and supporting tokenization utilities.

GUIDE:

Keep code clean and write to new files if one file is too large. 

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
- Keep local store contract aligned with `repo-ask/src/storage.js`:
	- Current primary format: one directory per doc (`local-store/<id>/content.md`, `local-store/<id>/metadata.json`, optional `local-store/<id>/images/*`).
	- Backward compatibility: legacy flat files (`<id>.md`, `<id>.json`, `<id>.txt`) are still read/deleted when present.
	- Indices: stored in `local-store-index/` with two index dimensions: keywords and bm25 for fast local retrieval.

## Capability expectations
- Understand and modify FastAPI routes and dummy data models in `dummy-servers/confluence_server.py` and `dummy-servers/jira_server.py`.
- Understand and modify VS Code extension runtime in `repo-ask/src/extension.js` (with `repo-ask/index.js` as entrypoint).
- Reuse tokenization helpers from `repo-ask/src/tokenization/*` and text-processing helpers from `repo-ask/src/textProcessing.js`.
- Understand document lifecycle in `repo-ask/src/extension/documentService/*`, LLM parsing helpers in `repo-ask/src/extension/llm.js`, and LM tool/chat routing in `repo-ask/src/extension/lmTools.js` and `repo-ask/src/extension/promptContext.js`.
- Understand command modules under `repo-ask/src/extension/commands/*` and chat handlers under `repo-ask/src/extension/chat/*`.
- Keep ranking behavior consistent with `documentService.rankLocalDocuments`: BM25-first ranking with IDF fallback.
- Run project commands for verification (`npm run compile`, `npx vsce package`, and relevant Python syntax checks).

## Implementation guidance
1. Scan relevant files before editing.
2. Create/update a short TODO plan for multi-step tasks.
3. Apply minimal patches and keep existing style.
4. Run focused validation commands.
5. Report what changed, what was validated, and any remaining risks.

Implementation additions:
- Prefer adding small, well-scoped helper functions (e.g. `parseRefreshArg`, `extractConfluenceIdentifierWithLlm`) that separate LLM prompts from execution logic.
- Validate page arguments by resolving with Confluence (`fetchConfluencePage`) before single-page refresh where the flow is ambiguous.
- Offer a safe fallback for ambiguous or unresolved args (e.g. prompt user with `Refresh All Docs` or perform full refresh only after explicit confirmation).
- Keep LLM output handling defensive: parse with `extractJsonObject`, validate shape, and fall back to heuristics.
- Keep sidebar interactions local to the webview (doc selection updates preview pane; no forced editor open from sidebar click).

## Success criteria for typical tasks
- `refresh` can sync one Confluence page by id/title/link, one Jira issue by key/id/link, and all Confluence pages when no argument is provided.
- `check` evaluates metadata relevance and returns references backed by local markdown/plain-text content.
- `rank` uses BM25 over local content with IDF fallback and is reused by sidebar search.
- Pre-process/post-process steps are integrated into refresh/annotate (`htmlToMarkdown`, tokenization keywords, LLM summary/keywords with fallback).
- Sidebar supports sync status, search, content preview, metadata edit/generate, delete doc, and add-to-prompts.
- Documentation reflects current behavior and commands.

Additional acceptance criteria (recent additions):
- `parseRefreshArg` extracts links/pageIds/titles using local heuristics first and falls back to LLM parsing only for extraction, not execution.
- Full refresh is only triggered explicitly (via user command or explicit clickable action), and ambiguous results provide a safe `Refresh All Docs` option rather than silently attempting an unresolved single-page refresh.
- For chat refresh flows, unresolved Confluence args are surfaced with an actionable fallback button instead of auto-downloading all docs.
- For tool refresh flows (`repoask_refresh`), unresolved args return a non-destructive error result instead of auto-downloading all docs.
- Chat participant behavior remains stable:
	- Explicit chat commands currently handled directly: `refresh`, `annotate`.
	- Refresh-like prompts are detected heuristically and routed through refresh handling.
	- Other prompts are answered as general prompt-context Q&A (not via direct `rank`/`check` command dispatch).

Quick validation checklist before merging any agent-driven change:
- Run `npm run compile` in `repo-ask` and ensure no JS diagnostics in the changed files.
- When changing dummy servers, run `python -m py_compile dummy-servers/confluence_server.py` and `python -m py_compile dummy-servers/jira_server.py`.
- Package the extension with `npx vsce package` to confirm manifest and contribution validity.

Current code map (verified):
- Extension runtime and commands: `repo-ask/src/extension.js`
- Document orchestration (refresh/annotate/rank): `repo-ask/src/extension/documentService/*`
- LLM tool/arg parsing helpers: `repo-ask/src/extension/llm.js`
- LM tool registration + chat refresh fallback UI: `repo-ask/src/extension/lmTools.js`
- Command registrations / definitions: `repo-ask/src/extension/commands.js`, `repo-ask/src/extension/commands/*` (e.g. `refreshParse.js`)
- Chat handlers & Prompt Context: `repo-ask/src/extension/chat/generalAnswer.js`, `repo-ask/src/extension/chat/agentPrompt.txt`, `repo-ask/src/extension/promptContext.js`
- Confluence/Jira API adapters: `repo-ask/src/confluenceApi.js`, `repo-ask/src/jiraApi.js`
- Local storage contract (doc directory with `content.md` + `metadata.json`, Indices in `local-store-index/`, plus legacy fallback): `repo-ask/src/storage.js`
- Ranking engines: `repo-ask/src/bm25.js`, `repo-ask/src/relevance.js`
- Text conversion + keyword/summary fallback: `repo-ask/src/textProcessing.js`
- Tokenization helpers: `repo-ask/src/tokenization/*` (extractors, ngrams, structural, patterns)
- Sidebar controller + UI: `repo-ask/src/extension/sidebarController.js`, `repo-ask/src/sidebar/index.html`, `repo-ask/src/sidebar/styles.css`, `repo-ask/src/sidebar/refreshPopup.html`
- Dummy servers: `dummy-servers/confluence_server.py`, `dummy-servers/jira_server.py`, `dummy-servers/generate_dummy_data.py`

Current behavior snapshot:
- Chat participant supports direct `refresh` and `annotate`; refresh-like prompts are auto-detected, and other prompts go through general prompt-context Q&A.
- `handleRefreshFromSource` parses args via `parseRefreshArg`, supports Jira issue extraction, resolves Confluence args before refresh, and provides safe fallback UI.
- `refresh` command refreshes one item (Confluence/Jira arg path) or all Confluence docs when input is empty.
- `annotate` command updates local metadata only (single doc or all docs).
- Sidebar search uses `rankLocalDocuments` (BM25 first, IDF fallback); selecting a doc updates embedded preview + metadata panel; metadata can be generated/saved; delete removes local doc-directory and legacy files; Add to Prompts writes `.github/prompts/*.prompt.md`.
- General prompt Q&A ranks metadata for context selection and streams explicit "Thinking" progress messages before returning model output.

Risks and notes:
- LLM responses can be noisy: always use `extractJsonObject` and validate outputs before executing commands.
- The extension relies on `vscode.lm` APIs; if not present, the code falls back to heuristics.
- Keep `engines.vscode` and `@types/vscode` aligned to avoid activation/typing mismatches.
- `repo-ask.refresh` command path currently parses the arg but still calls `refreshDocument(arg)` (original input) for non-Jira inputs; chat refresh/tool flows use parsed/resolved arg. Preserve this difference intentionally or align both paths explicitly when modifying refresh logic.
- Dummy servers now serve generated datasets via `generate_dummy_data.py`; keep resolve endpoints (`/rest/api/content/resolve`, `/rest/api/2/issue/resolve`) in sync with client parsing assumptions.