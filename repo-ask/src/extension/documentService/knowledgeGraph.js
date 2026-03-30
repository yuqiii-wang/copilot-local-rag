/**
 * knowledgeGraph.js — Unified knowledge graph generation for RepoAsk documents.
 *
 * Consolidates all KG-building logic previously scattered across:
 *   - sidebarController.js (buildKnowledgeGraphForForm — orchestration)
 *   - tools/llm.js          (generateKnowledgeGraph   — LLM call)
 *
 * The central export `buildKnowledgeGraph` accepts a primary document ID (or raw
 * content), an optional list of secondary page URLs/IDs, and optional reference
 * queries.  It resolves every piece from the local document store, expands any
 * Jira cross-references found in the primary content, then calls the LLM and
 * returns a validated Mermaid flowchart string.
 *
 * `saveKnowledgeGraph` persists the resulting Mermaid string back to the
 * document's stored metadata and triggers a BM25 keyword refresh.
 */

module.exports = function (context) {
    const {
        vscode,
        storagePath,
        readAllMetadata,
        readDocumentContent,
        writeDocumentFiles,
        getStoredMetadataById,
        finalizeBm25KeywordsForDocuments
    } = context;

    const { generateKnowledgeGraph: generateKgLlm, getJiraExtractionRegexes } = require('../tools/llm');

    // ── Internal helpers ───────────────────────────────────────────────────────

    /**
     * Resolve a single primary doc from one or more identifiers.
     * Accepts a numeric/string Confluence page ID, a Jira issue key, or a full
     * Confluence/Jira URL (pageId= query-param extraction).
     *
     * @param {string|null} primaryDocId   - Direct numeric or Jira ID
     * @param {string|null} confluenceLink - Full URL (optional fallback)
     * @returns {object|null} metadata record or null
     */
    function resolvePrimaryDoc(primaryDocId, confluenceLink) {
        const allMetadata = readAllMetadata(storagePath);
        if (primaryDocId) {
            const found = allMetadata.find(d => String(d.id) === String(primaryDocId) ||
                String(d.issueKey || '').toUpperCase() === String(primaryDocId).toUpperCase());
            if (found) return found;
        }
        if (confluenceLink) {
            try {
                const urlObj = new URL(confluenceLink);
                const m = urlObj.search.match(/pageId=(\d+)/);
                if (m && m[1]) {
                    return allMetadata.find(d => String(d.id) === String(m[1])) || null;
                }
            } catch (_) {}
        }
        return null;
    }

    /**
     * Collect content for a list of secondary URLs/IDs from the local store.
     * Returns a map of url → content string.
     *
     * @param {string[]} urls
     * @returns {{ contentMap: Object, resolvedUrls: string[] }}
     */
    function collectSecondaryContent(urls) {
        const allMetadata = readAllMetadata(storagePath);
        const contentMap = {};
        const resolvedUrls = [...urls];

        for (const url of urls) {
            let docContent = '';
            try {
                const urlObj = new URL(url);
                const pageIdMatch = urlObj.search.match(/pageId=(\d+)/);
                if (pageIdMatch && pageIdMatch[1]) {
                    const doc = allMetadata.find(d => String(d.id) === String(pageIdMatch[1]));
                    if (doc) docContent = readDocumentContent(storagePath, doc.id) || '';
                }
            } catch (_) {
                // Not a URL — try treating it as a direct doc ID or Jira key
                const doc = allMetadata.find(d =>
                    String(d.id) === String(url) ||
                    String(d.issueKey || '').toUpperCase() === String(url).toUpperCase()
                );
                if (doc) docContent = readDocumentContent(storagePath, doc.id) || '';
            }
            contentMap[url] = docContent;
        }

        return { contentMap, resolvedUrls };
    }

    /**
     * Expand Jira cross-references embedded in primary content and add them to
     * the secondary content map so the LLM can see what those tickets say.
     *
     * Falls back to a broad regex scan of `primaryContent` when `referencedJiraIds`
     * is absent from the primary doc's metadata (pre-dates the field).
     *
     * @param {object|null} primaryDoc
     * @param {string}      primaryContent
     * @param {string[]}    existingUrls    - Already-resolved secondary URL list (mutated in place)
     * @param {Object}      contentMap      - Content map (mutated in place)
     */
    function expandJiraReferences(primaryDoc, primaryContent, existingUrls, contentMap) {
        const allMetadata = readAllMetadata(storagePath);

        const jiraKeysToLoad = new Set(
            Array.isArray(primaryDoc?.referencedJiraIds) && primaryDoc.referencedJiraIds.length > 0
                ? primaryDoc.referencedJiraIds
                : (() => {
                    const found = new Set();
                    for (const re of getJiraExtractionRegexes(vscode)) {
                        const gr = new RegExp(re.source, re.flags.includes('i') ? 'gi' : 'g');
                        for (const m of (primaryContent || '').matchAll(gr)) {
                            found.add(m[0].toUpperCase());
                        }
                    }
                    return [...found];
                })()
        );

        for (const jiraKey of jiraKeysToLoad) {
            const refDoc = allMetadata.find(d =>
                String(d.issueKey || '').toUpperCase() === jiraKey ||
                String(d.title || '').toUpperCase().startsWith(jiraKey + ':') ||
                String(d.id).toUpperCase() === jiraKey
            );
            if (refDoc && !existingUrls.includes(jiraKey)) {
                const refContent = readDocumentContent(storagePath, refDoc.id) || '';
                existingUrls.push(jiraKey);
                contentMap[jiraKey] = `[${jiraKey}] ${refDoc.title || ''}\n${refContent.slice(0, 800)}`;
            }
        }
    }

    // ── Public API ─────────────────────────────────────────────────────────────

    /**
     * Build (or update) a Mermaid knowledge graph for a document.
     *
     * Workflow:
     *   1. Resolve the primary document from `primaryDocId` / `confluenceLink`.
     *   2. Read its stored content and existing KG (unless overridden by caller).
     *   3. Collect secondary URL/ID contents from the local store.
     *   4. Auto-expand any Jira tickets cross-referenced in the primary content.
     *   5. Merge caller-supplied `referenceQueries` with the doc's stored ones.
     *   6. Call the LLM (via tools/llm.js `generateKnowledgeGraph`) and return
     *      the resulting Mermaid string.
     *
     * @param {object} params
     * @param {string|null}   params.primaryDocId          - Confluence page ID or Jira key
     * @param {string|null}   [params.confluenceLink]      - Full Confluence/Jira URL (fallback resolver)
     * @param {string[]}      [params.secondaryUrls]       - Additional page URLs/IDs to include
     * @param {string[]}      [params.referenceQueries]    - User queries to seed the graph focus
     * @param {string}        [params.existingKnowledgeGraph] - Current mermaid (caller may have edited it)
     * @param {string}        [params.primaryContent]      - Override doc content (skips storage read)
     * @returns {Promise<string>} Mermaid flowchart string
     */
    async function buildKnowledgeGraph({
        primaryDocId = null,
        confluenceLink = null,
        secondaryUrls = [],
        referenceQueries = [],
        existingKnowledgeGraph,
        primaryContent: passedPrimaryContent,
        conversationSummary
    } = {}) {
        const urls = Array.isArray(secondaryUrls)
            ? secondaryUrls.filter(u => String(u || '').trim() && String(u) !== 'none')
            : [];

        // 1. Resolve primary document
        const primaryDoc = resolvePrimaryDoc(primaryDocId, confluenceLink);

        // 2. Collect secondary content from local store
        const { contentMap, resolvedUrls } = collectSecondaryContent(urls);

        // 3. Read primary content and existing KG
        let primaryContent = passedPrimaryContent || '';
        let kgToUse = typeof existingKnowledgeGraph === 'string' ? existingKnowledgeGraph.trim() : '';
        let queries = Array.isArray(referenceQueries) ? [...referenceQueries] : [];

        if (primaryDoc) {
            if (!primaryContent) {
                primaryContent = readDocumentContent(storagePath, primaryDoc.id) || '';
            }
            if (!kgToUse) {
                kgToUse = String(primaryDoc.knowledgeGraph || '').trim();
            }
            // Merge stored reference queries (dedup)
            const storedQueries = Array.isArray(primaryDoc.referencedQueries) ? primaryDoc.referencedQueries : [];
            queries = [...new Set([...queries, ...storedQueries])];

            // 4. Expand Jira cross-references into the content map
            expandJiraReferences(primaryDoc, primaryContent, resolvedUrls, contentMap);
        }

        // 5. Call LLM
        return await generateKgLlm(vscode, queries, resolvedUrls, contentMap, {
            primaryContent,
            existingKnowledgeGraph: kgToUse,
            conversationSummary: String(conversationSummary || '').trim() || undefined
        });
    }

    /**
     * Persist a Mermaid knowledge graph string to a document's stored metadata
     * and trigger a BM25 keyword refresh (so KG entity tokens are indexed).
     *
     * @param {string} docId    - Document ID
     * @param {string} mermaid  - Validated Mermaid flowchart string
     */
    async function saveKnowledgeGraph(docId, mermaid) {
        const metadata = getStoredMetadataById(docId);
        if (!metadata) {
            console.warn('[saveKnowledgeGraph] Document not found:', docId);
            return;
        }
        const content = readDocumentContent(storagePath, docId);
        if (!content) {
            console.warn('[saveKnowledgeGraph] No content for document:', docId);
            return;
        }
        writeDocumentFiles(storagePath, docId, content, { ...metadata, knowledgeGraph: String(mermaid || '') });
        if (typeof finalizeBm25KeywordsForDocuments === 'function') {
            await finalizeBm25KeywordsForDocuments([docId]);
        }
    }

    return { buildKnowledgeGraph, saveKnowledgeGraph };
};
