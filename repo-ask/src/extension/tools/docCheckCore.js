/**
 * Core document-check logic shared by the VS Code LM tool and the LangChain agent tool.
 *
 * Exported function: runDocCheck(input, deps) → Promise<string>
 * The caller is responsible for wrapping the return value in whatever result type
 * their framework expects (LanguageModelToolResult for VS Code, plain string for LangChain).
 */

const { ALLOWED_MODES, PARTIAL_CONTENT_NOTE } = require('../chat/prompts');

const MAX_PARTIAL_CHARS = 600;

/**
 * Execute the document-check operation and return a plain text result.
 *
 * @param {{ query?: string, mode?: string, ids?: string[], searchTerms?: string[], limit?: number }} input
 * @param {{ documentService: object, readAllMetadata: function, readDocumentMetadata: function, readDocumentContent: function, emptyStoreHint?: string }} deps
 * @returns {Promise<string>}
 */
async function runDocCheck(input, deps) {
    const { query, mode, ids, searchTerms, limit } = input || {};
    const { documentService, readAllMetadata, readDocumentMetadata, readDocumentContent, emptyStoreHint } = deps;
    const maxLimit = Math.max(1, Number(limit) || 3);

    // ── searchTerms → rank and return matching doc IDs ───────────────────────
    if (Array.isArray(searchTerms) && searchTerms.length > 0) {
        const searchQuery = searchTerms.join(' ');
        const ranked = documentService.rankLocalDocuments(searchQuery, maxLimit * 3);
        if (!ranked || ranked.length === 0) {
            return `No documents found for search terms: ${searchTerms.join(', ')}`;
        }
        const topDocs = ranked.slice(0, maxLimit);
        const lines = topDocs.map(d =>
            `ID: ${d.id} | Title: ${d.title || 'Untitled'} | Score: ${Math.round((d.score || 0) * 10) / 10}`
        );
        return `Found ${topDocs.length} relevant doc(s):\n${lines.join('\n')}`;
    }

    // ── Determine target IDs ─────────────────────────────────────────────────
    let targetIds;
    if (Array.isArray(ids) && ids.length > 0) {
        targetIds = ids.map(String);
    } else {
        const allMeta = readAllMetadata();
        if (!allMeta || allMeta.length === 0) {
            return emptyStoreHint || 'No documents found in store.';
        }
        targetIds = allMeta.map(m => String(m.id));
    }

    switch (mode) {
        case 'id_2_content': {
            const parts = [];
            for (const id of targetIds) {
                const content = readDocumentContent(id);
                const meta = readDocumentMetadata(id);
                if (content !== null && content !== undefined) {
                    parts.push(`### [${meta?.title || id}] (ID: ${id})\n${content}`);
                }
            }
            return parts.length ? parts.join('\n\n---\n\n') : 'No content found.';
        }

        case 'id_2_metadata': {
            const parts = [];
            for (const id of targetIds) {
                const meta = readDocumentMetadata(id);
                if (meta) parts.push(JSON.stringify(meta, null, 2));
            }
            return parts.length ? parts.join('\n\n---\n\n') : 'No metadata found.';
        }

        case 'id_2_content_partial': {
            const parts = [];
            for (const id of targetIds) {
                const content = readDocumentContent(id);
                const meta = readDocumentMetadata(id);
                if (content) {
                    const truncated = content.length > MAX_PARTIAL_CHARS;
                    const snippet = truncated ? content.slice(0, MAX_PARTIAL_CHARS) + '...' : content;
                    parts.push(
                        `### [${meta?.title || id}] (ID: ${id})\n${snippet}` +
                        (truncated ? `\n${PARTIAL_CONTENT_NOTE}` : '')
                    );
                }
            }
            return parts.length ? parts.join('\n\n---\n\n') : 'No content found.';
        }

        case 'id_2_metadata_4_summary': {
            const parts = [];
            for (const id of targetIds) {
                const meta = readDocumentMetadata(id);
                if (meta) {
                    parts.push(JSON.stringify({
                        id: meta.id,
                        title: meta.title || '',
                        summary: meta.summary || '',
                        tags: meta.tags || [],
                        type: meta.type || '',
                        url: meta.url || ''
                    }));
                }
            }
            return parts.length ? parts.join('\n') : 'No metadata found.';
        }

        case 'id_2_metadata_4_summary_kg': {
            // Include KG string and traverse relatedPages (1 hop) for context
            const visited = new Set(targetIds);
            const allIds = [...targetIds];
            for (const id of targetIds) {
                const meta = readDocumentMetadata(id);
                if (meta && Array.isArray(meta.relatedPages)) {
                    for (const relId of meta.relatedPages.map(String)) {
                        if (!visited.has(relId)) {
                            visited.add(relId);
                            allIds.push(relId);
                        }
                    }
                }
            }
            const parts = [];
            for (const id of allIds) {
                const meta = readDocumentMetadata(id);
                if (meta) {
                    const isTraversed = !targetIds.includes(id);
                    parts.push(JSON.stringify({
                        id: meta.id,
                        title: meta.title || '',
                        summary: meta.summary || '',
                        tags: meta.tags || [],
                        type: meta.type || '',
                        url: meta.url || '',
                        knowledgeGraph: meta.knowledgeGraph || '',
                        relatedPages: meta.relatedPages || [],
                        _traversed: isTraversed
                    }));
                }
            }
            return parts.length ? parts.join('\n') : 'No metadata found.';
        }

        default:
            return `Unknown mode: "${mode}". Allowed: ${ALLOWED_MODES.join(', ')}`;
    }
}

module.exports = { runDocCheck };
