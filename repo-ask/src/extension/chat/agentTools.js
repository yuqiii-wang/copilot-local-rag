/**
 * LangChain StructuredTool definitions for the RepoAsk agent.
 *
 * Each tool wraps a VS Code LM tool invocation (vscode.lm.invokeTool) inside a
 * LangChain StructuredTool so the LangChain agent loop can call them uniformly.
 * The underlying VS Code tool is still the authoritative implementation — these
 * are thin execution adapters, not reimplementations.
 */

const { tool } = require('@langchain/core/tools');
const { z } = require('zod');
const path = require('path');
const { DOC_CHECK_TOOL_DESCRIPTION } = require('./prompts');
const { ALLOWED_MODES } = require('../tools/vsCodeTools');
const { runDocCheck } = require('../tools/docCheckCore');

/**
 * Build LangChain StructuredTool instances for the RepoAsk agent.
 *
 * @param {Object} deps
 * @param {Object} deps.vscodeApi       - The `vscode` module
 * @param {Object} deps.options         - Chat request options
 * @param {string} deps.storagePath     - Local doc store root path (for file references)
 * @param {Object} deps.response        - VS Code chat response stream
 * @param {Object} deps.documentService - Document service (for rankLocalDocuments)
 * @param {Function} deps.readAllMetadata
 * @param {Function} deps.readDocumentMetadata
 * @param {Function} deps.readDocumentContent
 * @param {string}  [deps.emptyStoreHint]
 * @returns {Array} Array of LangChain StructuredTool instances
 */
function buildAgentTools({ vscodeApi, options, storagePath, response, documentService, readAllMetadata, readDocumentMetadata, readDocumentContent, emptyStoreHint }) {
    const docCheckTool = tool(
        async ({ query, mode, ids, searchTerms, limit }) => {
            try {
                const text = await runDocCheck(
                    { query, mode, ids, searchTerms, limit },
                    { documentService, readAllMetadata, readDocumentMetadata, readDocumentContent, emptyStoreHint }
                );

                // Emit VS Code file references so the Copilot UI shows the doc links
                if (storagePath && Array.isArray(ids) && ids.length > 0
                    && typeof response?.reference === 'function') {
                    for (const id of ids) {
                        const docPath = path.join(storagePath, id, 'content.md');
                        response.reference(vscodeApi.Uri.file(docPath));
                    }
                }

                return text || 'No content found.';
            } catch (err) {
                return `Tool error: ${err.message}`;
            }
        },
        {
            name: 'repoask_doc_check',
            description: DOC_CHECK_TOOL_DESCRIPTION,
            schema: z.object({
                query: z.string().describe(
                    'User question or keywords to match against local document metadata and content.'
                ),
                mode: z.enum(ALLOWED_MODES)
                    .default('id_2_title')
                    .describe(
                        'Read mode. id_2_title: return titles for provided doc ids. id_2_content_partial: quick snippet scan. id_2_content: full text. id_2_metadata_4_summary: evaluate by summaries. id_2_metadata_4_summary_kg: KG traversal for advanced search.'
                    ),
                ids: z.array(z.string()).optional().default([])
                    .describe(
                        'Specific document IDs to read. Omit to operate across all stored docs.'
                    ),
                searchTerms: z.array(z.string()).optional().default([])
                    .describe(
                        'Search terms proposed by LLM based on the user query and conversation history. Used to narrow results when no specific IDs are given.'
                    ),
                limit: z.number().optional().default(3)
                    .describe('Max number of results to return for searchTerms mode.')
            })
        }
    );

    return [docCheckTool];
}  

/**
 * Build a name→tool lookup map from a LangChain tool array.
 * @param {Array} tools
 * @returns {Object.<string, import('@langchain/core/tools').StructuredTool>}
 */
function buildToolMap(tools) {
    return Object.fromEntries(tools.map(t => [t.name, t]));
}

module.exports = { buildAgentTools, buildToolMap };
