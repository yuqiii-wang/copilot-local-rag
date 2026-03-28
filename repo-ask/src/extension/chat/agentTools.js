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
const { DOC_CHECK_TOOL_DESCRIPTION, ALLOWED_MODES } = require('./prompts');

/**
 * Build LangChain StructuredTool instances for the RepoAsk agent.
 *
 * @param {Object} deps
 * @param {Object} deps.vscodeApi     - The `vscode` module
 * @param {Object} deps.options       - Chat request options (provides toolInvocationToken)
 * @param {string} deps.storagePath   - Local doc store root path (for file references)
 * @param {Object} deps.response      - VS Code chat response stream (for response.reference)
 * @returns {Array} Array of LangChain StructuredTool instances
 */
function buildAgentTools({ vscodeApi, options, storagePath, response }) {
    /**
     * repoask_doc_check — reads documents from the local RepoAsk store.
     * Delegates to the registered VS Code "repoask_doc_check" LM tool.
     */
    const docCheckTool = tool(
        async ({ query, mode, ids, searchTerms }) => {
            try {
                const result = await vscodeApi.lm.invokeTool(
                    'repoask_doc_check',
                    {
                        input: {
                            query: query || '',
                            mode: mode || 'content_partial',
                            ids: ids || [],
                            searchTerms: searchTerms || []
                        },
                        toolInvocationToken: options?.request?.toolInvocationToken
                    }
                );

                // Emit VS Code file references so Copilot-style UI shows the docs
                if (storagePath && Array.isArray(ids) && ids.length > 0
                    && typeof response?.reference === 'function') {
                    for (const id of ids) {
                        const docPath = path.join(storagePath, id, 'content.md');
                        response.reference(vscodeApi.Uri.file(docPath));
                    }
                }

                return (result.content || [])
                    .filter(p => p instanceof vscodeApi.LanguageModelTextPart)
                    .map(p => p.value)
                    .join('\n')
                    .trim() || 'No content found.';
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
                    .default('content_partial')
                    .describe(
                        'Read mode. Use metadata.id to list docs, metadata.summary_kg for summaries with KG, content_partial for quick scan, content for full read.'
                    ),
                ids: z.array(z.string()).optional().default([])
                    .describe(
                        'Specific document IDs to read. Omit to operate across all stored docs.'
                    ),
                searchTerms: z.array(z.string()).optional().default([])
                    .describe(
                        'Search terms proposed by LLM based on the user query and conversation history. Used to narrow results when no specific IDs are given.'
                    )
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
