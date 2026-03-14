const { toToolResult, buildCheckAllDocsCommandLink } = require('./utils');

module.exports = function registerCheckTool(deps) {
    const { vscode, toolNames, documentService, readAllMetadata, emptyStoreHint } = deps;
    return vscode.lm.registerTool(toolNames.check, {
            async invoke(options) {
                const query = String(options?.input?.query || '').trim();
                const repAskConfig = vscode.workspace.getConfiguration('repoAsk');
                const initKeywordNum = repAskConfig.get('initKeywordNum') || 50;
                const rawLimit = Number(options?.input?.limit);
                const limit = Number.isFinite(rawLimit) && rawLimit > 0 ? Math.min(Math.floor(rawLimit), initKeywordNum) : 5;

                if (!query) {
                    return toToolResult('Missing required `query` input for check tool.', { references: [] });
                }

                const metadataList = readAllMetadata();
                if (metadataList.length === 0) {
                    return toToolResult(emptyStoreHint, { references: [] });
                }

                const agenticResult = documentService.checkLocalDocumentsAgentic(query, {
                    limit,
                    metadataCandidateLimit: Math.max(40, limit * 4)
                });

                if (!agenticResult.references || agenticResult.references.length === 0) {
                    return toToolResult(`No relevant documents found in local store. ${buildCheckAllDocsCommandLink(query)}`, { references: [] });
                }

                const confidentRefs = agenticResult.references.filter(r => r.score > 0);
                if (confidentRefs.length === 0) {
                    return toToolResult(`No confident local documents found for your query. Please ${buildCheckAllDocsCommandLink(query)}`, { references: [] });
                }

                const references = confidentRefs.map((ref) => ({
                    ...ref,
                    summary: ref.summary || 'No summary available',
                    reference: ref.reference || ''
                }));
                const lines = references.map((ref, index) => `${index + 1}. ${ref.title} (updated ${ref.last_updated || '-'})`);
                const summaryLines = [
                    `Top relevant RepoAsk references (agentic check):`,
                    `- Metadata scanned: ${agenticResult.metadataScanned}`,
                    `- Metadata candidates loaded for content: ${agenticResult.metadataCandidates}`,
                    `- Docs with content loaded: ${agenticResult.contentLoaded}`,
                    `- Metadata fallback used: ${agenticResult.usedMetadataFallback ? 'yes' : 'no'}`,
                    '',
                    ...lines,
                    '',
                    `Need broader confirmation? ${buildCheckAllDocsCommandLink(query)}`
                ];

                return toToolResult(summaryLines.join('\n'), {
                    references,
                    diagnostics: {
                        metadataScanned: agenticResult.metadataScanned,
                        metadataCandidates: agenticResult.metadataCandidates,
                        contentLoaded: agenticResult.contentLoaded,
                        usedMetadataFallback: agenticResult.usedMetadataFallback
                    }
                });
            }
        });

        };
