function createLanguageModelTools(deps) {
    const {
        vscode,
        context,
        documentService,
        parseRefreshArg,
        fetchConfluencePage,
        setSidebarSyncStatus,
        refreshSidebarView,
        readAllMetadata,
        readDocumentContent,
        findRelevantDocuments,
        tokenize,
        truncate,
        emptyStoreHint,
        toolNames
    } = deps;

    function toToolResult(text, data) {
        const parts = [new vscode.LanguageModelTextPart(String(text || ''))];
        if (vscode.LanguageModelDataPart && typeof vscode.LanguageModelDataPart.json === 'function' && data !== undefined) {
            parts.push(vscode.LanguageModelDataPart.json(data));
        }
        return new vscode.LanguageModelToolResult(parts);
    }

    function registerRepoAskLanguageModelTools() {
        if (!vscode.lm || typeof vscode.lm.registerTool !== 'function') {
            return [];
        }

        const refreshTool = vscode.lm.registerTool(toolNames.refresh, {
            prepareInvocation(options) {
                const arg = String(options?.input?.arg || '').trim();
                return {
                    invocationMessage: arg.length > 0
                        ? `Refreshing RepoAsk document source for: ${arg}`
                        : 'Refreshing all RepoAsk Confluence documents',
                    confirmationMessages: {
                        title: arg.length > 0 ? 'Refresh RepoAsk document source?' : 'Refresh all RepoAsk Confluence docs?',
                        message: arg.length > 0
                            ? `This will sync local-store from: ${arg}`
                            : 'This will sync all Confluence pages into local-store.'
                    }
                };
            },
            async invoke(options) {
                const arg = String(options?.input?.arg || '').trim();
                try {
                    if (!arg) {
                        setSidebarSyncStatus('downloading from confluence cloud ...');
                        await documentService.refreshAllDocuments();
                        setSidebarSyncStatus('');
                        refreshSidebarView();
                        return toToolResult('Refreshed all Confluence documents into local-store.', { refreshed: 'all' });
                    }

                    const parsed = await parseRefreshArg(vscode, arg);
                    if (parsed.found && parsed.source === 'regex-jira') {
                        setSidebarSyncStatus('downloading from jira ...');
                        await documentService.refreshJiraIssue(parsed.arg);
                        setSidebarSyncStatus('');
                        refreshSidebarView();
                        return toToolResult(`Refreshed Jira issue for: ${parsed.arg}`, { refreshed: parsed.arg, source: 'jira' });
                    }

                    if (parsed.found && parsed.arg) {
                        await fetchConfluencePage(parsed.arg);
                        setSidebarSyncStatus('downloading from confluence cloud ...');
                        await documentService.refreshDocument(parsed.arg);
                        setSidebarSyncStatus('');
                        refreshSidebarView();
                        return toToolResult(`Refreshed Confluence page for: ${parsed.arg}`, { refreshed: parsed.arg, source: 'confluence' });
                    }

                    return toToolResult(
                        'Could not resolve a Confluence page id/title/link or Jira issue key/id/link. Provide an explicit arg, or call this tool with empty arg to refresh all Confluence docs.',
                        { refreshed: false, reason: 'unresolved-arg' }
                    );
                } catch (error) {
                    setSidebarSyncStatus('');
                    return toToolResult(`Refresh failed: ${error.message}`, { refreshed: false, error: error.message });
                }
            }
        });

        const annotateTool = vscode.lm.registerTool(toolNames.annotate, {
            prepareInvocation(options) {
                const arg = String(options?.input?.arg || '').trim();
                return {
                    invocationMessage: arg.length > 0
                        ? `Annotating RepoAsk document: ${arg}`
                        : 'Annotating all RepoAsk local documents',
                    confirmationMessages: {
                        title: arg.length > 0 ? 'Annotate selected RepoAsk document?' : 'Annotate all RepoAsk local documents?',
                        message: arg.length > 0
                            ? `This will recompute summary and keywords for: ${arg}`
                            : 'This will recompute summary and keywords for all local documents.'
                    }
                };
            },
            async invoke(options) {
                const arg = String(options?.input?.arg || '').trim();
                try {
                    const result = arg.length > 0
                        ? await documentService.annotateDocumentByArg(arg)
                        : await documentService.annotateAllDocuments();
                    refreshSidebarView();
                    return toToolResult(result.message, { annotated: true, arg: arg || '' });
                } catch (error) {
                    return toToolResult(`Annotate failed: ${error.message}`, { annotated: false, error: error.message });
                }
            }
        });

        const rankTool = vscode.lm.registerTool(toolNames.rank, {
            async invoke(options) {
                const query = String(options?.input?.query || '').trim();
                const rawLimit = Number(options?.input?.limit);
                const limit = Number.isFinite(rawLimit) && rawLimit > 0 ? Math.min(Math.floor(rawLimit), 50) : 5;

                if (!query) {
                    return toToolResult('Missing required `query` input for rank tool.', { results: [] });
                }

                const ranked = documentService.rankLocalDocuments(query, limit);
                if (!ranked || ranked.length === 0) {
                    return toToolResult('No matching local documents found for the query.', { results: [] });
                }

                const results = ranked.map(item => ({
                    id: item.id,
                    title: item.title || 'Untitled',
                    score: Number(item.score?.toFixed ? item.score.toFixed(4) : item.score),
                    summary: truncate(item.summary || '', 220)
                }));
                const lines = results.map((item, index) => `${index + 1}. ${item.title} (score ${item.score})`);
                return toToolResult(`Top ranked RepoAsk documents:\n${lines.join('\n')}`, { results });
            }
        });

        const checkTool = vscode.lm.registerTool(toolNames.check, {
            async invoke(options) {
                const query = String(options?.input?.query || '').trim();
                const rawLimit = Number(options?.input?.limit);
                const limit = Number.isFinite(rawLimit) && rawLimit > 0 ? Math.min(Math.floor(rawLimit), 20) : 5;

                if (!query) {
                    return toToolResult('Missing required `query` input for check tool.', { references: [] });
                }

                const metadataList = readAllMetadata();
                if (metadataList.length === 0) {
                    return toToolResult(emptyStoreHint, { references: [] });
                }

                const relevantDocs = findRelevantDocuments(query, metadataList, tokenize).slice(0, limit);
                if (relevantDocs.length === 0) {
                    return toToolResult('No relevant documents found in local store.', { references: [] });
                }

                const references = relevantDocs.map(doc => {
                    const content = readDocumentContent(doc.id) || '';
                    return {
                        id: doc.id,
                        title: doc.title || 'Untitled',
                        author: doc.author || 'Unknown',
                        last_updated: doc.last_updated || '',
                        summary: truncate(doc.summary || 'No summary available', 220),
                        reference: truncate(content, 500)
                    };
                });

                const lines = references.map((ref, index) => `${index + 1}. ${ref.title} (updated ${ref.last_updated || '-'})`);
                return toToolResult(`Top relevant RepoAsk references:\n${lines.join('\n')}`, { references });
            }
        });

        return [refreshTool, annotateTool, rankTool, checkTool];
    }

    async function handleRefreshFromSource(sourceInput, response) {
        const parsed = await parseRefreshArg(vscode, sourceInput);

        if (parsed.found && parsed.source === 'regex-jira') {
            response.markdown(`Refreshing Jira issue for: ${parsed.arg}...`);
            try {
                setSidebarSyncStatus('downloading from jira ...');
                await documentService.refreshJiraIssue(parsed.arg);
                response.markdown('Refresh completed for the Jira issue.');
                setSidebarSyncStatus('');
                refreshSidebarView();
            } catch (error) {
                setSidebarSyncStatus('');
                const status = error?.response?.status;
                const detail = status ? `backend returned ${status}` : (error?.message || 'backend request failed');
                response.markdown(`Refresh failed for Jira issue ${parsed.arg} (${detail}).`);
            }
            return;
        }

        if (parsed.found && parsed.arg) {
            try {
                await fetchConfluencePage(parsed.arg);
            } catch (error) {
                const status = error?.response?.status;
                const detail = status ? `backend returned ${status}` : 'backend request failed';
                response.markdown(`Could not resolve the requested document (${detail}). Do you want to download all docs instead?`);
                appendRefreshAllDocsButton(response);
                return;
            }

            response.markdown(`Refreshing document for: ${parsed.arg}...`);
            try {
                setSidebarSyncStatus('downloading from confluence cloud ...');
                await documentService.refreshDocument(parsed.arg);
                response.markdown('Refresh completed for the resolved page.');
                setSidebarSyncStatus('');
                refreshSidebarView();
            } catch (error) {
                setSidebarSyncStatus('');
                const status = error?.response?.status;
                const detail = status ? `backend returned ${status}` : (error?.message || 'backend request failed');
                response.markdown(`Refresh failed for the requested page (${detail}). Do you want to download all docs instead?`);
                appendRefreshAllDocsButton(response);
            }
            return;
        }

        response.markdown(`I couldn't find a Confluence link, page id, or exact page title in your request. Do you want to download all docs instead?`);
        appendRefreshAllDocsButton(response);
    }

    function appendRefreshAllDocsButton(response) {
        response.button({
            command: 'repo-ask.refresh',
            title: 'Refresh All Docs',
            arguments: ['']
        });
    }

    return {
        registerRepoAskLanguageModelTools,
        handleRefreshFromSource
    };
}

module.exports = {
    createLanguageModelTools
};
