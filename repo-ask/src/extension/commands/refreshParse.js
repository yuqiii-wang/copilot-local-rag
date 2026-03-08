function registerRefreshAndParseCommands(deps) {
    const {
        vscode,
        sidebar,
        documentService,
        parseRefreshArg
    } = deps;

    const refreshDisposable = vscode.commands.registerCommand('repo-ask.refresh', async function (directArg) {
        const arg = typeof directArg === 'string' ? directArg : await vscode.window.showInputBox({
            prompt: 'Enter Confluence page id/title/link or Jira issue key, or leave empty to refresh all Confluence docs',
            placeHolder: 'e.g., 1, Technical Documentation Guide, Confluence URL, or PROJECT-1003'
        });

        try {
            // Do not clear status here to preserve "downloading..." state from UI
            // sidebar.setSidebarSyncStatus('');
            if (typeof sidebar.setSidebarSyncError === 'function') {
                sidebar.setSidebarSyncError('');
            }
            if (typeof sidebar.setSidebarSyncSuccess === 'function') {
                sidebar.setSidebarSyncSuccess('');
            }
            const formatRefreshStatus = (sourceLabel, progress = {}) => {
                const index = Number(progress?.index);
                const total = Number(progress?.total);
                const hasFraction = Number.isFinite(index) && Number.isFinite(total) && total > 0;
                const progressSuffix = hasFraction ? ` (${index}/${total})` : '';
                return `downloading from ${sourceLabel} ...${progressSuffix}`;
            };

            const createRefreshOptions = (sourceLabel) => ({
                onDocumentProcessed: ({ metadata, index, total }) => {
                    sidebar.upsertSidebarDocument(metadata);
                    sidebar.setSidebarSyncStatus(formatRefreshStatus(sourceLabel, { index, total }));
                }
            });

            if (arg && arg.trim().length > 0) {
                const parsed = await parseRefreshArg(vscode, arg.trim());
                sidebar.setSidebarSyncStatus('downloading from confluence/jira cloud ...');
                if (parsed.found && parsed.source === 'regex-jira') {
                    await documentService.refreshJiraIssue(parsed.arg, createRefreshOptions('jira'));
                    const successMsg = `Refreshed Jira issue for: ${parsed.arg}`;
                    vscode.window.showInformationMessage(successMsg);
                    if (typeof sidebar.setSidebarSyncSuccess === 'function') {
                        sidebar.setSidebarSyncSuccess(successMsg);
                    }
                } else {
                    const resolvedArg = parsed.found && parsed.arg ? parsed.arg : arg.trim();
                    await documentService.refreshDocument(resolvedArg, createRefreshOptions('confluence cloud'));
                    const successMsg = `Refreshed document for: ${resolvedArg}`;
                    vscode.window.showInformationMessage(successMsg);
                    if (typeof sidebar.setSidebarSyncSuccess === 'function') {
                        sidebar.setSidebarSyncSuccess(successMsg);
                    }
                }
            } else {
                const downloadingMessage = 'downloading from confluence/jira cloud ...';
                vscode.window.showInformationMessage(downloadingMessage);
                sidebar.setSidebarSyncStatus(downloadingMessage);
                await documentService.refreshAllDocuments(createRefreshOptions('confluence cloud'));
                const successMsg = 'Refreshed all documents';
                vscode.window.showInformationMessage(successMsg);
                if (typeof sidebar.setSidebarSyncSuccess === 'function') {
                    sidebar.setSidebarSyncSuccess(successMsg);
                }
            }

            sidebar.refreshSidebarView();
        } catch (error) {
            const errorMsg = error instanceof Error ? error.message : String(error);
            vscode.window.showErrorMessage(`Error refreshing documents: ${errorMsg}`);

            if (typeof sidebar.setSidebarSyncError === 'function') {
                const target = (arg && arg.trim().length > 0) ? `"${arg.trim()}"` : 'all documents';
                sidebar.setSidebarSyncError(`Failed to sync ${target}: ${errorMsg}`);
            }
        } finally {
            sidebar.setSidebarSyncStatus('');
            sidebar.refreshSidebarView();
        }
    });

    const parseArgDisposable = vscode.commands.registerCommand('repo-ask.parseArg', async function (sourceInput) {
        return await parseRefreshArg(vscode, sourceInput);
    });

    return [refreshDisposable, parseArgDisposable];
}

module.exports = {
    registerRefreshAndParseCommands
};
