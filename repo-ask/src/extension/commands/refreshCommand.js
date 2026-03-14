module.exports = function createRefreshCommand(deps) {
    const { vscode, documentService, sidebar, storagePath, readAllMetadata } = deps;
    const { parseRefreshArg } = require('../tools/llm');

    return vscode.commands.registerCommand('repo-ask.refresh', async (arg) => {
        sidebar.setSidebarSyncStatus('Downloading from source...');
        
        // Set up timeout
        const timeoutPromise = new Promise((_, reject) => {
            setTimeout(() => {
                reject(new Error('Refresh operation timed out after 10 seconds'));
            }, 10000); // 10-second timeout
        });
        
        try {
            let result;
            if (typeof arg === 'object' && arg.type === 'recursive' && arg.arg) {
                result = await Promise.race([
                    documentService.refreshConfluenceHierarchy(arg.arg, {
                        onDocumentProcessed: (data) => {
                            sidebar.setSidebarSyncStatus(`Downloading from source... (${data.index}/${data.total})`);
                        }
                    }),
                    timeoutPromise
                ]);
            } else if (arg) {
                const parsedArg = await parseRefreshArg(vscode, arg);
                if (parsedArg.source === 'regex-jira') {
                    result = await Promise.race([
                        documentService.refreshJiraIssue(arg, {
                            onDocumentProcessed: (data) => {
                                sidebar.setSidebarSyncStatus(`Downloading from source... (${data.index}/${data.total})`);
                            }
                        }),
                        timeoutPromise
                    ]);
                } else {
                    result = await Promise.race([
                        documentService.refreshDocument(arg, {
                            onDocumentProcessed: (data) => {
                                sidebar.setSidebarSyncStatus(`Downloading from source... (${data.index}/${data.total})`);
                            }
                        }),
                        timeoutPromise
                    ]);
                }
            } else {
                result = await Promise.race([
                    documentService.refreshAllDocuments({
                        onDocumentProcessed: (data) => {
                            sidebar.setSidebarSyncStatus(`Downloading from source... (${data.index}/${data.total})`);
                        }
                    }),
                    timeoutPromise
                ]);
            }
            
            sidebar.setSidebarSyncStatus('');
            sidebar.setSidebarSyncError('');
            sidebar.refreshSidebarView();
        } catch (error) {
            sidebar.setSidebarSyncStatus('');
            sidebar.setSidebarSyncError(error.message);
            vscode.window.showErrorMessage(`Refresh failed: ${error.message}`);
        }
    });
};
