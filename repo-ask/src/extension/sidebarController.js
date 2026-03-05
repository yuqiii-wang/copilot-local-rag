const fs = require('fs');
const path = require('path');

function createSidebarController(deps) {
    const {
        vscode,
        context,
        storagePath,
        documentService,
        readAllMetadata,
        readDocumentContent,
        deleteDocumentFiles
    } = deps;

    let docsWebviewView;
    let sidebarSyncStatus = '';

    const sidebarProvider = {
        resolveWebviewView: async (webviewView) => {
            try {
                docsWebviewView = webviewView;
                docsWebviewView.webview.options = {
                    enableScripts: true,
                    localResourceRoots: [vscode.Uri.joinPath(context.extensionUri, 'src', 'sidebar')]
                };

                docsWebviewView.webview.onDidReceiveMessage(async (message) => {
                    if (message?.command === 'openDoc' && message.docId) {
                        const metadata = readAllMetadata(storagePath).find(doc => String(doc.id) === String(message.docId));
                        const content = metadata
                            ? (readDocumentContent(storagePath, metadata.id) || 'No local markdown content found.')
                            : 'No local markdown content found.';

                        docsWebviewView.webview.postMessage({
                            command: 'docDetails',
                            payload: {
                                id: message.docId,
                                content,
                                metadata: documentService.formatMetadataEntries(metadata)
                            }
                        });
                    }

                    if (message?.command === 'searchDocs') {
                        const query = String(message.query || '').trim();
                        const results = query.length > 0
                            ? documentService.rankLocalDocuments(query, 50)
                            : readAllMetadata(storagePath)
                                .sort((a, b) => String(b.last_updated).localeCompare(String(a.last_updated)));

                        docsWebviewView.webview.postMessage({
                            command: 'searchResults',
                            payload: results.map(doc => ({
                                id: doc.id,
                                title: doc.title || 'Untitled'
                            }))
                        });
                    }

                    if (message?.command === 'addToPrompts') {
                        const docId = String(message.docId || '').trim();
                        if (!docId) {
                            vscode.window.showWarningMessage('Select a document first to add it to prompts.');
                            return;
                        }

                        const metadata = readAllMetadata(storagePath).find(doc => String(doc.id) === docId);
                        if (!metadata) {
                            vscode.window.showWarningMessage('Document metadata not found. Run refresh and try again.');
                            return;
                        }

                        const content = readDocumentContent(storagePath, metadata.id);
                        if (!content || String(content).trim().length === 0) {
                            vscode.window.showWarningMessage('Local document content is empty. Refresh this doc and try again.');
                            return;
                        }

                        try {
                            const createdPath = documentService.writeDocumentPromptFile(metadata, content);
                            vscode.window.showInformationMessage(`Added prompt: ${path.basename(createdPath)}`);
                        } catch (error) {
                            vscode.window.showErrorMessage(`Failed to add prompt: ${error.message}`);
                        }
                    }

                    if (message?.command === 'deleteDoc') {
                        const docId = String(message.docId || '').trim();
                        const docTitle = String(message.title || docId || 'this document').trim();
                        if (!docId) {
                            vscode.window.showWarningMessage('Select a document first to delete.');
                            return;
                        }

                        try {
                            const confirmation = await vscode.window.showWarningMessage(
                                `Delete local document "${docTitle}"?`,
                                { modal: true },
                                'Delete'
                            );
                            if (confirmation !== 'Delete') {
                                return;
                            }

                            const deletion = deleteDocumentFiles(storagePath, docId);
                            docsWebviewView.webview.postMessage({
                                command: 'docDeleted',
                                payload: { id: docId }
                            });
                            refreshSidebarView();
                            if (deletion.deletedCount > 0) {
                                vscode.window.showInformationMessage(`Deleted local files (.md/.json) for: ${docId}`);
                            } else {
                                const markdownPath = path.join(storagePath, `${docId}.md`);
                                const jsonPath = path.join(storagePath, `${docId}.json`);
                                vscode.window.showWarningMessage(
                                    `Delete may have failed for ${docId}. Manually delete these files if they still exist:\n- ${markdownPath}\n- ${jsonPath}`
                                );
                            }
                        } catch (error) {
                            const markdownPath = path.join(storagePath, `${docId}.md`);
                            const jsonPath = path.join(storagePath, `${docId}.json`);
                            vscode.window.showErrorMessage(
                                `Failed to delete ${docId}: ${error.message}. Please manually delete:\n- ${markdownPath}\n- ${jsonPath}`
                            );
                        }
                    }
                });

                docsWebviewView.onDidChangeVisibility(async () => {
                    if (!docsWebviewView.visible) {
                        return;
                    }

                    refreshSidebarView();
                });

                refreshSidebarView();
            } catch (error) {
                const reason = error && error.message ? error.message : 'unknown sidebar initialization error';
                if (webviewView && webviewView.webview) {
                    webviewView.webview.html = getSidebarErrorHtml(`RepoAsk sidebar failed to load: ${reason}`);
                }
            }
        }
    };

    function setSidebarSyncStatus(message) {
        sidebarSyncStatus = String(message || '');
        if (!docsWebviewView) {
            return;
        }

        docsWebviewView.webview.postMessage({
            command: 'syncStatus',
            payload: sidebarSyncStatus
        });
    }

    function refreshSidebarView() {
        if (!docsWebviewView) {
            return;
        }

        try {
            docsWebviewView.webview.html = getSidebarHtml(docsWebviewView.webview);
        } catch (error) {
            const reason = error && error.message ? error.message : 'unknown render error';
            docsWebviewView.webview.html = getSidebarErrorHtml(`RepoAsk sidebar failed to render: ${reason}`);
        }
    }

    function getSidebarHtml(webview) {
        const htmlPath = vscode.Uri.joinPath(context.extensionUri, 'src', 'sidebar', 'index.html');
        const cssPath = vscode.Uri.joinPath(context.extensionUri, 'src', 'sidebar', 'styles.css');

        const htmlTemplate = fs.readFileSync(htmlPath.fsPath, 'utf8');
        const cssUri = webview.asWebviewUri(cssPath).toString();
        const docs = readAllMetadata(storagePath).sort((a, b) => String(b.last_updated).localeCompare(String(a.last_updated)));

        return htmlTemplate
            .replace('__CSS_URI__', cssUri)
            .replace('__DOCS_DATA__', JSON.stringify(docs))
            .replace('__SYNC_STATUS__', JSON.stringify(sidebarSyncStatus));
    }

    function getSidebarErrorHtml(message) {
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RepoAsk Sidebar Error</title>
</head>
<body>
    <main style="padding: 12px; font-family: sans-serif;">
        <h2 style="margin: 0 0 8px 0;">RepoAsk Sidebar</h2>
        <p style="margin: 0;">${String(message || 'Unknown error')}</p>
    </main>
</body>
</html>`;
    }

    return {
        sidebarProvider,
        refreshSidebarView,
        setSidebarSyncStatus
    };
}

module.exports = {
    createSidebarController
};
