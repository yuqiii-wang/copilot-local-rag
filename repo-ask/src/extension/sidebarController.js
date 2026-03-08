const fs = require('fs');
const path = require('path');
const MarkdownIt = require('markdown-it');

const markdownRenderer = new MarkdownIt({
    html: false,
    linkify: true,
    breaks: true
});

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
    let sidebarSyncError = '';
    let sidebarSyncSuccess = '';

    const sidebarProvider = {
        resolveWebviewView: async (webviewView) => {
            try {
                docsWebviewView = webviewView;
                docsWebviewView.webview.options = {
                    enableScripts: true,
                    localResourceRoots: [
                        vscode.Uri.joinPath(context.extensionUri, 'src', 'sidebar'),
                        vscode.Uri.file(storagePath)
                    ]
                };

                docsWebviewView.webview.onDidReceiveMessage(async (message) => {
                    if (message?.command === 'clearSyncError') {
                        sidebarSyncError = '';
                    }

                    if (message?.command === 'clearSyncSuccess') {
                        sidebarSyncSuccess = '';
                    }

                    if (message?.command === 'refreshDocs') {
                        const { isAll, arg } = message;
                        if (isAll) {
                            vscode.commands.executeCommand('repo-ask.refresh', '');
                        } else if (arg) {
                            vscode.commands.executeCommand('repo-ask.refresh', String(arg).trim());
                        }
                    }

                    if (message?.command === 'openDoc' && message.docId) {
                        const metadata = readAllMetadata(storagePath).find(doc => String(doc.id) === String(message.docId));
                        const rawContent = metadata
                            ? (readDocumentContent(storagePath, metadata.id) || 'No local markdown content found.')
                            : 'No local markdown content found.';
                        const content = metadata
                            ? rewriteMarkdownImageLinksForWebview(rawContent, metadata.id, docsWebviewView.webview)
                            : rawContent;
                        const contentHtml = renderMarkdownForWebview(content);

                        docsWebviewView.webview.postMessage({
                            command: 'docDetails',
                            payload: {
                                id: message.docId,
                                content,
                                contentHtml,
                                metadata: metadata || null
                            }
                        });
                    }

                    if (message?.command === 'generateMetadata' && message.docId) {
                        const docId = String(message.docId);
                        docsWebviewView.webview.postMessage({
                            command: 'metadataGenerationState',
                            payload: {
                                docId,
                                isGenerating: true
                            }
                        });
                        try {
                            const updatedMetadata = await documentService.generateStoredMetadataById(docId);
                            upsertSidebarDocument(updatedMetadata);
                            docsWebviewView.webview.postMessage({
                                command: 'metadataUpdated',
                                payload: {
                                    id: updatedMetadata.id,
                                    metadata: updatedMetadata
                                }
                            });
                            vscode.window.showInformationMessage(`Generated summary and keywords for: ${updatedMetadata.title || updatedMetadata.id}`);
                        } catch (error) {
                            vscode.window.showErrorMessage(`Failed to generate metadata: ${error.message}`);
                        } finally {
                            docsWebviewView.webview.postMessage({
                                command: 'metadataGenerationState',
                                payload: {
                                    docId,
                                    isGenerating: false
                                }
                            });
                        }
                    }

                    if (message?.command === 'saveMetadata' && message.docId) {
                        try {
                            const updatedMetadata = documentService.updateStoredMetadataById(String(message.docId), {
                                summary: message.summary,
                                keywords: message.keywords
                            });
                            upsertSidebarDocument(updatedMetadata);
                            docsWebviewView.webview.postMessage({
                                command: 'metadataUpdated',
                                payload: {
                                    id: updatedMetadata.id,
                                    metadata: updatedMetadata
                                }
                            });
                            vscode.window.showInformationMessage(`Saved metadata for: ${updatedMetadata.title || updatedMetadata.id}`);
                        } catch (error) {
                            vscode.window.showErrorMessage(`Failed to save metadata: ${error.message}`);
                        }
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
                            docsWebviewView.webview.postMessage({ command: 'addToPromptsSuccess', payload: createdPath });
                        } catch (error) {
                            docsWebviewView.webview.postMessage({ command: 'addToPromptsError', payload: error.message });
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
                            if (typeof documentService.removeDocumentFromIndicesById === 'function') {
                                documentService.removeDocumentFromIndicesById(docId);
                            }
                            docsWebviewView.webview.postMessage({
                                command: 'docDeleted',
                                payload: { id: docId }
                            });
                            refreshSidebarView();
                            if (deletion.deletedCount > 0) {
                                vscode.window.showInformationMessage(`Deleted local files (.md/.json) for: ${docId}`);
                            } else {
                                const docFolderPath = path.join(storagePath, String(docId));
                                vscode.window.showWarningMessage(
                                    `Delete may have failed for ${docId}. Manually delete this folder if it still exists:\n- ${docFolderPath}`
                                );
                            }
                        } catch (error) {
                            const docFolderPath = path.join(storagePath, String(docId));
                            vscode.window.showErrorMessage(
                                `Failed to delete ${docId}: ${error.message}. Please manually delete:\n- ${docFolderPath}`
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

    function setSidebarSyncError(message) {
        sidebarSyncError = String(message || '');
        if (!docsWebviewView) {
            return;
        }

        docsWebviewView.webview.postMessage({
            command: 'syncError',
            payload: sidebarSyncError
        });
    }

    function setSidebarSyncSuccess(message) {
        sidebarSyncSuccess = String(message || '');
        if (!docsWebviewView) {
            return;
        }

        docsWebviewView.webview.postMessage({
            command: 'syncSuccess',
            payload: sidebarSyncSuccess
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

    function upsertSidebarDocument(metadata) {
        if (!docsWebviewView || !metadata || !metadata.id) {
            return;
        }

        docsWebviewView.webview.postMessage({
            command: 'docUpserted',
            payload: {
                id: metadata.id,
                title: metadata.title || 'Untitled',
                last_updated: metadata.last_updated || ''
            }
        });
    }

    function revealDocumentInSidebar(docId) {
        const id = String(docId || '').trim();
        if (!id || !docsWebviewView) {
            return false;
        }

        const metadata = readAllMetadata(storagePath).find(doc => String(doc.id) === id);
        if (!metadata) {
            return false;
        }

        const rawContent = readDocumentContent(storagePath, metadata.id) || 'No local markdown content found.';
        const content = rewriteMarkdownImageLinksForWebview(rawContent, metadata.id, docsWebviewView.webview);
        const contentHtml = renderMarkdownForWebview(content);

        if (typeof docsWebviewView.show === 'function') {
            docsWebviewView.show(true);
        }

        docsWebviewView.webview.postMessage({
            command: 'selectDoc',
            payload: {
                id,
                content,
                contentHtml,
                metadata
            }
        });

        return true;
    }

    function getSidebarHtml(webview) {
        const htmlPath = vscode.Uri.joinPath(context.extensionUri, 'src', 'sidebar', 'index.html');
        const cssPath = vscode.Uri.joinPath(context.extensionUri, 'src', 'sidebar', 'styles.css');
        const popupPath = vscode.Uri.joinPath(context.extensionUri, 'src', 'sidebar', 'refreshPopup.html');

        const htmlTemplate = fs.readFileSync(htmlPath.fsPath, 'utf8');
        const popupHtml = fs.existsSync(popupPath.fsPath) ? fs.readFileSync(popupPath.fsPath, 'utf8') : '';
        const cssUri = webview.asWebviewUri(cssPath).toString();
        const docs = readAllMetadata(storagePath).sort((a, b) => String(b.last_updated).localeCompare(String(a.last_updated)));

        return htmlTemplate
            .replace('__CSS_URI__', cssUri)
            .replace('__DOCS_DATA__', JSON.stringify(docs))
            .replace('__SYNC_STATUS__', JSON.stringify(sidebarSyncStatus))
            .replace('__SYNC_ERROR__', JSON.stringify(sidebarSyncError))
            .replace('__SYNC_SUCCESS__', JSON.stringify(sidebarSyncSuccess))
            .replace('__REFRESH_POPUP__', popupHtml);
    }

    function rewriteMarkdownImageLinksForWebview(markdownContent, docId, webview) {
        const markdown = String(markdownContent || '');
        const imagePattern = /!\[([^\]]*)\]\(([^)]+)\)/g;

        return markdown.replace(imagePattern, (fullMatch, alt, srcRaw) => {
            const source = normalizeMarkdownLinkTarget(srcRaw);
            if (!source || /^https?:\/\//i.test(source) || /^data:image\//i.test(source)) {
                return fullMatch;
            }

            const filePath = path.isAbsolute(source)
                ? source
                : path.join(storagePath, String(docId), source.replace(/\//g, path.sep));

            if (!fs.existsSync(filePath)) {
                return fullMatch;
            }

            const webviewUri = webview.asWebviewUri(vscode.Uri.file(filePath)).toString();
            return `![${String(alt || '').trim()}](${webviewUri})`;
        });
    }

    function normalizeMarkdownLinkTarget(rawValue) {
        const value = String(rawValue || '').trim();
        if (!value) {
            return '';
        }

        if (value.startsWith('<') && value.endsWith('>')) {
            return value.slice(1, -1).trim();
        }

        return value;
    }

    function renderMarkdownForWebview(markdownContent) {
        return markdownRenderer.render(String(markdownContent || ''));
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
        setSidebarSyncStatus,
        setSidebarSyncError,
        setSidebarSyncSuccess,
        upsertSidebarDocument,
        revealDocumentInSidebar
    };
}

module.exports = {
    createSidebarController
};
