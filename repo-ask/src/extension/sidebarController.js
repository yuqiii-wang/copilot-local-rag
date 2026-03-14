const fs = require('fs');
const path = require('path');
const { createOpenDocCommand, createMetadataCommands, createSearchCommand, createPromptsCommand, createDeleteCommand, createResetCommand } = require('./commands');


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

    // Create command instances
    const openDoc = createOpenDocCommand(deps);
    const { generateMetadata, saveMetadata } = createMetadataCommands(deps);
    const searchDocs = createSearchCommand(deps);
    const addToPrompts = createPromptsCommand(deps);
    const deleteDoc = createDeleteCommand(deps);
    const resetToDefaultDocs = createResetCommand(deps);

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
                            if (arg) {
                                vscode.commands.executeCommand('repo-ask.refresh', { type: 'recursive', arg: String(arg).trim() });
                            } else {
                                vscode.commands.executeCommand('repo-ask.refresh', '');
                            }
                        } else if (arg) {
                            vscode.commands.executeCommand('repo-ask.refresh', String(arg).trim());
                        }
                    }

                    if (message?.command === 'openDoc' && message.docId) {
                        await openDoc(message, docsWebviewView);
                    }

                    if (message?.command === 'generateMetadata' && message.docId) {
                        await generateMetadata(message, docsWebviewView, upsertSidebarDocument);
                    }

                    if (message?.command === 'saveMetadata' && message.docId) {
                        await saveMetadata(message, upsertSidebarDocument);
                    }

                    if (message?.command === 'searchDocs') {
                        await searchDocs(message, docsWebviewView);
                    }

                    if (message?.command === 'addToPrompts') {
                        await addToPrompts(message, docsWebviewView);
                    }

                    if (message?.command === 'deleteDoc') {
                        await deleteDoc(message, docsWebviewView, refreshSidebarView);
                    }

                    if (message?.command === 'resetToDefaultDocs') {
                        await resetToDefaultDocs(message, docsWebviewView, refreshSidebarView, context);
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
        const metadataHtmlPath = vscode.Uri.joinPath(context.extensionUri, 'src', 'sidebar', 'metadata.html');
        const docStoreHtmlPath = vscode.Uri.joinPath(context.extensionUri, 'src', 'sidebar', 'docStore.html');
        const metadataJsPath = vscode.Uri.joinPath(context.extensionUri, 'src', 'sidebar', 'metadata.js');
        const docStoreJsPath = vscode.Uri.joinPath(context.extensionUri, 'src', 'sidebar', 'docStore.js');

        const htmlTemplate = fs.readFileSync(htmlPath.fsPath, 'utf8');
        const popupHtml = fs.existsSync(popupPath.fsPath) ? fs.readFileSync(popupPath.fsPath, 'utf8') : '';
        const metadataHtml = fs.existsSync(metadataHtmlPath.fsPath) ? fs.readFileSync(metadataHtmlPath.fsPath, 'utf8') : '';
        const docStoreHtml = fs.existsSync(docStoreHtmlPath.fsPath) ? fs.readFileSync(docStoreHtmlPath.fsPath, 'utf8') : '';

        const cssUri = webview.asWebviewUri(cssPath).toString();
        const metadataJsUri = webview.asWebviewUri(metadataJsPath).toString();
        const docStoreJsUri = webview.asWebviewUri(docStoreJsPath).toString();

        const docs = readAllMetadata(storagePath).sort((a, b) => String(b.last_updated).localeCompare(String(a.last_updated)));

        return htmlTemplate
            .replace('__CSS_URI__', cssUri)
            .replace('__METADATA_JS_URI__', metadataJsUri)
            .replace('__DOC_STORE_JS_URI__', docStoreJsUri)
            .replace('__DOCS_DATA__', JSON.stringify(docs))
            .replace('__SYNC_STATUS__', JSON.stringify(sidebarSyncStatus))
            .replace('__SYNC_ERROR__', JSON.stringify(sidebarSyncError))
            .replace('__SYNC_SUCCESS__', JSON.stringify(sidebarSyncSuccess))
            .replace('__METADATA_HTML__', metadataHtml)
            .replace('__DOC_STORE_HTML__', docStoreHtml)
            .replace('__REFRESH_POPUP__', popupHtml);
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
        upsertSidebarDocument,
        revealDocumentInSidebar
    };
}

module.exports = {
    createSidebarController
};
