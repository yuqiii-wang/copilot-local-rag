module.exports = function createDeleteCommand(deps) {
    const { vscode, documentService, deleteDocumentFiles, storagePath, readAllMetadata, path } = deps;

    return async function deleteDoc(message, docsWebviewView, refreshSidebarView) {
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
    };
};
