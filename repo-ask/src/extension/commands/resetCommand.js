module.exports = function createResetCommand(deps) {
    const { vscode, documentService, deleteDocumentFiles, storagePath, readAllMetadata } = deps;

    return async function resetToDefaultDocs(message, docsWebviewView, refreshSidebarView, context) {
        try {
            const confirmation = await vscode.window.showWarningMessage(
                'Reset to default documents? This will delete all existing local documents and reload only the default docs.',
                { modal: true },
                'Reset'
            );
            if (confirmation !== 'Reset') {
                return;
            }

            // Remove all existing documents
            const allDocs = readAllMetadata(storagePath);
            for (const doc of allDocs) {
                deleteDocumentFiles(storagePath, doc.id);
                if (typeof documentService.removeDocumentFromIndicesById === 'function') {
                    documentService.removeDocumentFromIndicesById(doc.id);
                }
            }

            // Sync default docs
            documentService.syncDefaultDocs(context.extensionPath);

            refreshSidebarView();
            vscode.window.showInformationMessage('Reset to default documents completed.');
        } catch (error) {
            vscode.window.showErrorMessage(`Failed to reset to default docs: ${error.message}`);
        }
    };
};
