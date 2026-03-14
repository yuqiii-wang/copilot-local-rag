module.exports = function createPromptsCommand(deps) {
    const { vscode, documentService, readAllMetadata, readDocumentContent, storagePath } = deps;

    return async function addToPrompts(message, docsWebviewView) {
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
    };
};
