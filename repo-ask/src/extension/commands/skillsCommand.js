module.exports = function createSkillsCommand(deps) {
    const { vscode, documentService, readAllMetadata, readDocumentContent, storagePath } = deps;

    return async function addToSkills(message, docsWebviewView) {
        const docId = String(message.docId || '').trim();
        if (!docId) {
            vscode.window.showWarningMessage('Select a document first to add it to skills.');
            return;
        }

        const metadata = readAllMetadata(storagePath).find(doc => String(doc.id) === docId);
        if (!metadata) {
            vscode.window.showWarningMessage('Document metadata not found. Run refresh and try again.');
            return;
        }

        if (!metadata.summary || String(metadata.summary).trim().length === 0) {
            docsWebviewView.webview.postMessage({ command: 'addToSkillsError', payload: 'Please populate metadata summary either manually or via AI generation before adding to skill.' });
            return;
        }

        const content = readDocumentContent(storagePath, metadata.id);
        if (!content || String(content).trim().length === 0) {
            vscode.window.showWarningMessage('Local document content is empty. Refresh this doc and try again.');
            return;
        }

        try {
            const createdPath = documentService.writeDocumentSkillFile(metadata, content);
            docsWebviewView.webview.postMessage({ command: 'addToSkillsSuccess', payload: createdPath });
        } catch (error) {
            docsWebviewView.webview.postMessage({ command: 'addToSkillsError', payload: error.message });
        }
    };
};
