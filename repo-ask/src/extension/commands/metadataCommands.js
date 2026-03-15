module.exports = function createMetadataCommands(deps) {
    const { vscode, documentService } = deps;

    async function generateMetadata(message, docsWebviewView, upsertSidebarDocument) {
        if (!message?.docId) {
            return;
        }

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

    async function saveMetadata(message, upsertSidebarDocument) {
        if (!message?.docId) {
            return;
        }

        try {
            const updatedMetadata = documentService.updateStoredMetadataById(String(message.docId), {
                type: message.type,
                summary: message.summary,
                keywords: message.keywords,
                tags: message.tags
            });
            upsertSidebarDocument(updatedMetadata);
            vscode.window.showInformationMessage(`Saved metadata for: ${updatedMetadata.title || updatedMetadata.id}`);
        } catch (error) {
            vscode.window.showErrorMessage(`Failed to save metadata: ${error.message}`);
        }
    }

    return {
        generateMetadata,
        saveMetadata
    };
};
