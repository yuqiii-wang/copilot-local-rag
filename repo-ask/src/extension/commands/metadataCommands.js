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

            // Also generate and save knowledge graph using the same logic as the feedback form
            let finalMetadata = updatedMetadata;
            try {
                const mermaid = await documentService.buildKnowledgeGraph({
                    primaryDocId: updatedMetadata.confluencePageId || updatedMetadata.jiraId || docId,
                    confluenceLink: updatedMetadata.url || null,
                    secondaryUrls: [],
                    referenceQueries: [],
                    existingKnowledgeGraph: updatedMetadata.knowledgeGraph || undefined,
                    conversationSummary: String(updatedMetadata.feedback || updatedMetadata.summary || '').trim() || undefined
                });
                if (mermaid) {
                    await documentService.saveKnowledgeGraph(docId, mermaid);
                    finalMetadata = { ...updatedMetadata, knowledgeGraph: mermaid };
                }
            } catch (kgErr) {
                console.error('[generateMetadata] KG generation error:', kgErr);
            }

            upsertSidebarDocument(finalMetadata);
            docsWebviewView.webview.postMessage({
                command: 'metadataUpdated',
                payload: {
                    id: finalMetadata.id,
                    metadata: finalMetadata
                }
            });
            vscode.window.showInformationMessage(`Generated summary and keywords for: ${finalMetadata.title || finalMetadata.id}`);
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
