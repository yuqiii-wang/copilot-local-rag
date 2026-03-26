module.exports = function createSearchCommand(deps) {
    const { documentService, readAllMetadata, storagePath, vscode } = deps;

    return async function searchDocs(message, docsWebviewView) {
        const query = String(message.query || '').trim();
        const filterType = String(message.type || '').trim();
        // Use maxSearchResults setting as the search buffer; ranking will cap the final result
        const maxResults = Math.max(Number(vscode.workspace.getConfiguration('repoAsk').get('maxSearchResults')) || 5, 1);
        const searchBuffer = Math.max(maxResults * 10, 50);

        let results = query.length > 0
            ? documentService.rankLocalDocuments(query, searchBuffer)
            : readAllMetadata(storagePath)
                .sort((a, b) => String(b.last_updated).localeCompare(String(a.last_updated)));

        if (filterType) {
            // If doc has no type, we fallback treating it as 'confluence' due to historical data or just keep original logic
            results = results.filter(doc => (doc.type || 'confluence') === filterType);
        }

        // For unfiltered list or type-filtered list, still cap at maxResults
        if (!query.length) {
            results = results.slice(0, maxResults);
        }

        docsWebviewView.webview.postMessage({
            command: 'searchResults',
            payload: results.map(doc => ({
                id: doc.id,
                title: doc.title || 'Untitled'
            }))
        });
    };
};
