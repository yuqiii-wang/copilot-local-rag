module.exports = function createSearchCommand(deps) {
    const { documentService, readAllMetadata, storagePath } = deps;

    return async function searchDocs(message, docsWebviewView) {
        const query = String(message.query || '').trim();
        const filterType = String(message.type || '').trim();
        let results = query.length > 0
            ? documentService.rankLocalDocuments(query, 50)
            : readAllMetadata(storagePath)
                .sort((a, b) => String(b.last_updated).localeCompare(String(a.last_updated)));

        if (filterType) {
            // If doc has no type, we fallback treating it as 'confluence' due to historical data or just keep original logic
            results = results.filter(doc => (doc.type || 'confluence') === filterType);
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
