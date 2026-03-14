const { toToolResult } = require('./utils');

module.exports = function registerOptimizeQueryTool(deps) {
    const { vscode, documentService, readAllMetadata } = deps;
    return vscode.lm.registerTool('repoask_optimize_query', {
            async invoke(options) {
                const query = String(options?.input?.query || '').trim();

                if (!query) {
                    return toToolResult('Missing required `query` input for optimize query tool.', { results: [] });
                }

                // 1. Analyze and refine user queries to improve clarity and relevance
                const refinedQuery = refineQuery(query);

                // 2. Extract potential keywords from the polished query, with specific attention to platform-specific terms
                const extractedKeywords = extractKeywords(refinedQuery);

                // 3. Match extracted keywords against existing tags in the document metadata
                const metadataList = readAllMetadata(deps.storagePath);
                const matchedTags = matchKeywordsToTags(extractedKeywords, metadataList);

                // 4. Implement keyword matching algorithms to identify relevant documents based on the processed query
                const relevantDocuments = documentService.rankLocalDocuments(refinedQuery, 10);

                return toToolResult(
                    `Optimized query: ${refinedQuery}\nExtracted keywords: ${extractedKeywords.join(', ')}\nMatched tags: ${matchedTags.join(', ')}\nFound ${relevantDocuments.length} relevant documents`,
                    {
                        refinedQuery,
                        extractedKeywords,
                        matchedTags,
                        relevantDocuments
                    }
                );
            }
        });
};

function refineQuery(query) {
    // Simple query refinement logic
    // This could be enhanced with more sophisticated NLP techniques
    let refined = query.trim();
    
    // Add context for platform-specific terms
    if (refined.includes('jira')) {
        refined = `Jira issue: ${refined}`;
    }
    if (refined.includes('confluence')) {
        refined = `Confluence page: ${refined}`;
    }
    
    return refined;
}

function extractKeywords(query) {
    // Extract keywords from the query
    // This could be enhanced with more sophisticated keyword extraction
    const words = query.toLowerCase().split(/\s+/);
    const stopWords = new Set(['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by']);
    
    // Extract platform-specific terms and other keywords
    const keywords = words
        .filter(word => !stopWords.has(word) && word.length > 2)
        .concat(extractPlatformSpecificTerms(query));
    
    // Remove duplicates
    return [...new Set(keywords)];
}

function extractPlatformSpecificTerms(query) {
    // Extract platform-specific terms like "jira" and "confluence"
    const platformTerms = [];
    if (query.toLowerCase().includes('jira')) {
        platformTerms.push('jira');
    }
    if (query.toLowerCase().includes('confluence')) {
        platformTerms.push('confluence');
    }
    return platformTerms;
}

function matchKeywordsToTags(keywords, metadataList) {
    // Match extracted keywords against existing tags in document metadata
    const allTags = new Set();
    
    metadataList.forEach(metadata => {
        if (metadata.tags && Array.isArray(metadata.tags)) {
            metadata.tags.forEach(tag => {
                allTags.add(tag.toLowerCase());
            });
        }
    });
    
    // Find matching tags
    return keywords.filter(keyword => allTags.has(keyword.toLowerCase()));
}