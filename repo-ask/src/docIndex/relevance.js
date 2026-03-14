function scoreMetadata(questionTokens, metadata, tokenize) {
    const titleTokens = tokenize(metadata.title || '');
    const summaryTokens = tokenize(metadata.summary || '');
    const keywordTokens = (metadata.keywords || []).flatMap(keyword => tokenize(keyword));
    const extendedKeywordTokens = (metadata.extended_keywords || []).flatMap(keyword => tokenize(keyword));
    const topicTokens = tokenize(metadata.parent_confluence_topic || '');

    const tokenSet = new Set([...titleTokens, ...summaryTokens, ...keywordTokens, ...extendedKeywordTokens, ...topicTokens]);
    let score = 0;

    for (const token of questionTokens) {
        if (titleTokens.includes(token)) {
            score += 4;
        }
        if (keywordTokens.includes(token)) {
            score += 3;
        }
        if (extendedKeywordTokens.includes(token)) {
            score += 2;
        }
        if (summaryTokens.includes(token)) {
            score += 2;
        }
        if (topicTokens.includes(token)) {
            score += 2;
        }
    }

    const union = new Set([...questionTokens, ...tokenSet]);
    const overlapCount = questionTokens.filter(token => tokenSet.has(token)).length;
    const jaccard = union.size > 0 ? overlapCount / union.size : 0;
    score += jaccard * 10;

    return score;
}

function findRelevantDocuments(question, metadataList, tokenize) {
    const questionTokens = tokenize(question);

    const scoredDocs = metadataList.map(metadata => ({
        ...metadata,
        score: scoreMetadata(questionTokens, metadata, tokenize)
    }));

    return scoredDocs
        .filter(doc => doc.score >= 2.5)
        .sort((a, b) => b.score - a.score)
        .slice(0, 5);
}

function buildTokenCounts(tokens) {
    const counts = new Map();
    for (const token of tokens) {
        counts.set(token, (counts.get(token) || 0) + 1);
    }
    return counts;
}

function rankDocumentsByIdf(query, documents, tokenize, options = {}) {
    const limit = Number(options.limit) > 0 ? Number(options.limit) : 20;
    const minScore = typeof options.minScore === 'number' ? options.minScore : 0;
    const queryTokens = [...new Set(tokenize(query || ''))];

    if (queryTokens.length === 0 || !Array.isArray(documents) || documents.length === 0) {
        return [];
    }

    const docTokenCounts = documents.map(doc => {
        const docText = [
            doc.title || '',
            doc.summary || '',
            doc.parent_confluence_topic || '',
            ...(Array.isArray(doc.keywords) ? doc.keywords : []),
            ...(Array.isArray(doc.extended_keywords) ? doc.extended_keywords : []),
            doc.content || ''
        ].join(' ');

        return buildTokenCounts(tokenize(docText));
    });

    const docFrequency = new Map();
    for (const token of queryTokens) {
        let frequency = 0;
        for (const counts of docTokenCounts) {
            if (counts.has(token)) {
                frequency += 1;
            }
        }
        docFrequency.set(token, frequency);
    }

    const totalDocs = documents.length;
    const idf = new Map();
    for (const token of queryTokens) {
        const df = docFrequency.get(token) || 0;
        idf.set(token, Math.log((totalDocs + 1) / (df + 1)) + 1);
    }

    const scored = documents.map((doc, index) => {
        const counts = docTokenCounts[index];
        let score = 0;

        for (const token of queryTokens) {
            const tf = counts.get(token) || 0;
            if (tf > 0) {
                score += tf * (idf.get(token) || 0);
            }
        }

        return {
            ...doc,
            score
        };
    });

    return scored
        .filter(doc => doc.score > minScore)
        .sort((a, b) => b.score - a.score)
        .slice(0, limit);
}

module.exports = {
    findRelevantDocuments,
    rankDocumentsByIdf
};