module.exports = function(context) {
  const { vscode, storagePath, indexStoragePath, fetchConfluencePage, fetchAllConfluencePages, fetchJiraIssue, truncate, tokenize, htmlToMarkdown, generateKeywords, generateExtendedKeywords, generateSummary, readAllMetadata, writeDocumentFiles, readDocumentContent, rankDocumentsByIdf, bm25Index, keywordsIndex,   refreshDocument, refreshAllDocuments, refreshJiraIssue, notifyDocumentProcessed, processDocument, processJiraIssue, finalizeBm25KeywordsForDocuments, annotateDocumentByArg, annotateAllDocuments, annotateStoredDocument, generateAnnotationWithLlm, localizeMarkdownImageLinks, normalizeMarkdownLinkTarget, downloadImageAsset, downloadDataUriAsset, resolveAbsoluteImageUrl, isDataUri, determineImageExtension, mimeTypeToExtension, getKeywordConfig, buildKeywordOnlyIndexText, rebuildKeywordsIndexFromMetadata, normalizeKeywordsInput, cleanKeywords, normalizeMetadataKeywordFields, mergeKeywordsPreservingSignals, appendKeywordsToExisting, writeDocumentPromptFile, formatMetadataEntries, getStoredMetadataById, generateStoredMetadataById, updateStoredMetadataById, removeDocumentFromIndicesById, sanitizeFileSegment, getWorkspaceRootPath, getPageHtml, isLikelyHtml, extractHtmlTagData, resolveSourceUrl } = context;

function rankLocalDocuments(query, limit = 20) {
  const metadataList = readAllMetadata(storagePath).map(normalizeMetadataKeywordFields);
  if (metadataList.length === 0) {
    return [];
  }

  // Rank/search should use the latest metadata keywords as the keyword index corpus.
  rebuildKeywordsIndexFromMetadata(metadataList);
  const metadataById = Object.fromEntries(metadataList.map(item => [String(item.id), item]));
  const rankedByKeywords = keywordsIndex.rankDocuments(query, metadataById, {
    limit: 1000
  });
  const rankedByContent = bm25Index.rankDocuments(query, metadataById, {
    limit: 1000
  });
  const combinedScores = new Map();

  // Add explicit hits for keywords, id, title, tags, and type
  const lowerQuery = (query || '').toLowerCase().trim();
  const queryTerms = lowerQuery.split(/\s+/).filter(t => t.length > 0);
  for (const metadata of metadataList) {
    let exactHitScore = 0;
    const mId = String(metadata.id || '').toLowerCase();
    const mTitle = String(metadata.title || '').toLowerCase();
    const mKeywords = Array.isArray(metadata.keywords) ? metadata.keywords.map(k => String(k).toLowerCase()) : [];
    const mTags = Array.isArray(metadata.tags) ? metadata.tags.map(t => String(t).toLowerCase()) : [];
    const mType = String(metadata.type || '').toLowerCase();

    if (lowerQuery) {
        if (mId.includes(lowerQuery)) exactHitScore += 15; // Higher weight for exact ID match
        if (mTitle.includes(lowerQuery)) exactHitScore += 10;
        if (mKeywords.some(k => k.includes(lowerQuery))) exactHitScore += 8;
        if (mTags.some(t => t.includes(lowerQuery))) exactHitScore += 6;
        if (mType.includes(lowerQuery)) exactHitScore += 4;
    } else if (queryTerms.length > 0) {
        for (const term of queryTerms) {
            if (mId.includes(term)) exactHitScore += 3;
            if (mTitle.includes(term)) exactHitScore += 2;
            if (mKeywords.some(k => k.includes(term))) exactHitScore += 1.5;
            if (mTags.some(t => t.includes(term))) exactHitScore += 1.2;
            if (mType.includes(term)) exactHitScore += 1;
        }
    }
    
    if (exactHitScore > 0) {
        combinedScores.set(String(metadata.id), {
            ...metadata,
            score: exactHitScore
        });
    }
  }

  for (const doc of rankedByKeywords) {
    // Give keywords a slightly higher weight since they are explicit signals
    const id = String(doc.id);
    if (combinedScores.has(id)) {
      combinedScores.get(id).score += doc.score * 1.5;
    } else {
      combinedScores.set(id, {
        ...doc,
        score: doc.score * 1.5
      });
    }
  }
  for (const doc of rankedByContent) {
    const id = String(doc.id);
    if (combinedScores.has(id)) {
      combinedScores.get(id).score += doc.score;
    } else {
      combinedScores.set(id, {
        ...doc,
        score: doc.score
      });
    }
  }
  let combinedRanked = Array.from(combinedScores.values()).sort((a, b) => b.score - a.score).slice(0, limit * 2);
  
  // Group related documents and ensure logical flow
  combinedRanked = groupRelatedDocuments(combinedRanked).slice(0, limit);
  
  if (combinedRanked.length > 0) {
    return combinedRanked;
  }
  const fallbackCorpus = metadataList.map(metadata => ({
    ...metadata,
    content: readDocumentContent(storagePath, metadata.id) || ''
  }));
  return rankDocumentsByIdf(query, fallbackCorpus, tokenize, {
    limit,
    minScore: 0.01
  });
}

function groupRelatedDocuments(documents) {
  if (!Array.isArray(documents) || documents.length <= 1) {
    return documents;
  }
  
  const grouped = [];
  const used = new Set();
  
  // Start with the highest ranked document
  const topDoc = documents[0];
  grouped.push(topDoc);
  used.add(String(topDoc.id));
  
  // Find related documents based on shared keywords, tags, or parent topics
  for (let i = 1; i < documents.length; i++) {
    const currentDoc = documents[i];
    const currentId = String(currentDoc.id);
    
    if (used.has(currentId)) {
      continue;
    }
    
    // Check if current document is related to any already grouped document
    const isRelated = grouped.some(groupedDoc => {
      // Check shared keywords
      const sharedKeywords = (Array.isArray(currentDoc.keywords) ? currentDoc.keywords : [])
        .filter(k => (Array.isArray(groupedDoc.keywords) ? groupedDoc.keywords : []).includes(k));
      
      // Check shared tags
      const sharedTags = (Array.isArray(currentDoc.tags) ? currentDoc.tags : [])
        .filter(t => (Array.isArray(groupedDoc.tags) ? groupedDoc.tags : []).includes(t));
      
      // Check same parent topic
      const sameParent = currentDoc.parent_confluence_topic && 
                        groupedDoc.parent_confluence_topic &&
                        currentDoc.parent_confluence_topic === groupedDoc.parent_confluence_topic;
      
      // Check if documents are part of the same logical flow (e.g., same project or process)
      const sameContext = currentDoc.title && groupedDoc.title &&
                         (currentDoc.title.includes(groupedDoc.title) || 
                          groupedDoc.title.includes(currentDoc.title));
      
      return sharedKeywords.length > 0 || sharedTags.length > 0 || sameParent || sameContext;
    });
    
    if (isRelated) {
      grouped.push(currentDoc);
      used.add(currentId);
    }
  }
  
  // If we only have one document, add the next highest ranked documents that might be related
  if (grouped.length === 1 && documents.length > 1) {
    for (let i = 1; i < documents.length; i++) {
      const currentDoc = documents[i];
      const currentId = String(currentDoc.id);
      
      if (!used.has(currentId)) {
        grouped.push(currentDoc);
        used.add(currentId);
        if (grouped.length >= 5) break; // Limit to 5 related documents
      }
    }
  }
  
  return grouped;
}

function checkLocalDocumentsAgentic(query, options = {}) {
  const normalizedQuery = String(query || '').trim();
  const rawLimit = Number(options?.limit);
  const limit = Number.isFinite(rawLimit) && rawLimit > 0 ? Math.min(Math.floor(rawLimit), 50) : 5;
  const rawMetadataCandidateLimit = Number(options?.metadataCandidateLimit);
  const metadataCandidateLimit = Number.isFinite(rawMetadataCandidateLimit) && rawMetadataCandidateLimit > 0 ? Math.min(Math.floor(rawMetadataCandidateLimit), 1000) : Math.max(40, limit * 4);
  if (!normalizedQuery) {
    return {
      query: normalizedQuery,
      metadataScanned: 0,
      metadataCandidates: 0,
      contentLoaded: 0,
      usedMetadataFallback: false,
      references: []
    };
  }
  const metadataList = readAllMetadata(storagePath).map(normalizeMetadataKeywordFields);
  if (metadataList.length === 0) {
    return {
      query: normalizedQuery,
      metadataScanned: 0,
      metadataCandidates: 0,
      contentLoaded: 0,
      usedMetadataFallback: false,
      references: []
    };
  }
  const metadataCorpus = metadataList.map(doc => ({
    ...doc,
    content: `${String(doc.title || '')} ${(Array.isArray(doc.keywords) ? doc.keywords.join(' ') : '')} ${(Array.isArray(doc.tags) ? doc.tags.join(' ') : '')} ${String(doc.type || '')} ${String(doc.id || '')}`
  }));
  const rankedMetadata = rankDocumentsByIdf(normalizedQuery, metadataCorpus, tokenize, {
    limit: metadataList.length,
    minScore: 0
  });
  const positiveMetadata = rankedMetadata.filter(doc => Number(doc.score) > 0);
  const metadataCandidates = (positiveMetadata.length > 0 ? positiveMetadata : metadataList).slice(0, Math.min(metadataCandidateLimit, metadataList.length));
  const contentById = new Map();
  const contentCorpus = metadataCandidates.map(doc => {
    const content = readDocumentContent(storagePath, doc.id) || '';
    contentById.set(String(doc.id || ''), content);
    return {
      ...doc,
      content
    };
  });
  const rankedByContent = rankDocumentsByIdf(normalizedQuery, contentCorpus, tokenize, {
    limit,
    minScore: 0
  });
  let finalResults = rankedByContent.length > 0 ? rankedByContent : metadataCandidates.slice(0, limit * 2).map(doc => ({
    ...doc,
    score: Number(doc.score || 0)
  }));
  
  // Group related documents to ensure logical flow
  finalResults = groupRelatedDocuments(finalResults).slice(0, limit);
  const references = finalResults.map(doc => {
    const docId = String(doc.id || '');
    return {
      id: doc.id,
      title: doc.title || 'Untitled',
      author: doc.author || 'Unknown',
      last_updated: doc.last_updated || '',
      parent_confluence_topic: doc.parent_confluence_topic || '',
      summary: truncate(doc.summary || 'No summary available', 220),
      score: Number.isFinite(Number(doc.score)) ? Number(doc.score) : 0,
      reference: truncate(contentById.get(docId) || '', 500)
    };
  });
  const contentLoaded = contentCorpus.filter(doc => String(doc.content || '').trim().length > 0).length;
  return {
    query: normalizedQuery,
    metadataScanned: metadataList.length,
    metadataCandidates: metadataCandidates.length,
    contentLoaded,
    usedMetadataFallback: positiveMetadata.length === 0,
    references
  };
}

async function optimizeQueryAndRank(query, options = {}) {
  const normalizedQuery = String(query || '').trim();
  const rawLimit = Number(options?.limit);
  const limit = Number.isFinite(rawLimit) && rawLimit > 0 ? Math.min(Math.floor(rawLimit), 50) : 5;
  
  if (!normalizedQuery) {
    return {
      query: normalizedQuery,
      optimizedQuery: normalizedQuery,
      iterations: 0,
      confidence: 0,
      documents: []
    };
  }
  
  let currentQuery = normalizedQuery;
  let bestDocuments = [];
  let bestConfidence = 0;
  let iterations = 0;
  const maxIterations = 2;
  
  while (iterations < maxIterations) {
    // Rank documents with current query
    let rankedDocuments = rankLocalDocuments(currentQuery, limit * 2);
    
    // Group related documents to ensure logical flow
    rankedDocuments = groupRelatedDocuments(rankedDocuments).slice(0, limit);
    
    // Calculate confidence score
    const confidence = calculateConfidence(rankedDocuments, currentQuery);
    
    // Update best results if current confidence is higher
    if (confidence > bestConfidence) {
      bestConfidence = confidence;
      bestDocuments = rankedDocuments;
    }
    
    // Check if we've reached high confidence
    if (confidence >= 0.7) {
      break;
    }
    
    // Optimize the query for the next iteration
    const optimizedResult = optimizeQuery(currentQuery);
    currentQuery = optimizedResult.refinedQuery;
    
    iterations++;
  }
  
  return {
    query: normalizedQuery,
    optimizedQuery: currentQuery,
    iterations,
    confidence: bestConfidence,
    documents: bestDocuments
  };
}

function optimizeQuery(query) {
  // Simple query optimization logic
  // This could be enhanced with the LLM-based optimizeQueryTool
  let refined = query.trim();
  
  // Add context for platform-specific terms
  if (refined.includes('jira')) {
    refined = `Jira issue: ${refined}`;
  }
  if (refined.includes('confluence')) {
    refined = `Confluence page: ${refined}`;
  }
  
  // Extract keywords
  const keywords = extractKeywords(refined);
  
  return {
    refinedQuery: refined,
    keywords
  };
}

function extractKeywords(query) {
  const words = query.toLowerCase().split(/\s+/);
  const stopWords = new Set(['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by']);
  
  return words
    .filter(word => !stopWords.has(word) && word.length > 2)
    .filter((value, index, self) => self.indexOf(value) === index);
}

function calculateConfidence(documents, query) {
  if (!documents || documents.length === 0) {
    return 0;
  }
  
  // Calculate average score
  const averageScore = documents.reduce((sum, doc) => sum + (doc.score || 0), 0) / documents.length;
  
  // Normalize score to 0-1 range
  const normalizedScore = Math.min(averageScore / 10, 1);
  
  // Check if top document has a significantly higher score
  const topScore = documents[0].score || 0;
  const scoreRatio = topScore / averageScore;
  
  // Boost confidence if top document is much better than others
  const confidence = normalizedScore * (scoreRatio > 1.5 ? 1.2 : 1);
  
  return Math.min(confidence, 1);
}

  return {
    rankLocalDocuments,
    checkLocalDocumentsAgentic,
    optimizeQueryAndRank
  };
};
