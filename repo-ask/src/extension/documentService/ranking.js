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

  // Add explicit hits for keywords, id, and title
  const lowerQuery = (query || '').toLowerCase().trim();
  const queryTerms = lowerQuery.split(/\s+/).filter(t => t.length > 0);
  for (const metadata of metadataList) {
    let exactHitScore = 0;
    const mId = String(metadata.id || '').toLowerCase();
    const mTitle = String(metadata.title || '').toLowerCase();
    const mKeywords = Array.isArray(metadata.keywords) ? metadata.keywords.map(k => String(k).toLowerCase()) : [];

    if (lowerQuery && (mId.includes(lowerQuery) || mTitle.includes(lowerQuery) || mKeywords.some(k => k.includes(lowerQuery)))) {
        exactHitScore += 10;
    } else if (queryTerms.length > 0) {
        for (const term of queryTerms) {
            if (mId.includes(term) || mTitle.includes(term) || mKeywords.some(k => k.includes(term))) {
                exactHitScore += 2;
            }
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
  const combinedRanked = Array.from(combinedScores.values()).sort((a, b) => b.score - a.score).slice(0, limit);
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
    content: ''
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
  const finalResults = rankedByContent.length > 0 ? rankedByContent : metadataCandidates.slice(0, limit).map(doc => ({
    ...doc,
    score: Number(doc.score || 0)
  }));
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

  return {
    rankLocalDocuments,
    checkLocalDocumentsAgentic
  };
};
