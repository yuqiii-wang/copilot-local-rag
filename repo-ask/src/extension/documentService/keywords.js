module.exports = function(context) {
  const { vscode, storagePath, indexStoragePath, fetchConfluencePage, fetchAllConfluencePages, fetchJiraIssue, truncate, tokenize, htmlToMarkdown, generateKeywords, generateExtendedKeywords, generateSummary, readAllMetadata, writeDocumentFiles, readDocumentContent, rankDocumentsByIdf, bm25Index, keywordsIndex, rankLocalDocuments, checkLocalDocumentsAgentic, refreshDocument, refreshAllDocuments, refreshJiraIssue, notifyDocumentProcessed, processDocument, processJiraIssue, finalizeBm25KeywordsForDocuments, annotateDocumentByArg, annotateAllDocuments, annotateStoredDocument, generateAnnotationWithLlm, localizeMarkdownImageLinks, normalizeMarkdownLinkTarget, downloadImageAsset, downloadDataUriAsset, resolveAbsoluteImageUrl, isDataUri, determineImageExtension, mimeTypeToExtension,         writeDocumentPromptFile, formatMetadataEntries, getStoredMetadataById, generateStoredMetadataById, updateStoredMetadataById, removeDocumentFromIndicesById, sanitizeFileSegment, getWorkspaceRootPath, getPageHtml, isLikelyHtml, extractHtmlTagData, resolveSourceUrl } = context;

function getKeywordConfig() {
  const initKeywordNum = vscode.workspace.getConfiguration('repoAsk').get('initKeywordNum') || 40;
  return {
    DEFAULT_KEYWORD_LIMIT: initKeywordNum,
    TOKENIZATION_KEYWORD_LIMIT: Math.floor(initKeywordNum / 2),
    BM25_KEYWORD_LIMIT: initKeywordNum - Math.floor(initKeywordNum / 2)
  };
}

function buildKeywordOnlyIndexText(metadata) {
  if (!metadata || typeof metadata !== 'object') {
    return '';
  }
  const {
    DEFAULT_KEYWORD_LIMIT
  } = getKeywordConfig();
  const keywords = cleanKeywords(metadata.keywords, getKeywordConfig().DEFAULT_KEYWORD_LIMIT * 4);
  return keywords.join(' ');
}

function rebuildKeywordsIndexFromMetadata(metadataList = []) {
  const documents = (Array.isArray(metadataList) ? metadataList : []).map(item => ({
    id: item?.id,
    text: buildKeywordOnlyIndexText(item)
  })).filter(item => String(item.id || '').trim().length > 0);
  keywordsIndex.rebuildDocuments(documents);
}

function normalizeKeywordsInput(values) {
  if (Array.isArray(values)) {
    return values;
  }
  if (typeof values === 'string') {
    return values.split(',');
  }
  return [];
}

function cleanKeywords(values, limit = getKeywordConfig().DEFAULT_KEYWORD_LIMIT) {
  const keywordValues = normalizeKeywordsInput(values);
  if (keywordValues.length === 0) {
    return [];
  }
  const safeLimit = Number.isFinite(limit) && limit > 0 ? Math.floor(limit) : getKeywordConfig().DEFAULT_KEYWORD_LIMIT;
  return [...new Set(keywordValues.map(value => String(value || '').trim()).filter(value => value.length > 2))].slice(0, safeLimit);
}

function normalizeMetadataKeywordFields(metadata = {}) {
  const base = metadata && typeof metadata === 'object' ? metadata : {};
  const keywords = cleanKeywords(base.keywords, getKeywordConfig().DEFAULT_KEYWORD_LIMIT);
  return {
    ...base,
    keywords,
    extended_keywords: cleanKeywords(generateExtendedKeywords(keywords), 80)
  };
}

function mergeKeywordsPreservingSignals({
  structuralKeywords = [],
  modelKeywords = [],
  lexicalKeywords = [],
  limit = getKeywordConfig().DEFAULT_KEYWORD_LIMIT
} = {}) {
  const safeLimit = Number.isFinite(limit) && limit > 0 ? Math.floor(limit) : getKeywordConfig().DEFAULT_KEYWORD_LIMIT;
  const structural = cleanKeywords(structuralKeywords, safeLimit * 2);
  const model = cleanKeywords(modelKeywords, safeLimit * 2);
  const lexical = cleanKeywords(lexicalKeywords, safeLimit * 2);
  const merged = [];
  let index = 0;

  // Interleave structural and BM25 keywords so both sources remain visible.
  while (merged.length < safeLimit && (index < structural.length || index < model.length)) {
    if (index < structural.length && !merged.includes(structural[index])) {
      merged.push(structural[index]);
    }
    if (merged.length >= safeLimit) {
      break;
    }
    if (index < model.length && !merged.includes(model[index])) {
      merged.push(model[index]);
    }
    index += 1;
  }
  for (const keyword of lexical) {
    if (merged.length >= safeLimit) {
      break;
    }
    if (!merged.includes(keyword)) {
      merged.push(keyword);
    }
  }
  return merged;
}

function appendKeywordsToExisting(existingKeywords = [], addedKeywords = [], limit = getKeywordConfig().DEFAULT_KEYWORD_LIMIT) {
  const safeLimit = Number.isFinite(limit) && limit > 0 ? Math.floor(limit) : getKeywordConfig().DEFAULT_KEYWORD_LIMIT;
  const existing = cleanKeywords(existingKeywords, safeLimit * 2);
  const additions = cleanKeywords(addedKeywords, safeLimit * 2);
  const merged = [...existing];
  for (const keyword of additions) {
    if (merged.length >= safeLimit) {
      break;
    }
    if (!merged.includes(keyword)) {
      merged.push(keyword);
    }
  }
  return merged.slice(0, safeLimit);
}

  return {
    getKeywordConfig,
    buildKeywordOnlyIndexText,
    rebuildKeywordsIndexFromMetadata,
    normalizeKeywordsInput,
    cleanKeywords,
    normalizeMetadataKeywordFields,
    mergeKeywordsPreservingSignals,
    appendKeywordsToExisting
  };
};
