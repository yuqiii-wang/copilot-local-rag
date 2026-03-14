module.exports = function(context) {
  const { vscode, storagePath, indexStoragePath, fetchConfluencePage, fetchAllConfluencePages, fetchJiraIssue, truncate, tokenize, htmlToMarkdown, generateKeywords, generateExtendedKeywords, generateSummary, readAllMetadata, writeDocumentFiles, readDocumentContent, rankDocumentsByIdf, bm25Index, keywordsIndex, rankLocalDocuments, checkLocalDocumentsAgentic, refreshDocument, refreshAllDocuments, refreshJiraIssue, notifyDocumentProcessed, processDocument, processJiraIssue, finalizeBm25KeywordsForDocuments,     localizeMarkdownImageLinks, normalizeMarkdownLinkTarget, downloadImageAsset, downloadDataUriAsset, resolveAbsoluteImageUrl, isDataUri, determineImageExtension, mimeTypeToExtension, getKeywordConfig, buildKeywordOnlyIndexText, rebuildKeywordsIndexFromMetadata, normalizeKeywordsInput, cleanKeywords, normalizeMetadataKeywordFields, mergeKeywordsPreservingSignals, appendKeywordsToExisting, writeDocumentPromptFile, formatMetadataEntries, getStoredMetadataById, generateStoredMetadataById, updateStoredMetadataById, removeDocumentFromIndicesById, sanitizeFileSegment, getWorkspaceRootPath, getPageHtml, isLikelyHtml, extractHtmlTagData, resolveSourceUrl, extractJsonObject } = context;

async function annotateDocumentByArg(pageArg) {
  const allMetadata = readAllMetadata(storagePath);
  let metadata = allMetadata.find(item => String(item.id) === pageArg || String(item.title) === pageArg);
  if (!metadata) {
    const page = await fetchConfluencePage(pageArg);
    metadata = allMetadata.find(item => String(item.id) === String(page?.id));
  }
  if (!metadata) {
    return {
      message: `Document ${pageArg} is not in local store. Run refresh first.`
    };
  }
  const updated = await annotateStoredDocument(metadata);
  if (!updated) {
    return {
      message: `No local plain text content for ${metadata.title}. Run refresh first.`
    };
  }
  await finalizeBm25KeywordsForDocuments([metadata.id]);
  return {
    message: `Annotated document: ${metadata.title}`
  };
}

async function annotateAllDocuments() {
  const allMetadata = readAllMetadata(storagePath);
  let updatedCount = 0;
  const updatedIds = [];
  for (const metadata of allMetadata) {
    const updated = await annotateStoredDocument(metadata);
    if (updated) {
      updatedCount += 1;
      updatedIds.push(metadata.id);
    }
  }
  if (updatedCount > 0) {
    await finalizeBm25KeywordsForDocuments(updatedIds);
  }
  return {
    message: updatedCount > 0 ? `Annotated ${updatedCount} document(s)` : 'No local documents available to annotate. Run refresh first.'
  };
}

async function generateAnnotationWithLlm(metadata, content) {
  const originalKeywords = cleanKeywords(metadata?.keywords);
  const fallbackKeywords = generateKeywords(content);
  const fallbackSummary = generateSummary(content);
  function appendSynonymKeywords(baseKeywords, maxSynonyms = 6) {
    const orderedBase = cleanKeywords(baseKeywords);
    if (orderedBase.length === 0) {
      return [];
    }
    const synonymCandidates = cleanKeywords(generateExtendedKeywords(orderedBase), 80).filter(keyword => !orderedBase.includes(keyword)).slice(0, maxSynonyms);
    return cleanKeywords([...orderedBase, ...synonymCandidates]);
  }
  if (!vscode.lm || !vscode.LanguageModelChatMessage) {
    return {
      keywords: appendSynonymKeywords([...originalKeywords, ...fallbackKeywords]),
      summary: fallbackSummary
    };
  }
  try {
    const models = await vscode.lm.selectChatModels({});
    const model = models?.[0];
    if (!model) {
      return {
        keywords: appendSynonymKeywords([...originalKeywords, ...fallbackKeywords]),
        summary: fallbackSummary
      };
    }
    const prompt = [
      'You are annotating a local Confluence document metadata record.',
      'Return valid JSON only with shape: {"summary":"...","keywords":"keyword-a, keyword-b"}.',
      'Summary must be clear, relevant, and contain key information from the source content, with a maximum of 4 sentences.',
      'Keywords must be specific technical terms in one comma-separated string.',
      'Take existing document keywords (provided below) and filter out non-business-related words or stop words (e.g., "confluence", "jira", "and", "the", "that", etc.).',
      'Include a few close synonyms or related alternate terms that help retrieval.',
      `Title: ${metadata.title || ''}`,
      `Topic: ${metadata.parent_confluence_topic || ''}`,
      `Author: ${metadata.author || ''}`,
      `Existing Keywords: ${originalKeywords.join(', ')}`,
      `Existing Tags: ${metadata.tags ? metadata.tags.join(', ') : ''}`,
      `Existing Feedback: ${metadata.feedback || ''}`,
      'Document markdown content:',
      truncate(content, 100000)
    ].join('\\n');
    const response = await model.sendRequest([vscode.LanguageModelChatMessage.User(prompt)]);
    let responseText = '';
    for await (const fragment of response.text) {
      responseText += fragment;
    }
    const parsed = extractJsonObject(responseText) || {};
    const llmKeywords = cleanKeywords(parsed.keywords);
    const llmSummary = String(parsed.summary || '').trim();
    
    // Replace original keywords entirely with the LLM output, filling with fallback if needed
    const combinedLlmKeywords = cleanKeywords([...llmKeywords, ...fallbackKeywords]).slice(0, getKeywordConfig().DEFAULT_KEYWORD_LIMIT);
    const fallbackMergedKeywords = appendKeywordsToExisting(originalKeywords, fallbackKeywords, getKeywordConfig().DEFAULT_KEYWORD_LIMIT);
    
    return {
      keywords: llmKeywords.length > 0 ? appendSynonymKeywords(combinedLlmKeywords) : appendSynonymKeywords(fallbackMergedKeywords),
      summary: llmSummary || fallbackSummary
    };
  } catch {
    return {
      keywords: appendSynonymKeywords([...originalKeywords, ...fallbackKeywords]),
      summary: fallbackSummary
    };
  }
}

async function annotateStoredDocument(metadata) {
  const content = readDocumentContent(storagePath, metadata.id);
  if (!content) {
    return false;
  }
  const annotation = await generateAnnotationWithLlm(metadata, content);
  const updatedMetadata = normalizeMetadataKeywordFields({
    ...metadata,
    keywords: cleanKeywords(annotation.keywords, getKeywordConfig().DEFAULT_KEYWORD_LIMIT),
    summary: annotation.summary
  });
  writeDocumentFiles(storagePath, metadata.id, content, updatedMetadata);
  keywordsIndex.upsertDocument(updatedMetadata.id, buildKeywordOnlyIndexText(updatedMetadata));
  return true;
}

  return {
    annotateDocumentByArg,
    annotateAllDocuments,
    annotateStoredDocument,
    generateAnnotationWithLlm
  };
};
