module.exports = function(context) {
  const { fs, path, vscode, storagePath, indexStoragePath, fetchConfluencePage, fetchAllConfluencePages, fetchConfluencePageChildren, fetchJiraIssue, truncate, tokenize, htmlToMarkdown, generateKeywords, generateExtendedKeywords, generateSummary, readAllMetadata, writeDocumentFiles, readDocumentContent, rankDocumentsByIdf, bm25Index, keywordsIndex, rankLocalDocuments, checkLocalDocumentsAgentic,        annotateDocumentByArg, annotateAllDocuments, annotateStoredDocument, generateAnnotationWithLlm, localizeMarkdownImageLinks, normalizeMarkdownLinkTarget, downloadImageAsset, downloadDataUriAsset, resolveAbsoluteImageUrl, isDataUri, determineImageExtension, mimeTypeToExtension, getKeywordConfig, buildKeywordOnlyIndexText, rebuildKeywordsIndexFromMetadata, normalizeKeywordsInput, cleanKeywords, normalizeMetadataKeywordFields, mergeKeywordsPreservingSignals, appendKeywordsToExisting, writeDocumentPromptFile, formatMetadataEntries, getStoredMetadataById, generateStoredMetadataById, updateStoredMetadataById, removeDocumentFromIndicesById, sanitizeFileSegment, getWorkspaceRootPath, getPageHtml, isLikelyHtml, extractHtmlTagData, resolveSourceUrl } = context;

async function refreshDocument(pageArg, options = {}) {
  const page = await fetchConfluencePage(pageArg);
  const metadata = await processDocument(page);
  await finalizeBm25KeywordsForDocuments([metadata.id]);
  notifyDocumentProcessed(options, metadata, 1, 1);
}

async function refreshConfluenceHierarchy(pageArg, options = {}) {
  const rootPage = await fetchConfluencePage(pageArg);
  const pagesToProcess = [rootPage];
  const processedIds = new Set();
  let currentIndex = 0;
  const delay = (ms) => new Promise(res => setTimeout(res, ms));
  while (pagesToProcess.length > 0) {
    const page = pagesToProcess.shift();
    if (processedIds.has(page.id)) continue;
    processedIds.add(page.id);
    const metadata = await processDocument(page);
    await finalizeBm25KeywordsForDocuments([metadata.id]);
    currentIndex++;
    notifyDocumentProcessed(options, metadata, currentIndex, currentIndex + pagesToProcess.length);
    try {
      const childrenData = await fetchConfluencePageChildren(page.id);
      if (childrenData && childrenData.results) {
        for (const child of childrenData.results) {
          if (!processedIds.has(child.id)) pagesToProcess.push(child);
        }
      }
    } catch (e) {
      console.error(`Failed to fetch children for page ${page.id}`, e);
    }
    if (pagesToProcess.length > 0) await delay(10000);
  }
}

async function refreshAllDocuments(options = {}) {
  const pages = await fetchAllConfluencePages();
  const total = pages.length;
  const refreshedIds = [];
  for (let index = 0; index < total; index += 1) {
    const page = pages[index];
    const metadata = await processDocument(page);
    refreshedIds.push(metadata.id);
    notifyDocumentProcessed(options, metadata, index + 1, total);
  }
  await finalizeBm25KeywordsForDocuments(refreshedIds);
}

async function refreshJiraIssue(issueArg, options = {}) {
  if (typeof fetchJiraIssue !== 'function') {
    throw new Error('Jira integration is not configured.');
  }
  const issue = await fetchJiraIssue(issueArg);
  const metadata = await processJiraIssue(issue);
  await finalizeBm25KeywordsForDocuments([metadata.id]);
  notifyDocumentProcessed(options, metadata, 1, 1);
}

function notifyDocumentProcessed(options, metadata, index, total) {
  if (!options || typeof options.onDocumentProcessed !== 'function') {
    return;
  }
  options.onDocumentProcessed({
    metadata,
    index,
    total
  });
}

async function processDocument(page) {
  const rawContent = getPageHtml(page);
  const isHtmlContent = isLikelyHtml(rawContent);
  const htmlTagData = isHtmlContent ? extractHtmlTagData(rawContent) : {
    title: '',
    keywords: []
  };
  const sourceUrl = resolveSourceUrl(page);
  const markdownBaseContent = isHtmlContent ? htmlToMarkdown(rawContent) : String(rawContent || '').trim();
  const markdownContent = await localizeMarkdownImageLinks(markdownBaseContent, page.id, sourceUrl);
  const baseMetadata = {
    id: page.id,
    title: htmlTagData.title || page.title,
    author: page.author || 'Unknown',
    last_updated: page.last_updated || new Date().toISOString().slice(0, 10),
    parent_confluence_topic: page.parent_confluence_topic || page.space || 'General',
    url: sourceUrl,
    keywords: [],
    extended_keywords: [],
    summary: ''
  };
  const tokenizationKeywords = cleanKeywords(generateKeywords(markdownContent), getKeywordConfig().TOKENIZATION_KEYWORD_LIMIT);
  bm25Index.upsertDocument(page.id, markdownContent);
  const bm25Keywords = cleanKeywords(bm25Index.extractKeywordsForDocument(page.id, {
    limit: getKeywordConfig().BM25_KEYWORD_LIMIT
  }), getKeywordConfig().BM25_KEYWORD_LIMIT);
  const mergedKeywords = mergeKeywordsPreservingSignals({
    structuralKeywords: tokenizationKeywords,
    modelKeywords: bm25Keywords,
    limit: getKeywordConfig().DEFAULT_KEYWORD_LIMIT
  });
  const metadata = {
    ...baseMetadata,
    keywords: mergedKeywords,
    extended_keywords: cleanKeywords(generateExtendedKeywords(mergedKeywords), 80),
    summary: ''
  };
  writeDocumentFiles(storagePath, page.id, markdownContent, metadata);
  return metadata;
}

async function processJiraIssue(issue) {
  const fields = issue?.fields || {};
  const reporter = fields?.reporter?.displayName || 'Unknown';
  const projectKey = fields?.project?.key || 'Jira';
  const rawSummary = String(fields?.summary || issue?.summary || '').trim();
  const rawDescription = String(fields?.description || issue?.description || '').trim();
  const summaryIsHtml = isLikelyHtml(rawSummary);
  const descriptionIsHtml = isLikelyHtml(rawDescription);
  const summaryTagData = summaryIsHtml ? extractHtmlTagData(rawSummary) : {
    title: '',
    keywords: []
  };
  const descriptionTagData = descriptionIsHtml ? extractHtmlTagData(rawDescription) : {
    title: '',
    keywords: []
  };
  const summary = summaryIsHtml ? htmlToMarkdown(rawSummary).replace(/\s+/g, ' ').trim() : rawSummary;
  const description = descriptionIsHtml ? htmlToMarkdown(rawDescription) : rawDescription;
  const issueKey = String(issue?.key || '').trim();
  const htmlTitle = summaryTagData.title || descriptionTagData.title;
  const title = htmlTitle || (issueKey && summary ? `${issueKey}: ${summary}` : issueKey || summary || `Issue ${issue?.id || ''}`.trim());
  const contentSections = [`# ${title}`, '', `Issue Key: ${issueKey || '-'}`, `Issue ID: ${issue?.id || '-'}`, `Project: ${projectKey}`, `Type: ${fields?.issuetype?.name || '-'}`, `Status: ${fields?.status?.name || '-'}`, `Priority: ${fields?.priority?.name || '-'}`, `Reporter: ${reporter}`, `Assignee: ${fields?.assignee?.displayName || '-'}`, `Updated: ${fields?.updated || '-'}`, '', '## Description', description || 'No description provided.'];
  const markdownContent = await localizeMarkdownImageLinks(contentSections.join('\n'), issue?.id, resolveSourceUrl(issue));
  const baseMetadata = {
    id: issue?.id,
    title,
    author: reporter,
    last_updated: String(fields?.updated || new Date().toISOString().slice(0, 10)).slice(0, 10),
    parent_confluence_topic: `Jira ${projectKey}`,
    url: resolveSourceUrl(issue),
    keywords: [],
    extended_keywords: [],
    summary: ''
  };
  const tokenizationKeywords = cleanKeywords(generateKeywords(markdownContent), getKeywordConfig().TOKENIZATION_KEYWORD_LIMIT);
  bm25Index.upsertDocument(issue?.id, markdownContent);
  const bm25Keywords = cleanKeywords(bm25Index.extractKeywordsForDocument(issue?.id, {
    limit: getKeywordConfig().BM25_KEYWORD_LIMIT
  }), getKeywordConfig().BM25_KEYWORD_LIMIT);
  const mergedKeywords = mergeKeywordsPreservingSignals({
    structuralKeywords: tokenizationKeywords,
    modelKeywords: bm25Keywords,
    limit: getKeywordConfig().DEFAULT_KEYWORD_LIMIT
  });
  const metadata = {
    ...baseMetadata,
    keywords: mergedKeywords,
    extended_keywords: cleanKeywords(generateExtendedKeywords(mergedKeywords), 80),
    summary: ''
  };
  writeDocumentFiles(storagePath, issue?.id, markdownContent, metadata);
  return metadata;
}

async function finalizeBm25KeywordsForDocuments(docIds = []) {
  const metadataList = readAllMetadata(storagePath);
  if (!Array.isArray(metadataList) || metadataList.length === 0) {
    return;
  }
  const corpus = metadataList.map(item => ({
    id: item.id,
    text: readDocumentContent(storagePath, item.id) || ''
  }));
  bm25Index.rebuildDocuments(corpus);
  const targetIdSet = new Set((Array.isArray(docIds) ? docIds : []).map(value => String(value || '').trim()).filter(value => value.length > 0));
  if (targetIdSet.size === 0) {
    return;
  }
  for (const metadataEntry of metadataList) {
    const metadata = normalizeMetadataKeywordFields(metadataEntry);
    const id = String(metadata?.id || '').trim();
    if (!id || !targetIdSet.has(id)) {
      continue;
    }
    const content = readDocumentContent(storagePath, id);
    if (!content) {
      continue;
    }
    const bm25Keywords = cleanKeywords(bm25Index.extractKeywordsForDocument(id, {
      limit: getKeywordConfig().BM25_KEYWORD_LIMIT
    }), getKeywordConfig().BM25_KEYWORD_LIMIT);
    const tokenizationKeywords = cleanKeywords(generateKeywords(content), getKeywordConfig().TOKENIZATION_KEYWORD_LIMIT);
    const mergedKeywords = mergeKeywordsPreservingSignals({
      structuralKeywords: tokenizationKeywords,
      modelKeywords: bm25Keywords,
      limit: getKeywordConfig().DEFAULT_KEYWORD_LIMIT
    });
    if (mergedKeywords.length === 0) {
      continue;
    }
    const updatedMetadata = normalizeMetadataKeywordFields({
      ...metadata,
      keywords: mergedKeywords
    });
    writeDocumentFiles(storagePath, id, content, updatedMetadata);
  }
  const refreshedMetadata = readAllMetadata(storagePath).map(normalizeMetadataKeywordFields);
  rebuildKeywordsIndexFromMetadata(refreshedMetadata);
}

  function syncDefaultDocs(extensionPath) {
    const defaultDocsSrcDir = path.join(extensionPath, 'src', 'default_docs');
    if (!fs.existsSync(defaultDocsSrcDir)) {
      return;
    }

    const defaultDocFolders = fs.readdirSync(defaultDocsSrcDir).filter(f => fs.statSync(path.join(defaultDocsSrcDir, f)).isDirectory());
    let addedAny = false;
    for (const folder of defaultDocFolders) {
      if (!fs.existsSync(path.join(storagePath, folder))) {
        const srcFolder = path.join(defaultDocsSrcDir, folder);
        const mdPath = path.join(srcFolder, 'content.md');
        const jsonPath = path.join(srcFolder, 'metadata.json');
        
        if (fs.existsSync(mdPath) && fs.existsSync(jsonPath)) {
          try {
            const mdContent = fs.readFileSync(mdPath, 'utf8');
            const metadata = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
            writeDocumentFiles(storagePath, folder, mdContent, metadata);
            bm25Index.upsertDocument(folder, mdContent);
            addedAny = true;
          } catch (e) {
            console.error(`Failed to sync default doc ${folder}:`, e);
          }
        }
      }
    }
    if (addedAny) {
      context.rebuildKeywordsIndexFromMetadata(readAllMetadata(storagePath));
    }
  }

  return {
    syncDefaultDocs,
    refreshDocument,
    refreshConfluenceHierarchy,
    refreshAllDocuments,
    refreshJiraIssue,
    notifyDocumentProcessed,
    processDocument,
    processJiraIssue,
    finalizeBm25KeywordsForDocuments
  };
};
