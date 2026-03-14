const fs = require('fs');
const path = require('path');

module.exports = function(context) {
  const { vscode, storagePath, indexStoragePath, fetchConfluencePage, 
    fetchAllConfluencePages, fetchJiraIssue, truncate, tokenize, 
    htmlToMarkdown, generateKeywords, generateExtendedKeywords, 
    generateSummary, readAllMetadata, writeDocumentFiles, readDocumentContent, 
    rankDocumentsByIdf, bm25Index, keywordsIndex, rankLocalDocuments, 
    checkLocalDocumentsAgentic, refreshDocument, refreshAllDocuments, 
    refreshJiraIssue, notifyDocumentProcessed, processDocument, processJiraIssue, 
    finalizeBm25KeywordsForDocuments, annotateDocumentByArg, annotateAllDocuments,
     annotateStoredDocument, generateAnnotationWithLlm, localizeMarkdownImageLinks, 
     normalizeMarkdownLinkTarget, downloadImageAsset, downloadDataUriAsset, 
     resolveAbsoluteImageUrl, isDataUri, determineImageExtension, mimeTypeToExtension, 
     getKeywordConfig, buildKeywordOnlyIndexText, rebuildKeywordsIndexFromMetadata, 
     normalizeKeywordsInput, cleanKeywords, normalizeMetadataKeywordFields, 
     mergeKeywordsPreservingSignals, appendKeywordsToExisting, cheerio} = context;

function writeDocumentPromptFile(metadata, content) {
  const workspaceRoot = getWorkspaceRootPath();
  const promptsDir = path.join(workspaceRoot, '.github', 'prompts');

  if (!fs.existsSync(promptsDir)) {
    fs.mkdirSync(promptsDir, { recursive: true });
  }

  const safeTitle = sanitizeFileSegment(metadata.title || 'document');
  const safeId = sanitizeFileSegment(metadata.id || 'unknown');
  const fileName = `${safeTitle}-${safeId}.prompt.md`;
  const filePath = path.join(promptsDir, fileName);
  const promptText = [`# ${metadata.title || 'Untitled'}`, '', `Source ID: ${metadata.id || ''}`, `Author: ${metadata.author || 'Unknown'}`, `Last Updated: ${metadata.last_updated || ''}`, `Parent Topic: ${metadata.parent_confluence_topic || ''}`, '', '## Instructions', 'Use the following document content as authoritative context when answering questions about this topic.', '', '## Content', content].join('\n');
  fs.writeFileSync(filePath, promptText, 'utf8');
  return filePath;
}



function getPageHtml(page) {
  if (typeof page?.content === 'string') {
    return page.content;
  }
  if (typeof page?.body?.storage?.value === 'string') {
    return page.body.storage.value;
  }
  return '';
}

function isLikelyHtml(value) {
  const text = String(value || '').trim();
  return /<[a-z][\s\S]*>/i.test(text);
}

function extractHtmlTagData(html) {
  const $ = cheerio.load(String(html || ''));
  const extractedTitle = ($('title').first().text() || $('h1').first().text() || '').trim();
  const keywordCandidates = [];
  $('meta[name="keywords"], meta[name="news_keywords"], meta[property="article:tag"]').each((_, element) => {
    const content = $(element).attr('content');
    if (content) {
      keywordCandidates.push(...String(content).split(','));
    }
  });
  $('h1, h2, h3').each((_, element) => {
    const heading = $(element).text().trim();
    if (heading) {
      keywordCandidates.push(heading);
    }
  });
  return {
    title: extractedTitle,
    keywords: cleanKeywords(keywordCandidates)
  };
}

function resolveSourceUrl(source) {
  let candidate = source?.url || source?._links?.webui || source?._links?.self || source?.self || '';
  candidate = String(candidate || '').trim();

  if (candidate && !candidate.startsWith('http://') && !candidate.startsWith('https://')) {
    const isJira = source?.key || (source?.fields && source?.fields?.project);
    const configuration = vscode.workspace.getConfiguration('repoAsk');
    let base = '';
    
    if (isJira) {
      const jiraProfile = configuration.get('jira');
      const jiraObjectUrl = jiraProfile && typeof jiraProfile === 'object' ? jiraProfile.url : undefined;
      base = String(jiraObjectUrl || 'http://127.0.0.1:8002').replace(/\/$/, '');
    } else {
      const confProfile = configuration.get('confluence');
      const confObjectUrl = confProfile && typeof confProfile === 'object' ? confProfile.url : undefined;
      base = String(confObjectUrl || 'http://127.0.0.1:8001').replace(/\/$/, '');
    }

    if (!candidate.startsWith('/')) {
      candidate = '/' + candidate;
    }
    candidate = base + candidate;
  }
  
  return candidate;
}

function getStoredMetadataById(docId) {
  const safeId = String(docId || '').trim();
  if (!safeId) {
    return null;
  }
  const allMetadata = readAllMetadata(storagePath);
  const found = allMetadata.find(item => String(item.id) === safeId) || null;
  return found ? normalizeMetadataKeywordFields(found) : null;
}

async function generateStoredMetadataById(docId) {
  const metadata = getStoredMetadataById(docId);
  if (!metadata) {
    throw new Error(`Document ${docId} not found in local store.`);
  }
  const content = readDocumentContent(storagePath, metadata.id);
  if (!content) {
    throw new Error(`No local content found for document ${docId}.`);
  }
  const annotation = await generateAnnotationWithLlm(metadata, content);
  const updatedMetadata = normalizeMetadataKeywordFields({
    ...metadata,
    keywords: cleanKeywords(annotation.keywords, getKeywordConfig().DEFAULT_KEYWORD_LIMIT),
    summary: String(annotation.summary || '').trim()
  });
  writeDocumentFiles(storagePath, metadata.id, content, updatedMetadata);
  keywordsIndex.upsertDocument(updatedMetadata.id, buildKeywordOnlyIndexText(updatedMetadata));
  return updatedMetadata;
}

function updateStoredMetadataById(docId, patch = {}) {
  const metadata = getStoredMetadataById(docId);
  if (!metadata) {
    throw new Error(`Document ${docId} not found in local store.`);
  }
  const content = readDocumentContent(storagePath, metadata.id);
  if (!content) {
    throw new Error(`No local content found for document ${docId}.`);
  }
  const tokenizationKeywords = cleanKeywords(generateKeywords(content), getKeywordConfig().TOKENIZATION_KEYWORD_LIMIT);
  const manualKeywords = cleanKeywords(patch.keywords, getKeywordConfig().DEFAULT_KEYWORD_LIMIT);
  const nextKeywords = mergeKeywordsPreservingSignals({
    structuralKeywords: tokenizationKeywords,
    lexicalKeywords: manualKeywords,
    limit: getKeywordConfig().DEFAULT_KEYWORD_LIMIT
  });
  const nextSummary = String(patch.summary || '').trim();
  const nextType = patch.type !== undefined ? String(patch.type || '').trim() : metadata.type;
  const nextTags = cleanKeywords(patch.tags, getKeywordConfig().DEFAULT_KEYWORD_LIMIT);
  const nextFeedback = String(patch.feedback || '').trim();
  const updatedMetadata = normalizeMetadataKeywordFields({
    ...metadata,
    type: nextType,
    keywords: nextKeywords,
    tags: nextTags,
    feedback: nextFeedback,
    summary: nextSummary
  });
  writeDocumentFiles(storagePath, metadata.id, content, updatedMetadata);
  keywordsIndex.upsertDocument(updatedMetadata.id, buildKeywordOnlyIndexText(updatedMetadata));
  return updatedMetadata;
}

function removeDocumentFromIndicesById(docId) {
  bm25Index.removeDocument(docId);
  keywordsIndex.removeDocument(docId);
}

function sanitizeFileSegment(value) {
  return String(value || 'item').toLowerCase().replace(/[^a-z0-9-_ ]+/g, '').trim().replace(/\s+/g, '-').slice(0, 64) || 'item';
}

function getWorkspaceRootPath() {
  const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
  if (!workspaceFolder) {
    throw new Error('Open a workspace folder to add prompt files.');
  }
  return workspaceFolder.uri.fsPath;
}

  return {
    writeDocumentPromptFile,
    getStoredMetadataById,
    generateStoredMetadataById,
    updateStoredMetadataById,
    removeDocumentFromIndicesById,
    sanitizeFileSegment,
    getWorkspaceRootPath,
    getPageHtml,
    isLikelyHtml,
    extractHtmlTagData,
    resolveSourceUrl
  };
};
