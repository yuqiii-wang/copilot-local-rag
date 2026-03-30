module.exports = function(context) {
  const { fs, path, vscode, storagePath, indexStoragePath, fetchConfluencePage, fetchAllConfluencePages, fetchConfluencePageChildren, 
    fetchJiraIssue, truncate, tokenize, htmlToMarkdown, jiraTextToMarkdown, generateSynonyms, readAllMetadata, writeDocumentFiles, 
    readDocumentContent, localizeMarkdownImageLinks, getKeywordConfig, cleanKeywords, getStoredMetadataById,
    getPageHtml, isLikelyHtml, extractHtmlTagData, resolveSourceUrl, tokenization2bm25,
    buildCategorizedKeywords, normalizeCategorizedKeywords, flattenCategorizedKeywords } = context;
  const { extractMermaidKeywords } = require('./tokenization2keywords');
  const { getJiraExtractionRegexes } = require('../../mcp/jiraApi');
  const { extractMdKeywords } = require('./md2keywords');

  // Builds structural keywords from document content + title.
  // Used by both processDocument/processJiraIssue (initial sync) and finalizeBm25KeywordsForDocuments (BM25 refresh).
  function buildStructuralKeywords(docText, title) {
    const mdKeywords = extractMdKeywords(String(docText || ''));
    const titleKeywords = tokenize(String(title || ''));
    return [...titleKeywords, ...mdKeywords];
  }

  // Extracts all Jira issue keys explicitly referenced in text, using the configured jira.regex settings.
  function extractJiraReferences(text) {
    const found = new Set();
    const textStr = String(text || '');
    for (const regex of getJiraExtractionRegexes(vscode)) {
      const gr = new RegExp(regex.source, regex.flags.includes('i') ? 'gi' : 'g');
      for (const match of textStr.matchAll(gr)) {
        found.add(match[0].toUpperCase());
      }
    }
    return [...found];
  }

async function refreshDocument(pageArg, options = {}) {
  const page = await fetchConfluencePage(pageArg);
  const metadata = await processDocument(page);
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
}

async function refreshJiraIssue(issueArg, options = {}) {
  if (typeof fetchJiraIssue !== 'function') {
    throw new Error('Jira integration is not configured.');
  }
  const issue = await fetchJiraIssue(issueArg);
  const metadata = await processJiraIssue(issue);
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
  const existingMetadata = getStoredMetadataById(page.id) || {};
  const rawContent = getPageHtml(page);
  const isHtmlContent = isLikelyHtml(rawContent);
  const htmlTagData = isHtmlContent ? extractHtmlTagData(rawContent) : {
    title: '',
    keywords: []
  };
  const sourceUrl = resolveSourceUrl(page);
  const markdownBaseContent = isHtmlContent ? htmlToMarkdown(rawContent) : String(rawContent || '').trim();
  const markdownContent = await localizeMarkdownImageLinks(markdownBaseContent, page.id, sourceUrl);

  const title = htmlTagData.title || page.title;
  const existingSummary = String(existingMetadata.summary || '').trim();
  const kgMermaid = typeof existingMetadata.knowledgeGraph === 'string' ? existingMetadata.knowledgeGraph : '';

  const baseKeywords = buildCategorizedKeywords(title, existingSummary, markdownContent, { kgMermaid });
  const categorizedKeywords = buildCategorizedKeywords(title, existingSummary, markdownContent, {
    kgMermaid,
    synonymNGrams: generateSynonyms(flattenCategorizedKeywords(baseKeywords))
  });

  const referencedJiraIds = extractJiraReferences(markdownContent);

  const baseMetadata = {
    id: page.id,
    title,
    author: page.author || 'Unknown',
    last_updated: page.last_updated || new Date().toISOString().slice(0, 10),
    parent_confluence_topic: page.parent_confluence_topic || page.space || 'General',
    url: sourceUrl,
    type: 'confluence',
    keywords: categorizedKeywords,
    summary: '',
    tags: Array.isArray(existingMetadata.tags) ? existingMetadata.tags : [],
    feedback: String(existingMetadata.feedback || '').trim(),
    referencedQueries: Array.isArray(existingMetadata.referencedQueries) ? existingMetadata.referencedQueries : [],
    referencedJiraIds,
    knowledgeGraph: kgMermaid,
    relatedPages: Array.isArray(existingMetadata.relatedPages) ? existingMetadata.relatedPages : []
  };
  const metadata = {
    ...existingMetadata,
    ...baseMetadata,
    summary: existingSummary
  };
  writeDocumentFiles(storagePath, page.id, markdownContent, metadata);
  return metadata;
}

async function processJiraIssue(issue) {
  const existingMetadata = getStoredMetadataById(issue?.id) || {};
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
  const description = descriptionIsHtml ? htmlToMarkdown(rawDescription) : jiraTextToMarkdown(rawDescription);
  const issueKey = String(issue?.key || '').trim();
  const htmlTitle = summaryTagData.title || descriptionTagData.title;
  const title = htmlTitle || (issueKey && summary ? `${issueKey}: ${summary}` : issueKey || summary || `Issue ${issue?.id || ''}`.trim());
  const contentSections = [`# ${title}`, '', `Issue Key: ${issueKey || '-'}`, `Issue ID: ${issue?.id || '-'}`, `Project: ${projectKey}`, `Type: ${fields?.issuetype?.name || '-'}`, `Status: ${fields?.status?.name || '-'}`, `Priority: ${fields?.priority?.name || '-'}`, `Reporter: ${reporter}`, `Assignee: ${fields?.assignee?.displayName || '-'}`, `Updated: ${fields?.updated || '-'}`, '', '## Description', description || 'No description provided.'];
  const markdownContent = await localizeMarkdownImageLinks(contentSections.join('\n'), issue?.id, resolveSourceUrl(issue));

  const existingSummary = String(existingMetadata.summary || '').trim();
  const kgMermaid = typeof existingMetadata.knowledgeGraph === 'string' ? existingMetadata.knowledgeGraph : '';

  const baseKeywords = buildCategorizedKeywords(title, existingSummary, markdownContent, { kgMermaid });
  const categorizedKeywords = buildCategorizedKeywords(title, existingSummary, markdownContent, {
    kgMermaid,
    synonymNGrams: generateSynonyms(flattenCategorizedKeywords(baseKeywords))
  });

  const baseMetadata = {
    id: issue?.id,
    issueKey,
    title,
    author: reporter,
    last_updated: String(fields?.updated || new Date().toISOString().slice(0, 10)).slice(0, 10),
    parent_confluence_topic: `Jira ${projectKey}`,
    url: resolveSourceUrl(issue),
    type: 'jira',
    keywords: categorizedKeywords,
    summary: '',
    tags: Array.isArray(existingMetadata.tags) ? existingMetadata.tags : [],
    feedback: String(existingMetadata.feedback || '').trim(),
    referencedQueries: Array.isArray(existingMetadata.referencedQueries) ? existingMetadata.referencedQueries : [],
    knowledgeGraph: kgMermaid,
    relatedPages: Array.isArray(existingMetadata.relatedPages) ? existingMetadata.relatedPages : []
  };
  const metadata = {
    ...existingMetadata,
    ...baseMetadata,
    summary: existingSummary
  };
  writeDocumentFiles(storagePath, issue?.id, markdownContent, metadata);
  return metadata;
}

async function finalizeBm25KeywordsForDocuments(docIds = []) {
  const metadataList = readAllMetadata(storagePath);
  if (!Array.isArray(metadataList) || metadataList.length === 0) {
    return;
  }

  const targetIdSet = new Set((Array.isArray(docIds) ? docIds : []).map(value => String(value || '').trim()).filter(value => value.length > 0));
  if (targetIdSet.size === 0) {
    return;
  }

  // 1. Gather all documents and calculate IDF using ALL split content including ngrams
  const N = metadataList.length;
  const dfMap = new Map();
  const docTokensMap = new Map();
  const docLengthMap = new Map();
  let totalDocLength = 0;

  for (const meta of metadataList) {
    const docText = readDocumentContent(storagePath, meta.id) || '';
    const tokens = tokenization2bm25(docText);
    const tokenSet = new Set(tokens);
    
    docTokensMap.set(meta.id, tokens);
    docLengthMap.set(meta.id, tokens.length);
    totalDocLength += tokens.length;

    for (const token of tokenSet) {
      dfMap.set(token, (dfMap.get(token) || 0) + 1);
    }
  }

  const avgDocLength = totalDocLength / N || 1;

  // BM25 Parameters
  const k1 = 1.2;
  const b = 0.75;

  const idfMap = new Map();
  for (const [token, df] of dfMap.entries()) {
    const idf = Math.log((N - df + 0.5) / (df + 0.5) + 1);
    idfMap.set(token, idf);
  }

  const keywordConfig = getKeywordConfig();
  const topN = keywordConfig.BM25_KEYWORD_LIMIT || Math.floor((keywordConfig.DEFAULT_KEYWORD_LIMIT || 40) / 2);

  // 2. Score tokens and update metadata for target documents
  for (const docId of targetIdSet) {
    const tokens = docTokensMap.get(docId);
    const docText = readDocumentContent(storagePath, docId) || '';
    if (!tokens) continue;

    const docLen = docLengthMap.get(docId);
    const tfMap = new Map();
    for (const t of tokens) {
      tfMap.set(t, (tfMap.get(t) || 0) + 1);
    }

    const scores = [];
    for (const [token, tf] of tfMap.entries()) {
      const idf = idfMap.get(token) || 0;
      const tfScored = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (docLen / avgDocLength)));
      scores.push({ token, score: idf * tfScored });
    }

    scores.sort((a, b) => b.score - a.score);
    const bm25Keywords = scores.slice(0, topN).map(x => x.token);

    // 3. Rebuild categorized keywords with BM25 tokens filling the bm25 category
    const metaIndex = metadataList.findIndex(m => m.id === docId);
    if (metaIndex >= 0) {
      const meta = metadataList[metaIndex];
      const kgMermaid = typeof meta.knowledgeGraph === 'string' ? meta.knowledgeGraph : '';
      const existingSummary = String(meta.summary || '').trim();

      // Preserve LLM-annotated semantic keywords across BM25 refresh
      const oldNorm = normalizeCategorizedKeywords(meta.keywords);
      const existingSemantic = ['1gram', '2gram', '3gram', '4gram']
        .flatMap(g => {
          const slot = oldNorm.semantic[g];
          if (!slot) return [];
          return Array.isArray(slot) ? cleanKeywords(slot) : Object.keys(slot);
        });

      const bm25Base = buildCategorizedKeywords(
        meta.title,
        existingSummary,
        docText,
        { bm25Keywords, kgMermaid, existingSemantic }
      );
      meta.keywords = buildCategorizedKeywords(
        meta.title,
        existingSummary,
        docText,
        { bm25Keywords, kgMermaid, existingSemantic,
          synonymNGrams: generateSynonyms(flattenCategorizedKeywords(bm25Base)) }
      );
      writeDocumentFiles(storagePath, meta.id, docText, meta);
    }
  }
}

  function syncDefaultDocs(extensionPath) {
    const defaultDocsSrcDir = path.join(extensionPath, 'src', 'default_docs');
    if (!fs.existsSync(defaultDocsSrcDir)) {
      return;
    }

    const defaultDocFolders = fs.readdirSync(defaultDocsSrcDir).filter(f => fs.statSync(path.join(defaultDocsSrcDir, f)).isDirectory());
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
          } catch (e) {
            console.error(`Failed to sync default doc ${folder}:`, e);
          }
        }
      }
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
