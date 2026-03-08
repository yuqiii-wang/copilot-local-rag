module.exports = function(context) {
  const { vscode, storagePath, indexStoragePath, fetchConfluencePage, fetchAllConfluencePages, fetchJiraIssue, truncate, tokenize, htmlToMarkdown, generateKeywords, generateExtendedKeywords, generateSummary, readAllMetadata, writeDocumentFiles, readDocumentContent, rankDocumentsByIdf, bm25Index, keywordsIndex, rankLocalDocuments, checkLocalDocumentsAgentic, refreshDocument, refreshAllDocuments, refreshJiraIssue, notifyDocumentProcessed, processDocument, processJiraIssue, finalizeBm25KeywordsForDocuments, annotateDocumentByArg, annotateAllDocuments, annotateStoredDocument, generateAnnotationWithLlm,         getKeywordConfig, buildKeywordOnlyIndexText, rebuildKeywordsIndexFromMetadata, normalizeKeywordsInput, cleanKeywords, normalizeMetadataKeywordFields, mergeKeywordsPreservingSignals, appendKeywordsToExisting, writeDocumentPromptFile, formatMetadataEntries, getStoredMetadataById, generateStoredMetadataById, updateStoredMetadataById, removeDocumentFromIndicesById, sanitizeFileSegment, getWorkspaceRootPath, getPageHtml, isLikelyHtml, extractHtmlTagData, resolveSourceUrl } = context;

async function localizeMarkdownImageLinks(markdownContent, docId, sourceUrl) {
  const markdown = String(markdownContent || '');
  const imagePattern = /!\[([^\]]*)\]\(([^)]+)\)/g;
  const matches = [...markdown.matchAll(imagePattern)];
  if (!docId || matches.length === 0) {
    return markdown;
  }
  const imagesDir = path.join(storagePath, String(docId), 'images');
  fs.mkdirSync(imagesDir, {
    recursive: true
  });
  const localizedBySource = new Map();
  let imageIndex = 0;
  for (const match of matches) {
    const originalSrcRaw = String(match[2] || '').trim();
    const originalSrc = normalizeMarkdownLinkTarget(originalSrcRaw);
    if (!originalSrc || localizedBySource.has(originalSrc)) {
      continue;
    }
    try {
      imageIndex += 1;
      const downloadResult = await downloadImageAsset({
        source: originalSrc,
        sourceUrl,
        outputDir: imagesDir,
        imageIndex
      });
      if (downloadResult?.relativePath) {
        localizedBySource.set(originalSrc, downloadResult.relativePath);
      }
    } catch {
      // Keep original image URL if download fails.
    }
  }
  if (localizedBySource.size === 0) {
    return markdown;
  }
  return markdown.replace(imagePattern, (fullMatch, alt, src) => {
    const normalizedSource = normalizeMarkdownLinkTarget(src);
    const localizedPath = localizedBySource.get(normalizedSource);
    if (!localizedPath) {
      return fullMatch;
    }
    return `![${String(alt || '').trim()}](${localizedPath})`;
  });
}

function normalizeMarkdownLinkTarget(rawValue) {
  const value = String(rawValue || '').trim();
  if (!value) {
    return '';
  }
  if (value.startsWith('<') && value.endsWith('>')) {
    return value.slice(1, -1).trim();
  }
  return value;
}

async function downloadImageAsset({
  source,
  sourceUrl,
  outputDir,
  imageIndex
}) {
  if (isDataUri(source)) {
    return downloadDataUriAsset(source, outputDir, imageIndex);
  }
  const resolvedUrl = resolveAbsoluteImageUrl(source, sourceUrl);
  if (!resolvedUrl) {
    return null;
  }
  const response = await axios.get(resolvedUrl, {
    responseType: 'arraybuffer',
    timeout: 15000
  });
  const extension = determineImageExtension(source, response?.headers?.['content-type']);
  const fileName = `image-${String(imageIndex).padStart(3, '0')}${extension}`;
  const filePath = path.join(outputDir, fileName);
  fs.writeFileSync(filePath, Buffer.from(response.data));
  return {
    relativePath: `images/${fileName}`
  };
}

function downloadDataUriAsset(dataUri, outputDir, imageIndex) {
  const parsed = /^data:([^;,]+)?(?:;charset=[^;,]+)?(;base64)?,(.*)$/i.exec(String(dataUri || ''));
  if (!parsed) {
    return null;
  }
  const mimeType = String(parsed[1] || '').toLowerCase();
  const isBase64 = Boolean(parsed[2]);
  const payload = parsed[3] || '';
  const extension = determineImageExtension('', mimeType || 'image/png');
  const fileName = `image-${String(imageIndex).padStart(3, '0')}${extension}`;
  const filePath = path.join(outputDir, fileName);
  const bytes = isBase64 ? Buffer.from(payload, 'base64') : Buffer.from(decodeURIComponent(payload), 'utf8');
  fs.writeFileSync(filePath, bytes);
  return {
    relativePath: `images/${fileName}`
  };
}

function resolveAbsoluteImageUrl(source, sourceUrl) {
  const src = String(source || '').trim();
  if (!src) {
    return null;
  }
  if (/^https?:\/\//i.test(src)) {
    return src;
  }
  if (/^\/\//.test(src)) {
    try {
      const protocol = new URL(String(sourceUrl || '')).protocol || 'https:';
      return `${protocol}${src}`;
    } catch {
      return `https:${src}`;
    }
  }
  try {
    const baseUrl = new URL(String(sourceUrl || ''));
    return new URL(src, baseUrl).toString();
  } catch {
    return null;
  }
}

function isDataUri(value) {
  return /^data:image\//i.test(String(value || '').trim());
}

function determineImageExtension(source, contentType) {
  const fromContentType = mimeTypeToExtension(contentType);
  if (fromContentType) {
    return fromContentType;
  }
  const cleanSource = String(source || '').split('?')[0].split('#')[0];
  const ext = path.extname(cleanSource).toLowerCase();
  if (ext && ext.length <= 6) {
    return ext;
  }
  return '.png';
}

function mimeTypeToExtension(contentType) {
  const normalized = String(contentType || '').split(';')[0].trim().toLowerCase();
  if (normalized === 'image/jpeg' || normalized === 'image/jpg') {
    return '.jpg';
  }
  if (normalized === 'image/png') {
    return '.png';
  }
  if (normalized === 'image/gif') {
    return '.gif';
  }
  if (normalized === 'image/webp') {
    return '.webp';
  }
  if (normalized === 'image/svg+xml') {
    return '.svg';
  }
  if (normalized === 'image/bmp') {
    return '.bmp';
  }
  if (normalized === 'image/x-icon' || normalized === 'image/vnd.microsoft.icon') {
    return '.ico';
  }
  return '';
}

  return {
    localizeMarkdownImageLinks,
    normalizeMarkdownLinkTarget,
    downloadImageAsset,
    downloadDataUriAsset,
    resolveAbsoluteImageUrl,
    isDataUri,
    determineImageExtension,
    mimeTypeToExtension
  };
};
