const fs = require('fs');
const path = require('path');
const cheerio = require('cheerio');
const axios = require('axios');
const { extractJsonObject } = require('./llm');

function createDocumentService(deps) {
    const {
        vscode,
        storagePath,
        fetchConfluencePage,
        fetchAllConfluencePages,
        fetchJiraIssue,
        truncate,
        tokenize,
        htmlToMarkdown,
        generateKeywords,
        generateSummary,
        readAllMetadata,
        writeDocumentFiles,
        readDocumentContent,
        rankDocumentsByIdf
    } = deps;

    function rankLocalDocuments(query, limit = 20) {
        const metadataList = readAllMetadata(storagePath);
        if (metadataList.length === 0) {
            return [];
        }

        const corpus = metadataList.map(metadata => ({
            ...metadata,
            content: readDocumentContent(storagePath, metadata.id) || ''
        }));

        return rankDocumentsByIdf(query, corpus, tokenize, { limit, minScore: 0.01 });
    }

    async function refreshDocument(pageArg, options = {}) {
        const page = await fetchConfluencePage(pageArg);
        const metadata = await processDocument(page);
        notifyDocumentProcessed(options, metadata, 1, 1);
    }

    async function refreshAllDocuments(options = {}) {
        const pages = await fetchAllConfluencePages();
        const total = pages.length;

        for (let index = 0; index < total; index += 1) {
            const page = pages[index];
            const metadata = await processDocument(page);
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

        return {
            message: `Annotated document: ${metadata.title}`
        };
    }

    async function annotateAllDocuments() {
        const allMetadata = readAllMetadata(storagePath);
        let updatedCount = 0;

        for (const metadata of allMetadata) {
            const updated = await annotateStoredDocument(metadata);
            if (updated) {
                updatedCount += 1;
            }
        }

        return {
            message: updatedCount > 0
                ? `Annotated ${updatedCount} document(s)`
                : 'No local documents available to annotate. Run refresh first.'
        };
    }

    function writeDocumentPromptFile(metadata, content) {
        const workspaceRoot = getWorkspaceRootPath();
        const promptsDir = path.join(workspaceRoot, '.github', 'prompts');
        fs.mkdirSync(promptsDir, { recursive: true });

        const safeTitle = sanitizeFileSegment(metadata.title || 'document');
        const safeId = sanitizeFileSegment(metadata.id || 'unknown');
        const fileName = `${safeTitle}-${safeId}.prompt.md`;
        const filePath = path.join(promptsDir, fileName);

        const promptText = [
            `# ${metadata.title || 'Untitled'}`,
            '',
            `Source ID: ${metadata.id || ''}`,
            `Author: ${metadata.author || 'Unknown'}`,
            `Last Updated: ${metadata.last_updated || ''}`,
            `Parent Topic: ${metadata.parent_confluence_topic || ''}`,
            '',
            '## Instructions',
            'Use the following document content as authoritative context when answering questions about this topic.',
            '',
            '## Content',
            content
        ].join('\n');

        fs.writeFileSync(filePath, promptText, 'utf8');
        return filePath;
    }

    function formatMetadataEntries(metadata) {
        if (!metadata || typeof metadata !== 'object') {
            return [{ key: 'title', value: 'Unknown' }];
        }

        return Object.entries(metadata).map(([key, value]) => {
            if (Array.isArray(value)) {
                return { key, value: value.join(', ') };
            }
            if (value && typeof value === 'object') {
                return { key, value: JSON.stringify(value) };
            }
            return { key, value: String(value ?? '') };
        });
    }

    async function processDocument(page) {
        const rawContent = getPageHtml(page);
        const isHtmlContent = isLikelyHtml(rawContent);
        const htmlTagData = isHtmlContent ? extractHtmlTagData(rawContent) : { title: '', keywords: [] };
        const sourceUrl = resolveSourceUrl(page);
        const markdownBaseContent = isHtmlContent ? htmlToMarkdown(rawContent) : String(rawContent || '').trim();
        const markdownContent = await localizeMarkdownImageLinks(markdownBaseContent, page.id, sourceUrl);
        const incomingKeywords = cleanKeywords([
            ...(Array.isArray(page.keywords) ? page.keywords : []),
            ...htmlTagData.keywords
        ]);
        const baseMetadata = {
            id: page.id,
            title: htmlTagData.title || page.title,
            author: page.author || 'Unknown',
            last_updated: page.last_updated || new Date().toISOString().slice(0, 10),
            parent_confluence_topic: page.parent_confluence_topic || page.space || 'General',
            url: sourceUrl,
            keywords: [],
            summary: ''
        };

        const llmAnnotation = await generateAnnotationWithLlm(baseMetadata, markdownContent);
        const llmKeywords = cleanKeywords(llmAnnotation.keywords);
        const tokenizationKeywords = cleanKeywords(generateKeywords(markdownContent));
        const metadata = {
            ...baseMetadata,
            keywords: cleanKeywords([...incomingKeywords, ...tokenizationKeywords, ...llmKeywords]),
            summary: String(llmAnnotation.summary || '').trim() || generateSummary(markdownContent)
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
        const summaryTagData = summaryIsHtml ? extractHtmlTagData(rawSummary) : { title: '', keywords: [] };
        const descriptionTagData = descriptionIsHtml ? extractHtmlTagData(rawDescription) : { title: '', keywords: [] };
        const summary = summaryIsHtml
            ? htmlToMarkdown(rawSummary).replace(/\s+/g, ' ').trim()
            : rawSummary;
        const description = descriptionIsHtml
            ? htmlToMarkdown(rawDescription)
            : rawDescription;
        const issueKey = String(issue?.key || '').trim();
        const htmlTitle = summaryTagData.title || descriptionTagData.title;
        const title = htmlTitle
            || (issueKey && summary ? `${issueKey}: ${summary}` : (issueKey || summary || `Issue ${issue?.id || ''}`.trim()));
        const contentSections = [
            `# ${title}`,
            '',
            `Issue Key: ${issueKey || '-'}`,
            `Issue ID: ${issue?.id || '-'}`,
            `Project: ${projectKey}`,
            `Type: ${fields?.issuetype?.name || '-'}`,
            `Status: ${fields?.status?.name || '-'}`,
            `Priority: ${fields?.priority?.name || '-'}`,
            `Reporter: ${reporter}`,
            `Assignee: ${fields?.assignee?.displayName || '-'}`,
            `Updated: ${fields?.updated || '-'}`,
            '',
            '## Description',
            description || 'No description provided.'
        ];

        const markdownContent = await localizeMarkdownImageLinks(
            contentSections.join('\n'),
            issue?.id,
            resolveSourceUrl(issue)
        );
        const incomingKeywords = cleanKeywords([
            ...(Array.isArray(fields?.labels) ? fields.labels : []),
            ...summaryTagData.keywords,
            ...descriptionTagData.keywords
        ]);
        const baseMetadata = {
            id: issue?.id,
            title,
            author: reporter,
            last_updated: String(fields?.updated || new Date().toISOString().slice(0, 10)).slice(0, 10),
            parent_confluence_topic: `Jira ${projectKey}`,
            url: resolveSourceUrl(issue),
            keywords: [],
            summary: ''
        };

        const llmAnnotation = await generateAnnotationWithLlm(baseMetadata, markdownContent);
        const llmKeywords = cleanKeywords(llmAnnotation.keywords);
        const tokenizationKeywords = cleanKeywords(generateKeywords(markdownContent));
        const metadata = {
            ...baseMetadata,
            keywords: cleanKeywords([...incomingKeywords, ...tokenizationKeywords, ...llmKeywords]),
            summary: String(llmAnnotation.summary || '').trim() || generateSummary(markdownContent)
        };

        writeDocumentFiles(storagePath, issue?.id, markdownContent, metadata);
        return metadata;
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
        const candidate = source?.url
            || source?._links?.webui
            || source?._links?.self
            || source?.self
            || '';
        return String(candidate || '').trim();
    }

    async function localizeMarkdownImageLinks(markdownContent, docId, sourceUrl) {
        const markdown = String(markdownContent || '');
        const imagePattern = /!\[([^\]]*)\]\(([^)]+)\)/g;
        const matches = [...markdown.matchAll(imagePattern)];

        if (!docId || matches.length === 0) {
            return markdown;
        }

        const imagesDir = path.join(storagePath, String(docId), 'images');
        fs.mkdirSync(imagesDir, { recursive: true });

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

    async function downloadImageAsset({ source, sourceUrl, outputDir, imageIndex }) {
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
        const bytes = isBase64
            ? Buffer.from(payload, 'base64')
            : Buffer.from(decodeURIComponent(payload), 'utf8');

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

    function cleanKeywords(values) {
        if (!Array.isArray(values)) {
            return [];
        }

        return [...new Set(values
            .map(value => String(value || '').trim())
            .filter(value => value.length > 2))]
            .slice(0, 40);
    }

    async function generateAnnotationWithLlm(metadata, content) {
        const fallbackKeywords = generateKeywords(content);
        const fallbackSummary = generateSummary(content);

        if (!vscode.lm || !vscode.LanguageModelChatMessage) {
            return {
                keywords: fallbackKeywords,
                summary: fallbackSummary
            };
        }

        try {
            const models = await vscode.lm.selectChatModels({});
            const model = models?.[0];

            if (!model) {
                return {
                    keywords: fallbackKeywords,
                    summary: fallbackSummary
                };
            }

            const prompt = [
                'You are annotating a local Confluence document metadata record.',
                'Return valid JSON only with shape: {"summary":"...","keywords":["..."]}.',
                'Summary must be one concise paragraph under 220 characters.',
                'Keywords must be specific technical terms.',
                `Title: ${metadata.title || ''}`,
                `Topic: ${metadata.parent_confluence_topic || ''}`,
                `Author: ${metadata.author || ''}`,
                'Document markdown content:',
                truncate(content, 4000)
            ].join('\n');

            const response = await model.sendRequest([
                vscode.LanguageModelChatMessage.User(prompt)
            ]);

            let responseText = '';
            for await (const fragment of response.text) {
                responseText += fragment;
            }

            const parsed = extractJsonObject(responseText) || {};
            const llmKeywords = cleanKeywords(parsed.keywords);
            const llmSummary = String(parsed.summary || '').trim();

            return {
                keywords: cleanKeywords([...fallbackKeywords, ...llmKeywords]),
                summary: llmSummary || fallbackSummary
            };
        } catch {
            return {
                keywords: fallbackKeywords,
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
        const updatedMetadata = {
            ...metadata,
            keywords: annotation.keywords,
            summary: annotation.summary
        };

        writeDocumentFiles(storagePath, metadata.id, content, updatedMetadata);
        return true;
    }

    function sanitizeFileSegment(value) {
        return String(value || 'item')
            .toLowerCase()
            .replace(/[^a-z0-9-_ ]+/g, '')
            .trim()
            .replace(/\s+/g, '-')
            .slice(0, 64) || 'item';
    }

    function getWorkspaceRootPath() {
        const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
        if (!workspaceFolder) {
            throw new Error('Open a workspace folder to add prompt files.');
        }

        return workspaceFolder.uri.fsPath;
    }

    return {
        rankLocalDocuments,
        refreshDocument,
        refreshAllDocuments,
        refreshJiraIssue,
        annotateDocumentByArg,
        annotateAllDocuments,
        writeDocumentPromptFile,
        formatMetadataEntries
    };
}

module.exports = {
    createDocumentService
};