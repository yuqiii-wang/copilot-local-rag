const fs = require('fs');
const path = require('path');
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

    async function refreshDocument(pageArg) {
        const page = await fetchConfluencePage(pageArg);
        await processDocument(page);
    }

    async function refreshAllDocuments() {
        const pages = await fetchAllConfluencePages();
        for (const page of pages) {
            await processDocument(page);
        }
    }

    async function refreshJiraIssue(issueArg) {
        if (typeof fetchJiraIssue !== 'function') {
            throw new Error('Jira integration is not configured.');
        }

        const issue = await fetchJiraIssue(issueArg);
        await processJiraIssue(issue);
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
        const html = getPageHtml(page);
        const markdownContent = htmlToMarkdown(html);
        const incomingKeywords = cleanKeywords(Array.isArray(page.keywords) ? page.keywords : []);
        const baseMetadata = {
            id: page.id,
            title: page.title,
            author: page.author || 'Unknown',
            last_updated: page.last_updated || new Date().toISOString().slice(0, 10),
            parent_confluence_topic: page.parent_confluence_topic || page.space || 'General',
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
        const summary = String(fields?.summary || issue?.summary || '').trim();
        const description = String(fields?.description || issue?.description || '').trim();
        const issueKey = String(issue?.key || '').trim();
        const title = issueKey && summary ? `${issueKey}: ${summary}` : (issueKey || summary || `Issue ${issue?.id || ''}`.trim());
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

        const markdownContent = contentSections.join('\n');
        const incomingKeywords = cleanKeywords(Array.isArray(fields?.labels) ? fields.labels : []);
        const baseMetadata = {
            id: issue?.id,
            title,
            author: reporter,
            last_updated: String(fields?.updated || new Date().toISOString().slice(0, 10)).slice(0, 10),
            parent_confluence_topic: `Jira ${projectKey}`,
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