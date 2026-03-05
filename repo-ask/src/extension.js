const vscode = require('vscode');
const fs = require('fs');
const path = require('path');

const { fetchConfluencePage, fetchAllConfluencePages } = require('./confluenceApi');
const { fetchJiraIssue } = require('./jiraApi');
const {
    truncate,
    tokenize,
    htmlToMarkdown,
    generateKeywords,
    generateSummary
} = require('./textProcessing');
const {
    ensureStoragePath,
    readAllMetadata,
    readDocumentContent,
    deleteDocumentFiles,
    formatDocumentDetails,
    writeDocumentFiles
} = require('./storage');
const { findRelevantDocuments, rankDocumentsByIdf } = require('./relevance');
const { selectToolAndArg, parseRefreshArg } = require('./extension/llm');
const { createDocumentService } = require('./extension/documentService');

let storagePath;
let docsWebviewView;
let sidebarSyncStatus = '';
const EMPTY_STORE_HINT = 'No local documents found. Run `@repoask refresh` to sync to Confluence Cloud.';
const TOOL_NAMES = {
    refresh: 'repoask_refresh',
    annotate: 'repoask_annotate',
    rank: 'repoask_rank',
    check: 'repoask_check'
};

function activate(context) {
    storagePath = ensureStoragePath(context);

    const documentService = createDocumentService({
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
    });

    const checkDisposable = vscode.commands.registerCommand('repo-ask.check', async function (query) {
        const question = query || await vscode.window.showInputBox({
            prompt: 'Enter your question to check relevant documents',
            placeHolder: 'e.g., How to create a new Confluence page?'
        });

        if (!question) {
            return;
        }

        try {
            const metadataList = readAllMetadata(storagePath);
            if (metadataList.length === 0) {
                vscode.window.showInformationMessage('No local documents found. Run @repoask refresh to sync to Confluence Cloud.');
                return;
            }

            const relevantDocs = findRelevantDocuments(question, metadataList, tokenize);
            if (relevantDocs.length === 0) {
                vscode.window.showInformationMessage('No relevant documents found');
                return;
            }

            const items = relevantDocs.map(doc => ({
                label: doc.title,
                description: `Last updated: ${doc.last_updated}`,
                detail: truncate(doc.summary || 'No summary available', 120),
                doc
            }));

            const selected = await vscode.window.showQuickPick(items, {
                placeHolder: 'Select a document to view local reference content'
            });

            if (!selected) {
                return;
            }

            await vscode.commands.executeCommand('repo-ask.openDocumentDetails', selected.doc);
        } catch (error) {
            vscode.window.showErrorMessage(`Error checking documents: ${error.message}`);
        }
    });

    const openDocumentDetailsDisposable = vscode.commands.registerCommand('repo-ask.openDocumentDetails', async function (docArg) {
        const doc = typeof docArg === 'string'
            ? readAllMetadata(storagePath).find(metadata => String(metadata.id) === docArg)
            : docArg;

        if (!doc || !doc.id) {
            vscode.window.showWarningMessage('Document metadata not found. Run refresh and try again.');
            return;
        }

        const content = readDocumentContent(storagePath, doc.id) || 'No local markdown content found.';

        const document = await vscode.workspace.openTextDocument({
            language: 'plaintext',
            content: formatDocumentDetails(doc, content)
        });
        await vscode.window.showTextDocument(document, { preview: false });
    });

    const refreshDisposable = vscode.commands.registerCommand('repo-ask.refresh', async function (directArg) {
        const arg = typeof directArg === 'string' ? directArg : await vscode.window.showInputBox({
            prompt: 'Enter Confluence page id/title/link or Jira issue key, or leave empty to refresh all Confluence docs',
            placeHolder: 'e.g., 1, Technical Documentation Guide, Confluence URL, or PROJECT-1003'
        });

        try {
            setSidebarSyncStatus('');
            if (arg && arg.trim().length > 0) {
                const parsed = await parseRefreshArg(vscode, arg.trim());
                setSidebarSyncStatus('downloading from confluence cloud ...');
                if (parsed.found && parsed.source === 'regex-jira') {
                    await documentService.refreshJiraIssue(parsed.arg);
                    vscode.window.showInformationMessage(`Refreshed Jira issue for: ${parsed.arg}`);
                } else {
                    const resolvedArg = parsed.found && parsed.arg ? parsed.arg : arg.trim();
                    await documentService.refreshDocument(resolvedArg);
                    vscode.window.showInformationMessage(`Refreshed document for: ${resolvedArg}`);
                }
            } else {
                const downloadingMessage = 'downloading from confluence cloud ...';
                vscode.window.showInformationMessage(downloadingMessage);
                setSidebarSyncStatus(downloadingMessage);
                await documentService.refreshAllDocuments();
                vscode.window.showInformationMessage('Refreshed all documents');
            }

            refreshSidebarView(context.extensionUri);
        } catch (error) {
            vscode.window.showErrorMessage(`Error refreshing documents: ${error.message}`);
        } finally {
            setSidebarSyncStatus('');
            refreshSidebarView(context.extensionUri);
        }
    });

    const parseArgDisposable = vscode.commands.registerCommand('repo-ask.parseArg', async function (sourceInput) {
        return await parseRefreshArg(vscode, sourceInput);
    });

    const rankDisposable = vscode.commands.registerCommand('repo-ask.rank', async function (directQuery) {
        const query = typeof directQuery === 'string' ? directQuery : await vscode.window.showInputBox({
            prompt: 'Enter keywords to rank local documents',
            placeHolder: 'e.g., oauth token refresh'
        });

        if (!query || query.trim().length === 0) {
            return;
        }

        try {
            const rankedDocs = documentService.rankLocalDocuments(query.trim(), 10);
            if (rankedDocs.length === 0) {
                vscode.window.showInformationMessage('No matching local documents found for the query.');
                return;
            }

            const items = rankedDocs.map(doc => ({
                label: doc.title || 'Untitled',
                description: `IDF score: ${doc.score.toFixed(2)}`,
                detail: truncate(doc.summary || 'No summary available', 120),
                doc
            }));

            const selected = await vscode.window.showQuickPick(items, {
                placeHolder: 'Select a ranked document'
            });

            if (!selected) {
                return;
            }

            await vscode.commands.executeCommand('repo-ask.openDocumentDetails', selected.doc);
        } catch (error) {
            vscode.window.showErrorMessage(`Error ranking documents: ${error.message}`);
        }
    });

    const annotateDisposable = vscode.commands.registerCommand('repo-ask.annotate', async function (directArg) {
        const arg = typeof directArg === 'string' ? directArg : await vscode.window.showInputBox({
            prompt: 'Enter page id/title/link to annotate one doc, or leave empty to annotate all local docs',
            placeHolder: 'e.g., 1, Technical Documentation Guide, or a Confluence URL'
        });

        try {
            const result = arg && arg.trim().length > 0
                ? await documentService.annotateDocumentByArg(arg.trim())
                : await documentService.annotateAllDocuments();

            vscode.window.showInformationMessage(result.message);
            refreshSidebarView(context.extensionUri);
        } catch (error) {
            vscode.window.showErrorMessage(`Error annotating documents: ${error.message}`);
        }
    });

    const sidebarProvider = {
        resolveWebviewView: async (webviewView) => {
            try {
                docsWebviewView = webviewView;
                docsWebviewView.webview.options = {
                    enableScripts: true,
                    localResourceRoots: [vscode.Uri.joinPath(context.extensionUri, 'src', 'sidebar')]
                };

                docsWebviewView.webview.onDidReceiveMessage(async (message) => {
                if (message?.command === 'openDoc' && message.docId) {
                    const metadata = readAllMetadata(storagePath).find(doc => String(doc.id) === String(message.docId));
                    const content = metadata
                        ? (readDocumentContent(storagePath, metadata.id) || 'No local markdown content found.')
                        : 'No local markdown content found.';

                    docsWebviewView.webview.postMessage({
                        command: 'docDetails',
                        payload: {
                            id: message.docId,
                            content,
                            metadata: documentService.formatMetadataEntries(metadata)
                        }
                    });
                }

                if (message?.command === 'searchDocs') {
                    const query = String(message.query || '').trim();
                    const results = query.length > 0
                        ? documentService.rankLocalDocuments(query, 50)
                        : readAllMetadata(storagePath)
                            .sort((a, b) => String(b.last_updated).localeCompare(String(a.last_updated)));

                    docsWebviewView.webview.postMessage({
                        command: 'searchResults',
                        payload: results.map(doc => ({
                            id: doc.id,
                            title: doc.title || 'Untitled'
                        }))
                    });
                }

                if (message?.command === 'addToPrompts') {
                    const docId = String(message.docId || '').trim();
                    if (!docId) {
                        vscode.window.showWarningMessage('Select a document first to add it to prompts.');
                        return;
                    }

                    const metadata = readAllMetadata(storagePath).find(doc => String(doc.id) === docId);
                    if (!metadata) {
                        vscode.window.showWarningMessage('Document metadata not found. Run refresh and try again.');
                        return;
                    }

                    const content = readDocumentContent(storagePath, metadata.id);
                    if (!content || String(content).trim().length === 0) {
                        vscode.window.showWarningMessage('Local document content is empty. Refresh this doc and try again.');
                        return;
                    }

                    try {
                        const createdPath = documentService.writeDocumentPromptFile(metadata, content);
                        vscode.window.showInformationMessage(`Added prompt: ${path.basename(createdPath)}`);
                    } catch (error) {
                        vscode.window.showErrorMessage(`Failed to add prompt: ${error.message}`);
                    }
                }

                if (message?.command === 'deleteDoc') {
                    const docId = String(message.docId || '').trim();
                    const docTitle = String(message.title || docId || 'this document').trim();
                    if (!docId) {
                        vscode.window.showWarningMessage('Select a document first to delete.');
                        return;
                    }

                    try {
                        const confirmation = await vscode.window.showWarningMessage(
                            `Delete local document "${docTitle}"?`,
                            { modal: true },
                            'Delete'
                        );
                        if (confirmation !== 'Delete') {
                            return;
                        }

                        const deletion = deleteDocumentFiles(storagePath, docId);
                        docsWebviewView.webview.postMessage({
                            command: 'docDeleted',
                            payload: { id: docId }
                        });
                        refreshSidebarView(context.extensionUri);
                        if (deletion.deletedCount > 0) {
                            vscode.window.showInformationMessage(`Deleted local files (.md/.json) for: ${docId}`);
                        } else {
                            const markdownPath = path.join(storagePath, `${docId}.md`);
                            const jsonPath = path.join(storagePath, `${docId}.json`);
                            vscode.window.showWarningMessage(
                                `Delete may have failed for ${docId}. Manually delete these files if they still exist:\n- ${markdownPath}\n- ${jsonPath}`
                            );
                        }
                    } catch (error) {
                        const markdownPath = path.join(storagePath, `${docId}.md`);
                        const jsonPath = path.join(storagePath, `${docId}.json`);
                        vscode.window.showErrorMessage(
                            `Failed to delete ${docId}: ${error.message}. Please manually delete:\n- ${markdownPath}\n- ${jsonPath}`
                        );
                    }
                }
                });

                docsWebviewView.onDidChangeVisibility(async () => {
                    if (!docsWebviewView.visible) {
                        return;
                    }

                    refreshSidebarView(context.extensionUri);
                });

                refreshSidebarView(context.extensionUri);
            } catch (error) {
                const reason = error && error.message ? error.message : 'unknown sidebar initialization error';
                if (webviewView && webviewView.webview) {
                    webviewView.webview.html = getSidebarErrorHtml(`RepoAsk sidebar failed to load: ${reason}`);
                }
            }
        }
    };

    const webviewProviderDisposable = vscode.window.registerWebviewViewProvider('repo-ask-documents', sidebarProvider);
    const lmToolDisposables = registerRepoAskLanguageModelTools(context, documentService);

    let repoAskParticipant;
    if (vscode.chat && typeof vscode.chat.createChatParticipant === 'function') {
        repoAskParticipant = vscode.chat.createChatParticipant('repoask', async (request, chatContext, response) => {
            const prompt = request.prompt?.trim() || '';

        if (prompt.toLowerCase().startsWith('refresh')) {
            const refreshSource = prompt.replace(/^refresh\s*/i, '').trim();
            await handleRefreshFromSource(refreshSource || prompt, response, context.extensionUri, documentService);
            return;
        }

        if (prompt.toLowerCase().startsWith('annotate')) {
            const annotateArg = prompt.replace(/^annotate\s*/i, '').trim();
            response.markdown(`Annotating documents${annotateArg ? ` for ${annotateArg}` : ' (all local docs)'}...`);
            await vscode.commands.executeCommand('repo-ask.annotate', annotateArg);
            response.markdown('Annotation completed.');
            return;
        }

        const decision = await selectToolAndArg(vscode, prompt);

        if (decision && decision.tool === 'refresh') {
            await handleRefreshFromSource(decision.arg || prompt, response, context.extensionUri, documentService);
            return;
        }

        if (decision && decision.tool === 'annotate') {
            response.markdown(`Annotating documents for: ${decision.arg || '(all)'}...`);
            await vscode.commands.executeCommand('repo-ask.annotate', decision.arg || '');
            response.markdown('Annotation completed.');
            return;
        }

        if (decision && decision.tool === 'rank') {
            const ranked = documentService.rankLocalDocuments(decision.arg || prompt, 5);
            if (!ranked || ranked.length === 0) {
                response.markdown('No matching local documents found for the query.');
                return;
            }
            const lines = ranked.map((d, i) => `${i + 1}. **${d.title}** (score ${d.score.toFixed(2)})`);
            response.markdown(`Top ranked results:\n\n${lines.join('\n')}`);
            return;
        }

        const metadataList = readAllMetadata(storagePath);
        if (metadataList.length === 0) {
            response.markdown(EMPTY_STORE_HINT);
            return;
        }

        const relevantDocs = findRelevantDocuments(prompt, metadataList, tokenize);
        if (relevantDocs.length === 0) {
            response.markdown('No relevant documents found in local store. Try `RepoAsk: Refresh` first.');
            return;
        }

        const top = relevantDocs[0];
        const content = readDocumentContent(storagePath, top.id) || '';

        response.markdown(`Top match: **${top.title}** by ${top.author} (updated ${top.last_updated})`);
        response.markdown(`Summary: ${top.summary || 'No summary available'}`);
        response.markdown(`Keywords: ${(top.keywords || []).join(', ')}`);
        if (content) {
            response.markdown(`Reference:\n\n${truncate(content, 1200)}`);
        }
        });

        repoAskParticipant.iconPath = vscode.Uri.joinPath(context.extensionUri, 'media', 'icon.svg');
    }

    const baseSubscriptions = [
        checkDisposable,
        openDocumentDetailsDisposable,
        refreshDisposable,
        parseArgDisposable,
        rankDisposable,
        annotateDisposable,
        webviewProviderDisposable,
        ...lmToolDisposables
    ];

    if (repoAskParticipant) {
        baseSubscriptions.push(repoAskParticipant);
    }

    context.subscriptions.push(...baseSubscriptions);
}

function registerRepoAskLanguageModelTools(context, documentService) {
    if (!vscode.lm || typeof vscode.lm.registerTool !== 'function') {
        return [];
    }

    const toToolResult = (text, data) => {
        const parts = [new vscode.LanguageModelTextPart(String(text || ''))];
        if (vscode.LanguageModelDataPart && typeof vscode.LanguageModelDataPart.json === 'function' && data !== undefined) {
            parts.push(vscode.LanguageModelDataPart.json(data));
        }
        return new vscode.LanguageModelToolResult(parts);
    };

    const refreshTool = vscode.lm.registerTool(TOOL_NAMES.refresh, {
        prepareInvocation(options) {
            const arg = String(options?.input?.arg || '').trim();
            return {
                invocationMessage: arg.length > 0
                    ? `Refreshing RepoAsk document source for: ${arg}`
                    : 'Refreshing all RepoAsk Confluence documents',
                confirmationMessages: {
                    title: arg.length > 0 ? 'Refresh RepoAsk document source?' : 'Refresh all RepoAsk Confluence docs?',
                    message: arg.length > 0
                        ? `This will sync local-store from: ${arg}`
                        : 'This will sync all Confluence pages into local-store.'
                }
            };
        },
        async invoke(options) {
            const arg = String(options?.input?.arg || '').trim();
            try {
                if (!arg) {
                    setSidebarSyncStatus('downloading from confluence cloud ...');
                    await documentService.refreshAllDocuments();
                    setSidebarSyncStatus('');
                    refreshSidebarView(context.extensionUri);
                    return toToolResult('Refreshed all Confluence documents into local-store.', { refreshed: 'all' });
                }

                const parsed = await parseRefreshArg(vscode, arg);
                if (parsed.found && parsed.source === 'regex-jira') {
                    setSidebarSyncStatus('downloading from jira ...');
                    await documentService.refreshJiraIssue(parsed.arg);
                    setSidebarSyncStatus('');
                    refreshSidebarView(context.extensionUri);
                    return toToolResult(`Refreshed Jira issue for: ${parsed.arg}`, { refreshed: parsed.arg, source: 'jira' });
                }

                if (parsed.found && parsed.arg) {
                    await fetchConfluencePage(parsed.arg);
                    setSidebarSyncStatus('downloading from confluence cloud ...');
                    await documentService.refreshDocument(parsed.arg);
                    setSidebarSyncStatus('');
                    refreshSidebarView(context.extensionUri);
                    return toToolResult(`Refreshed Confluence page for: ${parsed.arg}`, { refreshed: parsed.arg, source: 'confluence' });
                }

                return toToolResult(
                    'Could not resolve a Confluence page id/title/link or Jira issue key/id/link. Provide an explicit arg, or call this tool with empty arg to refresh all Confluence docs.',
                    { refreshed: false, reason: 'unresolved-arg' }
                );
            } catch (error) {
                setSidebarSyncStatus('');
                return toToolResult(`Refresh failed: ${error.message}`, { refreshed: false, error: error.message });
            }
        }
    });

    const annotateTool = vscode.lm.registerTool(TOOL_NAMES.annotate, {
        prepareInvocation(options) {
            const arg = String(options?.input?.arg || '').trim();
            return {
                invocationMessage: arg.length > 0
                    ? `Annotating RepoAsk document: ${arg}`
                    : 'Annotating all RepoAsk local documents',
                confirmationMessages: {
                    title: arg.length > 0 ? 'Annotate selected RepoAsk document?' : 'Annotate all RepoAsk local documents?',
                    message: arg.length > 0
                        ? `This will recompute summary and keywords for: ${arg}`
                        : 'This will recompute summary and keywords for all local documents.'
                }
            };
        },
        async invoke(options) {
            const arg = String(options?.input?.arg || '').trim();
            try {
                const result = arg.length > 0
                    ? await documentService.annotateDocumentByArg(arg)
                    : await documentService.annotateAllDocuments();
                refreshSidebarView(context.extensionUri);
                return toToolResult(result.message, { annotated: true, arg: arg || '' });
            } catch (error) {
                return toToolResult(`Annotate failed: ${error.message}`, { annotated: false, error: error.message });
            }
        }
    });

    const rankTool = vscode.lm.registerTool(TOOL_NAMES.rank, {
        async invoke(options) {
            const query = String(options?.input?.query || '').trim();
            const rawLimit = Number(options?.input?.limit);
            const limit = Number.isFinite(rawLimit) && rawLimit > 0 ? Math.min(Math.floor(rawLimit), 50) : 5;

            if (!query) {
                return toToolResult('Missing required `query` input for rank tool.', { results: [] });
            }

            const ranked = documentService.rankLocalDocuments(query, limit);
            if (!ranked || ranked.length === 0) {
                return toToolResult('No matching local documents found for the query.', { results: [] });
            }

            const results = ranked.map(item => ({
                id: item.id,
                title: item.title || 'Untitled',
                score: Number(item.score?.toFixed ? item.score.toFixed(4) : item.score),
                summary: truncate(item.summary || '', 220)
            }));
            const lines = results.map((item, index) => `${index + 1}. ${item.title} (score ${item.score})`);
            return toToolResult(`Top ranked RepoAsk documents:\n${lines.join('\n')}`, { results });
        }
    });

    const checkTool = vscode.lm.registerTool(TOOL_NAMES.check, {
        async invoke(options) {
            const query = String(options?.input?.query || '').trim();
            const rawLimit = Number(options?.input?.limit);
            const limit = Number.isFinite(rawLimit) && rawLimit > 0 ? Math.min(Math.floor(rawLimit), 20) : 5;

            if (!query) {
                return toToolResult('Missing required `query` input for check tool.', { references: [] });
            }

            const metadataList = readAllMetadata(storagePath);
            if (metadataList.length === 0) {
                return toToolResult(EMPTY_STORE_HINT, { references: [] });
            }

            const relevantDocs = findRelevantDocuments(query, metadataList, tokenize).slice(0, limit);
            if (relevantDocs.length === 0) {
                return toToolResult('No relevant documents found in local store.', { references: [] });
            }

            const references = relevantDocs.map(doc => {
                const content = readDocumentContent(storagePath, doc.id) || '';
                return {
                    id: doc.id,
                    title: doc.title || 'Untitled',
                    author: doc.author || 'Unknown',
                    last_updated: doc.last_updated || '',
                    summary: truncate(doc.summary || 'No summary available', 220),
                    reference: truncate(content, 500)
                };
            });

            const lines = references.map((ref, index) => `${index + 1}. ${ref.title} (updated ${ref.last_updated || '-'})`);
            return toToolResult(`Top relevant RepoAsk references:\n${lines.join('\n')}`, { references });
        }
    });

    return [refreshTool, annotateTool, rankTool, checkTool];
}

async function handleRefreshFromSource(sourceInput, response, extensionUri, documentService) {
    const parsed = await parseRefreshArg(vscode, sourceInput);

    if (parsed.found && parsed.source === 'regex-jira') {
        response.markdown(`Refreshing Jira issue for: ${parsed.arg}...`);
        try {
            setSidebarSyncStatus('downloading from jira ...');
            await documentService.refreshJiraIssue(parsed.arg);
            response.markdown('Refresh completed for the Jira issue.');
            setSidebarSyncStatus('');
            if (extensionUri) {
                refreshSidebarView(extensionUri);
            }
        } catch (error) {
            setSidebarSyncStatus('');
            const status = error?.response?.status;
            const detail = status ? `backend returned ${status}` : (error?.message || 'backend request failed');
            response.markdown(`Refresh failed for Jira issue ${parsed.arg} (${detail}).`);
        }
        return;
    }

    if (parsed.found && parsed.arg) {
        try {
            await fetchConfluencePage(parsed.arg);
        } catch (error) {
            const status = error?.response?.status;
            const detail = status ? `backend returned ${status}` : 'backend request failed';
            response.markdown(`Could not resolve the requested document (${detail}). Do you want to download all docs instead?`);
            appendRefreshAllDocsButton(response);
            return;
        }

        response.markdown(`Refreshing document for: ${parsed.arg}...`);
        try {
            setSidebarSyncStatus('downloading from confluence cloud ...');
            await documentService.refreshDocument(parsed.arg);
            response.markdown('Refresh completed for the resolved page.');
            setSidebarSyncStatus('');
            if (extensionUri) {
                refreshSidebarView(extensionUri);
            }
        } catch (error) {
            setSidebarSyncStatus('');
            const status = error?.response?.status;
            const detail = status ? `backend returned ${status}` : (error?.message || 'backend request failed');
            response.markdown(`Refresh failed for the requested page (${detail}). Do you want to download all docs instead?`);
            appendRefreshAllDocsButton(response);
        }
        return;
    }

    response.markdown(`I couldn't find a Confluence link, page id, or exact page title in your request. Do you want to download all docs instead?`);
    appendRefreshAllDocsButton(response);
}

function appendRefreshAllDocsButton(response) {
    response.button({
        command: 'repo-ask.refresh',
        title: 'Refresh All Docs',
        arguments: ['']
    });
}

function setSidebarSyncStatus(message) {
    sidebarSyncStatus = String(message || '');
    if (!docsWebviewView) {
        return;
    }

    docsWebviewView.webview.postMessage({
        command: 'syncStatus',
        payload: sidebarSyncStatus
    });
}

function refreshSidebarView(extensionUri) {
    if (!docsWebviewView) {
        return;
    }

    try {
        docsWebviewView.webview.html = getSidebarHtml(docsWebviewView.webview, extensionUri);
    } catch (error) {
        const reason = error && error.message ? error.message : 'unknown render error';
        docsWebviewView.webview.html = getSidebarErrorHtml(`RepoAsk sidebar failed to render: ${reason}`);
    }
}

function getSidebarHtml(webview, extensionUri) {
    const htmlPath = vscode.Uri.joinPath(extensionUri, 'src', 'sidebar', 'index.html');
    const cssPath = vscode.Uri.joinPath(extensionUri, 'src', 'sidebar', 'styles.css');

    const htmlTemplate = fs.readFileSync(htmlPath.fsPath, 'utf8');
    const cssUri = webview.asWebviewUri(cssPath).toString();
    const docs = readAllMetadata(storagePath).sort((a, b) => String(b.last_updated).localeCompare(String(a.last_updated)));

    return htmlTemplate
        .replace('__CSS_URI__', cssUri)
        .replace('__DOCS_DATA__', JSON.stringify(docs))
        .replace('__SYNC_STATUS__', JSON.stringify(sidebarSyncStatus));
}

function getSidebarErrorHtml(message) {
    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RepoAsk Sidebar Error</title>
</head>
<body>
    <main style="padding: 12px; font-family: sans-serif;">
        <h2 style="margin: 0 0 8px 0;">RepoAsk Sidebar</h2>
        <p style="margin: 0;">${String(message || 'Unknown error')}</p>
    </main>
</body>
</html>`;
}

function deactivate() {}

module.exports = {
    activate,
    deactivate
};
