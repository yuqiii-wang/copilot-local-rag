const vscode = require('vscode');

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
const { parseRefreshArg } = require('./extension/llm');
const { createDocumentService } = require('./extension/documentService');
const { createSidebarController } = require('./extension/sidebarController');
const { createLanguageModelTools } = require('./extension/lmTools');
const { loadWorkspacePromptContext } = require('./extension/promptContext');

const EMPTY_STORE_HINT = 'No local documents found. Run `@repoask refresh` to sync to Confluence Cloud.';
const TOOL_NAMES = {
    refresh: 'repoask_refresh',
    annotate: 'repoask_annotate',
    rank: 'repoask_rank',
    check: 'repoask_check'
};

const LLM_RESPONSE_TIMEOUT_MS = 30000;

async function withTimeout(promise, timeoutMs, timeoutValue = null) {
    let timeoutId;
    const timeoutPromise = new Promise((resolve) => {
        timeoutId = setTimeout(() => resolve(timeoutValue), timeoutMs);
    });

    try {
        return await Promise.race([promise, timeoutPromise]);
    } finally {
        clearTimeout(timeoutId);
    }
}

async function answerGeneralPromptQuestion(vscodeApi, prompt, workspacePromptContext, response) {
    if (!vscodeApi.lm || !vscodeApi.LanguageModelChatMessage) {
        response.markdown('No language model is available in this VS Code session.');
        return;
    }

    const models = await withTimeout(vscodeApi.lm.selectChatModels({}), LLM_RESPONSE_TIMEOUT_MS, []);
    const model = models?.[0];
    if (!model) {
        response.markdown('No language model is available in this VS Code session.');
        return;
    }

    const contextText = String(workspacePromptContext || '').trim();
    const instruction = [
        'You are RepoAsk. Answer the user question using the provided markdown prompt context when possible.',
        'If the answer is not present in the provided context, clearly say so and provide the best possible guidance.',
        contextText
            ? `Workspace markdown prompt context:\n${contextText}`
            : 'Workspace markdown prompt context: (none found under .github/prompts)',
        `User question: ${prompt}`
    ].join('\n\n');

    const modelResponse = await withTimeout(model.sendRequest([
        vscodeApi.LanguageModelChatMessage.User(instruction)
    ]), LLM_RESPONSE_TIMEOUT_MS, null);

    if (!modelResponse || !modelResponse.text) {
        response.markdown('No answer returned by the language model.');
        return;
    }

    let output = '';
    for await (const fragment of modelResponse.text) {
        output += fragment;
    }

    response.markdown(output.trim() || 'No answer returned by the language model.');
}

function isRefreshPrompt(prompt) {
    const lowered = String(prompt || '').toLowerCase();
    return lowered.includes('refresh')
        || lowered.includes('sync')
        || lowered.includes('download')
        || lowered.includes('fetch')
        || lowered.includes('pull')
        || lowered.includes('import')
        || lowered.includes('update')
        || lowered.includes('confluence')
        || lowered.includes('jira')
        || /https?:\/\//i.test(prompt)
        || /(?:pageid=|\b)\d{1,8}(?:\b|$)/i.test(prompt)
        || /[A-Z][A-Z0-9_]+-\d+/i.test(prompt);
}

function setupExtension(context) {
    const storagePath = ensureStoragePath(context);

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

    const sidebar = createSidebarController({
        vscode,
        context,
        storagePath,
        documentService,
        readAllMetadata,
        readDocumentContent,
        deleteDocumentFiles
    });

    const lmTools = createLanguageModelTools({
        vscode,
        context,
        documentService,
        parseRefreshArg,
        fetchConfluencePage,
        setSidebarSyncStatus: sidebar.setSidebarSyncStatus,
        refreshSidebarView: sidebar.refreshSidebarView,
        upsertSidebarDocument: sidebar.upsertSidebarDocument,
        readAllMetadata: () => readAllMetadata(storagePath),
        readDocumentContent: (docId) => readDocumentContent(storagePath, docId),
        findRelevantDocuments,
        tokenize,
        truncate,
        emptyStoreHint: EMPTY_STORE_HINT,
        toolNames: TOOL_NAMES
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
            sidebar.setSidebarSyncStatus('');
            const formatRefreshStatus = (sourceLabel, progress = {}) => {
                const index = Number(progress?.index);
                const total = Number(progress?.total);
                const hasFraction = Number.isFinite(index) && Number.isFinite(total) && total > 0;
                const progressSuffix = hasFraction ? ` (${index}/${total})` : '';
                return `downloading from ${sourceLabel} ...${progressSuffix}`;
            };

            const createRefreshOptions = (sourceLabel) => ({
                onDocumentProcessed: ({ metadata, index, total }) => {
                    sidebar.upsertSidebarDocument(metadata);
                    sidebar.setSidebarSyncStatus(formatRefreshStatus(sourceLabel, { index, total }));
                }
            });

            if (arg && arg.trim().length > 0) {
                const parsed = await parseRefreshArg(vscode, arg.trim());
                sidebar.setSidebarSyncStatus('downloading from confluence/jira cloud ...');
                if (parsed.found && parsed.source === 'regex-jira') {
                    await documentService.refreshJiraIssue(parsed.arg, createRefreshOptions('jira'));
                    vscode.window.showInformationMessage(`Refreshed Jira issue for: ${parsed.arg}`);
                } else {
                    const resolvedArg = parsed.found && parsed.arg ? parsed.arg : arg.trim();
                    await documentService.refreshDocument(resolvedArg, createRefreshOptions('confluence cloud'));
                    vscode.window.showInformationMessage(`Refreshed document for: ${resolvedArg}`);
                }
            } else {
                const downloadingMessage = 'downloading from confluence/jira cloud ...';
                vscode.window.showInformationMessage(downloadingMessage);
                sidebar.setSidebarSyncStatus(downloadingMessage);
                await documentService.refreshAllDocuments(createRefreshOptions('confluence cloud'));
                vscode.window.showInformationMessage('Refreshed all documents');
            }

            sidebar.refreshSidebarView();
        } catch (error) {
            vscode.window.showErrorMessage(`Error refreshing documents: ${error.message}`);
        } finally {
            sidebar.setSidebarSyncStatus('');
            sidebar.refreshSidebarView();
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
            sidebar.refreshSidebarView();
        } catch (error) {
            vscode.window.showErrorMessage(`Error annotating documents: ${error.message}`);
        }
    });

    const webviewProviderDisposable = vscode.window.registerWebviewViewProvider('repo-ask-documents', sidebar.sidebarProvider);
    const lmToolDisposables = lmTools.registerRepoAskLanguageModelTools();

    let repoAskParticipant;
    if (vscode.chat && typeof vscode.chat.createChatParticipant === 'function') {
        repoAskParticipant = vscode.chat.createChatParticipant('repoask', async (request, chatContext, response) => {
            const prompt = request.prompt?.trim() || '';
            const workspacePromptContext = loadWorkspacePromptContext(vscode);
            const llmOptions = {
                workspacePromptContext: workspacePromptContext.text
            };

            if (!prompt) {
                response.markdown('Ask a question, or use `refresh` to sync content.');
                return;
            }

            if (prompt.toLowerCase().startsWith('refresh')) {
                const refreshSource = prompt.replace(/^refresh\s*/i, '').trim();
                await lmTools.handleRefreshFromSource(refreshSource || prompt, response, llmOptions);
                return;
            }

            if (prompt.toLowerCase().startsWith('annotate')) {
                const annotateArg = prompt.replace(/^annotate\s*/i, '').trim();
                response.markdown(`Annotating documents${annotateArg ? ` for ${annotateArg}` : ' (all local docs)'}...`);
                await vscode.commands.executeCommand('repo-ask.annotate', annotateArg);
                response.markdown('Annotation completed.');
                return;
            }

            if (isRefreshPrompt(prompt)) {
                await lmTools.handleRefreshFromSource(prompt, response, llmOptions);
                return;
            }

            try {
                await answerGeneralPromptQuestion(vscode, prompt, workspacePromptContext.text, response);
            } catch (error) {
                response.markdown(`Unable to answer with prompt context: ${error.message}`);
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

function deactivate() {}

module.exports = {
    setupExtension,
    deactivate
};
