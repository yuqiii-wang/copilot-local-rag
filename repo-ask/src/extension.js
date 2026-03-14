const vscode = require('vscode');

const { 
    fetchConfluencePage, 
    fetchAllConfluencePages, 
    fetchConfluencePageChildren,
    fetchJiraIssue
} = require('./mcp');
const {
    truncate,
    tokenize,
    htmlToMarkdown,
    generateKeywords,
    generateSummary,
    generateExtendedKeywords
} = require('./textProcessing');
const {
    ensureStoragePath,
    ensureIndexStoragePath,
    readAllMetadata,
    readDocumentContent,
    deleteDocumentFiles,
    formatDocumentDetails,
    writeDocumentFiles
} = require('./storage');
const { findRelevantDocuments, rankDocumentsByIdf } = require('./docIndex/relevance');
const { parseRefreshArg } = require('./extension/tools/llm');
const { createDocumentService } = require('./extension/documentService');
const { createSidebarController } = require('./extension/sidebarController');
const { createLanguageModelTools } = require('./extension/tools/lmTools');
const { loadWorkspacePromptContext } = require('./extension/promptContext');
const { answerGeneralPromptQuestion } = require('./extension/chat/generalAnswer');
const { answerCodePromptQuestion } = require('./extension/chat/codeAnswer');

const EMPTY_STORE_HINT = 'No local documents found. Use the sidebar popup to sync to Confluence Cloud.';
const TOOL_NAMES = {
    rank: 'repoask_rank',
    check: 'repoask_doc_check'
};

function setupExtension(context) {
    const storagePath = ensureStoragePath(context);
    const indexStoragePath = ensureIndexStoragePath(context);

    const documentService = createDocumentService({
        vscode,
        storagePath,
        indexStoragePath,
        fetchConfluencePage,
        fetchAllConfluencePages,
        fetchConfluencePageChildren,
        fetchJiraIssue,
        truncate,
        tokenize,
        htmlToMarkdown,
        generateKeywords,
        generateExtendedKeywords,
        generateSummary,
        readAllMetadata,
        writeDocumentFiles,
        readDocumentContent,
        rankDocumentsByIdf
    });

    documentService.syncDefaultDocs(context.extensionPath);

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
        readDocumentContent: (id) => readDocumentContent(storagePath, id),
        truncate,
        emptyStoreHint: EMPTY_STORE_HINT,
        toolNames: TOOL_NAMES
    });

    const webviewProviderDisposable = vscode.window.registerWebviewViewProvider('repo-ask-documents', sidebar.sidebarProvider);
    const lmToolDisposables = lmTools.registerRepoAskLanguageModelTools();

    let repoAskDocParticipant;
    let repoAskCodeParticipant;
    if (vscode.chat && typeof vscode.chat.createChatParticipant === 'function') {
        repoAskDocParticipant = vscode.chat.createChatParticipant('repoaskDoc', async (request, chatContext, response) => {
            const prompt = request.prompt?.trim() || '';
            const workspacePromptContext = loadWorkspacePromptContext(vscode);

            if (!prompt) {
                response.markdown('Ask a question.');
                return;
            }

            try {
                const forceCheckAllDocsButton = /^check\b/i.test(prompt);
                await answerGeneralPromptQuestion(vscode, prompt, workspacePromptContext.text, response, {
                    truncate,
                    tokenize,
                    rankDocumentsByIdf
                }, {
                    metadataList: readAllMetadata(storagePath),
                    forceCheckAllDocsButton,
                    request,
                    scenario: 'docs'
                });
            } catch (error) {
                response.markdown(`Unable to answer with prompt context: ${error.message}`);
            }
        });

        repoAskDocParticipant.iconPath = vscode.Uri.joinPath(context.extensionUri, 'media', 'icon.svg');

        repoAskCodeParticipant = vscode.chat.createChatParticipant('repoaskCode', async (request, chatContext, response) => {
            const prompt = request.prompt?.trim() || '';
            const workspacePromptContext = loadWorkspacePromptContext(vscode);

            if (!prompt) {
                response.markdown('Ask a question.');
                return;
            }

            try {
                await answerCodePromptQuestion(vscode, prompt, workspacePromptContext.text, response, {
                    truncate,
                    tokenize,
                    rankDocumentsByIdf
                }, {
                    metadataList: readAllMetadata(storagePath),
                    forceCheckAllDocsButton: false,
                    request,
                    scenario: 'code'
                });
            } catch (error) {
                response.markdown(`Unable to answer with prompt context: ${error.message}`);
            }
        });

        repoAskCodeParticipant.iconPath = vscode.Uri.joinPath(context.extensionUri, 'media', 'icon.svg');
    }

    const baseSubscriptions = [
        webviewProviderDisposable,
        ...lmToolDisposables
    ];

    if (repoAskDocParticipant) {
        baseSubscriptions.push(repoAskDocParticipant);
    }
    if (repoAskCodeParticipant) {
        baseSubscriptions.push(repoAskCodeParticipant);
    }

    context.subscriptions.push(...baseSubscriptions);
}

function deactivate() {}

module.exports = {
    setupExtension,
    deactivate
};
