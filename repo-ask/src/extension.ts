import vscode from 'vscode';
import EventEmitter from 'events';
import { httpManager } from './mcp';
import { refreshSkipWords } from './extension/documentService/tokenization2keywords/patternMatch';
import { 
    fetchConfluencePage, 
    fetchAllConfluencePages,
    fetchAllConfluencePagesMetaOnly,
    fetchConfluencePageMeta,
    fetchConfluencePageChildren,
    fetchJiraIssue
} from './mcp';
import {
    truncate,
    htmlToMarkdown,
    jiraTextToMarkdown,
    generateSummary,
    generateSynonyms
} from './extension/documentService/html2md';
import {
    ensureStoragePath,
    ensureIndexStoragePath,
    backfillStoredMetadataSchema,
    readAllMetadata,
    readDocumentMetadata,
    readDocumentContent,
    deleteDocumentFiles,

    writeDocumentFiles
} from './storage';
import { createLanguageModelTools } from './extension/tools/vsCodeTools';
import { createDocumentService } from './extension/documentService';
import { createSidebarController } from './extension/sidebarController';
import { createRefreshCommand, createShowLogActionButtonCommand, createCheckCodeLogicCommand, createAdvancedDocSearchCommand } from './extension/commands';
import { answerGeneralPromptQuestion } from './extension/chat/generalAnswer';
import { runAdvancedDocSearch } from './extension/chat/advancedDocSearch';
import { runSkillCommand } from './extension/chat/skillChat';

const EMPTY_STORE_HINT = 'No local documents found. Use the sidebar popup to sync to Confluence Cloud.';
const TOOL_NAMES = {
    docCheck: 'repoask_doc_check'
};

function setupExtension(context: any) {
    const storagePath = ensureStoragePath(context);
    const indexStoragePath = ensureIndexStoragePath(context);
    backfillStoredMetadataSchema(storagePath);

    const refreshCancelEmitter = new EventEmitter();

    const documentService = createDocumentService({
        vscode,
        storagePath,
        indexStoragePath,
        fetchConfluencePage,
        fetchConfluencePageMeta,
        fetchAllConfluencePages,
        fetchAllConfluencePagesMetaOnly,
        fetchConfluencePageChildren,
        fetchJiraIssue,
        truncate,
        htmlToMarkdown,
        jiraTextToMarkdown,
        generateSynonyms,
        generateSummary,
        readAllMetadata,
        writeDocumentFiles,
        readDocumentContent
    });

    // Initialise skip-words from settings and keep in sync on config changes
    refreshSkipWords(vscode);
    context.subscriptions.push(
        vscode.workspace.onDidChangeConfiguration(e => {
            if (e.affectsConfiguration('repoAsk.skipWords')) {
                refreshSkipWords(vscode);
            }
        })
    );

    documentService.syncDefaultDocs(context.extensionPath);

    const sidebar = createSidebarController({
        vscode,
        context,
        storagePath,
        documentService,
        readAllMetadata,
        readDocumentContent,
        deleteDocumentFiles,
        writeDocumentFiles,
        refreshCancelEmitter,
        setRefreshCanceled: (_val: unknown) => {},
        httpManager
    });

    const lmTools = createLanguageModelTools({
        vscode,
        context,
        documentService,
        fetchConfluencePage,
        setSidebarSyncStatus: sidebar.setSidebarSyncStatus,
        setSidebarSyncError: sidebar.setSidebarSyncError,
        refreshSidebarView: sidebar.refreshSidebarView,
        upsertSidebarDocument: sidebar.upsertSidebarDocument,
        readAllMetadata: () => readAllMetadata(storagePath),
        readDocumentMetadata: (id: string) => readDocumentMetadata(storagePath, id),
        readDocumentContent: (id: string) => readDocumentContent(storagePath, id),
        truncate,
        emptyStoreHint: EMPTY_STORE_HINT,
        toolNames: TOOL_NAMES
    });

    const webviewProviderDisposable = vscode.window.registerWebviewViewProvider('repo-ask-documents', sidebar.sidebarProvider);
    const lmToolDisposables = lmTools.registerRepoAskLanguageModelTools();

    // Register refresh command with 10-second timeout
    const refreshCommandDisposable = createRefreshCommand({
        vscode,
        documentService,
        sidebar,
        storagePath,
        readAllMetadata,
        readDocumentContent,
        writeDocumentFiles,
        refreshCancelEmitter,
        setRefreshCanceled: (_val: unknown) => {},
        httpManager
    });

    let repoaskParticipant;
    if (vscode.chat && typeof vscode.chat.createChatParticipant === 'function') {
        repoaskParticipant = vscode.chat.createChatParticipant('repoask', async (request, chatContext, response) => {
            const rawPrompt = request.prompt?.trim() || '';

            if (!rawPrompt) {
                response.markdown('Ask a question.');
                return;
            }

            // Route [ADV] prefix to the advanced agentic doc-search handler
            const ADV_PREFIX = '[ADV] ';
            const isAdvanced = rawPrompt.startsWith(ADV_PREFIX);
            const prompt = isAdvanced ? rawPrompt.slice(ADV_PREFIX.length).trim() : rawPrompt;

            if (!prompt) {
                response.markdown('Ask a question.');
                return;
            }

            try {
                // Build context from any files or pinned code the user has attached
                let attachedContext = '';
                if (request.references && request.references.length > 0) {
                    const contextParts: string[] = [];
                    for (const ref of request.references) {
                        try {
                            if (ref.value instanceof vscode.Uri) {
                                const bytes = await vscode.workspace.fs.readFile(ref.value);
                                const text = new TextDecoder().decode(bytes);
                                contextParts.push(`### File: ${ref.value.fsPath}\n\`\`\`\n${truncate(text, 8000)}\n\`\`\``);
                            } else if (ref.value && typeof ref.value === 'object' && (ref.value as any).uri && (ref.value as any).range) {
                                const bytes = await vscode.workspace.fs.readFile((ref.value as any).uri);
                                const lines = new TextDecoder().decode(bytes).split('\n');
                                const start = (ref.value as any).range.start.line;
                                const end = (ref.value as any).range.end.line;
                                const snippet = lines.slice(start, end + 1).join('\n');
                                const label = (ref as any).name || 'Pinned code';
                                contextParts.push(`### ${label} (${(ref.value as any).uri.fsPath}, lines ${start + 1}-${end + 1})\n\`\`\`\n${snippet}\n\`\`\``);
                            } else if (typeof ref.value === 'string' && ref.value.trim()) {
                                contextParts.push(`### ${(ref as any).name || 'Attached context'}\n${ref.value}`);
                            }
                        } catch (_) {
                            // Skip unreadable references
                        }
                    }
                    if (contextParts.length > 0) {
                        attachedContext = contextParts.join('\n\n');
                    }
                }

                const loggedPrompts = context.globalState.get('repoAsk.loggedPrompts', []);

                if (request.command === 'production-support') {
                    await runSkillCommand(vscode, prompt, response, {
                        documentService,
                        readAllMetadata: () => readAllMetadata(storagePath),
                        readDocumentContent: (id: string) => readDocumentContent(storagePath, id),
                        storagePath,
                        extensionPath: context.extensionPath
                    }, { request });
                } else if (isAdvanced) {
                    await runAdvancedDocSearch(vscode, prompt, response, {
                        readAllMetadata: () => readAllMetadata(storagePath),
                        readDocumentContent: (id: string) => readDocumentContent(storagePath, id),
                        storagePath,
                        documentService
                    }, { request });
                } else {
                    await answerGeneralPromptQuestion(vscode, prompt, attachedContext, response, {
                        documentService,
                        readDocumentContent: (id: string) => readDocumentContent(storagePath, id),
                        chatContext
                    }, {
                        metadataList: readAllMetadata(storagePath),
                        request,
                        scenario: 'docs',
                        showLogActionButton: sidebar.showLogActionButton,
                        loggedPrompts
                    });
                }
            } catch (error) {
                response.markdown(`Unable to answer with prompt context: ${error.message}`);
            }
        });

        repoaskParticipant.iconPath = vscode.Uri.joinPath(context.extensionUri, 'media', 'icon.svg');


    }

    // Register show log action button command
    const showLogActionButtonCommandDisposable = createShowLogActionButtonCommand({
        vscode,
        context,
        sidebar,
        documentService,
        readAllMetadata,
        storagePath
    });

    // Register check code logic command
    const checkCodeLogicCommandDisposable = createCheckCodeLogicCommand({ vscode });

    // Register advanced doc search command
    const advancedDocSearchCommandDisposable = createAdvancedDocSearchCommand({ vscode });

    const baseSubscriptions = [
        webviewProviderDisposable,
        ...lmToolDisposables,
        refreshCommandDisposable,
        showLogActionButtonCommandDisposable,
        checkCodeLogicCommandDisposable,
        advancedDocSearchCommandDisposable
    ];

    if (repoaskParticipant) {
        baseSubscriptions.push(repoaskParticipant);
    }

    context.subscriptions.push(...baseSubscriptions);
}

function deactivate() {}

export { setupExtension,
    deactivate
};
