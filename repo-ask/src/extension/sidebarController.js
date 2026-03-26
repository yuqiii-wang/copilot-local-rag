const fs = require('fs');
const path = require('path');
const vscode = require('vscode');
const { updateConfluencePage } = require('../mcp/confluenceApi');
const { mapFeedbackError } = require('./errMap');
const { createOpenDocCommand, createMetadataCommands, createSearchCommand, createPromptsCommand, createSkillsCommand, createDeleteCommand, createResetCommand } = require('./commands');
const { generateKnowledgeGraph, extractMermaidKeywords } = require('./tools/llm');


function createSidebarController(deps) {
    const {
        vscode,
        context,
        storagePath,
        documentService,
        readAllMetadata,
        readDocumentContent,
        deleteDocumentFiles,
        writeDocumentFiles,
        refreshCancelEmitter,
        setRefreshCanceled,
        httpManager
    } = deps;

    // Create command instances
    const openDoc = createOpenDocCommand(deps);
    const { generateMetadata, saveMetadata } = createMetadataCommands(deps);
    const searchDocs = createSearchCommand(deps);
    const addToPrompts = createPromptsCommand(deps);
    const addToSkills = createSkillsCommand(deps);
    const deleteDoc = createDeleteCommand(deps);
    const resetToDefaultDocs = createResetCommand(deps);

    let docsWebviewView;
    let sidebarSyncStatus = '';
    let sidebarSyncError = '';
    let sidebarSyncSuccess = '';


    const sidebarProvider = {
        resolveWebviewView: async (webviewView) => {
            try {
                docsWebviewView = webviewView;
                docsWebviewView.webview.options = {
                    enableScripts: true,
                    localResourceRoots: [
                        vscode.Uri.joinPath(context.extensionUri, 'src', 'sidebar'),
                        vscode.Uri.file(storagePath)
                    ]
                };

                docsWebviewView.webview.onDidReceiveMessage(async (message) => {
                    if (message?.command === 'clearSyncError') {
                        sidebarSyncError = '';
                    }

                    if (message?.command === 'clearSyncSuccess') {
                        sidebarSyncSuccess = '';
                    }

                    if (message?.command === 'refreshDocs') {
                        const { isAll, isFeedback, arg, fullIndexRefresh } = message;
                        if (isFeedback) {
                            vscode.commands.executeCommand('repo-ask.refresh', { type: 'feedback', fullIndexRefresh });
                        } else if (isAll) {
                            if (arg) {
                                vscode.commands.executeCommand('repo-ask.refresh', { type: 'recursive', arg: String(arg).trim(), fullIndexRefresh });
                            } else {
                                vscode.commands.executeCommand('repo-ask.refresh', { type: 'all', fullIndexRefresh });
                            }
                        } else if (arg) {
                            vscode.commands.executeCommand('repo-ask.refresh', { type: 'single', arg: String(arg).trim(), fullIndexRefresh });
                        }
                    }

                    if (message?.command === 'openDoc' && message.docId) {
                        await openDoc(message, docsWebviewView);
                    }

                    if (message?.command === 'generateMetadata' && message.docId) {
                        await generateMetadata(message, docsWebviewView, upsertSidebarDocument);
                    }

                    if (message?.command === 'saveMetadata' && message.docId) {
                        await saveMetadata(message, upsertSidebarDocument);
                    }

                    if (message?.command === 'searchDocs') {
                        await searchDocs(message, docsWebviewView);
                    }

                    if (message?.command === 'addToPrompts') {
                        await addToPrompts(message, docsWebviewView);
                    }

                    if (message?.command === 'addToSkills') {
                        await addToSkills(message, docsWebviewView);
                    }

                    if (message?.command === 'deleteDoc') {
                        await deleteDoc(message, docsWebviewView, refreshSidebarView);
                    }

                    if (message?.command === 'resetToDefaultDocs') {
                        await resetToDefaultDocs(message, docsWebviewView, refreshSidebarView, context);
                    }

                    if (message?.command === 'submitFeedback' && message.feedbackPayload) {
                        await handleSubmitFeedback(message.feedbackPayload);
                    }

                    if (message?.command === 'generateSummary' && message.conversationSummary) {
                        await handleGenerateSummary(message);
                    }

                    if (message?.command === 'generateKnowledgeGraph') {
                        await handleGenerateKnowledgeGraph(message);
                    }

                    if (message?.command === 'viewKnowledgeGraph' && message.mermaidText) {
                        const encoded = Buffer.from(message.mermaidText, 'utf8').toString('base64');
                        vscode.env.openExternal(vscode.Uri.parse(`https://mermaid.live/edit#base64:${encoded}`));
                    }

                    if (message?.command === 'getDocumentByID' && message.id) {
                        // Find document by ID and return its source URL
                        const allMetadata = readAllMetadata(storagePath);
                        const document = allMetadata.find(doc => String(doc.id) === String(message.id));
                        if (document) {
                            docsWebviewView.webview.postMessage({ 
                                command: 'documentFound', 
                                document 
                            });
                        }
                    }

                    if (message?.command === 'openDocStore') {
                        // Open the document store directory in file explorer
                        try {
                            const fs = require('fs');
                            const path = require('path');
                            const docStorePath = path.join(storagePath, 'documents');
                            
                            // Ensure the directory exists
                            if (!fs.existsSync(docStorePath)) {
                                fs.mkdirSync(docStorePath, { recursive: true });
                            }
                            
                            // Open the directory
                            vscode.commands.executeCommand('revealFileInOS', vscode.Uri.file(docStorePath));
                        } catch (error) {
                            console.error('Error opening document store:', error);
                            vscode.window.showErrorMessage('Failed to open document store directory');
                        }
                    }

                    if (message?.command === 'cancelRefresh') {
                        if (refreshCancelEmitter) {
                            refreshCancelEmitter.emit('cancel');
                        }
                        if (setRefreshCanceled) {
                            setRefreshCanceled(true);
                        }
                        if (httpManager) {
                            httpManager.cancelAll();
                        }
                    }
                });

                docsWebviewView.onDidChangeVisibility(async () => {
                    if (!docsWebviewView.visible) {
                        return;
                    }

                    refreshSidebarView();
                });

                refreshSidebarView();
            } catch (error) {
                const reason = error && error.message ? error.message : 'unknown sidebar initialization error';
                if (webviewView && webviewView.webview) {
                    webviewView.webview.html = getSidebarErrorHtml(`RepoAsk sidebar failed to load: ${reason}`);
                }
            }
        }
    };

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

    function setSidebarSyncError(message) {
        sidebarSyncError = String(message || '');
        if (!docsWebviewView) {
            return;
        }

        docsWebviewView.webview.postMessage({
            command: 'syncError',
            payload: sidebarSyncError
        });
    }



    function refreshSidebarView() {
        if (!docsWebviewView) {
            return;
        }

        try {
            docsWebviewView.webview.html = getSidebarHtml(docsWebviewView.webview);
        } catch (error) {
            const reason = error && error.message ? error.message : 'unknown render error';
            docsWebviewView.webview.html = getSidebarErrorHtml(`RepoAsk sidebar failed to render: ${reason}`);
        }
    }

    function upsertSidebarDocument(metadata) {
        if (!docsWebviewView || !metadata || !metadata.id) {
            return;
        }

        docsWebviewView.webview.postMessage({
            command: 'docUpserted',
            payload: {
                id: metadata.id,
                title: metadata.title || 'Untitled',
                last_updated: metadata.last_updated || ''
            }
        });
    }

    function revealDocumentInSidebar(docId) {
        const id = String(docId || '').trim();
        if (!id || !docsWebviewView) {
            return false;
        }

        const metadata = readAllMetadata(storagePath).find(doc => String(doc.id) === id);
        if (!metadata) {
            return false;
        }

        const rawContent = readDocumentContent(storagePath, metadata.id) || 'No local markdown content found.';
        const content = rewriteMarkdownImageLinksForWebview(rawContent, metadata.id, docsWebviewView.webview);
        const contentHtml = renderMarkdownForWebview(content);

        if (typeof docsWebviewView.show === 'function') {
            docsWebviewView.show(true);
        }

        docsWebviewView.webview.postMessage({
            command: 'selectDoc',
            payload: {
                id,
                content,
                contentHtml,
                metadata
            }
        });

        return true;
    }

    function getSidebarHtml(webview) {
        const htmlPath = vscode.Uri.joinPath(context.extensionUri, 'src', 'sidebar', 'index.html');
        const cssPath = vscode.Uri.joinPath(context.extensionUri, 'src', 'sidebar', 'styles.css');
        const popupPath = vscode.Uri.joinPath(context.extensionUri, 'src', 'sidebar', 'refreshPopup.html');
        const metadataHtmlPath = vscode.Uri.joinPath(context.extensionUri, 'src', 'sidebar', 'metadata.html');
        const docStoreHtmlPath = vscode.Uri.joinPath(context.extensionUri, 'src', 'sidebar', 'docStore.html');
        const feedbackHtmlPath = vscode.Uri.joinPath(context.extensionUri, 'src', 'sidebar', 'feedback.html');
        const metadataJsPath = vscode.Uri.joinPath(context.extensionUri, 'src', 'sidebar', 'metadata.js');
        const docStoreJsPath = vscode.Uri.joinPath(context.extensionUri, 'src', 'sidebar', 'docStore.js');
        const feedbackJsPath = vscode.Uri.joinPath(context.extensionUri, 'src', 'sidebar', 'feedback.js');

        const htmlTemplate = fs.readFileSync(htmlPath.fsPath, 'utf8');
        const popupHtml = fs.existsSync(popupPath.fsPath) ? fs.readFileSync(popupPath.fsPath, 'utf8') : '';
        const metadataHtml = fs.existsSync(metadataHtmlPath.fsPath) ? fs.readFileSync(metadataHtmlPath.fsPath, 'utf8') : '';
        const docStoreHtml = fs.existsSync(docStoreHtmlPath.fsPath) ? fs.readFileSync(docStoreHtmlPath.fsPath, 'utf8') : '';
        const feedbackHtml = fs.existsSync(feedbackHtmlPath.fsPath) ? fs.readFileSync(feedbackHtmlPath.fsPath, 'utf8') : '';

        const cssUri = webview.asWebviewUri(cssPath).toString();
        const metadataJsUri = webview.asWebviewUri(metadataJsPath).toString();
        const docStoreJsUri = webview.asWebviewUri(docStoreJsPath).toString();
        const feedbackJsUri = webview.asWebviewUri(feedbackJsPath).toString();

        const docs = readAllMetadata(storagePath).sort((a, b) => String(b.last_updated).localeCompare(String(a.last_updated)));

        return htmlTemplate
            .replace('__CSS_URI__', cssUri)
            .replace('__METADATA_JS_URI__', metadataJsUri)
            .replace('__DOC_STORE_JS_URI__', docStoreJsUri)
            .replace('__FEEDBACK_JS_URI__', feedbackJsUri)
            .replace('__DOCS_DATA__', JSON.stringify(docs))
            .replace('__SYNC_STATUS__', JSON.stringify(sidebarSyncStatus))
            .replace('__SYNC_ERROR__', JSON.stringify(sidebarSyncError))
            .replace('__SYNC_SUCCESS__', JSON.stringify(sidebarSyncSuccess))
            .replace('__METADATA_HTML__', metadataHtml)
            .replace('__DOC_STORE_HTML__', docStoreHtml)
            .replace('__FEEDBACK_HTML__', feedbackHtml)
            .replace('__REFRESH_POPUP__', popupHtml);
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

    async function handleSubmitFeedback(feedbackPayload) {
        try {
            console.log('[handleSubmitFeedback] Received feedbackPayload:', feedbackPayload);
            const configuration = vscode.workspace.getConfiguration('repoAsk');
            const confluenceUrl = configuration.get('logActionConfluenceUrl');

            if (!confluenceUrl) {
                const errMsg = 'Please set the logActionConfluenceUrl setting in RepoAsk configuration.';
                vscode.window.showErrorMessage(errMsg);
                console.error('[handleSubmitFeedback] ' + errMsg);
                if (docsWebviewView) {
                    docsWebviewView.webview.postMessage({
                        command: 'feedbackSubmitted',
                        success: false,
                        error: errMsg
                    });
                }
                return;
            }

            // Extract data points from feedback submission
            const { sourceQuery, confluencePageId, jiraId, confluenceLink, secondaryUrls } = feedbackPayload;

            // Implement mapping system: associate Source Query with reference queries
            let referenceQueries = [];
            if (sourceQuery && (confluencePageId || jiraId || confluenceLink)) {
                try {
                    const allMetadata = readAllMetadata(storagePath);
                    let targetDocument = null;

                    if (confluencePageId) {
                        targetDocument = allMetadata.find(doc => String(doc.id) === String(confluencePageId));
                    }
                    if (!targetDocument && jiraId) {
                        targetDocument = allMetadata.find(doc => String(doc.id) === String(jiraId));
                    }
                    if (!targetDocument && confluenceLink) {
                        try {
                            const url = new URL(confluenceLink);
                            const pageIdMatch = url.search.match(/pageId=(\d+)/);
                            if (pageIdMatch && pageIdMatch[1]) {
                                targetDocument = allMetadata.find(doc => String(doc.id) === String(pageIdMatch[1]));
                            }
                        } catch (urlError) {
                            const { errorMessage, detailedError } = mapFeedbackError(urlError);
                            vscode.window.showErrorMessage(errorMessage);
                            console.error('[handleSubmitFeedback] Error parsing Confluence link:', detailedError);
                        }
                    }

                    if (targetDocument) {
                        const currentReferenceQueries = Array.isArray(targetDocument.referencedQueries) ? targetDocument.referencedQueries : [];
                        if (!currentReferenceQueries.includes(sourceQuery)) {
                            const updatedReferenceQueries = [...currentReferenceQueries, sourceQuery];
                            const updatedMetadata = {
                                ...targetDocument,
                                referencedQueries: updatedReferenceQueries
                            };
                            try {
                                const content = readDocumentContent(storagePath, targetDocument.id);
                                if (content) {
                                    writeDocumentFiles(storagePath, targetDocument.id, content, updatedMetadata);
                                    vscode.window.showInformationMessage(`Added reference query to document: ${targetDocument.title || targetDocument.id}`);
                                    console.log(`[handleSubmitFeedback] Added reference query to document: ${targetDocument.id}`);
                                } else {
                                    vscode.window.showErrorMessage('No content found for document: ' + targetDocument.id);
                                    console.error('[handleSubmitFeedback] No content found for document:', targetDocument.id);
                                }
                            } catch (writeError) {
                                const { errorMessage, detailedError } = mapFeedbackError(writeError);
                                vscode.window.showErrorMessage(errorMessage);
                                console.error('[handleSubmitFeedback] Error writing document metadata:', detailedError);
                            }
                        } else {
                            console.log('[handleSubmitFeedback] Source query already in reference queries for document:', targetDocument.id);
                        }
                        referenceQueries = Array.isArray(targetDocument.referencedQueries) ? targetDocument.referencedQueries : [];
                    } else {
                        vscode.window.showErrorMessage('No document found for mapping source query: ' + sourceQuery);
                        console.log('[handleSubmitFeedback] No document found for mapping source query:', sourceQuery);
                    }
                } catch (mappingError) {
                    const { errorMessage, detailedError } = mapFeedbackError(mappingError);
                    vscode.window.showErrorMessage(errorMessage);
                    console.error('[handleSubmitFeedback] Error mapping source query to reference queries:', detailedError);
                }
            }

            // Use the knowledge_graph already generated and shown in the feedback UI.
            // Do NOT re-generate on submit — the user has already previewed it.
            const knowledgeGraphFromUi = typeof feedbackPayload.knowledge_graph === 'string'
                ? feedbackPayload.knowledge_graph.trim()
                : '';

            // Save the mermaid knowledge graph to the primary document metadata
            if (knowledgeGraphFromUi && (confluencePageId || jiraId || confluenceLink)) {
                try {
                    const allMetadata = readAllMetadata(storagePath);
                    let primaryDoc = null;
                    if (confluencePageId) primaryDoc = allMetadata.find(d => String(d.id) === String(confluencePageId));
                    if (!primaryDoc && jiraId) primaryDoc = allMetadata.find(d => String(d.id) === String(jiraId));
                    if (!primaryDoc && confluenceLink) {
                        try {
                            const linkUrl = new URL(confluenceLink);
                            const linkPageIdMatch = linkUrl.search.match(/pageId=(\d+)/);
                            if (linkPageIdMatch && linkPageIdMatch[1]) {
                                primaryDoc = allMetadata.find(d => String(d.id) === String(linkPageIdMatch[1]));
                            }
                        } catch (_) {}
                    }
                    if (primaryDoc) {
                        const content = readDocumentContent(storagePath, primaryDoc.id);
                        if (content) {
                            const mermaidKeywords = extractMermaidKeywords(knowledgeGraphFromUi);
                            const existingKeywords = Array.isArray(primaryDoc.keywords) ? primaryDoc.keywords : [];
                            const mergedKeywords = [...existingKeywords];
                            for (const kw of mermaidKeywords) {
                                if (!mergedKeywords.includes(kw)) mergedKeywords.push(kw);
                            }
                            const updatedMeta = { ...primaryDoc, knowledgeGraph: knowledgeGraphFromUi, keywords: mergedKeywords };
                            writeDocumentFiles(storagePath, primaryDoc.id, content, updatedMeta);
                            console.log('[handleSubmitFeedback] Saved knowledge graph to primary doc:', primaryDoc.id);
                        }
                    }
                } catch (saveError) {
                    console.error('[handleSubmitFeedback] Error saving knowledge graph to primary doc:', saveError);
                }
            }

            await updateConfluencePage(confluenceUrl, feedbackPayload);
            vscode.window.showInformationMessage('Feedback submitted successfully!');
            console.log('[handleSubmitFeedback] Feedback submitted successfully!');
            if (docsWebviewView) {
                docsWebviewView.webview.postMessage({
                    command: 'feedbackSubmitted',
                    success: true
                });
            }
        } catch (error) {
            console.error('[handleSubmitFeedback] Error submitting feedback:', error);
            const { errorMessage, detailedError } = mapFeedbackError(error);
            vscode.window.showErrorMessage(errorMessage);
            console.error('[handleSubmitFeedback] Detailed error:', detailedError);
            if (docsWebviewView) {
                docsWebviewView.webview.postMessage({
                    command: 'feedbackSubmitted',
                    success: false,
                    error: errorMessage,
                    detailedError: detailedError
                });
            }
        }
    }

    async function handleGenerateSummary(messageOrText) {
        const conversationSummary = typeof messageOrText === 'string'
            ? messageOrText
            : String(messageOrText?.conversationSummary || '');
        const formContext = typeof messageOrText === 'object' && messageOrText !== null ? messageOrText : {};
        try {
            let summary = '';
            const inputText = String(conversationSummary || '').trim();
            if (!inputText) {
                if (docsWebviewView) {
                    docsWebviewView.webview.postMessage({
                        command: 'populateSummary',
                        summary: ''
                    });
                }
                return;
            }
            
            // Try to use VS Code's built-in LLM to rewrite the provided conversation summary.
            if (vscode.lm && vscode.LanguageModelChatMessage) {
                try {
                    const shared = require('./chat/shared');
                    const model = await shared.selectDefaultChatModel(vscode);
                    if (model) {
                        const instruction = [
                            'You are a helpful assistant that rewrites conversation summaries.',
                            'Rewrite the following conversation summary into a clear, concise, and complete summary.',
                            'Keep all key details, decisions, and action items. Remove filler, redundancy, and unnecessary preamble.',
                            'Aim for the shortest version that preserves all essential information. Return only the rewritten summary text.',
                            '',
                            'Conversation Summary:',
                            inputText
                        ].join('\n');

                        const response = await model.sendRequest([
                            vscode.LanguageModelChatMessage.User(instruction)
                        ]);

                        let responseText = '';
                        if (response) {
                            if (response.stream) {
                                for await (const chunk of response.stream) {
                                    if (chunk instanceof vscode.LanguageModelTextPart) {
                                        responseText += chunk.value;
                                    }
                                }
                            } else if (response.text) {
                                for await (const fragment of response.text) {
                                    responseText += fragment;
                                }
                            }
                        }
                        summary = responseText.trim();
                    }
                } catch (llmError) {
                    console.error('LLM error:', llmError);
                    // Fallback to original content if LLM fails
                    summary = inputText;
                }
            } else {
                // Fallback to original content if LLM not available
                summary = inputText;
            }

            if (!String(summary || '').trim()) {
                summary = inputText;
            }
            
            if (docsWebviewView) {
                docsWebviewView.webview.postMessage({ 
                    command: 'populateSummary', 
                    summary 
                });
            }

            // After summary is sent, also build knowledge graph if form context includes a document ID
            if (formContext.confluencePageId || formContext.jiraId || formContext.confluenceLink) {
                try {
                    const mermaid = await buildKnowledgeGraphForForm(formContext);
                    if (docsWebviewView) {
                        docsWebviewView.webview.postMessage({ command: 'populateKnowledgeGraph', mermaid: mermaid || '' });
                    }
                } catch (kgErr) {
                    console.error('[handleGenerateSummary] KG generation error:', kgErr);
                    if (docsWebviewView) {
                        docsWebviewView.webview.postMessage({ command: 'populateKnowledgeGraph', mermaid: '' });
                    }
                }
            }
        } catch (error) {
            console.error('Error generating summary:', error);
            vscode.window.showErrorMessage('Failed to generate summary. Please try again.');
            
            // Ensure buttons are re-enabled even on error
            if (docsWebviewView) {
                docsWebviewView.webview.postMessage({ 
                    command: 'populateSummary', 
                    summary: conversationSummary
                });
                docsWebviewView.webview.postMessage({ command: 'populateKnowledgeGraph', mermaid: '' });
            }
        }
    }

    async function handleGenerateKnowledgeGraph(formContext) {
        try {
            const mermaid = await buildKnowledgeGraphForForm(formContext);
            if (docsWebviewView) {
                docsWebviewView.webview.postMessage({ command: 'populateKnowledgeGraph', mermaid: mermaid || '' });
            }
        } catch (error) {
            console.error('[handleGenerateKnowledgeGraph] Error:', error);
            if (docsWebviewView) {
                docsWebviewView.webview.postMessage({ command: 'populateKnowledgeGraph', mermaid: '' });
            }
        }
    }

    async function buildKnowledgeGraphForForm({ sourceQuery, confluencePageId, jiraId, confluenceLink, secondaryUrls }) {
        const urls = Array.isArray(secondaryUrls) ? secondaryUrls.filter(u => u && u !== 'none') : [];
        const allMetadata = readAllMetadata(storagePath);

        // Collect secondary URL content
        const contentMap = {};
        for (const url of urls) {
            let docContent = '';
            try {
                const urlObj = new URL(url);
                const pageIdMatch = urlObj.search.match(/pageId=(\d+)/);
                if (pageIdMatch && pageIdMatch[1]) {
                    const doc = allMetadata.find(d => String(d.id) === String(pageIdMatch[1]));
                    if (doc) docContent = readDocumentContent(storagePath, doc.id) || '';
                }
            } catch (_) {
                const doc = allMetadata.find(d => String(d.id) === String(url));
                if (doc) docContent = readDocumentContent(storagePath, doc.id) || '';
            }
            contentMap[url] = docContent;
        }

        // Resolve primary document
        let primaryDoc = null;
        if (confluencePageId) primaryDoc = allMetadata.find(d => String(d.id) === String(confluencePageId));
        if (!primaryDoc && jiraId) primaryDoc = allMetadata.find(d => String(d.id) === String(jiraId));
        if (!primaryDoc && confluenceLink) {
            try {
                const linkUrl = new URL(confluenceLink);
                const m = linkUrl.search.match(/pageId=(\d+)/);
                if (m && m[1]) primaryDoc = allMetadata.find(d => String(d.id) === String(m[1]));
            } catch (_) {}
        }

        let primaryContent = '';
        let existingKnowledgeGraph = '';
        let referenceQueries = [];
        if (primaryDoc) {
            primaryContent = readDocumentContent(storagePath, primaryDoc.id) || '';
            existingKnowledgeGraph = String(primaryDoc.knowledgeGraph || '').trim();
            referenceQueries = Array.isArray(primaryDoc.referencedQueries) ? primaryDoc.referencedQueries : [];
        }
        if (sourceQuery && !referenceQueries.includes(sourceQuery)) {
            referenceQueries = [...referenceQueries, sourceQuery];
        }

        return await generateKnowledgeGraph(vscode, referenceQueries, urls, contentMap, {
            primaryContent,
            existingKnowledgeGraph
        });
    }

async function getVSCodeUsername() {
        try {
            // Try to get user info from VS Code authentication first
            const sessions = await vscode.authentication.getSessions('github', ['user:email']);
            if (sessions && sessions.length > 0) {
                const username = sessions[0].account.label;
                if (username && username.trim()) {
                    return username;
                }
            }
        } catch (err) {
            console.log('[getVSCodeUsername] No GitHub session found or error:', err);
        }
        
        try {
            // Fallback to user's configured username alias
            const configuration = vscode.workspace.getConfiguration('repoAsk');
            const usernameAlias = configuration.get('usernameAlias');
            if (usernameAlias && String(usernameAlias).trim()) {
                return String(usernameAlias).trim();
            }
        } catch (err) {
            console.log('[getVSCodeUsername] Error getting username alias from settings:', err);
        }
        
        // Final hardcoded fallback
        return 'Anonymous';
    }

    async function showLogActionButton(firstUserQuery, firstRankedDocUrl, fullAiResponse, selectedDocument, queryStartTime) {
        const username = await getVSCodeUsername();
        if (docsWebviewView) {
            docsWebviewView.webview.postMessage({
                command: 'showFeedbackForm',
                firstUserQuery: firstUserQuery,
                firstRankedDocUrl: firstRankedDocUrl,
                fullAiResponse: fullAiResponse,
                selectedDocument: selectedDocument || null,
                queryStartTime: queryStartTime || 0,
                username: username
            });
        }
    }

    return {
        sidebarProvider,
        refreshSidebarView,
        setSidebarSyncStatus,
        setSidebarSyncError,
        upsertSidebarDocument,
        revealDocumentInSidebar,
        showLogActionButton
    };
}

module.exports = {
    createSidebarController
};
