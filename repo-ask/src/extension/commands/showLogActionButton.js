/**
 * Command to show the log action button and store logged prompts
 */

module.exports = function createShowLogActionButtonCommand(deps) {
    const { vscode, context, sidebar, documentService, readAllMetadata, storagePath } = deps;

    function normalizeUrl(url) {
        return String(url || '').trim().replace(/[)>.,;]+$/, '').replace(/\/$/, '');
    }

    function extractChatSignals(text) {
        const rawText = String(text || '');
        return {
            confluenceIds: [...new Set(Array.from(rawText.matchAll(/\b\d{5,}\b/g), match => match[0]))],
            jiraIds: [...new Set(Array.from(rawText.matchAll(/\b[A-Z][A-Z0-9]+-\d+\b/g), match => match[0].toUpperCase()))],
            urls: [...new Set(Array.from(rawText.matchAll(/https?:\/\/[^\s)\]>'"]+/gi), match => normalizeUrl(match[0])))]
        };
    }

    function deriveFeedbackTarget(document) {
        const metadata = document && typeof document === 'object' ? document : null;
        if (!metadata) {
            return null;
        }

        const url = String(metadata.url || metadata.link || metadata.source || '').trim();
        const rawId = String(metadata.id || '').trim();
        const jiraIdFromUrl = url.match(/\b([A-Z][A-Z0-9]+-\d+)\b/);
        const jiraIdFromTitle = String(metadata.title || '').match(/^([A-Z][A-Z0-9]+-\d+)\b/);
        const isJira = String(metadata.type || '').toLowerCase() === 'jira' || Boolean(jiraIdFromUrl || jiraIdFromTitle);

        return {
            id: rawId,
            title: String(metadata.title || '').trim(),
            type: String(metadata.type || '').trim(),
            url,
            confluencePageId: !isJira && /^\d+$/.test(rawId) ? rawId : '',
            jiraId: isJira ? String((jiraIdFromUrl && jiraIdFromUrl[1]) || (jiraIdFromTitle && jiraIdFromTitle[1]) || '').trim() : ''
        };
    }

    function selectHighestScoreDocument(firstUserQuery, firstRankedDocUrl, fullAiResponse) {
        if (!documentService || typeof documentService.rankLocalDocuments !== 'function') {
            return null;
        }

        const metadataList = typeof readAllMetadata === 'function' ? readAllMetadata(storagePath) : [];
        if (!Array.isArray(metadataList) || metadataList.length === 0) {
            return null;
        }

        // Try to extract AI-decided top doc directly from response
        const aiTopDocMatch = String(fullAiResponse || '').match(/\[TOP_DOC_URL:\s*(.+?),\s*TOP_DOC_ID:\s*(.+?)\]/);
        if (aiTopDocMatch && aiTopDocMatch[1] && aiTopDocMatch[2]) {
            const aiUrl = normalizeUrl(aiTopDocMatch[1]);
            const aiId = aiTopDocMatch[2].trim();
            // Find this document in metadata
            let aiDoc = metadataList.find(m => 
                String(m.id || '').trim() === aiId || 
                normalizeUrl(m.url || m.link || m.source) === aiUrl
            );
            
            if (aiDoc) {
                return deriveFeedbackTarget(aiDoc);
            } else {
                return {
                    id: aiId,
                    title: '',
                    type: '',
                    url: aiTopDocMatch[1],
                    confluencePageId: /^\d+$/.test(aiId) ? aiId : '',
                    jiraId: /^[A-Z][A-Z0-9]+-\d+$/.test(aiId) ? aiId : ''
                };
            }
        }
        
        return null; // Do not fall back to naive rank
    }

    return vscode.commands.registerCommand('repo-ask.showLogActionButton', async (firstUserQuery, firstRankedDocUrl, fullAiResponse, queryStartTime) => {
        // Store the logged prompt in globalState to archive the chat
        if (firstUserQuery && context) {
            const loggedPrompts = context.globalState.get('repoAsk.loggedPrompts', []);
            if (!loggedPrompts.includes(firstUserQuery)) {
                loggedPrompts.push(firstUserQuery);
                await context.globalState.update('repoAsk.loggedPrompts', loggedPrompts);
            }
        }

        const selectedDocument = selectHighestScoreDocument(firstUserQuery, firstRankedDocUrl, fullAiResponse);

        // Show the feedback form with the first user query, first ranked doc URL, and full AI response.
        if (sidebar) {
            sidebar.showLogActionButton(firstUserQuery, firstRankedDocUrl, fullAiResponse, selectedDocument, queryStartTime);
        }
    });
};