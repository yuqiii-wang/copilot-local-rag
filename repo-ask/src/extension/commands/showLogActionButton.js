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

        const combinedChatText = [String(firstUserQuery || '').trim(), String(fullAiResponse || '').trim()]
            .filter(Boolean)
            .join('\n\n');
        const signals = extractChatSignals(combinedChatText);
        const normalizedPreferredUrl = normalizeUrl(firstRankedDocUrl);
        const rankedDocs = documentService.rankLocalDocuments(combinedChatText, 25);
        const candidateMap = new Map();

        for (const doc of rankedDocs) {
            candidateMap.set(String(doc.id || '').trim(), {
                ...doc,
                rerankScore: Number(doc.score || 0)
            });
        }

        for (const metadata of metadataList) {
            const docId = String(metadata.id || '').trim();
            const docUrl = normalizeUrl(metadata.url || metadata.link || metadata.source);
            const matchesExplicitId = signals.confluenceIds.includes(docId) || signals.jiraIds.includes(docId);
            const matchesExplicitUrl = docUrl && signals.urls.includes(docUrl);
            const matchesPreferredUrl = docUrl && normalizedPreferredUrl && docUrl === normalizedPreferredUrl;
            if (!matchesExplicitId && !matchesExplicitUrl && !matchesPreferredUrl) {
                continue;
            }

            const existing = candidateMap.get(docId) || {
                ...metadata,
                rerankScore: 0
            };
            existing.rerankScore += matchesExplicitId ? 1000 : 0;
            existing.rerankScore += matchesExplicitUrl ? 800 : 0;
            existing.rerankScore += matchesPreferredUrl ? 200 : 0;
            candidateMap.set(docId, existing);
        }

        const bestDocument = Array.from(candidateMap.values())
            .sort((left, right) => Number(right.rerankScore || 0) - Number(left.rerankScore || 0))[0] || null;

        return deriveFeedbackTarget(bestDocument);
    }

    return vscode.commands.registerCommand('repo-ask.showLogActionButton', async (firstUserQuery, firstRankedDocUrl, fullAiResponse) => {
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
            sidebar.showLogActionButton(firstUserQuery, firstRankedDocUrl, fullAiResponse, selectedDocument);
        }
    });
};