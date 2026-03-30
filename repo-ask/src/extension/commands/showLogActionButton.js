/**
 * Command to show the log action button and store logged prompts
 */

module.exports = function createShowLogActionButtonCommand(deps) {
    const { vscode, context, sidebar, documentService, readAllMetadata, storagePath } = deps;
    const { getJiraExtractionRegexes } = require('../../mcp/jiraApi');

    function normalizeUrl(url) {
        return String(url || '').trim().replace(/[)>.,;]+$/, '').replace(/\/$/, '');
    }

    function findJiraMatches(text, firstOnly = false) {
        const found = [];
        const seen = new Set();
        const textStr = String(text || '');
        for (const regex of getJiraExtractionRegexes(vscode)) {
            const gr = new RegExp(regex.source, regex.flags.includes('i') ? 'gi' : 'g');
            for (const m of textStr.matchAll(gr)) {
                const key = m[0].toUpperCase();
                if (!seen.has(key)) {
                    seen.add(key);
                    found.push(key);
                    if (firstOnly) return found;
                }
            }
        }
        return found;
    }

    function isJiraId(str) {
        const s = String(str || '').trim();
        return getJiraExtractionRegexes(vscode).some(re => {
            const anchored = new RegExp('^(?:' + re.source + ')$', re.flags.includes('i') ? 'i' : '');
            return anchored.test(s);
        });
    }

    function extractChatSignals(text) {
        const rawText = String(text || '');
        return {
            confluenceIds: [...new Set(Array.from(rawText.matchAll(/\b\d{5,}\b/g), match => match[0]))],
            jiraIds: findJiraMatches(rawText),
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
        const jiraIdFromUrl = findJiraMatches(url, true)[0] || null;
        const jiraIdFromTitle = findJiraMatches(String(metadata.title || ''), true)[0] || null;
        const isJira = String(metadata.type || '').toLowerCase() === 'jira' || Boolean(jiraIdFromUrl || jiraIdFromTitle);

        return {
            id: rawId,
            title: String(metadata.title || '').trim(),
            type: String(metadata.type || '').trim(),
            url,
            confluencePageId: !isJira && /^\d+$/.test(rawId) ? rawId : '',
            jiraId: isJira ? String(jiraIdFromUrl || jiraIdFromTitle || '').trim() : ''
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
                    jiraId: isJiraId(aiId) ? aiId : ''
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