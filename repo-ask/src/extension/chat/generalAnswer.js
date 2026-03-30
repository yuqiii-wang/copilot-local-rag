const {
    looksLikeNotFoundAnswer,
    selectDefaultChatModel,
    emitThinking,
    withTimeout,
    LLM_RESPONSE_TIMEOUT_MS
} = require('./shared');
const { buildPhase2Prompt } = require('./prompts');

async function answerGeneralPromptQuestion(vscodeApi, prompt, workspacePromptContext, response, deps, options = {}) {
    const queryStartTime = Date.now();
    const { documentService, readDocumentContent } = deps;

    if (!vscodeApi.lm || !vscodeApi.LanguageModelChatMessage) {
        response.markdown('No language model is available in this VS Code session.');
        return;
    }

    const vsModel = await selectDefaultChatModel(vscodeApi, options);
    if (!vsModel) {
        response.markdown('No language model is available in this VS Code session.');
        return;
    }

    // ── Phase 1: Rank → read content by IDs ──────────────────────────────────
    let contentContext = '';
    let topDocFromSearch = null;

    if (documentService && typeof documentService.rankLocalDocuments === 'function') {
        const repAskConfig = vscodeApi.workspace.getConfiguration('repoAsk');
        const maxResults = Math.max(Number(repAskConfig.get('maxSearchResults')) || 5, 1);
        const ranked = documentService.rankLocalDocuments(prompt, maxResults);

        if (ranked && ranked.length > 0) {
            const confUrl = String((repAskConfig.get('confluence')?.url) || '').replace(/\/$/, '');
            const jiraUrl = String((repAskConfig.get('jira')?.url) || '').replace(/\/$/, '');

            const results = ranked.map(item => {
                let fullUrl = item.url || '';
                if (fullUrl && !fullUrl.startsWith('http')) {
                    const isJira = item.parent_confluence_topic && String(item.parent_confluence_topic).startsWith('Jira');
                    fullUrl = `${isJira ? jiraUrl : confUrl}${fullUrl.startsWith('/') ? '' : '/'}${fullUrl}`;
                }
                return { id: String(item.id), title: item.title || 'Untitled', url: fullUrl || 'None', summary: item.summary || '', score: item.score };
            });

            topDocFromSearch = results[0];

            const refLines = results.map(doc => {
                const score = typeof doc.score === 'number' ? ` — match score: ${Math.round(doc.score * 10) / 10}` : '';
                return `- [${doc.title}](${doc.url})${score}`;
            });
            response.markdown(`\n\n<details>\n<summary>Used ${results.length} references from ranking</summary>\n\n${refLines.join('\n')}\n</details>\n\n`);

            // Read content for each ranked doc ID (same path as search bar ranking)
            emitThinking(response, `Reading content from ${results.length} matched doc(s)...`);
            const contentParts = [];
            for (const doc of results) {
                const content = typeof readDocumentContent === 'function' ? readDocumentContent(doc.id) : null;
                if (content) {
                    contentParts.push(`### [${doc.title}] (ID: ${doc.id} | URL: ${doc.url})\n${content}`);
                } else if (doc.summary) {
                    contentParts.push(`### [${doc.title}] (ID: ${doc.id} | URL: ${doc.url})\nSummary: ${doc.summary}`);
                }
            }
            if (contentParts.length > 0) {
                contentContext = contentParts.join('\n\n---\n\n');
            }
        }
    }

    // ── Phase 2: Synthesize from content ─────────────────────────────────────
    emitThinking(response, 'Composing answer from retrieved documents...');

    const observations = workspacePromptContext
        ? `## Attached Context:\n${workspacePromptContext}\n\n## Retrieved Documents:\n${contentContext || 'No relevant documents found.'}`
        : contentContext || 'No relevant documents found.';

    const phase2Instruction = buildPhase2Prompt(observations, prompt);

    let finalAnswer = '';
    try {
        const synthMessages = [vscodeApi.LanguageModelChatMessage.User(phase2Instruction)];
        const synthResponse = await withTimeout(
            vsModel.sendRequest(synthMessages, {}, options.request?.token),
            LLM_RESPONSE_TIMEOUT_MS,
            null
        );

        if (synthResponse?.stream) {
            for await (const chunk of synthResponse.stream) {
                if (chunk instanceof vscodeApi.LanguageModelTextPart) {
                    finalAnswer += chunk.value;
                    response.markdown(chunk.value);
                }
            }
        }
    } catch (e) {
        console.error('Synthesis failed:', e);
        finalAnswer = 'Error synthesizing answer.';
        response.markdown(finalAnswer);
    }

    // ── Extract top doc URL from answer ───────────────────────────────────────
    let firstRankedDocUrl = '';
    const topDocMatch = finalAnswer.match(/\[TOP_DOC_URL:\s*(.+?),\s*TOP_DOC_ID:\s*(.+?)\]/);
    if (topDocMatch) {
        firstRankedDocUrl = topDocMatch[1].trim();
    }

    if (!/https?:\/\/[^\s\]'"()]+/.test(finalAnswer)) {
        if (topDocFromSearch?.url && topDocFromSearch.url !== 'None') {
            response.markdown(`\n\n**Reference:** [${topDocFromSearch.title}](${topDocFromSearch.url})`);
        } else {
            response.markdown('\n\n*No URL*');
        }
    }

    // ── Buttons ───────────────────────────────────────────────────────────────
    if (!finalAnswer || looksLikeNotFoundAnswer(finalAnswer)) {
        response.markdown('\n\n*Note: No relevant docs found. Try Advanced Doc Search to locate relevant documents.*');
        response.button({
            command: 'repo-ask.advancedDocSearch',
            title: 'Advanced Doc Search',
            arguments: [prompt]
        });
    } else {
        response.button({
            command: 'repo-ask.advancedDocSearch',
            title: 'Advanced Doc Search',
            arguments: [prompt]
        });
        response.button({
            command: 'repo-ask.showLogActionButton',
            title: 'Log Action',
            arguments: [prompt, firstRankedDocUrl || '[NO_URL]', finalAnswer, queryStartTime]
        });
        response.button({
            command: 'repo-ask.checkCodeLogic',
            title: 'Check Code Logic',
            arguments: [prompt, finalAnswer]
        });
    }
}

module.exports = {
    answerGeneralPromptQuestion
};
