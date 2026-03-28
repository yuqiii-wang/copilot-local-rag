const {
    looksLikeNotFoundAnswer,
    selectDefaultChatModel,
    runAgentLoop,
    emitThinking,
    withTimeout,
    LLM_RESPONSE_TIMEOUT_MS
} = require('./shared');
const { VsCodeChatModel } = require('./vscodeModel');
const { buildAgentTools } = require('./agentTools');
const { buildPhase1Template, buildPhase2Prompt } = require('./prompts');
const { historyToMessages } = require('./memory');

async function answerGeneralPromptQuestion(vscodeApi, prompt, workspacePromptContext, response, deps, options = {}) {
    const queryStartTime = Date.now();
    const {
        storagePath,
        documentService,
        chatContext
    } = deps;

    if (!vscodeApi.lm || !vscodeApi.LanguageModelChatMessage) {
        response.markdown('No language model is available in this VS Code session.');
        return;
    }

    const vsModel = await selectDefaultChatModel(vscodeApi, options);
    if (!vsModel) {
        response.markdown('No language model is available in this VS Code session.');
        return;
    }

    // ── Phase 0: Rank local docs to prime the agent context ──────────────────
    let initialRankedContext = 'No initial documents found.';
    let topDocFromSearch = null;

    if (documentService && typeof documentService.rankLocalDocuments === 'function') {
        const repAskConfig = vscodeApi.workspace.getConfiguration('repoAsk');
        const maxResults = Math.max(Number(repAskConfig.get('maxSearchResults')) || 5, 1);
        const ranked = documentService.rankLocalDocuments(prompt, Math.max(maxResults * 10, 50));

        if (ranked && ranked.length > 0) {
            const confUrl = String((repAskConfig.get('confluence')?.url) || '').replace(/\/$/, '');
            const jiraUrl = String((repAskConfig.get('jira')?.url) || '').replace(/\/$/, '');

            const results = ranked.map(item => {
                let fullUrl = item.url || '';
                if (fullUrl && !fullUrl.startsWith('http')) {
                    const isJira = item.parent_confluence_topic && String(item.parent_confluence_topic).startsWith('Jira');
                    fullUrl = `${isJira ? jiraUrl : confUrl}${fullUrl.startsWith('/') ? '' : '/'}${fullUrl}`;
                }
                return { id: item.id, title: item.title || 'Untitled', url: fullUrl || 'None', summary: item.summary || '', score: item.score };
            });

            topDocFromSearch = results[0];
            initialRankedContext = 'Found the following relevant documents:\n' + results.map(doc => {
                const score = typeof doc.score === 'number' ? ` | Score: ${Math.round(doc.score * 10) / 10}` : '';
                return `- ID: ${doc.id} | Title: ${doc.title} | URL: ${doc.url} | Summary: ${doc.summary}${score}`;
            }).join('\n');

            const refLines = results.map(doc => {
                const score = typeof doc.score === 'number' ? ` — match score: ${Math.round(doc.score * 10) / 10}` : '';
                return `- [${doc.title}](${doc.url})${score}`;
            });
            response.markdown(`\n\n<details>\n<summary>Used ${results.length} references from ranking</summary>\n\n${refLines.join('\n')}\n</details>\n\n`);
        }
    }

    // ── Phase 1: Tool-calling agent (LangChain) ───────────────────────────────
    // Build a VS Code LM adapter and bind the registered repoask_ tools for tool calling
    const vsTools = (vscodeApi.lm.tools || []).filter(t => t.name === 'repoask_doc_check');
    const agentModel = new VsCodeChatModel({
        vsModel,
        vscodeApi,
        cancellationToken: options.request?.token,
        vsTools
    });

    // LangChain StructuredTool executors (actual invocation via vscode.lm.invokeTool)
    const lcTools = buildAgentTools({ vscodeApi, options, storagePath, response });

    // Hydrate conversation history from VS Code's native chatContext.history
    const history = historyToMessages(chatContext, vscodeApi);

    // Build phase-1 messages using the shared template (includes MessagesPlaceholder for history)
    const phase1Messages = await buildPhase1Template().formatMessages({
        workspacePromptContext: workspacePromptContext || 'None.',
        initialRankedContext,
        history,
        prompt
    });

    const shortQuery = prompt.length > 60 ? prompt.slice(0, 57) + '…' : prompt;
    emitThinking(response, `Searching docs for: ${shortQuery}`);
    const { finalText: firstRoundOutput, messages: agentMessages } = await runAgentLoop({
        model: agentModel,
        lcTools,
        messages: phase1Messages,
        response,
        maxIterations: 7
    });

    // ── Phase 2: Synthesis (stream directly to response.markdown) ────────────
    emitThinking(response, 'Composing answer from retrieved documents...');

    const rawObservations = firstRoundOutput ||
        agentMessages.filter(m => typeof m._getType === 'function' && m._getType() === 'ai')
            .map(m => m.content).join('\n');
    const phase2Instruction = buildPhase2Prompt(rawObservations, prompt);

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
        console.error('Phase 2 synthesis failed:', e);
        finalAnswer = firstRoundOutput || 'Error synthesizing final answer.';
        response.markdown(finalAnswer);
    }

    // ── Extract top doc reference from LLM output ─────────────────────────────
    let firstRankedDocUrl = '';
    const topDocMatch = finalAnswer.match(/\[TOP_DOC_URL:\s*(.+?),\s*TOP_DOC_ID:\s*(.+?)\]/);
    if (topDocMatch) {
        firstRankedDocUrl = topDocMatch[1].trim();
    }

    // Fallback URL reference if the synthesis didn't emit a URL
    if (!/https?:\/\/[^\s\]'"()]+/.test(finalAnswer)) {
        if (topDocFromSearch?.url && topDocFromSearch.url !== 'None') {
            response.markdown(`\n\n**Reference:** [${topDocFromSearch.title}](${topDocFromSearch.url})`);
        } else {
            response.markdown('\n\n*No URL*');
        }
    }

    // ── Buttons / not-found note ──────────────────────────────────────────────
    if (!finalAnswer || looksLikeNotFoundAnswer(finalAnswer)) {
        response.markdown('\n\n*Note: No relevant docs found. Search the doc store for relevant doc IDs or keywords to refine.*');
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
