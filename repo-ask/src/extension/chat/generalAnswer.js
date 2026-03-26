const fs = require('fs');
const path = require('path');
const {
    looksLikeNotFoundAnswer,
    selectDefaultChatModel,
    runModelWithTools
} = require('./shared');

async function answerGeneralPromptQuestion(vscodeApi, prompt, workspacePromptContext, response, deps, options = {}) {
    const queryStartTime = Date.now();
    const {
        tokenize,
        storagePath,
        documentService
    } = deps;

    if (!vscodeApi.lm || !vscodeApi.LanguageModelChatMessage) {
        response.markdown('No language model is available in this VS Code session.');
        return;
    }

    const model = await selectDefaultChatModel(vscodeApi, options);
    if (!model) {
        response.markdown('No language model is available in this VS Code session.');
        return;
    }

    let initialRankedContext = 'No initial documents found.';
    let topDocFromSearch = null;
    if (documentService && typeof documentService.rankLocalDocuments === 'function') {
        const ranked = documentService.rankLocalDocuments(prompt, 10);

        if (ranked && ranked.length > 0) {
            const repAskConfig = vscodeApi.workspace.getConfiguration('repoAsk');
            const confProfile = repAskConfig.get('confluence');
            const confUrl = String((confProfile && typeof confProfile === 'object' ? confProfile.url : '') || '').replace(/\/$/, '');
            const jiraProfile = repAskConfig.get('jira');
            const jiraUrl = String((jiraProfile && typeof jiraProfile === 'object' ? jiraProfile.url : '') || '').replace(/\/$/, '');

            const results = ranked.map(item => {
                let fullUrl = item.url || '';
                if (fullUrl && !fullUrl.startsWith('http')) {
                    const isJira = item.parent_confluence_topic && String(item.parent_confluence_topic).startsWith('Jira');
                    const baseUrl = isJira ? jiraUrl : confUrl;
                    fullUrl = `${baseUrl}${fullUrl.startsWith('/') ? '' : '/'}${fullUrl}`;
                }
                return {
                    id: item.id,
                    title: item.title || 'Untitled',
                    url: fullUrl || 'None',
                    summary: item.summary || '',
                    keywords: item.keywords || []
                };
            });
            if (results.length > 0) {
                topDocFromSearch = results[0];
            }
            initialRankedContext = 'Found the following relevant documents:\n' + results.map(doc => 
                `- ID: ${doc.id} | Title: ${doc.title} | URL: ${doc.url} | Summary: ${doc.summary}`
            ).join('\n');
            
            const lines = results.map(item => `- [${item.title}](${item.url})`);
            const internalThinking = `\n\n<details>\n<summary>Used ${results.length} references from ranking</summary>\n\n${lines.join('\n')}\n</details>`;
            response.markdown(internalThinking + '\n\n');
        }
    }

    let instruction = [
        'You are RepoAsk Doc Agent. Your goal is to help the user answer general questions from the document store.',
        'Wait for tool results before explaining the final answer.',
        '- You MUST rely on the `local-store` via tools to find the answer.',
        '- Read content with `repoask_doc_check` for the most relevant documents identified in the Initial Ranked Documents Context below.',
        '- You MUST NOT hallucinate any information that is not explicitly present in the retrieved documents.',
        '- Do NOT output the final answer to the user yet. This is your internal thinking phase.',
        '- Gather all relevant information from the checked documents.',
        '- Identify which documents are actually relevant to answering the question.',
        '- Include all relevant document URLs and IDs in your internal analysis.',
        '',
        workspacePromptContext
            ? `## Attached Files and Code Context (provided by user):\n${workspacePromptContext}`
            : 'No attached files or pinned code.',
        `Initial Ranked Documents Context:\n${initialRankedContext}`,
        '',
        `User question: ${prompt}`
    ].join('\n\n');

    let toolsToUse = (vscodeApi.lm.tools || []).filter(t => t.name.startsWith('repoask_'));
    toolsToUse = toolsToUse.filter(t => t.name === 'repoask_doc_check');
    
    const firstRoundOutput = await runModelWithTools({
        vscodeApi,
        model,
        response,
        instruction,
        tools: toolsToUse,
        options: {
            ...options,
            storagePath
        }
    });

    const { emitThinking } = require('./shared');
    emitThinking(response, 'Synthesizing final answer...');

    let secondInstruction = [
        'You are an expert technical editor. Your task is to process the raw internal observations generated from a document search and provide a clear, concise final answer to the user.',
        '- Be as concise as possible. Avoid repeating information, filler phrases, and unnecessary preamble.',
        '- Prefer bullet points or short paragraphs. Aim for the minimum words needed to fully answer the question.',
        '- Judge which sources from the raw output are actually used and relevant to the user question.',
        '- Remove references or citations to any unused or irrelevant documents.',
        '- Format the final clear answer for the user.',
        '- You MUST include the clickable document link for the most relevant document used in your answer.',
        '- You MUST output the top doc URL and ID for the most relevant document used in your answer at the very bottom, formatted exactly as: `[TOP_DOC_URL: <url>, TOP_DOC_ID: <id>]`.',
        '- If no documents were relevant or used, explicitly state that you cannot answer the question based on the available documents, and format the doc reference as `[TOP_DOC_URL: [NO_URL], TOP_DOC_ID: [NO_ID]]`.',
        '',
        '--- RAW INTERNAL OBSERVATIONS ---',
        firstRoundOutput,
        '--- END RAW INTERNAL OBSERVATIONS ---',
        '',
        `User question: ${prompt}`
    ].join('\n\n');

    let finalAnswer = '';
    try {
        const secondRoundMessages = [
            vscodeApi.LanguageModelChatMessage.User(secondInstruction)
        ];
        
        const secondRoundResponse = await model.sendRequest(secondRoundMessages, {}, options.request?.token);
        
        if (secondRoundResponse.stream) {
            for await (const chunk of secondRoundResponse.stream) {
                if (chunk instanceof vscodeApi.LanguageModelTextPart) {
                    finalAnswer += chunk.value;
                    response.markdown(chunk.value);
                }
            }
        }
    } catch (e) {
        console.error("Second round LLM failed:", e);
        finalAnswer = firstRoundOutput || "Error synthesizing final answer.";
        response.markdown(finalAnswer);
    }

    // Extract the top doc URL and ID from the LLM's output
    let firstRankedDocUrl = '';
    let firstRankedDocId = '';
    const match = finalAnswer.match(/\[TOP_DOC_URL:\s*(.+?),\s*TOP_DOC_ID:\s*(.+?)\]/);
    if (match && match[1] && match[2]) {
        firstRankedDocUrl = match[1].trim();
        firstRankedDocId = match[2].trim();
    }

    // fallback when output doesn't contain url
    const isUrlMatch = finalAnswer.match(/https?:\/\/[^\s\]'"()]+/);
    if (!isUrlMatch) {
         if (topDocFromSearch && topDocFromSearch.url && topDocFromSearch.url !== 'None') {
             response.markdown(`\n\n**Reference:** [${topDocFromSearch.title}](${topDocFromSearch.url})`);
         } else {
             response.markdown(`\n\n*No URL*`);
         }
    }

    // Check if the answer indicates no relevant docs were found
    const isNotFoundAnswer = !finalAnswer || looksLikeNotFoundAnswer(finalAnswer);

    // Ensure we always send a response, even if empty
    if (isNotFoundAnswer) {
        response.markdown('\n\n*Note: No relevant docs found, you can search from doc store and find the doc id/title or more keywords to help locate the search*');
    } else {
        response.button({
            command: 'repo-ask.showLogActionButton',
            title: 'Log Action',
            arguments: [prompt, firstRankedDocUrl || '[NO_URL]', finalAnswer, queryStartTime]
        });
    }

}

module.exports = {
    answerGeneralPromptQuestion
};
