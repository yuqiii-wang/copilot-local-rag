const fs = require('fs');
const path = require('path');
const {
    looksLikeNotFoundAnswer,
    selectDefaultChatModel,
    runModelWithTools
} = require('./shared');

async function answerGeneralPromptQuestion(vscodeApi, prompt, workspacePromptContext, response, deps, options = {}) {
    const {
        tokenize,
        rankDocumentsByIdf
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

    const contextText = String(workspacePromptContext || '').trim();

    let instruction = [
        'You are RepoAsk Doc Agent. Your goal is to help the user answer general questions from the document store.',
        'Wait for tool results before explaining the final answer.',
        '- You MUST rely on the `local-store` via tools to find the answer.',
        '- Use `repoask_rank` tool with limit 10 to select the top 10 docs.',
        '- Read content with `repoask_doc_check` and base your answer solely on the retrieved text.',
        '- You MUST NOT hallucinate any information that is not explicitly present in the retrieved documents.',
        '- If no relevant documents are found or if the documents do not contain enough information to answer the question, you MUST explicitly state that you cannot answer the question based on the available documents.',
        '- You MUST cite the specific documents you used to form your answer, including document IDs and titles.',
        '- You MUST output the top doc URL in your final output, which will be used to populate the log action feedback URL and Confluence/Jira ID.',
        '- You MUST output a summary of the checked documents in relation to the question, even if ALL docs are irrelevant.',
        '',
        'Workspace guidelines: (none)',
        `User question: ${prompt}`
    ].join('\n\n');

    let toolsToUse = (vscodeApi.lm.tools || []).filter(t => t.name.startsWith('repoask_'));
    toolsToUse = toolsToUse.filter(t => t.name !== 'repoask_code_new_feat' && t.name !== 'repoask_code_explore');
    const finalText = await runModelWithTools({
        vscodeApi,
        model,
        response,
        instruction,
        tools: toolsToUse,
        options
    });

    // Get first ranked doc URL if available
    let firstRankedDocUrl = '';
    let hasRankedDocs = false;
    if (options.metadataList && options.metadataList.length > 0) {
        // Check if we have any documents at all
        hasRankedDocs = true;
        
        // Rank documents by relevance to the prompt
        const rankedDocs = rankDocumentsByIdf(
            prompt,
            options.metadataList.map((doc) => ({ ...doc, content: '' })),
            tokenize,
            { limit: 1, minScore: 0 }
        );
        if (rankedDocs && rankedDocs.length > 0) {
            const firstDoc = rankedDocs[0];
            // Assuming the doc has a URL property, adjust as needed
            firstRankedDocUrl = firstDoc.url || firstDoc.link || '';
        }
    }

    // Check if the answer indicates no relevant docs were found
    const isNotFoundAnswer = !finalText || looksLikeNotFoundAnswer(finalText);

    // Ensure we always send a response, even if empty
    if (isNotFoundAnswer) {
        response.markdown('No relevant docs found, you can search from doc store and find the doc id/title or more keywords to help locate the search');
    } else {
        response.markdown(finalText);
        response.button({
            command: 'repo-ask.showLogActionButton',
            title: 'Log Action',
            arguments: [prompt, firstRankedDocUrl, finalText]
        });
    }

}

module.exports = {
    answerGeneralPromptQuestion
};
