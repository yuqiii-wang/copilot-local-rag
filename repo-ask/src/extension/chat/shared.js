/**
 * Shared utilities for chat functionality
 */

/**
 * Timeout duration for LLM responses in milliseconds
 * @constant {number}
 */
exports.LLM_RESPONSE_TIMEOUT_MS = 30000;

/**
 * Converts text to sentence case
 * @param {string} text - The text to convert
 * @returns {string} The text in sentence case
 */
exports.toSentenceCase = function toSentenceCase(text) {
    const value = String(text || '').trim();
    if (!value) {
        return '';
    }

    const normalized = value.endsWith('.') ? value : `${value}.`;
    return normalized.charAt(0).toUpperCase() + normalized.slice(1);
};

/**
 * Emits a thinking progress message
 * @param {Object} response - The response object with a progress method
 * @param {string} message - The message to emit
 */
exports.emitThinking = function emitThinking(response, message) {
    if (typeof response.progress === 'function') {
        response.progress(message);
    }
};

/**
 * Checks if an answer indicates no relevant documents were found
 * @param {string} text - The answer text to check
 * @returns {boolean} True if the answer indicates no relevant docs were found
 */
exports.looksLikeNotFoundAnswer = function looksLikeNotFoundAnswer(text) {
    const value = String(text || '').toLowerCase();
    if (!value) {
        return false;
    }

    return value.includes('not found in the provided context')
        || value.includes('not present in the provided context')
        || value.includes('not enough information in the provided context')
        || value.includes('cannot find this in the provided context')
        || value.includes('unable to find this in the provided context')
        || value.includes('i do not have enough context')
        || value.includes('insufficient context');
};

/**
 * Adds timeout functionality to a promise
 * @param {Promise} promise - The promise to add timeout to
 * @param {number} timeoutMs - The timeout duration in milliseconds
 * @param {*} timeoutValue - The value to return if timeout occurs
 * @returns {Promise} A promise that resolves with the original promise result or timeout value
 */
exports.withTimeout = async function withTimeout(promise, timeoutMs, timeoutValue = null) {
    let timeoutId;
    const timeoutPromise = new Promise((resolve) => {
        timeoutId = setTimeout(() => resolve(timeoutValue), timeoutMs);
    });

    try {
        return await Promise.race([promise, timeoutPromise]);
    } finally {
        clearTimeout(timeoutId);
    }
};

/**
 * Collects the full text from a VS Code LM response stream.
 * @param {Object} vsApi - VS Code API object
 * @param {Object} response - LM response object with stream or text iterator
 * @returns {Promise<string>}
 */
exports.collectResponseText = async function collectResponseText(vsApi, response) {
    if (!response) return '';
    let text = '';
    if (response.stream) {
        for await (const chunk of response.stream) {
            if (vsApi.LanguageModelTextPart && chunk instanceof vsApi.LanguageModelTextPart) {
                text += chunk.value;
            } else if (typeof chunk === 'string') {
                text += chunk;
            } else if (chunk && typeof chunk.value === 'string') {
                text += chunk.value;
            }
        }
    } else if (response.text) {
        for await (const fragment of response.text) text += fragment;
    }
    return text;
};

/**
 * Selects the default chat model for this VS Code session.
 * @param {Object} vscodeApi - VS Code API object
 * @param {Object} [options] - Options containing the chat request object
 * @returns {Promise<Object|null>} Selected chat model or null if unavailable
 */
exports.selectDefaultChatModel = async function selectDefaultChatModel(vscodeApi, options = {}) {
    if (!vscodeApi?.lm || !vscodeApi?.LanguageModelChatMessage) {
        return null;
    }

    if (options.request && options.request.model) {
        console.log(`[RepoAsk] Selected chat model from request: ${options.request.model.name || options.request.model.id || 'unknown'}`);
        return options.request.model;
    }

    const gpt5MiniModels = await exports.withTimeout(
        vscodeApi.lm.selectChatModels({ family: 'gpt-5-mini' }),
        exports.LLM_RESPONSE_TIMEOUT_MS,
        []
    );

    const fallbackModel = gpt5MiniModels?.[0] || null;
    if (fallbackModel) {
        console.log(`[RepoAsk] Selected fallback chat model: ${fallbackModel.name || fallbackModel.id || 'unknown'}`);
    } else {
        console.log(`[RepoAsk] No chat model available.`);
    }

    return fallbackModel;
};

/**
 * Truncates text for concise thinking/progress updates.
 * @param {string} text - Source text
 * @param {number} maxLen - Maximum preview length
 * @returns {string} Preview text
 */
exports.previewForThinking = function previewForThinking(text, maxLen = 180) {
    const value = String(text || '').replace(/\s+/g, ' ').trim();
    if (!value) {
        return '';
    }
    if (value.length <= maxLen) {
        return value;
    }
    return `${value.slice(0, maxLen - 3)}...`;
};

/**
 * LangChain-based agent execution loop.
 *
 * Manages conversation state as a LangChain message array (HumanMessage, AIMessage,
 * ToolMessage) and drives a tool-calling loop until the model stops issuing tool calls
 * or the iteration cap is reached.  Tool execution is delegated entirely to the
 * LangChain StructuredTool instances produced by agentTools.buildAgentTools(), keeping
 * VS Code invocation details out of this layer.
 *
 * @param {Object}  params
 * @param {Object}  params.model          - VsCodeChatModel (BaseChatModel adapter) with VS Code tools bound
 * @param {Array}   params.lcTools        - LangChain StructuredTool array (from buildAgentTools)
 * @param {Array}   params.messages       - Initial LangChain message array (SystemMessage + HumanMessage)
 * @param {Object}  params.response       - VS Code chat response stream (for emitThinking progress)
 * @param {number}  [params.maxIterations=7] - Safety cap for the tool-call loop
 * @returns {Promise<{ finalText: string, messages: Array }>}
 */
exports.runAgentLoop = async function runAgentLoop({
    model,
    lcTools,
    messages,
    response,
    maxIterations = 7
}) {
    const { ToolMessage } = require('@langchain/core/messages');
    const { buildToolMap } = require('./agentTools');

    const toolMap = buildToolMap(lcTools);
    const state = { messages: [...messages], iterations: 0 };
    let finalText = '';

    while (state.iterations < maxIterations) {
        state.iterations++;

        let aiMessage;
        try {
            aiMessage = await exports.withTimeout(
                model.invoke(state.messages),
                exports.LLM_RESPONSE_TIMEOUT_MS,
                null
            );
        } catch (e) {
            finalText = finalText || `Language model error: ${e.message}`;
            break;
        }

        if (!aiMessage) {
            finalText = finalText || 'No response from language model.';
            break;
        }

        state.messages.push(aiMessage);

        const textPart = typeof aiMessage.content === 'string' ? aiMessage.content : '';
        if (textPart) finalText += textPart;

        const toolCalls = Array.isArray(aiMessage.tool_calls) ? aiMessage.tool_calls : [];
        if (toolCalls.length === 0) break;

        for (const toolCall of toolCalls) {
            const lcTool = toolMap[toolCall.name];
            let toolResult;

            if (lcTool) {
                const args = toolCall.args || {};
                const mode = args.mode || 'content_partial';
                const ids = Array.isArray(args.ids) ? args.ids : [];
                const terms = Array.isArray(args.searchTerms) && args.searchTerms.length > 0
                    ? args.searchTerms : null;
                const queryHint = terms
                    ? terms.join(', ')
                    : (args.query ? args.query.slice(0, 60) : '');
                const idHint = ids.length > 0 ? ` [${ids.slice(0, 4).join(', ')}${ids.length > 4 ? ', …' : ''}]` : '';

                let thinkMsg;
                if (mode === 'metadata.id') {
                    thinkMsg = 'Listing all document IDs...';
                } else if (mode === 'metadata') {
                    thinkMsg = ids.length > 0
                        ? `Fetching metadata for ${ids.length} doc(s)${idHint}`
                        : `Searching metadata${queryHint ? `: ${queryHint}` : '...'}`;
                } else if (mode === 'metadata.summary' || mode === 'metadata.summary_kg') {
                    thinkMsg = ids.length > 0
                        ? `Loading summaries & KG for ${ids.length} doc(s)${idHint}`
                        : `Scanning summaries & KG${queryHint ? ` for: ${queryHint}` : '...'}`;
                } else if (mode === 'content_partial') {
                    thinkMsg = ids.length > 0
                        ? `Scanning content of ${ids.length} doc(s)${idHint}`
                        : `Searching docs${queryHint ? `: ${queryHint}` : '...'}`;
                } else if (mode === 'content') {
                    thinkMsg = ids.length > 0
                        ? `Reading full content of ${ids.length} doc(s)${idHint}`
                        : `Reading documents${queryHint ? ` for: ${queryHint}` : '...'}`;
                } else {
                    thinkMsg = ids.length > 0
                        ? `Reading ${ids.length} doc(s)${idHint}`
                        : `Searching local doc store${queryHint ? `: ${queryHint}` : '...'}`;
                }
                exports.emitThinking(response, thinkMsg);
                try {
                    toolResult = await lcTool.invoke(toolCall.args);
                } catch (err) {
                    toolResult = `Tool error: ${err.message}`;
                }
            } else {
                toolResult = `Unknown tool: ${toolCall.name}`;
            }

            state.messages.push(
                new ToolMessage({ content: toolResult, tool_call_id: toolCall.id })
            );
        }
    }

    return { finalText, messages: state.messages };
};