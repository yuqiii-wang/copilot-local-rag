/**
 * LangChain BaseChatModel adapter for VS Code's Language Model API.
 *
 * Translates between LangChain's standard message format (HumanMessage, AIMessage,
 * ToolMessage, SystemMessage) and VS Code's proprietary LanguageModelChatMessage format,
 * enabling LangChain's agent patterns to drive VS Code's built-in LM.
 */

const { BaseChatModel } = require('@langchain/core/language_models/chat_models');
const { AIMessage } = require('@langchain/core/messages');

class VsCodeChatModel extends BaseChatModel {
    /**
     * @param {Object} fields
     * @param {Object} fields.vsModel           - VS Code model from vscode.lm.selectChatModels
     * @param {Object} fields.vscodeApi         - The `vscode` module
     * @param {Object} [fields.cancellationToken] - VS Code CancellationToken
     * @param {Array}  [fields.vsTools]         - VS Code tool definitions to pass to sendRequest
     */
    constructor(fields) {
        super(fields || {});
        this.vsModel = fields.vsModel;
        this.vscodeApi = fields.vscodeApi;
        this.cancellationToken = fields.cancellationToken || null;
        this.vsTools = fields.vsTools || [];
    }

    _llmType() {
        return 'vscode-lm';
    }

    /**
     * Returns a new model instance with VS Code tool definitions bound.
     * Follows the LangChain bindTools convention for tool-calling agents.
     * @param {Array} tools - VS Code tool objects (from vscode.lm.tools)
     */
    bindTools(tools) {
        return new VsCodeChatModel({
            vsModel: this.vsModel,
            vscodeApi: this.vscodeApi,
            cancellationToken: this.cancellationToken,
            vsTools: Array.isArray(tools) ? tools : []
        });
    }

    /**
     * Converts a LangChain message array to VS Code LanguageModelChatMessage array.
     * Handles: SystemMessage/HumanMessage → User, AIMessage (+ tool_calls) → Assistant,
     *          ToolMessage → User with LanguageModelToolResultPart.
     * @param {Array} messages - LangChain BaseMessage array
     * @returns {Array} VS Code LanguageModelChatMessage array
     */
    _toVsMessages(messages) {
        const {
            LanguageModelChatMessage,
            LanguageModelTextPart,
            LanguageModelToolCallPart,
            LanguageModelToolResultPart
        } = this.vscodeApi;

        return messages.map(msg => {
            const type = typeof msg._getType === 'function' ? msg._getType() : 'human';

            if (type === 'system' || type === 'human') {
                const content = typeof msg.content === 'string'
                    ? msg.content
                    : JSON.stringify(msg.content);
                return LanguageModelChatMessage.User(content);
            }

            if (type === 'ai') {
                const parts = [];
                if (typeof msg.content === 'string' && msg.content) {
                    parts.push(new LanguageModelTextPart(msg.content));
                }
                if (Array.isArray(msg.tool_calls) && msg.tool_calls.length > 0) {
                    for (const tc of msg.tool_calls) {
                        parts.push(new LanguageModelToolCallPart(tc.id, tc.name, tc.args || {}));
                    }
                }
                return LanguageModelChatMessage.Assistant(parts.length > 0 ? parts : '');
            }

            if (type === 'tool') {
                const resultText = typeof msg.content === 'string'
                    ? msg.content
                    : JSON.stringify(msg.content);
                return LanguageModelChatMessage.User([
                    new LanguageModelToolResultPart(
                        msg.tool_call_id,
                        [new LanguageModelTextPart(resultText)]
                    )
                ]);
            }

            // Fallback: treat as user message
            return LanguageModelChatMessage.User(String(msg.content || ''));
        });
    }

    /**
     * Core LangChain generation hook.
     * Sends the message array to VS Code LM; converts the response — including any
     * LanguageModelToolCallPart chunks — into a LangChain AIMessage with `tool_calls`.
     * @param {Array} messages - LangChain BaseMessage array
     * @returns {Promise<{generations: Array, llmOutput: Object}>}
     */
    async _generate(messages) {
        const vsMessages = this._toVsMessages(messages);
        const requestOptions = this.vsTools.length > 0 ? { tools: this.vsTools } : {};

        const modelResponse = await this.vsModel.sendRequest(
            vsMessages,
            requestOptions,
            this.cancellationToken
        );

        let textContent = '';
        const toolCalls = [];

        if (modelResponse.stream) {
            for await (const chunk of modelResponse.stream) {
                if (chunk instanceof this.vscodeApi.LanguageModelTextPart) {
                    textContent += chunk.value;
                } else if (chunk instanceof this.vscodeApi.LanguageModelToolCallPart) {
                    toolCalls.push({
                        id: chunk.callId,
                        name: chunk.name,
                        args: chunk.input || {}
                    });
                }
            }
        }

        const aiMessage = new AIMessage({
            content: textContent,
            ...(toolCalls.length > 0 && { tool_calls: toolCalls })
        });

        return {
            generations: [{ message: aiMessage, text: textContent }],
            llmOutput: {}
        };
    }
}

module.exports = { VsCodeChatModel };
