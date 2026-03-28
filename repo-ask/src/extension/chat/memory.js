/**
 * Session memory management for the RepoAsk chat agent.
 *
 * Converts VS Code's native chatContext.history into LangChain message objects
 * so prior turns in the same conversation are prepended to every agent invocation.
 * A sliding window keeps only the last MAX_HISTORY_TURNS turns to stay within
 * model context limits.
 *
 * VS Code's ChatRequestTurn / ChatResponseTurn are detected by duck-typing so
 * this module does not import `vscode` directly (it is injected as a parameter).
 */

const { HumanMessage, AIMessage } = require('@langchain/core/messages');

/** Maximum number of prior conversation turns to include in context. */
const MAX_HISTORY_TURNS = 5;

/**
 * Extract prior conversation turns from VS Code's chatContext.history and
 * convert them to LangChain HumanMessage / AIMessage pairs.
 *
 * Handles both ChatRequestTurn (user prompt) and ChatResponseTurn (assistant reply).
 * Only the last MAX_HISTORY_TURNS * 2 history entries are processed to cap context size.
 *
 * @param {Object}  chatContext - VS Code chatContext passed to the chat participant handler
 * @param {Object}  vscodeApi   - The `vscode` module (for ChatResponseMarkdownPart check)
 * @returns {Array<HumanMessage|AIMessage>} LangChain message array, oldest first
 */
function historyToMessages(chatContext, vscodeApi) {
    const history = chatContext?.history;
    if (!Array.isArray(history) || history.length === 0) return [];

    const windowEntries = history.slice(-(MAX_HISTORY_TURNS * 2));
    const messages = [];

    for (const entry of windowEntries) {
        // ChatRequestTurn: has a string `prompt` property
        if (typeof entry?.prompt === 'string') {
            const text = entry.prompt.trim();
            if (text) messages.push(new HumanMessage(text));
            continue;
        }

        // ChatResponseTurn: has a `response` array of parts
        if (Array.isArray(entry?.response)) {
            const MarkdownPart = vscodeApi?.ChatResponseMarkdownPart;
            const text = entry.response
                .filter(p => MarkdownPart ? p instanceof MarkdownPart : typeof p?.value?.value === 'string')
                .map(p => (typeof p.value === 'string' ? p.value : (p.value?.value || '')))
                .join('')
                .trim();
            if (text) messages.push(new AIMessage(text));
        }
    }

    return messages;
}

module.exports = { historyToMessages, MAX_HISTORY_TURNS };
