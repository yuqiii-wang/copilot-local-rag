/**
 * Command to trigger the Advanced Doc Search agentic loop.
 * Opens an @repoask chat turn prefixed with [ADV] so the participant
 * routes to the deeper iterative search handler.
 */

module.exports = function createAdvancedDocSearchCommand(deps) {
    const { vscode } = deps;

    return vscode.commands.registerCommand('repo-ask.advancedDocSearch', async (originalPrompt) => {
        const query = String(originalPrompt || '').trim();
        if (!query) {
            vscode.window.showErrorMessage('Advanced Doc Search requires an original query.');
            return;
        }

        // Open a new @repoask chat turn with the [ADV] prefix so the participant
        // can detect and route to the advanced search handler.
        await vscode.commands.executeCommand('workbench.action.chat.open', {
            query: `@repoask [ADV] ${query}`,
            isPartialQuery: false
        });
    });
};
