/**
 * Command to fact-check a workflow summary against the codebase and reveal code logic.
 * Opens VS Code's chat with the current agent, passing the conversation history as guide.
 */

const { buildCheckCodeLogicQuery } = require('../chat/prompts');

module.exports = function createCheckCodeLogicCommand(deps) {
    const { vscode } = deps;

    return vscode.commands.registerCommand('repo-ask.checkCodeLogic', async (originalPrompt, fullAiResponse) => {
        const workflowSummary = String(fullAiResponse || '').trim();
        const userQuestion = String(originalPrompt || '').trim();

        // Determine the project context (workspace or file)
        const workspaceFolders = vscode.workspace.workspaceFolders;
        let projectContext = '';

        if (workspaceFolders && workspaceFolders.length > 0) {
            // Use the first (primary) workspace folder
            const workspacePath = workspaceFolders[0].uri.fsPath;
            projectContext = workspacePath;
        } else if (vscode.window.activeTextEditor) {
            // No workspace, but a file is open
            const filePath = vscode.window.activeTextEditor.document.uri.fsPath;
            projectContext = filePath;
        } else {
            // No workspace and no open file
            vscode.window.showErrorMessage(
                'Cannot check code logic: No workspace folder or file is open. Please open a project or file.'
            );
            return;
        }

        const chatQuery = buildCheckCodeLogicQuery({ projectContext, workflowSummary, userQuestion });

        await vscode.commands.executeCommand('workbench.action.chat.open', {
            query: chatQuery,
            isPartialQuery: false
        });
    });
};
