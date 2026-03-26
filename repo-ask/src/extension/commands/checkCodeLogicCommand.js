/**
 * Command to fact-check a workflow summary against the codebase and reveal code logic.
 * Opens VS Code's chat with the current agent, passing the conversation history as guide.
 */

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

        const chatQuery = [
            `**Project Directory:** \`${projectContext}\`\n`,
            'The following is a summarized workflow derived from documentation:',
            '',
            `<details><summary>Workflow Summary (Click to expand)</summary>\n\n${workflowSummary}\n\n</details>`,
            '',
            '## Task',
            `Original question that produced the above summary: "${userQuestion}"`,
            '',
            'Using the workspace code:',
            '1. **Fact-check** — Verify whether the workflow described above accurately reflects what the code actually does. Point out any discrepancies.',
            '2. **Code Logic** — If the description is accurate (or mostly accurate), walk through the actual code logic: identify the relevant files, classes/functions, and execution flow that implement this workflow.',
        ].join('\n');

        await vscode.commands.executeCommand('workbench.action.chat.open', {
            query: chatQuery,
            isPartialQuery: false
        });
    });
};
