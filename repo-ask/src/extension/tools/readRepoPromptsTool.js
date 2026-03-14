const { toToolResult } = require('./utils');

module.exports = function registerReadRepoPromptsTool(deps) {
    const { vscode } = deps;
    return vscode.lm.registerTool('repoask_read_repo_prompts', {
            prepareInvocation() {
                return {
                    invocationMessage: 'Reading repository code guidelines prompts...',
                    confirmationMessages: {
                        title: 'Read repo prompts?',
                        message: 'This will read guidelines from .github/prompts/*.prompt.md'
                    }
                };
            },
            async invoke() {
                const fs = require('fs');
                const path = require('path');
                const workspaceFolder = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
                if (!workspaceFolder) {
                    return toToolResult('No workspace folder open', { contents: null });
                }

                const promptsDir = path.join(workspaceFolder, '.github', 'prompts');
                if (!fs.existsSync(promptsDir)) {
                    return toToolResult('No .github/prompts/ directory found.', { contents: [] });
                }

                try {
                    const files = fs.readdirSync(promptsDir).filter(f => f.endsWith('.prompt.md'));
                    if (files.length === 0) {
                        return toToolResult('No .prompt.md files found in .github/prompts/.', { contents: [] });
                    }

                    const contents = files.map(f => {
                        const content = fs.readFileSync(path.join(promptsDir, f), 'utf8');
                        return `--- File: ${f} ---\n${content}`;
                    });

                    return toToolResult(`Found ${files.length} prompt file(s):\n\n${contents.join('\n\n')}`, { contents });
                } catch (error) {
                    return toToolResult(`Failed to read prompts: ${error.message}`, { contents: null, error: error.message });
                }
            }
        });
};
