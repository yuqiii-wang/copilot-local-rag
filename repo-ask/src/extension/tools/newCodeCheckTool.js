const { toToolResult } = require('./utils');

module.exports = function registerNewCodeCheckTool(deps) {
    const { vscode } = deps;
    return vscode.lm.registerTool('repoask_new_code_check', {
            prepareInvocation() {
                return {
                    invocationMessage: 'Generating new code check vs main/master branch...',
                    confirmationMessages: {
                        title: 'Generate new code check?',
                        message: 'This will run git diff against main/master and return the changes for review.'
                    }
                };
            },
            async invoke() {
                const { execSync } = require('child_process');
                const workspaceFolder = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
                if (!workspaceFolder) {
                    return toToolResult('No workspace folder open', { diff: null });
                }
                
                try {
                    execSync('git rev-parse --is-inside-work-tree', { cwd: workspaceFolder, stdio: 'ignore' });
                    
                    const currentBranch = execSync('git branch --show-current', { cwd: workspaceFolder, encoding: 'utf8' }).trim();
                    if (currentBranch === 'main' || currentBranch === 'master') {
                        return toToolResult(`You are currently on the ${currentBranch} branch. Please switch to a dev branch first.`, { diff: null, error: `On ${currentBranch} branch` });
                    }
                } catch (e) {
                    return toToolResult('This workspace is not a valid git repository or git is not installed/permitted.', { diff: null, error: 'Not a git repository or git lacks permission' });
                }

                try {
                    let diff = '';
                    try {
                        diff = execSync('git diff origin/main...HEAD', { cwd: workspaceFolder, encoding: 'utf8' });
                    } catch (e1) {
                        try {
                            diff = execSync('git diff origin/master...HEAD', { cwd: workspaceFolder, encoding: 'utf8' });
                        } catch (e2) {
                            try {
                                diff = execSync('git diff main...HEAD', { cwd: workspaceFolder, encoding: 'utf8' });
                            } catch (e3) {
                                try {
                                    diff = execSync('git diff master...HEAD', { cwd: workspaceFolder, encoding: 'utf8' });
                                } catch (e4) {
                                    diff = execSync('git diff HEAD', { cwd: workspaceFolder, encoding: 'utf8' });
                                }
                            }
                        }
                    }
                    if (!diff || diff.trim().length < 50) {
                        const fs = require('fs');
                        const path = require('path');
                        const promptsDir = path.join(workspaceFolder, '.github', 'prompts');
                        let prompts = [];
                        if (fs.existsSync(promptsDir)) {
                            prompts = fs.readdirSync(promptsDir).filter(f => f.endsWith('.prompt.md')).map(f => {
                                return { file: f, content: fs.readFileSync(path.join(promptsDir, f), 'utf8') };
                            });
                        }
                        
                        if (prompts.length > 0) {
                            let promptsContent = prompts.map(p => `--- ${p.file} ---\n${p.content}`).join('\n\n');
                            return toToolResult(`Little or no code changes found.\n\nPrompts found in .github/prompts/:\n${promptsContent}\n\nINSTRUCTION FOR AI: Based on the little/no code change, check the prompts above. If they contain any Jira or task to do, understand the project architecture and start implementing the Jira description. If there's no clear Jira/task, do not write code but guide the user to load a Jira to prompts, run 'git checkout -b <new-branch>', and possibly manually add a TODO list.`, { diff: diff || '' });
                        } else {
                            return toToolResult(`Little or no code changes found, and no Jira/TODO prompts found in .github/prompts.\n\nINSTRUCTION FOR AI: Do not write code. Give a guide to the user explaining that they should load a Jira to prompts, run 'git checkout -b <new-branch>', and possibly manually add a TODO list.`, { diff: diff || '' });
                        }
                    }
                    return toToolResult(`Git diff:\n\n\`\`\`diff\n${diff}\n\`\`\``, { diff });
                } catch (error) {
                    return toToolResult(`Code check failed: ${error.message}`, { diff: null, error: error.message });
                }
            }
        });

        };
