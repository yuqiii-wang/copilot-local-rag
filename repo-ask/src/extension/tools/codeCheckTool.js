const { toToolResult } = require('./utils');

module.exports = function registerCodeDiffCheckTool(deps) {
    const { vscode } = deps;
    return vscode.lm.registerTool('repoask_code_diff_check', {
            prepareInvocation({ args }) {
                return {
                    invocationMessage: 'Generating new code check vs main/master branch...',
                    confirmationMessages: {
                        title: 'Generate new code check?',
                        message: 'This will run git diff against main/master and return the changes for review.'
                    }
                };
            },
            async invoke({ args }) {
                const { execSync } = require('child_process');
                const workspaceFolder = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
                if (!workspaceFolder) {
                    return toToolResult('No workspace folder open', { diff: null });
                }
                
                try {
                    execSync('git rev-parse --is-inside-work-tree', { cwd: workspaceFolder, stdio: 'ignore' });
                } catch (e) {
                    return toToolResult('This workspace is not a valid git repository or git is not installed/permitted.', { diff: null, error: 'Not a git repository or git lacks permission' });
                }

                // Extract Jira ID from user query if provided
                let jiraId = null;
                if (args && args.query) {
                    const jiraConfig = vscode.workspace.getConfiguration('repoAsk').get('jira');
                    if (jiraConfig && jiraConfig.regex) {
                        const regexPatterns = Array.isArray(jiraConfig.regex) ? jiraConfig.regex : [jiraConfig.regex];
                        for (const pattern of regexPatterns) {
                            const regex = new RegExp(pattern, 'g');
                            const match = regex.exec(args.query);
                            if (match) {
                                jiraId = match[0];
                                break;
                            }
                        }
                    }
                }

                // If Jira ID found, search for commits containing it
                if (jiraId) {
                    try {
                        const commits = execSync(`git log --grep="${jiraId}" --oneline`, { cwd: workspaceFolder, encoding: 'utf8' });
                        if (commits && commits.trim()) {
                            // Get the latest commit with the Jira ID
                            const latestCommit = commits.trim().split('\n')[0].split(' ')[0];
                            // Get the diff for that commit
                            const commitDiff = execSync(`git show ${latestCommit}`, { cwd: workspaceFolder, encoding: 'utf8' });
                            return toToolResult(`Found commits for Jira ID ${jiraId}:\n\n${commits}\n\nGit diff for latest commit:\n\n\`\`\`diff\n${commitDiff}\n\`\`\``, { diff: commitDiff, jiraId, commits });
                        } else {
                            return toToolResult(`No commits found for Jira ID ${jiraId}`, { diff: null, jiraId });
                        }
                    } catch (error) {
                        return toToolResult(`Error searching for commits with Jira ID ${jiraId}: ${error.message}`, { diff: null, error: error.message });
                    }
                }

                // Original functionality - check git diff
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