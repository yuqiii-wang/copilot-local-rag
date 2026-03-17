const { toToolResult } = require('./utils');
const fs = require('fs');
const np = require('path');

module.exports = function registerCodeExploreTool(deps) {
    const { vscode } = deps;
    return vscode.lm.registerTool('repoask_code_explore', {
            prepareInvocation() {
                return {
                    invocationMessage: 'Exploring codebase...',
                    confirmationMessages: {
                        title: 'Explore codebase?',
                        message: 'This will explore the project structure or search for code patterns locally.'
                    }
                };
            },
            async invoke(options) {
                const args = options.input || {};
                
                const workspaceFolder = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
                if (!workspaceFolder) {
                    return toToolResult('No workspace folder open', { result: null });
                }
                
                // The LLM sometimes sends `{}`. If it does, force it to give missing parameters.
                if (!args || !args.command) {
                    return toToolResult(`[LLM Input: ${JSON.stringify(args)}]\nPlease specify your input correctly formatted as a JSON object, for example: {"command": "ls", "path": "."} or {"command": "read", "path": "package.json"}`, { result: null, error: "Missing required command" });
                }

                const command = args.command;
                const path = args.path || '.';
                let pattern = args.pattern || '';
                
                try {
                    const inputTrace = `[LLM Input: ${JSON.stringify(args)}]`;
                    
                    if (command === 'ls') {
                        const targetPath = np.resolve(workspaceFolder, path);
                        let result = '';
                        try {
                            const files = fs.readdirSync(targetPath, { withFileTypes: true });
                            // Sort directories first, then files
                            const sortedFiles = files.sort((a, b) => {
                                if (a.isDirectory() === b.isDirectory()) return a.name.localeCompare(b.name);
                                return a.isDirectory() ? -1 : 1;
                            });
                            result = sortedFiles.map(f => `${f.isDirectory() ? '[DIR] ' : '[FILE]'} ${f.name}`).join('\n');
                            if (!result) result = "(Empty directory)";
                        } catch (err) {
                            return toToolResult(`${inputTrace}\nError reading directory: ${err.message}`, { result: null, error: err.message });
                        }
                        return toToolResult(`${inputTrace}\nDirectory listing for ${path}:\n\n\`\`\`\n${result}\n\`\`\``, { result });
                    } else if (command === 'read') {
                        const targetPath = np.resolve(workspaceFolder, path);
                        try {
                            const content = fs.readFileSync(targetPath, 'utf8');
                            return toToolResult(`${inputTrace}\nFile content for ${path}:\n\n\`\`\`\n${content}\n\`\`\``, { result: content });
                        } catch (err) {
                            return toToolResult(`${inputTrace}\nError reading file: ${err.message}`, { result: null, error: err.message });
                        }
                    } else if (command === 'search') {
                        // Use VS Code's builtin findFiles
                        if (!pattern) pattern = glob;
                        
                        // If model tries to search everything, gracefully fall back to directory listing
                        if (!pattern || pattern === '*' || pattern === '**/*') {
                            const targetPath = np.resolve(workspaceFolder, path);
                            let rootFiles = '';
                            try {
                                rootFiles = fs.readdirSync(targetPath, { withFileTypes: true })
                                    .sort((a, b) => {
                                        if (a.isDirectory() === b.isDirectory()) return a.name.localeCompare(b.name);
                                        return a.isDirectory() ? -1 : 1;
                                    })
                                    .map(f => `${f.isDirectory() ? '[DIR] ' : '[FILE]'} ${f.name}`)
                                    .join('\n');
                            } catch (e) {
                                rootFiles = `(Could not read directory ${path})`;
                            }
                            return toToolResult(`${inputTrace}\nSearch pattern missing or too broad. Here is the directory listing for '${path}' instead. Use 'ls' with a path to explore deeper, or 'search' with a specific pattern (e.g., '**/*api*.js').\n\n\`\`\`\n${rootFiles}\n\`\`\``, { result: rootFiles });
                        }
                        
                        // if pattern doesn't have directory wildcards but is just an extension, make it recursive
                        if (pattern.startsWith('*.') && !pattern.includes('/')) {
                            pattern = '**/' + pattern;
                        }
                        
                        // respect the path argument if provided
                        let searchPattern = pattern;
                        if (path && path !== '.' && path !== './') {
                            // Ensure the pattern is applied within the requested sub-path
                            searchPattern = path.endsWith('/') ? `${path}${pattern}` : `${path}/${pattern}`;
                        }
                        
                        const files = await vscode.workspace.findFiles(searchPattern, '{**/node_modules/**,**/.git/**,**/__pycache__/**,**/target/**}', 200);
                        const result = files.map(f => vscode.workspace.asRelativePath(f)).join('\n');
                        return toToolResult(`${inputTrace}\nSearch results for ${searchPattern}:\n\n\`\`\`\n${result || 'No results found.'}\n\`\`\``, { result });
                    } else {
                        return toToolResult(`${inputTrace}\nInvalid command. Use "ls", "read" or "search".`, { result: null, error: 'Invalid command' });
                    }
                } catch (error) {
                    return toToolResult(`Command execution failed: ${error.message}`, { result: null, error: error.message });
                }
            }
        });
};
