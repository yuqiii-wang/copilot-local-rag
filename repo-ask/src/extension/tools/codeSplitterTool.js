const { toToolResult } = require('./utils');
const path = require('path');
const fs = require('fs');

module.exports = function registerCodeSplitterTool(deps) {
    const { vscode } = deps;
    return vscode.lm.registerTool('repoask_code_splitter', {
        prepareInvocation(options) {
            return {
                invocationMessage: 'Searching for relevant classes/funcs via Langchain Tree-sitter CodeSplitter...',
                confirmationMessages: {
                    title: 'Search via CodeSplitter?',
                    message: `Search for ${options.input.query} using class/func context ${options.input.classesOrFuncs ? options.input.classesOrFuncs.join(', ') : ''}`
                }
            };
        },
        async invoke(options) {
            const { query, classesOrFuncs } = options.input;
            const workspaceFolder = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
            if (!workspaceFolder) {
                return toToolResult('No workspace folder open', { contents: null });
            }

            try {
                // Lazily require LangChain dependencies to speed up extension load
                const { RecursiveCharacterTextSplitter, SupportedTextSplitterLanguages } = require('@langchain/textsplitters');
                
                // Fallback to simpler glob search
                const glob = require('glob');
                
                // Collect search candidates
                const candidates = (classesOrFuncs && Array.isArray(classesOrFuncs) && classesOrFuncs.length > 0) 
                    ? classesOrFuncs 
                    : [query];

                // Gather some files
                let files = [];
                try {
                    files = glob.sync('**/*.{js,ts,jsx,tsx,py,java,cpp,go,rs}', { 
                        cwd: workspaceFolder, 
                        ignore: ['**/node_modules/**', '**/dist/**', '**/out/**', '**/.git/**', '**/build/**'],
                        nodir: true,
                        absolute: true
                    });
                } catch (e) {
                    return toToolResult(`Failed to search files: ${e.message}`, { contents: null, error: e.message });
                }

                if (files.length === 0) {
                    return toToolResult("No source files found in workspace.", { contents: null });
                }

                const maxSearchedFiles = 300;
                const relevantDocs = [];

                // Simple grep for the classes/funcs
                for (const file of files.slice(0, maxSearchedFiles)) {
                    try {
                        const content = fs.readFileSync(file, 'utf8');
                        const lowerContent = content.toLowerCase();
                        let isRelevant = false;
                        for (const cand of candidates) {
                            if (lowerContent.includes(cand.toLowerCase())) {
                                isRelevant = true;
                                break;
                            }
                        }
                        if (isRelevant) {
                            relevantDocs.push({ file, content });
                        }
                    } catch (e) {
                        // ignore unreadable files
                    }
                }

                if (relevantDocs.length === 0) {
                    return toToolResult("No source files found matching the proposed classes/funcs.", { contents: [] });
                }

                // Determine language heuristic and split using Tree-sitter based language splitters where applicable
                const extLanguageMap = {
                    '.js': "js",
                    '.ts': "ts",
                    '.jsx': "js",
                    '.tsx': "ts",
                    '.py': "python",
                    '.java': "java",
                    '.cpp': "cpp",
                    '.go': "go",
                    '.rs': "rust"
                };

                const results = [];
                const maxChunksReturned = 6;
                const maxDocsToSplit = 10;
                let chunksFound = 0;

                for (const doc of relevantDocs.slice(0, maxDocsToSplit)) {
                    const ext = path.extname(doc.file).toLowerCase();
                    const lang = extLanguageMap[ext];
                    
                    let chunks = [];
                    // Use LangChain splitters
                    if (lang) {
                        try {
                            const splitter = RecursiveCharacterTextSplitter.fromLanguage(lang, {
                                chunkSize: 1500,
                                chunkOverlap: 200,
                            });
                            chunks = await splitter.createDocuments([doc.content]);
                        } catch (err) {
                            // If language splitting fails, fall back to basic text splitter
                            const fallbackSplitter = new RecursiveCharacterTextSplitter({
                                chunkSize: 1500,
                                chunkOverlap: 200,
                            });
                            chunks = await fallbackSplitter.createDocuments([doc.content]);
                        }
                    } else {
                        const fallbackSplitter = new RecursiveCharacterTextSplitter({
                            chunkSize: 1500,
                            chunkOverlap: 200,
                        });
                        chunks = await fallbackSplitter.createDocuments([doc.content]);
                    }

                    // Filter chunks that actually mention the query or classes
                    for (const [index, chunk] of chunks.entries()) {
                        let hit = false;
                        for (const cand of candidates) {
                            if (chunk.pageContent.toLowerCase().includes(cand.toLowerCase())) {
                                hit = true;
                                break;
                            }
                        }
                        if (query && chunk.pageContent.toLowerCase().includes(query.toLowerCase())) {
                            hit = true;
                        }

                        if (hit) {
                            const relPath = path.relative(workspaceFolder, doc.file);
                            results.push(`## ${relPath} (Chunk ${index + 1})\n\`\`\`\n${chunk.pageContent}\n\`\`\``);
                            chunksFound++;

                            if (chunksFound >= maxChunksReturned) {
                                break;
                            }
                        }
                    }
                    if (chunksFound >= maxChunksReturned) {
                        break;
                    }
                }

                if (results.length === 0) {
                    return toToolResult("Files matched, but no specific chunks contained the search context after splitting.", { contents: [] });
                }

                const finalOutput = results.join('\n\n');
                return toToolResult(`Found relevant code chunks:\n\n${finalOutput}`, { contents: results });

            } catch (error) {
                return toToolResult(`Failed to run CodeSplitter: ${error.message}`, { contents: null, error: error.message });
            }
        }
    });
};
