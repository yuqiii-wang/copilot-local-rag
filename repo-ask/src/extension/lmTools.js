function createLanguageModelTools(deps) {
    const {
        vscode,
        context,
        documentService,
        parseRefreshArg,
        fetchConfluencePage,
        setSidebarSyncStatus,
        refreshSidebarView,
        upsertSidebarDocument,
        readAllMetadata,
        readDocumentContent,
        truncate,
        emptyStoreHint,
        toolNames
    } = deps;

    function toToolResult(text, data) {
        const parts = [new vscode.LanguageModelTextPart(String(text || ''))];
        if (vscode.LanguageModelDataPart && typeof vscode.LanguageModelDataPart.json === 'function' && data !== undefined) {
            parts.push(vscode.LanguageModelDataPart.json(data));
        }
        return new vscode.LanguageModelToolResult(parts);
    }

    function buildCheckAllDocsCommandLink(query) {
        const question = String(query || '').trim();
        if (!question) {
            return 'Run `repo-ask.checkAllDocs` to scan all docs.';
        }

        const encodedArgs = encodeURIComponent(JSON.stringify([question]));
        return `[Check ALL docs](command:repo-ask.checkAllDocs?${encodedArgs})`;
    }

    function formatRefreshStatus(sourceLabel, progress = {}) {
        const index = Number(progress?.index);
        const total = Number(progress?.total);
        const hasFraction = Number.isFinite(index) && Number.isFinite(total) && total > 0;
        const progressSuffix = hasFraction ? ` (${index}/${total})` : '';
        return `downloading from ${sourceLabel} ...${progressSuffix}`;
    }

    function registerRepoAskLanguageModelTools() {
        if (!vscode.lm || typeof vscode.lm.registerTool !== 'function') {
            return [];
        }

        const rankTool = vscode.lm.registerTool(toolNames.rank, {
            async invoke(options) {
                const query = String(options?.input?.query || '').trim();
                const rawLimit = Number(options?.input?.limit);
                const limit = Number.isFinite(rawLimit) && rawLimit > 0 ? Math.min(Math.floor(rawLimit), 50) : 5;

                if (!query) {
                    return toToolResult('Missing required `query` input for rank tool.', { results: [] });
                }

                const ranked = documentService.rankLocalDocuments(query, limit);
                if (!ranked || ranked.length === 0) {
                    return toToolResult('No matching local documents found for the query.', { results: [] });
                }

                const repAskConfig = vscode.workspace.getConfiguration('repoAsk');
                const confProfile = repAskConfig.get('confluence');
                const confUrl = String((confProfile && typeof confProfile === 'object' ? confProfile.url : '') || 'http://127.0.0.1:8001').replace(/\/$/, '');
                
                const jiraProfile = repAskConfig.get('jira');
                const jiraUrl = String((jiraProfile && typeof jiraProfile === 'object' ? jiraProfile.url : '') || 'http://127.0.0.1:8002').replace(/\/$/, '');

                const results = ranked.map(item => {
                    let fullUrl = item.url || '';
                    if (fullUrl && !fullUrl.startsWith('http')) {
                        const isJira = item.parent_confluence_topic && String(item.parent_confluence_topic).startsWith('Jira');
                        const baseUrl = isJira ? jiraUrl : confUrl;
                        fullUrl = `${baseUrl}${fullUrl.startsWith('/') ? '' : '/'}${fullUrl}`;
                    }
                    return {
                        ...item,
                        id: item.id,
                        title: item.title || 'Untitled',
                        url: fullUrl || 'None',
                        score: Number(item.score?.toFixed ? item.score.toFixed(4) : item.score),
                        summary: item.summary || '',
                        author: item.author || 'Unknown',
                        last_updated: item.last_updated || 'Unknown',
                        parent_confluence_topic: item.parent_confluence_topic || 'None',
                        keywords: item.keywords || []
                    };
                });
                const lines = results.map((item, index) => `${index + 1}. ${item.title} (score ${item.score})`);
                return toToolResult(`Top ranked RepoAsk documents:\n${lines.join('\n')}`, { results });
            }
        });

        const checkTool = vscode.lm.registerTool(toolNames.check, {
            async invoke(options) {
                const query = String(options?.input?.query || '').trim();
                const repAskConfig = vscode.workspace.getConfiguration('repoAsk');
                const initKeywordNum = repAskConfig.get('initKeywordNum') || 50;
                const rawLimit = Number(options?.input?.limit);
                const limit = Number.isFinite(rawLimit) && rawLimit > 0 ? Math.min(Math.floor(rawLimit), initKeywordNum) : 5;

                if (!query) {
                    return toToolResult('Missing required `query` input for check tool.', { references: [] });
                }

                const metadataList = readAllMetadata();
                if (metadataList.length === 0) {
                    return toToolResult(emptyStoreHint, { references: [] });
                }

                const agenticResult = documentService.checkLocalDocumentsAgentic(query, {
                    limit,
                    metadataCandidateLimit: Math.max(40, limit * 4)
                });

                if (!agenticResult.references || agenticResult.references.length === 0) {
                    return toToolResult(`No relevant documents found in local store. ${buildCheckAllDocsCommandLink(query)}`, { references: [] });
                }

                const confidentRefs = agenticResult.references.filter(r => r.score > 0);
                if (confidentRefs.length === 0) {
                    return toToolResult(`No confident local documents found for your query. Please ${buildCheckAllDocsCommandLink(query)}`, { references: [] });
                }

                const references = confidentRefs.map((ref) => ({
                    ...ref,
                    summary: ref.summary || 'No summary available',
                    reference: ref.reference || ''
                }));
                const lines = references.map((ref, index) => `${index + 1}. ${ref.title} (updated ${ref.last_updated || '-'})`);
                const summaryLines = [
                    `Top relevant RepoAsk references (agentic check):`,
                    `- Metadata scanned: ${agenticResult.metadataScanned}`,
                    `- Metadata candidates loaded for content: ${agenticResult.metadataCandidates}`,
                    `- Docs with content loaded: ${agenticResult.contentLoaded}`,
                    `- Metadata fallback used: ${agenticResult.usedMetadataFallback ? 'yes' : 'no'}`,
                    '',
                    ...lines,
                    '',
                    `Need broader confirmation? ${buildCheckAllDocsCommandLink(query)}`
                ];

                return toToolResult(summaryLines.join('\n'), {
                    references,
                    diagnostics: {
                        metadataScanned: agenticResult.metadataScanned,
                        metadataCandidates: agenticResult.metadataCandidates,
                        contentLoaded: agenticResult.contentLoaded,
                        usedMetadataFallback: agenticResult.usedMetadataFallback
                    }
                });
            }
        });

        const readMetadataTool = vscode.lm.registerTool('repoask_read_metadata', {
            prepareInvocation(options) {
                const ids = options?.input?.ids || [];
                return {
                    invocationMessage: ids.length > 0 ? `Reading metadata for ${ids.length} docs...` : 'Reading all metadata...'
                };
            },
            async invoke(options) {
                const ids = options?.input?.ids || [];
                const allMetadata = readAllMetadata();
                let filtered = allMetadata;
                if (ids.length > 0) {
                    filtered = allMetadata.filter(m => 
                        ids.includes(String(m.id)) || 
                        ids.includes(m.id) ||
                        ids.includes(m.title) ||
                        ids.some(id => String(m.title).includes(String(id)))
                    );
                }

                const repAskConfig = vscode.workspace.getConfiguration('repoAsk');
                const confProfile = repAskConfig.get('confluence');
                const confUrl = String((confProfile && typeof confProfile === 'object' ? confProfile.url : '') || 'http://127.0.0.1:8001').replace(/\/$/, '');
                
                const jiraProfile = repAskConfig.get('jira');
                const jiraUrl = String((jiraProfile && typeof jiraProfile === 'object' ? jiraProfile.url : '') || 'http://127.0.0.1:8002').replace(/\/$/, '');

                const summaryLines = filtered.map(m => {
                    let fullUrl = m.url || '';
                    if (fullUrl && !fullUrl.startsWith('http')) {
                        const isJira = m.parent_confluence_topic && String(m.parent_confluence_topic).startsWith('Jira');
                        const baseUrl = isJira ? jiraUrl : confUrl;
                        fullUrl = `${baseUrl}${fullUrl.startsWith('/') ? '' : '/'}${fullUrl}`;
                    }

                    const lines = [
                        `- [${m.id}] ${m.title || 'Untitled'}`,
                        `  URL: ${fullUrl || 'None'}`,
                        `  Jira ID / Confluence Title: ${m.title || 'Untitled'}`,
                        `  Author: ${m.author || 'Unknown'}`,
                        `  Last Updated: ${m.last_updated || 'Unknown'}`,
                        `  Parent Topic: ${m.parent_confluence_topic || 'None'}`,
                        `  Keywords: ${Array.isArray(m.keywords) ? m.keywords.join(', ') : 'None'}`,
                        `  Summary: ${m.summary || 'None'}`
                    ];
                    return lines.join('\n');
                });
                return toToolResult(`Found metadata for ${filtered.length} docs:\n${summaryLines.join('\n\n')}`, { metadata: filtered });
            }
        });

        const readContentTool = vscode.lm.registerTool('repoask_read_content', {
            prepareInvocation(options) {
                const ids = options?.input?.ids || [];
                return {
                    invocationMessage: ids.length > 0 ? `Reading content for ${ids.length} docs...` : 'Reading all content...'
                };
            },
            async invoke(options) {
                const ids = options?.input?.ids || [];
                const allMetadata = readAllMetadata();
                let filtered = allMetadata;
                if (ids.length > 0) {
                    filtered = allMetadata.filter(m => 
                        ids.includes(String(m.id)) || 
                        ids.includes(m.id) ||
                        ids.includes(m.title) ||
                        ids.some(id => String(m.title).includes(String(id)))
                    );
                }
                const results = [];
                for (const m of filtered) {
                    const content = readDocumentContent(m.id);
                    if (content) {
                        results.push({ id: m.id, title: m.title, content: content });
                    }
                }
                const summaryLines = results.map(r => `Doc [${r.id}] ${r.title}:\n${r.content}`);
                return toToolResult(`Found content for ${results.length} docs:\n\n${summaryLines.join('\n\n')}`, { contents: results });
            }
        });

        const newCodeCheckTool = vscode.lm.registerTool('repoask_new_code_check', {
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
                    if (!diff || !diff.trim()) {
                        return toToolResult('No code changes found compared to main/master.', { diff: '' });
                    }
                    return toToolResult(`Git diff:\n\n\`\`\`diff\n${diff}\n\`\`\``, { diff });
                } catch (error) {
                    return toToolResult(`Code check failed: ${error.message}`, { diff: null, error: error.message });
                }
            }
        });

        const readRepoPromptsTool = vscode.lm.registerTool('repoask_read_repo_prompts', {
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

        return [rankTool, checkTool, readMetadataTool, readContentTool, newCodeCheckTool, readRepoPromptsTool];
    }

    return {
        registerRepoAskLanguageModelTools
    };
}

module.exports = {
    createLanguageModelTools
};
