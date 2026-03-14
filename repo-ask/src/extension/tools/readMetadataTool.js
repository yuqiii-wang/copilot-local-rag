const { toToolResult } = require('./utils');

module.exports = function registerReadMetadataTool(deps) {
    const { vscode, readAllMetadata } = deps;
    return vscode.lm.registerTool('repoask_read_metadata', {
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

        };
