const { toToolResult } = require('./utils');

module.exports = function registerRankTool(deps) {
    const { vscode, toolNames, documentService } = deps;
    return vscode.lm.registerTool(toolNames.rank, {
            async invoke(options) {
                const query = String(options?.input?.query || '').trim();
                const rawLimit = Number(options?.input?.limit);
                const limit = Number.isFinite(rawLimit) && rawLimit > 0 ? Math.min(Math.floor(rawLimit), 50) : 5;

                if (!query) {
                    return toToolResult('Missing required `query` input for rank tool.', { results: [] });
                }

                const ranked = documentService.rankLocalDocuments(query, limit);
                if (!ranked || ranked.length === 0) {
                    return toToolResult('No matching local documents found for the query. Please search the document store in the sidebar to find the document ID, then use that ID in your query to retrieve the specific document.', { results: [] });
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

        };
