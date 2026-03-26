const { toToolResult, buildCheckAllDocsCommandLink } = require('./utils');

module.exports = function registerDocCheckTool(deps) {
    const { vscode, toolNames, documentService, readAllMetadata, readDocumentContent, emptyStoreHint } = deps;
    return vscode.lm.registerTool(toolNames.docCheck, {
            async invoke(options) {
                const query = String(options?.input?.query || '').trim();
                const mode = options?.input?.mode; // content, metadata, content_partial, metadata.summary, metadata.id
                const ids = options?.input?.ids || [];
                const repAskConfig = vscode.workspace.getConfiguration('repoAsk');

                const allowedModes = ['content', 'metadata', 'content_partial', 'metadata.summary', 'metadata.id'];
                if (!allowedModes.includes(mode)) {
                    return toToolResult(`Invalid mode '${mode}'. Allowed modes are: ${allowedModes.join(', ')}`, { references: [] });
                }

                const metadataList = readAllMetadata();
                if (metadataList.length === 0) {
                    return toToolResult(emptyStoreHint, { references: [] });
                }

                let filtered = metadataList;
                if (ids && ids.length > 0) {
                    filtered = metadataList.filter(m => 
                        ids.includes(String(m.id)) || 
                        ids.includes(m.id) ||
                        ids.includes(m.title) ||
                        ids.some(id => String(m.title).includes(String(id)))
                    );
                }

                const confProfile = repAskConfig.get('confluence');
                const confUrl = String((confProfile && typeof confProfile === 'object' ? confProfile.url : '') || '').replace(/\/$/, '');
                
                const jiraProfile = repAskConfig.get('jira');
                const jiraUrl = String((jiraProfile && typeof jiraProfile === 'object' ? jiraProfile.url : '') || '').replace(/\/$/, '');

                const results = [];
                const summaryLines = [];
                
                for (const m of filtered) {
                    const result = { id: m.id, title: m.title };
                    
                    if (mode.startsWith('metadata')) {
                        let fullUrl = m.url || '';
                        if (fullUrl && !fullUrl.startsWith('http')) {
                            const isJira = m.parent_confluence_topic && String(m.parent_confluence_topic).startsWith('Jira');
                            const baseUrl = isJira ? jiraUrl : confUrl;
                            fullUrl = `${baseUrl}${fullUrl.startsWith('/') ? '' : '/'}${fullUrl}`;
                        }
                        
                        if (mode === 'metadata') {
                            result.metadata = {
                                url: fullUrl,
                                author: m.author,
                                last_updated: m.last_updated,
                                parent_confluence_topic: m.parent_confluence_topic,
                                keywords: m.keywords,
                                summary: m.summary,
                                knowledgeGraph: m.knowledgeGraph
                            };
                            summaryLines.push(`- [${m.id}] ${m.title}`);
                            if (fullUrl) summaryLines.push(`  URL: ${fullUrl}`);
                            if (m.author) summaryLines.push(`  Author: ${m.author}`);
                            if (m.last_updated) summaryLines.push(`  Last Updated: ${m.last_updated}`);
                            if (m.parent_confluence_topic) summaryLines.push(`  Parent Topic: ${m.parent_confluence_topic}`);
                            if (m.keywords) summaryLines.push(`  Keywords: ${Array.isArray(m.keywords) ? m.keywords.join(', ') : m.keywords}`);
                            if (m.summary) summaryLines.push(`  Summary: ${m.summary}`);
                            if (m.knowledgeGraph) summaryLines.push(`  Knowledge Graph:\n${m.knowledgeGraph}`);
                            summaryLines.push('');
                        } else if (mode === 'metadata.summary') {
                            result.metadata = { summary: m.summary };
                            summaryLines.push(`- [${m.id}] ${m.title}`);
                            if (m.summary) summaryLines.push(`  Summary: ${m.summary}`);
                            summaryLines.push('');
                        } else if (mode === 'metadata.id') {
                            result.metadata = { id: m.id, title: m.title };
                            summaryLines.push(`- [${m.id}] ${m.title}`);
                        }
                    } else if (mode.startsWith('content')) {
                        let content = readDocumentContent(m.id);
                        if (content) {
                            if (mode === 'content_partial' && content.length > 1500) {
                                const chunkSize = Math.min(500, Math.floor(content.length / 3));
                                const topPart = content.substring(0, chunkSize);
                                const middleStart = Math.floor(content.length / 2) - Math.floor(chunkSize / 2);
                                const middlePart = content.substring(middleStart, middleStart + chunkSize);
                                const bottomPart = content.substring(content.length - chunkSize);
                                
                                content = `${topPart}\n......\n${middlePart}\n......\n${bottomPart}\n\n[Note]: This is partial content. If the partial content is likely related to user query, MUST read full content. To read full content, instruct LLM to use mode 'content' to read full content.`;
                            }
                            result.content = content;
                            summaryLines.push(`Doc [${m.id}] ${m.title}:`);
                            summaryLines.push(content);
                            summaryLines.push('');
                        }
                    }
                    results.push(result);
                }

                if (mode.startsWith('metadata')) {
                    summaryLines.unshift(`Metadata for ${results.length} docs:`);
                } else if (mode.startsWith('content')) {
                    summaryLines.unshift(`Content for ${results.filter(r => r.content).length} docs:`);
                }

                return toToolResult(summaryLines.join('\n'), { results });
            }
        });
};
