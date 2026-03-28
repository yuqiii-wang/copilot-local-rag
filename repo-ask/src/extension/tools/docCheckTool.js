const { toToolResult, buildCheckAllDocsCommandLink } = require('./utils');
const { DOC_CHECK_TOOL_DESCRIPTION, ALLOWED_MODES, PARTIAL_CONTENT_NOTE } = require('../chat/prompts');

module.exports = function registerDocCheckTool(deps) {
    const { vscode, toolNames, documentService, readAllMetadata, readDocumentContent, emptyStoreHint } = deps;
    return vscode.lm.registerTool(toolNames.docCheck, {
            async invoke(options) {
                const mode = options?.input?.mode; // content, metadata, content_partial, metadata.summary, metadata.summary_kg, metadata.id
                const ids = options?.input?.ids || [];
                const searchTerms = options?.input?.searchTerms || [];
                const repAskConfig = vscode.workspace.getConfiguration('repoAsk');

                if (!ALLOWED_MODES.includes(mode)) {
                    return toToolResult(`Invalid mode '${mode}'. Allowed modes are: ${ALLOWED_MODES.join(', ')}`, { references: [] });
                }

                const metadataList = readAllMetadata();
                if (metadataList.length === 0) {
                    return toToolResult(emptyStoreHint, { references: [] });
                }

                let filtered = metadataList;

                // Filter by explicit IDs first
                if (ids && ids.length > 0) {
                    filtered = metadataList.filter(m => 
                        ids.includes(String(m.id)) || 
                        ids.includes(m.id) ||
                        ids.includes(m.title) ||
                        ids.some(id => String(m.title).includes(String(id)))
                    );
                } else if (searchTerms && searchTerms.length > 0) {
                    // No explicit IDs — filter by LLM-proposed search terms against title/keywords/summary
                    const lowerTerms = searchTerms.map(t => String(t).toLowerCase().trim()).filter(Boolean);
                    filtered = metadataList.filter(m => {
                        const haystack = [
                            m.title || '',
                            Array.isArray(m.keywords) ? m.keywords.join(' ') : (m.keywords || ''),
                            m.summary || ''
                        ].join(' ').toLowerCase();
                        return lowerTerms.some(term => haystack.includes(term));
                    });
                    // Fallback: if nothing matched, return all (LLM terms may be too specific)
                    if (filtered.length === 0) {
                        filtered = metadataList;
                    }
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
                        } else if (mode === 'metadata.summary_kg') {
                            result.metadata = { id: m.id, title: m.title, summary: m.summary, knowledgeGraph: m.knowledgeGraph };
                            summaryLines.push(`- [${m.id}] ${m.title}`);
                            if (m.summary) summaryLines.push(`  Summary: ${m.summary}`);
                            if (m.knowledgeGraph) summaryLines.push(`  KG:\n${m.knowledgeGraph}`);
                            summaryLines.push('');
                        } else if (mode === 'metadata.summary') {
                            result.metadata = { summary: m.summary, knowledgeGraph: m.knowledgeGraph };
                            summaryLines.push(`- [${m.id}] ${m.title}`);
                            if (m.summary) summaryLines.push(`  Summary: ${m.summary}`);
                            if (m.knowledgeGraph) summaryLines.push(`  Knowledge Graph:\n${m.knowledgeGraph}`);
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
                                
                                content = `${topPart}\n......\n${middlePart}\n......\n${bottomPart}\n\n${PARTIAL_CONTENT_NOTE}`;
                            }
                            result.content = content;
                            if (m.knowledgeGraph) {
                                result.knowledgeGraph = m.knowledgeGraph;
                            }
                            summaryLines.push(`Doc [${m.id}] ${m.title}:`);
                            summaryLines.push(content);
                            if (m.knowledgeGraph) {
                                summaryLines.push(`\nKnowledge Graph:\n${m.knowledgeGraph}`);
                            }
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
