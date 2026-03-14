const { toToolResult } = require('./utils');

module.exports = function registerReadContentTool(deps) {
    const { vscode, readAllMetadata, readDocumentContent } = deps;
    return vscode.lm.registerTool('repoask_read_content', {
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

        };
