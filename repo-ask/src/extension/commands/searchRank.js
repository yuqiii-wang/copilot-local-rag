function registerCheckAndRankCommands(deps) {
    const {
        vscode,
        storagePath,
        documentService,
        readAllMetadata,
        readDocumentContent,
        findRelevantDocuments,
        tokenize,
        truncate
    } = deps;

    const checkDisposable = vscode.commands.registerCommand('repo-ask.check', async function (query) {
        const question = query || await vscode.window.showInputBox({
            prompt: 'Enter your question to check relevant documents',
            placeHolder: 'e.g., How to create a new Confluence page?'
        });

        if (!question) {
            return;
        }

        try {
            const metadataList = readAllMetadata(storagePath);
            if (metadataList.length === 0) {
                vscode.window.showInformationMessage('No local documents found. Use the sidebar to sync to Confluence Cloud.');
                return;
            }

            const relevantDocs = findRelevantDocuments(question, metadataList, tokenize);
            if (relevantDocs.length === 0) {
                vscode.window.showInformationMessage('No relevant documents found');
                return;
            }

            const items = relevantDocs.map(doc => ({
                label: doc.title,
                description: `Last updated: ${doc.last_updated}`,
                detail: truncate(doc.summary || 'No summary available', 120),
                doc
            }));

            const selected = await vscode.window.showQuickPick(items, {
                placeHolder: 'Select a document to view local reference content'
            });

            if (!selected) {
                return;
            }

            await vscode.commands.executeCommand('repo-ask.openDocumentDetails', selected.doc);
        } catch (error) {
            vscode.window.showErrorMessage(`Error checking documents: ${error.message}`);
        }
    });

    const rankDisposable = vscode.commands.registerCommand('repo-ask.rank', async function (directQuery) {
        const query = typeof directQuery === 'string' ? directQuery : await vscode.window.showInputBox({
            prompt: 'Enter keywords to rank local documents',
            placeHolder: 'e.g., oauth token refresh'
        });

        if (!query || query.trim().length === 0) {
            return;
        }

        try {
            const rankedDocs = documentService.rankLocalDocuments(query.trim(), 10);
            if (rankedDocs.length === 0) {
                vscode.window.showInformationMessage('No matching local documents found for the query.');
                return;
            }

            const items = rankedDocs.map(doc => ({
                label: doc.title || 'Untitled',
                description: `BM25 score: ${doc.score.toFixed(2)}`,
                detail: truncate(doc.summary || 'No summary available', 120),
                doc
            }));

            const selected = await vscode.window.showQuickPick(items, {
                placeHolder: 'Select a ranked document'
            });

            if (!selected) {
                return;
            }

            await vscode.commands.executeCommand('repo-ask.openDocumentDetails', selected.doc);
        } catch (error) {
            vscode.window.showErrorMessage(`Error ranking documents: ${error.message}`);
        }
    });

    const checkAllDocsDisposable = vscode.commands.registerCommand('repo-ask.checkAllDocs', async function (directQuery) {
        const query = typeof directQuery === 'string' ? directQuery : await vscode.window.showInputBox({
            prompt: 'Enter your question to check ALL local docs',
            placeHolder: 'e.g., summarize deployment rollback process'
        });

        const normalizedQuery = String(query || '').trim();
        if (!normalizedQuery) {
            return;
        }

        try {
            const metadataList = readAllMetadata(storagePath);
            if (metadataList.length === 0) {
                vscode.window.showInformationMessage('No local documents found. Use the sync button in the sidebar first.');
                return;
            }

            vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: "Checking all documents (Agentic RAG)",
                cancellable: false
            }, async (progress) => {
                try {
                    progress.report({ message: "Selecting chat model..." });
                    const models = await vscode.lm.selectChatModels({});
                    const model = models?.[0];
                    if (!model) {
                        vscode.window.showErrorMessage('No language model available in this VS Code session.');
                        return;
                    }

                    progress.report({ message: "Analyzing metadata for relevant files..." });
                    let metadataContext = '';
                    try {
                        metadataContext = metadataList.map(doc => {
                            return JSON.stringify({
                                id: doc.id,
                                title: doc.title,
                                summary: doc.summary,
                                keywords: doc.keywords
                            });
                        }).join('\n');
                    } catch(e) {
                         vscode.window.showErrorMessage('Error stringifying metadata context');
                         return;
                    }

                    const metaInstruction = [
                        'You are an agent deciding which documents are most relevant to answer the user query.',
                        'Review the following documents (one JSON object per line) and pick the IDs that might contain the answer.',
                        'Here are the available documents:',
                        metadataContext,
                        '',
                        `User query: ${normalizedQuery}`,
                        '',
                        'Respond ONLY with a raw JSON array format containing the string IDs of the most relevant documents (max 5).',
                        'Example output:',
                        '["12345", "67890"]',
                        'Do not use trailing commas, do not add markdown code blocks, do not add any other text.'
                    ].join('\n');

                    const metaResponse = await model.sendRequest([
                        vscode.LanguageModelChatMessage.User(metaInstruction)
                    ], {}, new vscode.CancellationTokenSource().token);

                    let metaText = '';
                    for await (const chunk of metaResponse.text) {
                        metaText += chunk;
                    }

                    let docIds = [];
                    try {
                        let cleanText = metaText.replace(/```json/gi, '').replace(/```/g, '').trim();
                        const jsonMatch = cleanText.match(/\[[\s\S]*\]/);
                        if (jsonMatch) {
                            docIds = JSON.parse(jsonMatch[0]);
                        } else {
                            docIds = JSON.parse(cleanText);
                        }
                    } catch (e) {
                        console.error('Failed to parse doc IDs:', e, 'Raw:', metaText);
                        docIds = [];
                    }

                    if (!Array.isArray(docIds) || docIds.length === 0) {
                        vscode.window.showInformationMessage('The LLM did not identify any relevant documents from the metadata.');
                        return;
                    }

                    const validDocIds = [...new Set(docIds.map(String))];

                    progress.report({ message: `Loading full content for ${validDocIds.length} docs and generating final answer...` });
                    
                    const contentContext = validDocIds.map(id => {
                        const metadata = metadataList.find(d => String(d.id) === id) 
                                      || metadataList.find(d => String(d.title) === id); 
                        const actualId = metadata ? metadata.id : id;
                        const content = metadata ? (readDocumentContent(storagePath, actualId) || '') : '';
                        return `## Doc: ${metadata?.title || id}\nID: ${actualId}\n\n${content}`;
                    }).join('\n\n');

                    const finalInstruction = [
                        'You are a helpful assistant answering the user query based ONLY on the provided documents.',
                        'If the answer is not in the documents, say so.',
                        '',
                        'Documents:',
                        contentContext,
                        '',
                        `User query: ${normalizedQuery}`
                    ].join('\n');

                    const finalResponse = await model.sendRequest([
                        vscode.LanguageModelChatMessage.User(finalInstruction)
                    ], {}, new vscode.CancellationTokenSource().token);

                    let finalText = '';
                    for await (const chunk of finalResponse.text) {
                        finalText += chunk;
                    }

                    const reportSections = [
                        `# Query: ${normalizedQuery}`,
                        '',
                        `**Evaluated Metadata Files**: ${metadataList.length}`,
                        `**Relevant Documents Picked by LLM**: ${docIds.length} (${docIds.join(', ')})`,
                        '',
                        '## Final Answer',
                        finalText
                    ];

                    const document = await vscode.workspace.openTextDocument({
                        language: 'markdown',
                        content: reportSections.join('\n')
                    });
                    await vscode.window.showTextDocument(document, { preview: false });

                } catch (err) {
                    vscode.window.showErrorMessage(`Agentic check failed: ${err.message}`);
                }
            });

        } catch (error) {
            vscode.window.showErrorMessage(`Error checking all documents: ${error.message}`);
        }
    });

    return [checkDisposable, rankDisposable, checkAllDocsDisposable];
}

module.exports = {
    registerCheckAndRankCommands
};
