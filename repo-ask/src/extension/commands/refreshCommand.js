module.exports = function createRefreshCommand(deps) {
    const {
        vscode,
        documentService,
        sidebar,
        storagePath,
        readAllMetadata,
        readDocumentContent,
        writeDocumentFiles
    } = deps;
    const { parseRefreshArg } = require('../tools/llm');

    function normalizeReferencedQueries(value) {
        if (Array.isArray(value)) {
            return value.map(item => String(item || '').trim()).filter(Boolean);
        }
        if (typeof value === 'string') {
            return value.split(',').map(item => item.trim()).filter(Boolean);
        }
        return [];
    }

    function extractConfluenceIdFromLink(link) {
        const raw = String(link || '').trim();
        if (!raw) {
            return '';
        }

        const pageIdMatch = raw.match(/[?&]pageId=(\d+)/i);
        if (pageIdMatch && pageIdMatch[1]) {
            return pageIdMatch[1];
        }

        const pathIdMatch = raw.match(/\/pages\/(\d+)/i);
        if (pathIdMatch && pathIdMatch[1]) {
            return pathIdMatch[1];
        }

        return '';
    }

    function mergeReferencedQueriesIntoMetadata(target, sourceQueries, metadataList) {
        if (!target || !sourceQueries || sourceQueries.size === 0) {
            return false;
        }

        const targetText = String(target).trim();
        if (!targetText) {
            return false;
        }

        const allMetadata = Array.isArray(metadataList) ? metadataList : [];
        const normalizedTarget = targetText.replace(/\/$/, '');
        const confluenceIdFromLink = extractConfluenceIdFromLink(targetText);

        let docMetadata = allMetadata.find(doc => String(doc?.id || '').trim() === targetText) || null;
        if (!docMetadata && confluenceIdFromLink) {
            docMetadata = allMetadata.find(doc => String(doc?.id || '').trim() === confluenceIdFromLink) || null;
        }
        if (!docMetadata && /^https?:\/\//i.test(targetText)) {
            docMetadata = allMetadata.find(doc => {
                const docUrl = String(doc?.url || '').trim().replace(/\/$/, '');
                return docUrl && docUrl === normalizedTarget;
            }) || null;
        }

        if (!docMetadata || !docMetadata.id || typeof readDocumentContent !== 'function' || typeof writeDocumentFiles !== 'function') {
            return false;
        }

        const content = readDocumentContent(storagePath, docMetadata.id);
        if (!content) {
            return false;
        }

        const existingReferencedQueries = normalizeReferencedQueries(docMetadata.referencedQueries);
        const mergedReferencedQueries = [...existingReferencedQueries];

        for (const query of sourceQueries) {
            const normalizedQuery = String(query || '').trim();
            if (!normalizedQuery) {
                continue;
            }
            if (!mergedReferencedQueries.includes(normalizedQuery)) {
                mergedReferencedQueries.push(normalizedQuery);
            }
        }

        if (mergedReferencedQueries.length === existingReferencedQueries.length) {
            return false;
        }

        writeDocumentFiles(storagePath, docMetadata.id, content, {
            ...docMetadata,
            referencedQueries: mergedReferencedQueries
        });

        return true;
    }

    return vscode.commands.registerCommand('repo-ask.refresh', async (arg) => {
        // Extract fullIndexRefresh flag if provided
        let fullIndexRefresh = false;
        let refreshArg = arg;
        
        if (typeof arg === 'object' && arg !== null) {
            fullIndexRefresh = arg.fullIndexRefresh || false;
            // Preserve structured command payloads for special refresh modes.
            if (arg.type === 'single') {
                refreshArg = arg.arg;
            } else if (arg.type === 'all') {
                refreshArg = undefined;
            } else {
                refreshArg = arg;
            }
        }
        
        sidebar.setSidebarSyncStatus('Downloading from source...');
        
        // Set up timeout
        const timeoutPromise = new Promise((_, reject) => {
            setTimeout(() => {
                reject(new Error('Refresh operation timed out after 10 seconds'));
            }, 10000); // 10-second timeout
        });
        
        try {
            let result;
            if (typeof refreshArg === 'object' && refreshArg.type === 'feedback') {
                // Handle feedback sync: load reference queries by ID first, then by links
                sidebar.setSidebarSyncStatus('Loading reference queries from feedback...');
                
                // Get all metadata to check for existing documents
                const allMetadata = readAllMetadata(storagePath);
                const existingIds = new Set(allMetadata.map(doc => String(doc.id)));
                
                // Track feedback targets and source-query mapping for metadata updates.
                const referenceQueries = new Set();
                const sourceQueriesByTarget = new Map();
                
                try {
                    // Get the feedback Confluence page URL from configuration
                    const configuration = vscode.workspace.getConfiguration('repoAsk');
                    const feedbackUrl = configuration.get('logActionConfluenceUrl');
                    
                    if (feedbackUrl) {
                        // Fetch the feedback page content
                        const axios = require('axios');
                        const cheerio = require('cheerio');
                        
                        const response = await axios.get(feedbackUrl, {
                            timeout: 10000,
                            maxContentLength: Infinity,
                            maxBodyLength: Infinity
                        });
                        
                        const $ = cheerio.load(response.data);
                        
                        // Extract feedback data from the table
                        $('table tbody tr').each((index, row) => {
                            let sourceQuery = '';
                            let confluencePageId = '';
                            let jiraId = '';
                            let confluenceLink = '';
                            
                            // Extract data from each row
                            $(row).find('li').each((i, li) => {
                                const text = $(li).text();
                                if (text.includes('Source Query:')) {
                                    sourceQuery = text.replace('Source Query:', '').trim();
                                } else if (text.includes('Confluence Page ID:')) {
                                    confluencePageId = text.replace('Confluence Page ID:', '').trim();
                                } else if (text.includes('Jira ID:')) {
                                    jiraId = text.replace('Jira ID:', '').trim();
                                } else if (text.includes('Confluence/Jira Link:')) {
                                    confluenceLink = text.replace('Confluence/Jira Link:', '').trim();
                                }
                            });
                            
                            const refreshTarget = String(confluencePageId || jiraId || confluenceLink || '').trim();
                            if (refreshTarget) {
                                referenceQueries.add(refreshTarget);

                                if (sourceQuery) {
                                    const currentQueries = sourceQueriesByTarget.get(refreshTarget) || new Set();
                                    currentQueries.add(sourceQuery);
                                    sourceQueriesByTarget.set(refreshTarget, currentQueries);
                                }
                            }
                        });
                    }
                } catch (error) {
                    console.error('Error fetching feedback from Confluence:', error);
                    // Fallback to existing reference queries
                    allMetadata.forEach(doc => {
                        if (Array.isArray(doc.referencedQueries)) {
                            doc.referencedQueries.forEach(query => referenceQueries.add(query));
                        }
                    });
                }
                
                const queriesArray = Array.from(referenceQueries);
                const total = queriesArray.length;
                let referencedQueryUpdates = 0;
                
                if (total > 0) {
                    for (let i = 0; i < total; i++) {
                        const query = queriesArray[i];
                        sidebar.setSidebarSyncStatus(`Loading reference queries... (${i + 1}/${total})`);
                        const refreshedDocIds = new Set();
                        const collectProcessedDocId = (payload) => {
                            const processedId = String(payload?.metadata?.id || '').trim();
                            if (processedId) {
                                refreshedDocIds.add(processedId);
                                existingIds.add(processedId);
                            }
                        };
                        
                        try {
                            // First, try to load by ID (if query is an ID)
                            if (/^\d+$/.test(query) || /^[A-Z]+-\d+$/.test(query)) {
                                if (!existingIds.has(query)) {
                                    // Try to refresh by ID
                                    try {
                                        await documentService.refreshDocument(query, {
                                            onDocumentProcessed: collectProcessedDocId
                                        });
                                        existingIds.add(query);
                                    } catch (e) {
                                        console.error(`Failed to refresh by ID ${query}:`, e);
                                        // If ID refresh fails, try as link
                                        try {
                                            await documentService.refreshDocument(query, {
                                                onDocumentProcessed: collectProcessedDocId
                                            });
                                        } catch (e2) {
                                            console.error(`Failed to refresh by link ${query}:`, e2);
                                        }
                                    }
                                }
                            } else {
                                // Try as link
                                try {
                                    await documentService.refreshDocument(query, {
                                        onDocumentProcessed: collectProcessedDocId
                                    });
                                } catch (e) {
                                    console.error(`Failed to refresh by link ${query}:`, e);
                                }
                            }

                            const sourceQueries = sourceQueriesByTarget.get(query);
                            if (sourceQueries && sourceQueries.size > 0) {
                                const latestMetadata = readAllMetadata(storagePath);
                                let didUpdate = false;

                                // Prefer IDs returned by refresh so URL-based feedback always maps to local metadata.
                                for (const refreshedId of refreshedDocIds) {
                                    if (mergeReferencedQueriesIntoMetadata(refreshedId, sourceQueries, latestMetadata)) {
                                        didUpdate = true;
                                    }
                                }

                                if (!didUpdate && mergeReferencedQueriesIntoMetadata(query, sourceQueries, latestMetadata)) {
                                    didUpdate = true;
                                }

                                if (didUpdate) {
                                    referencedQueryUpdates += 1;
                                }
                            }
                        } catch (e) {
                            console.error(`Error processing reference query ${query}:`, e);
                        }
                    }
                }
                
                if (referencedQueryUpdates > 0) {
                    sidebar.setSidebarSyncStatus(`Reference queries loaded successfully (${referencedQueryUpdates} metadata updates)`);
                } else {
                    sidebar.setSidebarSyncStatus('Reference queries loaded successfully');
                }
                await new Promise(resolve => setTimeout(resolve, 1000)); // Brief delay to show status
            } else if (typeof refreshArg === 'object' && refreshArg.type === 'recursive' && refreshArg.arg) {
                result = await Promise.race([
                    documentService.refreshConfluenceHierarchy(refreshArg.arg, {
                        onDocumentProcessed: (data) => {
                            sidebar.setSidebarSyncStatus(`Downloading from source... (${data.index}/${data.total})`);
                        }
                    }),
                    timeoutPromise
                ]);
            } else if (refreshArg) {
                const parsedArg = await parseRefreshArg(vscode, refreshArg);
                if (parsedArg.source === 'regex-jira') {
                    result = await Promise.race([
                        documentService.refreshJiraIssue(refreshArg, {
                            onDocumentProcessed: (data) => {
                                sidebar.setSidebarSyncStatus(`Downloading from source... (${data.index}/${data.total})`);
                            }
                        }),
                        timeoutPromise
                    ]);
                } else {
                    result = await Promise.race([
                        documentService.refreshDocument(refreshArg, {
                            onDocumentProcessed: (data) => {
                                sidebar.setSidebarSyncStatus(`Downloading from source... (${data.index}/${data.total})`);
                            }
                        }),
                        timeoutPromise
                    ]);
                }
            } else {
                result = await Promise.race([
                    documentService.refreshAllDocuments({
                        onDocumentProcessed: (data) => {
                            sidebar.setSidebarSyncStatus(`Downloading from source... (${data.index}/${data.total})`);
                        }
                    }),
                    timeoutPromise
                ]);
            }
            
            // Trigger full index refresh if requested
            if (fullIndexRefresh) {
                sidebar.setSidebarSyncStatus('Refreshing document index...');
                // Rebuild indexes
                const metadataList = readAllMetadata(storagePath);
                if (metadataList.length > 0) {
                    // This will rebuild the keywords index from metadata
                    documentService.rankLocalDocuments('');
                }
                sidebar.setSidebarSyncStatus('Document index refreshed successfully');
                await new Promise(resolve => setTimeout(resolve, 1000)); // Brief delay to show status
            }
            
            sidebar.setSidebarSyncStatus('');
            sidebar.setSidebarSyncError('');
            sidebar.refreshSidebarView();
        } catch (error) {
            sidebar.setSidebarSyncStatus('');
            sidebar.setSidebarSyncError(error.message);
            vscode.window.showErrorMessage(`Refresh failed: ${error.message}`);
        }
    });
};
