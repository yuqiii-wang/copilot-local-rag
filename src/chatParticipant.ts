import * as vscode from 'vscode';
import { RequestManager } from './requestManager';

export function registerChatParticipant(context: vscode.ExtensionContext) {
    const handler: vscode.ChatRequestHandler = async (request: vscode.ChatRequest, context: vscode.ChatContext, stream: vscode.ChatResponseStream, token: vscode.CancellationToken) => {
        
        // Check for commands
        if (request.command === 'help') {
            stream.markdown('How can I help you? You can ask me to perform tasks or use tools.');
            return;
        }

        let promptText = request.prompt;
        
        // Try to recover recordId from history if not an explicit new request
        let currentRecordId: string | undefined;

        // Check if we are processing a request from Sidebar
        const match = promptText.match(/process-request\s+([a-z0-9]+)/);
        if (match) {
            const requestId = match[1];
            const reqManager = RequestManager.getInstance();
            const requestData = reqManager.getRequest(requestId);
            if (requestData) {
                currentRecordId = requestData.id;
            }
        } else {
             // Look in history for hidden RecordID
             for (let i = context.history.length - 1; i >= 0; i--) {
                 const turn = context.history[i];
                 if (turn instanceof vscode.ChatResponseTurn) {
                     // Check response chunks (it's async iterable or array, depending on API version. Stable is usually string or markdown)
                     // In the extension API, `response` is an array of `ChatResponsePart`.
                     for (const part of turn.response) {
                         // Check for command button with arguments (Metadata storage)
                         const cmdPart = part as any;
                         if (cmdPart.value && cmdPart.value.command === 'repo-ask.feedbackAccept' && cmdPart.value.arguments && cmdPart.value.arguments[0]) {
                              const args = cmdPart.value.arguments[0];
                              if (args.recordId) {
                                  currentRecordId = args.recordId;
                                  break;
                              }
                         }

                         if (part instanceof vscode.ChatResponseMarkdownPart) {
                             const idMatch = part.value.value.match(/<!-- RecordID: ([^ ]+) -->/);
                             if (idMatch) {
                                 currentRecordId = idMatch[1];
                                 break;
                             }
                         }
                     }
                 }
                 if (currentRecordId) break;
             }
        }

        if (match) {
            const requestId = match[1];
            const reqManager = RequestManager.getInstance();
            const requestData = reqManager.getRequest(requestId);

            if (requestData) {
                stream.progress('Retrieving context and thinking...');
                
                // Fetch download content
                let loadedContext = "";
                if (requestData.urls && requestData.urls.length > 0) {
                     try {
                        const config = vscode.workspace.getConfiguration('repo-ask');
                        const backendUrl = config.get<string>('backendUrl') || 'http://localhost:14321';

                        // Assuming fetch is globally available in Node environment or use node-fetch if needed.
                        const response = await fetch(`${backendUrl}/download/fetch`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ 
                                urls: requestData.urls,
                                query: requestData.manualText || "" 
                            })
                         });
    
                        if (response.ok) {
                            try {
                                // FIX: Read text first to prevent crashing on empty/malformed JSON
                                const responseText = await response.text(); 
                                
                                // Try parsing the text
                                const data = responseText ? JSON.parse(responseText) : [];

                                // FIX: Ensure 'downloads' is actually an array before mapping
                                // (Handle cases where API returns { downloads: [...] } or just [...])
                                const downloads = Array.isArray(data) ? data : (data.downloads || []);

                                if (Array.isArray(downloads)) {
                                    // Summarize documents that lack comments using the selected language model
                                    for (const d of downloads) {
                                        // Ensure we preserve comment from backend if present
                                        if (d.found && !d.comment) {
                                            try {
                                                // Try to find a model for summarization
                                                let [sumModel] = await vscode.lm.selectChatModels({ family: 'gpt-3.5' });
                                                if (!sumModel) [sumModel] = await vscode.lm.selectChatModels();

                                                if (sumModel) {
                                                    const sumMessages = [
                                                        // Use User role to supply system-like instruction if System factory is unavailable
                                                        vscode.LanguageModelChatMessage.User('You are a helpful assistant. Summarize the following document in 1-2 short sentences to be used as a document comment.'),
                                                        vscode.LanguageModelChatMessage.User(`Document content:\n\n${d.content.substring(0, 12000)}`)
                                                    ];

                                                    let summary = '';
                                                    try {
                                                        const sumResp = await sumModel.sendRequest(sumMessages, {}, token);
                                                        for await (const frag of sumResp.text) {
                                                            summary += frag;
                                                        }
                                                    } catch (se) {
                                                        console.error('Summarization error', se);
                                                    }

                                                    d.comment = (summary || '').trim();
                                                } else {
                                                    // Fallback: extract first non-empty line(s)
                                                    const firstLines = d.content.split(/\n/).map((l:any)=>l.trim()).filter((l:any)=>l).slice(0,2).join(' ');
                                                    d.comment = firstLines;
                                                }

                                                // If we have a recordId, push the comment back to the backend
                                                if (d.comment && requestData.id) {
                                                    try {
                                                        await fetch(`${backendUrl}/data/update_docs`, {
                                                            method: 'POST',
                                                            headers: { 'Content-Type': 'application/json' },
                                                            body: JSON.stringify({ query: { id: requestData.id, ref_docs: [{ source: d.url, comment: d.comment }] } })
                                                        });
                                                    } catch (ue) { console.error('Failed to update record with doc comment', ue); }
                                                }
                                            } catch (e) {
                                                console.error('Error summarizing document', e);
                                            }
                                        }
                                    }

                                    // Rebuild loadedContext including any comments
                                    loadedContext = downloads
                                        .map((d: any) => `Source: ${d.url}${d.comment ? ('\nComment: ' + d.comment) : ''}\nContent:\n${d.content}\n`)
                                        .join('\n---\n\n');
                                } else {
                                    console.error("Unexpected JSON structure:", data);
                                    stream.markdown('_(Backend returned unexpected data structure)_');
                                }
                            } catch (parseErr) {
                                console.error("JSON Parse Error:", parseErr);
                                stream.markdown('_(Failed to parse backend response)_');
                            }
                        } else {
                            stream.markdown(`_(Backend returned status ${response.status})_`);
                        }
                     } catch (err) {
                         stream.markdown(`_(Error downloading content: ${err})_`);
                     }
                }

                // Construct system prompt / context
                const systemPrompt = `
You are a helpful assistant. Use the following context to answer the user's question.

Found text from images:
${requestData.ocrText || "None"}

Downloaded Document Context:
${loadedContext || "None (Failed to download or no links provided)"}

User Query:
${requestData.manualText || "Please analyze the provided context."}
`;
                // Delegate to Copilot (Language Model)
                try {
                    // Try to find a model, prioritizing GPT-4, then GPT-3.5, then any
                    let [model] = await vscode.lm.selectChatModels({ family: 'gpt-4' });
                    
                    if (!model) {
                         // Fallback to any available model
                        [model] = await vscode.lm.selectChatModels(); 
                    }

                    if (model) {
                         const messages = [
                             vscode.LanguageModelChatMessage.User(systemPrompt)
                         ];
                         
                         let fullResponse = "";
                         const chatResponse = await model.sendRequest(messages, {}, token);
                         for await (const fragment of chatResponse.text) {
                             fullResponse += fragment;
                             stream.markdown(fragment);
                         }

                        const feedbackId = Date.now().toString() + Math.random().toString();
                        
                        // Pass the ORIGINAL record ID if available
                        const recordId = requestData.id || "";
                        
                        try {
                             if (recordId) {
                                 const config = vscode.workspace.getConfiguration('repo-ask');
                                 const backendUrl = config.get<string>('backendUrl') || 'http://localhost:14321';
                                 
                                 await fetch(`${backendUrl}/data/record_feedback`, {
                                     method: 'POST',
                                     headers: { 'Content-Type': 'application/json' },
                                     body: JSON.stringify({
                                        id: recordId,
                                        status: 'pending',
                                        conversations: [{ ai_assistant: fullResponse, human: requestData.manualText || "" }]
                                     })
                                 });
                             }
                        } catch (e) {
                            console.error("Failed to update record with AI response", e);
                        }

                        stream.button({
                            command: 'repo-ask.feedbackAccept',
                            title: 'Accept',
                            arguments: [{ query: "", ai_answer: "", feedbackId, recordId }]
                        });
                        stream.button({
                            command: 'repo-ask.feedbackReject',
                            title: 'Reject',
                            arguments: [{ query: "", ai_answer: "", feedbackId, recordId }]
                        });
                        stream.button({
                            command: 'repo-ask.feedbackConfluence',
                            title: 'Add to Confluence',
                            arguments: [{ query: "", ai_answer: "", feedbackId, recordId }]
                        });
                    } else {
                        stream.markdown("Sorry, I could not find a suitable Language Model to generate a response. Here is the data I received:\n\n" + (requestData.manualText || "No text"));
                    }
                } catch (err: any) {
                    stream.markdown(`Error calling Language Model: ${err.message}`);
                }
                
                // Cleanup - optional
                // reqManager.deleteRequest(requestId);
                return;
            } else {
                stream.markdown("Request ID not found or expired.");
                return;
            }
        }


        stream.progress('Thinking...');
        
        // General Chat Query handling
        try {
            let [model] = await vscode.lm.selectChatModels({ family: 'gpt-4' });
            if (!model) [model] = await vscode.lm.selectChatModels({ family: 'gpt-3.5' });
            if (!model) [model] = await vscode.lm.selectChatModels();

            if (model) {
                // Construct messages from history to preserve context
                const messages: vscode.LanguageModelChatMessage[] = [];
                
                // Reconstruct history
                const reqManager = RequestManager.getInstance();
                let initialRequestData = currentRecordId ? reqManager.getRequestByRecordId(currentRecordId) : undefined;
                
                // If we found the original request data, we can rebuild the system prompt context
                // even if the history only contains the "process-request ..." command text.
                
                let contextInjected = false;
                let fullHistory = "";

                for (const turn of context.history) {
                    if (turn instanceof vscode.ChatRequestTurn) {
                        const text = turn.prompt;
                        messages.push(vscode.LanguageModelChatMessage.User(text));
                        fullHistory += `User: ${text}\n\n`;
                    } else if (turn instanceof vscode.ChatResponseTurn) {
                        // Aggregate response text
                        let text = "";
                        for (const part of turn.response) {
                             if (part instanceof vscode.ChatResponseMarkdownPart) {
                                 const val = part.value;
                                 text += (typeof val === 'string') ? val : val.value;
                             }
                        }
                        messages.push(vscode.LanguageModelChatMessage.Assistant(text));
                        fullHistory += `Assistant: ${text}\n\n`;
                    }
                }

                // If we have initial context data, we should prepend it as a System or Context User message
                // This ensures the model sees the documents again.
                if (initialRequestData && messages.length > 0) {
                     // We need to re-download the content? Or assume it's lost if we didn't cache the TEXT string?
                     // RequestManager stores `ocrText`, `manualText`. `urls` implies we need to fetch.
                     // But we can't do async fetch in the message construction easily if we want to be fast?
                     // Actually, we can.
                     
                     // Optimization: If the FIRST message in `messages` resembles the `process-request` command,
                     // we should probably REPLACE it with the full context prompt we generated originally.
                     // But we don't have that full prompt string cached.
                     
                     // Optimization: Only refetch if we don't have cached context, or relying on short-term cache in RequestManager
                     // To avoid freezing UI on every turn, we could cache the `loadedContext` in `RequestData`.
                     
                     let loadedContext = "";
                     if (initialRequestData.urls && initialRequestData.urls.length > 0) {
                        // Check if we have manually cached it (hack: let's cast RequestData to any to store it)
                        if ((initialRequestData as any).cachedContext) {
                            loadedContext = (initialRequestData as any).cachedContext;
                        } else {
                             try {
                                const config = vscode.workspace.getConfiguration('repo-ask');
                                const backendUrl = config.get<string>('backendUrl') || 'http://localhost:14321';
                                const response = await fetch(`${backendUrl}/download/fetch`, {
                                    method: 'POST',
                                    headers: { 'Content-Type': 'application/json' },
                                    body: JSON.stringify({ urls: initialRequestData.urls })
                                });
                                if (response.ok) {
                                    const txt = await response.text();
                                    const d = txt ? JSON.parse(txt) : [];
                                    const downloads = Array.isArray(d) ? d : (d.downloads || []);
                                    if (Array.isArray(downloads)) {
                                        // Summarize missing comments for each download
                                        for (const dwn of downloads) {
                                            if (dwn.found && !dwn.comment) {
                                                try {
                                                    let [sumModel] = await vscode.lm.selectChatModels({ family: 'gpt-3.5' });
                                                    if (!sumModel) [sumModel] = await vscode.lm.selectChatModels();
                                                    if (sumModel) {
                                                        const sumMessages = [
                                                            // Use User role to supply system-like instruction if System factory is unavailable
                                                            vscode.LanguageModelChatMessage.User('You are a helpful assistant. Summarize the following document in 1-2 short sentences to be used as a document comment.'),
                                                            vscode.LanguageModelChatMessage.User(`Document content:\n\n${dwn.content.substring(0, 12000)}`)
                                                        ];
                                                        let summary = '';
                                                        try {
                                                            const sumResp = await sumModel.sendRequest(sumMessages, {}, token);
                                                            for await (const frag of sumResp.text) { summary += frag; }
                                                        } catch (se) { console.error('Summarization error', se); }
                                                        dwn.comment = (summary || '').trim();

                                                        // Update backend record with comment if we have an id
                                                        if ((initialRequestData as any).id) {
                                                            try {
                                                                await fetch(`${backendUrl}/data/update_docs`, {
                                                                    method: 'POST',
                                                                    headers: { 'Content-Type': 'application/json' },
                                                                    body: JSON.stringify({ query: { id: (initialRequestData as any).id, ref_docs: [{ source: dwn.url, comment: dwn.comment }] } })
                                                                });
                                                            } catch (ue) { console.error('Failed to update record with doc comment', ue); }
                                                        }
                                                    }
                                                } catch (e) { console.error('Error summarizing document', e); }
                                            }
                                        }

                                        loadedContext = downloads.map((d: any) => `Source: ${d.url}${d.comment ? ('\nComment: ' + d.comment) : ''}\nContent:\n${d.content}\n`).join('\n---\n\n');
                                        (initialRequestData as any).cachedContext = loadedContext;
                                    }
                                }
                             } catch (e) { console.error("Error refetching context", e); }
                        }
                     }

                     const systemPrompt = `
You are a helpful assistant. Use the following context to answer the user's question.

Found text from images:
${initialRequestData.ocrText || "None"}

Downloaded Document Context:
${loadedContext || "None"}

Original User Query:
${initialRequestData.manualText || ""}
`;                   
                     // Insert at the beginning or replace the first message
                     // If we just prepend, the model sees:
                     // User: [Context... Original Query]
                     // User: "process-request..." (from history)
                     // Assistant: "Answer..." (from history)
                     // User: "New Query"
                     // This is acceptable.
                     
                     messages.unshift(vscode.LanguageModelChatMessage.User(systemPrompt));
                     fullHistory = `System Context:\n${systemPrompt}\n\n` + fullHistory;
                }

                // Add current prompt
                messages.push(vscode.LanguageModelChatMessage.User(promptText));
                fullHistory += `User: ${promptText}\n\n`;
                
                let fullResponse = "";
                const chatResponse = await model.sendRequest(messages, {}, token);
                for await (const fragment of chatResponse.text) {
                    fullResponse += fragment;
                    stream.markdown(fragment);
                }

                const feedbackId = Date.now().toString() + Math.random().toString();
                
                // AUTOMATICALLY LOG "PENDING" RECORD
                const config = vscode.workspace.getConfiguration('repo-ask');
                const backendUrl = config.get<string>('backendUrl') || 'http://localhost:14321';
                
                try {
                    if (currentRecordId) {
                         // Append to existing
                         await fetch(`${backendUrl}/data/record_feedback`, {
                             method: 'POST',
                             headers: { 'Content-Type': 'application/json' },
                             body: JSON.stringify({
                                id: currentRecordId,
                                status: 'pending',
                                conversations: [{ ai_assistant: fullResponse, human: promptText }]
                             })
                         });
                    } else {
                         // Create new
                         const response = await fetch(`${backendUrl}/data/record_docs`, {
                             method: 'POST',
                             headers: { 'Content-Type': 'application/json' },
                             body: JSON.stringify({
                                 query: {
                                     timestamp: new Date().toISOString(),
                                     ref_docs: [],
                                     conversations: [
                                         { ai_assistant: "", human: "" }, // Dummy start
                                         { ai_assistant: fullResponse, human: promptText }
                                     ],
                                     status: 'pending'
                                 }
                             })
                         });
                         if (response.ok) {
                             const json = (await response.json()) as any;
                             if (json.id) currentRecordId = json.id;
                         }
                    }
                } catch (e) {
                    console.error("Failed to log conversation automatically", e);
                }
                
                stream.button({
                    command: 'repo-ask.feedbackAccept',
                    title: 'Accept',
                    arguments: [{ query: "", ai_answer: "", feedbackId, recordId: currentRecordId }]
                });
                stream.button({
                    command: 'repo-ask.feedbackReject',
                    title: 'Reject',
                    arguments: [{ query: "", ai_answer: "", feedbackId, recordId: currentRecordId }]
                });
                stream.button({
                    command: 'repo-ask.feedbackConfluence',
                    title: 'Add to Confluence',
                    arguments: [{ query: "", ai_answer: "", feedbackId, recordId: currentRecordId }]
                });
            } else {
                stream.markdown("Sorry, I could not find a suitable Language Model to generate a response.");
            }
        } catch (err: any) {
            stream.markdown(`Error calling Language Model: ${err.message}`);
        }
    };

    const participant = vscode.chat.createChatParticipant('repo-ask.repo-ask', handler);
    participant.iconPath = vscode.Uri.joinPath(context.extensionUri, 'media', 'icon.svg');
    context.subscriptions.push(participant);
}
