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
        
        // Check if we are processing a request from Sidebar
        const match = promptText.match(/process-request\s+([a-z0-9]+)/);
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
                            body: JSON.stringify({ urls: requestData.urls })
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
                                    loadedContext = downloads
                                        .map((d: any) => `Source: ${d.url}\nContent:\n${d.content}\n`)
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
                        [model] = await vscode.lm.selectChatModels({ family: 'gpt-3.5' });
                    }
                    
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

                        stream.button({
                            command: 'repo-ask.feedbackAccept',
                            title: 'Accept',
                            arguments: [{ query: requestData.manualText, ai_answer: fullResponse }]
                        });
                        stream.button({
                            command: 'repo-ask.feedbackReject',
                            title: 'Reject',
                            arguments: [{ query: requestData.manualText, ai_answer: fullResponse }]
                        });
                        stream.button({
                            command: 'repo-ask.feedbackConfluence',
                            title: 'Add to Confluence',
                            arguments: [{ query: requestData.manualText, ai_answer: fullResponse }]
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

        // Basic Tool Usage Logic Check
        if (promptText.toLowerCase().includes('tool')) {
            stream.markdown('**Using tool: System Check**\n\n');
            await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate tool delay
            stream.markdown('Tool execution complete. All systems normal.\n\n');
            return; // Exit after tool usage for this demo
        }

        // General Chat Query handling
        try {
            let [model] = await vscode.lm.selectChatModels({ family: 'gpt-4' });
            if (!model) [model] = await vscode.lm.selectChatModels({ family: 'gpt-3.5' });
            if (!model) [model] = await vscode.lm.selectChatModels();

            if (model) {
                const messages = [
                    vscode.LanguageModelChatMessage.User(promptText)
                ];
                
                let fullResponse = "";
                const chatResponse = await model.sendRequest(messages, {}, token);
                for await (const fragment of chatResponse.text) {
                    fullResponse += fragment;
                    stream.markdown(fragment);
                }

                stream.button({
                    command: 'repo-ask.feedbackAccept',
                    title: 'Accept',
                    arguments: [{ query: promptText, ai_answer: fullResponse }]
                });
                stream.button({
                    command: 'repo-ask.feedbackReject',
                    title: 'Reject',
                    arguments: [{ query: promptText, ai_answer: fullResponse }]
                });
                stream.button({
                    command: 'repo-ask.feedbackConfluence',
                    title: 'Add to Confluence',
                    arguments: [{ query: promptText, ai_answer: fullResponse }]
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
