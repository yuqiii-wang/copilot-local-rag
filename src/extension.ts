import * as vscode from 'vscode';
import { SidebarProvider } from './sidebarProvider';
import { registerChatParticipant } from './chatParticipant';

export function activate(context: vscode.ExtensionContext) {
    console.log('Congratulations, your extension "repo-ask" is now active!');

    // Register Sidebar Provider
    const sidebarProvider = new SidebarProvider(context.extensionUri);
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider(
            "repo-ask.sidebarView",
            sidebarProvider
        )
    );

    const submittedFeedbackIds = new Set<string>();

    // Helper to send feedback directly to backend
    const sendFeedback = async (action: string, args: any) => {
        if (args.feedbackId && submittedFeedbackIds.has(args.feedbackId)) {
            vscode.window.showInformationMessage(`Feedback for this response has already been submitted.`);
            return;
        }

        // Hide buttons by invoking 'setContext' command if needed, but VS Code API doesn't allow removing buttons from previous turns easily.
        // However, checking submittedFeedbackIds prevents reprocessing.
        // To visually hide them isn't supported directly on past chat turns yet.

        const config = vscode.workspace.getConfiguration('repo-ask');
        const backendUrl = config.get<string>('backendUrl') || 'http://localhost:14321';
        
        // If we have a record ID, use it to UPDATE the existing record
        if (args.recordId) {
             try {
                const response = await fetch(`${backendUrl}/data/record_feedback`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        id: args.recordId,
                        status: action === 'addConfluence' ? 'added_confluence' : (action === 'accept' ? 'accepted' : 'rejected'),
                        conversations: [
                            // Only send the current turn (new backend logic appends it)
                            { ai_assistant: args.ai_answer || "", human: args.query || "" }
                        ]
                    })
                });
                
                if (response.ok) {
                    vscode.window.showInformationMessage(`Feedback (${action}) submitted successfully (Updated Record).`);
                    if (args.feedbackId) submittedFeedbackIds.add(args.feedbackId);
                    return;
                } else {
                     // Fallback if update fails (e.g. record not found on server)
                     console.warn(`Failed to update record ${args.recordId}, failing over to new record creation.`);
                }
             } catch (e) {
                 console.error("Error updating record:", e);
             }
        }

        const payload = {
            query: {
                timestamp: new Date().toISOString(),
                ref_docs: [], // Inline feedback might not track docs, or we could pass them if available
                conversations: [
                    { ai_assistant: "", human: "" },
                    { 
                        ai_assistant: args.ai_answer || "", 
                        human: args.query || "" 
                    }
                ],
                status: action === 'addConfluence' ? 'added_confluence' : (action === 'accept' ? 'accepted' : 'rejected')
            }
        };

        try {
            // Using record_docs to save the feedback as a new record since we don't have the original ID in this context
            const response = await fetch(`${backendUrl}/data/record_docs`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (response.ok) {
                vscode.window.showInformationMessage(`Feedback (${action}) submitted successfully.`);
                if (args.feedbackId) {
                    submittedFeedbackIds.add(args.feedbackId);
                }
            } else {
                vscode.window.showErrorMessage(`Failed to submit feedback: ${response.statusText}`);
            }
        } catch (error: any) {
            vscode.window.showErrorMessage(`Feedback Error: ${error.message}`);
        }
    };

    // Feedback Commands
    context.subscriptions.push(
        vscode.commands.registerCommand('repo-ask.feedbackAccept', (args) => {
            sendFeedback('accept', args);
        }),
        vscode.commands.registerCommand('repo-ask.feedbackReject', (args) => {
            sendFeedback('reject', args);
        }),
        vscode.commands.registerCommand('repo-ask.feedbackConfluence', (args) => {
            sendFeedback('addConfluence', args);
        })
    );

    // Register Command to open chat (or perform action from sidebar)
    context.subscriptions.push(
        vscode.commands.registerCommand('repo-ask.openChat', () => {
            vscode.commands.executeCommand('workbench.action.chat.open');
        })
    );

    // Register Chat Participant
    registerChatParticipant(context);
}

export function deactivate() {}
