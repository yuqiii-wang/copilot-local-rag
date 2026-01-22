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

    // Feedback Commands
    context.subscriptions.push(
        vscode.commands.registerCommand('repo-ask.feedbackAccept', (args) => {
            sidebarProvider.triggerFeedback('accept', args);
        }),
        vscode.commands.registerCommand('repo-ask.feedbackReject', (args) => {
            sidebarProvider.triggerFeedback('reject', args);
        }),
        vscode.commands.registerCommand('repo-ask.feedbackConfluence', (args) => {
            sidebarProvider.triggerFeedback('addConfluence', args);
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
