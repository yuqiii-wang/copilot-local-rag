import * as vscode from "vscode";
import * as fs from "fs";
import * as path from "path";
import { executeOCRRequest } from "./ocr/ocrHandler";
import { RequestManager } from "./requestManager";

export class SidebarProvider implements vscode.WebviewViewProvider {
  _view?: vscode.WebviewView;

  constructor(private readonly _extensionUri: vscode.Uri) {}

  public triggerFeedback(action: string, data?: any) {
    if (this._view) {
        this._view.show?.(true);
        this._view.webview.postMessage({
            type: 'triggerFeedback',
            action: action,
            data: data
        });
    }
  }

  public resolveWebviewView(
    webviewView: vscode.WebviewView,
    context: vscode.WebviewViewResolveContext,
    _token: vscode.CancellationToken
  ) {
    this._view = webviewView;

    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [this._extensionUri],
    };

    webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);

    webviewView.webview.onDidReceiveMessage(async (data) => {
      const config = vscode.workspace.getConfiguration('repo-ask');
      const backendUrl = config.get<string>('backendUrl') || 'http://localhost:14321';

      switch (data.type) {
        case "openChat": {
          vscode.commands.executeCommand("repo-ask.openChat");
          break;
        }
        case "uploadImage": {
            vscode.window.showInformationMessage(`Processing Image: ${data.fileName}`);
            try {
                // Reconstruct Buffer from the array data sent from webview
                const fileBuffer = Buffer.from(data.data);
                const result = await executeOCRRequest(fileBuffer, data.fileName);
                
                webviewView.webview.postMessage({
                    type: 'ocrResult',
                    text: result.ocr_text || '_No text found_'
                });
            } catch (error: any) {
                vscode.window.showErrorMessage(`OCR Error: ${error.message}`);
                webviewView.webview.postMessage({
                    type: 'error',
                    message: error.message
                });
            }
            break;
        }
        case "processAndChat": {
            try {
                let contextContent = "";
                
                // If there are URLs, download content for them
                if (data.urls && data.urls.length > 0) {
                     const response = await fetch(`${backendUrl}/download/fetch`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ urls: data.urls })
                     });

                    if (response.ok) {
                        try {
                            const text = await response.text();
                            const data = text ? JSON.parse(text) : [];
                            
                            // Ensure we actually have an array to map over
                            // (Handles direct array return [..] or wrapped object { downloads: [..] })
                            const downloads = Array.isArray(data) ? data : (data.downloads || []);

                            if (Array.isArray(downloads)) {
                                contextContent = downloads
                                    .map((d: any) => `Source: ${d.url}\nContent:\n${d.content}\n`)
                                    .join('\n---\n\n');
                            } else {
                                vscode.window.showWarningMessage("Download backend returned unexpected data structure.");
                            }
                        } catch (e) {
                            console.error("Failed to parse download response:", e);
                            vscode.window.showWarningMessage("Failed to process downloaded content.");
                        }
                    } else {
                        vscode.window.showWarningMessage("Failed to download some documents.");
                    }
                }

                // Construct final query
                const parts = [];
                // parts.push('@repo-ask'); // Optional: enforce using the participant
                if (data.manualText) parts.push(data.manualText);
                if (data.ocrText) parts.push(`Found text from images\n${data.ocrText}`);
                if (contextContent) parts.push(`Retrieved Context:\n${contextContent}`);
                
                const finalQuery = '@repo-ask ' + parts.join('\n\n');
                
                vscode.commands.executeCommand('workbench.action.chat.open', { query: finalQuery });
                webviewView.webview.postMessage({ type: 'processingComplete' });
                webviewView.webview.postMessage({ type: 'processingComplete' });
            } catch (error: any) {
                 vscode.window.showErrorMessage(`Error processing request: ${error.message}`);
                 webviewView.webview.postMessage({
                    type: 'error',
                    message: error.message
                });
            }
            break;
        }
        case "queryRag": {
            try {
                const skip = data.skip || 0;
                // Call backend RAG service with the query parameter
                const response = await fetch(`${backendUrl}/rag/retrieve`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ 
                        query: data.text || "",
                        skip: skip
                    })
                });

                if (!response.ok) {
                    throw new Error(`Backend error: ${response.statusText}`);
                }
                const results = await response.json();
                
                webviewView.webview.postMessage({
                    type: 'ragResult',
                    results: results,
                    isLoadMore: skip > 0
                });
            } catch (error: any) {
                 vscode.window.showErrorMessage(`RAG Error: ${error.message}`);
                 webviewView.webview.postMessage({
                    type: 'error',
                    message: error.message
                });
            }
            break;
        }
        case "submitFeedback": {
            try {
                const action = data.action; 
                
                // Use provided query/thinking/answer if available (from Chat Interaction), else fallback to current sidebar state
                let query = data.query;
                if (!query) {
                    const manualText = data.manualText || "";
                    const ocrText = data.ocrText || "";
                    if (manualText) query = manualText;
                    if (ocrText) query += (query ? "\n\nFound text from images\n" : "Found text from images\n") + ocrText;
                }
                
                const ai_thinking = data.ai_thinking || "";
                const ai_answer = data.ai_answer || "";

                // Determine endpoint based on action
                let endpoint = '';
                if (action === 'accept' || action === 'addConfluence') endpoint = '/rag/feedback/accept';
                else if (action === 'reject') endpoint = '/rag/feedback/reject';
                
                if (endpoint) {
                     const response = await fetch(`${backendUrl}${endpoint}`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            query: query,
                            ai_thinking: ai_thinking,
                            ai_answer: ai_answer,
                            user_comments: data.comments || ""
                        })
                    });
                    
                    if (response.ok) {
                        vscode.window.showInformationMessage(`Feedback (${action}) submitted successfully.`);
                    } else {
                        vscode.window.showErrorMessage("Failed to submit feedback.");
                    }
                }
            } catch (error: any) {
                vscode.window.showErrorMessage(`Feedback Error: ${error.message}`);
            }
            break;
        }
      }
    });
  }

  private _getHtmlForWebview(webview: vscode.Webview) {
    const htmlPath = path.join(this._extensionUri.fsPath, 'media', 'index.html');
    let htmlContent = fs.readFileSync(htmlPath, 'utf8');
    
    // If we had external resources, we would replace paths here. 
    // Since everything is embedded for now, just return the content.
    return htmlContent;
  }
}
