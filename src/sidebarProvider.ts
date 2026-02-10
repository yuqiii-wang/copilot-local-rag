import * as vscode from "vscode";
import * as fs from "fs";
import * as path from "path";
import { executeOCRRequest } from "./ocr/ocrHandler";
import { RequestManager } from "./requestManager";

export class SidebarProvider implements vscode.WebviewViewProvider {
  _view?: vscode.WebviewView;
  // State tracking for feedback
  private _lastDownloads: any[] = [];
  private _lastRagResults: any[] = [];
  private _lastOcrText: string = "";
  private _currentRecordId: string | null = null; // Track current session ID
  private _lastQuery: string = "";

  constructor(private readonly _extensionUri: vscode.Uri) {}

  private _generateUUID(): string {
      return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
          var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
          return v.toString(16);
      });
  }

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
                
                this._lastOcrText = result.ocr_text || "";

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
                const currentQuery = data.query || "";
                if (currentQuery !== this._lastQuery || !this._currentRecordId) {
                    this._currentRecordId = this._generateUUID();
                    this._lastQuery = currentQuery;
                }
                let contextContent = "";
                
                // If there are URLs, download content for them
                if (data.urls && data.urls.length > 0) {
                     const response = await fetch(`${backendUrl}/download/fetch`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ 
                            urls: data.urls,
                            query: data.manualText || data.query || "" 
                        })
                     });

                    if (response.ok) {
                        try {
                            const text = await response.text();
                            const data = text ? JSON.parse(text) : [];
                            
                            // Ensure we actually have an array to map over
                            // (Handles direct array return [..] or wrapped object { downloads: [..] })
                            const downloads = Array.isArray(data) ? data : (data.downloads || []);

                            if (Array.isArray(downloads)) {
                                this._lastDownloads = downloads; // Log for feedback
                                const missing = downloads.filter((d: any) => d.found === false);
                                if (missing.length > 0) {
                                    const missingMsg = missing.map((d: any) => d.url).join(', ');
                                    vscode.window.showWarningMessage(`File(s) not found: ${missingMsg}`);
                                }

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
                
                // Prepare request data for participant
                const reqManager = RequestManager.getInstance();
                // We pass the prompt via command hack or just let the user see it
                // Better approach: pass ID via hidden context if possible, or just rely on state.
                // CURRENT HACK: We embed a request ID in the prompt "process-request <id>"
                
                const requestId = reqManager.createRequest({
                    manualText: data.manualText,
                    ocrText: data.ocrText,
                    urls: data.urls,
                    id: this._currentRecordId // Pass the backend record ID so ChatParticipant can reference it in buttons
                });
                
                vscode.commands.executeCommand('workbench.action.chat.open', { query: finalQuery + `\n\nprocess-request ${requestId}` });
                
                // Record the docs immediately (Step 1)
                await this._recordInitialDocs(backendUrl, data.manualText, data.ocrText, this._lastQuery, "", this._currentRecordId);

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
                const currentQuery = data.text || "";
                if (currentQuery !== this._lastQuery || !this._currentRecordId) {
                    this._currentRecordId = this._generateUUID();
                    this._lastQuery = currentQuery;
                }
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
                const results = await response.json() as any[];
                this._lastRagResults = results; // Log for feedback
                
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
                
                if (!this._currentRecordId) {
                     this._currentRecordId = this._generateUUID();
                     await this._recordInitialDocs(backendUrl, "", "", data.query, data.ai_answer, this._currentRecordId);
                }

                if (this._currentRecordId) {
                    // Step 2: Update status
                    const endpoint = '/data/record_feedback';
                    const response = await fetch(`${backendUrl}${endpoint}`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 
                            id: this._currentRecordId,
                            status: action === 'addConfluence' ? 'added_confluence' : (action === 'accept' ? 'accepted' : 'rejected'),
                            conversations: [
                                { ai_assistant: data.ai_answer || "", human: data.query || this._lastQuery || "" }
                            ]
                        })
                    });
                    
                    if (response.ok) {
                        vscode.window.showInformationMessage(`Feedback (${action}) submitted successfully.`);
                    } else {
                        vscode.window.showErrorMessage("Failed to submit feedback.");
                    }
                } else {
                     vscode.window.showErrorMessage("Could not associate feedback with a record.");
                }

            } catch (error: any) {
                vscode.window.showErrorMessage(`Feedback Error: ${error.message}`);
            }
            break;
        }
      }
    });
  }

  private async _recordInitialDocs(backendUrl: string, manualText: string, ocrText: string, userQuery: string, aiAnswer: string = "", sessionId: string | null = null) {
      try {
            // Helper to determine type
            const getDocType = (source: string) => {
                const lower = source.toLowerCase();
                if (lower.includes('jira') || lower.includes('/browse/')) return 'jira';
                if (lower.includes('confluence') || lower.includes('/wiki/')) return 'confluence';
                return 'code';
            };

            // Collect docs from state
            const allDocs: any[] = [
                ...this._lastRagResults.map((r: any) => {
                    const src = r.id || r.link || 'graph';
                    return { 
                        source: src, 
                        type: getDocType(src),
                        title: r.title || r.metadata?.title || path.basename(src) || 'RAG Result',
                        score: r.score
                    };
                }),
                ...this._lastDownloads.map(d => ({ 
                    source: d.url, 
                    type: getDocType(d.url),
                    title: d.title || path.basename(d.url) || d.url,
                    comment: d.comment || undefined,
                    keywords: d.keywords || undefined
                }))
            ];

            if (this._lastOcrText) {
                allDocs.push({ 
                    source: 'user_upload',
                    type: 'code', // Default to code for OCR snippets as they are likely code screenshots in this context
                    content: this._lastOcrText.substring(0, 200) + '...',
                    title: 'OCR Image Upload'
                });
            }

            // Deduplicate docs based on normalized source
            const seenSources = new Set<string>();
            const refDocs = [];
            
            for (const doc of allDocs) {
                // simple normalization: remove file:/// prefix and decode, lower case
                let norm = doc.source.toLowerCase();
                if (norm.startsWith('file:///')) norm = norm.substring(8);
                if (norm.startsWith('file://')) norm = norm.substring(7);
                norm = decodeURIComponent(norm).replace(/\\/g, '/'); // standardize slashes
                
                if (!seenSources.has(norm)) {
                    seenSources.add(norm);
                    refDocs.push(doc);
                }
            }

            const payload = {
                query: {
                    id: sessionId, // Use frontend generated UUID
                    timestamp: new Date().toISOString(),
                    question: userQuery || manualText || (ocrText ? "OCR extraction" : ""),
                    ref_docs: refDocs,
                    conversations: [
                         { ai_assistant: "", human: "" },
                         { ai_assistant: aiAnswer, human: userQuery || "" }
                    ],
                    status: "pending"
                }
            };
            
            const response = await fetch(`${backendUrl}/data/record_docs`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (response.ok) {
                const res = await response.json() as any; // Fix unknown type syntax error
                if (res.id) {
                    this._currentRecordId = res.id;
                }
            }
      } catch (e) {
          console.error("Failed to record initial docs", e);
      }
  }

  private _getHtmlForWebview(webview: vscode.Webview) {
    const htmlPath = path.join(this._extensionUri.fsPath, 'media', 'index.html');
    let htmlContent = fs.readFileSync(htmlPath, 'utf8');
    
    // If we had external resources, we would replace paths here. 
    // Since everything is embedded for now, just return the content.
    return htmlContent;
  }
}
