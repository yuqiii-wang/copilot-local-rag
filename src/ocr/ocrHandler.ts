import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';


export async function executeOCRRequest(fileBuffer: Buffer, filename: string): Promise<any> {
    const config = vscode.workspace.getConfiguration('repo-ask');
    const backendUrl = config.get<string>('backendUrl', 'http://localhost:14321');
    const sanitizedBackendUrl = backendUrl.endsWith('/') ? backendUrl.slice(0, -1) : backendUrl;
    const ocrUrl = `${sanitizedBackendUrl}/ocr/`;

    // Use a dynamic import to avoid issues if Blob is not global or conflicts with DOM types
    let blob: Blob;
    try {
        blob = new Blob([fileBuffer]);
    } catch (e) {
        // Fallback for environments where global Blob might be broken
        const { Blob } = require('buffer');
        blob = new Blob([fileBuffer]);
    }

    const formData = new FormData();
    formData.append('file', blob, filename);

    const response = await fetch(ocrUrl, {
        method: 'POST',
        body: formData
    });

    if (!response.ok) {
        throw new Error(`Backend error: ${response.statusText}`);
    }

    return await response.json();
}
