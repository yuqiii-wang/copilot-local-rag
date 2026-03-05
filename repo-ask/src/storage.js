const fs = require('fs');
const path = require('path');
const vscode = require('vscode');

function getLocalStorePath(context) {
    const workspaceRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
    if (workspaceRoot) {
        return path.join(workspaceRoot, 'local-store');
    }
    return path.join(context.globalStorageUri.fsPath, 'local-store');
}

function ensureStoragePath(context) {
    const storagePath = getLocalStorePath(context);
    fs.mkdirSync(storagePath, { recursive: true });
    return storagePath;
}

function readAllMetadata(storagePath) {
    const files = fs.existsSync(storagePath)
        ? fs.readdirSync(storagePath).filter(file => file.endsWith('.json'))
        : [];

    const metadataList = [];
    for (const file of files) {
        const metadataPath = path.join(storagePath, file);
        try {
            metadataList.push(JSON.parse(fs.readFileSync(metadataPath, 'utf8')));
        } catch {
            // Ignore malformed metadata files
        }
    }
    return metadataList;
}

function writeDocumentFiles(storagePath, pageId, markdownContent, metadata) {
    const contentPath = path.join(storagePath, `${pageId}.md`);
    const metadataPath = path.join(storagePath, `${pageId}.json`);

    fs.writeFileSync(contentPath, markdownContent, 'utf8');
    fs.writeFileSync(metadataPath, JSON.stringify(metadata, null, 2), 'utf8');
}

function readDocumentContent(storagePath, docId) {
    const markdownPath = path.join(storagePath, `${docId}.md`);
    if (fs.existsSync(markdownPath)) {
        return fs.readFileSync(markdownPath, 'utf8');
    }

    const legacyTextPath = path.join(storagePath, `${docId}.txt`);
    if (fs.existsSync(legacyTextPath)) {
        return fs.readFileSync(legacyTextPath, 'utf8');
    }

    return null;
}

function deleteDocumentFiles(storagePath, docId) {
    const markdownPath = path.join(storagePath, `${docId}.md`);
    const metadataPath = path.join(storagePath, `${docId}.json`);
    const legacyTextPath = path.join(storagePath, `${docId}.txt`);

    let deletedMd = false;
    let deletedJson = false;

    for (const [kind, filePath] of [['md', markdownPath], ['json', metadataPath], ['txt', legacyTextPath]]) {
        try {
            if (fs.existsSync(filePath)) {
                fs.unlinkSync(filePath);
                if (kind === 'md') {
                    deletedMd = true;
                }
                if (kind === 'json') {
                    deletedJson = true;
                }
            }
        } catch {
        }
    }

    return {
        deletedMd,
        deletedJson,
        deletedCount: Number(deletedMd) + Number(deletedJson)
    };
}

function formatDocumentDetails(metadata, content) {
    return [
        'Content:',
        content || 'No content available.',
        '',
        'Metadata:',
        `- title: ${metadata.title || 'Unknown'}`
    ].join('\n');
}

module.exports = {
    ensureStoragePath,
    readAllMetadata,
    writeDocumentFiles,
    readDocumentContent,
    deleteDocumentFiles,
    formatDocumentDetails
};