const fs = require('fs');
const path = require('path');

const CONTENT_FILE_NAME = 'content.md';
const METADATA_FILE_NAME = 'metadata.json';
const IMAGES_DIR_NAME = 'images';

function getLocalStorePath(context) {
    return path.join(context.globalStorageUri.fsPath, 'local-store');
}

function getLocalIndexStorePath(context) {
    return path.join(context.globalStorageUri.fsPath, 'local-store-index');
}

function ensureStoragePath(context) {
    const storagePath = getLocalStorePath(context);
    fs.mkdirSync(storagePath, { recursive: true });
    return storagePath;
}

function ensureIndexStoragePath(context) {
    const indexPath = getLocalIndexStorePath(context);
    fs.mkdirSync(indexPath, { recursive: true });
    return indexPath;
}

function getDocumentDirectory(storagePath, docId) {
    return path.join(storagePath, String(docId));
}

function getDocumentContentPath(storagePath, docId) {
    return path.join(getDocumentDirectory(storagePath, docId), CONTENT_FILE_NAME);
}

function getDocumentMetadataPath(storagePath, docId) {
    return path.join(getDocumentDirectory(storagePath, docId), METADATA_FILE_NAME);
}

function getDocumentImagesDirectory(storagePath, docId) {
    return path.join(getDocumentDirectory(storagePath, docId), IMAGES_DIR_NAME);
}

function readAllMetadata(storagePath) {
    const entries = fs.existsSync(storagePath)
        ? fs.readdirSync(storagePath, { withFileTypes: true })
        : [];

    const byId = new Map();
    const metadataList = [];

    for (const entry of entries) {
        if (!entry.isDirectory()) {
            continue;
        }

        const metadataPath = path.join(storagePath, entry.name, METADATA_FILE_NAME);
        if (!fs.existsSync(metadataPath)) {
            continue;
        }

        try {
            const parsed = JSON.parse(fs.readFileSync(metadataPath, 'utf8'));
            if (parsed && parsed.id !== undefined && parsed.id !== null) {
                byId.set(String(parsed.id), parsed);
            }
        } catch {
            // Ignore malformed metadata files
        }
    }

    for (const entry of entries) {
        if (!entry.isFile() || !entry.name.endsWith('.json')) {
            continue;
        }

        const metadataPath = path.join(storagePath, entry.name);
        try {
            const parsed = JSON.parse(fs.readFileSync(metadataPath, 'utf8'));
            const parsedId = parsed && parsed.id !== undefined && parsed.id !== null
                ? String(parsed.id)
                : path.parse(entry.name).name;

            if (!byId.has(parsedId)) {
                byId.set(parsedId, parsed);
            }
        } catch {
            // Ignore malformed metadata files
        }
    }

    metadataList.push(...byId.values());
    return metadataList;
}

function writeDocumentFiles(storagePath, pageId, markdownContent, metadata) {
    const docDir = getDocumentDirectory(storagePath, pageId);
    const imagesDir = getDocumentImagesDirectory(storagePath, pageId);
    const contentPath = getDocumentContentPath(storagePath, pageId);
    const metadataPath = getDocumentMetadataPath(storagePath, pageId);

    fs.mkdirSync(docDir, { recursive: true });
    fs.mkdirSync(imagesDir, { recursive: true });

    fs.writeFileSync(contentPath, markdownContent, 'utf8');
    fs.writeFileSync(metadataPath, JSON.stringify(metadata, null, 2), 'utf8');
}

function readDocumentContent(storagePath, docId) {
    const markdownPath = getDocumentContentPath(storagePath, docId);
    if (fs.existsSync(markdownPath)) {
        return fs.readFileSync(markdownPath, 'utf8');
    }

    const legacyMarkdownPath = path.join(storagePath, `${docId}.md`);
    if (fs.existsSync(legacyMarkdownPath)) {
        return fs.readFileSync(legacyMarkdownPath, 'utf8');
    }

    const legacyTextPath = path.join(storagePath, `${docId}.txt`);
    if (fs.existsSync(legacyTextPath)) {
        return fs.readFileSync(legacyTextPath, 'utf8');
    }

    return null;
}

function deleteDocumentFiles(storagePath, docId) {
    const docDir = getDocumentDirectory(storagePath, docId);
    const markdownPath = getDocumentContentPath(storagePath, docId);
    const metadataPath = getDocumentMetadataPath(storagePath, docId);

    const legacyMarkdownPath = path.join(storagePath, `${docId}.md`);
    const legacyMetadataPath = path.join(storagePath, `${docId}.json`);
    const legacyTextPath = path.join(storagePath, `${docId}.txt`);

    let deletedMd = false;
    let deletedJson = false;

    for (const [kind, filePath] of [
        ['md', markdownPath],
        ['json', metadataPath],
        ['md', legacyMarkdownPath],
        ['json', legacyMetadataPath],
        ['txt', legacyTextPath]
    ]) {
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

    try {
        if (fs.existsSync(docDir)) {
            fs.rmSync(docDir, { recursive: true, force: true });
        }
    } catch {
    }

    return {
        deletedMd,
        deletedJson,
        deletedCount: Number(deletedMd) + Number(deletedJson)
    };
}

module.exports = {
    ensureStoragePath,
    ensureIndexStoragePath,
    getDocumentDirectory,
    getDocumentImagesDirectory,
    readAllMetadata,
    writeDocumentFiles,
    readDocumentContent,
    deleteDocumentFiles
};