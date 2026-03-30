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

function rewriteMetadataFile(metadataPath, fallbackId) {
    if (!fs.existsSync(metadataPath)) {
        return false;
    }

    try {
        const raw = JSON.parse(fs.readFileSync(metadataPath, 'utf8'));
        const normalized = normalizeStoredMetadataSchema(fallbackId, raw);
        const rawText = JSON.stringify(raw, null, 2);
        const normalizedText = JSON.stringify(normalized, null, 2);
        if (rawText !== normalizedText) {
            fs.writeFileSync(metadataPath, normalizedText, 'utf8');
            return true;
        }
    } catch {
        // Ignore malformed metadata files during backfill.
    }

    return false;
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

function backfillStoredMetadataSchema(storagePath) {
    const entries = fs.existsSync(storagePath)
        ? fs.readdirSync(storagePath, { withFileTypes: true })
        : [];

    for (const entry of entries) {
        if (entry.isDirectory()) {
            rewriteMetadataFile(path.join(storagePath, entry.name, METADATA_FILE_NAME), entry.name);
            continue;
        }

        if (entry.isFile() && entry.name.endsWith('.json')) {
            rewriteMetadataFile(path.join(storagePath, entry.name), path.parse(entry.name).name);
        }
    }
}

function normalizeStoredMetadataSchema(docId, metadata) {
    const base = metadata && typeof metadata === 'object' ? metadata : {};
    const normalizedReferencedQueries = Array.isArray(base.referencedQueries)
        ? [...new Set(base.referencedQueries.map(value => String(value || '').trim()).filter(Boolean))]
        : typeof base.referencedQueries === 'string'
            ? [...new Set(base.referencedQueries.split(',').map(value => value.trim()).filter(Boolean))]
            : [];

    const result = {
        ...base,
        id: base.id !== undefined && base.id !== null ? base.id : pageIdToId(docId),
        keywords: (base.keywords && typeof base.keywords === 'object' && !Array.isArray(base.keywords))
            ? base.keywords
            : (Array.isArray(base.keywords) ? base.keywords : {}),
        tags: Array.isArray(base.tags) ? base.tags : [],
        referencedQueries: normalizedReferencedQueries,
        summary: typeof base.summary === 'string' ? base.summary : '',
        feedback: typeof base.feedback === 'string' ? base.feedback : '',
        knowledgeGraph: typeof base.knowledgeGraph === 'string' ? base.knowledgeGraph : '',
        relatedPages: Array.isArray(base.relatedPages) ? base.relatedPages : []
    };
    delete result.synonyms;  // legacy field — synonyms now live in keywords.synonyms
    return result;
}

function pageIdToId(docId) {
    return String(docId);
}

function writeDocumentFiles(storagePath, pageId, markdownContent, metadata) {
    const docDir = getDocumentDirectory(storagePath, pageId);
    const imagesDir = getDocumentImagesDirectory(storagePath, pageId);
    const contentPath = getDocumentContentPath(storagePath, pageId);
    const metadataPath = getDocumentMetadataPath(storagePath, pageId);
    const normalizedMetadata = normalizeStoredMetadataSchema(pageId, metadata);

    fs.mkdirSync(docDir, { recursive: true });
    fs.mkdirSync(imagesDir, { recursive: true });

    fs.writeFileSync(contentPath, markdownContent, 'utf8');
    fs.writeFileSync(metadataPath, JSON.stringify(normalizedMetadata, null, 2), 'utf8');
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
    backfillStoredMetadataSchema,
    getDocumentDirectory,
    getDocumentImagesDirectory,
    readAllMetadata,
    writeDocumentFiles,
    readDocumentContent,
    deleteDocumentFiles
};