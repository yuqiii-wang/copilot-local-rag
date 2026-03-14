const fs = require('fs');
const path = require('path');

const DEFAULT_K1 = 1.5;
const DEFAULT_B = 0.75;

function createBm25Index(deps) {
    const {
        storePath,
        tokenize,
        indexFileName = 'bm25-index.json'
    } = deps;

    function ensureStorePath() {
        fs.mkdirSync(storePath, { recursive: true });
    }

    function getIndexPath() {
        ensureStorePath();
        return path.join(storePath, indexFileName);
    }

    function loadIndex() {
        const indexPath = getIndexPath();
        if (!fs.existsSync(indexPath)) {
            return { docs: {} };
        }

        try {
            const parsed = JSON.parse(fs.readFileSync(indexPath, 'utf8'));
            if (parsed && typeof parsed === 'object' && parsed.docs && typeof parsed.docs === 'object') {
                return parsed;
            }
        } catch {
            // Fall through to empty index.
        }

        return { docs: {} };
    }

    function saveIndex(index) {
        const indexPath = getIndexPath();
        fs.writeFileSync(indexPath, JSON.stringify(index, null, 2), 'utf8');
    }

    function normalizeTokenList(tokens) {
        return (Array.isArray(tokens) ? tokens : [])
            .map(token => String(token || '').trim().toLowerCase())
            .filter(token => token.length > 2 && /^[a-z0-9]+$/.test(token));
    }

    function buildTermFrequency(tokens) {
        const frequency = {};
        for (const token of tokens) {
            frequency[token] = (frequency[token] || 0) + 1;
        }
        return frequency;
    }

    function upsertDocument(docId, text) {
        const id = String(docId || '').trim();
        if (!id) {
            return;
        }

        const tokens = normalizeTokenList(tokenize(String(text || '')));
        const index = loadIndex();
        index.docs[id] = {
            id,
            length: tokens.length,
            tf: buildTermFrequency(tokens)
        };
        saveIndex(index);
    }

    function removeDocument(docId) {
        const id = String(docId || '').trim();
        if (!id) {
            return;
        }

        const index = loadIndex();
        if (index.docs[id]) {
            delete index.docs[id];
            saveIndex(index);
        }
    }

    function rebuildDocuments(documents) {
        const nextDocs = {};
        for (const entry of (Array.isArray(documents) ? documents : [])) {
            const id = String(entry?.id || '').trim();
            if (!id) {
                continue;
            }

            const tokens = normalizeTokenList(tokenize(String(entry?.text || '')));
            nextDocs[id] = {
                id,
                length: tokens.length,
                tf: buildTermFrequency(tokens)
            };
        }

        saveIndex({ docs: nextDocs });
    }

    function buildStats(index) {
        const docs = index?.docs || {};
        const docEntries = Object.values(docs);
        const totalDocs = docEntries.length;
        const totalLength = docEntries.reduce((sum, doc) => sum + Number(doc.length || 0), 0);
        const avgDocLength = totalDocs > 0 ? (totalLength / totalDocs) : 0;

        const docFreq = {};
        for (const doc of docEntries) {
            const terms = Object.keys(doc.tf || {});
            for (const term of terms) {
                docFreq[term] = (docFreq[term] || 0) + 1;
            }
        }

        return {
            docs,
            totalDocs,
            avgDocLength,
            docFreq
        };
    }

    function bm25Idf(term, stats) {
        const df = Number(stats.docFreq[term] || 0);
        const numerator = (stats.totalDocs - df + 0.5);
        const denominator = (df + 0.5);
        if (denominator <= 0) {
            return 0;
        }
        return Math.log(1 + (numerator / denominator));
    }

    function bm25TermScore(term, doc, stats, options = {}) {
        const tf = Number(doc?.tf?.[term] || 0);
        if (tf <= 0) {
            return 0;
        }

        const k1 = Number.isFinite(options.k1) ? options.k1 : DEFAULT_K1;
        const b = Number.isFinite(options.b) ? options.b : DEFAULT_B;
        const avgLength = stats.avgDocLength > 0 ? stats.avgDocLength : 1;
        const dl = Number(doc.length || 0);
        const idf = bm25Idf(term, stats);
        const denominator = tf + k1 * (1 - b + (b * dl / avgLength));
        if (denominator <= 0) {
            return 0;
        }

        return idf * ((tf * (k1 + 1)) / denominator);
    }

    function extractKeywordsForDocument(docId, options = {}) {
        const id = String(docId || '').trim();
        if (!id) {
            return [];
        }

        const index = loadIndex();
        const stats = buildStats(index);
        const doc = stats.docs[id];
        if (!doc) {
            return [];
        }

        const limit = Number.isFinite(options.limit) && options.limit > 0
            ? Math.floor(options.limit)
            : 20;

        const rankedTerms = Object.keys(doc.tf || {})
            .map((term) => ({
                term,
                score: bm25TermScore(term, doc, stats, options)
            }))
            .filter((item) => item.score > 0)
            .sort((a, b) => b.score - a.score)
            .slice(0, limit)
            .map((item) => item.term);

        return rankedTerms;
    }

    function rankDocuments(query, metadataById = {}, options = {}) {
        const queryTokens = [...new Set(normalizeTokenList(tokenize(String(query || ''))))];
        if (queryTokens.length === 0) {
            return [];
        }

        const index = loadIndex();
        const stats = buildStats(index);
        if (stats.totalDocs === 0) {
            return [];
        }

        const limit = Number.isFinite(options.limit) && options.limit > 0
            ? Math.floor(options.limit)
            : 20;

        const scored = Object.values(stats.docs)
            .map((doc) => {
                let score = 0;
                for (const term of queryTokens) {
                    score += bm25TermScore(term, doc, stats, options);
                }

                const metadata = metadataById[String(doc.id)] || {};
                return {
                    ...metadata,
                    id: metadata.id || doc.id,
                    score
                };
            })
            .filter((item) => Number(item.score) > 0)
            .sort((a, b) => b.score - a.score)
            .slice(0, limit);

        return scored;
    }

    function listDocumentIds() {
        const index = loadIndex();
        return Object.keys(index?.docs || {});
    }

    return {
        ensureStorePath,
        upsertDocument,
        removeDocument,
        rebuildDocuments,
        extractKeywordsForDocument,
        rankDocuments,
        listDocumentIds
    };
}

module.exports = {
    createBm25Index
};
