const { STOP_WORDS } = require('./patternMatch');
const { generateSynonyms } = require('./synonyms');

const MIN_WORD_LENGTH = 2;
const MAX_WORD_LENGTH = 64;

/**
 * Split a camelCase or PascalCase word into lowercase parts.
 * E.g. "TradeProcessor" → ["trade","processor"], "XMLParser" → ["xml","parser"]
 */
function splitCamelCase(word) {
    return word
        .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
        .replace(/([A-Z]+)([A-Z][a-z])/g, '$1 $2')
        .split(/\s+/)
        .map(w => w.toLowerCase())
        .filter(Boolean);
}

function isValidToken(token) {
    return (
        token.length >= MIN_WORD_LENGTH &&
        token.length <= MAX_WORD_LENGTH &&
        !STOP_WORDS.has(token) &&
        /[a-z]/.test(token)
    );
}

/**
 * Break text into clean lowercase word tokens, splitting camelCase and snake_case.
 * Compound tokens joined by - _ . + / are emitted as both the whole compound and
 * each individual part, e.g. "trade-1234-20260321" → ["trade-1234-20260321","trade","1234","20260321"].
 * Pure digit sequences (IDs, dates, codes) are also captured.
 * @param {string} text
 * @returns {string[]}
 */
function wordsFromText(text) {
    const rawText = String(text || '');
    const words = [];

    // ── Step 1: Compound tokens ────────────────────────────────────────────────
    // Alpha/digit segments joined by - _ . + /  (e.g. "trade-1234-20260321", "v1.2.3")
    // Emit the entire compound as one lowercase token, then each individual part.
    const compoundRe = /[A-Za-z0-9]+(?:[-_.+/][A-Za-z0-9]+)+/g;
    let m;
    while ((m = compoundRe.exec(rawText)) !== null) {
        const compound = m[0];
        words.push(compound.toLowerCase());                 // whole — one token
        // Pure decimal number (e.g. 123.456788): single dot between digit groups.
        // Do NOT split into parts — the whole token is the meaningful unit.
        // Multiple dots (e.g. 1.2.3) still split normally.
        if (/^\d+\.\d+$/.test(compound)) continue;
        for (const part of compound.split(/[-_.+/]/)) {
            if (!part) continue;
            if (/^\d+$/.test(part)) {
                // pure-digit segment (e.g. "1234", date "20260321")
                if (part.length >= MIN_WORD_LENGTH && part.length <= MAX_WORD_LENGTH) {
                    words.push(part);
                }
            } else {
                const hasCamel = /[A-Z]/.test(part) && /[a-z]/.test(part);
                if (hasCamel) {
                    for (const p of splitCamelCase(part)) {
                        if (isValidToken(p)) words.push(p);
                    }
                } else {
                    const lower = part.toLowerCase();
                    if (isValidToken(lower)) words.push(lower);
                }
            }
        }
    }

    // ── Step 2: Standalone word tokens (camelCase / snake_case) ───────────────
    // The underscore in the pattern captures snake_case as a single token for splitting.
    const wordTokens = rawText.match(/[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*/g) || [];
    for (const token of wordTokens) {
        const hasCamel = /[A-Z]/.test(token) && /[a-z]/.test(token);
        const hasSnake = token.includes('_');
        if (hasCamel) {
            for (const part of splitCamelCase(token)) {
                if (isValidToken(part)) words.push(part);
            }
        } else if (hasSnake) {
            for (const part of token.toLowerCase().split('_').filter(Boolean)) {
                if (isValidToken(part)) words.push(part);
            }
        } else {
            const lower = token.toLowerCase();
            if (isValidToken(lower)) words.push(lower);
        }
    }

    // ── Step 3: Pure digit sequences ──────────────────────────────────────────
    // Captures numeric IDs, dates, codes, etc. not already emitted from compounds.
    // Deduplication happens in tokenize() via Set.
    const digitRe = /\b\d{2,20}\b/g;
    while ((m = digitRe.exec(rawText)) !== null) {
        words.push(m[0]);
    }

    return words;
}

/**
 * Generate n-gram strings (space-joined) from a word array.
 * @param {string[]} words
 * @param {number} minN
 * @param {number} maxN
 * @returns {string[]}
 */
function buildNGrams(words, minN = 1, maxN = 4) {
    if (!words.length) return [];
    const ngrams = [];
    const upper = Math.min(maxN, words.length);
    for (let n = minN; n <= upper; n++) {
        for (let i = 0; i <= words.length - n; i++) {
            ngrams.push(words.slice(i, i + n).join(' '));
        }
    }
    return ngrams;
}

/**
 * Tokenize text into an array of unique lowercase tokens.
 * With includeNGrams: true, generates n-grams (default 1–4).
 * @param {string} text
 * @param {{ includeNGrams?: boolean, minN?: number, maxN?: number }} options
 * @returns {string[]}
 */
function tokenize(text, options = {}) {
    const words = wordsFromText(text);
    if (!options.includeNGrams) {
        return [...new Set(words)];
    }
    const minN = options.minN || 1;
    const maxN = options.maxN || 4;
    return [...new Set(buildNGrams(words, minN, maxN))];
}

/**
 * Extract meaningful keywords from a Mermaid flowchart diagram string.
 * @param {string} mermaidText
 * @returns {string[]}
 */
function extractMermaidKeywords(mermaidText) {
    const text = String(mermaidText || '').trim();
    if (!text) return [];

    const keywords = new Set();
    const lines = text.split('\n');

    for (const line of lines) {
        const trimmed = line.trim();
        if (/^graph\s+(TD|LR|BT|RL)/i.test(trimmed) || !trimmed) continue;

        const nodeLabelMatches = trimmed.matchAll(/\w+\s*[\[\({"]\s*"?([^"\]\)}"]+)"?\s*[\]\)}"]/g);
        for (const m of nodeLabelMatches) {
            if (m[1]) {
                const label = m[1].trim();
                if (label.length >= 2) {
                    keywords.add(label.toLowerCase());
                    for (const word of label.split(/\s+/)) {
                        if (word.length >= 2) keywords.add(word.toLowerCase());
                    }
                }
            }
        }

        const nodeIdMatches = trimmed.matchAll(/\b([A-Z][a-zA-Z0-9]+)\b/g);
        for (const m of nodeIdMatches) {
            if (m[1] && m[1].length >= 2) {
                const parts = m[1].replace(/([a-z0-9])([A-Z])/g, '$1 $2').toLowerCase().split(/\s+/);
                for (const part of parts) { if (part.length >= 2) keywords.add(part); }
                keywords.add(m[1].toLowerCase());
            }
        }

        const relLabelMatches = trimmed.matchAll(/--+>?\|([^|]+)\|/g);
        for (const m of relLabelMatches) {
            if (m[1]) {
                const label = m[1].trim();
                if (label.length >= 2) {
                    keywords.add(label.toLowerCase());
                    for (const word of label.split(/\s+/)) {
                        if (word.length >= 2) keywords.add(word.toLowerCase());
                    }
                }
            }
        }

        const quotedRelMatches = trimmed.matchAll(/--\s*"([^"]+)"\s*-->/g);
        for (const m of quotedRelMatches) {
            if (m[1]) {
                const label = m[1].trim();
                if (label.length >= 2) {
                    keywords.add(label.toLowerCase());
                    for (const word of label.split(/\s+/)) {
                        if (word.length >= 2) keywords.add(word.toLowerCase());
                    }
                }
            }
        }
    }

    const noise = new Set(['graph', 'td', 'lr', 'bt', 'rl', 'subgraph', 'end', 'style', 'class', 'click']);
    return [...keywords].filter(kw => !noise.has(kw));
}

module.exports = { tokenize, generateSynonyms, wordsFromText, buildNGrams, extractMermaidKeywords };
