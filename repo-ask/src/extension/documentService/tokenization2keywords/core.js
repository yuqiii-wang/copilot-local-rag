const { STOP_WORDS, PATTERN_URL_STRICT, PATTERN_NUM_STRICT } = require('./patternMatch');

const STRUCTURAL_SEP_REGEX = /[-_+=$\/]/;

/**
 * Returns compound phrases formed by structural-separator words.
 * e.g. "trade-event" → "trade event", "account_balance" → "account balance"
 * Kept separate from tokenize() so compound phrases are surfaced as primary keywords
 * in index.js and do NOT feed generate_ngrams (which would create cross-compound noise).
 */
function extractStructuralCompounds(text) {
    const rawText = String(text || '');
    if (!rawText.trim()) return [];
    const compounds = [];
    for (const rawWord of rawText.split(/\s+/)) {
        if (!STRUCTURAL_SEP_REGEX.test(rawWord)) continue;
        const stripped = rawWord.replace(/^[^\w]+|[^\w]+$/g, '');
        if (!stripped) continue;
        const allParts = stripped.toLowerCase().split(/[-_+=$\/]+/).filter(p => p.length > 0);
        if (allParts.length > 1) compounds.push(stripped.toLowerCase());
    }
    return [...new Set(compounds)];
}

function tokenize(text) {
    const rawText = String(text || '');
    if (!rawText.trim()) return [];

    let tokens = [];

    // Replace punctuations/symbols with space, but keep dashes so compounds
    // like "lst-1234-5678" survive as single tokens.
    // Underscore (_) is \w so snake_case identifiers survive as-is.
    const sanitizedText = rawText.replace(/[^\w\s\-]/g, ' ');

    // Split into words by spaces
    const words = sanitizedText.split(/\s+/);

    for (let word of words) {
        const tokenCandidate = word.trim().toLowerCase();

        if (tokenCandidate.length <= 1 || STOP_WORDS.has(tokenCandidate)) continue;

        if (PATTERN_URL_STRICT.test(tokenCandidate) || PATTERN_NUM_STRICT.test(tokenCandidate)) {
            tokens.push(tokenCandidate);
        } else {
            tokens.push(tokenCandidate);

            // Sub-words: split on remaining separators (dashes, commas, etc.)
            const subWords = tokenCandidate.split(/[,;"\s\-]+/).filter(t => t.length >= 2 && !STOP_WORDS.has(t));
            if (subWords.length > 1) {
                subWords.forEach(t => tokens.push(t));
            }
        }
    }

    return [...new Set(tokens)];
}

function generate_ngrams(tokens, n_min = 1, n_max = 5) {
    const ngrams = new Set();
    if (n_min <= 1) {
        tokens.forEach(token => ngrams.add(token));
    }
    const start_n = Math.max(2, n_min);
    for (let n = start_n; n <= n_max; n++) {
        if (tokens.length < n) continue;
        for (let i = 0; i <= tokens.length - n; i++) {
            const tokenSlice = tokens.slice(i, i + n);
            const dedupTokens = Array.from(new Set(tokenSlice));
            if (dedupTokens.length > 0) {
                const phrase = dedupTokens.join(' ');
                ngrams.add(phrase);
            }
        }
    }
    return Array.from(ngrams);
}

module.exports = { tokenize, generate_ngrams, extractStructuralCompounds };
