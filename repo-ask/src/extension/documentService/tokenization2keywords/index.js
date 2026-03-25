const { tokenize, generate_ngrams, extractStructuralCompounds } = require('./core');
const {
    patternTokenizer,
    extract_capital_sequences,
    STOP_WORDS,
} = require('./patternMatch');
const { generateSynonyms } = require('./synonyms');

/**
 * Splits a camelCase or snake_case identifier into word parts.
 * Returns ALL parts (no length filter) so the full phrase is preserved.
 * Caller is responsible for filtering parts when adding them individually.
 */
function splitCompoundIdentifier(token) {
    // camelCase / PascalCase: lowercase-to-uppercase OR consecutive-uppercase-then-lowercase
    // e.g. getUserById, FXEngine, HTMLParser
    if (/[a-z0-9][A-Z]/.test(token) || /[A-Z]{2,}[a-z]/.test(token)) {
        const parts = token
            .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
            .replace(/([A-Z]+)([A-Z][a-z])/g, '$1 $2')
            .toLowerCase()
            .split(/\s+/)
            .filter(p => p.length > 0);
        if (parts.length > 1) return parts;
    }
    // snake_case: contains underscore with multiple segments
    if (token.includes('_')) {
        const parts = token.toLowerCase().split('_').filter(p => p.length > 0);
        if (parts.length > 1) return parts;
    }
    return null;
}

const MAX_PARENTHESIS_KEYWORD_TOKENS = 10;

function extractParenthesizedKeywords(text, maxTokens = MAX_PARENTHESIS_KEYWORD_TOKENS) {
    const rawText = String(text || '');
    if (!rawText.trim()) {
        return [];
    }

    const phrases = [];
    const regex = /\(([^()]+)\)/g;
    let match;

    while ((match = regex.exec(rawText)) !== null) {
        const candidate = String(match[1] || '').replace(/\s+/g, ' ').trim();
        if (!candidate) {
            continue;
        }

        const tokenCount = candidate.split(/\s+/).filter(Boolean).length;
        if (tokenCount > 0 && tokenCount <= maxTokens) {
            phrases.push(candidate);
        }
    }

    return [...new Set(phrases)];
}

/**
 * Centralized API for tokenization functionality
 */
function tokenizeText(text, options = {}) {
    const {
        includeNGrams = true,
        includePatterns = true,
        nGramMin = 1,
        nGramMax = 4,
    } = options;

    let primaryTokens = [];
    let secondaryTokens = [];

    const parenthesizedKeywords = extractParenthesizedKeywords(text);
    if (parenthesizedKeywords.length > 0) {
        primaryTokens = primaryTokens.concat(parenthesizedKeywords);
    }

    // Structural-separator compound phrases as primary keywords
    // e.g. "trade-event" → "trade event"; "account_balance" → "account balance"
    const structuralCompounds = extractStructuralCompounds(text);
    if (structuralCompounds.length > 0) {
        primaryTokens = primaryTokens.concat(structuralCompounds);
    }

    // Base tokenization (single-word tokens only — compound phrases excluded to keep n-grams clean)
    const baseTokens = tokenize(text);
    secondaryTokens = secondaryTokens.concat(baseTokens);

    // N-grams and capital sequences
    if (includeNGrams) {
        const nGrams = generate_ngrams(baseTokens, nGramMin, nGramMax);
        secondaryTokens = secondaryTokens.concat(nGrams);

        const capitalSequences = extract_capital_sequences(text);
        if (capitalSequences && capitalSequences.length > 0) {
            primaryTokens = primaryTokens.concat(capitalSequences);
        }
    }

    // Pattern-based tokens: detect camelCase/snake_case identifiers in raw text and
    // add the split phrase + meaningful parts as high-priority primary keywords.
    if (includePatterns) {
        const CAMEL_BROAD = /\b[a-zA-Z][a-z0-9]*(?:[A-Z][a-z0-9]*)+\b/g;
        const SNAKE_BROAD = /\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b/g;
        const identifiers = new Set();
        let m;
        while ((m = CAMEL_BROAD.exec(text)) !== null) identifiers.add(m[0]);
        while ((m = SNAKE_BROAD.exec(text)) !== null) identifiers.add(m[0]);
        for (const id of identifiers) {
            const parts = splitCompoundIdentifier(id);
            if (parts && parts.length >= 2) {
                primaryTokens.push(parts.join(' ')); // e.g. "fx engine", "get user by id"
                parts.filter(p => p.length >= 2 && !STOP_WORDS.has(p)).forEach(p => primaryTokens.push(p));
            }
            secondaryTokens.push(id.toLowerCase());
        }
        // All other pattern tokens (emails, tickers, ISINs, etc.)
        const patternTokens = patternTokenizer(text);
        secondaryTokens = secondaryTokens.concat(patternTokens);
    }

    // Deduplicate tokens while preserving order
    const uniqueTokens = [];
    const seen = new Set();
    const allTokens = [...primaryTokens, ...secondaryTokens];

    for (const token of allTokens) {
        if (!seen.has(token)) {
            seen.add(token);
            uniqueTokens.push(token);
        }
    }

    return uniqueTokens;
}

module.exports = {
    tokenize: tokenizeText,
    generateSynonyms
};
