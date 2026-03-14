const { tokenize } = require('./tokenize');
const { generate_ngrams } = require('./ngrams');
const { patternTokenizer } = require('./patternTokenizer');
const { extract_capital_sequences } = require('./extractors');
const { generateExtendedKeywords } = require('./extendedKeywords');
const { generate_structural_regex } = require('./structural');
const { PATTERNS } = require('./patterns');

/**
 * Centralized API for tokenization functionality
 */
function tokenizeText(text, options = {}) {
    const {
        includeNGrams = true,
        includePatterns = true,
        nGramMin = 1,
        nGramMax = 3,
        prioritizeDashes = true
    } = options;
    
    let tokens = [];
    
    // Base tokenization
    const baseTokens = tokenize(text);
    tokens = tokens.concat(baseTokens);
    
    // Add n-grams if enabled
    if (includeNGrams) {
        const nGrams = generate_ngrams(baseTokens, nGramMin, nGramMax);
        tokens = tokens.concat(nGrams);
    }
    
    // Add pattern-based tokens if enabled
    if (includePatterns) {
        const patternTokens = patternTokenizer(text);
        tokens = tokens.concat(patternTokens);
    }
    
    // Deduplicate tokens while preserving order
    const uniqueTokens = [];
    const seen = new Set();
    
    for (const token of tokens) {
        if (!seen.has(token)) {
            seen.add(token);
            uniqueTokens.push(token);
        }
    }
    
    return uniqueTokens;
}

module.exports = {
    tokenize: tokenizeText,
    tokenizeOriginal: tokenize,
    generateNGrams: generate_ngrams,
    patternTokenizer,
    extractCapitalSequences: extract_capital_sequences,
    generateExtendedKeywords,
    generateStructuralRegex: generate_structural_regex,
    PATTERNS
};
