const { tokenize } = require('./core');

/**
 * Generates sliding window n-grams for n=1, 2, 3, and 4
 * Each complete n-gram is treated as a single token
 */
function generateSlidingWindowNgrams(tokens) {
    const ngrams = new Set();
    
    // 1-grams
    for (const token of tokens) {
        ngrams.add(token);
    }
    
    // 2-grams
    for (let i = 0; i <= tokens.length - 2; i++) {
        const ngram = tokens[i] + ' ' + tokens[i + 1];
        ngrams.add(ngram);
    }
    
    // 3-grams
    for (let i = 0; i <= tokens.length - 3; i++) {
        const ngram = tokens[i] + ' ' + tokens[i + 1] + ' ' + tokens[i + 2];
        ngrams.add(ngram);
    }
    
    // 4-grams
    for (let i = 0; i <= tokens.length - 4; i++) {
        const ngram = tokens[i] + ' ' + tokens[i + 1] + ' ' + tokens[i + 2] + ' ' + tokens[i + 3];
        ngrams.add(ngram);
    }
    
    return Array.from(ngrams);
}

/**
 * Tokenizes text and generates sliding window n-grams for BM25
 */
function tokenization2bm25(text) {
    const baseTokens = tokenize(text);
    return generateSlidingWindowNgrams(baseTokens);
}

module.exports = {
    tokenization2bm25,
    generateSlidingWindowNgrams
};
