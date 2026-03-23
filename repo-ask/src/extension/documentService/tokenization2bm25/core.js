const { STOP_WORDS } = require('../tokenization2keywords/patternMatch');
const { extractMdKeywordsForBm25 } = require('../md2keywords');

function tokenize(text) {
    const rawText = String(text || '');
    if (!rawText.trim()) return [];

    let tokens = extractMdKeywordsForBm25(rawText, STOP_WORDS);

    const sentences = rawText.split(/(?:[.!?\n]+)/).filter(s => s.trim().length > 0);
    for (const sentence of sentences) {
        const words = sentence.trim().split(/\s+/);
        for (let i = 0; i < words.length; i++) {
            let word = words[i];
            const cleanWord = word.toLowerCase()
                .replace(/^-+|-+$/g, '');
            if (cleanWord.length <= 2 || STOP_WORDS.has(cleanWord)) continue;

            tokens.push(cleanWord);

            // Split at , ; " and hyphens to create additional tokens
            const splitTokens = cleanWord.split(/[,;"\s-]+/).filter(t => t.length > 2 && !STOP_WORDS.has(t));
            splitTokens.forEach(token => tokens.push(token));

            // Add extra weight for words with hyphens or punctuation
            if (cleanWord.includes('-') || /[,;"@#$%^&*(){}[\]\\:;"'=+_]/.test(cleanWord)) {
                tokens.push(cleanWord, cleanWord);
            }
            if (cleanWord.length > 8) {
                tokens.push(cleanWord);
            }
        }
    }

    return tokens;
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

module.exports = { tokenize, generate_ngrams };
