const { STOP_WORDS } = require('../tokenization2keywords/patternMatch');

function tokenize(text) {
    const rawText = String(text || '');
    if (!rawText.trim()) return [];

    let tokens = [];

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
        }
    }

    return [...new Set(tokens)];
}


module.exports = { tokenize };
