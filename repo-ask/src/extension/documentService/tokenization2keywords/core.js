const { STOP_WORDS } = require('./patternMatch');
const { removeMdSyntax } = require('../md2keywords');

const urlRegexStrict = /^https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)$/i;
const numRegexStrict = /^-?\d+(\.\d+)?$/;

function tokenize(text) {
    const rawText = String(text || '');
    if (!rawText.trim()) return [];

    let tokens = [];

    // Remove markdown syntax: headers, bold, italic, inline code
    let processedText = removeMdSyntax(rawText);

    // Split into words by spaces
    const words = processedText.split(/\s+/);

    for (let word of words) {
        let cleanWord = word;
        // Strip leading/trailing punctuation typically wrapping words
        let strippedWord = cleanWord.replace(/^[.,;:"'(\[<]+|[.,;:"')\]>!?-]+$/g, '');
        
        let tokenCandidate = strippedWord || cleanWord;
        tokenCandidate = tokenCandidate.toLowerCase();

        if (tokenCandidate.length <= 2 || STOP_WORDS.has(tokenCandidate)) continue;

        if (urlRegexStrict.test(tokenCandidate) || numRegexStrict.test(tokenCandidate)) {
            tokens.push(tokenCandidate);
        } else {
            tokens.push(tokenCandidate);
            
            // Sub-words split
            const subWords = tokenCandidate.split(/[,;"\s-]+/).filter(t => t.length > 2 && !STOP_WORDS.has(t));
            if (subWords.length > 1) {
                subWords.forEach(t => tokens.push(t));
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
