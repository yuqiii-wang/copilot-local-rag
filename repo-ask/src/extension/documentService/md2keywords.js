function removeMdSyntax(text) {
    if (!text) return '';
    return text
        .replace(/^#+\s+/gm, '')
        .replace(/(?:\*\*|__)([^\*_]+)(?:\*\*|__)/g, '$1')
        .replace(/(?:\*|_)([^\*_]+)(?:\*|_)/g, '$1')
        .replace(/`([^`]+)`/g, '$1');
}

function extractMdEmphasis(text) {
    const emphases = [];

    // Title / Headers
    const titleMatch = text.match(/^#+\s+(.+)$/gm);
    if (titleMatch) {
        titleMatch.forEach(match => {
            const clean = match.replace(/^#+\s+/, '').trim();
            if (clean) emphases.push({ type: 'header', text: clean });
        });
    }

    // Bold
    const boldMatches = text.match(/(?:\*\*|__)([^\*_]+)(?:\*\*|__)/g);
    if (boldMatches) {
        boldMatches.forEach(match => {
            const clean = match.replace(/(?:\*\*|__)/g, '').trim();
            if (clean) emphases.push({ type: 'bold', text: clean });
        });
    }

    // Italic
    const italicMatches = text.match(/(?:\*|_)([^\*_]+)(?:\*|_)/g);
    if (italicMatches) {
        italicMatches.forEach(match => {
            const clean = match.replace(/(?:\*|_)/g, '').trim();
            if (clean) emphases.push({ type: 'italic', text: clean });
        });
    }

    return emphases;
}

function extractMdKeywordsForBm25(text, STOP_WORDS) {
    const rawText = String(text || '');
    let tokens = [];

    // First pass: extract markdown title, italic, and bold
    const titleMatch = rawText.match(/^#\s+(.+)$/m);
    if (titleMatch) {
        const titleText = titleMatch[1].trim();
        const titleWords = titleText.split(/\s+/);
        for (const word of titleWords) {
            const cleanWord = word.toLowerCase()
                .replace(/^-+|-+$/g, '');
            if (cleanWord.length > 2 && !STOP_WORDS.has(cleanWord)) {
                tokens.push(cleanWord, cleanWord, cleanWord);
                const splitTokens = cleanWord.split(/[,;"\s]+/).filter(t => t.length > 2 && !STOP_WORDS.has(t));
                splitTokens.forEach(token => tokens.push(token));
            }
        }
    }

    const boldMatches = rawText.match(/(?:\*\*|__)([^\*_]+)(?:\*\*|__)/g);
    if (boldMatches) {
        for (const match of boldMatches) {
            const boldText = match.replace(/(?:\*\*|__)/g, '').trim();
            const boldWords = boldText.split(/\s+/);
            for (const word of boldWords) {
                const cleanWord = word.toLowerCase()
                    .replace(/^-+|-+$/g, '');
                if (cleanWord.length > 2 && !STOP_WORDS.has(cleanWord)) {
                    tokens.push(cleanWord, cleanWord);
                    const splitTokens = cleanWord.split(/[,;"\s]+/).filter(t => t.length > 2 && !STOP_WORDS.has(t));
                    splitTokens.forEach(token => tokens.push(token));
                }
            }
        }
    }

    const italicMatches = rawText.match(/(?:\*|_)([^\*_]+)(?:\*|_)/g);
    if (italicMatches) {
        for (const match of italicMatches) {
            const italicText = match.replace(/(?:\*|_)/g, '').trim();
            const italicWords = italicText.split(/\s+/);
            for (const word of italicWords) {
                const cleanWord = word.toLowerCase()
                    .replace(/^-+|-+$/g, '');
                if (cleanWord.length > 2 && !STOP_WORDS.has(cleanWord)) {
                    tokens.push(cleanWord);
                    const splitTokens = cleanWord.split(/[,;"\s]+/).filter(t => t.length > 2 && !STOP_WORDS.has(t));
                    splitTokens.forEach(token => tokens.push(token));
                }
            }
        }
    }
    return tokens;
}

module.exports = {
    removeMdSyntax,
    extractMdEmphasis,
    extractMdKeywordsForBm25
};
