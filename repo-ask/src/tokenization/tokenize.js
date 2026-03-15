function tokenize(text) {
    const rawText = String(text || '');
    if (!rawText.trim()) return [];
    
    let tokens = [];
    
    // First pass: extract markdown title, italic, and bold
    // Extract title (h1)
    const titleMatch = rawText.match(/^#\s+(.+)$/m);
    if (titleMatch) {
        const titleText = titleMatch[1].trim();
        const titleWords = titleText.split(/\s+/);
        for (const word of titleWords) {
            const cleanWord = word.toLowerCase()
                .replace(/[^a-z0-9-]/g, '')
                .replace(/^-+|-+$/g, '');
            if (cleanWord.length > 2) {
                tokens.push(cleanWord);
                tokens.push(cleanWord);
                tokens.push(cleanWord);
            }
        }
    }
    
    // Extract bold text (**text** or __text__)
    const boldMatches = rawText.match(/(?:\*\*|__)([^\*_]+)(?:\*\*|__)/g);
    if (boldMatches) {
        for (const match of boldMatches) {
            const boldText = match.replace(/(?:\*\*|__)/g, '').trim();
            const boldWords = boldText.split(/\s+/);
            for (const word of boldWords) {
                const cleanWord = word.toLowerCase()
                    .replace(/[^a-z0-9-]/g, '')
                    .replace(/^-+|-+$/g, '');
                if (cleanWord.length > 2) {
                    tokens.push(cleanWord);
                    tokens.push(cleanWord);
                }
            }
        }
    }
    
    // Extract italic text (*text* or _text_)
    const italicMatches = rawText.match(/(?:\*|_)([^\*_]+)(?:\*|_)/g);
    if (italicMatches) {
        for (const match of italicMatches) {
            const italicText = match.replace(/(?:\*|_)/g, '').trim();
            const italicWords = italicText.split(/\s+/);
            for (const word of italicWords) {
                const cleanWord = word.toLowerCase()
                    .replace(/[^a-z0-9-]/g, '')
                    .replace(/^-+|-+$/g, '');
                if (cleanWord.length > 2) {
                    tokens.push(cleanWord);
                }
            }
        }
    }
    
    // Split into sentences/paragraphs based on punctuation and newlines
    const sentences = rawText.split(/(?:[.!?\n]+)/).filter(s => s.trim().length > 0);

    for (const sentence of sentences) {
        // Words inside this sentence/paragraph
        const words = sentence.trim().split(/\s+/);
        
        // Process individual words
        for (let i = 0; i < words.length; i++) {
            let word = words[i];
            
            // keep dashes for dashed-linked words, normalize rest
            const cleanWord = word.toLowerCase()
                .replace(/[^a-z0-9-]/g, '')
                .replace(/^-+|-+$/g, ''); // drop leading/trailing dashes
            if (cleanWord.length <= 2) continue;

            // Simple tokens
            tokens.push(cleanWord);
            
            // Favor long vocab and dashed-linked words by adding them extra times
            // This implicitly "favors" them when they are counted by bm25 or ngrams
            if (cleanWord.includes('-')) {
                tokens.push(cleanWord);
                tokens.push(cleanWord);
            }
            if (cleanWord.length > 8) {
                tokens.push(cleanWord);
            }
        }
    }
    
    return tokens;
}

module.exports = { tokenize };
