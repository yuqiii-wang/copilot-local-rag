function tokenize(text) {
    const rawText = String(text || '');
    if (!rawText.trim()) return [];
    
    // Split into sentences/paragraphs based on punctuation and newlines
    const sentences = rawText.split(/(?:[.!?\n]+)/).filter(s => s.trim().length > 0);
    
    let tokens = [];

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
