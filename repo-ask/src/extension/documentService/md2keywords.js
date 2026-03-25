const { STOP_WORDS } = require('./tokenization2keywords/patternMatch');

function extractMdEmphasis(text) {
    const emphases = [];
    let remainingText = text;

    // Title / Headers
    const titleMatch = remainingText.match(/^#+\s+(.+)$/gm);
    if (titleMatch) {
        titleMatch.forEach(match => {
            const clean = match.replace(/^#+\s+/, '').trim();
            if (clean) emphases.push({ type: 'header', text: clean });
        });
    }

    // Code blocks (```lang ... ``` and ~~~lang ... ~~~) - extract first to avoid matching inside
    const codeBlockRegex = /(```[\s\S]*?```|~~~[\s\S]*?~~~)/g;
    let codeBlockMatch;
    while ((codeBlockMatch = codeBlockRegex.exec(remainingText)) !== null) {
        const match = codeBlockMatch[0];
        const clean = match.replace(/```[\s\S]*?\n/, '').replace(/```$/, '').replace(/~~~[\s\S]*?\n/, '').replace(/~~~$/, '').trim();
        if (clean) emphases.push({ type: 'code', text: clean });
    }

    // Replace code blocks with placeholder to avoid matching inside them
    remainingText = remainingText.replace(codeBlockRegex, '');

    // Inline code (`code`)
    const inlineCodeRegex = /`([^`]+)`/g;
    let inlineCodeMatch;
    while ((inlineCodeMatch = inlineCodeRegex.exec(remainingText)) !== null) {
        const clean = inlineCodeMatch[1].trim();
        if (clean) emphases.push({ type: 'inlineCode', text: clean });
    }

    // Replace inline code with placeholder
    remainingText = remainingText.replace(inlineCodeRegex, '');

    // Images ![alt](url) - extract before links to avoid confusion
    const imageRegex = /!\[([^\]]*)\]\(([^)]+)\)/g;
    let imageMatch;
    while ((imageMatch = imageRegex.exec(remainingText)) !== null) {
        const alt = imageMatch[1].trim();
        if (alt) emphases.push({ type: 'image', text: alt });
    }
    remainingText = remainingText.replace(imageRegex, '');

    // Links [text](url)
    const linkRegex = /\[([^\]]+)\]\(([^)]+)\)/g;
    let linkMatch;
    while ((linkMatch = linkRegex.exec(remainingText)) !== null) {
        const linkText = linkMatch[1].trim();
        if (linkText) emphases.push({ type: 'link', text: linkText });
    }
    remainingText = remainingText.replace(linkRegex, '');

    // Bold
    const boldRegex = /(?:\*\*|__)([^\*_]+?)(?:\*\*|__)/g;
    let boldMatch;
    while ((boldMatch = boldRegex.exec(remainingText)) !== null) {
        const clean = boldMatch[1].trim();
        if (clean) emphases.push({ type: 'bold', text: clean });
    }
    remainingText = remainingText.replace(boldRegex, '');

    // Strikethrough
    const strikeRegex = /~~([^~]+?)~~/g;
    let strikeMatch;
    while ((strikeMatch = strikeRegex.exec(remainingText)) !== null) {
        const clean = strikeMatch[1].trim();
        if (clean) emphases.push({ type: 'strikethrough', text: clean });
    }
    remainingText = remainingText.replace(strikeRegex, '');

    // Italic - improved non-greedy regex that avoids matching inside other formatting
    const italicRegex = /(?:\*|_)([^\*_]+?)(?:\*|_)/g;
    let italicMatch;
    while ((italicMatch = italicRegex.exec(remainingText)) !== null) {
        const clean = italicMatch[1].trim();
        if (clean) emphases.push({ type: 'italic', text: clean });
    }

    return emphases;
}

function extractMdKeywords(text) {
    const rawText = String(text || '');
    let tokens = [];

    const emphases = extractMdEmphasis(rawText);
    
    for (const emphasis of emphases) {
        const { text: emphasisText } = emphasis;
        
        // Preserve dashes/underscores within words so compounds stay intact
        const sanitizedText = emphasisText.toLowerCase()
            .replace(/[^\w\s\-]/g, ' ')
            .replace(/^-+|-+$/g, ' ')
            .trim();
        
        const splitTokens = sanitizedText
            .split(/\s+/)
            .filter(t => t.length > 2 && t.length <= 50 && !STOP_WORDS.has(t));
        
        // Preserve the full emphasis phrase as an n-gram keyword (md syntax already stripped)
        if (splitTokens.length > 1) {
            tokens.push(splitTokens.join(' '));
        }
        splitTokens.forEach(token => tokens.push(token));
    }
    
    return [...new Set(tokens)];
}

module.exports = {
    extractMdEmphasis,
    extractMdKeywords
};
