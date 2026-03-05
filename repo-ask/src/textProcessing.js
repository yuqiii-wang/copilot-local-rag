const TurndownService = require('turndown');
const {
    generate_ngrams,
    identify_pattern,
    extract_capital_sequences
} = require('./tokenization');

const turndownService = new TurndownService({
    headingStyle: 'atx',
    codeBlockStyle: 'fenced',
    bulletListMarker: '-',
    emDelimiter: '_',
    strongDelimiter: '**'
});

function truncate(value, maxLen) {
    if (!value || value.length <= maxLen) {
        return value;
    }
    return `${value.slice(0, maxLen)}...`;
}

function normalizeText(text) {
    return String(text || '')
        .toLowerCase()
        .replace(/[^a-z0-9\s]/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();
}

function tokenize(text) {
    const normalized = normalizeText(text);
    if (!normalized) {
        return [];
    }
    return normalized.split(' ').filter(token => token.length > 2);
}

function htmlToMarkdown(html) {
    return turndownService.turndown(String(html || '')).trim();
}

function generateKeywords(text) {
    const baseTokens = tokenize(text);
    const capitalSequences = extract_capital_sequences(text || '');

    const structural = [];
    for (const token of baseTokens) {
        const pattern = identify_pattern(token);
        if (pattern) {
            structural.push(pattern);
        }
    }

    const ngrams = generate_ngrams(baseTokens, 1, 3);
    const keywords = [...new Set([...capitalSequences, ...structural, ...ngrams])];
    return keywords.filter(keyword => keyword.length > 2).slice(0, 40);
}

function generateSummary(text, maxLength = 220) {
    const sentences = String(text || '').split(/[.!?]+/).filter(s => s.trim().length > 0);
    let summary = '';
    let length = 0;

    for (const sentence of sentences) {
        const sentenceTrimmed = sentence.trim();
        if (length + sentenceTrimmed.length + 1 <= maxLength) {
            summary += sentenceTrimmed + '. ';
            length += sentenceTrimmed.length + 1;
        } else {
            break;
        }
    }

    return summary.trim() || truncate(String(text || '').trim(), maxLength) || 'No summary available';
}

module.exports = {
    truncate,
    tokenize,
    htmlToMarkdown,
    generateKeywords,
    generateSummary
};