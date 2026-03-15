const TurndownService = require('turndown');
const {
    tokenize: tokenizeFromTokenization,
    generateNGrams
} = require('./tokenization');

// Simple n-gram generator for backward compatibility
function generate_ngrams(tokens, minSize = 1, maxSize = 2) {
    const ngrams = [];
    for (let n = minSize; n <= maxSize; n++) {
        if (tokens.length >= n) {
            for (let i = 0; i <= tokens.length - n; i++) {
                ngrams.push(tokens.slice(i, i + n).join(' '));
            }
        }
    }
    return ngrams;
}

// Simple capital sequence extractor for backward compatibility
function extract_capital_sequences(text) {
    const capitalRegex = /\b[A-Z][A-Z]+\b/g;
    const matches = text.match(capitalRegex) || [];
    return [...new Set(matches)];
}

// Simple extended keywords generator for backward compatibility
function generateExtendedKeywordsFromTokenization(keywords) {
    return keywords.map(keyword => keyword.toLowerCase());
}

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

function extractPatternCandidateTokens(text) {
    const input = String(text || '');
    if (!input.trim()) {
        return [];
    }

    const tokenPattern = /https?:\/\/[^\s)]+|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|\b\d{4}-\d{2}-\d{2}\b|\b\d{2}\/\d{2}\/\d{4}\b|\b\d{8}\b|\b[$]?[0-9]{1,3}(?:,?[0-9]{3})*(?:\.[0-9]+)?[kKmMbB]?\b|\b[A-Za-z0-9]+(?:[-_/.][A-Za-z0-9]+)+\b|\b\d+[.,\-\/%]*[A-Z]+\b|\b[A-Za-z][A-Za-z0-9_]*\b/g;
    const matches = input.match(tokenPattern) || [];
    return [...new Set(matches.map(token => String(token || '').trim()).filter(token => token.length > 1))];
}

function addKeywordScore(scoreMap, keyword, score) {
    const value = String(keyword || '').trim();
    if (!value || value.length <= 2 || !Number.isFinite(score) || score <= 0) {
        return;
    }

    scoreMap.set(value, (scoreMap.get(value) || 0) + score);
}

function getMarkdownLineWeight(line, inCodeFence) {
    if (inCodeFence) {
        return 2;
    }

    const trimmed = String(line || '').trim();
    if (!trimmed) {
        return 0;
    }

    if (/^####\s+/.test(trimmed)) {
        return 7;
    }
    if (/^###\s+/.test(trimmed)) {
        return 8;
    }
    if (/^##\s+/.test(trimmed)) {
        return 10;
    }
    if (/^#\s+/.test(trimmed)) {
        return 12;
    }
    if (/^[-*+]\s+/.test(trimmed) || /^\d+\.\s+/.test(trimmed)) {
        return 5;
    }
    if (/^>\s+/.test(trimmed)) {
        return 4;
    }

    return 1;
}

function stripMarkdownPrefix(line) {
    return String(line || '')
        .replace(/^#{1,6}\s+/, '')
        .replace(/^[-*+]\s+/, '')
        .replace(/^\d+\.\s+/, '')
        .replace(/^>\s+/, '')
        .trim();
}

function htmlToMarkdown(html) {
    return turndownService.turndown(String(html || '')).trim();
}

function generateKeywords(text) {
    const sourceText = String(text || '');
    if (!sourceText.trim()) {
        return [];
    }

    const scoreMap = new Map();
    const lines = sourceText.split(/\r?\n/);
    let inCodeFence = false;

    for (const rawLine of lines) {
        const trimmedLine = String(rawLine || '').trim();
        if (/^```/.test(trimmedLine)) {
            inCodeFence = !inCodeFence;
            continue;
        }

        const lineWeight = getMarkdownLineWeight(trimmedLine, inCodeFence);
        if (lineWeight <= 0) {
            continue;
        }

        const lineText = stripMarkdownPrefix(rawLine);
        if (!lineText) {
            continue;
        }

        // Give title (h1) words even higher priority
        const isTitle = /^#\s+/.test(trimmedLine);
        const adjustedLineWeight = isTitle ? lineWeight * 1.5 : lineWeight;

        const patternCandidates = extractPatternCandidateTokens(lineText);
        for (const token of patternCandidates) {
            addKeywordScore(scoreMap, token, adjustedLineWeight * 3);
        }

        const capitalSequences = extract_capital_sequences(lineText);
        for (const capitalSequence of capitalSequences) {
            addKeywordScore(scoreMap, capitalSequence, adjustedLineWeight * 2);
        }

        const baseTokens = tokenizeFromTokenization(lineText);
        const ngrams = generate_ngrams(baseTokens, 1, 2);
        for (const ngram of ngrams) {
            const ngramWeight = ngram.includes(' ') ? adjustedLineWeight * 0.85 : adjustedLineWeight;
            addKeywordScore(scoreMap, ngram, ngramWeight);
        }
    }

    return [...scoreMap.entries()]
        .sort((left, right) => {
            if (right[1] !== left[1]) {
                return right[1] - left[1];
            }
            return left[0].localeCompare(right[0]);
        })
        .map(([keyword]) => keyword)
        .slice(0, 40);
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

function generateExtendedKeywords(keywords) {
    return generateExtendedKeywordsFromTokenization(Array.isArray(keywords) ? keywords : []);
}

module.exports = {
    truncate,
    tokenize: tokenizeFromTokenization,
    htmlToMarkdown,
    generateKeywords,
    generateSummary,
    generateExtendedKeywords
};
