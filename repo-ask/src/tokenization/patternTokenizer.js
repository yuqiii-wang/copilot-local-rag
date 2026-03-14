/**
 * Tokenizes text based on specific patterns
 * @param {string} text - The input text
 * @returns {string[]} Array of pattern-based tokens
 */
const { PATTERNS } = require('./patterns');

function patternTokenizer(text) {
    const rawText = String(text || '');
    if (!rawText.trim()) return [];
    
    const tokens = [];
    
    // 1. camelCase patterns
    const camelCaseRegex = /[A-Z][a-z]+(?:[A-Z][a-z]+)+/g;
    let match;
    while ((match = camelCaseRegex.exec(rawText)) !== null) {
        const camelCaseTerm = match[0];
        const dashedTerm = camelCaseToDashed(camelCaseTerm);
        tokens.push(dashedTerm);
    }
    
    // 2. snake_case patterns
    const snakeCaseRegex = /[a-z0-9]+(?:_[a-z0-9]+)+/g;
    while ((match = snakeCaseRegex.exec(rawText.toLowerCase())) !== null) {
        const snakeCaseTerm = match[0];
        const dashedTerm = snakeCaseTerm.replace(/_/g, '-');
        tokens.push(dashedTerm);
    }
    
    // 3. Use patterns from patterns.js
    for (const [name, pattern] of PATTERNS) {
        const regex = new RegExp(pattern.source, 'g');
        while ((match = regex.exec(rawText)) !== null) {
            let token = match[0];
            // Convert dates to dashed format
            if (name.includes('DATE')) {
                token = token.replace(/[\/]/g, '-');
            }
            tokens.push(token);
        }
    }
    
    // 4. URL patterns
    const urlRegex = /https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)/g;
    while ((match = urlRegex.exec(rawText)) !== null) {
        tokens.push(match[0]);
    }
    
    // 5. phone number patterns
    const phoneRegex = /\b\d{3}[-.]?\d{3}[-.]?\d{4}\b/g;
    while ((match = phoneRegex.exec(rawText)) !== null) {
        const phoneTerm = match[0].replace(/[-.]/g, '-');
        tokens.push(phoneTerm);
    }
    
    return tokens;
}

/**
 * Converts camelCase to dashed format
 * @param {string} camelCase - The camelCase string
 * @returns {string} The dashed format string
 */
function camelCaseToDashed(camelCase) {
    return camelCase
        .replace(/([a-z])([A-Z])/g, '$1-$2')
        .toLowerCase();
}

module.exports = { patternTokenizer };

