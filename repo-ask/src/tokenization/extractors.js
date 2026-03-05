function extract_capital_sequences(text) {
    const pattern = /\b[A-Z]\w*(?:\s+[A-Z]\w*)+\b/g;
    const matches = text.match(pattern) || [];
    return matches;
}

module.exports = {
    extract_capital_sequences
};