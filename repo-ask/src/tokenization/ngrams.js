function generate_ngrams(tokens, n_min = 1, n_max = 5) {
    const ngrams = [];

    if (n_min <= 1) {
        ngrams.push(...tokens);
    }

    const start_n = Math.max(2, n_min);

    for (let n = start_n; n <= n_max; n++) {
        if (tokens.length < n) {
            continue;
        }
        const phrases = [];
        for (let i = 0; i <= tokens.length - n; i++) {
            phrases.push(tokens.slice(i, i + n).join(' '));
        }
        ngrams.push(...phrases);
    }

    return ngrams;
}

module.exports = {
    generate_ngrams
};