function generate_structural_regex(text) {
    function get_char_type(char) {
        if (char.match(/\d/)) return 'digit';
        if (char.match(/[A-Z]/)) return 'upper';
        if (char.match(/[a-z]/)) return 'lower';
        return 'symbol';
    }

    const regex_parts = [];
    let current_type = null;
    let current_count = 0;
    let current_symbol_char = '';

    for (let i = 0; i < text.length; i++) {
        const char = text[i];
        const char_type = get_char_type(char);

        if (char_type === current_type) {
            current_count++;
        } else {
            if (current_type) {
                if (current_type === 'digit') {
                    regex_parts.push(`\\d${current_count > 1 ? `{${current_count}}` : ''}`);
                } else if (current_type === 'upper') {
                    regex_parts.push(`[A-Z]${current_count > 1 ? `{${current_count}}` : ''}`);
                } else if (current_type === 'lower') {
                    regex_parts.push(`[a-z]${current_count > 1 ? `{${current_count}}` : ''}`);
                } else {
                    const escaped_char = current_symbol_char.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
                    regex_parts.push(`${escaped_char}${current_count > 1 ? `{${current_count}}` : ''}`);
                }
            }
            current_type = char_type;
            current_count = 1;
            current_symbol_char = char_type === 'symbol' ? char : '';
        }
    }

    if (current_type) {
        if (current_type === 'digit') {
            regex_parts.push(`\\d${current_count > 1 ? `{${current_count}}` : ''}`);
        } else if (current_type === 'upper') {
            regex_parts.push(`[A-Z]${current_count > 1 ? `{${current_count}}` : ''}`);
        } else if (current_type === 'lower') {
            regex_parts.push(`[a-z]${current_count > 1 ? `{${current_count}}` : ''}`);
        } else {
            const escaped_char = current_symbol_char.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            regex_parts.push(`${escaped_char}${current_count > 1 ? `{${current_count}}` : ''}`);
        }
    }

    return regex_parts.join('');
}

module.exports = {
    generate_structural_regex
};