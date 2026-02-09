import re

# Regex Patterns
# UUID: 8-4-4-4-12 hex digits
PATTERN_UUID = r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
# ISIN: 2 letters, 9 alphanumeric, 1 digit check
PATTERN_ISIN = r'^[A-Za-z]{2}[A-Za-z0-9]{9}[0-9]$'
# CUSIP: 9 alphanumeric characters (Uppercase only to avoid matching common words like "Portfolio", "Calculate")
PATTERN_CUSIP = r'^[A-Z0-9]{9}$'
# Ticker: 3 to 5 uppercase letters (Simple heuristic to avoid noise, e.g. "THE")
PATTERN_TICKER = r'^[A-Z]{3,5}$' 

# Email
PATTERN_EMAIL = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
# SEDOL: 7 chars (no vowels to avoid acronyms, ends with check digit)
PATTERN_SEDOL = r'^[B-DF-HJ-NP-TV-Z0-9]{6}[0-9]$'
# LEI: 20 alphanumeric characters (Legal Entity Identifier)
PATTERN_LEI = r'^[A-Z0-9]{20}$'
# FIGI: 12 alphanumeric characters (no vowels)
PATTERN_FIGI = r'^[B-DF-HJ-NP-TV-Z0-9]{12}$'
# OSI Option Symbol: Root (1-6 letters) + YYMMDD + C/P + Strike (8 digits)
PATTERN_OPTION_OSI = r'^[A-Z]{1,6}\d{6}[CP]\d{8}$'

# Dates: YYYY-MM-DD or DD/MM/YYYY
PATTERN_DATE_ISO = r'^\d{4}-\d{2}-\d{2}$'
PATTERN_DATE_COMMON = r'^\d{2}/\d{2}/\d{4}$'
# Price: Digits, dot, 2 digits
PATTERN_PRICE = r'^\d+\.\d{2}$'
# Pure Number: Just digits
PATTERN_NUM = r'^\d+$'
# Quantity/Money: $1,000, 100k, 10M, 500.5. Can have commas, $, suffixes.
PATTERN_QUANTITY = r'^[$]?[0-9]{1,3}(?:,?[0-9]{3})*(?:\.[0-9]+)?[kKmMbB]?$'

# All Caps Word: At least 2 chars, allows underscores (e.g. MAX_VALUE, RISK_LIMIT)
PATTERN_ALL_CAPS = r'^[A-Z][A-Z0-9_]+$'

# Compact Date: YYYYMMDD
PATTERN_DATE_COMPACT = r'^\d{8}$'

# Digit-Punctuation-Capital: e.g. "100USD", "50%LIMIT", "2023-Q4"
# Digits, optional punctuation, then Capital letters
PATTERN_DIGIT_CAPS = r'^\d+[.,\-/%]*[A-Z]+$'

PATTERNS = [
    ("EMAIL", re.compile(PATTERN_EMAIL)),
    ("OPTION_OSI", re.compile(PATTERN_OPTION_OSI)),
    ("UUID", re.compile(PATTERN_UUID)),
    ("ISIN", re.compile(PATTERN_ISIN)),
    ("LEI", re.compile(PATTERN_LEI)),
    ("FIGI", re.compile(PATTERN_FIGI)),
    ("CUSIP", re.compile(PATTERN_CUSIP)),
    ("SEDOL", re.compile(PATTERN_SEDOL)),
    ("DATE_ISO", re.compile(PATTERN_DATE_ISO)),
    ("DATE_COMMON", re.compile(PATTERN_DATE_COMMON)),
    ("DATE_COMPACT", re.compile(PATTERN_DATE_COMPACT)),
    ("PRICE", re.compile(PATTERN_PRICE)),
    ("NUM", re.compile(PATTERN_NUM)),
    ("QUANTITY", re.compile(PATTERN_QUANTITY)),
    ("TICKER", re.compile(PATTERN_TICKER)),
    ("ALL_CAPS", re.compile(PATTERN_ALL_CAPS)),
    ("DIGIT_CAPS", re.compile(PATTERN_DIGIT_CAPS))
]


def generate_structural_regex(text: str) -> str:
    """
    Generates a regex representing the structural composition of the text.
    E.g. "123" -> "\\d{3}", "ABC" -> "[A-Z]{3}", "Abc" -> "[A-Z][a-z]{2}"
    """
    from itertools import groupby
    
    def get_char_type(char):
        if char.isdigit(): return 'digit'
        if char.isupper(): return 'upper'
        if char.islower(): return 'lower'
        return 'symbol'

    regex_parts = []
    
    for type_name, group in groupby(text, key=get_char_type):
        chars = list(group)
        count = len(chars)

        if type_name == 'digit':
            regex_parts.append(r'\d' + (f'{{{count}}}' if count > 1 else ''))
        elif type_name == 'upper':
            regex_parts.append(r'[A-Z]' + (f'{{{count}}}' if count > 1 else ''))
        elif type_name == 'lower':
            regex_parts.append(r'[a-z]' + (f'{{{count}}}' if count > 1 else ''))
        else: # symbol
            for sym, sym_group in groupby(chars):
                sym_count = len(list(sym_group))
                # re is imported globally
                regex_parts.append(re.escape(sym) + (f'{{{sym_count}}}' if sym_count > 1 else ''))
                
    return "".join(regex_parts)

def generate_ngrams(tokens: list, n_min: int = 1, n_max: int = 5) -> list[str]:
    """
    Generates n-grams from a list of tokens, from n_min to n_max.
    e.g. ["hello", "world"] -> ["hello", "world", "hello_world"]
    """
    ngrams = []
    
    # 1. Add single tokens (unigrams)
    if n_min <= 1:
        ngrams.extend(tokens)
        
    start_n = max(2, n_min)
    
    # 2. Add n-grams (bigrams, trigrams etc)
    for n in range(start_n, n_max + 1):
        if len(tokens) < n:
            continue
        # Using " " to join tokens to match TfidfVectorizer's default n-gram generation
        phrases = [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        ngrams.extend(phrases)
        
    return ngrams

def identify_pattern(token: str) -> str:
    """
    Checks if the token matches any registered pattern.
    Returns the structural regex of the token if matched (e.g. '\\d{7}'), else None.
    """
    # Strip common punctuation that might wrap the token for matching
    clean_token = token.strip(".,;:()[]{}'\"")
    
    for name, regex in PATTERNS:
        if regex.fullmatch(clean_token):
            return generate_structural_regex(clean_token)
    return None

def extract_capital_sequences(text: str) -> list[str]:
    """
    Extracts sequences of 2 or more capitalized words (potential entities/names).
    Example: "Risk Management Policy" -> ["Risk Management Policy"]
    """
    # Finds consecutive words starting with A-Z
    # Must be at least 2 words long
    pattern = r'\b[A-Z]\w*(?:\s+[A-Z]\w*)+\b'
    return re.findall(pattern, text)
