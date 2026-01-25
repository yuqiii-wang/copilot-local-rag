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
# Dates: YYYY-MM-DD or DD/MM/YYYY
PATTERN_DATE_ISO = r'^\d{4}-\d{2}-\d{2}$'
PATTERN_DATE_COMMON = r'^\d{2}/\d{2}/\d{4}$'
# Price: Digits, dot, 2 digits
PATTERN_PRICE = r'^\d+\.\d{2}$'
# Pure Number: Just digits
PATTERN_NUM = r'^\d+$'

PATTERNS = [
    ("UUID", re.compile(PATTERN_UUID)),
    ("ISIN", re.compile(PATTERN_ISIN)),
    ("CUSIP", re.compile(PATTERN_CUSIP)),
    ("DATE_ISO", re.compile(PATTERN_DATE_ISO)),
    ("DATE_COMMON", re.compile(PATTERN_DATE_COMMON)),
    ("PRICE", re.compile(PATTERN_PRICE)),
    ("NUM", re.compile(PATTERN_NUM)),
    ("TICKER", re.compile(PATTERN_TICKER))
]

def identify_pattern(token: str) -> str:
    """
    Checks if the token matches any registered pattern.
    Returns the pattern name (e.g. 'PATTERN_ISIN') or None.
    """
    # Strip common punctuation that might wrap the token for matching
    clean_token = token.strip(".,;:()[]{}'\"")
    
    for name, regex in PATTERNS:
        if regex.fullmatch(clean_token):
            return f"pattern_{name.lower()}"
    return None
