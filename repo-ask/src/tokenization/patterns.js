const PATTERN_UUID = /^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$/;
const PATTERN_ISIN = /^[A-Za-z]{2}[A-Za-z0-9]{9}[0-9]$/;
const PATTERN_CUSIP = /^[A-Z0-9]{9}$/;
const PATTERN_TICKER = /^[A-Z]{3,5}$/;
const PATTERN_EMAIL = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
const PATTERN_SEDOL = /^[B-DF-HJ-NP-TV-Z0-9]{6}[0-9]$/;
const PATTERN_LEI = /^[A-Z0-9]{20}$/;
const PATTERN_FIGI = /^[B-DF-HJ-NP-TV-Z0-9]{12}$/;
const PATTERN_OPTION_OSI = /^[A-Z]{1,6}\d{6}[CP]\d{8}$/;
const PATTERN_DATE_ISO = /^\d{4}-\d{2}-\d{2}$/;
const PATTERN_DATE_COMMON = /^\d{2}\/\d{2}\/\d{4}$/;
const PATTERN_PRICE = /^\d+\.\d{2}$/;
const PATTERN_NUM = /^\d+$/;
const PATTERN_QUANTITY = /^[$]?[0-9]{1,3}(?:,?[0-9]{3})*(?:\.[0-9]+)?[kKmMbB]?$/;
const PATTERN_ALL_CAPS = /^[A-Z][A-Z0-9_]+$/;
const PATTERN_DATE_COMPACT = /^\d{8}$/;
const PATTERN_DIGIT_CAPS = /^\d+[.,\-/%]*[A-Z]+$/;

const PATTERNS = [
    ['EMAIL', PATTERN_EMAIL],
    ['OPTION_OSI', PATTERN_OPTION_OSI],
    ['UUID', PATTERN_UUID],
    ['ISIN', PATTERN_ISIN],
    ['LEI', PATTERN_LEI],
    ['FIGI', PATTERN_FIGI],
    ['CUSIP', PATTERN_CUSIP],
    ['SEDOL', PATTERN_SEDOL],
    ['DATE_ISO', PATTERN_DATE_ISO],
    ['DATE_COMMON', PATTERN_DATE_COMMON],
    ['DATE_COMPACT', PATTERN_DATE_COMPACT],
    ['PRICE', PATTERN_PRICE],
    ['NUM', PATTERN_NUM],
    ['QUANTITY', PATTERN_QUANTITY],
    ['TICKER', PATTERN_TICKER],
    ['ALL_CAPS', PATTERN_ALL_CAPS],
    ['DIGIT_CAPS', PATTERN_DIGIT_CAPS]
];

module.exports = {
    PATTERNS
};