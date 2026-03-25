module.exports = function(context) {
  const { vscode } = context;
  const { generateSynonyms } = require('./tokenization2keywords');

function getKeywordConfig() {
  const initKeywordNum = vscode.workspace.getConfiguration('repoAsk').get('initKeywordNum') || 40;
  return {
    DEFAULT_KEYWORD_LIMIT: initKeywordNum,
    TOKENIZATION_KEYWORD_LIMIT: Math.floor(initKeywordNum / 2),
    BM25_KEYWORD_LIMIT: initKeywordNum - Math.floor(initKeywordNum / 2)
  };
}

function buildKeywordOnlyIndexText(metadata) {
  if (!metadata || typeof metadata !== 'object') {
    return '';
  }
  const {
    DEFAULT_KEYWORD_LIMIT
  } = getKeywordConfig();
  const keywords = cleanKeywords(metadata.keywords, DEFAULT_KEYWORD_LIMIT * 4);
  const tags = cleanKeywords(metadata.tags, DEFAULT_KEYWORD_LIMIT * 4);
  const extended = cleanKeywords(metadata.synonyms, 200);
  return [...keywords, ...tags, ...extended].join(' ');
}

function normalizeKeywordsInput(values) {
  if (Array.isArray(values)) {
    return values;
  }
  if (typeof values === 'string') {
    return values.split(',');
  }
  return [];
}

function cleanKeywords(values, limit = getKeywordConfig().DEFAULT_KEYWORD_LIMIT) {
  const keywordValues = normalizeKeywordsInput(values);
  if (keywordValues.length === 0) {
    return [];
  }
  const safeLimit = Number.isFinite(limit) && limit > 0 ? Math.floor(limit) : getKeywordConfig().DEFAULT_KEYWORD_LIMIT;
  return [...new Set(keywordValues.map(value => String(value || '').trim()).filter(value => value.length >= 2))].slice(0, safeLimit);
}

function normalizeMetadataKeywordFields(metadata = {}) {
  const base = metadata && typeof metadata === 'object' ? metadata : {};
  const allKeywords = cleanKeywords(base.keywords, 1000);

  // Expand all keywords:
  //   - keep the original token (including n-grams)
  //   - for n-grams: also add each individual gram as its own keyword
  //   - for camelCase identifiers: split and add each part
  //   - for snake_case identifiers: split and add each part
  const expanded = [];
  for (const kw of allKeywords) {
    expanded.push(kw); // always keep the original keyword/n-gram
    if (kw.includes(' ')) {
      // n-gram: add every individual gram too (e.g. "fx engine" → "fx", "engine")
      kw.split(/\s+/).filter(g => g.length > 2).forEach(g => expanded.push(g));
    } else if (/[a-z0-9][A-Z]/.test(kw)) {
      // camelCase: split into parts and add each (e.g. "getUserById" → "get", "user", "by", "id")
      const parts = kw
        .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
        .replace(/([A-Z]+)([A-Z][a-z])/g, '$1 $2')
        .toLowerCase()
        .split(/\s+/)
        .filter(p => p.length > 2);
      if (parts.length > 1) parts.forEach(p => expanded.push(p));
    } else if (kw.includes('_')) {
      // snake_case: split into parts and add each (e.g. "trade_event" → "trade", "event")
      const parts = kw.toLowerCase().split('_').filter(p => p.length > 2);
      if (parts.length > 1) parts.forEach(p => expanded.push(p));
    }
  }

  const keywords = cleanKeywords(expanded, getKeywordConfig().DEFAULT_KEYWORD_LIMIT);

  const tags = cleanKeywords(base.tags, getKeywordConfig().DEFAULT_KEYWORD_LIMIT);
  const referencedQueries = Array.isArray(base.referencedQueries)
    ? [...new Set(base.referencedQueries.map(value => String(value || '').trim()).filter(Boolean))]
    : typeof base.referencedQueries === 'string'
      ? [...new Set(base.referencedQueries.split(',').map(value => value.trim()).filter(Boolean))]
      : [];

  // Synonyms: generated synonyms only — n-grams now live in keywords themselves
  const extended = cleanKeywords(generateSynonyms(allKeywords), Infinity);

  return {
    ...base,
    keywords,
    tags,
    referencedQueries,
    synonyms: extended
  };
}

function mergeKeywordsPreservingSignals({
  structuralKeywords = [],
  modelKeywords = [],
  lexicalKeywords = [],
  limit = getKeywordConfig().DEFAULT_KEYWORD_LIMIT
} = {}) {
  const safeLimit = Number.isFinite(limit) && limit > 0 ? Math.floor(limit) : getKeywordConfig().DEFAULT_KEYWORD_LIMIT;
  const structural = cleanKeywords(structuralKeywords, safeLimit * 2);
  const model = cleanKeywords(modelKeywords, safeLimit * 2);
  const lexical = cleanKeywords(lexicalKeywords, safeLimit * 2);
  const merged = [];
  let index = 0;

  // Interleave BM25 (model) and markdown (structural) keywords so both sources remain visible.
  // BM25 keywords come first as requested
  while (merged.length < safeLimit && (index < model.length || index < structural.length)) {
    if (index < model.length && !merged.includes(model[index])) {
      merged.push(model[index]);
    }
    if (merged.length >= safeLimit) {
      break;
    }
    if (index < structural.length && !merged.includes(structural[index])) {
      merged.push(structural[index]);
    }
    index += 1;
  }
  for (const keyword of lexical) {
    if (merged.length >= safeLimit) {
      break;
    }
    if (!merged.includes(keyword)) {
      merged.push(keyword);
    }
  }

  // Post-merge: expand compound keywords so each gram also appears as an individual keyword.
  // This ensures grams are in the stored keywords, not only at search-time normalization.
  const expanded = [...merged];
  for (const kw of merged) {
    if (kw.includes(' ')) {
      for (const g of kw.split(/\s+/)) {
        const trimmed = g.replace(/[\s`.,;:!?\-_()\[\]{}<>\/\\@#$%^&*+=~|]/g, '').trim();
        if (trimmed.length >= 2 && !expanded.includes(trimmed)) expanded.push(trimmed);
      }
    }
    if (/[-_+=$\/]/.test(kw)) {
      for (const p of kw.split(/[-_+=$\/]+/)) {
        const trimmed = p.replace(/[\s`.,;:!?\-_()\[\]{}<>\/\\@#$%^&*+=~|]/g, '').trim();
        const lp = trimmed.toLowerCase();
        if (lp.length >= 2 && !expanded.includes(lp)) expanded.push(lp);
      }
    }
  }
  return expanded;
}

function appendKeywordsToExisting(existingKeywords = [], addedKeywords = [], limit = getKeywordConfig().DEFAULT_KEYWORD_LIMIT) {
  const safeLimit = Number.isFinite(limit) && limit > 0 ? Math.floor(limit) : getKeywordConfig().DEFAULT_KEYWORD_LIMIT;
  const existing = cleanKeywords(existingKeywords, safeLimit * 2);
  const additions = cleanKeywords(addedKeywords, safeLimit * 2);
  const merged = [...existing];
  for (const keyword of additions) {
    if (merged.length >= safeLimit) {
      break;
    }
    if (!merged.includes(keyword)) {
      merged.push(keyword);
    }
  }
  return merged.slice(0, safeLimit);
}

  return {
    getKeywordConfig,
    buildKeywordOnlyIndexText,
    normalizeKeywordsInput,
    cleanKeywords,
    normalizeMetadataKeywordFields,
    mergeKeywordsPreservingSignals,
    appendKeywordsToExisting
  };
};
