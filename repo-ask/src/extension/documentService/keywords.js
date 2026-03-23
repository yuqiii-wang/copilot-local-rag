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
  return [...new Set(keywordValues.map(value => String(value || '').trim()).filter(value => value.length > 2))].slice(0, safeLimit);
}

function normalizeMetadataKeywordFields(metadata = {}) {
  const base = metadata && typeof metadata === 'object' ? metadata : {};
  const allKeywords = cleanKeywords(base.keywords, 1000);
  
  // Move n-grams from primary keywords to extended keywords
  const nGrams = allKeywords.filter(kw => kw.includes(' '));
  const keywords = cleanKeywords(allKeywords.filter(kw => !kw.includes(' ')), getKeywordConfig().DEFAULT_KEYWORD_LIMIT);

  const tags = cleanKeywords(base.tags, getKeywordConfig().DEFAULT_KEYWORD_LIMIT);
  const referencedQueries = Array.isArray(base.referencedQueries)
    ? [...new Set(base.referencedQueries.map(value => String(value || '').trim()).filter(Boolean))]
    : typeof base.referencedQueries === 'string'
      ? [...new Set(base.referencedQueries.split(',').map(value => value.trim()).filter(Boolean))]
      : [];
      
  const extended = cleanKeywords(generateSynonyms(allKeywords), Infinity);
  const finalExtended = cleanKeywords([...new Set([...extended, ...nGrams])], Infinity);

  return {
    ...base,
    keywords,
    tags,
    referencedQueries,
    synonyms: finalExtended
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

  // Interleave structural and BM25 keywords so both sources remain visible.
  while (merged.length < safeLimit && (index < structural.length || index < model.length)) {
    if (index < structural.length && !merged.includes(structural[index])) {
      merged.push(structural[index]);
    }
    if (merged.length >= safeLimit) {
      break;
    }
    if (index < model.length && !merged.includes(model[index])) {
      merged.push(model[index]);
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
  return merged;
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
