module.exports = function(context) {
  const { vscode, storagePath, tokenize, readAllMetadata, readDocumentContent,
    normalizeCategorizedKeywords } = context;

  // Tokenizer that supports includeNGrams option
  const { tokenize: kwTokenize } = require('./tokenization2keywords');

  // Default cutoff ratio — docs scoring below this fraction of the top score are removed.
  const DEFAULT_TOP_SCORE_THRESHOLD_RATIO = 0.3;

  // ---------------------------------------------------------------------------
  // N-gram grouping helper
  // ---------------------------------------------------------------------------

  /**
   * Group a flat array of n-gram strings by word count into the same
   * slot names used in the keyword category objects.
   * @param {string[]} ngrams
   * @returns {{ '1gram': string[], '2gram': string[], '3gram': string[], '4gram': string[] }}
   */
  function groupNGramsBySize(ngrams) {
    const result = { '1gram': [], '2gram': [], '3gram': [], '4gram': [] };
    for (const ng of ngrams) {
      const n = String(ng || '').trim().split(/\s+/).length;
      const key = n >= 4 ? '4gram' : `${n}gram`;
      result[key].push(ng.toLowerCase());
    }
    return result;
  }

  // ---------------------------------------------------------------------------
  // Character-length multiplier
  // ---------------------------------------------------------------------------

  /**
   * Returns a score multiplier based on the non-whitespace char length of `token`.
   * Tokens shorter than 7 chars return 1.0 (no boost).
   * Each char at or beyond 7 adds 0.1: 7→1.1, 8→1.2, 9→1.3, …
   * @param {string} token
   * @returns {number}
   */
  function charLengthMultiplier(token) {
    const len = String(token || '').replace(/\s+/g, '').length;
    return len >= 7 ? 1.0 + (len - 6) * 0.1 : 1.0;
  }

  // ---------------------------------------------------------------------------
  // Referenced-query cross-doc boost (stub — extend with neighbour propagation)
  // ---------------------------------------------------------------------------
  function applyReferencedQueryNeighborBoost(scoredDocs) {
    return scoredDocs;
  }

  // ---------------------------------------------------------------------------
  // Main ranking function
  // ---------------------------------------------------------------------------

  /**
   * Score all stored documents against `query` using weight groups from `repoAsk.searchWeights`,
   * then sort and apply threshold cutoff.
   *
   * Scoring layers (additive):
   *   1. WHOLE_QUERY_WEIGHTS — whole query string matched (exact) against id/title/keywords/tags/type/referencedQueries
   *   2. TERM_WEIGHTS        — each 1-gram query token matched (substring) against the same fields
   *   3. KEYWORD_WEIGHTS     — per-category × per-gram: query n-grams matched against doc keyword count-maps,
   *                             each match contributes stored term-frequency × category weight
   *   4. Referenced-query neighbor boost (stub)
   *
   * @param {string} query
   * @param {number} limit  Hard cap before threshold is applied (default 20).
   * @returns {object[]}    Scored metadata entries sorted by score descending.
   */
  function rankLocalDocuments(query, limit = 20) {
    const queryStr = String(query || '').trim();
    if (!queryStr) return [];

    const config = vscode.workspace.getConfiguration('repoAsk');
    const SW  = config.get('searchWeights') || {};
    const maxResults             = Number(config.get('maxSearchResults')           || 5);
    const topScoreThresholdRatio = Number(config.get('topScoreThresholdRatio')     || DEFAULT_TOP_SCORE_THRESHOLD_RATIO);

    const WQ  = SW.WHOLE_QUERY_WEIGHTS     || {};
    const TW  = SW.TERM_WEIGHTS            || {};
    const KW  = SW.keywords                || {};
    const SYN = SW.synonyms                || {};

    const feedbackUrl = String(config.get('logActionConfluenceUrl') || '').trim().replace(/\/$/, '');
    const allMetadata = readAllMetadata(storagePath).filter(m => {
      if (!feedbackUrl) return true;
      const docUrl = String(m.url || '').trim().replace(/\/$/, '');
      return !docUrl || (
        !feedbackUrl.includes(docUrl) &&
        !docUrl.includes(feedbackUrl) &&
        !feedbackUrl.endsWith(`/${String(m.id)}`)
      );
    });
    if (!allMetadata.length) return [];
    console.log(`[RepoAsk] loaded ${allMetadata.length} docs from store`);

    const lowerQuery = queryStr.toLowerCase();
    const queryTerms = kwTokenize(queryStr);                                           // 1-grams
    const queryGrams = groupNGramsBySize(kwTokenize(queryStr, { includeNGrams: true })); // 1–4 grams

    // ── 0. Tag-pinned docs — full tag match guarantees inclusion ────────────
    // A doc is tag-pinned when any of its tags appears in full (case-insensitive)
    // somewhere inside the query string.
    const tagPinnedIds = new Set();
    for (const meta of allMetadata) {
      const docTags = (Array.isArray(meta.tags) ? meta.tags : [])
        .map(t => String(t).toLowerCase().trim()).filter(Boolean);
      if (docTags.some(t => lowerQuery.includes(t))) {
        tagPinnedIds.add(String(meta.id));
      }
    }
    if (tagPinnedIds.size) {
      console.log(`[RepoAsk] tag-pinned docs: ${[...tagPinnedIds].join(', ')}`);
    }

    const scoredDocs = [];

    for (const meta of allMetadata) {
      let score = 0;
      const breakdown = {};
      const track = (key, delta) => { if (delta) { score += delta; breakdown[key] = (breakdown[key] || 0) + delta; } };
      const norm       = normalizeCategorizedKeywords(meta.keywords);
      const lowerTitle = String(meta.title || '').toLowerCase();
      const lowerId    = String(meta.id    || '').toLowerCase();
      const lowerType  = String(meta.type  || '').toLowerCase();
      const tags  = (Array.isArray(meta.tags) ? meta.tags : []).map(t => String(t).toLowerCase());
      const refQs = (Array.isArray(meta.referencedQueries) ? meta.referencedQueries : [])
                      .map(rq => String(rq).toLowerCase());

      // norm[cat][gram] is now a count-map object { keyword: count }.
      // Build a flat set of all known keywords for WQ.keywords matching.
      const kwCats  = ['title', 'structural', 'semantic', 'bm25', 'kg', 'synonyms'];
      const kwGrams = ['1gram', '2gram', '3gram', '4gram'];
      const keywordFlatSet = new Set();
      for (const cat of kwCats)
        for (const gram of kwGrams)
          for (const kw of Object.keys(norm[cat][gram] || {})) keywordFlatSet.add(kw);

      // ── 1. WHOLE_QUERY_WEIGHTS ─────────────────────────────────────────────
      if (WQ.id     && lowerId.includes(lowerQuery))                                          track('WQ.id',                WQ.id);
      if (WQ.title  && lowerTitle.includes(lowerQuery))                                       track('WQ.title',             WQ.title);
      if (WQ.type   && lowerType.includes(lowerQuery))                                        track('WQ.type',              WQ.type);
      if (WQ.tags   && tags.some(t => t.includes(lowerQuery)))                                track('WQ.tags',              WQ.tags);
      if (WQ.keywords && keywordFlatSet.has(lowerQuery))                                      track('WQ.keywords',          WQ.keywords);
      if (WQ.referencedExact   && refQs.includes(lowerQuery))                                 track('WQ.referencedExact',   WQ.referencedExact);
      if (WQ.referencedPartial && refQs.some(rq => rq.includes(lowerQuery) || lowerQuery.includes(rq)))
                                                                                              track('WQ.referencedPartial', WQ.referencedPartial);

      // ── 2. TERM_WEIGHTS ────────────────────────────────────────────────────
      for (const term of queryTerms) {
        const t = term.toLowerCase();
        const clm = charLengthMultiplier(t);
        if (TW.id    && lowerId.includes(t))                    track('TW.id',         TW.id         * clm);
        if (TW.title && lowerTitle.includes(t))                 track('TW.title',      TW.title      * clm);
        if (TW.type  && lowerType.includes(t))                  track('TW.type',       TW.type       * clm);
        if (TW.tags  && tags.some(tag => tag.includes(t)))      track('TW.tags',       TW.tags       * clm);
        if (TW.referenced && refQs.some(rq => rq.includes(t))) track('TW.referenced', TW.referenced * clm);
        if (TW.keywords) {
          outer: for (const cat of ['title', 'structural', 'semantic', 'bm25', 'kg']) {
            for (const gram of kwGrams) {
              if ((norm[cat][gram] || {})[t] !== undefined) {
                track('TW.keywords', TW.keywords * clm);
                break outer;
              }
            }
          }
        }
      }

      // ── 3. KEYWORD_WEIGHTS — per-category × per-gram ──────────────────────
      for (const gramKey of ['1gram', '2gram', '3gram', '4gram']) {
        const qGrams = queryGrams[gramKey];
        if (!qGrams || !qGrams.length) continue;
        const qSet = new Set(qGrams);   // already lowercased by groupNGramsBySize

        // per keyword category — TF-weighted: score += doc-TF × weight × charLengthMultiplier
        for (const cat of ['title', 'structural', 'semantic', 'bm25', 'kg']) {
          const catWeight = Number((KW[cat] || {})[gramKey] || 0);
          if (!catWeight) continue;
          const catMap = norm[cat][gramKey] || {};
          let tfScore = 0;
          for (const qg of qSet) tfScore += (catMap[qg] || 0) * charLengthMultiplier(qg);
          track(`KW.${cat}.${gramKey}`, tfScore * catWeight);
        }

        // synonyms
        const synWeight = Number(SYN[gramKey] || 0);
        if (synWeight) {
          const legacySlot = Array.isArray(meta.synonyms) && gramKey === '1gram'
            ? meta.synonyms.map(s => String(s).toLowerCase()) : [];
          const synMap = norm.synonyms[gramKey] || {};
          let tfScore = 0;
          for (const qg of qSet) {
            const clm = charLengthMultiplier(qg);
            tfScore += (synMap[qg] || 0) * clm;
            if (legacySlot.includes(qg)) tfScore += clm;
          }
          track(`SYN.${gramKey}`, tfScore * synWeight);
        }
      }

      if (score > 0) scoredDocs.push({ ...meta, score, _breakdown: breakdown });
    }

    // ── 3b. Inject tag-pinned docs that did not score at all ─────────────────
    if (tagPinnedIds.size) {
      const scoredIds = new Set(scoredDocs.map(d => String(d.id)));
      for (const meta of allMetadata) {
        if (tagPinnedIds.has(String(meta.id)) && !scoredIds.has(String(meta.id))) {
          scoredDocs.push({ ...meta, score: 0, _breakdown: {} });
        }
      }
    }

    if (!scoredDocs.length) return [];

    console.log(`[RepoAsk] direct scorers for "${queryStr}": ${scoredDocs.length}`);
    for (const d of [...scoredDocs].sort((a, b) => b.score - a.score).slice(0, 10)) {
      const bk = d._breakdown || {};
      const parts = Object.entries(bk).sort((a, b) => b[1] - a[1])
        .map(([k, v]) => `${k}=${Number(v).toFixed(2)}`).join('  ');
      console.log(`    [${d.id}] "${d.title}"  score=${d.score.toFixed(2)}  { ${parts} }`);
    }

    // ── 4. Referenced-query neighbor boost ────────────────────────────────────
    let ranked = applyReferencedQueryNeighborBoost(scoredDocs);

    // ── 5. Sort → threshold cutoff → cap ──────────────────────────────────────
    ranked.sort((a, b) => b.score - a.score);
    const topScore = ranked[0]?.score || 0;
    if (topScore > 0) {
      const threshold = topScore * topScoreThresholdRatio;
      console.log(`[RepoAsk] ranked ${ranked.length} docs  topScore=${topScore.toFixed(2)}  threshold=${threshold.toFixed(2)}`);
      // Tag-pinned docs always pass through the threshold cutoff
      ranked = ranked.filter(d => d.score >= threshold || tagPinnedIds.has(String(d.id)));
    }

    const results = ranked.slice(0, Math.min(limit, maxResults));
    console.log(`[RepoAsk] query: "${queryStr}"`);
    for (let i = 0; i < results.length; i++) {
      const d = results[i];
      const bk = d._breakdown || {};
      const parts = Object.entries(bk)
        .sort((a, b) => b[1] - a[1])
        .map(([k, v]) => `${k}=${typeof v === 'number' ? v.toFixed(2) : v}`)
        .join('  ');
      console.log(`  #${i + 1} [${d.id}] "${d.title}"  score=${d.score.toFixed(2)}  { ${parts} }`);
    }
    return results.map(({ _breakdown, ...rest }) => rest);
  }

  return { rankLocalDocuments };
};