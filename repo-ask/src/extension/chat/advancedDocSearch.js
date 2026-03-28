/**
 * Advanced Doc Search — agentic iterative document search.
 *
 * Memory strategy:
 *   - `metadata.summary` is used directly as the per-doc memory cache.
 *     No LLM summarization is performed during search.
 *   - If a doc's title closely matches the query but it has no summary/keywords/KG,
 *     `documentService.generateStoredMetadataById` is called to populate them
 *     (same action as clicking "AI Gen" in the sidebar) before the evaluation round.
 *
 * Flow per round:
 *   1. Detect title-matched docs missing metadata → auto-generate their metadata.
 *   2. Filter active docs (excluding locally irrelevant ones).
 *   3. Ask LLM to evaluate summaries+KG → { relevantIds, irrelevantIds, searchTerms, satisfied, topDocId, topDocUrl, answer }.
 *   4. Mark irrelevantIds; if satisfied → stream answer + buttons and stop.
 *   5. Read full content for relevantIds (raw, no summarization).
 *   6. Repeat until maxRounds or satisfied.
 *
 * Final answer matches regular @repoask output:
 *   - Markdown answer with TOP_DOC_URL marker
 *   - Advanced Doc Search / Log Action / Check Code Logic buttons
 */

const { buildAdvancedSearchEvalPrompt, buildAdvancedSearchSynthesisPrompt, buildQueryOptimizePrompt } = require('./prompts');
const { selectDefaultChatModel, withTimeout, LLM_RESPONSE_TIMEOUT_MS, emitThinking } = require('./shared');

/**
 * Decide if a doc title is likely relevant to the query using simple word-overlap.
 * @param {string} title
 * @param {string} query
 * @returns {boolean}
 */
function titleLikelyRelevant(title, query) {
    const stopWords = new Set(['the', 'a', 'an', 'is', 'are', 'was', 'how', 'what', 'why', 'when', 'where', 'does', 'do', 'in', 'on', 'of', 'to', 'for', 'and', 'or', 'not']);
    const queryWords = query.toLowerCase().split(/\W+/).filter(w => w.length > 2 && !stopWords.has(w));
    const titleLower = title.toLowerCase();
    return queryWords.some(w => titleLower.includes(w));
}

/**
 * Parse the LLM's JSON evaluation response.
 * Tolerates markdown code fences.
 * @param {string} text
 * @returns {{ relevantIds: string[], irrelevantIds: string[], searchTerms: string[], satisfied: boolean, topDocId: string, topDocUrl: string, answer: string } | null}
 */
function parseEvalJson(text) {
    try {
        const stripped = text.replace(/```json\s*/gi, '').replace(/```/g, '').trim();
        const match = stripped.match(/\{[\s\S]+\}/);
        if (!match) return null;
        const parsed = JSON.parse(match[0]);
        return {
            relevantIds: Array.isArray(parsed.relevantIds) ? parsed.relevantIds.map(String) : [],
            irrelevantIds: Array.isArray(parsed.irrelevantIds) ? parsed.irrelevantIds.map(String) : [],
            searchTerms: Array.isArray(parsed.searchTerms) ? parsed.searchTerms.map(String) : [],
            satisfied: Boolean(parsed.satisfied),
            topDocId: typeof parsed.topDocId === 'string' ? parsed.topDocId.trim() : '',
            topDocUrl: typeof parsed.topDocUrl === 'string' ? parsed.topDocUrl.trim() : '',
            answer: typeof parsed.answer === 'string' ? parsed.answer.trim() : ''
        };
    } catch (_) {
        return null;
    }
}

/**
 * Parse the LLM's JSON query-optimization response.
 * @param {string} text
 * @returns {{ expandedQuery: string, keywords: string[], clarificationNeeded: boolean, clarificationMessage: string } | null}
 */
function parseOptimizeJson(text) {
    try {
        const stripped = text.replace(/```json\s*/gi, '').replace(/```/g, '').trim();
        const match = stripped.match(/\{[\s\S]+\}/);
        if (!match) return null;
        const parsed = JSON.parse(match[0]);
        return {
            expandedQuery: typeof parsed.expandedQuery === 'string' ? parsed.expandedQuery.trim() : '',
            keywords: Array.isArray(parsed.keywords) ? parsed.keywords.map(String) : [],
            clarificationNeeded: Boolean(parsed.clarificationNeeded),
            clarificationMessage: typeof parsed.clarificationMessage === 'string' ? parsed.clarificationMessage.trim() : ''
        };
    } catch (_) {
        return null;
    }
}

/**
 * Pre-search query optimizer.
 *
 * 1. Reads the doc matching `logActionConfluenceUrl` as a domain-context anchor.
 *    If found but missing summary, auto-generates it (same as clicking AI Gen).
 * 2. Builds an overview of the top 50 docs (id / title / summary / KG).
 * 3. Asks the LLM to expand ambiguous terms in the query based on the domain.
 *    If the query is too ambiguous, flags `clarificationNeeded`.
 *
 * @param {Object} vscodeApi
 * @param {string} query
 * @param {Array}  allMeta       - mutable snapshot (updated in-place for auto-gen)
 * @param {Object} deps          - { documentService, readDocumentContent }
 * @param {Object} response
 * @param {Object} options
 * @returns {Promise<{ expandedQuery: string, keywords: string[], clarificationNeeded: boolean, clarificationMessage: string }>}
 */
async function optimizeQuery(vscodeApi, query, allMeta, deps, response, options) {
    const { documentService, readDocumentContent } = deps;
    const config = vscodeApi.workspace.getConfiguration('repoAsk');
    const anchorUrl = String(config.get('logActionConfluenceUrl') || '').trim();

    // ── Resolve domain anchor document ───────────────────────────────────────
    let anchorContext = '';
    if (anchorUrl) {
        const anchorDoc = allMeta.find(m => {
            const docUrl = String(m.url || '').replace(/\/$/, '');
            if (!docUrl) return false;
            const normalAnchor = anchorUrl.replace(/\/$/, '');
            return normalAnchor.includes(docUrl) || docUrl.includes(normalAnchor) ||
                normalAnchor.endsWith(`/${String(m.id)}`);
        });

        if (anchorDoc) {
            let anchorMeta = anchorDoc;
            // Auto-gen metadata if summary is missing
            if (!anchorMeta.summary && documentService?.generateStoredMetadataById) {
                try {
                    emitThinking(response, `Generating domain context from anchor doc: "${anchorMeta.title}"`);
                    anchorMeta = await documentService.generateStoredMetadataById(anchorMeta.id);
                    const idx = allMeta.findIndex(x => String(x.id) === String(anchorMeta.id));
                    if (idx !== -1) allMeta[idx] = anchorMeta;
                    else allMeta.push(anchorMeta);
                } catch (e) {
                    console.warn(`[AdvancedDocSearch] Anchor doc auto-gen failed: ${e.message}`);
                }
            }

            if (anchorMeta.summary) {
                anchorContext = `Title: ${anchorMeta.title}\nSummary: ${anchorMeta.summary}`;
                if (anchorMeta.knowledgeGraph) {
                    anchorContext += `\nKG: ${anchorMeta.knowledgeGraph.slice(0, 500)}`;
                }
            } else if (typeof readDocumentContent === 'function') {
                const content = readDocumentContent(anchorDoc.id);
                if (content) {
                    anchorContext = `Title: ${anchorDoc.title}\nContent (partial): ${content.slice(0, 800)}`;
                }
            }
        }
    }

    // ── Build overview from top 50 docs (repoaskCheck: metadata.summary_kg, limit 50) ─
    const top50 = allMeta.slice(0, 50);
    const docsOverview = top50.map(m => {
        const parts = [`[${m.id}] ${m.title}`];
        if (m.summary) parts.push(`  Summary: ${m.summary}`);
        if (m.knowledgeGraph) parts.push(`  KG: ${m.knowledgeGraph.slice(0, 200)}`);
        return parts.join('\n');
    }).join('\n---\n');

    // ── Ask LLM to optimize query ─────────────────────────────────────────────
    const prompt = buildQueryOptimizePrompt({ query, anchorContext, docsOverview });
    try {
        const model = await selectDefaultChatModel(vscodeApi, options);
        if (!model) return { expandedQuery: query, keywords: [], clarificationNeeded: false, clarificationMessage: '' };

        const resp = await withTimeout(
            model.sendRequest(
                [vscodeApi.LanguageModelChatMessage.User(prompt)],
                {},
                options?.request?.token
            ),
            LLM_RESPONSE_TIMEOUT_MS,
            null
        );

        let text = '';
        if (resp?.stream) {
            for await (const chunk of resp.stream) {
                if (chunk instanceof vscodeApi.LanguageModelTextPart) text += chunk.value;
            }
        }
        const result = parseOptimizeJson(text);
        if (result) return result;
    } catch (e) {
        console.warn('[AdvancedDocSearch] Query optimization failed:', e.message);
    }

    return { expandedQuery: query, keywords: [], clarificationNeeded: false, clarificationMessage: '' };
}

/**
 * Emit the standard response footer: TOP_DOC_URL marker, then buttons.
 * @param {Object} response  - VS Code chat response stream
 * @param {string} query
 * @param {string} topDocUrl
 * @param {string} topDocId
 * @param {string} finalAnswer
 * @param {number} queryStartTime
 */
function emitButtons(response, query, topDocUrl, topDocId, finalAnswer, queryStartTime) {
    const topMarker = `\n\n[TOP_DOC_URL: ${topDocUrl || '[NO_URL]'}, TOP_DOC_ID: ${topDocId || '[NO_ID]'}]`;
    response.markdown(topMarker);

    response.button({
        command: 'repo-ask.advancedDocSearch',
        title: 'Advanced Doc Search',
        arguments: [query]
    });
    response.button({
        command: 'repo-ask.showLogActionButton',
        title: 'Log Action',
        arguments: [query, topDocUrl || '[NO_URL]', finalAnswer, queryStartTime]
    });
    response.button({
        command: 'repo-ask.checkCodeLogic',
        title: 'Check Code Logic',
        arguments: [query, finalAnswer]
    });
}

/**
 * Run the Advanced Doc Search agentic loop.
 *
 * @param {Object} vscodeApi         - The `vscode` module
 * @param {string} query             - The user's original question
 * @param {Object} response          - VS Code chat response stream
 * @param {Object} deps              - { readAllMetadata, readDocumentContent, documentService }
 * @param {Object} [options]         - Chat request options (for model selection and token)
 */
async function runAdvancedDocSearch(vscodeApi, query, response, deps, options) {
    const { readAllMetadata, readDocumentContent, documentService } = deps;
    const queryStartTime = Date.now();

    const config = vscodeApi.workspace.getConfiguration('repoAsk');
    const maxRounds = Math.max(1, Math.min(Number(config.get('maxAdvancedDocSearch')) || 3, 10));

    // Snapshot metadata at start; reload as needed after auto-gen
    let allMeta = typeof readAllMetadata === 'function' ? readAllMetadata() : [];

    // Per-search-session irrelevant doc set
    const localIrrelevantDocs = new Set();

    // docId → full content string (raw, no summarization)
    const readContent = {};

    // ── Query optimization pre-step ───────────────────────────────────────────
    emitThinking(response, 'Analyzing query and optimizing search terms...');
    const optimized = await optimizeQuery(vscodeApi, query, allMeta, deps, response, options);

    if (optimized.clarificationNeeded && optimized.clarificationMessage) {
        response.markdown(optimized.clarificationMessage);
        emitButtons(response, query, '', '', optimized.clarificationMessage, queryStartTime);
        return;
    }

    const effectiveQuery = (optimized.expandedQuery && optimized.expandedQuery !== query)
        ? optimized.expandedQuery
        : query;

    if (effectiveQuery !== query) {
        emitThinking(response, `Expanded query: "${effectiveQuery}"`);
        console.log(`[AdvancedDocSearch] Query optimized: "${query}" → "${effectiveQuery}"`);
    }

    let lastSearchTerms = optimized.keywords || [];

    for (let round = 1; round <= maxRounds; round++) {
        // ── Auto-generate metadata for title-matched docs missing summary ─────
        if (documentService && typeof documentService.generateStoredMetadataById === 'function') {
            const needsGen = allMeta.filter(m =>
                !localIrrelevantDocs.has(String(m.id)) &&
                !m.summary &&
                titleLikelyRelevant(m.title || '', effectiveQuery)
            );
            if (needsGen.length > 0) {
                const titles = needsGen.map(m => `"${m.title}"`).join(', ');
                emitThinking(response, `Generating missing metadata for: ${titles}`);
                console.log(`[AdvancedDocSearch] Round ${round}: auto-gen needed for ${needsGen.length} doc(s): ${titles}`);
            }
            for (const m of needsGen) {
                try {
                    const updated = await documentService.generateStoredMetadataById(m.id);
                    // Splice updated metadata back into our snapshot
                    const idx = allMeta.findIndex(x => String(x.id) === String(m.id));
                    if (idx !== -1) allMeta[idx] = updated;
                    else allMeta.push(updated);
                } catch (e) {
                    console.warn(`[AdvancedDocSearch] Auto-gen failed for "${m.title}" (${m.id}): ${e.message}`);
                }
            }
        }

        // ── Filter active candidates ──────────────────────────────────────────
        const activeMeta = allMeta.filter(m => !localIrrelevantDocs.has(String(m.id)));

        if (activeMeta.length === 0) {
            emitThinking(response, 'All documents have been evaluated — no remaining candidates.');
            console.log('[AdvancedDocSearch] No remaining candidates. Stopping.');
            break;
        }

        const searchTermsHint = lastSearchTerms.length > 0 ? ` | terms: ${lastSearchTerms.join(', ')}` : '';
        emitThinking(response, `Round ${round}/${maxRounds} — evaluating ${activeMeta.length} candidate doc(s)${searchTermsHint}`);
        console.log(`[AdvancedDocSearch] Round ${round}/${maxRounds}: ${activeMeta.length} candidates, irrelevant so far: ${localIrrelevantDocs.size}`);

        // ── Build summary list from metadata.summary (no LLM summarization) ──
        const summaryList = activeMeta.map(m => {
            const kg = m.knowledgeGraph ? m.knowledgeGraph.slice(0, 300) : '';
            return `[${m.id}] ${m.title}\nSummary: ${m.summary || '(no summary)'}${kg ? `\nKG: ${kg}` : ''}`;
        }).join('\n---\n');

        // Already-read content used in eval context
        const alreadyReadContent = Object.entries(readContent)
            .map(([id, c]) => {
                const meta = allMeta.find(m => String(m.id) === id);
                return `[${id}] ${meta?.title || id}: ${c.slice(0, 500)}`;
            })
            .join('\n---\n') || 'None yet.';

        const evalPromptText = buildAdvancedSearchEvalPrompt({
            query: effectiveQuery,
            round,
            maxRounds,
            summaryList,
            alreadyReadContent,
            searchTerms: lastSearchTerms
        });

        // ── Ask LLM to evaluate ───────────────────────────────────────────────
        let evalResult = null;
        try {
            const model = await selectDefaultChatModel(vscodeApi, options);
            if (!model) break;

            const evalResp = await withTimeout(
                model.sendRequest(
                    [vscodeApi.LanguageModelChatMessage.User(evalPromptText)],
                    {},
                    options?.request?.token
                ),
                LLM_RESPONSE_TIMEOUT_MS,
                null
            );

            let evalText = '';
            if (evalResp?.stream) {
                for await (const chunk of evalResp.stream) {
                    if (chunk instanceof vscodeApi.LanguageModelTextPart) evalText += chunk.value;
                }
            }
            evalResult = parseEvalJson(evalText);
        } catch (e) {
            console.error('[AdvancedDocSearch] Eval error:', e.message);
            break;
        }

        if (!evalResult) {
            console.warn('[AdvancedDocSearch] Could not parse eval JSON. Stopping.');
            break;
        }

        // ── Update irrelevant set and search terms ────────────────────────────
        for (const id of evalResult.irrelevantIds) {
            localIrrelevantDocs.add(id);
        }
        if (evalResult.irrelevantIds.length > 0) {
            const irrelevantTitles = evalResult.irrelevantIds
                .map(id => allMeta.find(m => String(m.id) === id))
                .filter(Boolean)
                .map(m => `"${m.title}"`)
                .join(', ');
            emitThinking(response, `Marked as irrelevant: ${irrelevantTitles || evalResult.irrelevantIds.join(', ')}`);
            console.log(`[AdvancedDocSearch] Round ${round}: irrelevant docs → ${irrelevantTitles}`);
        }
        if (evalResult.searchTerms.length > 0) {
            console.log(`[AdvancedDocSearch] Round ${round}: next search terms → ${evalResult.searchTerms.join(', ')}`);
        }
        lastSearchTerms = evalResult.searchTerms;

        // ── Satisfied: stream answer + buttons and stop ───────────────────────
        if (evalResult.satisfied && evalResult.answer) {
            const topTitle = allMeta.find(m => String(m.id) === evalResult.topDocId)?.title || evalResult.topDocId;
            emitThinking(response, `Sufficient information found. Top document: "${topTitle}"`);
            console.log(`[AdvancedDocSearch] Satisfied at round ${round}. Top doc: "${topTitle}" (${evalResult.topDocUrl})`);
            response.markdown(evalResult.answer);
            emitButtons(response, query, evalResult.topDocUrl, evalResult.topDocId, evalResult.answer, queryStartTime);
            return;
        }

        // ── Read full content (raw) for relevant IDs ──────────────────────────
        const toRead = evalResult.relevantIds.filter(id => !readContent[id]);
        if (toRead.length > 0) {
            const readTitles = toRead
                .map(id => allMeta.find(m => String(m.id) === id))
                .filter(Boolean)
                .map(m => `"${m.title}"`)
                .join(', ');
            emitThinking(response, `Reading full content for: ${readTitles || toRead.join(', ')}`);
            console.log(`[AdvancedDocSearch] Round ${round}: reading content for → ${readTitles}`);
            for (const id of toRead) {
                const content = typeof readDocumentContent === 'function' ? readDocumentContent(id) : '';
                if (content) readContent[id] = content;
            }
        } else if (round >= 2) {
            // Nothing new to read and not satisfied — stop to avoid idle loop
            break;
        }
    }

    // ── Final synthesis (fallback when loop ends without satisfied=true) ──────
    const contentEntries = Object.entries(readContent);
    if (contentEntries.length === 0) {
        response.markdown('No relevant documents found during advanced search.');
        emitButtons(response, query, '', '', 'No relevant documents found.', queryStartTime);
        return;
    }

    const readTitleList = contentEntries
        .map(([id]) => allMeta.find(m => String(m.id) === id)?.title || id)
        .map(t => `"${t}"`)
        .join(', ');
    emitThinking(response, `Synthesizing final answer from ${contentEntries.length} doc(s): ${readTitleList}`);
    console.log(`[AdvancedDocSearch] Synthesis from: ${readTitleList}`);

    const contentSummary = contentEntries.map(([id, c]) => {
        const meta = allMeta.find(m => String(m.id) === id);
        const url = meta?.url || '';
        return `**[${id}] ${meta?.title || id}**${url ? ` (${url})` : ''}:\n${c.slice(0, 3000)}`;
    }).join('\n\n---\n\n');

    let finalAnswer = '';
    let topDocUrl = '';
    let topDocId = '';

    try {
        const model = await selectDefaultChatModel(vscodeApi, options);
        if (model) {
            const synthPrompt = buildAdvancedSearchSynthesisPrompt({ query: effectiveQuery, contentSummary });
            const synthResp = await withTimeout(
                model.sendRequest(
                    [vscodeApi.LanguageModelChatMessage.User(synthPrompt)],
                    {},
                    options?.request?.token
                ),
                LLM_RESPONSE_TIMEOUT_MS,
                null
            );

            if (synthResp?.stream) {
                for await (const chunk of synthResp.stream) {
                    if (chunk instanceof vscodeApi.LanguageModelTextPart) {
                        finalAnswer += chunk.value;
                        response.markdown(chunk.value);
                    }
                }
            }

            // Extract TOP_DOC_URL from synthesis output
            const topMatch = finalAnswer.match(/\[TOP_DOC_URL:\s*(.+?),\s*TOP_DOC_ID:\s*(.+?)\]/);
            if (topMatch) {
                topDocUrl = topMatch[1].trim();
                topDocId = topMatch[2].trim();
            } else {
                // Fallback: use the first doc read
                const firstId = contentEntries[0]?.[0];
                const firstMeta = allMeta.find(m => String(m.id) === firstId);
                topDocId = firstId || '';
                topDocUrl = firstMeta?.url || '';
            }
        }
    } catch (e) {
        console.error('[AdvancedDocSearch] Synthesis error:', e.message);
        finalAnswer = contentEntries
            .map(([id]) => {
                const meta = allMeta.find(m => String(m.id) === id);
                return `- [${id}] ${meta?.title || id}`;
            })
            .join('\n');
        response.markdown(`**Found docs:**\n${finalAnswer}`);
        topDocId = contentEntries[0]?.[0] || '';
        topDocUrl = allMeta.find(m => String(m.id) === topDocId)?.url || '';
    }

    emitButtons(response, query, topDocUrl, topDocId, finalAnswer, queryStartTime);
}

module.exports = { runAdvancedDocSearch };

