/**
 * Shared prompt templates and tool descriptions for the RepoAsk agent.
 *
 * Single source of truth for all LLM instructions so that agentTools.js,
 * generalAnswer.js, and docCheckTool.js stay consistent without repeating strings.
 */

const { ChatPromptTemplate, MessagesPlaceholder } = require('@langchain/core/prompts');

// ─────────────────────────────────────────────────────────────────────────────
// Shared constants — imported by both docCheckTool.js and agentTools.js
// ─────────────────────────────────────────────────────────────────────────────

/** Allowed mode values for the repoask_doc_check tool. */
const ALLOWED_MODES = ['content', 'metadata', 'content_partial', 'metadata.summary', 'metadata.summary_kg', 'metadata.id'];

// ─────────────────────────────────────────────────────────────────────────────
// Tool description — kept here so agentTools.js (LangChain schema) and
// docCheckTool.js (VS Code LM registration) both import the same text.
// ─────────────────────────────────────────────────────────────────────────────
const DOC_CHECK_TOOL_DESCRIPTION = [
    'Read documents from the local RepoAsk store (synced from Confluence and Jira).',
    'Use mode "metadata.id" first to list available doc IDs and titles.',
    'Use mode "content_partial" for initial discovery of relevant docs.',
    'Escalate to mode "content" to read the full text of a specific doc.',
    'Prefer docs tagged with "confluence" or "jira" for team knowledge.',
    'For production support questions, also check docs tagged "production-support" or "incident-response".'
].join(' ');

// ─────────────────────────────────────────────────────────────────────────────
// Phase 1 — Investigation agent system prompt
// Uses MessagesPlaceholder for injected conversation history.
// ─────────────────────────────────────────────────────────────────────────────
const AGENT_SYSTEM_PROMPT = [
    'You are RepoAsk Doc Agent. Your goal is to help the user answer questions from the local document store.',
    'Wait for tool results before drawing conclusions.',
    '',
    'RULES:',
    '- You MUST use the repoask_doc_check tool to retrieve information.',
    '- First call with mode="metadata.id" to discover available documents.',
    '- Then call with mode="content_partial" for the most relevant IDs.',
    '- Escalate to mode="content" if partial content is clearly relevant.',
    '- You MUST NOT hallucinate information not present in retrieved documents.',
    '- Identify relevant documents; include their IDs and URLs in your analysis.',
    '',
    '## Attached Files and Code Context:\n{workspacePromptContext}',
    '',
    '## Initial Ranked Documents (use these IDs to start reading):\n{initialRankedContext}'
].join('\n');

/**
 * Build the Phase 1 ChatPromptTemplate.
 * Includes a MessagesPlaceholder for injected conversation history before the
 * current human turn, giving the agent full session context.
 *
 * Template variables: workspacePromptContext, initialRankedContext, history, prompt
 *
 * @returns {ChatPromptTemplate}
 */
function buildPhase1Template() {
    return ChatPromptTemplate.fromMessages([
        ['system', AGENT_SYSTEM_PROMPT],
        new MessagesPlaceholder('history'),
        ['human', '{prompt}']
    ]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 2 — Synthesis prompt (plain string, streamed directly to VS Code LM)
// ─────────────────────────────────────────────────────────────────────────────
const SYNTHESIS_SYSTEM = [
    'You are an expert technical editor.',
    'Process the raw observations from a document search and produce a clear, concise final answer.',
    '- Be concise. Use bullet points or short paragraphs.',
    '- Include only sources actually used in your answer; drop irrelevant references.',
    '- You MUST include a clickable Markdown link for the most relevant document.',
    '- At the very bottom output exactly: `[TOP_DOC_URL: <url>, TOP_DOC_ID: <id>]`',
    '- If no documents were relevant, state that clearly and output `[TOP_DOC_URL: [NO_URL], TOP_DOC_ID: [NO_ID]]`.'
].join('\n');

/**
 * Build the Phase 2 synthesis prompt string.
 * @param {string} rawObservations - Aggregated first-round agent output
 * @param {string} userPrompt      - Original user question
 * @returns {string}
 */
function buildPhase2Prompt(rawObservations, userPrompt) {
    return [
        SYNTHESIS_SYSTEM,
        '',
        '--- RAW INTERNAL OBSERVATIONS ---',
        rawObservations || '(no observations)',
        '--- END RAW INTERNAL OBSERVATIONS ---',
        '',
        `User question: ${userPrompt}`
    ].join('\n\n');
}

// ─────────────────────────────────────────────────────────────────────────────
// Partial content note — appended when doc content is truncated for scanning
// ─────────────────────────────────────────────────────────────────────────────
const PARTIAL_CONTENT_NOTE =
    "[Note]: This is partial content. If the partial content is likely related to user query, MUST read full content. To read full content, instruct LLM to use mode 'content' to read full content.";

// ─────────────────────────────────────────────────────────────────────────────
// Confluence ID extractor (llm.js — extractConfluenceIdentifierWithLlm)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @param {{ promptContext: string, rawInput: string }} vars
 * @returns {string}
 */
function buildConfluenceIdExtractorPrompt({ promptContext, rawInput }) {
    return [
        'You are a parser for Confluence sync arguments.',
        'From the SOURCE text, extract only one of the following if present: (1) full Confluence HTTP(S) link, (2) numeric Confluence page id, or (3) exact page title phrase.',
        'If none is present, return an empty string.',
        'Return valid JSON only with shape: {"arg":"..."}.',
        promptContext
            ? `Workspace prompt context:\n${promptContext}`
            : 'Workspace prompt context: (none)',
        `SOURCE: ${rawInput}`
    ].join('\n');
}

// ─────────────────────────────────────────────────────────────────────────────
// Knowledge graph builder (llm.js — generateKnowledgeGraph)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @param {{ queryList: string, primaryContent: string, secondaryContent: string, existingMermaid?: string }} vars
 * @returns {string}
 */
function buildKnowledgeGraphPrompt({ queryList, primaryContent, secondaryContent, existingMermaid }) {
    const parts = [
        'You are a knowledge graph builder that outputs Mermaid diagram syntax.',
        'Your task is to construct a knowledge graph as a Mermaid flowchart from the provided reference queries and document content.',
        '',
        '## Process',
        '1. Read the REFERENCE QUERIES below. Extract key entities from them — these are your starting anchor entities.',
        '2. Explore relationships between these anchor entities and other entities found in the PRIMARY CONTENT and SECONDARY CONTENT.',
        '3. Focus on non-lexical, semantic relationships (e.g., "depends on", "triggers", "is part of", "manages", "produces", "consumes") rather than simple keyword co-occurrence.',
        '4. Each node should represent a meaningful entity (system, process, concept, component, role).',
        '5. Each edge should represent a named relationship.',
        ''
    ];

    if (existingMermaid) {
        parts.push(
            '## Existing Knowledge Graph',
            'An existing Mermaid diagram is provided below. You MUST preserve all existing entities and relationships.',
            'You are encouraged to ADD new entities and new relationships discovered from the current queries and content.',
            'Only remove an existing entity if you are absolutely certain it is incorrect or contradicted by the new content.',
            '',
            '```mermaid',
            existingMermaid,
            '```',
            ''
        );
    }

    parts.push(
        '## Output Format',
        'Return ONLY a valid Mermaid flowchart diagram (no wrapping markdown code fences, no explanation).',
        'Start with "graph TD" or "graph LR".',
        'Use descriptive node IDs (e.g., FXEngine, TradeProcessor) and quoted labels where needed.',
        'Use arrow labels for relationships: A -->|relationship| B',
        '',
        '## Reference Queries (anchor entities)',
        queryList,
        '',
        '## Primary Content',
        primaryContent ? primaryContent.slice(0, 3000) : '(none)',
        '',
        '## Secondary Content',
        secondaryContent || '(none)'
    );

    return parts.join('\n');
}

// ─────────────────────────────────────────────────────────────────────────────
// Document annotation (annotation.js — generateAnnotationWithLlm)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @param {{ title: string, parentTopic: string, author: string, keywords: string, tags: string, feedback: string, content: string }} vars
 * @returns {string}
 */
function buildAnnotationPrompt({ title, parentTopic, author, keywords, tags, feedback, content }) {
    return [
        'You are annotating a local Confluence document metadata record.',
        'Return valid JSON only with shape: {"summary":"...","keywords":"keyword-a, keyword-b"}.',
        'Summary must be clear, concise, and contain only the most essential information from the source content, in 1-3 sentences maximum. Avoid filler phrases and repetition.',
        'Keywords must be specific technical terms in one comma-separated string.',
        'Take existing document keywords (provided below) and filter out non-business-related words or stop words (e.g., "confluence", "jira", "and", "the", "that", etc.).',
        'Include a few close synonyms or related alternate terms that help retrieval.',
        `Title: ${title || ''}`,
        `Topic: ${parentTopic || ''}`,
        `Author: ${author || ''}`,
        `Existing Keywords: ${keywords || ''}`,
        `Existing Tags: ${tags || ''}`,
        `Existing Feedback: ${feedback || ''}`,
        'Document markdown content:',
        content || ''
    ].join('\n');
}

// ─────────────────────────────────────────────────────────────────────────────
// Conversation summary rewrite (sidebarController.js)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @param {{ inputText: string }} vars
 * @returns {string}
 */
function buildSummaryRewritePrompt({ inputText }) {
    return [
        'You are a helpful assistant that rewrites conversation summaries.',
        'Rewrite the following conversation summary into a clear, concise, and complete summary.',
        'Keep all key details, decisions, and action items. Remove filler, redundancy, and unnecessary preamble.',
        'Aim for the shortest version that preserves all essential information. Return only the rewritten summary text.',
        '',
        'Conversation Summary:',
        inputText
    ].join('\n');
}

// ─────────────────────────────────────────────────────────────────────────────
// Check code logic chat query (checkCodeLogicCommand.js)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @param {{ projectContext: string, workflowSummary: string, userQuestion: string }} vars
 * @returns {string}
 */
function buildCheckCodeLogicQuery({ projectContext, workflowSummary, userQuestion }) {
    return [
        `**Project Directory:** \`${projectContext}\`\n`,
        'The following is a summarized workflow derived from documentation:',
        '',
        `<details><summary>Workflow Summary (Click to expand)</summary>\n\n${workflowSummary}\n\n</details>`,
        '',
        '## Task',
        `Original question that produced the above summary: "${userQuestion}"`,
        '',
        'Using the workspace code:',
        '1. **Fact-check** — Verify whether the workflow described above accurately reflects what the code actually does. Point out any discrepancies.',
        '2. **Code Logic** — If the description is accurate (or mostly accurate), walk through the actual code logic: identify the relevant files, classes/functions, and execution flow that implement this workflow.',
    ].join('\n');
}

// ─────────────────────────────────────────────────────────────────────────────
// Advanced Doc Search — evaluation and synthesis prompts
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Build the per-round evaluation prompt for the Advanced Doc Search agentic loop.
 * @param {{ query: string, round: number, maxRounds: number, summaryList: string, alreadyReadContent: string, searchTerms: string[] }} vars
 * @returns {string}
 */
function buildAdvancedSearchEvalPrompt({ query, round, maxRounds, summaryList, alreadyReadContent, searchTerms }) {
    return [
        'You are an agentic document relevance evaluator.',
        'IMPORTANT: Be extremely concise. No over-thinking. Return ONLY valid JSON — no markdown fences, no prose.',
        '',
        `User Query: ${query}`,
        searchTerms && searchTerms.length > 0
            ? `Previous search terms: ${searchTerms.join(', ')}`
            : '',
        `Round: ${round} of ${maxRounds}`,
        '',
        '## Already-read content from relevant docs:',
        alreadyReadContent || 'None yet.',
        '',
        '## Candidate docs (id / title / summary / KG):',
        summaryList,
        '',
        'Evaluate each candidate against the user query and already-read content. Then respond with JSON:',
        '{',
        '  "relevantIds": [],   // doc IDs to read full content next (max 4, only IDs not yet read)',
        '  "irrelevantIds": [], // doc IDs clearly unrelated — skipped for remaining rounds',
        '  "searchTerms": [],   // 2-4 refined terms for the next round',
        '  "satisfied": false,  // true when already-read content fully answers the user query',
        '  "topDocId": "",      // if satisfied: the single most relevant doc ID',
        '  "topDocUrl": "",     // if satisfied: the URL of that doc (from metadata)',
        '  "answer": ""         // if satisfied: answer as numbered/bullet steps (NOT a paragraph). cite doc IDs. markdown only.',
        '}'
    ].filter(l => l !== '').join('\n');
}

/**
 * Build the final synthesis prompt for Advanced Doc Search when no answer was produced during the loop.
 * @param {{ query: string, contentSummary: string }} vars
 * @returns {string}
 */
function buildAdvancedSearchSynthesisPrompt({ query, contentSummary }) {
    return [
        'Answer the user query based solely on the document content provided.',
        'Format rules:',
        '- Present the answer as a numbered list or bullet steps — NOT a wall of text or long paragraphs.',
        '- Each step/point should be one concise sentence. Add sub-bullets only when genuinely needed.',
        '- Cite the doc ID(s) used inline, e.g. [42].',
        '- Include a Markdown link to the most relevant doc.',
        'At the bottom output exactly: `[TOP_DOC_URL: <url>, TOP_DOC_ID: <id>]`',
        'If content is insufficient, state that as a single bullet and output `[TOP_DOC_URL: [NO_URL], TOP_DOC_ID: [NO_ID]]`.',
        '',
        `User Query: ${query}`,
        '',
        '## Document Content:',
        contentSummary || '(no content found)'
    ].join('\n');
}

// ─────────────────────────────────────────────────────────────────────────────
// Query optimizer (advancedDocSearch.js — optimizeQuery)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Build the query optimization prompt used before the Advanced Doc Search loop.
 * The LLM infers domain context from an anchor document and the docs overview,
 * then expands ambiguous abbreviations in the user query or asks for clarification.
 * @param {{ query: string, anchorContext: string, docsOverview: string }} vars
 * @returns {string}
 */
function buildQueryOptimizePrompt({ query, anchorContext, docsOverview }) {
    const parts = [
        'You are a domain-aware search query optimizer for a local document store.',
        'Your task: expand abbreviated or ambiguous terms in the user query based on the domain context.',
        '',
        'RULES:',
        '- Infer the domain ONLY from the anchor document and docs overview — do NOT assume.',
        '- Expand clear domain abbreviations (e.g., "fx" in a finance store → "foreign exchange").',
        '- If the query is too ambiguous to expand confidently, set clarificationNeeded=true and write a polite clarificationMessage asking for more context with a concrete example hint.',
        '- Propose 3–6 additional search keywords that represent the expanded meaning.',
        '- Return ONLY valid JSON — no markdown fences, no prose.',
        '',
    ];

    if (anchorContext) {
        parts.push(
            '## Domain Context (anchor document):',
            anchorContext,
            '',
        );
    }

    parts.push(
        `## Available Documents Overview (up to 50):`,
        docsOverview || '(no documents available)',
        '',
        `## User Query: ${query}`,
        '',
        'Return JSON:',
        '{',
        '  "expandedQuery": "<rewritten query with full terms, or original if no expansion needed>",',
        '  "keywords": ["keyword1", "keyword2"],',
        '  "clarificationNeeded": false,',
        '  "clarificationMessage": ""',
        '}',
    );

    return parts.join('\n');
}

// ─────────────────────────────────────────────────────────────────────────────
// Template registry — single map to retrieve and populate any template.
// All entries are (vars) => string so callers can use renderTemplate(key, vars).
// ─────────────────────────────────────────────────────────────────────────────
const TEMPLATES = {
    /** Phase 1 agent system prompt string (variable substitution done via ChatPromptTemplate). */
    AGENT_SYSTEM:            () => AGENT_SYSTEM_PROMPT,
    /** Phase 2 synthesis prompt. vars: { rawObservations, userPrompt } */
    SYNTHESIS:               ({ rawObservations, userPrompt }) => buildPhase2Prompt(rawObservations, userPrompt),
    /** Confluence ID extractor. vars: { promptContext, rawInput } */
    CONFLUENCE_ID_EXTRACTOR: (vars) => buildConfluenceIdExtractorPrompt(vars),
    /** Knowledge graph builder. vars: { queryList, primaryContent, secondaryContent, existingMermaid? } */
    KNOWLEDGE_GRAPH:         (vars) => buildKnowledgeGraphPrompt(vars),
    /** Document annotation. vars: { title, parentTopic, author, keywords, tags, feedback, content } */
    ANNOTATION:              (vars) => buildAnnotationPrompt(vars),
    /** Conversation summary rewrite. vars: { inputText } */
    SUMMARY_REWRITE:         (vars) => buildSummaryRewritePrompt(vars),
    /** Check code logic chat query. vars: { projectContext, workflowSummary, userQuestion } */
    CHECK_CODE_LOGIC:        (vars) => buildCheckCodeLogicQuery(vars),
    /** Partial content note appended to truncated doc reads. No vars. */
    PARTIAL_CONTENT_NOTE:    () => PARTIAL_CONTENT_NOTE,
    /** Advanced doc search evaluation. vars: { query, round, maxRounds, summaryList, alreadyReadContent, searchTerms } */
    ADVANCED_SEARCH_EVAL:    (vars) => buildAdvancedSearchEvalPrompt(vars),
    /** Advanced doc search synthesis. vars: { query, contentSummary } */
    ADVANCED_SEARCH_SYNTH:   (vars) => buildAdvancedSearchSynthesisPrompt(vars),
    /** Advanced doc search query optimizer. vars: { query, anchorContext, docsOverview } */
    QUERY_OPTIMIZE:          (vars) => buildQueryOptimizePrompt(vars)
};

/**
 * Retrieve and render a named template.
 * @param {keyof typeof TEMPLATES} key
 * @param {Object} [vars={}]
 * @returns {string}
 */
function renderTemplate(key, vars = {}) {
    const builder = TEMPLATES[key];
    if (!builder) throw new Error(`Unknown template key: "${key}"`);
    return builder(vars);
}

module.exports = {
    ALLOWED_MODES,
    DOC_CHECK_TOOL_DESCRIPTION,
    AGENT_SYSTEM_PROMPT,
    PARTIAL_CONTENT_NOTE,
    TEMPLATES,
    renderTemplate,
    buildPhase1Template,
    buildPhase2Prompt,
    buildConfluenceIdExtractorPrompt,
    buildKnowledgeGraphPrompt,
    buildAnnotationPrompt,
    buildSummaryRewritePrompt,
    buildCheckCodeLogicQuery,
    buildAdvancedSearchEvalPrompt,
    buildAdvancedSearchSynthesisPrompt,
    buildQueryOptimizePrompt
};
