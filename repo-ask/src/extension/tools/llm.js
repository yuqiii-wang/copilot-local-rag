function extractJsonObject(rawText) {
    if (!rawText) {
        return null;
    }

    const text = String(rawText).trim();
    try {
        return JSON.parse(text);
    } catch {
        const match = text.match(/\{[\s\S]*\}/);
        if (!match) {
            return null;
        }

        try {
            return JSON.parse(match[0]);
        } catch {
            return null;
        }
    }
}

function getJiraExtractionRegexes(vscode) {
    const configuration = vscode.workspace.getConfiguration('repoAsk');
    const jiraProfile = configuration.get('jira');
    const configuredList = Array.isArray(jiraProfile?.regex) && jiraProfile.regex.length > 0
        ? jiraProfile.regex
        : ['PROJECT-\\d+'];

    const compiled = [];
    for (const pattern of configuredList) {
        if (typeof pattern !== 'string' || pattern.trim().length === 0) {
            continue;
        }

        try {
            compiled.push(new RegExp(pattern, 'i'));
        } catch {
        }
    }

    return compiled;
}

const LLM_TIMEOUT_MS = 12000;

async function withTimeout(promise, timeoutMs, timeoutValue = null) {
    let timeoutId;
    const timeoutPromise = new Promise((resolve) => {
        timeoutId = setTimeout(() => resolve(timeoutValue), timeoutMs);
    });

    try {
        return await Promise.race([promise, timeoutPromise]);
    } finally {
        clearTimeout(timeoutId);
    }
}

/**
 * Collect all text from a LanguageModelChatResponse, handling both
 * the `stream` (AsyncIterable<LanguageModelTextPart>) and the
 * convenience `text` (AsyncIterable<string>) properties.
 */
async function collectResponseText(vscode, response) {
    if (!response) return '';
    let text = '';
    if (response.stream) {
        for await (const chunk of response.stream) {
            if (vscode.LanguageModelTextPart && chunk instanceof vscode.LanguageModelTextPart) {
                text += chunk.value;
            } else if (typeof chunk === 'string') {
                text += chunk;
            } else if (chunk && typeof chunk.value === 'string') {
                text += chunk.value;
            }
        }
    } else if (response.text) {
        for await (const fragment of response.text) {
            text += fragment;
        }
    }
    return text;
}

async function parseRefreshArg(vscode, sourceInput, options = {}) {
    const raw = String(sourceInput || '').trim();
    if (!raw) {
        return { found: false, arg: '', source: 'empty' };
    }

    const jiraRegexes = getJiraExtractionRegexes(vscode);

    const urlMatch = raw.match(/https?:\/\/[^\s)]+/i);
    if (urlMatch && urlMatch[0]) {
        const urlStr = urlMatch[0];
        if (urlStr.match(/\/browse\/[A-Za-z0-9\-]+/i)) {
            return { found: true, arg: urlStr, source: 'regex-jira' };
        }
        for (const regex of jiraRegexes) {
            if (urlStr.match(regex)) {
                return { found: true, arg: urlStr, source: 'regex-jira' };
            }
        }
        return { found: true, arg: urlStr, source: 'regex-url' }; // Default to confluence for now
    }

    // Pure number > 6 digits is a Confluence ID
    const pureNumMatch = raw.match(/^\d{7,}$/);
    if (pureNumMatch) {
        return { found: true, arg: raw, source: 'regex-id' };
    }

    for (const regex of jiraRegexes) {
        const jiraMatch = raw.match(regex);
        if (jiraMatch && jiraMatch[0]) {
            return { found: true, arg: jiraMatch[0], source: 'regex-jira' };
        }
    }

    const pageIdMatch = raw.match(/(?:pageid=)(\d+)/i) || raw.match(/\b(\d{1,8})\b/i);
    if (pageIdMatch && pageIdMatch[1]) {
        return { found: true, arg: pageIdMatch[1], source: 'regex-id' };
    }

    const candidateByLlm = await extractConfluenceIdentifierWithLlm(vscode, raw, options);
    if (candidateByLlm) {
        return { found: true, arg: candidateByLlm, source: 'llm' };
    }

    return { found: false, arg: '', source: 'none' };
}

async function extractConfluenceIdentifierWithLlm(vscode, rawInput, options = {}) {
    if (!vscode.lm || !vscode.LanguageModelChatMessage) {
        return null;
    }

    const workspacePromptContext = String(options.workspacePromptContext || '').trim();
    const boundedPromptContext = workspacePromptContext.slice(0, 12000);

    try {
        const shared = require('../chat/shared');
        const model = await shared.selectDefaultChatModel(vscode, options);
        if (!model) {
            return null;
        }

        const instruction = [
            'You are a parser for Confluence sync arguments.',
            'From the SOURCE text, extract only one of the following if present: (1) full Confluence HTTP(S) link, (2) numeric Confluence page id, or (3) exact page title phrase.',
            'If none is present, return an empty string.',
            'Return valid JSON only with shape: {"arg":"..."}.',
            boundedPromptContext
                ? `Workspace prompt context:\n${boundedPromptContext}`
                : 'Workspace prompt context: (none)',
            `SOURCE: ${rawInput}`
        ].join('\n');

        const response = await withTimeout(model.sendRequest([
            vscode.LanguageModelChatMessage.User(instruction)
        ]), LLM_TIMEOUT_MS, null);
        if (!response) {
            return null;
        }

        const responseText = await collectResponseText(vscode, response);

        const parsed = extractJsonObject(responseText);
        const arg = String(parsed?.arg || '').trim();
        return arg.length > 0 ? arg : null;
    } catch {
        return null;
    }
}

async function generateKnowledgeGraph(vscode, referenceQueries, secondaryUrls, contentMap, options = {}) {
    if (!vscode.lm || !vscode.LanguageModelChatMessage) {
        return '';
    }

    try {
        const shared = require('../chat/shared');
        const model = await shared.selectDefaultChatModel(vscode, options);
        if (!model) {
            return '';
        }

        const primaryContent = String(options.primaryContent || '').trim();
        const existingMermaid = String(options.existingKnowledgeGraph || '').trim();

        // Prepare content from secondary URLs
        const secondaryContent = (Array.isArray(secondaryUrls) ? secondaryUrls : []).map(url => {
            const content = (contentMap && contentMap[url]) || 'No content available';
            return `URL: ${url}\nContent: ${content.slice(0, 2000)}`;
        }).join('\n\n');

        const queryList = Array.isArray(referenceQueries) && referenceQueries.length > 0
            ? referenceQueries.join('\n')
            : '(none)';

        const instructionParts = [
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
            instructionParts.push(
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

        instructionParts.push(
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

        const instruction = instructionParts.join('\n');

        const response = await withTimeout(model.sendRequest([
            vscode.LanguageModelChatMessage.User(instruction)
        ]), LLM_TIMEOUT_MS * 3, null);
        if (!response) {
            return existingMermaid || '';
        }

        const responseText = await collectResponseText(vscode, response);

        // Clean up the response: strip markdown code fences if present
        let mermaid = responseText.trim();
        mermaid = mermaid.replace(/^```mermaid\s*/i, '').replace(/^```\s*/i, '').replace(/\s*```$/i, '').trim();

        // Validate it looks like a mermaid diagram
        if (!mermaid.match(/^graph\s+(TD|LR|BT|RL)/i)) {
            // If it doesn't start with graph directive, try to extract one
            const graphMatch = mermaid.match(/graph\s+(TD|LR|BT|RL)[\s\S]*/i);
            if (graphMatch) {
                mermaid = graphMatch[0].trim();
            } else {
                return existingMermaid || '';
            }
        }

        // Strip unnecessary double-quotes from node labels and edge labels
        // e.g. NodeId["Label"] -> NodeId[Label], -->|"label"| -> -->|label|
        mermaid = mermaid.replace(/(\[|\{|\()"([^"]*)"(\]|\}|\))/g, '$1$2$3');
        mermaid = mermaid.replace(/\|"([^"]*)"\ *\|/g, '|$1|');

        return mermaid;
    } catch (error) {
        console.error('Error generating knowledge graph:', error);
        return String(options.existingKnowledgeGraph || '');
    }
}

/**
 * Extract entity names and relationship labels from a mermaid diagram string.
 * Returns an array of keyword strings suitable for merging into document keywords.
 */
function extractMermaidKeywords(mermaidText) {
    const text = String(mermaidText || '').trim();
    if (!text) {
        return [];
    }

    const keywords = new Set();
    const lines = text.split('\n');

    for (const line of lines) {
        const trimmed = line.trim();
        // Skip the graph directive line
        if (/^graph\s+(TD|LR|BT|RL)/i.test(trimmed) || !trimmed) {
            continue;
        }

        // Extract node labels: NodeId["Label"] or NodeId("Label") or NodeId["Label"]
        const nodeLabelMatches = trimmed.matchAll(/\w+\s*[\[\({"]\s*"?([^"\]\)}"]+)"?\s*[\]\)}"]/g);
        for (const m of nodeLabelMatches) {
            if (m[1]) {
                const label = m[1].trim();
                if (label.length >= 2) {
                    keywords.add(label.toLowerCase());
                    // Also add individual words from multi-word labels
                    for (const word of label.split(/\s+/)) {
                        if (word.length >= 2) {
                            keywords.add(word.toLowerCase());
                        }
                    }
                }
            }
        }

        // Extract node IDs (PascalCase/camelCase identifiers)
        const nodeIdMatches = trimmed.matchAll(/\b([A-Z][a-zA-Z0-9]+)\b/g);
        for (const m of nodeIdMatches) {
            if (m[1] && m[1].length >= 2) {
                // Split camelCase into words
                const parts = m[1].replace(/([a-z0-9])([A-Z])/g, '$1 $2').toLowerCase().split(/\s+/);
                for (const part of parts) {
                    if (part.length >= 2) {
                        keywords.add(part);
                    }
                }
                // Also add the full identifier as-is (lowercased)
                keywords.add(m[1].toLowerCase());
            }
        }

        // Extract relationship labels: -->|label| or -- label -->
        const relLabelMatches = trimmed.matchAll(/--+>?\|([^|]+)\|/g);
        for (const m of relLabelMatches) {
            if (m[1]) {
                const label = m[1].trim();
                if (label.length >= 2) {
                    keywords.add(label.toLowerCase());
                    for (const word of label.split(/\s+/)) {
                        if (word.length >= 2) {
                            keywords.add(word.toLowerCase());
                        }
                    }
                }
            }
        }

        // Extract relationship labels: -- "label" -->
        const quotedRelMatches = trimmed.matchAll(/--\s*"([^"]+)"\s*-->/g);
        for (const m of quotedRelMatches) {
            if (m[1]) {
                const label = m[1].trim();
                if (label.length >= 2) {
                    keywords.add(label.toLowerCase());
                    for (const word of label.split(/\s+/)) {
                        if (word.length >= 2) {
                            keywords.add(word.toLowerCase());
                        }
                    }
                }
            }
        }
    }

    // Filter out common mermaid directives/noise
    const noise = new Set(['graph', 'td', 'lr', 'bt', 'rl', 'subgraph', 'end', 'style', 'class', 'click']);
    return [...keywords].filter(kw => !noise.has(kw));
}

module.exports = {
    extractJsonObject,
    parseRefreshArg,
    generateKnowledgeGraph,
    extractMermaidKeywords
};