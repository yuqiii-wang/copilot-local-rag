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
        if (!response || !response.text) {
            return null;
        }

        let responseText = '';
        for await (const fragment of response.text) {
            responseText += fragment;
        }

        const parsed = extractJsonObject(responseText);
        const arg = String(parsed?.arg || '').trim();
        return arg.length > 0 ? arg : null;
    } catch {
        return null;
    }
}

async function generateKnowledgeGraph(vscode, referenceQueries, secondaryUrls, contentMap, options = {}) {
    if (!vscode.lm || !vscode.LanguageModelChatMessage) {
        return {};
    }

    try {
        const shared = require('../chat/shared');
        const model = await shared.selectDefaultChatModel(vscode, options);
        if (!model) {
            return {};
        }

        // Prepare content from secondary URLs
        const secondaryContent = secondaryUrls.map(url => {
            const content = contentMap[url] || 'No content available';
            return `URL: ${url}\nContent: ${content.slice(0, 1000)}...`; // Truncate to avoid token limits
        }).join('\n\n');

        const instruction = [
            'You are a knowledge graph builder.',
            'Your task is to construct a structured knowledge graph from the provided reference queries and secondary content.',
            'Extract relevant entities and their relationships from the content.',
            'Organize the information into a graph structure with nodes and edges.',
            'Nodes should represent entities (e.g., systems, processes, concepts).',
            'Edges should represent relationships between entities (e.g., "uses", "depends on", "is part of").',
            'Return a JSON object with the following structure:',
            '{',
            '  "nodes": [',
            '    { "id": "unique-id", "label": "Entity Name", "type": "entity-type" }',
            '  ],',
            '  "edges": [',
            '    { "source": "source-node-id", "target": "target-node-id", "label": "relationship-type" }',
            '  ]',
            '}',
            '\nReference Queries:',
            referenceQueries.join('\n'),
            '\nSecondary Content:',
            secondaryContent
        ].join('\n');

        const response = await withTimeout(model.sendRequest([
            vscode.LanguageModelChatMessage.User(instruction)
        ]), LLM_TIMEOUT_MS, null);
        if (!response || !response.text) {
            return {};
        }

        let responseText = '';
        for await (const fragment of response.text) {
            responseText += fragment;
        }

        const parsed = extractJsonObject(responseText);
        return parsed || {};
    } catch (error) {
        console.error('Error generating knowledge graph:', error);
        return {};
    }
}

module.exports = {
    extractJsonObject,
    parseRefreshArg,
    generateKnowledgeGraph
};