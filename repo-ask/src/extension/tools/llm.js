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

async function selectToolAndArg(vscode, prompt, options = {}) {
    if (!prompt || String(prompt).trim().length === 0) {
        return { tool: 'none' };
    }

    const workspacePromptContext = String(options.workspacePromptContext || '').trim();
    const boundedPromptContext = workspacePromptContext.slice(0, 12000);

    if (vscode.lm && vscode.LanguageModelChatMessage) {
        try {
            const models = await withTimeout(vscode.lm.selectChatModels({}), LLM_TIMEOUT_MS, []);
            const model = models?.[0];
            if (model) {
                const instruction = [
                    'You are a helper that chooses which local repoask tool to run based on a user query.',
                    'Return only valid JSON with shape: {"tool":"refresh|annotate|rank|check|none","arg":"..."}.',
                    'Choose `refresh` when the user asks to refresh, sync, download, pull, fetch, import, or update docs/issues from Confluence or Jira.',
                    'For `refresh`, extract a Jira issue key/id/link or a Confluence page id/title/link into `arg` when present.',
                    boundedPromptContext
                        ? `Workspace prompt context:\n${boundedPromptContext}`
                        : 'Workspace prompt context: (none)',
                    'Analyze the user text and decide which tool is most appropriate. If uncertain, choose `check` and put the user query into `arg`.',
                    `User query: ${prompt}`
                ].join('\n');

                const response = await withTimeout(model.sendRequest([
                    vscode.LanguageModelChatMessage.User(instruction)
                ]), LLM_TIMEOUT_MS, null);
                if (!response || !response.text) {
                    return { tool: 'check', arg: prompt };
                }

                let responseText = '';
                for await (const fragment of response.text) {
                    responseText += fragment;
                }

                const parsed = extractJsonObject(responseText);
                if (parsed && parsed.tool) {
                    return { tool: parsed.tool, arg: parsed.arg || '' };
                }
            }
        } catch {
            // fallthrough to heuristics
        }
    }

    const lowered = String(prompt).toLowerCase();
    const urlMatch = prompt.match(/https?:\/\/[\w\-./?=&%]+/i);
    const pageIdMatch = prompt.match(/pageid=(\d+)|\b(\d{1,6})\b/i);
    const jiraRegexes = getJiraExtractionRegexes(vscode);
    const jiraMatch = jiraRegexes
        .map(regex => prompt.match(regex))
        .find(match => match && match[0]);

    if (
        lowered.includes('refresh') ||
        lowered.includes('sync') ||
        lowered.includes('download') ||
        lowered.includes('fetch') ||
        lowered.includes('pull') ||
        lowered.includes('import') ||
        lowered.includes('update') ||
        lowered.includes('confluence') ||
        lowered.includes('jira') ||
        urlMatch ||
        pageIdMatch ||
        jiraMatch
    ) {
        return {
            tool: 'refresh',
            arg: jiraMatch
                ? jiraMatch[0]
                : (urlMatch ? urlMatch[0] : (pageIdMatch ? (pageIdMatch[1] || pageIdMatch[2]) : prompt))
        };
    }

    if (lowered.includes('annotate')) {
        return { tool: 'annotate', arg: '' };
    }

    if (lowered.includes('rank') || lowered.includes('search')) {
        return { tool: 'rank', arg: prompt };
    }

    return { tool: 'check', arg: prompt };
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
        const models = await withTimeout(vscode.lm.selectChatModels({}), LLM_TIMEOUT_MS, []);
        const model = models?.[0];
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

module.exports = {
    extractJsonObject,
    selectToolAndArg,
    parseRefreshArg
};