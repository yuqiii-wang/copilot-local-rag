const vscode = require('vscode');
const { confluenceApiMap } = require('./apiMap');
const { httpManager, getAuthHeaders } = require('./httpManager');

const REQUEST_TIMEOUT_MS = 5000;

function getConfluenceConfig() {
    const configuration = vscode.workspace.getConfiguration('repoAsk');
    const profile = configuration.get('confluence');
    
    let url = '';
    let securityToken = '';
    
    if (profile && typeof profile === 'object') {
        url = profile.url || '';
        securityToken = profile.securityToken || '';
    }
    
    return {
        url: String(url).replace(/\/$/, ''),
        securityToken
    };
}

function extractConfluencePageIdFromArg(pageArg) {
    const raw = String(pageArg || '').trim();
    if (!raw) {
        return null;
    }
    const directMatch = raw.match(/(?:[?&]pageId=|\/pages\/|\/viewpage\/|\.action\/|\?pageId=)(\d+)/i);
    if (directMatch && directMatch[1]) {
        return directMatch[1];
    }
    return '';
}

async function fetchConfluencePage(pageArg) {
    let { url: base, securityToken } = getConfluenceConfig();
    
    if (String(pageArg).startsWith('http')) {
        base = new URL(pageArg).origin;
    } else if (!base) {
        throw new Error('Confluence base URL not configured. Please set the repoAsk.confluence.url setting.');
    }
    
    const headers = getAuthHeaders(securityToken);
    const pageId = extractConfluencePageIdFromArg(pageArg);
    
    if (pageId) {
        try {
            const storageUrl = confluenceApiMap.contentStorage(base, pageId);
            return await httpManager.request({
                method: 'GET',
                url: storageUrl,
                timeout: REQUEST_TIMEOUT_MS,
                headers
            });
        } catch (error) {
            // Continue to try other methods
        }
    }
    
    try {
        const storageUrl = confluenceApiMap.contentStorage(base, pageArg);
        return await httpManager.request({
            method: 'GET',
            url: storageUrl,
            timeout: REQUEST_TIMEOUT_MS,
            headers
        });
    } catch (error) {
        throw error || new Error('Failed to fetch Confluence page with provided argument.');
    }
}

async function fetchAllConfluencePages() {
    const { url: base, securityToken } = getConfluenceConfig();
    
    if (!base) {
        throw new Error('Confluence base URL not configured. Please set the repoAsk.confluence.url setting.');
    }
    
    const headers = getAuthHeaders(securityToken);
    
    return await httpManager.request({
        method: 'GET',
        url: confluenceApiMap.contentAll(base),
        timeout: REQUEST_TIMEOUT_MS,
        headers
    });
}

async function fetchConfluencePageChildren(pageId) {
    let { url: base, securityToken } = getConfluenceConfig();
    
    if (String(pageId).startsWith('http')) {
        base = new URL(pageId).origin;
    } else if (!base) {
        throw new Error('Confluence base URL not configured. Please set the repoAsk.confluence.url setting.');
    }
    
    const headers = getAuthHeaders(securityToken);
    
    return await httpManager.request({
        method: 'GET',
        url: confluenceApiMap.contentChildren(base, pageId),
        timeout: REQUEST_TIMEOUT_MS,
        headers
    });
}

function escapeHtml(value) {
    return String(value || '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function normalizeFeedbackPayload(feedbackPayload) {
    const payload = feedbackPayload && typeof feedbackPayload === 'object' ? feedbackPayload : {};
    return {
        datetime: String(payload.datetime || new Date().toISOString().slice(0, 16)).trim(),
        username: String(payload.username || 'Anonymous').trim(),
        elapsedTime: String(payload.elapsedTime || '').trim(),
        sourceQuery: String(payload.sourceQuery || '').trim(),
        conversationSummary: String(payload.conversationSummary || '').trim(),
        confluenceLink: String(payload.confluenceLink || '').trim(),
        confluencePageId: String(payload.confluencePageId || '').trim(),
        jiraId: String(payload.jiraId || '').trim(),
        tags: String(payload.tags || '').trim(),
        secondaryUrls: Array.isArray(payload.secondaryUrls) ? payload.secondaryUrls.map(String).filter(Boolean) : [],
        knowledge_graph: typeof payload.knowledge_graph === 'string' ? payload.knowledge_graph.trim() : ''
    };
}

function buildFeedbackRowHtml(feedbackPayload) {
    const normalized = normalizeFeedbackPayload(feedbackPayload);
    const details = [];

    if (normalized.sourceQuery) {
        details.push(`<li><strong>Source Query:</strong> ${escapeHtml(normalized.sourceQuery)}</li>`);
    }

    if (normalized.conversationSummary) {
        details.push(
            `<li><strong>Conversation Summary:</strong><pre style="white-space: pre-wrap; word-break: break-word; margin: 4px 0 0;">${escapeHtml(normalized.conversationSummary)}</pre></li>`
        );
    }

    if (normalized.confluenceLink) {
        const safeLink = escapeHtml(normalized.confluenceLink);
        const isHttpLink = /^https?:\/\//i.test(normalized.confluenceLink);
        const linkValue = isHttpLink ? `<a href="${safeLink}">${safeLink}</a>` : safeLink;
        details.push(`<li><strong>Confluence/Jira Link:</strong> ${linkValue}</li>`);
    }

    if (normalized.confluencePageId) {
        details.push(`<li><strong>Confluence Page ID:</strong> ${escapeHtml(normalized.confluencePageId)}</li>`);
    }

    if (normalized.jiraId) {
        details.push(`<li><strong>Jira ID:</strong> ${escapeHtml(normalized.jiraId)}</li>`);
    }

    if (normalized.tags) {
        details.push(`<li><strong>Tags:</strong> ${escapeHtml(normalized.tags)}</li>`);
    }

    if (normalized.secondaryUrls && normalized.secondaryUrls.length > 0) {
        const secondaryUrlsHtml = normalized.secondaryUrls.map(url => {
            const safeUrl = escapeHtml(url);
            const isHttpLink = /^https?:\/\//i.test(url);
            return isHttpLink ? `<a href="${safeUrl}">${safeUrl}</a>` : safeUrl;
        }).join('<br/>');
        details.push(`<li><strong>Secondary URLs/IDs:</strong><br/>${secondaryUrlsHtml}</li>`);
    }

    if (normalized.knowledge_graph) {
        details.push(
            `<li><strong>Knowledge Graph (Mermaid):</strong><pre style="white-space: pre-wrap; word-break: break-word; margin: 4px 0 0; font-family: monospace; font-size: 0.85em;">${escapeHtml(normalized.knowledge_graph)}</pre></li>`
        );
    }

    if (details.length === 0) {
        details.push('<li>No additional feedback details provided.</li>');
    }

    return [
        '<tr>',
        `<td>${escapeHtml(normalized.datetime)}</td>`,
        `<td>${escapeHtml(normalized.username)}</td>`,
        `<td>${escapeHtml(normalized.elapsedTime)}</td>`,
        `<td><ul>${details.join('')}</ul></td>`,
        '</tr>'
    ].join('');
}

function appendFeedbackToStorageValue(currentContent, feedbackPayload) {
    const rowHtml = buildFeedbackRowHtml(feedbackPayload);
    let content = String(currentContent || '');

    // Upgrade existing table headers if they don't have the new columns
    if (content.includes('<th>Date</th><th>User</th><th>Feedback</th>')) {
        content = content.replace('<th>Date</th><th>User</th><th>Feedback</th>', '<th>Date</th><th>Username</th><th>Elapsed Time (s)</th><th>Feedback</th>');
    }

    if (/<tbody[^>]*>/i.test(content)) {
        return content.replace(/<\/tbody>/i, `${rowHtml}</tbody>`);
    }

    if (/<table[^>]*>/i.test(content)) {
        return content.replace(/<\/table>/i, `<tbody>${rowHtml}</tbody></table>`);
    }

    const feedbackTable = [
        '<table>',
        '<tbody><tr><th>Date</th><th>Username</th><th>Elapsed Time (s)</th><th>Feedback</th></tr>',
        `${rowHtml}</tbody>`,
        '</table>'
    ].join('');

    if (/\<\/div\>\s*$/i.test(content)) {
        return content.replace(/\<\/div\>\s*$/i, `${feedbackTable}</div>`);
    }

    return `${content}${content ? '\n' : ''}${feedbackTable}`;
}

async function updateConfluencePage(pageId, feedbackPayload) {
    let { url: base, securityToken } = getConfluenceConfig();
    let pageIdForApi = pageId;
    
    if (String(pageId).startsWith('http')) {
        const url = new URL(pageId);
        base = url.origin;
        pageIdForApi = extractConfluencePageIdFromArg(pageId) || pageId;
    } else if (!base) {
        throw new Error('Confluence base URL not configured. Please set the repoAsk.confluence.url setting.');
    }
    
    const headers = getAuthHeaders(securityToken);
    headers['Content-Type'] = 'application/json';
    
    const currentPage = await fetchConfluencePage(pageId);
    let currentContent = currentPage.body?.storage?.value || '';
    const updatedContent = appendFeedbackToStorageValue(currentContent, feedbackPayload);
    
    const payload = {
        id: extractConfluencePageIdFromArg(pageId) || pageId,
        type: 'page',
        title: currentPage.title,
        body: {
            storage: {
                value: updatedContent,
                representation: 'storage'
            }
        },
        version: {
            number: (currentPage.version?.number || 1) + 1
        }
    };
    
    return await httpManager.request({
        method: 'PUT',
        url: confluenceApiMap.contentUpdate(base, pageIdForApi),
        timeout: REQUEST_TIMEOUT_MS,
        headers,
        data: payload
    });
}

async function createConfluencePage(title, content) {
    let { url: base, securityToken } = getConfluenceConfig();
    
    if (!base) {
        throw new Error('Confluence base URL not configured. Please set the repoAsk.confluence.url setting.');
    }
    
    const headers = getAuthHeaders(securityToken);
    headers['Content-Type'] = 'application/json';
    
    const payload = {
        type: 'page',
        title: title,
        body: {
            storage: {
                value: content,
                representation: 'storage'
            }
        },
        space: {
            key: 'PROJ'
        }
    };
    
    return await httpManager.request({
        method: 'POST',
        url: confluenceApiMap.contentCreate(base),
        timeout: REQUEST_TIMEOUT_MS,
        headers,
        data: payload
    });
}

module.exports = {
    fetchConfluencePage,
    fetchAllConfluencePages,
    fetchConfluencePageChildren,
    updateConfluencePage,
    createConfluencePage
};
