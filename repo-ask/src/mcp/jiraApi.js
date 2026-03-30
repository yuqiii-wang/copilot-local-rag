const vscode = require('vscode');
const { jiraApiMap } = require('./apiMap');
const { httpManager, getAuthHeaders } = require('./httpManager');

const REQUEST_TIMEOUT_MS = 15000;

function getJiraConfig() {
    const configuration = vscode.workspace.getConfiguration('repoAsk');
    const profile = configuration.get('jira');

    let securityToken = '';

    if (profile && typeof profile === 'object') {
        url = profile.url;
        securityToken = profile.securityToken || '';
    }

    return {
        url: String(url).replace(/\/$/, ''),
        securityToken
    };
}

async function fetchJiraIssue(issueArg) {
    let { url: base, securityToken } = getJiraConfig();
    let queryArg = issueArg;

    if (String(issueArg).startsWith('http')) {
        const parsed = new URL(issueArg);
        base = parsed.origin;
        const match = parsed.pathname.match(/\/browse\/([A-Za-z0-9\-]+)/i);
        if (match) {
            queryArg = match[1];
        }
    }

    const resolveUrl = jiraApiMap.issueResolve(base, queryArg);
    const headers = getAuthHeaders(securityToken);

    try {
        const responseData = await httpManager.request({
            method: 'GET',
            url: resolveUrl,
            timeout: REQUEST_TIMEOUT_MS,
            headers
        });
        return responseData;
    } catch {
        const responseData = await httpManager.request({
            method: 'GET',
            url: jiraApiMap.issue(base, queryArg),
            timeout: REQUEST_TIMEOUT_MS,
            headers
        });
        return responseData;
    }
}

async function fetchAllJiraIssues(project) {
    const { url: base, securityToken } = getJiraConfig();
    const headers = getAuthHeaders(securityToken);
    const query = project ? `?project=${encodeURIComponent(project)}` : '';
    
    const responseData = await httpManager.request({
        method: 'GET',
        url: jiraApiMap.search(base, query),
        timeout: REQUEST_TIMEOUT_MS,
        headers
    });
    
    const issues = Array.isArray(responseData?.issues) ? responseData.issues : [];
    return issues;
}

function getJiraExtractionRegexes(vsApi) {
    const configuration = (vsApi || vscode).workspace.getConfiguration('repoAsk');
    const jiraProfile = configuration.get('jira');
    const patternList = Array.isArray(jiraProfile?.regex) ? jiraProfile.regex : [];
    const compiled = [];
    for (const pattern of patternList) {
        if (typeof pattern !== 'string' || pattern.trim().length === 0) continue;
        try { compiled.push(new RegExp(pattern, 'i')); } catch { }
    }
    return compiled;
}

module.exports = {
    fetchJiraIssue,
    fetchAllJiraIssues,
    getJiraExtractionRegexes
};
