const axios = require('axios');
const vscode = require('vscode');
const { jiraApiMap } = require('./apiMap');

const DEFAULT_JIRA_BASE_URL = 'http://127.0.0.1:8002';
const REQUEST_TIMEOUT_MS = 15000;

function getJiraConfig() {
    const configuration = vscode.workspace.getConfiguration('repoAsk');
    const profile = configuration.get('jira');

    let url = DEFAULT_JIRA_BASE_URL;
    let securityToken = '';

    if (profile && typeof profile === 'object') {
        url = profile.url || url;
        securityToken = profile.securityToken || '';
    }

    return {
        url: String(url).replace(/\/$/, ''),
        securityToken
    };
}

function getHeaders(securityToken) {
    const headers = {};
    if (securityToken) {
        if (securityToken.startsWith('Bearer ') || securityToken.startsWith('Basic ')) {
            headers['Authorization'] = securityToken;
        } else if (securityToken.includes(':')) {
            headers['Authorization'] = `Basic ${Buffer.from(securityToken).toString('base64')}`;
        } else {
            headers['Authorization'] = `Bearer ${securityToken}`;
        }
    }
    return headers;
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
    const headers = getHeaders(securityToken);

    try {
        const response = await axios.get(resolveUrl, {
            timeout: REQUEST_TIMEOUT_MS,
            headers,
            maxContentLength: Infinity,
            maxBodyLength: Infinity
        });
        return response.data;
    } catch {
        const response = await axios.get(jiraApiMap.issue(base, queryArg), {
            timeout: REQUEST_TIMEOUT_MS,
            headers,
            maxContentLength: Infinity,
            maxBodyLength: Infinity
        });
        return response.data;
    }
}

async function fetchAllJiraIssues(project) {
    const { url: base, securityToken } = getJiraConfig();
    const headers = getHeaders(securityToken);
    const query = project ? `?project=${encodeURIComponent(project)}` : '';
    
    const response = await axios.get(jiraApiMap.search(base, query), {
        timeout: REQUEST_TIMEOUT_MS,
        headers,
        maxContentLength: Infinity,
        maxBodyLength: Infinity
    });
    
    const issues = Array.isArray(response.data?.issues) ? response.data.issues : [];
    return issues;
}

module.exports = {
    fetchJiraIssue,
    fetchAllJiraIssues
};