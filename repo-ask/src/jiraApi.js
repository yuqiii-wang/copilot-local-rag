const axios = require('axios');
const vscode = require('vscode');

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
        headers['Authorization'] = `Bearer ${securityToken}`;
    }
    return headers;
}

async function fetchJiraIssue(issueArg) {
    const { url: base, securityToken } = getJiraConfig();
    const encodedArg = encodeURIComponent(issueArg);
    const resolveUrl = `${base}/rest/api/2/issue/resolve?arg=${encodedArg}`;
    const headers = getHeaders(securityToken);

    try {
        const response = await axios.get(resolveUrl, { timeout: REQUEST_TIMEOUT_MS, headers });
        return response.data;
    } catch {
        const response = await axios.get(`${base}/rest/api/2/issue/${encodeURIComponent(issueArg)}`, { timeout: REQUEST_TIMEOUT_MS, headers });
        return response.data;
    }
}

async function fetchAllJiraIssues(project) {
    const { url: base, securityToken } = getJiraConfig();
    const headers = getHeaders(securityToken);
    const query = project ? `?project=${encodeURIComponent(project)}` : '';
    const response = await axios.get(`${base}/rest/api/2/search${query}`, { timeout: REQUEST_TIMEOUT_MS, headers });
    const issues = Array.isArray(response.data?.issues) ? response.data.issues : [];
    return issues;
}

module.exports = {
    fetchJiraIssue,
    fetchAllJiraIssues
};