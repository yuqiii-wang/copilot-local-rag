const axios = require('axios');
const vscode = require('vscode');

const DEFAULT_JIRA_BASE_URL = 'http://127.0.0.1:8002';
const REQUEST_TIMEOUT_MS = 15000;

function getJiraBaseUrl() {
    const configuration = vscode.workspace.getConfiguration('repoAsk');
    const profile = configuration.get('jira');
    const objectUrl = profile && typeof profile === 'object' ? profile.url : undefined;
    const legacyUrl = configuration.get('jiraBaseUrl');
    return String(objectUrl || legacyUrl || DEFAULT_JIRA_BASE_URL).replace(/\/$/, '');
}

async function fetchJiraIssue(issueArg) {
    const base = getJiraBaseUrl();
    const encodedArg = encodeURIComponent(issueArg);
    const resolveUrl = `${base}/rest/api/2/issue/resolve?arg=${encodedArg}`;

    try {
        const response = await axios.get(resolveUrl, { timeout: REQUEST_TIMEOUT_MS });
        return response.data;
    } catch {
        const response = await axios.get(`${base}/rest/api/2/issue/${encodeURIComponent(issueArg)}`, { timeout: REQUEST_TIMEOUT_MS });
        return response.data;
    }
}

async function fetchAllJiraIssues(project) {
    const base = getJiraBaseUrl();
    const query = project ? `?project=${encodeURIComponent(project)}` : '';
    const response = await axios.get(`${base}/rest/api/2/search${query}`, { timeout: REQUEST_TIMEOUT_MS });
    const issues = Array.isArray(response.data?.issues) ? response.data.issues : [];
    return issues;
}

module.exports = {
    fetchJiraIssue,
    fetchAllJiraIssues
};