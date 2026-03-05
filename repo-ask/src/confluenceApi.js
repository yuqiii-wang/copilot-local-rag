const axios = require('axios');
const vscode = require('vscode');

const DEFAULT_CONFLUENCE_BASE_URL = 'http://127.0.0.1:8001';
const REQUEST_TIMEOUT_MS = 15000;

function getConfluenceBaseUrl() {
    const configuration = vscode.workspace.getConfiguration('repoAsk');
    const profile = configuration.get('confluence');
    const objectUrl = profile && typeof profile === 'object' ? profile.url : undefined;
    const legacyUrl = configuration.get('confluenceBaseUrl');
    return String(objectUrl || legacyUrl || DEFAULT_CONFLUENCE_BASE_URL).replace(/\/$/, '');
}

async function fetchConfluencePage(pageArg) {
    const base = getConfluenceBaseUrl();
    const encodedArg = encodeURIComponent(pageArg);
    const resolveUrl = `${base}/rest/api/content/resolve?arg=${encodedArg}`;

    try {
        const response = await axios.get(resolveUrl, { timeout: REQUEST_TIMEOUT_MS });
        return response.data;
    } catch {
        const response = await axios.get(`${base}/rest/api/content/${encodeURIComponent(pageArg)}`, { timeout: REQUEST_TIMEOUT_MS });
        return response.data;
    }
}

async function fetchAllConfluencePages() {
    const base = getConfluenceBaseUrl();
    const response = await axios.get(`${base}/rest/api/content`, { timeout: REQUEST_TIMEOUT_MS });
    return response.data;
}

module.exports = {
    fetchConfluencePage,
    fetchAllConfluencePages
};