const axios = require('axios');
const vscode = require('vscode');

const DEFAULT_CONFLUENCE_BASE_URL = 'http://127.0.0.1:8001';
const REQUEST_TIMEOUT_MS = 15000;

function getConfluenceConfig() {
    const configuration = vscode.workspace.getConfiguration('repoAsk');
    const profile = configuration.get('confluence');
    
    let url = DEFAULT_CONFLUENCE_BASE_URL;
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

async function fetchConfluencePage(pageArg) {
    const { url: base, securityToken } = getConfluenceConfig();
    const encodedArg = encodeURIComponent(pageArg);
    const resolveUrl = `${base}/rest/api/content/resolve?arg=${encodedArg}`;
    const headers = getHeaders(securityToken);

    try {
        const response = await axios.get(resolveUrl, { timeout: REQUEST_TIMEOUT_MS, headers });
        return response.data;
    } catch {
        const response = await axios.get(`${base}/rest/api/content/${encodeURIComponent(pageArg)}`, { timeout: REQUEST_TIMEOUT_MS, headers });
        return response.data;
    }
}

async function fetchAllConfluencePages() {
    const { url: base, securityToken } = getConfluenceConfig();
    const headers = getHeaders(securityToken);
    const response = await axios.get(`${base}/rest/api/content`, { timeout: REQUEST_TIMEOUT_MS, headers });
    return response.data;
}

module.exports = {
    fetchConfluencePage,
    fetchAllConfluencePages
};