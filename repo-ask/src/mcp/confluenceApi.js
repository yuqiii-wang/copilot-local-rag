const axios = require('axios');
const vscode = require('vscode');
const { confluenceApiMap } = require('./apiMap');

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

function buildResolveCandidates(pageArg) {
    const raw = String(pageArg || '').trim();
    if (!raw) {
        return [];
    }
    const id = extractConfluencePageIdFromArg(raw);
    if (id) {
        return [id, raw];
    }
    return [raw];
}

async function fetchConfluencePage(pageArg) {
    let { url: base, securityToken } = getConfluenceConfig();
    if (String(pageArg).startsWith('http')) {
        base = new URL(pageArg).origin;
    }
    
    const headers = getHeaders(securityToken);
    
    let lastError = null;
    for (const candidate of buildResolveCandidates(pageArg)) {
        try {
            const resolveUrl = confluenceApiMap.contentResolve(base, candidate);
            const response = await axios.get(resolveUrl, {
                timeout: REQUEST_TIMEOUT_MS,
                headers,
                maxContentLength: Infinity,
                maxBodyLength: Infinity
            });
            return response.data;
        } catch (error) {
            lastError = error;
        }
        
        try {
            const storageUrl = confluenceApiMap.contentStorage(base, candidate);
            const response = await axios.get(storageUrl, {
                timeout: REQUEST_TIMEOUT_MS,
                headers,
                maxContentLength: Infinity,
                maxBodyLength: Infinity
            });
            return response.data;
        } catch (error) {
            lastError = error;
        }
    }
    
    throw lastError || new Error('Failed to fetch Confluence page with provided argument.');
}

async function fetchAllConfluencePages() {
    const { url: base, securityToken } = getConfluenceConfig();
    const headers = getHeaders(securityToken);
    
    const response = await axios.get(confluenceApiMap.contentAll(base), {
        timeout: REQUEST_TIMEOUT_MS,
        headers,
        maxContentLength: Infinity,
        maxBodyLength: Infinity
    });
    return response.data;
}

async function fetchConfluencePageChildren(pageId) {
    let { url: base, securityToken } = getConfluenceConfig();
    if (String(pageId).startsWith('http')) {
        base = new URL(pageId).origin;
    }
    const headers = getHeaders(securityToken);
    
    const response = await axios.get(confluenceApiMap.contentChildren(base, pageId), {
        timeout: REQUEST_TIMEOUT_MS,
        headers,
        maxContentLength: Infinity,
        maxBodyLength: Infinity
    });
    return response.data;
}

module.exports = {
    fetchConfluencePage,
    fetchAllConfluencePages,
    fetchConfluencePageChildren
};
