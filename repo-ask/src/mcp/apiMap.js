// mcp/apiMap.js

/**
 * apiMap provides localized path mappings for the underlying dummy APIs.
 * Instead of hardcoding the URL paths in the api modules, they use these maps.
 */

const confluenceApiMap = {
    contentResolve: (base, candidate) => `${base}/confluence/rest/api/content/${encodeURIComponent(candidate)}`,
    contentStorage: (base, candidate) => `${base}/confluence/rest/api/content/${encodeURIComponent(candidate)}?expand=body.storage`,
    contentAll: (base) => `${base}/rest/api/content?expand=body.storage`,
    contentChildren: (base, pageId) => `${base}/confluence/rest/api/content/${encodeURIComponent(pageId)}/child/page?expand=body.storage`
};

const jiraApiMap = {
    issueResolve: (base, queryArg) => `${base}/rest/api/2/issue/resolve?arg=${encodeURIComponent(queryArg)}`,
    issue: (base, queryArg) => `${base}/rest/api/2/issue/${encodeURIComponent(queryArg)}`,
    search: (base, query) => `${base}/rest/api/2/search${query}`
};

module.exports = {
    confluenceApiMap,
    jiraApiMap
};