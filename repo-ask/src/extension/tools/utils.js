const vscode = require('vscode');

function toToolResult(text, data) {
    const parts = [new vscode.LanguageModelTextPart(String(text || ''))];
    if (vscode.LanguageModelDataPart && typeof vscode.LanguageModelDataPart.json === 'function' && data !== undefined) {
        parts.push(vscode.LanguageModelDataPart.json(data));
    }
    return new vscode.LanguageModelToolResult(parts);
}

function buildCheckAllDocsCommandLink(query) {
    const question = String(query || '').trim();
    if (!question) {
        return 'Run `repo-ask.checkAllDocs` to scan all docs.';
    }
    const encodedArgs = encodeURIComponent(JSON.stringify([question]));
    return `[Check ALL docs](command:repo-ask.checkAllDocs?${encodedArgs})`;
}

function formatRefreshStatus(sourceLabel, progress = {}) {
    const index = Number(progress?.index);
    const total = Number(progress?.total);
    const hasFraction = Number.isFinite(index) && Number.isFinite(total) && total > 0;
    const progressSuffix = hasFraction ? ` (${index}/${total})` : '';
    return `downloading from ${sourceLabel} ...${progressSuffix}`;
}

module.exports = {
    toToolResult,
    buildCheckAllDocsCommandLink,
    formatRefreshStatus
};
