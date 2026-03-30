/**
 * vsCodeTools.js — VS Code extension tool utilities for RepoAsk.
 *
 * Contains:
 *   - Utility helpers (toToolResult, buildCheckAllDocsCommandLink, formatRefreshStatus)
 *   - VS Code LM tool registration (repoask_doc_check)
 *   - createLanguageModelTools factory
 *
 * Core document-check logic lives in docCheckCore.js and is shared with agentTools.js.
 */

const vscode = require('vscode');

const _pkg = require('../../../package.json');
const ALLOWED_MODES = _pkg?.contributes?.languageModelTools?.[0]?.inputSchema?.properties?.mode?.enum
    ?? ['id_2_content', 'id_2_metadata', 'id_2_content_partial', 'id_2_metadata_4_summary', 'id_2_metadata_4_summary_kg'];

// ─────────────────────────────────────────────────────────────────────────────
// Utility helpers (previously utils.js)
// ─────────────────────────────────────────────────────────────────────────────

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

// ─────────────────────────────────────────────────────────────────────────────
// VS Code LM tool registration (previously docCheckTool.js + lmTools.js)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Register the repoask_doc_check VS Code LM tool.
 * Delegates core logic to runDocCheck from docCheckCore.js.
 *
 * @param {object} deps
 * @returns {vscode.Disposable}
 */
function registerDocCheckTool(deps) {
    const { runDocCheck } = require('./docCheckCore');
    const { toolNames, documentService, readAllMetadata, readDocumentMetadata, readDocumentContent, emptyStoreHint } = deps;

    return vscode.lm.registerTool(toolNames.docCheck, {
        async invoke(options, _token) {
            try {
                const text = await runDocCheck(options.input || {}, {
                    documentService,
                    readAllMetadata,
                    readDocumentMetadata,
                    readDocumentContent,
                    emptyStoreHint
                });

                // For searchTerms mode, also attach structured data with ranked doc IDs
                const input = options.input || {};
                if (Array.isArray(input.searchTerms) && input.searchTerms.length > 0) {
                    const maxLimit = Math.max(1, Number(input.limit) || 3);
                    const ranked = documentService.rankLocalDocuments(input.searchTerms.join(' '), maxLimit * 3);
                    const topDocs = (ranked || []).slice(0, maxLimit);
                    return toToolResult(text, { docIds: topDocs.map(d => String(d.id)), ranked: topDocs });
                }

                return toToolResult(text);
            } catch (err) {
                console.error('[docCheckTool] Error:', err);
                return toToolResult(`Tool error: ${err.message}`);
            }
        }
    });
}

/**
 * Create and register all RepoAsk VS Code LM tools.
 *
 * @param {object} deps
 * @returns {{ registerRepoAskLanguageModelTools: function }}
 */
function createLanguageModelTools(deps) {
    function registerRepoAskLanguageModelTools() {
        if (!vscode.lm || typeof vscode.lm.registerTool !== 'function') {
            return [];
        }
        return [registerDocCheckTool(deps)];
    }

    return { registerRepoAskLanguageModelTools };
}

module.exports = {
    // Utilities
    toToolResult,
    buildCheckAllDocsCommandLink,
    formatRefreshStatus,
    // Registration
    createLanguageModelTools,
    // Schema
    ALLOWED_MODES
};

