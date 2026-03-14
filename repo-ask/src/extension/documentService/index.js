const fs = require('fs');
const path = require('path');
const cheerio = require('cheerio');
const axios = require('axios');
const { extractJsonObject } = require('./../tools/llm');
const { createBm25Index } = require('../../docIndex/bm25');

const ranking = require('./ranking');
const sync = require('./sync');
const annotation = require('./annotation');
const images = require('./images');
const keywords = require('./keywords');
const utils = require('./utils');

function createDocumentService(deps) {
  const {
  vscode,
  storagePath,
  indexStoragePath,
  fetchConfluencePage,
  fetchAllConfluencePages,
  fetchJiraIssue,
  truncate,
  tokenize,
  htmlToMarkdown,
  generateKeywords,
  generateExtendedKeywords,
  generateSummary,
  readAllMetadata,
  writeDocumentFiles,
  readDocumentContent,
  rankDocumentsByIdf
} = deps;
  const bm25Index = createBm25Index({
  storePath: indexStoragePath,
  indexFileName: 'bm25-index.json',
  tokenize
});
  const keywordsIndex = createBm25Index({
  storePath: indexStoragePath,
  indexFileName: 'keywords-index.json',
  tokenize
});
  bm25Index.ensureStorePath();

  const context = {
    ...deps,
    bm25Index,
    keywordsIndex,
    fs, path, cheerio, axios, extractJsonObject, createBm25Index
  };

  const proxyContext = new Proxy(context, {
    get(target, prop) {
      if (prop in target) {
        return target[prop];
      }
      return (...args) => {
        if (typeof target[prop] === 'function') {
          return target[prop](...args);
        }
        return undefined;
      };
    }
  });

  const _utils = utils(proxyContext); Object.assign(context, _utils);
  const _keywords = keywords(proxyContext); Object.assign(context, _keywords);
  const _images = images(proxyContext); Object.assign(context, _images);
  const _annotation = annotation(proxyContext); Object.assign(context, _annotation);
  const _sync = sync(proxyContext); Object.assign(context, _sync);
  const _ranking = ranking(proxyContext); Object.assign(context, _ranking);

  return {
    syncDefaultDocs: _sync.syncDefaultDocs,
    rankLocalDocuments: _ranking.rankLocalDocuments,
    checkLocalDocumentsAgentic: _ranking.checkLocalDocumentsAgentic,
    refreshDocument: _sync.refreshDocument,
    refreshConfluenceHierarchy: _sync.refreshConfluenceHierarchy,
    refreshAllDocuments: _sync.refreshAllDocuments,
    refreshJiraIssue: _sync.refreshJiraIssue,
    annotateDocumentByArg: _annotation.annotateDocumentByArg,
    annotateAllDocuments: _annotation.annotateAllDocuments,
    getStoredMetadataById: _utils.getStoredMetadataById,
    generateStoredMetadataById: _utils.generateStoredMetadataById,
    updateStoredMetadataById: _utils.updateStoredMetadataById,
    removeDocumentFromIndicesById: _utils.removeDocumentFromIndicesById,
    writeDocumentPromptFile: _utils.writeDocumentPromptFile,
    formatMetadataEntries: _utils.formatMetadataEntries
  };
}

module.exports = { createDocumentService };
