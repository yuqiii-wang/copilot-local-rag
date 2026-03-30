const fs = require('fs');
const path = require('path');
const cheerio = require('cheerio');
const axios = require('axios');
const { extractJsonObject } = require('./../tools/llm');
const { tokenize: tokenizeFromModule } = require('./tokenization2keywords');
const { tokenization2bm25 } = require('./tokenization2bm25');

const ranking = require('./ranking');
const sync = require('./sync');
const annotation = require('./annotation');
const images = require('./images');
const keywords = require('./keywords');
const utils = require('./utils');
const knowledgeGraph = require('./knowledgeGraph');

function createDocumentService(deps) {
  const {
  indexStoragePath,
} = deps;
  const tokenize = tokenizeFromModule;

  const context = {
    ...deps,
    tokenize,
    tokenization2bm25,
    fs, path, cheerio, axios, extractJsonObject
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
  const _knowledgeGraph = knowledgeGraph(proxyContext); Object.assign(context, _knowledgeGraph);

  return {
    syncDefaultDocs: _sync.syncDefaultDocs,
    rankLocalDocuments: _ranking.rankLocalDocuments,
    refreshDocument: _sync.refreshDocument,
    refreshConfluenceHierarchy: _sync.refreshConfluenceHierarchy,
    refreshAllDocuments: _sync.refreshAllDocuments,
    refreshJiraIssue: _sync.refreshJiraIssue,
    annotateDocumentByArg: _annotation.annotateDocumentByArg,
    annotateAllDocuments: _annotation.annotateAllDocuments,
    finalizeBm25KeywordsForDocuments: _sync.finalizeBm25KeywordsForDocuments,
    getStoredMetadataById: _utils.getStoredMetadataById,
    generateStoredMetadataById: _utils.generateStoredMetadataById,
    updateStoredMetadataById: _utils.updateStoredMetadataById,
    removeDocumentFromIndicesById: _utils.removeDocumentFromIndicesById,
    writeDocumentPromptFile: _utils.writeDocumentPromptFile,
    writeDocumentSkillFile: _utils.writeDocumentSkillFile,
    buildKnowledgeGraph: _knowledgeGraph.buildKnowledgeGraph,
    saveKnowledgeGraph: _knowledgeGraph.saveKnowledgeGraph
  };
}

module.exports = { createDocumentService };
