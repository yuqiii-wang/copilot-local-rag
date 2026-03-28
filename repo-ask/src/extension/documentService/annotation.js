module.exports = function(context) {
  const { vscode, storagePath, fetchConfluencePage, truncate, tokenize, generateSynonyms, generateSummary, readAllMetadata, writeDocumentFiles, readDocumentContent, getKeywordConfig, buildKeywordOnlyIndexText, cleanKeywords, normalizeMetadataKeywordFields, appendKeywordsToExisting, extractJsonObject, flattenCategorizedKeywords, mergeSemanticKeywords } = context;
  const { buildAnnotationPrompt } = require('../chat/prompts');

async function annotateDocumentByArg(pageArg) {
  const allMetadata = readAllMetadata(storagePath);
  let metadata = allMetadata.find(item => String(item.id) === pageArg || String(item.title) === pageArg);
  if (!metadata) {
    const page = await fetchConfluencePage(pageArg);
    metadata = allMetadata.find(item => String(item.id) === String(page?.id));
  }
  if (!metadata) {
    return {
      message: `Document ${pageArg} is not in local store. Run refresh first.`
    };
  }
  const updated = await annotateStoredDocument(metadata);
  if (!updated) {
    return {
      message: `No local plain text content for ${metadata.title}. Run refresh first.`
    };
  }
  await finalizeBm25KeywordsForDocuments([metadata.id]);
  return {
    message: `Annotated document: ${metadata.title}`
  };
}

async function annotateAllDocuments() {
  const allMetadata = readAllMetadata(storagePath);
  let updatedCount = 0;
  const updatedIds = [];
  for (const metadata of allMetadata) {
    const updated = await annotateStoredDocument(metadata);
    if (updated) {
      updatedCount += 1;
      updatedIds.push(metadata.id);
    }
  }
  if (updatedCount > 0) {
    await finalizeBm25KeywordsForDocuments(updatedIds);
  }
  return {
    message: updatedCount > 0 ? `Annotated ${updatedCount} document(s)` : 'No local documents available to annotate. Run refresh first.'
  };
}

async function generateAnnotationWithLlm(metadata, content) {
  const originalKeywords = cleanKeywords(flattenCategorizedKeywords(metadata?.keywords));
  const tokenizationSource = [metadata?.title || '', content || ''].filter(Boolean).join('\n');
  const fallbackKeywords = tokenize(tokenizationSource);
  const fallbackSummary = generateSummary(content);
  function appendSynonymKeywords(baseKeywords, maxSynonyms = 6) {
    const orderedBase = cleanKeywords(baseKeywords);
    if (orderedBase.length === 0) {
      return [];
    }
    const synonymCandidates = cleanKeywords(generateSynonyms(orderedBase), 80).filter(keyword => !orderedBase.includes(keyword)).slice(0, maxSynonyms);
    return cleanKeywords([...orderedBase, ...synonymCandidates]);
  }
  if (!vscode.lm || !vscode.LanguageModelChatMessage) {
    return {
      keywords: appendSynonymKeywords([...originalKeywords, ...fallbackKeywords]),
      summary: fallbackSummary
    };
  }
  try {
    const shared = require('../chat/shared');
    const model = await shared.selectDefaultChatModel(vscode);
    if (!model) {
      return {
        keywords: appendSynonymKeywords([...originalKeywords, ...fallbackKeywords]),
        summary: fallbackSummary
      };
    }
    const prompt = buildAnnotationPrompt({
      title: metadata.title,
      parentTopic: metadata.parent_confluence_topic,
      author: metadata.author,
      keywords: originalKeywords.join(', '),
      tags: metadata.tags ? metadata.tags.join(', ') : '',
      feedback: metadata.feedback || '',
      content: truncate(content, 100000)
    });
    const response = await model.sendRequest([vscode.LanguageModelChatMessage.User(prompt)]);
    let responseText = '';
    if (response) {
      if (response.stream) {
        for await (const chunk of response.stream) {
          if (chunk instanceof vscode.LanguageModelTextPart) {
            responseText += chunk.value;
          }
        }
      } else if (response.text) {
        for await (const fragment of response.text) {
          responseText += fragment;
        }
      }
    }
    const parsed = extractJsonObject(responseText) || {};
    const llmKeywords = cleanKeywords(parsed.keywords);
    const llmSummary = String(parsed.summary || '').trim();
    
    // Replace original keywords entirely with the LLM output, filling with fallback if needed
    const combinedLlmKeywords = cleanKeywords([...llmKeywords, ...fallbackKeywords]).slice(0, getKeywordConfig().DEFAULT_KEYWORD_LIMIT);
    const fallbackMergedKeywords = appendKeywordsToExisting(originalKeywords, fallbackKeywords, getKeywordConfig().DEFAULT_KEYWORD_LIMIT);
    
    return {
      keywords: llmKeywords.length > 0 ? appendSynonymKeywords(combinedLlmKeywords) : appendSynonymKeywords(fallbackMergedKeywords),
      summary: llmSummary || fallbackSummary
    };
  } catch {
    return {
      keywords: appendSynonymKeywords([...originalKeywords, ...fallbackKeywords]),
      summary: fallbackSummary
    };
  }
}

async function annotateStoredDocument(metadata) {
  const content = readDocumentContent(storagePath, metadata.id);
  if (!content) {
    return false;
  }
  const annotation = await generateAnnotationWithLlm(metadata, content);
  // Preserve all categorized keyword categories; place LLM keywords in `semantic`
  const semanticKeywords = cleanKeywords(annotation.keywords, getKeywordConfig().DEFAULT_KEYWORD_LIMIT);
  const updatedMetadata = normalizeMetadataKeywordFields({
    ...metadata,
    keywords: mergeSemanticKeywords(metadata.keywords, semanticKeywords),
    summary: annotation.summary
  });
  writeDocumentFiles(storagePath, metadata.id, content, updatedMetadata);
  return true;
}

  return {
    annotateDocumentByArg,
    annotateAllDocuments,
    annotateStoredDocument,
    generateAnnotationWithLlm
  };
};
