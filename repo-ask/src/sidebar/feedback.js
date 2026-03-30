// feedback.js

function setButtonLoadingState(button, isLoading, label) {
    if (!button) {
        return;
    }

    if (isLoading) {
        if (!button.dataset.defaultLabel) {
            button.dataset.defaultLabel = label || button.textContent || '';
        }
        button.dataset.loadingLabel = label || button.dataset.defaultLabel || 'Working...';
        button.disabled = true;
        button.classList.add('is-loading');
        button.textContent = button.dataset.loadingLabel;
        button.setAttribute('aria-busy', 'true');
        return;
    }

    button.classList.remove('is-loading');
    button.removeAttribute('aria-busy');
    button.disabled = false;
    button.textContent = label || button.dataset.defaultLabel || button.textContent || '';
}

async function settleSubmitState(submitBtn, success, label) {
    if (!submitBtn) {
        return;
    }

    const startedAt = Number(submitBtn.dataset.loadingStartedAt || 0);
    const elapsed = startedAt > 0 ? (Date.now() - startedAt) : 0;
    const minVisibleMs = 350;
    const waitMs = Math.max(0, minVisibleMs - elapsed);
    if (waitMs > 0) {
        await new Promise((resolve) => setTimeout(resolve, waitMs));
    }

    setButtonLoadingState(submitBtn, false, label);
    if (success) {
        submitBtn.disabled = true;
    }
}

function initFeedbackForm() {
    const feedbackSection = document.getElementById('feedback-section');
    const submitBtn = document.getElementById('submit-feedback-btn');
    const cancelBtn = document.getElementById('cancel-feedback-btn');
    const generateSummaryBtn = document.getElementById('generate-summary-btn');
    const datetimeInput = document.getElementById('datetime');
    const addSecondaryUrlBtn = document.getElementById('add-secondary-url-btn');
    const secondaryUrlsContainer = document.getElementById('secondary-urls-container');
    
    // Function to re-enable submit button on form changes
    function enableSubmitButton() {
        if (submitBtn) {
            submitBtn.disabled = false;
        }
    }
    
    // Set default datetime to current time
        const now = new Date();
        const formattedDate = now.toISOString().slice(0, 16);
        if (datetimeInput) {
            datetimeInput.value = formattedDate;
        }

        // Add event listener to confluence-link input to extract IDs
        const confluenceLinkInput = document.getElementById('confluence-link');
        const confluencePageIdGroup = document.querySelector('label[for="confluence-page-id"]').parentElement;
        const jiraIdGroup = document.querySelector('label[for="jira-id"]').parentElement;
        const confluencePageIdInput = document.getElementById('confluence-page-id');
        const jiraIdInput = document.getElementById('jira-id');
        
        // Function to sync ID and link fields
        function syncIdAndLink() {
            // Get current values
            const link = confluenceLinkInput.value.trim();
            const confluenceId = confluencePageIdInput.value.trim();
            const jiraId = jiraIdInput.value.trim();
            
            // Determine which ID is present
            if (confluenceId) {
                // Confluence ID is present, hide Jira field
                if (confluencePageIdGroup) confluencePageIdGroup.style.display = 'block';
                if (jiraIdGroup) jiraIdGroup.style.display = 'none';
                if (jiraIdInput) jiraIdInput.value = '';
                
                // Clear KG while loading, then fetch doc (KG will be populated via documentFound)
                if (typeof populateKnowledgeGraph === 'function') populateKnowledgeGraph('');
                vscode.postMessage({ 
                    command: 'getDocumentByID', 
                    id: confluenceId 
                });
            } else if (jiraId) {
                // Jira ID is present, hide Confluence field
                if (jiraIdGroup) jiraIdGroup.style.display = 'block';
                if (confluencePageIdGroup) confluencePageIdGroup.style.display = 'none';
                if (confluencePageIdInput) confluencePageIdInput.value = '';
                
                // Clear KG while loading, then fetch doc (KG will be populated via documentFound)
                if (typeof populateKnowledgeGraph === 'function') populateKnowledgeGraph('');
                vscode.postMessage({ 
                    command: 'getDocumentByID', 
                    id: jiraId 
                });
            } else if (link) {
                // Only link is present, extract ID from link
                // Try to extract Confluence page ID
                const confluenceMatch = link.match(/(?:[?&]pageId=|\/pages\/|\/viewpage\/|\.action\/|\?pageId=)(\d+)/i);
                if (confluenceMatch && confluenceMatch[1]) {
                    // It's a Confluence link, extract page ID
                    if (confluencePageIdGroup) confluencePageIdGroup.style.display = 'block';
                    if (jiraIdGroup) jiraIdGroup.style.display = 'none';
                    if (confluencePageIdInput) confluencePageIdInput.value = confluenceMatch[1];
                    if (jiraIdInput) jiraIdInput.value = '';
                    if (typeof populateKnowledgeGraph === 'function') populateKnowledgeGraph('');
                    vscode.postMessage({ command: 'getDocumentByID', id: confluenceMatch[1] });
                    return;
                }

                // Try to extract Jira issue ID
                const jiraMatch = link.match(/[A-Z]+-\d+/i);
                if (jiraMatch) {
                    // It's a Jira link, extract issue ID
                    if (jiraIdGroup) jiraIdGroup.style.display = 'block';
                    if (confluencePageIdGroup) confluencePageIdGroup.style.display = 'none';
                    if (jiraIdInput) jiraIdInput.value = jiraMatch[0].toUpperCase();
                    if (confluencePageIdInput) confluencePageIdInput.value = '';
                    if (typeof populateKnowledgeGraph === 'function') populateKnowledgeGraph('');
                    vscode.postMessage({ command: 'getDocumentByID', id: jiraMatch[0].toUpperCase() });
                    return;
                }

                // If no ID found, show both fields and clear KG
                if (confluencePageIdGroup) confluencePageIdGroup.style.display = 'block';
                if (jiraIdGroup) jiraIdGroup.style.display = 'block';
                if (typeof populateKnowledgeGraph === 'function') populateKnowledgeGraph('');
            } else {
                // No values, show both fields and clear KG
                if (confluencePageIdGroup) confluencePageIdGroup.style.display = 'block';
                if (jiraIdGroup) jiraIdGroup.style.display = 'block';
                if (typeof populateKnowledgeGraph === 'function') populateKnowledgeGraph('');
            }
        }
        
        // Add event listeners
        if (confluenceLinkInput) {
            confluenceLinkInput.addEventListener('input', syncIdAndLink);
            confluenceLinkInput.addEventListener('input', enableSubmitButton);
        }
        
        if (confluencePageIdInput) {
            confluencePageIdInput.addEventListener('input', syncIdAndLink);
            confluencePageIdInput.addEventListener('input', enableSubmitButton);
        }
        
        if (jiraIdInput) {
            jiraIdInput.addEventListener('input', syncIdAndLink);
            jiraIdInput.addEventListener('input', enableSubmitButton);
        }
        
        // Add listener for source query
        const sourceQueryInput = document.getElementById('source-query');
        if (sourceQueryInput) {
            sourceQueryInput.addEventListener('input', enableSubmitButton);
        }
        
        // Secondary URLs functionality
        function addSecondaryUrlItem() {
            const newItem = document.createElement('div');
            newItem.className = 'secondary-url-item';
            newItem.innerHTML = `
                <input type="text" class="form-input secondary-url-input" placeholder="Enter secondary URL or ID">
                <button type="button" class="remove-secondary-url-btn">Remove</button>
            `;
            secondaryUrlsContainer.appendChild(newItem);
            
            // Add event listener to the new input field
            const newInput = newItem.querySelector('.secondary-url-input');
            if (newInput) {
                newInput.addEventListener('input', enableSubmitButton);
            }
            
            // Add event listener to the new remove button
            const removeBtn = newItem.querySelector('.remove-secondary-url-btn');
            removeBtn.addEventListener('click', function() {
                newItem.remove();
                enableSubmitButton();
            });
        }
        
        // Add event listener for add secondary URL button
        if (addSecondaryUrlBtn) {
            addSecondaryUrlBtn.addEventListener('click', addSecondaryUrlItem);
        }
        
        // Add event listeners to existing remove buttons and inputs
        const existingRemoveBtns = document.querySelectorAll('.remove-secondary-url-btn');
        existingRemoveBtns.forEach(btn => {
            btn.addEventListener('click', function() {
                const item = this.closest('.secondary-url-item');
                if (item) {
                    item.remove();
                    enableSubmitButton();
                }
            });
        });
        
        const existingSecondaryInputs = document.querySelectorAll('.secondary-url-input');
        existingSecondaryInputs.forEach(input => {
            input.addEventListener('input', enableSubmitButton);
        });
    
    // Submit button handler
    if (submitBtn) {
        submitBtn.addEventListener('click', async () => {
            if (typeof renderSuccessMessage === 'function') renderSuccessMessage('');
            if (typeof renderSyncError === 'function') renderSyncError('');

            const sourceQueryRaw = document.getElementById('source-query')?.value || '';
            const conversationSummaryRaw = document.getElementById('conversation-summary')?.value || '';
            const confluenceLinkRaw = document.getElementById('confluence-link')?.value || '';
            const confluencePageIdRaw = document.getElementById('confluence-page-id')?.value || '';
            const jiraIdRaw = document.getElementById('jira-id')?.value || '';
            const usernameRaw = document.getElementById('username')?.value || 'Anonymous';
            const elapsedTimeRaw = document.getElementById('elapsed-time')?.value || '';
            const datetimeRaw = document.getElementById('datetime')?.value || '';
            const tagsRaw = document.getElementById('tags')?.value || '';

            // Collect secondary URLs/IDs
            const secondaryUrlInputs = document.querySelectorAll('.secondary-url-input');
            const secondaryUrls = Array.from(secondaryUrlInputs)
                .map(input => input.value.trim())
                .filter(url => url);

            const sourceQuery = sourceQueryRaw.trim();
            const conversationSummary = conversationSummaryRaw.trim();
            const confluenceLink = confluenceLinkRaw.trim();
            const confluencePageId = confluencePageIdRaw.trim();
            const jiraId = jiraIdRaw.trim();
            const username = usernameRaw.trim() || 'Anonymous';
            const elapsedTime = elapsedTimeRaw.trim();
            const datetime = datetimeRaw.trim();
            const tags = tagsRaw.trim();
            
            // Validate input
            if (!conversationSummary) {
                if (typeof renderSyncError === 'function') renderSyncError('Conversation Summary is required. Please paste or generate the full AI response before submitting.');
                return;
            }

            if (!sourceQuery || !datetime) {
                if (typeof renderSyncError === 'function') renderSyncError('Please fill in all required fields: Source Query and Datetime');
                return;
            }

            // Check if either Confluence page ID or Jira ID is provided
            if (!confluencePageId && !jiraId) {
                if (typeof renderSyncError === 'function') renderSyncError('Please provide either Confluence Page ID or Jira ID');
                return;
            }

            // Validate Confluence page ID format (should be numeric) if provided
            if (confluencePageId && !/^\d+$/.test(confluencePageId)) {
                if (typeof renderSyncError === 'function') renderSyncError('Confluence Page ID must be a numeric value');
                return;
            }

            // Validate Jira ID format (e.g., PROJ-123) if provided
            if (jiraId && !/^[A-Z]+-\d+$/.test(jiraId)) {
                if (typeof renderSyncError === 'function') renderSyncError('Jira ID must be in the format PROJ-123');
                return;
            }

            // Validate Confluence/Jira Link format if provided
            if (confluenceLink && !/^https?:\/\//i.test(confluenceLink)) {
                if (typeof renderSyncError === 'function') renderSyncError('Confluence/Jira Link must be a valid URL');
                return;
            }
            
            // Disable submit button and show loading state
            setButtonLoadingState(submitBtn, true, 'Submit');
            submitBtn.dataset.loadingStartedAt = String(Date.now());
            
            // Yield one frame so the loading style paints before posting to the extension host.
            await new Promise((resolve) => requestAnimationFrame(resolve));
            
            const feedbackPayload = {
                sourceQuery: sourceQueryRaw,
                conversationSummary: conversationSummaryRaw,
                confluenceLink: confluenceLinkRaw,
                confluencePageId: confluencePageIdRaw,
                jiraId: jiraIdRaw,
                username: usernameRaw,
                elapsedTime: elapsedTimeRaw,
                datetime: datetimeRaw,
                tags: tagsRaw,
                secondaryUrls: secondaryUrls.length > 0 ? secondaryUrls : ['none'],
                knowledge_graph: document.getElementById('knowledge-graph-raw')?.value?.trim() || ''
            };
            
            try {
                // Send feedback to extension
                vscode.postMessage({ 
                    command: 'submitFeedback', 
                    feedbackPayload
                });
            } catch (error) {
                console.error('Error submitting feedback:', error);
                if (typeof renderSyncError === 'function') renderSyncError(error.message || String(error));
                // Re-enable submit button
                settleSubmitState(submitBtn, false, 'Submit');
            }
        });
    }
    
    // Cancel button handler
    if (cancelBtn) {
        cancelBtn.addEventListener('click', () => {
            if (feedbackSection) {
                feedbackSection.style.display = 'none';
            }
            
            const viewerSection = document.querySelector('.viewer');
            if (viewerSection) {
                viewerSection.style.display = 'grid';
            }
        });
    }
    
    // Generate AI summary button handler
    if (generateSummaryBtn) {
        generateSummaryBtn.addEventListener('click', () => {
            if (typeof renderSuccessMessage === 'function') renderSuccessMessage('');
            if (typeof renderSyncError === 'function') renderSyncError('');

            const conversationSummary = document.getElementById('conversation-summary')?.value || '';
            if (!conversationSummary.trim()) {
                if (typeof renderSyncError === 'function') renderSyncError('Please enter a conversation summary first');
                return;
            }
            
            // Show spinner and disable buttons
            setButtonLoadingState(generateSummaryBtn, true, 'Generate AI Summary');
            
            // Disable submit button
            const submitBtn = document.getElementById('submit-feedback-btn');
            if (submitBtn) {
                submitBtn.disabled = true;
            }

            // Also disable KG button while summary is generating
            const kgBtn = document.getElementById('generate-kg-btn');
            if (kgBtn) kgBtn.disabled = true;

            // Collect form context so the extension can also build the knowledge graph
            const secondaryUrlInputs = document.querySelectorAll('.secondary-url-input');
            const secondaryUrls = Array.from(secondaryUrlInputs).map(i => i.value.trim()).filter(u => u);
            
            // Send request to generate summary (with form context for KG generation)
            vscode.postMessage({ 
                command: 'generateSummary', 
                conversationSummary,
                sourceQuery: document.getElementById('source-query')?.value || '',
                confluencePageId: document.getElementById('confluence-page-id')?.value || '',
                jiraId: document.getElementById('jira-id')?.value || '',
                confluenceLink: document.getElementById('confluence-link')?.value || '',
                secondaryUrls
            });
        });
    }

    // Generate Knowledge Graph button handler
    const generateKgBtn = document.getElementById('generate-kg-btn');
    if (generateKgBtn) {
        generateKgBtn.addEventListener('click', () => {
            if (typeof renderSuccessMessage === 'function') renderSuccessMessage('');
            if (typeof renderSyncError === 'function') renderSyncError('');

            const confluencePageId = document.getElementById('confluence-page-id')?.value?.trim() || '';
            const jiraId = document.getElementById('jira-id')?.value?.trim() || '';
            if (!confluencePageId && !jiraId) {
                if (typeof renderSyncError === 'function') renderSyncError('Please provide a Confluence Page ID or Jira ID before generating the knowledge graph.');
                return;
            }

            const secondaryUrlInputs = document.querySelectorAll('.secondary-url-input');
            const secondaryUrls = Array.from(secondaryUrlInputs).map(i => i.value.trim()).filter(u => u);

            setButtonLoadingState(generateKgBtn, true, 'Generate Knowledge Graph');

            // Get the existing knowledge graph so the AI can edit it based on primary and secondary doc contents
            const existingKnowledgeGraph = document.getElementById('knowledge-graph-raw')?.value?.trim() || '';

            vscode.postMessage({
                command: 'generateKnowledgeGraph',
                sourceQuery: document.getElementById('source-query')?.value || '',
                confluencePageId,
                jiraId,
                confluenceLink: document.getElementById('confluence-link')?.value || '',
                secondaryUrls,
                existingKnowledgeGraph,
                conversationSummary: document.getElementById('conversation-summary')?.value?.trim() || ''
            });
        });
    }

}

function escapeRegExp(value) {
    return String(value || '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function sanitizeSourceQuery(rawQuery, selectedDocument, fallbackUrl) {
    let query = String(rawQuery || '');
    if (!query.trim()) {
        return '';
    }

    const urlsToRemove = new Set();
    const selectedUrl = String(selectedDocument?.url || '').trim();
    const fallback = String(fallbackUrl || '').trim();
    if (selectedUrl) {
        urlsToRemove.add(selectedUrl);
    }
    if (fallback && fallback !== '[NO_URL]') {
        urlsToRemove.add(fallback);
    }

    for (const url of urlsToRemove) {
        query = query.replace(new RegExp(escapeRegExp(url), 'gi'), ' ');
    }

    const confluenceId = String(selectedDocument?.confluencePageId || '').trim();
    const jiraId = String(selectedDocument?.jiraId || '').trim();

    if (confluenceId) {
        query = query.replace(new RegExp(`\\bpageId\\s*=\\s*${escapeRegExp(confluenceId)}\\b`, 'gi'), ' ');
        query = query.replace(new RegExp(`\\b${escapeRegExp(confluenceId)}\\b`, 'g'), ' ');
    }
    if (jiraId) {
        query = query.replace(new RegExp(`\\b${escapeRegExp(jiraId)}\\b`, 'gi'), ' ');
    }

    // Strip any URL tokens that may still be present in the copied query.
    query = query.replace(/\bhttps?:\/\/[^\s<>"'`]+/gi, ' ');

    // Drop any structured top-doc tags if they appear in the copied query.
    query = query.replace(/\[TOP_DOC_URL:[^\]]*\]/gi, ' ');
    query = query.replace(/\[TOP_DOC_ID:[^\]]*\]/gi, ' ');

    return query.replace(/\s{2,}/g, ' ').trim();
}

// Function to show the feedback form
function showFeedbackForm(firstUserQuery, selectedDocumentOrUrl, fullAiResponse, queryStartTime, username) {
    const feedbackSection = document.getElementById('feedback-section');
    const viewerSection = document.querySelector('.viewer');
    const selectedDocument = selectedDocumentOrUrl && typeof selectedDocumentOrUrl === 'object'
        ? selectedDocumentOrUrl
        : null;
    const fallbackUrl = typeof selectedDocumentOrUrl === 'string' ? selectedDocumentOrUrl : '';

    if (feedbackSection) {
        feedbackSection.style.display = 'block';
    }

    if (viewerSection) {
        viewerSection.style.display = 'none';
    }

    // Calculate elapsed time if queryStartTime is provided
    if (queryStartTime) {
        const elapsedTimeField = document.getElementById('elapsed-time');
        if (elapsedTimeField) {
            const elapsedMs = Date.now() - queryStartTime;
            elapsedTimeField.value = Math.round(elapsedMs / 1000); // converting to seconds
        }
    }
    
    // Copy the first user query to the source query field
    if (firstUserQuery) {
        const sourceQueryField = document.getElementById('source-query');
        if (sourceQueryField) {
            sourceQueryField.value = sanitizeSourceQuery(firstUserQuery, selectedDocument, fallbackUrl);
        }
    }
    
    const confluenceLinkField = document.getElementById('confluence-link');
    const confluencePageIdField = document.getElementById('confluence-page-id');
    const jiraIdField = document.getElementById('jira-id');

    if (confluenceLinkField) {
        confluenceLinkField.value = selectedDocument?.url || fallbackUrl || '';
    }
    if (confluencePageIdField) {
        confluencePageIdField.value = selectedDocument?.confluencePageId || '';
    }
    if (jiraIdField) {
        jiraIdField.value = selectedDocument?.jiraId || '';
    }

    // Copy the full AI response into conversation summary when available.
    if (typeof fullAiResponse === 'string') {
        const summaryField = document.getElementById('conversation-summary');
        if (summaryField) {
            summaryField.value = fullAiResponse;
        }
    }
    
    // Load the stored knowledge graph from the selected primary document
    if (selectedDocument && selectedDocument.knowledgeGraph) {
        const kgRawField = document.getElementById('knowledge-graph-raw');
        if (kgRawField) {
            kgRawField.value = selectedDocument.knowledgeGraph;
        }
    }
    
    // Set the username field with the provided username, default to anonymous if not provided
    if (username) {
        const usernameField = document.getElementById('username');
        if (usernameField) {
            usernameField.value = username;
        }
    }
}

// Function to populate the summary field
function populateSummary(summary) {
    const summaryField = document.getElementById('conversation-summary');
    if (summaryField) {
        summaryField.value = summary;
    }
    
    // Reset the generate summary button
    const generateSummaryBtn = document.getElementById('generate-summary-btn');
    if (generateSummaryBtn) {
        setButtonLoadingState(generateSummaryBtn, false, 'Generate AI Summary');
    }

    // Re-enable KG button
    const generateKgBtn = document.getElementById('generate-kg-btn');
    if (generateKgBtn) {
        generateKgBtn.disabled = false;
    }
    
    // Re-enable submit button
    const submitBtn = document.getElementById('submit-feedback-btn');
    if (submitBtn) {
        submitBtn.disabled = false;
    }
}

// Function to populate the knowledge graph field
function populateKnowledgeGraph(mermaid) {
    const rawField = document.getElementById('knowledge-graph-raw');
    if (rawField) {
        rawField.value = mermaid || '';
    }
    // Re-enable KG generate button
    const generateKgBtn = document.getElementById('generate-kg-btn');
    if (generateKgBtn) {
        setButtonLoadingState(generateKgBtn, false, 'Generate Knowledge Graph');
    }
}

// Expose functions to global scope
window.initFeedbackForm = initFeedbackForm;
window.showFeedbackForm = showFeedbackForm;
window.populateSummary = populateSummary;
window.populateKnowledgeGraph = populateKnowledgeGraph;

// Handle messages from extension
window.addEventListener('message', (event) => {
    const message = event.data;
    if (message?.command === 'showFeedbackForm') {
        showFeedbackForm(message.firstUserQuery, message.selectedDocument || message.firstRankedDocUrl, message.fullAiResponse, message.queryStartTime, message.username);
    }
    if (message?.command === 'populateSummary') {
        populateSummary(String(message.summary || ''));
    }
    if (message?.command === 'feedbackSubmitted') {
        const success = message.success;
        const errorMessage = message.error;
        const detailedError = message.detailedError;
        const submitBtn = document.getElementById('submit-feedback-btn');

        if (success) {
            // Show success message
            if (typeof renderSuccessMessage === 'function') renderSuccessMessage('Feedback submitted successfully!');

            // Clear form
            document.getElementById('source-query').value = '';
            document.getElementById('conversation-summary').value = '';
            document.getElementById('confluence-link').value = '';
            document.getElementById('confluence-page-id').value = '';
            document.getElementById('jira-id').value = '';
            document.getElementById('datetime').value = new Date().toISOString().slice(0, 16);
            document.getElementById('tags').value = '';
            
            // Reset secondary URLs section
            const secondaryUrlsContainer = document.getElementById('secondary-urls-container');
            if (secondaryUrlsContainer) {
                // Clear all existing items except the first one
                const items = secondaryUrlsContainer.querySelectorAll('.secondary-url-item');
                items.forEach((item, index) => {
                    if (index > 0) {
                        item.remove();
                    } else {
                        // Clear the first item's input
                        const input = item.querySelector('.secondary-url-input');
                        if (input) {
                            input.value = '';
                        }
                    }
                });
            }

            // Reset knowledge graph section
            const kgRawField = document.getElementById('knowledge-graph-raw');
            if (kgRawField) kgRawField.value = '';


            settleSubmitState(submitBtn, true, 'Submit');

            // Hide feedback section and restore original content after a short delay
            setTimeout(() => {
                const feedbackSection = document.getElementById('feedback-section');
                if (feedbackSection) {
                    feedbackSection.style.display = 'none';
                }

                const viewerSection = document.querySelector('.viewer');
                if (viewerSection) {
                    viewerSection.style.display = 'grid';
                }
            }, 10000);
        } else {
            // Show error message, prefer detailed error if available
            let displayError = errorMessage || 'Unknown error';
            if (detailedError && detailedError !== errorMessage) {
                displayError += `\nDetails: ${detailedError}`;
            }
            console.error('[feedback.js] Feedback submission failed:', displayError, { errorMessage, detailedError });
            if (typeof renderSyncError === 'function') renderSyncError(displayError);

            // Re-enable submit button
            settleSubmitState(submitBtn, false, 'Submit');
        }
    }
    if (message?.command === 'populateKnowledgeGraph') {
        populateKnowledgeGraph(String(message.mermaid || ''));
    }
    if (message?.command === 'documentFound') {
        // Update link field with document URL if found
        const confluenceLinkInput = document.getElementById('confluence-link');
        if (confluenceLinkInput && message.document && message.document.source) {
            confluenceLinkInput.value = message.document.source;
        }
        // Load the stored knowledge graph for the selected primary doc
        if (typeof populateKnowledgeGraph === 'function') {
            populateKnowledgeGraph(String(message.document?.knowledgeGraph || ''));
        }
    }
});