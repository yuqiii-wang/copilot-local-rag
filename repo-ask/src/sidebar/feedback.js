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
                
                // Try to get link from local store via message to extension
                vscode.postMessage({ 
                    command: 'getDocumentByID', 
                    id: confluenceId 
                });
            } else if (jiraId) {
                // Jira ID is present, hide Confluence field
                if (jiraIdGroup) jiraIdGroup.style.display = 'block';
                if (confluencePageIdGroup) confluencePageIdGroup.style.display = 'none';
                if (confluencePageIdInput) confluencePageIdInput.value = '';
                
                // Try to get link from local store via message to extension
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
                    return;
                }

                // If no ID found, show both fields
                if (confluencePageIdGroup) confluencePageIdGroup.style.display = 'block';
                if (jiraIdGroup) jiraIdGroup.style.display = 'block';
            } else {
                // No values, show both fields
                if (confluencePageIdGroup) confluencePageIdGroup.style.display = 'block';
                if (jiraIdGroup) jiraIdGroup.style.display = 'block';
            }
        }
        
        // Add event listeners
        if (confluenceLinkInput) {
            confluenceLinkInput.addEventListener('input', syncIdAndLink);
        }
        
        if (confluencePageIdInput) {
            confluencePageIdInput.addEventListener('input', syncIdAndLink);
        }
        
        if (jiraIdInput) {
            jiraIdInput.addEventListener('input', syncIdAndLink);
        }
    
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
            const usernameRaw = document.getElementById('username')?.value || 'anonymous';
            const elapsedTimeRaw = document.getElementById('elapsed-time')?.value || '';
            const datetimeRaw = document.getElementById('datetime')?.value || '';
            const tagsRaw = document.getElementById('tags')?.value || '';

            const sourceQuery = sourceQueryRaw.trim();
            const conversationSummary = conversationSummaryRaw.trim();
            const confluenceLink = confluenceLinkRaw.trim();
            const confluencePageId = confluencePageIdRaw.trim();
            const jiraId = jiraIdRaw.trim();
            const username = usernameRaw.trim() || 'anonymous';
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
                tags: tagsRaw
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
            
            // Send request to generate summary
            vscode.postMessage({ 
                command: 'generateSummary', 
                conversationSummary 
            });
        });
    }
}

// Function to show the feedback form
function showFeedbackForm(firstUserQuery, selectedDocumentOrUrl, fullAiResponse, queryStartTime) {
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
            sourceQueryField.value = firstUserQuery;
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
    
    // Re-enable submit button
    const submitBtn = document.getElementById('submit-feedback-btn');
    if (submitBtn) {
        submitBtn.disabled = false;
    }
}

// Expose functions to global scope
window.initFeedbackForm = initFeedbackForm;
window.showFeedbackForm = showFeedbackForm;
window.populateSummary = populateSummary;

// Handle messages from extension
window.addEventListener('message', (event) => {
    const message = event.data;
    if (message?.command === 'showFeedbackForm') {
        showFeedbackForm(message.firstUserQuery, message.selectedDocument || message.firstRankedDocUrl, message.fullAiResponse, message.queryStartTime);
    }
    if (message?.command === 'populateSummary') {
        populateSummary(String(message.summary || ''));
    }
    if (message?.command === 'feedbackSubmitted') {
                const success = message.success;
                const errorMessage = message.error;
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
                    }, 2000);
                } else {
                    // Show error message
                    if (typeof renderSyncError === 'function') renderSyncError(errorMessage || 'Unknown error');
                    
                    // Re-enable submit button
                    settleSubmitState(submitBtn, false, 'Submit');
                }
            }
    if (message?.command === 'documentFound') {
        // Update link field with document URL if found
        const confluenceLinkInput = document.getElementById('confluence-link');
        if (confluenceLinkInput && message.document && message.document.source) {
            confluenceLinkInput.value = message.document.source;
        }
    }
});