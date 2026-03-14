
        const metadataListEl = document.getElementById('metadata-list');
        const generateMetadataBtnEl = document.getElementById('generate-metadata-btn');
        const cancelMetadataBtnEl = document.getElementById('cancel-metadata-btn');
        const metadataEditToggleBtnEl = document.getElementById('metadata-edit-toggle-btn');
        const summaryInputEl = document.getElementById('summary-input');
        const keywordsInputEl = document.getElementById('keywords-input');
        const tagsInputEl = document.getElementById('tags-input');
        const feedbackInputEl = document.getElementById('feedback-input');
        const typeInputEl = document.getElementById('type-input');
        const metadataPaneEl = document.querySelector('.metadata-pane');
        
        let selectedMetadata = null;
        let isMetadataEditMode = false;
        let isMetadataGenerating = false;

        function toMetadataItems(metadata) {
            if (!metadata || typeof metadata !== 'object') {
                return [];
            }
            return Object.entries(metadata)
                .filter(([key]) => key !== 'summary' && key !== 'keywords' && key !== 'tags' && key !== 'feedback' && key !== 'extended_keywords' && key !== 'type')
                .map(([key, value]) => {
                    if (Array.isArray(value)) return { key, value: value.join(', ') };
                    if (value && typeof value === 'object') return { key, value: JSON.stringify(value) };
                    return { key, value: String(value ?? '') };
                });
        }

        function setMetadataEditMode(isEditing) {
            isMetadataEditMode = Boolean(isEditing);
            const canEditDoc = Boolean(selectedDocId);
            const inputsEnabled = canEditDoc && isMetadataEditMode && !isMetadataGenerating;
            summaryInputEl.disabled = !inputsEnabled;
            keywordsInputEl.disabled = !inputsEnabled;
            if (typeof tagsInputEl !== 'undefined' && tagsInputEl) tagsInputEl.disabled = !inputsEnabled;
            if (typeof feedbackInputEl !== 'undefined' && feedbackInputEl) feedbackInputEl.disabled = !inputsEnabled;
            if (typeof typeInputEl !== 'undefined' && typeInputEl) typeInputEl.disabled = !inputsEnabled;
            metadataEditToggleBtnEl.disabled = !canEditDoc || isMetadataGenerating;
            metadataEditToggleBtnEl.textContent = isMetadataEditMode ? 'Save' : 'Edit';
            
            if (isMetadataEditMode) {
                if (generateMetadataBtnEl) generateMetadataBtnEl.style.display = 'none';
                if (cancelMetadataBtnEl) cancelMetadataBtnEl.style.display = 'inline-block';
            } else {
                if (generateMetadataBtnEl) generateMetadataBtnEl.style.display = 'inline-block';
                if (cancelMetadataBtnEl) cancelMetadataBtnEl.style.display = 'none';
            }
        }

        function setMetadataGeneratingState(isGenerating) {
            isMetadataGenerating = Boolean(isGenerating);
            if(metadataPaneEl) metadataPaneEl.classList.toggle('is-busy', isMetadataGenerating);
            if(generateMetadataBtnEl) generateMetadataBtnEl.classList.toggle('is-loading', isMetadataGenerating);
            const hasDoc = Boolean(selectedDocId);
            if(generateMetadataBtnEl) generateMetadataBtnEl.disabled = !hasDoc || isMetadataGenerating;
            if(metadataEditToggleBtnEl) metadataEditToggleBtnEl.disabled = !hasDoc || isMetadataGenerating;
            if(summaryInputEl) summaryInputEl.disabled = !hasDoc || !isMetadataEditMode || isMetadataGenerating;
            if(keywordsInputEl) keywordsInputEl.disabled = !hasDoc || !isMetadataEditMode || isMetadataGenerating;
            if (typeof tagsInputEl !== 'undefined' && tagsInputEl) tagsInputEl.disabled = !hasDoc || !isMetadataEditMode || isMetadataGenerating;
            if (typeof feedbackInputEl !== 'undefined' && feedbackInputEl) feedbackInputEl.disabled = !hasDoc || !isMetadataEditMode || isMetadataGenerating;
            if (typeof typeInputEl !== 'undefined' && typeInputEl) typeInputEl.disabled = !hasDoc || !isMetadataEditMode || isMetadataGenerating;
        }

        function renderMetadata(metadata) {
            selectedMetadata = metadata && typeof metadata === 'object' ? metadata : null;
            const items = toMetadataItems(selectedMetadata);
            const canEdit = Boolean(selectedDocId);
            if(generateMetadataBtnEl) generateMetadataBtnEl.disabled = !canEdit || isMetadataGenerating;
            setMetadataEditMode(false);

            if(summaryInputEl) summaryInputEl.value = selectedMetadata ? String(selectedMetadata.summary || '') : '';
            if (typeof typeInputEl !== 'undefined' && typeInputEl) typeInputEl.value = selectedMetadata ? String(selectedMetadata.type || 'custom') : 'custom';
            const keywordValues = selectedMetadata && Array.isArray(selectedMetadata.keywords) ? selectedMetadata.keywords : [];
            if(keywordsInputEl) keywordsInputEl.value = keywordValues.join(', ');
            const tagValues = selectedMetadata && Array.isArray(selectedMetadata.tags) ? selectedMetadata.tags : [];
            if(tagsInputEl) tagsInputEl.value = tagValues.join(', ');
            if(feedbackInputEl) feedbackInputEl.value = selectedMetadata ? String(selectedMetadata.feedback || '') : '';

            if (metadataListEl) {
                if (items.length === 0) {
                    metadataListEl.innerHTML = '<li class="metadata-item">title: -</li>';
                } else {
                    metadataListEl.innerHTML = items.map(item => `
                        <li class="metadata-item">${escapeHtml(item.key)}: ${escapeHtml(item.value)}</li>
                    `).join('');
                }
            }
        }
        
        if (generateMetadataBtnEl) {
            generateMetadataBtnEl.addEventListener('click', () => {
                if (!selectedDocId || isMetadataGenerating) return;
                setMetadataGeneratingState(true);
                vscode.postMessage({ command: 'generateMetadata', docId: selectedDocId });
            });
        }

        if (cancelMetadataBtnEl) {
            cancelMetadataBtnEl.addEventListener('click', () => {
                if (!selectedDocId || isMetadataGenerating) return;
                renderMetadata(selectedMetadata);
            });
        }

        if (metadataEditToggleBtnEl) {
            metadataEditToggleBtnEl.addEventListener('click', () => {
                if (!selectedDocId || isMetadataGenerating) return;
                if (!isMetadataEditMode) {
                    setMetadataEditMode(true);
                    if(summaryInputEl) summaryInputEl.focus();
                    return;
                }
                const keywords = String(keywordsInputEl ? keywordsInputEl.value : '')
                    .split(',').map(k => k.trim()).filter(k => k.length > 0).join(', ');
                const tags = String(tagsInputEl ? tagsInputEl.value : '')
                    .split(',').map(t => t.trim()).filter(t => t.length > 0).join(', ');
                vscode.postMessage({
                    command: 'saveMetadata',
                    docId: selectedDocId,
                    type: typeof typeInputEl !== 'undefined' && typeInputEl ? typeInputEl.value : 'custom',
                    summary: summaryInputEl ? summaryInputEl.value : '',
                    keywords,
                    tags,
                    feedback: feedbackInputEl ? feedbackInputEl.value : ''
                });
                setMetadataEditMode(false);
            });
        }

        window.addEventListener('message', event => {
            const message = event.data;
            if (message?.command === 'metadataUpdated') {
                const payload = message.payload || {};
                const updatedDocId = String(payload.id || '').trim();
                const metadata = payload.metadata || null;
                if (updatedDocId && String(selectedDocId) === updatedDocId) {
                    renderMetadata(metadata);
                }
            }
            if (message?.command === 'metadataGenerationState') {
                const payload = message.payload || {};
                const messageDocId = String(payload.docId || '').trim();
                if (messageDocId && String(selectedDocId) === messageDocId) {
                    setMetadataGeneratingState(Boolean(payload.isGenerating));
                }
            }
        });
