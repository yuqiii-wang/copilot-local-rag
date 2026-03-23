
        const docListEl = document.getElementById('doc-list');
        const searchInputEl = document.getElementById('search-input');
        const searchTypeFilterEl = document.getElementById('search-type-filter');
        const syncStatusEl = document.getElementById('sync-status');
        
        function renderSyncStatus(message) {
            const text = String(message || '').trim();
            const isDownloadingFromSource = text.toLowerCase().includes('downloading from');
            if(syncStatusEl) {
                syncStatusEl.textContent = text;
                syncStatusEl.classList.toggle('visible', text.length > 0);
                syncStatusEl.classList.toggle('loading', isDownloadingFromSource);
            }
            const refreshIconBtn = document.getElementById('open-refresh-popup-btn');
            if (refreshIconBtn) {
                if (isDownloadingFromSource) {
                    refreshIconBtn.classList.add('is-spinning');
                    refreshIconBtn.disabled = true;
                    refreshIconBtn.style.opacity = '0.5';
                    refreshIconBtn.style.cursor = 'not-allowed';
                    refreshIconBtn.title = 'Refresh in progress...';
                } else {
                    refreshIconBtn.classList.remove('is-spinning');
                    refreshIconBtn.disabled = false;
                    refreshIconBtn.style.opacity = '1';
                    refreshIconBtn.style.cursor = 'pointer';
                    refreshIconBtn.title = 'Refresh Documents';
                }
            }
        }

        function renderSyncError(message) {
            const errorBanner = document.getElementById('error-banner');
            const errorBannerText = document.getElementById('error-banner-text');
            const text = String(message || '').trim();
            if (errorBanner && errorBannerText) {
                if (text.length > 0) {
                    errorBannerText.textContent = `Error: ${text}`;
                    errorBanner.style.display = 'flex';
                    const successBanner = document.getElementById('success-banner');
                    if(successBanner) successBanner.style.display = 'none';
                } else {
                    errorBanner.style.display = 'none';
                }
            }
        }

        function renderSuccessMessage(message) {
            const successBanner = document.getElementById('success-banner');
            const successBannerText = document.getElementById('success-banner-text');
            const text = String(message || '').trim();
            if (successBanner && successBannerText) {
                if (text.length > 0) {
                    successBannerText.textContent = `${text}`;
                    successBanner.style.display = 'flex';
                    const errorBanner = document.getElementById('error-banner');
                    if(errorBanner) errorBanner.style.display = 'none';
                } else {
                    successBanner.style.display = 'none';
                }
            }
        }

        const closeErrorBtn = document.getElementById('close-error-banner-btn');
        if (closeErrorBtn) closeErrorBtn.addEventListener('click', () => {
            renderSyncError('');
            vscode.postMessage({ command: 'clearSyncError' });
        });

        const closeSuccessBtn = document.getElementById('close-success-banner-btn');
        if(closeSuccessBtn) closeSuccessBtn.addEventListener('click', () => {
            renderSuccessMessage('');
            vscode.postMessage({ command: 'clearSyncSuccess' });
        });

        function sortDisplayedDocs() {
            displayedDocs = displayedDocs.sort((a, b) => String(b.last_updated || '').localeCompare(String(a.last_updated || '')));
        }

        function render() {
            if (!docListEl) return;
            if (!Array.isArray(displayedDocs) || displayedDocs.length === 0) {
                docListEl.innerHTML = '<p class="empty">No local documents yet. Use the sync button to download from Confluence Cloud.</p>';
                return;
            }

            docListEl.innerHTML = displayedDocs.map(doc => {
                const title = escapeHtml(doc.title || 'Untitled');
                const isActive = String(doc.id) === String(selectedDocId) ? ' active' : '';
                return `
                    <div class="doc-item${isActive}" data-doc-id="${escapeHtml(doc.id)}">
                        <button class="doc-open-btn" data-doc-id="${escapeHtml(doc.id)}" title="Open ${title}">
                            <span class="doc-title">${title}</span>
                        </button>
                        <button class="doc-delete-btn" data-doc-id="${escapeHtml(doc.id)}" title="Delete ${title}" aria-label="Delete ${title}">×</button>
                    </div>
                `;
            }).join('');

            document.querySelectorAll('.doc-open-btn').forEach((buttonEl) => {
                buttonEl.addEventListener('click', () => {
                    const docId = buttonEl.getAttribute('data-doc-id');
                    selectedDocId = docId;
                    if(addToPromptsBtnEl) addToPromptsBtnEl.disabled = !selectedDocId;
                    if(addToSkillsBtnEl) addToSkillsBtnEl.disabled = !selectedDocId;
                    render();
                    vscode.postMessage({ command: 'openDoc', docId });
                });
            });

            document.querySelectorAll('.doc-delete-btn').forEach((buttonEl) => {
                buttonEl.addEventListener('click', () => {
                    const docId = buttonEl.getAttribute('data-doc-id');
                    const title = (displayedDocs.find(doc => String(doc.id) === String(docId))?.title || 'this document');
                    vscode.postMessage({ command: 'deleteDoc', docId, title });
                });
            });
        }

        function triggerSearch() {
            vscode.postMessage({
                command: 'searchDocs',
                query: searchInputEl ? searchInputEl.value || '' : '',
                type: searchTypeFilterEl ? searchTypeFilterEl.value || '' : ''
            });
        }

        if (searchInputEl) {
            searchInputEl.addEventListener('input', triggerSearch);
        }
        
        if (searchTypeFilterEl) {
            searchTypeFilterEl.addEventListener('change', triggerSearch);
        }

        window.addEventListener('message', event => {
            const message = event.data;

            if (message?.command === 'searchResults') {
                displayedDocs = Array.isArray(message.payload) ? message.payload : [];
                render();
            }

            if (message?.command === 'docUpserted') {
                const incoming = message.payload || {};
                const docId = String(incoming.id || '').trim();
                if (docId.length > 0) {
                    const existingIndex = displayedDocs.findIndex(doc => String(doc.id) === docId);
                    const nextDoc = {
                        id: incoming.id,
                        title: incoming.title || 'Untitled',
                        last_updated: incoming.last_updated || ''
                    };

                    if (existingIndex >= 0) {
                        displayedDocs[existingIndex] = {
                            ...displayedDocs[existingIndex],
                            ...nextDoc
                        };
                    } else {
                        displayedDocs.push(nextDoc);
                    }
                    if(typeof sortDisplayedDocs === 'function') sortDisplayedDocs();
                    render();
                }
            }

            if (message?.command === 'docDetails') {
                const payload = message.payload || {};
                if(docContentEl) docContentEl.innerHTML = payload.contentHtml || escapeHtml(payload.content || 'No local markdown content found.');
                if(typeof renderMetadata === 'function') renderMetadata(payload.metadata || null);
            }

            if (message?.command === 'selectDoc') {
                const payload = message.payload || {};
                const incomingId = String(payload.id || '').trim();
                if (!incomingId) return;
                selectedDocId = incomingId;
                if(addToPromptsBtnEl) addToPromptsBtnEl.disabled = !selectedDocId;
                if(addToSkillsBtnEl) addToSkillsBtnEl.disabled = !selectedDocId;
                if(docContentEl) docContentEl.innerHTML = payload.contentHtml || escapeHtml(payload.content || 'No local markdown content found.');
                if(typeof renderMetadata === 'function') renderMetadata(payload.metadata || null);
                render();
            }

            if (message?.command === 'syncStatus') renderSyncStatus(message.payload || '');
            if (message?.command === 'syncError') renderSyncError(message.payload || '');
            if (message?.command === 'syncSuccess') renderSuccessMessage(message.payload || '');
            if (message?.command === 'addToPromptsSuccess') renderSuccessMessage(`Added to prompts: ${message.payload}`);
            if (message?.command === 'addToPromptsError') renderSyncError(`${message.payload}`);
            if (message?.command === 'addToSkillsSuccess') renderSuccessMessage(`Added to skills: ${message.payload}`);
            if (message?.command === 'addToSkillsError') renderSyncError(`${message.payload}`);

            if (message?.command === 'docDeleted') {
                const deletedId = String(message.payload?.id || '').trim();
                if (deletedId) {
                    displayedDocs = displayedDocs.filter(doc => String(doc.id) !== deletedId);
                    if (String(selectedDocId) === deletedId) {
                        selectedDocId = null;
                        if(typeof renderMetadata === 'function') {
                            selectedMetadata = null;
                            setMetadataGeneratingState(false);
                            setMetadataEditMode(false);
                            renderMetadata(null);
                        }
                        if(addToPromptsBtnEl) addToPromptsBtnEl.disabled = true;
                        if(addToSkillsBtnEl) addToSkillsBtnEl.disabled = true;
                        if(docContentEl) docContentEl.textContent = 'Select a document to view local content.';
                    }
                    render();
                }
            }
        });
