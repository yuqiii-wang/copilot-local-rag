let recordsCache = [];
let page = 1;
let pageSize = 10;

async function fetchRecords() {
  const res = await fetch('/admin/records');
  const data = await res.json();
  recordsCache = data.records || [];
  // Sort records by Last Query Time descending (most recent first)
  recordsCache.sort((a, b) => {
    const ta = a && a.query && a.query.timestamp ? new Date(a.query.timestamp) : null;
    const tb = b && b.query && b.query.timestamp ? new Date(b.query.timestamp) : null;
    if (!ta && !tb) return 0;
    if (!ta) return 1; // put records without timestamps last
    if (!tb) return -1;
    return tb - ta; // descending
  });
  page = 1; renderPage();
}

function renderPage() {
  const container = document.getElementById('records');
  container.innerHTML = '';
  pageSize = parseInt(document.getElementById('pageSize').value, 10);
  const total = recordsCache.length;
  const totalPages = Math.max(1, Math.ceil(total / pageSize));
  if (page > totalPages) page = totalPages;
  const start = (page - 1) * pageSize;
  const end = Math.min(total, start + pageSize);

  const pageInfo = document.getElementById('pageInfo');
  if (pageInfo) pageInfo.textContent = `Page ${page} / ${totalPages} (${total} records)`;

  for (let i = start; i < end; i++) {
    const rec = recordsCache[i];
    const div = document.createElement('div'); div.className='record';
    const q = rec.query || {};
    const id = q.id || '(no id)';
    const heading = document.createElement('div'); heading.className='record-heading';
    const timeStr = q.timestamp ? new Date(q.timestamp).toLocaleString() : '';
    heading.innerHTML = `<div class='record-meta'><strong>Question:</strong> ${q.question||''}</div><div class='record-meta'><strong>Status:</strong> ${q.status||''}</div><div class='record-meta'><strong>Query Time:</strong> ${timeStr}</div>`;
    div.appendChild(heading);

    // Files summary (compact chips with delete)
    const filesSummary = document.createElement('div'); filesSummary.className='files-summary';
    if (q.ref_docs && q.ref_docs.length) {
      for (const rdSum of q.ref_docs) {
        let rName = (rdSum.source||'').replace(/\\/g, '/').split('/').pop() || rdSum.source || '';
        // Remove extension if present
        if (rName.lastIndexOf('.') > 0) {
             rName = rName.substring(0, rName.lastIndexOf('.'));
        }
        const fname = rName;
        
        const chip = document.createElement('span'); chip.className='file-chip';
        // clickable file name
        const nameSpan = document.createElement('span'); nameSpan.className='file-name'; nameSpan.textContent = fname;
        nameSpan.title = 'Click to edit comment/keywords';
        nameSpan.onclick = (ev) => {
          ev.stopPropagation();
          openEditModal(q.id, rdSum.source, rdSum.comment||'', rdSum.keywords || []);
        };
        chip.appendChild(nameSpan);

        // small grey close button
        const delSmall = document.createElement('button'); delSmall.textContent='✖'; delSmall.className='chip-close';
        delSmall.onclick = async (ev) => {
          ev.stopPropagation();
          if (!confirm('Delete this reference document from the record?')) return;
          const payload = { query: { id: q.id, ref_docs: [{ source: rdSum.source }] } };
          const r = await fetch('/admin/delete_doc', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
          if (r.ok) {
            alert('Deleted');
            const idx = q.ref_docs.findIndex(rditem => rditem.source === rdSum.source);
            if (idx >= 0) q.ref_docs.splice(idx,1);
            renderPage();
          } else alert('Failed to delete');
        };
        chip.appendChild(delSmall);
        filesSummary.appendChild(chip);
      }
    }
    div.appendChild(filesSummary);

    const docsDiv = document.createElement('div'); docsDiv.className='docs-wrap';
    if (q.ref_docs && q.ref_docs.length) {
      for (const rd of q.ref_docs) {
        const rdiv = document.createElement('div'); rdiv.className='refdoc';
        const src = rd.source||'';
        const comment = rd.comment||'';
        const keywords = (rd.keywords || []).join(', ');

        const srcLine = document.createElement('div'); srcLine.innerHTML = `<strong>Source:</strong> ${src.replace(/\\/g, '/')}`;
        const ta = document.createElement('textarea'); ta.value = comment; ta.placeholder='comment';
        const kw = document.createElement('input'); kw.value = keywords; kw.placeholder='comma-separated keywords'; kw.className='kw-input';

        const btnWrap = document.createElement('div'); btnWrap.className='btn-wrap';
        const saveBtn = document.createElement('button'); saveBtn.textContent='Save'; saveBtn.className='btn small primary';
        const viewBtn = document.createElement('button'); viewBtn.textContent='View'; viewBtn.className='btn small';
        const delDocBtn = document.createElement('button'); delDocBtn.textContent='Delete Doc'; delDocBtn.className='btn small danger';

        delDocBtn.onclick = async () => {
          if (!confirm('Delete this reference document from the record?')) return;
          const payload = { query: { id: q.id, ref_docs: [{ source: src }] } };
          const r = await fetch('/admin/delete_doc', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
          if (r.ok) {
            alert('Deleted');
            // remove from local cache and re-render
            const idx = q.ref_docs.findIndex(rditem => rditem.source === src);
            if (idx >= 0) q.ref_docs.splice(idx,1);
            renderPage();
          } else alert('Failed to delete');
        };

        saveBtn.onclick = async () => {
          const payload = { query: { id: q.id, ref_docs: [{ source: src, comment: ta.value, keywords: kw.value.split(',').map(s=>s.trim()).filter(Boolean) }] } };
          const r = await fetch('/admin/update_doc', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
          if (r.ok) {
            alert('Saved');
            // update local cache
            rd.comment = ta.value; rd.keywords = kw.value.split(',').map(s=>s.trim()).filter(Boolean);
          } else alert('Failed');
        };

        viewBtn.onclick = async () => {
          // fetch content via download API
          try {
            const body = { urls: [ src ] };
            const res = await fetch('/download/fetch', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body) });
            const data = await res.json();
            const d = Array.isArray(data) ? data[0] : (data.downloads && data.downloads[0]) || {};
            const content = d && d.content ? d.content : 'No content available';
            showModal(src, content);
          } catch (e) {
            showModal(src, 'Failed to fetch document: ' + e.message);
          }
        };

        btnWrap.appendChild(viewBtn); btnWrap.appendChild(saveBtn); btnWrap.appendChild(delDocBtn);
        rdiv.appendChild(srcLine);
        rdiv.appendChild(ta);
        rdiv.appendChild(kw);
        rdiv.appendChild(btnWrap);
        docsDiv.appendChild(rdiv);
      }
    } else {
      docsDiv.textContent = 'No ref_docs';
    }

    // Query-level chat and delete
    const showChatBtn = document.createElement('button'); showChatBtn.textContent = 'Show Chat'; showChatBtn.className='btn small';
    showChatBtn.onclick = (ev) => { ev.stopPropagation(); openChatModal(q.id, q.conversations || []); };

    const qdel = document.createElement('button'); qdel.textContent = 'Delete Query'; qdel.className='btn small danger';
    qdel.onclick = async () => {
      if (!confirm('Delete entire query record? This is irreversible.')) return;
      const r = await fetch('/admin/delete_query', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({ id: q.id }) });
      if (r.ok) {
        alert('Query deleted');
        // remove from cache and re-render
        const recIdx = recordsCache.findIndex(rc=> rc.query && rc.query.id === q.id);
        if (recIdx >=0) { recordsCache.splice(recIdx,1); renderPage(); }
      } else alert('Failed to delete query');
    };
    const foot = document.createElement('div'); foot.className='record-footer'; foot.appendChild(showChatBtn); foot.appendChild(qdel);
    div.appendChild(foot);

    container.appendChild(div);
  }
}

document.getElementById('prevPage').onclick = () => { if (page>1) { page--; renderPage(); } };
document.getElementById('nextPage').onclick = () => { page++; renderPage(); };
document.getElementById('pageSize').onchange = () => { page = 1; renderPage(); };

async function setWorking(working) {
  const trainBtn = document.getElementById('train');
  const visBtn = document.getElementById('visualize');
  const refreshBtn = document.getElementById('refresh');
  if (working) {
    trainBtn.disabled = true; visBtn.disabled = true; refreshBtn.disabled = true;
  } else {
    trainBtn.disabled = false; visBtn.disabled = false; refreshBtn.disabled = false;
  }
} 

// Helper: show a small spinner on the clicked button, remove button boundary, and store original text/style
function showButtonLoading(btn) {
  if (!btn) return;
  if (!btn.dataset.origText) btn.dataset.origText = btn.innerHTML;
  if (!btn.dataset.origStyle) btn.dataset.origStyle = btn.getAttribute('style') || '';
  btn.innerHTML = '⏳';
  // remove button visual boundary while spinner shown
  btn.style.border = 'none';
  btn.style.background = 'transparent';
  btn.style.padding = '6px 10px';
  btn.style.cursor = 'default';
  btn.disabled = true;
}

// Helper: restore button text, style, and enabled state
function hideButtonLoading(btn) {
  if (!btn) return;
  if (btn.dataset.origText) {
    btn.innerHTML = btn.dataset.origText;
    delete btn.dataset.origText;
  }
  // restore original inline style if any
  if (btn.dataset.origStyle !== undefined) {
    if (btn.dataset.origStyle) btn.setAttribute('style', btn.dataset.origStyle);
    else btn.removeAttribute('style');
    delete btn.dataset.origStyle;
  }
  btn.disabled = false;
}

let trainEventSource = null;

function startTrainStream() {
  if (trainEventSource) return;
  const logsEl = document.getElementById('trainLogs');
  logsEl.textContent = '';
  // hide image and show logs
  document.getElementById('graph').style.display = 'none';
  logsEl.style.display = 'block';

  trainEventSource = new EventSource('/admin/train_logs');
  trainEventSource.onmessage = function(e) {
    const data = e.data || '';
    if (data === '[TRAIN_DONE]' || data === '[END]') {
      closeTrainStream();
      return;
    }
    logsEl.textContent += data + '\n';
    logsEl.scrollTop = logsEl.scrollHeight;
  };
  trainEventSource.onerror = function(err) {
    console.error('Train stream error', err);
    closeTrainStream();
  };
}

function closeTrainStream() {
  if (trainEventSource) {
    trainEventSource.close();
    trainEventSource = null;
  }
  // hide logs and show image
  document.getElementById('trainLogs').style.display = 'none';
  document.getElementById('graph').style.display = 'block';
  // ensure latest image
  document.getElementById('graph').src = '/admin/graph.png?ts='+Date.now();
  setWorking(false);
}

async function pollStatus() {
  try {
    const s = await fetch('/admin/status');
    const st = await s.json();
    const busy = st.training || st.visualize;
    await setWorking(busy);
    if (st.training) {
      startTrainStream();
      setTimeout(pollStatus, 2000);
      return;
    }
    if (busy) {
      setTimeout(pollStatus, 2000);
    } else {
      // finished
      closeTrainStream();
      document.getElementById('graph').src = '/admin/graph.png?ts='+Date.now();
      fetchRecords();
      // Ensure any button-level spinners are cleared
      hideButtonLoading(document.getElementById('train'));
      hideButtonLoading(document.getElementById('visualize'));
    }
  } catch (e) { console.error('Status poll failed', e); setWorking(false); }
}

// initial status poll
pollStatus();

async function fetchAndRenderGrid(range) {
  try {
    const res = await fetch(`/admin/stats?range=${encodeURIComponent(range)}`);
    if (!res.ok) { alert('Failed to fetch stats'); return; }
    const data = await res.json();
    // Ensure recordsCache is loaded
    if (!recordsCache || recordsCache.length === 0) await fetchRecords();

    // Build map of last timestamp per question from recordsCache
    const lastMap = {};
    for (const rec of recordsCache) {
      const qtext = (rec.query || {}).question || '';
      const ts = (rec.query || {}).timestamp || '';
      if (!qtext || !ts) continue;
      try {
        if (!lastMap[qtext] || new Date(ts) > new Date(lastMap[qtext])) lastMap[qtext] = ts;
      } catch (e) { /* ignore bad dates */ }
    }

    const tbody = document.querySelector('#usageGrid tbody');
    tbody.innerHTML = '';
    const list = data.top_questions || [];
    // Sort by time descending
    list.sort((a,b) => {
      const ta = lastMap[a.question] ? new Date(lastMap[a.question]) : 0;
      const tb = lastMap[b.question] ? new Date(lastMap[b.question]) : 0;
      return tb - ta;
    });

    if (list.length === 0) {
      tbody.innerHTML = '<tr><td colspan="3">No data</td></tr>';
    } else {
      for (const item of list) {
        const qtext = item.question;
        const cnt = item.count;
        const rawLast = lastMap[qtext];
        const last = rawLast ? new Date(rawLast).toLocaleString() : '';
        const tr = document.createElement('tr');
        tr.innerHTML = `<td style="padding:6px; border-bottom:1px solid #eee">${qtext}</td>` +
                       `<td style="padding:6px; border-bottom:1px solid #eee; text-align:right">${cnt}</td>` +
                       `<td style="padding:6px; border-bottom:1px solid #eee; text-align:right">${last}</td>`;
        tbody.appendChild(tr);
      }
    }
    document.getElementById('gridArea').style.display = 'block';
    // hide logs while grid is visible (keep graph visible)
    document.getElementById('trainLogs').style.display = 'none';
  } catch (e) { console.error('Grid error', e); alert('Failed to render stats: ' + e.message); }
}

document.getElementById('showChart').onclick = async () => {
  const btn = document.getElementById('showChart');
  const grid = document.getElementById('gridArea');
  const recordsEl = document.getElementById('records');
  const logs = document.getElementById('trainLogs');
  const rng = document.getElementById('chartRange').value || '7d';
  const pager = document.querySelector('.pager');

  const prevBtn = document.getElementById('prevPage');
  const nextBtn = document.getElementById('nextPage');
  const pageInfo = document.getElementById('pageInfo');
  const pageSize = document.getElementById('pageSize');

  if (grid.style.display === 'block') {
    // Currently showing stats grid -> show records/files
    grid.style.display = 'none';
    recordsEl.style.display = 'block';
    // Restore pagination controls (keep Show Stats button visible)
    prevBtn.style.display = 'inline-block';
    nextBtn.style.display = 'inline-block';
    pageInfo.style.display = 'inline-block';
    pageSize.style.display = 'inline-block';
    // keep graph visibility unchanged
    logs.style.display = 'none';
    btn.textContent = 'Show Stats';
  } else {
    // Show stats grid (fetch if necessary)
    await fetchAndRenderGrid(rng);
    recordsEl.style.display = 'none';
    // Hide only pagination controls but keep Show Stats button and range selector visible
    prevBtn.style.display = 'none';
    nextBtn.style.display = 'none';
    pageInfo.style.display = 'none';
    pageSize.style.display = 'none';
    btn.textContent = 'Show Detail';
    // Keep button styled primary
  }
};

document.getElementById('train').onclick = async () => {
  // Prompt for optional initial learning rate (default shown)
  const lrInput = prompt('Init learning rate (default: 0.001):', '0.001');
  // If the user cancels the prompt, lrInput === null => do NOT start training
  if (lrInput === null) {
    return; // user cancelled
  }
  const lr = lrInput && lrInput.trim() ? lrInput.trim() : null;

  const trainBtn = document.getElementById('train');
  showButtonLoading(trainBtn);
  await setWorking(true);
  // Build query string
  const params = new URLSearchParams();
  if (lr) params.append('init_lr', lr);
  const url = '/admin/train' + (params.toString() ? ('?' + params.toString()) : '');

  try {
    const res = await fetch(url, { method: 'POST' });
    if (res.ok) {
      pollStatus();
      alert('Training started');
    } else { hideButtonLoading(trainBtn); await setWorking(false); alert('Failed to start training'); }
  } catch (e) {
    hideButtonLoading(trainBtn);
    await setWorking(false);
    alert('Failed to start training: ' + e.message);
  }
};


document.getElementById('visualize').onclick = async () => {
  const visBtn = document.getElementById('visualize');
  showButtonLoading(visBtn);
  await setWorking(true);
  try {
    const res = await fetch('/admin/visualize', { method: 'POST' });
    if (res.ok) {
      pollStatus();
      alert('Visualization started');
    } else { hideButtonLoading(visBtn); await setWorking(false); alert('Failed'); }
  } catch (e) {
    hideButtonLoading(visBtn);
    await setWorking(false);
    alert('Failed to start visualization: ' + e.message);
  }
};

function showModal(title, body) {
  document.getElementById('modalTitle').textContent = title.replace(/\\/g, '/');
  document.getElementById('modalBody').textContent = body;
  document.getElementById('viewerModal').style.display = 'flex';
}

document.getElementById('closeModal').onclick = () => { document.getElementById('viewerModal').style.display = 'none'; };

// Edit modal handlers
function openEditModal(qid, source, comment, keywords) {
  document.getElementById('editSource').textContent = source.replace(/\\/g, '/');
  document.getElementById('editComment').value = comment || '';
  document.getElementById('editKeywords').value = (keywords || []).join(', ');
  document.getElementById('editModal').style.display = 'flex';
  // store current context
  document.getElementById('editModal').dataset.qid = qid;
  document.getElementById('editModal').dataset.source = source;
}

document.getElementById('closeEditModal').onclick = () => { document.getElementById('editModal').style.display = 'none'; };

document.getElementById('editSave').onclick = async () => {
  const qid = document.getElementById('editModal').dataset.qid;
  const source = document.getElementById('editModal').dataset.source;
  const comment = document.getElementById('editComment').value;
  const keywords = document.getElementById('editKeywords').value.split(',').map(s=>s.trim()).filter(Boolean);
  const payload = { query: { id: qid, ref_docs: [{ source, comment, keywords }] } };
  const r = await fetch('/admin/update_doc', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
  if (r.ok) {
    alert('Saved');
    // update local cache
    const rec = recordsCache.find(rc => rc.query && rc.query.id === qid);
    if (rec) {
      const rd = rec.query.ref_docs.find(rr => rr.source === source);
      if (rd) { rd.comment = comment; rd.keywords = keywords; }
    }
    renderPage();
    document.getElementById('editModal').style.display = 'none';
  } else alert('Failed to save');
};

document.getElementById('editDelete').onclick = async () => {
  if (!confirm('Delete this reference document from the record?')) return;
  const qid = document.getElementById('editModal').dataset.qid;
  const source = document.getElementById('editModal').dataset.source;
  const payload = { query: { id: qid, ref_docs: [{ source }] } };
  const r = await fetch('/admin/delete_doc', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
  if (r.ok) {
    alert('Deleted');
    const rec = recordsCache.find(rc => rc.query && rc.query.id === qid);
    if (rec) {
      const idx = rec.query.ref_docs.findIndex(rr => rr.source === source);
      if (idx >= 0) rec.query.ref_docs.splice(idx,1);
    }
    renderPage();
    document.getElementById('editModal').style.display = 'none';
  } else alert('Failed to delete');
};

// Chat modal handlers
function openChatModal(qid, conversations) {
  const modal = document.getElementById('chatModal');
  modal.dataset.qid = qid;
  const container = document.getElementById('chatTurns');
  container.innerHTML = '';
  const convs = conversations && conversations.length ? conversations : [];

  if (convs.length === 0) {
    const empty = document.createElement('div'); empty.textContent = 'No conversation turns'; empty.style.color = '#666'; container.appendChild(empty);
  }

  convs.forEach((turn, idx) => {
    const wrap = document.createElement('div'); wrap.style.marginBottom = '8px';
    const aiLabel = document.createElement('div'); aiLabel.textContent = 'Assistant'; aiLabel.style.fontSize = '0.9rem';
    const aiTa = document.createElement('textarea'); aiTa.value = turn.ai_assistant || ''; aiTa.dataset.role = 'ai'; aiTa.style.width = '100%'; aiTa.style.height = '80px';
    const humanLabel = document.createElement('div'); humanLabel.textContent = 'Human'; humanLabel.style.fontSize = '0.9rem';
    const humanTa = document.createElement('textarea'); humanTa.value = turn.human || ''; humanTa.dataset.role = 'human'; humanTa.style.width = '100%'; humanTa.style.height = '80px';
    const del = document.createElement('button'); del.textContent = 'Delete Turn'; del.className = 'btn small danger'; del.onclick = () => { wrap.remove(); };
    wrap.appendChild(aiLabel); wrap.appendChild(aiTa); wrap.appendChild(humanLabel); wrap.appendChild(humanTa); wrap.appendChild(del);
    container.appendChild(wrap);
  });

  document.getElementById('chatModal').style.display = 'flex';
}

document.getElementById('closeChatModal').onclick = () => { document.getElementById('chatModal').style.display = 'none'; };

document.getElementById('addChatTurn').onclick = () => {
  const container = document.getElementById('chatTurns');
  // remove 'No conversation turns' placeholder if present
  const placeholder = container.querySelector('div');
  if (placeholder && placeholder.textContent === 'No conversation turns') container.innerHTML = '';
  const wrap = document.createElement('div'); wrap.style.marginBottom = '8px';
  const aiLabel = document.createElement('div'); aiLabel.textContent = 'Assistant'; aiLabel.style.fontSize = '0.9rem';
  const aiTa = document.createElement('textarea'); aiTa.dataset.role = 'ai'; aiTa.style.width = '100%'; aiTa.style.height = '80px';
  const humanLabel = document.createElement('div'); humanLabel.textContent = 'Human'; humanLabel.style.fontSize = '0.9rem';
  const humanTa = document.createElement('textarea'); humanTa.dataset.role = 'human'; humanTa.style.width = '100%'; humanTa.style.height = '80px';
  const del = document.createElement('button'); del.textContent = 'Delete Turn'; del.className = 'btn small danger'; del.onclick = () => { wrap.remove(); };
  wrap.appendChild(aiLabel); wrap.appendChild(aiTa); wrap.appendChild(humanLabel); wrap.appendChild(humanTa); wrap.appendChild(del);
  container.appendChild(wrap);
};

document.getElementById('saveChat').onclick = async () => {
  const modal = document.getElementById('chatModal');
  const qid = modal.dataset.qid;
  const container = document.getElementById('chatTurns');
  const wraps = Array.from(container.children);
  const convs = [];
  for (const w of wraps) {
    const tas = w.querySelectorAll('textarea');
    if (!tas || tas.length === 0) continue;
    const ai = tas[0].value || '';
    const human = tas[1] ? tas[1].value || '' : '';
    if (ai || human) convs.push({ ai_assistant: ai, human: human });
  }

  try {
    const res = await fetch('/data/record_feedback', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ id: qid, conversations: convs }) });
    if (res.ok) {
      alert('Saved');
      // update local cache
      const rec = recordsCache.find(rc => rc.query && rc.query.id === qid);
      if (rec) rec.query.conversations = convs;
      document.getElementById('chatModal').style.display = 'none';
      renderPage();
    } else {
      alert('Failed to save conversations');
    }
  } catch (e) {
    alert('Error saving conversations: ' + e.message);
  }
};

document.getElementById('refresh').onclick = fetchRecords;

// initial
fetchRecords();

/* Manual Record Logic */
const manualModal = document.getElementById('manualModal');
const manualUrlList = document.getElementById('manualUrlList');

document.getElementById('manualRecord').onclick = () => {
    // Clear and open
    document.getElementById('manualQuery').value = '';
    manualUrlList.innerHTML = '';
    addManualUrlRow('', ''); // Start with one row
    manualModal.style.display = 'flex';
};

document.getElementById('manualCancel').onclick = () => {
    manualModal.style.display = 'none';
};

function addManualUrlRow(url, score) {
    const div = document.createElement('div');
    div.style.display = 'flex';
    div.style.gap = '8px';
    div.style.marginBottom = '6px';
    
    const ui = document.createElement('input');
    ui.placeholder = 'Document URL / Source';
    ui.value = url || '';
    ui.style.flex = '1';

    const si = document.createElement('input');
    si.placeholder = 'Score';
    si.type = 'number';
    si.step = 'any';
    si.value = score !== undefined ? score : '';
    si.style.width = '80px';

    const del = document.createElement('button');
    del.textContent = '✕';
    del.className = 'btn small danger';
    del.onclick = () => div.remove();

    div.appendChild(ui);
    div.appendChild(si);
    div.appendChild(del);
    manualUrlList.appendChild(div);
}

document.getElementById('addUrlBtn').onclick = () => addManualUrlRow();

document.getElementById('manualSubmit').onclick = async () => {
    const q = document.getElementById('manualQuery').value.trim();
    if (!q) { alert('Question is required'); return; }

    const rows = Array.from(manualUrlList.children);
    const docs = [];
    
    // First pass: check if any scores were manually entered
    let hasManualScores = false;
    const entries = [];
    for (const r of rows) {
        const inputs = r.getElementsByTagName('input');
        const url = inputs[0].value.trim();
        const scoreStr = inputs[1].value.trim();
        if (url) {
            if (scoreStr !== '') hasManualScores = true;
            entries.push({ url, scoreStr });
        }
    }

    // Second pass: build document list with logic
    // If NO scores provided, distribute valid documents evenly (1.0 / count)
    // If ANY scores provided, use parsed value (or 0 if missing/invalid)
    const evenWeight = entries.length > 0 ? (1.0 / entries.length) : 0;

    for (const e of entries) {
        let finalScore;
        if (!hasManualScores) {
            finalScore = evenWeight;
        } else {
            const parsed = parseFloat(e.scoreStr);
            finalScore = isNaN(parsed) ? 0 : parsed;
        }
        docs.push({ url: e.url, score: finalScore });
    }
    
    // Validate scores? No, backend normalizes.
    
    const submitBtn = document.getElementById('manualSubmit');
    submitBtn.disabled = true;

    try {
        const res = await fetch('/admin/manual_record', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ question: q, docs: docs })
        });
        if (res.ok) {
            alert('Record added successfully');
            manualModal.style.display = 'none';
            fetchRecords();
        } else {
            const txt = await res.text();
            alert('Failed to add record: ' + txt);
        }
    } catch (e) {
        alert('Error: ' + e.message);
    } finally {
        submitBtn.disabled = false;
    }
};
// Test Query Logic
const testModal = document.getElementById('testQueryModal');
document.getElementById('testQuery').onclick = () => {
    document.getElementById('testQueryResults').innerHTML = '';
    document.getElementById('testQueryInput').value = '';
    testModal.style.display = 'flex';
};
document.getElementById('closeTestQuery').onclick = () => {
    testModal.style.display = 'none';
};
document.getElementById('runTestQuery').onclick = async () => {
    const q = document.getElementById('testQueryInput').value.trim();
    if (!q) return;
    const resDiv = document.getElementById('testQueryResults');
    resDiv.innerHTML = 'Searching...'; 
    try {
        const r = await fetch('/rag/retrieve', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: q, limit: 10 })
        });
        const docs = await r.json();
        resDiv.innerHTML = '';
        if (!docs || docs.length === 0) {
            resDiv.textContent = 'No results found.';
            return;
        }
        docs.forEach(d => {
            const item = document.createElement('div');
            item.style.marginBottom = '8px';
            item.style.padding = '8px';
            item.style.background = '#fff';
            item.style.border = '1px solid #eee';
            
            const head = document.createElement('div');
            head.style.fontWeight = 'bold';
            head.textContent = `${d.title} (Score: ${parseFloat(d.score).toFixed(2)})`;
            
            const ctx = document.createElement('div');
            ctx.style.fontSize = '0.85rem';
            ctx.style.color = '#555';
            ctx.textContent = d.snippet;
            
            const src = document.createElement('div');
            src.style.fontSize = '0.75rem';
            src.style.color = '#888';
            src.textContent = (d.link||'').replace(/\\/g, '/');
            
            item.appendChild(head);
            item.appendChild(ctx);
            item.appendChild(src);
            resDiv.appendChild(item);
        });
    } catch (e) {
        resDiv.textContent = 'Error: ' + e.message;
    }
};