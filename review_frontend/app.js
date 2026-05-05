let pages = [];
let currentIndex = -1;
let currentPageData = null;

async function init() {
  const resp = await fetch('/api/pages');
  pages = await resp.json();

  document.getElementById('page-count').textContent = pages.length;
  updateStats();
  renderSidebar();
}

function updateStats() {
  const approved = pages.filter(p => p.status === 'approved').length;
  const flagged  = pages.filter(p => p.status === 'flagged').length;
  const unseen   = pages.filter(p => p.status === 'unseen').length;
  document.getElementById('count-approved').textContent = `${approved} approved`;
  document.getElementById('count-flagged').textContent  = `${flagged} flagged`;
  document.getElementById('count-unseen').textContent   = `${unseen} unseen`;
}

function renderSidebar() {
  const list = document.getElementById('page-list');
  list.innerHTML = '';
  pages.forEach((page, idx) => {
    const item = document.createElement('div');
    item.className = `page-item ${page.status}${idx === currentIndex ? ' active' : ''}`;
    item.dataset.index = idx;
    item.onclick = () => loadPage(idx);
    item.innerHTML = `
      <img class="page-thumb" src="/image/thumbnail/${page.name}" alt="">
      <div>
        <div class="page-name">${page.name}</div>
        <div class="page-status status-${page.status}">${statusLabel(page.status)}</div>
      </div>
    `;
    list.appendChild(item);
  });
}

function statusLabel(status) {
  return status === 'approved' ? '✓ approved' : status === 'flagged' ? '⚑ flagged' : '— unseen';
}

async function loadPage(idx) {
  currentIndex = idx;
  const page = pages[idx];

  const resp = await fetch(`/api/page/${page.name}`);
  currentPageData = await resp.json();

  // Show panel
  document.getElementById('stages').style.display = '';
  document.getElementById('empty-state').style.display = 'none';
  document.getElementById('bottom-bar').style.display = '';

  // Header
  document.getElementById('panel-title').textContent = page.name;

  const bubbleCount = (currentPageData.boxes || []).length;
  const bubbleBadge = document.getElementById('bubble-badge');
  bubbleBadge.textContent = `${bubbleCount} bubble${bubbleCount !== 1 ? 's' : ''} detected`;
  bubbleBadge.style.display = '';

  const statusBadge = document.getElementById('status-badge');
  const st = currentPageData.review.status;
  statusBadge.textContent = statusLabel(st);
  statusBadge.className = `badge badge-${st === 'unseen' ? '' : st}`;
  statusBadge.style.display = '';

  // Images (cache-bust with timestamp so detection re-renders)
  const ts = Date.now();
  document.getElementById('img-detection').src = `/image/detection/${page.name}?t=${ts}`;
  document.getElementById('img-cleaned').src   = `/image/cleaned/${page.name}?t=${ts}`;
  document.getElementById('img-output').src    = `/image/output/${page.name}?t=${ts}`;

  // Detection meta
  const fixed = (currentPageData.boxes || []).filter(b => b.type === 'fixed').length;
  const free  = (currentPageData.boxes || []).filter(b => b.type === 'free').length;
  document.getElementById('meta-detection').textContent =
    `${bubbleCount} boxes — ${fixed} FIXED, ${free} FREE`;

  // Cleaning meta
  document.getElementById('meta-cleaning').textContent =
    currentPageData.translated ? 'Inpainting complete' : 'Not yet cleaned';

  // Translation meta
  const translations = currentPageData.translations || [];
  let metaHtml = translations.length
    ? `${translations.length} bubble${translations.length !== 1 ? 's' : ''} translated`
    : 'Not yet translated';
  if (translations.length) {
    metaHtml += '<table class="translations-table">';
    translations.forEach(t => {
      metaHtml += `<tr><td class="orig">${escHtml(t.original)}</td><td class="trans">${escHtml(t.translated)}</td></tr>`;
    });
    metaHtml += '</table>';
  }
  document.getElementById('meta-translation').innerHTML = metaHtml;

  // Notes
  document.getElementById('notes').value = currentPageData.review.notes || '';

  // Nav buttons
  document.getElementById('btn-prev').disabled = idx === 0;
  document.getElementById('btn-next').disabled = idx === pages.length - 1;

  renderSidebar();
}

function navigate(delta) {
  const next = currentIndex + delta;
  if (next >= 0 && next < pages.length) loadPage(next);
}

async function saveReview(status) {
  const page = pages[currentIndex];
  const notes = document.getElementById('notes').value;
  await fetch(`/api/review/${page.name}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ status, notes }),
  });
  pages[currentIndex].status = status;
  document.getElementById('status-badge').textContent = statusLabel(status);
  document.getElementById('status-badge').className = `badge badge-${status}`;
  updateStats();
  renderSidebar();
}

async function exportLog() {
  const resp = await fetch('/api/export');
  const data = await resp.json();
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'review_log.json';
  a.click();
  URL.revokeObjectURL(url);
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

init();
