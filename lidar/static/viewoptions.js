// viewoptions.js

function collectViewOptions() {
  return {
    colorMode: document.querySelector('input[name="grp1"]:checked')?.value,
    pointSize: document.getElementById('ptSize')?.value,
    fov: document.getElementById('fovVal')?.value,
    budget: document.getElementById('budVal')?.value,
    drawAll: document.getElementById('rdAll')?.checked,
    ruler: document.getElementById('ruler')?.checked,
    backgroundColor: document.getElementById('bkcolor')?.value
  };
}

function applyViewOptions(opts) {
  if (!opts) return;
  const emit = (el) => { try { OnUIEvent1(el); } catch (e) {} };

  if (opts.colorMode != null) {
    const colorRadio = document.querySelector(`input[name="grp1"][value="${opts.colorMode}"]`);
    if (colorRadio) { colorRadio.checked = true; emit(colorRadio); }
  }
  const fovEl = document.getElementById('fovVal');
  if (fovEl && opts.fov != null) { fovEl.value = opts.fov; emit(fovEl); }

  const ptEl = document.getElementById('ptSize');
  if (ptEl && opts.pointSize != null) { ptEl.value = opts.pointSize; emit(ptEl); }

  const budEl = document.getElementById('budVal');
  if (budEl && opts.budget != null) { budEl.value = opts.budget; emit(budEl); }

  const rdAllEl = document.getElementById('rdAll');
  if (rdAllEl && typeof opts.drawAll === 'boolean') { rdAllEl.checked = opts.drawAll; emit(rdAllEl); }

  const rulerEl = document.getElementById('ruler');
  if (rulerEl && typeof opts.ruler === 'boolean') { rulerEl.checked = opts.ruler; emit(rulerEl); }

  const bkEl = document.getElementById('bkcolor');
  if (bkEl && opts.backgroundColor) { bkEl.value = opts.backgroundColor; emit(bkEl); }

  if (typeof UpdateColorModeUI === 'function') UpdateColorModeUI();
}

function postViewOptions() {
  const opts = collectViewOptions();
  fetch('/view-options', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(opts)
  }).catch(() => {});
}

// Wire UI changes to backend
(function wireUIToBackend() {
  const ids = ['ptSize','fovVal','budVal','rdAll','ruler','bkcolor'];
  ids.forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener('change', postViewOptions);
    el.addEventListener('input', postViewOptions);
  });
  document.querySelectorAll('input[name="grp1"]').forEach(r => {
    r.addEventListener('change', postViewOptions);
  });
})();

// Decorate loadFile to reapply view options after reload
(function decorateLoadFile() {
  if (typeof loadFile !== 'function') return;
  const originalLoadFile = loadFile;

  window.loadFile = function(filePath) {
    originalLoadFile(filePath);
    const applyFromServer = () => {
      fetch('/view-options')
        .then(r => r.json())
        .then(applyViewOptions)
        .catch(() => {});
    };
    setTimeout(applyFromServer, 100);
    setTimeout(applyFromServer, 400);
    setTimeout(applyFromServer, 1200);
  };
})();
