// LungCare AI — main.js

/* ══ SLIDERS ══════════════════════════════════════════════════════════ */
function initSliders() {
  document.querySelectorAll('input[type="range"]').forEach(sl => {
    function update() {
      const pct = ((+sl.value - +sl.min) / (+sl.max - +sl.min)) * 100;
      sl.style.setProperty('--pct', pct + '%');
      const v = sl.closest('.slider-row')?.querySelector('.slider-val');
      if (v) v.textContent = sl.value;
    }
    sl.addEventListener('input', update);
    update();
  });
}

/* ══ MULTI-STEP FORM ══════════════════════════════════════════════════ */
let curStep = 0;
const TOTAL = 5;

function showStep(n) {
  document.querySelectorAll('.form-sec').forEach((s, i) => s.classList.toggle('active', i === n));
  document.querySelectorAll('.step-ind').forEach((ind, i) => {
    ind.classList.remove('active', 'done');
    if (i === n) ind.classList.add('active');
    else if (i < n) ind.classList.add('done');
  });
  const bar = document.querySelector('.prog-bar');
  if (bar) bar.style.width = ((n / (TOTAL - 1)) * 100) + '%';
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

window.nextStep = () => { if (curStep < TOTAL - 1) { curStep++; showStep(curStep); } };
window.prevStep = () => { if (curStep > 0) { curStep--; showStep(curStep); } };

/* ══ COLLECT DATA ═════════════════════════════════════════════════════ */
function collectData() {
  const data = {};
  document.querySelectorAll('[data-field]').forEach(el => {
    const k = el.dataset.field;
    if (el.type === 'range' || el.type === 'number') data[k] = parseFloat(el.value) || 0;
    else if (el.tagName === 'SELECT') data[k] = parseFloat(el.value) || 0;
    else if (el.type === 'radio' && el.checked) data[k] = parseFloat(el.value);
  });
  return data;
}

/* ══ SUBMIT ═══════════════════════════════════════════════════════════ */
window.submitAssessment = async function () {
  const ov = document.getElementById('loadingOv');
  if (ov) ov.classList.add('show');
  try {
    const data = collectData();
    const resp = await fetch('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) });
    const json = await resp.json();
    if (json.success) {
      localStorage.setItem('lungcare_result', JSON.stringify(json.result));
      window.location.href = '/results';
    } else { alert('Prediction failed: ' + json.error); if (ov) ov.classList.remove('show'); }
  } catch (e) { alert('Error: ' + e.message); if (ov) ov.classList.remove('show'); }
};

/* ══ RESULTS PAGE ═════════════════════════════════════════════════════ */
function renderResults() {
  const raw = localStorage.getItem('lungcare_result');
  if (!raw) { window.location.href = '/assessment'; return; }
  const r = JSON.parse(raw);
  const rec = r.recommendations;

  // Basics
  setText('ov-title', rec.risk_label + ' Detected');
  setText('ov-summary', rec.summary);
  setText('ts-row', 'Assessment completed: ' + r.timestamp);

  // Risk badge
  document.querySelectorAll('.risk-badge').forEach(el => {
    el.style.background = rec.risk_color + '20'; el.style.color = rec.risk_color;
    el.textContent = '● ' + rec.risk_label;
  });

  // Score gauge
  const score = r.risk_score;
  setText('score-num', score);
  const C = 2 * Math.PI * 60; // circumference for r=60
  document.querySelectorAll('.gauge-fill').forEach(el => {
    el.style.stroke = rec.risk_color;
    el.setAttribute('stroke-dasharray', C);
    setTimeout(() => el.setAttribute('stroke-dashoffset', C - (score / 100) * C), 300);
  });

  // Prob bars
  const cols = { Low: '#10b981', Medium: '#f59e0b', High: '#ef4444' };
  ['Low','Medium','High'].forEach(lvl => {
    const val = r.probabilities[lvl];
    const fill = document.querySelector(`.prob-fill[data-level="${lvl}"]`);
    const pct  = document.querySelector(`.prob-pct[data-level="${lvl}"]`);
    if (fill) setTimeout(() => { fill.style.width = val + '%'; fill.style.background = cols[lvl]; }, 400);
    if (pct) pct.textContent = val + '%';
  });

  // Warnings
  const wl = document.getElementById('warnList');
  if (wl) {
    if (!r.early_warnings.length) wl.innerHTML = '<div class="empty-warn">✅ No critical warning signs detected</div>';
    else wl.innerHTML = r.early_warnings.map(w =>
      `<div class="warn-item ${w.severity}"><span class="warn-icon">${w.icon}</span><span class="warn-text">${w.text}</span></div>`).join('');
  }

  // Patterns
  const pl = document.getElementById('patList');
  if (pl) pl.innerHTML = r.patterns.map(p =>
    `<div class="pat-item"><span class="pat-dot">◆</span><span class="pat-text">${p}</span></div>`).join('');

  // Rec panels
  renderRec('rec-clinical',  rec.clinical);
  renderRec('rec-avoid',     rec.avoid);
  renderRec('rec-exercise',  rec.exercise);
  renderRec('rec-food',      rec.food);

  // Doctor
  const db = document.getElementById('docBox');
  if (db) db.innerHTML = `<p>${rec.doctor}</p>`;

  // Next steps
  const ns = document.getElementById('nextSteps');
  if (ns && rec.next_steps) ns.innerHTML = rec.next_steps.map(s =>
    `<div class="next-step"><span class="step-tick">✓</span><span class="step-txt">${s}</span></div>`).join('');

  // Patient summary
  const ps = document.getElementById('patSummary');
  const pd = r.patient_data || {};
  const keyMap = {
    'Age':'Age','Gender':'Gender','Smoking':'Smoking (1-8)','Air Pollution':'Air Pollution',
    'Genetic Risk':'Genetic Risk','Alcohol use':'Alcohol Use',
    'Shortness of Breath':'Breathlessness','Coughing of Blood':'Blood in Cough',
    'Chest Pain':'Chest Pain','Weight Loss':'Weight Loss',
    'Fatigue':'Fatigue','Clubbing of Finger Nails':'Nail Clubbing'
  };
  if (ps) ps.innerHTML = Object.entries(keyMap).map(([k, lbl]) => {
    let v = pd[k] !== undefined ? pd[k] : '—';
    if (k === 'Gender') v = (+v === 1 ? 'Male' : 'Female');
    return `<div class="pat-row"><div class="pat-key">${lbl}</div><div class="pat-val">${v}</div></div>`;
  }).join('');
}

function renderRec(id, items) {
  const el = document.getElementById(id);
  if (!el || !items) return;
  el.innerHTML = `<div class="rec-list">${items.map(i =>
    `<div class="rec-item"><span class="rec-bullet">›</span><span class="rec-text">${i}</span></div>`).join('')}</div>`;
}

/* ══ TABS ═════════════════════════════════════════════════════════════ */
window.switchTab = function (name) {
  document.querySelectorAll('.rec-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.rec-panel').forEach(p => p.classList.remove('active'));
  document.querySelector(`.rec-tab[data-tab="${name}"]`)?.classList.add('active');
  document.querySelector(`.rec-panel[data-panel="${name}"]`)?.classList.add('active');
};

/* ══ PDF ══════════════════════════════════════════════════════════════ */
window.downloadReport = function () {
  const raw = localStorage.getItem('lungcare_result');
  if (!raw) return;
  sessionStorage.setItem('report_data', raw);
  window.open('/report', '_blank');
};

/* ══ REPORT PAGE ══════════════════════════════════════════════════════ */
function renderReport() {
  const raw = sessionStorage.getItem('report_data') || localStorage.getItem('lungcare_result');
  if (!raw) { document.body.innerHTML = '<p style="padding:40px;font-family:Inter">No data. Please run an assessment first.</p>'; return; }
  const r   = JSON.parse(raw);
  const rec = r.recommendations;
  const pd  = r.patient_data || {};

  const banner = document.getElementById('rpt-banner');
  if (banner) { banner.style.background = rec.risk_color; }
  setText('rpt-lvl', rec.risk_label);
  setText('rpt-score', 'Risk Score: ' + r.risk_score + ' / 100');
  setText('rpt-ts', r.timestamp);
  setText('rpt-summary', rec.summary);
  setText('rpt-doctor', rec.doctor);

  const pg = document.getElementById('rpt-patient');
  if (pg) {
    const flds = ['Age','Gender','Smoking','Air Pollution','Genetic Risk','Alcohol use','Shortness of Breath','Coughing of Blood','Chest Pain','Weight Loss','Fatigue','OccuPational Hazards'];
    pg.innerHTML = flds.map(k => {
      let v = pd[k] !== undefined ? pd[k] : '—';
      if (k === 'Gender') v = (+v === 1 ? 'Male' : 'Female');
      return `<div class="rpt-row"><div class="rpt-key">${k}</div><div class="rpt-val">${v}</div></div>`;
    }).join('');
  }

  renderRptList('rpt-clinical',  rec.clinical);
  renderRptList('rpt-avoid',     rec.avoid);
  renderRptList('rpt-exercise',  rec.exercise);
  renderRptList('rpt-food',      rec.food);

  const rw = document.getElementById('rpt-warnings');
  if (rw) rw.innerHTML = (r.early_warnings.length
    ? r.early_warnings.map(w => `<li>${w.icon} ${w.text}</li>`).join('')
    : '<li>No critical warnings detected</li>');

  setTimeout(() => window.print(), 700);
}

function renderRptList(id, items) {
  const el = document.getElementById(id);
  if (!el || !items) return;
  el.innerHTML = `<ul>${items.map(i => `<li>${i}</li>`).join('')}</ul>`;
}

/* ══ HELPERS ══════════════════════════════════════════════════════════ */
function setText(id, text) { const el = document.getElementById(id); if (el) el.textContent = text; }

/* ══ INIT ══════════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  initSliders();
  if (document.querySelector('.form-sec')) showStep(0);
  if (document.querySelector('.results-page')) renderResults();
  if (document.querySelector('.report-page')) renderReport();
});
