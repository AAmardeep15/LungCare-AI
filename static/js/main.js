/**
 * LungCare AI - Client-side Controller
 * Features: Dark/Light Mode, English/Hindi i18n, Assessment Wizard,
 * Chart.js Visualizations (Gauge, Donut, Radar, SHAP), What-If Simulator, History, PDF
 */

let currentLang = localStorage.getItem('lungcare_lang') || 'en';
let i18nData = {};
let simDebounceTimer = null;
let gaugeChart = null;
let donutChart = null;
let radarChart = null;
let shapChart = null;
let currentResult = null;

// ── 1. GLOBAL INITIALIZATION ─────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initI18n();
  initToast();
  initFAQ();
  initWizard();
  initResults();
});

// ── 2. THEME CONTROLLER ──────────────────────────────────────────────────────
function initTheme() {
  const savedTheme = localStorage.getItem('lungcare_theme') || 'light';
  document.documentElement.setAttribute('data-theme', savedTheme);
  updateThemeIcon(savedTheme);

  const btn = document.getElementById('themeToggleBtn');
  if (btn) {
    btn.addEventListener('click', () => {
      const current = document.documentElement.getAttribute('data-theme');
      const next = current === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      localStorage.setItem('lungcare_theme', next);
      updateThemeIcon(next);
      if (gaugeChart) updateChartThemes();
    });
  }
}

function updateThemeIcon(theme) {
  const icon = document.getElementById('themeIcon');
  if (icon) icon.textContent = theme === 'dark' ? '☀️' : '🌙';
}

function updateChartThemes() {
  // Re-render charts on theme change if on results page
  if (currentResult) {
    renderCharts(currentResult);
  }
}

// ── 3. I18N CONTROLLER (ENGLISH / HINDI) ────────────────────────────────────
async function initI18n() {
  await loadTranslations(currentLang);
  const btn = document.getElementById('langToggleBtn');
  if (btn) {
    btn.addEventListener('click', async () => {
      currentLang = currentLang === 'en' ? 'hi' : 'en';
      localStorage.setItem('lungcare_lang', currentLang);
      await loadTranslations(currentLang);
    });
  }
}

async function loadTranslations(lang) {
  try {
    const res = await fetch(`/static/i18n/${lang}.json`);
    if (res.ok) {
      i18nData = await res.json();
      applyTranslations(i18nData);
      const label = document.getElementById('currentLangLabel');
      if (label) label.textContent = lang === 'en' ? 'HI' : 'EN';
    }
  } catch (e) {
    console.warn('i18n load error:', e);
  }
}

function applyTranslations(dict) {
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    const val = getNestedValue(dict, key);
    if (val) el.textContent = val;
  });
}

function getNestedValue(obj, keyPath) {
  return keyPath.split('.').reduce((acc, part) => acc && acc[part], obj);
}

// ── 4. TOAST NOTIFICATIONS ───────────────────────────────────────────────────
function initToast() {
  window.showToast = function(msg, type = 'success') {
    const container = document.getElementById('toastContainer');
    if (!container) return;
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `<span>${type === 'error' ? '⚠️' : '✅'}</span><span>${msg}</span>`;
    container.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateX(30px)';
      setTimeout(() => toast.remove(), 300);
    }, 3500);
  };
}

// ── 5. FAQ ACCORDION ─────────────────────────────────────────────────────────
function initFAQ() {
  document.querySelectorAll('.faq-question').forEach(btn => {
    btn.addEventListener('click', () => {
      const item = btn.closest('.faq-item');
      item.classList.toggle('active');
    });
  });
}

// ── 6. ASSESSMENT 3-STEP WIZARD ──────────────────────────────────────────────
function initWizard() {
  const form = document.getElementById('assessmentForm');
  if (!form) return;

  let currentStep = 1;
  const totalSteps = 3;

  const step1 = document.getElementById('step1');
  const step2 = document.getElementById('step2');
  const step3 = document.getElementById('step3');
  const steps = [step1, step2, step3];

  const pill1 = document.getElementById('pillStep1');
  const pill2 = document.getElementById('pillStep2');
  const pill3 = document.getElementById('pillStep3');
  const pills = [pill1, pill2, pill3];

  const nextBtn = document.getElementById('nextBtn');
  const prevBtn = document.getElementById('prevBtn');
  const submitBtn = document.getElementById('submitBtn');
  const progressFill = document.getElementById('wizardProgressFill');

  // Slider bubble updates
  document.querySelectorAll('.range-slider').forEach(slider => {
    const bubble = document.getElementById(`bubble_${slider.id}`);
    const updateBubble = () => {
      if (bubble) {
        bubble.textContent = slider.value;
        const val = Number(slider.value);
        const min = Number(slider.min) || 1;
        const max = Number(slider.max) || 9;
        const percent = ((val - min) / (max - min)) * 100;
        bubble.style.left = `calc(${percent}% + (${8 - percent * 0.15}px))`;
      }
    };
    slider.addEventListener('input', updateBubble);
    updateBubble();
  });

  function showStep(s) {
    steps.forEach((el, idx) => {
      if (el) el.classList.toggle('active', idx + 1 === s);
    });
    pills.forEach((p, idx) => {
      if (p) {
        p.classList.toggle('active', idx + 1 === s);
        p.classList.toggle('completed', idx + 1 < s);
      }
    });

    if (progressFill) progressFill.style.width = `${(s / totalSteps) * 100}%`;

    if (prevBtn) prevBtn.style.visibility = s === 1 ? 'hidden' : 'visible';
    if (nextBtn) nextBtn.style.display = s === totalSteps ? 'none' : 'inline-flex';
    if (submitBtn) submitBtn.style.display = s === totalSteps ? 'inline-flex' : 'none';

    window.scrollTo({ top: 120, behavior: 'smooth' });
  }

  function validateStep(s) {
    if (s === 1) {
      const age = Number(document.getElementById('Age').value);
      if (isNaN(age) || age < 1 || age > 120) {
        window.showToast('Please enter a valid age between 1 and 120.', 'error');
        return false;
      }
      const gender = document.getElementById('Gender').value;
      if (!['1', '2'].includes(gender)) {
        window.showToast('Please select biological gender.', 'error');
        return false;
      }
    }
    return true;
  }

  if (nextBtn) {
    nextBtn.addEventListener('click', () => {
      if (!validateStep(currentStep)) return;
      if (currentStep < totalSteps) {
        currentStep++;
        showStep(currentStep);
      }
    });
  }

  if (prevBtn) {
    prevBtn.addEventListener('click', () => {
      if (currentStep > 1) {
        currentStep--;
        showStep(currentStep);
      }
    });
  }

  // Demo Sample Fillers
  setupSampleButtons();

  // Form Submit
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!validateStep(currentStep)) return;

    const overlay = document.getElementById('loadingOverlay');
    if (overlay) overlay.style.display = 'flex';

    const formData = new FormData(form);
    const payload = {};
    formData.forEach((val, key) => {
      payload[key] = Number(val);
    });

    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const data = await res.json();
      if (!res.ok || !data.success) {
        if (overlay) overlay.style.display = 'none';
        const errs = data.errors ? data.errors.join('<br>') : (data.error || 'Prediction failed');
        window.showToast(errs, 'error');
        return;
      }

      // Save result in sessionStorage & localStorage history
      sessionStorage.setItem('lungcare_assessment_result', JSON.stringify(data.result));
      saveToHistory(data.result);

      // Redirect to results page
      window.location.href = '/results';
    } catch (err) {
      if (overlay) overlay.style.display = 'none';
      window.showToast('Network error while analyzing. Please check your connection.', 'error');
    }
  });
}

function setupSampleButtons() {
  const samples = {
    low: {
      Age: 32, Gender: 1, 'Air Pollution': 2, 'Alcohol use': 1, 'Dust Allergy': 1,
      'OccuPational Hazards': 1, 'Genetic Risk': 1, 'chronic Lung Disease': 1,
      'Balanced Diet': 7, Obesity: 2, Smoking: 1, 'Passive Smoker': 1, 'Chest Pain': 1,
      'Coughing of Blood': 1, Fatigue: 1, 'Weight Loss': 1, 'Shortness of Breath': 1,
      Wheezing: 1, 'Swallowing Difficulty': 1, 'Clubbing of Finger Nails': 1,
      'Frequent Cold': 1, 'Dry Cough': 1, Snoring: 1
    },
    med: {
      Age: 52, Gender: 1, 'Air Pollution': 5, 'Alcohol use': 4, 'Dust Allergy': 4,
      'OccuPational Hazards': 4, 'Genetic Risk': 4, 'chronic Lung Disease': 3,
      'Balanced Diet': 4, Obesity: 4, Smoking: 4, 'Passive Smoker': 4, 'Chest Pain': 4,
      'Coughing of Blood': 3, Fatigue: 4, 'Weight Loss': 3, 'Shortness of Breath': 4,
      Wheezing: 3, 'Swallowing Difficulty': 3, 'Clubbing of Finger Nails': 2,
      'Frequent Cold': 4, 'Dry Cough': 4, Snoring: 3
    },
    high: {
      Age: 64, Gender: 1, 'Air Pollution': 8, 'Alcohol use': 7, 'Dust Allergy': 7,
      'OccuPational Hazards': 8, 'Genetic Risk': 7, 'chronic Lung Disease': 7,
      'Balanced Diet': 2, Obesity: 7, Smoking: 8, 'Passive Smoker': 8, 'Chest Pain': 7,
      'Coughing of Blood': 7, Fatigue: 8, 'Weight Loss': 7, 'Shortness of Breath': 8,
      Wheezing: 7, 'Swallowing Difficulty': 6, 'Clubbing of Finger Nails': 7,
      'Frequent Cold': 6, 'Dry Cough': 7, Snoring: 6
    }
  };

  const fill = (obj) => {
    Object.entries(obj).forEach(([key, val]) => {
      const input = document.querySelector(`[name="${key}"]`);
      if (input) {
        input.value = val;
        input.dispatchEvent(new Event('input'));
      }
    });
    window.showToast('Sample patient profile loaded.', 'success');
  };

  const btnLow = document.getElementById('btnFillLow');
  const btnMed = document.getElementById('btnFillMed');
  const btnHigh = document.getElementById('btnFillHigh');

  if (btnLow) btnLow.addEventListener('click', () => fill(samples.low));
  if (btnMed) btnMed.addEventListener('click', () => fill(samples.med));
  if (btnHigh) btnHigh.addEventListener('click', () => fill(samples.high));
}

function saveToHistory(res) {
  try {
    const raw = localStorage.getItem('lungcare_history');
    const list = raw ? JSON.parse(raw) : [];
    list.push(res);
    // keep maximum 50 records
    if (list.length > 50) list.shift();
    localStorage.setItem('lungcare_history', JSON.stringify(list));
  } catch (e) {
    console.warn('History save error:', e);
  }
}

// ── 7. RESULTS PAGE DASHBOARD ────────────────────────────────────────────────
function initResults() {
  const resultsPage = document.querySelector('.results-page');
  if (!resultsPage) return;

  const raw = sessionStorage.getItem('lungcare_assessment_result');
  if (!raw) {
    // If user opens /results directly without taking assessment, redirect to assessment
    window.location.href = '/assessment';
    return;
  }

  currentResult = JSON.parse(raw);
  populateResultsData(currentResult);
  renderCharts(currentResult);
  initTabs();
  initWhatIfSimulator(currentResult);

  // PDF Export
  const btnPDF = document.getElementById('btnExportPDF');
  if (btnPDF) {
    btnPDF.addEventListener('click', () => {
      const el = document.getElementById('printableReport');
      const opt = {
        margin: 10,
        filename: `LungCare-Report-${new Date().toISOString().slice(0, 10)}.pdf`,
        image: { type: 'jpeg', quality: 0.98 },
        html2canvas: { scale: 2, useCORS: true },
        jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' }
      };
      html2pdf().set(opt).from(el).save();
    });
  }

  // Save to history button
  const btnHist = document.getElementById('btnSaveHistory');
  if (btnHist) {
    btnHist.addEventListener('click', () => {
      saveToHistory(currentResult);
      window.showToast('Assessment saved to your local history.', 'success');
    });
  }
}

function populateResultsData(r) {
  const ts = document.getElementById('reportTimestamp');
  if (ts) ts.textContent = r.timestamp || new Date().toLocaleString();

  const scoreEl = document.getElementById('riskScoreVal');
  if (scoreEl) scoreEl.textContent = r.risk_score;

  const confEl = document.getElementById('confidenceVal');
  if (confEl) confEl.textContent = `${r.confidence}%`;

  const badge = document.getElementById('riskLevelBadge');
  if (badge) {
    badge.textContent = `${r.risk_level} Risk Tier`;
    const map = {
      Low: { bg: 'var(--risk-low-bg)', color: 'var(--risk-low)' },
      Medium: { bg: 'var(--risk-med-bg)', color: 'var(--risk-med)' },
      High: { bg: 'var(--risk-high-bg)', color: 'var(--risk-high)' }
    };
    const c = map[r.risk_level] || map.Low;
    badge.style.background = c.bg;
    badge.style.color = c.color;
    if (scoreEl) scoreEl.style.color = c.color;
  }

  // Probabilities text
  const p = r.probabilities || {};
  const pl = document.getElementById('probLow');
  const pm = document.getElementById('probMed');
  const ph = document.getElementById('probHigh');
  if (pl) pl.textContent = `${p.Low || 0}%`;
  if (pm) pm.textContent = `${p.Medium || 0}%`;
  if (ph) ph.textContent = `${p.High || 0}%`;

  // Early Warnings
  const ewContainer = document.getElementById('earlyWarningsContainer');
  if (ewContainer) {
    const warnings = r.early_warnings || [];
    if (warnings.length === 0) {
      ewContainer.innerHTML = '<p style="color:var(--text-muted); font-size:0.88rem;">No critical emergency indicators flagged.</p>';
    } else {
      ewContainer.innerHTML = warnings.map(w => `
        <div class="warning-item">
          <span style="font-size:1.2rem;">${w.icon}</span>
          <span style="font-weight:600;">${w.text}</span>
        </div>
      `).join('');
    }
  }

  // Pattern synergies
  const patContainer = document.getElementById('patternsContainer');
  if (patContainer) {
    const patterns = r.patterns || [];
    if (patterns.length === 0) {
      patContainer.innerHTML = '<p style="color:var(--text-muted); font-size:0.88rem;">No multi-factor risk interactions detected.</p>';
    } else {
      patContainer.innerHTML = patterns.map(pt => `
        <div class="pattern-item">
          <strong>✦ Synergy:</strong> ${pt}
        </div>
      `).join('');
    }
  }

  // Recommendations
  const recs = r.recommendations || {};
  populateRecList('recsClinical', recs.clinical);
  populateRecList('recsAvoid', recs.avoid);
  populateRecList('recsExercise', recs.exercise);
  populateRecList('recsFood', recs.food);

  const docEl = document.getElementById('recsDoctor');
  if (docEl && recs.doctor) docEl.textContent = recs.doctor;

  // Timeline
  const tlEl = document.getElementById('nextStepsTimeline');
  if (tlEl && recs.next_steps) {
    tlEl.innerHTML = recs.next_steps.map((step, idx) => `
      <div class="timeline-step">
        <strong>Phase ${idx + 1}:</strong> ${step}
      </div>
    `).join('');
  }
}

function populateRecList(containerId, items) {
  const el = document.getElementById(containerId);
  if (!el) return;
  if (!items || items.length === 0) {
    el.innerHTML = '<div style="color:var(--text-muted);">No specific guidance in this category.</div>';
    return;
  }
  el.innerHTML = items.map(item => `
    <div class="recs-item">
      <span class="recs-bullet">✓</span>
      <span>${item}</span>
    </div>
  `).join('');
}

function initTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const target = document.getElementById(btn.getAttribute('data-tab'));
      if (target) target.classList.add('active');
    });
  });
}

// ── 8. CHART.JS VISUALIZATIONS ───────────────────────────────────────────────
function renderCharts(r) {
  renderGauge(r.risk_score, r.risk_level);
  renderDonut(r.probabilities || {});
  renderRadar(r.patient_data || {});
  renderShapBars(r.top_factors || {});
}

function renderGauge(score, level) {
  const canvas = document.getElementById('gaugeCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (gaugeChart) gaugeChart.destroy();

  const colorMap = { Low: '#10b981', Medium: '#f59e0b', High: '#ef4444' };
  const gaugeColor = colorMap[level] || '#0d9488';

  gaugeChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      datasets: [{
        data: [score, 100 - score],
        backgroundColor: [gaugeColor, 'rgba(148, 163, 184, 0.2)'],
        borderWidth: 0,
        circumference: 180,
        rotation: 270
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '78%',
      plugins: {
        tooltip: { enabled: false }
      }
    }
  });
}

function renderDonut(probs) {
  const canvas = document.getElementById('donutChartCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (donutChart) donutChart.destroy();

  donutChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Low Risk', 'Medium Risk', 'High Risk'],
      datasets: [{
        data: [probs.Low || 0, probs.Medium || 0, probs.High || 0],
        backgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
        borderWidth: 2,
        borderColor: document.documentElement.getAttribute('data-theme') === 'dark' ? '#111827' : '#ffffff'
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      }
    }
  });
}

function renderRadar(p) {
  const canvas = document.getElementById('radarChartCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (radarChart) radarChart.destroy();

  // Aggregate into 5 clinical pillars (normalized out of 9)
  const lifestyle = ((p.Smoking || 1) + (p['Alcohol use'] || 1) + (10 - (p['Balanced Diet'] || 5))) / 3;
  const environment = ((p['Air Pollution'] || 1) + (p['OccuPational Hazards'] || 1) + (p['Passive Smoker'] || 1)) / 3;
  const genetic = ((p['Genetic Risk'] || 1) + (p['chronic Lung Disease'] || 1)) / 2;
  const thoracic = ((p['Chest Pain'] || 1) + (p['Coughing of Blood'] || 1) + (p['Dry Cough'] || 1) + (p.Wheezing || 1)) / 4;
  const systemic = ((p['Shortness of Breath'] || 1) + (p.Fatigue || 1) + (p['Weight Loss'] || 1) + (p['Clubbing of Finger Nails'] || 1)) / 4;

  radarChart = new Chart(ctx, {
    type: 'radar',
    data: {
      labels: ['Lifestyle', 'Environment', 'Genetic/Chronic', 'Thoracic Signs', 'Systemic Signs'],
      datasets: [{
        label: 'Patient Severity (1-9)',
        data: [lifestyle, environment, genetic, thoracic, systemic],
        backgroundColor: 'rgba(13, 148, 136, 0.25)',
        borderColor: '#0d9488',
        borderWidth: 2,
        pointBackgroundColor: '#0d9488'
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          min: 1,
          max: 9,
          ticks: { stepSize: 2, display: false },
          grid: { color: 'rgba(148, 163, 184, 0.2)' },
          pointLabels: {
            font: { size: 11, weight: '600' },
            color: document.documentElement.getAttribute('data-theme') === 'dark' ? '#cbd5e1' : '#475569'
          }
        }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });
}

function renderShapBars(topFactors) {
  const canvas = document.getElementById('shapBarCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (shapChart) shapChart.destroy();

  const raising = topFactors.raising || [];
  const lowering = topFactors.lowering || [];

  const labels = [];
  const values = [];
  const colors = [];

  // Raising factors (positive impact to high risk)
  raising.forEach(f => {
    labels.push(`${f.name} (Val: ${f.value})`);
    values.push(f.impact);
    colors.push('#ef4444');
  });

  // Lowering factors (protective impact)
  lowering.forEach(f => {
    labels.push(`${f.name} (Val: ${f.value})`);
    values.push(-f.impact);
    colors.push('#10b981');
  });

  shapChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Feature Risk Contribution (%)',
        data: values,
        backgroundColor: colors,
        borderRadius: 4
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          grid: { color: 'rgba(148, 163, 184, 0.15)' },
          ticks: {
            callback: (v) => `${v}%`
          }
        },
        y: {
          grid: { display: false },
          ticks: {
            font: { size: 10, weight: '600' },
            color: document.documentElement.getAttribute('data-theme') === 'dark' ? '#cbd5e1' : '#334155'
          }
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `Impact: ${ctx.raw > 0 ? '+' : ''}${ctx.raw}% on Risk Tier`
          }
        }
      }
    }
  });
}

// ── 9. WHAT-IF SIMULATOR ─────────────────────────────────────────────────────
function initWhatIfSimulator(r) {
  const p = r.patient_data || {};
  const sliders = document.querySelectorAll('.sim-slider');
  if (sliders.length === 0) return;

  // Initialize sliders with initial patient values
  sliders.forEach(s => {
    const feat = s.getAttribute('data-feature');
    if (p[feat] !== undefined) {
      s.value = p[feat];
      const valDisplay = document.getElementById(`simVal_${s.id.replace('sim_', '')}`);
      if (valDisplay) valDisplay.textContent = s.value;
    }
  });

  const runSimulation = () => {
    const modifications = {};
    sliders.forEach(s => {
      const feat = s.getAttribute('data-feature');
      modifications[feat] = Number(s.value);
    });

    fetch('/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        base_data: p,
        modifications: modifications,
        base_score: r.risk_score
      })
    })
    .then(res => res.json())
    .then(data => {
      if (data.success) {
        updateSimResults(data, r.risk_score);
      }
    })
    .catch(err => console.warn('Sim error:', err));
  };

  sliders.forEach(s => {
    s.addEventListener('input', () => {
      const valDisplay = document.getElementById(`simVal_${s.id.replace('sim_', '')}`);
      if (valDisplay) valDisplay.textContent = s.value;

      clearTimeout(simDebounceTimer);
      simDebounceTimer = setTimeout(runSimulation, 300);
    });
  });

  // Reset button
  const resetBtn = document.getElementById('btnResetSim');
  if (resetBtn) {
    resetBtn.addEventListener('click', () => {
      sliders.forEach(s => {
        const feat = s.getAttribute('data-feature');
        if (p[feat] !== undefined) {
          s.value = p[feat];
          const valDisplay = document.getElementById(`simVal_${s.id.replace('sim_', '')}`);
          if (valDisplay) valDisplay.textContent = s.value;
        }
      });
      runSimulation();
    });
  }
}

function updateSimResults(data, originalScore) {
  // Update gauge and score text
  renderGauge(data.new_risk_score, data.new_risk_level);
  renderDonut(data.new_probabilities);

  const scoreEl = document.getElementById('riskScoreVal');
  if (scoreEl) scoreEl.textContent = data.new_risk_score;

  const badge = document.getElementById('riskLevelBadge');
  if (badge) {
    badge.textContent = `${data.new_risk_level} Risk Tier`;
    const map = {
      Low: { bg: 'var(--risk-low-bg)', color: 'var(--risk-low)' },
      Medium: { bg: 'var(--risk-med-bg)', color: 'var(--risk-med)' },
      High: { bg: 'var(--risk-high-bg)', color: 'var(--risk-high)' }
    };
    const c = map[data.new_risk_level] || map.Low;
    badge.style.background = c.bg;
    badge.style.color = c.color;
    if (scoreEl) scoreEl.style.color = c.color;
  }

  // Delta badge
  const deltaBadge = document.getElementById('simRiskDeltaBadge');
  if (deltaBadge) {
    const delta = data.risk_delta;
    deltaBadge.className = 'sim-diff-badge';
    if (delta < 0) {
      deltaBadge.classList.add('improved');
      deltaBadge.textContent = `Risk Change: ▼ ${Math.abs(delta)} pts (Improved)`;
    } else if (delta > 0) {
      deltaBadge.classList.add('worsened');
      deltaBadge.textContent = `Risk Change: ▲ ${delta} pts (Elevated)`;
    } else {
      deltaBadge.classList.add('neutral');
      deltaBadge.textContent = 'Risk Change: 0 pts';
    }
  }
}
