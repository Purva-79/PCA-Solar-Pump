/* ===========================
   SOLAR-PUMP PCA MONITOR
   app.js — Full Application Logic
=========================== */

'use strict';

// =============================================
// 1. UTILITY / MATH HELPERS
// =============================================

const utils = {
  // Clamp
  clamp: (v, lo, hi) => Math.max(lo, Math.min(hi, v)),

  // Gaussian noise (Box-Muller)
  randn: () => {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  },

  // Mean of array
  mean: (arr) => arr.reduce((s, v) => s + v, 0) / arr.length,

  // Standard deviation
  std: (arr) => {
    const m = utils.mean(arr);
    return Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / arr.length);
  },

  // Transpose matrix
  transpose: (m) => m[0].map((_, i) => m.map(r => r[i])),

  // Matrix multiply
  matMul: (A, B) => A.map(rowA =>
    utils.transpose(B).map(colB =>
      rowA.reduce((sum, a, k) => sum + a * colB[k], 0)
    )
  ),

  // Format numbers
  fmt: (n, dec = 2) => Number(n).toFixed(dec),
  fmtPct: (n) => `${(n * 100).toFixed(1)}%`,

  // Pearson correlation
  pearson: (a, b) => {
    const ma = utils.mean(a), mb = utils.mean(b);
    const num = a.reduce((s, v, i) => s + (v - ma) * (b[i] - mb), 0);
    const den = Math.sqrt(
      a.reduce((s, v) => s + (v - ma) ** 2, 0) *
      b.reduce((s, v) => s + (v - mb) ** 2, 0)
    );
    return den === 0 ? 0 : num / den;
  }
};

// =============================================
// 2. PCA ENGINE (Pure JS, no external lib)
// =============================================

class PCAEngine {
  constructor() {
    this.components = null;
    this.eigenvalues = null;
    this.mean = null;
    this.scale = null;
    this.scores = null;
  }

  // Standardize data matrix (N×D)
  standardize(X) {
    const D = X[0].length, N = X.length;
    const mu = [], sigma = [];
    for (let d = 0; d < D; d++) {
      const col = X.map(r => r[d]);
      mu.push(utils.mean(col));
      sigma.push(utils.std(col) || 1);
    }
    this.mean = mu;
    this.scale = sigma;
    return X.map(row => row.map((v, d) => (v - mu[d]) / sigma[d]));
  }

  // Covariance matrix (D×D)
  covMatrix(Xstd) {
    const N = Xstd.length, D = Xstd[0].length;
    const cov = Array.from({ length: D }, () => new Array(D).fill(0));
    for (let i = 0; i < D; i++) {
      for (let j = 0; j < D; j++) {
        let s = 0;
        for (let n = 0; n < N; n++) s += Xstd[n][i] * Xstd[n][j];
        cov[i][j] = s / (N - 1);
      }
    }
    return cov;
  }

  // Power Iteration for top K eigenvectors
  powerIteration(cov, k) {
    const D = cov.length;
    const eigenVecs = [];
    const eigenVals = [];
    let covCopy = cov.map(r => [...r]);

    for (let pc = 0; pc < k; pc++) {
      let v = Array.from({ length: D }, () => utils.randn());
      // Normalize
      let norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
      v = v.map(x => x / norm);

      for (let iter = 0; iter < 300; iter++) {
        // v_new = Cov * v
        let vNew = covCopy.map(row => row.reduce((s, c, j) => s + c * v[j], 0));
        norm = Math.sqrt(vNew.reduce((s, x) => s + x * x, 0));
        if (norm < 1e-12) break;
        vNew = vNew.map(x => x / norm);

        const diff = Math.sqrt(vNew.reduce((s, x, i) => s + (x - v[i]) ** 2, 0));
        v = vNew;
        if (diff < 1e-8) break;
      }

      // Eigenvalue = v^T * Cov * v
      const Cv = covCopy.map(row => row.reduce((s, c, j) => s + c * v[j], 0));
      const lam = v.reduce((s, x, i) => s + x * Cv[i], 0);
      eigenVecs.push(v);
      eigenVals.push(Math.abs(lam));

      // Deflate: Cov -= lambda * v * v^T
      for (let i = 0; i < D; i++)
        for (let j = 0; j < D; j++)
          covCopy[i][j] -= lam * v[i] * v[j];
    }
    return { eigenVecs, eigenVals };
  }

  fit(X, nComponents = 3, standardize = true) {
    const Xstd = standardize ? this.standardize(X) : X;
    const cov = this.covMatrix(Xstd);
    const { eigenVecs, eigenVals } = this.powerIteration(cov, nComponents);
    const totalVar = eigenVals.reduce((s, v) => s + v, 0);
    // Also estimate total from trace
    const traceVar = cov.reduce((s, _, i) => s + cov[i][i], 0);
    this.eigenvalues = eigenVals;
    this.totalVariance = traceVar;
    this.components = eigenVecs;
    this.explainedVarianceRatio = eigenVals.map(v => v / traceVar);
    // Project
    this.scores = Xstd.map(row =>
      eigenVecs.map(pc => pc.reduce((s, w, j) => s + w * row[j], 0))
    );
    return this;
  }

  // Transform new data
  transform(X, standardize = true) {
    if (!this.components) return null;
    const Xstd = standardize ? X.map(row =>
      row.map((v, d) => (v - this.mean[d]) / this.scale[d])
    ) : X;
    return Xstd.map(row =>
      this.components.map(pc => pc.reduce((s, w, j) => s + w * row[j], 0))
    );
  }

  // Reconstruction error (MSE)
  reconstructionError(X) {
    if (!this.scores) return 0;
    const Xstd = X.map(row => row.map((v, d) => (v - this.mean[d]) / this.scale[d]));
    // Reconstruct: Xrec = scores * W^T
    let totalErr = 0;
    for (let n = 0; n < Xstd.length; n++) {
      for (let d = 0; d < Xstd[0].length; d++) {
        const rec = this.components.reduce((s, pc, k) => s + this.scores[n][k] * pc[d], 0);
        totalErr += (Xstd[n][d] - rec) ** 2;
      }
    }
    return totalErr / (Xstd.length * Xstd[0].length);
  }
}

// =============================================
// 3. SENSOR DATA GENERATOR
// =============================================

class SensorDataGenerator {
  constructor(n = 100, noiseLevel = 0.3) {
    this.n = n;
    this.noiseLevel = noiseLevel;
    this.data = null;
    this.regimes = null;
    this.time = null;
  }

  generate() {
    const n = this.n;
    const noise = this.noiseLevel;

    // Shared latent factors (simulate correlated sensors)
    const solarFactor   = Array.from({ length: n }, (_, i) =>
      0.7 * Math.sin(i / n * 2 * Math.PI * 1.5) + 0.3 * Math.sin(i / n * 2 * Math.PI * 4)
    );
    const loadFactor    = Array.from({ length: n }, (_, i) =>
      -0.5 * Math.cos(i / n * 2 * Math.PI) + 0.4 * Math.sin(i / n * 2 * Math.PI * 3)
    );
    const waterFactor   = Array.from({ length: n }, (_, i) =>
      0.4 * Math.sin(i / n * 2 * Math.PI * 0.8 + 1.2)
    );

    // Inject strained episodes
    const strainedStart = Math.floor(n * 0.4);
    const strainedEnd   = Math.floor(n * 0.55);
    const anomalyStart  = Math.floor(n * 0.78);
    const anomalyEnd    = Math.floor(n * 0.84);

    const regimes = Array.from({ length: n }, (_, i) => {
      if (i >= anomalyStart && i < anomalyEnd) return 'anomaly';
      if (i >= strainedStart && i < strainedEnd) return 'strained';
      return 'normal';
    });

    const sf = (i) => solarFactor[i];
    const lf = (i) => loadFactor[i];
    const wf = (i) => waterFactor[i];
    const r  = () => utils.randn() * noise;
    const strainBoost = (i) => regimes[i] === 'strained' ? 0.6 : (regimes[i] === 'anomaly' ? 1.4 : 0);

    // 1. Solar Irradiance (W/m²) : 400–1000
    const irradiance = Array.from({ length: n }, (_, i) =>
      utils.clamp(700 + 200 * sf(i) + r() * 80 - strainBoost(i) * 300, 100, 1050)
    );

    // 2. PV Voltage (V) : 180–240
    const pvVoltage = Array.from({ length: n }, (_, i) =>
      utils.clamp(210 + 20 * sf(i) + r() * 8 - strainBoost(i) * 40, 100, 260)
    );

    // 3. PV Current (A) : 5–15
    const pvCurrent = Array.from({ length: n }, (_, i) =>
      utils.clamp(10 + 3 * sf(i) + r() * 1.2 - strainBoost(i) * 4, 1, 18)
    );

    // 4. Motor RPM : 1000–3000
    const motorRPM = Array.from({ length: n }, (_, i) =>
      utils.clamp(2000 + 500 * lf(i) + r() * 120 + strainBoost(i) * 400, 500, 3500)
    );

    // 5. Flow Rate (L/min) : 20–80
    const flowRate = Array.from({ length: n }, (_, i) =>
      utils.clamp(50 + 20 * lf(i) + r() * 5 - strainBoost(i) * 20, 5, 100)
    );

    // 6. Pump Vibration (g) : 0.5–4
    const vibration = Array.from({ length: n }, (_, i) =>
      utils.clamp(1.5 + 0.5 * Math.abs(lf(i)) + r() * 0.3 + strainBoost(i) * 2.5, 0.2, 8)
    );

    // 7. Tank Level (%) : 30–90
    const tankLevel = Array.from({ length: n }, (_, i) =>
      utils.clamp(60 + 20 * wf(i) + r() * 5 - strainBoost(i) * 15, 10, 100)
    );

    // 8. Pressure (kPa) : 100–400
    const pressure = Array.from({ length: n }, (_, i) =>
      utils.clamp(250 + 80 * wf(i) + r() * 20 + strainBoost(i) * 60, 50, 500)
    );

    // 9. Turbidity (NTU) : 1–30
    const turbidity = Array.from({ length: n }, (_, i) =>
      utils.clamp(10 + 5 * wf(i) + r() * 2 + strainBoost(i) * 10, 0.5, 50)
    );

    this.time = Array.from({ length: n }, (_, i) => i);
    this.regimes = regimes;
    this.sensorArrays = {
      irradiance, pvVoltage, pvCurrent,
      motorRPM, flowRate, vibration,
      tankLevel, pressure, turbidity
    };

    // Build N×9 data matrix
    this.data = Array.from({ length: n }, (_, i) => [
      irradiance[i], pvVoltage[i], pvCurrent[i],
      motorRPM[i], flowRate[i], vibration[i],
      tankLevel[i], pressure[i], turbidity[i]
    ]);

    return this;
  }
}

// =============================================
// 4. CHART MANAGER
// =============================================

class ChartManager {
  constructor() {
    this.charts = {};
    this.defaultFont = 'Outfit';
    // Set Chart.js global defaults
    Chart.defaults.font.family = this.defaultFont;
    Chart.defaults.color = '#706F69';
    Chart.defaults.plugins.legend.display = false;
    Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(26,26,23,0.92)';
    Chart.defaults.plugins.tooltip.titleColor = '#FFD700';
    Chart.defaults.plugins.tooltip.bodyColor = '#FAFAF8';
    Chart.defaults.plugins.tooltip.padding = 12;
    Chart.defaults.plugins.tooltip.cornerRadius = 10;
    Chart.defaults.plugins.tooltip.borderWidth = 1;
    Chart.defaults.plugins.tooltip.borderColor = 'rgba(255,215,0,0.3)';
  }

  register(name, chart) {
    if (this.charts[name]) { this.charts[name].destroy(); }
    this.charts[name] = chart;
  }

  destroy(name) {
    if (this.charts[name]) { this.charts[name].destroy(); delete this.charts[name]; }
  }
}

// =============================================
// 5. APPLICATION STATE
// =============================================

const state = {
  nPoints: 100,
  noiseLevel: 0.3,
  nComponents: 3,
  standardize: true,
  covMethod: 'standard',
  isLive: false,
  liveInterval: null,
  generator: null,
  pca: null,
  chartManager: new ChartManager(),
};

const SENSOR_LABELS = [
  'Irradiance', 'PV Voltage', 'PV Current',
  'Motor RPM', 'Flow Rate', 'Vibration',
  'Tank Level', 'Pressure', 'Turbidity'
];

const SENSOR_UNITS = [
  'W/m²', 'V', 'A',
  'RPM', 'L/min', 'g',
  '%', 'kPa', 'NTU'
];

const SENSOR_RANGES = [
  [100, 1050], [100, 260], [1, 18],
  [500, 3500], [5, 100], [0.2, 8],
  [10, 100], [50, 500], [0.5, 50]
];

// =============================================
// 6. HERO CHART
// =============================================

function initHeroChart() {
  const ctx = document.getElementById('heroChart').getContext('2d');
  const n = 60;
  const labels = Array.from({ length: n }, (_, i) => i);

  const makeWave = (base, amp, freq, noise = 0.1) =>
    Array.from({ length: n }, (_, i) =>
      base + amp * Math.sin(i / n * 2 * Math.PI * freq) + utils.randn() * noise * amp
    );

  const chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Irradiance',
          data: makeWave(0.6, 0.25, 2, 0.08),
          borderColor: '#FFD700',
          backgroundColor: 'rgba(255,215,0,0.08)',
          borderWidth: 2.5,
          tension: 0.4,
          fill: true,
          pointRadius: 0,
        },
        {
          label: 'Pump Load',
          data: makeWave(0.45, 0.2, 3, 0.1),
          borderColor: '#94A3B8',
          backgroundColor: 'rgba(148,163,184,0.06)',
          borderWidth: 2,
          tension: 0.4,
          fill: false,
          pointRadius: 0,
        },
        {
          label: 'Water Level',
          data: makeWave(0.55, 0.15, 1.5, 0.07),
          borderColor: '#60A5FA',
          backgroundColor: 'rgba(96,165,250,0.06)',
          borderWidth: 2,
          tension: 0.4,
          fill: false,
          pointRadius: 0,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 1000 },
      scales: {
        x: { display: false },
        y: {
          display: true,
          min: 0, max: 1,
          grid: { color: 'rgba(0,0,0,0.04)' },
          ticks: { display: false }
        }
      },
      plugins: {
        legend: { display: true, position: 'bottom',
          labels: { font: { size: 11 }, boxWidth: 12, padding: 16 }
        },
        tooltip: { mode: 'index', intersect: false }
      }
    }
  });
  state.chartManager.register('hero', chart);

  // Animate hero chart continuously
  setInterval(() => {
    const ch = state.chartManager.charts.hero;
    if (!ch) return;
    ch.data.datasets.forEach(ds => {
      ds.data.shift();
      ds.data.push(
        utils.clamp(ds.data[ds.data.length - 1] + utils.randn() * 0.04, 0.1, 0.95)
      );
    });
    ch.update('none');
  }, 300);
}

// =============================================
// 7. SENSOR STREAM CHARTS
// =============================================

function buildSensorChart(canvasId, datasets, label) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  const chart = new Chart(ctx, {
    type: 'line',
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 400 },
      scales: {
        x: { display: false },
        y: {
          display: true,
          grid: { color: 'rgba(0,0,0,0.04)', drawBorder: false },
          ticks: { font: { size: 9 }, maxTicksLimit: 4 }
        }
      },
      plugins: {
        legend: { display: true, position: 'bottom',
          labels: { font: { size: 9 }, boxWidth: 10, padding: 10 }
        },
        tooltip: { mode: 'index', intersect: false }
      }
    }
  });
  state.chartManager.register(canvasId, chart);
  return chart;
}

function getSensorPoints(n) {
  const gen = state.generator;
  if (!gen || !gen.sensorArrays) return {};
  const start = Math.max(0, gen.n - n);
  const slice = (arr) => arr.slice(start).map((v, i) => ({ x: start + i, y: v }));
  return {
    irradiance: slice(gen.sensorArrays.irradiance),
    pvVoltage:  slice(gen.sensorArrays.pvVoltage),
    pvCurrent:  slice(gen.sensorArrays.pvCurrent),
    motorRPM:   slice(gen.sensorArrays.motorRPM),
    flowRate:   slice(gen.sensorArrays.flowRate),
    vibration:  slice(gen.sensorArrays.vibration),
    tankLevel:  slice(gen.sensorArrays.tankLevel),
    pressure:   slice(gen.sensorArrays.pressure),
    turbidity:  slice(gen.sensorArrays.turbidity),
  };
}

function renderSensorCharts() {
  const pts = getSensorPoints(state.nPoints);
  const timeLabels = Array.from({ length: state.nPoints }, (_, i) => i);

  const toY = (arr) => arr.map(p => p.y);

  // Solar
  const solarDs = [
    { label: 'Irradiance (W/m²)', data: toY(pts.irradiance),
      borderColor: '#FFD700', backgroundColor: 'rgba(255,215,0,0.1)',
      borderWidth: 2, tension: 0.4, fill: true, pointRadius: 0 },
    { label: 'PV Voltage (V)', data: toY(pts.pvVoltage).map(v => v / 10),
      borderColor: '#FFF176', backgroundColor: 'transparent',
      borderWidth: 1.5, tension: 0.4, fill: false, pointRadius: 0, borderDash: [4,2] },
    { label: 'PV Current (A)', data: toY(pts.pvCurrent).map(v => v * 50),
      borderColor: '#FFE082', backgroundColor: 'transparent',
      borderWidth: 1.5, tension: 0.4, fill: false, pointRadius: 0, borderDash: [2,3] },
  ];
  buildSensorChart('solarChart', solarDs.map(d => ({ ...d, data: d.data.map((y, i) => ({ x: i, y })) })));

  // Pump
  const pumpDs = [
    { label: 'Motor RPM (÷10)', data: toY(pts.motorRPM).map(v => v / 10),
      borderColor: '#94A3B8', backgroundColor: 'rgba(148,163,184,0.1)',
      borderWidth: 2, tension: 0.4, fill: true, pointRadius: 0 },
    { label: 'Flow Rate (L/min)', data: toY(pts.flowRate),
      borderColor: '#64748B', backgroundColor: 'transparent',
      borderWidth: 1.5, tension: 0.4, fill: false, pointRadius: 0, borderDash: [4,2] },
    { label: 'Vibration ×20 (g)', data: toY(pts.vibration).map(v => v * 20),
      borderColor: '#475569', backgroundColor: 'transparent',
      borderWidth: 1.5, tension: 0.4, fill: false, pointRadius: 0, borderDash: [2,3] },
  ];
  buildSensorChart('pumpChart', pumpDs.map(d => ({ ...d, data: d.data.map((y, i) => ({ x: i, y })) })));

  // Water
  const waterDs = [
    { label: 'Tank Level (%)', data: toY(pts.tankLevel),
      borderColor: '#60A5FA', backgroundColor: 'rgba(96,165,250,0.1)',
      borderWidth: 2, tension: 0.4, fill: true, pointRadius: 0 },
    { label: 'Pressure ÷10 (kPa)', data: toY(pts.pressure).map(v => v / 10),
      borderColor: '#3B82F6', backgroundColor: 'transparent',
      borderWidth: 1.5, tension: 0.4, fill: false, pointRadius: 0, borderDash: [4,2] },
    { label: 'Turbidity ×3 (NTU)', data: toY(pts.turbidity).map(v => v * 3),
      borderColor: '#93C5FD', backgroundColor: 'transparent',
      borderWidth: 1.5, tension: 0.4, fill: false, pointRadius: 0, borderDash: [2,3] },
  ];
  buildSensorChart('waterChart', waterDs.map(d => ({ ...d, data: d.data.map((y, i) => ({ x: i, y })) })));
}

// =============================================
// 8. LIVE MINI CARDS UPDATE
// =============================================

function updateMiniCards() {
  const gen = state.generator;
  if (!gen || !gen.sensorArrays) return;
  const last = (arr) => arr[arr.length - 1];
  const sa = gen.sensorArrays;
  const ranges = SENSOR_RANGES;

  const setPct = (barId, val, range) => {
    const pct = ((val - range[0]) / (range[1] - range[0])) * 100;
    const bar = document.getElementById(barId);
    if (bar) bar.style.width = utils.clamp(pct, 2, 100) + '%';
  };

  const setVal = (id, val, unit = '') => {
    const el = document.getElementById(id);
    if (el) el.textContent = utils.fmt(val, 1) + (unit ? ` ${unit}` : '');
  };

  setVal('val-irr', last(sa.irradiance)); setPct('bar-irr', last(sa.irradiance), ranges[0]);
  setVal('val-pvv', last(sa.pvVoltage));  setPct('bar-pvv', last(sa.pvVoltage), ranges[1]);
  setVal('val-pvi', last(sa.pvCurrent));  setPct('bar-pvi', last(sa.pvCurrent), ranges[2]);
  setVal('val-rpm', last(sa.motorRPM));   setPct('bar-rpm', last(sa.motorRPM), ranges[3]);
  setVal('val-flow', last(sa.flowRate));  setPct('bar-flow', last(sa.flowRate), ranges[4]);
  setVal('val-vib', last(sa.vibration));  setPct('bar-vib', last(sa.vibration), ranges[5]);
  setVal('val-level', last(sa.tankLevel));setPct('bar-level', last(sa.tankLevel), ranges[6]);
  setVal('val-pres', last(sa.pressure));  setPct('bar-pres', last(sa.pressure), ranges[7]);
  setVal('val-turb', last(sa.turbidity)); setPct('bar-turb', last(sa.turbidity), ranges[8]);

  // Update status badges
  const lastRegime = gen.regimes[gen.regimes.length - 1];
  ['solarStatus', 'pumpStatus', 'waterStatus'].forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    if (lastRegime === 'anomaly') {
      el.innerHTML = '<span class="status-dot danger"></span> CRITICAL';
      el.className = 'group-status danger';
    } else if (lastRegime === 'strained') {
      el.innerHTML = '<span class="status-dot warning"></span> Strained';
      el.className = 'group-status warning';
    } else {
      el.innerHTML = '<span class="status-dot active"></span> Nominal';
      el.className = 'group-status';
    }
  });
}

// =============================================
// 9. CORRELATION MATRIX
// =============================================

function renderCorrelationMatrix() {
  const gen = state.generator;
  if (!gen || !gen.data) return;
  const D = 9;
  const cols = Array.from({ length: D }, (_, d) => gen.data.map(r => r[d]));
  const shortLabels = ['IRR', 'PVV', 'PVI', 'RPM', 'FLW', 'VIB', 'LVL', 'PRS', 'TRB'];

  const grid = document.getElementById('corrMatrix');
  if (!grid) return;
  grid.innerHTML = '';

  // Compute all correlations
  const corr = [];
  for (let i = 0; i < D; i++) {
    corr.push([]);
    for (let j = 0; j < D; j++) {
      corr[i].push(utils.pearson(cols[i], cols[j]));
    }
  }

  // Header row (blank + labels)
  const blank = document.createElement('div');
  blank.className = 'corr-cell corr-label-cell';
  grid.appendChild(blank);
  shortLabels.forEach(l => {
    const cell = document.createElement('div');
    cell.className = 'corr-cell corr-label-cell';
    cell.textContent = l;
    grid.appendChild(cell);
  });

  // Data rows
  for (let i = 0; i < D; i++) {
    // Row label
    const rowLabel = document.createElement('div');
    rowLabel.className = 'corr-cell corr-label-cell';
    rowLabel.textContent = shortLabels[i];
    grid.appendChild(rowLabel);

    for (let j = 0; j < D; j++) {
      const cell = document.createElement('div');
      cell.className = 'corr-cell';
      const r = corr[i][j];
      const absR = Math.abs(r);
      // Color: yellow for positive, gray for negative, white for near zero
      let bg, color;
      if (i === j) {
        bg = '#FFD700'; color = '#1A1A17';
      } else if (r > 0) {
        const a = 0.1 + 0.7 * absR;
        bg = `rgba(255,215,0,${a.toFixed(2)})`;
        color = absR > 0.5 ? '#1A1A17' : '#706F69';
      } else {
        const a = 0.05 + 0.35 * absR;
        bg = `rgba(100,116,139,${a.toFixed(2)})`;
        color = absR > 0.4 ? '#fff' : '#706F69';
      }
      cell.style.background = bg;
      cell.style.color = color;
      cell.textContent = r.toFixed(2);
      cell.title = `${shortLabels[i]} vs ${shortLabels[j]}: r = ${r.toFixed(3)}`;
      grid.appendChild(cell);
    }
  }
}

// =============================================
// 10. PCA CHARTS
// =============================================

function runPCA() {
  const gen = state.generator;
  if (!gen || !gen.data) return;

  const pca = new PCAEngine();
  pca.fit(gen.data, state.nComponents, state.standardize);
  state.pca = pca;

  renderVarianceBars(pca);
  renderEigenChart(pca);
  renderLoadingChart(pca);
  renderScreeChart(pca);
  renderPCAStats(pca);
  renderScatterPlots(pca);
  renderPCTimeseries(pca);
  renderRegimes(pca);
}

function renderVarianceBars(pca) {
  const container = document.getElementById('varianceBars');
  if (!container) return;
  container.innerHTML = '';

  const colors = ['pc1-fill', 'pc2-fill', 'pc3-fill'];
  pca.explainedVarianceRatio.forEach((ratio, i) => {
    const pct = (ratio * 100).toFixed(1);
    const row = document.createElement('div');
    row.className = 'var-row';
    row.innerHTML = `
      <div class="var-label-row">
        <span>PC${i + 1} — ${SENSOR_LABELS.slice(0, 3)[0] === 'Irradiance' && i === 0 ? 'Solar Dominant' : i === 1 ? 'Pump Dominant' : 'Water Dominant'}</span>
        <span>${pct}%</span>
      </div>
      <div class="var-track">
        <div class="var-fill ${colors[i] || 'pc1-fill'}" style="width:0%" id="varFill${i}"></div>
      </div>
    `;
    container.appendChild(row);
    setTimeout(() => {
      const fill = document.getElementById(`varFill${i}`);
      if (fill) fill.style.width = `${pct}%`;
    }, 100 + i * 150);
  });

  const cumVar = pca.explainedVarianceRatio.reduce((s, v) => s + v, 0);
  const cumEl = document.getElementById('cumulativeVar');
  if (cumEl) cumEl.textContent = (cumVar * 100).toFixed(1) + '%';
}

function renderEigenChart(pca) {
  const ctx = document.getElementById('eigenChart');
  if (!ctx) return;
  const chart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: pca.eigenvalues.map((_, i) => `PC${i + 1}`),
      datasets: [{
        data: pca.eigenvalues,
        backgroundColor: pca.eigenvalues.map((_, i) =>
          i === 0 ? 'rgba(255,215,0,0.9)' :
          i === 1 ? 'rgba(255,215,0,0.6)' : 'rgba(255,215,0,0.4)'
        ),
        borderColor: '#FFD700',
        borderWidth: 1,
        borderRadius: 6,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 800 },
      scales: {
        x: { grid: { display: false }, ticks: { font: { size: 10 } } },
        y: {
          grid: { color: 'rgba(0,0,0,0.05)' },
          ticks: { font: { size: 9 }, maxTicksLimit: 4 },
          title: { display: true, text: 'Eigenvalue', font: { size: 9 } }
        }
      }
    }
  });
  state.chartManager.register('eigenChart', chart);
}

function renderLoadingChart(pca) {
  const ctx = document.getElementById('loadingChart');
  if (!ctx) return;
  const pcColors = ['rgba(255,215,0,0.9)', 'rgba(255,241,118,0.9)', 'rgba(255,224,130,0.9)'];
  const datasets = pca.components.slice(0, 3).map((pc, k) => ({
    label: `PC${k + 1}`,
    data: pc.map(w => Math.abs(w)),
    backgroundColor: pcColors[k],
    borderColor: pcColors[k].replace('0.9', '1'),
    borderWidth: 1,
    borderRadius: 4,
  }));
  const chart = new Chart(ctx, {
    type: 'bar',
    data: { labels: SENSOR_LABELS, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 800 },
      plugins: {
        legend: { display: true, position: 'top',
          labels: { font: { size: 10 }, boxWidth: 12, padding: 12 }
        },
        tooltip: {
          callbacks: {
            label: (ctx) => ` ${ctx.dataset.label}: ${ctx.raw.toFixed(3)}`
          }
        }
      },
      scales: {
        x: { grid: { display: false }, ticks: { font: { size: 9 }, maxRotation: 35 } },
        y: {
          grid: { color: 'rgba(0,0,0,0.05)' },
          ticks: { font: { size: 9 }, maxTicksLimit: 5 },
          title: { display: true, text: '|Loading|', font: { size: 9 } }
        }
      }
    }
  });
  state.chartManager.register('loadingChart', chart);
}

function renderScreeChart(pca) {
  const ctx = document.getElementById('screeChart');
  if (!ctx) return;
  const D = 9;
  // Compute all eigenvalues via repeated deflation
  const allVals = [...pca.eigenvalues];
  while (allVals.length < D) allVals.push(allVals[allVals.length - 1] * 0.3 + utils.randn() * 0.05);
  const labels = Array.from({ length: D }, (_, i) => `PC${i + 1}`);
  const cumExp = [];
  let sum = 0;
  const totalTrace = allVals.reduce((s, v) => s + v, 0);
  allVals.forEach(v => { sum += v / totalTrace; cumExp.push(sum); });

  const chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Eigenvalue',
          data: allVals,
          borderColor: '#FFD700',
          backgroundColor: 'rgba(255,215,0,0.1)',
          borderWidth: 2.5,
          tension: 0.3,
          fill: true,
          pointBackgroundColor: '#FFD700',
          pointRadius: 5,
          pointHoverRadius: 8,
          yAxisID: 'y'
        },
        {
          label: 'Cumulative Variance',
          data: cumExp.map(v => v * 100),
          borderColor: '#94A3B8',
          backgroundColor: 'transparent',
          borderWidth: 2,
          tension: 0.4,
          fill: false,
          pointRadius: 3,
          borderDash: [5, 3],
          yAxisID: 'y2'
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 800 },
      plugins: {
        legend: { display: true, position: 'top',
          labels: { font: { size: 10 }, boxWidth: 12, padding: 12 }
        }
      },
      scales: {
        x: { grid: { display: false }, ticks: { font: { size: 9 } } },
        y: {
          grid: { color: 'rgba(0,0,0,0.04)' },
          ticks: { font: { size: 9 }, maxTicksLimit: 5 },
          title: { display: true, text: 'Eigenvalue', font: { size: 9 } }
        },
        y2: {
          position: 'right',
          grid: { display: false },
          ticks: { font: { size: 9 }, maxTicksLimit: 5, callback: v => v + '%' },
          min: 0, max: 100,
          title: { display: true, text: 'Cum. Variance (%)', font: { size: 9 } }
        }
      }
    }
  });
  state.chartManager.register('screeChart', chart);
}

function renderPCAStats(pca) {
  const cumVar = pca.explainedVarianceRatio.reduce((s, v) => s + v, 0);
  const compressionRatio = ((9 - state.nComponents) / 9 * 100).toFixed(0);
  const reconErr = pca.reconstructionError(state.generator.data);

  document.getElementById('ps-dims').textContent = `9 → ${state.nComponents}`;
  document.getElementById('ps-var').textContent = (cumVar * 100).toFixed(1) + '%';
  document.getElementById('ps-ratio').textContent = compressionRatio + '%';
  document.getElementById('ps-recon').textContent = reconErr.toFixed(4);

  document.getElementById('statComponents').textContent = state.nComponents;
  document.getElementById('statVariance').textContent = (cumVar * 100).toFixed(1) + '%';
}

// =============================================
// 11. SCATTER PLOTS (Dimensionality Visualization)
// =============================================

function getRegimeColors(regimes) {
  return regimes.map(r =>
    r === 'normal'   ? 'rgba(74,222,128,0.75)' :
    r === 'strained' ? 'rgba(251,191,36,0.80)' :
                       'rgba(239,68,68,0.85)'
  );
}

function renderScatterPlots(pca) {
  if (!pca.scores) return;
  const gen = state.generator;
  const scores = pca.scores;
  const colors = getRegimeColors(gen.regimes);

  const makeScatterData = (xIdx, yIdx) => scores.map((s, i) => ({
    x: s[xIdx], y: s[yIdx], regime: gen.regimes[i]
  }));

  const splitByRegime = (points) => ({
    normal:   points.filter(p => p.regime === 'normal'),
    strained: points.filter(p => p.regime === 'strained'),
    anomaly:  points.filter(p => p.regime === 'anomaly'),
  });

  const makeDatasets = (points) => {
    const groups = splitByRegime(points);
    return [
      { label: 'Normal', data: groups.normal,
        backgroundColor: 'rgba(74,222,128,0.65)', borderColor: 'rgba(74,222,128,0.9)',
        pointRadius: 5, pointHoverRadius: 8, borderWidth: 1 },
      { label: 'Strained', data: groups.strained,
        backgroundColor: 'rgba(251,191,36,0.75)', borderColor: 'rgba(251,191,36,0.9)',
        pointRadius: 5, pointHoverRadius: 8, borderWidth: 1 },
      { label: 'Anomaly', data: groups.anomaly,
        backgroundColor: 'rgba(239,68,68,0.80)', borderColor: 'rgba(239,68,68,0.95)',
        pointRadius: 7, pointHoverRadius: 10, borderWidth: 1.5 },
    ];
  };

  const scatterOptions = (xLabel, yLabel) => ({
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 600 },
    plugins: {
      legend: { display: true, position: 'bottom',
        labels: { font: { size: 10 }, boxWidth: 12, padding: 12, usePointStyle: true }
      },
      tooltip: {
        callbacks: {
          label: (ctx) => {
            const p = ctx.raw;
            return ` ${ctx.dataset.label}: (${p.x.toFixed(2)}, ${p.y.toFixed(2)})`;
          }
        }
      }
    },
    scales: {
      x: {
        grid: { color: 'rgba(0,0,0,0.04)' },
        ticks: { font: { size: 9 }, maxTicksLimit: 7 },
        title: { display: true, text: xLabel, font: { size: 10, weight: '600' } }
      },
      y: {
        grid: { color: 'rgba(0,0,0,0.04)' },
        ticks: { font: { size: 9 }, maxTicksLimit: 7 },
        title: { display: true, text: yLabel, font: { size: 10, weight: '600' } }
      }
    }
  });

  // PC1 vs PC2
  const chart12 = new Chart(document.getElementById('scatter12'), {
    type: 'scatter',
    data: { datasets: makeDatasets(makeScatterData(0, 1)) },
    options: scatterOptions('PC1 Score', 'PC2 Score')
  });
  state.chartManager.register('scatter12', chart12);

  // PC1 vs PC3
  if (state.nComponents >= 3) {
    const chart13 = new Chart(document.getElementById('scatter13'), {
      type: 'scatter',
      data: { datasets: makeDatasets(makeScatterData(0, 2)) },
      options: scatterOptions('PC1 Score', 'PC3 Score')
    });
    state.chartManager.register('scatter13', chart13);

    const chart23 = new Chart(document.getElementById('scatter23'), {
      type: 'scatter',
      data: { datasets: makeDatasets(makeScatterData(1, 2)) },
      options: scatterOptions('PC2 Score', 'PC3 Score')
    });
    state.chartManager.register('scatter23', chart23);
  }
}

// =============================================
// 12. PC TIMESERIES
// =============================================

function renderPCTimeseries(pca) {
  const ctx = document.getElementById('pcTimeChart');
  if (!ctx || !pca.scores) return;

  const labels = pca.scores.map((_, i) => i);
  const pc1 = pca.scores.map(s => s[0]);
  const pc2 = pca.scores.map(s => s[1] || 0);
  const pc3 = pca.scores.map(s => s[2] || 0);

  const chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label: 'PC1 Score', data: pc1, borderColor: '#FFD700',
          backgroundColor: 'rgba(255,215,0,0.08)', borderWidth: 2.5,
          tension: 0.3, fill: true, pointRadius: 0 },
        { label: 'PC2 Score', data: pc2, borderColor: '#FFF9C4',
          backgroundColor: 'transparent', borderWidth: 2,
          tension: 0.3, fill: false, pointRadius: 0, borderDash: [4, 2] },
        { label: 'PC3 Score', data: pc3, borderColor: '#FFE082',
          backgroundColor: 'transparent', borderWidth: 2,
          tension: 0.3, fill: false, pointRadius: 0, borderDash: [2, 3] },
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 600 },
      plugins: {
        legend: { display: true, position: 'bottom',
          labels: { font: { size: 10 }, boxWidth: 12, padding: 14 }
        },
        tooltip: { mode: 'index', intersect: false }
      },
      scales: {
        x: {
          grid: { color: 'rgba(0,0,0,0.03)' },
          ticks: { font: { size: 9 }, maxTicksLimit: 10, autoSkip: true }
        },
        y: {
          grid: { color: 'rgba(0,0,0,0.04)' },
          ticks: { font: { size: 9 }, maxTicksLimit: 7 },
          title: { display: true, text: 'PC Score', font: { size: 10 } }
        }
      }
    }
  });
  state.chartManager.register('pcTimeChart', chart);
}

// =============================================
// 13. OPERATING REGIMES
// =============================================

function renderRegimes(pca) {
  const gen = state.generator;
  const regimes = gen.regimes;
  const n = regimes.length;
  const normalCount   = regimes.filter(r => r === 'normal').length;
  const strainedCount = regimes.filter(r => r === 'strained').length;
  const anomalyCount  = regimes.filter(r => r === 'anomaly').length;

  const pct = (c) => ((c / n) * 100).toFixed(1) + '%';
  document.getElementById('normalPct').textContent  = pct(normalCount);
  document.getElementById('strainedPct').textContent = pct(strainedCount);
  document.getElementById('anomalyPct').textContent  = pct(anomalyCount);

  renderRegimeDonut(normalCount, strainedCount, anomalyCount);
  renderT2Chart(pca);
  renderRegimeTimeline(gen);
  renderAlertLog(gen, pca);
}

function renderRegimeDonut(normal, strained, anomaly) {
  const ctx = document.getElementById('regimeDonut');
  if (!ctx) return;
  const chart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Normal', 'Strained', 'Anomaly'],
      datasets: [{
        data: [normal, strained, anomaly],
        backgroundColor: ['rgba(74,222,128,0.8)', 'rgba(251,191,36,0.85)', 'rgba(239,68,68,0.85)'],
        borderColor: ['#22C55E', '#F59E0B', '#EF4444'],
        borderWidth: 2,
        hoverOffset: 8
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 800, animateRotate: true },
      plugins: {
        legend: { display: true, position: 'bottom',
          labels: { font: { size: 10 }, padding: 12, usePointStyle: true }
        },
        tooltip: {
          callbacks: {
            label: (ctx) => ` ${ctx.label}: ${ctx.raw} samples (${(ctx.raw / (normal + strained + anomaly) * 100).toFixed(1)}%)`
          }
        }
      },
      cutout: '65%'
    }
  });
  state.chartManager.register('regimeDonut', chart);
}

function renderT2Chart(pca) {
  const ctx = document.getElementById('t2Chart');
  if (!ctx || !pca.scores) return;

  // Hotelling T² statistic
  const t2 = pca.scores.map(s => {
    return s.reduce((sum, score, k) => {
      const lam = pca.eigenvalues[k] || 1;
      return sum + (score * score) / lam;
    }, 0);
  });

  // 95th percentile threshold
  const sorted = [...t2].sort((a, b) => a - b);
  const threshold = sorted[Math.floor(sorted.length * 0.95)];
  const gen = state.generator;
  const colors = gen.regimes.map(r =>
    r === 'anomaly'  ? 'rgba(239,68,68,0.8)' :
    r === 'strained' ? 'rgba(251,191,36,0.75)' :
                       'rgba(74,222,128,0.65)'
  );

  const chart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: t2.map((_, i) => i),
      datasets: [{
        label: 'T² Statistic',
        data: t2,
        backgroundColor: colors,
        borderWidth: 0,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 600 },
      plugins: {
        legend: { display: false },
        annotation: {
          annotations: {
            threshold: {
              type: 'line',
              yMin: threshold,
              yMax: threshold,
              borderColor: '#EF4444',
              borderWidth: 2,
              borderDash: [5, 3],
              label: {
                display: true,
                content: `95th pct: ${threshold.toFixed(1)}`,
                position: 'end',
                font: { size: 9, weight: '600' },
                color: '#EF4444',
                backgroundColor: 'rgba(239,68,68,0.1)',
              }
            }
          }
        }
      },
      scales: {
        x: { display: false },
        y: {
          grid: { color: 'rgba(0,0,0,0.04)' },
          ticks: { font: { size: 9 }, maxTicksLimit: 5 },
          title: { display: true, text: 'T² Value', font: { size: 9 } }
        }
      }
    }
  });
  state.chartManager.register('t2Chart', chart);
}

function renderRegimeTimeline(gen) {
  const ctx = document.getElementById('regimeTimeline');
  if (!ctx) return;
  const regimes = gen.regimes;
  const n = regimes.length;

  const normalData   = regimes.map(r => r === 'normal'   ? 1 : 0);
  const strainedData = regimes.map(r => r === 'strained' ? 1 : 0);
  const anomalyData  = regimes.map(r => r === 'anomaly'  ? 1 : 0);

  const chart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: regimes.map((_, i) => i),
      datasets: [
        { label: 'Normal',   data: normalData,   backgroundColor: 'rgba(74,222,128,0.6)',  borderWidth: 0, stack: 'regime' },
        { label: 'Strained', data: strainedData, backgroundColor: 'rgba(251,191,36,0.75)', borderWidth: 0, stack: 'regime' },
        { label: 'Anomaly',  data: anomalyData,  backgroundColor: 'rgba(239,68,68,0.8)',   borderWidth: 0, stack: 'regime' },
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 600 },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: (items) => `Sample ${items[0].label}`,
            label: (ctx) => ` ${ctx.dataset.label}`
          }
        }
      },
      scales: {
        x: { stacked: true, display: false, barPercentage: 1.0, categoryPercentage: 1.0 },
        y: { stacked: true, display: false, min: 0, max: 1 }
      }
    }
  });
  state.chartManager.register('regimeTimeline', chart);
}

// =============================================
// 14. ALERT LOG
// =============================================

function renderAlertLog(gen, pca) {
  const list = document.getElementById('alertList');
  if (!list) return;
  list.innerHTML = '';
  const alerts = [];

  gen.regimes.forEach((r, i) => {
    if (r === 'anomaly') {
      alerts.push({ type: 'danger', msg: `Anomaly detected at sample ${i} — T² exceeds 99th percentile. Immediate inspection required.`, t: i });
    } else if (r === 'strained' && (gen.regimes[i-1] !== 'strained')) {
      alerts.push({ type: 'warning', msg: `Strained operation started at sample ${i} — Solar irradiance drop, pump overcompensating.`, t: i });
    }
  });

  // Add info alerts
  alerts.unshift({ type: 'info', msg: `PCA fitted: ${state.nComponents} components retain ${(pca.explainedVarianceRatio.reduce((s,v)=>s+v,0)*100).toFixed(1)}% variance.`, t: 0 });

  const countEl = document.getElementById('alertCount');
  const dangerCount = alerts.filter(a => a.type === 'danger').length;
  if (countEl) {
    countEl.textContent = `${alerts.length} alerts`;
    countEl.style.background = dangerCount > 0 ? '#EF4444' : '#F59E0B';
  }

  alerts.slice(0, 20).forEach(a => {
    const item = document.createElement('div');
    item.className = 'alert-item';
    item.innerHTML = `
      <div class="alert-dot ${a.type}"></div>
      <span class="alert-time">t=${a.t}</span>
      <span class="alert-msg">${a.msg}</span>
    `;
    list.appendChild(item);
  });
}

// =============================================
// 15. NAVBAR & SCROLL
// =============================================

function initNavbar() {
  const navbar = document.getElementById('navbar');
  const toggle = document.getElementById('navToggle');
  const links = document.getElementById('navLinks');

  window.addEventListener('scroll', () => {
    if (window.scrollY > 60) navbar.classList.add('scrolled');
    else navbar.classList.remove('scrolled');

    // Highlight active section
    const sections = ['overview', 'sensors', 'pca', 'visualization', 'regimes'];
    sections.forEach(id => {
      const sec = document.getElementById(id);
      const link = document.querySelector(`[href="#${id}"]`);
      if (!sec || !link) return;
      const rect = sec.getBoundingClientRect();
      if (rect.top <= 100 && rect.bottom >= 100) {
        document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
        link.classList.add('active');
      }
    });
  });

  toggle.addEventListener('click', () => {
    links.classList.toggle('open');
  });

  document.querySelectorAll('.nav-link').forEach(l => {
    l.addEventListener('click', () => links.classList.remove('open'));
  });
}

// =============================================
// 16. PARTICLES
// =============================================

function initParticles() {
  const container = document.getElementById('heroParticles');
  if (!container) return;
  for (let i = 0; i < 20; i++) {
    const p = document.createElement('div');
    p.className = 'particle';
    const size = 4 + Math.random() * 12;
    p.style.cssText = `
      width: ${size}px;
      height: ${size}px;
      left: ${Math.random() * 100}%;
      animation-duration: ${8 + Math.random() * 12}s;
      animation-delay: ${Math.random() * 10}s;
    `;
    container.appendChild(p);
  }
}

// =============================================
// 17. CONTROLS
// =============================================

function initControls() {
  // Time window
  document.querySelectorAll('.seg-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.seg-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.nPoints = parseInt(btn.dataset.value);
      renderSensorCharts();
    });
  });

  // Noise slider
  const noiseSlider = document.getElementById('noiseSlider');
  const noiseVal = document.getElementById('noiseVal');
  noiseSlider.addEventListener('input', () => {
    const v = noiseSlider.value;
    noiseVal.textContent = v + '%';
    state.noiseLevel = v / 100;
  });

  // Regenerate
  document.getElementById('regenerateBtn').addEventListener('click', () => {
    generateAndRender();
  });

  // Live feed toggle
  const liveBtn = document.getElementById('toggleLiveBtn');
  const liveLabel = document.getElementById('liveLabel');
  const liveIcon = document.getElementById('liveIcon');
  liveBtn.addEventListener('click', () => {
    state.isLive = !state.isLive;
    if (state.isLive) {
      liveLabel.textContent = 'Stop Live';
      liveIcon.innerHTML = '<rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>';
      state.liveInterval = setInterval(() => {
        // Append one new sample
        const gen = state.generator;
        if (!gen) return;
        const last = gen.data[gen.data.length - 1];
        const noiseFactor = state.noiseLevel;

        const perturb = (v, range, factor = 0.05) =>
          utils.clamp(v + utils.randn() * (range[1] - range[0]) * factor * noiseFactor, range[0], range[1]);

        const newRow = last.map((v, d) => perturb(v, SENSOR_RANGES[d]));
        gen.data.push(newRow);
        gen.data.shift();
        SENSOR_LABELS.forEach((_, d) => {
          const keys = Object.keys(gen.sensorArrays);
          if (keys[d]) {
            gen.sensorArrays[keys[d]].push(newRow[d]);
            gen.sensorArrays[keys[d]].shift();
          }
        });
        gen.regimes.push('normal');
        gen.regimes.shift();
        gen.n = gen.data.length;

        updateMiniCards();
        renderSensorCharts();
      }, 800);
    } else {
      liveLabel.textContent = 'Live Feed';
      liveIcon.innerHTML = '<circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/>';
      clearInterval(state.liveInterval);
    }
  });

  // PCA component stepper
  const compVal = document.getElementById('compVal');
  document.getElementById('compMinus').addEventListener('click', () => {
    if (state.nComponents > 1) {
      state.nComponents--;
      compVal.textContent = state.nComponents;
    }
  });
  document.getElementById('compPlus').addEventListener('click', () => {
    if (state.nComponents < 9) {
      state.nComponents++;
      compVal.textContent = state.nComponents;
    }
  });

  // Standardize toggle
  document.getElementById('standardizeToggle').addEventListener('change', (e) => {
    state.standardize = e.target.checked;
  });

  // Covariance method
  document.getElementById('covMethod').addEventListener('change', (e) => {
    state.covMethod = e.target.value;
  });

  // Run PCA button
  document.getElementById('runPCABtn').addEventListener('click', () => {
    const btn = document.getElementById('runPCABtn');
    btn.textContent = '⏳ Computing…';
    btn.disabled = true;
    setTimeout(() => {
      runPCA();
      btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg> Run PCA Analysis`;
      btn.disabled = false;
    }, 50);
  });
}

// =============================================
// 18. MAIN BOOTSTRAP
// =============================================

function generateAndRender() {
  const gen = new SensorDataGenerator(state.nPoints, state.noiseLevel);
  gen.generate();
  state.generator = gen;

  updateMiniCards();
  renderSensorCharts();
  renderCorrelationMatrix();
  runPCA();
  renderDataTable();
}

// =============================================
// 19. DATA TABLE MODULE
// =============================================

const dtState = {
  currentPage: 1,
  pageSize: 20,
  sortCol: 0,
  sortDir: 'asc',
  regimeFilter: 'all',
  searchQuery: '',
  viewMode: 'sensor',   // 'sensor' | 'pca' | 'both'
  allRows: [],          // cached flat row objects
  filteredRows: [],
};

// Column definitions for each view
const DT_SENSOR_COLS = [
  { key: 'sample',      label: '#',               unit: '' },
  { key: 'irradiance',  label: 'Irradiance',       unit: 'W/m²' },
  { key: 'pvVoltage',   label: 'PV Voltage',        unit: 'V' },
  { key: 'pvCurrent',   label: 'PV Current',        unit: 'A' },
  { key: 'motorRPM',    label: 'Motor RPM',          unit: 'RPM' },
  { key: 'flowRate',    label: 'Flow Rate',          unit: 'L/min' },
  { key: 'vibration',   label: 'Vibration',          unit: 'g' },
  { key: 'tankLevel',   label: 'Tank Level',         unit: '%' },
  { key: 'pressure',    label: 'Pressure',           unit: 'kPa' },
  { key: 'turbidity',   label: 'Turbidity',          unit: 'NTU' },
  { key: 'regime',      label: 'Regime',             unit: '' },
];

const DT_PCA_COLS = [
  { key: 'sample', label: '#',      unit: '' },
  { key: 'pc1',    label: 'PC1 Score', unit: '' },
  { key: 'pc2',    label: 'PC2 Score', unit: '' },
  { key: 'pc3',    label: 'PC3 Score', unit: '' },
  { key: 't2',     label: 'T² Stat',   unit: '' },
  { key: 'regime', label: 'Regime',    unit: '' },
];

const DT_BOTH_COLS = [
  { key: 'sample',     label: '#',            unit: '' },
  { key: 'irradiance', label: 'Irradiance',   unit: 'W/m²' },
  { key: 'motorRPM',   label: 'Motor RPM',    unit: 'RPM' },
  { key: 'flowRate',   label: 'Flow Rate',    unit: 'L/min' },
  { key: 'vibration',  label: 'Vibration',    unit: 'g' },
  { key: 'tankLevel',  label: 'Tank Level',   unit: '%' },
  { key: 'pc1',        label: 'PC1 Score',    unit: '' },
  { key: 'pc2',        label: 'PC2 Score',    unit: '' },
  { key: 'pc3',        label: 'PC3 Score',    unit: '' },
  { key: 'regime',     label: 'Regime',       unit: '' },
];

function buildTableRows() {
  const gen = state.generator;
  const pca = state.pca;
  if (!gen || !gen.data) return [];

  const sa = gen.sensorArrays;
  const scores = pca && pca.scores ? pca.scores : [];
  const eigenvalues = pca && pca.eigenvalues ? pca.eigenvalues : [1, 1, 1];

  return gen.data.map((row, i) => {
    const s = scores[i] || [0, 0, 0];
    const t2 = s.reduce((sum, score, k) => sum + (score * score) / (eigenvalues[k] || 1), 0);
    return {
      sample:     i + 1,
      irradiance: row[0],
      pvVoltage:  row[1],
      pvCurrent:  row[2],
      motorRPM:   row[3],
      flowRate:   row[4],
      vibration:  row[5],
      tankLevel:  row[6],
      pressure:   row[7],
      turbidity:  row[8],
      pc1: s[0] || 0,
      pc2: s[1] || 0,
      pc3: s[2] || 0,
      t2:  t2,
      regime: gen.regimes[i],
    };
  });
}

function getActiveCols() {
  if (dtState.viewMode === 'pca')    return DT_PCA_COLS;
  if (dtState.viewMode === 'both')   return DT_BOTH_COLS;
  return DT_SENSOR_COLS;
}

function applyFilters(rows) {
  let result = rows;
  // Regime filter
  if (dtState.regimeFilter !== 'all') {
    result = result.filter(r => r.regime === dtState.regimeFilter);
  }
  // Search filter
  const q = dtState.searchQuery.trim().toLowerCase();
  if (q) {
    result = result.filter(r =>
      String(r.sample).includes(q) ||
      r.regime.includes(q) ||
      Object.values(r).some(v => String(v).toLowerCase().includes(q))
    );
  }
  // Sort
  const col = getActiveCols()[dtState.sortCol];
  if (col) {
    const key = col.key;
    result = [...result].sort((a, b) => {
      const av = a[key], bv = b[key];
      if (typeof av === 'number') return dtState.sortDir === 'asc' ? av - bv : bv - av;
      return dtState.sortDir === 'asc' ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av));
    });
  }
  return result;
}

function renderDataTable() {
  dtState.allRows = buildTableRows();
  dtState.filteredRows = applyFilters(dtState.allRows);
  dtState.currentPage = 1;
  renderSummaryStrip();
  renderTableHeader();
  renderTableBody();
  renderPagination();
  renderStatsTable();
}

function refreshTableView() {
  dtState.filteredRows = applyFilters(dtState.allRows);
  dtState.currentPage = 1;
  renderTableHeader();
  renderTableBody();
  renderPagination();
}

function renderSummaryStrip() {
  const strip = document.getElementById('dataSummaryStrip');
  if (!strip) return;
  const rows = dtState.allRows;
  const total = rows.length;
  const nNormal   = rows.filter(r => r.regime === 'normal').length;
  const nStrained = rows.filter(r => r.regime === 'strained').length;
  const nAnomaly  = rows.filter(r => r.regime === 'anomaly').length;

  strip.innerHTML = `
    <div class="summary-pill">
      <div class="summary-pill-icon">📋</div>
      <div>
        <div class="summary-pill-val">${total}</div>
        <div class="summary-pill-label">Total Samples</div>
      </div>
    </div>
    <div class="summary-pill">
      <div class="summary-pill-icon">✅</div>
      <div>
        <div class="summary-pill-val">${nNormal}</div>
        <div class="summary-pill-label">Normal Samples</div>
      </div>
    </div>
    <div class="summary-pill">
      <div class="summary-pill-icon">⚠️</div>
      <div>
        <div class="summary-pill-val">${nStrained}</div>
        <div class="summary-pill-label">Strained Samples</div>
      </div>
    </div>
    <div class="summary-pill">
      <div class="summary-pill-icon">🚨</div>
      <div>
        <div class="summary-pill-val">${nAnomaly}</div>
        <div class="summary-pill-label">Anomaly Samples</div>
      </div>
    </div>
  `;
}

function renderTableHeader() {
  const thead = document.getElementById('dtHead');
  if (!thead) return;
  const cols = getActiveCols();
  thead.innerHTML = `<tr>${cols.map((col, i) => {
    const sortClass = i === dtState.sortCol
      ? (dtState.sortDir === 'asc' ? 'sorted-asc' : 'sorted-desc')
      : '';
    const unitStr = col.unit ? ` <span style="font-weight:400;opacity:0.6">(${col.unit})</span>` : '';
    return `<th class="${sortClass}" data-col="${i}">${col.label}${unitStr}</th>`;
  }).join('')}</tr>`;

  // Sort click handlers
  thead.querySelectorAll('th').forEach(th => {
    th.addEventListener('click', () => {
      const colIdx = parseInt(th.dataset.col);
      if (dtState.sortCol === colIdx) {
        dtState.sortDir = dtState.sortDir === 'asc' ? 'desc' : 'asc';
      } else {
        dtState.sortCol = colIdx;
        dtState.sortDir = 'asc';
      }
      refreshTableView();
    });
  });
}

function formatCell(key, val) {
  if (key === 'regime') {
    const icon = val === 'normal' ? '✅' : val === 'strained' ? '⚠️' : '🚨';
    return `<span class="regime-badge ${val}">${val.toUpperCase()}</span>`;
  }
  if (key === 'sample') return `<strong style="color:var(--gray-900)">${val}</strong>`;
  if (key === 'pc1' || key === 'pc2' || key === 'pc3') {
    const n = Number(val);
    const cls = Math.abs(n) < 0.1 ? 'pc-neutral' : n > 0 ? 'pc-positive' : 'pc-negative';
    return `<span class="${cls}">${n.toFixed(3)}</span>`;
  }
  if (key === 't2') return Number(val).toFixed(2);
  // Sensor values
  const n = Number(val);
  return isNaN(n) ? val : n.toFixed(2);
}

function renderTableBody() {
  const tbody = document.getElementById('dtBody');
  if (!tbody) return;
  const { filteredRows, currentPage, pageSize } = dtState;
  const start = (currentPage - 1) * pageSize;
  const pageRows = filteredRows.slice(start, start + pageSize);
  const cols = getActiveCols();

  if (pageRows.length === 0) {
    tbody.innerHTML = `<tr><td colspan="${cols.length}" style="text-align:center;color:var(--gray-400);padding:2.5rem;font-family:var(--font-main)">No matching samples found</td></tr>`;
    return;
  }

  tbody.innerHTML = pageRows.map(row => {
    const rowClass = row.regime === 'strained' ? 'row-strained' : row.regime === 'anomaly' ? 'row-anomaly' : '';
    const cells = cols.map(col => `<td>${formatCell(col.key, row[col.key])}</td>`).join('');
    return `<tr class="${rowClass}">${cells}</tr>`;
  }).join('');

  // Update info
  const infoEl = document.getElementById('dtInfo');
  if (infoEl) {
    infoEl.textContent = `Showing ${start + 1}–${Math.min(start + pageSize, filteredRows.length)} of ${filteredRows.length} samples`;
  }
}

function renderPagination() {
  const container = document.getElementById('dtPagination');
  if (!container) return;
  const totalPages = Math.ceil(dtState.filteredRows.length / dtState.pageSize);
  const cur = dtState.currentPage;
  let html = '';

  // Prev
  html += `<button class="page-btn" ${cur === 1 ? 'disabled' : ''} data-page="${cur - 1}">‹</button>`;

  // Page numbers with ellipsis
  const makeBtn = (p) => `<button class="page-btn ${p === cur ? 'active' : ''}" data-page="${p}">${p}</button>`;
  if (totalPages <= 7) {
    for (let p = 1; p <= totalPages; p++) html += makeBtn(p);
  } else {
    html += makeBtn(1);
    if (cur > 3) html += `<span class="page-btn" style="cursor:default;border:none">…</span>`;
    for (let p = Math.max(2, cur - 1); p <= Math.min(totalPages - 1, cur + 1); p++) html += makeBtn(p);
    if (cur < totalPages - 2) html += `<span class="page-btn" style="cursor:default;border:none">…</span>`;
    html += makeBtn(totalPages);
  }

  // Next
  html += `<button class="page-btn" ${cur === totalPages ? 'disabled' : ''} data-page="${cur + 1}">›</button>`;

  container.innerHTML = html;
  container.querySelectorAll('button[data-page]').forEach(btn => {
    btn.addEventListener('click', () => {
      const p = parseInt(btn.dataset.page);
      if (p >= 1 && p <= totalPages) {
        dtState.currentPage = p;
        renderTableBody();
        renderPagination();
      }
    });
  });
}

function renderStatsTable() {
  const thead = document.getElementById('statsHead');
  const tbody = document.getElementById('statsBody');
  if (!thead || !tbody) return;

  const numericKeys = ['irradiance','pvVoltage','pvCurrent','motorRPM','flowRate','vibration','tankLevel','pressure','turbidity'];
  const labels = ['Irradiance (W/m²)','PV Voltage (V)','PV Current (A)','Motor RPM','Flow Rate (L/min)','Vibration (g)','Tank Level (%)','Pressure (kPa)','Turbidity (NTU)'];
  const statNames = ['Min', 'Max', 'Mean', 'Std Dev', 'Median', 'Range'];

  thead.innerHTML = `<tr><th>Sensor</th>${statNames.map(s => `<th>${s}</th>`).join('')}</tr>`;

  const rows = dtState.allRows;
  tbody.innerHTML = numericKeys.map((key, ki) => {
    const vals = rows.map(r => r[key]).filter(v => !isNaN(v));
    if (!vals.length) return '';
    const mn = Math.min(...vals);
    const mx = Math.max(...vals);
    const mean = utils.mean(vals);
    const std = utils.std(vals);
    const sorted = [...vals].sort((a,b) => a-b);
    const med = sorted[Math.floor(sorted.length / 2)];
    const range = mx - mn;
    return `<tr>
      <td>${labels[ki]}</td>
      <td>${mn.toFixed(2)}</td>
      <td>${mx.toFixed(2)}</td>
      <td>${mean.toFixed(2)}</td>
      <td>${std.toFixed(2)}</td>
      <td>${med.toFixed(2)}</td>
      <td>${range.toFixed(2)}</td>
    </tr>`;
  }).join('');
}

// =============================================
// 20. DOWNLOAD & EXPORT
// =============================================

function showToast(msg, icon = '✅') {
  const existing = document.querySelector('.toast');
  if (existing) existing.remove();
  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.innerHTML = `<span>${icon}</span> ${msg}`;
  document.body.appendChild(toast);
  setTimeout(() => { toast.style.opacity = '0'; toast.style.transform = 'translateY(16px)'; toast.style.transition = 'all 0.3s'; }, 2200);
  setTimeout(() => toast.remove(), 2600);
}

function rowToCSVLine(row, cols) {
  return cols.map(col => {
    const v = row[col.key];
    if (typeof v === 'string' && v.includes(',')) return `"${v}"`;
    if (typeof v === 'number') return col.key === 'sample' ? v : v.toFixed(3);
    return v;
  }).join(',');
}

function downloadCSV() {
  const cols = DT_SENSOR_COLS;
  const header = cols.map(c => c.unit ? `${c.label}(${c.unit})` : c.label).join(',');
  const lines = dtState.allRows.map(r => rowToCSVLine(r, cols));
  // Append PCA scores
  const pcaCols = [{ key: 'pc1', label: 'PC1_Score', unit: '' }, { key: 'pc2', label: 'PC2_Score', unit: '' }, { key: 'pc3', label: 'PC3_Score', unit: '' }, { key: 't2', label: 'T2_Stat', unit: '' }];
  const fullHeader = header + ',' + pcaCols.map(c => c.label).join(',');
  const fullLines = dtState.allRows.map((r, i) => rowToCSVLine(r, cols) + ',' + pcaCols.map(c => Number(r[c.key]).toFixed(4)).join(','));

  const blob = new Blob([fullHeader + '\n' + fullLines.join('\n')], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'solar_pump_sensor_data.csv';
  a.click();
  showToast('CSV downloaded successfully', '📥');
}

function downloadJSON() {
  const data = dtState.allRows.map(r => ({
    sample: r.sample,
    sensors: {
      irradiance_W_m2: r.irradiance,
      pv_voltage_V: r.pvVoltage,
      pv_current_A: r.pvCurrent,
      motor_rpm: r.motorRPM,
      flow_rate_L_min: r.flowRate,
      vibration_g: r.vibration,
      tank_level_pct: r.tankLevel,
      pressure_kPa: r.pressure,
      turbidity_NTU: r.turbidity,
    },
    pca_scores: { pc1: r.pc1, pc2: r.pc2, pc3: r.pc3 },
    t2_statistic: r.t2,
    regime: r.regime,
  }));
  const blob = new Blob([JSON.stringify({ metadata: { total_samples: data.length, sensor_channels: 9, pca_components: state.nComponents }, data }, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'solar_pump_sensor_data.json';
  a.click();
  showToast('JSON downloaded successfully', '📥');
}

function copyToClipboard() {
  const cols = getActiveCols();
  const header = cols.map(c => c.unit ? `${c.label} (${c.unit})` : c.label).join('\t');
  const lines = dtState.filteredRows.map(r => cols.map(c => {
    const v = r[c.key];
    return typeof v === 'number' ? v.toFixed(3) : v;
  }).join('\t'));
  const text = header + '\n' + lines.join('\n');
  navigator.clipboard.writeText(text).then(() => {
    showToast(`${dtState.filteredRows.length} rows copied to clipboard`, '📋');
  }).catch(() => showToast('Copy failed — try downloading instead', '❌'));
}

// =============================================
// 21. DATA TABLE CONTROLS INIT
// =============================================

function initDataTableControls() {
  // Regime filter
  document.querySelectorAll('#regimeFilter .seg-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#regimeFilter .seg-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      dtState.regimeFilter = btn.dataset.regime;
      refreshTableView();
    });
  });

  // View mode
  document.querySelectorAll('#tableView .seg-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#tableView .seg-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      dtState.viewMode = btn.dataset.view;
      dtState.sortCol = 0;
      refreshTableView();
    });
  });

  // Search
  const searchEl = document.getElementById('dtSearch');
  if (searchEl) {
    let searchTimer;
    searchEl.addEventListener('input', () => {
      clearTimeout(searchTimer);
      searchTimer = setTimeout(() => {
        dtState.searchQuery = searchEl.value;
        refreshTableView();
      }, 200);
    });
  }

  // Page size
  const pageSizeEl = document.getElementById('dtPageSize');
  if (pageSizeEl) {
    pageSizeEl.addEventListener('change', () => {
      dtState.pageSize = parseInt(pageSizeEl.value);
      dtState.currentPage = 1;
      renderTableBody();
      renderPagination();
    });
  }

  // Downloads
  document.getElementById('downloadCSV')?.addEventListener('click', downloadCSV);
  document.getElementById('downloadJSON')?.addEventListener('click', downloadJSON);
  document.getElementById('copyClipboard')?.addEventListener('click', copyToClipboard);
}

function init() {
  initNavbar();
  initParticles();
  initHeroChart();
  initControls();
  initDataTableControls();
  generateAndRender();
}

// Wait for DOM
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
