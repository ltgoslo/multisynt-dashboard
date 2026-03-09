// Multilingual Training Progress Dashboard
// ==========================================

let DATA = null;
let currentLang = null; // set on init from first language
let currentShot = "0";
let currentTaskSelection = "__filtered__";
let currentPromptAgg = "max";
let currentNormalization = "baseline";
let currentMetric = null;
let checkedTasks = new Set();
const showStderr = true;
let showPromptDeviation = true;

// HPLT-E quality filter state
let filterCriteria = {
  monotonicity: {
    enabled: true, minStep: 5000, maxStep: 50000, threshold: 0.5, direction: ">=",
    label: "Monotonicity", description: "Spearman \u03C1 (step vs. score)",
    tooltip: "Spearman rank correlation between checkpoint step number and benchmark score. Measures whether performance improves monotonically during training. Default threshold: \u2265 0.5.",
  },
  snr: {
    enabled: false, minStep: 5000, maxStep: 50000, threshold: 3.0, direction: ">=",
    label: "Signal-to-noise ratio (SNR)", description: "Signal-to-noise ratio",
    tooltip: "Ratio of mean signal (score minus random baseline) to mean prompt standard deviation across checkpoints. Measures whether the benchmark signal is distinguishable from prompt-induced noise. Note: unlike the original HPLT-E implementation, the random baseline is subtracted from the signal so that chance-level performance yields SNR \u2248 0. Default threshold: \u2265 3.",
  },
  cv: {
    enabled: false, minStep: 5000, maxStep: 50000, threshold: 15.0, direction: "<=",
    label: "Stable pretraining (CV)", description: "Coefficient of variation (%)",
    tooltip: "Standard deviation divided by mean score across checkpoints, as percentage. Measures score stability during training, following the original HPLT-E implementation. Default threshold: \u2264 15%.",
  },
  mad: {
    enabled: false, minStep: 5000, maxStep: 50000, threshold: 5.0, direction: "<=",
    label: "Prompt sensitivity (MAD)", description: "Median MAD across prompts",
    tooltip: "Median Absolute Deviation of scores across prompt variants, taken as the median over all checkpoints. Default threshold: \u2264 5.",
  },
  consistency: {
    enabled: false, minStep: 5000, maxStep: 50000, threshold: 0.5, direction: ">=",
    label: "Ranking consistency", description: "Kendall \u03C4 (model rankings)",
    tooltip: "Average Kendall\u2019s Tau correlation of model rankings between successive checkpoints. Measures whether the relative ordering of models is preserved across training. Default threshold: \u2265 0.5.",
  },
  promptSwitch: {
    enabled: false, minStep: 5000, maxStep: 50000, threshold: 20.0, direction: "<=",
    label: "Prompt-switch rate", description: "Best-prompt change rate (%)",
    tooltip: "Fraction of checkpoints where the best-performing prompt variant changes, as a percentage.",
  },
  nonRandom: {
    enabled: false, minStep: 5000, maxStep: 50000, threshold: 5.0, direction: ">=",
    label: "Non-randomness", description: "Max score \u2212 random baseline",
    tooltip: "Difference between the maximum score and the task\u2019s random baseline. Verifies the model learned beyond chance. Default threshold: \u2265 5.",
  },
};
let filterResults = {}; // {bench: {criterionName: {value, pass}}}
let allFilterBenchmarks = new Set();

// Per-language default filter criteria overrides (keys not listed use the global defaults above)
const FILTER_DEFAULTS = {
  French: {
    monotonicity: { enabled: true, minStep: 1000, maxStep: 45000, threshold: 0.25 },
    snr: { enabled: true, minStep: 5000, maxStep: 45000, threshold: 1 },
    cv: { enabled: false, minStep: 5000, maxStep: 50000, threshold: 15 },
    mad: { enabled: false, minStep: 5000, maxStep: 50000, threshold: 5 },
    consistency: { enabled: true, minStep: 5000, maxStep: 45000, threshold: 0.5 },
    promptSwitch: { enabled: false, minStep: 5000, maxStep: 50000, threshold: 20 },
    nonRandom: { enabled: true, minStep: 5000, maxStep: 45000, threshold: 5 },
  },
  Spanish: {
    monotonicity: { enabled: true, minStep: 1000, maxStep: 45000, threshold: 0.25 },
    snr: { enabled: true, minStep: 5000, maxStep: 45000, threshold: 1 },
    cv: { enabled: false, minStep: 5000, maxStep: 50000, threshold: 15 },
    mad: { enabled: true, minStep: 5000, maxStep: 45000, threshold: 5 },
    consistency: { enabled: true, minStep: 5000, maxStep: 45000, threshold: 0 },
    promptSwitch: { enabled: false, minStep: 5000, maxStep: 50000, threshold: 20 },
    nonRandom: { enabled: true, minStep: 5000, maxStep: 45000, threshold: 5 },
  },
};

function applyFilterDefaults(lang) {
  const defaults = FILTER_DEFAULTS[lang];
  if (!defaults) return;
  for (const [name, vals] of Object.entries(defaults)) {
    if (!filterCriteria[name]) continue;
    const cfg = filterCriteria[name];
    cfg.enabled = vals.enabled;
    cfg.minStep = vals.minStep;
    cfg.maxStep = vals.maxStep;
    cfg.threshold = vals.threshold;
  }
}

const MODEL_COLORS = [
  "#6366f1", "#f43f5e", "#10b981", "#f59e0b", "#8b5cf6",
  "#06b6d4", "#ec4899", "#84cc16", "#14b8a6", "#f97316",
];

const METRIC_DISPLAY = {
  acc: "accuracy", acc_norm: "accuracy (normalized)", f1: "F1",
  em: "exact match", exact_match: "exact match", exact: "exact match",
  fscore: "F-score", bleu: "BLEU", bleu_max: "BLEU (best ref.)",
  bleu_avg: "BLEU (avg ref.)", bleu_acc: "BLEU accuracy", chrf: "chrF",
  rougeL_max: "ROUGE-L (best ref.)", rougeL_avg: "ROUGE-L (avg ref.)",
  rougeL_acc: "ROUGE-L accuracy", rouge1_max: "ROUGE-1 (best ref.)",
  rouge1_acc: "ROUGE-1 accuracy", rouge2_max: "ROUGE-2 (best ref.)",
  rouge2_acc: "ROUGE-2 accuracy", mcc: "MCC",
};

const METRIC_SCALES = {
  acc: "unit", acc_norm: "unit", f1: "unit", em: "unit", exact: "unit",
  exact_match: "unit", fscore: "unit", bleu_acc: "unit", rougeL_acc: "unit",
  rouge1_acc: "unit", rouge2_acc: "unit", mcc: "unit",
  bleu: "percent", bleu_max: "percent", bleu_avg: "percent",
  chrf: "percent", rougeL_max: "percent", rougeL_avg: "percent",
  rouge1_max: "percent", rouge2_max: "percent",
};

const JSON_DOWNLOAD_ICON = {
  width: 24, height: 24,
  path: "M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z",
};

function exportChartDataAsJSON(gd) {
  const exportData = {
    metadata: {
      language: currentLang, shot: currentShot + "-shot",
      task_selection: currentTaskSelection,
      prompt_aggregation: currentPromptAgg,
      normalization: currentNormalization,
      title: gd.layout.title?.text || "",
      exported_at: new Date().toISOString(),
    },
    series: [],
  };
  for (const trace of gd.data) {
    if (trace.fill === "toself") continue;
    exportData.series.push({
      name: trace.name || null,
      x: Array.from(trace.x),
      y: Array.from(trace.y),
      ...(trace.error_y?.array ? { error: Array.from(trace.error_y.array) } : {}),
    });
  }
  const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "chart-data.json";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

const PLOTLY_CONFIG = {
  responsive: true, displaylogo: false,
  modeBarButtons: [[
    {
      name: "Download plot as PNG", icon: Plotly.Icons.camera,
      click: (gd) => Plotly.downloadImage(gd, { format: "png", width: 1600, height: 900, scale: 3, filename: "chart" }),
    },
    {
      name: "Download plot as SVG", icon: Plotly.Icons.camera,
      click: (gd) => Plotly.downloadImage(gd, { format: "svg", width: 1600, height: 900, filename: "chart" }),
    },
  ], [{
    name: "Export data as JSON", icon: JSON_DOWNLOAD_ICON,
    click: (gd) => exportChartDataAsJSON(gd),
  }]],
};

// ============================================================
// Data access helpers
// ============================================================

function getLangData() {
  return DATA.languages[currentLang];
}

function getMetricsSetup() {
  return getLangData().metrics_setup;
}

function getModels() {
  return getLangData().models;
}

function hexToRgb(hex) {
  return [parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16), parseInt(hex.slice(5, 7), 16)];
}

function hexToRgba(hex, alpha) {
  const [r, g, b] = hexToRgb(hex);
  return `rgba(${r},${g},${b},${alpha})`;
}

// ============================================================
// Score access (prompt aggregation aware)
// ============================================================

function getScore(progressData, step, bench, shot, metric) {
  metric = metric || getMetricsSetup()[bench]?.main_metric;
  const obj = progressData[step]?.[bench]?.[shot]?.[metric];
  if (obj === undefined || obj === null) return undefined;
  if (typeof obj === "number") return currentPromptAgg === "stdev" ? 0 : obj;
  if (currentPromptAgg === "stdev") return obj.prompt_sd != null ? obj.prompt_sd : undefined;
  return obj[currentPromptAgg];
}

function getStderr(progressData, step, bench, shot, metric) {
  metric = metric || getMetricsSetup()[bench]?.main_metric;
  const obj = progressData[step]?.[bench]?.[shot]?.[metric];
  if (!obj || typeof obj === "number") return undefined;
  const se = obj[currentPromptAgg + "_stderr"];
  return (se !== undefined && se !== null) ? se : undefined;
}

function getPromptSE(progressData, step, bench, shot, metric) {
  metric = metric || getMetricsSetup()[bench]?.main_metric;
  const obj = progressData[step]?.[bench]?.[shot]?.[metric];
  if (!obj || typeof obj === "number") return undefined;
  const sd = obj.prompt_sd, n = obj.n_prompts;
  if (sd == null || n == null || n < 2) return undefined;
  return sd / Math.sqrt(n);
}

function getCombinedSE(progressData, step, bench, shot, metric) {
  if (currentPromptAgg === "stdev") return undefined;
  const sampSe = showStderr ? getStderr(progressData, step, bench, shot, metric) : undefined;
  const promptSe = showPromptDeviation ? getPromptSE(progressData, step, bench, shot, metric) : undefined;
  if (sampSe == null && promptSe == null) return undefined;
  return Math.sqrt((sampSe || 0) ** 2 + (promptSe || 0) ** 2);
}

function toDisplayScale(value, benchmark, metric) {
  const base = metric || null;
  const scale = base ? (METRIC_SCALES[base] || "unit") : getMetricsSetup()[benchmark].metric_scale;
  return scale === "unit" ? value * 100 : value;
}

function scaleStderr(se, benchmark, metric) {
  if (se == null) return undefined;
  return toDisplayScale(se, benchmark, metric);
}

// ============================================================
// Normalization
// ============================================================

function baselineNorm(raw, benchmark) {
  const info = getMetricsSetup()[benchmark];
  const base = info.random_baseline;
  return Math.max(0, toDisplayScale(raw, benchmark) - toDisplayScale(base, benchmark));
}

function applyNorm(raw, benchmark, allRaw, metric) {
  if (currentPromptAgg === "stdev") return toDisplayScale(raw, benchmark, metric);
  if (currentNormalization === "none") return toDisplayScale(raw, benchmark, metric);
  if (currentNormalization === "baseline") return baselineNorm(raw, benchmark);
  return toDisplayScale(raw, benchmark, metric);
}

function getNormYLabel() {
  if (currentPromptAgg === "stdev") return "prompt stdev (0\u2013100)";
  if (currentNormalization === "baseline") return "score \u2212 random baseline";
  return "score (0\u2013100)";
}

function getMetricYLabel(benchmark, metric) {
  const m = metric || getMetricsSetup()[benchmark].main_metric;
  return METRIC_DISPLAY[m] || m;
}

function autoSetNormalization() {
  currentNormalization = "baseline";
  document.getElementById("norm-select").value = currentNormalization;
}

// ============================================================
// Selection helpers
// ============================================================

function isAggregateSelection(sel) {
  return sel === "__all__" || sel === "__all_macro__" || sel === "__custom__" || sel === "__filtered__" || sel.startsWith("__cat__");
}

function getBenchmarksForSelection(sel) {
  const ms = getMetricsSetup();
  if (sel === "__all__" || sel === "__all_macro__" || sel === "__filtered__") return Object.keys(ms);
  if (sel === "__custom__") return [];
  if (sel.startsWith("__cat__")) {
    const c = sel.slice(7);
    return Object.keys(ms).filter((b) => ms[b].category === c);
  }
  if (ms[sel]) return [sel];
  return [];
}

function isMacroSelection() {
  return currentTaskSelection === "__all_macro__";
}

function getMacroGroups(benchmarks) {
  const benchSet = benchmarks instanceof Set ? benchmarks : new Set(benchmarks);
  const categoryGroups = {};
  const ms = getMetricsSetup();
  for (const bench of benchSet) {
    const info = ms[bench];
    if (!info) continue;
    const cat = info.category;
    if (!categoryGroups[cat]) categoryGroups[cat] = [];
    categoryGroups[cat].push(bench);
  }
  return Object.values(categoryGroups);
}

function aggregateScores(benchmarks, scoreFn, macro) {
  if (macro) {
    const groups = getMacroGroups(benchmarks);
    let groupSum = 0, groupCount = 0, groupSe2 = 0;
    for (const group of groups) {
      let sum = 0, count = 0, se2 = 0;
      for (const bench of group) {
        const r = scoreFn(bench);
        if (r === undefined) continue;
        const s = (typeof r === "number") ? r : r.score;
        if (s === undefined) continue;
        sum += s; count++;
        const se = (typeof r === "object" && r.stderr != null) ? r.stderr : 0;
        se2 += se * se;
      }
      if (count > 0) {
        groupSum += sum / count;
        groupSe2 += se2 / (count * count);
        groupCount++;
      }
    }
    if (groupCount === 0) return null;
    return { score: groupSum / groupCount, count: groupCount, stderr: Math.sqrt(groupSe2) / groupCount };
  } else {
    let sum = 0, count = 0, se2 = 0;
    for (const bench of benchmarks) {
      const r = scoreFn(bench);
      if (r === undefined) continue;
      const s = (typeof r === "number") ? r : r.score;
      if (s === undefined) continue;
      sum += s; count++;
      const se = (typeof r === "object" && r.stderr != null) ? r.stderr : 0;
      se2 += se * se;
    }
    if (count === 0) return null;
    return { score: sum / count, count, stderr: Math.sqrt(se2) / count };
  }
}

// ============================================================
// Metric selector
// ============================================================

function getEffectiveMetric(benchmark) {
  return currentMetric || getMetricsSetup()[benchmark]?.main_metric;
}

function populateMetricSelector(benchmarks) {
  const select = document.getElementById("metric-select");
  const control = document.getElementById("metric-control");
  if (!select || !control) return;
  const ms = getMetricsSetup();
  let metrics = null;
  for (const bench of benchmarks) {
    const info = ms[bench];
    if (!info || !info.available_metrics) continue;
    const set = new Set(info.available_metrics);
    metrics = metrics ? new Set([...metrics].filter((m) => set.has(m))) : set;
  }
  if (!metrics || metrics.size <= 1) { hideMetricSelector(); return; }
  const mainMetric = ms[benchmarks[0]]?.main_metric;
  select.innerHTML = "";
  const ordered = [];
  if (mainMetric && metrics.has(mainMetric)) ordered.push(mainMetric);
  for (const m of [...metrics].sort()) if (m !== mainMetric) ordered.push(m);
  for (const m of ordered) {
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = (METRIC_DISPLAY[m] || m) + (m === mainMetric ? " (default)" : "");
    select.appendChild(opt);
  }
  if (currentMetric && metrics.has(currentMetric)) select.value = currentMetric;
  else { currentMetric = mainMetric; select.value = mainMetric; }
  control.style.display = "";
}

function hideMetricSelector() {
  const control = document.getElementById("metric-control");
  if (control) control.style.display = "none";
  currentMetric = null;
}

// ============================================================
// Initialization
// ============================================================

async function init() {
  try {
    const response = await fetch("data.json");
    DATA = await response.json();
    const languages = Object.keys(DATA.languages);
    if (languages.length === 0) throw new Error("No languages found in data");
    currentLang = languages[0];
    applyFilterDefaults(currentLang);
    buildTabs(languages);
    const hasURLState = restoreStateFromURL();
    checkedTasks = new Set(Object.keys(getMetricsSetup()));
    populateTaskDropdown();
    bindEventListeners();
    if (hasURLState) {
      syncUIFromState();
    } else {
      currentTaskSelection = "__filtered__";
      document.getElementById("task-select").value = "__filtered__";
    }
    buildCheckboxes();
    allFilterBenchmarks = new Set(Object.keys(getMetricsSetup()));
    autoSetNormalization();
    renderChart();
    pushStateToURL();
  } catch (err) {
    console.error("init failed:", err);
    const el = document.getElementById("chart");
    if (el) el.innerHTML = "<pre style='color:red;padding:1rem;'>" + err.stack + "</pre>";
  }
}

function buildTabs(languages) {
  const nav = document.getElementById("tab-nav");
  nav.innerHTML = "";
  languages.forEach((lang, i) => {
    const btn = document.createElement("button");
    btn.className = "tab-btn" + (i === 0 ? " active" : "");
    btn.dataset.lang = lang;
    btn.textContent = lang;
    nav.appendChild(btn);
  });
}

// ============================================================
// Dropdown
// ============================================================

function populateTaskDropdown() {
  const select = document.getElementById("task-select");
  const ms = getMetricsSetup();
  // Remove old optgroups (keep the first 4 static options)
  while (select.children.length > 4) select.removeChild(select.lastChild);

  // Category aggregates
  const categories = {};
  for (const [bench, info] of Object.entries(ms)) {
    const cat = info.category;
    if (!categories[cat]) categories[cat] = [];
    categories[cat].push(bench);
  }
  if (Object.keys(categories).length > 1) {
    const catGroup = document.createElement("optgroup");
    catGroup.label = "Aggregate by Category";
    for (const catName of Object.keys(categories).sort()) {
      const opt = document.createElement("option");
      opt.value = "__cat__" + catName;
      opt.textContent = catName;
      catGroup.appendChild(opt);
    }
    select.appendChild(catGroup);
  }

  // Individual tasks
  const taskGroup = document.createElement("optgroup");
  taskGroup.label = "Individual Tasks";
  const entries = Object.entries(ms).map(([bench, info]) => ({ value: bench, label: info.pretty_name }));
  entries.sort((a, b) => a.label.localeCompare(b.label));
  for (const entry of entries) {
    const opt = document.createElement("option");
    opt.value = entry.value;
    opt.textContent = entry.label;
    taskGroup.appendChild(opt);
  }
  select.appendChild(taskGroup);
}

// ============================================================
// Event listeners
// ============================================================

function bindEventListeners() {
  document.getElementById("tab-nav").addEventListener("click", (e) => {
    const btn = e.target.closest(".tab-btn");
    if (!btn) return;
    document.querySelector(".tab-btn.active")?.classList.remove("active");
    btn.classList.add("active");
    currentLang = btn.dataset.lang;
    applyFilterDefaults(currentLang);
    // Rebuild UI for new language
    populateTaskDropdown();
    const prevSel = currentTaskSelection;
    // Reset selection if current individual task doesn't exist in new language
    if (!isAggregateSelection(prevSel) && !getMetricsSetup()[prevSel]) {
      currentTaskSelection = "__filtered__";
      document.getElementById("task-select").value = "__filtered__";
    }
    allFilterBenchmarks = new Set(Object.keys(getMetricsSetup()));
    checkedTasks = new Set(Object.keys(getMetricsSetup()));
    buildCheckboxes();
    autoSetNormalization();
    filterPanelRendered = false;
    renderChart();
    pushStateToURL();
  });

  document.querySelectorAll(".shot-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelector(".shot-btn.active").classList.remove("active");
      btn.classList.add("active");
      currentShot = btn.dataset.shot;
      renderChart();
      pushStateToURL();
    });
  });

  document.getElementById("prompt-agg-select").addEventListener("change", (e) => {
    currentPromptAgg = e.target.value;
    renderChart();
    pushStateToURL();
  });

  document.getElementById("norm-select").addEventListener("change", (e) => {
    currentNormalization = e.target.value;
    renderChart();
    pushStateToURL();
  });

  document.getElementById("metric-select").addEventListener("change", (e) => {
    currentMetric = e.target.value;
    renderChart();
    pushStateToURL();
  });

  document.getElementById("prompt-dev-toggle").addEventListener("change", (e) => {
    showPromptDeviation = e.target.checked;
    renderChart();
    pushStateToURL();
  });

  const promptDevLabel = document.getElementById("prompt-dev-label");
  if (promptDevLabel) {
    promptDevLabel.style.cursor = "help";
    promptDevLabel.style.textDecoration = "underline";
    promptDevLabel.style.textDecorationStyle = "dotted";
    promptDevLabel.style.textUnderlineOffset = "3px";
    attachTooltip(promptDevLabel, () => ({
      title: "Prompt uncertainty",
      body: "Include prompt-template uncertainty in the error band. Computed as the standard deviation across prompt variants divided by \u221An, combined in quadrature with the sampling standard error.",
      footer: "",
    }));
  }

  document.getElementById("task-select").addEventListener("change", (e) => {
    currentTaskSelection = e.target.value;
    if (currentTaskSelection === "__filtered__") {
      allFilterBenchmarks = new Set(Object.keys(getMetricsSetup()));
      checkedTasks = new Set(allFilterBenchmarks);
      showFilterUI();
      runFilter();
    } else {
      hideFilterUI();
      const benchmarks = getBenchmarksForSelection(currentTaskSelection);
      if (benchmarks.length > 0) checkedTasks = new Set(benchmarks);
    }
    syncCheckboxStates();
    autoSetNormalization();
    renderChart();
    pushStateToURL();
  });

  document.getElementById("select-all-btn").addEventListener("click", () => {
    if (currentTaskSelection === "__filtered__") {
      allFilterBenchmarks = new Set(Object.keys(getMetricsSetup()));
      syncCheckboxStates();
      runFilter();
      renderChart();
    } else {
      checkedTasks = new Set(Object.keys(getMetricsSetup()));
      currentTaskSelection = "__all__";
      document.getElementById("task-select").value = "__all__";
      syncCheckboxStates();
      autoSetNormalization();
      renderChart();
    }
  });

  document.getElementById("select-none-btn").addEventListener("click", () => {
    if (currentTaskSelection === "__filtered__") {
      allFilterBenchmarks.clear();
      syncCheckboxStates();
      runFilter();
      renderChart();
    } else {
      checkedTasks.clear();
      syncCheckboxStates();
      renderChart();
    }
  });
}

// ============================================================
// Task checkboxes
// ============================================================

function buildCheckboxes() {
  const grid = document.getElementById("checkbox-grid");
  grid.innerHTML = "";
  const ms = getMetricsSetup();
  const grouped = {};
  for (const [bench, info] of Object.entries(ms)) {
    const cat = info.category;
    if (!grouped[cat]) grouped[cat] = [];
    grouped[cat].push(bench);
  }
  for (const cat of Object.keys(grouped).sort()) {
    const catDiv = document.createElement("div");
    catDiv.className = "checkbox-category";
    const catBenches = grouped[cat];
    const headerDiv = document.createElement("div");
    headerDiv.className = "checkbox-category-header";
    const groupCheckbox = document.createElement("input");
    groupCheckbox.type = "checkbox";
    groupCheckbox.dataset.cat = cat;
    const source0 = currentTaskSelection === "__filtered__" ? allFilterBenchmarks : checkedTasks;
    groupCheckbox.checked = catBenches.every((b) => source0.has(b));
    groupCheckbox.indeterminate = !groupCheckbox.checked && catBenches.some((b) => source0.has(b));
    groupCheckbox.addEventListener("change", () => {
      const source = currentTaskSelection === "__filtered__" ? allFilterBenchmarks : checkedTasks;
      for (const b of catBenches) {
        if (groupCheckbox.checked) source.add(b); else source.delete(b);
      }
      syncCheckboxStates();
      onTaskCheckboxChange();
    });
    headerDiv.addEventListener("click", (e) => { if (e.target !== groupCheckbox) groupCheckbox.click(); });
    const h4 = document.createElement("h4");
    h4.textContent = cat;
    headerDiv.appendChild(groupCheckbox);
    headerDiv.appendChild(h4);
    catDiv.appendChild(headerDiv);
    for (const bench of catBenches) {
      const label = document.createElement("label");
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = checkedTasks.has(bench);
      checkbox.dataset.bench = bench;
      checkbox.addEventListener("change", () => {
        if (currentTaskSelection === "__filtered__") {
          if (checkbox.checked) allFilterBenchmarks.add(bench); else allFilterBenchmarks.delete(bench);
        } else {
          if (checkbox.checked) checkedTasks.add(bench); else checkedTasks.delete(bench);
        }
        syncCheckboxStates();
        onTaskCheckboxChange();
      });
      label.appendChild(checkbox);
      label.appendChild(document.createTextNode(" " + (ms[bench].pretty_name || bench)));
      catDiv.appendChild(label);
    }
    grid.appendChild(catDiv);
  }
}

function onTaskCheckboxChange() {
  if (currentTaskSelection === "__filtered__") {
    allFilterBenchmarks = new Set();
    document.querySelectorAll("#checkbox-grid input[data-bench]").forEach((cb) => {
      if (cb.checked) allFilterBenchmarks.add(cb.dataset.bench);
    });
    runFilter();
    renderChart();
    return;
  }
  if (checkedTasks.size === 1) {
    const bench = [...checkedTasks][0];
    currentTaskSelection = bench;
    if (getMetricsSetup()[bench]) document.getElementById("task-select").value = bench;
    autoSetNormalization();
    renderChart();
    return;
  }
  currentTaskSelection = "__custom__";
  document.getElementById("task-select").value = "__custom__";
  autoSetNormalization();
  renderChart();
}

function syncCheckboxStates() {
  const source = currentTaskSelection === "__filtered__" ? allFilterBenchmarks : checkedTasks;
  document.querySelectorAll("#checkbox-grid input[data-bench]").forEach((cb) => {
    cb.checked = source.has(cb.dataset.bench);
  });
  const ms = getMetricsSetup();
  document.querySelectorAll("#checkbox-grid input[data-cat]").forEach((gcb) => {
    const cat = gcb.dataset.cat;
    const catBenches = Object.entries(ms).filter(([, info]) => info.category === cat).map(([b]) => b);
    const allChecked = catBenches.length > 0 && catBenches.every((b) => source.has(b));
    const someChecked = catBenches.some((b) => source.has(b));
    gcb.checked = allChecked;
    gcb.indeterminate = !allChecked && someChecked;
  });
}

// ============================================================
// Tooltip
// ============================================================

let tooltipTimeout = null;

function showTooltip(event, title, body, footer, meta) {
  const tooltip = document.getElementById("custom-tooltip");
  const titleEl = document.getElementById("tooltip-title");
  const metaEl = document.getElementById("tooltip-meta");
  const bodyEl = document.getElementById("tooltip-body");
  const footerEl = document.getElementById("tooltip-footer");
  titleEl.textContent = title || ""; titleEl.style.display = title ? "" : "none";
  metaEl.textContent = meta || ""; metaEl.style.display = meta ? "" : "none";
  bodyEl.textContent = body || ""; bodyEl.style.display = body ? "" : "none";
  footerEl.textContent = footer || ""; footerEl.style.display = footer ? "" : "none";
  positionTooltip(tooltip, event);
  tooltip.classList.add("visible");
}

function hideTooltip() {
  clearTimeout(tooltipTimeout);
  document.getElementById("custom-tooltip").classList.remove("visible");
}

function positionTooltip(tooltip, event) {
  const pad = 12;
  tooltip.style.left = "0px"; tooltip.style.top = "0px";
  tooltip.classList.add("visible");
  const rect = tooltip.getBoundingClientRect();
  let x = event.clientX + pad, y = event.clientY + pad;
  if (x + rect.width > window.innerWidth - pad) x = event.clientX - rect.width - pad;
  if (y + rect.height > window.innerHeight - pad) y = event.clientY - rect.height - pad;
  tooltip.style.left = x + "px"; tooltip.style.top = y + "px";
}

function attachTooltip(element, contentFn) {
  element.addEventListener("mouseenter", (e) => {
    tooltipTimeout = setTimeout(() => { const c = contentFn(); if (c) showTooltip(e, c.title, c.body, c.footer, c.meta); }, 300);
  });
  element.addEventListener("mousemove", (e) => {
    const tooltip = document.getElementById("custom-tooltip");
    if (tooltip.classList.contains("visible")) positionTooltip(tooltip, e);
  });
  element.addEventListener("mouseleave", () => hideTooltip());
}

// ============================================================
// Chart rendering
// ============================================================

const ALL_SHOTS = ["0", "1", "5"];

function getPlotlyLayout(overrides) {
  const result = Object.assign({
    font: { family: "Inter, system-ui, sans-serif", size: 13 },
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    margin: { l: 60, r: 20, t: 50, b: 60 },
    autosize: true, hovermode: "closest",
  }, overrides);
  // Ensure axes have no frame lines and use visible gridlines
  const axisDefaults = { showline: false, zeroline: false, gridcolor: "#d8dce3" };
  result.xaxis = Object.assign({ automargin: true }, axisDefaults, result.xaxis);
  result.yaxis = Object.assign({}, axisDefaults, result.yaxis);
  return result;
}

function plotChart(traces, layout) {
  Plotly.newPlot("chart", traces, layout, PLOTLY_CONFIG);
  const chartEl = document.getElementById("chart");
  chartEl.on("plotly_hover", onChartHover);
  chartEl.on("plotly_unhover", hideTooltip);
}

function onChartHover(data) {
  if (!data.points || !data.points.length) return;
  const pt = data.points[0];
  if (pt.y == null) return;
  const scoreStr = Number(pt.y).toFixed(1);
  const cd = pt.customdata;
  let seStr = "";
  if (cd && typeof cd === "object" && cd.stderr != null && cd.stderr > 0) {
    seStr = " \u00b1 " + Number(cd.stderr).toFixed(1);
  } else if (typeof cd === "number" && cd > 0) {
    seStr = " \u00b1 " + Number(cd).toFixed(1);
  }
  let body;
  if (isAggregateSelection(currentTaskSelection)) {
    const countStr = cd && typeof cd === "object" ? cd.count : "";
    const unit = isMacroSelection() ? "groups" : "tasks";
    body = "Average: " + scoreStr + seStr + (countStr ? " (" + countStr + " " + unit + ")" : "");
  } else {
    body = "Score: " + scoreStr + seStr;
  }
  const title = (pt.data.name || "") + " \u2014 Step " + pt.x;
  showTooltip(data.event, title, body, "", "");
}

function updateDescription() {
  const descEl = document.getElementById("task-description");
  if (!descEl) return;
  const sel = currentTaskSelection;
  if (isAggregateSelection(sel)) {
    descEl.textContent = getAggregateDescription();
    descEl.style.display = "block";
  } else {
    descEl.style.display = "none";
  }
}

function getAggregateDescription() {
  const sel = currentTaskSelection;
  const count = sel === "__custom__" ? checkedTasks.size : getBenchmarksForSelection(sel).filter((b) => checkedTasks.has(b)).length;
  const macro = isMacroSelection();
  let scope = "";
  if (sel === "__all_macro__") {
    const groups = getMacroGroups(checkedTasks);
    scope = "all " + count + " tasks (" + groups.length + " categories, group-averaged)";
  } else if (sel === "__all__") scope = "all " + count + " tasks (task-averaged)";
  else if (sel === "__filtered__") scope = count + " signal-filtered tasks (task-averaged, HPLT-E criteria)";
  else if (sel === "__custom__") scope = count + " selected tasks (task-averaged)";
  else if (sel.startsWith("__cat__")) scope = count + " tasks in the \"" + sel.slice(7) + "\" category";

  const avgDesc = macro
    ? "Scores are first averaged within each task category, then averaged across categories. This gives equal weight to each category regardless of how many tasks it contains. "
    : "";
  const normDescs = {
    none: "Scores are shown on their native metric scales without normalization, then averaged.",
    baseline: "Each task score has the random baseline subtracted (clamped at 0), then averaged across tasks. This accounts for different chance levels across tasks (e.g. 25% for 4-choice QA vs. 50% for binary classification).",
  };
  const normDesc = normDescs[currentNormalization] || "";
  return "Aggregate score across " + scope + ". " + avgDesc + normDesc;
}

function renderChart() {
  const sel = currentTaskSelection;
  // Show/hide metric selector
  if (isAggregateSelection(sel)) {
    hideMetricSelector();
  } else if (getMetricsSetup()[sel]) {
    populateMetricSelector([sel]);
  }
  // Update description
  updateDescription();
  // Show/hide filter
  if (sel === "__filtered__") {
    if (allFilterBenchmarks.size === 0) allFilterBenchmarks = new Set(Object.keys(getMetricsSetup()));
    showFilterUI();
    runFilter();
  } else {
    hideFilterUI();
  }
  if (isAggregateSelection(sel)) renderAggregateProgressChart();
  else renderSingleProgressChart(sel);
}

function makeBandTrace(xValues, yValues, seValues, color) {
  const upper = [], lower = [], xs = [];
  for (let i = 0; i < xValues.length; i++) {
    if (yValues[i] != null && seValues[i] != null) {
      xs.push(xValues[i]);
      upper.push(yValues[i] + seValues[i]);
      lower.push(yValues[i] - seValues[i]);
    }
  }
  if (xs.length === 0) return null;
  return {
    x: xs.concat(xs.slice().reverse()),
    y: upper.concat(lower.slice().reverse()),
    fill: "toself", fillcolor: hexToRgba(color, 0.15),
    line: { color: "transparent" },
    showlegend: false, hoverinfo: "skip",
  };
}

function getModelSteps(modelDir) {
  const progress = getModels()[modelDir].progress;
  return Object.keys(progress)
    .filter((k) => k !== "main" && !isNaN(Number(k)))
    .map(Number)
    .sort((a, b) => a - b);
}

function getAllSteps() {
  const models = getModels();
  const allSteps = new Set();
  for (const m of Object.keys(models)) {
    for (const s of getModelSteps(m)) allSteps.add(s);
  }
  return [...allSteps].sort((a, b) => a - b);
}

function renderAggregateProgressChart() {
  const models = getModels();
  const macro = isMacroSelection();
  const wantSE = showStderr || showPromptDeviation;
  const traces = [];
  const allYValues = [];

  // Compute y-range across all shots for stable axes
  for (const shot of ALL_SHOTS) {
    for (const [modelDir, modelData] of Object.entries(models)) {
      const steps = getModelSteps(modelDir);
      for (const step of steps) {
        const result = aggregateScores(checkedTasks, (bench) => {
          const raw = getScore(modelData.progress, step, bench, shot);
          if (raw === undefined) return undefined;
          return applyNorm(raw, bench, null);
        }, macro);
        if (result) allYValues.push(result.score);
      }
    }
  }
  const yRange = computeYRange(allYValues);

  for (const [modelDir, modelData] of Object.entries(models)) {
    const steps = getModelSteps(modelDir);
    const color = modelData.color || MODEL_COLORS[0];

    const aggResults = steps.map((step) => {
      return aggregateScores(checkedTasks, (bench) => {
        const raw = getScore(modelData.progress, step, bench, currentShot);
        if (raw === undefined) return undefined;
        const score = applyNorm(raw, bench, null);
        const se = wantSE ? scaleStderr(getCombinedSE(modelData.progress, step, bench, currentShot), bench) : undefined;
        return { score, stderr: se };
      }, macro);
    });

    const scores = aggResults.map((r) => r ? r.score : null);
    const aggSes = aggResults.map((r) => r ? r.stderr : null);

    if (wantSE) {
      const band = makeBandTrace(steps, scores, aggSes, color);
      if (band) traces.push(band);
    }
    traces.push({
      x: steps, y: scores, mode: "lines+markers",
      name: modelData.display_name,
      line: { color, width: 2.5 }, marker: { size: 5 },
      customdata: aggResults.map((r) => r ? { count: r.count, stderr: r.stderr } : null),
      hoverinfo: "none",
    });
  }

  const avgLabel = macro ? "group-avg" : "task-avg";
  const taskLabel = currentTaskSelection === "__filtered__"
    ? checkedTasks.size + " signal-filtered tasks"
    : "all tasks";
  const layout = getPlotlyLayout({
    title: { text: currentLang + " \u2014 " + taskLabel + " \u2014 " + avgLabel + " (" + currentShot + "-shot)", font: { size: 16 } },
    xaxis: { title: "training step", dtick: 5000 },
    yaxis: { title: getNormYLabel(), range: yRange },
    legend: {
      x: 0.01, y: 0.99, xanchor: "left", yanchor: "top",
      bgcolor: "rgba(255,255,255,0.8)", bordercolor: "#e2e8f0", borderwidth: 1,
    },
  });
  plotChart(traces, layout);
}

function renderSingleProgressChart(benchmark) {
  const info = getMetricsSetup()[benchmark];
  if (!info) return;
  const metric = getEffectiveMetric(benchmark);
  const models = getModels();
  const wantSE = showStderr || showPromptDeviation;
  const traces = [];
  const allYValues = [];

  // y-range across all shots
  for (const shot of ALL_SHOTS) {
    for (const [modelDir, modelData] of Object.entries(models)) {
      for (const step of getModelSteps(modelDir)) {
        const raw = getScore(modelData.progress, step, benchmark, shot, metric);
        if (raw != null) {
          allYValues.push(currentNormalization === "none"
            ? toDisplayScale(raw, benchmark, metric)
            : applyNorm(raw, benchmark, null, metric));
        }
      }
    }
  }
  const yRange = currentNormalization === "none" ? [0, computeYMax(allYValues)] : computeYRange(allYValues);

  for (const [modelDir, modelData] of Object.entries(models)) {
    const steps = getModelSteps(modelDir);
    const color = modelData.color || MODEL_COLORS[0];
    const ys = steps.map((s) => {
      const raw = getScore(modelData.progress, s, benchmark, currentShot, metric);
      if (raw == null) return null;
      return currentNormalization === "none"
        ? toDisplayScale(raw, benchmark, metric)
        : applyNorm(raw, benchmark, null, metric);
    });
    const ses = wantSE ? steps.map((s) => {
      const se = getCombinedSE(modelData.progress, s, benchmark, currentShot, metric);
      return scaleStderr(se, benchmark, metric);
    }) : null;
    if (wantSE && ses) {
      const band = makeBandTrace(steps, ys, ses, color);
      if (band) traces.push(band);
    }
    traces.push({
      x: steps, y: ys, mode: "lines+markers",
      name: modelData.display_name,
      line: { color, width: 2.5 }, marker: { size: 5 },
      customdata: ses || ys.map(() => null),
      hoverinfo: "none",
    });
  }

  const yLabel = currentPromptAgg === "stdev" ? getNormYLabel() : (currentNormalization === "none" ? getMetricYLabel(benchmark, metric) : getNormYLabel());
  const layout = getPlotlyLayout({
    title: { text: currentLang + " \u2014 " + info.pretty_name + " (" + currentShot + "-shot)", font: { size: 16 } },
    xaxis: { title: "training step", dtick: 5000 },
    yaxis: { title: yLabel, range: yRange },
    legend: {
      x: 0.01, y: 0.99, xanchor: "left", yanchor: "top",
      bgcolor: "rgba(255,255,255,0.8)", bordercolor: "#e2e8f0", borderwidth: 1,
    },
  });
  plotChart(traces, layout);
}

// ============================================================
// Y-axis helpers
// ============================================================

function computeYRange(values) {
  if (!values.length) return [0, 100];
  const mx = Math.max(...values);
  return [0, Math.min(mx + Math.max(mx * 0.15, 2), 115)];
}

function computeYMax(values) {
  if (!values.length) return 100;
  const mx = Math.max(...values);
  return Math.min(mx + Math.max(mx * 0.15, 2), 115);
}

// ============================================================
// HPLT-E Quality Filter
// ============================================================

function computeSpearmanRank(x, y) {
  if (x.length < 3) return null;
  const n = x.length;
  function rankArray(arr) {
    const sorted = arr.map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
    const ranks = new Array(n);
    let i = 0;
    while (i < n) {
      let j = i;
      while (j < n - 1 && sorted[j + 1].v === sorted[j].v) j++;
      const avgRank = (i + j) / 2 + 1;
      for (let k = i; k <= j; k++) ranks[sorted[k].i] = avgRank;
      i = j + 1;
    }
    return ranks;
  }
  const rx = rankArray(x), ry = rankArray(y);
  const mx = rx.reduce((a, b) => a + b, 0) / n;
  const my = ry.reduce((a, b) => a + b, 0) / n;
  let num = 0, dx2 = 0, dy2 = 0;
  for (let i = 0; i < n; i++) {
    const dx = rx[i] - mx, dy = ry[i] - my;
    num += dx * dy; dx2 += dx * dx; dy2 += dy * dy;
  }
  const denom = Math.sqrt(dx2 * dy2);
  return denom === 0 ? 0 : num / denom;
}

function computeKendallTau(x, y) {
  const n = x.length;
  if (n < 2) return null;
  let concordant = 0, discordant = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const dx = x[i] - x[j], dy = y[i] - y[j];
      if (dx * dy > 0) concordant++;
      else if (dx * dy < 0) discordant++;
    }
  }
  const pairs = n * (n - 1) / 2;
  return pairs === 0 ? 0 : (concordant - discordant) / pairs;
}

function evaluateThreshold(criterionName, value, threshold) {
  if (value === null || value === undefined) return null;
  const lowerIsBetter = { cv: true, mad: true, promptSwitch: true };
  return lowerIsBetter[criterionName] ? value <= threshold : value >= threshold;
}

function getNoiseScaleFactor(benchmark) {
  const info = getMetricsSetup()[benchmark];
  if (currentNormalization === "none") return info.metric_scale === "unit" ? 100 : 1;
  if (currentNormalization === "baseline") {
    const range = info.max_performance - info.random_baseline;
    return range === 0 ? 0 : 100 / range;
  }
  return info.metric_scale === "unit" ? 100 : 1;
}

function medianOf(arr) {
  if (arr.length === 0) return null;
  const s = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 === 0 ? (s[mid - 1] + s[mid]) / 2 : s[mid];
}

function filterDisplayScale(rawScore, benchmark) {
  // Convert raw score to percentage scale (0-100) without baseline normalization,
  // matching HPLT-E which always computes filter criteria on raw percentage scores.
  const info = getMetricsSetup()[benchmark];
  return info.metric_scale === "unit" ? rawScore * 100 : rawScore;
}

function filterNoiseScale(benchmark) {
  // Scale factor to convert raw prompt_sd/prompt_mad to percentage scale
  const info = getMetricsSetup()[benchmark];
  return info.metric_scale === "unit" ? 100 : 1;
}

function computeFilterCriteriaForBench(benchmark, shot) {
  // Compute each criterion per model, then take the median across models (HPLT-E approach)
  // All criteria are computed on raw percentage-scale scores (not baseline-normalized)
  const models = getModels();
  const modelDirs = Object.keys(models);
  const info = getMetricsSetup()[benchmark];
  if (!info) return {};
  const mainMetric = info.main_metric;
  const noiseFactor = filterNoiseScale(benchmark);
  const results = {};

  for (const [name, cfg] of Object.entries(filterCriteria)) {
    // Ordering consistency is a cross-model criterion (ranks models at each step)
    if (name === "consistency") {
      const windowSteps = getAllSteps().filter((s) => s >= cfg.minStep && s <= cfg.maxStep);
      if (windowSteps.length < 2 || modelDirs.length < 2) {
        results[name] = { value: null, pass: null };
        continue;
      }
      const stepRankings = [];
      for (const step of windowSteps) {
        const scores = [];
        let valid = true;
        for (const modelDir of modelDirs) {
          const obj = models[modelDir].progress[String(step)]?.[benchmark]?.[shot]?.[mainMetric];
          if (!obj) { valid = false; break; }
          const rawScore = obj[currentPromptAgg];
          if (rawScore == null) { valid = false; break; }
          scores.push(filterDisplayScale(rawScore, benchmark));
        }
        if (valid) stepRankings.push(scores);
      }
      if (stepRankings.length < 2) {
        results[name] = { value: null, pass: null };
        continue;
      }
      const taus = [];
      for (let i = 0; i < stepRankings.length - 1; i++) {
        const tau = computeKendallTau(stepRankings[i], stepRankings[i + 1]);
        taus.push(tau != null ? tau : 0);
      }
      const value = taus.reduce((a, b) => a + b, 0) / taus.length;
      results[name] = { value, pass: evaluateThreshold(name, value, cfg.threshold) };
      continue;
    }

    const perModelValues = [];
    for (const modelDir of modelDirs) {
      const progressData = models[modelDir].progress;
      const steps = Object.keys(progressData)
        .filter((k) => k !== "main" && !isNaN(Number(k)))
        .map(Number).sort((a, b) => a - b);
      const windowSteps = steps.filter((s) => s >= cfg.minStep && s <= cfg.maxStep);
      const pairs = [];
      for (const s of windowSteps) {
        const obj = progressData[String(s)]?.[benchmark]?.[shot]?.[mainMetric];
        if (!obj) continue;
        const rawScore = obj[currentPromptAgg];
        if (rawScore == null) continue;
        const score = filterDisplayScale(rawScore, benchmark);
        pairs.push({ step: s, score, rawScore, obj });
      }
      if (pairs.length < 3 && name !== "nonRandom") continue;
      if (pairs.length < 1) continue;

      let value = null;
      switch (name) {
        case "monotonicity":
          value = computeSpearmanRank(pairs.map((p) => p.step), pairs.map((p) => p.score));
          break;
        case "snr": {
          const scores = pairs.map((p) => p.score);
          const noiseVals = currentPromptAgg === "median"
            ? pairs.map((p) => 1.4826 * (p.obj.prompt_mad || 0) * noiseFactor)
            : pairs.map((p) => (p.obj.prompt_sd || 0) * noiseFactor);
          const baseline = filterDisplayScale(info.random_baseline || 0, benchmark);
          const meanSignal = Math.max(0, scores.reduce((a, b) => a + b, 0) / scores.length - baseline);
          const meanNoise = noiseVals.reduce((a, b) => a + b, 0) / noiseVals.length;
          value = meanNoise > 1e-10 ? meanSignal / (meanNoise + 1e-8) : (meanSignal > 0 ? Infinity : 0);
          break;
        }
        case "cv": {
          const scores = pairs.map((p) => p.score);
          const n = scores.length;
          const mean = scores.reduce((a, b) => a + b, 0) / n;
          const stdDev = n > 1 ? Math.sqrt(scores.reduce((s, v) => s + (v - mean) ** 2, 0) / (n - 1)) : 0;
          value = Math.abs(mean) > 1e-10 ? (stdDev / Math.abs(mean)) * 100 : (stdDev > 0 ? Infinity : 0);
          break;
        }
        case "mad": {
          const mads = pairs.map((p) => p.obj.prompt_mad).filter((v) => v != null);
          if (mads.length === 0) break;
          const scaled = mads.map((m) => m * noiseFactor).sort((a, b) => a - b);
          value = scaled.length % 2 === 0
            ? (scaled[scaled.length / 2 - 1] + scaled[scaled.length / 2]) / 2
            : scaled[Math.floor(scaled.length / 2)];
          break;
        }
        case "promptSwitch": {
          const indices = pairs.map((p) => p.obj.max_prompt_idx);
          if (indices.some((v) => v == null)) break;
          let switches = 0;
          for (let i = 1; i < indices.length; i++) if (indices[i] !== indices[i - 1]) switches++;
          value = indices.length > 1 ? (switches / (indices.length - 1)) * 100 : 0;
          break;
        }
        case "nonRandom": {
          const scores = pairs.map((p) => p.score);
          const maxScore = Math.max(...scores);
          const baseline = filterDisplayScale(info.random_baseline || 0, benchmark);
          value = Math.max(0, maxScore - baseline);
          break;
        }
      }
      if (value != null) perModelValues.push(value);
    }

    const medianValue = medianOf(perModelValues);
    results[name] = { value: medianValue, pass: evaluateThreshold(name, medianValue, cfg.threshold) };
  }
  return results;
}

function runFilter() {
  if (currentTaskSelection !== "__filtered__") return;

  // Compute criteria per benchmark as the median across all models (HPLT-E approach)
  filterResults = {};
  for (const bench of allFilterBenchmarks) {
    filterResults[bench] = computeFilterCriteriaForBench(bench, currentShot);
  }

  checkedTasks = new Set();
  for (const bench of allFilterBenchmarks) {
    let passes = true;
    const res = filterResults[bench] || {};
    for (const [name, cfg] of Object.entries(filterCriteria)) {
      if (!cfg.enabled) continue;
      const r = res[name];
      if (r && r.pass === false) { passes = false; break; }
    }
    if (passes) checkedTasks.add(bench);
  }

  syncCheckboxStates();
  renderFilterTable();
}

function renderFilterPanel() {
  const grid = document.getElementById("filter-criteria-grid");
  if (!grid) return;
  grid.innerHTML = "";
  const criterionOrder = ["monotonicity", "snr", "cv", "mad", "consistency", "promptSwitch", "nonRandom"];
  for (const name of criterionOrder) {
    const cfg = filterCriteria[name];
    const card = document.createElement("div");
    card.className = "filter-criterion" + (cfg.enabled ? "" : " disabled");
    const header = document.createElement("div");
    header.className = "filter-criterion-header";
    const cb = document.createElement("input");
    cb.type = "checkbox"; cb.checked = cfg.enabled; cb.id = "filter-cb-" + name;
    cb.addEventListener("change", () => {
      cfg.enabled = cb.checked;
      card.classList.toggle("disabled", !cfg.enabled);
      runFilter(); renderChart(); pushStateToURL();
    });
    const label = document.createElement("label");
    label.className = "criterion-label"; label.htmlFor = cb.id;
    label.textContent = cfg.label;
    if (cfg.tooltip) {
      label.style.cursor = "help";
      label.style.textDecoration = "underline";
      label.style.textDecorationStyle = "dotted";
      label.style.textUnderlineOffset = "3px";
      attachTooltip(label, () => ({ title: cfg.label, body: cfg.tooltip, footer: "" }));
    }
    const desc = document.createElement("span");
    desc.className = "criterion-desc"; desc.textContent = cfg.description;
    header.appendChild(cb); header.appendChild(label); header.appendChild(desc);
    card.appendChild(header);

    const controls = document.createElement("div");
    controls.className = "filter-criterion-controls";
    const minLabel = document.createElement("label");
    minLabel.textContent = "Steps: ";
    const minInput = document.createElement("input");
    minInput.type = "number"; minInput.value = cfg.minStep; minInput.min = 1000; minInput.max = 50000; minInput.step = 1000;
    minInput.addEventListener("change", () => { cfg.minStep = parseInt(minInput.value) || 1000; runFilter(); renderChart(); pushStateToURL(); });
    minLabel.appendChild(minInput); controls.appendChild(minLabel);
    const dash = document.createElement("span"); dash.textContent = "\u2013"; dash.style.color = "var(--fg-muted)";
    controls.appendChild(dash);
    const maxInput = document.createElement("input");
    maxInput.type = "number"; maxInput.value = cfg.maxStep; maxInput.min = 1000; maxInput.max = 50000; maxInput.step = 1000;
    maxInput.addEventListener("change", () => { cfg.maxStep = parseInt(maxInput.value) || 50000; runFilter(); renderChart(); pushStateToURL(); });
    controls.appendChild(maxInput);
    const threshLabel = document.createElement("label");
    threshLabel.textContent = "Threshold " + cfg.direction + " ";
    const threshInput = document.createElement("input");
    threshInput.type = "number"; threshInput.className = "threshold-input";
    threshInput.value = cfg.threshold; threshInput.step = name === "monotonicity" ? 0.1 : 1;
    threshInput.addEventListener("change", () => { cfg.threshold = parseFloat(threshInput.value) || 0; runFilter(); renderChart(); pushStateToURL(); });
    threshLabel.appendChild(threshInput); controls.appendChild(threshLabel);
    card.appendChild(controls); grid.appendChild(card);
  }
}

function renderFilterTable() {
  const table = document.getElementById("filter-table");
  const summary = document.getElementById("filter-summary");
  if (!table) return;
  table.innerHTML = "";
  const ms = getMetricsSetup();
  const criterionOrder = ["monotonicity", "snr", "cv", "mad", "consistency", "promptSwitch", "nonRandom"];
  const criterionHeaders = ["Monotonicity", "SNR", "CV", "MAD", "Consistency", "Switch rate", "Non-Randomness"];

  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  const benchTh = document.createElement("th"); benchTh.textContent = "Benchmark"; headerRow.appendChild(benchTh);
  for (let i = 0; i < criterionOrder.length; i++) {
    const cfg = filterCriteria[criterionOrder[i]];
    const th = document.createElement("th");
    th.textContent = criterionHeaders[i];
    if (!cfg.enabled) th.style.opacity = "0.4";
    if (cfg.tooltip) {
      attachTooltip(th, () => ({ title: cfg.label + " (median across models)", body: cfg.tooltip, footer: "" }));
    }
    headerRow.appendChild(th);
  }
  thead.appendChild(headerRow);
  table.appendChild(thead);

  const sorted = [...allFilterBenchmarks].sort((a, b) =>
    (ms[a]?.pretty_name || a).localeCompare(ms[b]?.pretty_name || b));
  const tbody = document.createElement("tbody");
  let passCount = 0;
  for (const bench of sorted) {
    const passes = checkedTasks.has(bench);
    if (passes) passCount++;
    const tr = document.createElement("tr");
    tr.className = passes ? "pass-row" : "fail-row";
    const nameTd = document.createElement("td");
    nameTd.textContent = ms[bench]?.pretty_name || bench;
    tr.appendChild(nameTd);
    const res = filterResults[bench] || {};
    for (const name of criterionOrder) {
      const cfg = filterCriteria[name];
      const r = res[name];
      const td = document.createElement("td");
      if (!r || r.value === null || r.value === undefined) {
        td.className = "na"; td.textContent = "N/A";
      } else {
        td.textContent = r.value === Infinity ? "\u221E" : r.value.toFixed(2);
        td.className = cfg.enabled ? (r.pass ? "pass" : "fail") : "na";
      }
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  if (summary) summary.textContent = passCount + " / " + allFilterBenchmarks.size + " benchmarks pass";
}

let filterPanelRendered = false;
let filterPanelExpanded = false;

function showFilterUI() {
  const panel = document.getElementById("filter-panel");
  if (panel) panel.style.display = "";
  const heading = document.querySelector("#task-checkboxes .checkbox-header h3");
  if (heading) heading.textContent = "Tasks eligible for quality filtering";
  if (!filterPanelRendered) {
    renderFilterPanel();
    filterPanelRendered = true;
    const toggleBtn = document.getElementById("filter-panel-toggle");
    const panelContent = document.getElementById("filter-panel-content");
    if (toggleBtn && panelContent) {
      toggleBtn.onclick = () => {
        filterPanelExpanded = !filterPanelExpanded;
        panelContent.style.display = filterPanelExpanded ? "" : "none";
        toggleBtn.textContent = filterPanelExpanded ? "Hide signal criteria" : "Show signal criteria";
      };
    }
    // Download / upload buttons
    const dlBtn = document.getElementById("filter-download-btn");
    if (dlBtn) dlBtn.onclick = () => exportFilterCriteria();
    const ulBtn = document.getElementById("filter-upload-btn");
    const ulInput = document.getElementById("filter-upload-input");
    if (ulBtn && ulInput) {
      ulBtn.onclick = () => ulInput.click();
      ulInput.onchange = () => {
        const file = ulInput.files[0];
        if (!file) return;
        file.text().then((json) => importFilterCriteria(json));
        ulInput.value = "";
      };
    }
  }
}

function hideFilterUI() {
  const panel = document.getElementById("filter-panel");
  if (panel) panel.style.display = "none";
  filterPanelRendered = false;
  const heading = document.querySelector("#task-checkboxes .checkbox-header h3");
  if (heading) heading.textContent = "Tasks included in aggregation";
}

// ============================================================
// Criteria download / upload
// ============================================================

function exportFilterCriteria() {
  const data = {};
  for (const [name, cfg] of Object.entries(filterCriteria)) {
    data[name] = { enabled: cfg.enabled, minStep: cfg.minStep, maxStep: cfg.maxStep, threshold: cfg.threshold };
  }
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = "filter-criteria-" + (currentLang || "default").toLowerCase() + ".json";
  document.body.appendChild(a); a.click();
  document.body.removeChild(a); URL.revokeObjectURL(url);
}

function importFilterCriteria(json) {
  let data;
  try { data = JSON.parse(json); } catch { return; }
  for (const [name, vals] of Object.entries(data)) {
    if (!filterCriteria[name]) continue;
    const cfg = filterCriteria[name];
    if (typeof vals.enabled === "boolean") cfg.enabled = vals.enabled;
    if (typeof vals.minStep === "number") cfg.minStep = vals.minStep;
    if (typeof vals.maxStep === "number") cfg.maxStep = vals.maxStep;
    if (typeof vals.threshold === "number") cfg.threshold = vals.threshold;
  }
  filterPanelRendered = false;
  if (currentTaskSelection === "__filtered__") {
    showFilterUI();
    runFilter();
  }
  renderChart();
  pushStateToURL();
}

// ============================================================
// URL state management
// ============================================================

function pushStateToURL() {
  const p = new URLSearchParams();
  if (currentLang) p.set("lang", currentLang);
  p.set("shot", currentShot);
  p.set("task", currentTaskSelection);
  p.set("pagg", currentPromptAgg);
  p.set("norm", currentNormalization);
  if (currentMetric) p.set("metric", currentMetric);
  p.set("pdev", showPromptDeviation ? "1" : "0");
  // Encode filter criteria compactly: name=enabled,min,max,thresh
  const fc = [];
  for (const [name, cfg] of Object.entries(filterCriteria)) {
    fc.push(name + ":" + (cfg.enabled ? "1" : "0") + "," + cfg.minStep + "," + cfg.maxStep + "," + cfg.threshold);
  }
  p.set("fc", fc.join(";"));
  history.replaceState(null, "", "?" + p.toString());
}

function restoreStateFromURL() {
  const p = new URLSearchParams(window.location.search);
  if (!p.has("lang") && !p.has("task")) return false; // no state in URL
  const languages = Object.keys(DATA.languages);
  if (p.has("lang") && languages.includes(p.get("lang"))) currentLang = p.get("lang");
  if (p.has("shot")) currentShot = p.get("shot");
  if (p.has("pagg")) currentPromptAgg = p.get("pagg");
  if (p.has("norm")) currentNormalization = p.get("norm");
  if (p.has("metric")) currentMetric = p.get("metric");
  if (p.has("pdev")) showPromptDeviation = p.get("pdev") === "1";
  if (p.has("task")) currentTaskSelection = p.get("task");
  // Restore filter criteria
  if (p.has("fc")) {
    for (const part of p.get("fc").split(";")) {
      const [name, vals] = part.split(":");
      if (!name || !vals || !filterCriteria[name]) continue;
      const [en, min, max, thresh] = vals.split(",");
      const cfg = filterCriteria[name];
      cfg.enabled = en === "1";
      if (!isNaN(Number(min))) cfg.minStep = Number(min);
      if (!isNaN(Number(max))) cfg.maxStep = Number(max);
      if (!isNaN(Number(thresh))) cfg.threshold = Number(thresh);
    }
  }
  return true;
}

function syncUIFromState() {
  // Language tabs
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.lang === currentLang);
  });
  // Shot buttons
  document.querySelectorAll(".shot-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.shot === currentShot);
  });
  // Dropdowns
  document.getElementById("task-select").value = currentTaskSelection;
  document.getElementById("prompt-agg-select").value = currentPromptAgg;
  document.getElementById("norm-select").value = currentNormalization;
  // Prompt deviation toggle
  document.getElementById("prompt-dev-toggle").checked = showPromptDeviation;
}

// ============================================================
// Entry point
// ============================================================

document.addEventListener("DOMContentLoaded", init);
