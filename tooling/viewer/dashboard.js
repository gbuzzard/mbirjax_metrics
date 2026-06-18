"use strict";
// Client logic for the mbirjax metrics dashboard.  Data is pre-parsed by
// build_dashboard.py and embedded as window.__METRICS__; this script reads it
// and renders.  Charts are uPlot (vendored, inlined); no network, no framework.

const M = window.__METRICS__;
const $ = (id) => document.getElementById(id);

// ---- colours (canvas can't read CSS vars, so these are fixed; chosen to read
// in both light and dark mode) -------------------------------------------------
const DEVC = { 1: "#378ADD", 2: "#1D9E75", 4: "#D85A30" };           // by device count
const SIZEC = ["#378ADD", "#1D9E75", "#D85A30", "#7F77DD", "#BA7517"]; // by size index
const GEOMC = { cone: "#D85A30", parallel: "#378ADD" };               // by geometry
const PLATC = { gpu: "#185fa5", cpu: "#BA7517" };                     // history: by platform
const IDEAL = "#9b9b94", FAILC = "#E24B4A", REFC = "#1f1f1d"; // ideal=grey, gate=red, reference=near-black
const BRANCH_DASH = [null, [5, 3], [2, 2], [6, 2, 2, 2]];
const devColor = (n) => DEVC[n] || SIZEC[n % SIZEC.length];

const OP_ORDER = ["direct_filter", "forward", "back", "vcd_nonconst"];
const GEOM_ORDER = ["parallel", "cone"];

// Expected (ideal) time-scaling per op, for the roughly cubical sweep shapes.
// The x-axis is sinogram entries (∝ N³ for cubic), so cost ∝ N^k maps to x^(k/3):
//   filter ∝ sinogram entries (N³) → x¹ ; forward/back ∝ voxels (N³) → x¹ ;
//   vcd ∝ voxels·views (N⁴) → x^(4/3).
const IDEAL_EXP = { direct_filter: 1, forward: 1, back: 1, vcd_nonconst: 4 / 3 };
const IDEAL_BASIS = { direct_filter: "sinogram entries", forward: "voxels", back: "voxels", vcd_nonconst: "voxels · views" };

const state = { platform: null, branch: null, go: null, ref: "none", view: "plot", openTile: null, runDate: null };

// Displayed name for each reference ("golden" stays the internal key).
const REF_LABEL = { golden: "baseline", main: "main", best: "best-ever" };

// ---- generic helpers ---------------------------------------------------------
const uniq = (a) => [...new Set(a)];
const cellKey = (c) => `${c.geom}|${c.op}|${c.size}|${c.ndev}`;
const sizeVol = (s) => s.split("x").reduce((p, n) => p * (+n || 1), 1);
const runsFor = (p, b) => M.runs.filter((r) => r.platform === p && r.branch === b).sort((a, b2) => a.date.localeCompare(b2.date));
const latestRun = (p, b) => { const r = runsFor(p, b); return r.length ? r[r.length - 1] : null; };
// The run currently being viewed: the one the user picked (state.runDate), else latest.
function currentRun() {
  const rs = runsFor(state.platform, state.branch);
  if (!rs.length) return null;
  if (state.runDate) { const m = rs.find((r) => r.date === state.runDate); if (m) return m; }
  return rs[rs.length - 1];
}
// A run's position in time: the commit's date when recorded, else the collection
// date.  Lets older prerelease checkouts sit at their real point on the timeline.
const runTime = (r) => (r.commit_date ? Date.parse(r.commit_date) / 1000 : dateToUnix(r.date));
const runDateLabel = (r) => (r.commit_date ? r.commit_date.slice(0, 10) : dateLabel(r.date));
// Branch/date/sha provenance string for the active comparison reference.
function refProvenance() {
  if (state.ref === "golden") {
    const g = M.golden[state.platform]; if (!g) return "";
    const d = g.commit_date ? g.commit_date.slice(0, 10) : null;
    return `${g.branch || "?"}${g.commit ? " @ " + g.commit : ""}${d ? " · " + d : ""}`;
  }
  if (state.ref === "main") { const m = M.main[state.platform]; return m ? `${m.branch || "main"}${m.version ? " · v" + m.version : ""}` : ""; }
  if (state.ref === "best") return "per-config best-ever";
  return "";
}
const branchesFor = (p) => uniq(M.runs.filter((r) => r.platform === p).map((r) => r.branch)).sort();
const findCell = (run, key) => run.cells.find((c) => cellKey(c) === key) || null;
const dateToUnix = (d) => Date.UTC(+d.slice(0, 4), +d.slice(4, 6) - 1, +d.slice(6, 8)) / 1000;
const dateLabel = (d) => `${d.slice(0, 4)}-${d.slice(4, 6)}-${d.slice(6, 8)}`;

function fillSelect(id, values, current, labels) {
  $(id).innerHTML = values.map((v, i) =>
    `<option value="${v}" ${v == current ? "selected" : ""}>${labels ? labels[i] : v}</option>`).join("");
}
function fmtGB(mb) { return mb == null ? "—" : (mb / 1024).toFixed(mb / 1024 < 10 ? 2 : 1) + " GB"; }
function fmtNum(v) { if (v == null) return ""; if (v >= 100) return v.toFixed(0); if (v >= 1) return v.toFixed(1); if (v > 0) return v.toFixed(2); return "0"; }
// Log axes: label only exact powers of ten, blank the minor ticks (otherwise
// uPlot tries to label every minor gridline, which reads as a stack of noise).
function logFmt(v) {
  if (v == null) return "";
  const l = Math.log10(v);
  if (Math.abs(l - Math.round(l)) > 1e-6) return "";
  return v >= 1 ? String(v) : v.toString();
}

// ---- uPlot wrapper -----------------------------------------------------------
// specs: [{label, color, ys, dash, pointsOnly, width, psize}]
function linePlot(el, xs, specs, o) {
  o = o || {};
  const cs = getComputedStyle(document.body);
  const axc = (cs.getPropertyValue("--muted").trim() || "#888");
  const grc = (cs.getPropertyValue("--border").trim() || "#ddd");
  const bg = (cs.getPropertyValue("--bg").trim() || "#fff");
  const data = [xs, ...specs.map((s) => s.ys)];
  const series = [{}, ...specs.map((s) => ({
    stroke: s.color, width: s.width == null ? 2 : s.width, dash: s.dash || undefined,
    spanGaps: true,  // bridge null cells (e.g. a failed non-dividing size) so the curve stays connected
    points: { show: true, size: s.psize == null ? 5 : s.psize, stroke: s.ring || s.color,
      fill: s.fillPoints ? s.color : ((s.pointsOnly || s.hollow) ? bg : s.color), width: s.pw == null ? 1 : s.pw },
    ...(s.pointsOnly ? { paths: () => null } : {}),
  }))];
  const xAxis = { scale: "x", stroke: axc, grid: { stroke: grc, width: 1 }, ticks: { stroke: grc, size: 4 },
    font: "11px " + (cs.fontFamily || "sans-serif") };
  if (o.xSplits) { xAxis.splits = () => o.xSplits; xAxis.filter = (u, sp) => sp; } // keep ALL custom
  // ticks (uPlot's default log filter would otherwise drop non-power-of-10 ones, e.g. the 512³ tick)
  if (o.xLabels) { xAxis.values = (u, sp) => sp.map((v) => o.xLabels[v] != null ? o.xLabels[v] : ""); }
  if (o.xLabelText) xAxis.label = o.xLabelText;
  const yAxis = { scale: "y", stroke: axc, grid: { stroke: grc, width: 1 }, ticks: { stroke: grc, size: 4 },
    font: "11px sans-serif", size: 52 };
  if (o.yLog) yAxis.values = (u, sp) => sp.map(logFmt);
  else if (o.yfmt) yAxis.values = (u, sp) => sp.map((v) => v == null ? "" : o.yfmt(v));
  if (o.yLabelText) yAxis.label = o.yLabelText;
  const xScale = { distr: o.xLog ? 3 : 1, time: !!o.xTime };
  if (o.xRange) xScale.range = o.xRange;
  const opts = {
    width: o.width || el.clientWidth || 320, height: o.height || 210,
    scales: { x: xScale, y: { distr: o.yLog ? 3 : 1 } },
    series, axes: [xAxis, yAxis], legend: { show: false },
    // drag a region to zoom (both axes on the scaling panels, x-only on the
    // time-series history panels); double-click resets.
    cursor: { points: { size: 7 }, drag: { x: true, y: !o.xTime } },
  };
  if (el._u) { el._u.destroy(); el._u = null; }
  el.innerHTML = "";
  try { el._u = new uPlot(opts, data, el); } catch (e) { el.innerHTML = "<p class='muted'>chart error: " + e.message + "</p>"; return null; }
  // Optional: a plain click (vs a drag, which zooms) selects the nearest point.
  if (o.onPick) {
    const over = el.querySelector(".u-over");
    if (over) over.addEventListener("click", () => {
      const u = el._u; const idx = u.cursor.idx;
      if (idx == null) return;
      let best = -1, bestD = Infinity;
      for (let si = 1; si < u.data.length; si++) {
        const v = u.data[si][idx]; if (v == null) continue;
        const d = Math.abs(u.valToPos(v, "y") - u.cursor.top);
        if (d < bestD) { bestD = d; best = si; }
      }
      if (best > 0 && bestD < 40) o.onPick(specs[best - 1], idx);
    });
  }
  return el._u;
}

// ---- header / selectors ------------------------------------------------------
function goOptions() {
  const run = latestRun(state.platform, state.branch);
  if (!run) return [];
  const combos = uniq(run.cells.map((c) => c.geom + "|" + c.op));
  return combos.sort((a, b) => {
    const [ga, oa] = a.split("|"), [gb, ob] = b.split("|");
    return GEOM_ORDER.indexOf(ga) - GEOM_ORDER.indexOf(gb) || OP_ORDER.indexOf(oa) - OP_ORDER.indexOf(ob);
  });
}
function syncGoSelect() {
  const opts = goOptions();
  if (!opts.includes(state.go)) state.go = opts.includes("cone|vcd_nonconst") ? "cone|vcd_nonconst" : opts[0];
  fillSelect("op", opts, state.go, opts.map((s) => s.replace("|", " · ")));
}

// ---- tiles + drill-down ------------------------------------------------------
function renderTiles() {
  const run = currentRun();
  const box = $("tiles");
  if (!run) { box.innerHTML = "<p class='muted'>no runs.</p>"; return; }
  const nRuns = runsFor(state.platform, state.branch).length;
  const failed = run.cells.filter((c) => c.failed);
  const hard = run.gate.hard.length;
  const tf = run.tests ? (run.tests.failures || []).length : 0;
  const tiles = [
    { id: "cells", lbl: "configs measured", val: run.cells.filter((c) => !c.failed).length,
      bad: failed.length > 0, sub: failed.length ? failed.length + " failed — click for details" : "all ran", click: failed.length > 0 },
    { id: "gate", lbl: "hard gate hits", val: hard, bad: hard > 0,
      sub: hard > 0 ? "click for details" : "all passed — click", click: true },
    { id: "tests", lbl: "tests failed", val: tf, bad: tf > 0,
      sub: tf > 0 ? "click for details" : (run.tests ? run.tests.passed + " passed" : "no log"), click: tf > 0 },
    { id: "run", lbl: "run shown", val: runDateLabel(run), bad: false,
      sub: state.platform.toUpperCase() + " · " + run.commit + (run.dirty ? " · dirty" : "") + (nRuns > 1 ? " · pick in history" : ""), click: false },
  ];
  box.innerHTML = tiles.map((t) =>
    `<div class="tile ${t.click ? "click" : ""} ${state.openTile === t.id ? "open" : ""}" data-id="${t.id}" data-click="${!!t.click}">
       <div class="lbl">${t.lbl}</div><div class="val ${t.bad ? "bad" : ""}">${t.val}</div><div class="sub">${t.sub}</div></div>`
  ).join("");
  box.querySelectorAll(".tile").forEach((el) => {
    if (el.dataset.click === "true") el.onclick = () => {
      state.openTile = state.openTile === el.dataset.id ? null : el.dataset.id;
      renderTiles(); renderDetail();
    };
  });
}
function renderDetail() {
  const run = currentRun();
  const box = $("detail");
  if (!run || !state.openTile) { box.innerHTML = ""; return; }
  let title = "", body = "";
  if (state.openTile === "gate") {
    title = "Gate";
    const gc = run.gate_config || {};
    const cmp = (run.gate.compared_to || []).join(", ") || "the prior run and the golden snapshot";
    const pct = (v) => v == null ? "?" : v + "%";
    body =
      `<p class="muted">After each run the engine compares this run against ${cmp}, per config and metric, and flags changes. Memory and correctness are deterministic so they hard-fail; timing is noisy so it only warns.</p>
       <p><b>Hard</b> (fails the gate): a correctness fingerprint that drifts beyond tolerance (${gc.fp_rtol_single ?? "?"} single-shot / ${gc.fp_rtol_iter ?? "?"} iterative), a structural change, a config that went ok→fail, a config that was expected but is now absent, and — GPU only — peak-memory growth above ${pct(gc.mem_hard_pct)}.</p>
       <p><b>Soft</b> (warn only): speedup drop above ${pct(gc.speedup_warn_pct)}, absolute-time increase above ${pct(gc.time_soft_pct)}, CPU memory, and sweep add/drop.</p>`;
    body += run.gate.hard.length
      ? `<h3>Hard-gate hits this run</h3><ul>${run.gate.hard.map((h) => `<li class="bad"><span class="basis">vs ${h.basis || "?"}</span> — ${h.text}</li>`).join("")}</ul>`
      : `<p class="muted">No hard-gate hits this run (gate result: ${run.gate.result || "?"}).</p>`;
  } else if (state.openTile === "tests") {
    title = "Failing tests";
    body = `<ul>${(run.tests.failures || []).map((f) => `<li class="bad">${f}</li>`).join("") || "<li>none</li>"}</ul>`;
  } else if (state.openTile === "cells") {
    title = "Failed configs";
    body = `<ul>${run.cells.filter((c) => c.failed).map((c) => `<li class="bad">${cellKey(c)}${c.oom ? " — OOM" : ""}${c.error ? " — " + c.error : ""}</li>`).join("") || "<li>none</li>"}</ul>`;
  }
  box.innerHTML = `<div class="detail-box"><h3>${title}</h3>${body}</div>`;
}

// ---- scaling view: data ------------------------------------------------------
function gridFor(run, geom, op) {
  const cs = run.cells.filter((c) => c.geom === geom && c.op === op);
  const sizes = uniq(cs.map((c) => c.size)).sort((a, b) => sizeVol(a) - sizeVol(b));
  const ndevs = uniq(cs.map((c) => c.ndev)).sort((a, b) => a - b);
  const at = (s, n) => cs.find((c) => c.size === s && c.ndev === n) || null;
  return { sizes, ndevs, at, cells: cs };
}
function refVal(geom, op, size, nd, metric) {
  const key = `${geom}|${op}|${size}|${nd}`;
  if (state.ref === "golden") { const g = M.golden[state.platform]; return g && g.cells[key] ? g.cells[key][metric] : null; }
  if (state.ref === "main") { const m = M.main[state.platform]; return m && m.cells[key] ? m.cells[key][metric] : null; }
  if (state.ref === "best") { const r = M.records[state.platform + "|" + state.branch]; const e = r && r[key]; return e && e[metric] ? e[metric].value : null; }
  return null;
}
// reference overlay series for the absolute (vs-size) panels
function refSeries(geom, op, sizes, ndevs, metric, div) {
  if (state.ref === "none") return [];
  const devs = state.ref === "main" ? [1] : ndevs;
  const out = [];
  devs.forEach((nd) => {
    const ys = sizes.map((s) => { const v = refVal(geom, op, s, nd, metric); return v != null ? v / div : null; });
    if (ys.some((y) => y != null)) out.push({ label: `${state.ref} n=${nd}`, color: REFC, ys, width: 4, fillPoints: true, psize: 4 });
  });
  return out;
}
// Interpolate a y for each failed config so it can be marked ON its curve.
// Between two good points -> interpolate along the connecting segment (in the
// panel's log/linear space).  At an endpoint -> extend the nearest two goods'
// slope; with a single good point -> follow the ideal slope; no goods -> skip.
function interpFails(xs, ys, failIdx, xLog, yLog, idealYs) {
  if (!failIdx || !failIdx.length) return null;
  const tx = (x) => xLog ? Math.log10(x) : x;
  const ty = (y) => yLog ? Math.log10(y) : y;
  const ity = (v) => yLog ? Math.pow(10, v) : v;
  const good = []; ys.forEach((y, i) => { if (y != null) good.push(i); });
  const out = xs.map(() => null);
  failIdx.forEach((i) => {
    const left = good.filter((j) => j < i).pop();
    const right = good.find((j) => j > i);
    if (left != null && right != null) {
      const f = (tx(xs[i]) - tx(xs[left])) / (tx(xs[right]) - tx(xs[left]));
      out[i] = ity(ty(ys[left]) + f * (ty(ys[right]) - ty(ys[left])));
    } else if (good.length >= 2 && (left != null || right != null)) {
      const g0 = (left != null) ? left : right;
      const g1 = good.filter((j) => j !== g0).reduce((a, b) => Math.abs(b - g0) < Math.abs(a - g0) ? b : a);
      const slope = (ty(ys[g1]) - ty(ys[g0])) / (tx(xs[g1]) - tx(xs[g0]));
      out[i] = ity(ty(ys[g0]) + slope * (tx(xs[i]) - tx(xs[g0])));
    } else if (good.length === 1 && idealYs && idealYs[i] != null && idealYs[good[0]] != null) {
      out[i] = ity(ty(ys[good[0]]) + ty(idealYs[i]) - ty(idealYs[good[0]]));
    }
  });
  return out.some((v) => v != null) ? out : null;
}
// red-ring markers for hard-gate cells of this op, on the panel whose metric matches
function gateSeries(run, geom, op, sizes, metricWord, div) {
  const hits = run.gate.hard.filter((h) => h.cell && h.cell.startsWith(geom + "|" + op + "|") && (h.text || "").toLowerCase().includes(metricWord));
  if (!hits.length) return null;
  const bySize = {};
  hits.forEach((h) => { const p = h.cell.split("|"); bySize[p[2]] = +p[3]; });
  const field = metricWord === "memory" ? "mem_mb" : "min_ms";
  const ys = sizes.map((s) => {
    if (!(s in bySize)) return null;
    const c = findCell(run, geom + "|" + op + "|" + s + "|" + bySize[s]);
    return c && c[field] != null ? c[field] / div : null;
  });
  return ys.some((y) => y != null) ? { label: "gate fail", color: FAILC, ys, pointsOnly: true, psize: 9, pw: 3 } : null;
}

function renderScaling() {
  const run = currentRun();
  const [geom, op] = state.go.split("|");
  $("sv-meta").textContent = run ? `${geom} · ${op} — ${state.branch} @ ${run.commit} · ${dateLabel(run.date)}` : "";
  if (state.view === "table") { $("sv-plot").style.display = "none"; $("sv-table").style.display = ""; renderScalingTable(run, geom, op); return; }
  $("sv-plot").style.display = ""; $("sv-table").style.display = "none";
  const g = gridFor(run, geom, op);
  const { sizes, ndevs, at } = g;
  if (!sizes.length) { $("pTime").innerHTML = "<p class='muted'>no cells.</p>"; return; }
  const PLAT = (state.platform || "").toUpperCase();
  $("capTime").textContent = `${PLAT}: time vs size · ideal ∝ ${IDEAL_BASIS[op] || "voxels"}`;
  $("capMem").textContent = `${PLAT}: memory vs size · ideal ∝ voxels`;
  $("capSpeed").textContent = `${PLAT}: speedup vs devices`;
  $("capShard").textContent = `${PLAT}: per-device memory ÷ sino shard`;
  const xvol = sizes.map(sizeVol);
  // Tick labels at the measured sizes — but collapse near-identical volumes
  // (e.g. 512³ vs the non-dividing 513³, <1% apart) to one tick so both the
  // small and large ends get a readable label instead of colliding.
  const xticks = [], xLabels = {};
  sizes.forEach((s) => { const v = sizeVol(s); if (!xticks.length || v / xticks[xticks.length - 1] > 1.03) xticks.push(v); xLabels[v] = s; });
  // Pad the log x-range so the smallest/largest ticks don't sit on the axes
  // (their labels would otherwise be clipped at the panel edges).
  const xr = [xticks[0] / 1.7, xticks[xticks.length - 1] * 1.7];
  const w = $("pTime").clientWidth || 460;

  // anchor the ideal at the fastest measured point (smallest size, most devices)
  const fastN = ndevs[ndevs.length - 1], aV = xvol[0];
  const aT = at(sizes[0], fastN), aM = at(sizes[0], fastN);

  // --- time vs size (log-log, minutes) ---
  // uPlot paints series[1] LAST (on top), so order matters: gate markers + the
  // reference go FIRST (low index → drawn on top), current curves + ideal after,
  // otherwise the reference hides behind a near-coincident current curve.
  const texp = IDEAL_EXP[op] != null ? IDEAL_EXP[op] : 1;
  const timeCurves = ndevs.map((nd) => ({ label: "n=" + nd, color: devColor(nd),
    ys: sizes.map((s) => { const c = at(s, nd); return c && !c.failed && c.min_ms != null ? c.min_ms / 60000 : null; }) }));
  const timeIdeal = (aT && aT.min_ms != null) ? [{ label: "ideal", color: IDEAL, dash: [5, 4], width: 1.5, psize: 0,
    ys: xvol.map((v) => (aT.min_ms / 60000) * Math.pow(v / aV, texp)) }] : [];
  const gT = gateSeries(run, geom, op, sizes, "time", 60000);
  // big red dots for failed configs, placed on the curve at the failing size
  const timeFails = ndevs.map((nd, ci) => {
    const fi = sizes.map((s, i) => { const c = at(s, nd); return (c && c.failed) ? i : -1; }).filter((i) => i >= 0);
    const yy = interpFails(xvol, timeCurves[ci].ys, fi, true, true, timeIdeal.length ? timeIdeal[0].ys : null);
    return yy ? { label: "failed", color: devColor(nd), ring: FAILC, ys: yy, pointsOnly: true, fillPoints: true, psize: 11, pw: 3 } : null;
  }).filter(Boolean);
  const timeSpecs = [...timeFails, ...(gT ? [gT] : []), ...refSeries(geom, op, sizes, ndevs, "min_ms", 60000), ...timeCurves, ...timeIdeal];
  linePlot($("pTime"), xvol, timeSpecs, { width: w, xLog: true, yLog: true, xSplits: xticks, xLabels, xRange: xr, yLabelText: "minutes" });

  // --- memory vs size (log-log, GB) ---  (same draw-order rule as the time panel)
  const memCurves = ndevs.map((nd) => ({ label: "n=" + nd, color: devColor(nd),
    ys: sizes.map((s) => { const c = at(s, nd); return c && !c.failed && c.mem_mb != null ? c.mem_mb / 1024 : null; }) }));
  const memIdeal = (aM && aM.mem_mb != null) ? [{ label: "ideal", color: IDEAL, dash: [5, 4], width: 1.5, psize: 0,
    ys: xvol.map((v) => (aM.mem_mb / 1024) * (v / aV)) }] : [];
  const gM = gateSeries(run, geom, op, sizes, "memory", 1024);
  const memFails = ndevs.map((nd, ci) => {
    const fi = sizes.map((s, i) => { const c = at(s, nd); return (c && c.failed) ? i : -1; }).filter((i) => i >= 0);
    const yy = interpFails(xvol, memCurves[ci].ys, fi, true, true, memIdeal.length ? memIdeal[0].ys : null);
    return yy ? { label: "failed", color: devColor(nd), ring: FAILC, ys: yy, pointsOnly: true, fillPoints: true, psize: 11, pw: 3 } : null;
  }).filter(Boolean);
  const memSpecs = [...memFails, ...(gM ? [gM] : []), ...refSeries(geom, op, sizes, ndevs, "mem_mb", 1024), ...memCurves, ...memIdeal];
  linePlot($("pMem"), xvol, memSpecs, { width: w, xLog: true, yLog: true, xSplits: xticks, xLabels, xRange: xr, yLabelText: "GB" });

  // --- speedup vs devices (one curve per size; ideal linear) ---
  const w2 = $("pSpeed").clientWidth || 460;
  const speedCurves = sizes.map((s, i) => { const base = at(s, ndevs[0]);
    return { label: s, color: SIZEC[i % SIZEC.length],
      ys: ndevs.map((nd) => { const c = at(s, nd); return c && base && !c.failed && !base.failed ? (base.min_ms / c.min_ms) * ndevs[0] : null; }) }; });
  const speedIdeal = ndevs.slice();
  const speedFails = sizes.map((s, ci) => {
    const fi = ndevs.map((nd, i) => { const c = at(s, nd); return (c && c.failed) ? i : -1; }).filter((i) => i >= 0);
    const yy = interpFails(ndevs, speedCurves[ci].ys, fi, false, false, speedIdeal);
    return yy ? { label: "failed", color: SIZEC[ci % SIZEC.length], ring: FAILC, ys: yy, pointsOnly: true, fillPoints: true, psize: 11, pw: 3 } : null;
  }).filter(Boolean);
  const speedSpecs = [...speedFails, ...speedCurves, { label: "ideal", color: IDEAL, dash: [5, 4], width: 1.5, psize: 0, ys: speedIdeal }];
  linePlot($("pSpeed"), ndevs, speedSpecs, { width: w2, xSplits: ndevs, xLabels: Object.fromEntries(ndevs.map((n) => [n, String(n)])), yfmt: (v) => v.toFixed(0) + "×", yLabelText: "speedup", xLabelText: "devices" });

  // --- per-device memory ÷ sino shard (one curve per size; ideal 2x) ---
  const shardCurves = sizes.map((s, i) => ({ label: s, color: SIZEC[i % SIZEC.length],
    ys: ndevs.map((nd) => { const c = at(s, nd); if (!c || c.failed || c.mem_mb == null) return null;
      const shardMB = (sizeVol(s) * 4 / nd) / (1024 * 1024); return c.mem_mb / shardMB; }) }));
  const shardFails = sizes.map((s, ci) => {
    const fi = ndevs.map((nd, i) => { const c = at(s, nd); return (c && c.failed) ? i : -1; }).filter((i) => i >= 0);
    const yy = interpFails(ndevs, shardCurves[ci].ys, fi, false, false, ndevs.map(() => 2));
    return yy ? { label: "failed", color: SIZEC[ci % SIZEC.length], ring: FAILC, ys: yy, pointsOnly: true, fillPoints: true, psize: 11, pw: 3 } : null;
  }).filter(Boolean);
  const shardSpecs = [...shardFails, ...shardCurves, { label: "ideal 2×", color: IDEAL, dash: [5, 4], width: 1.5, psize: 0, ys: ndevs.map(() => 2) }];
  linePlot($("pShard"), ndevs, shardSpecs, { width: w2, xSplits: ndevs, xLabels: Object.fromEntries(ndevs.map((n) => [n, String(n)])), yfmt: (v) => v.toFixed(1) + "×", yLabelText: "mem ÷ shard", xLabelText: "devices" });

  renderScalingLegend(ndevs, sizes);
}
function renderScalingLegend(ndevs, sizes) {
  const k = (c, t, dash) => `<span class="k"><span class="sw" style="background:${c};${dash ? "height:0;border-top:2px dashed " + c : ""}"></span>${t}</span>`;
  // failed-config marker = curve-coloured centre with a red ring (centre shown
  // neutral here since the colour varies per curve)
  const ringDot = (t) => `<span class="k"><span style="width:13px;height:13px;border-radius:50%;background:var(--surface2);border:3px solid ${FAILC};box-sizing:border-box;display:inline-block"></span>${t}</span>`;
  const devs = ndevs.map((n) => k(devColor(n), "n=" + n)).join("");
  const szs = sizes.map((s, i) => k(SIZEC[i % SIZEC.length], s)).join("");
  // active comparison: solid black swatch + display name + provenance (branch @ commit)
  const refNote = state.ref !== "none"
    ? `<span class="k"><span class="sw" style="background:${REFC};height:4px"></span>${REF_LABEL[state.ref] || state.ref}${refProvenance() ? " (" + refProvenance() + ")" : ""}</span>` : "";
  // Top legend sits above time & memory (device-count curves + the overlay ref);
  // the second legend sits above speedup & shard (size curves).
  $("sv-legend").innerHTML =
    `<span class="grp">${devs}</span>` +
    `<span class="grp">${k(IDEAL, "ideal", true)}${ringDot("failed config")}<span class="k"><span class="ring"></span>gate fail</span>${refNote}</span>`;
  $("sv-legend2").innerHTML =
    `<span class="grp">${szs}</span>` +
    `<span class="grp">${k(IDEAL, "ideal", true)}${ringDot("failed config")}</span>`;
}

function renderScalingTable(run, geom, op) {
  const g = gridFor(run, geom, op);
  const { sizes, ndevs, at } = g;
  const box = $("sv-table");
  if (!sizes.length) { box.innerHTML = "<p class='muted'>no cells.</p>"; return; }
  // pick one time unit for the whole table by the max time
  const allMs = g.cells.filter((c) => !c.failed && c.min_ms != null).map((c) => c.min_ms);
  const useMin = allMs.length && Math.max(...allMs) >= 60000;
  const tUnit = useMin ? "min" : "s", tDiv = useMin ? 60000 : 1000;
  const fmtT = (ms) => ms == null ? "—" : (ms / tDiv).toFixed(2);
  const refActive = state.ref !== "none";
  const dCell = (cur, ref, lowerBetter) => {
    if (cur == null || ref == null || ref === 0) return "<td class='num'>—</td>";
    const d = ((cur - ref) / Math.abs(ref)) * 100;
    if (Math.abs(d) < 1) return `<td class='num'>${d > 0 ? "+" : ""}${d.toFixed(1)}%</td>`;
    const worse = lowerBetter ? d > 0 : d < 0;
    return `<td class='num ${worse ? "up" : "dn"}'>${d > 0 ? "+" : ""}${d.toFixed(1)}%</td>`;
  };
  const tbl = (title, field, div, fmt, unit) => {
    let h = `<table class='grid'><caption>${title}${unit ? " (" + unit + ")" : ""}${refActive ? " · Δ vs " + state.ref : ""}</caption><thead><tr><th>devices</th>`;
    sizes.forEach((s) => { h += `<th>${s}</th>`; if (refActive) h += "<th>Δ</th>"; });
    h += "</tr></thead><tbody>";
    ndevs.forEach((nd) => {
      h += `<tr><td class='l'>n=${nd}</td>`;
      sizes.forEach((s) => {
        const c = at(s, nd);
        if (c && c.failed) { h += `<td class='fail-cell'>${c.oom ? "OOM" : "FAIL"}</td>`; if (refActive) h += "<td></td>"; return; }
        const cur = c ? c[field] : null;
        h += `<td class='num'>${cur == null ? "—" : fmt(cur)}</td>`;
        if (refActive) h += dCell(cur, refVal(geom, op, s, nd, field), true);
      });
      h += "</tr>";
    });
    return h + "</tbody></table>";
  };
  box.innerHTML = tbl("time", "min_ms", tDiv, fmtT, tUnit) + tbl("memory", "mem_mb", 1024, fmtGB, "");
}

// ---- history strip -----------------------------------------------------------
function aggregate(run) {
  const focus = Math.max(...run.cells.map((c) => sizeVol(c.size)));
  const focusSize = run.cells.map((c) => c.size).find((s) => sizeVol(s) === focus);
  const out = { vcd: {}, mem: {}, gate: run.gate.hard.length };
  GEOM_ORDER.forEach((gm) => {
    const vc = run.cells.find((c) => c.geom === gm && c.op === "vcd_nonconst" && c.size === focusSize && c.ndev === 1 && !c.failed);
    out.vcd[gm] = vc && vc.min_ms != null ? vc.min_ms / 60000 : null;
    const mems = run.cells.filter((c) => c.geom === gm && c.size === focusSize && c.ndev === 1 && !c.failed && c.mem_mb != null).map((c) => c.mem_mb);
    out.mem[gm] = mems.length ? Math.max(...mems) / 1024 : null;
  });
  return out;
}
// Click a history point -> show that run (and switch platform/branch to match).
function pickRun(spec, idx) {
  const t = spec._xs[idx];
  const r = runsFor(spec.meta.platform, spec.meta.branch).find((x) => runTime(x) === t);
  if (!r) return;
  state.platform = spec.meta.platform; state.branch = spec.meta.branch;
  state.runDate = r.date; state.openTile = null;
  fillSelect("platform", M.platforms, state.platform);
  fillSelect("branch", branchesFor(state.platform), state.branch);
  renderAll();
}
function renderHistory() {
  // The history spans BOTH platforms and all branches; x is commit time
  // (falls back to collection date for older runs).
  const xs = uniq(M.runs.map(runTime)).sort((a, b) => a - b);
  const aggByPB = {};  // "platform|branch" -> runTime -> aggregate
  M.runs.forEach((r) => { const key = r.platform + "|" + r.branch; (aggByPB[key] = aggByPB[key] || {})[runTime(r)] = aggregate(r); });

  // colour = platform, line-style = geometry (cone solid, parallel dashed).
  const specsFor = (pick) => {
    const out = [];
    M.platforms.forEach((plat) => M.branches.forEach((b) => GEOM_ORDER.forEach((gm) => {
      const agg = aggByPB[plat + "|" + b]; if (!agg) return;
      const ys = xs.map((t) => { const a = agg[t]; return a && a[pick][gm] != null ? a[pick][gm] : null; });
      if (ys.some((y) => y != null)) out.push({ label: `${plat} ${gm}`, color: PLATC[plat] || IDEAL,
        dash: gm === "parallel" ? [5, 3] : undefined, ys, _xs: xs, meta: { platform: plat, branch: b } });
    })));
    return out;
  };
  // With a single point, uPlot's auto time-range sprawls across years; pin a ±12h window.
  const xr = xs.length < 2 ? [xs[0] - 43200, xs[0] + 43200] : null;
  const opts = (yl) => ({ xTime: true, xRange: xr, yLog: true, yLabelText: yl, yfmt: fmtNum, onPick: pickRun });
  linePlot($("hVcd"), xs, specsFor("vcd"), { width: $("hVcd").clientWidth || 320, ...opts("min") });
  linePlot($("hMem"), xs, specsFor("mem"), { width: $("hMem").clientWidth || 320, ...opts("GB") });
  const gateSpecs = [];
  M.platforms.forEach((plat) => M.branches.forEach((b) => {
    const agg = aggByPB[plat + "|" + b]; if (!agg) return;
    const ys = xs.map((t) => { const a = agg[t]; return a ? a.gate : null; });
    if (ys.some((y) => y != null)) gateSpecs.push({ label: `${plat} ${b}`, color: PLATC[plat] || IDEAL, ys, _xs: xs, meta: { platform: plat, branch: b } });
  }));
  linePlot($("hGate"), xs, gateSpecs, { width: $("hGate").clientWidth || 320, xTime: true, xRange: xr, yLabelText: "count", yfmt: (v) => v.toFixed(0), onPick: pickRun });

  const k = (c, t, dash) => `<span class="k"><span class="sw" style="background:${c};${dash ? "height:0;border-top:2px dashed " + c : ""}"></span>${t}</span>`;
  $("hist-legend").innerHTML =
    `<span class="grp">${M.platforms.map((p) => k(PLATC[p] || IDEAL, p)).join("")}</span>` +
    `<span class="grp">${k("#888", "cone (solid)")}${k("#888", "parallel (dashed)", true)}</span>`;
}

// ---- orchestration -----------------------------------------------------------
function renderAll() { renderTiles(); renderDetail(); syncGoSelect(); renderScaling(); renderHistory(); }
function onPlatform() {
  state.platform = $("platform").value;
  const bs = branchesFor(state.platform);
  if (!bs.includes(state.branch)) state.branch = bs[0];
  fillSelect("branch", bs, state.branch);
  state.openTile = null; state.runDate = null; renderAll();
}
function init() {
  $("gen").textContent = `generated ${M.generated_utc} · ${M.repo_name}`;
  $("footer").innerHTML = `${M.runs.length} run(s) · platforms ${M.platforms.join(", ")} · branches ${M.branches.join(", ")} · regenerate with <code>action_scripts/build_dashboard.sh</code>`;
  if (!M.runs.length) { $("tiles").innerHTML = "<p class='muted'>No runs found under results/.</p>"; return; }

  state.platform = M.platforms.includes("gpu") ? "gpu" : M.platforms[0];
  state.branch = branchesFor(state.platform)[0];
  fillSelect("platform", M.platforms, state.platform);
  fillSelect("branch", branchesFor(state.platform), state.branch);

  $("platform").onchange = onPlatform;
  $("branch").onchange = () => { state.branch = $("branch").value; state.openTile = null; state.runDate = null; renderAll(); };
  $("op").onchange = () => { state.go = $("op").value; renderScaling(); };
  $("ref").value = state.ref;
  $("ref").onchange = () => { state.ref = $("ref").value; renderScaling(); };
  $("view-seg").innerHTML = `<button data-v="plot" class="on">plot</button><button data-v="table">table</button>`;
  $("view-seg").querySelectorAll("button").forEach((b) => b.onclick = () => {
    state.view = b.dataset.v; $("view-seg").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b)); renderScaling();
  });
  let rt; window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => { renderScaling(); renderHistory(); }, 160); });
  renderAll();
}
init();
