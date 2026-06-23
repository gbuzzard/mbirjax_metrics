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
const THROTC = "#E8950C";                                    // amber: a GPU ran hot / throttled (timing unreliable)
const CORRC = "#a32d2d";                                      // deep red: a CORRECTNESS divergence (more severe than a perf hit)
const HOT_C = 85, HOT_HBM = 95;                              // a cell is "hot" if a GPU core>=85C / HBM>=95C / any throttle reason
const BRANCH_DASH = [null, [5, 3], [2, 2], [6, 2, 2, 2]];
const devColor = (n) => DEVC[n] || SIZEC[n % SIZEC.length];

const OP_ORDER = ["direct_filter", "forward", "back", "vcd_nonconst", "denoise"];
const GEOM_ORDER = ["parallel", "cone", "translation", "multiaxis_parallel", "denoiser"];
// History line-style by geometry (one solid + one dashed within each group); short legend labels.
const GEOM_DASH = { cone: undefined, parallel: [5, 3], translation: undefined, multiaxis_parallel: [5, 3], denoiser: undefined };
const GEOM_LABEL = { cone: "cone", parallel: "parallel", translation: "translation", multiaxis_parallel: "multiaxis", denoiser: "denoiser" };
// History geometry GROUPS: a toggle swaps the set shown.  cone/parallel headline the vcd recon;
// translation/multiaxis don't run vcd (only projectors + filter), so they headline back-projection;
// the denoiser is its own group (a single geometry) headlining its one op, denoise (the vcd analog).
const HIST_GROUPS = [
  { id: "cp", label: "cone + parallel", geoms: ["cone", "parallel"], op: "vcd_nonconst", opLabel: "VCD" },
  { id: "tm", label: "translation + multiaxis", geoms: ["translation", "multiaxis_parallel"], op: "back", opLabel: "back-projection" },
  { id: "dn", label: "denoiser", geoms: ["denoiser"], op: "denoise", opLabel: "denoise" },
];

// Expected (ideal) time-scaling per op, for the roughly cubical sweep shapes.
// The x-axis is sinogram entries (∝ N³ for cubic), so cost ∝ N^k maps to x^(k/3):
//   filter ∝ sinogram entries (N³) → x¹ ; forward/back ∝ voxels (N³) → x¹ ;
//   vcd ∝ voxels·views (N⁴) → x^(4/3).
const IDEAL_EXP = { direct_filter: 1, forward: 1, back: 1, vcd_nonconst: 4 / 3 };
const IDEAL_BASIS = { direct_filter: "sinogram entries", forward: "voxels", back: "voxels", vcd_nonconst: "voxels · views" };

const ui_state = { platform: null, branch: null, go: null, ref: "none", view: "plot", openTile: null, runKey: null, histN: 1, histGroup: "cp", histBranch: "all" };

// Displayed name for each reference (internal key -> label).  References are now derived from the
// tracked runs themselves (latest main/prerelease tip, this branch's prior run) + best-ever.
const REF_LABEL = { main: "main", prerelease: "prerelease", prior: "prior run", best: "best-ever" };

// ---- generic helpers ---------------------------------------------------------
const uniq = (a) => [...new Set(a)];
const cellKey = (c) => `${c.geom}|${c.op}|${c.size}|${c.ndev}`;
// "geom|op|size|ndev" -> "geom, op, size, n_devices=N" (the human config label, shared by the banner
// and the correctness/perf drill-downs).
const cellCoords = (k) => { const p = (k || "").split("|"); return p.length === 4 ? `${p[0]}, ${p[1]}, ${p[2]}, n_devices=${p[3]}` : (k || ""); };
const sizeVol = (s) => s.split("x").reduce((p, n) => p * (+n || 1), 1);
// Sort by COMMIT time (not collection date), so "latest" = the most recent COMMIT.  Otherwise an
// add_run of an OLD commit (collected recently) would sort last and become the default run shown.
const runsFor = (p, b) => M.runs.filter((r) => r.platform === p && r.branch === b).sort((a, b2) => runTime(a) - runTime(b2));
const latestRun = (p, b) => { const r = runsFor(p, b); return r.length ? r[r.length - 1] : null; };
// The run currently being viewed: the one the user picked (ui_state.runKey), else latest.
function currentRun() {
  const rs = runsFor(ui_state.platform, ui_state.branch);
  if (!rs.length) return null;
  if (ui_state.runKey) { const m = rs.find((r) => runKey(r) === ui_state.runKey); if (m) return m; }
  return rs[rs.length - 1];
}
// A run's position in time: the commit's date when recorded, else the collection
// date.  Lets older prerelease checkouts sit at their real point on the timeline.
const runTime = (r) => (r.commit_date ? Date.parse(r.commit_date) / 1000 : dateToUnix(r.date));
// Unique handle for a picked run.  The collection `date` (YYYYMMDD) is NOT unique — two commits
// measured the same day share it — so a click resolved by date alone returned the earlier-committed
// run even though the tooltip (which keys on commit time) named the one clicked.  Key on the commit
// too: commit-sha # collection-date # commit-time disambiguates same-day commits and re-measures.
const runKey = (r) => (r.commit_full || r.commit || "?") + "#" + (r.date || "") + "#" + (r.commit_date || "");
const runDateLabel = (r) => (r.commit_date ? r.commit_date.slice(0, 10) : dateLabel(r.date));
// Commit date AND time, to the minute (e.g. "2026-06-17 23:00") — the unambiguous run stamp.
// ISO is "YYYY-MM-DDTHH:MM:SS±zz"; slice to the minute and swap the T for a space.
const commitMinute = (r) => (r.commit_date ? r.commit_date.slice(0, 16).replace("T", " ") : (r.date ? dateLabel(r.date) : "?"));
// ---- correctness (severity split, design note D1/D6) -------------------------------------------
// The "reviewed-through" watermark (a single date from results/correctness_acks.yaml): a run whose
// commit is dated <= this is acknowledged — kept visible but greyed and dropped from the banner/badge.
// Compare by the commit's LOCAL calendar date (runDateLabel -> "YYYY-MM-DD"), NOT a UTC timestamp, so it
// matches clear_correctness.py / the build analyzer (both use commit_date[:10]).  An evening commit in a
// negative-offset zone is "the 22nd" locally even though it's already the 23rd in UTC — a UTC compare
// would leave it un-acknowledged and keep the banner up after clear_correctness said "nothing to clear".
// Correctness findings are computed corpus-wide by build_dashboard (prior + cross-device + vs-main),
// each {reference, cell, basis, discrepancies[]}.  Perf hard hits stay on gate.hard (kind != correctness).
const runCorr = (r) => (r.correctness || []);
const runCorrCells = (r) => new Set(runCorr(r).map((f) => f.cell)).size;   // distinct divergent configs
const runPerfHard = (r) => (r.gate && r.gate.hard || []).filter((h) => h.kind !== "correctness");
const runIncorrect = (r) => runCorr(r).length > 0;            // diverges on at least one cell/reference
const runAcked = (r) => M.cleared_through != null && runDateLabel(r) <= M.cleared_through;
const runAlert = (r) => runIncorrect(r) && !runAcked(r);     // unacknowledged -> drives banner/badge/markers
// A gate basis / compared_to string is the prior RUN's filename ("prior:regression_<plat>_<ts>_<sha>.yaml").
// Render it like a run entry — "GPU · <commit> · <commit date+time>" — by resolving its sha to that run.
function priorLabel(fn, plat) {
  const m = /_([0-9a-f]{7,40})\.ya?ml$/.exec(fn || "");
  const sha = m ? m[1] : null;
  if (!sha) return fn || "?";
  const r = M.runs.find((x) => x.platform === plat && x.commit_full && x.commit_full.startsWith(sha));
  return r ? `${plat.toUpperCase()} · ${r.commit} · ${commitMinute(r)}` : `${plat.toUpperCase()} · ${sha.slice(0, 10)}`;
}
// The run shown for a given platform on the currently-selected branch: honour an explicitly
// picked run (ui_state.runKey) only on the active platform; otherwise the latest for that platform.
function runOnPlat(plat) {
  const rs = runsFor(plat, ui_state.branch);
  if (!rs.length) return null;
  // Anchor BOTH platforms to the shown run's COMMIT, so every split tile (and the per-platform drill-down)
  // reflects exactly that commit.  A platform that did not measure this commit returns null here -> shows
  // '—'; we do NOT substitute its latest run (that would mix commits).  Only an absent anchor commit (old
  // data with no commit_full) falls back to latest, so the page still renders.
  const cur = currentRun();
  if (!cur || !cur.commit_full) return rs[rs.length - 1];
  return rs.filter((r) => r.commit_full === cur.commit_full).sort((a, b) => runTime(b) - runTime(a))[0] || null;
}
// The run backing the active reference overlay: the tracked main/prerelease tip, or this branch's
// immediately-preceding run.  best-ever is records-derived (not a single run) -> handled separately.
function refRun() {
  if (ui_state.ref === "main") return latestRun(ui_state.platform, "main");
  if (ui_state.ref === "prerelease") return latestRun(ui_state.platform, "prerelease");
  if (ui_state.ref === "prior") {
    const rs = runsFor(ui_state.platform, ui_state.branch), cur = currentRun();
    const i = cur ? rs.indexOf(cur) : -1;
    return i > 0 ? rs[i - 1] : null;
  }
  return null;
}
// Branch/sha/date provenance string for the active comparison reference.
function refProvenance() {
  if (ui_state.ref === "best") return "per-config best-ever";
  const r = refRun(); if (!r) return "";
  const d = r.commit_date ? r.commit_date.slice(0, 10) : (r.date ? runDateLabel(r) : "");
  return `${r.branch || "?"}${r.commit ? " @ " + r.commit : ""}${d ? " · " + d : ""}`;
}
const branchesFor = (p) => uniq(M.runs.filter((r) => r.platform === p).map((r) => r.branch)).sort();
const findCell = (run, key) => run.cells.find((c) => cellKey(c) === key) || null;
// CAUSAL signal: the driver actually clamped clocks (a clocks_throttle_reasons.* fired during the
// cell).  This is what degrades a measured time — distinct from merely running hot.  (Only the new
// engine records throttle reasons, so this stays empty on runs measured before that landed.)
function cellThrottled(c) {
  return !!(c && c.gpu && c.gpu.some((g) => g.thr && g.thr.length));
}
// A cell worth flagging because a GPU ran hot OR throttled during it.  Re-derived client-side from
// the per-GPU temps (c.gpu, present only for flagged cells) so OLD runs measured before the throttle
// detector was tightened still light up — plus the engine's own `throttled` flag.  Superset of
// cellThrottled: "hot" (temperature) is advisory; "throttled" (clocks clamped) is causal.
function cellHot(c) {
  if (!c) return false;
  if (c.throttled || cellThrottled(c)) return true;
  return !!(c.gpu && c.gpu.some((g) => (g.t || 0) >= HOT_C || (g.mt || 0) >= HOT_HBM));
}
// "which GPU, how hot, why" for the tooltip (the hottest GPU on the cell).
function hotGpuStr(c) {
  const gs = c && c.gpu;
  if (!gs || !gs.length) return c && c.throttled ? "throttled" : "";
  const w = gs.reduce((a, b) => ((b.t || 0) > (a.t || 0) ? b : a), gs[0]);
  let s = `GPU${w.i} ${w.t}°C`;
  if (w.mt != null) s += ` · HBM ${w.mt}°C`;
  if (w.sm != null) s += ` · sm ${w.sm}` + (w.mem != null ? `/mem ${w.mem}` : "") + " MHz";
  if (w.thr) s += ` · ${w.thr.join(", ")}`;
  return s;
}
// Tooltip warning text: lead with the ui_state word so "hot" (advisory) reads distinctly from
// "throttled" (the driver clamped clocks, so the time is suspect).
function hotWarn(c) { return (cellThrottled(c) ? "throttled" : "hot") + " · " + hotGpuStr(c); }
// Run-level thermal summary for the "run shown" tile: the worst severity present (throttled beats
// hot), the device counts (n) it hit, and the peak core temp — null when the run was thermally fine.
function runThermal(run) {
  const acc = { throttled: { devs: new Set(), t: 0 }, hot: { devs: new Set(), t: 0 } };
  (run.cells || []).forEach((c) => {
    const sev = cellThrottled(c) ? "throttled" : (cellHot(c) ? "hot" : null);
    if (!sev) return;
    if (c.ndev != null) acc[sev].devs.add(c.ndev);
    const pk = Math.max(0, ...((c.gpu || []).map((g) => g.t || 0)));
    if (pk > acc[sev].t) acc[sev].t = pk;
  });
  const sev = acc.throttled.devs.size ? "throttled" : (acc.hot.devs.size ? "hot" : null);
  return sev ? { sev, ndevs: [...acc[sev].devs].sort((a, b) => a - b), peak: acc[sev].t } : null;
}
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
  // Optional data-domain padding: extend the x-domain on each end with a null-y column, so the
  // extreme points/ticks aren't jammed against the panel edge — WITHOUT pinning a fixed scale range,
  // which would hard-clamp the scale and block drag-zoom (#6).  Auto-range keeps zoom working (the
  // prepended pad column shifts the data index by padOff, undone for the tooltip/click callbacks).
  //   o.xPad    — multiplicative factor (log axes, e.g. the scaling size axis)
  //   o.xPadAdd — additive amount in x-units (linear axes, e.g. 5% of the span on the time-history
  //               axis, so the first/last run isn't on the boundary where a click misses — the
  //               tooltip's 30px grab still finds it, but u.cursor.idx comes back null right at the edge).
  const padOff = ((o.xPad || o.xPadAdd) && xs.length > 1) ? 1 : 0;
  const xLo = o.xPadAdd ? xs[0] - o.xPadAdd : xs[0] / o.xPad;
  const xHi = o.xPadAdd ? xs[xs.length - 1] + o.xPadAdd : xs[xs.length - 1] * o.xPad;
  const X = padOff ? [xLo, ...xs, xHi] : xs;
  const S = padOff ? specs.map((s) => ({ ...s, ys: [null, ...s.ys, null] })) : specs;
  const data = [X, ...S.map((s) => s.ys)];
  const series = [{}, ...S.map((s) => ({
    stroke: s.color, width: s.width == null ? 2 : s.width, dash: s.dash || undefined,
    spanGaps: true,  // bridge null cells (e.g. a failed non-dividing size) so the curve stays connected
    // psize 0 means "no markers" (e.g. the ideal line) — hide them via show:false.  Passing
    // size:0 makes uPlot compute a NEGATIVE arc radius and throw mid-draw, which aborted every
    // redraw and silently broke drag-zoom (the zoom's commit redraw never finished).
    points: { show: s.psize !== 0, size: s.psize == null ? 5 : s.psize, stroke: s.ring || s.color,
      fill: s.fillPoints ? s.color : ((s.pointsOnly || s.hollow) ? bg : s.color), width: s.pw == null ? 1 : s.pw },
    ...(s.pointsOnly ? { paths: () => null } : {}),
  }))];
  const xAxis = { scale: "x", stroke: axc, grid: { stroke: grc, width: 1 }, ticks: { stroke: grc, size: 4 },
    font: "11px " + (cs.fontFamily || "sans-serif") };
  // Custom ticks at the measured sizes/devices — but RANGE-AWARE: only the ticks inside the current
  // scale window.  (A fixed full list would keep out-of-range ticks after a zoom and fight uPlot's
  // ranging, which blocked drag-zoom.)  filter keeps all of them (uPlot's default log filter would
  // otherwise drop the non-power-of-10 ones, e.g. the 512³ tick).
  if (o.xSplits) { xAxis.splits = (u, ai, mn, mx) => o.xSplits.filter((v) => v >= mn && v <= mx); xAxis.filter = (u, sp) => sp; }
  if (o.xLabels) { xAxis.values = (u, sp) => sp.map((v) => o.xLabels[v] != null ? o.xLabels[v] : ""); }
  if (o.xLabelText) xAxis.label = o.xLabelText;
  const yAxis = { scale: "y", stroke: axc, grid: { stroke: grc, width: 1 }, ticks: { stroke: grc, size: 4 },
    font: "11px sans-serif", size: 52 };
  if (o.yLog) yAxis.values = (u, sp) => sp.map(logFmt);
  else if (o.yfmt) yAxis.values = (u, sp) => sp.map((v) => v == null ? "" : o.yfmt(v));
  if (o.yLabelText) yAxis.label = o.yLabelText;
  // uPlot's default log range snaps min/max OUT to the enclosing powers of 10.  On a drag-zoom that
  // expands the selection back out (a ~1-decade pick on the size axis snaps to the full 1e7..1e10
  // view), so you never get the region you dragged.  o.tightLog gives the log scales an identity
  // range — the scale is exactly its data extent (or the zoom selection), no power-of-10 rounding.
  // (It's a function, not a fixed array, so it still zooms/resets — unlike the old hard-clamp.)
  const tightLog = (u, mn, mx) => [mn, mx];
  // Pad a scale by `frac` of its span on each end — log-space for log scales, linear otherwise — to
  // give the panel a little breathing room at the edges (applies to the initial view and to zooms).
  // A ZERO-span input (e.g. a single-device CPU run whose speedup is always 1× -> min==max) must NOT
  // be returned as-is: uPlot's tick splitter then loops forever building an array (RangeError /
  // multi-second hang).  Open such a range up to a small non-zero window instead.
  const padRange = (frac, isLog) => (u, mn, mx) => {
    if (mn == null || mx == null) return [mn, mx];
    if (isLog) {
      if (!(mn > 0) || !(mx > 0)) return [mn, mx];           // non-positive: leave it to uPlot
      let a = Math.log10(mn), b = Math.log10(mx);
      if (b - a < 1e-9) { a -= 0.5; b += 0.5; } else { const p = (b - a) * frac; a -= p; b += p; }
      return [Math.pow(10, a), Math.pow(10, b)];
    }
    let lo = mn, hi = mx;
    if (hi - lo < 1e-12) { const d = Math.abs(lo) * 0.1 || 1; lo -= d; hi += d; }
    else { const p = (hi - lo) * frac; lo -= p; hi += p; }
    return [lo, hi];
  };
  const xScale = { distr: o.xLog ? 3 : 1, time: !!o.xTime };
  if (o.xRange) xScale.range = o.xRange;
  else if (o.padAll != null) xScale.range = padRange(o.padAll, o.xLog);
  else if (o.tightLog && o.xLog) xScale.range = tightLog;
  const yScale = { distr: o.yLog ? 3 : 1 };
  if (o.yPad != null) yScale.range = padRange(o.yPad, o.yLog);
  else if (o.padAll != null) yScale.range = padRange(o.padAll, o.yLog);
  else if (o.tightLog && o.yLog) yScale.range = tightLog;
  // Truly-nearest drawn point to the cursor, in 2-D px (shared by the click-to-pick and hover-tooltip
  // handlers).  Previously this snapped to uPlot's cursor.idx — the nearest x-COLUMN — then took the
  // nearest series in y.  When two runs sit close in x (two commits a few hours apart on a multi-day
  // axis), a few px of cursor motion flips the snapped column and the pick jumps to a far run.  Scanning
  // every drawn (series, column) by Euclidean px distance instead picks the dot the eye is closest to.
  const nearestPoint = (u, maxPx) => {
    const cl = u.cursor.left, ct = u.cursor.top;
    if (cl == null || ct == null || cl < 0 || ct < 0) return null;
    let bSi = -1, bDi = -1, bD = Infinity;
    for (let si = 1; si < u.data.length; si++) {
      const col = u.data[si];
      for (let di = 0; di < col.length; di++) {
        const v = col[di]; if (v == null) continue;
        const dx = u.valToPos(u.data[0][di], "x") - cl, dy = u.valToPos(v, "y") - ct;
        const d = dx * dx + dy * dy;
        if (d < bD) { bD = d; bSi = si; bDi = di; }
      }
    }
    return (bSi > 0 && bD <= maxPx * maxPx) ? { si: bSi, di: bDi } : null;
  };
  // Optional hover tooltip: o.tooltip(spec, idx) -> HTML string (or null to hide).
  let tip = null;
  if (o.tooltip) { tip = document.createElement("div"); tip.className = "u-tip"; }
  const hooks = {};
  if (o.tooltip) hooks.setCursor = [(u) => {
    const cl = u.cursor.left, ct = u.cursor.top;
    // hide while off-plot or mid drag-zoom
    if (cl == null || ct == null || cl < 0 || ct < 0 || (u.select && u.select.width > 1)) { tip.style.display = "none"; return; }
    const np = nearestPoint(u, 30), oi = np ? np.di - padOff : -1;
    const html = (np && oi >= 0 && oi < xs.length) ? o.tooltip(specs[np.si - 1], oi) : null;
    if (!html) { tip.style.display = "none"; return; }
    tip.innerHTML = html; tip.style.display = "block";
    const ob = u.over.getBoundingClientRect(), eb = el.getBoundingClientRect();
    let lx = ob.left - eb.left + cl + 14;
    if (lx + tip.offsetWidth > el.clientWidth) lx = Math.max(0, ob.left - eb.left + cl - tip.offsetWidth - 14);
    tip.style.left = lx + "px"; tip.style.top = (ob.top - eb.top + ct + 8) + "px";
  }];
  // Linked x-zoom: when one plot in a sync group zooms (or resets) its x-scale, mirror the range to
  // the others.  Propagate ONLY when a peer's range actually differs, so the cascade converges
  // instead of looping (uPlot commits setScale on rAF, so a sync flag wouldn't span the callbacks).
  if (o.syncX) { const group = o.syncX;
    hooks.setScale = [(u, key) => {
      if (key !== "x") return;
      const mn = u.scales.x.min, mx = u.scales.x.max;
      group.forEach((p) => { if (p !== u && (p.scales.x.min !== mn || p.scales.x.max !== mx)) p.setScale("x", { min: mn, max: mx }); });
    }];
  }
  // Optional custom triangle marks at data points (o.marks: [{x, y}]) — uPlot only draws CIRCLE
  // points, so failing-test flags are painted as red triangles in a draw hook (canvas/device px).
  if (o.marks && o.marks.length) hooks.draw = [(u) => {
    const ctx = u.ctx, dpr = u.pxRatio || 1, h = 9 * dpr;   // ~match the 'ran hot' ring (psize 14)
    ctx.save();
    o.marks.forEach((m) => {
      const x = u.valToPos(m.x, "x", true), y = u.valToPos(m.y, "y", true);
      if (!isFinite(x) || !isFinite(y)) return;
      if (m.shape === "x") {                                 // a CORRECTNESS divergence: a bold ✕ (white-haloed)
        const r = h * 1.15;
        const cross = () => { ctx.beginPath(); ctx.moveTo(x - r, y - r); ctx.lineTo(x + r, y + r);
                              ctx.moveTo(x + r, y - r); ctx.lineTo(x - r, y + r); ctx.stroke(); };
        ctx.lineCap = "round";
        ctx.lineWidth = 4 * dpr; ctx.strokeStyle = "#fff"; cross();        // halo for contrast
        ctx.lineWidth = 2.2 * dpr; ctx.strokeStyle = m.color || CORRC; cross();
      } else {                                               // a FAILING-TESTS flag: a red triangle
        ctx.beginPath(); ctx.moveTo(x, y - h); ctx.lineTo(x - h, y + h); ctx.lineTo(x + h, y + h); ctx.closePath();
        ctx.fillStyle = m.color || FAILC; ctx.fill();
        ctx.lineWidth = 1.2 * dpr; ctx.strokeStyle = "#fff"; ctx.stroke();   // white edge for contrast
      }
    });
    ctx.restore();
  }];
  // "Run shown" overlay (o.showNow): a full-height red guide at the CURRENT run's commit-x, drawn in
  // drawClear so it sits BEHIND the grid + data + marks (never obscures them), plus a red ring around
  // that run's point(s) on its OWN platform's series, drawn just UNDER the ✕/throttle/test marks.  The
  // ring lands on the blue (GPU) or amber (CPU) line, so it also says which platform you're viewing.
  // currentRun() is read LIVE, so navigating runs only needs el._u.redraw() (no re-render -> zoom survives).
  if (o.showNow) {
    const drawNow = (u, ring) => {
      const cur = (typeof currentRun === "function") ? currentRun() : null;
      if (!cur) return;
      const nx = u.valToPos(runTime(cur), "x", true);
      if (!isFinite(nx)) return;
      const ctx = u.ctx, dpr = u.pxRatio || 1;
      if (!ring) {
        ctx.save(); ctx.strokeStyle = "rgba(163,45,45,0.55)"; ctx.lineWidth = 3 * dpr;
        ctx.beginPath(); ctx.moveTo(nx, u.bbox.top); ctx.lineTo(nx, u.bbox.top + u.bbox.height); ctx.stroke(); ctx.restore();
        return;
      }
      const di = X.indexOf(runTime(cur));
      if (di < 0) return;
      ctx.save(); ctx.strokeStyle = CORRC; ctx.lineWidth = 2 * dpr;
      S.forEach((s) => {
        if (!s.meta || s.pointsOnly || s.meta.platform !== cur.platform || s.meta.branch !== cur.branch) return;
        const v = s.ys[di]; if (v == null) return;
        const y = u.valToPos(v, "y", true); if (!isFinite(y)) return;
        ctx.beginPath(); ctx.arc(nx, y, 13 * dpr, 0, 2 * Math.PI); ctx.stroke();
      });
      ctx.restore();
    };
    hooks.drawClear = [(u) => drawNow(u, false)];
    hooks.draw = hooks.draw ? [(u) => drawNow(u, true), ...hooks.draw] : [(u) => drawNow(u, true)];
  }
  const opts = {
    width: o.width || el.clientWidth || 320, height: o.height || 210,
    scales: { x: xScale, y: yScale },
    series, axes: [xAxis, yAxis], legend: { show: false },
    // drag a region to zoom (uPlot built-in): 2-D box zoom on the scaling panels, x-only on the
    // time-series history panels.  Double-click resets.  (The fixed-array x-range that previously
    // hard-clamped the scale is gone — see xPad — so the drag actually takes now.)
    // drag.dist: a drag shorter than this (px) is NOT a zoom — it falls through to a plain click, so
    // a click with a pixel or two of jitter still selects the point instead of zooming a sliver.
    cursor: { points: { size: 7 }, drag: { x: true, y: !o.xTime, dist: 6 } },
    hooks: (hooks.setCursor || hooks.setScale || hooks.draw || hooks.drawClear) ? hooks : undefined,
  };
  if (el._u) { el._u.destroy(); el._u = null; }
  el.innerHTML = "";
  try { el._u = new uPlot(opts, data, el); } catch (e) { el.innerHTML = "<p class='muted'>chart error: " + e.message + "</p>"; return null; }
  if (tip) el.appendChild(tip);
  if (o.syncX) o.syncX.push(el._u);
  // Optional: a plain click (vs a drag, which zooms) selects the nearest point.
  if (o.onPick) {
    const over = el.querySelector(".u-over");
    if (over) over.addEventListener("click", () => {
      const u = el._u; const np = nearestPoint(u, 40);
      if (!np) return;
      const oi = np.di - padOff;
      if (oi >= 0 && oi < xs.length) o.onPick(specs[np.si - 1], oi);
    });
  }
  return el._u;
}

// ---- header / selectors ------------------------------------------------------
function goOptions() {
  const run = latestRun(ui_state.platform, ui_state.branch);
  if (!run) return [];
  const combos = uniq(run.cells.map((c) => c.geom + "|" + c.op));
  return combos.sort((a, b) => {
    const [ga, oa] = a.split("|"), [gb, ob] = b.split("|");
    return GEOM_ORDER.indexOf(ga) - GEOM_ORDER.indexOf(gb) || OP_ORDER.indexOf(oa) - OP_ORDER.indexOf(ob);
  });
}
function syncGoSelect() {
  const opts = goOptions();
  if (!opts.includes(ui_state.go)) ui_state.go = opts.includes("cone|vcd_nonconst") ? "cone|vcd_nonconst" : opts[0];
  fillSelect("op", opts, ui_state.go, opts.map((s) => s.replace("|", " · ")));
}

// ---- tiles + drill-down ------------------------------------------------------
// Headline numbers for one platform's run of the selected branch (null if none).
function platMetrics(plat) {
  const run = runOnPlat(plat);
  if (!run) return null;
  return { run,
    cells: run.cells.length,   // TOTAL configs attempted (matches status_nightly's "cells N");
                               // failures are surfaced via cellsFailed (the sub + drill-down + red markers)
    cellsFailed: run.cells.filter((c) => c.failed).length,
    gate: run.gate.hard.length,
    correctness: runCorrCells(run),   // distinct divergent configs (own severity tier)
    perfHard: runPerfHard(run).length,  // hard hits that are NOT correctness (memory / structural / ok->fail)
    // The authoritative count is the pytest SUMMARY's `failed` (what status_nightly uses).  The
    // `failures` node-id LIST is only for the drill-down and can be empty if the log format hid the
    // names (e.g. pytest-xdist without -ra), so counting it would under-report (showed 0 vs 3).
    testsFailed: run.tests ? (run.tests.failed || 0) : 0,
    testsPassed: run.tests ? run.tests.passed : null };
}
function renderTiles() {
  const box = $("tiles");
  const cur = currentRun();
  if (!cur) { box.innerHTML = "<p class='muted'>no runs.</p>"; return; }
  // The first three tiles always show BOTH platforms (cpu + gpu) for the selected branch.
  const mets = {}; M.platforms.forEach((p) => mets[p] = platMetrics(p));
  const anyBad = (isBad) => M.platforms.some((p) => mets[p] && isBad(mets[p]));
  const pvs = (pick, isBad) => `<div class="pvs">` + M.platforms.map((p) => {
    const m = mets[p];
    if (!m) return `<span class="pv none" title="no ${p.toUpperCase()} run for this commit">${p.toUpperCase()}<b>—</b></span>`;
    return `<span class="pv">${p.toUpperCase()}<b class="${isBad(m) ? "bad" : ""}">${pick(m)}</b></span>`;
  }).join("") + `</div>`;
  const health = [
    { id: "cells", lbl: "configs measured", body: pvs((m) => m.cells, (m) => m.cellsFailed > 0),
      click: anyBad((m) => m.cellsFailed > 0), sub: anyBad((m) => m.cellsFailed > 0) ? "failures — click" : "all ran" },
    { id: "correctness", lbl: "correctness", body: pvs((m) => m.correctness, (m) => m.correctness > 0),
      click: true, sub: anyBad((m) => m.correctness > 0) ? "DIVERGENT — click" : "fingerprints match" },
    { id: "gate", lbl: "performance regressions", body: pvs((m) => m.perfHard, (m) => m.perfHard > 0),
      click: true, sub: "click for details" },
    { id: "tests", lbl: "tests failed", body: pvs((m) => m.testsFailed, (m) => m.testsFailed > 0),
      click: anyBad((m) => m.testsFailed > 0), sub: anyBad((m) => m.testsFailed > 0) ? "failures — click" : "none failing" },
  ];
  // Flags for the shown run: tint the tile + ⚠ badge(s).  Failing tests and throttling are RED
  // (warn-throttled); merely running hot is amber (warn-hot).  Both badges show if both apply.
  const therm = runThermal(cur);
  const tf = (cur.tests && cur.tests.failed) || 0;
  const corr = runCorrCells(cur);
  const corrAlert = runAlert(cur);          // incorrect AND not acknowledged -> dominant red
  const corrAcked = corr > 0 && !corrAlert; // incorrect but acknowledged -> muted note, audit trail
  const sev = (tf || (therm && therm.sev === "throttled")) ? "throttled" : (therm ? "hot" : null);
  // Correctness OUTRANKS perf/thermal: an unacknowledged INCORRECT run is red regardless of the rest.
  const warnTile = corrAlert ? " incorrect" : (sev ? ` warn-${sev}` : "");
  const warnLine =
    (corrAlert ? `<div class="warnflag incorrect">⚠ INCORRECT — ${corr} divergent config${corr > 1 ? "s" : ""}</div>`
       : corrAcked ? `<div class="warnflag muted">✓ ${corr} correctness divergence${corr > 1 ? "s" : ""} (acknowledged)</div>` : "")
    + (therm ? `<div class="warnflag ${therm.sev}">⚠ ${therm.sev === "throttled" ? "throttled" : "ran hot"} · n=${therm.ndevs.join(", ")}${therm.peak ? ` · up to ${therm.peak}°C` : ""}</div>` : "")
    + (tf ? `<div class="warnflag throttled">⚠ ${tf} test${tf > 1 ? "s" : ""} failed</div>` : "");
  // Run navigation (this tile): ◀/▶ step through THIS platform+branch's runs by commit time; the ⇄
  // toggle jumps to the OTHER platform's run of the SAME commit (greyed if that platform never measured
  // this commit).  Both keep the open drill-down so you can step and watch the same panel update.
  const series = runsFor(ui_state.platform, ui_state.branch).slice().sort((a, b) => runTime(a) - runTime(b));
  const idx = series.findIndex((r) => runKey(r) === runKey(cur));   // cur, not ui_state.runKey (null until you navigate)
  const prevR = idx > 0 ? series[idx - 1] : null;
  const nextR = (idx >= 0 && idx < series.length - 1) ? series[idx + 1] : null;
  const otherPlat = M.platforms.find((p) => p !== ui_state.platform);
  const otherR = (otherPlat && cur.commit_full)
    ? (runsFor(otherPlat, ui_state.branch).filter((r) => r.commit_full === cur.commit_full)
         .sort((a, b) => runTime(b) - runTime(a))[0] || null)
    : null;
  const runTile =
    `<div class="tile${warnTile}" data-click="false">
       <div class="lbl">run shown</div>
       <div class="runnav">
         <button class="rn-step" data-dir="prev" ${prevR ? "" : "disabled"} title="${prevR ? "older run · " + commitMinute(prevR) : "oldest run"}">◀</button>
         <button class="rn-step" data-dir="next" ${nextR ? "" : "disabled"} title="${nextR ? "newer run · " + commitMinute(nextR) : "newest run"}">▶</button>
         ${idx >= 0 && series.length > 1 ? `<span class="rn-pos">${idx + 1}/${series.length}</span>` : ""}
         <button class="rn-plat" ${otherR ? "" : "disabled"} title="${otherR ? "same commit on " + (otherPlat || "").toUpperCase() : "no " + (otherPlat || "other").toUpperCase() + " run for this commit"}">⇄&nbsp;${(otherPlat || "").toUpperCase()}</button>
       </div>
       <div class="when">${commitMinute(cur)}</div>
       <div class="sub"><b>${ui_state.branch}</b> · <b>${ui_state.platform.toUpperCase()}</b> · ${cur.commit}${cur.dirty ? " · dirty" : ""}</div>
       ${warnLine}
     </div>`;
  box.innerHTML = health.map((t) =>
    `<div class="tile ${t.click ? "click" : ""} ${ui_state.openTile === t.id ? "open" : ""}" data-id="${t.id}" data-click="${!!t.click}">
       <div class="lbl">${t.lbl}</div>${t.body}<div class="sub">${t.sub}</div></div>`
  ).join("") + runTile;
  box.querySelectorAll(".tile").forEach((el) => {
    if (el.dataset.click === "true") el.onclick = () => {
      ui_state.openTile = ui_state.openTile === el.dataset.id ? null : el.dataset.id;
      renderTiles(); renderDetail();
    };
  });
  // Run-shown nav: ◀/▶ step within the series, ⇄ swaps to the other platform's same-commit run.
  box.querySelectorAll(".rn-step").forEach((b) => b.onclick = () => showRun(b.dataset.dir === "prev" ? prevR : nextR));
  const platBtn = box.querySelector(".rn-plat");
  if (platBtn) platBtn.onclick = () => showRun(otherR);
}
function renderDetail() {
  const box = $("detail");
  if (!ui_state.openTile) { box.innerHTML = ""; return; }
  const titles = { gate: "Performance regressions — hard-gate hits", correctness: "Correctness — fingerprint divergences",
                   tests: "Failing tests", cells: "Failed configs" };
  const pct = (v) => v == null ? "?" : v + "%";
  // The drill-down covers BOTH platforms (matching the tiles).  Gate/correctness get a one-time
  // threshold explanation since the thresholds are identical across platforms.
  let intro = "";
  const anyRun = M.platforms.map(runOnPlat).find(Boolean);
  const gc = (anyRun && anyRun.gate_config) || {};
  if (ui_state.openTile === "gate") {
    intro = `<p>Each run is compared per config + metric against its reference run(s).  This panel shows <b>performance</b> regressions — correctness has its own tile. <b>Hard:</b> structural change, ok→fail, expected-but-absent, GPU peak-memory &gt;${pct(gc.mem_hard_pct)}. <b>Soft:</b> speedup drop &gt;${pct(gc.speedup_warn_pct)}, time &gt;${pct(gc.time_soft_pct)}, CPU memory, sweep add/drop.</p>`;
  } else if (ui_state.openTile === "correctness") {
    const ct = M.corr_tol || {};
    intro = `<p>Correctness compares the recon <b>fingerprint</b> against four references — the <b>prior run</b> on this branch, the latest <b>main</b>, <b>single-device n=1</b> within the same run, and the <b>other platform</b> (CPU↔GPU) at the same commit. Flags a float64 {sum, mean, l2norm} relative change beyond ${ct.single ?? "?"} (single-shot) / ${ct.iter ?? "?"} (iterated VCD) / ${ct.xdev ?? "?"} (cross-device) / ${ct.xplat ?? "?"} (cross-platform), or a shape/dtype change.${M.cleared_through ? " Divergences on commits dated ≤ " + M.cleared_through + " are acknowledged." : ""}</p>`;
  }
  const section = (plat) => {
    const run = runOnPlat(plat);
    const head = `<h4>${plat.toUpperCase()}${run ? ` · ${run.commit} · ${commitMinute(run)}` : ""}</h4>`;
    if (!run) return head + `<p class="muted">no ${plat.toUpperCase()} run for this commit.</p>`;
    if (ui_state.openTile === "correctness") {
      const fs = runCorr(run);
      if (!fs.length) return head + `<p class="muted">fingerprints match the references — no divergence.</p>`;
      // Group findings by cell; within a cell, one "vs <reference>" block per reference (prior / main /
      // single-device) with that reference's bulleted discrepancies.
      const byCell = {};
      fs.forEach((f) => { (byCell[f.cell] = byCell[f.cell] || []).push(f); });
      const blocks = Object.keys(byCell).map((cell) => {
        const coords = cellCoords(cell);
        const refs = byCell[cell].map((f) =>
          `<div class="vsref">vs ${f.basis}</div><ul class="discr">${f.discrepancies.map((d) => `<li>${d}</li>`).join("")}</ul>`).join("");
        return `<div class="hitcell"><div class="hitcoords">${coords}</div>${refs}</div>`;
      }).join("");
      return head + blocks;
    }
    if (ui_state.openTile === "gate") {
      const hits = runPerfHard(run);
      const cmp = (run.gate.compared_to || []).map((c) => priorLabel(c, plat)).join(", ") || "its reference run(s)";
      if (!hits.length) return head + `<p class="muted">no performance regressions (result: ${run.gate.result || "?"}).</p>`;
      // Group hits by cell: one coords line + a bullet per discrepancy.
      const byCell = {};
      hits.forEach((h) => { const k = h.cell || "—"; (byCell[k] = byCell[k] || []).push(h); });
      const blocks = Object.keys(byCell).map((cell) => {
        const coords = cell === "—" ? "(run-level)" : cellCoords(cell);
        const bullets = byCell[cell].map((h) => `<li>${h.detail || h.text}</li>`).join("");
        return `<div class="hitcell"><div class="hitcoords">${coords}</div><ul class="discr">${bullets}</ul></div>`;
      }).join("");
      return head + `<p class="muted">vs ${cmp}</p>${blocks}`;
    }
    if (ui_state.openTile === "tests") {
      const t = run.tests, f = (t && t.failures) || [];
      return head + (f.length ? `<ul>${f.map((x) => `<li class="bad">${x}</li>`).join("")}</ul>`
        : (t && t.failed ? `<p class="bad">${t.failed} failing</p><p class="muted">(test names not captured in this log)</p>`
        : `<p class="muted">${t ? t.passed + " passed, none failing" : "no test log"}.</p>`));
    }
    const f = run.cells.filter((c) => c.failed);  // "cells"
    return head + (f.length ? `<ul>${f.map((c) => `<li class="bad">${cellKey(c)}${c.oom ? " — OOM" : ""}${c.error ? " — " + c.error : ""}</li>`).join("")}</ul>`
      : `<p class="muted">all configs ran.</p>`);
  };
  box.innerHTML = `<div class="detail-box"><h3>${titles[ui_state.openTile] || ""}</h3>${intro}${M.platforms.map(section).join("")}</div>`;
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
  if (ui_state.ref === "best") { const r = M.records[ui_state.platform + "|" + ui_state.branch]; const e = r && r[key]; return e && e[metric] ? e[metric].value : null; }
  const run = refRun(); if (!run) return null;
  const c = findCell(run, key);
  return c ? c[metric] : null;
}
// reference overlay series for the absolute (vs-size) panels.  No device-count restriction: a
// reference run carries whatever device counts it measured (main is n=1-only, prerelease shards
// parallel, etc.), and refVal returns null where the ref lacks a cell -> spanGaps bridges it.
function refSeries(geom, op, sizes, ndevs, metric, div) {
  if (ui_state.ref === "none") return [];
  const lab = REF_LABEL[ui_state.ref] || ui_state.ref;
  const out = [];
  ndevs.forEach((nd) => {
    const ys = sizes.map((s) => { const v = refVal(geom, op, s, nd, metric); return v != null ? v / div : null; });
    if (ys.some((y) => y != null)) out.push({ label: `${lab} n=${nd}`, color: REFC, ys, width: 4, fillPoints: true, psize: 4 });
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
// Thermal markers on cells whose timing may be suspect — two tiers per device curve: a filled amber
// disc where the driver actually throttled (causal), a hollow amber ring where a GPU merely ran hot
// (advisory).  (A hot GPU gates the slowest-device multi-GPU timing, so a 2x jump with no code change
// is usually this, not a regression.)
function throttleSeries(run, geom, op, sizes, ndevs, field, div) {
  const out = [];
  ndevs.forEach((nd) => {
    const cells = sizes.map((s) => findCell(run, geom + "|" + op + "|" + s + "|" + nd));
    const val = (c) => (c && !c.failed && c[field] != null) ? c[field] / div : null;
    // confirmed throttle (clocks clamped) -> filled amber disc; ran hot only -> hollow amber ring
    const thr = cells.map((c) => (c && cellThrottled(c)) ? val(c) : null);
    const hot = cells.map((c) => (c && cellHot(c) && !cellThrottled(c)) ? val(c) : null);
    if (thr.some((y) => y != null)) out.push({ label: "throttled n=" + nd, color: THROTC, ys: thr, pointsOnly: true, fillPoints: true, psize: 12, pw: 2.5 });
    if (hot.some((y) => y != null)) out.push({ label: "hot n=" + nd, color: THROTC, ys: hot, pointsOnly: true, hollow: true, psize: 14, pw: 2.5 });
  });
  return out;
}

function renderScaling() {
  const run = currentRun();
  const [geom, op] = ui_state.go.split("|");
  $("sv-meta").textContent = run ? `${geom} · ${op} — ${ui_state.branch} @ ${run.commit} · ${commitMinute(run)}` : "";
  if (ui_state.view === "table") { $("sv-plot").style.display = "none"; $("sv-table").style.display = ""; renderScalingTable(run, geom, op); return; }
  $("sv-plot").style.display = ""; $("sv-table").style.display = "none";
  const g = gridFor(run, geom, op);
  const { sizes, ndevs, at } = g;
  if (!sizes.length) { $("pTime").innerHTML = "<p class='muted'>no cells.</p>"; return; }
  const PLAT = (ui_state.platform || "").toUpperCase();
  // The ∝-voxels/views "ideal" reference doesn't hold for the translation geometry — suppress it
  // (line + caption note) there; keep it for parallel/cone/multiaxis.
  const showIdeal = geom !== "translation";
  $("capTime").textContent = `${PLAT}: time vs size${showIdeal ? ` · ideal ∝ ${IDEAL_BASIS[op] || "voxels"}` : ""}`;
  $("capMem").textContent = `${PLAT}: memory vs size${showIdeal ? " · ideal ∝ voxels" : ""}`;
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
  const w = $("pTime").clientWidth || 460;
  // Hover tooltips: a rich cell readout (config + result) over a measured curve; a plain label+value
  // over the ideal/reference/gate overlays.  `fb` formats the overlay's y for the panel's unit.
  const cellLine = (c) => `<span class="tdim">time</span> ${c.min_ms != null ? (c.min_ms / 1000).toFixed(2) + " s" : "—"}`
    + `<br><span class="tdim">peak mem</span> ${fmtGB(c.mem_mb)}`
    + (c.speedup != null ? `<br><span class="tdim">speedup</span> ${c.speedup.toFixed(2)}×` : "")
    + (cellHot(c) ? `<br><span class="thr">⚠ ${hotWarn(c)}</span>` : "");
  const sizeTip = (fb) => (spec, idx) => {
    const size = sizes[idx], m = /n=(\d+)/.exec(spec.label || "");
    if (!m || spec.color === REFC) { const y = spec.ys[idx]; return y == null ? null : `<b>${geom} · ${op}</b> · ${size}<br>${spec.label}: ${fb(y)}`; }
    const c = at(size, +m[1]); if (!c) return null;
    const head = `<b>${geom} · ${op}</b> · ${size} · n=${m[1]}`;
    return c.failed ? `${head}<br><span class="bad">${c.oom ? "OOM" : "FAILED"}</span>${c.error ? " — " + c.error : ""}` : `${head}<br>${cellLine(c)}`;
  };
  const devTip = (spec, idx) => {
    const nd = ndevs[idx], size = spec.label;
    if (!/^\d+x\d+x\d+$/.test(size)) { const y = spec.ys[idx]; return y == null ? null : `n=${nd}<br>${spec.label}: ${fmtNum(y)}`; }
    const c = at(size, nd); if (!c) return null;
    const head = `<b>${geom} · ${op}</b> · ${size} · n=${nd}`;
    return c.failed ? `${head}<br><span class="bad">${c.oom ? "OOM" : "FAILED"}</span>` : `${head}<br>${cellLine(c)}`;
  };

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
  const timeIdeal = (showIdeal && aT && aT.min_ms != null) ? [{ label: "ideal", color: IDEAL, dash: [5, 4], width: 1.5, psize: 0,
    ys: xvol.map((v) => (aT.min_ms / 60000) * Math.pow(v / aV, texp)) }] : [];
  const gT = gateSeries(run, geom, op, sizes, "time", 60000);
  // big red dots for failed configs, placed on the curve at the failing size
  const timeFails = ndevs.map((nd, ci) => {
    const fi = sizes.map((s, i) => { const c = at(s, nd); return (c && c.failed) ? i : -1; }).filter((i) => i >= 0);
    const yy = interpFails(xvol, timeCurves[ci].ys, fi, true, true, timeIdeal.length ? timeIdeal[0].ys : null);
    return yy ? { label: "failed n=" + nd, color: devColor(nd), ring: FAILC, ys: yy, pointsOnly: true, fillPoints: true, psize: 11, pw: 3 } : null;
  }).filter(Boolean);
  const timeSpecs = [...timeFails, ...(gT ? [gT] : []), ...throttleSeries(run, geom, op, sizes, ndevs, "min_ms", 60000), ...refSeries(geom, op, sizes, ndevs, "min_ms", 60000), ...timeCurves, ...timeIdeal];
  linePlot($("pTime"), xvol, timeSpecs, { width: w, xLog: true, yLog: true, tightLog: true, yPad: 0.06, xSplits: xticks, xLabels, xPad: 1.7, yLabelText: "minutes", tooltip: sizeTip((y) => y.toFixed(2) + " min") });

  // --- memory vs size (log-log, GB) ---  (same draw-order rule as the time panel)
  const memCurves = ndevs.map((nd) => ({ label: "n=" + nd, color: devColor(nd),
    ys: sizes.map((s) => { const c = at(s, nd); return c && !c.failed && c.mem_mb != null ? c.mem_mb / 1024 : null; }) }));
  const memIdeal = (showIdeal && aM && aM.mem_mb != null) ? [{ label: "ideal", color: IDEAL, dash: [5, 4], width: 1.5, psize: 0,
    ys: xvol.map((v) => (aM.mem_mb / 1024) * (v / aV)) }] : [];
  const gM = gateSeries(run, geom, op, sizes, "memory", 1024);
  const memFails = ndevs.map((nd, ci) => {
    const fi = sizes.map((s, i) => { const c = at(s, nd); return (c && c.failed) ? i : -1; }).filter((i) => i >= 0);
    const yy = interpFails(xvol, memCurves[ci].ys, fi, true, true, memIdeal.length ? memIdeal[0].ys : null);
    return yy ? { label: "failed n=" + nd, color: devColor(nd), ring: FAILC, ys: yy, pointsOnly: true, fillPoints: true, psize: 11, pw: 3 } : null;
  }).filter(Boolean);
  const memSpecs = [...memFails, ...(gM ? [gM] : []), ...throttleSeries(run, geom, op, sizes, ndevs, "mem_mb", 1024), ...refSeries(geom, op, sizes, ndevs, "mem_mb", 1024), ...memCurves, ...memIdeal];
  linePlot($("pMem"), xvol, memSpecs, { width: w, xLog: true, yLog: true, tightLog: true, yPad: 0.06, xSplits: xticks, xLabels, xPad: 1.7, yLabelText: "GB", tooltip: sizeTip((y) => y.toFixed(2) + " GB") });

  // --- speedup vs devices (one curve per size; ideal linear) ---
  const w2 = $("pSpeed").clientWidth || 460;
  const speedCurves = sizes.map((s, i) => { const base = at(s, ndevs[0]);
    return { label: s, color: SIZEC[i % SIZEC.length],
      ys: ndevs.map((nd) => { const c = at(s, nd); return c && base && !c.failed && !base.failed ? (base.min_ms / c.min_ms) * ndevs[0] : null; }) }; });
  const speedIdeal = ndevs.slice();
  const speedFails = sizes.map((s, ci) => {
    const fi = ndevs.map((nd, i) => { const c = at(s, nd); return (c && c.failed) ? i : -1; }).filter((i) => i >= 0);
    const yy = interpFails(ndevs, speedCurves[ci].ys, fi, false, false, speedIdeal);
    return yy ? { label: s, color: SIZEC[ci % SIZEC.length], ring: FAILC, ys: yy, pointsOnly: true, fillPoints: true, psize: 11, pw: 3 } : null;
  }).filter(Boolean);
  const speedSpecs = [...speedFails, ...speedCurves, { label: "ideal", color: IDEAL, dash: [5, 4], width: 1.5, psize: 0, ys: speedIdeal }];
  linePlot($("pSpeed"), ndevs, speedSpecs, { width: w2, padAll: 0.07, xSplits: ndevs, xLabels: Object.fromEntries(ndevs.map((n) => [n, String(n)])), yfmt: (v) => v.toFixed(0) + "×", yLabelText: "speedup", xLabelText: "devices", tooltip: devTip });

  // --- per-device memory ÷ sino shard (one curve per size; ideal 2x) ---
  const shardCurves = sizes.map((s, i) => ({ label: s, color: SIZEC[i % SIZEC.length],
    ys: ndevs.map((nd) => { const c = at(s, nd); if (!c || c.failed || c.mem_mb == null) return null;
      const shardMB = (sizeVol(s) * 4 / nd) / (1024 * 1024); return c.mem_mb / shardMB; }) }));
  const shardFails = sizes.map((s, ci) => {
    const fi = ndevs.map((nd, i) => { const c = at(s, nd); return (c && c.failed) ? i : -1; }).filter((i) => i >= 0);
    const yy = interpFails(ndevs, shardCurves[ci].ys, fi, false, false, ndevs.map(() => 2));
    return yy ? { label: s, color: SIZEC[ci % SIZEC.length], ring: FAILC, ys: yy, pointsOnly: true, fillPoints: true, psize: 11, pw: 3 } : null;
  }).filter(Boolean);
  const shardSpecs = [...shardFails, ...shardCurves, { label: "ideal 2×", color: IDEAL, dash: [5, 4], width: 1.5, psize: 0, ys: ndevs.map(() => 2) }];
  linePlot($("pShard"), ndevs, shardSpecs, { width: w2, padAll: 0.07, xSplits: ndevs, xLabels: Object.fromEntries(ndevs.map((n) => [n, String(n)])), yfmt: (v) => v.toFixed(1) + "×", yLabelText: "mem ÷ shard", xLabelText: "devices", tooltip: devTip });

  renderScalingLegend(ndevs, sizes, showIdeal);
}
function renderScalingLegend(ndevs, sizes, showIdeal) {
  const k = (c, t, dash) => `<span class="k"><span class="sw" style="background:${c};${dash ? "height:0;border-top:2px dashed " + c : ""}"></span>${t}</span>`;
  // failed-config marker = curve-coloured centre with a red ring (centre shown
  // neutral here since the colour varies per curve)
  const ringDot = (t) => `<span class="k"><span style="width:13px;height:13px;border-radius:50%;background:var(--surface2);border:3px solid ${FAILC};box-sizing:border-box;display:inline-block"></span>${t}</span>`;
  const devs = ndevs.map((n) => k(devColor(n), "n=" + n)).join("");
  const szs = sizes.map((s, i) => k(SIZEC[i % SIZEC.length], s)).join("");
  // active comparison: solid black swatch + display name + provenance (branch @ commit)
  const refNote = ui_state.ref !== "none"
    ? `<span class="k"><span class="sw" style="background:${REFC};height:4px"></span>${REF_LABEL[ui_state.ref] || ui_state.ref}${refProvenance() ? " (" + refProvenance() + ")" : ""}</span>` : "";
  // Top legend sits above time & memory (device-count curves + the overlay ref);
  // the second legend sits above speedup & shard (size curves).
  $("sv-legend").innerHTML =
    `<span class="grp">${devs}</span>` +
    `<span class="grp">${showIdeal ? k(IDEAL, "ideal", true) : ""}${ringDot("failed config")}<span class="k"><span class="ring"></span>gate fail</span><span class="k"><span class="ring" style="border-color:${THROTC}"></span>ran hot</span><span class="k"><span class="dot" style="background:${THROTC}"></span>throttled</span>${refNote}</span>`;
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
  const refActive = ui_state.ref !== "none";
  const dCell = (cur, ref, lowerBetter) => {
    if (cur == null || ref == null || ref === 0) return "<td class='num'>—</td>";
    const d = ((cur - ref) / Math.abs(ref)) * 100;
    if (Math.abs(d) < 1) return `<td class='num'>${d > 0 ? "+" : ""}${d.toFixed(1)}%</td>`;
    const worse = lowerBetter ? d > 0 : d < 0;
    return `<td class='num ${worse ? "up" : "dn"}'>${d > 0 ? "+" : ""}${d.toFixed(1)}%</td>`;
  };
  const tbl = (title, field, div, fmt, unit) => {
    let h = `<table class='grid'><caption>${title}${unit ? " (" + unit + ")" : ""}${refActive ? " · Δ vs " + ui_state.ref : ""}</caption><thead><tr><th>devices</th>`;
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
// Headline-op time + peak memory at each geometry's LARGEST size, for device count `n` (gate count
// is n-independent).  `geoms` is the active group; `timeOp` is its headline op (vcd for cone/parallel,
// back for translation/multiaxis — they don't run vcd).  Sizes differ per geometry now, so the
// "largest size" is computed PER geometry (a global max would miss the smaller new geometries).
function aggregate(run, n, geoms, timeOp) {
  // Keep the source cell behind each point (timeCell/memCell) so the history can flag GPU
  // throttling — the amber ring + tooltip warning — exactly as the scaling panels do.
  const out = { time: {}, mem: {}, timeCell: {}, memCell: {}, gate: run.gate.hard.length,
                gatePerf: runPerfHard(run).length,   // perf-only hard count (correctness is its own signal)
                testsFailed: (run.tests && run.tests.failed) || 0,
                corrAlert: runAlert(run) };   // unacknowledged correctness divergence -> ✕ marker
  geoms.forEach((gm) => {
    const gmSizes = run.cells.filter((c) => c.geom === gm).map((c) => c.size);
    if (!gmSizes.length) { out.time[gm] = out.mem[gm] = out.timeCell[gm] = out.memCell[gm] = null; return; }
    const focus = Math.max(...gmSizes.map(sizeVol));
    const focusSize = gmSizes.find((s) => sizeVol(s) === focus);
    const tc = run.cells.find((c) => c.geom === gm && c.op === timeOp && c.size === focusSize && c.ndev === n && !c.failed);
    out.time[gm] = tc && tc.min_ms != null ? tc.min_ms / 60000 : null;
    out.timeCell[gm] = tc || null;
    const mems = run.cells.filter((c) => c.geom === gm && c.size === focusSize && c.ndev === n && !c.failed && c.mem_mb != null);
    const mc = mems.length ? mems.reduce((a, b) => (b.mem_mb > a.mem_mb ? b : a)) : null;
    out.mem[gm] = mc ? mc.mem_mb / 1024 : null;
    out.memCell[gm] = mc;
  });
  return out;
}
// Click a history point -> show that run (and switch platform/branch to match).
// Load a specific run into the run-dependent views (tiles / detail / scaling), syncing the platform +
// branch selectors.  History is left untouched so its zoom survives.  By default the open drill-down is
// PRESERVED (so the run-shown ◀/▶ can step while you watch the same panel update); pass resetOpen=true to
// close it (History clicks do, since you're navigating away from a specific point).
// Re-highlight the "run shown" guide/ring on the History panels WITHOUT re-rendering them (the draw
// hooks read currentRun() live, so a redraw() suffices) — so the marker tracks navigation while the
// panels' zoom survives.
function refreshHistoryNow() {
  ["hVcd", "hMem", "hGate"].forEach((id) => { const el = $(id); if (el && el._u) el._u.redraw(); });
}
function showRun(r, resetOpen) {
  if (!r) return;
  ui_state.platform = r.platform; ui_state.branch = r.branch; ui_state.runKey = runKey(r);
  if (resetOpen) ui_state.openTile = null;
  fillSelect("platform", M.platforms, ui_state.platform);
  fillSelect("branch", branchesFor(ui_state.platform), ui_state.branch);
  renderTiles(); renderDetail(); syncGoSelect(); renderScaling(); refreshHistoryNow();
}
function pickRun(spec, idx) {
  const t = spec._xs[idx];
  showRun(runsFor(spec.meta.platform, spec.meta.branch).find((x) => runTime(x) === t), true);
}
function renderHistory() {
  // The history spans BOTH platforms and all branches; x is commit time
  // (falls back to collection date for older runs).
  const xs = uniq(M.runs.map(runTime)).sort((a, b) => a - b);
  const n = ui_state.histN;
  const group = HIST_GROUPS.find((g) => g.id === ui_state.histGroup) || HIST_GROUPS[0];
  // Branch filter: "all" shows every branch (colour=platform, style=geometry); selecting one
  // restricts all three panels to that branch only.
  const branches = (ui_state.histBranch && ui_state.histBranch !== "all")
    ? M.branches.filter((b) => b === ui_state.histBranch) : M.branches;
  $("hCapVcd").textContent = `${group.opLabel} time at largest size (n=${n})`;
  $("hCapMem").textContent = `peak memory at largest size (n=${n})`;
  const aggByPB = {};  // "platform|branch" -> runTime -> aggregate (for the active geometry group)
  M.runs.forEach((r) => { const key = r.platform + "|" + r.branch; (aggByPB[key] = aggByPB[key] || {})[runTime(r)] = aggregate(r, n, group.geoms, group.op); });

  // colour = platform, line-style = geometry (one solid + one dashed per group — see GEOM_DASH).
  const cellField = (pick) => (pick === "time" ? "timeCell" : "memCell");
  const specsFor = (pick) => {
    const out = [], markers = [];
    M.platforms.forEach((plat) => branches.forEach((b) => group.geoms.forEach((gm) => {
      const agg = aggByPB[plat + "|" + b]; if (!agg) return;
      const ys = xs.map((t) => { const a = agg[t]; return a && a[pick][gm] != null ? a[pick][gm] : null; });
      if (!ys.some((y) => y != null)) return;
      const meta = { platform: plat, branch: b, geom: gm, pick };
      out.push({ label: `${plat} ${gm}`, color: PLATC[plat] || IDEAL,
        dash: GEOM_DASH[gm], ys, _xs: xs, meta });
      // two-tier thermal markers, re-derived client-side like the scaling panels: a filled amber disc
      // where the driver throttled (causal), a hollow amber ring where a GPU merely ran hot (advisory).
      const valAt = (t) => { const a = agg[t]; return a ? a[pick][gm] : null; };
      const cellAt = (t) => { const a = agg[t]; return a ? a[cellField(pick)][gm] : null; };
      const thr = xs.map((t) => { const c = cellAt(t); return (c && cellThrottled(c)) ? valAt(t) : null; });
      const hot = xs.map((t) => { const c = cellAt(t); return (c && cellHot(c) && !cellThrottled(c)) ? valAt(t) : null; });
      if (thr.some((y) => y != null)) markers.push({ label: `throttled ${plat} ${gm}`, color: THROTC,
        ys: thr, _xs: xs, meta, pointsOnly: true, fillPoints: true, psize: 12, pw: 2.5 });
      if (hot.some((y) => y != null)) markers.push({ label: `hot ${plat} ${gm}`, color: THROTC,
        ys: hot, _xs: xs, meta, pointsOnly: true, hollow: true, psize: 14, pw: 2.5 });
    })));
    // markers FIRST -> lowest series index -> drawn on top and win the hover tie at a flagged point
    return [...markers, ...out];
  };
  // With a single point, uPlot's auto time-range sprawls across years; pin a ±12h window.
  const DAY = 86400;
  const xr = xs.length < 2 ? [xs[0] - DAY / 2, xs[0] + DAY / 2] : null;
  // Pad the time domain by 5% of its span on each end so the first/last run isn't on the boundary
  // (where a click misses) — proportional, so it scales as the history grows instead of a fixed day.
  const xpad = xs.length > 1 ? (xs[xs.length - 1] - xs[0]) * 0.05 : 0;
  // hover tooltip: the same identity as the "run shown" tile (branch · platform · commit date+time).
  const histTip = (spec, idx) => {
    const r = runsFor(spec.meta.platform, spec.meta.branch).find((x) => runTime(x) === spec._xs[idx]);
    if (!r) return null;
    const y = spec.ys[idx], m = spec.meta;
    // value line: a throttle-marker spec shares its run-line's identity, so show "<plat> <geom>"
    // rather than its internal "throttled …" label.
    const vlabel = m.geom ? `${m.platform} ${m.geom}` : spec.label;
    let warn = "";
    if (m.geom && m.pick) {
      const a = (aggByPB[m.platform + "|" + m.branch] || {})[spec._xs[idx]];
      const c = a ? a[cellField(m.pick)][m.geom] : null;
      if (c && cellHot(c)) warn = `<br><span class="thr">⚠ ${hotWarn(c)}</span>`;
    }
    // correctness divergence flag — the most severe, shown first (same source as the ✕ marker / tile).
    const corrN = runCorrCells(r);
    const corrWarn = corrN ? `<br><span class="corr">✕ ${corrN} correctness divergence${corrN > 1 ? "s" : ""}${runAcked(r) ? " (acknowledged)" : ""}</span>` : "";
    // tests-failed flag — same info as the run-shown tile's red badge.
    const tf = (r.tests && r.tests.failed) || 0;
    const testWarn = tf ? `<br><span class="bad">⚠ ${tf} test${tf > 1 ? "s" : ""} failed</span>` : "";
    // unit follows the panel: minutes for the time panel, GB for memory (gate has no pick -> bare count).
    const unit = m.pick === "mem" ? " GB" : m.pick === "time" ? " min" : "";
    return `<b>${r.branch}</b><br>${r.platform.toUpperCase()} · ${commitMinute(r)}`
      + `<br><span class="tdim">commit</span> ${r.commit}${r.dirty ? " · dirty" : ""}`
      + (y != null ? `<br><span class="tdim">${vlabel}</span> ${fmtNum(y)}${unit}` : "") + warn + corrWarn + testWarn;
  };
  // All three history plots share one x (commit time): a sync group links their zoom so dragging
  // any one re-ranges all three to the same window (and a double-click reset clears all three).
  const histGroup = [];
  // Red triangle on a run that had FAILING TESTS (a run-level flag, not per-cell like the thermal
  // markers): one per (platform, branch) failing run, sat on its drawn point for the panel's metric.
  // run-level overlay marks sat on the run's drawn point for the panel's metric: a red triangle for
  // FAILING TESTS, a bold ✕ for an unacknowledged CORRECTNESS divergence (flag = the aggregate field).
  const runMarks = (pick, flag, extra) => {
    const out = [];
    M.platforms.forEach((plat) => branches.forEach((b) => {
      const agg = aggByPB[plat + "|" + b]; if (!agg) return;
      xs.forEach((t) => {
        const a = agg[t]; if (!a || !a[flag]) return;
        let y = null;
        for (const gm of group.geoms) { if (a[pick][gm] != null) { y = a[pick][gm]; break; } }
        if (y != null) out.push({ x: t, y, ...extra });
      });
    }));
    return out;
  };
  const allMarks = (pick) => [...runMarks(pick, "testsFailed", {}), ...runMarks(pick, "corrAlert", { shape: "x", color: CORRC })];
  const opts = (yl) => ({ xTime: true, xRange: xr, xPadAdd: xpad, yLog: true, yLabelText: yl, yfmt: fmtNum, onPick: pickRun, tooltip: histTip, syncX: histGroup, showNow: true });
  linePlot($("hVcd"), xs, specsFor("time"), { width: $("hVcd").clientWidth || 320, ...opts("min"), marks: allMarks("time") });
  linePlot($("hMem"), xs, specsFor("mem"), { width: $("hMem").clientWidth || 320, ...opts("GB"), marks: allMarks("mem") });
  const gateSpecs = [];
  M.platforms.forEach((plat) => branches.forEach((b) => {
    const agg = aggByPB[plat + "|" + b]; if (!agg) return;
    const ys = xs.map((t) => { const a = agg[t]; return a ? a.gatePerf : null; });
    if (ys.some((y) => y != null)) gateSpecs.push({ label: `${plat} ${b}`, color: PLATC[plat] || IDEAL, ys, _xs: xs, meta: { platform: plat, branch: b } });
  }));
  linePlot($("hGate"), xs, gateSpecs, { width: $("hGate").clientWidth || 320, xTime: true, xRange: xr, xPadAdd: xpad, yLabelText: "count", yfmt: (v) => v.toFixed(0), onPick: pickRun, tooltip: histTip, syncX: histGroup, showNow: true });

  const k = (c, t, dash) => `<span class="k"><span class="sw" style="background:${c};${dash ? "height:0;border-top:2px dashed " + c : ""}"></span>${t}</span>`;
  $("hist-legend").innerHTML =
    `<span class="grp">${M.platforms.map((p) => k(PLATC[p] || IDEAL, p)).join("")}</span>` +
    `<span class="grp">${group.geoms.map((gm) => k("#888", `${GEOM_LABEL[gm]} (${GEOM_DASH[gm] ? "dashed" : "solid"})`, !!GEOM_DASH[gm])).join("")}` +
    `<span class="k"><span class="ring" style="border-color:${THROTC}"></span>ran hot</span>` +
    `<span class="k"><span class="dot" style="background:${THROTC}"></span>throttled</span>` +
    `<span class="k"><span class="tri" style="border-bottom-color:${FAILC}"></span>tests failed</span>` +
    `<span class="k"><span class="ring" style="border-color:${CORRC}"></span>run shown</span>` +
    `<span class="k"><span class="cx" style="color:${CORRC}">✕</span>incorrect</span></span>`;
}

// Device-count choices for the History `n` selector = the counts present in the ACTIVE group's
// geometries (cone/parallel have 1/2/4; translation/multiaxis are n=1 until sharding lands).  Clamps
// ui_state.histN into range so switching groups can't leave it on a count the new group lacks.
function syncHistN() {
  const group = HIST_GROUPS.find((g) => g.id === ui_state.histGroup) || HIST_GROUPS[0];
  const devs = uniq(M.runs.flatMap((r) => r.cells.filter((c) => group.geoms.includes(c.geom)).map((c) => c.ndev)))
    .filter((x) => x != null).sort((a, b) => a - b);
  if (!devs.length) devs.push(1);
  if (!devs.includes(ui_state.histN)) ui_state.histN = devs[0];
  fillSelect("histN", devs, ui_state.histN);
}

// ---- correctness banner + tab badge (the dashboard IS the alert — design note D5) ----------------
// The favicon doubles as a passive signal: a red "!" tile when any divergence is unacknowledged, so a
// pinned/bookmarked tab flags it without being opened.  SVG data-URI -> no asset, no infra.
function setFavicon(n) {
  let link = $("favicon");
  if (!link) { link = document.createElement("link"); link.id = "favicon"; link.rel = "icon"; document.head.appendChild(link); }
  const svg = n > 0
    ? `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><rect width="16" height="16" rx="3" fill="${CORRC}"/><text x="8" y="12.5" font-size="13" font-weight="bold" text-anchor="middle" fill="#fff" font-family="sans-serif">!</text></svg>`
    : `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><circle cx="8" cy="8" r="6" fill="${PLATC.gpu}"/></svg>`;
  link.href = "data:image/svg+xml," + encodeURIComponent(svg);
}
function renderBanner() {
  const box = $("corr-banner"); if (!box) return;
  // the alert inbox: the LATEST run per (platform, branch) that is unacknowledged-incorrect — not every
  // historical run (those show as ✕ marks in History).  A branch auto-clears when its latest run is clean.
  const bad = [];
  M.platforms.forEach((p) => branchesFor(p).forEach((b) => { const r = latestRun(p, b); if (r && runAlert(r)) bad.push(r); }));
  bad.sort((a, b) => runTime(b) - runTime(a));
  setFavicon(bad.length);
  document.title = bad.length ? `⚠(${bad.length}) mbirjax metrics` : "mbirjax metrics";
  if (!bad.length) { box.style.display = "none"; box.innerHTML = ""; return; }
  const since = bad.reduce((a, b) => (runTime(b) < runTime(a) ? b : a));
  // Each run is a clickable .cb-run row: a header line + a bulleted list of its divergent configs.
  // The click handler sits on .cb-run only, so clicks on the nested config bullets bubble up to it.
  const item = (r) => {
    const cells = [...new Set(runCorr(r).map((f) => f.cell).filter(Boolean))];
    const bullets = cells.length ? cells.map((c) => `<li>${cellCoords(c)}</li>`).join("") : `<li>${runCorrCells(r)} config(s)</li>`;
    return `<li class="cb-run" data-rk="${runKey(r)}" data-plat="${r.platform}" data-branch="${r.branch}">`
      + `<div class="cb-runhead"><b>${r.branch}</b> · ${r.platform.toUpperCase()} · ${commitMinute(r)}</div>`
      + `<ul class="cb-cfgs">${bullets}</ul></li>`;
  };
  box.style.display = "block";
  box.innerHTML = `<div class="cb-head">✕ ${bad.length} unacknowledged correctness divergence${bad.length > 1 ? "s" : ""} since ${runDateLabel(since)}</div>`
    + `<ul class="cb-list">${bad.map(item).join("")}</ul>`
    + `<div class="cb-foot">vs the prior run on each branch · click a row to view it (again to hide) · clear reviewed runs with <code>action_scripts/clear_correctness.sh</code></div>`;
  box.querySelectorAll(".cb-run").forEach((li) => li.onclick = () => {
    // Toggle: a second click on the row whose correctness detail is already open collapses it; clicking
    // a DIFFERENT row switches to (and opens) that run instead.
    const isOpen = ui_state.openTile === "correctness" && ui_state.runKey === li.dataset.rk
                   && ui_state.platform === li.dataset.plat && ui_state.branch === li.dataset.branch;
    if (isOpen) { ui_state.openTile = null; renderAll(); return; }
    ui_state.platform = li.dataset.plat; ui_state.branch = li.dataset.branch; ui_state.runKey = li.dataset.rk; ui_state.openTile = "correctness";
    fillSelect("platform", M.platforms, ui_state.platform);
    fillSelect("branch", branchesFor(ui_state.platform), ui_state.branch);
    renderAll();
    $("tiles").scrollIntoView({ behavior: "smooth", block: "start" });
  });
}

// ---- orchestration -----------------------------------------------------------
function renderAll() { renderBanner(); renderTiles(); renderDetail(); syncGoSelect(); renderScaling(); renderHistory(); }
// Default branch for a platform = the one with the MOST RECENT run (by commit time), NOT the
// alphabetically-first.  Keeps the run-shown tile on the newest run after a branch rename / new branch
// (e.g. greg/conebeam_sharding -> greg/sharding_extensions: the newest run is on the new branch).
function defaultBranch(plat) {
  const rs = M.runs.filter((r) => r.platform === plat);
  return rs.length ? rs.reduce((a, b) => (runTime(b) > runTime(a) ? b : a)).branch : (branchesFor(plat)[0] || null);
}
function onPlatform() {
  ui_state.platform = $("platform").value;
  const bs = branchesFor(ui_state.platform);
  if (!bs.includes(ui_state.branch)) ui_state.branch = defaultBranch(ui_state.platform);
  fillSelect("branch", bs, ui_state.branch);
  ui_state.openTile = null; ui_state.runKey = null; renderAll();
}
function init() {
  // The repo name at the end of the header line links to the repo (plain text if the URL is unknown).
  const repo = M.repo_url
    ? `<a class="repolink" href="${M.repo_url}" target="_blank" rel="noopener">${M.repo_name}</a>`
    : M.repo_name;
  $("gen").innerHTML = `generated ${M.generated} · ${repo}`;
  $("footer").innerHTML = `${M.runs.length} run(s) · platforms ${M.platforms.join(", ")} · branches ${M.branches.join(", ")} · regenerate with <code>action_scripts/build_dashboard.sh</code>`;
  if (!M.runs.length) { $("tiles").innerHTML = "<p class='muted'>No runs found under results/.</p>"; return; }

  ui_state.platform = M.platforms.includes("gpu") ? "gpu" : M.platforms[0];
  ui_state.branch = defaultBranch(ui_state.platform);
  fillSelect("platform", M.platforms, ui_state.platform);
  fillSelect("branch", branchesFor(ui_state.platform), ui_state.branch);

  $("platform").onchange = onPlatform;
  $("branch").onchange = () => { ui_state.branch = $("branch").value; ui_state.openTile = null; ui_state.runKey = null; renderAll(); };
  $("op").onchange = () => { ui_state.go = $("op").value; renderScaling(); };
  $("ref").value = ui_state.ref;
  $("ref").onchange = () => { ui_state.ref = $("ref").value; renderScaling(); };
  // History branch filter: "all" (default) overlays every branch; pick one to isolate it.
  fillSelect("histBranch", ["all", ...M.branches], ui_state.histBranch, ["all branches", ...M.branches]);
  $("histBranch").onchange = () => { ui_state.histBranch = $("histBranch").value; renderHistory(); };
  // History geometry-group toggle: swaps cone+parallel <-> translation+multiaxis (different headline op).
  $("hist-group-seg").innerHTML = HIST_GROUPS.map((g) =>
    `<button data-g="${g.id}" class="${g.id === ui_state.histGroup ? "on" : ""}">${g.label}</button>`).join("");
  $("hist-group-seg").querySelectorAll("button").forEach((b) => b.onclick = () => {
    ui_state.histGroup = b.dataset.g;
    $("hist-group-seg").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
    syncHistN(); renderHistory();
  });
  // History device-count selector (n): the device counts present in the ACTIVE group (the new
  // geometries only have n=1 until sharding lands, so the choices shrink when that group is selected).
  syncHistN();
  $("histN").onchange = () => { ui_state.histN = +$("histN").value; renderHistory(); };
  $("view-seg").innerHTML = `<button data-v="plot" class="on">plot</button><button data-v="table">table</button>`;
  $("view-seg").querySelectorAll("button").forEach((b) => b.onclick = () => {
    ui_state.view = b.dataset.v; $("view-seg").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b)); renderScaling();
  });
  let rt; window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => { renderScaling(); renderHistory(); }, 160); });
  renderAll();
}
init();
