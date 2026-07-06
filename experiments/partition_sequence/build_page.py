"""Build partition_sequence.html from the study trajectory JSONs in data/.

Study: mbirjax/experiments/partition_sequence/partition_sequence_plan.md (results section).
Each JSON is one run of the study harness (run_study.py, alongside): per-iteration masked
NRMSE vs the dataset's converged reference, native change %, cumulative wall time, and peak
GPU memory.

Per dataset the page shows a summary table, then two linked plots:
  LEFT  (NRMSE vs iteration):  curves nearly collapse -- convergence per iteration is
                               almost schedule-independent.
  RIGHT (NRMSE vs wall time):  the same runs fan out by the per-iteration COST of the tail
                               granularity; a slider truncates it at a chosen iteration.
Hover a sequence name to highlight its curves in both plots.

Round 2 supersedes round 1 for candidates present in both (same seed => identical
trajectory prefix; round 2 ran a higher iteration cap).

Run:  python build_page.py   (writes partition_sequence.html next to this file)
"""
import glob
import json
import os

import ps_config

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(HERE, '..', '..', 'tooling', 'dashboard', 'vendor')

# Which results to render is driven by config.yaml's `page` block: per dataset a tag, the
# data/ rounds (precedence order, later loses), and readable NRMSE targets.  sino/recon
# shapes and the noise floor AUTO-DERIVE from the run JSONs / <tag>_floor.json when present;
# the optional floor/sino/recon in config are fallbacks for legacy rounds.
CFG = ps_config.load()
SKIP = ('floor', 'chunk', 'mono', 'reference')


def load(dataset, rounds):
    """Return (runs, sino_shape, recon_shape); shapes come from the first run JSON that
    carries them (falls back to None so the caller can use the config value)."""
    runs, sino_shape, recon_shape = {}, None, None
    for rnd in reversed(rounds):                 # later entries in `rounds` lose
        for path in sorted(glob.glob(os.path.join(HERE, 'data', rnd, f'{dataset}_*.json'))):
            r = json.load(open(path))
            name = r['label'][len(dataset) + 1:]
            if any(s in name for s in SKIP):
                if r.get('sino_shape') and sino_shape is None:   # e.g. the reference run
                    sino_shape, recon_shape = r['sino_shape'], r.get('recon_shape')
                continue
            runs[name] = r
            if r.get('sino_shape'):
                sino_shape, recon_shape = r['sino_shape'], r.get('recon_shape')
    out = []
    for name, r in sorted(runs.items()):
        rows = [x for x in r['rows'] if x['nrmse_vs_ref'] is not None]
        if not rows:
            continue
        out.append({'name': name, 'seq': r['sequence'],
                    'it': [x['iteration'] for x in rows],
                    't': [round(x['time_s'], 2) for x in rows],
                    'nrmse': [round(x['nrmse_vs_ref'], 6) for x in rows],
                    'peak': round(max(r['peak_gib_per_device']), 2)
                            if r['peak_gib_per_device'] else None})
    return out, sino_shape, recon_shape


def load_floor(dataset, rounds, fallback):
    """Auto-derived noise floor from <tag>_floor.json (written by run_study), else the
    config fallback, else None."""
    for rnd in reversed(rounds):
        path = os.path.join(HERE, 'data', rnd, f'{dataset}_floor.json')
        if os.path.exists(path):
            return json.load(open(path)).get('floor_median')
    return fallback


def target_cells(run, targets):
    cells = []
    for tgt in targets:
        hit = next(((it, t) for it, t, n in zip(run['it'], run['t'], run['nrmse'])
                    if n <= tgt), None)
        cells.append(f'{hit[0]} / {hit[1]:.0f}s' if hit else '&mdash;')
    return cells


def reference_note(dataset):
    # Read the actual reference run so the note reflects whether it truly hit 0.01% change
    # (the z62 1024^3 reference was capped at its iteration limit, not converged).
    for f in glob.glob(os.path.join(HERE, 'data', '*', f'{dataset}_reference.json')):
        rows = json.load(open(f))['rows']
        iters, final = len(rows), rows[-1]['change_pct']
        if final < 0.01:
            tail = f'{iters} iterations to the 0.01% per-iteration-change threshold'
        else:
            tail = (f'{iters} iterations (iteration cap reached first; final change '
                    f'{final:.4f}%, so slightly short of the 0.01% target)')
        return (f'Reference (NRMSE is measured against it): default sequence '
                f'<code>[0, 2, 4, 6, 7]</code>, {tail}.')
    return ''


def shape(s):
    return '&times;'.join(str(x) for x in s) if s else '?'


def main():
    data = {}
    sections = []
    for entry in CFG['page']['datasets']:
        ds, rounds, targets = entry['tag'], entry['rounds'], entry['targets']
        runs, sino, recon = load(ds, rounds)
        if not runs:
            continue
        sino = sino or entry.get('sino')             # auto-derived, else config fallback
        recon = recon or entry.get('recon')
        floor = load_floor(ds, rounds, entry.get('floor'))
        data[ds] = {'runs': runs, 'floor': floor}

        hdr = ''.join(f'<th>iter / sec&nbsp;@&nbsp;{t:g}</th>' for t in targets)
        body = ''
        for r in runs:
            cells = ''.join(f'<td>{c}</td>' for c in target_cells(r, targets))
            peak = f'{r["peak"]:.2f}' if r["peak"] is not None else '&mdash;'
            body += (f'<tr><td>{r["name"]}</td><td class="idx">{r["seq"]}</td>'
                     f'{cells}<td>{peak}</td></tr>')
        floor_txt = f'{floor:.4f}' if floor else 'n/a'
        ref_note = reference_note(ds)

        sections.append(f'''
<section class="ds" id="sec-{ds}">
  <h2>{ds} &mdash; sinogram {shape(sino)}, reconstruction {shape(recon)}</h2>
  <table class="sum">
    <tr><th>sequence</th><th>indices</th>{hdr}<th>peak GiB</th></tr>
    {body}
    <tr class="floornote"><td colspan="{len(targets) + 3}">noise floor (5 seeds):
      NRMSE {floor_txt} &mdash; differences smaller than this are run-to-run noise, not
      schedule differences</td></tr>
  </table>
  <p class="refnote">{ref_note}</p>
  <div class="legend" id="leg-{ds}"></div>
  <div class="row">
    <div class="plot" id="it-{ds}"></div>
    <div class="plotwrap">
      <div class="plot" id="tm-{ds}"></div>
      <div class="sliderbox">show iterations &le;
        <input type="range" id="sl-{ds}" min="1" value="1">
        <b id="slv-{ds}"></b></div>
    </div>
  </div>
</section>''')

    intro = '''
<p>This page compares <b>granularity schedules</b> (&ldquo;partition sequences&rdquo;) for
mbirjax's VCD reconstruction.  A schedule lists, for each iteration, how many <b>subsets</b>
the image voxels are split into; the reconstruction updates one subset at a time within an
iteration.  <b>Coarser</b> (fewer subsets) updates more voxels together and costs more per
iteration; <b>finer</b> (more subsets) is cheaper per iteration but each update sees less of
the image.  We reconstruct several real CT scans &mdash; subsampled 4&ndash;8&times; so many
experiments run cheaply &mdash; and track how each schedule's error falls.</p>
<ul class="defs">
<li><b>NRMSE</b>: normalized RMS error of a reconstruction against a fully converged
<i>reference</i> (the default schedule run until it changes &lt;0.01% per iteration),
measured inside the region of reconstruction with the end slices dropped.  Lower is better
&mdash; it is the &ldquo;distance to the answer.&rdquo;</li>
<li><b>Noise floor</b> (dashed line / table note): reconstructions vary slightly run-to-run
because the random voxel partitions differ.  Running the default schedule with 5 seeds and
measuring the typical NRMSE between those runs gives this floor; schedule differences
<i>below</i> it are noise, not signal.</li>
<li><b>Checkpointed run</b>: to record NRMSE at every iteration for free, each recon is
stepped one iteration at a time and resumed exactly (checkpointed <code>vcd_recon</code>),
reproducing a single continuous run bit-for-bit.</li>
</ul>
<p><b>How to read.</b> <b>Left</b> = NRMSE vs iteration: the schedules nearly overlap
&mdash; convergence per iteration barely depends on the schedule.  <b>Right</b> = the same
runs vs wall-clock time: they spread out, because the schedule mostly sets the <i>cost per
iteration</i> (via the finest granularity in its tail) and the <i>peak memory</i> (via
whether it starts at granularity&nbsp;1, an all-voxel update).  Hover a name to highlight its
curves; drag the slider to truncate the time plot at a chosen iteration.  Study record:
<code>mbirjax/experiments/partition_sequence/</code>.</p>'''

    tmpl = TEMPLATE
    tmpl = tmpl.replace('__UPLOT_CSS__', open(os.path.join(VENDOR, 'uPlot.min.css')).read())
    tmpl = tmpl.replace('__UPLOT_JS__', open(os.path.join(VENDOR, 'uPlot.iife.min.js')).read())
    nav = ('<p class="nav"><b>Jump to dataset:</b> '
           + ' &nbsp;·&nbsp; '.join(f'<a href="#sec-{d}">{d}</a>' for d in data)
           + '</p>')
    tmpl = tmpl.replace('__INTRO__', intro + nav)
    tmpl = tmpl.replace('__SECTIONS__', ''.join(sections))
    tmpl = tmpl.replace('__DATA_JSON__', json.dumps(data))

    out = os.path.join(HERE, 'partition_sequence.html')
    open(out, 'w').write(tmpl)
    print(f'wrote {out} ({os.path.getsize(out) / 1e6:.2f} MB, '
          f'{sum(len(d["runs"]) for d in data.values())} runs, {len(data)} datasets)')


# Plain template (NOT an f-string) so the JS braces below stay literal.
TEMPLATE = r'''<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>mbirjax partition-sequence study</title>
<style>__UPLOT_CSS__
body { font: 14px/1.5 -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px;
       max-width: 1360px; background: #fff; color: #1a1a1a; }
h1 { font-size: 22px; } h2 { margin: 30px 0 6px; font-size: 18px; }
.intro, .defs { max-width: 900px; color: #333; }
.nav { max-width: 900px; }
.nav a { margin: 0 2px; }
.defs li { margin: 3px 0; }
.row { display: flex; gap: 24px; flex-wrap: wrap; margin: 6px 0 10px; }
.plot { width: 600px; min-height: 360px; flex: 0 0 600px; }
.plotwrap { display: inline-block; }
.sliderbox { font-size: 12px; color: #444; margin: 4px 0 0 60px;
              position: relative; z-index: 2; }
.sliderbox input { vertical-align: middle; width: 320px; }
/* Summary tables only (uPlot builds its own <table> internals). */
table.sum { border-collapse: collapse; margin: 8px 0 10px; font-size: 13px; }
table.sum td, table.sum th { border: 1px solid #ccc; padding: 3px 10px; text-align: right; }
table.sum td:first-child, table.sum th:first-child,
table.sum td.idx, table.sum th:nth-child(2) { text-align: left; }
table.sum td.idx { color: #666; font-family: ui-monospace, monospace; font-size: 12px; }
table.sum tr.floornote td { text-align: left; color: #666; font-style: italic;
                            background: #fafafa; }
.refnote { color: #555; font-size: 12px; margin: 2px 0 6px; }
.legend { max-width: 1300px; margin: 4px 0 8px; }
.chip { display: inline-block; margin: 0 12px 5px 0; cursor: pointer; font-size: 12px;
        white-space: nowrap; }
.chip:hover { text-decoration: underline; }
.chip .sw { display: inline-block; width: 11px; height: 11px; margin-right: 4px;
            border-radius: 2px; vertical-align: middle; }
</style>
<script>__UPLOT_JS__</script></head><body>
<h1>Partition-sequence study &mdash; VCD convergence vs granularity schedule</h1>
__INTRO__
<div id="plots">__SECTIONS__</div>
<script>
// Testing hook for headless/hidden preview tabs, where layout for fresh elements can lag and
// uPlot canvases stay at the 300x150 default.  Open with #force-visible to spoof visibility.
// Harmless in a normal browser.
if (location.hash === '#force-visible') {
  Object.defineProperty(document, 'visibilityState', { get: () => 'visible', configurable: true });
  Object.defineProperty(document, 'hidden', { get: () => false, configurable: true });
}
const DATA = __DATA_JSON__;
const COLORS = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b",
                "#e377c2","#7f7f7f","#bcbd22","#17becf","#aec7e8","#ffbb78",
                "#98df8a","#ff9896","#c5b0d5","#c49c94"];
const $ = id => document.getElementById(id);

// uPlot's built-in log splitter can hang on tight non-power-of-10 bounds; make ticks ourselves.
function logTicks(mn, mx) {
  const out = [];
  for (let e = Math.floor(Math.log10(mn)); Math.pow(10, e) <= mx * 1.0001; e++)
    for (const m of [1, 2, 5]) {
      const v = m * Math.pow(10, e);
      if (v >= mn * 0.9999 && v <= mx * 1.0001) out.push(v);
    }
  return out.length >= 2 ? out : [mn, mx];
}
const fmtNrmse = v => v == null ? null : (v >= 0.01 ? v.toFixed(2) : v.toExponential(0));
// Lighten a hex color toward white (dims non-highlighted series).
function dim(hex) {
  const n = parseInt(hex.slice(1), 16), r = n >> 16, g = (n >> 8) & 255, b = n & 255;
  const f = v => Math.round(v + (255 - v) * 0.82);
  return `rgb(${f(r)},${f(g)},${f(b)})`;
}
function floorLine(floor) {
  return u => {
    const y = Math.round(u.valToPos(floor, 'y', true)), c = u.ctx;
    c.save(); c.strokeStyle = '#999'; c.setLineDash([6, 5]); c.lineWidth = 1;
    c.beginPath(); c.moveTo(u.bbox.left, y); c.lineTo(u.bbox.left + u.bbox.width, y);
    c.stroke(); c.restore();
  };
}
// Union-x builder: one shared x vector; each series null where it has no point there.
// `keep(r,j)` optionally filters points (used by the time slider).
function buildData(runs, xKey, keep) {
  const xset = new Set();
  runs.forEach(r => r[xKey].forEach((v, j) => { if (!keep || keep(r, j)) xset.add(v); }));
  const xs = [...xset].sort((a, b) => a - b);
  const idx = new Map(xs.map((v, i) => [v, i]));
  const arr = [xs];
  runs.forEach(r => {
    const ys = new Array(xs.length).fill(null);
    r[xKey].forEach((v, j) => { if (!keep || keep(r, j)) ys[idx.get(v)] = r.nrmse[j]; });
    arr.push(ys);
  });
  return arr;
}
// A panel owns a mutable uPlot it can DESTROY and rebuild (some first-on-page uPlot
// constructions fail to size; rebuilding once layout is settled fixes them).
function makePanel(el, title, xLabel, runs, initialData, floor) {
  const st = { hi: -1, u: null, data: initialData };
  // Redraw the highlighted series on TOP (all series are dimmed when a highlight is active,
  // so the hovered one must be re-stroked above them in full color).
  function drawHi(u) {
    if (st.hi < 0) return;
    const s = st.hi + 1, xd = u.data[0], yd = u.data[s], c = u.ctx;
    const dpr = u.pxRatio || window.devicePixelRatio || 1;
    c.save(); c.lineWidth = 3.5 * dpr; c.strokeStyle = COLORS[st.hi % COLORS.length];
    c.beginPath(); let on = false;
    for (let i = 0; i < xd.length; i++) {
      if (yd[i] == null) continue;  // span gaps (do NOT reset the pen)
      const px = u.valToPos(xd[i], 'x', true), py = u.valToPos(yd[i], 'y', true);
      if (!on) { c.moveTo(px, py); on = true; } else c.lineTo(px, py);
    }
    c.stroke(); c.restore();
  }
  const drawHooks = [];
  if (floor) drawHooks.push(floorLine(floor));
  drawHooks.push(drawHi);
  function construct() {
    const series = [{}];
    runs.forEach((r, i) => {
      const c = COLORS[i % COLORS.length];
      series.push({ label: r.name, width: 2, spanGaps: true, points: { show: false },
                    stroke: () => st.hi < 0 ? c : dim(c) });  // dim ALL while a highlight is active
    });
    st.u = new uPlot({
      title, width: el.clientWidth || 600, height: 360,
      scales: { x: { time: false }, y: { distr: 3 } },
      axes: [ { label: xLabel },
              { label: 'NRMSE vs reference',
                splits: (u, ai, mn, mx) => logTicks(mn, mx),
                values: (u, sp) => sp.map(fmtNrmse) } ],
      series, legend: { show: false }, hooks: { draw: drawHooks },
    }, st.data, el);
    el._u = st.u;
  }
  construct();
  return {
    highlight(k) { st.hi = k; st.u.redraw(); },
    setData(d) { st.data = d; st.u.setData(d); },
    healIfNeeded() {  // rebuild an undersized canvas; return true once correctly sized
      const cv = el.querySelector('canvas');
      if (cv && cv.width >= 400) return true;
      st.u.destroy(); el.innerHTML = ''; construct();
      return false;
    },
  };
}
function build() {
  const panels = [];
  for (const [ds, d] of Object.entries(DATA)) {
    try {
      const runs = d.runs;
      const itP = makePanel($('it-' + ds), 'NRMSE vs iteration', 'iteration', runs,
                            buildData(runs, 'it'), d.floor);
      const maxIt = Math.max(...runs.flatMap(r => r.it));
      const timeData = m => buildData(runs, 't', (r, j) => r.it[j] <= m);
      const tmP = makePanel($('tm-' + ds), 'NRMSE vs wall time (seconds)', 'seconds', runs,
                            timeData(maxIt), d.floor);
      // Shared legend: hover highlights the sequence in BOTH plots.
      const leg = $('leg-' + ds);
      runs.forEach((r, i) => {
        const chip = document.createElement('span');
        chip.className = 'chip';
        chip.innerHTML = `<span class="sw" style="background:${COLORS[i % COLORS.length]}"></span>`
                         + r.name + ` <span style="color:#999">${JSON.stringify(r.seq)}</span>`;
        chip.onmouseenter = () => { itP.highlight(i); tmP.highlight(i); };
        chip.onmouseleave = () => { itP.highlight(-1); tmP.highlight(-1); };
        leg.appendChild(chip);
      });
      // Slider truncates the time plot at a chosen iteration.
      const sl = $('sl-' + ds), out = $('slv-' + ds);
      sl.min = 1; sl.max = maxIt; sl.value = maxIt; out.textContent = maxIt;
      sl.oninput = () => { out.textContent = sl.value; tmP.setData(timeData(+sl.value)); };
      panels.push(itP, tmP);
    } catch (e) { console.error('build failed for ' + ds, e); }
  }
  // Heal any plot that failed to size on first construct.  A long multi-dataset page can
  // finish layout well after load, so poll (rebuild undersized plots) until all are sized
  // or ~2.5 s elapses.
  let tries = 0;
  const timer = setInterval(() => {
    const allOk = panels.every(p => p.healIfNeeded());
    if (allOk || ++tries > 25) clearInterval(timer);
  }, 100);
}
window.addEventListener('load', build);

</script></body></html>'''


if __name__ == '__main__':
    main()
