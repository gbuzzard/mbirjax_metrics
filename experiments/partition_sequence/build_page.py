"""Build partition_sequence.html from the study trajectory JSONs in data/.

Study: mbirjax/experiments/partition_sequence/partition_sequence_plan.md (results section).
Each JSON is one run of the study harness (mbirjax_applications/partition_sequence/
run_study.py): per-iteration masked NRMSE vs the dataset's converged reference, native
change %, cumulative wall time, and peak GPU memory.

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

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(HERE, '..', '..', 'tooling', 'dashboard', 'vendor')

# (dataset, rounds in precedence order, noise-floor median, readable NRMSE targets,
#  sino shape, recon shape).  Shapes queried once from the caches (build_cache.py output).
DATASETS = [
    ('lilly', ['round2', 'round1'], 0.00499, [0.05, 0.02, 0.01], (225, 356, 470), (470, 470, 356)),
    ('z62', ['round2', 'round1'], 0.01235, [0.05, 0.02], (101, 512, 512), (512, 512, 512)),
    ('sic', ['round2', 'round1'], 0.00451, [0.10, 0.08], (201, 512, 512), (512, 512, 512)),
    ('z62_2x', ['scale2x'], None, [0.05, 0.02], (201, 1024, 1024), (1024, 1024, 1024)),
]
SKIP = ('floor', 'chunk', 'mono', 'reference')


def load(dataset, rounds):
    runs = {}
    for rnd in reversed(rounds):                 # later entries in `rounds` lose
        for path in sorted(glob.glob(os.path.join(HERE, 'data', rnd, f'{dataset}_*.json'))):
            r = json.load(open(path))
            name = r['label'][len(dataset) + 1:]
            if any(s in name for s in SKIP):
                continue
            runs[name] = r
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
    return out


def target_cells(run, targets):
    cells = []
    for tgt in targets:
        hit = next(((it, t) for it, t, n in zip(run['it'], run['t'], run['nrmse'])
                    if n <= tgt), None)
        cells.append(f'{hit[0]} / {hit[1]:.0f}s' if hit else '&mdash;')
    return cells


def shape(s):
    return '&times;'.join(str(x) for x in s)


def main():
    data = {}
    sections = []
    for ds, rounds, floor, targets, sino, recon in DATASETS:
        runs = load(ds, rounds)
        if not runs:
            continue
        data[ds] = {'runs': runs, 'floor': floor}

        hdr = ''.join(f'<th>iter / sec&nbsp;@&nbsp;{t:g}</th>' for t in targets)
        body = ''
        for r in runs:
            cells = ''.join(f'<td>{c}</td>' for c in target_cells(r, targets))
            peak = f'{r["peak"]:.2f}' if r["peak"] is not None else '&mdash;'
            body += (f'<tr><td>{r["name"]}</td><td class="idx">{r["seq"]}</td>'
                     f'{cells}<td>{peak}</td></tr>')
        floor_txt = f'{floor:.4f}' if floor else 'n/a'

        sections.append(f'''
<section class="ds">
  <h2>{ds} &mdash; sinogram {shape(sino)}, reconstruction {shape(recon)}</h2>
  <table class="sum">
    <tr><th>sequence</th><th>indices</th>{hdr}<th>peak GiB</th></tr>
    {body}
    <tr class="floornote"><td colspan="{len(targets) + 3}">noise floor (5 seeds):
      NRMSE {floor_txt} &mdash; differences smaller than this are run-to-run noise, not
      schedule differences</td></tr>
  </table>
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
    tmpl = tmpl.replace('__INTRO__', intro)
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
.defs li { margin: 3px 0; }
.row { display: flex; gap: 24px; flex-wrap: wrap; margin: 6px 0 10px; }
.plot { width: 600px; height: 360px; flex: 0 0 600px; }
.plotwrap { display: inline-block; }
.sliderbox { font-size: 12px; color: #444; margin: 4px 0 0 60px; }
.sliderbox input { vertical-align: middle; width: 320px; }
/* Summary tables only (uPlot builds its own <table> internals). */
table.sum { border-collapse: collapse; margin: 8px 0 10px; font-size: 13px; }
table.sum td, table.sum th { border: 1px solid #ccc; padding: 3px 10px; text-align: right; }
table.sum td:first-child, table.sum th:first-child,
table.sum td.idx, table.sum th:nth-child(2) { text-align: left; }
table.sum td.idx { color: #666; font-family: ui-monospace, monospace; font-size: 12px; }
table.sum tr.floornote td { text-align: left; color: #666; font-style: italic;
                            background: #fafafa; }
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
const RO = new ResizeObserver(entries => {
  for (const e of entries) {
    const u = e.target._u, w = Math.round(e.contentRect.width);
    if (u && w > 0 && Math.abs(u.width - w) > 1) u.setSize({ width: w, height: 360 });
  }
});

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
const fmtNrmse = v => v >= 0.01 ? v.toFixed(2) : v.toExponential(0);
// Lighten a hex color toward white (used to dim non-highlighted series).
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
// Union-x builder: one shared x vector, each series null where it has no point there
// (spanGaps draws through).  `keep(j)` optionally filters points (used by the time slider).
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
function makePlot(el, title, xLabel, runs, dataArr, floor) {
  let hi = -1;
  const series = [{}];
  runs.forEach((r, i) => {
    const c = COLORS[i % COLORS.length];
    series.push({ label: r.name, width: 2, spanGaps: true, points: { show: false },
                  stroke: () => (hi < 0 || hi === i) ? c : dim(c) });
  });
  const u = new uPlot({
    title, width: el.clientWidth || 600, height: 360,
    scales: { x: { time: false }, y: { distr: 3 } },
    axes: [ { label: xLabel },
            { label: 'NRMSE vs reference',
              splits: (u, ai, mn, mx) => logTicks(mn, mx),
              values: (u, sp) => sp.map(fmtNrmse) } ],
    series, legend: { show: false },
    hooks: floor ? { draw: [floorLine(floor)] } : {},
  }, dataArr, el);
  el._u = u;
  RO.observe(el);
  function highlight(k) {
    hi = k;
    for (let i = 1; i < u.series.length; i++)
      u.series[i].width = (k >= 0 && i - 1 === k) ? 3.5 : (k < 0 ? 2 : 1);
    u.redraw();
  }
  return { u, highlight };
}
function build() {
  for (const [ds, d] of Object.entries(DATA)) {
    const runs = d.runs;
    const itP = makePlot($('it-' + ds), 'NRMSE vs iteration', 'iteration', runs,
                         buildData(runs, 'it'), d.floor);
    const maxIt = Math.max(...runs.flatMap(r => r.it));
    const timeData = m => buildData(runs, 't', (r, j) => r.it[j] <= m);
    const tmP = makePlot($('tm-' + ds), 'NRMSE vs wall time (seconds)', 'seconds', runs,
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
    sl.max = maxIt; sl.value = maxIt; out.textContent = maxIt;
    sl.oninput = () => { out.textContent = sl.value; tmP.u.setData(timeData(+sl.value)); };
  }
}
// Build after layout is settled (two rAFs): constructing before the containers have real
// geometry left the first plots' canvases at the browser default size.
window.addEventListener('load', () =>
  requestAnimationFrame(() => requestAnimationFrame(build)));
</script></body></html>'''


if __name__ == '__main__':
    main()
