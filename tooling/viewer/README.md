# Metrics dashboard

A static, self-contained HTML viewer for the performance time series in this repo.
`build_dashboard.py` reads the YAML result files and emits a single
`dashboard/index.html` with the data, styles, scripts, and the charting library all
inlined — no server, no network, opens straight from `file://`.

## Run it

From the `mbirjax` conda environment (the only dependency is PyYAML, already in that env):

```bash
conda activate mbirjax
action_scripts/build_dashboard.sh        # -> dashboard/index.html
open dashboard/index.html                # macOS; or just double-click it
```

`action_scripts/build_dashboard.sh` is a thin wrapper around `tooling/viewer/build_dashboard.py`
(which you can also run directly). Re-run it whenever new nightly results land. The generated
`dashboard/` is gitignored (it is derived from the tracked YAML); the source in this directory is
what's committed. Knobs live in the CONFIG block at the top of `build_dashboard.py` — no CLI args.

## What it reads

| source | used for |
|---|---|
| `results/<plat>/<branch>/regression_<plat>_<date>.yaml` | per-run cells, gate, time series |
| `results/<plat>/<branch>/records_<plat>.yaml` | best-ever record book ("best-ever" reference) |
| `results/<plat>/<branch>/tests_<plat>_<date>.txt` | pytest counts + failing node-ids |

The "compare against" overlays (main / prerelease / prior run / best-ever) are all derived from the
run files above — there are no separate reference snapshots.

## The views

- **Health tiles** — cells measured, hard-gate hits, tests failed, latest run. Fail counts show in
  red; clicking a tile with failures opens a detail panel (gate entries with their comparison basis,
  failing test node-ids, or failed/OOM cells).
- **Scaling** (centerpiece, one op at a time) — four log-scaled panels: time vs size (minutes),
  memory vs size (GB), speedup vs devices, and per-device memory ÷ sino shard, each with an ideal
  reference. A "compare against" selector (none / main / prerelease / prior run / best-ever) overlays
  that reference and drives the table's red/green (colored only when |Δ| ≥ 1%); hard-gate cells are marked
  with a red ring. A plot/table toggle shows the raw numbers (GB; time in s or min).
- **History** — three headline aggregates over time (VCD time and peak memory at the largest size,
  and the hard-gate regression count), with branches overlaid; drag a chart to zoom a date range.

## Files

| file | role |
|---|---|
| `build_dashboard.py` | the reader/generator (run this) |
| `template.html` | page skeleton with `{{...}}` inline placeholders |
| `dashboard.css` | styles (light + dark) |
| `dashboard.js` | client logic for the views |
| `vendor/uPlot.iife.min.js`, `vendor/uPlot.min.css` | vendored charting lib ([uPlot](https://github.com/leeoniya/uPlot) v1.6.31, MIT) |
