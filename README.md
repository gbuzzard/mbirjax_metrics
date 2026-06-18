# mbirjax_metrics

Performance-regression data and tooling for [mbirjax](https://mbirjax.readthedocs.io/).

This is the standalone, version-controlled home for mbirjax's performance time series and the tools
that gather and display it. Keeping it separate from the mbirjax library means the results survive
mbirjax's branch churn and are never pushed to the library's `main`.

**Live dashboard:** <https://gbuzzard.github.io/mbirjax_metrics/> — rebuilt from the YAML time series
and published automatically by a GitHub Action on every push to `main` (`.github/workflows/pages.yml`).

## Quick start

```bash
conda activate mbirjax
action_scripts/build_dashboard.sh     # -> dashboard/index.html
open dashboard/index.html             # macOS; or double-click the file
```

The only dependency for building the dashboard is PyYAML, which the `mbirjax` env already provides.

## Layout

- **`action_scripts/`** — the top-level entry points (build the dashboard, add a run, run the
  nightly). Start here; see `action_scripts/README.md`.
- **`tooling/`**
  - `scaling_tests/` — the measurement **engine** (`performance_tracking.py`): sweeps geometry × op ×
    size × device-count, records min time / peak memory / speedup + a tolerant correctness
    fingerprint per config, and runs the diff/gate. Also the nightly launcher (`run_nightly.py`) and
    the reference-capture scripts.
  - `regression/` — the unattended **nightly harness** (`run_regression.sh`): fire-on-change per
    tracked branch — clone the tip → run tests → measure → push.
  - `viewer/` — the **dashboard generator** (`build_dashboard.py`): reads the YAML time series and
    emits a single self-contained `dashboard/index.html` (no server, no network).
- **`results/<platform>/<branch>/`** — the **time series**: one
  `regression_<plat>_<commit-time>_<sha8>.yaml` per measured commit (named and sorted by *commit*
  time), plus a `records_<plat>.yaml` best-ever book and the pytest log.
- **`golden/`** — reference snapshots the gate compares against: the per-branch `golden_<plat>.yaml`
  drift/accept reference, the `main_baseline_<plat>.yaml` released-`main` reference, and small
  `.npy` cross-version deep-diff arrays.
- **`state/`** — per-platform fire-on-change bookkeeping (the last-measured sha per branch).
- **`dashboard/`** — the generated `index.html` (gitignored; regenerate with `build_dashboard.sh`).

## The dashboard

Health tiles (configs measured, hard-gate hits, tests, the run shown), a per-op **Scaling** view
(time/memory vs problem size and speedup / per-device-memory-÷-shard vs device count, on log axes
with ideal references and a "compare against" overlay), and a **History** timeline spanning both
platforms. Details in `tooling/viewer/README.md`.

## How a run is gated

After each run the engine compares it — per config and metric — against the prior run and the
reference snapshot(s). Memory and correctness are deterministic, so they **hard-fail**; timing is
noisy, so it only **warns**. The full criteria and thresholds are described in
`tooling/regression/README.md` and surfaced in the dashboard's gate tile.
