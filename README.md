# mbirjax_metrics

The performance & correctness dashboard for [mbirjax](https://mbirjax.readthedocs.io/) — an
automatically-updated record of how fast, how memory-hungry, and how **correct** mbirjax's
reconstruction operators are over time, on both CPU and GPU.

**Live dashboard:** <https://gbuzzard.github.io/mbirjax_metrics/>

The dashboard rebuilds and republishes automatically whenever new measurements are pushed, so that
link is always current — you don't need to run anything to read it. (Why a separate repo from mbirjax
itself: it keeps the performance time series out of the library's history, so the data survives branch
churn and is never pushed to the library's `main`.)

## How to read it

The dashboard explains itself. Open the live page and expand the **"How to read this dashboard"** panel
at the top — it walks through the tiles, the red correctness banner, the History and Scaling views, and
the colours & marks.

That guide lives **inside the dashboard** (authored in
[`tooling/dashboard/template.html`](tooling/dashboard/template.html) and shipped in the page itself), so
it can never drift from the UI it describes — which is why it isn't duplicated here or in the mbirjax
docs; both just point to the live page.

## Running or extending it
You don't need a server — the dashboard is a single self-contained page generated from the YAML time
series in `results/`.
- Build it locally: `conda activate mbirjax && action_scripts/build_dashboard.sh`, then open `dashboard/index.html`.
- Add a one-off measurement, run the nightly by hand, or check the schedule — see **[`action_scripts/README.md`](action_scripts/README.md)**.
- How runs are measured, gated, and scheduled — see the READMEs under **[`tooling/`](tooling/)**.
