# mbirjax_metrics

The performance & correctness dashboard for [mbirjax](https://mbirjax.readthedocs.io/) — an
automatically-updated record of how fast, how memory-hungry, and how **correct** mbirjax's
reconstruction operators are over time, on both CPU and GPU.

**Live dashboard:** <https://gbuzzard.github.io/mbirjax_metrics/>

The dashboard rebuilds and republishes automatically whenever new measurements are pushed, so that
link is always current. Everything below is a guide to **reading** it — you don't need to run
anything. (Why a separate repo from mbirjax itself: it keeps the performance time series out of the
library's history, so the data survives branch churn and is never pushed to the library's `main`.)

## What you're looking at

A scheduled job measures each tracked mbirjax branch — running its reconstruction operators (the FBP
filter, forward projection, back projection, and the iterative VCD recon) across a range of problem
sizes and device counts, on both CPU and GPU — and records, for every configuration: **run time**,
**peak memory**, and a numeric **fingerprint** of the output used to catch correctness changes. The
dashboard is the view onto that growing time series. Read it top to bottom:

### 1 · Tiles — health of the shown run, at a glance
A row of cards for the currently-selected run, each split **CPU | GPU**:
- **configs measured** — how many (geometry × op × size × device-count) cells ran, and how many failed.
- **correctness** — how many configs diverge from a trusted reference (see *Correctness* below). Red = divergent.
- **performance regressions** — configs whose time or memory regressed versus the reference.
- **tests failed** — unit-test failures from that commit.
- **run shown** — which commit you're viewing (branch · platform · date).

Click any card to drill into the specifics.

### 2 · Correctness banner (red, top of page)
If any branch's latest run produces a **different reconstruction** than its reference, a red banner
lists the offending configs — and the browser tab gets a ⚠ badge so you notice without even
switching to it. This is the loudest signal on the page: correctness is treated as more important
than speed. It clears when the divergence goes away or is acknowledged as reviewed.

### 3 · History — trends over time
Time-series panels (commit time on the x-axis) spanning both platforms and all branches: **time** and
**peak memory** at the largest size, plus a **performance-regressions** count. Use the controls to pick a
**branch**, a **geometry group** (cone + parallel, or translation + multiaxis), and a **device
count**. **Click any point to load that run** into the tiles and scaling views.

### 4 · Scaling — how the selected run behaves
For the chosen run and operation:
- **time vs size** and **memory vs size** (log–log), each with an "ideal" slope for reference.
- **speedup vs devices** and **per-device memory** — i.e. does the work actually shard across GPUs?
- **compare against** overlays the same curves from `main`, `prerelease`, the prior run, or the best-ever.

## Reading the colors & marks
- **Colour = platform:** GPU is blue, CPU is amber.
- **Line style = geometry** within each group (one solid, one dashed).
- **Red ✕** = a correctness fail (output mismatch between reference and the indicated run) · **red △** = failing tests · **amber ring** = a GPU that ran
  hot · **amber disc** = a GPU that throttled (so that point's timing is unreliable).

## Correctness — what "diverges" means
Each output's fingerprint is checked against up to four references:
- the **prior run** on the same branch — did this commit change the result?
- the latest **main** — does this branch still match the canonical answer?
- **single- vs multi-device** within the same run — does sharding change the result?
- the **other platform** — do CPU and GPU agree?

A change beyond a small tolerance is flagged. Reviewed or expected changes can be **acknowledged** so
they stop alerting (without erasing the record).

## Running or extending it
You don't need a server — the dashboard is a single self-contained page generated from the YAML time
series in `results/`.
- Build it locally: `conda activate mbirjax && action_scripts/build_dashboard.sh`, then open `dashboard/index.html`.
- Add a one-off measurement, run the nightly by hand, or check the schedule — see **[`action_scripts/README.md`](action_scripts/README.md)**.
- How runs are measured, gated, and scheduled — see the READMEs under **[`tooling/`](tooling/)**.
