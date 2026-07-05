# Partition-sequence study — results and figures

Interactive results page for the VCD partition-sequence study (July 2026).
Study plan + findings prose: `mbirjax/experiments/partition_sequence/partition_sequence_plan.md`.
Harness: `mbirjax_applications/partition_sequence/` (cache builder + study runner).

- `data/round1|round2|scale2x/*.json` — raw per-run trajectories (per-iteration masked NRMSE
  vs converged reference, change %, wall time, peak GPU memory).  The durable record; the
  cluster scratch copies are purgeable.
- `build_page.py` — distills the JSONs into `partition_sequence.html` (self-contained:
  vendored uPlot inlined).  Rerun after adding data.
- `partition_sequence.html` — per dataset: NRMSE vs iteration (curves nearly collapse) and
  NRMSE vs wall time (curves fan out by tail-granularity cost), noise floor dashed, legend
  click-to-toggle; summary tables of iterations/seconds to NRMSE targets + peak memory.

Open with `#force-visible` appended to the URL when viewing through headless/hidden-tab
tooling (uPlot sizing needs live layout; the page also self-heals with a retry).
