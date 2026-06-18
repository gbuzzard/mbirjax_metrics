# Nightly regression harness

A standing, **fire-on-change** check: it watches a few mbirjax branches and, whenever one moves,
measures every geometry × op × size × device-count (min time + peak memory + a tolerant correctness
fingerprint), diffs against that branch's own previous run, and flags regressions.  (Cross-branch
context — vs `main`/`prerelease` — and best-ever drift are shown on the dashboard, not gated here.)
Runs on a Mac via launchd (working) and, eventually, on the cluster via scrontab + slurm (pending the
slurm script).

## Layout (canonical home: this repo)

The harness and data live entirely in `mbirjax_metrics` — **edit them here directly** (no deploy step):
```
action_scripts/          top-level entry points + run_configs.env (the run knobs); see its README
tooling/scaling_tests/   engine: scaling_common.py, performance_tracking.py, run_nightly.py,
                         run_performance_local.py
tooling/regression/      this wrapper: run_regression.sh, regression.env, enable/disable_nightly.sh,
                         com.mbirjax.regression.plist, cluster_preamble.sh.example, README.md
results/<plat>/<branch>/ regression_<plat>_<commit-time>_<sha8>.yaml (time series) + records_<plat>.yaml
                         (best-ever) + tests_*.txt
state/<plat>/<branch>    last MEASURED commit per branch (fire-on-change)
```
`mbirjax` itself is only the **library under test**: the nightly fresh-clones it and makes a throwaway
worktree per changed branch (no fixed checkout).

## What a run does (`run_regression.sh`)

1. **Bootstrap**: source the node preamble (cluster proxy/modules), update-or-clone the persistent
   metrics clone at `$WORK_DIR/metrics`, and re-exec that copy — so remote harness / `run_configs.env`
   / engine changes are always picked up.
2. Activate the **dedicated** conda env; install the harness deps (matplotlib, ruamel).
3. For each tracked branch: `git ls-remote` its head; **skip if unchanged** (vs `state/`).
4. Per changed branch: worktree at that commit → `pip install -e "$WT[<extras>]"` → run its tests →
   run the perf engine (`run_nightly.py`, `lib_root=$WT`). The engine auto-detects sharding capability
   **per geometry** and measures an unported geometry single-device (n=1) — so `main` (no sharding) and
   `prerelease` (parallel only) are tracked cleanly alongside the dev branch.
5. Commit + push `results/` + `state/` (**push failure is non-fatal** — retried next run).
6. Exit non-zero **only** on a hard-gate perf regression → the cron/slurm mail is a real alert.

The published dashboard rebuilds itself separately: a GitHub Action regenerates it from the pushed
YAML and deploys to Pages (see the repo README), so the nightly only needs to push data.

## One-time setup

1. Clone the metrics repo somewhere stable (the bootstrap/entrypoint clone).
2. Create the dedicated env: `conda create -n mbirjax_regression python=3.11` (the wrapper installs the
   rest each run). **Don't reuse your dev `mbirjax` env** — the per-branch editable installs churn it.
3. Set the run knobs in `action_scripts/run_configs.env` (`TRACKED_BRANCHES`, `INSTALL_EXTRAS_*`,
   `CONDA_PYTHON`) and the infra in `regression.env` (`POLL_SCHEDULE`, `NOTIFY`, `WORK_DIR`,
   `PREAMBLE_FILE`, `TOKEN_FILE`). The nightly pulls the repo before each run, so committed+pushed edits
   propagate automatically.
4. For the unattended push, create a fine-grained PAT with write access to `mbirjax_metrics` only via
   `action_scripts/create_token.sh` (see `create_token_instructions.md`); point `TOKEN_FILE` at it.

**Schedule it**
- macOS: `action_scripts/enable_nightly.sh` (loads a launchd agent) / `disable_nightly.sh`.
- Cluster: scrontab + `nightly_regression.slurm` — pending (enable/disable stub this for now).

## Verify before scheduling
```
action_scripts/run_one_night.sh     # one manual pass: clone -> test -> measure -> push
```
Confirm a `results/<plat>/<branch>/regression_<plat>_<...>.yaml` appears and `state/<plat>/<branch>`
updates; a second immediate run should report no changed branch (fire-on-change working).

## Knobs
- **`action_scripts/run_configs.env`** (the ones you edit): `TRACKED_BRANCHES`,
  `INSTALL_EXTRAS_{cpu,gpu}` (`cuda12,test` GPU / `test` CPU), `CONDA_PYTHON`.
- **`regression.env`** (infra): `ENABLED` (kill-switch — prefer enable/disable_nightly) · `POLL_SCHEDULE`
  · `METRICS_URL`/`MBIRJAX_URL` · `WORK_DIR` · `CONDA_ENV` · `HARNESS_DEPS` · `RUN_TESTS` /
  `TEST_CPU_DEVICES` · `PREAMBLE_FILE` · `TOKEN_FILE` · `NOTIFY`.

## Notes / current limits
- `enable_nightly.sh` supports a **daily** `POLL_SCHEDULE` (`M H * * *`) on macOS; richer specs need
  more launchd entries. Cluster scheduling (scrontab + slurm) is not yet written.
- Per-branch **test** results are logged but **not gated/diffed** (the perf engine is the alert path).
- The gate compares each run only against **this branch's own previous run** (commit-over-commit).
  Cross-branch comparison (vs `main`/`prerelease`) and best-ever drift are surfaced on the dashboard,
  derived from the tracked runs themselves — there are no hand-captured reference snapshots.
