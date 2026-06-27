# Nightly regression harness

A standing, **fire-on-change** check: it watches a few mbirjax branches and, whenever one moves,
measures every geometry × op × size × device-count (min time + peak memory + a tolerant correctness
fingerprint), diffs against that branch's own previous run, and flags regressions.  (Cross-branch
context — vs `main`/`prerelease` — and best-ever drift are shown on the dashboard, not gated here.)
Runs on a Mac via launchd and on Purdue's Gautschi cluster via a SLURM `scrontab` entry — both
installed/removed by `enable_nightly.sh` / `disable_nightly.sh`.

## Layout (canonical home: this repo)

The harness and data live entirely in `mbirjax_metrics` — **edit them here directly** (no deploy step):
```
action_scripts/          top-level entry points + run_configs.env (the run knobs); see its README
tooling/scaling_tests/   engine: scaling_common.py, performance_tracking.py, run_nightly.py,
                         run_performance_local.py
tooling/regression/      this wrapper: run_regression.sh, lib_env.sh (shared dedicated-env +
                         worktree-install mechanism, also used by action_scripts/add_run.sh),
                         regression.env, enable/disable/status_nightly.sh, recent_runs.py (status),
                         com.mbirjax.regression.plist (macOS), nightly_regression.slurm +
                         cluster_preamble.sh.example (cluster), sbatch_submit.sh (the add_run /
                         run_one_night --sbatch helper), README.md
results/<plat>/<branch>/ regression_<plat>_<commit-time>_<sha8>.yaml (time series) + a sibling
                         _table.yaml (browsable geometry/op/size/n view, auto-written per run by the
                         engine via tooling/scaling_tests/regression_to_table.py) + records_<plat>.yaml
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
   **per geometry** and measures an unported geometry single-device (n=1) — so a branch that hasn't ported
   a given geometry yet (e.g. `main`) is tracked cleanly alongside a fully-ported dev branch.
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

**Schedule it** — `action_scripts/enable_nightly.sh` / `disable_nightly.sh` (platform-aware):
- macOS: loads/removes a launchd agent.
- Cluster (Gautschi): writes/removes a managed `scrontab` block — a daily batch job submitted with
  the `SLURM_*` options from `run_configs.env` (account/partition/QoS/GPUs/walltime) that runs the
  wrapper. One-time cluster prep first: copy the preamble
  (`cp tooling/regression/cluster_preamble.sh.example "$HOME/load_conda_cuda.sh"`) and create the
  push token (`action_scripts/create_token.sh`). Smoke-test before scheduling, either by running
  `tooling/regression/run_regression.sh` in an interactive GPU session, or `sbatch
  tooling/regression/nightly_regression.slurm` from your standing checkout.

**Is it on? / what's it done?** — `tooling/regression/status_nightly.sh` (read-only) reports both
layers that must hold for a nightly to run — the schedule (loaded launchd agent / installed
`scrontab` block) **and** the `ENABLED` kill-switch in `regression.env` — with a one-line verdict
(✅ will run · ⏸ scheduled but `ENABLED=0` · ❌ not scheduled). It then prints the **last wake** time
(from the launchd / scrontab log) and a **table of recent runs** — commit time · `meas` (the date it was
measured) · Cpu/Gpu · branch · sha · configs · gate · tests · thermal flag (ran hot / throttled · device
counts · peak temp) — plus a **CORRECTNESS** summary of any unacknowledged divergences. These are read
from the persistent metrics clone the nightly writes to (`$WORK_DIR/metrics`), falling back to this
checkout's `results/`. The
summary (`recent_runs.py`) reuses the dashboard's own `build_dashboard.collect_data()` rather than
re-parsing YAML — same parser, same record shape, run under the same `mbirjax`-env Python the
dashboard build uses (or `MBIRJAX_STATUS_PYTHON`); only the small thermal rule mirrors `dashboard.js`.
Without a PyYAML-capable Python it lists filenames only. On the cluster it also shows any nightly
currently in `squeue`.

## Verify before scheduling
```
action_scripts/run_one_night.sh     # one manual pass: clone -> test -> measure -> push
```
Confirm a `results/<plat>/<branch>/regression_<plat>_<...>.yaml` appears and `state/<plat>/<branch>`
updates; a second immediate run should report no changed branch (fire-on-change working).

## Knobs
- **`action_scripts/run_configs.env`** (the ones you edit): `TRACKED_BRANCHES`,
  `INSTALL_EXTRAS_{cpu,gpu}` (`cuda12,test` GPU / `test` CPU), `CONDA_PYTHON`, `MACOS_NIGHTLY_TIME`.
- **`regression.env`** (infra): `ENABLED` (kill-switch — prefer enable/disable_nightly) · `POLL_SCHEDULE`
  (cluster cadence; macOS uses `MACOS_NIGHTLY_TIME`) · `METRICS_URL`/`MBIRJAX_URL` · `WORK_DIR` ·
  `CONDA_ENV` · `HARNESS_DEPS` · `RUN_TESTS` / `TEST_CPU_DEVICES` · `PREAMBLE_FILE` · `TOKEN_FILE` · `NOTIFY`.

## Notes / current limits
- **Schedule:** macOS runs at `MACOS_NIGHTLY_TIME` (a daily local "HH:MM" in `run_configs.env`) — pick a
  time the Mac is normally **awake**, because a scheduled wake from sleep is a *dark wake* that won't fire
  a LaunchAgent (so a middle-of-the-night time on a slept laptop never fires). The cluster ignores it and
  uses the full `POLL_SCHEDULE` cron expression (passed straight to `scrontab`); its job uses QoS `normal`
  (the `ai` H100 partition rejects `standby`). Most of the cost is the measurement sweeps — fire-on-change
  exits in seconds otherwise.
- Per-branch **test** results are logged but **not gated/diffed** (the perf engine is the alert path).
- **Compile cache:** workers share a persistent XLA cache at `~/.mbirjax/jax_compile_cache` so the same
  shapes aren't recompiled every run (cuts the lull after each `[measure …]`). It's keyed on jaxlib
  version + HLO, so it never serves stale kernels; it only trims warmup/setup time, not the measured
  `min_ms`. Safe to delete anytime; it just grows over time.
- The **engine** gate compares each run only against **this branch's own previous run**
  (commit-over-commit). The **dashboard** layers on the broader correctness checks (vs `main`,
  single-vs-multi-device, and CPU↔GPU) and the perf "compare against" overlays — all derived from the
  tracked runs themselves, with no hand-captured reference snapshots.
