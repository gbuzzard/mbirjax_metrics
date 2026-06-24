# action_scripts

Top-level entry points + config for this metrics repo. The scripts are thin wrappers around the
engine/harness in `tooling/`; run them from the `mbirjax` conda environment (they try to activate it
for you), and each keeps the terminal open on a nonzero exit instead of closing it.

| script | purpose |
|---|---|
| `build_dashboard.sh` | Rebuild the static dashboard (`dashboard/index.html`) from the YAML time series and open it locally. (The live site is rebuilt automatically by a GitHub Action; see the repo README.) Wraps `tooling/dashboard/build_dashboard.py`. |
| `add_run.sh` | Measure a **specific mbirjax commit** and add it to the tracked time series — e.g. to seed an older prerelease run onto the timeline. `--local` measures the **committed `HEAD`** of the branch in your current mbirjax checkout (uncommitted changes to tracked files are rejected — it's the commit, *not* your live working tree); `<ref>` measures a branch/tag/sha from `MBIRJAX_REPO` (default `../mbirjax`); no args prints help. Either way it checks out that commit into a **throwaway git worktree** (your tree is untouched) and measures it through the **same pipeline as the nightly** — the dedicated `mbirjax_regression` env with the worktree `pip install -e`'d into it (shared via `tooling/regression/lib_env.sh`) — so a seeded point is comparable to the nightly runs around it. Your dev `mbirjax` env is never touched; the first run pulls jax into the dedicated env (slow once). Installing the worktree (not just `PYTHONPATH`) is required: a modern editable install registers a `sys.meta_path` finder that takes precedence over `PYTHONPATH`, so without it the engine would silently measure whatever mbirjax is already installed in the env. On a SLURM cluster, append `--sbatch` to **submit** the run as a batch job on a GPU node (resources from `run_configs.env`'s `SLURM_*` knobs) instead of running it in this session. |
| `run_one_night.sh` | Run **one nightly pass** by hand — the faithful single invocation of the harness (`tooling/regression/run_regression.sh`): for each tracked branch whose remote tip moved, clone it, run the tests + the perf engine, write results, and push. Use it to verify the pipeline before enabling the scheduled nightly. On a SLURM cluster, add `--sbatch` to **submit** it as a batch job on a GPU node instead of running it in the session. |
| `enable_nightly.sh` / `disable_nightly.sh` | Start / stop the **scheduled** nightly. Platform-aware: macOS uses a launchd agent; Gautschi uses a managed SLURM `scrontab` block (resources from `run_configs.env`'s `SLURM_*` knobs). Wrap `tooling/regression/{enable,disable}_nightly.sh`. |
| `status_nightly.sh` | **Is the nightly on, and what has it done?** Read-only check of both layers that must hold — the schedule (launchd agent / `scrontab` block) **and** the `ENABLED` kill-switch — with a one-line verdict (✅ will run · ⏸ paused via `ENABLED=0` · ❌ not scheduled), then the **last wake** time, a **table of recent runs** (commit time · `meas` date · Cpu/Gpu · branch · sha · configs · gate · tests · thermal flag), and a **CORRECTNESS** summary of any unacknowledged divergences. Wraps `tooling/regression/status_nightly.sh`. |
| `clear_correctness.sh` | **Acknowledge reviewed correctness divergences** through a date — writes `results/correctness_acks.yaml` so they drop off the dashboard banner / browser-tab badge (record kept). No args prints the status and confirms "clear through today? [Y/n]"; `--status` previews only. Wraps `tooling/dashboard/clear_correctness.py`. |
| `create_token.sh` | One-time setup of the fine-grained GitHub PAT used for the unattended push. See `create_token_instructions.md`. |

## Run knobs — `run_configs.env`

The run-time knobs you edit. The harness sources this (via `tooling/regression/regression.env`); because
each run pulls the metrics repo before measuring, edits here propagate to the nightly automatically.

| knob | scope | what it sets |
|---|---|---|
| `TRACKED_BRANCHES` | all | mbirjax branches to watch; each is measured only when its remote tip moves (fire-on-change). |
| `INSTALL_EXTRAS_cpu` / `INSTALL_EXTRAS_gpu` | all | pip extras for each branch's editable install (`test` = pytest + xdist; `cuda12` = `jax[cuda12]`). |
| `CONDA_PYTHON` | all | Python version for the dedicated `mbirjax_regression` env — used only when the harness must create it. |
| `MACOS_NIGHTLY_TIME` | macOS | local 24-h `HH:MM` the launchd nightly runs. Pick a time the Mac is **awake** — a scheduled wake from sleep is a "dark wake" that won't fire a LaunchAgent. Re-run `enable_nightly.sh` after changing. |
| `SLURM_ACCOUNT` · `SLURM_PARTITION` · `SLURM_QOS` | cluster | SLURM account / partition (`ai`, H100) / QoS (`normal`; `standby` isn't accepted on `ai`). |
| `SLURM_GPUS_PER_NODE` | cluster | GPUs for the sweep (4 → the full n=4 sharding sweep; 2 → n=1,2 only). |
| `SLURM_NTASKS` | cluster | CPU cores. |
| `SLURM_WALLTIME` | cluster | walltime ceiling; fire-on-change exits in seconds on a no-change night, so it's just a cap. |

Harness *infrastructure* (URLs, paths, credentials, the schedule cadence, the `ENABLED` kill-switch)
lives in `regression.env`, not here.

## One-time setup

The nightly runs on a **Mac** (launchd, the CPU sweep) and on **Gautschi** (SLURM `scrontab`, the GPU
sweep) — they write disjoint paths, so both can track the same branches in parallel. Set up each
machine once. In both cases the dedicated `mbirjax_regression` conda env is auto-created on the first
run, and the standing checkout below is only the entry point — each run fresh-clones its own working
copy under `WORK_DIR`.

### macOS (CPU)

From a shell where `conda` is on your PATH:

1. **Clone** the metrics repo to a stable location (the launchd agent points at *this* checkout's
   `run_regression.sh`, so don't delete it):
   ```
   git clone https://github.com/gbuzzard/mbirjax_metrics ~/mbirjax_metrics && cd ~/mbirjax_metrics
   ```
2. **Push token (optional on macOS)** — git can push via your macOS keychain; for a scoped PAT
   instead, run `action_scripts/create_token.sh` (the harness uses the token file when present, else
   falls back to the keychain). No node preamble is needed — conda is already on PATH, and the
   `PREAMBLE_FILE` in `regression.env` is skipped when that file is absent.
3. **Tune** `run_configs.env` (`TRACKED_BRANCHES`, `INSTALL_EXTRAS_cpu="test"`, `CONDA_PYTHON`).
4. **Smoke-test, then schedule:**
   ```
   REG_SMOKE=1 bash tooling/regression/run_regression.sh   # ~1-2 min plumbing check (no push)
   action_scripts/run_one_night.sh                          # one real pass (measures + pushes)
   action_scripts/enable_nightly.sh                         # load the launchd agent (daily at MACOS_NIGHTLY_TIME)
   action_scripts/status_nightly.sh                         # confirm it's on (schedule + ENABLED)
   ```
   `disable_nightly.sh` unloads it; logs land in `~/.mbirjax/regression/launchd.{out,err}.log`.

### Cluster (Gautschi, GPU)

On a Gautschi login node:

1. **Clone** the metrics repo to a stable location (the standing entry point):
   ```
   git clone https://github.com/gbuzzard/mbirjax_metrics ~/mbirjax_metrics && cd ~/mbirjax_metrics
   ```
2. **Preamble** — copy the template to the path `regression.env` expects (`PREAMBLE_FILE`); it
   `module load`s conda + cuda and exports the RCAC proxy so compute nodes can reach GitHub/PyPI:
   ```
   cp tooling/regression/cluster_preamble.sh.example "$HOME/load_conda_cuda.sh"
   ```
3. **Push token (required)** — the fine-grained GitHub PAT for the unattended push (no keychain on a
   compute node):
   ```
   action_scripts/create_token.sh        # writes ~/.config/mbirjax/metrics_credentials (chmod 600)
   ```
4. **Tune** `run_configs.env` if needed (the `SLURM_*` knobs: account `bouman`, partition `ai`, QoS
   `normal`, 4 GPUs, walltime).
5. **Smoke-test, then schedule** — from an interactive GPU session
   (`sinteractive -A bouman -N1 -n56 --gpus-per-node=4 -p ai -t 2:00:00`):
   ```
   REG_SMOKE=1 bash tooling/regression/run_regression.sh   # ~1-2 min plumbing check (no push)
   action_scripts/run_one_night.sh                          # one real pass (measures + pushes)
   action_scripts/enable_nightly.sh                         # install the scrontab schedule
   action_scripts/status_nightly.sh                         # confirm it's on (schedule + ENABLED)
   ```
   `disable_nightly.sh` removes it; `status_nightly.sh` (or `scrontab -l` / `squeue --me`) inspect it.
   To pre-flight the SLURM directives without running, `sbatch --test-only tooling/regression/nightly_regression.slurm`.

See `tooling/dashboard/README.md` (dashboard) and `tooling/regression/README.md` (nightly) for details.
