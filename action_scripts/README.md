# action_scripts

Top-level entry points + config for this metrics repo. The scripts are thin wrappers around the
engine/harness in `tooling/`; run them from the `mbirjax` conda environment (they try to activate it
for you), and each keeps the terminal open on a nonzero exit instead of closing it.

| script | purpose |
|---|---|
| `build_dashboard.sh` | Rebuild the static dashboard (`dashboard/index.html`) from the YAML time series and open it locally. (The live site is rebuilt automatically by a GitHub Action; see the repo README.) Wraps `tooling/viewer/build_dashboard.py`. |
| `add_run.sh` | Measure a **specific mbirjax commit** and add it to the tracked time series — e.g. to seed an older prerelease run onto the timeline. `--local` measures the branch checked out in your current mbirjax repo (no uncommitted changes); `<ref>` measures a branch/tag/sha from `MBIRJAX_REPO` (default `../mbirjax`); no args prints help. Checks out into a throwaway worktree (your tree is untouched). |
| `run_one_night.sh` | Run **one nightly pass** by hand — the faithful single invocation of the harness (`tooling/regression/run_regression.sh`): for each tracked branch whose remote tip moved, clone it, run the tests + the perf engine, write results, and push. Use it to verify the pipeline before enabling the scheduled nightly. |
| `enable_nightly.sh` / `disable_nightly.sh` | Start / stop the **scheduled** nightly. Platform-aware: macOS uses a launchd agent; Gautschi uses a managed SLURM `scrontab` block (resources from `run_configs.env`'s `SLURM_*` knobs). Wrap `tooling/regression/{enable,disable}_nightly.sh`. |
| `create_token.sh` | One-time setup of the fine-grained GitHub PAT used for the unattended push. See `create_token_instructions.md`. |

**`run_configs.env`** — the run-time knobs you edit: `TRACKED_BRANCHES`, `INSTALL_EXTRAS_cpu/gpu`,
`CONDA_PYTHON`, and the cluster `SLURM_*` knobs (account/partition/QoS/GPUs/cores/walltime). The
harness sources it (via `tooling/regression/regression.env`); edits propagate to the nightly
automatically (it pulls the metrics repo before each run). Harness infrastructure (URLs, paths,
credentials, schedule, the `ENABLED` kill-switch) stays in `regression.env`.

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
   action_scripts/enable_nightly.sh                         # load the launchd agent (daily at POLL_SCHEDULE)
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
   ```
   `disable_nightly.sh` removes it; `scrontab -l` and `squeue --me` inspect it. To pre-flight the
   SLURM directives without running, `sbatch --test-only tooling/regression/nightly_regression.slurm`.

See `tooling/viewer/README.md` (dashboard) and `tooling/regression/README.md` (nightly) for details.
